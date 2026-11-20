# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the helpers in maxtext.layers.fsdp_prefetch.

Pipeline-level gradient parity against the regular scanned layer stack is tested in
nnx_decoders_test.py (TestNNXDecoderPrefetchGradParity).
"""

import re
import types
import unittest
from unittest import mock

from absl.testing import absltest
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from flax import nnx
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch
from maxtext.layers import fsdp_prefetch_inhomogeneous
from maxtext.layers import fsdp_prefetch_overlap_transform
from maxtext.utils import sharding


def _abstract_mesh(shape, names):
  return AbstractMesh(shape, names, axis_types=(AxisType.Explicit,) * len(names))


class FsdpPrefetchHelpersTest(unittest.TestCase):
  """Tests for the layer-slicing, stacking, and sharding helpers."""

  def test_slice_and_expand0(self):
    tree = {"w": jnp.arange(12.0).reshape(3, 4)}
    layer = fsdp_prefetch._slice_at(tree, 1)  # pylint: disable=protected-access
    np.testing.assert_array_equal(layer["w"], np.arange(4.0, 8.0))
    sub = fsdp_prefetch._slice_range(tree, 1, 3)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sub["w"], np.arange(4.0, 12.0).reshape(2, 4))
    self.assertEqual(fsdp_prefetch._expand0(layer)["w"].shape, (1, 4))  # pylint: disable=protected-access

  def test_needs_minor_dim_layout_pin(self):
    mesh = _abstract_mesh((4, 2, 2), ("fsdp", "expert", "tensor"))
    w_3d = jnp.ones((2, 4, 8, 16))  # [layers, E, F, D] -> 3D per layer
    w_2d = jnp.ones((2, 32, 16))  # [layers, E*F, D] -> 2D per layer
    # 3D weight FSDP-sharded on minor-most dim (dim 2) needs layout pinning.
    self.assertTrue(
        fsdp_prefetch._needs_minor_dim_layout_pin(  # pylint: disable=protected-access
            w_3d, NamedSharding(mesh, PartitionSpec("expert", "tensor", "fsdp")), mesh
        )
    )
    # 3D weight FSDP-sharded on a non-minor dim (dim 1) does not need layout pinning.
    self.assertFalse(
        fsdp_prefetch._needs_minor_dim_layout_pin(  # pylint: disable=protected-access
            w_3d, NamedSharding(mesh, PartitionSpec("expert", "fsdp", "tensor")), mesh
        )
    )
    # 2D flat weight FSDP-sharded on dim 0 or dim 1 does not need layout pinning.
    self.assertFalse(
        fsdp_prefetch._needs_minor_dim_layout_pin(  # pylint: disable=protected-access
            w_2d, NamedSharding(mesh, PartitionSpec(("expert", "fsdp"), "tensor")), mesh
        )
    )
    self.assertFalse(
        fsdp_prefetch._needs_minor_dim_layout_pin(  # pylint: disable=protected-access
            w_2d, NamedSharding(mesh, PartitionSpec("tensor", "fsdp")), mesh
        )
    )

  def test_explicit_all_gather_without_fsdp_sharding_returns_none(self):
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"))
    w = jnp.ones((4, 8))
    # A size-1 FSDP axis needs no gather; the caller then falls back to resharding.
    out = fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
        w, mesh, PartitionSpec("fsdp", "tensor"), PartitionSpec(None, "tensor")
    )
    self.assertIsNone(out)

  def test_mesh_uses_fsdp(self):
    self.assertTrue(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((4, 2), ("fsdp", "tensor"))))
    self.assertTrue(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((1, 2), ("fsdp", "fsdp_transpose"))))
    self.assertFalse(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((1, 8), ("fsdp", "tensor"))))
    self.assertFalse(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((8,), ("data",))))

  def test_mesh_uses_fsdp_with_context_parallelism(self):
    # Context parallelism co-shards weights with FSDP, so an FSDP x context mesh is prefetched.
    self.assertTrue(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((2, 2), ("fsdp", "context"))))
    # Without an FSDP axis of size > 1 there is nothing to prefetch: a pure context-parallel mesh keeps
    # the regular scan, where XLA gathers the weights sharded over `context` inside each layer.
    self.assertFalse(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((1, 4), ("fsdp", "context"))))

  def test_per_layer_sharding_drops_layer_axis(self):
    # Multi-device abstract mesh (matches the mesh_uses_fsdp=True regime; independent of size-one axis removal).
    abstract_mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    abstract_stacked = jax.ShapeDtypeStruct(
        (2, 4, 8), jnp.float32, sharding=NamedSharding(abstract_mesh, PartitionSpec(None, "fsdp", "tensor"))
    )
    abstract_sharding = fsdp_prefetch._per_layer_sharding(abstract_stacked, abstract_mesh)  # pylint: disable=protected-access
    self.assertEqual(tuple(abstract_sharding.spec), ("fsdp", "tensor"))

    # Concrete 1-device (1, 1) mesh with explicit axes requires keeping size-1 axes in jax.typeof.
    if hasattr(jax.config, "jax_remove_size_one_mesh_axis_from_type"):
      previous_flag = jax.config.jax_remove_size_one_mesh_axis_from_type
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", False)
      self.addCleanup(jax.config.update, "jax_remove_size_one_mesh_axis_from_type", previous_flag)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"), axis_types=(AxisType.Explicit,) * 2)
    stacked = jax.device_put(jnp.ones((2, 4, 8)), NamedSharding(mesh, PartitionSpec(None, "fsdp", "tensor")))
    layer_sharding = fsdp_prefetch._per_layer_sharding(stacked, mesh)  # pylint: disable=protected-access
    self.assertEqual(tuple(layer_sharding.spec), ("fsdp", "tensor"))

  def test_remove_fsdp_from_sharding(self):
    mesh = _abstract_mesh((4, 2, 2), ("fsdp", "expert", "tensor"))
    tree = {
        "co_sharded": NamedSharding(mesh, PartitionSpec(("fsdp", "expert"), "tensor")),
        "routed": NamedSharding(mesh, PartitionSpec("expert", "fsdp", None)),
        "compound": NamedSharding(mesh, PartitionSpec(("expert", "fsdp"), None)),
    }
    stripped = sharding.remove_fsdp_from_sharding(tree)
    self.assertEqual(tuple(stripped["co_sharded"].spec), (None, "tensor"))
    self.assertEqual(tuple(stripped["routed"].spec), ("expert", None, None))
    self.assertEqual(tuple(stripped["compound"].spec), ("expert", None))

  def test_sc_rs_padded_minor_size(self):
    # 18432 = 144 * 128 (144 = 9 * 16, odd_part(144 // 3) = 3) is already valid.
    self.assertEqual(fsdp_prefetch._sc_rs_padded_minor_size(18432), 18432)  # pylint: disable=protected-access
    # DeepSeek-V3 vocab 129280 = 1010 * 128 pads up to 147456 = 1152 * 128 (1152 = 9 * 128).
    self.assertEqual(fsdp_prefetch._sc_rs_padded_minor_size(129280), 147456)  # pylint: disable=protected-access
    # Power-of-two 32768 = 256 * 128 pads up to 36864 = 288 * 128 (288 = 9 * 32).
    self.assertEqual(fsdp_prefetch._sc_rs_padded_minor_size(32768), 36864)  # pylint: disable=protected-access


class FsdpPrefetchSparseCoreTest(unittest.TestCase):
  """Tests for automatic SparseCore pinning and post-scheduler overlap transform registration."""

  def test_on_sparse_core_is_identity_when_disabled_or_for_vectors(self):
    fn = lambda x: x  # pylint: disable=unnecessary-lambda-assignment
    self.assertIs(fsdp_prefetch._on_sparse_core(fn, -1), fn)  # pylint: disable=protected-access
    self.assertIs(fsdp_prefetch._on_sparse_core(fn, None), fn)  # pylint: disable=protected-access
    # Vectors and narrow minor dims stay with XLA.
    self.assertIs(fsdp_prefetch._on_sparse_core(fn, 1, jnp.ones((256,))), fn)  # pylint: disable=protected-access
    self.assertIs(fsdp_prefetch._on_sparse_core(fn, 1, jnp.ones((256, 64))), fn)  # pylint: disable=protected-access
    self.assertIsNot(fsdp_prefetch._on_sparse_core(fn, 1, jnp.ones((8, 256))), fn)  # pylint: disable=protected-access

  def test_default_sparse_core_id_pins_sc1_on_tpu7x_only(self):
    def fake_mesh(device_kind):
      dev = types.SimpleNamespace(device_kind=device_kind, platform="tpu")
      return types.SimpleNamespace(devices=np.array([dev], dtype=object))

    self.assertEqual(fsdp_prefetch._default_sparse_core_id(fake_mesh("TPU7x")), 1)  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_mesh("TPU v6 lite")))  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_mesh("cpu")))  # pylint: disable=protected-access

  def test_pinned_all_gather_and_dim1_reduce_scatter_lower_to_sparse_core_region(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    w = jax.ShapeDtypeStruct((16, 512), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec("fsdp", "tensor")))

    def gather(x, core_id):
      return fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
          x, mesh, PartitionSpec("fsdp", "tensor"), PartitionSpec(None, "tensor"), reduced=True, sparse_core_id=core_id
      )

    g = jax.ShapeDtypeStruct(
        (256, 512),
        jnp.float32,
        sharding=NamedSharding(mesh, PartitionSpec(None, None, unreduced=frozenset({"fsdp"}))),
    )

    def scatter_dim1(x, core_id):
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          x, mesh, PartitionSpec(None, "fsdp"), PartitionSpec(None, None), sparse_core_id=core_id
      )

    with jax.sharding.use_abstract_mesh(mesh):
      pinned = jax.jit(lambda x: gather(x, 1)).trace(w).lower(lowering_platforms=("tpu",)).as_text()
      unpinned = jax.jit(lambda x: gather(x, None)).trace(w).lower(lowering_platforms=("tpu",)).as_text()
      rs_pinned = jax.jit(lambda x: scatter_dim1(x, 1)).trace(g).lower(lowering_platforms=("tpu",)).as_text()
    # Both collectives run in a SparseCore offload region pinned to SparseCore 1, and the dim-1 2D
    # reduce-scatter moves the scatter dimension to axis 0 inside the region.
    self.assertIn('_xla_compute_type = "sparseoffload"', pinned)
    self.assertRegex(pinned, r"core_ids.{0,8}\[1\]")
    self.assertNotIn("sparseoffload", unpinned)
    self.assertIn('_xla_compute_type = "sparseoffload"', rs_pinned)
    self.assertRegex(rs_pinned, r"core_ids.{0,8}\[1\]")
    self.assertIn("dimension = 0", rs_pinned)

  def test_pinned_3d_non_major_reduce_scatter_moves_scatter_dim_to_axis0(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "expert"))
    # MLA `self_attention.out.kernel` shape [heads, v_head_dim, emb] sharded on dim=2 over ('fsdp', 'expert').
    g_dim2 = jax.ShapeDtypeStruct(
        (8, 128, 512),
        jnp.float32,
        sharding=NamedSharding(mesh, PartitionSpec(None, None, None, unreduced=frozenset({"fsdp", "expert"}))),
    )
    # 3D routed expert weight shape [experts, emb, mlp] sharded on dim=1 over 'fsdp'.
    g_dim1 = jax.ShapeDtypeStruct(
        (8, 256, 256),
        jnp.float32,
        sharding=NamedSharding(mesh, PartitionSpec("expert", None, None, unreduced=frozenset({"fsdp"}))),
    )

    def scatter_dim2(x):
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          x, mesh, PartitionSpec(None, None, ("fsdp", "expert")), PartitionSpec(None, None, None), sparse_core_id=1
      )

    def scatter_dim1(x):
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          x, mesh, PartitionSpec("expert", "fsdp", None), PartitionSpec("expert", None, None), sparse_core_id=1
      )

    with jax.sharding.use_abstract_mesh(mesh):
      hlo_dim2 = jax.jit(scatter_dim2).trace(g_dim2).lower(lowering_platforms=("tpu",)).as_text()
      hlo_dim1 = jax.jit(scatter_dim1).trace(g_dim1).lower(lowering_platforms=("tpu",)).as_text()

    for hlo in (hlo_dim2, hlo_dim1):
      self.assertIn('_xla_compute_type = "sparseoffload"', hlo)
      self.assertRegex(hlo, r"core_ids.{0,8}\[1\]")
      self.assertIn("dimension = 0", hlo)
      self.assertNotIn("dimension = 1", hlo)
      self.assertNotIn("dimension = 2", hlo)

  def test_overlap_transform_registered_on_tpu_mesh_only(self):
    def fake_mesh(platform):
      dev = types.SimpleNamespace(platform=platform, device_kind="TPU7x" if platform == "tpu" else "cpu")
      return types.SimpleNamespace(devices=np.array([dev], dtype=object))

    with mock.patch.object(fsdp_prefetch_overlap_transform, "register_transform") as register:
      fsdp_prefetch._register_overlap_transform(fake_mesh("cpu"))  # pylint: disable=protected-access
      register.assert_not_called()
      fsdp_prefetch._register_overlap_transform(fake_mesh("tpu"))  # pylint: disable=protected-access
      register.assert_called_once_with(platforms="tpu")

  def test_reschedules_unrolled_two_layer_scan_body_starts_and_dones(self):
    # Models an unroll=2 backward scan while body with two layers on SparseCore 1 ("sparse_core:1"):
    #   Layer 1: RS(k+1) and AG(k-1) overlapped with Layer 1 compute (dot1_a -> dot1_b),
    #   Midpoint optimization_barrier separating Layer 1 from Layer 2,
    #   Layer 2: RS(k) and AG(k-2) overlapped with Layer 2 compute (dot2_a -> dot2_b).
    # In the raw schedule, starts are delayed after compute begins and dones sit right after starts.
    order = (
        "param",
        "dot1_a",
        "rs1_start",
        "rs1_done",
        "rs1_dus",
        "slice1",
        "ag1_start",
        "ag1_done",
        "dot1_b",
        "slice2",
        "mid_barrier",
        "gte_y2",
        "gte_p2",
        "dot2_a",
        "rs2_start",
        "rs2_done",
        "rs2_dus",
        "ag2_start",
        "ag2_done",
        "dot2_b",
        "root",
    )
    opcodes = {
        "param": "kParameter",
        "dot1_a": "kDot",
        "rs1_start": "kAsyncStart",
        "rs1_done": "kAsyncDone",
        "rs1_dus": "kDynamicUpdateSlice",
        "slice1": "kDynamicSlice",
        "ag1_start": "kAllGatherStart",
        "ag1_done": "kAllGatherDone",
        "dot1_b": "kDot",
        "slice2": "kDynamicSlice",
        "mid_barrier": "kOptimizationBarrier",
        "gte_y2": "kGetTupleElement",
        "gte_p2": "kGetTupleElement",
        "dot2_a": "kDot",
        "rs2_start": "kAsyncStart",
        "rs2_done": "kAsyncDone",
        "rs2_dus": "kDynamicUpdateSlice",
        "ag2_start": "kAllGatherStart",
        "ag2_done": "kAllGatherDone",
        "dot2_b": "kDot",
        "root": "kTuple",
    }
    operands = {
        "param": (),
        "dot1_a": ("param",),
        "rs1_start": ("param",),
        "rs1_done": ("rs1_start",),
        "rs1_dus": ("param", "rs1_done"),
        "slice1": ("param",),
        "ag1_start": ("slice1",),
        "ag1_done": ("ag1_start",),
        "dot1_b": ("dot1_a",),
        "slice2": ("param",),
        "mid_barrier": ("dot1_b", "slice2"),
        "gte_y2": ("mid_barrier",),
        "gte_p2": ("mid_barrier",),
        "dot2_a": ("gte_y2", "ag1_done"),
        "rs2_start": ("dot1_b",),
        "rs2_done": ("rs2_start",),
        "rs2_dus": ("rs1_dus", "rs2_done"),
        "ag2_start": ("gte_p2",),
        "ag2_done": ("ag2_start",),
        "dot2_b": ("dot2_a",),
        "root": ("dot2_b", "ag2_done", "rs2_dus"),
    }
    data_formatting = {"rs1_dus", "slice1", "slice2", "mid_barrier", "gte_y2", "gte_p2", "rs2_dus"}
    queues = {
        "rs1_start": "sparse_core:1",
        "ag1_start": "sparse_core:1",
        "rs2_start": "sparse_core:1",
        "ag2_start": "sparse_core:1",
    }
    graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        data_formatting=data_formatting,
        queues=queues,
    )
    rescheduled = fsdp_prefetch_overlap_transform.reschedule(graph)
    fsdp_prefetch_overlap_transform.validate_order(graph, rescheduled)
    pos = {name: i for i, name in enumerate(rescheduled)}

    # Layer 1 starts (rs1_start, ag1_start) hoist before Layer 1 compute (dot1_a, dot1_b).
    self.assertLess(pos["rs1_start"], pos["dot1_a"])
    self.assertLess(pos["ag1_start"], pos["dot1_a"])
    # ag1_start has an earlier deadline (dot2_a) than rs1_start (root), so ag1_start issues first on
    # SparseCore 1, ag1_done sinks to the end of Layer 1 (after dot1_b, before dot2_a), and rs1_done
    # sinks past Layer 2 compute to the end of the while body ahead of ag2_done and rs2_done.
    self.assertLess(pos["ag1_start"], pos["rs1_start"])
    self.assertGreater(pos["ag1_done"], pos["dot1_b"])
    self.assertLess(pos["ag1_done"], pos["dot2_a"])
    self.assertGreater(pos["rs1_done"], pos["dot2_b"])
    # Layer 2 starts (rs2_start, ag2_start) sit after Layer 1 compute (dot1_b) and before Layer 2 compute (dot2_a).
    self.assertGreater(pos["rs2_start"], pos["dot1_b"])
    self.assertGreater(pos["ag2_start"], pos["dot1_b"])
    self.assertLess(pos["rs2_start"], pos["dot2_a"])
    self.assertLess(pos["ag2_start"], pos["dot2_a"])
    # Layer 2 dones (rs2_done, ag2_done) sink after Layer 2 compute (dot2_b) and immediately before root.
    self.assertGreater(pos["rs2_done"], pos["dot2_b"])
    self.assertGreater(pos["ag2_done"], pos["dot2_b"])
    self.assertLess(pos["ag2_done"], pos["root"])
    # SparseCore 1 FIFO order is preserved: start(A) < start(B) <=> done(A) < done(B).
    sc_starts = sorted(queues, key=pos.__getitem__)
    done_of = {"rs1_start": "rs1_done", "ag1_start": "ag1_done", "rs2_start": "rs2_done", "ag2_start": "ag2_done"}
    sc_dones = [done_of[s] for s in sc_starts]
    self.assertEqual(sc_dones, sorted(sc_dones, key=pos.__getitem__))

  def test_async_queue_separates_sc0_and_sc1_with_nested_consumer_core_ids(self):
    sc0_text = (
        "%call-start.10 = ((bf16[4,2048]{1,0}), bf16[32,2048]{1,0}) call-start(%slice.1), "
        'async_execution_thread="sparsecore", '
        'backend_config={"operation_queue_id":"0",'
        '"sparse_core_config":{"consumers":[{"core_ids":[]}],"producers":[{"core_ids":[]}],"core_ids":["0"]}}'
    )
    sc1_text = (
        "%call-start.11 = ((bf16[32,2048]{1,0}), bf16[512,2048]{1,0}) call-start(%slice.2), "
        'async_execution_thread="sparsecore", '
        'backend_config={"operation_queue_id":"0",'
        '"sparse_core_config":{"consumers":[{"core_ids":[]}],"core_ids":["1"]}}'
    )
    # Also verify stringified `backend_config="{...}"` emitted by XLA HLO printers.
    sc0_stringified = (
        "%call-start.12 = ((bf16[4,2048]{1,0}), bf16[32,2048]{1,0}) call-start(%slice.1), "
        'async_execution_thread="sparsecore", '
        r'backend_config="{\"operation_queue_id\":\"0\",'
        r'\"sparse_core_config\":{\"consumers\":[{\"core_ids\":[]}],\"core_ids\":[\"0\"]}}"'
    )
    sc1_stringified = (
        "%call-start.13 = ((bf16[32,2048]{1,0}), bf16[512,2048]{1,0}) call-start(%slice.2), "
        'async_execution_thread="sparsecore", '
        r'backend_config="{\"operation_queue_id\":\"0\",'
        r'\"sparse_core_config\":{\"consumers\":[{\"core_ids\":[]}],\"core_ids\":[\"1\"]}}"'
    )
    q_sc0 = fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc0_text)
    q_sc1 = fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc1_text)
    self.assertEqual(q_sc0, "sparsecore:0")
    self.assertEqual(q_sc1, "sparsecore:1")
    self.assertEqual(fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc0_stringified), "sparsecore:0")
    self.assertEqual(fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc1_stringified), "sparsecore:1")

    # Edge cases: non-SparseCore `backend_config`, malformed `backend_config`, regex fallback, and host/non-async ops.
    self.assertIsNone(
        fsdp_prefetch_overlap_transform.async_queue(
            "kAsyncStart", '%ag = call-start(%p), async_execution_thread="sparsecore", backend_config={"other":1}'
        )
    )
    self.assertIsNone(
        fsdp_prefetch_overlap_transform.async_queue(
            "kAsyncStart", '%ag = call-start(%p), async_execution_thread="sparsecore", backend_config="{not_json"'
        )
    )
    self.assertEqual(
        fsdp_prefetch_overlap_transform.async_queue(
            "kAsyncStart", '%ag = call-start(%p), async_execution_thread="sparsecore", attrs={"core_ids": [1]}'
        ),
        "sparsecore:1",
    )
    self.assertIsNone(
        fsdp_prefetch_overlap_transform.async_queue(
            "kAsyncStart", '%ag = call-start(%p), async_execution_thread="sparsecore"'
        )
    )
    self.assertIsNone(fsdp_prefetch_overlap_transform.async_queue("kDot", sc1_text))
    self.assertIsNone(
        fsdp_prefetch_overlap_transform.async_queue(
            "kAsyncStart", '%h = call-start(%p), async_execution_thread="host", attrs={"core_ids": [0]}'
        )
    )

    # An SC1 FSDP prefetch issued before an SC0 MoE EP collective must not have its done capped before
    # the SC0 done (since SC0 and SC1 have independent hardware FIFO queues).
    order = ("param", "sc1_ag_start", "sc1_ag_done", "sc0_ep_start", "sc0_ep_done", "moe_dot", "next_dot", "root")
    opcodes = {
        "param": "kParameter",
        "sc1_ag_start": "kAsyncStart",
        "sc1_ag_done": "kAsyncDone",
        "sc0_ep_start": "kAsyncStart",
        "sc0_ep_done": "kAsyncDone",
        "moe_dot": "kDot",
        "next_dot": "kDot",
        "root": "kTuple",
    }
    operands = {
        "param": (),
        "sc1_ag_start": ("param",),
        "sc1_ag_done": ("sc1_ag_start",),
        "sc0_ep_start": ("param",),
        "sc0_ep_done": ("sc0_ep_start",),
        "moe_dot": ("sc0_ep_done",),
        "next_dot": ("moe_dot", "sc1_ag_done"),
        "root": ("next_dot",),
    }
    graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        queues={"sc1_ag_start": q_sc1, "sc0_ep_start": q_sc0},
    )
    rescheduled = fsdp_prefetch_overlap_transform.reschedule(graph)
    fsdp_prefetch_overlap_transform.validate_order(graph, rescheduled)
    pos = {name: i for i, name in enumerate(rescheduled)}
    self.assertGreater(pos["sc1_ag_done"], pos["moe_dot"])
    self.assertLess(pos["sc1_ag_done"], pos["next_dot"])

  def test_ignores_pipelined_vmem_dma_async_ops_and_loop_fences(self):
    def fake_inst(name, opcode_name, operand_list, text):
      return types.SimpleNamespace(
          name=name,
          opcode=types.SimpleNamespace(name=opcode_name),
          operands=lambda operand_list=operand_list: operand_list,
          to_string=lambda text=text: text,
      )

    # Inner recurrence loop body containing only intra-core VMEM DMAs (`dma_priority` / `dynamic-slice-start`),
    # where XLA's loop pipeliner consumes a previous-iteration DMA from a loop parameter tuple (`kGetTupleElement`).
    b_param = fake_inst("b_param", "kParameter", [], "%b_param = (bf16[8,128]{1,0}) parameter(0)")
    b_gte = fake_inst(
        "b_gte", "kGetTupleElement", [b_param], "%b_gte = bf16[8,128]{1,0} get-tuple-element(%b_param), index=0"
    )
    b_dma_done = fake_inst(
        "dynamic-slice-done.3",
        "kAsyncDone",
        [b_gte],
        '%dynamic-slice-done.3 = bf16[8,128S(1)]{1,0} dynamic-slice-done(%b_gte), backend_config={"dma_priority":0}',
    )
    b_dma_start = fake_inst(
        "dynamic-slice-start.4",
        "kAsyncStart",
        [b_dma_done],
        "%dynamic-slice-start.4 = ((bf16[8,128]{1,0}), bf16[8,128S(1)]{1,0}) dynamic-slice-start(%dynamic-slice-done.3), "
        'backend_config={"dma_priority":0}',
    )
    b_root = fake_inst("b_root", "kTuple", [b_dma_start], "%b_root = tuple(%dynamic-slice-start.4)")
    pipelined_body = types.SimpleNamespace(
        name="gdn_chunk_post_fusion_pipelined_body",
        instructions=lambda: [b_param, b_gte, b_dma_done, b_dma_start, b_root],
    )

    # Outer layer computation: issues an SC1 FSDP weight prefetch before the inner VMEM-DMA loop, and
    # drains the pipelined VMEM DMA after the inner loop (`dynamic-slice-done.12` whose operand 0 is `gte`,
    # even without `backend_config` on the done).
    param = fake_inst("param", "kParameter", [], "%param = bf16[32,2048]{1,0} parameter(0)")
    ag_start = fake_inst(
        "ag_start",
        "kAsyncStart",
        [param],
        "%ag_start = ((bf16[32,2048]{1,0}), bf16[256,2048]{1,0}) call-start(%param), "
        'async_execution_thread="sparsecore", backend_config={"sparse_core_config":{"core_ids":["1"]}}',
    )
    ag_done = fake_inst(
        "ag_done",
        "kAsyncDone",
        [ag_start],
        '%ag_done = bf16[256,2048]{1,0} call-done(%ag_start), async_execution_thread="sparsecore"',
    )
    vmem_prologue = fake_inst(
        "dynamic-slice-start.1",
        "kAsyncStart",
        [param],
        "%dynamic-slice-start.1 = ((bf16[32,2048]{1,0}), bf16[8,128S(1)]{1,0}) dynamic-slice-start(%param), "
        r'backend_config="{\"dma_priority\":0}"',
    )
    host_start = fake_inst(
        "host_offload_start",
        "kAsyncStart",
        [param],
        "%host_offload_start = ((bf16[32,2048]{1,0}), bf16[32,2048]{1,0:S(5)}) call-start(%param)",
    )
    host_done = fake_inst(
        "host_offload_done",
        "kAsyncDone",
        [host_start],
        "%host_offload_done = bf16[32,2048]{1,0} call-done(%host_offload_start)",
    )
    chunk_while = fake_inst(
        "chunk_while",
        "kWhile",
        [vmem_prologue],
        "%chunk_while = ((bf16[8,128]{1,0})) while(%dynamic-slice-start.1), "
        "condition=%chunk_cond, body=%gdn_chunk_post_fusion_pipelined_body",
    )
    while_gte = fake_inst(
        "while_gte", "kGetTupleElement", [chunk_while], "%while_gte = get-tuple-element(%chunk_while), index=0"
    )
    vmem_epilogue = fake_inst(
        "dynamic-slice-done.12",
        "kAsyncDone",
        [while_gte],
        "%dynamic-slice-done.12 = bf16[8,128S(1)]{1,0} dynamic-slice-done(%while_gte)",
    )
    next_dot = fake_inst(
        "next_dot",
        "kDot",
        [vmem_epilogue, ag_done, host_done],
        "%next_dot = dot(%dynamic-slice-done.12, %ag_done, %host_offload_done)",
    )
    root = fake_inst("root", "kTuple", [next_dot], "%root = tuple(%next_dot)")
    outer_seq = [
        param,
        ag_start,
        ag_done,
        host_start,
        host_done,
        vmem_prologue,
        chunk_while,
        while_gte,
        vmem_epilogue,
        next_dot,
        root,
    ]
    outer_comp = types.SimpleNamespace(name="sublayer_body", instructions=lambda: outer_seq)
    entry_while = fake_inst(
        "outer_while",
        "kWhile",
        [param],
        "%outer_while = tuple() while(%param), condition=%outer_cond, body=%sublayer_body",
    )
    entry_seq = [param, entry_while, root]
    entry_comp = types.SimpleNamespace(name="main", instructions=lambda: entry_seq)

    sequences = {
        "gdn_chunk_post_fusion_pipelined_body": [b_param, b_gte, b_dma_done, b_dma_start, b_root],
        "sublayer_body": outer_seq,
        "main": entry_seq,
    }
    updated_sequences = {}
    fake_schedule = types.SimpleNamespace(
        sequence=lambda comp: sequences[comp.name],
        set_sequence=lambda comp, seq: updated_sequences.__setitem__(comp.name, [i.name for i in seq]),
        verify=lambda: None,
    )
    fake_module = types.SimpleNamespace(
        name="test_module",
        schedule=lambda: fake_schedule,
        set_schedule=lambda _: None,
        to_string=lambda *_: "HloModule test_module\nENTRY %main () -> () {\n}\n",
        computations=lambda: [pipelined_body, outer_comp, entry_comp],
        make_nonfusion_computations=lambda: [pipelined_body, outer_comp, entry_comp],
    )
    all_bodies, async_bodies = fsdp_prefetch_overlap_transform._while_and_async_body_names(fake_module)  # pylint: disable=protected-access
    self.assertIn("gdn_chunk_post_fusion_pipelined_body", all_bodies)
    # VMEM-DMA-only loop must not be classified as a collective-issuing async body / fence.
    self.assertNotIn("gdn_chunk_post_fusion_pipelined_body", async_bodies)
    self.assertIn("sublayer_body", async_bodies)

    rewritten = fsdp_prefetch_overlap_transform.transform_module(fake_module)
    self.assertEqual(set(rewritten), {"sublayer_body", "main"})
    self.assertNotIn("gdn_chunk_post_fusion_pipelined_body", updated_sequences)
    pos = {name: i for i, name in enumerate(updated_sequences["sublayer_body"])}
    # `ag_done` overlaps the entire `chunk_while` loop instead of being fenced before `chunk_while`.
    self.assertLess(pos["ag_start"], pos["chunk_while"])
    self.assertGreater(pos["ag_done"], pos["chunk_while"])
    self.assertLess(pos["ag_done"], pos["next_dot"])

  def test_only_queued_async_work_fences_nested_loops(self):
    sparse_core_start = (
        '%call-start.1 = ((), ()) async-start(), async_execution_thread="sparsecore", calls=%c, '
        'backend_config={"sparse_core_config":{"core_ids":["1"]}}'
    )
    ring_permute_start = "%collective-permute-start = (bf16[8], bf16[8]) collective-permute-start(%x), channel_id=1"
    self.assertTrue(fsdp_prefetch_overlap_transform.issues_queued_async_work("kAsyncStart", sparse_core_start))
    self.assertFalse(
        fsdp_prefetch_overlap_transform.issues_queued_async_work("kCollectivePermuteStart", ring_permute_start)
    )

    def _inst(opcode, text):
      return types.SimpleNamespace(opcode=types.SimpleNamespace(name=opcode), to_string=lambda t=text: t)

    def _comp(name, insts):
      return types.SimpleNamespace(name=name, instructions=lambda items=insts: items)

    fake_module = types.SimpleNamespace(
        make_nonfusion_computations=lambda: [
            _comp(
                "outer",
                [
                    _inst("kWhile", "%w1 = () while(%p), body=%sc_body"),
                    _inst("kWhile", "%w2 = () while(%p), body=%ring_body"),
                ],
            ),
            _comp("sc_body", [_inst("kAsyncStart", sparse_core_start)]),
            _comp("ring_body", [_inst("kCollectivePermuteStart", ring_permute_start)]),
        ]
    )
    while_bodies, async_bodies = fsdp_prefetch_overlap_transform._while_and_async_body_names(  # pylint: disable=protected-access
        fake_module
    )
    self.assertEqual(while_bodies, {"sc_body", "ring_body"})
    self.assertEqual(async_bodies, {"sc_body"})

  def test_unfenced_nested_loop_lets_reduce_scatter_overlap_it(self):
    # A layer body whose attention runs in a nested loop (ring attention). The deferred gradient
    # reduce-scatter is independent of the loop, so it should start before and finish after it unless
    # the loop is a fence.
    order = ("param", "pre", "rs_start", "rs_done", "ring_loop", "post", "root")
    opcodes = {
        "param": "kParameter",
        "pre": "kDot",
        "rs_start": "kAsyncStart",
        "rs_done": "kAsyncDone",
        "ring_loop": "kWhile",
        "post": "kDot",
        "root": "kTuple",
    }
    operands = {
        "param": (),
        "pre": ("param",),
        "rs_start": ("param",),
        "rs_done": ("rs_start",),
        "ring_loop": ("pre",),
        "post": ("ring_loop",),
        "root": ("post", "rs_done"),
    }
    queues = {"rs_start": "sparsecore:1"}

    def schedule(fences):
      graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
          order=order, opcodes=opcodes, operands=operands, queues=queues, fences=fences
      )
      rescheduled = fsdp_prefetch_overlap_transform.reschedule(graph)
      fsdp_prefetch_overlap_transform.validate_order(graph, rescheduled)
      return {name: i for i, name in enumerate(rescheduled)}

    pos = schedule(fences=())
    self.assertLess(pos["rs_start"], pos["ring_loop"])
    self.assertGreater(pos["rs_done"], pos["post"])
    fenced = schedule(fences=("ring_loop",))
    self.assertLess(fenced["rs_done"], fenced["ring_loop"])


class FsdpPrefetchAllGatherTest(unittest.TestCase):
  """Lowers the FSDP weight all-gather against abstract multi-device meshes (no devices needed)."""

  def _lower_gather(self, mesh, spec, shape):
    """Returns (output ShapeDtypeStruct, all_gather StableHLO lines) of `_apply_sharding_hint` in explicit mode."""
    w = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=NamedSharding(mesh, spec))

    def gather(x):
      return fsdp_prefetch._apply_sharding_hint(x, mesh, ShardMode.EXPLICIT)  # pylint: disable=protected-access

    with jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(gather, w)
      # An abstract mesh has no devices, so the lowering platform must be given explicitly.
      hlo = jax.jit(gather).trace(w).lower(lowering_platforms=("tpu",)).as_text()
    return out, [line for line in hlo.splitlines() if "stablehlo.all_gather" in line]

  def test_explicit_mode_gathers_fsdp_axis_with_tagged_all_gather(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    out, gathers = self._lower_gather(mesh, PartitionSpec("fsdp", "tensor"), (8, 6))
    self.assertEqual(out.shape, (8, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), (None, "tensor"))  # Non-FSDP axes stay sharded.
    # Typed `reduced` over the gathered axis, so its gradient is unreduced and reduced once by the tagged RS.
    self.assertEqual(out.sharding.spec.reduced, frozenset({"fsdp"}))
    self.assertEqual(len(gathers), 1)
    # One all-gather over the 4-way 'fsdp' axis of dim 0; the 2-way 'tensor' shard of dim 1 is kept.
    self.assertIn("all_gather_dim = 0", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertNotIn('name = "tensor"', gathers[0])
    self.assertIn("(tensor<2x3xf32>) -> tensor<8x3xf32>", gathers[0])

  def test_explicit_mode_gathers_multiple_fsdp_axes_of_one_dim(self):
    mesh = _abstract_mesh((2, 2, 2), ("tensor", "fsdp", "fsdp_transpose"))
    out, gathers = self._lower_gather(mesh, PartitionSpec(None, ("tensor", "fsdp", "fsdp_transpose")), (3, 16))
    self.assertEqual(out.shape, (3, 16))
    self.assertEqual(tuple(out.sharding.spec.partitions), (None, "tensor"))
    self.assertEqual(out.sharding.spec.reduced, frozenset({"fsdp", "fsdp_transpose"}))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 1", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertIn('name = "fsdp_transpose"', gathers[0])
    self.assertIn("(tensor<3x2xf32>) -> tensor<3x8xf32>", gathers[0])

  def test_explicit_mode_gathers_co_sharded_expert_axis_with_fsdp(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "expert"))
    # Non-routed weights (attention, dense MLP, shared experts) are sharded over ('fsdp', 'expert').
    out, gathers = self._lower_gather(mesh, PartitionSpec(("fsdp", "expert"), None), (16, 6))
    self.assertEqual(out.shape, (16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), (None, None))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 0", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertIn('name = "expert"', gathers[0])
    self.assertIn("(tensor<2x6xf32>) -> tensor<16x6xf32>", gathers[0])

  def test_explicit_mode_preserves_routed_expert_axis(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "expert"))
    # MoE routed expert weights have 'expert' on dim 0 and 'fsdp' on dim 1; only 'fsdp' is gathered.
    out, gathers = self._lower_gather(mesh, PartitionSpec("expert", "fsdp", None), (4, 16, 6))
    self.assertEqual(out.shape, (4, 16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), ("expert", None, None))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 1", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertNotIn('name = "expert"', gathers[0])
    self.assertIn("(tensor<2x4x6xf32>) -> tensor<2x16x6xf32>", gathers[0])

  def test_explicit_mode_preserves_leading_expert_axis_on_compound_dim(self):
    mesh = _abstract_mesh((2, 4), ("expert", "fsdp"))
    # Compound ('expert', 'fsdp') on the expert dim (e.g. moe_flat_fsdp_weights) keeps 'expert' sharded.
    out, gathers = self._lower_gather(mesh, PartitionSpec(("expert", "fsdp"), None), (16, 6))
    self.assertEqual(out.shape, (16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), ("expert", None))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 0", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertNotIn('name = "expert"', gathers[0])
    self.assertIn("(tensor<2x6xf32>) -> tensor<8x6xf32>", gathers[0])

  def test_explicit_mode_gathers_co_sharded_context_axis_with_fsdp(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "context"))
    # Under context parallelism, `embed`-like weight dimensions are sharded over ('fsdp', 'context').
    out, gathers = self._lower_gather(mesh, PartitionSpec(("fsdp", "context"), None), (16, 6))
    self.assertEqual(out.shape, (16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), (None, None))
    self.assertEqual(out.sharding.spec.reduced, frozenset({"fsdp", "context"}))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 0", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertIn('name = "context"', gathers[0])
    self.assertIn("(tensor<2x6xf32>) -> tensor<16x6xf32>", gathers[0])

  def test_explicit_mode_preserves_routed_expert_axis_with_context(self):
    mesh = _abstract_mesh((2, 2, 2), ("fsdp", "context", "expert"))
    # Routed expert weights under context parallelism: 'expert' on dim 0, ('fsdp', 'context') on dim 1.
    out, gathers = self._lower_gather(mesh, PartitionSpec("expert", ("fsdp", "context"), None), (4, 16, 6))
    self.assertEqual(out.shape, (4, 16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), ("expert", None, None))
    self.assertEqual(out.sharding.spec.reduced, frozenset({"fsdp", "context"}))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 1", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertIn('name = "context"', gathers[0])
    self.assertNotIn('name = "expert"', gathers[0])
    self.assertIn("(tensor<2x4x6xf32>) -> tensor<2x16x6xf32>", gathers[0])

  def test_non_minor_fsdp_axis_falls_back_to_resharding(self):
    mesh = _abstract_mesh((2, 4), ("tensor", "fsdp"))
    # 'fsdp' is not the minor-most axis of dim 0, so a tiled gather would not give the target layout.
    out = fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
        jnp.ones((8, 6)), mesh, PartitionSpec(("fsdp", "tensor"), None), PartitionSpec("tensor", None)
    )
    self.assertIsNone(out)

  def test_leaf_without_named_sharding_is_left_alone(self):
    mesh = _abstract_mesh((4,), ("fsdp",))
    leaf = np.ones((2, 3))
    with mock.patch.object(fsdp_prefetch.jax, "typeof", return_value=types.SimpleNamespace(sharding=None)):
      self.assertIs(fsdp_prefetch._apply_sharding_hint(leaf, mesh, ShardMode.EXPLICIT), leaf)  # pylint: disable=protected-access
      self.assertIsNone(fsdp_prefetch._per_layer_sharding(leaf, mesh))  # pylint: disable=protected-access


class FsdpPrefetchReduceScatterTest(unittest.TestCase):
  """Lowers the tagged dW reduce-scatter against abstract multi-device meshes (no devices needed)."""

  def _lower_scatter(self, mesh, src_spec, shape, unreduced=True, unreduced_axes=None):
    """Returns (output ShapeDtypeStruct, reduce_scatter StableHLO lines) for a gradient of a `src_spec` weight.

    With `unreduced`, the gradient is typed as the partial sums of a `reduced` gathered weight, over
    `unreduced_axes` (default: the FSDP axes of size > 1).
    """
    gathered = sharding.remove_fsdp_from_sharding(NamedSharding(mesh, src_spec)).spec
    if unreduced_axes is None:
      unreduced_axes = (a for a in mesh.axis_names if a.startswith("fsdp") and mesh.shape[a] > 1)
    g_spec = PartitionSpec(*gathered, unreduced=frozenset(unreduced_axes)) if unreduced else gathered
    g = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=NamedSharding(mesh, g_spec))

    def scatter(x):
      out = fsdp_prefetch._explicit_fsdp_reduce_scatter(x, mesh, src_spec, gathered)  # pylint: disable=protected-access
      return x if out is None else out

    with jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(scatter, g)
      hlo = jax.jit(scatter).trace(g).lower(lowering_platforms=("tpu",)).as_text().splitlines()
    # A reduce_scatter op spans its reduction region; its attributes are on the line closing it (`}) {...}`).
    ops = []
    for i, line in enumerate(hlo):
      if "stablehlo.reduce_scatter" in line:
        end = next(j for j in range(i, len(hlo)) if hlo[j].lstrip().startswith("})"))
        ops.append(" ".join(hlo[i : end + 1]))
    return out, ops

  def test_scatters_fsdp_axis_with_tagged_reduce_scatter(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    out, scatters = self._lower_scatter(mesh, PartitionSpec("fsdp", "tensor"), (8, 6))
    self.assertEqual(out.shape, (8, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), ("fsdp", "tensor"))
    self.assertEqual(out.sharding.spec.unreduced, frozenset())
    self.assertEqual(len(scatters), 1)
    self.assertIn("scatter_dimension = 0", scatters[0])
    self.assertIn("(tensor<8x3xf32>) -> tensor<2x3xf32>", scatters[0])

  def test_flat_expert_weight_scatters_only_fsdp_of_compound_dim(self):
    mesh = _abstract_mesh((2, 4), ("expert", "fsdp"))
    # moe_flat_fsdp_weights: dim 0 is sharded over ('expert', 'fsdp'); only 'fsdp' is reduce-scattered.
    out, scatters = self._lower_scatter(mesh, PartitionSpec(("expert", "fsdp"), None), (16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), (("expert", "fsdp"), None))
    self.assertEqual(len(scatters), 1)
    self.assertIn("scatter_dimension = 0", scatters[0])
    self.assertIn("(tensor<8x6xf32>) -> tensor<2x6xf32>", scatters[0])

  def test_scatters_co_sharded_context_axis_with_fsdp(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "context"))
    # The gradient of a weight gathered over ('fsdp', 'context') holds partial sums over both axes (the
    # batch over 'fsdp', the sequence over 'context') and is reduce-scattered over both at once.
    out, scatters = self._lower_scatter(
        mesh, PartitionSpec(("fsdp", "context"), None), (16, 6), unreduced_axes={"fsdp", "context"}
    )
    self.assertEqual(tuple(out.sharding.spec.partitions), (("fsdp", "context"), None))
    self.assertEqual(out.sharding.spec.unreduced, frozenset())
    self.assertEqual(len(scatters), 1)
    self.assertIn("scatter_dimension = 0", scatters[0])
    self.assertIn("(tensor<16x6xf32>) -> tensor<2x6xf32>", scatters[0])

  def test_already_reduced_gradient_falls_back_to_resharding(self):
    """A summed (not unreduced) gradient must not be reduce-scattered again: that would scale it by the FSDP size."""
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    gathered = PartitionSpec(None, "tensor")
    g = jax.ShapeDtypeStruct((8, 6), jnp.float32, sharding=NamedSharding(mesh, gathered))
    with jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(
          lambda x: fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
              x, mesh, PartitionSpec("fsdp", "tensor"), gathered
          ),
          g,
      )
    self.assertIsNone(out)

  def test_two_scattered_dims_fall_back_to_resharding(self):
    mesh = _abstract_mesh((2, 2), ("fsdp", "fsdp_transpose"))
    out = fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
        jnp.ones((4, 4)), mesh, PartitionSpec("fsdp", "fsdp_transpose"), PartitionSpec(None, None)
    )
    self.assertIsNone(out)


class FsdpPrefetchReducedGradTest(unittest.TestCase):
  """Gather -> layer -> tagged reduce-scatter on CPU devices: the FSDP reduction happens exactly once."""

  def test_gradient_matches_reference_with_single_tagged_reduce_scatter(self):
    n = jax.device_count()
    if n < 2:
      self.skipTest("Needs at least 2 devices.")
    mesh = jax.make_mesh((2,), ("fsdp",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    src, gathered = PartitionSpec("fsdp", None), PartitionSpec(None, None)
    rng = np.random.default_rng(0)
    x_np, wd_np, wm_np = (rng.standard_normal(s).astype(np.float32) for s in ((4, 8), (8, 6), (8, 6)))

    def layer_loss(x, wd, wm):
      # A dense consumer and a nested shard_map consumer (like the MoE block) of the gathered weights.
      wm_spec = PartitionSpec(None, None, reduced=jax.typeof(wm).sharding.spec.reduced)
      m = jax.shard_map(
          lambda xs, w: jnp.tanh(xs @ w),
          mesh=mesh,
          in_specs=(PartitionSpec("fsdp", None), wm_spec),
          out_specs=PartitionSpec("fsdp", None),
      )(x, wm)
      return jnp.sin(jnp.einsum("bd,df->bf", x, wd, out_sharding=PartitionSpec("fsdp", None))).sum() + (m * m).sum()

    def prefetch_grads(x, wd, wm):
      gathered_w = [fsdp_prefetch._apply_sharding_hint(w, mesh, ShardMode.EXPLICIT) for w in (wd, wm)]  # pylint: disable=protected-access
      for w in gathered_w:
        self.assertEqual(jax.typeof(w).sharding.spec.reduced, frozenset({"fsdp"}))
      _, vjp = jax.vjp(lambda a, b: layer_loss(x, a, b), *gathered_w)
      grads = vjp(jnp.ones((), jnp.float32))
      for g in grads:
        self.assertEqual(jax.typeof(g).sharding.spec.unreduced, frozenset({"fsdp"}))
      return tuple(fsdp_prefetch._explicit_fsdp_reduce_scatter(g, mesh, src, gathered) for g in grads)  # pylint: disable=protected-access

    with jax.set_mesh(mesh):
      args = (
          jax.device_put(x_np, PartitionSpec("fsdp", None)),
          jax.device_put(wd_np, src),
          jax.device_put(wm_np, src),
      )
      f = jax.jit(prefetch_grads)
      out = f(*args)
      hlo = f.lower(*args).as_text()

    def ref_loss(wd, wm):
      m = jnp.tanh(x_np @ wm)
      return jnp.sin(x_np @ wd).sum() + (m * m).sum()

    for got, want in zip(out, jax.grad(ref_loss, argnums=(0, 1))(wd_np, wm_np)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
      self.assertEqual(tuple(got.sharding.spec.partitions), ("fsdp", None))
    # One tagged reduce-scatter per weight and no all-reduce: the gradient is reduced exactly once.
    self.assertEqual(hlo.count("stablehlo.reduce_scatter"), 2)
    self.assertNotIn("stablehlo.all_reduce", hlo)

  def test_3d_non_major_sparse_core_reduce_scatter_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs at least 4 devices.")
    mesh = jax.make_mesh(
        (2, 2), ("fsdp", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit), devices=jax.devices()[:4]
    )
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((4, 4, 128)).astype(np.float32)
    w_out_np = (rng.standard_normal((4, 128, 256)) / 128**0.5).astype(np.float32)
    src_spec = PartitionSpec(None, None, ("fsdp", "expert"))
    gathered_spec = PartitionSpec(None, None, None)

    def loss_fn(x, w):
      return jnp.sin(
          jnp.einsum(
              "bhd,hde->be",
              x,
              w,
              precision=jax.lax.Precision.HIGHEST,
              out_sharding=PartitionSpec(("fsdp", "expert"), None),
          )
      ).sum()

    def prefetch_grad(x, w):
      wg = fsdp_prefetch._apply_sharding_hint(w, mesh, ShardMode.EXPLICIT, sparse_core_id=1)  # pylint: disable=protected-access
      _, vjp_fn = jax.vjp(lambda w_: loss_fn(x, w_), wg)
      (dw,) = vjp_fn(jnp.ones((), jnp.float32))
      return fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          dw, mesh, src_spec, gathered_spec, sparse_core_id=1
      )

    with jax.set_mesh(mesh):
      x_dev = jax.device_put(x_np, PartitionSpec(("fsdp", "expert"), None, None))
      w_dev = jax.device_put(w_out_np, src_spec)
      got = jax.jit(prefetch_grad)(x_dev, w_dev)

    want = jax.grad(lambda w: jnp.sin(jnp.einsum("bhd,hde->be", x_np, w, precision=jax.lax.Precision.HIGHEST)).sum())(
        w_out_np
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    self.assertEqual(tuple(got.sharding.spec.partitions), (None, None, ("fsdp", "expert")))

  def test_explicit_fsdp_weight_and_embed_lookup_gradients_match_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs at least 4 devices.")
    mesh = jax.make_mesh(
        (2, 2), ("fsdp", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit), devices=jax.devices()[:4]
    )
    rng = np.random.default_rng(1)
    # 512 minor elements (4 * 128, not divisible by 3) pads to 768 (6 * 128) in _sc_rs_padded_minor_size.
    w_logits_np = rng.standard_normal((8, 512)).astype(np.float32)
    w_gate_np = rng.standard_normal((8, 6)).astype(np.float32)
    w_embed_np = rng.standard_normal((512, 8)).astype(np.float32)
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    ids_np = rng.integers(0, 512, size=(4, 4), dtype=np.int32)
    out_sharding = NamedSharding(mesh, PartitionSpec(("fsdp", "expert"), None, None))

    def step_loss(w_logits, wg, w_embed, x, ids):
      # 1. logits_dense: sharded on ('fsdp', 'expert') on dim 0, padded minor dim in backward RS.
      wl = fsdp_prefetch.explicit_fsdp_weight(w_logits, mesh)
      logits = jnp.einsum(
          "bd,dv->bv", x, wl, precision=jax.lax.Precision.HIGHEST, out_sharding=PartitionSpec(("fsdp", "expert"), None)
      )
      # 2. gate.kernel: `wg` is the gathered weight (reduced over 'fsdp' and, via pcast, 'expert').
      gate_out = jnp.einsum(
          "bd,de->be", x, wg, precision=jax.lax.Precision.HIGHEST, out_sharding=PartitionSpec(("fsdp", "expert"), None)
      )
      # 3. token_embedder: forward gather + backward padded scatter-add -> reduce-scatter.
      emb_out = fsdp_prefetch.explicit_fsdp_embed_lookup(w_embed, ids, out_sharding, mesh)
      return (logits * logits).sum() + (gate_out * gate_out).sum() + (emb_out * emb_out).sum()

    def step_grads(w_logits, w_gate, w_embed, x, ids):
      # As in the prefetch pipeline, differentiate w.r.t. the gathered gate weight: its gradient comes
      # back unreduced over both 'fsdp' and 'expert' and is then reduce-scattered explicitly.
      wg = fsdp_prefetch._apply_sharding_hint(w_gate, mesh, ShardMode.EXPLICIT)  # pylint: disable=protected-access
      _, vjp_fn = jax.vjp(lambda wl, g, we: step_loss(wl, g, we, x, ids), w_logits, wg, w_embed)
      dl, dg, de = vjp_fn(jnp.ones((), jnp.float32))
      dg = fsdp_prefetch._explicit_fsdp_reduce_scatter(  # pylint: disable=protected-access
          dg, mesh, PartitionSpec("fsdp", None), PartitionSpec(None, None)
      )
      return dl, dg, de

    def ref_loss(w_logits, w_gate, w_embed):
      logits = jnp.einsum("bd,dv->bv", x_np, w_logits, precision=jax.lax.Precision.HIGHEST)
      gate_out = jnp.einsum("bd,de->be", x_np, w_gate, precision=jax.lax.Precision.HIGHEST)
      emb_out = w_embed[ids_np]
      return (logits * logits).sum() + (gate_out * gate_out).sum() + (emb_out * emb_out).sum()

    with jax.set_mesh(mesh):
      wl_dev = jax.device_put(w_logits_np, PartitionSpec(("fsdp", "expert"), None))
      wg_dev = jax.device_put(w_gate_np, PartitionSpec("fsdp", None))
      we_dev = jax.device_put(w_embed_np, PartitionSpec(None, ("fsdp", "expert")))
      x_dev = jax.device_put(x_np, PartitionSpec(("fsdp", "expert"), None))
      ids_dev = jax.device_put(ids_np, PartitionSpec(("fsdp", "expert"), None))
      grad_fn = jax.jit(step_grads)
      got_grads = grad_fn(wl_dev, wg_dev, we_dev, x_dev, ids_dev)
      hlo = grad_fn.lower(wl_dev, wg_dev, we_dev, x_dev, ids_dev).as_text()

    want_grads = jax.grad(ref_loss, argnums=(0, 1, 2))(w_logits_np, w_gate_np, w_embed_np)
    for got, want in zip(got_grads, want_grads):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    self.assertEqual(tuple(got_grads[0].sharding.spec.partitions), (("fsdp", "expert"), None))
    self.assertEqual(tuple(got_grads[1].sharding.spec.partitions), ("fsdp", None))
    self.assertEqual(tuple(got_grads[2].sharding.spec.partitions), (None, ("fsdp", "expert")))
    # All three weights use reduce-scatter (with 512 -> 768 minor-dim padding for logits and embed).
    self.assertGreaterEqual(hlo.count("stablehlo.reduce_scatter"), 3)
    self.assertIn("768xf32", hlo)


class FsdpPrefetchContextParallelGradTest(unittest.TestCase):
  """Gather -> layer -> reduce-scatter on an FSDP x context mesh, with sequence-sharded activations."""

  def _grads(self, src):
    """(gradient, reference gradient, StableHLO, traced types) of a weight sharded `src` on (fsdp, context)."""
    mesh = jax.make_mesh((2, 2), ("fsdp", "context"), axis_types=(AxisType.Explicit,) * 2, devices=jax.devices()[:4])
    act = PartitionSpec("fsdp", "context", None)  # [batch, seq, dim]: batch over 'fsdp', sequence over 'context'.
    gathered = sharding.remove_fsdp_from_sharding(NamedSharding(mesh, src)).spec
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 8, 8)).astype(np.float32)
    w_np = rng.standard_normal((8, 6)).astype(np.float32)
    seen = {}

    def grads(x, w):
      wg = fsdp_prefetch._apply_sharding_hint(w, mesh, ShardMode.EXPLICIT)  # pylint: disable=protected-access
      seen["reduced"] = jax.typeof(wg).sharding.spec.reduced
      _, vjp = jax.vjp(lambda w_: jnp.sin(jnp.einsum("bsd,df->bsf", x, w_, out_sharding=act)).sum(), wg)
      (g,) = vjp(jnp.ones((), jnp.float32))
      seen["unreduced"] = jax.typeof(g).sharding.spec.unreduced
      out = fsdp_prefetch._explicit_fsdp_reduce_scatter(g, mesh, src, gathered)  # pylint: disable=protected-access
      seen["explicit_reduce_scatter"] = out is not None
      return out if out is not None else jax.sharding.reshard(g, NamedSharding(mesh, src))

    with jax.set_mesh(mesh):
      x = jax.device_put(x_np, act)
      w = jax.device_put(w_np, src)
      f = jax.jit(grads)
      got = f(x, w)
      hlo = f.lower(x, w).as_text()
    want = jax.grad(lambda w_: jnp.sin(jnp.einsum("bsd,df->bsf", x_np, w_)).sum())(w_np)
    return got, want, hlo, seen

  def test_co_sharded_weight_gradient_is_reduce_scattered_once(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    src = PartitionSpec(("fsdp", "context"), None)
    got, want, hlo, seen = self._grads(src)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    self.assertEqual(tuple(got.sharding.spec.partitions), (("fsdp", "context"), None))
    # Gathered `reduced` over both axes, so the gradient comes back `unreduced` over both and one
    # reduce-scatter sums it over the batch ('fsdp') and the sequence ('context') at once.
    self.assertEqual(seen["reduced"], frozenset({"fsdp", "context"}))
    self.assertEqual(seen["unreduced"], frozenset({"fsdp", "context"}))
    self.assertTrue(seen["explicit_reduce_scatter"])
    self.assertEqual(hlo.count("stablehlo.reduce_scatter"), 1)
    self.assertNotIn("all_reduce", hlo)

  def test_weight_replicated_over_context_gradient_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # A weight sharded over FSDP only (not co-sharded over context) still contracts over the
    # sequence-sharded context axis in backward: it is typed `reduced` over both axes, and its
    # unreduced gradient is reduce-scattered over `fsdp` and summed over `context`.
    src = PartitionSpec("fsdp", None)
    got, want, hlo, seen = self._grads(src)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    self.assertEqual(tuple(got.sharding.spec.partitions), ("fsdp", None))
    self.assertEqual(seen["reduced"], frozenset({"fsdp", "context"}))
    self.assertEqual(seen["unreduced"], frozenset({"fsdp", "context"}))
    self.assertTrue(seen["explicit_reduce_scatter"])
    self.assertEqual(hlo.count("stablehlo.reduce_scatter"), 1)
    self.assertEqual(hlo.count("stablehlo.all_reduce"), 1)

  def test_explicit_fsdp_embed_lookup_fallbacks_on_abstract_mesh(self):
    # When the embedding dimension itself is sharded (`out_spec[-1] is not None`), fall back to `.at[ids].get`.
    mesh = _abstract_mesh((2, 2), ("fsdp", "tensor"))
    w_embed = jax.ShapeDtypeStruct((64, 8), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec("fsdp", None)))
    ids = jax.ShapeDtypeStruct((4, 8), jnp.int32, sharding=NamedSharding(mesh, PartitionSpec("fsdp", None)))
    out_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None, "tensor"))
    with jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(
          lambda we, i: fsdp_prefetch.explicit_fsdp_embed_lookup(we, i, out_sharding, mesh), w_embed, ids
      )
    self.assertEqual(tuple(out.sharding.spec.partitions), ("fsdp", None, "tensor"))

  def test_explicit_fsdp_embed_lookup_with_context_parallelism(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # Include a size-1 `data` axis on batch so `batch_axes` filters size-1 axes while keeping `fsdp` and `context`.
    mesh = jax.make_mesh(
        (1, 2, 2), ("data", "fsdp", "context"), axis_types=(AxisType.Explicit,) * 3, devices=jax.devices()[:4]
    )
    rng = np.random.default_rng(2)
    w_embed_np = rng.standard_normal((512, 8)).astype(np.float32)
    ids_np = rng.integers(0, 512, size=(4, 8), dtype=np.int32)
    out_sharding = NamedSharding(mesh, PartitionSpec(("data", "fsdp"), "context", None))

    def loss(w_embed, ids):
      emb = fsdp_prefetch.explicit_fsdp_embed_lookup(w_embed, ids, out_sharding, mesh)
      return (emb * emb).sum()

    want = jax.grad(lambda we: (we[ids_np] * we[ids_np]).sum())(w_embed_np)
    for emb_spec, ids_spec, expected_all_reduces in (
        (PartitionSpec(None, ("fsdp", "context")), PartitionSpec(("data", "fsdp"), "context"), 0),
        (PartitionSpec(None, ("fsdp", "context")), PartitionSpec("fsdp", None), 0),
        (PartitionSpec(None, "fsdp"), PartitionSpec(("data", "fsdp"), "context"), 1),
    ):
      with self.subTest(emb_spec=emb_spec, ids_spec=ids_spec):
        with jax.set_mesh(mesh):
          we_dev = jax.device_put(w_embed_np, emb_spec)
          ids_dev = jax.device_put(ids_np, ids_spec)
          grad_fn = jax.jit(jax.grad(loss))
          got = grad_fn(we_dev, ids_dev)
          hlo = grad_fn.lower(we_dev, ids_dev).as_text()
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
        self.assertEqual(tuple(got.sharding.spec.partitions), tuple(emb_spec))
        self.assertEqual(hlo.count("stablehlo.reduce_scatter"), 1)
        self.assertEqual(hlo.count("stablehlo.all_reduce"), expected_all_reduces)


class _ToyLayer(nnx.Module):
  """tanh(tanh(x @ kernel)), naming its input `decoder_layer_input` and its hidden activation `mlpwi`."""

  def __init__(self, dim, rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)) / dim**0.5)

  def __call__(self, x):
    x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
    h = ad_checkpoint.checkpoint_name(jnp.tanh(x @ self.kernel[...]), "mlpwi")
    return jnp.tanh(h), None


class FsdpPrefetchLayerInputOffloadTest(unittest.TestCase):
  """`decoder_layer_input: offload` in the prefetch pipeline: inputs saved in host memory, same gradients."""

  def _grads(self, decoder_layer_input, length=5):
    """(pipeline gradients, reference gradients, jaxpr text) for a stack of `length` toy layers."""
    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:4])
    dim, batch = 8, 8
    layers = [_ToyLayer(dim, nnx.Rngs(params=i)) for i in range(length)]
    graphdef = nnx.graphdef(layers[0])
    params = jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(l, nnx.Param) for l in layers])
    x_np = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=ShardMode.EXPLICIT,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        decoder_layer_input=decoder_layer_input,
    )
    # As remat_policy=custom builds it: `decoder_layer_input` saved or offloaded, `mlpwi` saved.
    offload = decoder_layer_input == "offload"
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["mlpwi"] + ([] if offload else ["decoder_layer_input"]),
        names_which_can_be_offloaded=["decoder_layer_input"] if offload else [],
        offload_src="device",
        offload_dst="pinned_host",
    )

    def loss(p, x):
      y, _, _ = fsdp_prefetch.apply_layers_with_fsdp_prefetch(
          nnx.merge(graphdef, p), x, length=length, mesh=mesh, config=config, remat_policy=policy
      )
      return (y * y).sum()

    def ref_loss(p, x):
      for i in range(length):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, i=i: v[i], p))(x)
      return (x * x).sum()

    with jax.set_mesh(mesh):
      p_sharded = jax.tree.map(lambda v: jax.device_put(v, PartitionSpec(None, "fsdp", None)), params)
      x = jax.device_put(x_np, PartitionSpec("fsdp", None))
      f = jax.jit(jax.grad(loss))
      grads = f(p_sharded, x)
      # The jaxpr, since CPU lowering does not keep host memory placement (device and host memory are the same).
      jaxpr = str(jax.make_jaxpr(jax.grad(loss))(p_sharded, x))
    return grads, jax.grad(ref_loss)(params, x_np), jaxpr

  def test_offloaded_inputs_match_reference_and_on_device(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # 6 layers: the middle layers' loop (4 layers, unroll=2) has 2 trips, so their inputs are offloaded.
    got, ref, jaxpr = self._grads("offload", length=6)
    for g, r in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
      np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)
    on_device, _, device_jaxpr = self._grads("device", length=6)
    for g, d in zip(jax.tree.leaves(got), jax.tree.leaves(on_device)):
      np.testing.assert_allclose(np.asarray(g), np.asarray(d), rtol=1e-6, atol=1e-6)
    # 3 device_puts: the host buffer for the middle layers' inputs, each input's move into it (one
    # dynamic_update_slice per layer in the forward loop body) and its move back (backward loop body).
    # The per-layer remat does not offload a second copy of the input (which would add 2 per remat site).
    self.assertEqual(len(re.findall(r"= device_put\b", jaxpr)), 3, jaxpr)
    self.assertNotRegex(device_jaxpr, r"= device_put\b")

  def test_single_trip_loop_keeps_inputs_on_device(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # 5 layers: the middle layers' loop has 1 trip (XLA unrolls it into the parent computation, where
    # host copies break the FIFO ordering of FSDP collectives), so nothing is offloaded.
    got, ref, jaxpr = self._grads("offload", length=5)
    for g, r in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
      np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)
    self.assertNotRegex(jaxpr, r"= device_put\b")


class FsdpPrefetchContextParallelPipelineTest(unittest.TestCase):
  """The prefetch pipeline on an FSDP x context mesh (sequence-sharded activations) matches a plain loop."""

  def _grads(self, length, shard_mode, defer=False, remat_policy=None, decoder_layer_input="device"):
    """(pipeline gradients, reference gradients) for a stack of `length` toy layers on a 2 x 2 mesh."""
    axis_type = AxisType.Explicit if shard_mode == ShardMode.EXPLICIT else AxisType.Auto
    mesh = jax.make_mesh((2, 2), ("fsdp", "context"), axis_types=(axis_type,) * 2, devices=jax.devices()[:4])
    dim, batch, seq = 8, 4, 8
    layers = [_ToyLayer(dim, nnx.Rngs(params=i)) for i in range(length)]
    graphdef = nnx.graphdef(layers[0])
    params = jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(l, nnx.Param) for l in layers])
    x_np = np.random.default_rng(0).standard_normal((batch, seq, dim)).astype(np.float32)
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=shard_mode,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=defer,
        decoder_layer_input=decoder_layer_input,
    )

    def loss(p, x):
      y, _, _ = fsdp_prefetch.apply_layers_with_fsdp_prefetch(
          nnx.merge(graphdef, p), x, length=length, mesh=mesh, config=config, remat_policy=remat_policy
      )
      return (y * y).sum()

    def ref_loss(p, x):
      for i in range(length):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, i=i: v[i], p))(x)
      return (x * x).sum()

    with jax.set_mesh(mesh):
      # Weights co-sharded over ('fsdp', 'context') as under context parallelism; the batch is
      # sharded over 'fsdp' and the sequence over 'context'.
      w_sharding = NamedSharding(mesh, PartitionSpec(None, ("fsdp", "context"), None))
      p_sharded = jax.tree.map(lambda v: jax.device_put(v, w_sharding), params)
      x = jax.device_put(x_np, NamedSharding(mesh, PartitionSpec("fsdp", "context", None)))
      grads = jax.jit(jax.grad(loss))(p_sharded, x)
    return grads, jax.grad(ref_loss)(params, x_np)

  def _assert_parity(self, length, shard_mode, **kwargs):
    got, ref = self._grads(length, shard_mode, **kwargs)
    for g, r in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
      np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)
      if shard_mode == ShardMode.EXPLICIT:
        # The reduce-scattered gradient keeps the weight's ('fsdp', 'context') sharding.
        self.assertEqual(tuple(g.sharding.spec.partitions), (None, ("fsdp", "context"), None))

  def test_matches_reference_across_layer_counts_and_shard_modes(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # 1 layer: no loop; 3 layers: single-trip loops; 6 layers: two-trip unroll=2 loops in both passes.
    for length in (1, 3, 6):
      for shard_mode, defer in ((ShardMode.EXPLICIT, False), (ShardMode.EXPLICIT, True), (ShardMode.AUTO, False)):
        with self.subTest(length=length, shard_mode=shard_mode, defer=defer):
          self._assert_parity(length, shard_mode, defer=defer)

  def test_matches_reference_with_saved_residuals(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # The remat policy saves the sequence-sharded hidden activation of every layer.
    policy = jax.checkpoint_policies.save_only_these_names("mlpwi")
    self._assert_parity(6, ShardMode.EXPLICIT, defer=True, remat_policy=policy)

  def test_matches_reference_with_offloaded_layer_inputs(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["mlpwi"],
        names_which_can_be_offloaded=["decoder_layer_input"],
        offload_src="device",
        offload_dst="pinned_host",
    )
    self._assert_parity(6, ShardMode.EXPLICIT, defer=True, remat_policy=policy, decoder_layer_input="offload")


class _ToySubLayer(nnx.Module):
  """tanh(x @ kernel) (+ bias): a sub-layer of `_ToyBlock`."""

  def __init__(self, dim, rngs, bias=False):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)) / dim**0.5)
    self.has_bias = bias
    self.bias = nnx.Param(0.1 * jax.random.normal(rngs.params(), (dim,)) if bias else jnp.zeros((0,)))

  def __call__(self, x):
    x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
    y = ad_checkpoint.checkpoint_name(jnp.tanh(x @ self.kernel[...]), "mlpwi")
    return (y + self.bias[...] if self.has_bias else y), None


class _ToyBlock(nnx.Module):
  """An inhomogeneous block: `num_a` identical sub-layers, then one with a bias (2 sub-layer groups)."""

  def __init__(self, num_a, dim, rngs):
    self.sub_names = tuple(f"a{i}" for i in range(num_a)) + ("b",)
    for i in range(num_a):
      setattr(self, f"a{i}", _ToySubLayer(dim, rngs))
    self.b = _ToySubLayer(dim, rngs, bias=True)

  def prefetch_sublayer_names(self):
    return self.sub_names

  def __call__(self, x):
    for name in self.sub_names:
      x, _ = getattr(self, name)(x)
    return x, None


class FsdpPrefetchInhomogeneousDeferredTest(unittest.TestCase):
  """Per-sub-layer prefetch of inhomogeneous blocks with deferred dW reduce-scatters on CPU devices."""

  def _grads(self, num_blocks, num_a, defer, decoder_layer_input="device"):
    """(gradients of the prefetch pipeline, reference gradients, lowered HLO) for a toy block stack."""
    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:4])
    dim, batch = 8, 8
    blocks = [_ToyBlock(num_a, dim, nnx.Rngs(params=b)) for b in range(num_blocks)]
    graphdef = nnx.graphdef(blocks[0])
    params = jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(b, nnx.Param) for b in blocks])
    x_np = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=ShardMode.EXPLICIT,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=defer,
        decoder_layer_input=decoder_layer_input,
    )
    offload = decoder_layer_input == "offload"
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["mlpwi"] + ([] if offload else ["decoder_layer_input"]),
        names_which_can_be_offloaded=["decoder_layer_input"] if offload else [],
        offload_src="device",
        offload_dst="pinned_host",
    )

    def loss(p, x):
      layers = nnx.merge(graphdef, p)
      stages = fsdp_prefetch_inhomogeneous.sublayer_stages(layers)
      y, _, _ = fsdp_prefetch_inhomogeneous.apply_layers_with_fsdp_prefetch(
          layers, stages, x, length=num_blocks, mesh=mesh, config=config, remat_policy=policy
      )
      return (y * y).sum()

    def ref_loss(p, x):
      for b in range(num_blocks):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, b=b: v[b], p))(x)
      return (x * x).sum()

    with jax.set_mesh(mesh):
      # Kernels are FSDP-sharded on their input dimension; biases are replicated (untagged gradients).
      p_sharded = jax.tree.map(
          lambda v: jax.device_put(v, PartitionSpec(None, "fsdp", None) if v.ndim == 3 else PartitionSpec()), params
      )
      x = jax.device_put(x_np, PartitionSpec("fsdp", None))
      f = jax.jit(jax.grad(loss))
      grads = f(p_sharded, x)
      hlo = f.lower(p_sharded, x).as_text()
    return grads, jax.grad(ref_loss)(params, x_np), hlo

  def _assert_matches_reference(self, num_blocks, num_a):
    """Asserts deferred gradients match an unsharded reference and the non-deferred pipeline."""
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    grads, ref, hlo = self._grads(num_blocks, num_a, defer=True)
    for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(ref)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    # The kernel gradients are reduce-scattered by the deferred reduce-scatter.
    self.assertIn("stablehlo.reduce_scatter", hlo)
    # Deferral only reorders the reduce-scatters: same gradients as without it.
    no_defer, _, _ = self._grads(num_blocks, num_a, defer=False)
    for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(no_defer)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6)

  def test_three_sublayer_group(self):
    # Groups [a0, a1, a2] (entry, same-group and exit steps) and [b] (entry is exit).
    self._assert_matches_reference(num_blocks=3, num_a=3)

  def test_one_block(self):
    # The first block step's placeholder flush targets the sub-layer whose deferred dW is flushed last.
    self._assert_matches_reference(num_blocks=1, num_a=3)

  def test_group_sizes(self):
    # 1 (entry is exit), 2 (entry and exit) and 5 (unrolled same-group steps) identical sub-layers.
    for num_a in (1, 2, 5):
      with self.subTest(num_a=num_a):
        self._assert_matches_reference(num_blocks=2, num_a=num_a)

  def test_offloaded_sublayer_inputs_match_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    for defer in (False, True):
      with self.subTest(defer=defer):
        got, ref, _ = self._grads(num_blocks=2, num_a=3, defer=defer, decoder_layer_input="offload")
        for g, r in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
          np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)
        on_device, _, _ = self._grads(num_blocks=2, num_a=3, defer=defer, decoder_layer_input="device")
        for g, d in zip(jax.tree.leaves(got), jax.tree.leaves(on_device)):
          np.testing.assert_allclose(np.asarray(g), np.asarray(d), rtol=1e-6, atol=1e-6)


class FsdpPrefetchInhomogeneousContextParallelTest(unittest.TestCase):
  """Per-sub-layer prefetch of inhomogeneous blocks on an FSDP x context mesh matches a plain loop."""

  def _grads(self, kernel_spec, defer):
    """(pipeline gradients, reference gradients) for 2 toy blocks with sequence-sharded activations."""
    mesh = jax.make_mesh((2, 2), ("fsdp", "context"), axis_types=(AxisType.Explicit,) * 2, devices=jax.devices()[:4])
    num_blocks, num_a, dim, batch, seq = 2, 3, 8, 4, 8
    blocks = [_ToyBlock(num_a, dim, nnx.Rngs(params=b)) for b in range(num_blocks)]
    graphdef = nnx.graphdef(blocks[0])
    params = jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(b, nnx.Param) for b in blocks])
    x_np = np.random.default_rng(0).standard_normal((batch, seq, dim)).astype(np.float32)
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=ShardMode.EXPLICIT,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=defer,
        decoder_layer_input="device",
    )
    policy = jax.checkpoint_policies.save_only_these_names("mlpwi", "decoder_layer_input")

    def loss(p, x):
      layers = nnx.merge(graphdef, p)
      stages = fsdp_prefetch_inhomogeneous.sublayer_stages(layers)
      y, _, _ = fsdp_prefetch_inhomogeneous.apply_layers_with_fsdp_prefetch(
          layers, stages, x, length=num_blocks, mesh=mesh, config=config, remat_policy=policy
      )
      return (y * y).sum()

    def ref_loss(p, x):
      for b in range(num_blocks):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, b=b: v[b], p))(x)
      return (x * x).sum()

    with jax.set_mesh(mesh):
      # Batch over 'fsdp' and sequence over 'context'; kernels sharded on their input dimension.
      p_sharded = jax.tree.map(lambda v: jax.device_put(v, kernel_spec if v.ndim == 3 else PartitionSpec()), params)
      x = jax.device_put(x_np, PartitionSpec("fsdp", "context", None))
      grads = jax.jit(jax.grad(loss))(p_sharded, x)
    return grads, jax.grad(ref_loss)(params, x_np)

  def test_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    # Kernels co-sharded over ('fsdp', 'context'), and sharded over 'fsdp' only (gradients are then
    # reduce-scattered over 'fsdp' and summed over 'context').
    for kernel_spec in (PartitionSpec(None, ("fsdp", "context"), None), PartitionSpec(None, "fsdp", None)):
      for defer in (False, True):
        with self.subTest(kernel_spec=kernel_spec, defer=defer):
          got, ref = self._grads(kernel_spec, defer)
          for g, r in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
            np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)


class _ToyQuantLayer(nnx.Module):
  """Two-projection layer `(x @ w1) @ w2` with non-square weights for Qwix interception tests."""

  def __init__(self, dim, hidden_dim, rngs):
    self.w1 = nnx.Param(jax.random.normal(rngs.params(), (dim, hidden_dim)) / dim**0.5)
    self.w2 = nnx.Param(jax.random.normal(rngs.params(), (hidden_dim, dim)) / hidden_dim**0.5)

  def __call__(self, x):
    x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
    h = ad_checkpoint.checkpoint_name(jnp.tanh(jnp.einsum("bd,dh->bh", x, self.w1[...])), "mlpwi")
    return jnp.einsum("bh,hd->bd", h, self.w2[...]), None


class _ToyQuantModel(nnx.Module):
  """Root module wrapped by `qwix.quantize_model`, which deactivates interception when `__call__` returns."""

  def __init__(self, length, dim, hidden_dim):
    @nnx.split_rngs(splits=length)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def make_layers(rngs):
      return _ToyQuantLayer(dim, hidden_dim, rngs)

    self.layers = make_layers(nnx.Rngs(params=0))
    self.length = length

  def __call__(self, x, *, mesh=None, config=None, remat_policy=None, prefetch=True):
    if not prefetch or mesh is None:
      graphdef, params, state = nnx.split(self.layers, nnx.Param, ...)
      for i in range(self.length):
        layer = nnx.merge(
            graphdef,
            jax.tree.map(lambda v, i=i: v[i], params),
            jax.tree.map(lambda v, i=i: v[i], state),
        )
        x, _ = layer(x)
      return x
    y, _, _ = fsdp_prefetch.apply_layers_with_fsdp_prefetch(
        self.layers, x, length=self.length, mesh=mesh, config=config, remat_policy=remat_policy
    )
    return y


class FsdpPrefetchQwixQuantizationTest(unittest.TestCase):
  """`custom_vjp` backward pass preserves Qwix quantization traced during the root module's `__call__`."""

  def test_rejects_explicit_sharding_when_qwix_lacks_explicit_sharding_support(self):
    import qwix  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.configs import types as config_types  # pylint: disable=import-outside-toplevel

    mesh = _abstract_mesh((4,), ("fsdp",))
    length, batch, dim, hidden_dim = 2, 4, 8, 16
    base_model = _ToyQuantModel(length, dim, hidden_dim)
    rules = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
        )
    ]
    q_model = qwix.quantize_model(base_model, qwix.QtProvider(rules), jnp.ones((batch, dim)))
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=ShardMode.EXPLICIT,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        decoder_layer_input="device",
    )
    with unittest.mock.patch.object(fsdp_prefetch, "qwix_supports_explicit_sharding", return_value=False):
      with self.assertRaisesRegex(ValueError, "google/qwix#455"):
        q_model(jnp.ones((batch, dim)), mesh=mesh, config=config, remat_policy=None, prefetch=True)

    with unittest.mock.patch.object(config_types, "_qwix_supports_explicit_sharding", return_value=False):
      with self.assertRaisesRegex(ValueError, "google/qwix#455"):
        pyconfig.initialize(
            [
                "",
                "base_output_directory=gs://unused",
                "run_name=test",
                "prefetch_fsdp_weights=True",
                "shard_mode=explicit",
                "quantization=fp8_full",
                "use_qwix_quantization=True",
            ]
        )

  def test_backward_pass_preserves_qwix_fp8_quantization_and_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    if not fsdp_prefetch.qwix_supports_explicit_sharding():
      self.skipTest("Requires qwix with explicit sharding support (google/qwix#455).")
    import qwix  # pylint: disable=import-outside-toplevel

    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:4])
    length, batch, dim, hidden_dim = 4, 8, 16, 32
    base_model = _ToyQuantModel(length, dim, hidden_dim)
    rules = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
        )
    ]
    q_model = qwix.quantize_model(base_model, qwix.QtProvider(rules), jnp.ones((batch, dim)))
    graphdef, params, state = nnx.split(q_model, nnx.Param, ...)
    x_np = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)
    config = types.SimpleNamespace(
        param_scan_axis=0,
        shard_mode=ShardMode.EXPLICIT,
        debug_sharding=False,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        decoder_layer_input="device",
    )
    custom_policy = jax.checkpoint_policies.save_only_these_names("mlpwi", "decoder_layer_input")

    for policy in (None, custom_policy):
      with self.subTest(remat_policy="full" if policy is None else "custom"):

        def loss(p, x, policy=policy):
          m = nnx.merge(graphdef, p, state)
          y = m(x, mesh=mesh, config=config, remat_policy=policy, prefetch=True)
          return (y * y).sum()

        def ref_loss(p, x):
          m = nnx.merge(graphdef, p, state)
          y = m(x, mesh=mesh, config=config, remat_policy=None, prefetch=False)
          return (y * y).sum()

        with jax.set_mesh(mesh):
          p_sharded = jax.tree.map(lambda v: jax.device_put(v, PartitionSpec(None, "fsdp", None)), params)
          x = jax.device_put(x_np, PartitionSpec("fsdp", None))
          grads = jax.jit(jax.grad(loss))(p_sharded, x)
          ref_grads = jax.jit(jax.grad(ref_loss))(p_sharded, x)
          grad_jaxpr = str(jax.make_jaxpr(jax.grad(loss))(p_sharded, x))

        # The backward pass must use float8_e5m2 gradients (proving `_bwd` reused the forward-traced VJP layout)
        # and match the quantized non-prefetched reference gradients.
        self.assertIn("float8_e5m2", grad_jaxpr)
        for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads)):
          np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
