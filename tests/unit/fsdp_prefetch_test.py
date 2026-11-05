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

import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from flax import nnx
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch
from maxtext.layers import fsdp_prefetch_inhomogeneous
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

  def test_metadata_layer_spec_and_param_layer_specs(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    rules = (("embed", "fsdp"), ("mlp", "tensor"), ("layers", None))
    var = nnx.Param(
        jnp.ones((8, 3, 4)),
        out_sharding=("embed", "layers", "mlp"),
        eager_sharding=False,
        **{nnx.PARTITION_NAME: "layers"},
    )
    no_meta = nnx.Param(jnp.ones((3, 4)))
    # pylint: disable=protected-access
    spec = fsdp_prefetch._metadata_layer_spec(var, mesh, rules)
    self.assertEqual(tuple(spec), ("fsdp", "tensor"))
    self.assertIsNone(fsdp_prefetch._metadata_layer_spec(no_meta, mesh, rules))

    tree = (var, no_meta)
    auto_cfg = types.SimpleNamespace(shard_mode=ShardMode.AUTO, logical_axis_rules=rules)
    explicit_cfg = types.SimpleNamespace(shard_mode=ShardMode.EXPLICIT, logical_axis_rules=rules)
    specs = fsdp_prefetch._param_layer_specs(tree, 2, mesh, auto_cfg)
    self.assertEqual(tuple(specs[0]), ("fsdp", "tensor"))
    self.assertIsNone(specs[1])
    self.assertIsNone(fsdp_prefetch._param_layer_specs(tree, 2, mesh, explicit_cfg))


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
    # The tag that lets XLA's FSDP scheduler run the gather on a dedicated SparseCore.
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

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
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

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
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

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
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

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
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

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

  def _lower_scatter(self, mesh, src_spec, shape, unreduced=True):
    """Returns (output ShapeDtypeStruct, reduce_scatter StableHLO lines) for a gradient of a `src_spec` weight.

    With `unreduced`, the gradient is typed as the partial sums of a `reduced` gathered weight.
    """
    gathered = sharding.remove_fsdp_from_sharding(NamedSharding(mesh, src_spec)).spec
    fsdp_axes = frozenset(a for a in mesh.axis_names if a.startswith("fsdp") and mesh.shape[a] > 1)
    g_spec = PartitionSpec(*gathered, unreduced=fsdp_axes) if unreduced else gathered
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
    self.assertIn('xla_explicit_fsdp = "yes"', scatters[0])
    self.assertIn("(tensor<8x3xf32>) -> tensor<2x3xf32>", scatters[0])

  def test_flat_expert_weight_scatters_only_fsdp_of_compound_dim(self):
    mesh = _abstract_mesh((2, 4), ("expert", "fsdp"))
    # moe_flat_fsdp_weights: dim 0 is sharded over ('expert', 'fsdp'); only 'fsdp' is reduce-scattered.
    out, scatters = self._lower_scatter(mesh, PartitionSpec(("expert", "fsdp"), None), (16, 6))
    self.assertEqual(tuple(out.sharding.spec.partitions), (("expert", "fsdp"), None))
    self.assertEqual(len(scatters), 1)
    self.assertIn("scatter_dimension = 0", scatters[0])
    self.assertIn("(tensor<8x6xf32>) -> tensor<2x6xf32>", scatters[0])
    self.assertIn('xla_explicit_fsdp = "yes"', scatters[0])

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
    # The kernel gradients are reduce-scattered by the tagged (deferred) reduce-scatter.
    self.assertIn("stablehlo.reduce_scatter", hlo)
    self.assertIn('xla_explicit_fsdp = "yes"', hlo)
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


class FsdpPrefetchAutoModeTest(unittest.TestCase):
  """Homogeneous and inhomogeneous prefetch under `ShardMode.AUTO` recover FSDP axes from NNX metadata."""

  def _annotate_kernels(self, params):
    def _tag(v):
      if not isinstance(v, nnx.Variable) or v.get_value().ndim != 3:
        return v
      return v.replace(
          out_sharding=("layers", "embed", "mlp"),
          sharding_names=("layers", "embed", "mlp"),
          **{nnx.PARTITION_NAME: "layers"},
      )

    return jax.tree.map(_tag, params, is_leaf=lambda x: isinstance(x, nnx.Variable))

  def test_homogeneous_auto_mode_emits_tagged_all_gather_and_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Auto,), devices=jax.devices()[:4])
    dim, batch, length = 8, 8, 6
    layers = [_ToyLayer(dim, nnx.Rngs(params=i)) for i in range(length)]
    graphdef = nnx.graphdef(layers[0])
    params = self._annotate_kernels(
        jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(l, nnx.Param) for l in layers])
    )
    x_np = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)

    def ref_loss(p, x):
      for i in range(length):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, i=i: v[i], p))(x)
      return (x * x).sum()

    ref_grads = jax.grad(ref_loss)(params, x_np)
    for decoder_layer_input in ("device", "offload"):
      with self.subTest(decoder_layer_input=decoder_layer_input):
        config = types.SimpleNamespace(
            param_scan_axis=0,
            shard_mode=ShardMode.AUTO,
            debug_sharding=False,
            prefetch_fsdp_defer_grad_reduce_scatter=True,
            decoder_layer_input=decoder_layer_input,
            logical_axis_rules=(("embed", "fsdp"), ("mlp", None), ("layers", None)),
        )

        def loss(p, x, cfg=config):
          y, _, _ = fsdp_prefetch.apply_layers_with_fsdp_prefetch(
              nnx.merge(graphdef, p), x, length=length, mesh=mesh, config=cfg
          )
          return (y * y).sum()

        with jax.set_mesh(mesh):
          p_sharded = jax.tree.map(lambda v: jax.device_put(v, PartitionSpec(None, "fsdp", None)), params)
          x = jax.device_put(x_np, PartitionSpec("fsdp", None))
          f = jax.jit(jax.grad(loss))
          grads = f(p_sharded, x)
          hlo = f.lower(p_sharded, x).as_text()

        for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads)):
          np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
        self.assertIn("stablehlo.all_gather", hlo)
        self.assertIn('xla_explicit_fsdp = "yes"', hlo)

  def test_inhomogeneous_auto_mode_emits_tagged_all_gather_and_matches_reference(self):
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")
    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Auto,), devices=jax.devices()[:4])
    dim, batch, num_blocks, num_a = 8, 8, 2, 3
    blocks = [_ToyBlock(num_a, dim, nnx.Rngs(params=b)) for b in range(num_blocks)]
    graphdef = nnx.graphdef(blocks[0])
    params = self._annotate_kernels(
        jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.state(b, nnx.Param) for b in blocks])
    )
    x_np = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)

    def ref_loss(p, x):
      for b in range(num_blocks):
        x, _ = nnx.merge(graphdef, jax.tree.map(lambda v, b=b: v[b], p))(x)
      return (x * x).sum()

    ref_grads = jax.grad(ref_loss)(params, x_np)
    for decoder_layer_input in ("device", "offload"):
      with self.subTest(decoder_layer_input=decoder_layer_input):
        config = types.SimpleNamespace(
            param_scan_axis=0,
            shard_mode=ShardMode.AUTO,
            debug_sharding=False,
            prefetch_fsdp_defer_grad_reduce_scatter=True,
            decoder_layer_input=decoder_layer_input,
            logical_axis_rules=(("embed", "fsdp"), ("mlp", None), ("layers", None)),
        )

        def loss(p, x, cfg=config):
          layers = nnx.merge(graphdef, p)
          stages = fsdp_prefetch_inhomogeneous.sublayer_stages(layers)
          y, _, _ = fsdp_prefetch_inhomogeneous.apply_layers_with_fsdp_prefetch(
              layers, stages, x, length=num_blocks, mesh=mesh, config=cfg
          )
          return (y * y).sum()

        with jax.set_mesh(mesh):
          p_sharded = jax.tree.map(
              lambda v: jax.device_put(v, PartitionSpec(None, "fsdp", None) if v.ndim == 3 else PartitionSpec()), params
          )
          x = jax.device_put(x_np, PartitionSpec("fsdp", None))
          f = jax.jit(jax.grad(loss))
          grads = f(p_sharded, x)
          hlo = f.lower(p_sharded, x).as_text()

        for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads)):
          np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
        self.assertIn("stablehlo.all_gather", hlo)
        self.assertIn('xla_explicit_fsdp = "yes"', hlo)


if __name__ == "__main__":
  unittest.main()
