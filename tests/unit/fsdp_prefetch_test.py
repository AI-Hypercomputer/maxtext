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

import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch
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


if __name__ == "__main__":
  unittest.main()
