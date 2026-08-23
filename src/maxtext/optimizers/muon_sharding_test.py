# Copyright 2023–2026 Google LLC
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

"""Integration tests for sharded Muon optimizer."""

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

jax.config.update("jax_num_cpu_devices", 8)

import numpy as np
from optax.contrib import _muon as optax_muon

from maxtext.optimizers import muon as _muon
from maxtext.optimizers import reshape_utils


class MuonShardingTest(parameterized.TestCase):

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_scale_by_muon_matches_optax_contrib(self, axis_type):
    """Tests that scale_by_muon matches optax.contrib.scale_by_muon on sharded batch array."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None, None))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (4, 8, 16)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (4, 8, 16)), sharding)

    params = {"w": w}
    grads = {"w": g}

    local_dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)}
    optax_dim_nums = {"w": optax_muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2)}

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers=local_dim_nums,
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers=optax_dim_nums)

    local_state = local_transform.init(params)
    optax_state = optax_transform.init(params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(grads, optax_state, params)

    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, optax_updates["w"].sharding)
    else:
      self.assertEqual(local_updates["w"].shape, w.shape)
      self.assertFalse(jnp.isnan(local_updates["w"]).any())

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_2d_tensor_reduction_and_output_axes(self, axis_type):
    """Tests that scale_by_muon handles 2D tensors with sharded reduction and output axes without all-to-all."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y"))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (8, 16)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (8, 16)), sharding)

    params = {"w": w}
    grads = {"w": g}

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": _muon.MuonDimensionNumbers(sharding=sharding)},
    )
    local_state = local_transform.init(params)
    local_updates, _ = local_transform.update(grads, local_state, params)

    self.assertEqual(local_updates["w"].shape, (8, 16))
    self.assertFalse(jnp.isnan(local_updates["w"]).any())
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, sharding)

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_3d_unsharded_batch_with_sharded_matrix_axes(self, axis_type):
    """Tests 3D tensor (4, 8, 16) where batch axis 0 is unsharded and matrix axes (1, 2) are sharded,
    triggering all-to-all.
    """
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x", "y"))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (4, 8, 16)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (4, 8, 16)), sharding)

    params = {"w": w}
    grads = {"w": g}

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
    optax_dim_num = optax_muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2)

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": dim_num},
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_dim_num})

    local_state = local_transform.init(params)
    unsharded_params = jax.device_get(params)
    unsharded_grads = jax.device_get(grads)
    optax_state = optax_transform.init(unsharded_params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(unsharded_grads, optax_state, unsharded_params)

    self.assertEqual(local_updates["w"].shape, (4, 8, 16))
    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, sharding)

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_3d_unsharded_batch_with_sharded_matrix_axes_without_all_to_all(self, axis_type):
    """Tests 3D tensor where batch axis is unsharded and matrix axes are sharded, without all-to-all."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x", "y"))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (4, 8, 16)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (4, 8, 16)), sharding)

    params = {"w": w}
    grads = {"w": g}

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
    optax_dim_num = optax_muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2)

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": dim_num},
        use_all_to_all=False,
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_dim_num})

    local_state = local_transform.init(params)
    unsharded_params = jax.device_get(params)
    unsharded_grads = jax.device_get(grads)
    optax_state = optax_transform.init(unsharded_params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(unsharded_grads, optax_state, unsharded_params)

    self.assertEqual(local_updates["w"].shape, (4, 8, 16))
    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, sharding)

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_3d_batch_axis(self, axis_type):
    """Tests 3D tensor (8, 16, 32) where batch axis 0 is sharded across mesh ('x', 'y') and matrix axes are unsharded."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("x", "y"), None, None))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (8, 16, 32)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (8, 16, 32)), sharding)

    params = {"w": w}
    grads = {"w": g}

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
    optax_dim_num = optax_muon.MuonDimensionNumbers(reduction_axis=1, output_axis=2)

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": dim_num},
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_dim_num})

    local_state = local_transform.init(params)
    optax_state = optax_transform.init(params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(grads, optax_state, params)

    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, optax_updates["w"].sharding)
    else:
      self.assertEqual(local_updates["w"].shape, w.shape)
      self.assertFalse(jnp.isnan(local_updates["w"]).any())

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_4d_mixed_batch_axes(self, axis_type):
    """Tests 4D tensor (2, 4, 8, 16) with mixed batch axes: axis 0 sharded ('x'), axis 1 unsharded (None),
    axis 2 sharded ('y').
    """
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None, "y", None))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (2, 4, 8, 16)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (2, 4, 8, 16)), sharding)

    params = {"w": w}
    grads = {"w": g}

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
    optax_dim_num = optax_muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3)

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": dim_num},
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_dim_num})

    local_state = local_transform.init(params)
    unsharded_params = jax.device_get(params)
    unsharded_grads = jax.device_get(grads)
    optax_state = optax_transform.init(unsharded_params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(unsharded_grads, optax_state, unsharded_params)

    self.assertEqual(local_updates["w"].shape, (2, 4, 8, 16))
    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, sharding)

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_4d_all_sharded_batch_axes(self, axis_type):
    """Tests 4D tensor (2, 4, 16, 32) where both batch axes (0, 1) are sharded across mesh ('x', 'y')."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y", None, None))

    key = jax.random.key(42)
    key_w, key_g = jax.random.split(key)
    w = jax.device_put(jax.random.normal(key_w, (2, 4, 16, 32)), sharding)
    g = jax.device_put(jax.random.normal(key_g, (2, 4, 16, 32)), sharding)

    params = {"w": w}
    grads = {"w": g}

    dim_num = _muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
    optax_dim_num = optax_muon.MuonDimensionNumbers(reduction_axis=2, output_axis=3)

    local_transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": dim_num},
    )
    optax_transform = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_dim_num})

    local_state = local_transform.init(params)
    optax_state = optax_transform.init(params)

    local_updates, _ = local_transform.update(grads, local_state, params)
    optax_updates, _ = optax_transform.update(grads, optax_state, params)

    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, optax_updates["w"].sharding)
    else:
      self.assertEqual(local_updates["w"].shape, w.shape)
      self.assertFalse(jnp.isnan(local_updates["w"]).any())

  def test_mixed_mesh_axis_types_raises_error(self):
    """Tests that a mesh with mixed (Explicit and Auto) axis types raises a ValueError."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y"))
    w = jax.device_put(jax.random.normal(jax.random.key(0), (8, 16)), sharding)
    params = {"w": w}

    transform = _muon.scale_by_muon(
        weight_dimension_numbers={"w": _muon.MuonDimensionNumbers(sharding=sharding)},
    )
    state = transform.init(params)
    with self.assertRaisesRegex(ValueError, "Mixed mesh axis types"):
      transform.update(params, state, params)

  def test_2d_tensor_reshape_and_xxt_sharding(self):
    """Tests 2D tensor reshape, xxt, and b_times_x without all-to-all."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y"))
    w = jax.device_put(jax.random.normal(jax.random.key(0), (8, 16)), sharding)

    reshape_fn, unreshape_fn, flat_sharding = reshape_utils.get_reshape_fns(
        w,
        reduction_axes=(0,),
        output_axes=(1,),
        sharding=sharding,
    )
    w_flat = reshape_fn(w)
    self.assertEqual(w_flat.shape, (1, 1, 8, 16))
    self.assertEqual(
        w_flat.sharding.spec,
        jax.sharding.PartitionSpec(None, None, "x", "y"),
    )
    w_xxt = _muon.xxt(w_flat, flat_sharding=flat_sharding)
    self.assertEqual(w_xxt.shape, (1, 1, 8, 8))
    self.assertEqual(
        w_xxt.sharding.spec,
        jax.sharding.PartitionSpec(None, None, None, "x"),
    )
    w_restored = unreshape_fn(w_flat)
    self.assertEqual(w_restored.shape, (8, 16))
    self.assertEqual(w_restored.sharding, sharding)
    chex.assert_trees_all_close(w_restored, w)

  def test_3d_tensor_reshape_and_xxt_sharding_all_to_all(self):
    """Tests 3D tensor with unsharded batch dimension triggering all-to-all with padding."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, 2, 2)),
        axis_names=("a", "b", "c"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "b", "c"))
    x = jax.device_put(jax.random.normal(jax.random.key(0), (4, 8, 16)), sharding)

    reshape_fn, unreshape_fn, flat_sharding = reshape_utils.get_reshape_fns(
        x,
        reduction_axes=(1,),
        output_axes=(2,),
        sharding=sharding,
    )
    x_flat = reshape_fn(x)
    # Batch size 4 padded to total matrix shards = 2*2 = 4.
    self.assertEqual(x_flat.shape, (1, 4, 8, 16))
    self.assertEqual(
        x_flat.sharding.spec,
        jax.sharding.PartitionSpec(None, ("b", "c"), None, None),
    )

    xxt = _muon.xxt(x_flat, flat_sharding=flat_sharding)
    self.assertEqual(xxt.shape, (1, 4, 8, 8))
    self.assertEqual(
        xxt.sharding.spec,
        jax.sharding.PartitionSpec(None, ("b", "c"), None, None),
    )

    x_restored = unreshape_fn(x_flat)
    self.assertEqual(x_restored.shape, (4, 8, 16))
    self.assertEqual(x_restored.sharding, sharding)
    chex.assert_trees_all_close(x_restored, x)

  def test_3d_tensor_reshape_and_xxt_sharding_without_all_to_all(self):
    """Tests 3D tensor with unsharded batch dimension and use_all_to_all=False."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, 2, 2)),
        axis_names=("a", "b", "c"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "b", "c"))
    x = jax.device_put(jax.random.normal(jax.random.key(0), (4, 8, 16)), sharding)

    reshape_fn, unreshape_fn, flat_sharding = reshape_utils.get_reshape_fns(
        x,
        reduction_axes=(1,),
        output_axes=(2,),
        sharding=sharding,
        use_all_to_all=False,
    )
    x_flat = reshape_fn(x)
    self.assertEqual(x_flat.shape, (1, 4, 8, 16))
    self.assertEqual(
        x_flat.sharding.spec,
        jax.sharding.PartitionSpec(None, None, "b", "c"),
    )

    xxt = _muon.xxt(x_flat, flat_sharding=flat_sharding)
    self.assertEqual(xxt.shape, (1, 4, 8, 8))
    self.assertEqual(
        xxt.sharding.spec,
        jax.sharding.PartitionSpec(None, None, None, "b"),
    )

    x_restored = unreshape_fn(x_flat)
    self.assertEqual(x_restored.shape, (4, 8, 16))
    self.assertEqual(x_restored.sharding, sharding)
    chex.assert_trees_all_close(x_restored, x)

  def test_transposed_matrix_axes(self):
    """Tests tensor with m > n where rows and columns are swapped during reshape."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, 2, 2)),
        axis_names=("a", "b", "c"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("a", "b", "c"))
    # m = 16 (reduction), n = 8 (output) -> m > n, so reshape transposes.
    x = jax.device_put(jax.random.normal(jax.random.key(0), (4, 16, 8)), sharding)

    reshape_fn, unreshape_fn, flat_sharding = reshape_utils.get_reshape_fns(
        x,
        reduction_axes=(1,),
        output_axes=(2,),
        sharding=sharding,
    )
    x_flat = reshape_fn(x)
    self.assertEqual(x_flat.shape, (4, 1, 8, 16))
    # Row axis is output_axis (c), col axis is reduction_axis (b)
    self.assertEqual(x_flat.sharding.spec, jax.sharding.PartitionSpec("a", None, "c", "b"))

    xxt = _muon.xxt(x_flat, flat_sharding=flat_sharding)
    self.assertEqual(xxt.shape, (4, 1, 8, 8))
    self.assertEqual(xxt.sharding.spec, jax.sharding.PartitionSpec("a", None, None, "c"))

    x_restored = unreshape_fn(x_flat)
    self.assertEqual(x_restored.shape, (4, 16, 8))
    self.assertEqual(x_restored.sharding, sharding)
    chex.assert_trees_all_close(x_restored, x)

  def test_get_dim_mesh_axes(self):
    """Tests get_dim_mesh_axes extraction across different dimension configurations."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, 2, 2)),
        axis_names=("a", "b", "c"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "a", ("b", "c")))

    self.assertEqual(reshape_utils.get_dim_mesh_axes(None, 0), ())
    self.assertEqual(reshape_utils.get_dim_mesh_axes(sharding, -1), ())
    self.assertEqual(reshape_utils.get_dim_mesh_axes(sharding, 0), ())
    self.assertEqual(reshape_utils.get_dim_mesh_axes(sharding, 1), ("a",))
    self.assertEqual(reshape_utils.get_dim_mesh_axes(sharding, 2), ("b", "c"))
    self.assertEqual(reshape_utils.get_dim_mesh_axes(sharding, 3), ())

  def test_get_flat_mesh_axes(self):
    """Tests get_flat_mesh_axes concatenation across specified dimensions."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, 2, 2)),
        axis_names=("a", "b", "c"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("a", None, ("b", "c")))

    self.assertEqual(reshape_utils.get_flat_mesh_axes(None, (0, 1)), ())
    self.assertEqual(reshape_utils.get_flat_mesh_axes(sharding, ()), ())
    self.assertEqual(reshape_utils.get_flat_mesh_axes(sharding, (0, 1)), ("a",))
    self.assertEqual(reshape_utils.get_flat_mesh_axes(sharding, (0, 2)), ("a", "b", "c"))
    self.assertEqual(reshape_utils.get_flat_mesh_axes(sharding, (1, 2)), ("b", "c"))

  def test_is_explicit_axes(self):
    """Tests is_explicit_axes validation and classification."""
    devices = jax.devices("cpu")
    self.assertFalse(reshape_utils.is_explicit_axes(None))

    mesh_explicit = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    sharding_explicit = jax.sharding.NamedSharding(mesh_explicit, jax.sharding.PartitionSpec("x", "y"))
    self.assertTrue(reshape_utils.is_explicit_axes(sharding_explicit))

    mesh_auto = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )
    sharding_auto = jax.sharding.NamedSharding(mesh_auto, jax.sharding.PartitionSpec("x", "y"))
    self.assertFalse(reshape_utils.is_explicit_axes(sharding_auto))

    mesh_mixed = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Auto,
        ),
    )
    sharding_mixed = jax.sharding.NamedSharding(mesh_mixed, jax.sharding.PartitionSpec("x", "y"))
    with self.assertRaisesRegex(ValueError, "Mixed mesh axis types"):
      reshape_utils.is_explicit_axes(sharding_mixed)

  @parameterized.parameters(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto)
  def test_sharded_all_to_all_with_non_zero_padding_jitted(self, axis_type):
    """Tests 3D tensor with unsharded batch size 3 padded to 8 shards compiled under @jax.jit."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x", "y"))
    # Batch size 3 on 2*4=8 shards -> padding = 5 -> padded batch = 8
    key = jax.random.key(42)
    w = jax.device_put(jax.random.normal(key, (3, 8, 16)), sharding)
    grads = {"w": w}
    params = {"w": w}

    dim_nums = {"w": _muon.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1, sharding=sharding)}
    opt = _muon.muon(learning_rate=1e-3, muon_weight_dimension_numbers=dim_nums)
    state = opt.init(params)

    @jax.jit
    def step(p, s, g):
      return opt.update(g, s, p)

    updates, _ = step(params, state, grads)
    self.assertEqual(updates["w"].shape, (3, 8, 16))
    self.assertEqual(updates["w"].sharding, sharding)
    self.assertFalse(jnp.isnan(updates["w"]).any())


if __name__ == "__main__":
  parameterized.absltest.main()
