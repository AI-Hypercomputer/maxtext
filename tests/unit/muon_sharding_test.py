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

import os
import subprocess
import sys
import unittest

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from maxtext.optimizers import muon as _muon
from maxtext.optimizers import reshape_utils
import numpy as np
from optax.contrib import _muon as optax_muon
import pytest

_REQUIRED_CPU_DEVICES = 8

pytestmark = pytest.mark.cpu_only


@pytest.mark.cpu_only
def test_sharded_muon_on_cpu_mesh():
  """Runs the Muon sharding tests in a subprocess with forced 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_REQUIRED_CPU_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run(
      [sys.executable, __file__],
      env=env,
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "MUON_SHARDING_TESTS_PASSED" in result.stdout


class MuonShardingTest(parameterized.TestCase):
  __test__ = False

  def setUp(self):
    super().setUp()
    try:
      cpu_devices = jax.devices("cpu")
    except (RuntimeError, ValueError):
      self.skipTest(f"MuonShardingTest requires {_REQUIRED_CPU_DEVICES} CPU devices; CPU" " backend unavailable.")
    if len(cpu_devices) < _REQUIRED_CPU_DEVICES:
      self.skipTest(
          f"MuonShardingTest requires {_REQUIRED_CPU_DEVICES} CPU devices;" " executed via test_sharded_muon_on_cpu_mesh."
      )

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

    local_dim_nums = {"w": _muon.ShardedMuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)}
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
    """Tests that scale_by_muon handles 2D tensors with sharded matrix axes."""
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
        weight_dimension_numbers={"w": _muon.ShardedMuonDimensionNumbers(sharding=sharding)},
    )
    optax_transform = optax_muon.scale_by_muon(
        weight_dimension_numbers={"w": optax_muon.MuonDimensionNumbers()},
    )
    local_updates, _ = local_transform.update(grads, local_transform.init(params), params)
    unsharded_params = jax.device_get(params)
    unsharded_grads = jax.device_get(grads)
    optax_updates, _ = optax_transform.update(unsharded_grads, optax_transform.init(unsharded_params), unsharded_params)

    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=1, output_axis=2, sharding=sharding)
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
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

    dim_num = _muon.ShardedMuonDimensionNumbers(reduction_axis=2, output_axis=3, sharding=sharding)
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
        weight_dimension_numbers={"w": _muon.ShardedMuonDimensionNumbers(sharding=sharding)},
    )
    state = transform.init(params)
    with self.assertRaisesRegex(ValueError, "Mixed mesh axis types"):
      transform.update(params, state, params)

  @parameterized.named_parameters(
      (
          "2d_unshards_matrix",
          (8, 16),
          jax.sharding.PartitionSpec("x", "y"),
          True,
          jax.sharding.PartitionSpec(None, None, None),
      ),
      (
          "3d_local_batch_1_replicates_matrix",
          (2, 8, 16),
          jax.sharding.PartitionSpec("x", "y", None),
          True,
          jax.sharding.PartitionSpec(("x",), None, None),
      ),
      (
          "3d_unsharded_batch_all_to_all",
          (8, 8, 16),
          jax.sharding.PartitionSpec(None, "x", "y"),
          True,
          jax.sharding.PartitionSpec(("x", "y"), None, None),
      ),
      (
          "3d_sharded_batch_padded_all_to_all",
          (6, 8, 16),
          jax.sharding.PartitionSpec("x", "y", None),
          True,
          jax.sharding.PartitionSpec(("x", "y"), None, None),
      ),
      (
          "3d_disabled_all_to_all_transposed",
          (4, 16, 8),
          jax.sharding.PartitionSpec("x", "y", None),
          False,
          jax.sharding.PartitionSpec("x", None, "y"),
      ),
  )
  def test_get_reshape_fns_flat_sharding(self, shape, spec, use_all_to_all, expected_flat_spec):
    """Tests flat_sharding.spec and round-trip reconstruction across sharding modes."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    sharding = jax.sharding.NamedSharding(mesh, spec)
    x = jax.device_put(jax.random.normal(jax.random.key(0), shape), sharding)

    reshape_fn, unreshape_fn, flat_sharding = reshape_utils.get_reshape_fns(
        x,
        reduction_axes=(-2,),
        output_axes=(-1,),
        sharding=sharding,
        use_all_to_all=use_all_to_all,
    )
    x_flat = reshape_fn(x)
    self.assertEqual(flat_sharding.spec, expected_flat_spec)
    self.assertEqual(x_flat.sharding.spec, expected_flat_spec)
    x_restored = unreshape_fn(x_flat)
    self.assertEqual(x_restored.sharding, sharding)
    chex.assert_trees_all_close(x_restored, x)

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
  def test_sharded_batch_and_matrix_axes_with_padding_matches_optax_contrib(self, axis_type):
    """Tests sharded batch with local padding matches optax.contrib."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 4),
        axis_names=("x", "y"),
        axis_types=(axis_type, axis_type),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y", None))
    w = jax.device_put(jax.random.normal(jax.random.key(11), (6, 8, 16)), sharding)
    params = {"w": w}
    local_opt = _muon.scale_by_muon(
        weight_dimension_numbers={"w": _muon.ShardedMuonDimensionNumbers(1, 2, sharding=sharding)}
    )
    optax_opt = optax_muon.scale_by_muon(weight_dimension_numbers={"w": optax_muon.MuonDimensionNumbers(1, 2)})
    local_updates, _ = jax.jit(local_opt.update)(params, local_opt.init(params), params)
    unsharded_params = jax.device_get(params)
    optax_updates, _ = optax_opt.update(unsharded_params, optax_opt.init(unsharded_params), unsharded_params)
    chex.assert_trees_all_close(local_updates, optax_updates, rtol=1e-5, atol=1e-5)
    if axis_type == jax.sharding.AxisType.Explicit:
      self.assertEqual(local_updates["w"].sharding, sharding)


if __name__ == "__main__":
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(MuonShardingTest)
  runner = unittest.TextTestRunner(verbosity=2)
  res = runner.run(suite)
  if not res.wasSuccessful():
    sys.exit(1)
  print("MUON_SHARDING_TESTS_PASSED")
