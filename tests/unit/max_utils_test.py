# Copyright 2023–2025 Google LLC
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

"""Tests for the common Max Utils"""
import os
import sys
import unittest
import time
import pytest


import jax
from jax import numpy as jnp
from jax import random

from flax import linen as nn
from flax import nnx

import optax

from maxtext.configs import pyconfig
from maxtext.utils import max_utils
from maxtext.utils.train_utils import setup_train_loop
from tests.utils.test_helpers import get_test_config_path
from unittest import mock


class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""

  def test_l2norm_pytree(self):
    x = {"a": jax.numpy.array([0, 2, 0]), "b": jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))


class MaxUtilsPytree(unittest.TestCase):
  """Tests initialization of training and decode states in max_utils.py"""

  def setUp(self):
    self.model = nn.Dense(features=5)
    self.key1, self.key2 = random.split(random.key(0))
    self.input = random.normal(self.key1, (10,))  # Dummy input data
    self.params = self.model.init(self.key2, self.input)

  def test_calculate_num_params_from_pytree(self):
    example_tree = [
        [1, "a", object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jnp.array([1, 2, 3]),
    ]
    self.assertEqual(max_utils.calculate_num_params_from_pytree(example_tree), 17)
    # Model params
    self.assertEqual(max_utils.calculate_num_params_from_pytree(self.params), 55)


class MaxUtilsT5XCrossEntropy(unittest.TestCase):
  """Tests for the cross entropy functions in max_utils.py"""

  def test_t5x_cross_entropy(self):
    # Generate random targets and logits
    key = jax.random.PRNGKey(0)
    targets = jax.random.randint(key, shape=(48, 2048), dtype=jax.numpy.int32, minval=1, maxval=10)
    logits = jax.random.uniform(key, shape=(48, 2048, 4096), dtype=jax.numpy.float32)

    # Calculate xent from optax implementation
    optax_xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # Calculate xent from custom T5X implementation
    one_hot_targets = jax.nn.one_hot(targets, 4096)
    t5x_xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    t5x_xent = nn.with_logical_constraint(t5x_xent, ("activation_batch", "activation_length"))

    # Compare results
    self.assertTrue(jax.numpy.allclose(optax_xent, t5x_xent, rtol=1e-05, atol=1e-08, equal_nan=False))

  def test_cross_entropy_with_z_loss(self):
    """Tests the exact mathematical output of the z-loss penalty."""
    # Shape [2, 3] to test across multiple dimensions
    logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]])
    # Target indices: [2, 1], [0, 2]
    targets = jnp.array([[2, 1], [0, 2]])
    one_hot_targets = jax.nn.one_hot(targets, 3)

    z_loss_multiplier = 1e-4

    # 1. Run without z-loss
    total_loss_no_z, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=0.0)

    # 2. Run with z-loss
    total_loss_with_z, z_loss_only = max_utils.cross_entropy_with_logits(
        logits, one_hot_targets, z_loss=z_loss_multiplier
    )

    # 3. Calculate expected z-loss manually
    # Expected log_z = log(sum(exp(logits), axis=-1))
    expected_log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    expected_z_loss = z_loss_multiplier * jnp.square(expected_log_z)

    # Compare isolated z_loss component
    self.assertTrue(jnp.allclose(z_loss_only, expected_z_loss, rtol=1e-5, atol=1e-8))

    # Compare total loss aggregation
    self.assertTrue(jnp.allclose(total_loss_with_z, total_loss_no_z + z_loss_only, rtol=1e-5, atol=1e-8))


class MaxUtilsCustomMesh(unittest.TestCase):
  """Tests for the is_valid_custom_mesh function in max_utils.py"""

  def test_empty_value(self):
    self.assertFalse(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], ""))

  def test_valid_64x4(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], "hybrid_ring_64x4"))

  def test_valid_32x8(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 32, 1, 1, 8, 1, 1], "hybrid_ring_32x8"))

  def test_invalid_64x4(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "hybrid_ring_64x4")

  def test_invalid_strategy(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "invalid_strategy")


class UnscanTest(unittest.TestCase):
  """Test unscanning utility."""

  def init_pyconfig(self, **kwargs):
    """init pyconfig"""
    init_kwargs = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "model_name": "llama3.1-8b",
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **init_kwargs,
    )
    return config

  @pytest.mark.tpu_only
  def test_unscan_train_state_params(self):
    """Test unscan_train_state_params logic and performance with a real model."""
    # Initialize a configuration for an 8B model.
    config = self.init_pyconfig()

    _, _, sharding, _, mesh, *_, state = setup_train_loop(config, None)

    scan_axis = config.param_scan_axis
    num_layers = config.base_num_decoder_layers

    # Make a copy to unscan, leaving the original state intact.
    if hasattr(state, "model"):
      _, params_state, _ = nnx.split(state.model, nnx.Param, ...)
      params_to_unscan = {"params": params_state.to_pure_dict()}
    else:
      params_to_unscan = jax.tree_util.tree_map(lambda x: x, state.params)
    if hasattr(sharding, "model"):
      _, sharding_params, _ = nnx.split(sharding.model, nnx.Param, ...)
      sharding_to_unscan = {"params": sharding_params.to_pure_dict()}
    else:
      sharding_to_unscan = jax.tree_util.tree_map(lambda x: x, sharding.params)

    # Time the unscan operation.
    start_time = time.time()
    max_utils.unscan_train_state_params(
        params_to_unscan,
        sharding_to_unscan,
        mesh,
        scan_axis,
        [("layers", num_layers)],
    )
    jax.block_until_ready(params_to_unscan)
    end_time = time.time()
    print(f"\nUnscanning 8B model took: {end_time - start_time:.4f} seconds.\n")

    # Assertions to verify correctness.
    decoder_params = params_to_unscan["params"]["decoder"]
    self.assertNotIn("layers", decoder_params)
    self.assertIn("layers_0", decoder_params)
    self.assertIn(f"layers_{num_layers-1}", decoder_params)

    # Check shape of one of the unstacked tensors.
    # The exact key might differ based on model implementation, adjust if needed.
    unstacked_shape = decoder_params["layers_5"]["mlp"]["wi_0"]["kernel"].shape
    expected_shape = (config.base_emb_dim, config.base_mlp_dim)
    self.assertEqual(unstacked_shape, expected_shape)

    # Check that the original state is unchanged.

    if hasattr(state, 'model'):
      _, params_state, _ = nnx.split(state.model, nnx.Param, ...)
      state_decoder_params = params_state.to_pure_dict()["decoder"]
      self.assertIn("layers", state_decoder_params)
    else:
      self.assertIn("layers", state.params["params"]["decoder"])

    if hasattr(state, 'model'):
      self.assertNotIn("layers_0", state_decoder_params)
    else:
      self.assertNotIn("layers_0", state.params["params"]["decoder"])


class TestGpuDistributedInitialization(unittest.TestCase):
  """Tests using CUDA_VISIBLE_DEVICES / SLURM_STEP_GPUS for jax.distributed.initialize."""

  @mock.patch.dict(
      os.environ,
      {
          "JAX_COORDINATOR_IP": "10.0.0.1",
          "JAX_COORDINATOR_PORT": "1234",
          "NNODES": "1",
          "NODE_RANK": "0",
          "CUDA_VISIBLE_DEVICES": "0,2,3",
      },
  )
  @mock.patch("jax.distributed.initialize")
  @mock.patch("jax.devices")
  @mock.patch("maxtext.utils.max_logging.log")
  def test_initialize_jax_for_gpu_valid_devices(self, _mock_log, _mock_devices, mock_init):
    """Verifies that a comma-separated string of IDs is correctly parsed."""
    raw_keys = {"jax_distributed_initialization_timeout": 300}
    max_utils.initialize_jax_for_gpu(raw_keys)
    _, kwargs = mock_init.call_args
    self.assertEqual(kwargs["local_device_ids"], [0, 2, 3])
    self.assertEqual(kwargs["coordinator_address"], "10.0.0.1:1234")

  @mock.patch.dict(
      os.environ,
      {
          "JAX_COORDINATOR_IP": "10.0.0.1",
          "JAX_COORDINATOR_PORT": "1234",
          "NNODES": "1",
          "NODE_RANK": "0",
          "CUDA_VISIBLE_DEVICES": "GPU-8f2e3072-...",
      },
  )
  @mock.patch("jax.distributed.initialize")
  @mock.patch("jax.devices")
  @mock.patch("maxtext.utils.max_logging.log")
  def test_initialize_jax_for_gpu_invalid_devices(self, _mock_log, _mock_devices, mock_init):
    """Verifies fallback behavior when parsing fails (e.g., UUIDs)."""
    raw_keys = {"jax_distributed_initialization_timeout": 300}
    max_utils.initialize_jax_for_gpu(raw_keys)
    _, kwargs = mock_init.call_args
    self.assertIsNone(kwargs.get("local_device_ids"))
    self.assertEqual(kwargs["coordinator_address"], "10.0.0.1:1234")

  @mock.patch.dict(
      os.environ,
      {
          "JAX_COORDINATOR_IP": "10.0.0.1",
          "JAX_COORDINATOR_PORT": "1234",
          "NNODES": "1",
          "NODE_RANK": "0",
      },
      clear=True,
  )
  @mock.patch("jax.config.update")
  @mock.patch("jax.distributed.initialize")
  @mock.patch("jax.devices")
  @mock.patch("maxtext.utils.max_logging.log")
  def test_initialize_jax_for_gpu_no_devices(self, _mock_log, _mock_devices, mock_init, mock_config_update):
    """When coordinator env is set but neither CUDA_VISIBLE_DEVICES nor SLURM_STEP_GPUS is set, JAX uses all devices
    (config) and init gets no local ids.
    """
    raw_keys = {"jax_distributed_initialization_timeout": 300}
    max_utils.initialize_jax_for_gpu(raw_keys)
    _, kwargs = mock_init.call_args
    self.assertIsNone(kwargs.get("local_device_ids"))
    self.assertEqual(kwargs["coordinator_address"], "10.0.0.1:1234")
    mock_config_update.assert_has_calls(
        [
            mock.call("jax_cuda_visible_devices", "all"),
            mock.call("jax_rocm_visible_devices", "all"),
        ]
    )

  @mock.patch("jax.distributed.initialize")
  @mock.patch("jax.devices")
  @mock.patch("maxtext.utils.max_logging.log")
  def test_initialize_jax_for_gpu_uses_slurm_when_cuda_unset(self, mock_log, _mock_devices, mock_init):
    """Uses SLURM_STEP_GPUS when CUDA_VISIBLE_DEVICES is absent (loop over env_var_list)."""
    env = {
        "JAX_COORDINATOR_IP": "10.0.0.1",
        "JAX_COORDINATOR_PORT": "1234",
        "NNODES": "1",
        "NODE_RANK": "0",
        "SLURM_STEP_GPUS": "1,3",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      os.environ.pop("CUDA_VISIBLE_DEVICES", None)
      raw_keys = {"jax_distributed_initialization_timeout": 300}
      max_utils.initialize_jax_for_gpu(raw_keys)
    _, kwargs = mock_init.call_args
    self.assertEqual(kwargs["local_device_ids"], [1, 3])
    mock_log.assert_any_call("Using SLURM_STEP_GPUS to initialize JAX distributed system: 1,3")

  @mock.patch.dict(
      os.environ,
      {
          "JAX_COORDINATOR_IP": "10.0.0.1",
          "JAX_COORDINATOR_PORT": "1234",
          "NNODES": "1",
          "NODE_RANK": "0",
          "CUDA_VISIBLE_DEVICES": "0,2",
          "SLURM_STEP_GPUS": "4,5,6",
      },
  )
  @mock.patch("jax.distributed.initialize")
  @mock.patch("jax.devices")
  @mock.patch("maxtext.utils.max_logging.log")
  def test_initialize_jax_for_gpu_prefers_cuda_visible_devices_in_loop(self, mock_log, _mock_devices, mock_init):
    """First matching env var in the list wins; CUDA_VISIBLE_DEVICES is checked before SLURM_STEP_GPUS."""
    raw_keys = {"jax_distributed_initialization_timeout": 300}
    max_utils.initialize_jax_for_gpu(raw_keys)
    _, kwargs = mock_init.call_args
    self.assertEqual(kwargs["local_device_ids"], [0, 2])
    mock_log.assert_any_call("Using CUDA_VISIBLE_DEVICES to initialize JAX distributed system: 0,2")


class TestMaybePad(unittest.TestCase):
  """Tests that maybe_pad satisfies its contract."""

  def test_odd_shape_padded(self):
    inputs = jnp.ones((9, 8))
    tile_size = 4

    target_padding_amount = 3
    target = jnp.concat((inputs, jnp.zeros((3, 8))))

    padded, padding_amount = max_utils.maybe_pad(inputs, tile_size)
    self.assertTrue(jnp.equal(padded, target).all())
    self.assertEqual(padding_amount, target_padding_amount)

  def test_regular_shape_unpadded(self):
    inputs = jnp.ones((12, 13))
    tile_size = 4

    target_padding_amount = 0
    target = jnp.ones((12, 13))

    padded, padding_amount = max_utils.maybe_pad(inputs, tile_size)
    self.assertTrue(jnp.equal(padded, target).all())
    self.assertEqual(padding_amount, target_padding_amount)


class TestMaybeInitializeJaxDistributedSystem(unittest.TestCase):
  """Tests for maybe_initialize_jax_distributed_system.

  Note: jax.distributed.initialize is mocked in these tests.
  We intentionally avoid actually calling jax.distributed.initialize
  because doing so within pytest can lead to unpredictable
  interactions with the test runner.
  """

  def setUp(self):
    patcher = mock.patch("jax.distributed.is_initialized", return_value=False)
    self.mock_is_initialized = patcher.start()
    self.addCleanup(patcher.stop)

  def _base_keys(self, **overrides):
    """Return a minimal raw_keys dict with all required fields, accepting per-test overrides."""
    keys = {
        "skip_jax_distributed_system": False,
        "enable_single_controller": False,
        "inference_benchmark_test": False,
        "compile_topology": False,
        "hardware": "tpu",
        "enable_emergency_checkpoint": False,
        "jax_distributed_initialization_timeout": 300,
        "enable_multi_tier_checkpointing": False,
        "local_checkpoint_directory": "/tmp/ckpt",
        "multi_tier_checkpointing_backup_interval_minutes": 5,
        "run_name": "test_run",
        "mtc_data_parallelism": 1,
        "enable_checkpointing": True,
        "compile_topology_num_slices": -1,
    }
    keys.update(overrides)
    return keys

  @mock.patch("jax.distributed.initialize")
  def test_skip_flag_exits_early(self, mock_init):
    max_utils.maybe_initialize_jax_distributed_system(self._base_keys(skip_jax_distributed_system=True))
    mock_init.assert_not_called()

  @mock.patch("jax.distributed.initialize")
  def test_single_controller_exits_early(self, mock_init):
    max_utils.maybe_initialize_jax_distributed_system(self._base_keys(enable_single_controller=True))
    mock_init.assert_not_called()

  @mock.patch("jax.distributed.is_initialized", return_value=True)
  @mock.patch("jax.distributed.initialize")
  def test_already_initialized_exits_early(self, mock_init, _is_init):
    max_utils.maybe_initialize_jax_distributed_system(self._base_keys())
    mock_init.assert_not_called()

  @mock.patch("jax.distributed.initialize")
  def test_inference_benchmark_exits_early(self, mock_init):
    max_utils.maybe_initialize_jax_distributed_system(self._base_keys(inference_benchmark_test=True))
    mock_init.assert_not_called()

  @mock.patch("jax.distributed.initialize")
  def test_compile_topology_exits_early(self, mock_init):
    max_utils.maybe_initialize_jax_distributed_system(self._base_keys(compile_topology=True))
    mock_init.assert_not_called()

  @mock.patch("maxtext.utils.max_utils.initialize_jax_for_gpu")
  def test_gpu_backend_calls_initialize_jax_for_gpu(self, mock_gpu_init):
    raw_keys = self._base_keys(hardware="gpu")
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_gpu_init.assert_called_once_with(raw_keys)

  @mock.patch("maxtext.utils.max_utils.initialize_jax_for_cpu")
  def test_cpu_backend_calls_initialize_jax_for_cpu(self, mock_cpu_init):
    raw_keys = self._base_keys(hardware="cpu")
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_cpu_init.assert_called_once_with(raw_keys)

  @mock.patch("jax.distributed.initialize")
  def test_gpu_multiprocess_no_emergency_calls_jax_init(self, mock_init):
    raw_keys = self._base_keys(hardware="gpu_multiprocess")
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_init.assert_called_once_with(initialization_timeout=self._base_keys()["jax_distributed_initialization_timeout"])

  # create=True: initialize_distributed_to_device_ids only exists in multi-host orbax builds, not single-host.
  @mock.patch("orbax.checkpoint.multihost.initialize_distributed_to_device_ids", create=True)
  @mock.patch("orbax.checkpoint.multihost.initialize_runtime_to_distributed_ids")
  @mock.patch("jax.distributed.initialize")
  def test_gpu_multiprocess_with_emergency_calls_ocp_multihost(self, mock_init, mock_runtime, mock_device):
    raw_keys = self._base_keys(hardware="gpu_multiprocess", enable_emergency_checkpoint=True)
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_init.assert_called_once_with(initialization_timeout=self._base_keys()["jax_distributed_initialization_timeout"])
    mock_runtime.assert_called_once()
    mock_device.assert_called_once()

  @mock.patch("maxtext.utils.max_utils.initialize_multi_tier_checkpointing")
  def test_tpu_multi_tier_checkpointing(self, mock_mtc):
    raw_keys = self._base_keys(enable_multi_tier_checkpointing=True)
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_mtc.assert_called_once_with(
        local_checkpoint_directory=self._base_keys()["local_checkpoint_directory"],
        backup_interval_minutes=self._base_keys()["multi_tier_checkpointing_backup_interval_minutes"],
        run_name=self._base_keys()["run_name"],
        jax_initialization_timeout_seconds=self._base_keys()["jax_distributed_initialization_timeout"],
        data_parallelism=self._base_keys()["mtc_data_parallelism"],
    )

  @mock.patch("jax.distributed.initialize")
  def test_tpu_checkpointing_no_emergency_calls_jax_init(self, mock_init):
    raw_keys = self._base_keys(enable_checkpointing=True, compile_topology_num_slices=-1)
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_init.assert_called_once_with(initialization_timeout=self._base_keys()["jax_distributed_initialization_timeout"])

  @mock.patch("maxtext.utils.max_utils.initialize_jax_for_tpu_with_emergency_checkpointing")
  def test_tpu_checkpointing_with_emergency(self, mock_tpu_emergency):
    raw_keys = self._base_keys(
        enable_checkpointing=True, compile_topology_num_slices=-1, enable_emergency_checkpoint=True
    )
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_tpu_emergency.assert_called_once_with(raw_keys)

  @mock.patch("jax.distributed.initialize")
  def test_tpu_no_checkpointing_does_not_call_jax_init(self, mock_init):
    raw_keys = self._base_keys(enable_checkpointing=False, enable_multi_tier_checkpointing=False)
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)
    mock_init.assert_not_called()


if __name__ == "__main__":
  unittest.main()
