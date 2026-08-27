#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the DiLoCo implementation in diloco.py"""


import os
import shutil
import tempfile
from tempfile import gettempdir
import unittest

import chex
from flax.experimental import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax.sharding
from maxtext.common import checkpointing
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco import utils as diloco_utils
from maxtext.trainers.diloco.utils import spmd_diloco_checkpointing as diloco_checkpoint_utils
from maxtext.trainers.pre_train.train_compile import main as train_compile_main
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax
import pytest


class SimpleNNXModel(nnx.Module):
  """A simple state for testing a minimal model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.dense = nnx.Linear(
        2,
        1,
        kernel_init=nnx.initializers.constant(jnp.asarray([[2.0], [1.0]])),
        bias_init=nnx.initializers.ones_init(),
        rngs=rngs,
    )

  def __call__(self, x):
    return self.dense(x)


@pytest.mark.integration_test
class DiLoCoTest(unittest.TestCase):

  @pytest.mark.tpu_only
  def test_diloco_training_simulation_with_mesh(self):
    """Runs a simulation of DiLoCo training on a mesh and asserts correctness."""
    num_replicas = 2
    num_steps = 4

    devices = jax.devices()
    if len(devices) < num_replicas:
      self.skipTest(f"Test requires {num_replicas} devices, but only {len(devices)} are available.")

    mesh_devices = np.array(devices[:num_replicas]).reshape(1, num_replicas)
    mesh = jax.sharding.Mesh(mesh_devices, axis_names=("data", "diloco"))

    test_config = initialize_pydantic(
        [
            "",
            get_test_config_path(),
            f"dcn_diloco_parallelism={num_replicas}",
            "ici_diloco_parallelism=1",
            "diloco_outer_momentum=0.9",
            "diloco_outer_lr=1.0",
            f"diloco_sync_period={num_steps-1}",
        ]
    )

    with jax.set_mesh(mesh):
      tx = optax.sgd(learning_rate=0.1)
      rngs = nnx.Rngs(params=jax.random.key(seed=42))
      model = SimpleNNXModel(rngs=rngs)
      graphdef, params = nnx.split(model)

      if test_config.pure_nnx:
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        # diloco_test_state expects a TrainStateNNX instance when pure_nnx is True.
        initial_test_state = TrainStateNNX(model, optimizer)

        # For NNX, train_step needs to take the TrainStateNNX and mutate it

        def _test_train_step(state, batch, prng_key: diloco.PRNGKey):
          del prng_key

          def loss_fn(model, batch):
            inputs, labels = batch
            logits = jax.vmap(model)(inputs)
            residual = logits - labels
            return jnp.mean(jnp.square(residual))

          loss, grads = nnx.value_and_grad(loss_fn)(state.model, batch)
          state.optimizer.update(state.model, grads)
          return state, loss

      else:

        def nnx_apply_fn(params, inputs):
          model_replica = nnx.merge(graphdef, params)
          return model_replica(inputs)

        # 2. Vmap this new wrapper function
        vmapped_apply = jax.vmap(nnx_apply_fn, in_axes=(None, 0))

        def _test_train_step(state: train_state.TrainState, batch, prng_key: diloco.PRNGKey):
          """A simple MSE loss train step to enable numerics testing."""
          del prng_key

          def loss_fn(params, batch):
            inputs, labels = batch
            logits = vmapped_apply(params, inputs)
            residual = logits - labels
            sq_residual = jnp.square(residual)
            msq_residual = jnp.mean(sq_residual)
            return msq_residual

          loss, grad = jax.value_and_grad(loss_fn)(state.params, batch)
          return state.apply_gradients(grads=grad), loss

        initial_test_state = train_state.TrainState.create(
            apply_fn=vmapped_apply,
            params=params,
            tx=tx,
        )

      diloco_test_state, _ = diloco.build_diloco_state(test_config, lambda: initial_test_state)
      chex.assert_equal(diloco_test_state.step, 0)
      if test_config.pure_nnx:
        _, params_pure, _ = nnx.split(initial_test_state.model, nnx.Param, ...)

        # diloco_test_state.params might contain nnx.Variables instead of pure arrays.
        # We need to unwrap them if they do.
        diloco_params_pure = jax.tree_util.tree_map(
            lambda x: x.value if hasattr(x, "value") else x,
            diloco_test_state.params,
        )
        chex.assert_trees_all_equal(diloco_params_pure, params_pure.to_pure_dict())
      else:
        chex.assert_trees_all_equal(diloco_test_state.params, initial_test_state.params)

      diloco_train_step = diloco.build_diloco_train_step(test_config, _test_train_step)
      inputs = jnp.array(
          [
              [[0.0, 1.0], [1.0, 0.0]],  # First replica inputs.
              [[1.0, 0.0], [0.0, 1.0]],  # Second replica inputs.
          ]
      )
      labels = jnp.array(
          [
              [[1.0], [2.0]],  # First replica labels.
              [[2.0], [3.0]],  # Second replica labels.
          ]
      )

      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "diloco"))
      inputs = jax.device_put(inputs, sharding)
      labels = jax.device_put(labels, sharding)

      # Run the first step (no synchronization).
      # Replica 0:
      #   Data: [[0, 1], [1, 0]]
      #   Labels: [[1], [2]]
      #   Weights: w = [[2], [1]]
      #   Bias: b = [1]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[2], [1]] + [1])) ^ 2 )
      #   = mean( ([[1], [2]] - [[2], [3]]) ^ 2 )
      #   = mean( ([-1, 1]) ^ 2 ) = mean( [1, 1] )
      #   = 1.0
      #
      # Replica 1:
      #   Data: [[1, 0], [0, 1]]
      #   Labels: [[2], [3]]
      #   Weights: w = [[2], [1]]
      #   Bias: b = [1]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[2], [1]] + [1])) ^ 2 )
      #   = mean( ([[2], [3]] - [[3], [2]]) ^ 2 )
      #   = mean( ([-1, 1]) ^ 2 ) = mean( [1, 1] )
      #   = 1.0
      diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
      chex.assert_equal(diloco_test_state.step, 1.0)
      chex.assert_equal(loss, 1.0)
      # Assert no updates to the global model yet (no synchronization)
      if test_config.pure_nnx:
        _, params_pure, _ = nnx.split(initial_test_state.model, nnx.Param, ...)

        # diloco_test_state.params might contain nnx.Variables instead of pure arrays.
        # We need to unwrap them if they do.
        diloco_params_pure = jax.tree_util.tree_map(
            lambda x: x.value if hasattr(x, "value") else x,
            diloco_test_state.params,
        )
        chex.assert_trees_all_equal(diloco_params_pure, params_pure.to_pure_dict())
      else:
        chex.assert_trees_all_equal(diloco_test_state.params, initial_test_state.params)

      # Run the second step (no synchronization).
      # Replica 0:
      #   Data: [[0, 1], [1, 0]]
      #   Labels: [[1], [2]]
      #   Weights: w = [[1.9], [0.9]]
      #   Bias: b = [0.8]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.9], [0.9]] + [0.8])) ^ 2 )
      #   = mean( ([[1], [2]] - [[1.7], [2.7]]) ^ 2 )
      #   = mean( ([-0.7, 0.7]) ^ 2 ) = mean( [0.49, 0.49] )
      #   = 0.49
      #
      # Replica 1:
      #   Data: [[1, 0], [0, 1]]
      #   Labels: [[2], [3]]
      #   Weights: w = [[1.9], [1.1]]
      #   Bias: b = [1]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.9], [1.1]] + [1])) ^ 2 )
      #   = mean( ([[2], [3]] - [[2.9], [2.1]]) ^ 2 )
      #   = mean( ([-0.9, 0.9]) ^ 2 ) = mean( [0.81, 0.81] )
      #   = 0.81
      diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
      chex.assert_equal(diloco_test_state.step, 2.0)
      chex.assert_trees_all_close(loss, 0.49, rtol=1e-2, atol=1e-2)
      # Assert no updates to the global model yet (no synchronization)
      if test_config.pure_nnx:
        _, params_pure, _ = nnx.split(initial_test_state.model, nnx.Param, ...)

        # diloco_test_state.params might contain nnx.Variables instead of pure arrays.
        # We need to unwrap them if they do.
        diloco_params_pure = jax.tree_util.tree_map(
            lambda x: x.value if hasattr(x, "value") else x,
            diloco_test_state.params,
        )
        chex.assert_trees_all_equal(diloco_params_pure, params_pure.to_pure_dict())
      else:
        chex.assert_trees_all_equal(diloco_test_state.params, initial_test_state.params)

      # Run the third step, which synchronizes afterwards.
      # Replica 0:
      #   Data: [[0, 1], [1, 0]]
      #   Labels: [[1], [2]]
      #   Weights: w = [[1.83], [0.83]]
      #   Bias: b = [0.66]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.83], [0.83]] + [0.66])) ^ 2 )
      #   = mean( ([[1], [2]] - [[1.49], [2.49]]) ^ 2 )
      #   = mean( ([-0.49, 0.49]) ^ 2 ) = mean( [0.2401, 0.2401] )
      #   = 0.2401
      #
      # Replica 1:
      #   Data: [[1, 0], [0, 1]]
      #   Labels: [[2], [3]]
      #   Weights: w = [[1.81], [1.19]]
      #   Bias: b = [1.]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.81], [1.19]] + [1])) ^ 2 )
      #   = mean( ([[2], [3]] - [[2.81], [2.19]]) ^ 2 )
      #   = mean( ([-0.81, 0.81]) ^ 2 ) = mean( [0.6561, 0.6561] )
      #   = 0.6561
      #
      # After these are averaged, the model differences are computed to create a
      # pseudo-gradient update to the outer_params and applied via a momentum
      # based outer optimizer.
      diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
      chex.assert_equal(diloco_test_state.step, 3.0)
      chex.assert_trees_all_close(loss, 0.2401, rtol=1e-2, atol=1e-2)
      # Assert that inner and outer parameters are all equal now that
      # synchronization has happened.
      if test_config.pure_nnx:
        _, inner_params, _ = nnx.split(diloco_test_state.inner_state.model, nnx.Param, ...)
        inner_params_pure = jax.tree_util.tree_map(
            lambda x: x.value if hasattr(x, "value") else x,
            inner_params.to_pure_dict(),
        )
        diloco_params_pure_3 = jax.tree_util.tree_map(
            lambda x: x.value if hasattr(x, "value") else x,
            diloco_test_state.params,
        )
        chex.assert_trees_all_equal(
            diloco_params_pure_3,
            jax.tree.map(lambda arr: arr[0, ...], inner_params_pure),
        )
        chex.assert_trees_all_equal(
            diloco_params_pure_3,
            jax.tree.map(lambda arr: arr[1, ...], inner_params_pure),
        )
      else:
        chex.assert_trees_all_equal(
            diloco_test_state.params,
            jax.tree.map(lambda arr: arr[0, ...], diloco_test_state.inner_state.params),
        )
        chex.assert_trees_all_equal(
            diloco_test_state.params,
            jax.tree.map(lambda arr: arr[1, ...], diloco_test_state.inner_state.params),
        )

      # Run the fourth step (no synchronization).
      # Replica 0:
      #   Data: [[0, 1], [1, 0]]
      #   Labels: [[1], [2]]
      #   Weights: w = [[1.5345], [1.0494]]
      #   Bias: b = [0.5839]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.5345], [1.0494]]] + [0.5839])) ^ 2 )
      #   = mean( ([[1], [2]] - [[1.6333], [2.1184]]) ^ 2 )
      #   = mean( ([-0.6333, 0.1184]) ^ 2 ) = mean( [0.4010, 0.0140] )
      #   ~ 0.2075
      #
      # Replica 1:
      #   Data: [[1, 0], [0, 1]]
      #   Labels: [[2], [3]]
      #   Weights: w = [[1.5345], [1.0494]]
      #   Bias: b = [0.5839]
      #   Loss = mean((y - pred)^2) =
      #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
      #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.5345], [1.0494]] + [0.5839])) ^ 2 )
      #   = mean( ([[2], [3]] - [[2.1184], [1.6333]]) ^ 2 )
      #   = mean( ([-0.1184, 1.3667]) ^ 2 ) = mean( [0.0140, 1.8678] )
      #   ~ 0.94
      step_three_outer_params = diloco_test_state.params
      diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
      chex.assert_equal(diloco_test_state.step, 4.0)
      chex.assert_trees_all_close(loss, 0.207545, rtol=1e-2, atol=1e-2)
      # Assert no updates to the global model since previous step (no
      # synchronization).
      chex.assert_trees_all_equal(diloco_test_state.params, step_three_outer_params)

  @pytest.mark.tpu_backend
  def test_diloco_qwen3_moe_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_diloco_qwen3_moe.pickle")
    train_compile_main(
        (
            None,
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=tpu7x-16",
            "compile_topology_num_slices=2",
            "ici_fsdp_parallelism=-1",
            "dcn_diloco_parallelism=2",
            "enable_diloco=true",
            "model_name=qwen3-30b-a3b",
            "override_model_config=True",
            "base_emb_dim=32",
            "base_num_decoder_layers=1",
            "base_mlp_dim=64",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "head_dim=8",
        )
    )

  @pytest.mark.tpu_backend
  def test_diloco_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_diloco.pickle")
    train_compile_main(
        (
            None,
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=tpu7x-8",
            "compile_topology_num_slices=2",
            "ici_fsdp_parallelism=-1",
            "dcn_diloco_parallelism=2",
            "enable_diloco=true",
            "model_name=gemma2-2b",
            "override_model_config=True",
            "base_emb_dim=32",
            "base_num_decoder_layers=1",
            "base_mlp_dim=64",
            "base_num_query_heads=1",
            "base_num_kv_heads=1",
            "head_dim=4",
        )
    )

  @pytest.mark.cpu_only
  @pytest.mark.tpu_backend
  def test_streaming_diloco_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_streaming_diloco.pickle")
    train_compile_main(
        (
            None,
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=tpu7x-8",
            "compile_topology_num_slices=2",
            "ici_fsdp_parallelism=-1",
            "dcn_diloco_parallelism=2",
            "enable_diloco=true",
            "enable_streaming_diloco=true",
            "num_diloco_fragments=2",
            "model_name=gemma2-2b",
            "override_model_config=True",
            "base_emb_dim=32",
            "base_num_decoder_layers=2",
            "base_mlp_dim=64",
            "base_num_query_heads=1",
            "base_num_kv_heads=1",
            "head_dim=4",
        )
    )

  def test_fragmented_tree_manipulator_scanned_filter(self):
    """Tests that parameters matching regex but lacking leading layer dim are NOT marked scanned."""
    num_layers = 4
    config = initialize_pydantic(
        [
            "",
            get_test_config_path(),
            "enable_diloco=true",
            "enable_streaming_diloco=true",
            "num_diloco_fragments=3",
            f"base_num_decoder_layers={num_layers}",
        ]
    )
    # Scanned param has leading dim = num_layers; non-scanned param matches regex name but lacks leading layer dim
    params_tree = {
        "decoder": {
            "layers": jnp.ones((num_layers, 16, 16)),
            "layers_outside_pipeline": jnp.ones((16, 16)),  # Lacks leading layer dim = 4
        }
    }
    manipulator = diloco_utils.FragmentedTreeManipulator.create(params_tree, config)
    # Check that layers_outside_pipeline is NOT treated as scanned
    scanned_map = manipulator.keypath_to_is_scanned
    for keystr, is_scanned in scanned_map.items():
      if "layers_outside_pipeline" in keystr:
        self.assertFalse(is_scanned)
      elif "decoder/layers" in keystr:
        self.assertTrue(is_scanned)

  def test_streaming_diloco_requires_scan_layers(self):
    """Tests that enable_streaming_diloco=True raises ValueError if scan_layers=False."""
    with self.assertRaises(ValueError) as ctx:
      initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "enable_diloco=true",
              "enable_streaming_diloco=true",
              "num_diloco_fragments=2",
              "scan_layers=false",
          ]
      )
    self.assertIn("enable_streaming_diloco=True requires scan_layers=True", str(ctx.exception))

  def test_apply_flat_fragment_shapedtypestruct(self):
    """Tests that FragmentedTreeManipulator handles ShapeDtypeStruct leaves during abstract tracing."""
    num_layers = 2
    config = initialize_pydantic(
        [
            "",
            get_test_config_path(),
            "enable_diloco=true",
            "enable_streaming_diloco=true",
            "num_diloco_fragments=2",
            f"base_num_decoder_layers={num_layers}",
        ]
    )
    abstract_tree = {"decoder": {"layers": jax.ShapeDtypeStruct((num_layers, 8), jnp.float32)}}
    manipulator = diloco_utils.FragmentedTreeManipulator.create(abstract_tree, config)
    frag = manipulator.get_flat_fragment(abstract_tree, fragment_idx=1)
    res = manipulator.apply_flat_fragment(abstract_tree, fragment_idx=1, flat_fragment=frag)
    self.assertIsInstance(res["decoder"]["layers"], jax.ShapeDtypeStruct)

  def test_diloco_requires_pure_nnx(self):
    """Tests that enable_diloco=True raises ValueError if pure_nnx=False."""
    with self.assertRaises(ValueError) as ctx:
      initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "enable_diloco=true",
              "pure_nnx=false",
          ]
      )
    self.assertIn("enable_diloco=True requires pure_nnx=True", str(ctx.exception))

  def test_diloco_checkpoint_saving_and_normal_resume(self):
    """Tests that DiLoCo checkpoints save outer params under params/params for direct normal pre-training resume."""
    temp_dir = tempfile.mkdtemp()
    try:
      rngs = nnx.Rngs(params=jax.random.key(0))
      model = SimpleNNXModel(rngs=rngs)
      tx = optax.adamw(1e-3)
      nnx_train_state = TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

      config = initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "enable_diloco=true",
              "dcn_diloco_parallelism=2",
              "num_diloco_replicas=2",
              f"checkpoint_dir={temp_dir}",
              "enable_checkpointing=true",
          ]
      )
      diloco_state, _ = diloco.build_diloco_state(config, lambda: nnx_train_state)

      mgr = checkpointing.create_orbax_checkpoint_manager(
          checkpoint_dir=temp_dir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=1,
          use_ocdbt=True,
          use_zarr3=True,
      )
      checkpointing.save_checkpoint(mgr, 10, diloco_state, config, force=True)
      mgr.wait_until_finished()

      items_path = os.path.join(temp_dir, "10", "items")

      # 1. Verify DiLoCo self-restoration
      abstract_nnx = nnx.eval_shape(lambda: nnx.state(nnx_train_state))
      restored_diloco = diloco_checkpoint_utils.restore_diloco_checkpoint(
          items_path, abstract_nnx, 96, use_ocdbt=True, use_zarr3=True, config=config
      )
      self.assertIsInstance(restored_diloco, diloco.DiLoCoTrainState)

      # 2. Verify normal pre-training weights-only restoration
      fresh_model = SimpleNNXModel(rngs=nnx.Rngs(params=jax.random.key(1)))
      abstract_params = nnx.split_state(nnx.state(fresh_model), nnx.Param, ...)[0]
      restored_params = checkpointing.load_params_from_path(
          items_path,
          abstract_params,
          checkpoint_storage_concurrent_gb=96,
          use_ocdbt=True,
          use_zarr3=True,
      )
      problems = checkpointing._weight_mismatches(  # pylint: disable=protected-access
          abstract_params.to_pure_dict(), restored_params.to_pure_dict()
      )
      self.assertEqual(len(problems), 0)

      # 3. Verify normal pre-training full state restoration
      normal_config = initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "enable_diloco=false",
              f"load_full_state_path={items_path}",
          ]
      )
      restored_full = checkpointing._load_linen_checkpoint_into_nnx(  # pylint: disable=protected-access
          items_path,
          abstract_nnx,
          checkpoint_storage_concurrent_gb=96,
          use_ocdbt=True,
          use_zarr3=True,
          config=normal_config,
      )
      restored_full_params = nnx.split_state(restored_full["model"], nnx.Param, ...)[0].to_pure_dict()
      problems_full = checkpointing._weight_mismatches(  # pylint: disable=protected-access
          abstract_params.to_pure_dict(), restored_full_params
      )
      self.assertEqual(len(problems_full), 0)

    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  def test_diloco_automatic_checkpoint_resumption(self):
    """Tests that DiLoCo automatically resumes from checkpoint_manager when no explicit paths are given."""
    temp_dir = tempfile.mkdtemp()
    try:
      rngs = nnx.Rngs(params=jax.random.key(0))
      model = SimpleNNXModel(rngs=rngs)
      tx = optax.adamw(1e-3)
      nnx_train_state = TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

      config = initialize_pydantic(
          [
              "",
              get_test_config_path(),
              "enable_diloco=true",
              "dcn_diloco_parallelism=2",
              "num_diloco_replicas=2",
              f"checkpoint_dir={temp_dir}",
              "enable_checkpointing=true",
          ]
      )
      diloco_state, _ = diloco.build_diloco_state(config, lambda: nnx_train_state)
      diloco_state = diloco_state.replace(step=jnp.array(5, dtype=jnp.int32))

      mgr = checkpointing.create_orbax_checkpoint_manager(
          checkpoint_dir=temp_dir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=1,
          use_ocdbt=True,
          use_zarr3=True,
      )
      checkpointing.save_checkpoint(mgr, 5, diloco_state, config, force=True)
      mgr.wait_until_finished()

      # Create new checkpoint manager for resumption (simulating next run with same run_name / checkpoint_dir)
      resume_mgr = checkpointing.create_orbax_checkpoint_manager(
          checkpoint_dir=temp_dir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=1,
          use_ocdbt=True,
          use_zarr3=True,
      )
      latest = checkpointing.latest_step(resume_mgr)
      self.assertEqual(latest, 5)

      # Build abstract state for restoration
      abstract_diloco_state, _ = diloco.build_diloco_state(config, lambda: nnx_train_state)
      abstract_unboxed = nnx.eval_shape(lambda: abstract_diloco_state)

      restored, raw_params = checkpointing.load_state_if_possible(
          resume_mgr,
          None,
          config.load_parameters_path,
          config.load_full_state_path,
          96,
          abstract_unboxed,
          maxtext_config=config,
      )

      self.assertIsNotNone(restored)
      self.assertIsNone(raw_params)
      restored_items = restored["items"]
      self.assertIsInstance(restored_items, diloco.DiLoCoTrainState)
      self.assertEqual(int(restored_items.step), 5)

      # Ensure no unmaterialized ShapeDtypeStruct leaves exist
      has_sds = any(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree_util.tree_leaves(restored_items))
      self.assertFalse(has_sds, "Restored state must not contain ShapeDtypeStruct placeholders!")

      # Verify weights match
      expected_params = nnx.split_state(nnx.state(model), nnx.Param, ...)[0].to_pure_dict()
      diloco_outer_params = restored_items.params
      if isinstance(diloco_outer_params, dict) and "params" in diloco_outer_params:
        diloco_outer_params = diloco_outer_params["params"]
      self.assertEqual(len(checkpointing._weight_mismatches(expected_params, diloco_outer_params)), 0)  # pylint: disable=protected-access
    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)
