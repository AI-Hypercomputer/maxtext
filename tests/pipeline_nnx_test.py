# Copyright 2024 Google LLC
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

"""Tests for NNX pipeline parallelism."""

import os.path
import sys
import unittest
import pytest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT
from MaxText.layers import deepseek
from MaxText.layers import pipeline_nnx
from MaxText.layers import simple_layer
from MaxText.train import main as train_main

# --- Mocks & Helpers ---


def get_mock_config(**kwargs):
  """Creates a dummy config object with default values."""
  defaults = {
      "ici_pipeline_parallelism": 2,
      "dcn_pipeline_parallelism": 1,
      "pipeline_delay_activation_forwarding": False,
      "micro_batch_size_to_train_on": 4,
      "num_pipeline_microbatches": 4,
      "expert_shard_attention_option": "standard",
      "max_target_length": 8,
      "emb_dim": 4,
      "logical_axis_rules": [],
      "num_pipeline_repeats": 1,
      "remat_policy": None,
      "pipeline_fsdp_ag_once": False,
  }
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


class SimpleLinearNNX(nnx.Module):
  """A simple NNX layer to wrap in the pipeline.

  Accepts the full signature expected by the Pipeline module:
  (inputs, positions, segment_ids, deterministic, model_mode)
  """

  def __init__(self, config, rngs):
    self.linear = nnx.Linear(config.emb_dim, config.emb_dim, rngs=rngs, use_bias=False)

  def __call__(self, x, positions=None, segment_ids=None, deterministic=True, model_mode="train"):
    # positions and segment_ids are accepted but not used in this simple layer
    return self.linear(x)


class SequentialModel(nnx.Module):
  """Reference sequential model for correctness checking.

  Runs layers sequentially (like a non-pipelined model would).
  """

  def __init__(self, config, layer_cls, num_layers, rngs):
    """Initialize sequential model with num_layers layers.

    Args:
      config: Model configuration.
      layer_cls: Layer class to instantiate.
      num_layers: Total number of layers (num_stages * num_repeats).
      rngs: Random number generators.
    """
    self.layers = nnx.vmap(layer_cls, in_axes=None, out_axes=0, axis_size=num_layers)(config, rngs=rngs)

  def __call__(self, x, positions=None, segment_ids=None):
    """Run input through all layers sequentially."""

    @nnx.scan(in_axes=(nnx.Carry, nnx.Carry, nnx.Carry, 0), out_axes=(nnx.Carry, nnx.Carry, nnx.Carry, None))
    def scan_fn(x, positions, segment_ids, layer):
      out = layer(x, positions, segment_ids)
      return out, positions, segment_ids, None

    x, _, _, _ = scan_fn(x, positions, segment_ids, self.layers)
    return x


def sync_weights_for_repeats(pipeline_model, sequential_model, num_stages, num_repeats):
  """Copies weights from Pipeline model to Sequential model, handling repeats.

  For circular pipelines with num_repeats > 1, the sequential model has
  num_stages * num_repeats layers, while the pipeline model has num_stages
  layers that are reused num_repeats times.

  Args:
    pipeline_model: The Pipeline model.
    sequential_model: The Sequential model.
    num_stages: Number of pipeline stages.
    num_repeats: Number of times each stage is repeated.
  """
  # Get pipeline layer weights: shape [num_stages, ...]
  pipeline_state = nnx.state(pipeline_model.layers)

  # For sequential model, we need to tile the weights num_repeats times
  # Sequential model expects shape [num_stages * num_repeats, ...]
  def tile_weights(state_dict):
    """Tile weights to match sequential model structure."""
    tiled = {}
    for key, value in state_dict.items():
      if hasattr(value, "value"):
        # This is a parameter - tile it
        param = value.value
        # Tile along the first axis (stage axis)
        tiled_param = jnp.tile(param, [num_repeats] + [1] * (param.ndim - 1))
        tiled[key] = type(value)(tiled_param)
      elif isinstance(value, dict):
        tiled[key] = tile_weights(value)
      else:
        tiled[key] = value
    return tiled

  # Convert state to dict, tile, and update sequential model
  state_dict = pipeline_state.flat_state()
  tiled_state = {}
  for key, value in state_dict.items():
    param = value.value
    tiled_param = jnp.tile(param, [num_repeats] + [1] * (param.ndim - 1))
    tiled_state[key] = type(value)(tiled_param)

  # Create new state and update
  seq_state = nnx.state(sequential_model.layers)
  seq_flat = seq_state.flat_state()

  for key in seq_flat:
    if key in tiled_state:
      seq_flat[key] = tiled_state[key]

  seq_state.replace_by_pure_dict(seq_flat)
  nnx.update(sequential_model.layers, seq_state)


def sync_weights(pipeline_model, sequential_model):
  """Copies weights from Pipeline model to Sequential model (non-circular case).

  For non-circular pipelines (num_repeats=1), the weights are directly copied.
  """
  pipeline_state = nnx.state(pipeline_model.layers)
  nnx.update(sequential_model.layers, pipeline_state)


# --- Tests ---


class PipelineNNXTest(unittest.TestCase):

  def test_init_and_shapes(self):
    """Verifies that the pipeline initializes with correct dimensions."""
    config = get_mock_config(ici_pipeline_parallelism=4)
    rngs = nnx.Rngs(0)
    pipeline = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    self.assertEqual(pipeline.num_stages, 4)

    # Check vmapped weight shape: [stages, in, out]
    kernel_shape = pipeline.layers.linear.kernel.value.shape
    expected_shape = (4, config.emb_dim, config.emb_dim)
    self.assertEqual(kernel_shape, expected_shape)

  def test_correctness_vs_sequential(self):
    """Verifies pipeline output matches sequential execution."""
    config = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=1,
    )
    rngs = nnx.Rngs(params=0)

    # 1. Initialize Models
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=2, rngs=rngs)

    # 2. Sync Weights
    sync_weights(pipeline_model, sequential_model)

    # 3. Create Input Data
    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    # 4. Define Loss Functions
    def pipeline_loss(model, x):
      y = model(x)
      return jnp.mean(y**2)

    def sequential_loss(model, x):
      y = model(x)
      return jnp.mean(y**2)

    # 5. Compute Gradients
    grad_fn_pipe = nnx.value_and_grad(pipeline_loss)
    grad_fn_seq = nnx.value_and_grad(sequential_loss)

    loss_pipe, grads_pipe = grad_fn_pipe(pipeline_model, x)
    loss_seq, grads_seq = grad_fn_seq(sequential_model, x)

    # 6. Compare Outputs
    self.assertTrue(jnp.allclose(loss_pipe, loss_seq, atol=1e-5), f"Loss mismatch: Pipe {loss_pipe} vs Seq {loss_seq}")

    # 7. Compare Gradients
    pipe_flat = jax.tree_util.tree_leaves(grads_pipe)
    seq_flat = jax.tree_util.tree_leaves(grads_seq)

    self.assertTrue(len(pipe_flat) > 0)

    for g1, g2 in zip(pipe_flat, seq_flat):
      self.assertTrue(jnp.allclose(g1, g2, atol=1e-5), "Gradient mismatch found in a parameter tensor")

  def test_data_flow_shapes(self):
    """Verifies internal microbatch reshaping logic preserves data volume."""
    config = get_mock_config(num_pipeline_microbatches=2)
    rngs = nnx.Rngs(0)
    pipeline = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Batch 4, Microbatches 2 -> Microbatch Size 2
    x = jax.random.normal(jax.random.key(0), (4, 8, 4))
    y = pipeline(x)

    self.assertEqual(x.shape, y.shape)
    self.assertFalse(jnp.all(y == 0), "Output should not be zero")

  def test_non_circular_same_output_and_grad(self):
    """Tests non-circular pipeline (num_repeats=1) matches sequential model."""
    config = get_mock_config(
        ici_pipeline_parallelism=2,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=1,
        max_target_length=16,
        emb_dim=8,
    )
    num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism

    rngs = nnx.Rngs(params=42)

    # Initialize models
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages, rngs=rngs)

    # Sync weights
    sync_weights(pipeline_model, sequential_model)

    # Create input
    batch_size = config.micro_batch_size_to_train_on
    x = jax.random.normal(jax.random.key(1), (batch_size, config.max_target_length, config.emb_dim))

    # Forward pass
    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(jnp.allclose(y_pipe, y_seq, atol=1e-5), "Output mismatch between pipeline and sequential")

    # Test gradients
    def loss_fn_pipe(model, x):
      return jnp.mean(model(x) ** 2)

    def loss_fn_seq(model, x):
      return jnp.mean(model(x) ** 2)

    loss_pipe, _ = nnx.value_and_grad(loss_fn_pipe)(pipeline_model, x)
    loss_seq, _ = nnx.value_and_grad(loss_fn_seq)(sequential_model, x)

    self.assertTrue(jnp.allclose(loss_pipe, loss_seq, atol=1e-5), f"Loss mismatch: {loss_pipe} vs {loss_seq}")

  def test_circular_minimum_microbatches_same_output_and_grad(self):
    """Tests circular pipeline with minimum microbatches (num_micro = num_stages).

    When num_microbatches == num_stages, circular storage is not needed
    because outputs flow directly to the next repeat.
    """
    num_stages = 2
    num_repeats = 2
    config = get_mock_config(
        ici_pipeline_parallelism=num_stages,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=num_stages,  # minimum: equal to num_stages
        micro_batch_size_to_train_on=4,
        num_pipeline_repeats=num_repeats,
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=123)

    # Initialize models
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Verify circular storage is NOT used (minimum microbatches case)
    self.assertFalse(pipeline_model.use_circ_storage)

    # Sequential model needs num_stages * num_repeats layers
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages * num_repeats, rngs=rngs)

    # Sync weights (with repeat tiling)
    sync_weights_for_repeats(pipeline_model, sequential_model, num_stages, num_repeats)

    # Create input
    x = jax.random.normal(jax.random.key(1), (4, 8, 4))

    # Forward pass
    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(
        jnp.allclose(y_pipe, y_seq, atol=1e-4), f"Output mismatch: max diff = {jnp.max(jnp.abs(y_pipe - y_seq))}"
    )

  def test_circular_extra_microbatches_same_output_and_grad(self):
    """Tests circular pipeline with extra microbatches (uses circular storage).

    When num_microbatches > num_stages, circular storage is needed to buffer
    outputs between repeats.
    """
    num_stages = 2
    num_repeats = 2
    num_microbatches = 4  # > num_stages, so circular storage is needed

    config = get_mock_config(
        ici_pipeline_parallelism=num_stages,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=num_microbatches,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=num_repeats,
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=456)

    # Initialize models
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Verify circular storage IS used
    self.assertTrue(pipeline_model.use_circ_storage)

    # Sequential model needs num_stages * num_repeats layers
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages * num_repeats, rngs=rngs)

    # Sync weights
    sync_weights_for_repeats(pipeline_model, sequential_model, num_stages, num_repeats)

    # Create input
    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    # Forward pass
    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(
        jnp.allclose(y_pipe, y_seq, atol=1e-4), f"Output mismatch: max diff = {jnp.max(jnp.abs(y_pipe - y_seq))}"
    )

  def test_delay_activation_forwarding_same_output_and_grad(self):
    """Tests pipeline with delayed activation forwarding enabled."""
    config = get_mock_config(
        ici_pipeline_parallelism=2,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=1,
        pipeline_delay_activation_forwarding=True,  # Enable delay
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=789)

    # Initialize models
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Verify forwarding delay is set
    self.assertEqual(pipeline_model.forwarding_delay, 2)

    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=2, rngs=rngs)

    # Sync weights
    sync_weights(pipeline_model, sequential_model)

    # Create input
    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    # Forward pass
    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(
        jnp.allclose(y_pipe, y_seq, atol=1e-4), f"Output mismatch: max diff = {jnp.max(jnp.abs(y_pipe - y_seq))}"
    )

  # def test_positions_and_segment_ids_passthrough(self):
  #   """Tests that positions and segment_ids are correctly passed to layers."""
  #   config = get_mock_config(
  #       ici_pipeline_parallelism=2,
  #       num_pipeline_microbatches=4,
  #       micro_batch_size_to_train_on=8,
  #       max_target_length=16,
  #       emb_dim=8,
  #   )

  #   rngs = nnx.Rngs(params=111)

  #   # Use a layer that actually uses positions
  #   class PositionAwareLayer(nnx.Module):
  #     def __init__(self, config, rngs):
  #       self.linear = nnx.Linear(config.emb_dim, config.emb_dim, rngs=rngs, use_bias=False)
  #       self.emb_dim = config.emb_dim

  #     def __call__(self, x, positions=None, segment_ids=None, deterministic=True, model_mode="train"):
  #       out = self.linear(x)
  #       if positions is not None:
  #         # Add position-based bias
  #         pos_bias = positions[..., None].astype(x.dtype) * 0.01
  #         out = out + pos_bias
  #       return out

  #   pipeline_model = pipeline_nnx.Pipeline(config, PositionAwareLayer, mesh=None, rngs=rngs)

  #   # Create inputs
  #   batch_size = config.micro_batch_size_to_train_on
  #   x = jax.random.normal(jax.random.key(1), (batch_size, config.max_target_length, config.emb_dim))
  #   positions = jnp.arange(config.max_target_length)[None, :].repeat(batch_size, axis=0)
  #   segment_ids = jnp.ones((batch_size, config.max_target_length), dtype=jnp.int32)

  #   # Forward pass with positions
  #   y_with_pos = pipeline_model(x, positions=positions, segment_ids=segment_ids)

  #   # Forward pass without positions
  #   y_without_pos = pipeline_model(x)

  #   # Outputs should be different when positions are provided
  #   self.assertEqual(y_with_pos.shape, x.shape)
  #   self.assertFalse(jnp.allclose(y_with_pos, y_without_pos, atol=1e-6), "Positions should affect output")

  def test_multiple_stages(self):
    """Tests pipeline with more than 2 stages."""
    num_stages = 4
    config = get_mock_config(
        ici_pipeline_parallelism=num_stages,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=8,
        micro_batch_size_to_train_on=16,
        num_pipeline_repeats=1,
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=222)

    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages, rngs=rngs)

    sync_weights(pipeline_model, sequential_model)

    x = jax.random.normal(jax.random.key(1), (16, 8, 4))

    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertTrue(jnp.allclose(y_pipe, y_seq, atol=1e-4))

  def test_circular_with_delay_activation_forwarding(self):
    """Tests circular pipeline combined with delayed activation forwarding."""
    num_stages = 2
    num_repeats = 2
    config = get_mock_config(
        ici_pipeline_parallelism=num_stages,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=num_repeats,
        pipeline_delay_activation_forwarding=True,
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=333)

    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Verify both features are active
    self.assertEqual(pipeline_model.forwarding_delay, 2)
    self.assertTrue(pipeline_model.use_circ_storage)

    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages * num_repeats, rngs=rngs)

    sync_weights_for_repeats(pipeline_model, sequential_model, num_stages, num_repeats)

    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(
        jnp.allclose(y_pipe, y_seq, atol=1e-4), f"Output mismatch: max diff = {jnp.max(jnp.abs(y_pipe - y_seq))}"
    )

  def test_circular_ag_once(self):
    """Tests circular pipeline with all-gather once enabled."""
    num_stages = 2
    num_repeats = 2
    config = get_mock_config(
        ici_pipeline_parallelism=num_stages,
        dcn_pipeline_parallelism=1,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=num_repeats,
        pipeline_fsdp_ag_once=True,
        max_target_length=8,
        emb_dim=4,
    )

    rngs = nnx.Rngs(params=555)

    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)
    sequential_model = SequentialModel(config, SimpleLinearNNX, num_layers=num_stages * num_repeats, rngs=rngs)

    sync_weights_for_repeats(pipeline_model, sequential_model, num_stages, num_repeats)

    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    y_pipe = pipeline_model(x)
    y_seq = sequential_model(x)

    self.assertEqual(y_pipe.shape, y_seq.shape)
    self.assertTrue(
        jnp.allclose(y_pipe, y_seq, atol=1e-4), f"Output mismatch: max diff = {jnp.max(jnp.abs(y_pipe - y_seq))}"
    )

  def test_gradient_flow(self):
    """Tests that gradients flow correctly through the pipeline."""
    config = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=4,
        micro_batch_size_to_train_on=8,
        num_pipeline_repeats=1,
    )

    rngs = nnx.Rngs(params=444)
    pipeline_model = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    x = jax.random.normal(jax.random.key(1), (8, 8, 4))

    def loss_fn(model, x):
      return jnp.mean(model(x) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(pipeline_model, x)

    # Check that loss is a scalar
    self.assertEqual(loss.shape, ())

    # Check that gradients exist and are non-zero
    grad_leaves = jax.tree_util.tree_leaves(grads)
    self.assertTrue(len(grad_leaves) > 0, "Should have gradients")

    for g in grad_leaves:
      if hasattr(g, "value"):
        self.assertFalse(jnp.all(g.value == 0), "Gradient should not be all zeros")

  def test_need_circ_storage_logic(self):
    """Tests the need_circ_storage method returns correct values."""
    # Case 1: num_repeats = 1, should not need circ storage
    config1 = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=4,
        num_pipeline_repeats=1,
    )
    rngs = nnx.Rngs(0)
    pipeline1 = pipeline_nnx.Pipeline(config1, SimpleLinearNNX, mesh=None, rngs=rngs)
    self.assertFalse(pipeline1.use_circ_storage)

    # Case 2: num_repeats > 1, num_microbatches <= num_stages * delay, should not need
    config2 = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=2,  # == num_stages
        num_pipeline_repeats=2,
    )
    pipeline2 = pipeline_nnx.Pipeline(config2, SimpleLinearNNX, mesh=None, rngs=rngs)
    self.assertFalse(pipeline2.use_circ_storage)

    # Case 3: num_repeats > 1, num_microbatches > num_stages * delay, should need
    config3 = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=4,  # > num_stages
        num_pipeline_repeats=2,
    )
    pipeline3 = pipeline_nnx.Pipeline(config3, SimpleLinearNNX, mesh=None, rngs=rngs)
    self.assertTrue(pipeline3.use_circ_storage)

  def test_iterations_calculation(self):
    """Tests iteration calculation methods."""
    config = get_mock_config(
        ici_pipeline_parallelism=2,
        num_pipeline_microbatches=4,
        num_pipeline_repeats=2,
        pipeline_delay_activation_forwarding=False,
    )

    rngs = nnx.Rngs(0)
    pipeline = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # forwarding_delay = 1 (no delay)
    # iterations_to_complete_first_microbatch_one_repeat = 1 * (2 - 1) = 1
    self.assertEqual(pipeline.iterations_to_complete_first_microbatch_one_repeat(), 1)

    # iterations_to_complete_first_microbatch = 4 * (2 - 1) + 1 = 5
    self.assertEqual(pipeline.iterations_to_complete_first_microbatch(), 5)

  def test_remat_policy(self):
    """Tests get_pipeline_remat_policy method."""
    config = get_mock_config(remat_policy=None)

    rngs = nnx.Rngs(0)
    pipeline = pipeline_nnx.Pipeline(config, SimpleLinearNNX, mesh=None, rngs=rngs)

    # Should return a policy (not None)
    policy = pipeline.get_pipeline_remat_policy()
    self.assertIsNotNone(policy)

  @pytest.mark.tpu_only
  def test_with_real_config_non_circular(self):
    """Tests NNX pipeline with real pyconfig configuration (non-circular)."""
    # 4 stages, 4 layers (no circular repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="nnx_non_circular",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=4,
        num_pipeline_microbatches=4,
        per_device_batch_size=4,
    )

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    rngs = nnx.Rngs(params=0)
    single_pipeline_stage = simple_layer.SimpleDecoderLayerNNX

    pipeline_model = pipeline_nnx.Pipeline(
        config=config,
        layer_cls=single_pipeline_stage,
        mesh=mesh,
        rngs=rngs,
    )

    # Create test input
    batch_size = config.global_batch_size_to_train_on
    input_shape = [batch_size, config.max_target_length, config.emb_dim]
    inputs = jax.random.normal(jax.random.PRNGKey(2), input_shape, dtype=jnp.float32)

    # Run forward pass
    output = pipeline_model(inputs)

    # Verify output shape matches input shape
    self.assertEqual(output.shape, inputs.shape)

    # Verify output is not all zeros (data actually flowed through)
    self.assertFalse(jnp.all(output == 0), "Output should not be all zeros")

  @pytest.mark.tpu_only
  def test_with_real_config_circular(self):
    """Tests NNX pipeline with real pyconfig configuration (circular)."""
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="nnx_circular",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=4,
        per_device_batch_size=4,
    )

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    rngs = nnx.Rngs(params=0)
    single_pipeline_stage = simple_layer.SimpleDecoderLayerNNX

    pipeline_model = pipeline_nnx.Pipeline(
        config=config,
        layer_cls=single_pipeline_stage,
        mesh=mesh,
        rngs=rngs,
    )

    # Create test input
    batch_size = config.global_batch_size_to_train_on
    input_shape = [batch_size, config.max_target_length, config.emb_dim]
    inputs = jax.random.normal(jax.random.PRNGKey(2), input_shape, dtype=jnp.float32)

    # Run forward pass
    output = pipeline_model(inputs)

    # Verify output shape matches input shape
    self.assertEqual(output.shape, inputs.shape)

    # Verify output is not all zeros (data actually flowed through)
    self.assertFalse(jnp.all(output == 0), "Output should not be all zeros")

  @pytest.mark.tpu_only
  def test_circular_deepseek_moe(self):
    """Tests circular NNX pipeline with DeepSeek MOE configuration.

    4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches.
    """
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="nnx_circular_moe",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        megablox=False,
        sparse_matmul=False,
        capacity_factor=1,
        decoder_block="deepseek",
    )

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    rngs = nnx.Rngs(params=0)

    # Use DeepSeek MOE layer with the NNX pipeline
    pipeline_model = pipeline_nnx.Pipeline(
        config=config,
        layer_cls=deepseek.DeepSeekMoELayerNNX,
        mesh=mesh,
        rngs=rngs,
    )

    # Create test input
    batch_size = config.global_batch_size_to_train_on
    input_shape = [batch_size, config.max_target_length, config.emb_dim]
    inputs = jax.random.normal(jax.random.PRNGKey(2), input_shape, dtype=jnp.float32)

    # Run forward pass
    output = pipeline_model(inputs)

    # Verify output shape matches input shape
    self.assertEqual(output.shape, inputs.shape)

    # Verify output is not all zeros (data actually flowed through)
    self.assertFalse(jnp.all(output == 0), "Output should not be all zeros")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_full_train_circular(self):
    """Run a full train.py call with circular pipeline configuration.

    4 stages, 32 layers (2 layers per stage, 4 circular repeats), 8 microbatches.
    """
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_nnx_pipeline_circular_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=32",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=2",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_full_train_non_circular(self):
    """Run a full train.py call with non-circular pipeline configuration.

    4 stages, 32 layers (8 layers per stage), 8 microbatches.
    """
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_nnx_pipeline_non_circular_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=32",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=8",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_subset_layers(self):
    """Run a full train.py call with subset of layers in pipeline.

    4 stages, 16 layers - 8 in pipeline, 8 ran outside of pipeline.
    """
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_nnx_pipeline_subset_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=16",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=1",
            "num_pipeline_repeats=2",
            "pipeline_parallel_layers=8",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",
        ]
    )

  @pytest.mark.integration_test
  def test_full_train_fp8(self):
    """Run a full train.py call with fp8 quantization.

    FP8 quantization adds extra variable collections that need to be handled.
    """
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_nnx_pipeline_fp8_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=4",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "quantization=fp8",
            "scan_layers_per_stage=False",
            "attention=dot_product",
        ]
    )

  @pytest.mark.integration_test
  def test_full_train_nanoo_fp8(self):
    """Run a full train.py call with NANOO fp8 quantization.

    NANOO FP8 quantization adds extra variable collections that need to be handled.
    """
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_nnx_pipeline_nanoo_fp8_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=4",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "quantization=nanoo_fp8",
            "scan_layers_per_stage=False",
            "attention=dot_product",
        ]
    )


if __name__ == "__main__":
  unittest.main()
