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
"""multi_token_prediction_test"""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.layers import multi_token_prediction  # The class under test
from maxtext.layers import embeddings
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.common.common_types import Config
from maxtext.layers.nnx_decoders import NNXDecoderLayer
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils

from tests.utils.test_helpers import get_test_config_path


TEST_LAYER_NUM = 1


class MultiTokenPredictionLayerTest(unittest.TestCase):
  """Unit tests for the standalone MultiTokenPredictionLayer."""

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="multi_token_prediction_layer_test",
        skip_jax_distributed_system=True,
        per_device_batch_size=8,
        base_emb_dim=16,
        base_mlp_dim=32,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        head_dim=8,
        max_target_length=128,
        vocab_size=128,
    )
    self.rng = jax.random.PRNGKey(42)  # Base RNG for setup
    self.rngs = nnx.Rngs(params=self.rng, dropout=self.rng)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    self.mtp_layer = multi_token_prediction.MultiTokenPredictionLayer(
        config=self.cfg,
        mesh=self.mesh,
        layer_number=TEST_LAYER_NUM,
        transformer_layer_module=NNXDecoderLayer,
        rngs=self.rngs,
    )

    # Dimensions directly from the config object
    self.batch_size = int(self.cfg.per_device_batch_size)
    self.seq_len = self.cfg.max_target_length
    self.embed_dim = self.cfg.base_emb_dim

    # Prepare Dummy Input Data
    prev_hidden_state_shape = (self.batch_size, self.seq_len, self.embed_dim)
    target_embedding_shape = (self.batch_size, self.seq_len, self.embed_dim)
    data_rng1, data_rng2, _ = jax.random.split(self.rng, 3)

    self.prev_hidden_state = jax.random.normal(data_rng1, prev_hidden_state_shape, dtype=self.cfg.dtype)
    self.target_token_embedding = jax.random.normal(data_rng2, target_embedding_shape, dtype=self.cfg.dtype)
    self.position_ids = jnp.arange(self.seq_len, dtype=jnp.int32).reshape(1, -1).repeat(self.batch_size, axis=0)
    # Simulate a simple case with no padding.
    self.decoder_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    max_logging.log("Setup complete.")

  def test_multi_token_prediction_layer_output(self):
    """Tests the basic forward pass and output shape of MultiTokenPredictionLayer."""

    output_hidden_state = self.mtp_layer(
        self.prev_hidden_state,
        self.target_token_embedding,
        position_ids=self.position_ids,
        decoder_segment_ids=self.decoder_segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    # Assertions using unittest methods
    expected_output_shape = (self.batch_size, self.seq_len, self.embed_dim)

    # Check shape
    self.assertEqual(
        output_hidden_state.shape,
        expected_output_shape,
        f"Expected output shape {expected_output_shape}, but got {output_hidden_state.shape}",
    )
    # TODO(@parambole) to check the fixed inputs in the unit test with expected values
    # Check dtype
    self.assertEqual(
        output_hidden_state.dtype,
        self.cfg.dtype,
        f"Expected output dtype {self.cfg.dtype}, but got {output_hidden_state.dtype}",
    )

    # Check for NaNs/Infs
    self.assertFalse(jnp.isnan(output_hidden_state).any(), "Output contains NaNs")
    self.assertFalse(jnp.isinf(output_hidden_state).any(), "Output contains Infs")

    max_logging.log("\nMultiTokenPredictionLayer unittest-style test passed!")
    max_logging.log(f"  Config Batch: {self.batch_size}, SeqLen: {self.seq_len}, EmbedDim: {self.embed_dim}")
    max_logging.log(f"  Output shape: {output_hidden_state.shape}")


# A lightweight wrapper model for robustly testing the MTPBlock.
class MTPBlockTestModel(nnx.Module):
  """A lightweight wrapper model for testing the MTPBlock."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    """Initializes the MTP block and its dependencies for the test."""
    self.config = config
    self.mesh = mesh
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    self._shared_embedding = embeddings.Embed(
        num_embeddings=self.config.vocab_size,
        num_features=self.config.base_emb_dim,
        config=self.config,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    class MockDecoderForMTP:
      """A mock decoder that simulates the behavior needed by MTPBlock."""

      def __init__(self, config: Config):
        self.config = config
        self.model_mode = MODEL_MODE_TRAIN

      def _apply_embedding(self, _shared_embedding, input_ids, _position_ids, _deterministic, model_mode):
        """Returns a zero tensor with the correct embedding shape."""
        batch_size, seq_len = input_ids.shape
        embed_dim = self.config.base_emb_dim
        return jnp.zeros((batch_size, seq_len, embed_dim), dtype=self.config.dtype)

      def apply_output_head(self, _shared_embedding, hidden_state, _deterministic, model_mode):
        """Returns a zero tensor with the correct logit shape."""
        batch_size, seq_len, _ = hidden_state.shape
        return jnp.zeros((batch_size, seq_len, self.config.vocab_size), dtype=self.config.dtype)

    self.decoder = MockDecoderForMTP(config=self.config)

    self.mtp_block = multi_token_prediction.MultiTokenPredictionBlock(
        config=self.config,
        mesh=self.mesh,
        transformer_layer_module=NNXDecoderLayer,
        decoder=self.decoder,
        rngs=self.rngs,
    )

  def __call__(
      self,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      *,
      position_ids,
      decoder_segment_ids,
      model_mode,
      deterministic,
      mutable=None,
  ):
    return self.mtp_block(
        self._shared_embedding,
        main_hidden_state,
        input_ids,
        target_ids,
        target_mask,
        position_ids=position_ids,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        deterministic=deterministic,
    )

  def shared_embedding(self):
    """Returns the shared embedding."""
    return self._shared_embedding


class MultiTokenPredictionBlockTest(unittest.TestCase):
  """Unit tests for the MultiTokenPredictionBlock."""

  def setUp(self):
    super().setUp()
    num_devices = jax.device_count()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="mtp_block_test",
        skip_jax_distributed_system=True,
        mtp_num_layers=2,
        base_emb_dim=16,
        base_mlp_dim=32,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        head_dim=8,
        max_target_length=128,
        vocab_size=128,
    )
    self.nnx_rngs = nnx.Rngs(params=0)
    self.rng = jax.random.PRNGKey(43)
    self.rngs = nnx.Rngs(params=self.rng, dropout=self.rng)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    data_rng, self.init_rng = jax.random.split(self.rng)

    self.batch_size, self.seq_len, self.embed_dim = num_devices, 8, self.cfg.base_emb_dim
    key1, key2, key3 = jax.random.split(data_rng, 3)
    self.main_hidden_state = jax.random.normal(key1, (self.batch_size, self.seq_len, self.embed_dim))
    self.input_ids = jax.random.randint(key2, (self.batch_size, self.seq_len), 0, self.cfg.vocab_size)
    self.target_ids = jax.random.randint(key3, (self.batch_size, self.seq_len), 0, self.cfg.vocab_size)
    self.target_mask = jnp.ones_like(self.target_ids)
    self.position_ids = jnp.arange(self.seq_len, dtype=jnp.int32).reshape(1, -1).repeat(self.batch_size, axis=0)
    self.decoder_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    self.test_model = MTPBlockTestModel(
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def test_sow_functionality(self):
    """Verifies that the block correctly sows losses and weights."""
    _ = self.test_model(
        main_hidden_state=self.main_hidden_state,
        input_ids=self.input_ids,
        target_ids=self.target_ids,
        target_mask=self.target_mask,
        position_ids=self.position_ids,
        decoder_segment_ids=self.decoder_segment_ids,
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )
    state = nnx.state(self.test_model)

    # Check for the existence of the 'losses' and 'weights' attributes.
    self.assertTrue(hasattr(state.mtp_block, "losses"))
    self.assertTrue(hasattr(state.mtp_block, "weights"))

    # Access the actual data tuple inside the .value attribute.
    losses_val = state.mtp_block.losses[...]
    weights_val = state.mtp_block.weights[...]

    self.assertEqual(len(losses_val), self.cfg.mtp_num_layers)
    self.assertEqual(len(weights_val), self.cfg.mtp_num_layers)

  def test_loss_aggregation_logic(self):
    """
    Tests the full 'sow and reap' cycle, mimicking the logic from train.py
    to ensure the final loss calculation is correct.
    """
    # Run the forward pass and capture the sown variables.
    _ = self.test_model(
        main_hidden_state=self.main_hidden_state,
        input_ids=self.input_ids,
        target_ids=self.target_ids,
        target_mask=self.target_mask,
        position_ids=self.position_ids,
        decoder_segment_ids=self.decoder_segment_ids,
        model_mode=MODEL_MODE_TRAIN,
        deterministic=False,
    )
    state = nnx.state(self.test_model)

    # This section of the test now *becomes* the logic from train.py
    # -------------------------------------------------------------
    final_loss_for_gradient = 100.0  # A dummy main loss
    mtp_loss_for_logging = 0.0

    # Use the standard utility to get the data.
    mtp_losses_var = getattr(state.mtp_block, "losses", None)
    mtp_weights_var = getattr(state.mtp_block, "weights", None)

    # Perform the aggregation logic exactly as in `loss_fn`.
    if mtp_losses_var and mtp_weights_var:
      sum_of_all_mtp_losses = jnp.sum(jnp.array(mtp_losses_var[...]))
      sum_of_all_mtp_weights = jnp.sum(jnp.array(mtp_weights_var[...]))

      self.assertGreater(sum_of_all_mtp_weights, 0)

      avg_mtp_loss = sum_of_all_mtp_losses / (sum_of_all_mtp_weights + 1e-8)
      scaled_mtp_loss = avg_mtp_loss * self.cfg.mtp_loss_scaling_factor

      final_loss_for_gradient += scaled_mtp_loss
      mtp_loss_for_logging = scaled_mtp_loss
    # -------------------------------------------------------------

    # Assert that the final values are correct.
    # The final loss should have increased from its base value.
    self.assertGreater(final_loss_for_gradient, 100.0)
    # The logged MTP loss should be a valid, positive number.
    self.assertGreater(mtp_loss_for_logging, 0.0)
    self.assertFalse(jnp.isnan(mtp_loss_for_logging).any())


class TestRollAndMask(unittest.TestCase):
  """Test class for utility functions supporting Roll and Mask."""

  def test_mtp_roll_and_mask_shapes(self):
    """
    Validates that roll_and_mask works correctly on the specific tensor shapes
    that will be passed during training. The primary use case involves tensors
    with a [batch, sequence_length] shape.
    """
    batch_size = 4
    seq_len = 8
    # Create a dummy input tensor that mimics `input_ids` or `target_ids`.
    # The values are sequential for easy validation.
    # Shape: [4, 8]
    input_tensor = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape((batch_size, seq_len))

    # --- Test Case 1: Default left shift by 1 ---
    # This is the most common operation inside the MTP loop.
    rolled_by_1 = multi_token_prediction.roll_and_mask(input_tensor, shift=-1)

    # Manually construct the expected output using jnp
    expected_1 = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 0],  # First row rolled left, last element masked
            [9, 10, 11, 12, 13, 14, 15, 0],  # Second row rolled left
            [17, 18, 19, 20, 21, 22, 23, 0],
            [25, 26, 27, 28, 29, 30, 31, 0],
        ],
        dtype=jnp.int32,
    )

    self.assertEqual(
        rolled_by_1.shape,
        (batch_size, seq_len),
        "Shape should be preserved after rolling.",
    )
    self.assertTrue(
        jnp.array_equal(rolled_by_1, expected_1),
        "Array content is incorrect after shift by -1.",
    )

    # --- Test Case 2: Larger left shift by 3 ---
    # This simulates a later step in a hypothetical MTP loop.
    rolled_by_3 = multi_token_prediction.roll_and_mask(input_tensor, shift=-3)

    # Manually construct the expected output using jnp
    expected_3 = jnp.array(
        [
            [3, 4, 5, 6, 7, 0, 0, 0],  # First row rolled left by 3, last 3 masked
            [11, 12, 13, 14, 15, 0, 0, 0],
            [19, 20, 21, 22, 23, 0, 0, 0],
            [27, 28, 29, 30, 31, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    self.assertEqual(
        rolled_by_3.shape,
        (batch_size, seq_len),
        "Shape should be preserved after rolling.",
    )
    self.assertTrue(
        jnp.array_equal(rolled_by_3, expected_3),
        "Array content is incorrect after shift by -3.",
    )

    # --- Test Case 3: Shift of 0 (edge case) ---
    # This should result in no change to the tensor.
    rolled_by_0 = multi_token_prediction.roll_and_mask(input_tensor, shift=0)
    self.assertTrue(jnp.array_equal(rolled_by_0, input_tensor), "A shift of 0 should be a no-op.")


class TestRollAndMaskBySegment(unittest.TestCase):
  """Unit tests for roll_and_mask_by_segment."""

  def setUp(self):
    super().setUp()
    self.seq_len = 8

  def _make_data(self, values, dtype=jnp.int32):
    return jnp.array(values, dtype=dtype).reshape((1, -1))

  def test_no_segment_ids_falls_back_to_roll_and_mask(self):
    """When segment_ids is None, behaves exactly like roll_and_mask."""
    x = self._make_data([10, 20, 30, 40, 50, 60, 70, 80])
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=None)
    expected = multi_token_prediction.roll_and_mask(x, shift=-1)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_single_segment_no_boundaries(self):
    """With a single segment, only the last position is masked (same as no-packing)."""
    x = self._make_data([10, 20, 30, 40, 50, 60, 70, 80])
    seg = self._make_data([1, 1, 1, 1, 1, 1, 1, 1])
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    expected = self._make_data([20, 30, 40, 50, 60, 70, 80, 0])
    self.assertTrue(jnp.array_equal(result, expected), f"Got {result}")

  def test_two_segments_masks_boundary(self):
    """Cross-segment boundary positions are zeroed out."""
    # Doc A: positions 0-3, Doc B: positions 4-7
    x = self._make_data([10, 20, 30, 40, 50, 60, 70, 80])
    seg = self._make_data([1, 1, 1, 1, 2, 2, 2, 2])
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    # Position 3 (end of doc A) should be 0 because next token is cross-doc.
    # Position 7 (end of doc B) should be 0 because it's the last position.
    expected = self._make_data([20, 30, 40, 0, 60, 70, 80, 0])
    self.assertTrue(jnp.array_equal(result, expected), f"Got {result}")

  def test_padding_segment_masked(self):
    """Positions with segment_id == 0 (padding) are masked."""
    x = self._make_data([10, 20, 30, 40, 0, 0, 0, 0])
    seg = self._make_data([1, 1, 1, 1, 0, 0, 0, 0])
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    # Position 3 (end of doc) → 0; positions 4-6 (padding) → 0
    expected = self._make_data([20, 30, 40, 0, 0, 0, 0, 0])
    self.assertTrue(jnp.array_equal(result, expected), f"Got {result}")

  def test_three_segments_boundaries(self):
    """Three segments: boundaries at doc ends are all masked."""
    x = self._make_data([10, 20, 30, 40, 50, 60, 70, 80])
    seg = self._make_data([1, 1, 2, 2, 2, 3, 3, 3])
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    # Boundaries at positions 1 (end of seg 1), 4 (end of seg 2), 7 (end of seg 3)
    expected = self._make_data([20, 0, 40, 50, 0, 70, 80, 0])
    self.assertTrue(jnp.array_equal(result, expected), f"Got {result}")

  def test_2d_tensor_shape_preserved(self):
    """Shape is preserved after segment-aware roll."""
    batch, seq = 3, 16
    x = jnp.arange(batch * seq).reshape((batch, seq))
    seg = jnp.ones((batch, seq), dtype=jnp.int32)
    seg = seg.at[:, 8:].set(2)  # split each row into two segments
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    self.assertEqual(result.shape, (batch, seq))

  def test_3d_tensor_shape_preserved(self):
    """3D tensors (e.g., embeddings) are handled correctly."""
    batch, seq, feat = 2, 10, 4
    x = jnp.arange(batch * seq * feat, dtype=jnp.float32).reshape((batch, seq, feat))
    seg = jnp.ones((batch, seq), dtype=jnp.int32)
    seg = seg.at[:, 5:].set(2)
    result = multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg)
    self.assertEqual(result.shape, (batch, seq, feat))
    # Boundary positions should be all-zero.
    self.assertTrue(jnp.all(result[:, 4, :] == 0), "Cross-segment boundary should be zero")
    self.assertTrue(jnp.all(result[:, 9, :] == 0), "Last position should be zero")
    # Non-boundary positions should be non-zero (shifted).
    self.assertTrue(jnp.all(result[:, 0, :] != 0), "First position should not be masked")

  def test_shift_not_minus_one_raises(self):
    """Only shift=-1 is supported."""
    x = self._make_data([10, 20, 30, 40])
    seg = self._make_data([1, 1, 1, 1])
    with self.assertRaises(AssertionError):
      multi_token_prediction.roll_and_mask_by_segment(x, segment_ids=seg, shift=-2)


if __name__ == "__main__":
  unittest.main()
