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
""" multi_token_prediction_test """

import os.path
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from MaxText import max_logging, pyconfig
from MaxText import maxtext_utils
from MaxText.globals import PKG_DIR
from MaxText.layers import models
from MaxText.layers import multi_token_prediction  # The class under test

TEST_LAYER_NUM = 1


class MultiTokenPredictionLayerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="multi_token_prediction_layer_test",
        skip_jax_distributed_system=True,
    )
    self.rng = jax.random.PRNGKey(42)  # Base RNG for setup
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    # Instantiate the Layer
    self.mtp_layer = multi_token_prediction.MultiTokenPredictionLayer(
        config=self.cfg, mesh=self.mesh, layer_number=TEST_LAYER_NUM, transformer_layer_module=models.DecoderLayer
    )

    # Dimensions directly from the config object
    self.batch_size = int(self.cfg.per_device_batch_size)
    self.seq_len = self.cfg.max_target_length
    self.embed_dim = self.cfg.base_emb_dim

    # Prepare Dummy Input Data
    prev_hidden_state_shape = (self.batch_size, self.seq_len, self.embed_dim)
    target_embedding_shape = (self.batch_size, self.seq_len, self.embed_dim)
    data_rng1, data_rng2, init_rng = jax.random.split(self.rng, 3)

    self.prev_hidden_state = jax.random.normal(data_rng1, prev_hidden_state_shape, dtype=self.cfg.dtype)
    self.target_token_embedding = jax.random.normal(data_rng2, target_embedding_shape, dtype=self.cfg.dtype)
    self.position_ids = jnp.arange(self.seq_len, dtype=jnp.int32).reshape(1, -1).repeat(self.batch_size, axis=0)
    # Simulate a simple case with no padding.
    self.decoder_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    # Initialize Layer Parameters
    init_rngs = {"params": init_rng}
    self.variables = self.mtp_layer.init(
        init_rngs,
        self.prev_hidden_state,
        self.target_token_embedding,
        self.position_ids,
        self.decoder_segment_ids,
        deterministic=True,
    )
    max_logging.log("Setup complete.")

  def test_multi_token_prediction_layer_output(self):
    """Tests the basic forward pass and output shape of MultiTokenPredictionLayer."""

    output_hidden_state = self.mtp_layer.apply(
        self.variables,
        self.prev_hidden_state,
        self.target_token_embedding,
        self.position_ids,
        decoder_segment_ids=self.decoder_segment_ids,
        deterministic=True,
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


if __name__ == "__main__":
  unittest.main()
