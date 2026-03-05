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

"""Gemma3 Multimodal tests."""

import sys

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import pytest


class TestGemma3Multimodal(absltest.TestCase):
  """Test Gemma3 Multimodal functionality."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def init_gemma3_config(self, **kwargs):
    """Init pyconfig for Gemma3."""
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test_gemma3",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=32,
        base_emb_dim=256,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        head_dim=64,
        decoder_block="gemma3",
        model_name="gemma3-4b",
        use_multimodal=True,
        # Vision config
        image_size_for_vit=256,
        patch_size_for_vit=16,
        conv_stride_for_vit=16,
        hidden_size_for_vit=256,
        num_attention_heads_for_vit=4,
        num_hidden_layers_for_vit=2,
        num_channels_for_vit=3,
        **kwargs,
    )
    return config

  @pytest.mark.tpu_only
  def test_gemma3_multimodal_forward(self):
    """Test Gemma3 multimodal forward pass with dummy images."""
    config = self.init_gemma3_config()
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(
        config=config, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN
    )

    batch_size = config.global_batch_size_to_train_on
    seq_len = config.max_target_length
    
    # Text inputs
    ids = jax.random.randint(self.rng, (batch_size, seq_len), 0, config.vocab_size)
    decoder_positions = jnp.stack([
        jnp.arange(seq_len, dtype=jnp.int32)
        for _ in range(batch_size)
    ])
    decoder_segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR

    # Image inputs [B, N, H, W, C]
    # N=1 for now as per code
    image_shape = (batch_size, 1, config.image_size_for_vit, config.image_size_for_vit, config.num_channels_for_vit)
    encoder_images = jax.random.normal(self.rng, image_shape)

    # Initialize model
    variables = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        encoder_images=encoder_images,
        enable_dropout=False,
    )

    # Forward pass
    logits = model.apply(
        variables,
        ids,
        decoder_positions,
        decoder_segment_ids,
        encoder_images=encoder_images,
        enable_dropout=False,
        rngs={"aqt": self.rng},
    )

    # Check output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    self.assertEqual(logits.shape, expected_shape)
    # Check that logits are not NaN
    self.assertTrue(jnp.all(jnp.isfinite(logits)))

if __name__ == "__main__":
  absltest.main()
