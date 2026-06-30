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

"""Tests for vla_decode model components."""

import sys
import unittest
from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.models.jasmine import DynamicsMaskGIT
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
try:
  from maxtext.tests.utils.test_helpers import get_test_config_path
except ModuleNotFoundError:
  from tests.utils.test_helpers import get_test_config_path



class DynamicsMaskGITTest(unittest.TestCase):
  """Tests for DynamicsMaskGIT."""

  def setUp(self):
    super().setUp()
    # We need both params and default keys for Attention layers
    self.rngs = nnx.Rngs(params=0, default=1)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 16,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    model_dim = 128
    ffn_dim = 256
    num_latents = 100
    latent_action_dim = 32
    num_blocks = 2
    num_heads = 4
    seq_len = 16
    batch_size = 2
    num_spatial_patches = 16

    model = DynamicsMaskGIT(
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        num_latents=num_latents,
        latent_action_dim=latent_action_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=0.0,
        mask_limit=0.0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        use_flash_attention=False,
        config=self.cfg,
        mesh=self.mesh,
        num_spatial_patches=num_spatial_patches,
        temporal_seq_len=seq_len,
        decode=False,
        rngs=self.rngs,
    )

    # Mock inputs:
    # video_tokens_BTN shape: (B, T, N)
    # latent_actions_BTm11L shape: (B, T-1, 1, L)
    video_tokens_BTN = jnp.zeros((batch_size, seq_len, num_spatial_patches), dtype=jnp.int32)
    latent_actions_BTm11L = jnp.zeros((batch_size, seq_len - 1, 1, latent_action_dim), dtype=jnp.float32)

    logits_BTNV, mask = model(video_tokens_BTN, latent_actions_BTm11L)

    self.assertEqual(logits_BTNV.shape, (batch_size, seq_len, num_spatial_patches, num_latents))
    self.assertEqual(mask.shape, (batch_size, seq_len, num_spatial_patches))

  def test_sample(self):
    model_dim = 128
    ffn_dim = 256
    num_latents = 100
    latent_action_dim = 32
    num_blocks = 2
    num_heads = 4
    seq_len = 16
    batch_size = 2
    num_spatial_patches = 16

    model = DynamicsMaskGIT(
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        num_latents=num_latents,
        latent_action_dim=latent_action_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=0.0,
        mask_limit=0.0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        use_flash_attention=False,
        config=self.cfg,
        mesh=self.mesh,
        num_spatial_patches=num_spatial_patches,
        temporal_seq_len=seq_len,
        decode=False,
        rngs=self.rngs,
    )

    # Mock inputs for sampling:
    # token_idxs_BTN: (B, T_condition, N)
    # action_tokens_EL: (B * (T - 1), L)
    T_condition = 1
    token_idxs_BTN = jnp.zeros((batch_size, T_condition, num_spatial_patches), dtype=jnp.int32)
    action_tokens_EL = jnp.zeros((batch_size * (seq_len - 1), latent_action_dim), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)

    final_token_idxs_BSN, final_logits_BSNV = model.sample(
        token_idxs_BTN=token_idxs_BTN,
        action_tokens_EL=action_tokens_EL,
        seq_len=seq_len,
        steps=2, # small steps for fast test
        temperature=1.0,
        sample_argmax=True,
        rng=rng,
    )

    self.assertEqual(final_token_idxs_BSN.shape, (batch_size, seq_len, num_spatial_patches))
    self.assertEqual(final_logits_BSNV.shape, (batch_size, seq_len, num_spatial_patches, num_latents))

if __name__ == "__main__":
  unittest.main()
