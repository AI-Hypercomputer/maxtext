# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Wan 2.1 DiT transformer (`transformer_wan.py`)."""

import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.wan.transformer_wan import WanModel, WanRotaryPosEmbed, get_sinusoidal_embeddings

# pylint: enable=wrong-import-position

# Tiny Wan 2.1 transformer (14B: 40 layers, 40 heads x 128, ffn 13824, text_dim 4096, 16 latent channels).
_CONFIG = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 16,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": 2,
    "rope_max_seq_len": 32,
}
_BATCH, _FRAMES, _HEIGHT, _WIDTH, _TEXT_LEN = 2, 3, 8, 8, 7


def _inputs():
  latents = jax.random.normal(jax.random.key(0), (_BATCH, _CONFIG["in_channels"], _FRAMES, _HEIGHT, _WIDTH))
  timestep = jnp.array([10.0, 500.0])
  text = jax.random.normal(jax.random.key(1), (_BATCH, _TEXT_LEN, _CONFIG["text_dim"]))
  return latents, timestep, text


def _single_device_mesh():
  """The blocks constrain activations to the Wan mesh axes; a 1-device mesh makes those constraints no-ops."""
  axes = ("data", "fsdp", "context", "tensor")  # `mesh_axes` of wan2.1-t2v-14b.yml
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(axes)), axes)


class WanModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.set_mesh(_single_device_mesh()))

  @parameterized.named_parameters(("scanned", True), ("unscanned", False))
  def test_output_has_latent_shape(self, scan_layers):
    model = WanModel(rngs=nnx.Rngs(0), scan_layers=scan_layers, **_CONFIG)
    latents, timestep, text = _inputs()
    out = model(latents, timestep, text)
    self.assertEqual(out.shape, latents.shape)
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_output_depends_on_timestep_and_text(self):
    model = WanModel(rngs=nnx.Rngs(0), **_CONFIG)
    latents, timestep, text = _inputs()
    out = np.asarray(model(latents, timestep, text))
    np.testing.assert_allclose(np.asarray(model(latents, timestep, text)), out)  # deterministic
    self.assertFalse(np.allclose(np.asarray(model(latents, timestep + 100.0, text)), out))
    self.assertFalse(np.allclose(np.asarray(model(latents, timestep, text + 1.0)), out))


class EmbeddingTest(unittest.TestCase):

  def test_rotary_pos_embed_has_one_row_per_patch(self):
    rope = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=32)
    freqs = rope(jnp.zeros((1, _FRAMES, _HEIGHT, _WIDTH, 4)))  # channels-last video
    num_patches = _FRAMES * (_HEIGHT // 2) * (_WIDTH // 2)
    self.assertEqual(freqs.shape[:3], (1, 1, num_patches))

  def test_sinusoidal_embeddings(self):
    emb = get_sinusoidal_embeddings(jnp.array([0.0, 1.0, 999.0]), embedding_dim=16)
    self.assertEqual(emb.shape, (3, 16))
    self.assertTrue(np.all(np.abs(np.asarray(emb)) <= 1.0 + 1e-6))
    self.assertFalse(np.allclose(np.asarray(emb[0]), np.asarray(emb[1])))


if __name__ == "__main__":
  unittest.main()
