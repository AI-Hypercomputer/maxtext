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

"""Tests for the Wan 2.1 3D causal VAE (`autoencoder_kl_wan.py`)."""

import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from flax import nnx
import jax
import numpy as np

from maxtext.m3.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache

# pylint: enable=wrong-import-position

# Tiny VAE (Wan 2.1: base_dim 96, z_dim 16); same downsampling: 4x in time (causal), 8x in space.
_CONFIG = {
    "base_dim": 8,
    "z_dim": 4,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
}
_BATCH, _FRAMES, _HEIGHT, _WIDTH = 1, 5, 16, 16


class AutoencoderKLWanTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.vae = AutoencoderKLWan(rngs=nnx.Rngs(0), **_CONFIG)

  def test_latent_shape(self):
    self.assertEqual(AutoencoderKLWan.latent_shape(_CONFIG, 2, 81, 480, 832), (2, 4, 21, 60, 104))
    self.assertEqual(AutoencoderKLWan.latent_shape(_CONFIG, 1, 1, 16, 16), (1, 4, 1, 2, 2))

  def test_encode_decode_shapes(self):
    video = jax.random.uniform(jax.random.key(0), (_BATCH, 3, _FRAMES, _HEIGHT, _WIDTH), minval=-1.0, maxval=1.0)
    posterior = self.vae.encode(video, AutoencoderKLWanCache(self.vae)).latent_dist
    latents = posterior.mode()  # channels-last (B, F, H, W, z_dim)
    b, z, f, h, w = AutoencoderKLWan.latent_shape(_CONFIG, _BATCH, _FRAMES, _HEIGHT, _WIDTH)
    self.assertEqual(latents.shape, (b, f, h, w, z))

    decoded = self.vae.decode(latents, AutoencoderKLWanCache(self.vae)).sample  # channels-last (B, F, H, W, 3)
    self.assertEqual(decoded.shape, (_BATCH, _FRAMES, _HEIGHT, _WIDTH, 3))
    self.assertTrue(np.all(np.isfinite(np.asarray(decoded))))

  def test_decode_accepts_channels_first_latents(self):
    latents = jax.random.normal(jax.random.key(1), AutoencoderKLWan.latent_shape(_CONFIG, _BATCH, _FRAMES, 16, 16))
    decoded = self.vae.decode(latents, AutoencoderKLWanCache(self.vae)).sample
    self.assertEqual(decoded.shape, (_BATCH, _FRAMES, 16, 16, 3))


if __name__ == "__main__":
  unittest.main()
