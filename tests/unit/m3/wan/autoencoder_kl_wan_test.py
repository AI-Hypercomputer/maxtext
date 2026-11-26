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

import functools
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from flax import nnx
import jax
import numpy as np

from maxtext.m3.models.wan.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AutoencoderKLWanCache,
    AvgDown3D,
    DupUp3D,
    patchify,
    unpatchify,
)

# pylint: enable=wrong-import-position

# Tiny VAE (Wan 2.1: base_dim 96, z_dim 16); same downsampling: 4x in time (causal), 8x in space.
_CONFIG = {
    "base_dim": 8,
    "z_dim": 4,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "latents_mean": [0.0] * 4,
    "latents_std": [1.0] * 4,
}
# Tiny Wan 2.2 residual VAE (Wan 2.2 5B: base_dim 160, decoder_base_dim 256, z_dim 48, patch_size 2, 4x16x16).
_RESIDUAL_CONFIG = {
    "base_dim": 8,
    "decoder_base_dim": 16,
    "z_dim": 12,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "latents_mean": [0.0] * 12,
    "latents_std": [1.0] * 12,
}
_BATCH, _FRAMES, _HEIGHT, _WIDTH = 1, 5, 16, 16


# Eager VAE init and 3D-conv calls dispatch every op separately on CPU, so jit init and encode/decode and build each
# VAE lazily at most once per process: half of the tests below need no VAE, and only one needs the residual VAE.
@functools.cache
def _vae() -> AutoencoderKLWan:
  return nnx.jit(lambda: AutoencoderKLWan(rngs=nnx.Rngs(0), **_CONFIG))()


@functools.cache
def _residual_vae() -> AutoencoderKLWan:
  return nnx.jit(lambda: AutoencoderKLWan(rngs=nnx.Rngs(0), **_RESIDUAL_CONFIG))()


@nnx.jit
def _encode_mode(vae: AutoencoderKLWan, video: jax.Array) -> jax.Array:
  """Jitted encode returning posterior mode latents."""
  return vae.encode(video, AutoencoderKLWanCache(vae)).latent_dist.mode()


@nnx.jit
def _decode_sample(vae: AutoencoderKLWan, latents: jax.Array) -> jax.Array:
  """Jitted decode returning reconstructed video frames."""
  return vae.decode(latents, AutoencoderKLWanCache(vae)).sample


# Keep this class on one xdist worker so the cached VAEs are built once, not once per worker.
@pytest.mark.xdist_group("AutoencoderKLWanTest")
class AutoencoderKLWanTest(unittest.TestCase):

  @property
  def vae(self) -> AutoencoderKLWan:
    return _vae()

  @property
  def residual_vae(self) -> AutoencoderKLWan:
    return _residual_vae()

  def test_latent_shape(self):
    self.assertEqual(AutoencoderKLWan.latent_shape(_CONFIG, 2, 81, 480, 832), (2, 4, 21, 60, 104))
    self.assertEqual(AutoencoderKLWan.latent_shape(_CONFIG, 1, 1, 16, 16), (1, 4, 1, 2, 2))
    self.assertEqual(AutoencoderKLWan.latent_shape(_RESIDUAL_CONFIG, 2, 81, 704, 1280), (2, 12, 21, 44, 80))

  def test_encode_decode_shapes(self):
    video = jax.random.uniform(jax.random.key(0), (_BATCH, 3, _FRAMES, _HEIGHT, _WIDTH), minval=-1.0, maxval=1.0)
    latents = _encode_mode(self.vae, video)  # channels-last (B, F, H, W, z_dim)
    b, z, f, h, w = AutoencoderKLWan.latent_shape(_CONFIG, _BATCH, _FRAMES, _HEIGHT, _WIDTH)
    self.assertEqual(latents.shape, (b, f, h, w, z))

    decoded = _decode_sample(self.vae, latents)  # channels-last (B, F, H, W, 3)
    self.assertEqual(decoded.shape, (_BATCH, _FRAMES, _HEIGHT, _WIDTH, 3))
    self.assertTrue(np.all(np.isfinite(np.asarray(decoded))))

  def test_decode_accepts_channels_first_latents(self):
    latents = jax.random.normal(jax.random.key(1), AutoencoderKLWan.latent_shape(_CONFIG, _BATCH, _FRAMES, 16, 16))
    decoded = _decode_sample(self.vae, latents)
    self.assertEqual(decoded.shape, (_BATCH, _FRAMES, 16, 16, 3))

  def test_residual_vae_encode_decode_shapes(self):
    height, width = 32, 32
    video = jax.random.uniform(jax.random.key(2), (_BATCH, 3, _FRAMES, height, width), minval=-1.0, maxval=1.0)
    latents = _encode_mode(self.residual_vae, video)
    b, z, f, h, w = AutoencoderKLWan.latent_shape(_RESIDUAL_CONFIG, _BATCH, _FRAMES, height, width)
    self.assertEqual(latents.shape, (b, f, h, w, z))

    decoded = _decode_sample(self.residual_vae, latents)
    self.assertEqual(decoded.shape, (_BATCH, _FRAMES, height, width, 3))
    self.assertTrue(np.all(np.isfinite(np.asarray(decoded))))

  def test_latents_stats_must_match_z_dim(self):
    # No stats -> the 16-entry Wan 2.1 defaults, which don't fit z_dim=12.
    config = {k: v for k, v in _RESIDUAL_CONFIG.items() if k not in ("latents_mean", "latents_std")}
    with self.assertRaisesRegex(ValueError, "z_dim=12"):
      nnx.eval_shape(lambda: AutoencoderKLWan(rngs=nnx.Rngs(0), **config))

  def test_shortcut_channel_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, "AvgDown3D"):
      AvgDown3D(in_channels=3, out_channels=8, factor_t=1, factor_s=2)  # 3 * 4 = 12 not divisible by 8
    with self.assertRaisesRegex(ValueError, "DupUp3D"):
      DupUp3D(in_channels=8, out_channels=3, factor_t=1, factor_s=2)  # 3 * 4 = 12 not divisible by 8


class PatchifyTest(unittest.TestCase):
  """Pins the channel order of diffusers' einops "b c f (h q) (w r) -> b (c r q) f h w" (CPU, no torch)."""

  def test_channel_order(self):
    # x[0, 0, h, w, c] = 4h + 2w + c; output slot c*4 + r*2 + q holds x[0, 0, q, r, c] = 4q + 2r + c.
    x = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)  # (B, F, H, W, C)
    out = np.asarray(patchify(x, patch_size=2))
    self.assertEqual(out.shape, (1, 1, 1, 1, 8))
    np.testing.assert_array_equal(out[0, 0, 0, 0], [0, 4, 2, 6, 1, 5, 3, 7])

  def test_round_trip(self):
    x = np.random.default_rng(0).standard_normal((2, 3, 8, 6, 3)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(unpatchify(patchify(x, patch_size=2), patch_size=2)), x)

  def test_patch_size_one_is_identity(self):
    x = np.random.default_rng(1).standard_normal((1, 2, 4, 4, 3)).astype(np.float32)
    for patch_size in (None, 1):
      np.testing.assert_array_equal(np.asarray(patchify(x, patch_size)), x)
      np.testing.assert_array_equal(np.asarray(unpatchify(x, patch_size)), x)


if __name__ == "__main__":
  unittest.main()
