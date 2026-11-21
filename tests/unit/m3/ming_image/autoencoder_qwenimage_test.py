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

"""Ming-Image VAE: diffusers `AutoencoderKLQwenImage` (RGBA) converted to the Wan VAE (`input_channels=4`)."""

import os
import tempfile
import unittest

import pytest

diffusers = pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext_diffusers import convert_diffusers_component, model_init_config
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING, MING_IMAGE_HF_TO_MAXTEXT
from maxtext.m3.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

# Ming-Image-0.1-Design vae/config.json with a tiny base_dim (real: 96).
_VAE_CONFIG = {
    "_class_name": "AutoencoderKLQwenImage",
    "attn_scales": [],
    "base_dim": 8,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "input_channels": 4,
    "latents_mean": None,
    "latents_std": None,
    "num_res_blocks": 1,
    "temperal_downsample": [False, True, True],
    "z_dim": 4,
}


def _load_converted(model, state):
  """Loads a converted (string-keyed) tree into an NNX model, as the inference pipeline does."""
  flat = {tuple(int(p) if p.isdigit() else p for p in path): v for path, v in flatten_dict(state).items()}
  params = nnx.state(model, nnx.Param)
  nnx.replace_by_pure_dict(params, unflatten_dict(flat))
  nnx.update(model, params)
  return model


class MingImageVaeMappingTest(unittest.TestCase):

  def test_registered(self):
    self.assertIs(DIFFUSERS_PARAM_MAPPING["ming-image"], MING_IMAGE_HF_TO_MAXTEXT)
    self.assertEqual(set(MING_IMAGE_HF_TO_MAXTEXT.COMPONENTS), {"transformer", "vae", "mllm", "connector", "mlp"})

  def test_init_config_keeps_input_channels(self):
    config = model_init_config(AutoencoderKLWan, _VAE_CONFIG)
    self.assertEqual(config["input_channels"], 4)
    model = nnx.eval_shape(lambda: AutoencoderKLWan(**config, rngs=nnx.Rngs(0)))
    self.assertEqual(model.encoder.conv_in.conv.kernel.shape[-2], 4)
    self.assertEqual(model.decoder.conv_out.conv.kernel.shape[-1], 4)

  def test_rgb_default_unchanged(self):
    config = model_init_config(AutoencoderKLWan, {k: v for k, v in _VAE_CONFIG.items() if k != "input_channels"})
    model = nnx.eval_shape(lambda: AutoencoderKLWan(**config, rngs=nnx.Rngs(0)))
    self.assertEqual(model.encoder.conv_in.conv.kernel.shape[-2], 3)
    self.assertEqual(model.decoder.conv_out.conv.kernel.shape[-1], 3)


@unittest.skipIf(torch is None, "needs torch (diffusers reference AutoencoderKLQwenImage)")
class QwenImageVaeParityTest(unittest.TestCase):
  """Converts a tiny diffusers `AutoencoderKLQwenImage` with the checkpoint converter and compares outputs."""

  def test_converted_vae_matches_diffusers(self):
    torch.manual_seed(0)
    hf_config = {k: v for k, v in _VAE_CONFIG.items() if not k.startswith("_")}
    hf_model = diffusers.AutoencoderKLQwenImage(**hf_config).eval()
    with torch.no_grad():  # Norm scales init to ones and biases to zeros: randomize so every mapping is checked.
      for name, param in hf_model.named_parameters():
        if "norm" in name:
          param.copy_(1.0 + 0.1 * torch.randn_like(param))
        elif not param.any():
          param.copy_(0.1 * torch.randn_like(param))
    with tempfile.TemporaryDirectory() as repo_dir:  # laid out like the Ming-Image repo's `vae/` folder
      hf_model.save_pretrained(os.path.join(repo_dir, "vae"))
      state, model_config = convert_diffusers_component(
          repo_dir, "vae", MING_IMAGE_HF_TO_MAXTEXT, scan_layers=False, weights_dtype=np.float32
      )
    self.assertEqual(model_config["input_channels"], 4)
    model = _load_converted(AutoencoderKLWan(**model_config, rngs=nnx.Rngs(0)), state)

    # One RGBA image, as Ming-Image encodes / decodes it: (B, C, T=1, H, W).
    image = np.random.default_rng(0).uniform(-1.0, 1.0, (1, 4, 1, 32, 32)).astype(np.float32)
    with torch.no_grad():
      expected_mean = hf_model.encode(torch.from_numpy(image)).latent_dist.mode()
      expected_image = hf_model.decode(expected_mean).sample.numpy()
    expected_mean = expected_mean.numpy()
    with jax.default_matmul_precision("float32"):
      mean = model.encode(image, AutoencoderKLWanCache(model)).latent_dist.mode()  # (B, F, H, W, z)
      decoded = model.decode(np.transpose(expected_mean, (0, 2, 3, 4, 1)), AutoencoderKLWanCache(model)).sample
    self.assertEqual(decoded.shape[-1], 4)
    np.testing.assert_allclose(np.transpose(np.asarray(mean), (0, 4, 1, 2, 3)), expected_mean, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.transpose(np.asarray(decoded), (0, 4, 1, 2, 3)), expected_image, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
  unittest.main()
