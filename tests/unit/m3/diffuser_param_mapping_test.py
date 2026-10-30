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

"""Tests for the diffusers -> MaxText checkpoint conversion of Wan 2.1 (WAN2_1_HF_TO_MAXTEXT)."""

import os
import tempfile
import unittest

import pytest

diffusers = pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import numpy as np
import transformers

from maxtext.checkpoint_conversion.to_maxtext_diffusers import convert_diffusers_component, model_init_config
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING, WAN2_1_HF_TO_MAXTEXT
from maxtext.checkpoint_conversion.utils.utils import validate_and_filter_param_map_keys
from maxtext.m3.models.wan import AutoencoderKLWan, UMT5EncoderModel, WanModel
from maxtext.m3.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

# Tiny configs with the Wan 2.1 diffusers config keys (real model: 40 layers, 40 heads x 128, ...).
_TRANSFORMER_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 8,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "ffn_dim": 32,
    "freq_dim": 16,
    "image_dim": None,
    "in_channels": 4,
    "num_attention_heads": 2,
    "num_layers": 3,
    "out_channels": 4,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 64,
    "text_dim": 16,
}
_I2V_CONFIG = dict(_TRANSFORMER_CONFIG, image_dim=8, added_kv_proj_dim=16)
_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 8,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 4,
}
# UMT5 text encoder (umt5-xxl: d_model 4096, 24 layers, 64 heads x 64, d_ff 10240).
_TEXT_ENCODER_CONFIG = {
    "d_ff": 24,
    "d_kv": 4,
    "d_model": 16,
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-6,
    "num_heads": 2,
    "num_layers": 3,
    "relative_attention_max_distance": 16,
    "relative_attention_num_buckets": 8,
    "vocab_size": 64,
}


def _abstract_shapes(component, config, scan_layers):
  if component == "transformer":
    extra = {"image_seq_len": 4} if config["image_dim"] else {}
    model = nnx.eval_shape(lambda: WanModel(**config, rngs=nnx.Rngs(0), scan_layers=scan_layers, **extra))
  elif component == "text_encoder":
    model = nnx.eval_shape(lambda: UMT5EncoderModel(**config, rngs=nnx.Rngs(0)))
  else:
    model = nnx.eval_shape(lambda: AutoencoderKLWan(**config, rngs=nnx.Rngs(0)))
  params = nnx.state(model, nnx.Param).to_pure_dict()
  return {"-".join(map(str, path)): tuple(leaf.shape) for path, leaf in flatten_dict(params).items()}


def _hf_shape(maxtext_key, target_shape, hook_name):
  """Inverts the hook on shapes: the PyTorch-layout shape the HF tensor would have."""
  if hook_name == "transpose":
    return target_shape[::-1]
  if hook_name == "conv":  # Flax (*spatial, in, out) <- PyTorch (out, in, *spatial)
    return (target_shape[-1], target_shape[-2], *target_shape[:-2])
  if hook_name == "reshape":
    return (*target_shape, 1, 1, 1)
  return target_shape


class Wan21ParamMappingTest(parameterized.TestCase):

  def test_registered(self):
    self.assertIs(DIFFUSERS_PARAM_MAPPING["wan2.1"], WAN2_1_HF_TO_MAXTEXT)
    self.assertEqual(set(WAN2_1_HF_TO_MAXTEXT.COMPONENTS), {"transformer", "vae", "text_encoder"})

  @parameterized.named_parameters(
      ("transformer_t2v", "transformer", _TRANSFORMER_CONFIG, False),
      ("transformer_t2v_scanned", "transformer", _TRANSFORMER_CONFIG, True),
      ("transformer_i2v", "transformer", _I2V_CONFIG, False),
      ("vae", "vae", _VAE_CONFIG, False),
      ("text_encoder", "text_encoder", _TEXT_ENCODER_CONFIG, False),
  )
  def test_mapping_covers_model(self, component, config, scan_layers):
    target = _abstract_shapes(component, config, scan_layers)
    mapping = WAN2_1_HF_TO_MAXTEXT.param_mapping(component, config, scan_layers)
    hooks = WAN2_1_HF_TO_MAXTEXT.hook_fns(component, config, scan_layers)
    # Raises if any model param has no mapping entry.
    keys = validate_and_filter_param_map_keys(mapping.keys(), set(target))
    self.assertLen(keys, len(target))

    hf_keys = []
    for key in keys:
      sources = mapping[key] if isinstance(mapping[key], list) else [mapping[key]]
      shape = target[key][1:] if isinstance(mapping[key], list) else target[key]
      if isinstance(mapping[key], list):
        self.assertLen(sources, config["num_layers"], key)
      hf_keys += sources
      hook = hooks.get(key)
      hook_name = getattr(hook, "__name__", None)
      weight = np.zeros(_hf_shape(key, shape, hook_name), np.float32)
      out = hook(weight, shape) if hook else weight
      self.assertEqual(out.shape, shape, key)
    self.assertLen(set(hf_keys), len(hf_keys), "an HF tensor is mapped twice")

  def test_param_dtype(self):
    dtype = WAN2_1_HF_TO_MAXTEXT.param_dtype
    self.assertEqual(dtype("transformer", "blocks-0-attn1-query-kernel", "bfloat16"), np.dtype("bfloat16"))
    for key in (
        "blocks-0-norm2-layer_norm-scale",
        "blocks-0-adaln_scale_shift_table",
        "condition_embedder-time_proj-kernel",
    ):
      self.assertEqual(dtype("transformer", key, "bfloat16"), np.dtype(np.float32), key)
    self.assertIsNone(dtype("vae", "decoder-conv_in-conv-kernel", "bfloat16"))
    self.assertIsNone(dtype("text_encoder", "blocks-0-attn-query-kernel", "bfloat16"))

  def test_model_init_config_drops_non_init_keys(self):
    hf_config = dict(_TRANSFORMER_CONFIG, _class_name="WanTransformer3DModel", _diffusers_version="0.33.0", foo=1)
    self.assertEqual(model_init_config(WanModel, hf_config), _TRANSFORMER_CONFIG)
    self.assertEqual(model_init_config(AutoencoderKLWan, dict(_VAE_CONFIG, _class_name="AutoencoderKLWan")), _VAE_CONFIG)


@unittest.skipIf(torch is None, "needs torch (Hugging Face reference UMT5EncoderModel)")
class UMT5ConversionParityTest(unittest.TestCase):
  """Converts a tiny HF `UMT5EncoderModel` with the checkpoint converter and compares the NNX model with it."""

  _HF_CONFIG = {
      "vocab_size": 96,
      "d_model": 32,
      "d_kv": 8,
      "d_ff": 48,
      "num_layers": 3,
      "num_heads": 4,
      "relative_attention_num_buckets": 8,
      "relative_attention_max_distance": 16,
      "feed_forward_proj": "gated-gelu",
      "layer_norm_epsilon": 1e-6,
      "dropout_rate": 0.0,
  }

  def test_converted_text_encoder_matches_hf(self):
    torch.manual_seed(0)
    hf_model = transformers.UMT5EncoderModel(transformers.UMT5Config.from_dict(self._HF_CONFIG)).eval()
    with torch.no_grad():  # Norm scales init to ones: randomize them so the mapping of every param is checked.
      for name, param in hf_model.named_parameters():
        if "layer_norm" in name:
          param.copy_(1.0 + 0.1 * torch.randn_like(param))
    with tempfile.TemporaryDirectory() as repo_dir:  # laid out like a diffusers repo's `text_encoder/` folder
      hf_model.save_pretrained(os.path.join(repo_dir, "text_encoder"))
      state, model_config = convert_diffusers_component(
          repo_dir, "text_encoder", WAN2_1_HF_TO_MAXTEXT, scan_layers=False, weights_dtype=np.float32
      )
    self.assertEqual(model_config["num_layers"], self._HF_CONFIG["num_layers"])
    self.assertTrue(model_config["is_gated_act"])
    self.assertEqual(model_config["dense_act_fn"], "gelu_new")

    # Load the converted tree into the NNX model, as the inference pipeline does.
    model = UMT5EncoderModel(**model_config, precision=jax.lax.Precision.HIGHEST, rngs=nnx.Rngs(0))
    flat = {tuple(int(p) if p.isdigit() else p for p in path): v for path, v in flatten_dict(state).items()}
    params = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(params, unflatten_dict(flat))
    nnx.update(model, params)

    # Length 40 > relative_attention_max_distance exercises the logarithmic buckets; rows are padded differently.
    batch, length = 3, 40
    input_ids = np.random.default_rng(0).integers(2, self._HF_CONFIG["vocab_size"], size=(batch, length))
    mask = (np.arange(length)[None, :] < np.array([length, 17, 1])[:, None]).astype(np.int64)
    input_ids = np.where(mask > 0, input_ids, 0)
    with torch.no_grad():
      expected = hf_model(torch.from_numpy(input_ids), torch.from_numpy(mask)).last_hidden_state.numpy()
    np.testing.assert_allclose(np.asarray(jax.jit(model.__call__)(input_ids, mask)), expected, atol=1e-5, rtol=1e-5)


def _randomize_torch_params(model):
  """Norm scales init to ones and some biases to zeros: randomize them so the mapping of every param is checked."""
  with torch.no_grad():
    for name, param in model.named_parameters():
      if "norm" in name:
        param.copy_(1.0 + 0.1 * torch.randn_like(param))
      elif not param.any():
        param.copy_(0.1 * torch.randn_like(param))


def _load_converted(model, state):
  """Loads a converted (string-keyed) tree into an NNX model, as the inference pipeline does."""
  flat = {tuple(int(p) if p.isdigit() else p for p in path): v for path, v in flatten_dict(state).items()}
  params = nnx.state(model, nnx.Param)
  nnx.replace_by_pure_dict(params, unflatten_dict(flat))
  nnx.update(model, params)
  return model


def _convert(hf_model, component):
  with tempfile.TemporaryDirectory() as repo_dir:  # laid out like a diffusers repo's `<component>/` folder
    hf_model.save_pretrained(os.path.join(repo_dir, component))
    return convert_diffusers_component(
        repo_dir, component, WAN2_1_HF_TO_MAXTEXT, scan_layers=False, weights_dtype=np.float32
    )


@unittest.skipIf(torch is None, "needs torch (diffusers reference WanTransformer3DModel / AutoencoderKLWan)")
class DiffusersConversionParityTest(unittest.TestCase):
  """Converts tiny diffusers Wan models with the checkpoint converter and compares the NNX models with them.

  Local check only: CI's cpu-unit environment has no torch, so it skips this class.
  """

  def test_converted_transformer_matches_diffusers(self):
    torch.manual_seed(0)
    hf_config = dict(_TRANSFORMER_CONFIG, patch_size=tuple(_TRANSFORMER_CONFIG["patch_size"]))
    hf_model = diffusers.WanTransformer3DModel(**hf_config).eval()
    _randomize_torch_params(hf_model)
    state, model_config = _convert(hf_model, "transformer")

    axes = ("data", "fsdp", "context", "tensor")  # `mesh_axes` of wan2.1-t2v-14b.yml; constraints are no-ops on 1 device
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(axes)), axes)
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((2, _TRANSFORMER_CONFIG["in_channels"], 3, 8, 8), dtype=np.float32)
    timestep = np.array([10.0, 700.0], np.float32)
    text = rng.standard_normal((2, 7, _TRANSFORMER_CONFIG["text_dim"]), dtype=np.float32)
    with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
      model = WanModel(**model_config, attention="dot_product", rngs=nnx.Rngs(0))  # model_config has scan_layers
      _load_converted(model, state)
      actual = np.asarray(model(latents, timestep, text))
    with torch.no_grad():
      expected = hf_model(
          torch.from_numpy(latents), torch.from_numpy(timestep), torch.from_numpy(text), return_dict=False
      )[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)

  def test_converted_vae_matches_diffusers(self):
    torch.manual_seed(0)
    vae_config = dict(_VAE_CONFIG, num_res_blocks=1)
    hf_model = diffusers.AutoencoderKLWan(**vae_config).eval()
    _randomize_torch_params(hf_model)
    state, model_config = _convert(hf_model, "vae")
    model = _load_converted(AutoencoderKLWan(**model_config, rngs=nnx.Rngs(0)), state)

    video = np.random.default_rng(0).uniform(-1.0, 1.0, (1, 3, 5, 16, 16)).astype(np.float32)  # (B, C, F, H, W)
    with torch.no_grad():
      expected_mean = hf_model.encode(torch.from_numpy(video)).latent_dist.mean
      expected_video = hf_model.decode(expected_mean).sample.numpy()
    expected_mean = expected_mean.numpy()
    with jax.default_matmul_precision("float32"):
      mean = model.encode(video, AutoencoderKLWanCache(model)).latent_dist.mode()  # channels-last (B, F, H, W, z)
      # Decode the reference latents so the decoder is compared on identical inputs.
      decoded = model.decode(np.transpose(expected_mean, (0, 2, 3, 4, 1)), AutoencoderKLWanCache(model)).sample
    np.testing.assert_allclose(np.transpose(np.asarray(mean), (0, 4, 1, 2, 3)), expected_mean, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.transpose(np.asarray(decoded), (0, 4, 1, 2, 3)), expected_video, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
  absltest.main()
