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

"""Tests for the Wan 2.1 diffusers -> MaxText parameter mapping (WAN2_1_HF_TO_MAXTEXT)."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.traverse_util import flatten_dict
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext_diffusers import model_init_config
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING, WAN2_1_HF_TO_MAXTEXT
from maxtext.checkpoint_conversion.utils.utils import validate_and_filter_param_map_keys
from maxtext.m3.models.wan import AutoencoderKLWan, UMT5EncoderModel, WanModel

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
    model = nnx.eval_shape(lambda: AutoencoderKLWan(**config, rngs=nnx.Rngs(0), mesh=None))
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


if __name__ == "__main__":
  absltest.main()
