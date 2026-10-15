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

"""Parity tests of the NNX UMT5 text encoder against Hugging Face `transformers.UMT5EncoderModel`."""

import os
import tempfile
import unittest

from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import numpy as np
import transformers

from maxtext.checkpoint_conversion.to_maxtext_diffusers import convert_diffusers_component
from maxtext.checkpoint_conversion.utils.param_mapping import WAN2_1_HF_TO_MAXTEXT
from maxtext.m3.models.wan.text_encoder_wan import UMT5EncoderModel, relative_position_bucket

try:
  import torch
except ImportError:
  torch = None

# Tiny UMT5 (umt5-xxl: d_model 4096, 24 layers, 64 heads x 64, d_ff 10240, 32 buckets, max distance 128).
_TINY_CONFIG = {
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


def _nnx_from_converted(state: dict, model_config: dict) -> UMT5EncoderModel:
  """Builds the NNX model and loads a converted param tree into it (as the inference pipeline does)."""
  model = UMT5EncoderModel(**model_config, precision=jax.lax.Precision.HIGHEST, rngs=nnx.Rngs(0))
  flat = {tuple(int(p) if p.isdigit() else p for p in path): v for path, v in flatten_dict(state).items()}
  params = nnx.state(model, nnx.Param)
  nnx.replace_by_pure_dict(params, unflatten_dict(flat))
  nnx.update(model, params)
  return model


@unittest.skipIf(torch is None, "needs torch (Hugging Face reference UMT5EncoderModel)")
class UMT5EncoderParityTest(unittest.TestCase):
  """NNX vs torch on a tiny config.

  Weights go HF -> NNX through the checkpoint converter (`convert_diffusers_component` with the
  `WAN2_1_HF_TO_MAXTEXT` mapping), so this also covers the conversion of the `text_encoder` component.
  Run: `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest src/maxtext/m3/models/wan/text_encoder_wan_test.py`.
  """

  def _convert(self, hf_model):
    """Saves `hf_model` like a diffusers `text_encoder/` folder and converts it with the checkpoint converter."""
    with tempfile.TemporaryDirectory() as repo_dir:
      hf_model.save_pretrained(os.path.join(repo_dir, "text_encoder"))
      return convert_diffusers_component(
          repo_dir, "text_encoder", WAN2_1_HF_TO_MAXTEXT, scan_layers=False, weights_dtype=np.float32
      )

  def test_matches_hf(self):
    torch.manual_seed(0)
    hf_model = transformers.UMT5EncoderModel(transformers.UMT5Config.from_dict(_TINY_CONFIG)).eval()
    with torch.no_grad():  # Norm scales init to ones: randomize them so the mapping of every param is checked.
      for name, param in hf_model.named_parameters():
        if "layer_norm" in name:
          param.copy_(1.0 + 0.1 * torch.randn_like(param))

    state, model_config = self._convert(hf_model)
    self.assertEqual(model_config["num_layers"], _TINY_CONFIG["num_layers"])
    self.assertTrue(model_config["is_gated_act"])
    self.assertEqual(model_config["dense_act_fn"], "gelu_new")
    model = _nnx_from_converted(state, model_config)

    # Length 40 > relative_attention_max_distance exercises the logarithmic buckets; rows are padded differently.
    batch, length = 3, 40
    rng = np.random.default_rng(0)
    input_ids = rng.integers(2, _TINY_CONFIG["vocab_size"], size=(batch, length))
    seq_lens = np.array([length, 17, 1])
    mask = (np.arange(length)[None, :] < seq_lens[:, None]).astype(np.int64)
    input_ids = np.where(mask > 0, input_ids, 0)

    with torch.no_grad():
      expected = hf_model(torch.from_numpy(input_ids), torch.from_numpy(mask)).last_hidden_state.numpy()
    actual = np.asarray(jax.jit(model.__call__)(input_ids, mask))
    self.assertEqual(actual.shape, (batch, length, _TINY_CONFIG["d_model"]))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    # Without a mask every token is attended.
    with torch.no_grad():
      expected = hf_model(torch.from_numpy(input_ids)).last_hidden_state.numpy()
    np.testing.assert_allclose(np.asarray(model(input_ids)), expected, atol=1e-5, rtol=1e-5)

  def test_relative_position_bucket_matches_hf(self):
    for num_buckets, max_distance in ((32, 128), (8, 16)):
      config = transformers.UMT5Config.from_dict(
          {"relative_attention_num_buckets": num_buckets, "relative_attention_max_distance": max_distance}
      )
      attention = transformers.models.umt5.modeling_umt5.UMT5Attention(config, has_relative_attention_bias=True)
      relative_position = np.arange(-2048, 2049)
      # pylint: disable-next=protected-access
      expected = attention._relative_position_bucket(torch.from_numpy(relative_position)).numpy()
      np.testing.assert_array_equal(relative_position_bucket(relative_position, num_buckets, max_distance), expected)


class UMT5EncoderShardingTest(unittest.TestCase):

  def test_params_have_logical_axes(self):
    init_config = {k: v for k, v in _TINY_CONFIG.items() if k not in ("feed_forward_proj", "dropout_rate")}
    model = nnx.eval_shape(lambda: UMT5EncoderModel(**init_config, rngs=nnx.Rngs(0)))
    specs = flatten_dict(nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(model, nnx.Param))))
    self.assertEqual(tuple(specs[("token_embedding", "embedding")]), ("vocab", "embed"))
    self.assertEqual(tuple(specs[("blocks", 0, "attn", "query", "kernel")]), ("embed", "heads"))
    self.assertEqual(tuple(specs[("blocks", 0, "attn", "proj_attn", "kernel")]), ("heads", "embed"))
    self.assertEqual(tuple(specs[("blocks", 0, "ffn", "wo", "kernel")]), ("mlp", "embed"))
    self.assertEqual(tuple(specs[("final_norm", "scale")]), ("norm",))


if __name__ == "__main__":
  unittest.main()
