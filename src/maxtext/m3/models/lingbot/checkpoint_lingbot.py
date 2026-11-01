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

"""LingBot-World checkpoint helpers for `to_maxtext_diffusers.py` (no jax / torch imports at module level).

`robbyant/lingbot-world-base-cam` ships the Wan 2.1 VAE and Wan's own umt5-xxl encoder as raw `.pth` state dicts in the
original Wan layout, without diffusers `config.json` files. This module provides their configs (the MaxText model
`__init__` arguments) and the T5 key renaming to Hugging Face `UMT5EncoderModel` names; the VAE is renamed by diffusers'
`convert_wan_vae_to_diffusers`. Referenced by dotted path from `LINGBOT_WORLD_FAST_HF_TO_MAXTEXT.COMPONENT_SOURCES`.
"""

import re

# Wan 2.1 VAE (`AutoencoderKLWan` arguments; latents_mean / latents_std are the model defaults).
WAN21_VAE_CONFIG = {
    "base_dim": 96,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
}

# google/umt5-xxl encoder (`UMT5EncoderModel` arguments).
UMT5_XXL_CONFIG = {
    "vocab_size": 256384,
    "d_model": 4096,
    "d_kv": 64,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1e-6,
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
}

# Wan T5 (`wan/modules/t5.py`, `T5Encoder` with per-layer relative position embeddings) -> HF UMT5 names.
_T5_BLOCK_RENAMES = {
    "norm1.weight": "layer.0.layer_norm.weight",
    "attn.q.weight": "layer.0.SelfAttention.q.weight",
    "attn.k.weight": "layer.0.SelfAttention.k.weight",
    "attn.v.weight": "layer.0.SelfAttention.v.weight",
    "attn.o.weight": "layer.0.SelfAttention.o.weight",
    "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
    "norm2.weight": "layer.1.layer_norm.weight",
    "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
    "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
    "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
}
_T5_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def convert_wan_t5_to_hf(state_dict: dict) -> dict:
  """Renames a Wan umt5-xxl encoder state dict to Hugging Face `UMT5EncoderModel` keys (tensors are not copied)."""
  converted = {}
  for key, value in state_dict.items():
    if key == "token_embedding.weight":
      converted["shared.weight"] = value
    elif key == "norm.weight":
      converted["encoder.final_layer_norm.weight"] = value
    else:
      match = _T5_BLOCK_KEY.match(key)
      if match is None or match.group(2) not in _T5_BLOCK_RENAMES:
        raise KeyError(f"Unexpected Wan T5 encoder key: {key}")
      converted[f"encoder.block.{match.group(1)}.{_T5_BLOCK_RENAMES[match.group(2)]}"] = value
  return converted
