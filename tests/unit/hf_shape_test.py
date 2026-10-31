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

"""Tests for Hugging Face checkpoint shape mappings."""

from types import SimpleNamespace

import numpy as np
import pytest

from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping

pytestmark = [pytest.mark.cpu_only, pytest.mark.decoupled_target]
OLMO3_MODELS = ("olmo3-7b", "olmo3-7b-pt", "olmo3-32b")


@pytest.mark.parametrize("model_name", OLMO3_MODELS)
@pytest.mark.parametrize(
    "attention_config,query_size,kv_size",
    [
        ({"num_key_value_heads": 4}, 24, 24),
        ({"num_key_value_heads": 2, "head_dim": None}, 24, 12),
        ({"num_key_value_heads": 2, "head_dim": 8}, 32, 16),
    ],
)
def test_olmo3_shape_mapping(model_name, attention_config, query_size, kv_size):
  """OLMo3 has post-MLP normalization and QK norms across all heads."""
  config = {
      "hidden_size": 24,
      "intermediate_size": 40,
      "vocab_size": 33,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      **attention_config,
  }
  expected = {
      "model.embed_tokens.weight": [33, 24],
      "model.norm.weight": [24],
      "lm_head.weight": [33, 24],
  }
  for layer in range(2):
    prefix = f"model.layers.{layer}"
    expected.update(
        {
            f"{prefix}.self_attn.q_proj.weight": [query_size, 24],
            f"{prefix}.self_attn.k_proj.weight": [kv_size, 24],
            f"{prefix}.self_attn.v_proj.weight": [kv_size, 24],
            f"{prefix}.self_attn.o_proj.weight": [24, query_size],
            f"{prefix}.self_attn.q_norm.weight": [query_size],
            f"{prefix}.self_attn.k_norm.weight": [kv_size],
            f"{prefix}.mlp.gate_proj.weight": [40, 24],
            f"{prefix}.mlp.up_proj.weight": [40, 24],
            f"{prefix}.mlp.down_proj.weight": [24, 40],
            f"{prefix}.post_attention_layernorm.weight": [24],
            f"{prefix}.post_feedforward_layernorm.weight": [24],
        }
    )
  assert hf_shape.HF_SHAPE[model_name](config) == expected


@pytest.mark.parametrize("model_name", OLMO3_MODELS)
@pytest.mark.parametrize("scan_layers", [False, True])
def test_olmo3_shapes_cover_mapping_targets(model_name, scan_layers):
  """Every exporter target has a shape, including an incomplete scan cycle."""
  config = {
      "hidden_size": 24,
      "intermediate_size": 40,
      "vocab_size": 33,
      "num_hidden_layers": 5,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
  }
  maxtext_config = SimpleNamespace(inhomogeneous_layer_cycle_interval=4)
  mapping = param_mapping.PARAM_MAPPING[model_name](config, maxtext_config, scan_layers)
  hooks = param_mapping.HOOK_FNS[model_name](config, maxtext_config, scan_layers, saving_to_hf=True)
  targets = {target for value in mapping.values() for target in (value if isinstance(value, list) else [value])}
  shapes = hf_shape.HF_SHAPE[model_name](config)
  assert set(shapes) == targets
  assert len(shapes) == 58  # Three global tensors plus eleven per decoder layer.
  assert set(hooks) == set(mapping)


def test_olmo3_shapes_roundtrip_existing_hooks():
  """The new shapes also work with the existing kernel and norm adapters."""
  config = {
      "hidden_size": 24,
      "intermediate_size": 40,
      "vocab_size": 33,
      "num_hidden_layers": 1,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
  }
  maxtext_config = SimpleNamespace(inhomogeneous_layer_cycle_interval=4)
  mapping = param_mapping.PARAM_MAPPING["olmo3-7b"](config, maxtext_config)
  to_mt = param_mapping.HOOK_FNS["olmo3-7b"](config, maxtext_config)
  to_hf = param_mapping.HOOK_FNS["olmo3-7b"](config, maxtext_config, saving_to_hf=True)
  shapes = hf_shape.HF_SHAPE["olmo3-7b"](config)
  for mt_key, hf_key in mapping.items():
    shape = shapes[hf_key]
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mt_shape = shape[::-1] if mt_key.endswith("kernel") else shape
    converted = to_mt[mt_key](original, mt_shape)
    restored = to_hf[mt_key](converted, shape)
    np.testing.assert_array_equal(restored, original, err_msg=hf_key)


def test_llama_shape_mapping_is_unchanged():
  """OLMo3-specific norms must not leak into an existing model family."""
  shapes = hf_shape.HF_SHAPE["llama3.1-8b"](
      {
          "hidden_size": 24,
          "intermediate_size": 40,
          "vocab_size": 33,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 6,
      }
  )
  assert shapes["model.layers.0.input_layernorm.weight"] == [24]
  assert shapes["model.layers.0.self_attn.k_proj.weight"] == [12, 24]
  assert "model.layers.0.post_feedforward_layernorm.weight" not in shapes
  assert "model.layers.0.self_attn.q_norm.weight" not in shapes
