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

"""Tier 0 startup configuration and CLI-override audit against HuggingFace config.json."""

from __future__ import annotations

import json
import pathlib
from typing import Any

DEEPSEEK_V4_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("hidden_size", "emb_dim"),
    ("num_hidden_layers", "base_num_decoder_layers"),
    ("num_attention_heads", "num_query_heads"),
    ("num_key_value_heads", "num_kv_heads"),
    ("n_routed_experts", "num_experts"),
    ("n_shared_experts", "shared_experts"),
    ("num_experts_per_tok", "num_experts_per_tok"),
    ("moe_intermediate_size", "moe_mlp_dim"),
    ("intermediate_size", "mlp_dim"),
    ("q_lora_rank", "q_lora_rank"),
    ("kv_lora_rank", "kv_lora_rank"),
    ("qk_rope_head_dim", "qk_rope_head_dim"),
    ("qk_nope_head_dim", "qk_nope_head_dim"),
    ("v_head_dim", "v_head_dim"),
    ("index_n_heads", "indexer_n_heads"),
    ("index_head_dim", "indexer_head_dim"),
    ("index_topk", "indexer_topk"),
    ("rms_norm_eps", "normalization_layer_epsilon"),
    ("routed_scaling_factor", "routed_scaling_factor"),
    ("n_group", "n_routing_groups"),
    ("topk_group", "topk_routing_groups"),
    ("vocab_size", "vocab_size"),
)


def audit_deepseek_v4_config(
    maxtext_config: Any,
    hf_config_or_path: dict[str, Any] | str | pathlib.Path,
) -> list[dict[str, Any]]:
  """Compares a resolved MaxText pyconfig against an official HuggingFace DeepSeek-V4 config.

  Args:
    maxtext_config: Resolved MaxText config object or mapping.
    hf_config_or_path: Parsed HF config.json dict or path to config.json.

  Returns:
    List of mismatch dictionaries. Empty list indicates full compliance.
  """
  if isinstance(hf_config_or_path, (str, pathlib.Path)):
    hf_cfg = json.loads(pathlib.Path(hf_config_or_path).read_text())
  else:
    hf_cfg = dict(hf_config_or_path)

  get_mt = (lambda k: maxtext_config[k]) if isinstance(maxtext_config, dict) else (lambda k: getattr(maxtext_config, k))

  mismatches: list[dict[str, Any]] = []
  for hf_key, mt_key in DEEPSEEK_V4_FIELD_MAP:
    if hf_key not in hf_cfg:
      continue
    hf_val = hf_cfg[hf_key]
    mt_val = get_mt(mt_key)
    if hf_val != mt_val:
      mismatches.append(
          {
              "hf_key": hf_key,
              "mt_key": mt_key,
              "hf_val": hf_val,
              "mt_val": mt_val,
          }
      )

  if not bool(get_mt("float32_gate_logits")):
    mismatches.append(
        {
            "hf_key": "Gate.forward(fp32_policy)",
            "mt_key": "float32_gate_logits",
            "hf_val": True,
            "mt_val": get_mt("float32_gate_logits"),
        }
    )
  return mismatches
