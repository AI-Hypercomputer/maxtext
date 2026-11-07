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

"""Audits 1:1 parameter parity between a HuggingFace config.json and a MaxText model YAML."""

import argparse
import json
import math
import sys
from typing import Any
import yaml

# Required core Transformer keys (HF key -> MaxText key).
REQUIRED_KEY_MAP: dict[str, str] = {
    "hidden_size": "base_emb_dim",
    "num_hidden_layers": "base_num_decoder_layers",
    "num_attention_heads": "base_num_query_heads",
    "num_key_value_heads": "base_num_kv_heads",
    "intermediate_size": "base_mlp_dim",
    "rms_norm_eps": "normalization_layer_epsilon",
    "vocab_size": "vocab_size",
}

# Optional keys across common model families (Llama, Gemma, Qwen, Mixtral, DeepSeek, GPT-OSS).
OPTIONAL_KEY_MAP: dict[str, str | list[str]] = {
    "rope_theta": "rope_max_timescale",
    "max_position_embeddings": [
        "max_position_embeddings",
        "base_max_position_embeddings",
    ],
    "head_dim": "head_dim",
    "tie_word_embeddings": "logits_via_embedding",
    "sliding_window": "sliding_window_size",
    "attn_logit_softcapping": "attn_logit_softcapping",
    "final_logit_softcapping": "final_logit_softcapping",
    "query_pre_attn_scalar": "query_pre_attn_scalar",
    "moe_intermediate_size": "base_moe_mlp_dim",
    "num_local_experts": "num_experts",
    "num_experts": "num_experts",
    "n_routed_experts": "num_experts",
    "num_experts_per_tok": "num_experts_per_tok",
    "n_shared_experts": "shared_experts",
    "routed_scaling_factor": "routed_scaling_factor",
    "n_group": "n_routing_groups",
    "topk_group": "topk_routing_group",
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "v_head_dim": "v_head_dim",
}

ROPE_SCALING_KEY_MAP: dict[str, str] = {
    "factor": "rope_factor",
    "original_max_position_embeddings": "original_max_position_embeddings",
    "low_freq_factor": "rope_low_freq_factor",
    "high_freq_factor": "rope_high_freq_factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "mscale": "mscale",
    "mscale_all_dim": "mscale_all_dim",
}


def values_match(v1: Any, v2: Any, rel_tol: float = 1e-6) -> bool:
  if isinstance(v1, bool) or isinstance(v2, bool):
    return bool(v1) is bool(v2)
  if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
    return math.isclose(float(v1), float(v2), rel_tol=rel_tol, abs_tol=1e-12)
  return v1 == v2


def _resolve_mt_key(mt_cfg: dict[str, Any], candidates: str | list[str]) -> str | None:
  if isinstance(candidates, str):
    return candidates if candidates in mt_cfg else None
  for c in candidates:
    if c in mt_cfg:
      return c
  return None


def audit_configs(
    hf_cfg: dict[str, Any],
    mt_cfg: dict[str, Any],
    extra_map: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str, Any, Any]], list[tuple[str, str, Any, Any]]]:
  """Compares HF config.json and MaxText YAML, returning (matches, mismatches)."""
  if "text_config" in hf_cfg and isinstance(hf_cfg["text_config"], dict):
    hf_cfg = hf_cfg["text_config"]

  matches: list[tuple[str, str, Any, Any]] = []
  mismatches: list[tuple[str, str, Any, Any]] = []
  checked_hf_keys: set[str] = set()

  def check_pair(hf_label: str, mt_label: str, hf_val: Any, mt_val: Any) -> None:
    if mt_val is not None and values_match(hf_val, mt_val):
      matches.append((hf_label, mt_label, hf_val, mt_val))
    else:
      mismatches.append((hf_label, mt_label, hf_val, mt_val))

  for hf_key, mt_key in REQUIRED_KEY_MAP.items():
    if hf_key in hf_cfg:
      checked_hf_keys.add(hf_key)
      check_pair(hf_key, mt_key, hf_cfg[hf_key], mt_cfg.get(mt_key))

  combined_optional = dict(OPTIONAL_KEY_MAP)
  if extra_map:
    combined_optional.update(extra_map)

  for hf_key, mt_candidates in combined_optional.items():
    if hf_key in hf_cfg and hf_cfg[hf_key] is not None:
      checked_hf_keys.add(hf_key)
      mt_key = _resolve_mt_key(mt_cfg, mt_candidates)
      if mt_key is not None:
        check_pair(hf_key, mt_key, hf_cfg[hf_key], mt_cfg[mt_key])

  rope_cfg = hf_cfg.get("rope_scaling")
  if isinstance(rope_cfg, dict):
    checked_hf_keys.add("rope_scaling")
    for hf_subkey, mt_key in ROPE_SCALING_KEY_MAP.items():
      if hf_subkey in rope_cfg and rope_cfg[hf_subkey] is not None and mt_key in mt_cfg:
        check_pair(
            f"rope_scaling.{hf_subkey}",
            mt_key,
            rope_cfg[hf_subkey],
            mt_cfg[mt_key],
        )

  # Auto-discover any remaining scalar keys with identical names in both configs.
  for key, hf_val in hf_cfg.items():
    if key in checked_hf_keys or not isinstance(hf_val, (int, float, str, bool)):
      continue
    if key in mt_cfg and isinstance(mt_cfg[key], (int, float, str, bool)):
      check_pair(key, key, hf_val, mt_cfg[key])

  return matches, mismatches


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hf-config", required=True, help="Path to HuggingFace config.json")
  parser.add_argument("--maxtext-yml", required=True, help="Path to MaxText model .yml")
  parser.add_argument(
      "--key-map",
      default=None,
      help="Optional JSON file mapping custom HF keys to MaxText keys",
  )
  args = parser.parse_args()

  with open(args.hf_config, "r", encoding="utf-8") as f:
    hf_cfg = json.load(f)
  with open(args.maxtext_yml, "r", encoding="utf-8") as f:
    mt_cfg = yaml.safe_load(f)
  extra_map = None
  if args.key_map:
    with open(args.key_map, "r", encoding="utf-8") as f:
      extra_map = json.load(f)

  matches, mismatches = audit_configs(hf_cfg, mt_cfg, extra_map=extra_map)
  print(f"{'HF Key':<36} | {'MaxText Key':<30} | {'HF Value':<14} | {'MT Value':<14} | Status")
  print("-" * 110)
  for hf_k, mt_k, hf_v, mt_v in matches:
    print(f"{hf_k:<36} | {mt_k:<30} | {str(hf_v):<14} | {str(mt_v):<14} | PASS")
  for hf_k, mt_k, hf_v, mt_v in mismatches:
    print(f"{hf_k:<36} | {mt_k:<30} | {str(hf_v):<14} | {str(mt_v):<14} | FAIL")

  if mismatches:
    print(f"\nAUDIT FAILED: {len(mismatches)} mismatch(es) detected.", file=sys.stderr)
    return 1
  print(f"\nAUDIT PASSED: All {len(matches)} mapped parameters match.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
