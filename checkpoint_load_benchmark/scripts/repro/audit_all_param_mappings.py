# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-model blast-radius audit across all PARAM_MAPPING models in MaxText.

Verifies for every `(model, scan_layers)` entry in `param_mapping.PARAM_MAPPING`:
  1. `dequantize.is_companion_scale_key` never triggers false positives on BF16 checkpoint metadata.
  2. `load_dynamic.rename_legacy_hf_keys` is a strict no-op on already-canonical mapped keys.
Exits non-zero if any collision, rename regression, or mapping construction failure occurs.
"""

import argparse
import sys
import types
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import dequantize
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils import param_mapping


def _flat_keys(item):
  if isinstance(item, (list, tuple)):
    for sub in item:
      yield from _flat_keys(sub)
  elif isinstance(item, str):
    yield item


def _flat_tuples(item):
  if isinstance(item, list):
    for sub in item:
      yield from _flat_tuples(sub)
  elif isinstance(item, tuple):
    yield item


def audit_param_mappings(models: list[str] | None = None, quiet: bool = False) -> int:
  """Audits `PARAM_MAPPING` models across `scan_layers={False, True}` and returns exit code (`0` = pass)."""
  print("=== Auditing PARAM_MAPPING models (scan_layers=False & True) ===")
  selected = sorted(models) if models else sorted(param_mapping.PARAM_MAPPING)
  total_checked = 0
  errors: list[str] = []

  for model in selected:
    cfg_key = model.replace("-Instruct", "")
    if cfg_key not in param_mapping.PARAM_MAPPING:
      errors.append(f"Unknown model {model!r} (not in PARAM_MAPPING)")
      continue
    if cfg_key not in hf_model_configs.HF_MODEL_CONFIGS:
      if not quiet:
        print(f"[SKIP NO HF_CONFIG] {model}")
      continue
    hf_cfg = hf_model_configs.HF_MODEL_CONFIGS[cfg_key].to_dict()
    q_cfg = hf_cfg.get("quantization_config")

    # Shrink layer/expert counts so mapping construction is fast while preserving structure.
    tiny_hf = dict(hf_cfg)
    if cfg_key != "kimi-k3":
      if isinstance(tiny_hf.get("num_hidden_layers"), int) and tiny_hf["num_hidden_layers"] > 8:
        tiny_hf["num_hidden_layers"] = 8
    for key in ("n_routed_experts", "num_local_experts", "num_experts"):
      if isinstance(tiny_hf.get(key), int) and tiny_hf[key] > 4:
        tiny_hf[key] = 4
    if "first_k_dense_replace" in tiny_hf and tiny_hf["first_k_dense_replace"] >= tiny_hf.get("num_hidden_layers", 8):
      tiny_hf["first_k_dense_replace"] = 1
    if "num_hash_layers" in tiny_hf and tiny_hf["num_hash_layers"] >= tiny_hf.get("num_hidden_layers", 8):
      tiny_hf["num_hash_layers"] = 2

    for scan in (False, True):
      mt_cfg = types.SimpleNamespace(
          model_name=model,
          scan_layers=scan,
          param_scan_axis=1,
          num_decoder_layers=tiny_hf.get("num_hidden_layers", 4),
          base_num_decoder_layers=tiny_hf.get("num_hidden_layers", 4),
          first_num_dense_layers=tiny_hf.get("first_k_dense_replace", 1),
          first_num_hash_layers=tiny_hf.get("num_hash_layers", 1),
          num_experts=tiny_hf.get("n_routed_experts", tiny_hf.get("num_local_experts", tiny_hf.get("num_experts", 2))),
          mtp_num_layers=1 if "deepseek" in model else 0,
          compress_ratios=tiny_hf.get("compress_ratios", [0, 0, 128, 4, 128, 4, 128, 4])[
              : tiny_hf.get("num_hidden_layers", 8)
          ],
          weight_block_size=None,
          quantization="fp8" if model.endswith("-fp8") else "",
          mhc_expansion_rate=tiny_hf.get("hc_mult", 2),
          o_groups=tiny_hf.get("o_groups", 2),
          layer_cycle_interval=tiny_hf.get("full_attention_interval", 2),
          inhomogeneous_layer_cycle_interval=tiny_hf.get("full_attention_interval", 4),
          attention_pattern_length=4,
          share_kv_projections=False,
          v_norm_with_scale=False,
          use_multimodal=False,
          ming_talker_dit_num_layers=2,
          ming_talker_aggregator_num_layers=2,
          ming_vae_num_layers=2,
          num_unscanned_layers=0,
      )
      try:
        pmap = param_mapping.PARAM_MAPPING[cfg_key](tiny_hf, mt_cfg, scan_layers=scan)
      except (ValueError, NotImplementedError) as exc:
        if scan and "Scan layers is not supported" in str(exc):
          continue
        msg = f"[PMAP ERR] {model} scan={scan}: {exc}"
        print(msg, file=sys.stderr)
        errors.append(msg)
        continue
      except Exception as exc:  # pylint: disable=broad-exception-caught
        msg = f"[PMAP ERR] {model} scan={scan}: {exc}"
        print(msg, file=sys.stderr)
        errors.append(msg)
        continue
      total_checked += 1
      hf_keys = set(_flat_keys(list(pmap.values())))
      tuples = set(_flat_tuples(list(pmap.values())))
      scale_suffix_keys = [k for k in hf_keys if k.endswith(("_scale_inv", ".scale"))]
      scale_any_keys = [k for k in hf_keys if "scale" in k]
      paired_mapped_scales = [k for k in scale_suffix_keys if dequantize.companion_weight_key(k) in hf_keys]

      # Check 1: Simulate BF16 checkpoint metadata; `is_companion_scale_key` must never fire on BF16 weights.
      bf16_meta = {k: jax.ShapeDtypeStruct((16, 16), jnp.bfloat16) for k in hf_keys}
      false_companions = [k for k in hf_keys if dequantize.is_companion_scale_key(k, bf16_meta)]
      if false_companions:
        msg = f"[BUG] False companion scale on BF16 for {model} (scan={scan}): {false_companions[:3]}"
        print(msg, file=sys.stderr)
        errors.append(msg)

      # Check 2: Legacy prefix rename on `{k: k}` must be a strict no-op for already-canonical keys.
      renamed = load_dynamic.rename_legacy_hf_keys({k: k for k in hf_keys}, pmap)
      if set(renamed.keys()) != hf_keys:
        msg = f"[BUG] rename_legacy_hf_keys mutated canonical keys for {model} (scan={scan})"
        print(msg, file=sys.stderr)
        errors.append(msg)

      if not quiet and not scan and (q_cfg or scale_any_keys or tuples or paired_mapped_scales):
        qm = q_cfg.get("quant_method") if isinstance(q_cfg, dict) else q_cfg
        print(
            f"  MODEL {model:28s} | q_cfg={str(qm):18s} | tuples={len(tuples):3d}"
            f" | scale_keys={len(scale_any_keys):3d} | paired_suffix={len(paired_mapped_scales)}"
        )

  if total_checked == 0:
    print("[FAIL] Audited 0 configurations.", file=sys.stderr)
    return 1
  if errors:
    print(f"[FAIL] {len(errors)} audit error(s) across {total_checked} configurations.", file=sys.stderr)
    return 1
  print(
      f"[PASS] Audited {total_checked} (model, scan_layers) configurations:"
      " 0 false scale collisions, 0 canonical-key rename regressions."
  )
  return 0


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models", nargs="+", default=None, help="Optional subset of PARAM_MAPPING model names.")
  parser.add_argument("--quiet", action="store_true", help="Print only summary and failure diagnostics.")
  args = parser.parse_args(argv)
  return audit_param_mappings(models=args.models, quiet=args.quiet)


if __name__ == "__main__":
  sys.exit(main())
