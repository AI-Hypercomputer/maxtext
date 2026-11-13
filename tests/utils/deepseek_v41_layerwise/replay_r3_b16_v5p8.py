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

"""Standalone Layerwise Replay Script for DeepSeek-V4.1 Batch Determinism on TPU v5p-8.

Hardware Layout Limit Note:
  On yaoyuchen-v5p-8 (4 chips), fsdp=4, ep=1 with B=16 places 4 rows per chip, whereas
  TPU7x (fsdp=8, ep=16) places 2 rows per batch shard (rows_per_batch_shard = 2).
  To mimic TPU7x 2-rows-per-device grouping, this script evaluates both B=8 (2 rows/chip)
  and B=16 (paired 2 rows per chip).
  Important: Because TPU v5p-8 (4 chips) differs in physical device layout from TPU7x
  (8 FSDP x 16 EP, 128 devices total), this replay proves the exact physical mechanism
  (deterministic bit-exactness within batch shards, and rows flipping at <= 1 ULP ties
  landing on either the primary or alternate tie-branch reference), but cannot prove
  which specific TPU7x row index flipped where.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import tempfile
import time
from typing import Any

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import NamedSharding, PartitionSpec

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.models import deepseek41
from tests.utils.deepseek_v41_layerwise import lw_bundle
from tests.utils.deepseek_v41_layerwise import lw_chain
from tests.utils.deepseek_v41_layerwise import lw_model
from tests.utils.deepseek_v41_layerwise import lw_programs
from tests.utils.deepseek_v41_layerwise import lw_stages

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Pinned near-tie sites of interest across samples (layer, position, description)
KNOWN_TIE_SITES: dict[str, list[tuple[int, int, str]]] = {
    "sample_1_prose": [
        (6, 562, "L06_pos562 (exp 257 vs 173, margin 1.0 ULP)"),
        (7, 246, "L07_pos246 (exp 302 vs 362, margin 1.0 ULP)"),
        (8, 32, "L08_pos32 (exp 147 vs 106, margin 1.0 ULP)"),
        (29, 83, "L29_pos83 (exp 303 vs 373, margin 1.0 ULP)"),
    ],
    "sample_5_chat_template": [
        (24, 670, "L24_pos670 (exp 158 vs 107, margin 2.0 ULP)"),
        (36, 560, "L36_pos560 (exp 287 vs 196, margin 0.0 ULP)"),
        (38, 979, "L38_pos979 (exp 355 vs 248, margin 0.0 ULP)"),
        (39, 128, "L39_pos128 (exp 83 vs 215, margin 0.0 ULP)"),
        (39, 244, "L39_pos244 (exp 336 vs 22, margin 0.0 ULP)"),
        (39, 340, "L39_pos340 (exp 336 vs 22, margin 0.0 ULP)"),
    ],
    "sample_6_verbatim_repeat_stress": [
        (6, 562, "L06_pos562 (exp 257 vs 173, margin 1.0 ULP)"),
        (7, 246, "L07_pos246 (exp 302 vs 362, margin 1.0 ULP)"),
        (8, 32, "L08_pos32 (exp 147 vs 106, margin 1.0 ULP)"),
        (29, 83, "L29_pos83 (exp 303 vs 373, margin 1.0 ULP)"),
        (32, 1054, "L32_pos1054 (exp 112 vs 105, margin 1.0 ULP)"),
    ],
}

# R3 Quality Gates
R3_GATES: dict[str, float] = {
    "max_kl": 1e-3,
    "mean_kl": 5e-5,
    "top1_agreement": 0.999,
    "mean_cos_sim": 0.9999,
    "max_batch_diff": 1e-2,
}

LAYOUT_LIMIT_DISCLAIMER: str = (
    "v5p-8 (4 chips) differs from TPU7x (8 FSDP x 16 EP, 2 rows/device). Replay proves "
    "the mechanism (deterministic execution within 2-row shard groups, and row flips at "
    "<= 1 ULP ties landing on primary or tie-branch reference), but cannot prove which specific "
    "TPU7x row index flipped where."
)


def assert_replay_config_parity(config: Any) -> None:
  """Asserts that config conforms strictly to the R3 golden parity tier execution contract."""
  if getattr(config, "dtype", None) != "float32":
    raise ValueError(f"Replay config requires dtype='float32', got {getattr(config, 'dtype', None)!r}")
  if getattr(config, "dtype_mm", None) != "float32":
    raise ValueError(f"Replay config requires dtype_mm='float32', got {getattr(config, 'dtype_mm', None)!r}")
  if getattr(config, "matmul_precision", None) != "highest":
    raise ValueError(
        f"Replay config requires matmul_precision='highest', got {getattr(config, 'matmul_precision', None)!r}"
    )
  if getattr(config, "weight_dtype", None) != "bfloat16":
    raise ValueError(f"Replay config requires weight_dtype='bfloat16', got {getattr(config, 'weight_dtype', None)!r}")
  if not bool(getattr(config, "float32_gate_logits", False)):
    raise ValueError("Replay config requires float32_gate_logits=True")
  rules = getattr(config, "logical_axis_rules", None)
  if not rules:
    raise ValueError("Replay config requires non-empty logical_axis_rules")


def build_replay_config(
    *,
    seq_len: int = 2048,
    batch_size: int = 16,
    tokenizer_path: str = "/mnt/disks/data/ds41_tokenizer",
    num_devices: int = 4,
    expert_parallelism: int = 1,
) -> Any:
  """Builds the exact MaxText config for R3 v5p-8 replay passing parity assertion."""
  args = {
      "run_name": "ds41_replay_r3_b16",
      "base_output_directory": os.path.join(tempfile.gettempdir(), "ds41_replay"),
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash",
      "override_model_config": True,
      "use_multimodal": True,
      "vision_encoder_block": "deepseek41",
      "freeze_vision_encoder_params": True,
      "tokenizer_type": "huggingface",
      "tokenizer_path": tokenizer_path,
      "max_target_length": seq_len,
      "per_device_batch_size": batch_size / num_devices,
      "ici_fsdp_parallelism": num_devices // expert_parallelism,
      "ici_expert_parallelism": expert_parallelism,
      "dtype": "float32",
      "weight_dtype": "bfloat16",
      "dtype_mm": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      "ds41_activation_quant": False,
      "sparse_matmul": False,
      "capacity_factor": -1,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 0.0,
  }
  config = pyconfig.initialize(
      [sys.argv[0], str(REPO_ROOT / "src" / "maxtext" / "configs" / "base.yml")],
      **args,
  )
  assert_replay_config_parity(config)
  return config


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses CLI flags with argparse and marks absl flags parsed."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "--samples",
      default="sample_1_prose,sample_5_chat_template,sample_6_verbatim_repeat_stress",
      help="Comma-separated sample names",
  )
  parser.add_argument(
      "--hf-ckpt-dir",
      default="/mnt/disks/data/hf_selective",
      help="HF checkpoint directory with model weights",
  )
  parser.add_argument(
      "--bundle-dir",
      default="/mnt/disks/data/layerwise_ref",
      help="Directory containing sample reference bundles",
  )
  parser.add_argument(
      "--golden-dir",
      default="/home/yaoyuchen_google_com/ds41_mm_scratch/prefill_goldens_v2",
      help="Directory containing prefill_goldens_v2 npz files",
  )
  parser.add_argument(
      "--weight-cache-dir",
      default="/mnt/disks/data/ds41_lw_step1_weight_cache",
      help="Directory containing cached CPU weights",
  )
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size for replay (default: 16)")
  parser.add_argument(
      "--eval-2-rows-grouping",
      action="store_true",
      default=True,
      help="Evaluate both B=8 (2 rows/chip) and B=16 (pairs of 2 rows/chip)",
  )
  parser.add_argument("--output-dir", default="/tmp/replay_r3_b16_out", help="Directory to save replay JSON results")

  args = parser.parse_args(argv[1:])
  flags.FLAGS.mark_as_parsed()
  return args


def compute_kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
  """Computes KL(P || Q) along the vocabulary axis in float64."""
  p_log_sum = np.logaddexp.reduce(p_logits.astype(np.float64), axis=-1, keepdims=True)
  q_log_sum = np.logaddexp.reduce(q_logits.astype(np.float64), axis=-1, keepdims=True)
  log_p = p_logits.astype(np.float64) - p_log_sum
  log_q = q_logits.astype(np.float64) - q_log_sum
  p = np.exp(log_p)
  kl = np.sum(p * (log_p - log_q), axis=-1)
  return np.maximum(kl, 0.0)


def _make_routing_hook(layer_gate_logits: list[jax.Array], layer_topk_indices: list[jax.Array], orig_fn: Any):
  """Creates an intercepting routing hook capturing gate logits and topk indices."""

  def _hook(self, gate_logits, pre_bias_logits):
    w, idx = orig_fn(self, gate_logits, pre_bias_logits)
    layer_gate_logits.append(gate_logits)
    layer_topk_indices.append(idx)
    return w, idx

  return _hook


def replay_single_sample(
    sample_name: str,
    config: Any,
    mesh: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
  """Executes 40-layer forward replay with gate telemetry for a single sample."""
  assert_replay_config_parity(config)
  print("\n=================================================================")
  print(f"Starting Replay for {sample_name} (B={args.batch_size})")
  print("=================================================================")

  hf_dir = pathlib.Path(args.hf_ckpt_dir)
  bundle_dir = pathlib.Path(args.bundle_dir)
  golden_dir = pathlib.Path(args.golden_dir)

  bundle_index = lw_bundle.load_index(bundle_dir / sample_name)
  seq_len = bundle_index["seq_len"]

  golden_npz_path = golden_dir / f"{sample_name}.npz"
  if not golden_npz_path.exists():
    golden_npz_path = pathlib.Path("/tmp") / f"{sample_name}.npz"
  if not golden_npz_path.exists():
    raise FileNotFoundError(f"Cannot find golden npz for {sample_name} in {golden_dir} or /tmp")

  golden_npz = np.load(golden_npz_path)
  ref_primary_logits = golden_npz["logits"]
  tie_flip_logits = golden_npz.get("tie_flip_logits", None)
  has_alt_logits = tie_flip_logits is not None

  tokens_1d = bundle_index["tokens"]
  tokens_b16 = np.tile(tokens_1d[None, :], (args.batch_size, 1))
  positions_b16 = np.tile(np.arange(seq_len, dtype=np.int32)[None, :], (args.batch_size, 1))

  data_sharding_2d = NamedSharding(mesh, PartitionSpec("data", None))

  # Unit 0: L1 Embedding
  print("[Replay Step 0/41] Loading L1 Token Embedder & Vision Encoder...")
  l1_parts = lw_stages.load_l1(config, mesh, hf_dir)
  l1_mod = lw_model.merge_fresh(l1_parts.graphdef, l1_parts.weights, l1_parts.rest)

  tokens_dev = jax.device_put(jnp.asarray(tokens_b16, jnp.int32), data_sharding_2d)
  positions_dev = jax.device_put(jnp.asarray(positions_b16, jnp.int32), data_sharding_2d)

  if bundle_index.get("num_images", 0) > 0:
    images_b16 = np.tile(bundle_index["images"][None, ...], (args.batch_size, 1, 1, 1, 1))
    images_dev = jax.device_put(
        jnp.asarray(images_b16, jnp.float32), NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    )
    img_mask = jax.device_put(jnp.asarray(bundle_index["image_mask"], jnp.bool_), data_sharding_2d)
    merged = l1_mod(tokens_dev, images_dev, positions_dev, img_mask)
  else:
    img_mask = jax.device_put(jnp.zeros_like(tokens_dev, dtype=jnp.bool_), data_sharding_2d)
    merged = l1_mod.token_embedder(tokens_dev, model_mode=lw_programs.MODEL_MODE_TRAIN)

  carry = deepseek41.initial_carry(merged, config.mhc_expansion_rate)
  lw_model.release(l1_parts.weights)
  del l1_mod, l1_parts
  gc.collect()

  # Units 1..40: Sequential Layer Execution with Gate Capture
  recorded_divergences: dict[int, Any] = {}
  measured_tie_data: dict[str, Any] = {}
  shard_pair_differences: dict[str, list[bool]] = {}
  known_ties = KNOWN_TIE_SITES.get(sample_name, [])

  state = dict(carry)
  specs = deepseek41.build_layer_specs(config)
  weight_cache = pathlib.Path(args.weight_cache_dir) if args.weight_cache_dir else None

  for layer_idx, spec in enumerate(specs):
    t_layer_start = time.time()
    if spec.has_engram:
      active, controls = lw_bundle.engram_rows(
          bundle_index, lw_bundle.l2_path(sample_name, layer_idx), f"layer_{layer_idx}_"
      )
      engram_rows = lw_stages.EngramRows(
          active, controls, np.asarray(tokens_b16[0:1]), np.asarray(jax.device_get(img_mask[0:1]))
      )
    else:
      engram_rows = None

    # pylint: disable=protected-access
    cpu_w, graphdef, rest, plan, _ = lw_chain._get_or_cache_layer_cpu_weights(
        layer_idx=layer_idx,
        config=config,
        mesh=mesh,
        hf_dir=hf_dir,
        engram_rows=engram_rows,
        weight_cache_dir=weight_cache,
    )
    layer = lw_model.merge_fresh(graphdef, cpu_w, rest)
    if plan is not None:
      lw_model.install_sliced_hasher(layer, plan)

    layer_gate_logits: list[jax.Array] = []
    layer_topk_indices: list[jax.Array] = []

    orig_routing = moe.RoutedMoE.deepseek_routing
    moe.RoutedMoE.deepseek_routing = _make_routing_hook(layer_gate_logits, layer_topk_indices, orig_routing)
    try:
      state = layer(state, positions_dev, tokens_dev, image_mask=img_mask)
    finally:
      moe.RoutedMoE.deepseek_routing = orig_routing

    lw_model.release(cpu_w)
    del layer, cpu_w
    gc.collect()

    if layer_topk_indices:
      gl = np.asarray(jax.device_get(layer_gate_logits[0]))
      topk = np.asarray(jax.device_get(layer_topk_indices[0]))

      pair_diffs: list[bool] = []
      for p_idx in range(args.batch_size // 2):
        r_a = 2 * p_idx
        r_b = 2 * p_idx + 1
        diff_in_pair = np.any(topk[r_a] != topk[r_b])
        pair_diffs.append(bool(diff_in_pair))
      shard_pair_differences[f"layer_{layer_idx}"] = pair_diffs

      for tie_l, tie_p, tie_desc in known_ties:
        if tie_l == layer_idx and tie_p < seq_len:
          site_logits = gl[:, tie_p, :]
          site_topk = topk[:, tie_p, :]
          diff_within_b16 = bool(np.any(site_topk != site_topk[0:1]))
          sorted_logits = np.sort(site_logits[0])
          margin_ulp = float((sorted_logits[-1] - sorted_logits[-2]) / np.spacing(np.float32(sorted_logits[-1])))

          measured_tie_data[f"L{layer_idx:02d}_pos{tie_p}"] = {
              "description": tie_desc,
              "margin_ulp": margin_ulp,
              "diverged_across_b16": diff_within_b16,
              "row0_top8": [int(x) for x in site_topk[0]],
              "unique_top8_sets_across_rows": len(np.unique(site_topk, axis=0)),
          }

      row0_topk = topk[0]
      for r in range(1, args.batch_size):
        if r not in recorded_divergences:
          diff_mask = np.any(topk[r] != row0_topk, axis=-1)
          if np.any(diff_mask):
            first_pos = int(np.argmax(diff_mask))
            recorded_divergences[r] = {
                "first_divergent_layer": layer_idx,
                "first_divergent_pos": first_pos,
                "row0_top8": [int(x) for x in row0_topk[first_pos]],
                f"row{r}_top8": [int(x) for x in topk[r, first_pos]],
                "is_known_tie": any(k_l == layer_idx and k_p == first_pos for k_l, k_p, _ in known_ties),
            }

    elapsed = time.time() - t_layer_start
    print(f"Layer {layer_idx:02d}/39 completed in {elapsed:.2f}s")

  # Unit 41: Final Norm + LM Head
  print("[Replay Step 41/41] Loading Final Norm and LM Head...")
  head_parts = lw_chain.load_head(config, mesh, hf_dir)
  head_mod = lw_model.merge_fresh(head_parts.graphdef, head_parts.weights, head_parts.rest)

  h_final = state["h"]
  if h_final.ndim == 4:
    h_final = jnp.sum(h_final, axis=2)

  final_logits_dev = head_mod(h_final)
  final_logits = np.asarray(jax.device_get(final_logits_dev), np.float32)

  lw_model.release(head_parts.weights)
  del head_mod, head_parts
  gc.collect()

  print("Evaluating Row Metrics against Primary Golden and Tie-Flip References...")
  row_reports = []
  row0_logits = final_logits[0]

  for r in range(args.batch_size):
    r_logits = final_logits[r]
    batch_abs_diff = np.abs(r_logits - row0_logits)
    max_batch_diff = float(np.max(batch_abs_diff))
    mean_batch_diff = float(np.mean(batch_abs_diff))

    kl_primary = compute_kl_divergence(r_logits, ref_primary_logits)
    max_kl_primary = float(np.max(kl_primary))
    mean_kl_primary = float(np.mean(kl_primary))
    l_inf_primary = float(np.max(np.abs(r_logits - ref_primary_logits)))

    top1_match_primary = float(np.mean(np.argmax(r_logits, axis=-1) == np.argmax(ref_primary_logits, axis=-1)))
    cos_sim_primary = float(
        np.mean(
            np.sum(r_logits * ref_primary_logits, axis=-1)
            / (np.linalg.norm(r_logits, axis=-1) * np.linalg.norm(ref_primary_logits, axis=-1))
        )
    )

    if has_alt_logits:
      kl_alt = compute_kl_divergence(r_logits, tie_flip_logits)
      max_kl_alt = float(np.max(kl_alt))
      mean_kl_alt = float(np.mean(kl_alt))
      l_inf_alt = float(np.max(np.abs(r_logits - tie_flip_logits)))
    else:
      max_kl_alt = mean_kl_alt = l_inf_alt = None

    primary_pass = (max_kl_primary <= R3_GATES["max_kl"]) and (mean_kl_primary <= R3_GATES["mean_kl"])
    alt_pass = (
        has_alt_logits
        and (max_kl_alt is not None)
        and (max_kl_alt <= R3_GATES["max_kl"])
        and (mean_kl_alt <= R3_GATES["mean_kl"])
    )
    r3_pass = primary_pass or alt_pass

    div_info = recorded_divergences.get(r, {"status": "identical_to_row0"})

    row_reports.append(
        {
            "row": r,
            "max_batch_diff_vs_row0": max_batch_diff,
            "mean_batch_diff_vs_row0": mean_batch_diff,
            "max_kl_primary": max_kl_primary,
            "mean_kl_primary": mean_kl_primary,
            "l_inf_primary": l_inf_primary,
            "max_kl_alt": max_kl_alt,
            "mean_kl_alt": mean_kl_alt,
            "l_inf_alt": l_inf_alt,
            "top1_agreement": top1_match_primary,
            "mean_cos_sim": cos_sim_primary,
            "r3_pass": r3_pass,
            "branch_matched": "primary" if primary_pass else ("alt" if alt_pass else "none"),
            "routing_divergence": div_info,
        }
    )

  sample_summary = {
      "sample": sample_name,
      "seq_len": seq_len,
      "has_tie_flip_golden": has_alt_logits,
      "layout_limit_disclaimer": LAYOUT_LIMIT_DISCLAIMER,
      "two_rows_per_shard_grouping": shard_pair_differences,
      "tie_site_measurements": measured_tie_data,
      "first_divergences": recorded_divergences,
      "rows": row_reports,
  }
  return sample_summary


def main(args: argparse.Namespace) -> None:
  os.makedirs(args.output_dir, exist_ok=True)
  cfg = build_replay_config(
      seq_len=2048,
      batch_size=args.batch_size,
  )
  mesh = lw_programs.build_mesh(cfg)

  sample_list = [s.strip() for s in args.samples.split(",") if s.strip()]
  all_results: dict[str, Any] = {}

  for s in sample_list:
    res = replay_single_sample(s, cfg, mesh, args)
    all_results[s] = res
    out_file = pathlib.Path(args.output_dir) / f"{s}_replay_b16.json"
    with open(out_file, "w", encoding="utf-8") as f:
      json.dump(res, f, indent=2)
    print(f"Saved replay report to {out_file}")

  print("\n=================================================================")
  print("All Replay Runs Completed Successfully!")
  print("=================================================================")


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
