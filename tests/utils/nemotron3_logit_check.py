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

"""Router-tie-aware full-model logit check for Nemotron-3-Nano-Omni.

Nemotron's router biases sit within about 0.1 of each other, so exact fp32 top-6 ties occur on real text and a single
flipped expert choice spreads through the Mamba state. Free-routing KL is then dominated by ties rather than by
implementation error. This check runs the same full-model forward as `forward_pass_logit_checker` against HF goldens
(tests/assets/golden_logits/golden_data_nemotron3-nano-omni-30b-a3b.jsonl, falling back to
gs://maxtext-test-assets/golden_data_nemotron3-nano-omni-30b-a3b.jsonl; generated with a patched HF
reference whose generator is kept with the bringup harness) and adds router replay:

  free    MaxText routes freely (what `forward_pass_logit_checker` does). Reported only.
  replay  HF top-k is replayed in every MoE layer through `forced_routed_experts`, and MaxText's own top-k is recorded
          through `return_routed_experts`. If MaxText's own choice equals HF's at every cell that is not a near-tie
          (HF rank-6/7 margin <= 2 fp32 ulp of the layer's max |bias|), this is the same computation as replaying HF
          only at disagreeing near-tie cells, because every layer then sees the same input.

Gates (pre-registered in the bringup DECISIONS.md):
  fp32 (D-P3-9): replay run has no disagreement outside near-ties, max KL <= 1e-4 (go/maxtext-model-bringup §6.1) and
                 mean KL < 1e-3, max KL <= 1e-2 (model-bringup skill, references/logits_checker.md).
  bf16 (D-P3-10): replay of HF bf16 routing; mean and max KL(HF_fp32 || MaxText) <= 1.25x the same statistic of
                 KL(HF_fp32 || HF_bf16), per prompt.

  python3 -m tests.utils.nemotron3_logit_check src/maxtext/configs/base.yml model_name=nemotron3-nano-omni-30b-a3b \
      load_parameters_path=gs://.../0/items scan_layers=false attention=dot_product dtype=float32 \
      activations_in_float32=true matmul_precision=high weight_dtype=float32 max_target_length=1025 \
      --golden_logits_path=golden_s1025.pkl --golden_dtype=fp32 --output_jsonl=p4_logits.jsonl
"""

import argparse
import json
import os
import pickle
import re
import sys
import time

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jsonlines
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
import numpy as np

FP32_DOC_MAX_KL = 1e-4
FP32_SKILL_MEAN_KL = 1e-3
FP32_SKILL_MAX_KL = 1e-2
BF16_FLOOR_RATIO = 1.25


def load_goldens(path: str) -> list[dict]:
  if path.endswith((".pkl", ".pickle")):
    with open(path, "rb") as f:
      return pickle.load(f)
  with jsonlines.open(path, "r") as f:
    return list(f)


def kl_per_position(ref_logits: np.ndarray, cand_logits: np.ndarray, chunk: int = 256) -> np.ndarray:
  """KL(P_ref || Q_cand) per position [S], float64."""
  out = np.zeros(ref_logits.shape[0], dtype=np.float64)
  for t0 in range(0, ref_logits.shape[0], chunk):
    r = ref_logits[t0 : t0 + chunk].astype(np.float64)
    q = cand_logits[t0 : t0 + chunk].astype(np.float64)
    log_p = r - r.max(-1, keepdims=True)
    log_p -= np.log(np.exp(log_p).sum(-1, keepdims=True))
    log_q = q - q.max(-1, keepdims=True)
    log_q -= np.log(np.exp(log_q).sum(-1, keepdims=True))
    out[t0 : t0 + chunk] = np.maximum((np.exp(log_p) * (log_p - log_q)).sum(-1), 0.0)
  return out


def nll(logits: np.ndarray, ids: np.ndarray) -> float:
  lg = logits[:-1].astype(np.float64)
  lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
  return float(np.mean(lse - lg[np.arange(lg.shape[0]), ids[1:]]))


def kl_stats(kl: np.ndarray) -> dict:
  return {
      "mean_kl": float(kl.mean()),
      "p95_kl": float(np.percentile(kl, 95)),
      "max_kl": float(kl.max()),
      "argmax_pos": int(kl.argmax()),
  }


def routing_disagreements(own: dict[int, np.ndarray], golden: dict, dtype: str, seq_len: int) -> dict:
  """Cells where MaxText's own top-k set differs from HF's; near-tie per Phase 2 D4."""
  hf_topk = np.asarray(golden[f"router_topk_{dtype}"])
  margins = np.asarray(golden[f"router_margin_{dtype}"])
  thresholds = np.asarray(golden["d4_threshold"])
  tie_cells, non_tie = 0, []
  for li, layer in enumerate(np.asarray(golden["moe_layers"]).tolist()):
    mine = np.sort(own[layer][:seq_len], axis=-1)
    ref = np.sort(hf_topk[li, :seq_len], axis=-1)
    for t in np.nonzero((mine != ref).any(-1))[0].tolist():
      margin = float(margins[li, t])
      if margin <= float(thresholds[li]):
        tie_cells += 1
      else:
        non_tie.append({"layer": layer, "pos": t, "margin_6_7": margin, "d4_threshold": float(thresholds[li])})
  return {"d4_tie_disagreements": tie_cells, "non_d4_disagreements": len(non_tie), "non_d4_cells": non_tie[:20]}


def build_forced(golden: dict, dtype: str, num_layers: int, batch: int, length: int) -> np.ndarray:
  """[batch, length, num_layers, k] HF top-k; non-MoE layers and padding positions hold -1 (unused slot)."""
  hf_topk = np.asarray(golden[f"router_topk_{dtype}"], dtype=np.int32)
  seq_len, k = hf_topk.shape[1], hf_topk.shape[2]
  forced = np.full((length, num_layers, k), -1, dtype=np.int32)
  for li, layer in enumerate(np.asarray(golden["moe_layers"]).tolist()):
    forced[:seq_len, layer, :] = hf_topk[li]
  return np.broadcast_to(forced, (batch, length, num_layers, k)).copy()


def own_topk_by_layer(intermediates) -> dict[int, np.ndarray]:
  """MaxText's own top-k per decoder layer, from the `selected_experts` intermediates."""
  out = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(intermediates):
    keys = [str(getattr(p, "key", getattr(p, "name", p))) for p in path]
    if "selected_experts" not in keys:
      continue
    layer = next((int(m.group(1)) for k in keys if (m := re.fullmatch(r"layers_(\d+)", k))), None)
    if layer is None:
      raise ValueError(f"selected_experts intermediate outside a decoder layer: {'/'.join(keys)}")
    arr = np.asarray(jax.experimental.multihost_utils.process_allgather(leaf, tiled=True))
    out[layer] = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0] if arr.ndim > 2 else arr
  return out


def aot_record(name: str, compiled, gate_gib: float) -> dict:
  """Builds an AOT memory footprint record for gate comparison."""
  m = compiled.memory_analysis()
  g = 2**30
  met = {
      "args_gib": m.argument_size_in_bytes / g,
      "out_gib": m.output_size_in_bytes / g,
      "temp_gib": m.temp_size_in_bytes / g,
      "alias_gib": m.alias_size_in_bytes / g,
  }
  met["peak_hbm_gib"] = met["args_gib"] + met["out_gib"] + met["temp_gib"] - met["alias_gib"]
  passed = met["peak_hbm_gib"] <= gate_gib
  return {
      "suite": "p4_aot",
      "check_id": name,
      "metrics": met,
      "gate_gib": gate_gib,
      "passed": passed,
      "status": "PASS" if passed else "FAIL",
  }


def main(config, args) -> int:
  assert config.return_routed_experts, "pass return_routed_experts=true so MaxText's own routing is recorded"
  assert not config.scan_layers, "nemotron3 supports only scan_layers=false"
  if args.gate == "d_p3_10" and args.golden_dtype != "bf16":
    raise ValueError("--gate=d_p3_10 compares against the HF bf16 floor; pass --golden_dtype=bf16.")
  if args.aot:
    # Compiles the exact programs for `compile_topology` without TPUs; the host path (goldens, forced arrays) still runs.
    from maxtext.trainers.pre_train import train_compile  # pylint: disable=import-outside-toplevel

    mesh = train_compile.get_topology_mesh(config)
    _, model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  else:
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  graphdef, state = nnx.split(model)

  @jax.jit
  def forward(step_state, ids, positions, segment_ids, forced):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      m = nnx.merge(graphdef, step_state)
      logits = m(
          decoder_input_tokens=ids,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          enable_dropout=False,
          forced_routed_experts=forced,
      )
      return logits, nnx.pop(m, nnx.Intermediate)

  @jax.jit
  def forward_free(step_state, ids, positions, segment_ids):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      m = nnx.merge(graphdef, step_state)
      logits = m(
          decoder_input_tokens=ids,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          enable_dropout=False,
      )
      return logits, nnx.pop(m, nnx.Intermediate)

  goldens = load_goldens(args.golden_logits_path)
  if args.max_records:
    goldens = goldens[: args.max_records]
  modes = [m for m in args.modes.split(",") if m]
  batch = max(1, int(config.global_batch_size_to_train_on or 1))
  length = config.max_target_length
  flags = {
      k: str(getattr(config, k))
      for k in (
          "dtype",
          "weight_dtype",
          "matmul_precision",
          "activations_in_float32",
          "attention",
          "sparse_matmul",
          "megablox",
          "ici_fsdp_parallelism",
          "ici_expert_parallelism",
          "ici_tensor_parallelism",
      )
  }
  records, failed = [], 0
  for gi, golden in enumerate(goldens):
    ids_1d = np.asarray(golden["tokens"], dtype=np.int32)
    seq_len = ids_1d.shape[0]
    assert seq_len <= length, f"golden length {seq_len} > max_target_length {length}"
    ids = np.zeros((batch, length), np.int32)
    ids[:, :seq_len] = ids_1d
    segment_ids = np.zeros((batch, length), np.int32)
    segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR
    positions = np.broadcast_to(np.arange(length, dtype=np.int32), (batch, length)).copy()
    hf_fp32 = np.asarray(golden["logits"], dtype=np.float32)[:seq_len]
    floor = None
    if args.golden_dtype == "bf16":
      hf_bf16 = np.asarray(golden["logits_bf16"], dtype=np.float32)[:seq_len]
      floor = kl_stats(kl_per_position(hf_fp32, hf_bf16))

    for mode in modes:
      t = time.time()
      if args.aot:
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        sds = [jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep) for x in (ids, positions, segment_ids)]
        if mode == "replay":
          forced = build_forced(golden, args.golden_dtype, config.num_decoder_layers, batch, length)
          compiled = forward.lower(state, *sds, jax.ShapeDtypeStruct(forced.shape, forced.dtype, sharding=rep)).compile()
        else:
          compiled = forward_free.lower(state, *sds).compile()
        rec = aot_record(
            f"{args.check_id_prefix}.aot.{config.compile_topology}.S{length}.{mode}", compiled, args.aot_gate_gib
        )
        rec["flags"] = flags
        failed += not rec["passed"]
        records.append(rec)
        print(f"NEMOTRON3_LOGIT_CHECK {json.dumps(rec)}", flush=True)
        continue
      if mode == "replay":
        forced = build_forced(golden, args.golden_dtype, config.num_decoder_layers, batch, length)
        logits, inter = forward(state, ids, positions, segment_ids, forced)
      else:
        logits, inter = forward_free(state, ids, positions, segment_ids)
      logits = np.asarray(jax.experimental.multihost_utils.process_allgather(logits[:1], tiled=True), np.float32)
      logits = logits.reshape(-1, length, logits.shape[-1])[0, :seq_len]
      routing = routing_disagreements(own_topk_by_layer(inter), golden, args.golden_dtype, seq_len)
      kl = kl_per_position(hf_fp32, logits)
      metrics = {
          **kl_stats(kl),
          "top1_agreement": float(np.mean(hf_fp32.argmax(-1) == logits.argmax(-1))),
          "nll_maxtext": nll(logits, ids_1d),
          "nll_hf_fp32": float(golden.get("nll_fp32", float("nan"))),
          **routing,
          "seq_len": seq_len,
          "seconds": round(time.time() - t, 1),
      }
      tag, passed, rule = "diagnostic", None, "reported"
      if mode == "replay" and args.gate == "d_p3_9":
        doc_ok = metrics["max_kl"] <= FP32_DOC_MAX_KL and routing["non_d4_disagreements"] == 0
        skill_ok = metrics["mean_kl"] < FP32_SKILL_MEAN_KL and metrics["max_kl"] <= FP32_SKILL_MAX_KL
        metrics.update(doc_max_kl_1e4=doc_ok, skill_mean_1e3_max_1e2=skill_ok)
        tag, passed, rule = "doc+skill", doc_ok and skill_ok, "D-P3-9"
      elif mode == "replay" and args.gate == "d_p3_10":
        metrics.update({f"floor_{k}": v for k, v in floor.items()})
        metrics["mean_ratio"] = metrics["mean_kl"] / max(floor["mean_kl"], 1e-30)
        metrics["max_ratio"] = metrics["max_kl"] / max(floor["max_kl"], 1e-30)
        metrics["kl_vs_hf_bf16_mean"] = float(kl_per_position(hf_bf16, logits).mean())
        tag, rule = "decision", "D-P3-10"
        passed = metrics["mean_ratio"] <= BF16_FLOOR_RATIO and metrics["max_ratio"] <= BF16_FLOOR_RATIO
      elif floor is not None:
        metrics.update({f"floor_{k}": v for k, v in floor.items()})
      rec = {
          "suite": "p4_logits",
          "check_id": f"{args.check_id_prefix}.{gi}.{args.golden_dtype}.{mode}",
          "tag": tag,
          "rule": rule,
          "prompt": str(golden.get("prompt", ""))[:80],
          "flags": flags,
          "golden": os.path.basename(args.golden_logits_path),
          "metrics": metrics,
          "passed": passed,
          "status": "INFO" if passed is None else ("PASS" if passed else "FAIL"),
      }
      failed += passed is False
      records.append(rec)
      print(f"NEMOTRON3_LOGIT_CHECK {json.dumps(rec)}", flush=True)

  if args.output_jsonl:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    with open(args.output_jsonl, "a", encoding="utf-8") as f:
      for rec in records:
        f.write(json.dumps(rec) + "\n")
  print(f"NEMOTRON3_LOGIT_CHECK_SUMMARY records={len(records)} failed={failed}", flush=True)
  return 1 if failed else 0


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "--golden_logits_path",
      required=True,
      help=".jsonl or .pkl golden, e.g. gs://maxtext-test-assets/golden_data_nemotron3-nano-omni-30b-a3b.jsonl",
  )
  parser.add_argument("--golden_dtype", choices=["fp32", "bf16"], default="fp32", help="HF routing to replay.")
  parser.add_argument("--modes", default="replay,free", help="Comma-separated subset of replay,free.")
  parser.add_argument("--gate", choices=["d_p3_9", "d_p3_10", "none"], default="d_p3_9")
  parser.add_argument("--output_jsonl", default="")
  parser.add_argument("--check_id_prefix", default="p4.logits")
  parser.add_argument("--max_records", type=int, default=0)
  parser.add_argument("--aot", action="store_true", help="Compile for compile_topology and report peak HBM; no run.")
  parser.add_argument("--aot_gate_gib", type=float, default=85.23, help="Peak HBM gate (90%% of v5p per-chip HBM).")
  test_args, remaining = parser.parse_known_args()
  sys.exit(main(pyconfig.initialize_pydantic([sys.argv[0]] + remaining), test_args))
