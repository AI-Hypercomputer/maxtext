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

"""Chained full-model layerwise forward of DeepSeek-V4-Flash (284B) at the e2e logit-test config.

Runs embedding + S1 (layers 0..2), B0..B18 (layers 3..40) and S3 (layers 41..42 + head) one unit at a time on a
small slice. Each unit restores only its own weights and consumes the previous unit's output, so the final logits
are a full 43-layer forward over the same tokens, padding, positions and config as forward_pass_logit_checker in
2_test_deepseek.sh. Reports KL vs the golden logits (checker method), optional KL vs the e2e logits, and per-unit
wall clock and memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_layerwise as vl  # pylint: disable=g-import-not-at-top,wrong-import-position

from absl import logging as absl_logging  # pylint: disable=wrong-import-order,wrong-import-position
import jax  # pylint: disable=wrong-import-order,wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-order,wrong-import-position
from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.trainers.pre_train import train_compile  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-order,wrong-import-position
import torch  # pylint: disable=wrong-import-order,wrong-import-position
import torch.nn.functional as F  # pylint: disable=wrong-import-order,wrong-import-position
import yaml  # pylint: disable=wrong-import-order,wrong-import-position

views = vl.views
UNITS = ("S1",) + tuple(f"B{b}" for b in range(vl.NUM_SCANNED_BLOCKS)) + ("S3",)
DEFAULT_GOLDEN = "gs://maxtext-test-assets/golden_data_deepseek4-284b.jsonl"


def make_chain_config(stage_kwargs: dict, aot_topology: str = "", overrides=()):
  """deepseek4-284b on a 5-layer (3 prefix + 1 scanned block) skeleton with the e2e stage flags."""
  cfg_dir = os.path.dirname(pyconfig.__file__)
  with open(os.path.join(cfg_dir, "models", "deepseek4-284b.yml"), encoding="utf-8") as f:
    compress_ratios = yaml.safe_load(f)["compress_ratios"]
  kwargs = {
      "model_name": "deepseek4-284b",
      "override_model_config": True,
      "run_name": "ds4_chain",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "base_num_decoder_layers": 5,
      "compress_ratios": list(compress_ratios[:5]),
      "scan_layers": True,
      "ici_fsdp_parallelism": -1,
      **stage_kwargs,
  }
  for kv in overrides:
    k, v = kv.split("=", 1)
    kwargs[k] = yaml.safe_load(v)
  if aot_topology:
    kwargs.update(compile_topology=aot_topology, compile_topology_num_slices=1)
  return pyconfig.initialize(["", os.path.join(cfg_dir, "base.yml")], **kwargs)


def logit_kwargs(args) -> dict:
  """2_test_deepseek.sh logit-stage flags."""
  return {
      "attention": "dot_product",
      "per_device_batch_size": 1,
      "indexer_topk": args.indexer_topk,
      "max_target_length": args.seq,
      "sparse_matmul": True,
      "megablox": False,
      "dtype": args.dtype,
      "weight_dtype": args.dtype,
      "activations_in_float32": False,
      "matmul_precision": "highest",
      "float32_logits": False,
      "float32_qk_product": False,
  }


def load_golden(path: str) -> dict:
  local = path
  if path.startswith("gs://"):
    local = os.path.join("/tmp", os.path.basename(path))
    if not os.path.exists(local):
      subprocess.run(["gcloud", "storage", "cp", path, local], check=True)
  with open(local, encoding="utf-8") as f:
    return json.loads(f.readline())


def kl_per_token(model_logits: np.ndarray, golden_logits: np.ndarray, clip_eps: float = 1e-6) -> np.ndarray:
  """Per-token D_KL(P_golden || Q_model) as forward_pass_logit_checker reports it (both sides clipped, renormalized)."""
  v = min(model_logits.shape[-1], golden_logits.shape[-1])
  m = torch.from_numpy(np.ascontiguousarray(model_logits[..., :v], dtype=np.float32))
  g = torch.from_numpy(np.ascontiguousarray(golden_logits[..., :v], dtype=np.float32))
  p = torch.clamp(F.softmax(g, dim=-1), min=clip_eps)
  q = torch.clamp(F.softmax(m, dim=-1), min=clip_eps)
  p = p / p.sum(dim=-1, keepdim=True)
  q = q / q.sum(dim=-1, keepdim=True)
  return (p * (torch.log(p) - torch.log(q))).sum(dim=-1).numpy()


def kl_summary(name: str, kl: np.ndarray, model_logits: np.ndarray, ref_logits: np.ndarray) -> dict:
  top1 = float(np.mean(np.argmax(model_logits, -1) == np.argmax(ref_logits, -1)))
  res = {"mean": float(kl.mean()), "max": float(kl.max()), "argmax": int(kl.argmax()), "kl0": float(kl[0]), "top1": top1}
  vl.log(
      f"KL vs {name}: mean={res['mean']:.4e} max={res['max']:.4e}@{res['argmax']} kl[0]={res['kl0']:.4e} top1={top1:.4f}"
  )
  return res


def device_mem_gib() -> dict:
  """Max over local devices of current and peak HBM bytes in use."""
  cur = peak = 0
  for d in jax.local_devices():
    stats = d.memory_stats() or {}
    cur = max(cur, stats.get("bytes_in_use", 0))
    peak = max(peak, stats.get("peak_bytes_in_use", 0))
  return {"hbm_in_use_gib": cur / 2**30, "hbm_peak_gib": peak / 2**30}


def head_logits(sv, h, params, bias):
  m = sv._merge(params, bias)  # pylint: disable=protected-access
  return m.decoder.apply_output_head(m.token_embedder, m.decoder.hc_head(h), True, views.MODEL_MODE_TRAIN)


def unit_fn(sv, unit: str, tokens, segs, pos):
  """(h, params, bias) -> unit output; S1 ignores h and embeds tokens, S3 returns (h_out, logits)."""
  if unit == "S1":
    return lambda h, p, b: sv.s1(sv.embed(p, b, tokens, pos), p, b, tokens, segs, pos)
  if unit == "S3":

    def s3(h, p, b):
      h_out = sv.blocks(h, p, b, tokens, segs, pos)
      return h_out, head_logits(sv, h_out, p, b)

    return s3
  return lambda h, p, b: sv.blocks(h, p, b, tokens, segs, pos)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
  parser.add_argument("--seq", type=int, default=512)
  parser.add_argument("--indexer_topk", type=int, default=4)
  parser.add_argument("--golden_logits_path", default=DEFAULT_GOLDEN)
  parser.add_argument("--e2e_logits_path", default="", help="forward_pass_logit_checker --output_logits_path jsonl")
  parser.add_argument("--unscanned_ckpt", default=vl.DEFAULT_UNSCANNED_CKPT)
  parser.add_argument("--tid2eid_path", default=vl.DEFAULT_TID2EID_PATH)
  parser.add_argument("--units", default=",".join(UNITS), help="prefix of the chain, for smoke runs")
  parser.add_argument("--override", action="append", default=[], help="extra pyconfig key=value (e2e LOGIT_EXTRA_FLAGS)")
  parser.add_argument("--save_dir", default="", help="local or gs:// dir for per-unit h_out and final logits")
  parser.add_argument("--out_json", default="")
  parser.add_argument("--max_kl", type=float, default=0.0, help="assert max KL vs golden below this when > 0")
  parser.add_argument("--aot_topology", default="", help="e.g. v5p-8: only AOT-compile each distinct unit function")
  args = parser.parse_args()
  absl_logging.set_verbosity(absl_logging.WARNING)
  units = args.units.split(",")
  if units != list(UNITS[: len(units)]):
    raise ValueError(f"--units must be a prefix of {UNITS}")

  if args.aot_topology:
    # B1..B18 compile to B0's program.
    units = [u for u in units if u in ("S1", "B0", "S3")]
  # B1..B18 lower to identical HLO, so the persistent cache compiles the block program once.
  jax.config.update("jax_compilation_cache_dir", "/tmp/jax_chain_cache")
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

  t_start = time.time()
  cfg = make_chain_config(logit_kwargs(args), args.aot_topology, args.override)
  vl.replicate_batch_axis(cfg)
  mesh = train_compile.get_topology_mesh(cfg) if args.aot_topology else views.maxtext_utils.get_mesh_from_config(cfg)
  model = vl.init_model_on_cpu(cfg, mesh)
  vl.log(
      f"Config: dtype={args.dtype} seq={args.seq} indexer_topk={cfg.indexer_topk} overrides={args.override} "
      f"mesh={dict(mesh.shape)}"
  )

  golden = load_golden(args.golden_logits_path)
  seq_len = len(golden["tokens"])
  ids = np.zeros((1, args.seq), np.int32)
  ids[0, :seq_len] = golden["tokens"]
  segs_np = np.zeros((1, args.seq), np.int32)
  segs_np[0, :seq_len] = 1
  tokens, segs = jnp.asarray(ids), jnp.asarray(segs_np)
  pos = jnp.arange(args.seq, dtype=jnp.int32)[None]
  act_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
  h_shape = (1, args.seq, cfg.mhc_expansion_rate, cfg.emb_dim)

  h = jnp.zeros(h_shape, act_dtype)
  timings, saved = [], {}
  logits = None
  for unit in units:
    t0 = time.time()
    vl.restore_subgroup_weights(
        model,
        cfg,
        mesh,
        unit,
        1,
        unscanned_ckpt=args.unscanned_ckpt,
        tid2eid_path=args.tid2eid_path,
        abstract=bool(args.aot_topology),
        with_globals=unit == "S1",
    )
    t_restore = time.time() - t0
    sv = views.MaxTextSubgroups(model, cfg, mesh)
    fn = unit_fn(sv, unit, tokens, segs, pos)
    t1 = time.time()
    with views.maxtext_context(cfg, mesh):
      if args.aot_topology:
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        h_abs = jax.ShapeDtypeStruct(h_shape, act_dtype, sharding=rep)
        mem = jax.jit(fn).lower(h_abs, sv.params, sv.bias).compile().memory_analysis()
        vl.log(
            f"AOT OK {unit} ({args.aot_topology}): args {mem.argument_size_in_bytes / 2**30:.2f} GiB, temp "
            f"{mem.temp_size_in_bytes / 2**30:.2f} GiB per device"
        )
        del sv, fn
        gc.collect()
        continue
      out = jax.jit(fn)(h, sv.params, sv.bias)
      jax.block_until_ready(out)
    t_run = time.time() - t1
    if unit == "S3":
      h, logits = out
    else:
      h = out
    saved[f"h_out.{unit}"] = np.asarray(h, dtype=np.float32)
    rec = {"unit": unit, "restore_s": t_restore, "run_s": t_run, "rss_peak_gib": vl.rss_gib(), **device_mem_gib()}
    timings.append(rec)
    vl.log(f"UNIT {unit}: restore {t_restore:.1f}s run(compile+exec) {t_run:.1f}s hbm_peak {rec['hbm_peak_gib']:.1f} GiB")
    del sv, fn, out
    gc.collect()

  if args.aot_topology:
    vl.log(f"AOT PASSED for {units}.")
    return
  total_s = time.time() - t_start
  results = {"dtype": args.dtype, "seq": args.seq, "overrides": args.override, "units": timings, "total_s": total_s}
  vl.log(f"Chain of {len(units)} units finished in {total_s:.1f}s.")

  if logits is not None:
    model_logits = np.asarray(logits, dtype=np.float32)[0, :seq_len]
    golden_logits = np.asarray(golden["logits"], dtype=np.float32)
    results["kl_vs_golden"] = kl_summary("golden", kl_per_token(model_logits, golden_logits), model_logits, golden_logits)
    saved["logits"] = model_logits
    if args.e2e_logits_path:
      e2e = load_golden(args.e2e_logits_path)
      e2e_logits = np.asarray(e2e["logits"], dtype=np.float32)[:seq_len]
      results["kl_vs_e2e"] = kl_summary("e2e", kl_per_token(model_logits, e2e_logits), model_logits, e2e_logits)
      results["kl_e2e_vs_golden"] = kl_summary(
          "golden (e2e logits)", kl_per_token(e2e_logits, golden_logits), e2e_logits, golden_logits
      )

  if args.save_dir:
    vl.save_outputs(os.path.join(args.save_dir, f"chain_{args.dtype}.safetensors"), saved)
  if args.out_json:
    with open(args.out_json, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
  print("CHAIN_RESULTS " + json.dumps({k: v for k, v in results.items() if k != "units"}), flush=True)
  if args.max_kl > 0 and not results.get("kl_vs_golden", {}).get("max", np.inf) < args.max_kl:
    raise AssertionError(f"chained max KL vs golden {results.get('kl_vs_golden')} >= {args.max_kl}")


if __name__ == "__main__":
  main()
