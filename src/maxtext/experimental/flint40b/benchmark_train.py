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

"""Multi-host 64-chip TPU v7x (`tpu7x:4x4x4`) trainer for `RecursiveSI/flint` 40B."""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
from concurrent import futures
import json
import os
import socket
import sys
import tempfile
import time
from typing import Any

from absl import app
from absl import flags
from etils import epath
import jax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
import numpy as np

from maxtext.experimental.flint40b import aot_compile
from maxtext.experimental.flint40b import data
from maxtext.experimental.flint40b import flint_checkpoint
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_train
from maxtext.experimental.flint40b import flint_types

_TPU_V7X_PEAK_BF16_TFLOPS_PER_TC: float = 1153.5  # 2,307 dense BF16 TFLOP/s per chip / 2 TensorCores.

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Local or gs:// output directory for `benchmark_summary.json`,"
    " `metrics.jsonl`, profiler traces, and optional XLA dumps.",
)
_STEPS = flags.DEFINE_integer(
    "steps",
    18,
    "Total training steps to execute (warmup + profile + steady-state).",
)
_BENCH_WARMUP_STEPS = flags.DEFINE_integer(
    "bench_warmup_steps",
    4,
    "Number of unprofiled warmup steps before profiler capture.",
)
_PROFILE_STEPS = flags.DEFINE_integer(
    "profile_steps",
    3,
    "Number of steps to capture with `jax.profiler` (0 disables).",
)
_PROFILE_ALL_PROCESSES = flags.DEFINE_boolean(
    "profile_all_processes",
    False,
    "Capture a profiler trace on every process instead of only process 0.",
)
_SEED = flags.DEFINE_integer(
    "seed",
    42,
    "Deterministic PRNG seed across all hosts.",
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    -1.0,
    "Peak AdamW learning rate (<0 selects config override, else 3e-5 on c4 or" " 1e-9 on synthetic).",
)
_CONFIG_OVERRIDES_JSON = flags.DEFINE_string(
    "config_overrides_json",
    "",
    "JSON dict of `flint_40b_config` overrides. Empty (default) runs the"
    " optimized 256K recipe (`flint_config.default_recipe_overrides()`); `{}`"
    " runs the unoptimized dataclass defaults.",
)
_EXTRA_LIBTPU_ARGS = flags.DEFINE_string(
    "extra_libtpu_args",
    "",
    "Additional space-separated XLA/libtpu compiler flags appended last.",
)
_OPTIMIZED_LIBTPU_ARGS = flags.DEFINE_boolean(
    "optimized_libtpu_args",
    True,
    "Append the tuned TPU v7x compiler flags" " (`flint_config.optimized_v7x_libtpu_args`) for the recipe shape.",
)
_DUMP_XLA = flags.DEFINE_boolean(
    "dump_xla",
    False,
    "Whether to dump the train-step HLO on every host into"
    " `<output_dir>/xla_dump/<hostname>` (requires --output_dir).",
)
_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    "synthetic",
    ["synthetic", "c4"],
    "Training token source (`synthetic` fixed batch or pre-tokenized `c4`" " ArrayRecord).",
)
_C4_DATA_DIR = flags.DEFINE_string(
    "c4_data_dir",
    "",
    "Pre-tokenized C4 ArrayRecord directory, file, or glob with wildcards only in"
    " the final path component (required for --dataset_type=c4).",
)
_C4_SHUFFLE_SEED = flags.DEFINE_integer(
    "c4_shuffle_seed",
    -1,
    "Optional deterministic epoch-shuffle seed for C4 (<0 disables shuffling).",
)
_CKPT_DIR = flags.DEFINE_string(
    "ckpt_dir",
    "",
    "Optional HuggingFace safetensors directory loaded via" " `load_hf_checkpoint`.",
)
_SKIP_JAX_DISTRIBUTED = flags.DEFINE_boolean(
    "skip_jax_distributed",
    False,
    "Skip `jax.distributed.initialize()` (e.g. single-process runs).",
)
_JAX_DISTRIBUTED_TIMEOUT = flags.DEFINE_integer(
    "jax_distributed_timeout",
    300,
    "Timeout in seconds for `jax.distributed.initialize()`.",
)


def _profile_dir(proc_idx: int) -> str:
  """Returns the `jax.profiler` trace directory for this process."""
  if not _OUTPUT_DIR.value:
    return tempfile.mkdtemp(prefix="flint40b_profile_")
  root = os.path.join(_OUTPUT_DIR.value, "profiler")
  if _PROFILE_ALL_PROCESSES.value:
    return os.path.join(root, f"proc_{proc_idx:03d}")
  return root


def _metrics_to_host(
    metrics: flint_types.FlintTrainMetrics,
) -> dict[str, float]:
  """Transfers scalar `FlintTrainMetrics` from device to Python floats."""
  loss, ce_loss, aux_loss, grad_norm, lr = jax.device_get(
      (
          metrics.loss,
          metrics.ce_loss,
          metrics.aux_loss,
          metrics.grad_norm,
          metrics.learning_rate,
      )
  )
  return {
      "loss": float(np.asarray(loss)),
      "ce_loss": float(np.asarray(ce_loss)),
      "aux_loss": float(np.asarray(aux_loss)),
      "grad_norm": float(np.asarray(grad_norm)),
      "learning_rate": float(np.asarray(lr)),
  }


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
  if _STEPS.value <= 0:
    raise app.UsageError(f"--steps must be >= 1, got {_STEPS.value}.")
  if _DATASET_TYPE.value == "c4" and not _C4_DATA_DIR.value:
    raise app.UsageError("--c4_data_dir is required when --dataset_type=c4.")
  if _DUMP_XLA.value and not _OUTPUT_DIR.value:
    raise app.UsageError("--output_dir is required when --dump_xla=true.")

  jax.config.update("jax_enable_compilation_cache", False)

  raw_overrides = _CONFIG_OVERRIDES_JSON.value.strip()
  overrides: dict[str, Any] = json.loads(raw_overrides) if raw_overrides else flint_config.default_recipe_overrides()
  if not isinstance(overrides, dict):
    raise app.UsageError("--config_overrides_json must decode to a JSON dict.")
  if _LEARNING_RATE.value >= 0.0:
    overrides["learning_rate"] = float(_LEARNING_RATE.value)
  elif "learning_rate" not in overrides:
    overrides["learning_rate"] = 3e-5 if _DATASET_TYPE.value == "c4" else 1e-9
  if "total_steps" not in overrides:
    overrides["total_steps"] = int(_STEPS.value)
  if "warmup_steps" not in overrides:
    overrides["warmup_steps"] = min(
        int(overrides["total_steps"]),
        max(1, round(0.1 * int(_STEPS.value))) if int(_STEPS.value) >= 10 else 0,
    )

  cfg = flint_config.flint_40b_config(**overrides)
  total_params, active_params, _, _ = flint_config.count_flint_params(cfg.model)
  total_tflops_per_tc, weight_tflops_per_tc, attn_tflops_per_tc = flint_config.calculate_tflops_per_device(cfg)

  xla_dump_dir: str | None = None
  if _DUMP_XLA.value:
    xla_dump_dir = os.path.join(_OUTPUT_DIR.value, "xla_dump", socket.gethostname())

  libtpu_args = aot_compile.resolve_libtpu_args(
      cfg,
      optimized_libtpu_args=bool(_OPTIMIZED_LIBTPU_ARGS.value),
      extra_libtpu_args=_EXTRA_LIBTPU_ARGS.value,
  )
  pjrt_compiler_opts, absl_synced = aot_compile.sync_compiler_and_dump_flags(
      libtpu_args,
      xla_dump_dir=xla_dump_dir,
      hlo_module_re=".*train_step.*",
  )

  if not _SKIP_JAX_DISTRIBUTED.value and not jax.distributed.is_initialized():
    jax.distributed.initialize(initialization_timeout=_JAX_DISTRIBUTED_TIMEOUT.value)

  devices = jax.devices()
  proc_idx = jax.process_index()
  proc_cnt = jax.process_count()
  is_primary = proc_idx == 0

  if is_primary:
    dev0: Any = devices[0]
    sample_coords = dev0.coords if hasattr(dev0, "coords") else None
    print(
        f"[Flint-40B] Initialized JAX cluster: processes={proc_cnt},"
        f" global_devices={len(devices)},"
        f" local_devices={jax.local_device_count()},"
        f" device_kind={devices[0].device_kind!r},"
        f" sample_coords={sample_coords}, absl_synced={absl_synced},"
        f" pjrt_opts={len(pjrt_compiler_opts)},"
        f" total_params={total_params:,}, active_params={active_params:,},"
        f" model_tflops_per_tc={total_tflops_per_tc:.2f}",
        flush=True,
    )

  mesh = flint_sharding.create_mesh(cfg.sharding, devices=list(devices))
  abs_state, abs_batch = aot_compile.make_abstract_sharded_inputs(cfg, mesh)

  t_comp0 = time.perf_counter()
  train_step_fn: Any = flint_train.make_train_step(cfg, mesh, donate_state=True)
  lowered = train_step_fn.lower(abs_state, abs_batch)
  t_lower = time.perf_counter() - t_comp0
  compiled = lowered.compile(compiler_options=pjrt_compiler_opts)
  t_compile = time.perf_counter() - t_comp0

  mem: Any = compiled.memory_analysis()
  if mem is None:
    raise RuntimeError("compiled.memory_analysis() returned None.")
  gib = float(1024**3)
  total_hbm_gib = (
      int(mem.generated_code_size_in_bytes)
      + int(mem.argument_size_in_bytes)
      + max(0, int(mem.output_size_in_bytes) - int(mem.alias_size_in_bytes))
      + int(mem.temp_size_in_bytes)
  ) / gib
  temp_gib = int(mem.temp_size_in_bytes) / gib
  host_temp_gib = int(mem.host_temp_size_in_bytes) / gib

  if is_primary:
    print(
        f"[Flint-40B] Compiled train_step in {t_compile:.2f}s"
        f" (lower={t_lower:.2f}s): total_aot_hbm={total_hbm_gib:.3f} GiB/TC,"
        f" temp={temp_gib:.3f} GiB/TC, host_temp={host_temp_gib:.3f} GiB/TC",
        flush=True,
    )

  t_init0 = time.perf_counter()
  loaded_params = None
  if _CKPT_DIR.value.strip():
    loaded_params = flint_checkpoint.load_hf_checkpoint(_CKPT_DIR.value.strip(), cfg, mesh)
  state = flint_train.init_train_state(jax.random.key(_SEED.value), mesh, cfg, params=loaded_params)

  batch_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
  c4_ds: data.ArrayRecordC4 | None = None
  prefetch_pool: futures.ThreadPoolExecutor | None = None

  if _DATASET_TYPE.value == "synthetic":
    synthetic_batch = data.make_synthetic_batch(
        0,
        cfg.train.global_batch_size,
        cfg.train.seq_len,
        cfg.model.vocab_size,
        batch_sharding,
        seed=_SEED.value + 1,
    )
    jax.block_until_ready(synthetic_batch)

    def next_batch() -> flint_types.FlintBatch:
      return synthetic_batch

  else:
    shuffle_seed = _C4_SHUFFLE_SEED.value if _C4_SHUFFLE_SEED.value >= 0 else None
    c4_ds = data.ArrayRecordC4(
        _C4_DATA_DIR.value,
        cfg.train.seq_len,
        shuffle_seed=shuffle_seed,
    )
    if is_primary:
      records_used = _STEPS.value * cfg.train.global_batch_size * c4_ds.chunks_per_row
      epoch_cov = (_STEPS.value * cfg.train.global_batch_size) / c4_ds.num_rows
      print(
          f"[Flint-40B] C4 ArrayRecord: {c4_ds.num_rows} rows"
          f" ({c4_ds.total_records} records,"
          f" chunks_per_seq={c4_ds.chunks_per_row}) of {cfg.train.seq_len}"
          f" tokens (shuffle_seed={shuffle_seed}, records_used={records_used},"
          f" epoch_coverage={epoch_cov:.4f})",
          flush=True,
      )
    prefetch_depth = min(8, _STEPS.value)
    prefetch_pool = futures.ThreadPoolExecutor(max_workers=prefetch_depth)
    pending_batches: collections.deque[tuple[int, futures.Future[dict[int, np.ndarray]]]] = collections.deque()
    next_submit_step = 0
    for _ in range(prefetch_depth):
      pending_batches.append(
          (
              next_submit_step,
              prefetch_pool.submit(
                  c4_ds.prefetch_host_rows,
                  next_submit_step,
                  cfg.train.global_batch_size,
                  batch_sharding,
              ),
          )
      )
      next_submit_step += 1

    def _next_c4_batch() -> flint_types.FlintBatch:
      nonlocal next_submit_step
      assert c4_ds is not None and prefetch_pool is not None
      step_i, fut_i = pending_batches.popleft()
      rows_i = fut_i.result()
      if next_submit_step < _STEPS.value:
        pending_batches.append(
            (
                next_submit_step,
                prefetch_pool.submit(
                    c4_ds.prefetch_host_rows,
                    next_submit_step,
                    cfg.train.global_batch_size,
                    batch_sharding,
                ),
            )
        )
        next_submit_step += 1
      return c4_ds.make_batch_from_prefetched(step_i, rows_i, cfg.train.global_batch_size, batch_sharding)

    next_batch: Callable[[], flint_types.FlintBatch] = _next_c4_batch

  jax.block_until_ready(state)
  multihost_utils.sync_global_devices("flint_init_done")
  t_init = time.perf_counter() - t_init0
  if is_primary:
    print(
        f"[Flint-40B] Sharded 40B state + {_DATASET_TYPE.value} data" f" initialized in {t_init:.2f}s",
        flush=True,
    )

  total_steps = _STEPS.value
  warmup_end = min(max(0, _BENCH_WARMUP_STEPS.value), total_steps)
  prof_end = min(warmup_end + max(0, _PROFILE_STEPS.value), total_steps)

  step_records: list[dict[str, Any]] = []
  per_step_times_s: list[float] = []
  steady_step_times_s: list[float] = []
  profiler_trace_dir: str | None = None

  def _record_step(
      step_idx: int,
      phase: str,
      dt_s: float,
      wall_ts_s: float,
      metrics_dev: flint_types.FlintTrainMetrics,
  ) -> None:
    m_host = _metrics_to_host(metrics_dev)
    rec: dict[str, Any] = {
        "step": step_idx,
        "phase": phase,
        "step_time_s": round(dt_s, 6),
        "step_time_ms": round(dt_s * 1000.0, 3),
        "wall_timestamp_s": round(wall_ts_s, 6),
        **m_host,
    }
    step_records.append(rec)
    per_step_times_s.append(dt_s)
    if phase == "steady":
      steady_step_times_s.append(dt_s)
    if is_primary:
      print(
          f"[{phase.capitalize():6s} step {step_idx:03d}]"
          f" step_time={dt_s:.4f}s ({dt_s * 1000.0:.1f} ms)"
          f" loss={m_host['loss']:.5f} ce={m_host['ce_loss']:.5f}"
          f" aux={m_host['aux_loss']:.5f} grad_norm={m_host['grad_norm']:.4f}"
          f" lr={m_host['learning_rate']:.3e}",
          flush=True,
      )

  try:
    # Phase 1: Unprofiled warmup steps.
    for step in range(warmup_end):
      t0 = time.perf_counter()
      state, metrics = compiled(state, next_batch())
      jax.block_until_ready(metrics)
      dt = time.perf_counter() - t0
      _record_step(step, "warmup", dt, time.time(), metrics)

    # Phase 2: Profiler capture window.
    if prof_end > warmup_end:
      multihost_utils.sync_global_devices("flint_pre_profile")
      profile_this_proc = is_primary or _PROFILE_ALL_PROCESSES.value
      trace_dir = _profile_dir(proc_idx) if profile_this_proc else None
      if trace_dir is not None:
        jax.profiler.start_trace(trace_dir)
      for step in range(warmup_end, prof_end):
        t0 = time.perf_counter()
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
          state, metrics = compiled(state, next_batch())
          jax.block_until_ready(metrics)
        dt = time.perf_counter() - t0
        _record_step(step, "profile", dt, time.time(), metrics)
      if trace_dir is not None:
        jax.profiler.stop_trace()
        print(
            f"[Profiler process {proc_idx:03d}] trace_dir={trace_dir}",
            flush=True,
        )
        if is_primary:
          profiler_trace_dir = trace_dir
      multihost_utils.sync_global_devices("flint_post_profile")

    # Phase 3: Unprofiled steady-state measurement with 1-step async overlap.
    rem_steps = total_steps - prof_end
    pipelined_step_time_s: float | None = None
    if rem_steps > 0:
      multihost_utils.sync_global_devices("flint_pre_steady")
      t_phase3_start = time.perf_counter()
      t_last_complete = t_phase3_start
      prev_step = prof_end
      state, prev_metrics = compiled(state, next_batch())

      for step in range(prof_end + 1, total_steps):
        batch_cur = next_batch()
        state, cur_metrics = compiled(state, batch_cur)
        jax.block_until_ready(prev_metrics)
        t_now = time.perf_counter()
        _record_step(
            prev_step,
            "steady",
            t_now - t_last_complete,
            time.time(),
            prev_metrics,
        )
        t_last_complete = t_now
        prev_step = step
        prev_metrics = cur_metrics

      jax.block_until_ready((state, prev_metrics))
      t_end = time.perf_counter()
      _record_step(
          prev_step,
          "steady",
          t_end - t_last_complete,
          time.time(),
          prev_metrics,
      )
      pipelined_step_time_s = (t_end - t_phase3_start) / rem_steps
  finally:
    if prefetch_pool is not None:
      prefetch_pool.shutdown(wait=True, cancel_futures=True)
    if c4_ds is not None:
      c4_ds.close()

  steady_samples = (
      steady_step_times_s[1:] if len(steady_step_times_s) >= 2 else (steady_step_times_s or per_step_times_s[-3:])
  )
  median_step_time_s = float(np.median(steady_samples))
  mean_step_time_s = float(np.mean(steady_samples))
  best_step_time_s = float(np.min(steady_samples))
  primary_step_s = median_step_time_s
  primary_step_ms = primary_step_s * 1000.0

  num_chips = max(1, len(devices) // 2)
  global_tokens_per_step = cfg.train.global_batch_size * cfg.train.seq_len
  tokens_per_s_per_chip = global_tokens_per_step / (primary_step_s * num_chips)
  tflops_per_tc = total_tflops_per_tc / primary_step_s
  bf16_mfu_pct = (tflops_per_tc / _TPU_V7X_PEAK_BF16_TFLOPS_PER_TC) * 100.0
  final_rec = step_records[-1]

  summary: dict[str, Any] = {
      "num_processes": proc_cnt,
      "num_devices": len(devices),
      "num_chips": num_chips,
      "device_kind": devices[0].device_kind,
      "seq_len": cfg.train.seq_len,
      "global_batch_size": cfg.train.global_batch_size,
      "grad_accum_steps": cfg.train.grad_accum_steps,
      "fsdp_size": cfg.sharding.fsdp_size,
      "cp_size": cfg.sharding.cp_size,
      "ep_size": cfg.sharding.ep_size,
      "param_dtype": cfg.train.param_dtype,
      "compute_dtype": cfg.train.compute_dtype,
      "dataset_type": _DATASET_TYPE.value,
      "learning_rate": cfg.train.learning_rate,
      "optimized_libtpu_args": bool(_OPTIMIZED_LIBTPU_ARGS.value),
      "libtpu_init_args": libtpu_args,
      "config_overrides": overrides,
      "total_params": total_params,
      "active_params": active_params,
      "lower_sec": round(t_lower, 3),
      "compile_sec": round(t_compile, 3),
      "init_sec": round(t_init, 3),
      "total_aot_hbm_gib": round(total_hbm_gib, 3),
      "temp_gib": round(temp_gib, 3),
      "host_temp_gib": round(host_temp_gib, 3),
      "median_step_time_s": round(median_step_time_s, 4),
      "median_step_time_ms": round(primary_step_ms, 2),
      "mean_step_time_s": round(mean_step_time_s, 4),
      "best_step_time_s": round(best_step_time_s, 4),
      "pipelined_step_time_s": (round(pipelined_step_time_s, 4) if pipelined_step_time_s is not None else None),
      "global_tokens_per_step": global_tokens_per_step,
      "tokens_per_s_per_chip": round(tokens_per_s_per_chip, 2),
      "model_tflops_per_tc": round(total_tflops_per_tc, 3),
      "weight_tflops_per_tc": round(weight_tflops_per_tc, 3),
      "attn_tflops_per_tc": round(attn_tflops_per_tc, 3),
      "tflops_per_tc": round(tflops_per_tc, 2),
      "bf16_mfu_pct": round(bf16_mfu_pct, 2),
      "initial_loss": round(float(step_records[0]["loss"]), 5),
      "final_loss": round(float(final_rec["loss"]), 5),
      "final_ce_loss": round(float(final_rec["ce_loss"]), 5),
      "final_grad_norm": round(float(final_rec["grad_norm"]), 5),
      "steady_step_times_s": [round(x, 4) for x in steady_step_times_s],
      "all_step_times_s": [round(x, 4) for x in per_step_times_s],
      "step_losses": [round(float(r["loss"]), 5) for r in step_records],
      "step_grad_norms": [round(float(r["grad_norm"]), 5) for r in step_records],
      "profiler_trace_dir": profiler_trace_dir,
  }

  if is_primary:
    print("=== FLINT-40B 64-CHIP TRAINING SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if _OUTPUT_DIR.value:
      out_dir = epath.Path(_OUTPUT_DIR.value)
      out_dir.mkdir(parents=True, exist_ok=True)
      summary_path = out_dir / "benchmark_summary.json"
      summary_path.write_text(json.dumps(summary, indent=2))
      metrics_path = out_dir / "metrics.jsonl"
      metrics_path.write_text("".join(json.dumps(rec) + "\n" for rec in step_records))
      print(
          f"Saved summary to {summary_path} and metrics to {metrics_path}",
          flush=True,
      )

  sys.stdout.flush()
  sys.stderr.flush()
  multihost_utils.sync_global_devices("flint_benchmark_done")


if __name__ == "__main__":
  app.run(main)
