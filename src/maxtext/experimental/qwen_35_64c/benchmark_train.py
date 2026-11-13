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

"""Multi-host 64-chip TPU v7x (tpu7x:4x4x4) trainer for Qwen3.5-397B-A17B (synthetic or C4)."""

from __future__ import annotations

from collections.abc import Sequence
import concurrent.futures
import itertools
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("MAXTEXT_GMM_V2_IN_KERNEL_FP8_DEQUANT", "1")

# pylint: disable=g-import-not-at-top
from absl import app
from absl import flags
from absl import logging
from etils import epath
import numpy as np

from maxtext.experimental.qwen_35_64c import aot_compile

# pylint: enable=g-import-not-at-top

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Local or gs:// output directory for JSON summary, profiler traces, and" " optional XLA dumps.",
)
_STEPS = flags.DEFINE_integer(
    "steps",
    18,
    "Total training steps to execute (warmup + profiled + steady-state).",
)
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    4,
    "Number of unprofiled warmup steps before profiler capture.",
)
_PROFILE_STEPS = flags.DEFINE_integer(
    "profile_steps",
    3,
    "Number of steps to capture with jax.profiler on process 0 (0 disables).",
)
_PROFILE_ALL_PROCESSES = flags.DEFINE_boolean(
    "profile_all_processes",
    False,
    "Capture a jax.profiler trace on every process"
    " (<output_dir>/profiler/proc_NNN) instead of only process 0, for comparing"
    " per-host MoE load and collective waits.",
)
_SEED = flags.DEFINE_integer("seed", 42, "Deterministic PRNG seed across all hosts.")
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    1e-9,
    "AdamW learning rate. Default 1e-9 preserves 100% identical AdamW HLO (no"
    " constant folding) while preventing synthetic-token router load collapse.",
)
_CONFIG_OVERRIDES_JSON = flags.DEFINE_string(
    "config_overrides_json",
    "{}",
    "Optional JSON dict overriding Qwen35Config fields.",
)
_EXTRA_LIBTPU_ARGS = flags.DEFINE_string(
    "extra_libtpu_args",
    "",
    "Additional space-separated --xla_tpu_... compiler flags.",
)
_DUMP_XLA = flags.DEFINE_boolean(
    "dump_xla",
    False,
    "Whether to dump HLO on process 0 into <output_dir>/xla_task0.",
)
_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    "synthetic",
    ["synthetic", "c4"],
    "synthetic: one fixed random batch reused every step. c4: pre-tokenized" " Qwen3.5 C4 ArrayRecord rows.",
)
_C4_DATA_DIR = flags.DEFINE_string(
    "c4_data_dir",
    "",
    "Pre-tokenized C4 ArrayRecord directory (local or gs://), required when" " --dataset_type=c4.",
)
_LOG_GRAD_NORM = flags.DEFINE_boolean(
    "log_grad_norm",
    False,
    "Record per-step FP32 global gradient L2 norm for convergence parity runs.",
)
_SKIP_JAX_DISTRIBUTED = flags.DEFINE_boolean(
    "skip_jax_distributed",
    False,
    "Skip jax.distributed.initialize() (e.g. for single-process testing).",
)
_JAX_DIST_TIMEOUT = flags.DEFINE_integer(
    "jax_distributed_timeout",
    300,
    "Timeout in seconds for jax.distributed.initialize().",
)


def _guess_is_worker_zero() -> bool:
  for env_key in ("TPU_WORKER_ID", "JOB_COMPLETION_INDEX", "PROCESS_INDEX"):
    val = os.environ.get(env_key)
    if val is not None:
      return val == "0"
  if "jax_task_id" in flags.FLAGS:
    return int(getattr(flags.FLAGS, "jax_task_id", 0) or 0) == 0
  return True


def _configure_compiler_flags(extra_args: str, xla_dump_dir: str | None) -> tuple[dict[str, str], int]:
  """Configures LIBTPU_INIT_ARGS, XLA_FLAGS, and PJRT compiler_options before TPU init."""
  tokens = list(aot_compile.BASE_LIBTPU_ARGS.split())
  if extra_args.strip():
    tokens.extend(extra_args.strip().split())
  libtpu_str = " ".join(tokens)

  existing_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
  os.environ["LIBTPU_INIT_ARGS"] = f"{libtpu_str} {existing_libtpu}".strip() if existing_libtpu else libtpu_str

  raw_opts = aot_compile.parse_xla_compiler_options(libtpu_str)
  dump_opts: dict[str, str] = {}
  if xla_dump_dir:
    dump_flags = " ".join(
        [
            f"--xla_dump_to={xla_dump_dir}",
            "--xla_dump_hlo_module_re=jit_train_step",
            "--xla_dump_large_constants=true",
            "--xla_dump_hlo_as_proto=true",
        ]
    )
    existing_xla = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = f"{existing_xla} {dump_flags}".strip() if existing_xla else dump_flags
    dump_opts = aot_compile.parse_xla_compiler_options(dump_flags)

  synced = 0
  for k, v in {**raw_opts, **dump_opts}.items():
    if k in flags.FLAGS:
      flags.FLAGS[k].parse(v)
      synced += 1

  # Pass dump_opts exclusively via XLA_FLAGS / C++ absl.flags; PJRT
  # CompileOptions rejects string "true" for DebugOptions boolean fields.
  return dict(raw_opts), synced


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
  if _DATASET_TYPE.value == "c4" and not _C4_DATA_DIR.value:
    raise ValueError("--c4_data_dir is required when --dataset_type=c4.")

  overrides = json.loads(_CONFIG_OVERRIDES_JSON.value)
  for k, v in list(overrides.items()):
    if isinstance(v, list) and ("tile" in k or "sublayers" in k):
      overrides[k] = tuple(v)
  overrides.setdefault("learning_rate", _LEARNING_RATE.value)

  xla_dump_dir = None
  if _DUMP_XLA.value and _OUTPUT_DIR.value and _guess_is_worker_zero():
    xla_dump_dir = os.path.join(_OUTPUT_DIR.value.rstrip("/"), "xla_task0")
  pjrt_compiler_opts, absl_synced = _configure_compiler_flags(_EXTRA_LIBTPU_ARGS.value, xla_dump_dir)

  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top

  if not _SKIP_JAX_DISTRIBUTED.value and not jax.distributed.is_initialized():
    jax.distributed.initialize(initialization_timeout=_JAX_DIST_TIMEOUT.value)

  # The persistent compilation cache keys on the pre-transform HLO, so a cache
  # hit would skip MaxOverlapTransform. Disable before the first compile.
  jax.config.update("jax_enable_compilation_cache", False)

  from maxtext.experimental import qwen_35_64c as qwen35_jax  # pylint: disable=g-import-not-at-top
  from maxtext.experimental.qwen_35_64c import max_async_overlap_transform  # pylint: disable=g-import-not-at-top

  devices = jax.devices()
  proc_idx = jax.process_index()
  proc_cnt = jax.process_count()
  is_primary = proc_idx == 0

  if is_primary:
    sample_coords = getattr(devices[0], "coords", None)
    print(
        f"[Qwen3.5 64c] Initialized JAX cluster: processes={proc_cnt},"
        f" global_devices={len(devices)},"
        f" local_devices={jax.local_device_count()},"
        f" device_kind={devices[0].device_kind!r},"
        f" sample_coords={sample_coords}, absl_synced={absl_synced},"
        f" pjrt_opts={len(pjrt_compiler_opts)}",
        flush=True,
    )

  cfg = qwen35_jax.full_config(**overrides)

  mesh = qwen35_jax.create_mesh(cfg, devices=list(devices), aot_topology=None)
  max_async_overlap_transform.reset_transform_stats()
  if cfg.use_max_async_overlap_transform:
    reg_platforms = max_async_overlap_transform.register_transform(
        bundle_quant_scales=cfg.max_overlap_bundle_quant,
        bundle_scalar_all_reduce=cfg.max_overlap_bundle_scalar_ar,
        queue_filter=cfg.max_overlap_queues,
        order_mode=cfg.max_overlap_order_mode,
    )
  else:
    reg_platforms = []
    max_async_overlap_transform.unregister_transform()

  t_comp0 = time.perf_counter()
  abstract_params = qwen35_jax.get_abstract_params(cfg, mesh)
  abstract_opt_state = qwen35_jax.get_abstract_opt_state(cfg, mesh, abstract_params)
  global_b = cfg.global_batch_size
  data_sharding = NamedSharding(mesh, qwen35_jax.data_pspec(cfg))
  tok_spec = jax.ShapeDtypeStruct((global_b, cfg.seq_len), jnp.int32, sharding=data_sharding)

  log_gn = _LOG_GRAD_NORM.value
  train_step_fn = qwen35_jax.make_train_step(cfg, mesh, donate_state=True, return_grad_norm=log_gn)
  lowered = train_step_fn.lower(
      abstract_params,
      abstract_opt_state,
      tok_spec,
      tok_spec,
      tok_spec,
      tok_spec,
  )
  t_lower = time.perf_counter() - t_comp0
  compiled = lowered.compile(compiler_options=pjrt_compiler_opts)
  t_compile = time.perf_counter() - t_comp0

  max_overlap_stats = max_async_overlap_transform.get_last_transform_stats()
  if (
      is_primary
      and cfg.use_max_async_overlap_transform
      and (t_compile - t_lower) > 5.0
      and max_overlap_stats.get("modules_transformed", 0) == 0
  ):
    raise RuntimeError(f"MaxOverlapTransform ran on 0 modules! reg_platforms={reg_platforms}")

  mem = compiled.memory_analysis()
  gib = 1024**3
  total_hbm_gib = (
      mem.generated_code_size_in_bytes
      + mem.argument_size_in_bytes
      + max(0, mem.output_size_in_bytes - mem.alias_size_in_bytes)
      + mem.temp_size_in_bytes
  ) / gib
  temp_gib = mem.temp_size_in_bytes / gib
  host_temp_gib = mem.host_temp_size_in_bytes / gib

  if is_primary:
    print(
        f"[Qwen3.5 64c] Compiled train_step in {t_compile:.2f}s"
        f" (lower={t_lower:.2f}s): total_hbm={total_hbm_gib:.3f} GiB,"
        f" temp={temp_gib:.3f} GiB, host_temp={host_temp_gib:.3f} GiB/TC,"
        f" max_overlap={max_overlap_stats}",
        flush=True,
    )

  t_init0 = time.perf_counter()
  params = qwen35_jax.init_random_params(cfg, mesh, jax.random.PRNGKey(_SEED.value))
  opt_state = qwen35_jax.init_opt_state(cfg, mesh, params)
  if _DATASET_TYPE.value == "synthetic":
    synthetic_batch = qwen35_jax.init_synthetic_batch(cfg, mesh, seed=_SEED.value + 1)
    jax.block_until_ready(synthetic_batch)

    def next_batch():
      return synthetic_batch

  else:
    from maxtext.experimental.qwen_35_64c import data as data_lib  # pylint: disable=g-import-not-at-top

    c4 = data_lib.ArrayRecordC4(_C4_DATA_DIR.value, cfg.seq_len)
    if is_primary:
      print(
          f"[Qwen3.5 64c] C4 ArrayRecord: {c4.num_rows} rows of" f" {cfg.seq_len} tokens",
          flush=True,
      )
    data_steps = itertools.count()
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_batch = prefetch_pool.submit(c4.make_batch, next(data_steps), global_b, data_sharding)

    def next_batch():
      nonlocal pending_batch
      batch = pending_batch.result()
      pending_batch = prefetch_pool.submit(c4.make_batch, next(data_steps), global_b, data_sharding)
      return batch

  jax.block_until_ready((params, opt_state))
  multihost_utils.sync_global_devices("qwen35_init_done")
  t_init = time.perf_counter() - t_init0
  if is_primary:
    print(
        "[Qwen3.5 64c] Sharded 397B params + AdamW + data" f" ({_DATASET_TYPE.value}) initialized in {t_init:.2f}s",
        flush=True,
    )

  total_steps = _STEPS.value
  warmup_end = min(_WARMUP_STEPS.value, total_steps)
  prof_end = min(warmup_end + max(0, _PROFILE_STEPS.value), total_steps)

  per_step_times_s: list[float] = []
  step_losses: list[float] = []
  step_grad_norms: list[float] = []
  profiler_trace_dir: str | None = None

  def _run_step(p_in, o_in, b_in):
    out = compiled(p_in, o_in, *b_in)
    if log_gn:
      return out[0], out[1], out[2], out[3]
    return out[0], out[1], out[2], None

  # Phase 1: Warmup steps
  for step in range(warmup_end):
    t0 = time.perf_counter()
    params, opt_state, loss, gn = _run_step(params, opt_state, next_batch())
    jax.block_until_ready((loss, gn) if gn is not None else loss)
    dt = time.perf_counter() - t0
    loss_val = float(np.asarray(loss))
    per_step_times_s.append(dt)
    step_losses.append(loss_val)
    gn_sfx = ""
    if gn is not None:
      gn_val = float(np.asarray(gn))
      step_grad_norms.append(gn_val)
      gn_sfx = f" grad_norm={gn_val:.4f}"
    if is_primary:
      print(
          f"[Warmup step {step:02d}] step_time={dt:.4f}s ({dt*1000:.1f} ms)" f" loss={loss_val:.4f}{gn_sfx}",
          flush=True,
      )

  # Phase 2: JAX profiler trace capture
  if prof_end > warmup_end:
    multihost_utils.sync_global_devices("qwen35_pre_profile")
    should_profile = is_primary or _PROFILE_ALL_PROCESSES.value
    proc_trace_dir: str | None = None
    if should_profile:
      if _OUTPUT_DIR.value:
        base_prof_dir = os.path.join(_OUTPUT_DIR.value.rstrip("/"), "profiler")
        proc_trace_dir = (
            os.path.join(base_prof_dir, f"proc_{proc_idx:03d}") if _PROFILE_ALL_PROCESSES.value else base_prof_dir
        )
      else:
        proc_trace_dir = tempfile.mkdtemp(prefix=f"qwen35_profiler_proc{proc_idx:03d}_")
      jax.profiler.start_trace(proc_trace_dir)
      logging.info(
          "Started JAX profiler trace on process %d into %s",
          proc_idx,
          proc_trace_dir,
      )
      if is_primary:
        profiler_trace_dir = proc_trace_dir
    for step in range(warmup_end, prof_end):
      t0 = time.perf_counter()
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
        params, opt_state, loss, gn = _run_step(params, opt_state, next_batch())
        jax.block_until_ready((loss, gn) if gn is not None else loss)
      dt = time.perf_counter() - t0
      loss_val = float(np.asarray(loss))
      per_step_times_s.append(dt)
      step_losses.append(loss_val)
      gn_sfx = ""
      if gn is not None:
        gn_val = float(np.asarray(gn))
        step_grad_norms.append(gn_val)
        gn_sfx = f" grad_norm={gn_val:.4f}"
      if is_primary:
        print(
            f"[Profile step {step:02d}] step_time={dt:.4f}s ({dt*1000:.1f} ms)" f" loss={loss_val:.4f}{gn_sfx}",
            flush=True,
        )
    if should_profile:
      jax.profiler.stop_trace()
      logging.info(
          "Completed JAX profiler trace on process %d: %s",
          proc_idx,
          proc_trace_dir,
      )
    multihost_utils.sync_global_devices("qwen35_post_profile")

  # Phase 3: Clean unprofiled steady-state measurement
  steady_step_times_s: list[float] = []
  rem_steps = total_steps - prof_end
  pipelined_window_step_s: float | None = None

  if rem_steps > 0:
    indiv_steps = max(1, rem_steps // 2) if rem_steps >= 4 else rem_steps
    pipe_steps = rem_steps - indiv_steps

    for step in range(prof_end, prof_end + indiv_steps):
      t0 = time.perf_counter()
      params, opt_state, loss, gn = _run_step(params, opt_state, next_batch())
      jax.block_until_ready((loss, gn) if gn is not None else loss)
      dt = time.perf_counter() - t0
      loss_val = float(np.asarray(loss))
      per_step_times_s.append(dt)
      steady_step_times_s.append(dt)
      step_losses.append(loss_val)
      gn_sfx = ""
      if gn is not None:
        gn_val = float(np.asarray(gn))
        step_grad_norms.append(gn_val)
        gn_sfx = f" grad_norm={gn_val:.4f}"
      if is_primary:
        print(
            f"[Steady step {step:02d}] step_time={dt:.4f}s ({dt*1000:.1f} ms)" f" loss={loss_val:.4f}{gn_sfx}",
            flush=True,
        )

    if pipe_steps > 0:
      multihost_utils.sync_global_devices("qwen35_pre_pipelined_window")
      t_pipe0 = time.perf_counter()
      pipe_losses = []
      pipe_gns = []
      for step in range(prof_end + indiv_steps, total_steps):
        params, opt_state, loss_pipe, gn_pipe = _run_step(params, opt_state, next_batch())
        pipe_losses.append(loss_pipe)
        if gn_pipe is not None:
          pipe_gns.append(gn_pipe)
      jax.block_until_ready((params, opt_state, pipe_losses, pipe_gns))
      t_pipe_total = time.perf_counter() - t_pipe0
      pipelined_window_step_s = t_pipe_total / pipe_steps
      pipe_loss_vals = [float(np.asarray(l)) for l in pipe_losses]
      pipe_gn_vals = [float(np.asarray(g)) for g in pipe_gns]
      step_losses.extend(pipe_loss_vals)
      step_grad_norms.extend(pipe_gn_vals)
      loss_val = pipe_loss_vals[-1]
      if is_primary:
        for i, lv in enumerate(pipe_loss_vals):
          gn_s = f" grad_norm={pipe_gn_vals[i]:.4f}" if pipe_gn_vals else ""
          print(
              f"[Pipelined step {prof_end + indiv_steps + i:02d}]" f" loss={lv:.4f}{gn_s}",
              flush=True,
          )
        print(
            f"[Pipelined {pipe_steps} steps] total={t_pipe_total:.4f}s ->"
            f" avg_step_time={pipelined_window_step_s:.4f}s"
            f" ({pipelined_window_step_s*1000:.2f} ms),"
            f" final_loss={loss_val:.4f}",
            flush=True,
        )

  num_chips = len(devices) // 2
  global_tokens_per_step = global_b * cfg.seq_len
  median_indiv_s = (
      float(np.median(steady_step_times_s)) if steady_step_times_s else float(np.median(per_step_times_s[-3:]))
  )
  best_indiv_s = float(np.min(steady_step_times_s)) if steady_step_times_s else float(np.min(per_step_times_s))
  primary_step_s = pipelined_window_step_s if pipelined_window_step_s is not None else median_indiv_s
  primary_step_ms = primary_step_s * 1000.0
  tok_per_s_per_chip = global_tokens_per_step / (primary_step_s * max(1, num_chips))

  summary = {
      "num_processes": proc_cnt,
      "num_devices": len(devices),
      "num_chips": num_chips,
      "device_kind": devices[0].device_kind,
      "compile_sec": round(t_compile, 3),
      "lower_sec": round(t_lower, 3),
      "init_sec": round(t_init, 3),
      "total_aot_hbm_gib": round(total_hbm_gib, 3),
      "temp_gib": round(temp_gib, 3),
      "host_temp_gib": round(host_temp_gib, 3),
      "max_overlap_stats": max_overlap_stats,
      "primary_step_time_s": round(primary_step_s, 4),
      "primary_step_time_ms": round(primary_step_ms, 2),
      "median_indiv_step_time_s": round(median_indiv_s, 4),
      "best_indiv_step_time_s": round(best_indiv_s, 4),
      "pipelined_window_step_time_s": (
          round(pipelined_window_step_s, 4) if pipelined_window_step_s is not None else None
      ),
      "tokens_per_sec_per_chip": round(tok_per_s_per_chip, 2),
      "global_tokens_per_step": global_tokens_per_step,
      "steady_step_times_s": [round(x, 4) for x in steady_step_times_s],
      "all_step_times_s": [round(x, 4) for x in per_step_times_s],
      "step_losses": [round(x, 5) for x in step_losses],
      "step_grad_norms": [round(x, 5) for x in step_grad_norms],
      "profiler_trace_dir": profiler_trace_dir,
      "learning_rate": cfg.learning_rate,
      "router_aux_loss_coef": cfg.router_aux_loss_coef,
      "grad_accum_steps": cfg.grad_accum_steps,
      "dataset_type": _DATASET_TYPE.value,
      "fsdp_size": cfg.fsdp_size,
      "ep_size": cfg.ep_size,
      "decouple_ep_d2d": cfg.decouple_ep_d2d,
  }

  if is_primary:
    print("=== QWEN3.5-397B 64-CHIP TRAINING SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if _OUTPUT_DIR.value:
      out_path = epath.Path(_OUTPUT_DIR.value)
      out_path.mkdir(parents=True, exist_ok=True)
      summary_path = out_path / "benchmark_summary.json"
      summary_path.write_text(json.dumps(summary, indent=2) + "\n")
      print(f"Saved summary to {summary_path}", flush=True)

  sys.stdout.flush()
  sys.stderr.flush()
  multihost_utils.sync_global_devices("qwen35_benchmark_done")


if __name__ == "__main__":
  app.run(main)
