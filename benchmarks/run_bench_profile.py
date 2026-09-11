"""Benchmark and XProf profiling runner for MaxText decode inference."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils


def main():
  parser = argparse.ArgumentParser(description="MaxText Benchmark and XProf Profiler")
  parser.add_argument(
      "--mode",
      choices=["baseline", "diff"],
      required=True,
      help="Mode: baseline (on-the-fly dequant) or diff (serve_fp8_weight native compute)",
  )
  parser.add_argument("--output_dir", default="/dev/shm/xprof", help="Directory to save XProf traces")
  parser.add_argument(
      "--checkpoint_path",
      default="/dev/shm/maxtext_qwen3.5_35b_fp8_pertensor_v3",
      help="Path to per-tensor quantized checkpoint",
  )
  parser.add_argument(
      "--warmup_steps",
      type=int,
      default=15,
      help="Number of warmup steps (excluding JIT compilation step 0)",
  )
  parser.add_argument(
      "--profile_steps",
      type=int,
      default=2,
      help="Number of profiled steps captured in XProf trace",
  )
  parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for decode step")
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size for decode step")
  args = parser.parse_args()

  print(f"=== Starting MaxText Profiling Run: mode={args.mode} ===", flush=True)

  quantization_mode = "serve_fp8_weight" if args.mode == "diff" else ""

  # Initialize pyconfig FIRST so jax.distributed.initialize() happens before XLA backend init
  cfg = pyconfig.initialize(
      ["", "src/maxtext/configs/base.yml"],
      model_name="qwen3.5-35b-a3b-fp8",
      override_model_config=True,
      load_parameters_path=args.checkpoint_path,
      quantization=quantization_mode,
      weight_block_size=None,
      sparse_matmul=True,
      use_tokamax_gmm=True,
      use_gmm_v2=True,
      scan_layers=True,
      attention="dot_product",
      per_device_batch_size=args.batch_size,
      max_target_length=max(128, args.seq_len * 2),
      async_checkpointing=False,
      run_name=f"profile_{args.mode}",
  )

  print(f"JAX Devices: {jax.devices()}", flush=True)
  print(f"TPU Device Count: {jax.device_count()}, Process Count: {jax.process_count()}", flush=True)

  print(f"Loading {args.mode} model from {args.checkpoint_path}...", flush=True)
  t_load_start = time.perf_counter()
  model, _ = model_creation_utils.from_pretrained(cfg, model_mode=MODEL_MODE_TRAIN)
  t_load = time.perf_counter() - t_load_start
  print(f"Model loaded successfully in {t_load:.2f}s.", flush=True)

  @nnx.jit
  def forward_step(m, tok, pos, seg):
    return m(
        decoder_input_tokens=tok,
        decoder_positions=pos,
        decoder_segment_ids=seg,
        enable_dropout=False,
    )

  # Prepare test inputs
  tok = jnp.full((args.batch_size, args.seq_len), 42, dtype=jnp.int32)
  pos = jnp.arange(args.seq_len, dtype=jnp.int32)[None, :].repeat(args.batch_size, axis=0)
  seg = jnp.full((args.batch_size, args.seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR, dtype=jnp.int32)

  # Step 0: JIT compilation
  print("Running Step 0 (JIT Compilation)...", flush=True)
  t0 = time.perf_counter()
  logits = forward_step(model, tok, pos, seg)
  logits.block_until_ready()
  t_compile = time.perf_counter() - t0
  is_finite = bool(jnp.all(jnp.isfinite(logits)))
  print(f"Step 0 finished in {t_compile:.3f}s. Logits finite: {is_finite}, shape: {logits.shape}", flush=True)

  if not is_finite:
    raise RuntimeError("Non-finite logits detected in forward pass!")

  # Warmup steps (1 to args.warmup_steps)
  warmup_times = []
  print(f"Running {args.warmup_steps} warmup steps...", flush=True)
  for step_idx in range(1, args.warmup_steps + 1):
    t0 = time.perf_counter()
    logits = forward_step(model, tok, pos, seg)
    logits.block_until_ready()
    dt = time.perf_counter() - t0
    warmup_times.append(dt)
    print(f"  Warmup step {step_idx}/{args.warmup_steps}: {dt * 1000:.2f} ms", flush=True)

  # Profiled steps with XProf
  trace_dir = os.path.join(args.output_dir, args.mode)
  os.makedirs(trace_dir, exist_ok=True)
  print(f"Starting XProf trace in: {trace_dir}...", flush=True)
  jax.profiler.start_trace(trace_dir)

  profile_times = []
  for step_idx in range(1, args.profile_steps + 1):
    t0 = time.perf_counter()
    logits = forward_step(model, tok, pos, seg)
    logits.block_until_ready()
    dt = time.perf_counter() - t0
    profile_times.append(dt)
    print(f"  Profiled step {step_idx}/{args.profile_steps}: {dt * 1000:.2f} ms", flush=True)

  print("Stopping XProf trace...", flush=True)
  jax.profiler.stop_trace()
  print(f"XProf trace stopped and exported successfully to {trace_dir}.", flush=True)

  # Collect metrics
  warmup_arr = np.array(warmup_times) * 1000.0  # ms
  profile_arr = np.array(profile_times) * 1000.0  # ms

  metrics = {
      "mode": args.mode,
      "quantization": quantization_mode,
      "load_time_sec": t_load,
      "compile_step0_sec": t_compile,
      "warmup_steps": args.warmup_steps,
      "warmup_mean_ms": float(np.mean(warmup_arr)),
      "warmup_std_ms": float(np.std(warmup_arr)),
      "warmup_min_ms": float(np.min(warmup_arr)),
      "warmup_max_ms": float(np.max(warmup_arr)),
      "profile_steps": args.profile_steps,
      "profile_mean_ms": float(np.mean(profile_arr)),
      "profile_std_ms": float(np.std(profile_arr)),
      "profile_times_ms": profile_arr.tolist(),
      "trace_dir": trace_dir,
      "logits_finite": is_finite,
      "logits_shape": list(logits.shape),
  }

  summary_path = os.path.join(trace_dir, "metrics.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"\n=== Summary for {args.mode} ===", flush=True)
  print(json.dumps(metrics, indent=2), flush=True)
  print(f"Metrics written to {summary_path}", flush=True)


if __name__ == "__main__":
  main()
