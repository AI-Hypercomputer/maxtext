# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarking and verification script for Canonical GDN kernel on Cloud TPU.

Authoritative 2-way comparison:
1. Pure JAX GDN (Reference)
2. Canonical Decoupled GDN Kernel (use_gdn_kernel=True)
"""

import argparse
import builtins
import functools
import glob
import os
import shutil
import sys
import time
import types
from typing import Any, Tuple

print = functools.partial(builtins.print, flush=True)

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# Force highest precision for TPU MXU 2/3-pass FP32 simulation
jax.config.update("jax_default_matmul_precision", "highest")

try:
  from maxtext.models import hybrid_bwd_analytical_pipeline
  from maxtext.models import qwen3
except ImportError:
  from maxtext.src.maxtext.models import hybrid_bwd_analytical_pipeline
  from maxtext.src.maxtext.models import qwen3


def create_model_configs(
    hidden_size: int = 4096,
    num_key_heads: int = 16,
    num_value_heads: int = 64,
    head_dim: int = 128,
    conv_kernel_dim: int = 4,
    chunk_size: int = 64,
    dtype: Any = jnp.float32,
    use_qk_norm: bool = True,
) -> Tuple[types.SimpleNamespace, types.SimpleNamespace]:
  """Creates configurations for Pure JAX (Reference) and Canonical GDN Kernel."""
  if dtype is None:
    dtype = jnp.float32
  base_dict = dict(
      emb_dim=hidden_size,
      gdn_num_value_heads=num_value_heads,
      gdn_num_key_heads=num_key_heads,
      gdn_key_head_dim=head_dim,
      gdn_value_head_dim=head_dim,
      gdn_conv_kernel_dim=conv_kernel_dim,
      dtype=dtype,
      weight_dtype=dtype,
      matmul_precision="highest",
      normalization_layer_epsilon=1e-6,
      gdn_chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm,
      load_balance_loss_weight=0.0,
      scan_layers=False,
      using_pipeline_parallelism=False,
      logical_axis_rules=(),
  )

  # 1. Pure JAX GDN config (Reference)
  pure_jax_config = types.SimpleNamespace(
      **base_dict,
      use_gdn_kernel=False,
  )

  # 2. Canonical Decoupled GDN Kernel config (Decoupled v1.5)
  gdn_kernel_config = types.SimpleNamespace(
      **base_dict,
      use_gdn_kernel=True,
  )

  return pure_jax_config, gdn_kernel_config


def create_jitted_train_step(
    model: nnx.Module,
    input_shape: Tuple[int, ...],
    step_scope: str = "TrainStep",
    fwd_scope: str = "Fwd",
    bwd_scope: str = "Bwd",
    remat: bool | str = False,
):
  """Creates a pure functional, JIT-compiled training step with position-aware loss."""
  is_remat = (
      remat.lower() in ("full", "true", "yes", "1")
      if isinstance(remat, str)
      else bool(remat)
  )
  graphdef, params = nnx.split(model)

  proj_key = jax.random.PRNGKey(99)
  projection = jax.random.normal(proj_key, input_shape)

  @jax.jit
  def pure_train_step(params, x):
    with jax.named_scope(step_scope):
      m = nnx.merge(graphdef, params)

      def loss_fn(m_inner):
        with jax.named_scope(fwd_scope):
          if is_remat:
            out = jax.checkpoint(lambda m, inp: m(inp))(m_inner, x)
          else:
            out = m_inner(x)
          y = out[0] if isinstance(out, tuple) else out
          loss = jnp.mean(y * projection.astype(y.dtype))
          return loss, out

      with jax.named_scope(bwd_scope):
        (loss, y), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
      return loss, y, grads

  return pure_train_step, params


def create_jitted_forward(model: nnx.Module, scope_name: str = "Fwd"):
  """Creates a pure functional, JIT-compiled forward pass."""
  graphdef, params = nnx.split(model)

  @jax.jit
  def pure_forward(params, x):
    with jax.named_scope(scope_name):
      m = nnx.merge(graphdef, params)
      out = m(x)
      return out

  return pure_forward, params


def print_numerical_correctness_table(
    out_ref: Any,
    out_test: Any,
    loss_ref: Any,
    loss_test: Any,
    grads_ref: Any,
    grads_test: Any,
    tolerance: float = 1e-4,
    abs_tolerance: float = 1e-5,
    comparison_name: str = "Candidate vs Reference",
    diff_records: dict[str, Any] | None = None,
) -> bool:
  """Prints a numerical correctness comparison table between two implementations."""
  print(
      "\n========================================================================================="
  )
  print(f">>> NUMERICAL CORRECTNESS: {comparison_name}")
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Tensor / Parameter':<40} | {'Max Abs Diff':<12} | {'Relative Diff':<13} |"
      f" {'Tolerance':<10} | {'Status'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  rows = []

  # 1. Forward Output
  ref_t = np.asarray(out_ref[0] if isinstance(out_ref, tuple) else out_ref)
  test_t = np.asarray(out_test[0] if isinstance(out_test, tuple) else out_test)
  abs_d_fwd = float(np.max(np.abs(ref_t - test_t)))
  rel_d_fwd = abs_d_fwd / (float(np.max(np.abs(ref_t))) + 1e-7)
  match_fwd = (rel_d_fwd <= tolerance) or (abs_d_fwd <= abs_tolerance)
  rows.append(("Forward Output", abs_d_fwd, rel_d_fwd, match_fwd))

  # 2. Loss Scalar
  lp = float(loss_ref)
  la = float(loss_test)
  abs_d_loss = abs(lp - la)
  rel_d_loss = abs_d_loss / (abs(lp) + 1e-7)
  match_loss = (rel_d_loss <= tolerance) or (abs_d_loss <= abs_tolerance)
  rows.append(("Loss Scalar", abs_d_loss, rel_d_loss, match_loss))

  # 3. Parameter Gradients
  ref_leaves = jax.tree_util.tree_leaves_with_path(grads_ref)
  test_leaves = jax.tree_util.tree_leaves_with_path(grads_test)

  for (path_ref, g_ref), (_, g_test) in zip(ref_leaves, test_leaves):
    if not hasattr(g_ref, "shape") or not hasattr(g_test, "shape"):
      continue
    path_parts = []
    for k in path_ref:
      if hasattr(k, "key"):
        path_parts.append(str(k.key))
      elif hasattr(k, "name"):
        path_parts.append(str(k.name))
      elif hasattr(k, "idx"):
        path_parts.append(str(k.idx))
      else:
        path_parts.append(str(k))
    name = ".".join(path_parts)
    g_ref_np = np.asarray(g_ref)
    g_test_np = np.asarray(g_test)
    abs_d = float(np.max(np.abs(g_ref_np - g_test_np)))
    rel_d = abs_d / (float(np.max(np.abs(g_ref_np))) + 1e-7)
    is_m = (rel_d <= tolerance) or (abs_d <= abs_tolerance)
    rows.append((name, abs_d, rel_d, is_m))

  overall_diverged = False
  for name, abs_d, rel_d, is_m in rows:
    if not is_m:
      overall_diverged = True
      status = "❌ DIVERGED"
    else:
      status = "✅ MATCH"
    if diff_records is not None:
      diff_records[name] = {
          "abs_diff": abs_d,
          "rel_diff": rel_d,
          "match": is_m,
          "tolerance": tolerance,
          "abs_tolerance": abs_tolerance,
      }
    print(
        f"  {name:<40} | {abs_d:<12.2e} | {rel_d:<13.2e} |"
        f" {tolerance:<10.2e} | {status}"
    )
  print(sep)
  return overall_diverged


def get_device_memory_stats() -> dict[str, Any] | None:
  """Returns memory stats dict from jax.devices()[0] if supported, else None."""
  try:
    dev = jax.devices()[0]
    if hasattr(dev, "memory_stats"):
      stats = dev.memory_stats()
      if stats and "bytes_in_use" in stats:
        return stats
  except Exception:
    pass
  return None


def get_compiled_memory_analysis(jit_fn: Any, params: Any, inputs: Any) -> Any | None:
  """Extracts static HBM memory analysis from XLA compiler."""
  if hasattr(jit_fn, "memory_analysis"):
    try:
      return jit_fn.memory_analysis()
    except Exception:
      pass
  if hasattr(jit_fn, "_cached_memory_analysis") and jit_fn._cached_memory_analysis is not None:
    return jit_fn._cached_memory_analysis
  try:
    lowered = jit_fn.lower(params, inputs)
    compiled = lowered.compile()
    if hasattr(compiled, "memory_analysis"):
      return compiled.memory_analysis()
  except Exception:
    pass
  return None


def run_memory_profile_analysis(
    kernel_names: list[str],
    fwd_fns: list[Any],
    train_fns: list[Any],
    params_list: list[Any],
    inputs: Any,
    seq_len: int,
    batch_size: int,
    policy_label: str = "",
):
  """Measures and displays comparative HBM memory usage across implementations."""
  title_suffix = (
      f" ({policy_label}S={seq_len}, B={batch_size}, Dtype=FP32)"
      if policy_label
      else f" (S={seq_len}, B={batch_size}, Dtype=FP32)"
  )
  print(
      "\n========================================================================================="
  )
  print(
      f">>> HBM MEMORY PROFILING & COMPARATIVE ANALYSIS{title_suffix}"
  )
  print(
      "========================================================================================="
  )

  fwd_act_mbs = []
  train_peak_mbs = []
  bwd_peak_mbs = []
  fwd_compiled_mbs = []
  train_compiled_mbs = []
  dev_peak_train_mbs = []
  breakdown_rows = []

  for name, fwd_fn, train_fn, p in zip(
      kernel_names, fwd_fns, train_fns, params_list
  ):
    mem_before_fwd = get_device_memory_stats()
    out_fwd = fwd_fn(p, inputs)
    jax.block_until_ready(out_fwd)
    mem_after_fwd = get_device_memory_stats()
    fwd_analysis = get_compiled_memory_analysis(fwd_fn, p, inputs)

    mem_before_train = get_device_memory_stats()
    out_train = train_fn(p, inputs)
    jax.block_until_ready(out_train)
    mem_after_train = get_device_memory_stats()
    train_analysis = get_compiled_memory_analysis(train_fn, p, inputs)

    dev_in_use_fwd = (mem_after_fwd["bytes_in_use"] / (1024**2)) if mem_after_fwd else 0.0
    dev_peak_fwd = (mem_after_fwd.get("peak_bytes_in_use", 0) / (1024**2)) if mem_after_fwd else 0.0
    dev_in_use_train = (mem_after_train["bytes_in_use"] / (1024**2)) if mem_after_train else 0.0
    dev_peak_train = (mem_after_train.get("peak_bytes_in_use", 0) / (1024**2)) if mem_after_train else 0.0

    if fwd_analysis is not None:
      fwd_act_mb = fwd_analysis.temp_size_in_bytes / (1024**2)
      fwd_peak_compiled_mb = (
          fwd_analysis.argument_size_in_bytes
          + fwd_analysis.temp_size_in_bytes
          + fwd_analysis.output_size_in_bytes
      ) / (1024**2)
    else:
      fwd_act_mb = dev_in_use_fwd
      fwd_peak_compiled_mb = dev_peak_fwd

    if train_analysis is not None:
      train_peak_compiled_mb = (
          train_analysis.argument_size_in_bytes
          + train_analysis.temp_size_in_bytes
          + train_analysis.output_size_in_bytes
      ) / (1024**2)
      train_peak_mb = train_peak_compiled_mb
      bwd_peak_mb = max(train_peak_compiled_mb - fwd_peak_compiled_mb, 0.0)
    else:
      train_peak_mb = dev_peak_train if dev_peak_train > 0 else dev_in_use_train
      bwd_peak_mb = max(train_peak_mb - fwd_act_mb, 0.0)
      train_peak_compiled_mb = train_peak_mb

    fwd_act_mbs.append(fwd_act_mb)
    train_peak_mbs.append(train_peak_mb)
    bwd_peak_mbs.append(bwd_peak_mb)
    fwd_compiled_mbs.append(fwd_peak_compiled_mb)
    train_compiled_mbs.append(train_peak_compiled_mb)
    dev_peak_train_mbs.append(dev_peak_train)

    if fwd_analysis is not None and train_analysis is not None:
      breakdown_rows.append((
          name,
          "Forward",
          fwd_analysis.argument_size_in_bytes / (1024**2),
          fwd_analysis.temp_size_in_bytes / (1024**2),
          fwd_analysis.output_size_in_bytes / (1024**2),
          fwd_peak_compiled_mb,
          dev_in_use_fwd,
          dev_peak_fwd,
      ))
      breakdown_rows.append((
          name,
          "Backward (Est.)",
          train_analysis.argument_size_in_bytes / (1024**2),
          max(train_analysis.temp_size_in_bytes - fwd_analysis.temp_size_in_bytes, 0) / (1024**2),
          train_analysis.output_size_in_bytes / (1024**2),
          bwd_peak_mb,
          dev_in_use_train,
          dev_peak_train,
      ))
      breakdown_rows.append((
          name,
          "Train Step",
          train_analysis.argument_size_in_bytes / (1024**2),
          train_analysis.temp_size_in_bytes / (1024**2),
          train_analysis.output_size_in_bytes / (1024**2),
          train_peak_compiled_mb,
          dev_in_use_train,
          dev_peak_train,
      ))
    else:
      breakdown_rows.append((name, "Forward", 0.0, fwd_act_mb, 0.0, fwd_peak_compiled_mb, dev_in_use_fwd, dev_peak_fwd))
      breakdown_rows.append((name, "Backward (Est.)", 0.0, bwd_peak_mb, 0.0, bwd_peak_mb, dev_in_use_train, dev_peak_train))
      breakdown_rows.append((name, "Train Step", 0.0, train_peak_mb, 0.0, train_peak_mb, dev_in_use_train, dev_peak_train))

  # 1. Comparative Summary Table
  ref_fwd = fwd_act_mbs[0] if fwd_act_mbs[0] > 0 else 1.0
  ref_train = train_peak_mbs[0] if train_peak_mbs[0] > 0 else 1.0

  summary_header = (
      f"  {'Kernel Implementation':<36} | {'Fwd Activation Mem':<20} |"
      f" {'Est. Backward Mem':<18} | {'Peak Train HBM':<18} |"
      f" {'Fwd Ratio vs Pure':<20} | {'Train Ratio vs Pure'}"
  )
  separator = "  " + "-" * len(summary_header)
  print("\nComparative HBM Memory Usage:")
  print(separator)
  print(summary_header)
  print(separator)

  for i in range(len(kernel_names)):
    f_mb = fwd_act_mbs[i]
    b_mb = bwd_peak_mbs[i]
    t_mb = train_peak_mbs[i]
    f_ratio = f_mb / ref_fwd if ref_fwd > 0 else 1.0
    t_ratio = t_mb / ref_train if ref_train > 0 else 1.0
    f_pct = (f_ratio - 1.0) * 100.0
    t_pct = (t_ratio - 1.0) * 100.0

    if i == 0:
      f_str = "1.00x (ref)"
      t_str = "1.00x (ref)"
    else:
      f_color = "🟢" if f_ratio <= 1.0 else "🔴"
      t_color = "🟢" if t_ratio <= 1.0 else "🔴"
      f_str = f"{f_ratio:.2f}x ({f_color} {f_pct:+.0f}%)"
      t_str = f"{t_ratio:.2f}x ({t_color} {t_pct:+.0f}%)"

    print(
        f"  [{i + 1}] {kernel_names[i]:<32} | {f_mb:>16.2f} MB |"
        f" {b_mb:>14.2f} MB | {t_mb:>14.2f} MB | {f_str:<20} | {t_str}"
    )
  print(separator)

  # 2. Detailed Buffer Breakdown
  if breakdown_rows:
    print("\nDetailed Memory Breakdown (XLA Compiled Buffers & Allocator):")
    b_header = (
        f"  {'Implementation':<32} | {'Pass':<16} | {'Argument':<12} |"
        f" {'Temp / Scratch':<14} | {'Output':<10} | {'Peak Total':<12} |"
        f" {'Dev In-Use':<12} | {'Dev Peak'}"
    )
    b_sep = "  " + "-" * len(b_header)
    print(b_sep)
    print(b_header)
    print(b_sep)
    for impl, scope, arg, tmp, out, pk, dev_u, dev_pk in breakdown_rows:
      print(
          f"  {impl:<32} | {scope:<16} | {arg:>9.2f} MB | {tmp:>11.2f} MB |"
          f" {out:>7.2f} MB | {pk:>9.2f} MB | {dev_u:>9.2f} MB | {dev_pk:>9.2f} MB"
      )
    print(b_sep)

  return fwd_act_mbs, train_peak_mbs, bwd_peak_mbs


def print_latency_comparison(
    kernel_names: list[str],
    fwd_lats: list[float],
    bwd_lats: list[float],
    train_lats: list[float],
    policy_label: str = "",
) -> None:
  """Prints a comprehensive 2-way latency and speedup comparison table."""
  print(
      "\n========================================================================================="
  )
  print(
      f">>> LATENCY & SPEEDUP: COMPARISON {policy_label}({kernel_names[0]} vs {kernel_names[1]})"
  )
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Pass / Step':<20} | {kernel_names[0]:<28} |"
      f" {kernel_names[1]:<32} | {'Speedup':<12} | {'Winner'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  passes = [
      ("Forward Pass", [fwd_lats[0], fwd_lats[1]]),
      ("Backward Pass", [bwd_lats[0], bwd_lats[1]]),
      ("Full Training Step", [train_lats[0], train_lats[1]]),
  ]

  for step_name, lats in passes:
    p_val = lats[0]
    k_val = lats[1]

    p_str = f"{p_val:>25.2f} ms" if not np.isnan(p_val) and p_val > 0 else "N/A"
    k_str = f"{k_val:>29.2f} ms" if not np.isnan(k_val) and k_val > 0 else "FAILED"

    if (not np.isnan(p_val) and p_val > 0) and (not np.isnan(k_val) and k_val > 0):
      speedup = p_val / k_val
      speedup_str = f"{speedup:>9.2f}x"
      if speedup > 1.05:
        winner = f"🏆 {kernel_names[1]}"
      elif speedup < 0.95:
        winner = f"🏆 {kernel_names[0]}"
      else:
        winner = "≈ Parity"
    else:
      speedup_str = "N/A"
      winner = "None"

    print(
        f"  {step_name:<20} | {p_str:>28} |"
        f" {k_str:>32} | {speedup_str:<12} | {winner}"
    )
  print(sep)


def print_tradeoff_table(
    ref_name: str,
    kernel_name: str,
    fwd_ref: float,
    fwd_k: float,
    bwd_ref: float,
    bwd_k: float,
    train_ref: float,
    train_k: float,
    fwd_mem_ref: float,
    fwd_mem_k: float,
    train_mem_ref: float,
    train_mem_k: float,
    policy_label: str = "",
) -> None:
  """Prints quantitative trade-off analysis of Canonical GDN Kernel vs Pure JAX Reference."""
  print(
      "\n========================================================================================="
  )
  print(
      f">>> QUANTITATIVE TRADE-OFF {policy_label}: {kernel_name} vs {ref_name}"
  )
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Metric':<30} | {ref_name:<28} | {kernel_name:<32} |"
      f" {'Difference / Savings':<22} | {'Advantage'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  metrics = [
      ("Forward Pass Latency", fwd_ref, fwd_k, "ms", True),
      ("Backward Pass Latency", bwd_ref, bwd_k, "ms", True),
      ("Full Training Step Latency", train_ref, train_k, "ms", True),
      ("Forward Activation Memory", fwd_mem_ref, fwd_mem_k, "MB", True),
      ("Peak Train Step Memory", train_mem_ref, train_mem_k, "MB", True),
  ]

  for label, val_ref, val_k, unit, lower_is_better in metrics:
    val_ref_is_valid = not np.isnan(val_ref) and val_ref > 0
    val_k_is_valid = not np.isnan(val_k) and val_k > 0

    val_ref_str = f"{val_ref:>25.2f} {unit}" if val_ref_is_valid else "N/A"
    val_k_str = f"{val_k:>29.2f} {unit}" if val_k_is_valid else "FAILED"

    if val_ref_is_valid and val_k_is_valid:
      diff = val_k - val_ref
      pct = (diff / (val_ref + 1e-7)) * 100.0
      diff_str = f"{diff:+.2f} {unit} ({pct:+.1f}%)"
      if abs(pct) < 1.0:
        advantage = "≈ Parity"
      elif (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better):
        advantage = f"🏆 {kernel_name} ({abs(pct):.1f}% better)"
      else:
        advantage = f"🏆 {ref_name} ({abs(pct):.1f}% better)"
    else:
      diff_str = "N/A"
      advantage = "N/A"

    print(
        f"  {label:<30} | {val_ref_str:>28} | {val_k_str:>32} |"
        f" {diff_str:>22} | {advantage}"
    )
  print(sep)


def print_cross_policy_summary(results: dict[str, dict[str, Any]]) -> None:
  """Prints an executive summary table comparing results across remat policies."""
  print(
      "\n========================================================================================================================="
  )
  print(
      ">>> EXECUTIVE SUMMARY: CROSS-POLICY COMPARISON (remat=none vs remat=full)"
  )
  print(
      "========================================================================================================================="
  )
  # 1. Numerical Correctness
  has_diffs = any("diff_records" in res and res["diff_records"] for res in results.values())
  if has_diffs:
    num_header = (
        f"  {'Tensor / Parameter':<40} | {'Rel Diff (none)':<16} |"
        f" {'Rel Diff (full)':<16} | {'Tolerance':<10} | {'Status'}"
    )
    num_sep = "  " + "-" * (len(num_header) - 2)
    print("\n1. Numerical Correctness (Relative Gradient Difference vs Pure JAX):")
    print(num_sep)
    print(num_header)
    print(num_sep)

    all_names = []
    for policy in ("none", "full"):
      if policy in results and "diff_records" in results[policy]:
        for k in results[policy]["diff_records"]:
          if k not in all_names:
            all_names.append(k)

    for name in all_names:
      none_rec = results.get("none", {}).get("diff_records", {}).get(name)
      full_rec = results.get("full", {}).get("diff_records", {}).get(name)
      none_str = f"{none_rec['rel_diff']:<16.2e}" if none_rec else f"{'N/A':<16}"
      full_str = f"{full_rec['rel_diff']:<16.2e}" if full_rec else f"{'N/A':<16}"
      tol_val = (full_rec or none_rec or {}).get("tolerance", 1e-4)
      match_none = none_rec.get("match", True) if none_rec else True
      match_full = full_rec.get("match", True) if full_rec else True
      status = "✅ MATCH" if (match_none and match_full) else "❌ DIVERGED"
      print(
          f"  {name:<40} | {none_str} | {full_str} |"
          f" {tol_val:<10.2e} | {status}"
      )
    print(num_sep)

  # 2. Latency & Speedup
  header = (
      f"  {'Configuration / Implementation':<42} | {'Remat':<6} |"
      f" {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Train (ms)':<12} |"
      f" {'Speedup vs Pure':<16} | {'Winner'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print("\n2. Latency & Speedup Summary:")
  print(sep)
  print(header)
  print(sep)

  for policy in ("none", "full"):
    if policy not in results:
      continue
    res = results[policy]
    fwd_p = res["t_fwd_pure"]
    bwd_p = res["t_bwd_pure"]
    trn_p = res["t_train_pure"]
    fwd_k = res["t_fwd_kernel"]
    bwd_k = res["t_bwd_kernel"]
    trn_k = res["t_train_kernel"]
    p_str = f"{trn_p:>10.2f} ms" if not np.isnan(trn_p) and trn_p > 0 else "FAILED"
    k_str = f"{trn_k:>10.2f} ms" if not np.isnan(trn_k) and trn_k > 0 else "FAILED"
    speedup = trn_p / trn_k if (not np.isnan(trn_p) and trn_p > 0 and not np.isnan(trn_k) and trn_k > 0) else 0.0
    speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
    winner = "🏆 Canonical GDN" if speedup > 1.05 else ("🏆 Pure JAX" if speedup < 0.95 and speedup > 0 else "≈ Parity")

    print(
        f"  {'Pure JAX GDN (Reference)':<42} | {policy:<6} |"
        f" {fwd_p:>8.2f} ms | {bwd_p:>8.2f} ms | {p_str} |"
        f" {'1.00x (ref)':<16} | ref"
    )
    print(
        f"  {'Canonical GDN Kernel (use_gdn_kernel=True)':<42} | {policy:<6} |"
        f" {fwd_k:>8.2f} ms | {bwd_k:>8.2f} ms | {k_str} |"
        f" {speedup_str:<16} | {winner}"
    )
    print(sep)

  # 3. HBM Memory Footprint Summary
  mem_header = (
      f"  {'Configuration / Implementation':<42} | {'Remat':<6} |"
      f" {'Fwd Act (MB)':<12} | {'Est Bwd (MB)':<12} | {'Peak Train (MB)':<15} |"
      f" {'Mem Ratio vs Pure':<18} | {'Savings vs Pure'}"
  )
  mem_sep = "  " + "-" * (len(mem_header) - 2)
  print("\n3. HBM Memory Footprint Summary:")
  print(mem_sep)
  print(mem_header)
  print(mem_sep)

  for policy in ("none", "full"):
    if policy not in results:
      continue
    res = results[policy]
    fwd_p_mem = res["fwd_act_mbs"][0]
    bwd_p_mem = res["bwd_peak_mbs"][0]
    trn_p_mem = res["train_peak_mbs"][0]
    fwd_k_mem = res["fwd_act_mbs"][1]
    bwd_k_mem = res["bwd_peak_mbs"][1]
    trn_k_mem = res["train_peak_mbs"][1]
    mem_ratio = trn_k_mem / trn_p_mem if trn_p_mem > 0 else 1.0
    mem_savings_pct = (1.0 - mem_ratio) * 100.0
    mem_savings_str = f"🟢 -{mem_savings_pct:.1f}%" if mem_savings_pct >= 0 else f"🔴 +{abs(mem_savings_pct):.1f}%"

    print(
        f"  {'Pure JAX GDN (Reference)':<42} | {policy:<6} |"
        f" {fwd_p_mem:>10.2f} MB | {bwd_p_mem:>10.2f} MB | {trn_p_mem:>13.2f} MB |"
        f" {'1.00x (ref)':<18} | ref"
    )
    print(
        f"  {'Canonical GDN Kernel (use_gdn_kernel=True)':<42} | {policy:<6} |"
        f" {fwd_k_mem:>10.2f} MB | {bwd_k_mem:>10.2f} MB | {trn_k_mem:>13.2f} MB |"
        f" {f'{mem_ratio:.2f}x':<18} | {mem_savings_str}"
    )
    print(mem_sep)


def print_3way_latency_summary(
    pure_none_times: list[float],
    pure_full_times: list[float],
    kernel_none_times: list[float],
    seq_len: int,
    batch_size: int,
    hidden_size: int,
) -> None:
  """Prints a 3-way latency and speedup comparison table for the structured profile."""
  print(
      "\n========================================================================================================================="
  )
  print(
      f">>> STRUCTURED 3-WAY PROFILE LATENCY & SPEEDUP (Full Qwen3.5-397B Layer: S={seq_len}, B={batch_size}, H={hidden_size}, FP32)"
  )
  print(
      "========================================================================================================================="
  )
  header = (
      f"  {'Section / Implementation':<42} | {'Remat':<6} | {'Steps':<5} |"
      f" {'Avg Train (ms)':<15} | {'Min (ms)':<10} | {'Max (ms)':<10} |"
      f" {'Speedup vs Pure(none)':<22} | {'Speedup vs Pure(full)'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  t_pn_avg = float(np.mean(pure_none_times)) if pure_none_times else float("nan")
  t_pn_min = float(np.min(pure_none_times)) if pure_none_times else float("nan")
  t_pn_max = float(np.max(pure_none_times)) if pure_none_times else float("nan")

  t_pf_avg = float(np.mean(pure_full_times)) if pure_full_times else float("nan")
  t_pf_min = float(np.min(pure_full_times)) if pure_full_times else float("nan")
  t_pf_max = float(np.max(pure_full_times)) if pure_full_times else float("nan")

  t_kn_avg = float(np.mean(kernel_none_times)) if kernel_none_times else float("nan")
  t_kn_min = float(np.min(kernel_none_times)) if kernel_none_times else float("nan")
  t_kn_max = float(np.max(kernel_none_times)) if kernel_none_times else float("nan")

  sp_kn_vs_pn = (t_pn_avg / t_kn_avg) if (t_kn_avg > 0 and t_pn_avg > 0) else 1.0
  sp_kn_vs_pf = (t_pf_avg / t_kn_avg) if (t_kn_avg > 0 and t_pf_avg > 0) else 1.0
  sp_pf_vs_pn = (t_pn_avg / t_pf_avg) if (t_pf_avg > 0 and t_pn_avg > 0) else 1.0

  print(
      f"  {'Section 1: Pure JAX GDN (Reference)':<42} | {'none':<6} | {len(pure_none_times):<5} |"
      f" {t_pn_avg:>11.2f} ms   | {t_pn_min:>7.2f} ms | {t_pn_max:>7.2f} ms |"
      f" {'1.00x (ref)':<22} | {sp_pf_vs_pn:>6.2f}x"
  )
  print(
      f"  {'Section 2: Pure JAX GDN':<42} | {'full':<6} | {len(pure_full_times):<5} |"
      f" {t_pf_avg:>11.2f} ms   | {t_pf_min:>7.2f} ms | {t_pf_max:>7.2f} ms |"
      f" {sp_pf_vs_pn:>6.2f}x                | {'1.00x (ref)':<22}"
  )
  kn_vs_pn_winner = f"{sp_kn_vs_pn:.2f}x (🏆 WINNER)" if sp_kn_vs_pn > 1.05 else f"{sp_kn_vs_pn:.2f}x"
  kn_vs_pf_winner = f"{sp_kn_vs_pf:.2f}x (🏆 WINNER)" if sp_kn_vs_pf > 1.05 else f"{sp_kn_vs_pf:.2f}x"
  print(
      f"  {'Section 3: Canonical GDN Kernel (v1.5)':<42} | {'none':<6} | {len(kernel_none_times):<5} |"
      f" {t_kn_avg:>11.2f} ms   | {t_kn_min:>7.2f} ms | {t_kn_max:>7.2f} ms |"
      f" {kn_vs_pn_winner:<22} | {kn_vs_pf_winner}"
  )
  print(sep)


def print_3mode_memory_table(
    mem_pure_none: Any | None,
    mem_pure_full: Any | None,
    mem_kernel_none: Any | None,
) -> None:
  """Prints comparative compiled HBM memory usage across the 3 benchmark modes."""
  print(
      "\n========================================================================================================================="
  )
  print(
      ">>> HBM MEMORY PROFILING & COMPILATION ANALYSIS (3-WAY COMPARISON, FP32)"
  )
  print(
      "========================================================================================================================="
  )
  header = (
      f"  {'Section / Implementation':<42} | {'Remat':<6} | {'Argument (MB)':<14} |"
      f" {'Temp/Scratch (MB)':<18} | {'Peak HBM (MB)':<14} | {'Ratio vs Pure(none)':<20} | {'Savings vs Pure(none)'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  def get_sizes(mem):
    if mem is not None:
      arg_mb = getattr(mem, "argument_size_in_bytes", 0) / (1024**2)
      tmp_mb = getattr(mem, "temp_size_in_bytes", 0) / (1024**2)
      out_mb = getattr(mem, "output_size_in_bytes", 0) / (1024**2)
      pk_mb = arg_mb + tmp_mb + out_mb
      return arg_mb, tmp_mb, out_mb, pk_mb
    return 0.0, 0.0, 0.0, 0.0

  arg_pn, tmp_pn, out_pn, pk_pn = get_sizes(mem_pure_none)
  arg_pf, tmp_pf, out_pf, pk_pf = get_sizes(mem_pure_full)
  arg_kn, tmp_kn, out_kn, pk_kn = get_sizes(mem_kernel_none)

  ref_pk = pk_pn if pk_pn > 0 else 1.0

  # Section 1
  print(
      f"  {'Section 1: Pure JAX GDN (Reference)':<42} | {'none':<6} |"
      f" {arg_pn:>11.2f} MB | {tmp_pn:>15.2f} MB | {pk_pn:>11.2f} MB |"
      f" {'1.00x (ref)':<20} | {'ref'}"
  )

  # Section 2
  r_pf = pk_pf / ref_pk if ref_pk > 0 else 1.0
  sav_pf = (1.0 - r_pf) * 100.0
  sav_pf_str = f"🟢 -{sav_pf:.1f}%" if sav_pf >= 0 else f"🔴 +{abs(sav_pf):.1f}%"
  print(
      f"  {'Section 2: Pure JAX GDN':<42} | {'full':<6} |"
      f" {arg_pf:>11.2f} MB | {tmp_pf:>15.2f} MB | {pk_pf:>11.2f} MB |"
      f" {f'{r_pf:.2f}x':<20} | {sav_pf_str}"
  )

  # Section 3
  r_kn = pk_kn / ref_pk if ref_pk > 0 else 1.0
  sav_kn = (1.0 - r_kn) * 100.0
  sav_kn_str = f"🟢 -{sav_kn:.1f}%" if sav_kn >= 0 else f"🔴 +{abs(sav_kn):.1f}%"
  print(
      f"  {'Section 3: Canonical GDN Kernel (v1.5)':<42} | {'none':<6} |"
      f" {arg_kn:>11.2f} MB | {tmp_kn:>15.2f} MB | {pk_kn:>11.2f} MB |"
      f" {f'{r_kn:.2f}x':<20} | {sav_kn_str}"
  )
  print(sep)


def run_gdn_comparison(
    batch_size: int | None = None,
    seq_len: int | None = None,
    iters: int | None = None,
    warmup: int | None = None,
    dtype_str: str | None = None,
    hidden_size: int = 4096,
    num_key_heads: int = 16,
    num_value_heads: int = 64,
    head_dim: int = 128,
    conv_kernel_dim: int = 4,
    chunk_size: int = 64,
    remat_policy: str = "structured",
    gap_seconds: float | None = None,
):
  backend = jax.default_backend()
  print(f"\nDevice: {jax.devices()[0]} ({backend})")
  print(
      "Precision: jax_default_matmul_precision = highest (TPU MXU multi-pass"
      " FP32 simulation)"
  )

  if backend == "cpu":
    hybrid_bwd_analytical_pipeline.ensure_cpu_interpret_registered()

  # Hardware defaults: Dedicate strictly to 8k sequence length on TPU in FP32
  if backend == "tpu":
    dtype = jnp.float32 if dtype_str is None else getattr(jnp, dtype_str)
    batch = 1 if batch_size is None else batch_size
    slen = 8192 if seq_len is None else seq_len
    num_iters = 5 if iters is None else iters
    num_warmup = 2 if warmup is None else warmup
    gap = 2.0 if gap_seconds is None else gap_seconds
  else:
    print("⚠️  Running on CPU: Using reduced dims and CPU interpret mode.")
    dtype = jnp.float32 if dtype_str is None else getattr(jnp, dtype_str)
    batch = 1 if batch_size is None else batch_size
    slen = 128 if seq_len is None else seq_len
    num_iters = 2 if iters is None else iters
    num_warmup = 1 if warmup is None else warmup
    gap = 0.5 if gap_seconds is None else gap_seconds

  print(f"Config: Batch={batch}, SeqLen={slen}, Dtype={dtype}")
  print(
      f"Model: H={hidden_size}, K_Heads={num_key_heads},"
      f" V_Heads={num_value_heads}, HeadDim={head_dim}, ChunkSize={chunk_size},"
      f" Structured 3-Mode Profiling ({num_iters} steps per mode, {gap:.1f}s gap)"
  )

  pure_jax_cfg, gdn_kernel_cfg = create_model_configs(
      hidden_size=hidden_size,
      num_key_heads=num_key_heads,
      num_value_heads=num_value_heads,
      head_dim=head_dim,
      conv_kernel_dim=conv_kernel_dim,
      chunk_size=chunk_size,
      dtype=dtype,
      use_qk_norm=True,
  )

  print("\nInitializing models...")
  pure_jax_model = qwen3.Qwen3NextGatedDeltaNet(
      config=pure_jax_cfg, rngs=nnx.Rngs(0)
  )
  gdn_kernel_model = qwen3.Qwen3NextGatedDeltaNet(
      config=gdn_kernel_cfg, rngs=nnx.Rngs(0)
  )

  _, params_state = nnx.split(gdn_kernel_model)
  nnx.update(pure_jax_model, params_state)
  print("✅ Both models synchronized with identical weights.")

  key = jax.random.PRNGKey(42)
  inputs = jax.random.normal(key, (batch, slen, hidden_size), dtype=dtype)

  # Create the 3 JIT-compiled training step functions
  print("\n--- Creating Functional JIT Training Steps ---")
  # Mode 1: Pure JAX remat=none
  jit_train_pure_none, params_pure_none = create_jitted_train_step(
      pure_jax_model,
      inputs.shape,
      step_scope="PureJAX_RematNone_Step",
      fwd_scope="PureJAX_RematNone_Fwd",
      bwd_scope="PureJAX_RematNone_Bwd",
      remat=False,
  )
  # Mode 2: Pure JAX remat=full
  jit_train_pure_full, params_pure_full = create_jitted_train_step(
      pure_jax_model,
      inputs.shape,
      step_scope="PureJAX_RematFull_Step",
      fwd_scope="PureJAX_RematFull_Fwd",
      bwd_scope="PureJAX_RematFull_Bwd",
      remat=True,
  )
  # Mode 3: Canonical GDN Kernel remat=none
  jit_train_kernel_none, params_kernel_none = create_jitted_train_step(
      gdn_kernel_model,
      inputs.shape,
      step_scope="GdnKernel_RematNone_Step",
      fwd_scope="GdnKernel_RematNone_Fwd",
      bwd_scope="GdnKernel_RematNone_Bwd",
      remat=False,
  )

  # =========================================================================
  # 1. WARMUP & JIT COMPILATION STRICTLY OUTSIDE THE TRACE
  # =========================================================================
  print(
      "\n========================================================================================="
  )
  print(
      f">>> STEP 1: PRE-WARMUP & JIT COMPILATION (STRICTLY OUTSIDE TRACE, {num_warmup} warmups each)"
  )
  print(
      "========================================================================================="
  )

  # Mode 1: Pure JAX remat=none
  print(f"[{time.strftime('%X')}] Compiling Mode 1: Pure JAX (remat=none)...")
  compiled_pure_none = jit_train_pure_none.lower(params_pure_none, inputs).compile()
  mem_pure_none = compiled_pure_none.memory_analysis() if hasattr(compiled_pure_none, "memory_analysis") else None
  jit_train_pure_none._cached_memory_analysis = mem_pure_none
  print(f"[{time.strftime('%X')}] Warming up Mode 1: Pure JAX (remat=none) ({num_warmup} warmups)...")
  for _ in range(num_warmup):
    res_pure_none = compiled_pure_none(params_pure_none, inputs)
    jax.block_until_ready(res_pure_none)
  loss_pure_none, out_pure_none, grads_pure_none = res_pure_none

  # Mode 2: Pure JAX remat=full
  print(f"[{time.strftime('%X')}] Compiling Mode 2: Pure JAX (remat=full)...")
  compiled_pure_full = jit_train_pure_full.lower(params_pure_full, inputs).compile()
  mem_pure_full = compiled_pure_full.memory_analysis() if hasattr(compiled_pure_full, "memory_analysis") else None
  jit_train_pure_full._cached_memory_analysis = mem_pure_full
  print(f"[{time.strftime('%X')}] Warming up Mode 2: Pure JAX (remat=full) ({num_warmup} warmups)...")
  for _ in range(num_warmup):
    res_pure_full = compiled_pure_full(params_pure_full, inputs)
    jax.block_until_ready(res_pure_full)
  loss_pure_full, out_pure_full, grads_pure_full = res_pure_full

  # Mode 3: Canonical GDN Kernel remat=none
  print(f"[{time.strftime('%X')}] Compiling Mode 3: Canonical GDN Kernel (remat=none)...")
  compiled_kernel_none = jit_train_kernel_none.lower(params_kernel_none, inputs).compile()
  mem_kernel_none = compiled_kernel_none.memory_analysis() if hasattr(compiled_kernel_none, "memory_analysis") else None
  jit_train_kernel_none._cached_memory_analysis = mem_kernel_none
  print(f"[{time.strftime('%X')}] Warming up Mode 3: Canonical GDN Kernel (remat=none) ({num_warmup} warmups)...")
  for _ in range(num_warmup):
    res_kernel_none = compiled_kernel_none(params_kernel_none, inputs)
    jax.block_until_ready(res_kernel_none)
  loss_kernel_none, out_kernel_none, grads_kernel_none = res_kernel_none
  print(f"[{time.strftime('%X')}] ✅ All 3 modes compiled and warmed up outside trace.")

  # Numerical correctness checks (outside trace)
  tol = 1e-3 if backend == "cpu" else 1e-4
  abs_tol = 1e-5
  policy_diffs_kernel = {}
  diverged_kernel = print_numerical_correctness_table(
      out_ref=out_pure_none,
      out_test=out_kernel_none,
      loss_ref=loss_pure_none,
      loss_test=loss_kernel_none,
      grads_ref=grads_pure_none,
      grads_test=grads_kernel_none,
      tolerance=tol,
      abs_tolerance=abs_tol,
      comparison_name="Canonical GDN Kernel (remat=none) vs Pure JAX (remat=none)",
      diff_records=policy_diffs_kernel,
  )

  policy_diffs_remat = {}
  diverged_remat = print_numerical_correctness_table(
      out_ref=out_pure_none,
      out_test=out_pure_full,
      loss_ref=loss_pure_none,
      loss_test=loss_pure_full,
      grads_ref=grads_pure_none,
      grads_test=grads_pure_full,
      tolerance=tol,
      abs_tolerance=abs_tol,
      comparison_name="Pure JAX (remat=full) vs Pure JAX (remat=none)",
      diff_records=policy_diffs_remat,
  )

  overall_numerical_diverged = diverged_kernel or diverged_remat

  # Memory profiling table (outside trace)
  print_3mode_memory_table(
      mem_pure_none,
      mem_pure_full,
      mem_kernel_none,
  )

  # =========================================================================
  # 2. START TRACE (PRISTINE TRACE CONTAINING ONLY THE 3 STRUCTURED SECTIONS)
  # =========================================================================
  log_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/xprof_traces")
  os.makedirs(log_dir, exist_ok=True)
  print(
      "\n========================================================================================="
  )
  print(f">>> STEP 2: STARTING PRISTINE XPROF TRACE (log_dir={log_dir})")
  print(
      "========================================================================================="
  )

  tracing_active = False
  try:
    jax.profiler.start_trace(log_dir)
    tracing_active = True
    print(f"[{time.strftime('%X')}] ✅ jax.profiler.start_trace active.")
  except Exception as e:
    print(f"⚠️ Failed to start JAX profiler trace: {e}")

  # --- Section 1: Pure JAX remat=none ---
  print(f"\n[{time.strftime('%X')}] >>> Tracing Section 1: Pure JAX remat=none ({num_iters} steps)...")
  pure_none_times = []
  for step_i in range(num_iters):
    t_s = time.perf_counter()
    with jax.named_scope("PureJAX_RematNone_Step"):
      with jax.profiler.StepTraceAnnotation("PureJAX_RematNone_Step", step_num=step_i):
        with jax.profiler.TraceAnnotation(f"PureJAX_RematNone_Step_{step_i}"):
          res = compiled_pure_none(params_pure_none, inputs)
          jax.block_until_ready(res)
    dur = (time.perf_counter() - t_s) * 1000.0
    pure_none_times.append(dur)
    print(f"    [Section 1: Pure JAX remat=none] Step {step_i + 1}/{num_iters}: {dur:.2f} ms")

  # --- Gap 1: Intentional gap with device sync ---
  print(f"\n[{time.strftime('%X')}] >>> Gap 1: Device sync & intentional {gap:.1f}s gap between Section 1 and Section 2...")
  jax.block_until_ready(res)
  with jax.profiler.TraceAnnotation(f"Gap1_Sleep_{gap:.1f}s"):
    time.sleep(gap)

  # --- Section 2: Pure JAX remat=full ---
  print(f"\n[{time.strftime('%X')}] >>> Tracing Section 2: Pure JAX remat=full ({num_iters} steps)...")
  pure_full_times = []
  for step_i in range(num_iters):
    t_s = time.perf_counter()
    with jax.named_scope("PureJAX_RematFull_Step"):
      with jax.profiler.StepTraceAnnotation("PureJAX_RematFull_Step", step_num=step_i):
        with jax.profiler.TraceAnnotation(f"PureJAX_RematFull_Step_{step_i}"):
          res = compiled_pure_full(params_pure_full, inputs)
          jax.block_until_ready(res)
    dur = (time.perf_counter() - t_s) * 1000.0
    pure_full_times.append(dur)
    print(f"    [Section 2: Pure JAX remat=full] Step {step_i + 1}/{num_iters}: {dur:.2f} ms")

  # --- Gap 2: Intentional gap with device sync ---
  print(f"\n[{time.strftime('%X')}] >>> Gap 2: Device sync & intentional {gap:.1f}s gap between Section 2 and Section 3...")
  jax.block_until_ready(res)
  with jax.profiler.TraceAnnotation(f"Gap2_Sleep_{gap:.1f}s"):
    time.sleep(gap)

  # --- Section 3: Canonical GDN Kernel remat=none ---
  print(f"\n[{time.strftime('%X')}] >>> Tracing Section 3: Canonical GDN Kernel remat=none ({num_iters} steps)...")
  kernel_none_times = []
  for step_i in range(num_iters):
    t_s = time.perf_counter()
    with jax.named_scope("GdnKernel_RematNone_Step"):
      with jax.profiler.StepTraceAnnotation("GdnKernel_RematNone_Step", step_num=step_i):
        with jax.profiler.TraceAnnotation(f"GdnKernel_RematNone_Step_{step_i}"):
          res = compiled_kernel_none(params_kernel_none, inputs)
          jax.block_until_ready(res)
    dur = (time.perf_counter() - t_s) * 1000.0
    kernel_none_times.append(dur)
    print(f"    [Section 3: Canonical GDN Kernel remat=none] Step {step_i + 1}/{num_iters}: {dur:.2f} ms")

  # --- Stop Trace ---
  if tracing_active:
    print(f"\n[{time.strftime('%X')}] Stopping XProf trace...")
    try:
      jax.profiler.stop_trace()
      print(f"[{time.strftime('%X')}] ✅ jax.profiler.stop_trace completed. Trace written to: {log_dir}")
    except Exception as e:
      print(f"⚠️ Failed to stop JAX profiler trace: {e}")

  # =========================================================================
  # 3. POST-TRACE ARTIFACTS & EXECUTIVE SUMMARY
  # =========================================================================
  xplane_files = glob.glob(os.path.join(log_dir, "**/*.xplane.pb"), recursive=True)
  print(f"\nDiscovered {len(xplane_files)} generated .xplane.pb file(s) in {log_dir}:")
  for xf in xplane_files:
    sz = os.path.getsize(xf)
    print(f"  📁 {xf} ({sz:,} bytes)")
    target_named_copy = os.path.join(log_dir, "ghostlite_structured_3modes.xplane.pb")
    if os.path.abspath(xf) != os.path.abspath(target_named_copy):
      try:
        shutil.copy(xf, target_named_copy)
      except Exception:
        pass
    try:
      os.makedirs("/tmp/xprof_traces", exist_ok=True)
      shutil.copy(xf, os.path.join("/tmp/xprof_traces", os.path.basename(xf)))
      shutil.copy(xf, "/tmp/xprof_traces/ghostlite_structured_3modes.xplane.pb")
    except Exception:
      pass

  print_3way_latency_summary(
      pure_none_times=pure_none_times,
      pure_full_times=pure_full_times,
      kernel_none_times=kernel_none_times,
      seq_len=slen,
      batch_size=batch,
      hidden_size=hidden_size,
  )

  return overall_numerical_diverged


# Backwards compatibility alias
run_analytical_comparison = run_gdn_comparison


class HybridGdnBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")
    hybrid_bwd_analytical_pipeline.ensure_cpu_interpret_registered()

  def test_benchmark_397b_8k_fp32(self):
    """Primary benchmark testing Pure JAX vs Canonical GDN Kernel (Decoupled v1.5) in FP32 at full Qwen3.5-397B config (S=8192, H=4096, V=64, K=16)."""
    backend = jax.default_backend()
    if backend == "tpu":
      print(
          "\n========================================================================================="
      )
      print(
          ">>> BENCHMARK: Dedicated 8k FP32 Comparison (Pure JAX vs Canonical GDN Kernel - Full Qwen3.5-397B Config)"
      )
      print(
          "========================================================================================="
      )
      diverged = run_gdn_comparison(
          batch_size=1,
          seq_len=8192,
          iters=5,
          warmup=2,
          dtype_str="float32",
          hidden_size=4096,
          num_key_heads=16,
          num_value_heads=64,
          head_dim=128,
          conv_kernel_dim=4,
          chunk_size=64,
          gap_seconds=2.0,
      )
    else:
      print(
          "\n========================================================================================="
      )
      print(">>> CPU HERMETIC VERIFICATION: Structured 3-Mode FP32 Comparison (S=128, B=1)")
      print(
          "========================================================================================="
      )
      diverged = run_gdn_comparison(
          batch_size=1,
          seq_len=128,
          iters=2,
          warmup=1,
          dtype_str="float32",
          hidden_size=2048,
          num_key_heads=8,
          num_value_heads=16,
          head_dim=128,
          conv_kernel_dim=4,
          chunk_size=64,
          gap_seconds=0.5,
      )
    self.assertFalse(
        diverged, "GDN Kernel gradients diverged beyond tolerance in FP32!"
    )


# Backwards compatibility alias for external imports
if __name__ != "__main__":
  class HybridGdnAnalyticalBenchmarkTest(HybridGdnBenchmarkTest):
    """Backwards compatibility alias for external imports."""
    __test__ = False


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark GDN Kernel")
  parser.add_argument("--batch_size", type=int, default=None)
  parser.add_argument("--seq_len", type=int, default=None)
  parser.add_argument("--iters", type=int, default=None)
  parser.add_argument("--warmup", type=int, default=None)
  parser.add_argument("--dtype", type=str, default=None)
  parser.add_argument("--hidden_size", type=int, default=4096)
  parser.add_argument("--num_key_heads", type=int, default=16)
  parser.add_argument("--num_value_heads", type=int, default=64)
  parser.add_argument("--head_dim", type=int, default=128)
  parser.add_argument("--conv_kernel_dim", type=int, default=4)
  parser.add_argument("--chunk_size", type=int, default=64)
  parser.add_argument(
      "--remat",
      type=str,
      default="structured",
      choices=["full", "none", "both", "structured"],
      help="Remat policy: 'full', 'none', 'both', or 'structured'",
  )
  parser.add_argument(
      "--gap_seconds",
      type=float,
      default=2.0,
      help="Gap between sections in seconds (default: 2.0)",
  )

  if "--benchmark" in sys.argv:
    sys.argv.remove("--benchmark")
    args, _ = parser.parse_known_args()
    run_gdn_comparison(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        dtype_str=args.dtype,
        hidden_size=args.hidden_size,
        num_key_heads=args.num_key_heads,
        num_value_heads=args.num_value_heads,
        head_dim=args.head_dim,
        conv_kernel_dim=args.conv_kernel_dim,
        chunk_size=args.chunk_size,
        remat_policy=args.remat,
        gap_seconds=args.gap_seconds,
    )
  else:
    absltest.main()
