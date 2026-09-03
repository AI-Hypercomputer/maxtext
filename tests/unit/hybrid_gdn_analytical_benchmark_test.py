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
    remat_policy: str = "both",
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
    num_iters = 10 if iters is None else iters
    num_warmup = 3 if warmup is None else warmup
  else:
    print("⚠️  Running on CPU: Using reduced dims and CPU interpret mode.")
    dtype = jnp.float32 if dtype_str is None else getattr(jnp, dtype_str)
    batch = 1 if batch_size is None else batch_size
    slen = 128 if seq_len is None else seq_len
    num_iters = 3 if iters is None else iters
    num_warmup = 1 if warmup is None else warmup

  print(f"Config: Batch={batch}, SeqLen={slen}, Dtype={dtype}")
  print(
      f"Model: H={hidden_size}, K_Heads={num_key_heads},"
      f" V_Heads={num_value_heads}, HeadDim={head_dim}, ChunkSize={chunk_size},"
      f" RematPolicy={remat_policy}"
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

  # 1. Compile & Analyze Forward Pass (shared across remat policies)
  print("\n--- Compiling Forward Passes (FP32) ---")
  jit_fwd_pure, params_pure = create_jitted_forward(
      pure_jax_model, scope_name="PureJAX_Fwd"
  )
  jit_fwd_kernel, params_kernel = create_jitted_forward(
      gdn_kernel_model, scope_name="GdnKernel_Fwd"
  )

  pure_fwd_ok = False
  try:
    lowered_fwd_pure = jit_fwd_pure.lower(params_pure, inputs)
    compiled_fwd_pure = lowered_fwd_pure.compile()
    if hasattr(compiled_fwd_pure, "memory_analysis"):
      jit_fwd_pure._cached_memory_analysis = compiled_fwd_pure.memory_analysis()
    pure_fwd_ok = True
  except Exception as e:
    print(f"⚠️ Pure JAX forward compilation failed: {e}")

  kernel_fwd_ok = False
  try:
    lowered_fwd_kernel = jit_fwd_kernel.lower(params_kernel, inputs)
    compiled_fwd_kernel = lowered_fwd_kernel.compile()
    if hasattr(compiled_fwd_kernel, "memory_analysis"):
      jit_fwd_kernel._cached_memory_analysis = compiled_fwd_kernel.memory_analysis()
    kernel_fwd_ok = True
  except Exception as e:
    print(f"⚠️ Canonical GDN Kernel forward compilation failed: {e}")

  # Determine policies to benchmark
  if isinstance(remat_policy, bool):
    remat_str = "full" if remat_policy else "none"
  else:
    remat_str = str(remat_policy).lower()

  if remat_str == "both":
    policies = ["full", "none"]
  elif remat_str in ("full", "none"):
    policies = [remat_str]
  else:
    raise ValueError(f"Unknown remat_policy: {remat_policy}. Expected 'full', 'none', or 'both'.")

  tol = 1e-3 if backend == "cpu" else 1e-4
  abs_tol = 1e-5
  overall_numerical_diverged = False
  results = {}

  log_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/xprof_traces")
  os.makedirs(log_dir, exist_ok=True)
  print(
      "\n========================================================================================="
  )
  print(f">>> STARTING XPROF TRACE (log_dir={log_dir})")
  print(
      "========================================================================================="
  )

  tracing_active = False
  try:
    jax.profiler.start_trace(log_dir)
    tracing_active = True
    print("✅ jax.profiler.start_trace active.")
  except Exception as e:
    print(f"⚠️ Failed to start JAX profiler trace: {e}")

  def timed_benchmark(name, step_name, func, p, x):
    print(f"Benchmarking {name} ({step_name}) under trace...")
    t0 = time.time()
    for step_i in range(num_iters):
      with jax.profiler.StepTraceAnnotation(step_name, step_num=step_i):
        out = func(p, x)
        jax.block_until_ready(out)
    t_avg = (time.time() - t0) / num_iters * 1000.0
    print(f"  -> {t_avg:.2f} ms")
    return t_avg

  # Benchmark Forward Passes
  print(f"\nWarming up forward passes ({num_warmup} warmups each)...")
  for _ in range(num_warmup):
    if pure_fwd_ok:
      jax.block_until_ready(jit_fwd_pure(params_pure, inputs))
    if kernel_fwd_ok:
      jax.block_until_ready(jit_fwd_kernel(params_kernel, inputs))

  t_fwd_pure = timed_benchmark("Pure JAX Forward", "PureJAX_Fwd", jit_fwd_pure, params_pure, inputs) if pure_fwd_ok else float("nan")
  t_fwd_kernel = timed_benchmark("Canonical GDN Kernel Forward", "GdnKernel_Fwd", jit_fwd_kernel, params_kernel, inputs) if kernel_fwd_ok else float("nan")

  for policy in policies:
    use_remat = (policy == "full")
    print(f"\n{'='*90}")
    print(f">>> EVALUATING CONFIGURATION: remat={policy.upper()} (Pure JAX vs Canonical GDN Kernel)")
    print(f"{'='*90}")

    jit_train_pure, _ = create_jitted_train_step(
        pure_jax_model,
        inputs.shape,
        fwd_scope=f"PureJAX_Fwd_{policy}",
        bwd_scope=f"PureJAX_Bwd_{policy}",
        remat=use_remat,
    )
    jit_train_kernel, _ = create_jitted_train_step(
        gdn_kernel_model,
        inputs.shape,
        fwd_scope=f"GdnKernel_Fwd_{policy}",
        bwd_scope=f"GdnKernel_Bwd_{policy}",
        remat=use_remat,
    )

    pure_train_ok = False
    try:
      print(f"[{time.strftime('%X')}] Compiling Pure JAX training step (remat={policy})...")
      lowered_pure = jit_train_pure.lower(params_pure, inputs)
      compiled_train_pure = lowered_pure.compile()
      if hasattr(compiled_train_pure, "memory_analysis"):
        try:
          jit_train_pure._cached_memory_analysis = compiled_train_pure.memory_analysis()
        except Exception:
          pass
      loss_pure, out_pure, grads_pure = compiled_train_pure(params_pure, inputs)
      jax.block_until_ready((loss_pure, out_pure, grads_pure))
      pure_train_ok = True
      print(f"[{time.strftime('%X')}] ✅ Pure JAX training step (remat={policy}) complete.")
    except Exception as e:
      print(f"⚠️ [{time.strftime('%X')}] Pure JAX train step (remat={policy}) failed: {e}")
      loss_pure, out_pure, grads_pure = None, None, None

    kernel_train_ok = False
    try:
      print(f"[{time.strftime('%X')}] Compiling Canonical GDN Kernel training step (remat={policy})...")
      lowered_kernel = jit_train_kernel.lower(params_kernel, inputs)
      compiled_train_kernel = lowered_kernel.compile()
      if hasattr(compiled_train_kernel, "memory_analysis"):
        try:
          jit_train_kernel._cached_memory_analysis = compiled_train_kernel.memory_analysis()
        except Exception:
          pass
      loss_kernel, out_kernel, grads_kernel = compiled_train_kernel(params_kernel, inputs)
      jax.block_until_ready((loss_kernel, out_kernel, grads_kernel))
      kernel_train_ok = True
      print(f"[{time.strftime('%X')}] ✅ Canonical GDN Kernel training step (remat={policy}) complete.")
    except Exception as e:
      print(f"⚠️ [{time.strftime('%X')}] Canonical GDN Kernel train step (remat={policy}) failed: {e}")
      loss_kernel, out_kernel, grads_kernel = None, None, None

    # Numerical equivalence check
    policy_diverged = False
    policy_diffs = {}
    if pure_train_ok and kernel_train_ok:
      policy_diverged = print_numerical_correctness_table(
          out_ref=out_pure,
          out_test=out_kernel,
          loss_ref=loss_pure,
          loss_test=loss_kernel,
          grads_ref=grads_pure,
          grads_test=grads_kernel,
          tolerance=tol,
          abs_tolerance=abs_tol,
          comparison_name=f"Pure JAX vs Canonical GDN Kernel (remat={policy})",
          diff_records=policy_diffs,
      )
      if policy_diverged:
        overall_numerical_diverged = True
      else:
        print(
            f"\n✅ Canonical GDN Kernel (remat={policy}) matched Pure JAX within FP32 tolerance (< {tol:.0e})!"
        )
    else:
      policy_diverged = True
      overall_numerical_diverged = True

    # Memory Profile Analysis
    fwd_act_mbs, train_peak_mbs, bwd_peak_mbs = run_memory_profile_analysis(
        kernel_names=[
            f"Pure JAX GDN (remat={policy})",
            f"Canonical GDN Kernel (remat={policy})",
        ],
        fwd_fns=[jit_fwd_pure, jit_fwd_kernel],
        train_fns=[jit_train_pure, jit_train_kernel],
        params_list=[params_pure, params_kernel],
        inputs=inputs,
        seq_len=slen,
        batch_size=batch,
        policy_label=f"remat={policy}, ",
    )

    # Warmup and Timed Benchmark
    print(f"\nWarming up train step kernels (remat={policy}, {num_warmup} warmups each)...")
    if pure_train_ok:
      for _ in range(num_warmup):
        jax.block_until_ready(jit_train_pure(params_pure, inputs))
    if kernel_train_ok:
      for _ in range(num_warmup):
        jax.block_until_ready(jit_train_kernel(params_kernel, inputs))

    if pure_train_ok:
      t_train_pure = timed_benchmark(
          f"Pure JAX Train Step (remat={policy})",
          f"PureJAX_Train_{policy}",
          jit_train_pure,
          params_pure,
          inputs,
      )
      t_bwd_pure = max(t_train_pure - t_fwd_pure, 0.0)
    else:
      t_train_pure = float("nan")
      t_bwd_pure = float("nan")

    if kernel_train_ok:
      t_train_kernel = timed_benchmark(
          f"Canonical GDN Kernel Train Step (remat={policy})",
          f"GdnKernel_Train_{policy}",
          jit_train_kernel,
          params_kernel,
          inputs,
      )
      t_bwd_kernel = max(t_train_kernel - t_fwd_kernel, 0.0)
    else:
      t_train_kernel = float("nan")
      t_bwd_kernel = float("nan")

    kernel_names_policy = [
        f"Pure JAX GDN (remat={policy})",
        f"Canonical GDN Kernel (remat={policy})",
    ]
    fwd_lats = [t_fwd_pure, t_fwd_kernel]
    bwd_lats = [t_bwd_pure, t_bwd_kernel]
    train_lats = [t_train_pure, t_train_kernel]

    print_latency_comparison(
        kernel_names=kernel_names_policy,
        fwd_lats=fwd_lats,
        bwd_lats=bwd_lats,
        train_lats=train_lats,
        policy_label=f"[remat={policy}] ",
    )

    print_tradeoff_table(
        ref_name=kernel_names_policy[0],
        kernel_name=kernel_names_policy[1],
        fwd_ref=t_fwd_pure,
        fwd_k=t_fwd_kernel,
        bwd_ref=t_bwd_pure,
        bwd_k=t_bwd_kernel,
        train_ref=t_train_pure,
        train_k=t_train_kernel,
        fwd_mem_ref=fwd_act_mbs[0],
        fwd_mem_k=fwd_act_mbs[1],
        train_mem_ref=train_peak_mbs[0],
        train_mem_k=train_peak_mbs[1],
        policy_label=f"[remat={policy}] ",
    )

    results[policy] = {
        "t_fwd_pure": t_fwd_pure,
        "t_fwd_kernel": t_fwd_kernel,
        "t_bwd_pure": t_bwd_pure,
        "t_bwd_kernel": t_bwd_kernel,
        "t_train_pure": t_train_pure,
        "t_train_kernel": t_train_kernel,
        "fwd_act_mbs": fwd_act_mbs,
        "bwd_peak_mbs": bwd_peak_mbs,
        "train_peak_mbs": train_peak_mbs,
        "diverged": policy_diverged,
        "diff_records": policy_diffs,
    }

  if tracing_active:
    try:
      jax.profiler.stop_trace()
      print(f"✅ jax.profiler.stop_trace completed. Trace written to: {log_dir}")
    except Exception as e:
      print(f"⚠️ Failed to stop JAX profiler trace: {e}")

  # Discover generated XPlane files
  xplane_files = glob.glob(os.path.join(log_dir, "**/*.xplane.pb"), recursive=True)
  print(f"\nDiscovered {len(xplane_files)} generated .xplane.pb file(s) in {log_dir}:")
  for xf in xplane_files:
    sz = os.path.getsize(xf)
    print(f"  📁 {xf} ({sz:,} bytes)")
    try:
      os.makedirs("/tmp/xprof_traces", exist_ok=True)
      shutil.copy(xf, os.path.join("/tmp/xprof_traces", os.path.basename(xf)))
    except Exception:
      pass

  # If multiple policies were evaluated, print cross-policy summary table
  if len(policies) > 1:
    print_cross_policy_summary(results)

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
          iters=10,
          warmup=3,
          dtype_str="float32",
          hidden_size=4096,
          num_key_heads=16,
          num_value_heads=64,
          head_dim=128,
          conv_kernel_dim=4,
          chunk_size=64,
          remat_policy="both",
      )
    else:
      print(
          "\n========================================================================================="
      )
      print(">>> CPU HERMETIC VERIFICATION: FP32 Comparison (S=128, B=1)")
      print(
          "========================================================================================="
      )
      diverged = run_gdn_comparison(
          batch_size=1,
          seq_len=128,
          iters=3,
          warmup=1,
          dtype_str="float32",
          hidden_size=2048,
          num_key_heads=8,
          num_value_heads=16,
          head_dim=128,
          conv_kernel_dim=4,
          chunk_size=64,
          remat_policy="both",
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
      default="both",
      choices=["full", "none", "both"],
      help="Remat policy: 'full', 'none', or 'both'",
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
    )
  else:
    absltest.main()
