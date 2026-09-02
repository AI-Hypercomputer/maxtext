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

"""Benchmarking and verification script for Analytical Hybrid GDN kernel in MaxText.

Compares:
1. Pure JAX GDN
2. Analytical Hybrid GDN (isolated kernel with cached t_inv & closed-form
systolic matmuls)
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
  """Creates configurations for Pure JAX and Analytical GDN in FP32."""
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

  # 1. Pure JAX GDN config
  pure_jax_config = types.SimpleNamespace(
      **base_dict,
      use_gdn_kernel=False,
      use_hybrid_gdn=False,
      use_hybrid_gdn_bwd=False,
      use_hybrid_gdn_analytical=False,
  )

  # 2. Analytical Hybrid GDN config (Decoupled Conv1D Backward)
  analytical_config = types.SimpleNamespace(
      **base_dict,
      use_gdn_kernel=True,
      use_hybrid_gdn=False,
      use_hybrid_gdn_bwd=False,
      use_hybrid_gdn_analytical=True,
  )

  return pure_jax_config, analytical_config


def create_jitted_train_step(
    model: nnx.Module,
    input_shape: Tuple[int, ...],
    fwd_scope: str = "Fwd",
    bwd_scope: str = "Bwd",
):
  """Creates a pure functional, JIT-compiled training step with position-aware loss."""
  graphdef, params = nnx.split(model)

  proj_key = jax.random.PRNGKey(99)
  projection = jax.random.normal(proj_key, input_shape)

  @jax.jit
  def pure_train_step(params, x):
    m = nnx.merge(graphdef, params)

    def loss_fn(m_inner):
      with jax.named_scope(fwd_scope):
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


def print_forward_output_table(
    out_pure: Any,
    out_ana: Any,
    tolerance: float = 1e-4,
    abs_tolerance: float = 1e-5,
) -> bool:
  """Prints a formatted comparison table of forward output differences."""
  print(
      "\n========================================================================================="
  )
  print(">>> FORWARD OUTPUT EQUIVALENCE TABLE (FP32)")
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Comparison':<45} | {'Max AbsDiff':<12} | {'Rel Diff':<10} |"
      f" {'Status'}"
  )
  separator = "  " + "-" * (len(header) - 2)
  print(header)
  print(separator)

  pure_t = np.asarray(out_pure[0] if isinstance(out_pure, tuple) else out_pure)
  ana_t = np.asarray(out_ana[0] if isinstance(out_ana, tuple) else out_ana)

  abs_d = float(np.max(np.abs(pure_t - ana_t)))
  ref_max = float(np.max(np.abs(pure_t)))
  rel_d = abs_d / (ref_max + 1e-7)
  is_match = (rel_d <= tolerance) or (abs_d <= abs_tolerance)
  status_str = "✅ MATCH" if is_match else "❌ DIVERGED"
  print(
      f"  {'Pure JAX vs Analytical GDN':<45} | {abs_d:<12.2e} | {rel_d:<10.2e} |"
      f" {status_str}"
  )
  print(separator)
  return not is_match


def print_loss_scalar_table(
    loss_pure: Any,
    loss_ana: Any,
    tolerance: float = 1e-4,
    abs_tolerance: float = 1e-5,
) -> bool:
  """Prints a formatted comparison table of loss scalar differences."""
  print(
      "\n========================================================================================="
  )
  print(">>> LOSS SCALAR EQUIVALENCE TABLE (FP32)")
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Comparison':<40} | {'Pure JAX':<12} | {'Analytical':<12} |"
      f" {'AbsDiff':<12} | {'Rel Diff':<10} | {'Status'}"
  )
  separator = "  " + "-" * (len(header) - 2)
  print(header)
  print(separator)

  lp = float(loss_pure)
  la = float(loss_ana)
  abs_d = abs(lp - la)
  ref_val = abs(lp)
  rel_d = abs_d / (ref_val + 1e-7)
  is_match = (rel_d <= tolerance) or (abs_d <= abs_tolerance)
  status_str = "✅ MATCH" if is_match else "❌ DIVERGED"
  print(
      f"  {'Pure JAX vs Analytical GDN':<40} | {lp:<12.6e} | {la:<12.6e} |"
      f" {abs_d:<12.2e} | {rel_d:<10.2e} | {status_str}"
  )
  print(separator)
  return not is_match


def print_gradient_comparison_table(
    grads_ref: Any,
    grads_test: Any,
    tolerance: float = 1e-4,
    abs_tolerance: float = 1e-5,
    label: str = "Pure vs Analytical",
) -> bool:
  """Prints an itemized per-parameter gradient comparison table."""
  print(f"\n  --- Detailed Parameter Gradient Breakdown ({label}) ---")
  header = (
      f"  {'Parameter Path':<40} | {'Max AbsDiff':<12} | {'Rel Diff':<10} |"
      f" {'Status'}"
  )
  print(header)
  print("  " + "-" * len(header))

  ref_leaves = jax.tree_util.tree_leaves_with_path(grads_ref)
  test_leaves = jax.tree_util.tree_leaves_with_path(grads_test)

  overall_diverged = False
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
    ref_max = float(np.max(np.abs(g_ref_np)))
    rel_d = abs_d / (ref_max + 1e-7)

    is_match = (rel_d <= tolerance) or (abs_d <= abs_tolerance)
    if not is_match:
      overall_diverged = True
      status_str = "❌ DIVERGED"
    else:
      status_str = "✅ MATCH"

    print(f"  {name:<40} | {abs_d:<12.2e} | {rel_d:<10.2e} | {status_str}")

  return overall_diverged


def print_numerical_correctness_table(
    out_pure: Any,
    out_ana: Any,
    loss_pure: Any,
    loss_ana: Any,
    grads_pure: Any,
    grads_ana: Any,
    tolerance: float = 1e-4,
    abs_tolerance: float = 1e-5,
) -> bool:
  """Prints a unified 2-way numerical correctness comparison table."""
  print(
      "\n========================================================================================="
  )
  print(">>> NUMERICAL CORRECTNESS TABLE: 2-WAY COMPARISON (Pure JAX vs Analytical GDN)")
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
  pure_t = np.asarray(out_pure[0] if isinstance(out_pure, tuple) else out_pure)
  ana_t = np.asarray(out_ana[0] if isinstance(out_ana, tuple) else out_ana)
  abs_d_fwd = float(np.max(np.abs(pure_t - ana_t)))
  rel_d_fwd = abs_d_fwd / (float(np.max(np.abs(pure_t))) + 1e-7)
  match_fwd = (rel_d_fwd <= tolerance) or (abs_d_fwd <= abs_tolerance)
  rows.append(("Forward Output", abs_d_fwd, rel_d_fwd, match_fwd))

  # 2. Loss Scalar
  lp = float(loss_pure)
  la = float(loss_ana)
  abs_d_loss = abs(lp - la)
  rel_d_loss = abs_d_loss / (abs(lp) + 1e-7)
  match_loss = (rel_d_loss <= tolerance) or (abs_d_loss <= abs_tolerance)
  rows.append(("Loss Scalar", abs_d_loss, rel_d_loss, match_loss))

  # 3. Parameter Gradients
  ref_leaves = jax.tree_util.tree_leaves_with_path(grads_pure)
  test_leaves = jax.tree_util.tree_leaves_with_path(grads_ana)

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
):
  """Measures and displays comparative HBM memory usage across implementations."""
  print(
      "\n========================================================================================="
  )
  print(
      f">>> HBM MEMORY PROFILING & COMPARATIVE ANALYSIS (S={seq_len},"
      f" B={batch_size}, Dtype=FP32)"
  )
  print(
      "========================================================================================="
  )

  fwd_act_mbs = []
  train_peak_mbs = []
  bwd_peak_mbs = []
  fwd_compiled_mbs = []
  train_compiled_mbs = []
  breakdown_rows = []

  for name, fwd_fn, train_fn, p in zip(
      kernel_names, fwd_fns, train_fns, params_list
  ):
    # 1. Forward Pass Memory
    mem_before_fwd = get_device_memory_stats()
    out_fwd = fwd_fn(p, inputs)
    jax.block_until_ready(out_fwd)
    mem_after_fwd = get_device_memory_stats()
    fwd_analysis = get_compiled_memory_analysis(fwd_fn, p, inputs)

    # 2. Train Step Memory
    mem_before_train = get_device_memory_stats()
    out_train = train_fn(p, inputs)
    jax.block_until_ready(out_train)
    mem_after_train = get_device_memory_stats()
    train_analysis = get_compiled_memory_analysis(train_fn, p, inputs)

    dev_in_use_fwd = (mem_after_fwd["bytes_in_use"] / (1024**2)) if mem_after_fwd else 0.0
    dev_peak_fwd = (mem_after_fwd.get("peak_bytes_in_use", 0) / (1024**2)) if mem_after_fwd else 0.0
    dev_in_use_train = (mem_after_train["bytes_in_use"] / (1024**2)) if mem_after_train else 0.0
    dev_peak_train = (mem_after_train.get("peak_bytes_in_use", 0) / (1024**2)) if mem_after_train else 0.0

    # Calculate Forward Activation Memory
    if fwd_analysis is not None:
      fwd_act_mb = fwd_analysis.temp_size_in_bytes / (1024**2)
      fwd_peak_compiled_mb = (
          fwd_analysis.argument_size_in_bytes
          + fwd_analysis.temp_size_in_bytes
          + fwd_analysis.output_size_in_bytes
          - fwd_analysis.alias_size_in_bytes
      ) / (1024**2)
    else:
      if mem_after_fwd and mem_before_fwd:
        fwd_act_mb = max(
            (mem_after_fwd.get("peak_bytes_in_use", 0)
             - mem_before_fwd.get("bytes_in_use", 0))
            / (1024**2),
            0.0,
        )
      else:
        fwd_act_mb = 0.0
      fwd_peak_compiled_mb = dev_peak_fwd

    # Calculate Peak Training Step Memory
    if train_analysis is not None:
      train_peak_compiled_mb = (
          train_analysis.argument_size_in_bytes
          + train_analysis.temp_size_in_bytes
          + train_analysis.output_size_in_bytes
          - train_analysis.alias_size_in_bytes
      ) / (1024**2)
      train_peak_mb = train_peak_compiled_mb
    else:
      if mem_after_train:
        train_peak_mb = mem_after_train.get("peak_bytes_in_use", 0) / (1024**2)
      else:
        train_peak_mb = 0.0

    # If runtime peak is available and higher, record runtime peak
    if mem_after_train and "peak_bytes_in_use" in mem_after_train:
      dev_peak = mem_after_train["peak_bytes_in_use"] / (1024**2)
      if train_peak_mb == 0.0:
        train_peak_mb = dev_peak

    bwd_peak_mb = max(train_peak_mb - fwd_act_mb, 0.0)

    fwd_act_mbs.append(fwd_act_mb)
    train_peak_mbs.append(train_peak_mb)
    bwd_peak_mbs.append(bwd_peak_mb)
    fwd_compiled_mbs.append(fwd_peak_compiled_mb)
    train_compiled_mbs.append(
        train_peak_compiled_mb if train_analysis is not None else train_peak_mb
    )

    # Detailed breakdown rows
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
      breakdown_rows.append((
          name,
          "Forward",
          0.0,
          fwd_act_mb,
          0.0,
          fwd_peak_compiled_mb,
          dev_in_use_fwd,
          dev_peak_fwd,
      ))
      breakdown_rows.append((
          name,
          "Backward (Est.)",
          0.0,
          bwd_peak_mb,
          0.0,
          bwd_peak_mb,
          dev_in_use_train,
          dev_peak_train,
      ))
      breakdown_rows.append((
          name,
          "Train Step",
          0.0,
          train_peak_mb,
          0.0,
          train_peak_mb,
          dev_in_use_train,
          dev_peak_train,
      ))

  # 1. Comparative Summary Table
  ref_fwd = fwd_act_mbs[0] if fwd_act_mbs[0] > 0 else 1.0
  ref_train = train_peak_mbs[0] if train_peak_mbs[0] > 0 else 1.0

  summary_header = (
      f"  {'Kernel Implementation':<32} | {'Fwd Activation Mem':<20} |"
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
        f"  [{i + 1}] {kernel_names[i]:<28} | {f_mb:>16.2f} MB |"
        f" {b_mb:>14.2f} MB | {t_mb:>14.2f} MB | {f_str:<20} | {t_str}"
    )
  print(separator)

  # 2. Detailed Buffer Breakdown (if compiled analysis available)
  if breakdown_rows:
    print("\nDetailed Memory Breakdown (XLA Compiled Buffers & Allocator):")
    b_header = (
        f"  {'Implementation':<28} | {'Pass':<16} | {'Argument':<12} |"
        f" {'Temp / Scratch':<14} | {'Output':<10} | {'Peak Total':<12} |"
        f" {'Dev In-Use':<12} | {'Dev Peak'}"
    )
    b_sep = "  " + "-" * len(b_header)
    print(b_sep)
    print(b_header)
    print(b_sep)
    for impl, scope, arg, tmp, out, pk, dev_u, dev_pk in breakdown_rows:
      print(
          f"  {impl:<28} | {scope:<16} | {arg:>9.2f} MB | {tmp:>11.2f} MB |"
          f" {out:>7.2f} MB | {pk:>9.2f} MB | {dev_u:>9.2f} MB | {dev_pk:>9.2f} MB"
      )
    print(b_sep)

  # 3. 2-Way Comparative Memory Profile Table
  if len(kernel_names) == 2:
    print(
        "\n========================================================================================="
    )
    print(
        f">>> MEMORY PROFILE: 2-WAY COMPARISON ({kernel_names[0]} vs {kernel_names[1]})"
    )
    print(
        "========================================================================================="
    )
    m_header = (
        f"  {'Memory Metric':<28} | {kernel_names[0]:<14} | {kernel_names[1]:<15} |"
        f" {'Savings Ratio':<14} | {'Savings (%)':<12} | {'Winner'}"
    )
    m_sep = "  " + "-" * (len(m_header) - 2)
    print(m_sep)
    print(m_header)
    print(m_sep)

    mem_metrics = [
        ("Peak Compiled Memory", train_compiled_mbs[0], train_compiled_mbs[1]),
        ("Forward Activation Memory", fwd_act_mbs[0], fwd_act_mbs[1]),
        ("Peak Training Step Memory", train_peak_mbs[0], train_peak_mbs[1]),
    ]

    for metric_name, m_pure, m_ana in mem_metrics:
      if m_ana > 0 and m_pure > 0:
        ratio = m_pure / m_ana
        diff_pct = (1.0 - (m_ana / m_pure)) * 100.0
        ratio_str = f"{ratio:.2f}x"
        color = "🟢" if diff_pct >= 0 else "🔴"
        pct_str = f"{color} {diff_pct:+.1f}%"
        winner = f"🏆 {kernel_names[1]}" if ratio >= 1.0 else f"🏆 {kernel_names[0]}"
      else:
        ratio_str, pct_str, winner = "N/A", "N/A", "N/A"

      print(
          f"  {metric_name:<28} | {m_pure:>11.2f} MB | {m_ana:>12.2f} MB |"
          f" {ratio_str:>14} | {pct_str:>12} | {winner}"
      )
    print(m_sep)
  else:
    min_mem_idx = int(np.argmin(train_peak_mbs))
    print(
        f"🏆 Most Memory Efficient (Train Step): [{min_mem_idx + 1}]"
        f" {kernel_names[min_mem_idx]} ({train_peak_mbs[min_mem_idx]:.2f} MB)\n"
    )


def print_2way_latency_comparison(
    kernel_names: list[str],
    fwd_lats: list[float],
    bwd_lats: list[float],
    train_lats: list[float],
) -> None:
  """Prints a clean 2-way latency and speedup comparison table (Pure JAX vs Analytical GDN)."""
  print(
      "\n========================================================================================="
  )
  print(
      f">>> LATENCY & SPEEDUP: 2-WAY COMPARISON ({kernel_names[0]} vs {kernel_names[1]})"
  )
  print(
      "========================================================================================="
  )
  header = (
      f"  {'Pass / Step':<24} | {kernel_names[0]:<14} | {kernel_names[1]:<15} |"
      f" {'Speedup Ratio':<14} | {'Speedup (%)':<12} | {'Champion'}"
  )
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  passes = [
      ("Forward Pass", fwd_lats[0], fwd_lats[1]),
      ("Backward Pass", bwd_lats[0], bwd_lats[1]),
      ("Full Training Step", train_lats[0], train_lats[1]),
  ]

  for step_name, t_pure, t_ana in passes:
    if t_ana > 0:
      ratio = t_pure / t_ana
      pct = (ratio - 1.0) * 100.0
      ratio_str = f"{ratio:.2f}x"
      color = "🟢" if pct >= 0 else "🔴"
      pct_str = f"{color} {pct:+.1f}%"
      champ = f"🏆 {kernel_names[1]}" if ratio >= 1.0 else f"🏆 {kernel_names[0]}"
    else:
      ratio_str, pct_str, champ = "N/A", "N/A", "N/A"

    print(
        f"  {step_name:<24} | {t_pure:>11.2f} ms | {t_ana:>12.2f} ms |"
        f" {ratio_str:>14} | {pct_str:>12} | {champ}"
    )
  print(sep)


def print_pairwise_grid(
    metric_name: str,
    kernel_names: list[str],
    latencies: list[float],
) -> Tuple[str, float]:
  """Backwards-compatible helper returning best name and latency."""
  best_idx = int(np.argmin(latencies))
  return kernel_names[best_idx], latencies[best_idx]


print_3x3_pairwise_grid = print_pairwise_grid


def run_analytical_comparison(
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
      f" V_Heads={num_value_heads}, HeadDim={head_dim}, ChunkSize={chunk_size}"
  )

  pure_jax_cfg, analytical_cfg = create_model_configs(
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
  analytical_model = qwen3.Qwen3NextGatedDeltaNet(
      config=analytical_cfg, rngs=nnx.Rngs(0)
  )

  _, params_state = nnx.split(analytical_model)
  nnx.update(pure_jax_model, params_state)
  print("✅ Models synchronized with identical weights.")

  key = jax.random.PRNGKey(42)
  inputs = jax.random.normal(key, (batch, slen, hidden_size), dtype=dtype)

  print("\n--- Checking Numerical Equivalence in FP32 ---")
  jit_train_pure, params_pure = create_jitted_train_step(
      pure_jax_model,
      inputs.shape,
      fwd_scope="PureJAX_Fwd",
      bwd_scope="PureJAX_Bwd",
  )
  jit_train_analytical, params_analytical = create_jitted_train_step(
      analytical_model,
      inputs.shape,
      fwd_scope="Analytical_Fwd",
      bwd_scope="Analytical_Bwd",
  )

  loss_pure, out_pure, grads_pure = jit_train_pure(params_pure, inputs)
  jax.block_until_ready((loss_pure, out_pure, grads_pure))

  loss_ana, out_ana, grads_ana = jit_train_analytical(params_analytical, inputs)
  jax.block_until_ready((loss_ana, out_ana, grads_ana))

  out_pure_tensor = out_pure[0] if isinstance(out_pure, tuple) else out_pure
  out_ana_tensor = out_ana[0] if isinstance(out_ana, tuple) else out_ana

  tol = 1e-3 if backend == "cpu" else 1e-4
  abs_tol = 1e-5

  overall_numerical_diverged = print_numerical_correctness_table(
      out_pure=out_pure,
      out_ana=out_ana,
      loss_pure=loss_pure,
      loss_ana=loss_ana,
      grads_pure=grads_pure,
      grads_ana=grads_ana,
      tolerance=tol,
      abs_tolerance=abs_tol,
  )

  if not overall_numerical_diverged:
    print(
        "\n✅ All implementations matched within FP32 tolerance across"
        " forward outputs, loss scalars, and parameter gradients!"
    )
  else:
    print(
        "\n⚠️ Divergence detected beyond tolerance across implementations!"
    )

  # Performance Benchmark & XProf Tracing
  print("\n--- Performance Benchmark & XProf Tracing (FP32) ---")

  jit_fwd_pure, _ = create_jitted_forward(
      pure_jax_model, scope_name="PureJAX_Fwd"
  )
  jit_fwd_ana, _ = create_jitted_forward(
      analytical_model, scope_name="Analytical_Fwd"
  )

  kernel_names = [
      "Pure JAX GDN",
      "Analytical GDN",
  ]
  fwd_fns = [jit_fwd_pure, jit_fwd_ana]
  train_fns = [jit_train_pure, jit_train_analytical]
  params_list = [params_pure, params_analytical]

  # Memory Profile Analysis (HBM Usage)
  run_memory_profile_analysis(
      kernel_names=kernel_names,
      fwd_fns=fwd_fns,
      train_fns=train_fns,
      params_list=params_list,
      inputs=inputs,
      seq_len=slen,
      batch_size=batch,
  )

  # Warmup all forward and train step functions before profiling
  print(
      f"\nWarming up kernels ({num_warmup} warmups each to complete JIT"
      " compilation)..."
  )
  for name, fn, p in [
      ("Pure JAX Forward", jit_fwd_pure, params_pure),
      ("Pure JAX Train Step", jit_train_pure, params_pure),
      ("Analytical GDN Forward", jit_fwd_ana, params_analytical),
      (
          "Analytical GDN Train Step",
          jit_train_analytical,
          params_analytical,
      ),
  ]:
    for _ in range(num_warmup):
      out = fn(p, inputs)
      jax.block_until_ready(out)
  print("✅ Warmup complete. All JIT compilations finished.")

  log_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/xprof_traces")
  os.makedirs(log_dir, exist_ok=True)
  print(
      f"\n========================================================================================="
  )
  print(f">>> STARTING XPROF TRACE (log_dir={log_dir})")
  print(
      f"========================================================================================="
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

  # [1] Pure JAX GDN
  t_fwd_pure = timed_benchmark(
      "Pure JAX Forward", "PureJAX_Fwd", jit_fwd_pure, params_pure, inputs
  )
  t_train_pure = timed_benchmark(
      "Pure JAX Train Step", "PureJAX_Bwd", jit_train_pure, params_pure, inputs
  )

  # [2] Analytical GDN
  t_fwd_ana = timed_benchmark(
      "Analytical GDN Forward",
      "Analytical_Fwd",
      jit_fwd_ana,
      params_analytical,
      inputs,
  )
  t_train_ana = timed_benchmark(
      "Analytical GDN Train Step",
      "Analytical_Bwd",
      jit_train_analytical,
      params_analytical,
      inputs,
  )

  if tracing_active:
    try:
      jax.profiler.stop_trace()
      print(
          f"✅ jax.profiler.stop_trace completed. Trace written to: {log_dir}"
      )
    except Exception as e:
      print(f"⚠️ Failed to stop JAX profiler trace: {e}")

  # Discover generated XPlane files
  xplane_files = glob.glob(
      os.path.join(log_dir, "**/*.xplane.pb"), recursive=True
  )
  print(
      f"\nDiscovered {len(xplane_files)} generated .xplane.pb file(s) in"
      f" {log_dir}:"
  )
  for xf in xplane_files:
    sz = os.path.getsize(xf)
    print(f"  📁 {xf} ({sz:,} bytes)")
    try:
      os.makedirs("/tmp/xprof_traces", exist_ok=True)
      shutil.copy(xf, os.path.join("/tmp/xprof_traces", os.path.basename(xf)))
    except Exception:
      pass

  # XPlane files are saved to TEST_UNDECLARED_OUTPUTS_DIR for post-run upload.

  t_bwd_pure = max(t_train_pure - t_fwd_pure, 0.0)
  t_bwd_ana = max(t_train_ana - t_fwd_ana, 0.0)

  fwd_lats = [t_fwd_pure, t_fwd_ana]
  bwd_lats = [t_bwd_pure, t_bwd_ana]
  train_lats = [t_train_pure, t_train_ana]

  print_2way_latency_comparison(
      kernel_names=kernel_names,
      fwd_lats=fwd_lats,
      bwd_lats=bwd_lats,
      train_lats=train_lats,
  )

  best_fwd, best_fwd_lat = print_pairwise_grid(
      "Forward Pass", kernel_names, fwd_lats
  )
  best_bwd, best_bwd_lat = print_pairwise_grid(
      "Backward Pass", kernel_names, bwd_lats
  )
  best_train, best_train_lat = print_pairwise_grid(
      "Full Training Step", kernel_names, train_lats
  )

  print(
      "========================================================================================="
  )
  print(
      f">>> OVERALL BENCHMARK CONCLUSION & BEST KERNEL (S={slen}, B={batch},"
      " Dtype=FP32)"
  )
  print(
      "========================================================================================="
  )
  print(f"  • Forward Pass Champion:       {best_fwd} ({best_fwd_lat:.2f} ms)")
  print(f"  • Backward Pass Champion:      {best_bwd} ({best_bwd_lat:.2f} ms)")
  print(
      f"  • Full Training Step Champion: {best_train} ({best_train_lat:.2f} ms)"
  )
  print(
      "=========================================================================================\n"
  )

  return overall_numerical_diverged


class HybridGdnAnalyticalBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")
    hybrid_bwd_analytical_pipeline.ensure_cpu_interpret_registered()

  def test_benchmark_8k_fp32(self):
    """Primary benchmark testing Pure JAX vs Analytical GDN in FP32 at 8k with Qwen3.5-397B dimensions."""
    backend = jax.default_backend()
    if backend == "tpu":
      print(
          "\n========================================================================================="
      )
      print(
          ">>> BENCHMARK: Dedicated 8k FP32 Comparison (Pure JAX vs Analytical"
          " GDN - Qwen3.5-397B)"
      )
      print(
          "========================================================================================="
      )
      diverged = run_analytical_comparison(
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
      )
    else:
      print(
          "\n========================================================================================="
      )
      print(">>> CPU HERMETIC VERIFICATION: FP32 Comparison (S=128, B=1)")
      print(
          "========================================================================================="
      )
      diverged = run_analytical_comparison(
          batch_size=1,
          seq_len=128,
          iters=3,
          warmup=1,
          dtype_str="float32",
          hidden_size=4096,
          num_key_heads=16,
          num_value_heads=64,
          head_dim=128,
          conv_kernel_dim=4,
          chunk_size=64,
      )
    self.assertFalse(
        diverged, "Analytical GDN gradients diverged beyond tolerance in FP32!"
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark Analytical GDN")
  parser.add_argument("--batch_size", type=int, default=None)
  parser.add_argument("--seq_len", type=int, default=None)
  parser.add_argument("--iters", type=int, default=None)
  parser.add_argument("--warmup", type=int, default=None)
  parser.add_argument("--dtype", type=str, default=None)

  if "--benchmark" in sys.argv:
    sys.argv.remove("--benchmark")
    args, _ = parser.parse_known_args()
    run_analytical_comparison(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        dtype_str=args.dtype,
    )
  else:
    absltest.main()
