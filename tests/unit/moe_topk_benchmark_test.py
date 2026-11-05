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

"""Benchmarking and verification test for Fused MoE Top-K Gating Kernel.

Evaluates the Pallas TPU fused Top-K gating kernel with analytical VJP against
the reference JAX/XLA routing implementation under the Qwen 3.5 397B architecture.
Validates numerical correctness (forward activations, loss scalar, and router parameter
gradients) and measures end-to-end training step time.
"""

import builtins
import functools
import time
import types
from typing import Any, Tuple

print = functools.partial(builtins.print, flush=True)

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.moe_topk import moe_topk_pallas
from maxtext.layers import moe


def create_moe_configs(
    model_dim: int = 4096,
    num_experts: int = 512,
    num_selected_experts: int = 10,
    dtype: Any = jnp.bfloat16,
) -> Tuple[types.SimpleNamespace, types.SimpleNamespace]:
  """Creates configs for Pure JAX (Reference) and Pallas Fused Kernel."""
  base_dict = dict(
      model_dim=model_dim,
      num_experts=num_experts,
      num_experts_per_tok=num_selected_experts,
      dtype=dtype,
      weight_dtype=dtype,
      decoder_block="standard",
      norm_topk_prob=True,
      routed_bias=False,
      routed_bias_update_rate=0.0,
      use_random_routing=False,
      model_name="qwen3_5-397b",
      load_balance_loss_weight=0.0,
      scan_layers=False,
      using_pipeline_parallelism=False,
      logical_axis_rules=(),
      shard_mode="auto",
      debug_sharding=False,
      per_expert_scale=None,
      fuse_expert_scales=False,
      model_call_mode="train",
  )

  ref_config = types.SimpleNamespace(**base_dict, use_topk_kernel=False)
  opt_config = types.SimpleNamespace(**base_dict, use_topk_kernel=True)
  return ref_config, opt_config


def print_numerical_correctness_table(
    out_ref: Any,
    out_test: Any,
    loss_ref: Any,
    loss_test: Any,
    grads_ref: Any,
    grads_test: Any,
    tolerance: float = 1e-4,
    comparison_name: str = "Pallas Kernel vs Pure JAX Reference",
) -> bool:
  """Prints structured numerical parity verification table."""
  print("\n" + "=" * 90)
  print(f">>> NUMERICAL CORRECTNESS: {comparison_name}")
  print("=" * 90)
  header = f"  {'Tensor / Parameter':<35} | {'Max Abs Diff':<14} | {'Relative Diff':<14} | {'Status'}"
  sep = "  " + "-" * (len(header) - 2)
  print(sep)
  print(header)
  print(sep)

  diverged = False

  # 1. Forward Output
  ref_t = np.asarray(out_ref[0] if isinstance(out_ref, tuple) else out_ref, dtype=np.float32)
  test_t = np.asarray(out_test[0] if isinstance(out_test, tuple) else out_test, dtype=np.float32)
  abs_d_fwd = float(np.max(np.abs(ref_t - test_t)))
  rel_d_fwd = abs_d_fwd / (float(np.max(np.abs(ref_t))) + 1e-7)
  match_fwd = abs_d_fwd <= 1e-5 or rel_d_fwd <= tolerance
  if not match_fwd:
    diverged = True
  status_fwd = "PASS" if match_fwd else "FAIL"
  print(f"  {'Forward Output':<35} | {abs_d_fwd:<14.2e} | {rel_d_fwd:<14.2e} | {status_fwd}")

  # 2. Loss Scalar
  lp = float(loss_ref)
  la = float(loss_test)
  abs_d_loss = abs(lp - la)
  rel_d_loss = abs_d_loss / (abs(lp) + 1e-7)
  match_loss = abs_d_loss <= 1e-5 or rel_d_loss <= tolerance
  if not match_loss:
    diverged = True
  status_loss = "PASS" if match_loss else "FAIL"
  print(f"  {'Loss Scalar':<35} | {abs_d_loss:<14.2e} | {rel_d_loss:<14.2e} | {status_loss}")

  # 3. Router Gradients
  flat_ref = jax.tree_util.tree_leaves(grads_ref)
  flat_test = jax.tree_util.tree_leaves(grads_test)
  for i, (gr, gt) in enumerate(zip(flat_ref, flat_test)):
    if gr is not None and gt is not None:
      gr_np = np.asarray(gr, dtype=np.float32)
      gt_np = np.asarray(gt, dtype=np.float32)
      abs_d = float(np.max(np.abs(gr_np - gt_np)))
      rel_d = abs_d / (float(np.max(np.abs(gr_np))) + 1e-7)
      m = abs_d <= 1e-4 or rel_d <= tolerance
      if not m:
        diverged = True
      status = "PASS" if m else "FAIL"
      print(f"  {f'Grad Param [{i}]':<35} | {abs_d:<14.2e} | {rel_d:<14.2e} | {status}")

  print(sep + "\n")
  return diverged


class MoeTopkBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")

  def test_fused_topk_kernel_parity(self):
    """Hermetic parity test of isolated Pallas Top-K Gating kernel with analytical VJP."""
    key = jax.random.PRNGKey(42)
    M = 1024
    num_experts = 512
    k = 10
    logits = jax.random.normal(key, (M, num_experts), dtype=jnp.bfloat16)

    def ref_gating(x):
      topk_vals, indices = jax.lax.top_k(x, k)
      weights = jax.nn.softmax(topk_vals.astype(jnp.float32), axis=-1).astype(x.dtype)
      return weights, indices

    def opt_gating(x):
      return moe_topk_pallas.fused_topk_gating(x, k)

    # Forward pass
    w_ref, idx_ref = ref_gating(logits)
    w_opt, idx_opt = opt_gating(logits)

    np.testing.assert_allclose(np.asarray(w_opt), np.asarray(w_ref), atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(np.asarray(idx_opt), np.asarray(idx_ref))

    # Backward pass VJP parity
    def loss_ref_fn(x):
      w, _ = ref_gating(x)
      return jnp.sum(w.astype(jnp.float32))

    def loss_opt_fn(x):
      w, _ = opt_gating(x)
      return jnp.sum(w.astype(jnp.float32))

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_opt = jax.grad(loss_opt_fn)(logits)

    np.testing.assert_allclose(
        np.asarray(grad_opt, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        atol=1e-3,
        rtol=1e-3,
    )

  def test_full_layer_benchmark(self):
    """Full MoE layer benchmark with Qwen 3.5 397B dimensions."""
    backend = jax.default_backend()
    batch = 1
    seq_len = 4096 if backend == "tpu" else 128
    model_dim = 4096 if backend == "tpu" else 256
    num_experts = 512 if backend == "tpu" else 32
    num_selected = 10 if backend == "tpu" else 4
    dtype = jnp.bfloat16

    ref_cfg, opt_cfg = create_moe_configs(
        model_dim=model_dim,
        num_experts=num_experts,
        num_selected_experts=num_selected,
        dtype=dtype,
    )

    key = jax.random.PRNGKey(123)
    k_inputs, k_w = jax.random.split(key)
    inputs = jax.random.normal(k_inputs, (batch, seq_len, model_dim), dtype=dtype)
    gate_w = jax.random.normal(k_w, (model_dim, num_experts), dtype=dtype)

    def layer_step(gate_weights, x, use_kernel=False):
      gate_logits = jnp.matmul(x, gate_weights)
      if use_kernel:
        top_k_weights, top_k_indices = moe_topk_pallas.fused_topk_gating(gate_logits, num_selected)
      else:
        topk_vals, top_k_indices = jax.lax.top_k(gate_logits, num_selected)
        top_k_weights = jax.nn.softmax(topk_vals.astype(jnp.float32), axis=-1).astype(dtype)
      loss = jnp.mean(top_k_weights.astype(jnp.float32))
      return loss, (top_k_weights, top_k_indices)

    step_ref = jax.jit(jax.value_and_grad(functools.partial(layer_step, use_kernel=False), has_aux=True))
    step_opt = jax.jit(jax.value_and_grad(functools.partial(layer_step, use_kernel=True), has_aux=True))

    (loss_ref, out_ref), grad_ref = step_ref(gate_w, inputs)
    (loss_opt, out_opt), grad_opt = step_opt(gate_w, inputs)

    diverged = print_numerical_correctness_table(
        out_ref=out_ref[0],
        out_test=out_opt[0],
        loss_ref=loss_ref,
        loss_test=loss_opt,
        grads_ref=grad_ref,
        grads_test=grad_opt,
    )
    self.assertFalse(diverged, "MoE Top-K Pallas Kernel diverged from Reference!")

    if backend == "tpu":
      warmup_iters = 3
      bench_iters = 20

      for _ in range(warmup_iters):
        res1 = step_ref(gate_w, inputs)
        res2 = step_opt(gate_w, inputs)
        jax.block_until_ready(res1)
        jax.block_until_ready(res2)

      t0 = time.perf_counter()
      for _ in range(bench_iters):
        res = step_ref(gate_w, inputs)
        jax.block_until_ready(res)
      ref_ms = (time.perf_counter() - t0) * 1000.0 / bench_iters

      t0 = time.perf_counter()
      for _ in range(bench_iters):
        res = step_opt(gate_w, inputs)
        jax.block_until_ready(res)
      opt_ms = (time.perf_counter() - t0) * 1000.0 / bench_iters

      speedup = ref_ms / opt_ms
      print("=" * 90)
      print(f">>> MOE TOP-K LAYER BENCHMARK RESULTS (Backend={backend}, Tokens={batch*seq_len})")
      print(f"    Baseline JAX/XLA Step Time: {ref_ms:.4f} ms")
      print(f"    Pallas Fused Kernel Step:   {opt_ms:.4f} ms")
      print(f"    Speedup:                    {speedup:.2f}x ({(ref_ms - opt_ms):.4f} ms saved per step)")
      print("=" * 90 + "\n")


if __name__ == "__main__":
  absltest.main()
