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

"""Directional finite-difference gradient checker in float32 with frozen routing/indexer selections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common import common_types as ctypes
from maxtext.layers import attention_compressed
from maxtext.layers import moe


class FrozenTopKVar(nnx.Variable):
  """Non-Param NNX Variable holding frozen discrete top-k indices across unscanned or scanned layers."""


@dataclasses.dataclass
class FDCheckDirectionResult:
  """Result for a single random direction across multiple step sizes."""

  direction_idx: int
  analytical: float
  fd_estimates: dict[float, float]
  abs_errors: dict[float, float]
  rel_errors: dict[float, float]
  min_rel_error: float
  best_eps: float


@dataclasses.dataclass
class FDCheckReport:
  """Summary report for directional finite-difference gradient check."""

  num_directions: int
  epsilons: tuple[float, ...]
  directions: list[FDCheckDirectionResult]
  max_min_rel_error: float
  passed: bool
  rtol: float


@contextlib.contextmanager
def freeze_discrete_selections(
    root_module: nnx.Module,
    moe_indices_by_path: dict[tuple[Any, ...], np.ndarray] | None = None,
    indexer_indices_by_path: dict[tuple[Any, ...], np.ndarray] | None = None,
):
  """Freezes RoutedMoE and DeepseekV4Indexer top-k indices using non-Param FrozenTopKVar attributes.

  Because FrozenTopKVar is an nnx.Variable (and not nnx.Param), NNXDecoder._apply_layers_sequentially
  automatically slices stacked [num_scanned_blocks, B, S, k] arrays along axis 0 inside jax.lax.scan
  while keeping them outside the differentiated nnx.Param tree.
  """
  patched_modules: list[nnx.Module] = []
  moe_indices_by_path = moe_indices_by_path or {}
  indexer_indices_by_path = indexer_indices_by_path or {}

  for path, node in nnx.iter_graph(root_module):
    if isinstance(node, moe.RoutedMoE) and path in moe_indices_by_path:
      arr = jnp.asarray(moe_indices_by_path[path], dtype=jnp.int32)
      meta = {"param_scan_axis": 0, nnx.PARTITION_NAME: "scanned_blocks"} if arr.ndim == 4 else {}
      node._frozen_topk = FrozenTopKVar(arr, **meta)  # pylint: disable=protected-access
      patched_modules.append(node)
    elif isinstance(node, attention_compressed.DeepseekV4Indexer) and path in indexer_indices_by_path:
      arr = jnp.asarray(indexer_indices_by_path[path], dtype=jnp.int32)
      meta = {"param_scan_axis": 0, nnx.PARTITION_NAME: "scanned_blocks"} if arr.ndim == 4 else {}
      node._frozen_topk = FrozenTopKVar(arr, **meta)  # pylint: disable=protected-access
      patched_modules.append(node)

  orig_get_topk = moe.RoutedMoE.get_topk
  orig_indexer_call = attention_compressed.DeepseekV4Indexer.__call__

  def _frozen_get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None, forced_routed_experts=None):
    frozen_var = getattr(self, "_frozen_topk", None)
    if frozen_var is not None and forced_routed_experts is None:
      top_k_indices = frozen_var.get_value().astype(jnp.int32)
      valid_mask = moe.valid_expert_mask(top_k_indices, self.num_experts)
      gather_indices = jnp.where(valid_mask, top_k_indices, 0)
      if self.is_hash_routing or self.config.model_name.startswith(("deepseek3", "deepseek4", "kimi-k2")):
        top_k_weights = jnp.take_along_axis(pre_bias_logits, gather_indices, axis=-1)
      elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
        top_k_weights = jnp.take_along_axis(router_probs, gather_indices, axis=-1).astype(self.dtype)
      else:
        top_k_weights = jnp.take_along_axis(gate_logits, gather_indices, axis=-1)
      if self.config.decoder_block in (ctypes.DecoderBlockType.DEEPSEEK, ctypes.DecoderBlockType.DEEPSEEK4):
        top_k_weights = self.deepseek_scale_weights(top_k_weights)
        top_k_weights = top_k_weights * valid_mask
      return top_k_weights, top_k_indices
    return orig_get_topk(
        self,
        gate_logits,
        pre_bias_logits,
        rngs=rngs,
        input_ids=input_ids,
        forced_routed_experts=forced_routed_experts,
    )

  def _frozen_indexer_call(self, *args, **kwargs):
    final_indices, index_scores = orig_indexer_call(self, *args, **kwargs)
    frozen_var = getattr(self, "_frozen_topk", None)
    if frozen_var is not None:
      return frozen_var.get_value().astype(jnp.int32), index_scores
    return final_indices, index_scores

  moe.RoutedMoE.get_topk = _frozen_get_topk
  attention_compressed.DeepseekV4Indexer.__call__ = _frozen_indexer_call
  try:
    yield
  finally:
    moe.RoutedMoE.get_topk = orig_get_topk
    attention_compressed.DeepseekV4Indexer.__call__ = orig_indexer_call
    for mod in patched_modules:
      if hasattr(mod, "_frozen_topk"):
        delattr(mod, "_frozen_topk")


def capture_and_stack_selections(
    root_module: nnx.Module,
    forward_fn: Callable[[], Any],
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[tuple[Any, ...], np.ndarray]]:
  """Runs forward_fn once and captures per-module (or stacked scan-block) top-k selections."""
  moe_records: list[np.ndarray] = []
  idx_records: list[np.ndarray] = []

  orig_get_topk = moe.RoutedMoE.get_topk
  orig_indexer_call = attention_compressed.DeepseekV4Indexer.__call__

  def _rec_moe(arr):
    moe_records.append(np.asarray(arr, dtype=np.int32))

  def _rec_idx(arr):
    idx_records.append(np.asarray(arr, dtype=np.int32))

  def _cap_get_topk(self, gate_logits, *args, **kwargs):
    weights, indices = orig_get_topk(self, gate_logits, *args, **kwargs)
    jax.debug.callback(_rec_moe, indices, ordered=True)
    return weights, indices

  def _cap_indexer_call(self, *args, **kwargs):
    indices, scores = orig_indexer_call(self, *args, **kwargs)
    jax.debug.callback(_rec_idx, indices, ordered=True)
    return indices, scores

  moe.RoutedMoE.get_topk = _cap_get_topk
  attention_compressed.DeepseekV4Indexer.__call__ = _cap_indexer_call
  try:
    forward_fn()
    jax.effects_barrier()
  finally:
    moe.RoutedMoE.get_topk = orig_get_topk
    attention_compressed.DeepseekV4Indexer.__call__ = orig_indexer_call

  # Map captured records back to module paths in execution order.
  moe_paths: list[tuple[Any, ...]] = []
  idx_paths: list[tuple[Any, ...]] = []
  scanned_moe_paths: list[tuple[Any, ...]] = []
  scanned_idx_paths: list[tuple[Any, ...]] = []

  for path, node in nnx.iter_graph(root_module):
    if isinstance(node, moe.RoutedMoE):
      if "scanned_blocks" in path:
        scanned_moe_paths.append(path)
      else:
        moe_paths.append(path)
    elif isinstance(node, attention_compressed.DeepseekV4Indexer):
      if "scanned_blocks" in path:
        scanned_idx_paths.append(path)
      else:
        idx_paths.append(path)

  moe_by_path: dict[tuple[Any, ...], np.ndarray] = {}
  idx_by_path: dict[tuple[Any, ...], np.ndarray] = {}

  cursor = 0
  for p in moe_paths:
    moe_by_path[p] = moe_records[cursor]
    cursor += 1
  if scanned_moe_paths:
    rem = len(moe_records) - cursor
    n_sub = len(scanned_moe_paths)
    n_blocks = rem // n_sub
    for s_idx, p in enumerate(scanned_moe_paths):
      moe_by_path[p] = np.stack([moe_records[cursor + b * n_sub + s_idx] for b in range(n_blocks)], axis=0)

  cursor = 0
  for p in idx_paths:
    idx_by_path[p] = idx_records[cursor]
    cursor += 1
  if scanned_idx_paths:
    rem = len(idx_records) - cursor
    n_sub = len(scanned_idx_paths)
    n_blocks = rem // n_sub
    for s_idx, p in enumerate(scanned_idx_paths):
      idx_by_path[p] = np.stack([idx_records[cursor + b * n_sub + s_idx] for b in range(n_blocks)], axis=0)

  return moe_by_path, idx_by_path


def directional_fd_check(
    scalar_fn: Callable[[Any], jax.Array],
    params: Any,
    *,
    num_directions: int = 4,
    epsilons: Sequence[float] = (1e-2, 5e-3, 1e-3),
    seed: int = 0,
    rtol: float = 1e-3,
) -> FDCheckReport:
  """Checks analytical gradient <g, v> against central FD [f(w+eps*v) - f(w-eps*v)] / (2*eps) in float32."""
  eval_fn = jax.jit(scalar_fn)
  grad_fn = jax.jit(jax.grad(scalar_fn))
  grads = grad_fn(params)

  leaves, treedef = jax.tree_util.tree_flatten(params)
  grad_leaves, _ = jax.tree_util.tree_flatten(grads)

  grad_norm_sq = sum(
      float(np.sum(np.asarray(g.get_value() if isinstance(g, nnx.Variable) else g, dtype=np.float64) ** 2))
      for g in grad_leaves
  )
  use_grad_modulation = grad_norm_sq > 0.0

  rng = np.random.default_rng(seed)
  results: list[FDCheckDirectionResult] = []

  for d_idx in range(num_directions):
    raw_v = []
    norm_sq = 0.0
    for leaf, g_leaf in zip(leaves, grad_leaves):
      arr = np.asarray(leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf)
      if np.issubdtype(arr.dtype, np.floating):
        z = rng.standard_normal(arr.shape).astype(np.float64)
        if use_grad_modulation:
          g_arr = np.asarray(g_leaf.get_value() if isinstance(g_leaf, nnx.Variable) else g_leaf, dtype=np.float64)
          v_leaf = (g_arr * np.exp(0.5 * z)).astype(np.float32)
        else:
          v_leaf = z.astype(np.float32)
        norm_sq += float(np.sum(v_leaf.astype(np.float64) ** 2))
        raw_v.append(v_leaf)
      else:
        raw_v.append(np.zeros_like(arr))
    inv_norm = 1.0 / max(np.sqrt(norm_sq), 1e-12)
    unit_v = [v * np.float32(inv_norm) for v in raw_v]

    analytical = 0.0
    for g_leaf, v_leaf in zip(grad_leaves, unit_v):
      g_arr = np.asarray(g_leaf.get_value() if isinstance(g_leaf, nnx.Variable) else g_leaf, dtype=np.float64)
      analytical += float(np.sum(g_arr * v_leaf.astype(np.float64)))

    fd_estimates: dict[float, float] = {}
    abs_errors: dict[float, float] = {}
    rel_errors: dict[float, float] = {}

    for eps in epsilons:
      eps_f32 = np.float32(eps)

      def _perturb(leaf, v_leaf, sign, eps_val=eps_f32):
        if isinstance(leaf, nnx.Variable):
          return leaf.replace(value=leaf.get_value() + sign * eps_val * jnp.asarray(v_leaf, dtype=leaf.get_value().dtype))
        return leaf + sign * eps_val * jnp.asarray(v_leaf, dtype=leaf.dtype)

      params_pos = jax.tree_util.tree_unflatten(treedef, [_perturb(l, v, 1.0) for l, v in zip(leaves, unit_v)])
      params_neg = jax.tree_util.tree_unflatten(treedef, [_perturb(l, v, -1.0) for l, v in zip(leaves, unit_v)])

      f_pos = float(eval_fn(params_pos))
      f_neg = float(eval_fn(params_neg))
      fd_val = (f_pos - f_neg) / (2.0 * float(eps_f32))
      abs_err = abs(analytical - fd_val)
      denom = max(abs(analytical), abs(fd_val), 1e-12)
      rel_err = abs_err / denom

      fd_estimates[float(eps)] = fd_val
      abs_errors[float(eps)] = abs_err
      rel_errors[float(eps)] = rel_err

    best_eps = min(rel_errors, key=rel_errors.get)
    min_rel = rel_errors[best_eps]
    results.append(
        FDCheckDirectionResult(
            direction_idx=d_idx,
            analytical=analytical,
            fd_estimates=fd_estimates,
            abs_errors=abs_errors,
            rel_errors=rel_errors,
            min_rel_error=min_rel,
            best_eps=best_eps,
        )
    )

  max_min_rel = max((r.min_rel_error for r in results), default=0.0)
  return FDCheckReport(
      num_directions=num_directions,
      epsilons=tuple(float(e) for e in epsilons),
      directions=results,
      max_min_rel_error=max_min_rel,
      passed=max_min_rel <= rtol,
      rtol=rtol,
  )
