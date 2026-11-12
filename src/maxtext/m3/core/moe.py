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

"""Shared routed-expert execution, load-balancing statistics, and overlap hooks.

Model modules own router score computation (softmax, sigmoid, group-limited
top-k, routing bias, shared experts) and pass selected `(weights, indices)` to
a frozen `MoEExecutor` resolved from `(config, sharding)` at construction:

    self.executor = moe.moe_from_config(config, self.sharding)
    ...
    out = self.executor(x, weights, indices, wi_0, wi_1, wo)
    # or for fused (E, D, 2*M) / (E, D, 2, M) gate_up projections:
    out = self.executor.fused(x, weights, indices, gate_up_proj, down_proj)

Resolving `PartitionSpec` axes from `Sharding` at construction avoids reading
`jax.typeof(x).sharding` at trace time, which reports replicated `P(None, ...)`
on `Auto` mesh axes inside `jax.jit` and `ScannedLayers`. Because `MoEExecutor`
is a frozen dataclass of hashable primitives, Flax NNX treats it as a static
attribute across `nnx.split`, `nnx.merge`, `jax.jit`, and layer scan transforms.
"""

from collections.abc import Callable
import contextlib
import dataclasses
import functools
import math
from typing import Any, Literal

# pylint: disable=too-many-positional-arguments

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.m3.core import kernels
from maxtext.m3.core.sharding import MeshAxes, Sharding, TensorType, ambient_mesh_is_explicit, sharding_from_config

ExecutorKind = Literal["dense", "ragged_dot", "gmm_v2", "tokamax"]
SHARDED_EXECUTOR_KINDS: tuple[str, ...] = ("gmm_v2", "tokamax")

_COMBINE_DTYPE = jnp.float32
_TOKAMAX_M_ALIGN = 256
_TOKAMAX_KN_ALIGN = 128


def _align_up(value: int, multiple: int) -> int:
  """Rounds `value` up to a positive multiple of `multiple`."""
  return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


# ==============================================================================
# Permutation & grouped-matmul primitives
# ==============================================================================


@jax.custom_vjp
def _permute_rows(inputs: jax.Array, perm: jax.Array) -> jax.Array:
  """Permutes axis 0 by bijection `perm` with gather-only inverse VJP."""
  return inputs[perm]


def _permute_rows_fwd(inputs: jax.Array, perm: jax.Array) -> tuple[jax.Array, jax.Array]:
  return inputs[perm], perm


def _permute_rows_bwd(perm: jax.Array, grad: jax.Array) -> tuple[jax.Array, None]:
  return grad[jnp.argsort(perm)], None


_permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


def _matmul_precision(dtype: Any) -> jax.lax.Precision:
  """Returns single-pass MXU `DEFAULT` for `bfloat16` and `HIGHEST` otherwise."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16):
    return jax.lax.Precision.DEFAULT
  return jax.lax.Precision.HIGHEST


def _pallas_matmul_precision_context(dtype: Any) -> contextlib.AbstractContextManager[Any]:
  """Enables `float32` MXU passes inside Pallas `gmm_v2`/`tgmm_v2` on `float32` inputs."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
    return jax.default_matmul_precision("float32")
  return contextlib.nullcontext()


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _pallas_grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
) -> jax.Array:
  """Runs grouped matmul `lhs @ rhs[g]` on the TPU `gmm_v2` backend."""
  group_sizes_i32 = group_sizes.astype(jnp.int32)
  valid_count = jnp.sum(group_sizes_i32)
  valid_rows = (jnp.arange(lhs.shape[0], dtype=jnp.int32) < valid_count)[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend_kind == "gmm_v2":
    if not is_tpu:
      out = jax.lax.ragged_dot(
          lhs_safe,
          rhs,
          group_sizes_i32,
          precision=_matmul_precision(lhs.dtype),
          preferred_element_type=jnp.float32,
      ).astype(out_dtype)
      return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel

    m, k = lhs.shape
    _, _, n = rhs.shape
    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    with _pallas_matmul_precision_context(lhs.dtype):
      out_pad = gmm_v2.gmm_v2(
          lhs=lhs_pad,
          rhs=rhs_pad,
          group_sizes=group_sizes_i32,
          rhs_scale=None,
          tile_info=gmm_v2.calculate_tiling,
          preferred_element_type=out_dtype,
          maybe_quantize_lhs=False,
      )
    out = out_pad[:m, :n].astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  raise ValueError(f"Unsupported Pallas grouped matmul backend {backend_kind!r}.")


def _pallas_grouped_matmul_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  out = _pallas_grouped_matmul(lhs, rhs, group_sizes, backend_kind, is_tpu, out_dtype)
  return out, (lhs, rhs, group_sizes.astype(jnp.int32))


def _pallas_grouped_matmul_bwd(
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    grad: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  """Backward pass for the TPU `gmm_v2` / `tgmm_v2` grouped matmuls."""
  lhs, rhs, group_sizes = residuals
  m, k = lhs.shape
  num_groups, _, n = rhs.shape
  valid_count = jnp.sum(group_sizes)
  valid_rows = (jnp.arange(m, dtype=jnp.int32) < valid_count)[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend_kind == "gmm_v2":
    if not is_tpu:
      prec = _matmul_precision(lhs.dtype)
      _, vjp_fn = jax.vjp(
          lambda l, r: jax.lax.ragged_dot(
              l,
              r,
              group_sizes,
              precision=prec,
              preferred_element_type=jnp.float32,
          ).astype(out_dtype),
          lhs_safe,
          rhs,
      )
      grad_safe = jnp.where(valid_rows, grad.astype(out_dtype), jnp.zeros((), dtype=out_dtype))
      dlhs, drhs = vjp_fn(grad_safe)
      dlhs = jnp.where(valid_rows, dlhs, jnp.zeros((), dtype=lhs.dtype))
      return dlhs, drhs, None

    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel
    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2  # pylint: disable=import-outside-toplevel

    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    grad_safe = jnp.where(valid_rows, grad.astype(lhs.dtype), jnp.zeros((), dtype=lhs.dtype))
    grad_pad = jnp.pad(grad_safe, ((0, m_pad - m), (0, n_pad - n)))
    kernel_transpose_rhs = jax.dtypes.itemsize_bits(rhs_pad.dtype) >= 8
    dlhs_rhs = rhs_pad if kernel_transpose_rhs else rhs_pad.swapaxes(1, 2)
    with _pallas_matmul_precision_context(lhs.dtype):
      dlhs_pad = gmm_v2.gmm_v2(
          lhs=grad_pad,
          rhs=dlhs_rhs,
          group_sizes=group_sizes,
          rhs_scale=None,
          tile_info=gmm_v2.calculate_tiling,
          preferred_element_type=lhs_pad.dtype,
          maybe_quantize_lhs=False,
          transpose_rhs=kernel_transpose_rhs,
      )
      drhs_pad = tgmm_v2.tgmm_v2(
          lhs=lhs_pad,
          rhs=grad_pad,
          group_sizes=group_sizes,
          num_actual_groups=num_groups,
          rhs_scale=None,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=rhs_pad.dtype,
          tile_info=tgmm_v2.calculate_tgmm_tiling,
      )
    dlhs = jnp.where(valid_rows, dlhs_pad[:m, :k].astype(lhs.dtype), jnp.zeros((), dtype=lhs.dtype))
    return dlhs, drhs_pad[:, :k, :n].astype(rhs.dtype), None

  raise ValueError(f"Unsupported Pallas grouped matmul backend {backend_kind!r}.")


_pallas_grouped_matmul.defvjp(_pallas_grouped_matmul_fwd, _pallas_grouped_matmul_bwd)


def grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    backend: str = "ragged_dot",
    is_tpu: bool = False,
    preferred_element_type: Any = None,
) -> jax.Array:
  """Grouped matmul dispatcher across `ragged_dot`, `gmm_v2`, and `tokamax`."""
  if backend not in ("ragged_dot", "gmm_v2", "tokamax"):
    raise ValueError(f"Unknown grouped matmul backend {backend!r}; expected 'ragged_dot', 'gmm_v2', or 'tokamax'.")

  out_dtype = jnp.dtype(lhs.dtype if preferred_element_type is None else preferred_element_type)
  group_sizes_i32 = group_sizes.astype(jnp.int32)
  valid_rows = (jnp.arange(lhs.shape[0], dtype=jnp.int32) < jnp.sum(group_sizes_i32))[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend == "ragged_dot" or not is_tpu:
    out = jax.lax.ragged_dot(
        lhs_safe,
        rhs,
        group_sizes_i32,
        precision=_matmul_precision(lhs.dtype),
        preferred_element_type=jnp.float32,
    ).astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  if backend == "tokamax":
    import tokamax  # pylint: disable=import-outside-toplevel

    m, k = lhs.shape
    _, _, n = rhs.shape
    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    out_pad = tokamax.ragged_dot(
        lhs=lhs_pad,
        rhs=rhs_pad,
        group_sizes=tokamax.RaggedDotGroupSizes(group_sizes_i32, m_pad),
        preferred_element_type=out_dtype,
        precision=_matmul_precision(lhs.dtype),
        implementation="mosaic",
    )
    out = out_pad[:m, :n].astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  return _pallas_grouped_matmul(lhs, rhs, group_sizes_i32, backend, is_tpu, out_dtype)


# ==============================================================================
# Executor descriptor & config resolution
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class MoEExecutor:
  """Static routed-expert execution descriptor resolved from `(config, sharding)`."""

  kind: ExecutorKind = "dense"
  batch: MeshAxes = None
  experts: MeshAxes = None
  mlp: MeshAxes = None
  capacity_factor: float = -1.0
  ragged_buffer_factor: float = -1.0
  use_ragged_sort: bool = False
  length: MeshAxes = None

  def __call__(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array,
      wi_0: jax.Array,
      wi_1: jax.Array,
      wo: jax.Array,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
  ) -> jax.Array:
    """Executes routed experts with separate `(wi_0, wi_1, wo)` projections."""
    return routed_experts(
        x,
        weights,
        indices,
        wi_0,
        wi_1,
        wo,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

  def fused(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array,
      gate_up_proj: jax.Array,
      down_proj: jax.Array,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
  ) -> jax.Array:
    """Executes routed experts with 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
    return routed_experts_fused(
        x,
        weights,
        indices,
        gate_up_proj,
        down_proj,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


DENSE = MoEExecutor("dense")
RAGGED_DOT = MoEExecutor("ragged_dot")


_EP_PARALLELISM_ATTRS: tuple[str, ...] = ("ici_expert_parallelism", "dcn_expert_parallelism")
_TP_PARALLELISM_ATTRS: tuple[str, ...] = (
    "ici_tensor_parallelism",
    "dcn_tensor_parallelism",
    "ici_tensor_sequence_parallelism",
    "dcn_tensor_sequence_parallelism",
)

_UNSUPPORTED_BOOL_OFF_MOE_FIELDS: tuple[str, ...] = (
    "use_random_routing",
    "use_te_grouped_gemm",
    "te_moe_block",
    "te_moe_disable_cudnn_gmm_fusion",
    "te_moe_quantize_before_fsdp_all_gather",
    "fp8_moe",
    "fp8_moe_fake_quant",
    "moe_flat_fsdp_weights",
    "shard_exp_on_fsdp",
    "shard_embed_moe_on_fsdp",
    "use_2d_fsdp_sharding",
    "moe_fsdp_use_two_stage_all_gather",
    "moe_wo_fsdp_on_mlp",
    "moe_use_fused_gmm3d",
    "moe_dispatch_no_expert_sharding",
    "moe_use_megatron_seq_aux_loss",
    "moe_exclude_padding_from_routing",
    "float32_gate_logits",
    "merge_gating_gmm",
    "moe_pin_sparse_core_all_gathers",
    "moe_pin_sparse_core_ep_all_gathers",
    "moe_pin_sparse_core_fsdp_all_gathers",
    "moe_pin_sparse_core_fsdp_all_gathers_fwd_only",
    "moe_quantize_token_all_gather",
    "moe_accumulate_chunk_wgrad",
    "moe_accumulate_wi_dlhs",
    "moe_gmm_v2_dlhs_transpose_rhs",
    "moe_gmm_v2_tail_chunks",
    "moe_bwd_prequant_before_unsort",
    "moe_combine_bwd_direct_qarray",
    "moe_3d_dlhs_partial_sum",
    "moe_chunk_barrier",
    "moe_log_max_load_ratio",
    "log_required_ragged_buffer_factor",
    "router_topk_matmul_vjp",
    "moe_topk_before_ep_all_gather",
    "moe_tc_ragged_sort",
    "moe_tc_ragged_weights_on_activation",
    "moe_tc_ragged_3d_gmm",
    "moe_tc_ragged_3d_dispatch",
    "moe_tc_window_routing",
    "moe_use_direct_token_gather",
    "moe_packed_routing_sort",
    "moe_tc_routing_offload_free",
    "moe_unsliced_prefused_wi",
    "prefuse_moe_weights",
    "moe_tc_combine",
    "moe_tc_permute",
    "use_gather_mosaic_kernel",
    "ragged_sort_use_single_sparsecore",
    "ragged_unsort_spread_masked_indices",
    "ragged_gather_fallback",
    "ragged_gather_reduce_fallback",
    "use_gmm_v2_heuristic_tiling",
    "defer_small_all_reduces",
    "use_batch_split_schedule",
    "use_manual_quantization",
    "moe_fused_gmm3d_prequant_dy",
    "moe_fused_gmm3d_early_dwo_rs",
    "moe_fused_gmm3d_sc_routed_gather",
    "moe_fused_gmm3d_exact_topk_combine",
    "moe_fused_gmm3d_exact_topk_out_2d",
    "moe_fused_gmm3d_inv_perm_combine",
    "moe_fused_gmm3d_relayout_x_2d_to_3d",
    "moe_fused_gmm3d_fused_quant_relayout",
    "moe_fused_gmm3d_overlap_dcoeffs_scatter",
)

_UNSUPPORTED_BOOL_ON_MOE_FIELDS: tuple[str, ...] = (
    "float32_weight_sum",
    "use_custom_sort_vjp",
    "moe_tc_ragged_mask_padding",
    "moe_fused_gmm3d_gmm_combine",
    "moe_fused_gmm3d_split_gate_bwd",
    "moe_fused_gmm3d_split_dw_gate",
)

_UNSUPPORTED_EMPTY_STR_MOE_FIELDS: tuple[str, ...] = (
    "te_gmm_quantization",
    "moe_quantize_combine_bwd_method",
    "moe_quantize_dispatch_fwd_method",
    "moe_quantize_combine_fwd_method",
    "moe_quantize_dispatch_bwd_method",
)

_UNSUPPORTED_ENUM_STR_MOE_FIELDS: dict[str, str] = {
    "moe_weight_quantization_granularity": "per_channel",
    "moe_routing_index_math": "sort",
}

_UNSUPPORTED_FLOAT_MOE_FIELDS: dict[str, float] = {
    "eval_ragged_buffer_factor": -1.0,
    "first_phase_ragged_buffer_factor": 0.0,
}

_UNSUPPORTED_INT_MOE_FIELDS: dict[str, int] = {
    "te_ep_overflow_check_every_n_steps": 20,
    "num_moe_token_chunks": 1,
    "num_moe_emb_chunks": 0,
    "expert_replicas_per_device": 0,
    "expert_replica_max_send": 4,
    "retry_dropless_first_steps": 0,
    "batch_split_factor": 1,
    "moe_wi_ag_latency_ns": 0,
    "moe_fsdp_all_gather_sparse_core_id": 0,
    "moe_ep_all_gather_sparse_core_id": 1,
    "moe_tc_ragged_gather_block_size": 1024,
    "moe_tc_ragged_reduce_block_size": 896,
    "moe_tc_ragged_flatten_block_size": 0,
    "moe_tc_combine_block_tokens": 256,
    "moe_tc_combine_expert_chunks": 1,
    "moe_tc_combine_col_chunks": 1,
    "moe_tc_permute_block_tokens": 512,
    "moe_tc_permute_expert_chunks": 1,
    "moe_tc_permute_col_chunks": 1,
    "moe_fused_gmm3d_gate_tile_m": 768,
    "moe_fused_gmm3d_gate_tile_k": 1024,
    "moe_fused_gmm3d_dgrad_tile_rows": 256,
    "moe_fused_gmm3d_dgrad_bucket_rows": 128,
    "moe_fused_gmm3d_dgrad_tile_n": 512,
    "moe_fused_gmm3d_dx_tile_m": 512,
    "moe_fused_gmm3d_dx_tile_k": 3072,
    "moe_fused_gmm3d_dx_tile_n": 2048,
    "moe_fused_gmm3d_dw_tile_k": 1024,
    "moe_fused_gmm3d_dgrad_tile_k": 1024,
    "moe_fused_gmm3d_wgrad_schedule_mode": 0,
    "ragged_gather_cost_estimate_flops": -1,
    "ragged_gather_reduce_cost_estimate_flops": -1,
    "ragged_gather_cost_estimate_bytes_accessed": -1,
    "ragged_gather_reduce_cost_estimate_bytes_accessed": -1,
    "wi_tile_fwd_batch_seq": 512,
    "wi_tile_fwd_embed_dim": 1024,
    "wi_tile_fwd_mlp_dim": 1024,
    "eval_wi_tile_fwd_batch_seq": 512,
    "eval_wi_tile_fwd_embed_dim": 1024,
    "eval_wi_tile_fwd_mlp_dim": 1024,
    "wi_tile_dlhs_batch_seq": 512,
    "wi_tile_dlhs_embed_dim": 1024,
    "wi_tile_dlhs_mlp_dim": 1024,
    "wi_tile_drhs_batch_seq": 512,
    "wi_tile_drhs_embed_dim": 1024,
    "wi_tile_drhs_mlp_dim": 1024,
    "wo_tile_fwd_batch_seq": 512,
    "wo_tile_fwd_embed_dim": 1024,
    "wo_tile_fwd_mlp_dim": 1024,
    "eval_wo_tile_fwd_batch_seq": 512,
    "eval_wo_tile_fwd_embed_dim": 1024,
    "eval_wo_tile_fwd_mlp_dim": 1024,
    "wo_tile_dlhs_batch_seq": 512,
    "wo_tile_dlhs_embed_dim": 1024,
    "wo_tile_dlhs_mlp_dim": 1024,
    "wo_tile_drhs_batch_seq": 512,
    "wo_tile_drhs_embed_dim": 1024,
    "wo_tile_drhs_mlp_dim": 1024,
}


def _parallelism_product(config: Any, *attrs: str) -> int:
  """Returns the product of positive (> 1) parallelism attributes on `config`."""
  size = 1
  for attr in attrs:
    val = int(getattr(config, attr, 1) or 1)
    if val > 1:
      size *= val
  return size


def _has_auto_parallelism(config: Any, *attrs: str) -> bool:
  """Returns whether any parallelism attribute in `attrs` uses a negative auto-fill sentinel (`-1`)."""
  for attr in attrs:
    raw = getattr(config, attr, 1)
    if raw is not None and int(raw) < 0:
      return True
  return False


def _active_mesh_or_none(mesh: Any = None) -> Any | None:
  """Returns `mesh` if provided and non-empty, else the active abstract or concrete JAX mesh."""
  if mesh is not None and not getattr(mesh, "empty", False):
    return mesh
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty:
    return abstract_mesh
  try:
    concrete_mesh = jax.sharding.get_mesh()
  except ValueError:
    concrete_mesh = None
  if concrete_mesh is not None and not getattr(concrete_mesh, "empty", False):
    return concrete_mesh
  return None


def _is_moe_dropless_fallback_set(val: Any) -> bool:
  """Returns whether `moe_dropless_fallback` is set to a non-default active mode."""
  if val is None:
    return False
  if isinstance(val, str):
    return val.strip().lower() not in ("", "none")
  return True


def unsupported_moe_settings(config: Any) -> dict[str, Any]:
  """Returns legacy MoE knobs set to non-default values on `config` that m3 does not implement."""
  unsupported: dict[str, Any] = {}
  for field in _UNSUPPORTED_BOOL_OFF_MOE_FIELDS:
    if bool(getattr(config, field, False)):
      unsupported[field] = getattr(config, field)
  for field in _UNSUPPORTED_BOOL_ON_MOE_FIELDS:
    val = getattr(config, field, True)
    if val is not None and not bool(val):
      unsupported[field] = val
  for field in _UNSUPPORTED_EMPTY_STR_MOE_FIELDS:
    val = getattr(config, field, "")
    if _is_moe_dropless_fallback_set(val):
      unsupported[field] = val
  for field, default_str in _UNSUPPORTED_ENUM_STR_MOE_FIELDS.items():
    val = getattr(config, field, default_str)
    if val is not None and (not isinstance(val, str) or val.strip().lower() != default_str):
      unsupported[field] = val
  for field, default_flt in _UNSUPPORTED_FLOAT_MOE_FIELDS.items():
    val = getattr(config, field, default_flt)
    if val is not None and (isinstance(val, bool) or float(val) != default_flt):
      unsupported[field] = val
  for field, default_int in _UNSUPPORTED_INT_MOE_FIELDS.items():
    val = getattr(config, field, default_int)
    if val is not None and (isinstance(val, bool) or val != default_int):
      unsupported[field] = val
  fallback = getattr(config, "moe_dropless_fallback", None)
  if _is_moe_dropless_fallback_set(fallback):
    unsupported["moe_dropless_fallback"] = fallback
  return unsupported


def _resolve_executor_kind(config: Any) -> ExecutorKind:
  """Resolves `MoEExecutor.kind` following legacy MoE backend flag precedence."""
  # Legacy precedence (`layers/moe.py:3098-3189`, `megablox/ops.py:320-346`, `configs/types.py:7956`):
  # `sparse_matmul=False` runs dense einsum; `use_gmm_v2` requires `use_tokamax_gmm`; `use_tokamax_gmm=True`
  # selects `"gmm_v2"` when `use_gmm_v2=True` else `"tokamax"` (even if `megablox=True`); `megablox=True` without
  # `use_tokamax_gmm` selects the legacy Megablox kernel, which m3 refuses rather than remapping to `gmm_v2`.
  sparse_matmul = bool(getattr(config, "sparse_matmul", True))
  use_tokamax_gmm = bool(getattr(config, "use_tokamax_gmm", False))
  use_gmm_v2 = bool(getattr(config, "use_gmm_v2", False))
  megablox = bool(getattr(config, "megablox", False))

  if use_gmm_v2 and not use_tokamax_gmm:
    raise ValueError("use_gmm_v2=True requires use_tokamax_gmm=True.")
  if not sparse_matmul:
    return "dense"
  if use_tokamax_gmm and use_gmm_v2:
    return "gmm_v2"
  if use_tokamax_gmm:
    return "tokamax"
  if megablox:
    raise ValueError(
        "m3 MoE does not implement the legacy Megablox kernel (megablox=True with use_tokamax_gmm=False). "
        "Set use_tokamax_gmm=True and use_gmm_v2=True for Tokamax GMM v2 (sharded EP/TP on TPU), "
        "use_tokamax_gmm=True and use_gmm_v2=False for Tokamax v1 ragged_dot, or "
        "megablox=False for jax.lax.ragged_dot (EP=1, TP=1)."
    )
  return "ragged_dot"


def _require_sharded_gmm_for_ep_or_tp(
    config: Any,
    *,
    mesh: Any = None,
    sharding: Sharding | None = None,
) -> None:
  """Rejects unsharded `ragged_dot` when EP or TP exceeds 1, uses `-1` auto-fill, or shards in the active mesh."""
  ep_cfg = _parallelism_product(config, *_EP_PARALLELISM_ATTRS)
  tp_cfg = _parallelism_product(config, *_TP_PARALLELISM_ATTRS)
  ep_auto = _has_auto_parallelism(config, *_EP_PARALLELISM_ATTRS)
  tp_auto = _has_auto_parallelism(config, *_TP_PARALLELISM_ATTRS)

  mesh_ep = 1
  mesh_tp = 1
  active_mesh = _active_mesh_or_none(mesh)
  if active_mesh is not None:
    shape = getattr(active_mesh, "shape", {})
    mesh_ep = max(mesh_ep, int(shape.get("expert", 1) or 1))
    mesh_tp = max(
        mesh_tp,
        int(shape.get("tensor", 1) or 1) * int(shape.get("tensor_sequence", 1) or 1),
    )
    if sharding is not None:
      wi_spec = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
      mesh_ep = max(mesh_ep, kernels.mesh_axis_size(active_mesh, wi_spec[0]))
      mesh_tp = max(mesh_tp, kernels.mesh_axis_size(active_mesh, wi_spec[2]))

  if ep_cfg > 1 or tp_cfg > 1 or ep_auto or tp_auto or mesh_ep > 1 or mesh_tp > 1:
    raise ValueError(
        "Expert parallelism (expert > 1 or -1 auto-fill) or tensor parallelism (tensor > 1 or -1 auto-fill) "
        "with sparse_matmul=True requires a sharded GMM backend (gmm_v2 or tokamax) so tokens dispatch via "
        "shard_map without all-gathering expert weights."
    )


def _validate_executor_backend_config(config: Any, *, mesh: Any = None) -> ExecutorKind:
  """Validates resolved MoE backend flags against `use_ragged_sort` and EP/TP."""
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  kind = _resolve_executor_kind(config)
  if use_ragged_sort and kind not in SHARDED_EXECUTOR_KINDS:
    raise ValueError("use_ragged_sort=True requires sparse_matmul=True with a sharded GMM backend (gmm_v2 or tokamax).")
  if kind == "ragged_dot":
    _require_sharded_gmm_for_ep_or_tp(config, mesh=mesh)
  return kind


def validate_moe_config(
    config: Any,
    *,
    check_executor_backend: bool = True,
    require_moe: bool = False,
    mesh: Any = None,
) -> None:
  """Validates m3 MoE configuration fields, failing closed on unsupported modes."""
  num_experts = int(getattr(config, "num_experts", 1) or 1)
  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  use_ring_of_experts = bool(getattr(config, "use_ring_of_experts", False))

  if capacity_factor == 0.0:
    raise ValueError("capacity_factor cannot be 0.0; set < 0 for dropless routing or > 0 for capacity dropping.")
  if ragged_buffer_factor == 0.0:
    raise ValueError("ragged_buffer_factor cannot be 0.0; set < 0 for worst-case buffer or > 0 to bound buffer size.")

  unsupported = unsupported_moe_settings(config)
  if unsupported:
    raise ValueError(f"m3 MoE does not support {unsupported} (see m3/README.md).")

  if bool(getattr(config, "use_gmm_v2", False)) and not bool(getattr(config, "use_tokamax_gmm", False)):
    raise ValueError("use_gmm_v2=True requires use_tokamax_gmm=True.")

  if not require_moe and num_experts <= 1 and not use_ragged_sort and not use_ring_of_experts:
    return

  if use_ring_of_experts:
    raise NotImplementedError(
        "m3 MoE uses ragged_all_to_all expert parallelism; use_ring_of_experts=True is not supported."
    )
  if not check_executor_backend:
    unwired = {
        field: getattr(config, field)
        for field in ("use_tokamax_gmm", "use_gmm_v2", "use_ragged_sort")
        if bool(getattr(config, field, False))
    }
    if unwired:
      raise ValueError(
          f"Model {getattr(config, 'model_name', '<unknown>')!r} does not use MoEExecutor yet; "
          f"m3 cannot apply {unwired}."
      )
    return

  _validate_executor_backend_config(config, mesh=mesh)


def moe_from_config(config: Any, sharding: Sharding | None = None) -> MoEExecutor:
  """Resolves a static `MoEExecutor` from `config` and `sharding`."""
  active_mesh = _active_mesh_or_none()
  validate_moe_config(config, require_moe=True, mesh=active_mesh)
  sharding = sharding or sharding_from_config(config)

  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))

  cap = capacity_factor if capacity_factor > 0.0 else -1.0
  rbf = ragged_buffer_factor if ragged_buffer_factor > 0.0 else -1.0

  kind = _resolve_executor_kind(config)
  if kind == "dense":
    return MoEExecutor(
        kind="dense",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
    )

  if kind == "ragged_dot":
    _require_sharded_gmm_for_ep_or_tp(config, mesh=active_mesh, sharding=sharding)
    return MoEExecutor(
        kind="ragged_dot",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
    )

  tokens_spec = sharding("moe_tokens", ("batch", "length", "embed"), TensorType.ACTIVATION)
  wi_spec = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
  return MoEExecutor(
      kind=kind,
      batch=tokens_spec[0],
      experts=wi_spec[0],
      mlp=wi_spec[2],
      capacity_factor=cap,
      ragged_buffer_factor=rbf,
      use_ragged_sort=use_ragged_sort,
      length=tokens_spec[1],
  )


# ==============================================================================
# Load-balancing loss & dense combine helpers
# ==============================================================================


def combine_weights(weights: jax.Array, indices: jax.Array, num_experts: int) -> jax.Array:
  """Projects per-token top-k weights into a dense `(batch, length, num_experts)` table."""
  valid = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid, indices, 0)
  safe_weights = jnp.where(valid, weights, jnp.zeros_like(weights))
  return jnp.sum(jax.nn.one_hot(safe_indices, num_experts, dtype=weights.dtype) * safe_weights[..., None], axis=-2)


def load_balance_loss(
    indices: jax.Array,
    probs: jax.Array,
    *,
    weight: float = 1.0,
    mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Switch-Transformer load-balance loss masking padding and binarizing `segment_ids`."""
  if probs.ndim != 3 or indices.ndim != 3 or probs.shape[:2] != indices.shape[:2]:
    raise ValueError(
        f"Expected 3D indices (B, S, K) and probs (B, S, E) with matching (B, S); "
        f"got indices={indices.shape}, probs={probs.shape}."
    )
  batch, length, num_experts = probs.shape
  top_k = indices.shape[-1]
  probs_f32 = probs.astype(jnp.float32)

  valid_slots = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid_slots, indices, 0)
  expert_mask = jax.nn.one_hot(safe_indices, num_experts, dtype=jnp.float32) * valid_slots[..., None].astype(jnp.float32)

  token_valid: jax.Array | None = None
  if mask is not None:
    if mask.shape != (batch, length):
      raise ValueError(f"mask shape {mask.shape} must match (batch, length)=({batch}, {length}).")
    token_valid = mask != 0
  if segment_ids is not None:
    if segment_ids.shape != (batch, length):
      raise ValueError(f"segment_ids shape {segment_ids.shape} must match (batch, length)=({batch}, {length}).")
    seg_valid = segment_ids != 0
    token_valid = seg_valid if token_valid is None else (token_valid & seg_valid)

  if token_valid is None:
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / top_k
    density_prob = jnp.mean(probs_f32, axis=1)
    return (jnp.mean(density * density_prob) * (num_experts**2) * float(weight)).astype(jnp.float32)

  m = token_valid.astype(jnp.float32)
  seq_counts = jnp.sum(m, axis=1, keepdims=True)
  safe_counts = jnp.maximum(seq_counts, 1.0)
  seq_active = (seq_counts[:, 0] > 0).astype(jnp.float32)

  masked_expert_mask = expert_mask * m[:, :, None, None]
  density = jnp.sum(masked_expert_mask, axis=(1, 2)) / (safe_counts * top_k)
  density_prob = jnp.sum(probs_f32 * m[:, :, None], axis=1) / safe_counts
  per_seq = jnp.mean(density * density_prob, axis=-1)
  active_denom = jnp.maximum(jnp.sum(seq_active), 1.0)
  return ((jnp.sum(per_seq * seq_active) / active_denom) * (num_experts**2) * float(weight)).astype(jnp.float32)


# ==============================================================================
# Expert MLP & EP dispatch/combine execution
# ==============================================================================


def _apply_gated_activation(
    gate: jax.Array,
    up: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any,
) -> jax.Array:
  """Computes `activation(gate) * up` with optional symmetric/upper SwiGLU clipping."""
  if swiglu_limit is not None and swiglu_limit > 0:
    limit = jnp.asarray(swiglu_limit, dtype=gate.dtype)
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
  return (activation(gate) * up).astype(out_dtype)


def _truncate_slots_by_capacity(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    expert_cap: int,
    prior_expert_counts: jax.Array | None = None,
) -> jax.Array:
  """Drops slots whose per-expert token-major rank exceeds `expert_cap` in O(M) memory."""
  num_slots = safe_idx.shape[0]
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  counts = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
  starts = jnp.cumsum(counts) - counts
  sorted_exp = safe_idx[order]
  rank_in_exp = jnp.arange(num_slots, dtype=jnp.int32) - starts[sorted_exp]
  if prior_expert_counts is not None:
    rank_in_exp = rank_in_exp + prior_expert_counts[sorted_exp]
  keep_sorted = slot_valid[order] & (rank_in_exp < int(expert_cap))
  return keep_sorted[jnp.argsort(order)]


def _truncate_slots_by_buffer(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    buf_cap: int,
) -> jax.Array:
  """Keeps at most `buf_cap` valid slots in expert-sorted order."""
  num_slots = safe_idx.shape[0]
  if buf_cap >= num_slots:
    return slot_valid
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  keep_sorted = slot_valid[order] & (jnp.arange(num_slots, dtype=jnp.int32) < int(buf_cap))
  return keep_sorted[jnp.argsort(order)]


def _ep_ragged_exchange(
    operand: jax.Array,
    out_len: int,
    counts_matrix: jax.Array,
    ep_id: jax.Array,
    ep_axis: str | tuple[str, ...],
    *,
    is_tpu: bool,
) -> jax.Array:
  """Exchanges variable-length token slices across `ep_axis`.

  `counts_matrix[src, dst]` gives the number of rows sent from `src` to `dst`.
  On TPU, dispatches directly to `jax.lax.ragged_all_to_all` using sender-side
  destination offsets. On CPU (where XLA's CPU thunk emitter lacks
  `ragged-all-to-all`), computes the bit-identical `[out_len, d_dim]` buffer via
  token-only `jax.lax.all_to_all`.
  """
  ep_size = counts_matrix.shape[0]
  d_dim = operand.shape[1]
  send_sz = counts_matrix[ep_id]
  in_off = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(send_sz)[:-1]], axis=0)
  out_off_all = jnp.cumsum(
      jnp.concatenate([jnp.zeros((1, ep_size), dtype=jnp.int32), counts_matrix], axis=0),
      axis=0,
  )[:-1]
  out_off = out_off_all[ep_id]
  recv_sz = counts_matrix[:, ep_id]

  if is_tpu:
    out_buf = jnp.zeros((out_len, d_dim), dtype=operand.dtype)
    return jax.lax.ragged_all_to_all(
        operand,
        out_buf,
        in_off,
        send_sz,
        out_off,
        recv_sz,
        axis_name=ep_axis,
    )

  in_len = operand.shape[0]
  cap = max(1, min(int(in_len), int(out_len)))
  col = jnp.arange(cap, dtype=jnp.int32)[None, :]
  gather_idx = in_off[:, None] + col
  valid_send = col < send_sz[:, None]
  safe_gather = jnp.clip(gather_idx, 0, max(int(in_len) - 1, 0))
  staged = jnp.where(valid_send[:, :, None], operand[safe_gather], jnp.zeros((), dtype=operand.dtype))
  recvd = jax.lax.all_to_all(staged, ep_axis, split_axis=0, concat_axis=0)

  recv_ends = jnp.cumsum(recv_sz)
  recv_starts = recv_ends - recv_sz
  row_idx = jnp.arange(out_len, dtype=jnp.int32)
  src_idx = jnp.searchsorted(recv_ends, row_idx, side="right")
  valid_row = row_idx < recv_ends[-1]
  safe_src = jnp.minimum(src_idx, ep_size - 1)
  pos_in_src = jnp.clip(row_idx - recv_starts[safe_src], 0, cap - 1)
  return jnp.where(valid_row[:, None], recvd[safe_src, pos_in_src], jnp.zeros((), dtype=operand.dtype))


def _dense_separate(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Runs every expert on every token and combines selected expert outputs."""
  if capacity_factor > 0.0:
    batch, length, _ = x.shape
    top_k = indices.shape[-1]
    num_tokens = batch * length
    flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)
    cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
    eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
    eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
  else:
    eff_indices = indices
    eff_weights = weights
  table = combine_weights(eff_weights, eff_indices, num_experts)
  prec = _matmul_precision(x.dtype)
  gate = jnp.einsum("bse,neh->bsnh", x, wi_0, precision=prec)
  up = jnp.einsum("bse,neh->bsnh", x, wi_1, precision=prec)
  hidden = _apply_gated_activation(gate, up, activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype)
  weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(wo.dtype)
  return jnp.einsum("bsnh,nhe->bse", weighted_hidden, wo, precision=prec, preferred_element_type=_COMBINE_DTYPE)


def _dense_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Dense reference execution for 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim == 4:
    if capacity_factor > 0.0:
      batch, length, _ = x.shape
      top_k = indices.shape[-1]
      num_tokens = batch * length
      flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
      slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
      safe_idx = jnp.where(slot_valid, flat_idx, 0)
      cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
      slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
      eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
      eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
    else:
      eff_indices = indices
      eff_weights = weights
    table = combine_weights(eff_weights, eff_indices, num_experts)
    prec = _matmul_precision(x.dtype)
    gu = jnp.einsum("bse,negh->bsngh", x, gate_up_proj, precision=prec)
    hidden = _apply_gated_activation(
        gu[..., 0, :], gu[..., 1, :], activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype
    )
    weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(down_proj.dtype)
    return jnp.einsum("bsnh,nhe->bse", weighted_hidden, down_proj, precision=prec, preferred_element_type=_COMBINE_DTYPE)
  wi_0, wi_1 = jnp.split(gate_up_proj, 2, axis=-1)
  return _dense_separate(
      x,
      weights,
      indices,
      wi_0,
      wi_1,
      down_proj,
      num_experts=num_experts,
      activation=activation,
      swiglu_limit=swiglu_limit,
      capacity_factor=capacity_factor,
  )


def _run_local_expert_mlp(
    sorted_x: jax.Array,
    projections: tuple[jax.Array, ...],
    group_sizes: jax.Array,
    *,
    fused: bool,
    matmul_fn: Callable[..., jax.Array],
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any = None,
) -> jax.Array:
  """Runs gate/up + activation + down grouped matmuls on sorted rows."""
  if fused:
    w_gu, w_dn = projections
    gate_up = matmul_fn(sorted_x, w_gu, group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    hidden = _apply_gated_activation(
        gate,
        up,
        activation=activation,
        swiglu_limit=swiglu_limit,
        out_dtype=sorted_x.dtype,
    )
    return matmul_fn(hidden, w_dn, group_sizes, preferred_element_type=out_dtype)

  wi_0, wi_1, wo = projections
  gate = matmul_fn(sorted_x, wi_0, group_sizes)
  up = matmul_fn(sorted_x, wi_1, group_sizes)
  hidden = _apply_gated_activation(
      gate,
      up,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=sorted_x.dtype,
  )
  return matmul_fn(hidden, wo, group_sizes, preferred_element_type=out_dtype)


def _grouped_unsharded(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
) -> jax.Array:
  """Portable `ragged_dot` grouped execution without `shard_map`."""
  if fused and projections[0].ndim == 4:
    w_gu_4d, w_dn = projections
    projections = (
        jnp.reshape(w_gu_4d, (w_gu_4d.shape[0], w_gu_4d.shape[1], 2 * w_gu_4d.shape[3])),
        w_dn,
    )

  batch, length, embed = x.shape
  top_k = indices.shape[-1]
  num_tokens = batch * length
  num_slots = num_tokens * top_k

  flat_idx = jnp.reshape(indices, (num_slots,)).astype(jnp.int32)
  flat_w = jnp.reshape(weights, (num_slots,))
  slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
  safe_idx = jnp.where(slot_valid, flat_idx, 0)

  if executor.capacity_factor > 0.0:
    cap = max(1, int(math.ceil((num_slots / num_experts) * executor.capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
  if executor.ragged_buffer_factor > 0.0:
    buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
    slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)

  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  perm = jnp.argsort(sort_key, stable=True)
  inv_perm = jnp.argsort(perm)
  group_sizes = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
  valid_count = jnp.sum(group_sizes)

  slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
  slot_rows = jnp.reshape(x, (num_tokens, embed))[slot_token_ids]
  sorted_x = _permute_rows(slot_rows, perm)
  sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)

  matmul_fn = functools.partial(grouped_matmul, backend="ragged_dot", is_tpu=False)
  expert_out = _run_local_expert_mlp(
      sorted_x,
      projections,
      group_sizes,
      fused=fused,
      matmul_fn=matmul_fn,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=_COMBINE_DTYPE,
  )
  unsorted_out = _permute_rows(expert_out, inv_perm)
  eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
  safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
  weighted = safe_out * eff_w[:, None]
  combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, embed)), axis=1)
  return jnp.reshape(combined, (batch, length, embed))


def _can_use_sparsecore_ragged_sort(num_rows: int, hidden_size: int, dtype: Any) -> bool:
  """Returns True when SparseCore `ragged_sort` kernels support `(num_rows, hidden_size, dtype)` on the active TPU."""
  dt = jnp.dtype(dtype)
  if dt not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
    return False
  if num_rows <= 0 or hidden_size <= 0:
    return False
  if (num_rows * jax.dtypes.itemsize_bits(dt)) % 32 != 0:
    return False
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=import-outside-toplevel

  if not pltpu.is_tpu_device():
    return False
  try:
    tpu_info = pltpu.get_tpu_info()
  except ValueError:
    return False
  sc_info = tpu_info.sparse_core
  if sc_info is None:
    return False
  num_lanes = int(tpu_info.num_lanes)
  if hidden_size % num_lanes != 0:
    return False
  num_cores = int(sc_info.num_cores) * int(sc_info.num_subcores)
  num_simd_lanes = int(sc_info.num_lanes)
  preferred_num_stages = 4
  num_col_parts = 1
  while num_cores // num_col_parts > num_simd_lanes:
    next_col_parts = num_col_parts * 2
    if num_cores % next_col_parts != 0:
      break
    if hidden_size % (num_lanes * next_col_parts) != 0:
      break
    if hidden_size // (num_lanes * next_col_parts) < preferred_num_stages:
      break
    num_col_parts = next_col_parts
  return (num_cores // num_col_parts) <= num_simd_lanes


def _sharded_grouped(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
) -> jax.Array:
  """Distributed `shard_map` execution supporting `EP >= 1`, `TP >= 1`, and `FSDP >= 1`."""
  mesh = kernels.mesh_in_scope()
  is_tpu = kernels.is_tpu_mesh(mesh)

  batch_ax = kernels.resolve_mesh_axes(mesh, executor.batch)
  experts_ax = kernels.resolve_mesh_axes(mesh, executor.experts)
  mlp_ax = kernels.resolve_mesh_axes(mesh, executor.mlp)

  batch_shards = kernels.mesh_axis_size(mesh, batch_ax)
  ep_size = kernels.mesh_axis_size(mesh, experts_ax)
  tp_size = kernels.mesh_axis_size(mesh, mlp_ax)
  batch_names = kernels.mesh_axis_names(batch_ax)
  ep_axes = kernels.mesh_axis_names(experts_ax)
  tp_axes = kernels.mesh_axis_names(mlp_ax)
  extra_ep = tuple(ax for ax in ep_axes if ax not in batch_names)
  extra_ep_size = kernels.mesh_axis_size(mesh, extra_ep) if extra_ep else 1
  raw_length_ax = kernels.resolve_mesh_axes(mesh, executor.length)
  non_ep_claimed_axes = set(batch_names) | set(tp_axes)
  expected_length_axes = [
      ax for ax in kernels.mesh_axis_names(raw_length_ax) if ax not in non_ep_claimed_axes and int(mesh.shape[ax]) > 1
  ]
  expected_cand_length_ax: MeshAxes = (
      expected_length_axes[0]
      if len(expected_length_axes) == 1
      else (tuple(expected_length_axes) if expected_length_axes else None)
  )
  expected_length_shards = kernels.mesh_axis_size(mesh, expected_cand_length_ax)
  expected_length_ax = (
      expected_cand_length_ax
      if (expected_length_shards > 1 and x.shape[1] > 1 and x.shape[1] % expected_length_shards == 0)
      else None
  )

  claimed_axes = non_ep_claimed_axes | set(ep_axes)
  cand_length_axes = [
      ax for ax in kernels.mesh_axis_names(raw_length_ax) if ax not in claimed_axes and int(mesh.shape[ax]) > 1
  ]
  cand_length_ax: MeshAxes = (
      cand_length_axes[0] if len(cand_length_axes) == 1 else (tuple(cand_length_axes) if cand_length_axes else None)
  )
  length_shards = kernels.mesh_axis_size(mesh, cand_length_ax)
  length_ax = cand_length_ax if (length_shards > 1 and x.shape[1] > 1 and x.shape[1] % length_shards == 0) else None

  if not extra_ep:
    if x.shape[0] % batch_shards != 0:
      raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {batch_ax!r}.")
    tokens_spec = P(batch_ax, length_ax, None)
  elif cand_length_axes and x.shape[1] > 1 and x.shape[1] % (length_shards * extra_ep_size) == 0:
    if x.shape[0] % batch_shards != 0:
      raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {batch_ax!r}.")
    seq_axes = (*cand_length_axes, *extra_ep)
    tokens_spec = P(batch_ax, seq_axes[0] if len(seq_axes) == 1 else seq_axes, None)
  elif length_ax is None and x.shape[1] > 1 and x.shape[1] % extra_ep_size == 0:
    if x.shape[0] % batch_shards != 0:
      raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {batch_ax!r}.")
    extra_ax: MeshAxes = extra_ep[0] if len(extra_ep) == 1 else extra_ep
    tokens_spec = P(batch_ax, extra_ax, None)
  elif x.shape[0] % (batch_shards * extra_ep_size) == 0:
    merged_batch = (*batch_names, *extra_ep)
    tokens_spec = P(merged_batch[0] if len(merged_batch) == 1 else merged_batch, length_ax, None)
  else:
    raise ValueError(
        f"Token shape {x.shape[:2]} cannot partition disjointly across batch={batch_ax!r} ({batch_shards}) "
        f"and extra expert axes {extra_ep!r} ({extra_ep_size})."
    )
  if num_experts % ep_size != 0:
    raise ValueError(
        f"num_experts ({num_experts}) must be divisible by expert parallelism size ({ep_size}) on {experts_ax!r}."
    )

  if fused:
    w_gu_global, w_dn_global = projections
    is_4d_fused = w_gu_global.ndim == 4
    if is_4d_fused:
      mlp_dim = w_gu_global.shape[3]
      proj_specs: tuple[P, ...] = (P(experts_ax, None, None, mlp_ax), P(experts_ax, mlp_ax, None))
    else:
      if tp_size > 1:
        raise ValueError(
            "3D fused gate_up_proj cannot be tensor-parallel sharded across tp_size > 1 without replicating expert weights; "
            "store gate_up_proj as 4D (num_experts, emb_dim, 2, moe_mlp_dim)."
        )
      mlp_dim = w_gu_global.shape[2] // 2
      proj_specs = (P(experts_ax, None, None), P(experts_ax, mlp_ax, None))
    if w_dn_global.shape[1] != mlp_dim:
      raise ValueError(f"down_proj MLP dim ({w_dn_global.shape[1]}) must match gate_up MLP dim ({mlp_dim}).")
  else:
    wi_0_global, wi_1_global, wo_global = projections
    is_4d_fused = False
    mlp_dim = wi_0_global.shape[2]
    if wi_1_global.shape != wi_0_global.shape or wo_global.shape[1] != mlp_dim:
      raise ValueError(
          f"Mismatched expert weight shapes: wi_0={wi_0_global.shape}, wi_1={wi_1_global.shape}, wo={wo_global.shape}."
      )
    proj_specs = (P(experts_ax, None, mlp_ax), P(experts_ax, None, mlp_ax), P(experts_ax, mlp_ax, None))

  if mlp_dim % tp_size != 0:
    raise ValueError(f"Expert MLP dim ({mlp_dim}) must be divisible by tensor parallelism size ({tp_size}).")

  matmul_fn = functools.partial(grouped_matmul, backend=executor.kind, is_tpu=is_tpu)
  e_loc = num_experts // ep_size
  m_tp = mlp_dim // tp_size

  def _slice_local_projections(proj_local: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    if not fused:
      return proj_local
    w_gu, w_dn = proj_local
    if is_4d_fused:
      return (jnp.reshape(w_gu, (e_loc, w_gu.shape[1], 2 * m_tp)), w_dn)
    return (w_gu, w_dn)

  def per_device(x_sh: jax.Array, weights_sh: jax.Array, indices_sh: jax.Array, *proj_sh: jax.Array) -> jax.Array:
    proj_tuple: tuple[jax.Array, ...] = tuple(proj_sh)
    local_projs = _slice_local_projections(proj_tuple)

    b_loc, seq_len, d_dim = x_sh.shape
    top_k = indices_sh.shape[-1]
    num_tokens = b_loc * seq_len
    num_slots = num_tokens * top_k

    flat_x = jnp.reshape(x_sh, (num_tokens, d_dim))
    flat_idx = jnp.reshape(indices_sh, (num_slots,)).astype(jnp.int32)
    flat_w = jnp.reshape(weights_sh, (num_slots,))
    slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
    slot_rows = flat_x[slot_token_ids]

    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)

    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      prior_exp = None
      if ep_size > 1:
        ep_axis_pre: str | tuple[str, ...] = ep_axes[0] if len(ep_axes) == 1 else ep_axes
        local_counts_pre = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(
            jnp.int32
        )
        all_counts_pre = jax.lax.all_gather(local_counts_pre, ep_axis_pre, axis=0, tiled=False)
        ep_id_pre = jax.lax.axis_index(ep_axis_pre)
        prior_exp = (jnp.cumsum(all_counts_pre, axis=0) - all_counts_pre)[ep_id_pre]
      slot_valid = _truncate_slots_by_capacity(
          safe_idx,
          slot_valid,
          num_experts=num_experts,
          expert_cap=expert_cap,
          prior_expert_counts=prior_exp,
      )

    if ep_size == 1:
      if executor.ragged_buffer_factor > 0.0:
        buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
        slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)

      sort_key = jnp.where(slot_valid, safe_idx, num_experts)
      perm = jnp.argsort(sort_key, stable=True)
      inv_perm = jnp.argsort(perm)
      group_sizes = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
      valid_count = jnp.sum(group_sizes)

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(num_slots, d_dim, slot_rows.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        sorted_x = ragged_sort.a2a_ragged_sort(slot_rows, perm, valid_count)
      else:
        sorted_x = _permute_rows(slot_rows, perm)
        sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)

      expert_out = _run_local_expert_mlp(
          sorted_x,
          local_projs,
          group_sizes,
          fused=fused,
          matmul_fn=matmul_fn,
          activation=activation,
          swiglu_limit=swiglu_limit,
          out_dtype=_COMBINE_DTYPE,
      )

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(num_slots, d_dim, expert_out.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        unsorted_out = ragged_sort.a2a_ragged_unsort(expert_out, inv_perm, num_slots)
      else:
        unsorted_out = _permute_rows(expert_out, inv_perm)

      eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
      safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
      weighted = safe_out * eff_w[:, None]
      combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
      if tp_axes:
        combined = jax.lax.psum(combined, tp_axes)
      return jnp.reshape(combined, (b_loc, seq_len, d_dim))

    # Multi-device Expert Parallelism (EP > 1) via ragged_all_to_all.
    ep_axis: str | tuple[str, ...] = ep_axes[0] if len(ep_axes) == 1 else ep_axes
    ep_id = jax.lax.axis_index(ep_axis)

    recv_buf_len = ep_size * num_slots
    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      recv_buf_len = min(recv_buf_len, e_loc * expert_cap)
    if executor.ragged_buffer_factor > 0.0:
      rbf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
      recv_buf_len = min(recv_buf_len, rbf_cap)

    local_exp_counts = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
    all_exp_counts = jax.lax.all_gather(local_exp_counts, ep_axis, axis=0, tiled=False)

    if recv_buf_len < ep_size * num_slots:
      by_dst = jnp.transpose(jnp.reshape(all_exp_counts, (ep_size, ep_size, e_loc)), (1, 0, 2))
      flat_by_dst = jnp.reshape(by_dst, (ep_size, ep_size * e_loc))
      cum_end = jnp.cumsum(flat_by_dst, axis=1)
      cum_start = cum_end - flat_by_dst
      clamped_flat = jnp.clip(recv_buf_len - cum_start, 0, flat_by_dst)
      all_exp_counts = jnp.reshape(
          jnp.transpose(jnp.reshape(clamped_flat, (ep_size, ep_size, e_loc)), (1, 0, 2)),
          (ep_size, num_experts),
      )
      allowed_local = all_exp_counts[ep_id]
      pre_key = jnp.where(slot_valid, safe_idx, num_experts)
      pre_order = jnp.argsort(pre_key, stable=True)
      pre_starts = jnp.cumsum(local_exp_counts) - local_exp_counts
      pre_exp = safe_idx[pre_order]
      pre_rank = jnp.arange(num_slots, dtype=jnp.int32) - pre_starts[pre_exp]
      keep_pre = slot_valid[pre_order] & (pre_rank < allowed_local[pre_exp])
      slot_valid = keep_pre[jnp.argsort(pre_order)]

    send_key = jnp.where(slot_valid, safe_idx, num_experts)
    send_perm = jnp.argsort(send_key, stable=True)
    inv_send_perm = jnp.argsort(send_perm)
    send_x = _permute_rows(slot_rows, send_perm)

    counts_matrix = jnp.sum(jnp.reshape(all_exp_counts, (ep_size, ep_size, e_loc)), axis=2)
    recv_x = _ep_ragged_exchange(send_x, recv_buf_len, counts_matrix, ep_id, ep_axis, is_tpu=is_tpu)

    recv_exp_by_src = jax.lax.dynamic_slice_in_dim(all_exp_counts, ep_id * e_loc, e_loc, axis=1)
    local_group_sizes = jnp.sum(recv_exp_by_src, axis=0)
    chunk_sizes = jnp.reshape(recv_exp_by_src, (ep_size * e_loc,))
    chunk_ends = jnp.cumsum(chunk_sizes)
    recv_valid_end = chunk_ends[-1]
    row_idx = jnp.arange(recv_buf_len, dtype=jnp.int32)
    chunk_idx = jnp.searchsorted(chunk_ends, row_idx, side="right")
    recv_local_exp = jnp.where(row_idx < recv_valid_end, chunk_idx % e_loc, e_loc)

    local_perm = jnp.argsort(recv_local_exp, stable=True)
    inv_local_perm = jnp.argsort(local_perm)

    if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(recv_buf_len, d_dim, recv_x.dtype):
      from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

      sorted_recv_x = ragged_sort.a2a_ragged_sort(recv_x, local_perm, recv_valid_end)
    else:
      sorted_recv_x = _permute_rows(recv_x, local_perm)
      sorted_recv_x = jnp.where((row_idx < recv_valid_end)[:, None], sorted_recv_x, 0)

    expert_out = _run_local_expert_mlp(
        sorted_recv_x,
        local_projs,
        local_group_sizes,
        fused=fused,
        matmul_fn=matmul_fn,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(recv_buf_len, d_dim, expert_out.dtype):
      from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

      recv_unsorted = ragged_sort.a2a_ragged_unsort(expert_out, inv_local_perm, recv_valid_end)
    else:
      recv_unsorted = _permute_rows(expert_out, inv_local_perm)
      recv_unsorted = jnp.where((row_idx < recv_valid_end)[:, None], recv_unsorted, 0)

    returned_sorted_slots = _ep_ragged_exchange(recv_unsorted, num_slots, counts_matrix.T, ep_id, ep_axis, is_tpu=is_tpu)
    unsorted_slots = _permute_rows(returned_sorted_slots, inv_send_perm)

    eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
    safe_slots = jnp.where(slot_valid[:, None], unsorted_slots.astype(_COMBINE_DTYPE), 0.0)
    weighted = safe_slots * eff_w[:, None]
    combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
    if tp_axes:
      combined = jax.lax.psum(combined, tp_axes)
    return jnp.reshape(combined, (b_loc, seq_len, d_dim))

  sharded_fn = jax.shard_map(
      per_device,
      mesh=mesh,
      in_specs=(tokens_spec, tokens_spec, tokens_spec, *proj_specs),
      out_specs=tokens_spec,
      check_vma=False,
  )
  orig_x_spec = getattr(jax.typeof(x).sharding, "spec", None)
  x_in = kernels.ensure_explicit_spec(x, tokens_spec, mesh)
  weights_in = kernels.ensure_explicit_spec(weights, tokens_spec, mesh)
  indices_in = kernels.ensure_explicit_spec(indices, tokens_spec, mesh)
  proj_in = tuple(kernels.ensure_explicit_spec(p, spec, mesh) for p, spec in zip(projections, proj_specs))
  out = sharded_fn(x_in, weights_in, indices_in, *proj_in)
  if extra_ep:
    expected_out_spec = P(batch_ax, expected_length_ax, None)
    if orig_x_spec is not None and ambient_mesh_is_explicit():
      out = kernels.ensure_explicit_spec(out, orig_x_spec, mesh)
    elif tokens_spec != expected_out_spec:
      out = jax.lax.with_sharding_constraint(out, jax.sharding.NamedSharding(mesh, expected_out_spec))
  return out


# ==============================================================================
# Public entrypoints
# ==============================================================================


def _validate_routed_inputs(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    num_experts: int,
) -> None:
  """Validates 3D token, routing weight, and routing index tensor shapes."""
  if x.ndim != 3 or weights.ndim != 3 or indices.ndim != 3:
    raise ValueError(
        f"Expected 3D x (B, S, D), weights (B, S, K), and indices (B, S, K); "
        f"got x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if x.shape[:2] != weights.shape[:2] or weights.shape != indices.shape:
    raise ValueError(
        f"Mismatched batch/length/top_k shapes: x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if num_experts < 1:
    raise ValueError(f"num_experts must be >= 1; got {num_experts}.")


def routed_experts(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
) -> jax.Array:
  """Executes selected experts with separate `(wi_0, wi_1, wo)` projections."""
  if wi_0.ndim != 3 or wi_1.ndim != 3 or wo.ndim != 3:
    raise ValueError(f"Expected 3D wi_0, wi_1, wo; got {wi_0.shape}, {wi_1.shape}, {wo.shape}.")
  num_experts = wi_0.shape[0]
  _validate_routed_inputs(x, weights, indices, num_experts)
  if wi_0.shape[1] != x.shape[2] or wo.shape[0] != num_experts or wo.shape[2] != x.shape[2]:
    raise ValueError(f"Incompatible expert weight shapes for x={x.shape}: wi_0={wi_0.shape}, wo={wo.shape}.")

  projections = (wi_0.astype(x.dtype), wi_1.astype(x.dtype), wo.astype(x.dtype))
  match executor.kind:
    case "dense":
      out = _dense_separate(
          x,
          weights,
          indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      out = _grouped_unsharded(
          x,
          weights,
          indices,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights,
          indices,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")


def routed_experts_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
) -> jax.Array:
  """Executes selected experts with fused 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim not in (3, 4) or down_proj.ndim != 3:
    raise ValueError(
        f"Expected 3D or 4D gate_up_proj and 3D down_proj; got gate_up_proj={gate_up_proj.shape}, "
        f"down_proj={down_proj.shape}."
    )
  if gate_up_proj.ndim == 3 and gate_up_proj.shape[2] % 2 != 0:
    raise ValueError(f"3D gate_up_proj last dim must be even (2 * mlp_dim); got {gate_up_proj.shape}.")
  if gate_up_proj.ndim == 4 and gate_up_proj.shape[2] != 2:
    raise ValueError(f"4D gate_up_proj axis 2 must have size 2 (gate, up); got {gate_up_proj.shape}.")

  num_experts = gate_up_proj.shape[0]
  _validate_routed_inputs(x, weights, indices, num_experts)
  if gate_up_proj.shape[1] != x.shape[2] or down_proj.shape[0] != num_experts or down_proj.shape[2] != x.shape[2]:
    raise ValueError(
        f"Incompatible fused expert weight shapes for x={x.shape}: "
        f"gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}."
    )

  projections = (gate_up_proj.astype(x.dtype), down_proj.astype(x.dtype))
  match executor.kind:
    case "dense":
      out = _dense_fused(
          x,
          weights,
          indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      out = _grouped_unsharded(
          x,
          weights,
          indices,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights,
          indices,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")
