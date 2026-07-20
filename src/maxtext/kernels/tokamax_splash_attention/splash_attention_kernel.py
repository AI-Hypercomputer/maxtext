# pylint: skip-file
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.tokamax_splash_attention import base
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as mask_lib
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info as mask_info_lib


P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


def _base2_stats_to_natural_base(stats: dict[str, jax.Array], mask_value: float) -> dict[str, jax.Array]:
  """Converts base-2 residual stats to natural-base user stats."""
  stats["logsumexp"] = jnp.where(
      stats["logsumexp"] == mask_value,
      mask_value,
      stats["logsumexp"] / LOG2E,
  )
  stats["max_logits"] = jnp.where(
      stats["max_logits"] == mask_value,
      mask_value,
      stats["max_logits"] / LOG2E,
  )
  return stats


SegmentIds = base.SegmentIds

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  # TODO: Remove these 3 params, they're only kept for backwards compatibility.
  block_q_dq: int | None = None
  block_kv_dq: int | None = None
  use_fused_bwd_kernel: bool = True

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  fwd_cost_estimate: pl.CostEstimate | None = None
  bwd_cost_estimate: pl.CostEstimate | None = None

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False
  # The fused bwd kernel accumulates dq at every grid step. To safely avoid
  # read/write conflicts we conservatively avoid *any* in-kernel reductions.
  # This parameter allows to override this behavior and specifies the number of
  # reduction steps. For now, only 3 or all the kv steps are supported.
  dq_reduction_steps: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps is not None and self.dq_reduction_steps != 3:
      raise ValueError(f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 3 or" " None are supported.")
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


to_i32 = lambda x: x.astype(jnp.int32)


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask)
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)

        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(q_sequence_ref[...], (1, repeats))  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("Mask function must return a boolean-valued array, but got:" f" {computed_mask.dtype}")
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = jnp.tile(kv_segment_ids_ref[k_slice, :], (1, repeats))  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
    return qk, mask
  else:
    qk = cap_logits(qk)
  return qk, None


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    block_mask = block_mask_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask != 1
    should_run = block_mask != 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
  else:
    should_not_mask = False
    should_run = True
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate *= LOG2E

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    sink = None
    if sinks_ref is not None:
      sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      if config.use_base2_exp:
        sink *= LOG2E

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = exp(sink - jnp.full_like(l_scratch_ref, max_logit_estimate))

  def body(kv_compute_index, _, has_partial_mask=False):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp:
      q *= LOG2E

    qk_dims = NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
    else:
      k = k_ref[:, slice_k]
    qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    assert qk.shape == (bq, bkv_compute)
    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    qk, mask = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[:, None]  # pytype: disable=attribute-error
      assert m_curr.shape == (bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - jnp.tile(m_next, (1, bkv_repeats)))
    else:
      s_curr = exp(qk - max_logit_estimate)
    if mask is not None:
      s_curr = jnp.where(mask, s_curr, 0.0)
    assert s_curr.shape == (bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    sv_dims = NN_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
    else:
      v = v_ref[:, slice_k]
    o_curr = lax.dot_general(s_curr, v, sv_dims)

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, head_dim_v_repeats))
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(0, num_iters, partial(body, has_partial_mask=True), None, unroll=True)

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    safe_l = jnp.where(l == 0.0, 1.0, l)
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = jnp.tile(jnp.where(l == 0.0, 0.0, 1.0 / safe_l), (1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = jnp.where(l == 0.0, mask_value, m + log(safe_l))
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (bq, NUM_LANES)
      l_linear_ref[...] = l.astype(l_linear_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (bq, NUM_LANES)
      max_logits_ref[...] = m.astype(max_logits_ref.dtype)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
  if x is None:
    return 0

  if jnp.issubdtype(x.dtype, jnp.floating):
    info = jnp.finfo
  elif jnp.issubdtype(x.dtype, jnp.integer):
    info = jnp.iinfo
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a" f" {len(k.shape)}-dim one.")

  if k.shape[-1] != head_dim_qk:
    raise ValueError(f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:" f" {k.shape[-1]}.")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA/GQA, expected number of 'query' heads ({num_q_heads}) to"
        f" be a multiple of the number of 'key' heads ({num_kv_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same " "leading dimensions.")

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  dynamic_grid = mask_info.active_rows is not None

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError("Invalid shape for q segment_ids: " f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}")
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError("Invalid shape for kv segment_ids: " f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}")
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(f"Only one of {config.max_logit_const=} and" f" {max_logit_value=} can be set.")
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(jnp.atleast_1d(max_logit_value), (num_q_heads,))

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  def unravel(f):
    def index_map(h, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h, i, j):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(lambda h, i, j: from_head_minor((h, i, 0), q_layout))
  out_index_map = unravel(lambda h, i, j: (h, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h, i, j: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor((bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, NUM_LANES), (0,))
    kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,))
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
    sinks = jnp.broadcast_to(sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads))
  else:
    in_specs += [None]

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,))
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
  else:
    in_specs.append(None)

  out_shapes = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((None, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(lambda h, i, j, *_: (h, i, 0))

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32) if fuse_reciprocal else None,
        # l_linear
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32) if not fuse_reciprocal else None,
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]
    out_specs += [
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map) if fuse_reciprocal else None,
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map) if not fuse_reciprocal else None,
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map),
    ]
  else:
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = 2 * q_seq_len * kv_seq_len * head_dim_qk + 2 * q_seq_len * kv_seq_len * head_dim_v

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

    inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
  ]
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(*vmem_inputs, out_shapes, fwd_mask_sparsity)

  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]
    grid = (num_q_heads, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (num_q_heads, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # m_scratch
                pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # l_scratch
                pltpu.VMEM((bq, head_dim_v), jnp.float32),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": (config.use_experimental_scheduler)},
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out

  # If there is no compute to do within an attention block, then we want to
  # initialize the output and residuals to default values. Otherwise, we will
  # read uninitialized memory. This is a common case in ring attention.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x

    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  if save_residuals:
    assert max_logits is not None
    max_logits = init_if_empty(max_logits[..., 0], mask_value)

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(logsumexp[..., 0], mask_value)
    else:
      assert l_linear is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log

      l = init_if_empty(l_linear[..., 0], 0.0)
      safe_l = jnp.where(l == 0.0, 1.0, l)
      logsumexp = jnp.where(l == 0.0, mask_value, max_logits + log(safe_l))
      l_inv = jnp.where(l == 0.0, 0.0, 1.0 / safe_l)
      out = (out * l_inv[..., None]).astype(out.dtype)
  else:
    # If we're not saving residuals, then we can't fuse the reciprocal
    # out of the kernel.
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(out, name=config.residual_checkpoint_name)
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(logsumexp, name=config.residual_checkpoint_name)
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "save_residuals",
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ),
)
def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dkv_mask_info

  ret = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats = _base2_stats_to_natural_base(stats, mask_value)
    return out, stats
  else:
    return ret


def _splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], base.SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats = _base2_stats_to_natural_base(stats, mask_value)
  residuals = q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info
  if save_residuals:
    return (out, stats), residuals
  else:
    return out, residuals


def _flash_attention_dq_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  grid_idx = pl.program_id(1)
  if block_mask_ref is not None:
    kv_index = active_cols_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index = grid_idx % kv_steps
    should_not_mask = False
    should_initialize = kv_index == 0
    should_write = kv_index == kv_steps - 1

  @pl.when(should_initialize)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def body(has_partial_mask: bool = False):
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp:
      q *= LOG2E
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk, mask = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    if mask is not None:
      p = jnp.where(mask, p, 0.0)
    dp_dims = NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    dp = lax.dot_general(
        do.astype(v.dtype),
        v,
        dp_dims,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    dq_dims = NN_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype),
        k,
        dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(should_not_mask)
  def _():
    body()

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    body(has_partial_mask=True)

  @pl.when(should_write)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # aliases
    dq_alias,
    dk_alias,
    dv_alias,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    # Scratch
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    mask_value: float,
    q_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: SplashConfig,
):
  del mask_next_ref, active_cols_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(should_initialize, q_head_index_per_kv_head == 0)
      should_write = jnp.logical_and(should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
  else:
    should_not_mask = False
    should_run = True

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def body(i, _, has_partial_mask=False):

    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS
    if config.use_base2_exp:
      scaled_q = q * LOG2E
    else:
      scaled_q = q

    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    logsumexp = logsumexp_ref[:1, :]
    do = do_ref[...]
    di = di_ref[:1, :]

    qk_dims = NT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(k, scaled_q, qk_dims, preferred_element_type=jnp.float32)

    qk, mask = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    if mask is not None:
      p = jnp.where(mask, p, 0.0)
    dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v,
        do,
        NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    dk_dims = NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dk = lax.dot_general(ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32)
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T.astype(k.dtype),
          k,
          NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        # Compute block size == memory block size
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  num_iters = k_ref.shape[0 if config.k_layout is HEAD_DIM_MINOR else 1] // bkv_compute

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(0, num_iters, partial(body, has_partial_mask=True), None, unroll=True)

  if dq_scratch_ref is not None:
    if dq_alias is not None:
      dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
    else:
      dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  else:
    q_head = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _():
      dk_ref[...] = dk_alias[...] + dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + dv_scratch_ref[...].astype(dv_ref.dtype)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
    dkv_mask_sparsity: float,
    return_fp32_grads: bool = False,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA/GQA, expected number of 'query' heads ({num_q_heads}) to"
        f" be a multiple of the number of 'key' heads ({num_kv_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same " "leading dimensions.")

  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  q_index_map = unravel(lambda h, i, j: from_head_minor((h, i, 0), config.q_layout))
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map)

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,))
    kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (kv_seq_len, NUM_LANES), (0,))
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,))
  else:
    q_sequence = None
    in_specs.append(None)

  dq_reduction_steps = config.dq_reduction_steps
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  if dq_reduction_steps == 3:
    dq_index_map = unravel(lambda h, i, j: (j % 3, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_dtype = jnp.float32 if return_fp32_grads else q.dtype
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), dq_dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = jnp.float32 if return_fp32_grads or kv_steps > 4 else q.dtype
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *q.shape), q_dtype)

  in_specs += [dq_alias_spec]

  if bkv == bkv_compute:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM((bq, head_dim_qk), jnp.float32)

  if dynamic_grid and q_heads_per_kv_head != 1:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = jnp.float32 if return_fp32_grads else k.dtype
    dv_type = jnp.float32 if return_fp32_grads else v.dtype

  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(k.shape, dk_type),
      jax.ShapeDtypeStruct(v.shape, dv_type),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=config.use_experimental_scheduler,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  num_args = sum(1 for x in args if x is not None)
  input_output_aliases = {}
  if dq_reduction_steps == 3:
    if dynamic_grid and q_heads_per_kv_head != 1:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif dynamic_grid and q_heads_per_kv_head != 1:
    input_output_aliases = {num_args: 1, num_args + 1: 2}

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM((bkv, head_dim_qk), jnp.float32),
      pltpu.VMEM((bkv, head_dim_v), jnp.float32),
  ]

  def _bwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      logsumexp: jax.Array,
      do: jax.Array,
      di: jax.Array,
      partial_mask_blocks: jax.Array | None,
      q_sequence: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    total_matmul_flops_per_head = (
        2 * q_seq_len * kv_seq_len * head_dim_qk  # qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dv
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dp
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dq
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dk
    )

    estimated_flops = int(total_matmul_flops_per_head * num_q_heads * mask_sparsity_factor)

    exp_flops = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    if config.attn_logits_soft_cap is None:
      tanh_flops = 0
    else:
      tanh_flops = 2 * num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    estimated_transcendentals = int(exp_flops + tanh_flops)

    inputs_ = [
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        partial_mask_blocks,
        q_sequence,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))

    estimated_bytes = input_bytes + output_bytes

    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=estimated_bytes,
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",) * len(grid)),
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv)
  dq = dq_unreduced.sum(axis=0)
  if return_fp32_grads:
    return dq.astype(jnp.float32), dk.astype(jnp.float32), dv.astype(jnp.float32)
  dq = dq.astype(q.dtype)
  dk = dk.astype(k.dtype)
  dv = dv.astype(v.dtype)
  return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    res: base.SplashResidualsType,
    grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]],
    return_fp32_grads: bool = False,
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    base.SegmentIds | None,  # segment_ids
    jax.Array | None,  # segment_ids
    jax.Array | None,  # max_logit_estimate
]:
  # If `save_residuals` is True, `_splash_attention_fwd` returns `(out, stats)`,
  # so we unpack the gradients, otherwise it returns `out` and `grads` is just
  # `do`.
  if save_residuals:
    do, _ = grads
  else:
    do = grads
  del save_residuals, fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      config.block_q_dkv,
      config.block_kv_dkv,
      config.block_kv_dkv_compute,
  )
  q, k, v, segment_ids, sinks, o, logsumexp, dkv_mask_info = res

  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pytype: disable=attribute-error
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,
      bkv=bkv_dkv_memory,
      bkv_compute=bkv_dkv_compute,
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
      return_fp32_grads=return_fp32_grads,
  )
  dsinks = None
  if sinks is not None:
    logsumexp_ = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
    sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - logsumexp_[..., None].astype(jnp.float32))
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      None,  # segment_ids
      dsinks,  # sinks
      None,  # max_logit_estimate
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
) -> base.SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> base.SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError("The sharding must divide the mask blocks evenly between devices") from exc

    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None

    def mask_info_spec(mask_info):
      if mask_info is None:
        return None
      return MaskInfo(  # pytype: disable=wrong-arg-types
          mask_next=_resolve_spec(mask_info.mask_next),
          active_rows=_resolve_spec(mask_info.active_rows),
          active_cols=_resolve_spec(mask_info.active_cols),
          num_active_blocks=_resolve_spec(mask_info.num_active_blocks),
          block_mask=_resolve_spec(mask_info.block_mask),
          partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated
          if mask_info.partial_mask_blocks is not None
          else None,
          q_sequence=_resolve_spec(mask_info.q_sequence),
      )

    return SplashAttentionKernel(
        mask_info_spec(self.fwd_mask_info),
        mask_info_spec(self.dkv_mask_info),
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dkv_mask_info = MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    return SplashAttentionKernel(MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs)


def _make_splash_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
):
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )

    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
):
  if (mesh is not None) != (mask_spec is not None):
    raise ValueError("Either both or neither of mesh and mask_spec must be specified.")

  if mask_spec is not None and len(mask_spec) != 1:
    raise ValueError("Only shard over the query sequence dimension.")

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if config is None:
    config = SplashConfig.get_default()

  # This is the only mode that supports the dynamic grid.
  config = dataclasses.replace(config, dq_reduction_steps=3)

  def process_mask_shard(mask):
    process_mask_fn = functools.partial(
        mask_info_lib._process_dynamic_mask,
        downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,
    )

    fwd_mask_info = process_mask_fn(mask, (config.block_q, config.block_kv), is_dkv=False)

    dkv_mask_info = None
    if config.has_backward_blocks:
      dkv_mask_info = process_mask_fn(mask, (config.block_q_dkv, config.block_kv_dkv), is_dkv=True)

    return fwd_mask_info, dkv_mask_info

  kwargs = dict(
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      dkv_mask_sparsity=1.0,
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dkv_mask_info = process_mask_shard(mask)
    kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
    return kernel

  mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
      mask_next=mask_spec,
      active_rows=None,
      active_cols=None,
      num_active_blocks=None,
      block_mask=mask_spec,
      partial_mask_blocks=mask_spec,
      q_sequence=None,
  )
  out_specs = (
      mask_info_specs,
      mask_info_specs if config.has_backward_blocks else None,
  )

  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=mask_spec,
      out_specs=out_specs,
      check_vma=False,
  )
  def process_all_shards(mask):
    return process_mask_shard(mask)

  fwd_mask_info, dkv_mask_info = process_all_shards(mask)
  kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
  kernel_spec = SplashAttentionKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, q_seq_shards=1)

make_splash_mqa_single_device = partial(make_splash_mqa, q_seq_shards=1)

make_dynamic_splash_mqa = partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = partial(_make_dynamic_splash_attention, is_mqa=False)
