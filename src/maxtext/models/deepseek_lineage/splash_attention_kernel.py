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
from typing import Any, NamedTuple

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.models.deepseek_lineage import base
from maxtext.models.deepseek_lineage import splash_attention_mask as mask_lib
from maxtext.models.deepseek_lineage import splash_attention_mask_info as mask_info_lib
import numpy as np

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


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


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


def _generate_blockwise_dropout_mask(
    prng_key: jax.Ref | jax.Array,
    head_idx: jax.Array | int,
    q_block_idx: jax.Array | int,
    kv_block_idx: jax.Array | int,
    q_block_size: int,
    kv_block_size: int,
    dropout_rate: float,
) -> jax.Array:
  """Generates the dropout mask of a single (head, q block, kv block) tile.

  The mask is a pure function of the key and the block coordinates, so the
  forward and backward kernels regenerate the identical mask without either
  materializing the full [heads, q, kv] mask in HBM or saving it as a residual.
  True means "dropped".

  The block coordinates are indices into the canonical dropout grid, which is
  independent of the kernel's block sizes; callers go through
  `_dropout_mask_tile` rather than calling this directly.

  The batch dimension is not folded in here: the kernels are written for a
  single batch element, so the caller is responsible for folding a batch index
  into `prng_key` before it reaches the kernel.
  """
  # TODO: Optimize key derivation by using one fold_in instead of three.
  # We can collapse the serial chain (head -> q_block -> kv_block) if we pass static
  # grid extents (n_q_blocks, n_kv_blocks) from config:
  # block_id = (head_idx * n_q_blocks + q_block_idx) * n_kv_blocks + kv_block_idx
  # sub_key = jax.random.fold_in(prng_key[...], block_id)
  sub_key = prng_key[...]
  sub_key = jax.random.fold_in(sub_key, head_idx)
  sub_key = jax.random.fold_in(sub_key, q_block_idx)
  sub_key = jax.random.fold_in(sub_key, kv_block_idx)
  # TODO: Avoid float round-trip in bernoulli mask generation.
  # jax.random.bernoulli builds float32 uniform then does f32 compare.
  # We can use raw bits and compare with threshold directly:
  # bits = jax.random.bits(sub_key, (q_block_size, kv_block_size), jnp.uint32)
  # return bits < np.uint32(dropout_rate * 2**32)
  return jax.random.bernoulli(sub_key, dropout_rate, (q_block_size, kv_block_size))


def _dropout_mask_tile(
    prng_key: jax.Ref | jax.Array,
    *,
    head_idx: jax.Array | int,
    q_block_idx: jax.Array | int,
    kv_block_idx: jax.Array | int,
    q_block_size: int,
    kv_block_size: int,
    canonical_q: int,
    canonical_kv: int,
    dropout_rate: float,
) -> jax.Array:
  """Builds the [q_block_size, kv_block_size] dropout mask of one kernel tile.

  The randomness lives on a fixed `canonical_q x canonical_kv` grid laid over
  the whole attention matrix and is keyed by absolute canonical-block
  coordinates, not by the kernel's own tiling. A tile is assembled from the
  canonical blocks it covers, so an element's dropout bit is a function of its
  (head, query, key) position alone -- the same property a dense mask array
  indexed by absolute coordinates has, without materializing one. The forward
  and the backward may therefore tile the attention matrix differently and
  still agree on which weights were dropped.

  Tile sizes are whole multiples of the canonical sizes (validated in
  `SplashConfig.__post_init__`) and tile offsets are tile-aligned, so a
  canonical block is never split across two tiles. When a tile size equals the
  canonical size this collapses to a single draw keyed on the tile's own index,
  i.e. it is bit-identical to keying on the tile directly.
  """
  nq, rem = divmod(q_block_size, canonical_q)
  assert rem == 0, f"{q_block_size=} must be a multiple of {canonical_q=}"
  nkv, rem = divmod(kv_block_size, canonical_kv)
  assert rem == 0, f"{kv_block_size=} must be a multiple of {canonical_kv=}"

  # Block indices, not element offsets: the canonical index of a tile's first
  # block is exact without a division on a traced value.
  q_base = q_block_idx * nq
  kv_base = kv_block_idx * nkv
  rows = []
  for a in range(nq):
    cols = [
        _generate_blockwise_dropout_mask(
            prng_key,
            head_idx=head_idx,
            q_block_idx=q_base + a,
            kv_block_idx=kv_base + b,
            q_block_size=canonical_q,
            kv_block_size=canonical_kv,
            dropout_rate=dropout_rate,
        )
        for b in range(nkv)
    ]
    rows.append(cols[0] if nkv == 1 else jnp.concatenate(cols, axis=1))
  return rows[0] if nq == 1 else jnp.concatenate(rows, axis=0)


def _check_dropout_args(
    config: "SplashConfig",
    prng_key: jax.Array | None,
) -> jax.Array | None:
  """Validates the dropout arguments and converts the key for Pallas."""

  if not config.dropout_rate:
    return None
  if prng_key is None:
    raise ValueError("A prng_key is required when config.dropout_rate > 0; got None.")
  # A key passed as a regular operand fails to lower ("AssertionError:
  # key<pl>"), so it travels through scalar prefetch as a Pallas key instead.
  return pltpu.to_pallas_key(prng_key)


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
  num_stacked_q_heads: int = 1
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
  # Skip the wasted causal-diagonal QK matmul. On a partial-mask (diagonal) block,
  # split the QK matmul into a qk_diag_grid x qk_diag_grid sub-grid over (kv rows,
  # q cols) and skip every sub-tile that lies entirely above the causal line
  # (kv > q), filling mask_value instead of computing it. Those entries are
  # overwritten to mask_value by _apply_mask_and_soft_cap regardless, so the result
  # is bit-exact; the elementwise/softmax ops still run on the full assembled tile.
  # PRECONDITION (enforced below): pure CausalMask + aligned SQUARE blocks with a
  # single compute tile per block (block_q == block_kv == block_kv_compute, and the
  # backward trio), and sequence length a multiple of the block. Any other config
  # raises (it does not silently corrupt); a non-causal mask raises too.
  qk_diag_skip: bool = False
  # Granularity of the diagonal skip: grid=2 -> quadrants (skip 1/4 of the diagonal
  # block, per-block waste 1/2 -> 1/4); grid=4 -> 4x4 (skip 6/16, waste -> 1/8).
  # Larger grid skips more of the triangle but uses smaller (less MXU-efficient)
  # matmuls; grid=4 is a good default at S=4096/block=2048. Must be a power of 2.
  qk_diag_grid: int = 2
  # Skip the wasted causal-diagonal SV matmul. On a partial-mask (diagonal)
  # block, split the SV matmul into a qk_diag_grid x qk_diag_grid sub-grid and
  # skip sub-tiles strictly above the diagonal (kj > qi) during dot_general.
  sv_diag_skip: bool = False
  # Attention dropout probability, applied to the softmax weights. The mask is
  # generated inside the kernel per (head, q block, kv block) tile from a
  # `prng_key` passed at call time, so it costs no HBM traffic and is never
  # saved for the backward pass. Requires a `prng_key`; 0.0 disables it and
  # leaves the kernel bit-identical to the no-dropout version.
  dropout_rate: float = 0.0
  # Granularity of the grid the dropout mask is drawn on. Keying the mask on
  # absolute canonical coordinates (instead of kernel tiles) allows fwd and
  # bwd to use different block sizes. Every tile size must be a multiple of the
  # canonical size. Recommended to set to the smaller of the fwd/bwd block sizes (coarsest
  # common grid). Set smaller to decouple block sizes that aren't multiples.
  dropout_block_q: int | None = None
  dropout_block_kv: int | None = None

  @property
  def active_dropout_block_q(self) -> int:
    assert self.dropout_block_q is not None
    return self.dropout_block_q

  @property
  def active_dropout_block_kv(self) -> int:
    assert self.dropout_block_kv is not None
    return self.dropout_block_kv

  def __post_init__(self):
    if not 0.0 <= self.dropout_rate < 1.0:
      raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}.")
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps is not None and self.dq_reduction_steps != 3:
      raise ValueError(f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 3 or" " None are supported.")
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")

    if self.dropout_rate:
      # The backward regenerates the mask rather than reloading it, so the two
      # passes have to agree on which weights were dropped. They are keyed on a
      # canonical grid of absolute coordinates instead of on their own tiles,
      # which makes agreement independent of the tiling; all that is left to
      # check is that each tile covers a whole number of canonical blocks.
      if self.dropout_block_q is None or self.dropout_block_kv is None:
        raise ValueError("dropout_block_q and dropout_block_kv must be set if dropout_rate" " > 0.")
      if self.dropout_block_q <= 0:
        raise ValueError(f"dropout_block_q must be positive, got {self.dropout_block_q}.")
      if self.dropout_block_kv <= 0:
        raise ValueError(f"dropout_block_kv must be positive, got {self.dropout_block_kv}.")

      if self.block_q % self.dropout_block_q != 0:
        raise ValueError(f"block_q={self.block_q} must be a multiple of" f" dropout_block_q={self.dropout_block_q}.")
      if self.block_q_dkv is not None and self.block_q_dkv % self.dropout_block_q != 0:
        raise ValueError(
            f"block_q_dkv={self.block_q_dkv} must be a multiple of" f" dropout_block_q={self.dropout_block_q}."
        )

      assert self.block_kv_compute is not None
      if self.block_kv_compute % self.dropout_block_kv != 0:
        raise ValueError(
            f"block_kv_compute={self.block_kv_compute} must be a multiple of"
            f" dropout_block_kv={self.dropout_block_kv}."
        )
      if self.block_kv_dkv_compute is not None and self.block_kv_dkv_compute % self.dropout_block_kv != 0:
        raise ValueError(
            f"block_kv_dkv_compute={self.block_kv_dkv_compute} must be a"
            f" multiple of dropout_block_kv={self.dropout_block_kv}."
        )

    if self.qk_diag_skip:
      # The skip fills mask_value for sub-tiles where kv > q, relying on the mask to
      # mask EXACTLY those. That holds only for aligned SQUARE blocks (kv-band > q-band
      # <=> fully above the causal line, single compute tile per block) — enforce it or
      # the skip silently corrupts. Causality is checked in _make_splash_attention.
      if not (self.block_q == self.block_kv == self.block_kv_compute):
        raise ValueError(
            "qk_diag_skip requires square forward blocks "
            "(block_q == block_kv == block_kv_compute); got "
            f"{self.block_q}/{self.block_kv}/{self.block_kv_compute}."
        )
      if self.has_backward_blocks and not (self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute):
        raise ValueError(
            "qk_diag_skip requires square backward blocks "
            "(block_q_dkv == block_kv_dkv == block_kv_dkv_compute); got "
            f"{self.block_q_dkv}/{self.block_kv_dkv}/{self.block_kv_dkv_compute}."
        )
      if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
        raise ValueError(f"qk_diag_grid must be a power of 2 >= 2; got {self.qk_diag_grid}.")

    if self.sv_diag_skip:
      # The skip assumes the kv > q region is masked. That holds only for aligned
      # SQUARE blocks (kv-band > q-band <=> fully above the causal line, single
      # compute tile per block) — enforce it or the skip silently corrupts.
      # Causality is checked in _make_splash_attention.
      if not (self.block_q == self.block_kv == self.block_kv_compute):
        raise ValueError(
            "sv_diag_skip requires square forward blocks "
            "(block_q == block_kv == block_kv_compute); got "
            f"{self.block_q}/{self.block_kv}/{self.block_kv_compute}."
        )
      if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
        raise ValueError(f"qk_diag_grid must be a power of 2 >= 2; got {self.qk_diag_grid}.")

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
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask.astype(jnp.bool_))
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(
            q_sequence_ref[...], (1, repeats)  # pyrefly: ignore[unsupported-operation]
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]  # pyrefly: ignore[unsupported-operation]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)
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
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    masks = [m.astype(jnp.bool_) for m in masks]
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    if mask.ndim == 2 and qk.ndim == 3:
      mask = jnp.expand_dims(mask, axis=0)

    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    prng_key_ref,
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
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
    # Dropout needs the q block index too; it is only tracked on the dynamic
    # grid, where `grid_idx` is a position in the sparse (i, j) work list.
    i = active_rows_ref[grid_idx].astype(jnp.int32) if dropout_rate else 0
  else:
    should_not_mask = False
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1
    i = grid_idx // kv_steps

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    assert num_stacked_q_heads == 1
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
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = exp(
          sink
          - jnp.full_like(l_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      )

  def body(kv_compute_index, _, has_partial_mask=False):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)
    assert l_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
    if config.use_base2_exp:
      q *= LOG2E

    head_dim_qk = q.shape[-1]
    # Collapse the head and sequence dimensions for a larger matmul.
    q_flat = q.reshape((num_stacked_q_heads * bq, head_dim_qk))

    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
      qk_dims = NT_DIM_NUMBERS
    else:
      k = k_ref[:, slice_k]
      qk_dims = NN_DIM_NUMBERS

    _g = config.qk_diag_grid
    if (
        config.qk_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.k_layout == HEAD_DIM_MINOR
        and bq % _g == 0
        and bkv_compute % _g == 0
    ):
      # Diagonal skip (forward): qk tile is [q, kv]. On an aligned square diagonal
      # block, sub-tile (q-band qi, kv-band kj) with kj > qi is fully above the causal
      # boundary (kv > q) -> masked to mask_value anyway -> skip its matmul.
      sq = bq // _g
      sk = bkv_compute // _g
      q_parts = [q_flat[i * sq : (i + 1) * sq, :] for i in range(_g)]
      k_parts = [k[j * sk : (j + 1) * sk, :] for j in range(_g)]
      rows = []
      for qi in range(_g):  # q row-band
        cols = []
        for kj in range(_g):  # kv col-band
          if kj > qi:  # fully masked -> skip matmul
            cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
          else:
            cols.append(lax.dot_general(q_parts[qi], k_parts[kj], qk_dims, preferred_element_type=float32))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_flat = jnp.concatenate(rows, axis=0)
    else:
      qk_flat = lax.dot_general(q_flat, k, qk_dims, preferred_element_type=float32)
    qk = qk_flat.reshape((num_stacked_q_heads, bq, bkv_compute))

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

    qk = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]  # pyrefly: ignore[missing-attribute]
      assert m_curr.shape == (num_stacked_q_heads, bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (num_stacked_q_heads, bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(
          qk - jnp.tile(m_next, (1, 1, bkv_repeats))
      )  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    else:
      s_curr = exp(qk - max_logit_estimate)  # pyrefly: ignore[unsupported-operation]
    assert s_curr.shape == (num_stacked_q_heads, bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    assert l_curr.shape == (num_stacked_q_heads, bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    # Dropout is applied *after* the running softmax denominator has been
    # accumulated from the undropped weights, so it does not renormalize over
    # the survivors: only the numerator (the s @ v product) sees the mask.
    if dropout_rate:
      global_kv_block_idx = j * (bkv // bkv_compute) + kv_compute_index
      # One mask per stacked head, assembled into the (heads, bq, bkv) tile:
      # writing them in with `.at[head].set` would lower to a scatter, which
      # Mosaic does not implement. Built with an explicit loop rather than a
      # comprehension: `i` is also a comprehension target above, and an inlined
      # comprehension (PEP 709) that reads it would bind the shadowed local.
      head_masks = []
      for head in range(num_stacked_q_heads):
        head_masks.append(
            _dropout_mask_tile(
                prng_key_ref,
                head_idx=h * num_stacked_q_heads + head,
                q_block_idx=i,
                kv_block_idx=global_kv_block_idx,
                q_block_size=bq,
                kv_block_size=bkv_compute,
                canonical_q=config.active_dropout_block_q,
                canonical_kv=config.active_dropout_block_kv,
                dropout_rate=dropout_rate,
            )
        )
      dropout_mask = jnp.stack(head_masks, axis=0)
      s_curr = jnp.where(dropout_mask, 0.0, s_curr) / (1.0 - dropout_rate)

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * bq, bkv_compute))

    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS

    if (
        config.sv_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.v_layout == HEAD_DIM_MINOR
        and bq % _g == 0
        and bkv_compute % _g == 0
    ):
      sq = bq // _g
      sk = bkv_compute // _g
      s_parts = [[s_curr_flat[i * sq : (i + 1) * sq, j * sk : (j + 1) * sk] for j in range(_g)] for i in range(_g)]
      v_parts = [v[j * sk : (j + 1) * sk, :] for j in range(_g)]
      o_rows = []
      for qi in range(_g):
        o_row = lax.dot_general(s_parts[qi][0], v_parts[0], sv_dims)
        for kj in range(1, qi + 1):
          o_row = o_row + lax.dot_general(s_parts[qi][kj], v_parts[kj], sv_dims)
        o_rows.append(o_row)
      o_curr_flat = jnp.concatenate(o_rows, axis=0)
    else:
      o_curr_flat = lax.dot_general(s_curr_flat, v, sv_dims)
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, bq, head_dim_v))

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute

  @pl.when(should_not_mask)
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    lax.fori_loop(0, num_iters, partial(body, has_partial_mask=True), None, unroll=True)

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = jnp.tile(1.0 / l, (1, 1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      l_linear_ref[...] = l.astype(l_linear_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
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
    prng_key: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  num_stacked_q_heads = config.num_stacked_q_heads
  prng_key = _check_dropout_args(config, prng_key)

  if num_stacked_q_heads > 1 and (sinks is not None or max_logit_value is not None):
    raise ValueError("Stacked heads are not supported with sinks or max_logit_value.")

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
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if num_q_heads % num_stacked_q_heads != 0:
    raise ValueError(f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}")

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if q_heads_per_kv_head % num_stacked_q_heads != 0:
    raise ValueError(f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}")

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same " "leading dimensions.")

  if bkv % bkv_compute:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
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
    def index_map(h_block, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h_block, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h_block, i, j):
      del i  # Unused.
      first_h_in_block = h_block * num_stacked_q_heads
      prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(lambda h_block, i, j: from_head_minor((h_block, i, 0), q_layout))
  out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(h_block, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h_block, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
          q_index_map,
      ),
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
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)  # pyrefly: ignore[bad-argument-type]
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)  # pyrefly: ignore[bad-argument-type]
    )
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
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,))
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

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
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  out_shapes = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(lambda h_block, i, j, *_: (h_block, i, 0))

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32) if fuse_reciprocal else None,
        # l_linear
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32) if not fuse_reciprocal else None,
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]
    out_specs += [
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map) if fuse_reciprocal else None,
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map) if not fuse_reciprocal else None,
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map),
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
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      *vmem_inputs, out_shapes, fwd_mask_sparsity  # pyrefly: ignore[bad-argument-count, bad-argument-type]
  )

  grid_size_h = num_q_heads // num_stacked_q_heads
  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (grid_size_h, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (grid_size_h, kv_steps * (q_seq_len // bq))
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
            num_stacked_q_heads=num_stacked_q_heads,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM((num_stacked_q_heads, bq, NUM_LANES), jnp.float32),  # m_scratch
                pltpu.VMEM((num_stacked_q_heads, bq, NUM_LANES), jnp.float32),  # l_scratch
                pltpu.VMEM((num_stacked_q_heads, bq, head_dim_v), jnp.float32),  # o_scratch
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
        prng_key,
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

      l = l_linear[..., 0]
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
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
    prng_key: jax.Array | None = None,
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

  ret = _splash_attention_forward(
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
      prng_key=prng_key,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      stats["max_logits"] = stats["max_logits"] / LOG2E
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
    prng_key: jax.Array | None = None,
) -> tuple[tuple[jax.Array], base.SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats = _splash_attention_forward(
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
      prng_key=prng_key,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats["logsumexp"] = stats["logsumexp"] / LOG2E
    stats["max_logits"] = stats["max_logits"] / LOG2E
  # `prng_key` is a residual so the backward regenerates the identical dropout
  # mask; it is the key itself, not the mask, so this costs nothing in HBM.
  residuals = (
      q,
      k,
      v,
      segment_ids,
      sinks,
      out,
      logsumexp,
      dkv_mask_info,
      prng_key,
  )
  if save_residuals:
    return (out, stats), residuals  # pyrefly: ignore[bad-return]
  else:
    return out, residuals  # pyrefly: ignore[bad-return]


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
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
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

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)  # pyrefly: ignore[unsupported-operation]
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
    prng_key_ref,
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
  del mask_next_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
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
    if dropout_rate:
      # Mirrors the fwd indices: rows are kv blocks and cols are q blocks here.
      q_head = pl.program_id(0)
      q_index = active_cols_ref[grid_idx].astype(jnp.int32)
    else:
      q_head = 0
      q_index = 0
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

  # TODO: Update docstring explaining the accumulation logic

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
    dropout_mask = None
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
    _g = config.qk_diag_grid
    if config.qk_diag_skip and has_partial_mask and bkv_compute % _g == 0 and bq % _g == 0:
      # Diagonal skip (backward dkv): qk tile is [kv, q]. On an aligned square diagonal
      # block, sub-tile (kv-band ki, q-band qj) with ki > qj is fully above the causal
      # boundary (kv > q) -> overwritten to mask_value anyway -> skip its matmul; compute
      # only ki <= qj sub-tiles; assemble the full tile for the single exp/ds/dv/dk.
      sk = bkv_compute // _g
      sq = bq // _g
      k_parts = [k[i * sk : (i + 1) * sk, :] for i in range(_g)]
      q_parts = [scaled_q[j * sq : (j + 1) * sq, :] for j in range(_g)]
      _mm = lambda kk, qq: lax.dot_general(kk, qq, qk_dims, preferred_element_type=jnp.float32)
      rows = []
      for ki in range(_g):  # kv row-band
        cols = []
        for qj in range(_g):  # q col-band
          if ki > qj:  # fully masked -> skip matmul
            cols.append(jnp.full((sk, sq), mask_value, dtype=jnp.float32))
          else:
            cols.append(_mm(k_parts[ki], q_parts[qj]))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_uncapped = jnp.concatenate(rows, axis=0)
    else:
      qk_uncapped = lax.dot_general(k, scaled_q, qk_dims, preferred_element_type=jnp.float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)

    if dropout_rate:
      # Regenerated, not saved: the canonical blocks this tile covers are the
      # same ones the forward covered over the same (query, key) range, so the
      # bits match even though the backward tiles the matrix differently. Here
      # the tile is [kv, q] rather than [q, kv], hence the transpose (only
      # float32 transposes lower).
      global_kv_block_idx = kv_index * (bkv // bkv_compute) + i
      dropout_mask = _dropout_mask_tile(
          prng_key_ref,
          head_idx=q_head,
          q_block_idx=q_index,
          kv_block_idx=global_kv_block_idx,
          q_block_size=bq,
          kv_block_size=bkv_compute,
          canonical_q=config.active_dropout_block_q,
          canonical_kv=config.active_dropout_block_kv,
          dropout_rate=dropout_rate,
      )
      dropout_mask = dropout_mask.astype(jnp.float32).T.astype(jnp.bool_)
      # dv sees the dropped weights, matching the forward's numerator.
      pr = jnp.where(dropout_mask, 0.0, p) / (1.0 - dropout_rate)
    else:
      pr = p

    dv = lax.dot(pr.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v,
        do,
        NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    if dropout_rate:
      assert dropout_mask is not None
      dp = jnp.where(dropout_mask, 0.0, dp) / (1.0 - dropout_rate)
    # `p` here is deliberately the *undropped* softmax weight: `di` is
    # rowsum(do * o), which already equals rowsum(dp_dropped * p).
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
    prng_key: jax.Array | None = None,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  prng_key = _check_dropout_args(config, prng_key)
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
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
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
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

    grid_size = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
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
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

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
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), q.dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
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
    dk_type = k.dtype
    dv_type = v.dtype

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
      prng_key,
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
      mask_info.partial_mask_blocks,  # pyrefly: ignore[bad-argument-type]
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
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
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    base.SegmentIds | None,  # segment_ids
    jax.Array | None,  # segment_ids
    jax.Array | None,  # max_logit_estimate
    jax.Array | None,  # prng_key
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
  q, k, v, segment_ids, sinks, o, logsumexp, dkv_mask_info, prng_key = res

  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pyrefly: ignore[missing-attribute]
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,  # pyrefly: ignore[bad-argument-type]
      bkv=bkv_dkv_memory,  # pyrefly: ignore[bad-argument-type]
      bkv_compute=bkv_dkv_compute,  # pyrefly: ignore[bad-argument-type]
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,  # pyrefly: ignore[bad-argument-type]
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
      prng_key=prng_key,
  )
  dsinks = None
  if sinks is not None:
    logsumexp_ = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
    sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - logsumexp_[..., None].astype(jnp.float32))
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))  # pyrefly: ignore[bad-argument-type]
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
      None,  # prng_key
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
    prng_key: jax.Array | None = None,
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
      prng_key=prng_key,
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
    mask_info_specs = MaskInfo(
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),  # pyrefly: ignore[bad-argument-type]
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),  # pyrefly: ignore[bad-argument-type]
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),  # pyrefly: ignore[bad-argument-type]
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),  # pyrefly: ignore[bad-argument-type]
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),  # pyrefly: ignore[bad-argument-type]
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated  # pyrefly: ignore[bad-argument-type]
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),  # pyrefly: ignore[bad-argument-type]
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
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

  if (config.qk_diag_skip or config.sv_diag_skip) and not isinstance(mask, mask_lib.CausalMask):
    # The skip assumes kv > q is ALWAYS masked — a pure-causal property. Any mask
    # that admits a valid kv > q entry (bidirectional, local/sliding window, custom)
    # would be silently corrupted, so fail loud. (Square-block preconditions are
    # enforced in SplashConfig.__post_init__.)
    param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
    raise ValueError(
        f"{param_name}=True requires a pure CausalMask (the skip assumes the"
        f" kv > q region is masked); got {type(mask).__name__}. Disable"
        f" {param_name} for non-causal masks."
    )

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

  mask_info_specs = MaskInfo(
      mask_next=mask_spec,  # pyrefly: ignore[bad-argument-type]
      active_rows=None,
      active_cols=None,
      num_active_blocks=None,
      block_mask=mask_spec,  # pyrefly: ignore[bad-argument-type]
      partial_mask_blocks=mask_spec,  # pyrefly: ignore[bad-argument-type]
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
