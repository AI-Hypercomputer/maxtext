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

from collections.abc import Callable, Sequence
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
import numpy as np

try:
  from tokamax._src.ops.splash_attention import base
  from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib
  from tokamax._src.ops.splash_attention import splash_attention_mask_info as mask_info_lib
except ImportError:
  from tokamax._src.ops.experimental.tpu.splash_attention import base
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as mask_info_lib

P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)


def _is_fp8(x) -> bool:
  return x.dtype == jnp.float8_e4m3fn


def _unquantized_dtype(x):
  """bf16 for fp8 inputs; otherwise the input dtype (pre-cl/978108475 behavior)."""
  return jnp.bfloat16 if _is_fp8(x) else x.dtype


def dkv_q_blocks_are_contiguous(mask_info: mask_info_lib.MaskInfo, q_blocks_per_shard: int) -> bool:
  """Returns whether VMEM dq reduction (`dq_reduction_steps=1`) is safe.

  It keeps a q block's dq in VMEM across consecutive dkv grid steps and writes
  it once, so in every shard each q block must appear in exactly one run of
  consecutive steps. Trivially true without a dynamic grid, where it is unused.

  Args:
    mask_info: dKV `MaskInfo` with concrete (non-traced) arrays.
    q_blocks_per_shard: Number of q blocks in each q sequence shard.
  """
  if mask_info.active_cols is None:
    return True
  num_active = np.asarray(mask_info.num_active_blocks).reshape(-1)
  shard_cols = np.asarray(mask_info.active_cols).reshape(num_active.size, -1)
  for cols, n in zip(shard_cols, num_active):
    cols = cols[:n].astype(np.int32)
    run_starts = cols[np.diff(cols, prepend=-1) != 0]
    if not np.array_equal(np.sort(run_starts), np.arange(q_blocks_per_shard)):
      return False
  return True


LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


def _make_unvmap_primitive(name: str, reduce_fn: Callable[..., jax.Array]) -> jax.extend.core.Primitive:
  """Primitive that reduces `vmap` batch dims so `MaskInfo` stays unbatched."""
  prim = jax.extend.core.Primitive(name)
  prim.def_impl(lambda x: x)
  prim.def_abstract_eval(lambda x: jax.core.ShapedArray(x.shape, x.dtype))
  jax.interpreters.batching.primitive_batchers[prim] = lambda xs, bdims: (
      prim.bind(xs[0] if bdims[0] is None else reduce_fn(xs[0], axis=bdims[0])),
      None,
  )
  jax.interpreters.mlir.register_lowering(prim, jax.interpreters.mlir.lower_fun(lambda x: x, multiple_results=False))
  return prim


_unvmap_any_p = _make_unvmap_primitive("splash_unvmap_any", jnp.any)


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
    n_q_blocks: int | jax.Array,
    n_kv_blocks: int | jax.Array,
) -> jax.Array:
  """Generates the dropout mask of a single (head, q block, kv block) tile.

  The mask is a pure function of the key and the block coordinates, so the
  forward and backward kernels regenerate the identical mask without either
  materializing the full [heads, q, kv] mask in HBM or saving it as a residual.
  Returns float32 weights in {0.0, 1.0 / (1.0 - dropout_rate)}.

  The block coordinates are indices into the canonical dropout grid, which is
  independent of the kernel's block sizes; callers go through
  `_dropout_mask_tile` rather than calling this directly.
  """
  tile_id = (head_idx * n_q_blocks + q_block_idx) * n_kv_blocks + kv_block_idx
  sub_key = jax.random.fold_in(prng_key[...], tile_id)

  # Direct integer threshold mask generation in uint32 ALU, avoiding float32 round-trip.
  threshold = np.uint32(dropout_rate * (2**32))
  bits = jax.random.bits(sub_key, (q_block_size, kv_block_size), dtype=jnp.uint32)
  dropped = bits < threshold

  # Pre-scaled mask returns float multiplier directly.
  scale = jnp.float32(1.0 / (1.0 - dropout_rate))
  return jnp.where(dropped, 0.0, scale)


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
    n_q_blocks: int | jax.Array,
    n_kv_blocks: int | jax.Array,
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

  # Fast-path: single canonical block tile (1:1), avoids loop,
  # list, and concat overhead.
  if nq == 1 and nkv == 1:
    return _generate_blockwise_dropout_mask(
        prng_key,
        head_idx=head_idx,
        q_block_idx=q_block_idx,
        kv_block_idx=kv_block_idx,
        q_block_size=canonical_q,
        kv_block_size=canonical_kv,
        dropout_rate=dropout_rate,
        n_q_blocks=n_q_blocks,
        n_kv_blocks=n_kv_blocks,
    )

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
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
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


def _above_diag(q_band: int, q_size: int, kv_band: int, kv_size: int) -> bool:
  """Whether a diagonal-block sub-tile lies entirely in kv > q (masked)."""
  return kv_band * kv_size >= (q_band + 1) * q_size


def _resolve_diag_grids(
    grid: Any,
) -> tuple[tuple[int, int], tuple[int, int]]:
  """Resolves `SplashConfig.qk_diag_grid` to ((fwd q, kv), (dkv q, kv))."""

  def is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

  if is_int(grid):
    return (int(grid), int(grid)), (int(grid), int(grid))
  if isinstance(grid, Sequence) and len(grid) == 2:
    if all(is_int(g) for g in grid):
      pair = (int(grid[0]), int(grid[1]))
      return pair, pair
    if all(isinstance(g, Sequence) and len(g) == 2 and all(is_int(n) for n in g) for g in grid):
      fwd, dkv = grid
      return (int(fwd[0]), int(fwd[1])), (int(dkv[0]), int(dkv[1]))
  raise ValueError(
      "qk_diag_grid must be an int, a (q, kv) pair, or a pair of (q, kv)" f" pairs (forward, dkv); got {grid!r}."
  )


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

  # TODO(b/461754630): Remove these 3 params, they're only kept for backwards compatibility.
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
  # reduction steps. For now, only 3 or all the kv steps are supported. 1
  # accumulates dq in VMEM; it requires a dynamic dkv grid that visits every q
  # block in one run of consecutive steps (see
  # `dkv_q_blocks_are_contiguous`).
  dq_reduction_steps: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False
  # Skip the wasted causal-diagonal work. Bit-exact: the output, dQ, dK and dV
  # are identical to qk_diag_skip=False.
  # Forward: requires block_q == block_kv (block_kv_compute may be smaller). On
  # a diagonal block, compute tile c covers kv [c * bkc, (c + 1) * bkc) of the
  # block (bkc = block_kv_compute). q rows [0, c * bkc) see no kv in it, so
  # their QK, softmax and SV are skipped (an exact no-op for the online
  # softmax). Rows [c * bkc, (c + 1) * bkc) form the causal triangle: the QK
  # matmul is split into a qk_diag_grid sub-grid and sub-tiles fully above the
  # diagonal are filled with mask_value instead of being computed, and the SV
  # matmul runs on the visible kv prefix of each row band. Rows below the
  # triangle are one dense matmul.
  # Backward (dkv): requires square tiles (block_q_dkv == block_kv_dkv ==
  # block_kv_dkv_compute); the QK^T sub-tiles fully above the diagonal are
  # skipped the same way.
  # PRECONDITION (enforced): pure CausalMask with offset 0, and the block
  # conditions above. Any other config raises; it never silently corrupts.
  qk_diag_skip: bool = False
  # Sub-grid of the diagonal skip, as (q splits, kv splits) of the triangle:
  #   g (int)         -> (g, g) in both phases (default 2).
  #   (gq, gkv)       -> the same grid in both phases.
  #   [(gq, gkv), (gq, gkv)] -> forward grid, dkv grid.
  # A finer grid skips more of the triangle but issues smaller matmuls. Every
  # entry must be a power of 2 (1 = no split along that dimension). Also used
  # by sv_diag_skip (forward grid).
  qk_diag_grid: int | tuple[int, int] | Sequence[tuple[int, int]] = 2
  # Skip the wasted causal-diagonal SV matmul. On a partial-mask (diagonal)
  # block, split the SV matmul into the forward qk_diag_grid sub-grid and
  # skip sub-tiles fully above the diagonal during dot_general. Redundant
  # with qk_diag_skip, which already skips SV in the forward.
  sv_diag_skip: bool = False
  # With qk_diag_skip, run the dkv kernel's causal diagonal block as
  # qk_diag_grid kv row-bands, each over only the queries it attends to, so the
  # softmax and dv/dp/dk/dq work also skip the strictly-above-diagonal tiles.
  # The bands accumulate dq into the VMEM dq scratch, so in the fused kernel
  # this needs dq_reduction_steps=1 (qk_diag_skip forces square dkv blocks,
  # which otherwise have no dq scratch); without it the stock path is used.
  diag_band_skip: bool = False
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
  # [experimental] In the fused dkv kernel, reuse the
  # LOG2E-scaled q for the dK matmul and undo the scale with dk /= LOG2E
  # (exact grad w.r.t. the rounded fp8 q; one fewer q tile live).
  # False = previous behavior.
  dkv_reuse_scaled_q: bool = False
  # [experimental] q arrives already multiplied by LOG2E (the caller folds it
  # into the softmax scale before the fp8 cast, so q is rounded to fp8 once).
  # The kernel skips its own q scaling and returns dq, dk w.r.t. the
  # pre-scaled input (both divided by LOG2E). Subsumes dkv_reuse_scaled_q.
  q_prescaled_log2e: bool = False

  @property
  def qk_diag_grid_fwd(self) -> tuple[int, int]:
    return _resolve_diag_grids(self.qk_diag_grid)[0]

  @property
  def qk_diag_grid_dkv(self) -> tuple[int, int]:
    return _resolve_diag_grids(self.qk_diag_grid)[1]

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

    if self.dq_reduction_steps not in (None, 1, 3):
      raise ValueError(f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 1, 3 or" " None are supported.")
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")
    if self.q_prescaled_log2e and not self.use_base2_exp:
      raise ValueError("q_prescaled_log2e requires use_base2_exp=True.")

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

    # Validate first (this also accepts np.integer entries), then store the
    # canonical form: the config is a static jit argument, so it must hash
    # (lists become tuples).
    fwd_grid, dkv_grid = _resolve_diag_grids(self.qk_diag_grid)
    if isinstance(self.qk_diag_grid, Sequence):
      canonical_grid = tuple(tuple(int(n) for n in g) if isinstance(g, Sequence) else int(g) for g in self.qk_diag_grid)
    else:
      canonical_grid = int(self.qk_diag_grid)
    object.__setattr__(self, "qk_diag_grid", canonical_grid)
    if self.qk_diag_skip or self.sv_diag_skip:
      for phase, grid in (("forward", fwd_grid), ("dkv", dkv_grid)):
        # 1 is allowed: no split along that dimension.
        if any(g < 1 or g & (g - 1) for g in grid):
          raise ValueError(f"qk_diag_grid entries must each be a power of 2; got {phase}" f" grid {grid}.")
      assert self.block_kv_compute is not None
      if any(self.block_kv_compute % g for g in fwd_grid):
        raise ValueError(f"Forward qk_diag_grid {fwd_grid} must divide" f" block_kv_compute={self.block_kv_compute}.")

    if self.qk_diag_skip:
      # The skip fills mask_value / skips work exactly where kv > q, so it needs
      # the diagonal blocks to be aligned (block_q == block_kv); causality and
      # offset 0 are checked in _make_splash_attention.
      assert self.block_kv_compute is not None
      if self.block_q != self.block_kv:
        raise ValueError(
            "qk_diag_skip requires block_q == block_kv in the forward; got" f" {self.block_q}/{self.block_kv}."
        )
      if self.block_kv % self.block_kv_compute:
        raise ValueError(
            f"qk_diag_skip requires block_kv_compute={self.block_kv_compute}" f" to divide block_kv={self.block_kv}."
        )
      if self.has_backward_blocks:
        assert self.block_q_dkv is not None
        assert self.block_kv_dkv_compute is not None
        if not (self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute):
          raise ValueError(
              "qk_diag_skip requires square backward blocks "
              "(block_q_dkv == block_kv_dkv == block_kv_dkv_compute); got "
              f"{self.block_q_dkv}/{self.block_kv_dkv}/"
              f"{self.block_kv_dkv_compute}."
          )
        if self.block_q_dkv % dkv_grid[0] or self.block_kv_dkv_compute % dkv_grid[1]:
          raise ValueError(
              f"dkv qk_diag_grid {dkv_grid} must divide the dkv blocks"
              f" {self.block_q_dkv}/{self.block_kv_dkv_compute}."
          )

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
    # TODO(b/461753221): Select better parameters based on a heuristic.
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
    q_steps: int,
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
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
    # Dropout needs the q block index too; it is only tracked on the dynamic
    # grid, where `grid_idx` is a position in the sparse (i, j) work list.
    i = active_rows_ref[grid_idx].astype(jnp.int32) if dropout_rate else 0
  else:
    should_not_mask = False
    should_run = True
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
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
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
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      n_q_blocks = q_steps * (bq // canonical_q)
      n_kv_blocks = kv_steps * (bkv // canonical_kv)
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
                canonical_q=canonical_q,
                canonical_kv=canonical_kv,
                dropout_rate=dropout_rate,
                n_q_blocks=n_q_blocks,
                n_kv_blocks=n_kv_blocks,
            )
        )
      dropout_scale = jnp.stack(head_masks, axis=0)
      s_curr = s_curr * dropout_scale

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * bq, bkv_compute))

    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS

    _g, gk = config.qk_diag_grid_fwd
    gq = _g
    if (
        config.sv_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.v_layout == HEAD_DIM_MINOR
        and bq % gq == 0
        and bkv_compute % gk == 0
    ):
      sq = bq // gq
      sk = bkv_compute // gk
      v_parts = [v[t * sk : (t + 1) * sk, :] for t in range(gk)]
      o_rows = []
      for qi in range(gq):
        o_row = None
        for kj in range(gk):
          if _above_diag(qi, sq, kj, sk):
            continue
          part = lax.dot_general(
              s_curr_flat[qi * sq : (qi + 1) * sq, kj * sk : (kj + 1) * sk],
              v_parts[kj],
              sv_dims,
              preferred_element_type=jnp.float32,
          )
          o_row = part if o_row is None else o_row + part
        o_rows.append(o_row)
      o_curr_flat = jnp.concatenate(o_rows, axis=0)
    else:
      o_curr_flat = lax.dot_general(s_curr_flat, v, sv_dims, preferred_element_type=jnp.float32)
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, bq, head_dim_v))

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  def body_diag(tile: int) -> None:
    """`qk_diag_skip` forward on a diagonal block, static compute tile `tile`.

    The block is aligned (bq == bkv, causal, offset 0), so the tile covers kv
    [tile * bkv_compute, (tile + 1) * bkv_compute) of the block and q rows
    [0, r0) with r0 = tile * bkv_compute see none of it: their m and l stay
    unchanged and their o gets nothing added, exactly what the online softmax
    would do with an all-masked row (alpha = 1, s = 0). Only rows [r0, bq) are
    processed.

    Args:
      tile: Static index of the compute tile along the kv block.
    """
    r0 = tile * bkv_compute
    nq = bq - r0
    gq, gk = config.qk_diag_grid_fwd
    sq, sk = bkv_compute // gq, bkv_compute // gk
    slice_k = pl.ds(tile * bkv_compute, bkv_compute)
    m_prev = m_scratch_ref[:, r0:bq, :]
    l_prev = l_scratch_ref[:, r0:bq, :]

    if config.q_layout == HEAD_DIM_MINOR:
      q = q_ref[:, r0:bq, :]
    else:
      q = q_ref[:, :, r0:bq].mT
    if config.use_base2_exp and not config.q_prescaled_log2e:  # As in `body`.
      if q.dtype == jnp.float8_e4m3fn:
        q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
        q *= LOG2E
    q_flat = q.reshape((nq, q.shape[-1]))

    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
      qk_dims = NT_DIM_NUMBERS
      k_parts = [k[t * sk : (t + 1) * sk, :] for t in range(gk)]
    else:
      k = k_ref[:, slice_k]
      qk_dims = NN_DIM_NUMBERS
      k_parts = [k[:, t * sk : (t + 1) * sk] for t in range(gk)]

    # Rows [0, bkv_compute) of the slice are the causal triangle: compute the
    # sub-tiles on or below the diagonal, fill the rest with mask_value (which
    # the mask would write there anyway).
    qk_rows = []
    for qi in range(gq):
      q_part = q_flat[qi * sq : (qi + 1) * sq, :]
      cols = []
      for kj in range(gk):
        if _above_diag(qi, sq, kj, sk):
          cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
        else:
          cols.append(lax.dot_general(q_part, k_parts[kj], qk_dims, preferred_element_type=float32))
      qk_rows.append(jnp.concatenate(cols, axis=1))
    if nq > bkv_compute:  # Rows below the triangle see the whole tile.
      qk_rows.append(
          lax.dot_general(
              q_flat[bkv_compute:, :],
              k,
              qk_dims,
              preferred_element_type=float32,
          )
      )
    qk = jnp.concatenate(qk_rows, axis=0).reshape((1, nq, bkv_compute))

    rows = lambda ref: None if ref is None else ref.at[r0:bq]
    qk = _apply_mask_and_soft_cap(
        qk,
        mask_value,
        rows(mask_ref),
        rows(q_sequence_ref),
        rows(q_segment_ids_ref),
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + tile * bkv_compute,
        bq=nq,
        mask_function=mask_function,
        has_partial_mask=True,
    )

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]  # pyrefly: ignore[missing-attribute]
      m_next = jnp.maximum(m_prev, m_curr)
      bkv_repeats = bkv_compute // NUM_LANES
      s_curr = exp(
          qk - jnp.tile(m_next, (1, 1, bkv_repeats))
      )  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    else:
      m_next = None
      s_curr = exp(qk - max_logit_estimate)  # pyrefly: ignore[unsupported-operation]
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      m_scratch_ref[:, r0:bq, :] = m_next
      l_scratch_ref[:, r0:bq, :] = l_curr + alpha * l_prev
    else:
      alpha = None
      l_scratch_ref[:, r0:bq, :] = l_curr + l_prev

    if dropout_rate:
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      dropout_scale = _dropout_mask_tile(
          prng_key_ref,
          head_idx=h,
          q_block_idx=i,
          kv_block_idx=j * (bkv // bkv_compute) + tile,
          q_block_size=bq,
          kv_block_size=bkv_compute,
          canonical_q=canonical_q,
          canonical_kv=canonical_kv,
          dropout_rate=dropout_rate,
          n_q_blocks=q_steps * (bq // canonical_q),
          n_kv_blocks=kv_steps * (bkv // canonical_kv),
      )[r0:, :]
      s_curr = s_curr * dropout_scale[None]

    # SV over the triangle: each row band only multiplies the kv prefix it can
    # see (the rest of its s is exactly 0). One dot per band over the prefix,
    # not a sum of per-sub-tile dots, keeps the f32 accumulation identical to
    # the dense matmul. Bands are at least 256 rows: on TPU7x an M=128 dot
    # rounds differently from the dense one (o off by 1 bf16 ulp).
    s_flat = s_curr.reshape((nq, bkv_compute))
    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS
    sv_bands = 1  # Largest power of 2 <= gq that keeps bands >= 256 rows.
    while sv_bands * 2 <= gq and bkv_compute // (sv_bands * 2) >= 256:
      sv_bands *= 2
    band = bkv_compute // sv_bands
    o_rows = []
    for b in range(sv_bands):
      kv_end = (b + 1) * band
      if config.v_layout == HEAD_DIM_MINOR:
        v_pre = v[:kv_end, :]
      else:
        v_pre = v[:, :kv_end]
      o_rows.append(
          lax.dot_general(
              s_flat[b * band : (b + 1) * band, :kv_end],
              v_pre,
              sv_dims,
              preferred_element_type=jnp.float32,
          )
      )
    if nq > bkv_compute:
      o_rows.append(
          lax.dot_general(
              s_flat[bkv_compute:, :],
              v,
              sv_dims,
              preferred_element_type=jnp.float32,
          )
      )
    o_curr = jnp.concatenate(o_rows, axis=0).reshape((1, nq, head_dim_v))

    o_prev = o_scratch_ref[:, r0:bq, :]
    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[:, r0:bq, :] = alpha_o * o_prev + o_curr
    else:
      o_scratch_ref[:, r0:bq, :] = o_prev + o_curr

  assert bkv % bkv_compute == 0
  num_iters = k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  # body_diag may only run on diagonal blocks, so the partial blocks must be
  # known: that needs the block mask (without it every block is "partial").
  use_body_diag = config.qk_diag_skip and block_mask_ref is not None and num_stacked_q_heads == 1 and bq == bkv

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(jnp.logical_not(should_not_mask), should_run))
  def _():
    if use_body_diag:
      for c in range(num_iters):  # Static c: body_diag slices rows by it.
        body_diag(c)
    else:
      lax.fori_loop(0, num_iters, partial(body, has_partial_mask=True), None, unroll=True)

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = jnp.tile(jnp.where(l == 0.0, 0.0, 1.0 / l), (1, 1, head_dim_v_repeats))
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


def _segment_block_overlap(
    segment_ids: base.SegmentIds,
    q_idx: jax.Array,
    kv_idx: jax.Array,
    *,
    q_blocks: int,
    kv_blocks: int,
    bq: int,
    bkv: int,
) -> jax.Array:
  """Returns whether each `(q_idx, kv_idx)` tile can share any segment id.

  Conservative: it tests whether the `[min, max]` id intervals of the two
  blocks intersect. If the blocks really do share an id, that id lies in both
  intervals, so the intervals must intersect. There are therefore no false
  negatives and pruning on this result never drops a contributing tile. False
  positives (e.g. q ids `{1, 5}` against kv id `{3}`) merely leave a fully
  masked tile scheduled. This argument does not assume ids are sorted.

  The result is reduced over any `vmap` batch dimension so the whole batch
  shares one schedule, which is what keeps `pallas_call` batch fusion intact.

  Args:
    segment_ids: Unbatched q / kv segment ids.
    q_idx: Integer array of q block indices.
    kv_idx: Integer array of kv block indices, broadcastable against `q_idx`.
    q_blocks: Number of q blocks, `q_seq_len // bq`.
    kv_blocks: Number of kv blocks, `kv_seq_len // bkv`.
    bq: q block size.
    bkv: kv block size.

  Returns:
    A boolean array of the broadcast shape of `q_idx` and `kv_idx`.
  """
  q_segs = segment_ids.q.reshape(q_blocks, bq)  # pyrefly: ignore[missing-attribute]
  kv_segs = segment_ids.kv.reshape(kv_blocks, bkv)  # pyrefly: ignore[missing-attribute]
  q_lo = _onehot_lookup(jnp.min(q_segs, axis=-1), q_idx)
  q_hi = _onehot_lookup(jnp.max(q_segs, axis=-1), q_idx)
  kv_lo = _onehot_lookup(jnp.min(kv_segs, axis=-1), kv_idx)
  kv_hi = _onehot_lookup(jnp.max(kv_segs, axis=-1), kv_idx)
  return _unvmap_any_p.bind((q_hi >= kv_lo) & (kv_hi >= q_lo))


def _onehot_lookup(table: jax.Array, idx: jax.Array) -> jax.Array:
  """Returns `table[idx]` for in-range `idx` without emitting an XLA gather.

  On TPU7x, XLA offloads large gathers to SparseCore, whose lowering fails for
  the schedule-sized index arrays used here (T >= 16K). A one-hot select-reduce
  over the last axis is the same `O(len(idx) * len(table))` shape as `row_eq`
  below and fuses into a single elementwise-plus-reduce.

  Args:
    table: `[n]` array (may carry a leading `vmap` batch dim).
    idx: Integer array with values in `[0, n)`.

  Returns:
    An array of `idx.shape` with `table`'s dtype.
  """
  onehot = idx[..., None] == jnp.arange(table.shape[-1], dtype=idx.dtype)
  return jnp.sum(jnp.where(onehot, table, jnp.zeros_like(table)), axis=-1)


def _refine_mask_info_with_segments(
    mask_info: MaskInfo,
    segment_ids: base.SegmentIds,
    *,
    q_blocks: int,
    kv_blocks: int,
    bq: int,
    bkv: int,
    is_dkv: bool,
) -> tuple[
    MaskInfo,
    jax.Array | np.ndarray | None,
    jax.Array | np.ndarray | None,
]:
  """Prunes `mask_info` tiles whose q and kv `segment_ids` ranges are disjoint."""
  num_rows, num_cols = (kv_blocks, q_blocks) if is_dkv else (q_blocks, kv_blocks)
  if mask_info.active_rows is None:
    m_size = num_rows * num_cols
    rows, cols = jnp.unravel_index(jnp.arange(m_size, dtype=jnp.int32), (num_rows, num_cols))
    num_active = jnp.array([m_size], dtype=jnp.int32)
  else:
    rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
    cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
    m_size = rows.size
    num_active = jnp.asarray(mask_info.num_active_blocks, dtype=jnp.int32)

  block_mask = (
      jnp.full((m_size,), 2, dtype=jnp.int8)
      if mask_info.block_mask is None
      else jnp.asarray(mask_info.block_mask).reshape(m_size)
  )
  idx = jnp.arange(m_size, dtype=jnp.int32)
  valid_static = (block_mask > 0) & (idx < num_active[0])

  rows_c = jnp.clip(rows, 0, num_rows - 1)
  cols_c = jnp.clip(cols, 0, num_cols - 1)
  seg_any = _segment_block_overlap(
      segment_ids,
      cols_c if is_dkv else rows_c,
      rows_c if is_dkv else cols_c,
      q_blocks=q_blocks,
      kv_blocks=kv_blocks,
      bq=bq,
      bkv=bkv,
  )

  active = valid_static & seg_any
  # Every output row needs >= 1 scheduled step (with block_mask=0 if inactive)
  # so `bounds_start`/`bounds_end` still zero-initialize and write that row.
  # Per-tile row lookups go through the `row_eq` one-hot rather than
  # `table[rows_c]`, and compaction below uses `lax.sort` rather than
  # `x[argsort(...)]`: this function emits no XLA gather (see `_onehot_lookup`).
  row_eq = rows_c[:, None] == jnp.arange(num_rows, dtype=jnp.int32)[None, :]
  row_has_active = jnp.any(row_eq & active[:, None], axis=0)
  first_in_row = jnp.argmax(row_eq, axis=0).astype(jnp.int32)
  keep = active | (
      (is_dkv | jnp.any(active))
      & ~jnp.any(row_eq & row_has_active[None, :], axis=1)
      & (idx == jnp.sum(jnp.where(row_eq, first_in_row[None, :], 0), axis=1))
  )

  block_mask = jnp.where(active, block_mask, 0).astype(block_mask.dtype)
  mask_next = None if mask_info.mask_next is None else jnp.asarray(mask_info.mask_next).reshape(m_size)

  # A stable sort on `~keep` moves kept tiles to the front in schedule order.
  new_num_active = jnp.sum(keep.astype(jnp.int32), keepdims=True)
  payload = (rows_c, cols_c, block_mask) + (() if mask_next is None else (mask_next,))
  _, *payload = lax.sort(((~keep).astype(jnp.int32), *payload), num_keys=1, is_stable=True)
  is_active_slot = idx < new_num_active[0]
  rows_s, cols_s, block_mask, *mask_next_s = [jnp.where(is_active_slot, x, 0).astype(x.dtype) for x in payload]
  bounds_start, bounds_end = mask_info_lib.find_bounds(jnp.where(is_active_slot, payload[0], -1))
  row_dtype = jnp.int32 if mask_info.active_rows is None else mask_info.active_rows.dtype
  col_dtype = jnp.int32 if mask_info.active_cols is None else mask_info.active_cols.dtype
  return (
      mask_info._replace(
          active_rows=rows_s.astype(row_dtype),
          active_cols=cols_s.astype(col_dtype),
          block_mask=block_mask,
          mask_next=mask_next_s[0] if mask_next_s else None,
          num_active_blocks=new_num_active,
      ),
      bounds_start,
      bounds_end,
  )


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

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError("Invalid shape for q segment_ids: " f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}")
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError("Invalid shape for kv segment_ids: " f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}")
    mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
        mask_info,
        segment_ids,
        q_blocks=q_seq_len // bq,
        kv_blocks=kv_steps,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
    )
  else:
    bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  dynamic_grid = mask_info.active_rows is not None

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
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), _unquantized_dtype(q)),
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
            q_steps=q_seq_len // bq,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,  # pyrefly: ignore[bad-argument-type]
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
    return out, stats  # pyrefly: ignore[bad-return]
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

  # TODO(b/461753121): add some higher order AD check that isn't save_residuals based.
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
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
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
        k_slice=pl.ds(0, bkv),
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)  # pyrefly: ignore[unsupported-operation]
    dp_dims = NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    dp = lax.dot_general(
        do if _is_fp8(v) else do.astype(v.dtype),
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
        ds if _is_fp8(k) else ds.astype(k.dtype),
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
    kv_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: SplashConfig,
    use_vmem_dq_reduction: bool = False,
):
  del mask_next_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  should_initialize_dq = should_write_dq = True
  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    if use_vmem_dq_reduction:
      # A q block's kv steps are consecutive, so dq stays in VMEM across them.
      col = lambda g: active_cols_ref[g].astype(jnp.int32)
      last = pl.num_programs(1) - 1
      should_initialize_dq = jnp.logical_or(grid_idx == 0, col(jnp.maximum(grid_idx - 1, 0)) != col(grid_idx))
      should_write_dq = jnp.logical_or(
          grid_idx == last,
          col(jnp.minimum(grid_idx + 1, last)) != col(grid_idx),
      )
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

  # TODO(b/447656860): Update docstring explaining the accumulation logic

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
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        scaled_q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
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
    gq, gk = config.qk_diag_grid_dkv
    sk_band, sq_band = bkv_compute // gk, bq // gq
    if (
        config.qk_diag_skip
        and config.diag_band_skip
        and has_partial_mask
        and not dropout_rate
        and (dq_scratch_ref is not None or dq_ref is None)
        and gq == gk
        and bkv_compute % gk == 0
        and bq % gq == 0
        and sk_band % NUM_LANES == 0
        and sq_band % NUM_LANES == 0
    ):
      _g = gq
      sk, sq = sk_band, sq_band
      # Row-band diagonal skip: on an aligned causal diagonal block, kv band ki
      # only attends to queries [ki * sq, bq), so run the backward for each band
      # on that contiguous query suffix instead of the full tile.
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      dk_dims = NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
      q_view = lambda ref, s: None if ref is None else ref.at[:, s]
      for ki in range(_g):
        q0, k0 = ki * sq, ki * sk
        qs = pl.ds(q0, bq - q0)
        if config.q_layout == HEAD_DIM_MINOR:
          q_b, scaled_q_b = q[q0:, :], scaled_q[q0:, :]
        else:
          q_b, scaled_q_b = q[:, q0:], scaled_q[:, q0:]
        k_b, v_b, do_b = k[k0 : k0 + sk, :], v[k0 : k0 + sk, :], do[q0:, :]
        qk_b = lax.dot_general(k_b, scaled_q_b, qk_dims, preferred_element_type=jnp.float32)
        p_b = exp(
            _apply_mask_and_soft_cap(
                qk_b,
                mask_value,
                q_view(mask_ref, qs),
                q_view(q_sequence_ref, qs),
                q_view(q_segment_ids_ref, qs),
                kv_segment_ids_ref,
                attn_logits_soft_cap=attn_logits_soft_cap,
                k_slice=pl.ds(i * bkv_compute + k0, sk),
                k_offset=kv_index * bkv + i * bkv_compute + k0,
                bq=bq - q0,
                k_in_lanes=False,
                mask_function=mask_function,
                has_partial_mask=True,
            )
            - logsumexp[:, q0:]
        )
        dp_b = lax.dot_general(v_b, do_b, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        ds_b = (dp_b - di[:, q0:]) * p_b
        if attn_logits_soft_cap is not None:
          d = jnp.tanh(qk_b / attn_logits_soft_cap)
          ds_b = ds_b * (1 - d * d)
        kv_rows = pl.ds(i * bkv_compute + k0, sk)
        dv_scratch_ref[kv_rows, :] += lax.dot(p_b.astype(do.dtype), do_b, preferred_element_type=jnp.float32).astype(
            dv_scratch_ref.dtype
        )
        dk_b = lax.dot_general(
            ds_b.astype(do.dtype),
            scaled_q_b if config.dkv_reuse_scaled_q else q_b,
            dk_dims,
            preferred_element_type=jnp.float32,
        )
        if config.use_base2_exp and (config.dkv_reuse_scaled_q or config.q_prescaled_log2e):
          dk_b = dk_b / LOG2E
        dk_scratch_ref[kv_rows, :] += dk_b.astype(dk_scratch_ref.dtype)
        if dq_scratch_ref is not None:
          dq_b = lax.dot_general(
              ds_b.T if _is_fp8(k) else ds_b.T.astype(k.dtype),
              k_b,
              NN_DIM_NUMBERS,
              preferred_element_type=jnp.float32,
          )
          if config.q_prescaled_log2e:
            dq_b = dq_b / LOG2E
          dq_scratch_ref[q0:, :] += dq_b
      return

    if config.qk_diag_skip and has_partial_mask and bkv_compute % gk == 0 and bq % gq == 0:
      # Diagonal skip (backward dkv): qk tile is [kv, q]. On an aligned square
      # diagonal block, a sub-tile (kv-band ki, q-band qj) that lies entirely in
      # kv > q is overwritten to mask_value anyway -> skip its matmul; assemble
      # the full tile for the single exp/ds/dv/dk.
      sk = bkv_compute // gk
      sq = bq // gq
      k_parts = [k[t * sk : (t + 1) * sk, :] for t in range(gk)]
      # `scaled_q` is kept in its original layout here (it is always the RHS of
      # the qk matmul), so the q sequence axis is the minor one under
      # SEQ_MINOR and the major one under HEAD_DIM_MINOR.
      if config.q_layout == HEAD_DIM_MINOR:
        q_parts = [scaled_q[t * sq : (t + 1) * sq, :] for t in range(gq)]
      else:
        q_parts = [scaled_q[:, t * sq : (t + 1) * sq] for t in range(gq)]
      _mm = lambda kk, qq: lax.dot_general(kk, qq, qk_dims, preferred_element_type=jnp.float32)
      rows = []
      for ki in range(gk):  # kv row-band
        cols = []
        for qj in range(gq):  # q col-band
          if _above_diag(qj, sq, ki, sk):  # fully masked -> skip matmul
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
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)

    dropout_scale: jax.Array | None = None
    if dropout_rate:
      # Regenerated, not saved: the canonical blocks this tile covers are the
      # same ones the forward covered over the same (query, key) range, so the
      # bits match even though the backward tiles the matrix differently. Here
      # the tile is [kv, q] rather than [q, kv], hence the transpose (only
      # float32 transposes lower).
      global_kv_block_idx = kv_index * (bkv // bkv_compute) + i
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      n_q_blocks = q_steps * (bq // canonical_q)
      n_kv_blocks = kv_steps * (bkv // canonical_kv)
      dropout_scale = _dropout_mask_tile(
          prng_key_ref,
          head_idx=q_head,
          q_block_idx=q_index,
          kv_block_idx=global_kv_block_idx,
          q_block_size=bq,
          kv_block_size=bkv_compute,
          canonical_q=canonical_q,
          canonical_kv=canonical_kv,
          dropout_rate=dropout_rate,
          n_q_blocks=n_q_blocks,
          n_kv_blocks=n_kv_blocks,
      )
      dropout_scale = dropout_scale.T
      # dv sees the dropped weights, matching the forward's numerator.
      pr = p * dropout_scale
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
      assert dropout_scale is not None
      dp = dp * dropout_scale
    # `p` here is deliberately the *undropped* softmax weight: `di` is
    # rowsum(do * o), which already equals rowsum(dp_dropped * p).
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    dk_dims = NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dk = lax.dot_general(
        ds.astype(do.dtype),
        scaled_q if config.dkv_reuse_scaled_q else q,
        dk_dims,
        preferred_element_type=jnp.float32,
    )
    # Post-dK correction to undo the LOG2E scaling.
    if config.use_base2_exp and (config.dkv_reuse_scaled_q or config.q_prescaled_log2e):
      dk = dk / LOG2E
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T if _is_fp8(k) else ds.T.astype(k.dtype),
          k,
          NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if config.q_prescaled_log2e:
        dq = dq / LOG2E
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

    @pl.when(should_initialize_dq)
    def _init_dq_scratch():
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

    @pl.when(should_write_dq)
    def _write_dq_scratch():
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
  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if segment_ids is not None:
    mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
        mask_info,
        segment_ids,
        q_blocks=q_steps,
        kv_blocks=kv_steps,
        bq=bq,
        bkv=bkv,
        is_dkv=True,
    )
  else:
    bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  dynamic_grid = mask_info.active_rows is not None

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

  # On the dynamic grid (`grid = (num_q_heads, grid_size)`), GQA/MQA normally
  # accumulates `dk`/`dv` across the `q_heads_per_kv_head` query heads in a KV
  # group through an aliased HBM buffer (`dk_ref = dk_alias + dk_scratch`),
  # cycling through `j = 0 .. kv_steps - 1` within each `q_head`. Pallas TPU's
  # 2-deep input/output DMA pipeline requires at least 3 distinct blocks in that
  # cycle (matching `dq_reduction_steps = 3`); when `kv_steps <= 2`, the next
  # `q_head` prefetches `dk_alias` before the previous `q_head`'s store lands in
  # HBM (or coalesces the load when `kv_steps == 1`). Write per-`q_head` outputs
  # and reduce outside the kernel instead.
  reduce_dkv_over_q_heads = dynamic_grid and q_heads_per_kv_head != 1 and kv_steps <= 2
  alias_dkv = dynamic_grid and q_heads_per_kv_head != 1 and not reduce_dkv_over_q_heads

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    if reduce_dkv_over_q_heads:
      prefix = (h,)
    elif is_mqa:
      prefix = ()
    else:
      prefix = (_div(h, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      (bkv, head_dim_qk) if (is_mqa and not reduce_dkv_over_q_heads) else (None, bkv, head_dim_qk),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      (bkv, head_dim_v) if (is_mqa and not reduce_dkv_over_q_heads) else (None, bkv, head_dim_v),
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
  # TODO(b/447656860): Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  # TODO(b/447656860): Remove the sublane expansion once Mosaic has all retilings
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

  use_vmem_dq_reduction = dynamic_grid and config.dq_reduction_steps == 1
  dq_reduction_steps = 3 if dynamic_grid else config.dq_reduction_steps
  if use_vmem_dq_reduction:
    dq_reduction_steps = None  # dq is not accumulated or aliased in HBM.
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  if use_vmem_dq_reduction:
    dq_spec = pl.BlockSpec((None, bq, head_dim_qk), o_index_map)
    dq_shape = jax.ShapeDtypeStruct(q.shape, _unquantized_dtype(q))
  elif dq_reduction_steps == 3:
    dq_index_map = (
        (lambda h, g, r, c, *_: (g % 3, h, to_i32(c[g]), 0))
        if dynamic_grid
        else unravel(lambda h, i, j: (j % 3, h, i, 0))
    )
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), _unquantized_dtype(q))
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    if _is_fp8(q):
      q_dtype = jnp.bfloat16
    else:
      q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *q.shape), q_dtype)

  in_specs += [dq_alias_spec]

  if bkv == bkv_compute and not use_vmem_dq_reduction:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM((bq, head_dim_qk), jnp.float32)

  if alias_dkv:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = jnp.float32 if reduce_dkv_over_q_heads else _unquantized_dtype(k)
    dv_type = jnp.float32 if reduce_dkv_over_q_heads else _unquantized_dtype(v)

  dk_shape = (num_q_heads, kv_seq_len, head_dim_qk) if reduce_dkv_over_q_heads else k.shape
  dv_shape = (num_q_heads, kv_seq_len, head_dim_v) if reduce_dkv_over_q_heads else v.shape
  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(dk_shape, dk_type),
      jax.ShapeDtypeStruct(dv_shape, dv_type),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      kv_steps=kv_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
      use_vmem_dq_reduction=use_vmem_dq_reduction,
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
    if alias_dkv:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif alias_dkv:
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
  dq = dq_unreduced if use_vmem_dq_reduction else dq_unreduced.sum(axis=0)
  if reduce_dkv_over_q_heads:
    dk = dk.reshape(num_kv_heads, q_heads_per_kv_head, kv_seq_len, head_dim_qk).sum(axis=1)
    dv = dv.reshape(num_kv_heads, q_heads_per_kv_head, kv_seq_len, head_dim_v).sum(axis=1)
    if is_mqa:
      dk = jnp.squeeze(dk, axis=0)
      dv = jnp.squeeze(dv, axis=0)
  dq = dq.astype(_unquantized_dtype(q))
  dk = dk.astype(_unquantized_dtype(k))
  dv = dv.astype(_unquantized_dtype(v))
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

  if (config.qk_diag_skip or config.sv_diag_skip) and not (isinstance(mask, mask_lib.CausalMask) and mask.offset == 0):
    # The skip assumes kv > q is ALWAYS masked and kv <= q never is — a property
    # of the pure causal mask with offset 0. Any mask that admits a valid kv > q
    # entry (bidirectional, local/sliding window, positive offset, custom) would
    # be silently corrupted, so fail loud. (Block-shape preconditions are
    # enforced in SplashConfig.__post_init__.)
    param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
    raise ValueError(
        f"{param_name}=True requires a pure CausalMask with offset 0 (the skip"
        f" assumes the kv > q region is masked); got {mask!r}. Disable"
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
        return_dynamic_grid=config.dq_reduction_steps in (1, 3),
    )
    assert bq_dkv is not None
    q_blocks_per_shard = mask.shape[0] // q_seq_shards // bq_dkv
    if config.dq_reduction_steps == 1 and not (dkv_q_blocks_are_contiguous(dkv_mask_info, q_blocks_per_shard)):
      raise ValueError(
          "dq_reduction_steps=1 requires the dkv grid to visit every q block in"
          " one run of consecutive steps, which this mask and block sizes do"
          " not. Use dq_reduction_steps=3 or None."
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

  if config.qk_diag_skip or config.sv_diag_skip:
    # The skip treats every partial block as an aligned causal diagonal block,
    # which only a static pure CausalMask guarantees; a dynamic mask can have
    # off-diagonal or non-causal partial blocks, which would be corrupted.
    param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
    raise ValueError(
        f"{param_name}=True is not supported with dynamic masks (the skip"
        " assumes a static pure CausalMask with offset 0)."
    )

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
      dkv_mask_info = process_mask_fn(
          mask, (config.block_q_dkv, config.block_kv_dkv), is_dkv=True  # pyrefly: ignore[bad-argument-type]
      )

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
