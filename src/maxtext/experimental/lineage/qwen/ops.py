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

"""Pure-JAX Dual-SparseCore collectives, core-minor [ICI, core] EP, FP8 helpers, norms, and barriers."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any

import jax
from jax.experimental.compute_on import compute_on
from jax.experimental.layout import Layout
from jax.experimental.layout import with_layout_constraint
from jax.experimental.xla_metadata import set_xla_metadata
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen.config import QwenConfig

# Matches `--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=250000`
# so `>= 250 KB` dense/routing collectives offload to SparseCore while smaller
# `< 250 KB` tensors/scales run on fast TensorCore ICI.
_SC_MIN_COLLECTIVE_BYTES = 250_000

ALLOWED_REDUCTION_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
ALLOWED_FP8_DTYPES = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))


def assert_bf16_or_fp32_reduction(
    x: jax.Array, op_name: str = "reduction"
) -> None:
  """Enforces the Global Mixed-Precision Contract: BF16+ mandatory on ALL reductions."""
  dt = jnp.dtype(x.dtype)
  if dt not in ALLOWED_REDUCTION_DTYPES:
    raise ValueError(
        f"{op_name} requires bfloat16 or float32 reduction dtype, got {dt}. "
        "FP8 reductions (ReduceScatter / AllReduce) are strictly forbidden."
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Fp8DispatchedTokens:
  """Pre-quantized 3D/2D FP8 dispatched EP tokens with zero-byte bf16 STE carrier.

  - `ste_carrier`: 0-byte lazy `broadcast(0, [T_ep, ...])` carrying `dx_ep` in
  bwd.
  - `codes_ep`: `[T_ep, D // 128, 128]` (or `[T_ep, D]`) `float8_e4m3fn`
  gathered on SC0.
  - `scale`: `[1, 1]` `float32` per-EP-group dynamic quantization scale.
  """

  ste_carrier: Any
  codes_ep: jax.Array
  scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.codes_ep.shape

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.bfloat16 if self.ste_carrier is None else self.ste_carrier.dtype

  @property
  def ndim(self) -> int:
    return self.codes_ep.ndim


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoePartialTokens:
  """Partial EP output tokens carrying pre-quantized `fp8_e5m2` `dy` in backward.

  - `y_ep`: `[T_ep, D // 128, 128]` `bfloat16` partial EP output in forward
    (0-byte broadcast in backward).
  - `dy_codes`: `[T_ep, D // 128, 128]` `float8_e5m2` pre-quantized cotangent in
    backward (0-byte broadcast in forward).
  - `dy_scale`: `[1, 1]` `float32` per-EP-group `dy` scale in backward.
  """

  y_ep: jax.Array
  dy_codes: jax.Array
  dy_scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.y_ep.shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.y_ep.dtype

  @property
  def ndim(self) -> int:
    return self.y_ep.ndim


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoeSavedRouting:
  """Compact per-microbatch MoE routing metadata saved from FWD for BWD reuse.

  Eliminates redundant BWD router/sort/metadata compute and SC0 metadata
  AllGathers (`~0.15 GiB` across all 92L of Qwen MAX).
  """

  slot_order: jax.Array
  local_gs: jax.Array
  count: jax.Array
  reduce_meta: Any
  w_rows: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Fp8DenseWeight:
  """Gathered `float8_e4m3fn` dense weight (`gather8` on SC1) with BF16+ STE carrier.

  - `w_ste`: 0-byte lazy `broadcast(0, shape)` carrying `dw_full`
  (`bf16`/`fp32`) in bwd.
  - `codes`: `float8_e4m3fn` gathered weight (`[K, N]` or `[D, H, C]`).
  - `scale`: `float32` weight dequantization scale (`[1, 1]` or `[1, ...]` or
  `[K_blocks, ...]`).
  """

  w_ste: Any
  codes: jax.Array
  scale: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.codes.shape

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.bfloat16 if self.w_ste is None else self.w_ste.dtype

  @property
  def ndim(self) -> int:
    return self.codes.ndim


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class BwdRoutedWeightPrefetch:
  """Zero-forward-byte carrier transporting prefetched weights/tokens across backward scan."""

  wi_codes: jax.Array
  wi_scale: jax.Array
  wo_codes: jax.Array
  wo_scale: jax.Array
  disp_codes_ep: jax.Array | None = None
  disp_scale: jax.Array | None = None


def zero_bwd_rw_prefetch(
    wi_codes_shape: tuple[int, ...],
    wi_scale_shape: tuple[int, ...],
    wo_codes_shape: tuple[int, ...],
    wo_scale_shape: tuple[int, ...],
    *,
    disp_codes_shape: tuple[int, ...] | None = None,
    disp_scale_shape: tuple[int, ...] | None = None,
    dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> BwdRoutedWeightPrefetch:
  """Creates 0-byte lazy broadcast placeholders for `BwdRoutedWeightPrefetch` in forward."""
  z_w = jnp.zeros((), dtype=dtype)
  z_f8 = jnp.zeros((), dtype=jnp.float8_e4m3fn)
  z_f32 = jnp.zeros((), dtype=jnp.float32)
  return BwdRoutedWeightPrefetch(
      wi_codes=jax.lax.broadcast(z_w, wi_codes_shape),
      wi_scale=jax.lax.broadcast(z_f32, wi_scale_shape),
      wo_codes=jax.lax.broadcast(z_w, wo_codes_shape),
      wo_scale=jax.lax.broadcast(z_f32, wo_scale_shape),
      disp_codes_ep=(
          jax.lax.broadcast(z_f8, disp_codes_shape)
          if disp_codes_shape is not None
          else None
      ),
      disp_scale=(
          jax.lax.broadcast(z_f32, disp_scale_shape)
          if disp_scale_shape is not None
          else None
      ),
  )


def merge_bwd_rw3_slots(
    slot_mb0: BwdRoutedWeightPrefetch,
    slot_mb1: BwdRoutedWeightPrefetch,
) -> BwdRoutedWeightPrefetch:
  """Zero-residual fanout sharing prefetched `attn_3` weights across `mb1` (`B4`) and `mb0` (`B3`)."""
  wi_cs = slot_mb0.wi_codes.shape
  wi_ss = slot_mb0.wi_scale.shape
  wo_cs = slot_mb0.wo_codes.shape
  wo_ss = slot_mb0.wo_scale.shape

  @jax.custom_vjp
  def _merge(
      s0: BwdRoutedWeightPrefetch, s1: BwdRoutedWeightPrefetch
  ) -> BwdRoutedWeightPrefetch:
    del s0, s1
    return zero_bwd_rw_prefetch(wi_cs, wi_ss, wo_cs, wo_ss)

  def _merge_fwd(s0: BwdRoutedWeightPrefetch, s1: BwdRoutedWeightPrefetch):
    del s0, s1
    return zero_bwd_rw_prefetch(wi_cs, wi_ss, wo_cs, wo_ss), ()

  def _merge_bwd(_, g_pref: BwdRoutedWeightPrefetch):
    return g_pref, g_pref

  _merge.defvjp(_merge_fwd, _merge_bwd)
  return _merge(slot_mb0, slot_mb1)


def _ep_core_id(cfg: QwenConfig) -> int | None:
  return 0 if (cfg.use_pallas_kernels and cfg.ep_sc_pinning == "sc0") else None


def _fsdp_ag_core_id(
    cfg: QwenConfig, *, is_bwd_remat: bool = False
) -> int | None:
  if not cfg.use_pallas_kernels:
    return None
  if cfg.fsdp_ag_sc_pinning == "sc1":
    return 1
  if cfg.fsdp_ag_sc_pinning == "sc1_fwd_only" and not is_bwd_remat:
    return 1
  return None


def _fsdp_rs_core_id(cfg: QwenConfig) -> int | None:
  return (
      1
      if (cfg.use_pallas_kernels and cfg.fsdp_rs_sc_pinning == "sc1")
      else None
  )


def _dense_rs_core_id(cfg: QwenConfig) -> int | None:
  if not cfg.use_pallas_kernels or cfg.fsdp_rs_sc_pinning == "unpinned":
    return None
  if cfg.bwd_sc_balance_mode == "wi_dense_rs_sc0":
    return 0
  return 1


def _moe_rs_core_id(cfg: QwenConfig, *, is_wi: bool = False) -> int | None:
  if not cfg.use_pallas_kernels or cfg.fsdp_rs_sc_pinning == "unpinned":
    return None
  if is_wi and cfg.bwd_sc_balance_mode in ("wi_rs_sc0", "wi_dense_rs_sc0"):
    return 0
  if (not is_wi) and cfg.bwd_sc_balance_mode == "dwo_sc0":
    return 0
  return 1


def sc_all_gather(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    axis: int = 0,
    min_collective_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Single-array all-gather pinned to a specific SparseCore (`0` or `1`)."""
  if isinstance(axis_name, str):
    axis_size = jax.lax.axis_size(axis_name)
  else:
    axis_size = functools.reduce(
        lambda a, b: a * b, (jax.lax.axis_size(n) for n in axis_name), 1
    )
  gathered_bytes = x.size * x.dtype.itemsize * axis_size
  if core_id is None or gathered_bytes < min_collective_bytes:
    return jax.lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=True)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _ag(inp: jax.Array) -> jax.Array:
    return jax.lax.all_gather(inp, axis_name=axis_name, axis=axis, tiled=True)

  return _ag(x)


def sc_psum_scatter(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    scatter_dimension: int = 0,
    min_collective_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Single-array BF16/FP32 psum_scatter pinned to SparseCore (`0` or `1`)."""
  assert_bf16_or_fp32_reduction(x, "sc_psum_scatter")
  input_bytes = x.size * x.dtype.itemsize
  if core_id is None or input_bytes < min_collective_bytes:
    return jax.lax.psum_scatter(
        x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=True
    )

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _rs(inp: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(
        inp,
        axis_name=axis_name,
        scatter_dimension=scatter_dimension,
        tiled=True,
    )

  return _rs(x)


def _next_pow2(n: int) -> int:
  return 1 << (n - 1).bit_length()


def _is_pow2(n: int) -> bool:
  return n > 0 and (n & (n - 1)) == 0


def _pick_3d_col_split(cols: int) -> tuple[int, int] | None:
  """Splits `cols` (`% 128 == 0`) into `(h, w)` with `h % 8 == 0`, `w % 128 == 0`."""
  for w in (1024, 512, 256, 128):
    if cols % w == 0:
      h = cols // w
      if h >= 8 and h % 8 == 0:
        return h, w
  return None


def _needs_col_pad_for_sc_3d_rs(cols: int) -> bool:
  """Returns True when `cols` fails SC ND `reduce-scatter` span divisibility."""
  if cols <= 0 or cols % 128 != 0:
    return False
  tiles = cols // 128
  rem = tiles
  for p in (2, 3, 5, 7, 11, 13, 17, 19):
    while rem % p == 0:
      rem //= p
  return (tiles % 2 != 0) or (rem > 1)


def sc_3d_all_gather(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    axis: int = 0,
    min_collective_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
    force_align: bool = False,
) -> jax.Array:
  """3D SC `all_gather` lifting sub-8-row 2D shards (`R_local < 8`) to 3D `T(8, 128)`."""
  if isinstance(axis_name, str):
    axis_size = jax.lax.axis_size(axis_name)
  else:
    axis_size = functools.reduce(
        lambda a, b: a * b, (jax.lax.axis_size(n) for n in axis_name), 1
    )
  gathered_bytes = x.size * x.dtype.itemsize * axis_size
  use_align = (
      (
          force_align
          or (core_id is not None and gathered_bytes >= min_collective_bytes)
      )
      and x.ndim == 2
      and axis == 0
      and x.shape[0] < 8
  )
  if use_align:
    split = _pick_3d_col_split(x.shape[1])
    if split is not None:
      h_dim, w_dim = split
      out_3d = sc_all_gather(
          x.reshape(x.shape[0], h_dim, w_dim),
          axis_name,
          core_id=core_id,
          axis=0,
          min_collective_bytes=min_collective_bytes,
      )
      return out_3d.reshape(out_3d.shape[0], x.shape[1])
  return sc_all_gather(
      x,
      axis_name,
      core_id=core_id,
      axis=axis,
      min_collective_bytes=min_collective_bytes,
  )


def sc_3d_psum_scatter(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    scatter_dimension: int = 0,
    min_collective_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
    force_align: bool = False,
) -> jax.Array:
  """3D SC `psum_scatter` avoiding `AllReduceOffload` fallback purely in JAX.

  1. Odd / large-prime column tile counts (e.g. `lm_head` `62080 = 485 * 128`):
     zero-pads axis 1 to the next power-of-2 multiple of `128` (`65536`),
     runs native SC 3D `reduce-scatter`, and slices back to `[:, :orig_cols]`.
  2. Small per-shard row counts (`R_local < 8` on `q35_512c`, or `R_local == 8`
     with non-pow2 cols at `axis_size >= 1024` on `qmax_512c` `in_proj_qkvz`):
     reshapes 2D `[R, C]` -> 3D `[R, H, W]` (`H % 8 == 0`, `W % 128 == 0`) so
     `T(8, 128)` tiles `(H, W)` and `basic_unit_shape` keeps `dim0 = 1`
     (`units_per_shard = R_local`, even, zero `kPad`), then restores `[R_local, C]`.
  """
  assert_bf16_or_fp32_reduction(x, "sc_3d_psum_scatter")
  if isinstance(axis_name, str):
    axis_size = jax.lax.axis_size(axis_name)
  else:
    axis_size = functools.reduce(
        lambda a, b: a * b, (jax.lax.axis_size(n) for n in axis_name), 1
    )
  input_bytes = x.size * x.dtype.itemsize
  should_align = (
      (
          force_align
          or (core_id is not None and input_bytes >= min_collective_bytes)
      )
      and x.ndim == 2
      and scatter_dimension == 0
      and axis_size > 1
      and x.shape[0] % axis_size == 0
  )
  if not should_align:
    return sc_psum_scatter(
        x,
        axis_name,
        core_id=core_id,
        scatter_dimension=scatter_dimension,
        min_collective_bytes=min_collective_bytes,
    )

  rows, orig_cols = x.shape
  r_local = rows // axis_size
  cur = x
  cols = orig_cols

  if _needs_col_pad_for_sc_3d_rs(cols):
    padded_cols = _next_pow2(cols // 128) * 128
    pad_amt = padded_cols - cols
    if pad_amt > 0:
      cur = jnp.pad(cur, ((0, 0), (0, pad_amt)))
      cols = padded_cols

  need_3d = (r_local < 8) or (
      r_local == 8
      and axis_size >= 1024
      and cols % 128 == 0
      and not _is_pow2(cols // 128)
  )
  split = _pick_3d_col_split(cols) if need_3d else None
  if split is not None:
    h_dim, w_dim = split
    out_3d = sc_psum_scatter(
        cur.reshape(rows, h_dim, w_dim),
        axis_name,
        core_id=core_id,
        scatter_dimension=0,
        min_collective_bytes=min_collective_bytes,
    )
    out = out_3d.reshape(r_local, cols)
  else:
    out = sc_psum_scatter(
        cur,
        axis_name,
        core_id=core_id,
        scatter_dimension=0,
        min_collective_bytes=min_collective_bytes,
    )

  if cols != orig_cols:
    out = out[:, :orig_cols]
  return out


def psum_scatter_bf16_or_fp32(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    scatter_dimension: int = 0,
) -> jax.Array:
  """TensorCore ICI `psum_scatter` enforcing BF16+ reduction dtype."""
  assert_bf16_or_fp32_reduction(x, "psum_scatter")
  return jax.lax.psum_scatter(
      x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=True
  )


def psum_bf16_or_fp32(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
) -> jax.Array:
  """AllReduce `psum` wrapper enforcing BF16+ reduction dtype."""
  assert_bf16_or_fp32_reduction(x, "psum")
  return jax.lax.psum(x, axis_name=axis_name)


def sc_all_reduce(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None = None,
    min_collective_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """AllReduce `psum` pinned optionally to SparseCore, enforcing BF16+ dtype."""
  assert_bf16_or_fp32_reduction(x, "sc_all_reduce")
  input_bytes = x.size * x.dtype.itemsize
  if core_id is None or input_bytes < min_collective_bytes:
    return jax.lax.psum(x, axis_name=axis_name)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _ar(inp: jax.Array) -> jax.Array:
    return jax.lax.psum(inp, axis_name=axis_name)

  return _ar(x)


def sc_pmax(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None = None,
) -> jax.Array:
  """AllReduce `pmax` pinned optionally to SparseCore, enforcing BF16+ dtype."""
  assert_bf16_or_fp32_reduction(x, "sc_pmax")
  if core_id is None:
    return jax.lax.pmax(x, axis_name=axis_name)
  orig_shape = x.shape
  # SparseCore collective offload requires rank >= 1 (CHECK_GT(rank, 0)).
  x_1d = x.reshape(1) if x.ndim == 0 else x

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )
  def _pmax(inp: jax.Array) -> jax.Array:
    return jax.lax.pmax(inp, axis_name=axis_name)

  out = _pmax(x_1d)
  return out.reshape(orig_shape) if not orig_shape else out


# ---------------------------------------------------------------------------
# Lineage-style T(8, 128) zero-copy routed weight tiling across SparseCore
# ---------------------------------------------------------------------------


def routed_w_to_tiles(w: jax.Array) -> jax.Array:
  """Reshapes `[..., E, rows, cols]` -> `[..., E, rows // 8, 8, cols]`."""
  if w.shape[-2] % 8 != 0:
    return w
  *prefix, rows, cols = w.shape
  return w.reshape(*prefix, rows // 8, 8, cols)


def routed_w_from_tiles(w: jax.Array) -> jax.Array:
  """Restores `[..., E, rows // 8, 8, cols]` -> `[..., E, rows, cols]`."""
  if w.ndim < 4 or w.shape[-2] != 8:
    return w
  *prefix, r_tiles, sub, cols = w.shape
  return w.reshape(*prefix, r_tiles * sub, cols)


def constrain_activation_bank(
    bank: jax.Array, *, enabled: bool = True
) -> jax.Array:
  """Pins `[R, 2F]` or `[R // 8, 8, 2F]` bf16 activation bank to `T(8, 128), (2, 1)`."""
  if not enabled or bank.shape[-1] % 128 != 0:
    return bank
  if bank.ndim == 2 and bank.shape[0] % 8 != 0:
    return bank
  tiling = (
      ((8, 128), (2, 1)) if jnp.dtype(bank.dtype).itemsize == 2 else ((8, 128),)
  )
  layout = Layout(major_to_minor=tuple(range(bank.ndim)), tiling=tiling)
  return with_layout_constraint(bank, layout)


def bank_to_tiles(bank_2d: jax.Array) -> jax.Array:
  """Views `[R, 2F]` activation bank as `[R // 8, 8, 2F]` so `T(8, 128)` survives carries/VJPs."""
  if bank_2d.ndim != 2 or bank_2d.shape[0] % 8 != 0:
    return bank_2d
  return bank_2d.reshape(bank_2d.shape[0] // 8, 8, bank_2d.shape[1])


def bank_from_tiles(
    bank_tiles: jax.Array, *, constrain: bool = True
) -> jax.Array:
  """Restores `[R // 8, 8, 2F]` -> `[R, 2F]` with `T(8, 128)` layout constraint."""
  b2d = (
      bank_tiles.reshape(-1, bank_tiles.shape[-1])
      if bank_tiles.ndim == 3
      else bank_tiles
  )
  return constrain_activation_bank(b2d, enabled=constrain)


def alloc_activation_bank(
    rows: int,
    cols: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    constrain: bool = True,
) -> jax.Array:
  """Allocates uninitialized `[rows, cols]` activation bank in `T(8, 128)` tiling."""
  raw = jax.lax.empty((rows, cols), dtype=dtype)
  return constrain_activation_bank(raw, enabled=constrain)


# ---------------------------------------------------------------------------
# Forward-only layout & optimization barrier primitives
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def forward_layout_constraint(
    x: jax.Array, major_to_minor: tuple[int, ...]
) -> jax.Array:
  """Pins `with_layout_constraint` in forward only; identity in backward."""
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor))


def _fwd_layout_fwd(x: jax.Array, major_to_minor: tuple[int, ...]):
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor)), None


def _fwd_layout_bwd(major_to_minor: tuple[int, ...], _, cotangent: jax.Array):
  del major_to_minor
  return (cotangent,)


forward_layout_constraint.defvjp(_fwd_layout_fwd, _fwd_layout_bwd)


def first_scalar(arr: jax.Array) -> jax.Array:
  """Reads element `(0, ..., 0)` directly without flattening (`reshape(-1)` copies tiled 4D arrays!)."""
  return arr[(0,) * arr.ndim] if arr.ndim > 0 else arr


def _collect_dep_scalars(dep: Any) -> jax.Array:
  """Recursively extracts zero-scaled scalar tie terms without pulling unused grad carriers."""
  if isinstance(dep, Fp8DispatchedTokens):
    return (
        first_scalar(dep.codes_ep).astype(jnp.float32) * 0.0
        + first_scalar(dep.scale).astype(jnp.float32) * 0.0
    )
  if isinstance(dep, Fp8DenseWeight):
    return (
        first_scalar(dep.codes).astype(jnp.float32) * 0.0
        + first_scalar(dep.scale).astype(jnp.float32) * 0.0
    )
  if isinstance(dep, MoePartialTokens):
    return first_scalar(dep.y_ep).astype(jnp.float32) * 0.0
  if isinstance(dep, (tuple, list)):
    acc = jnp.zeros((), dtype=jnp.float32)
    for item in dep:
      if item is not None:
        acc = acc + _collect_dep_scalars(item)
    return acc
  if hasattr(dep, "ndim"):
    return first_scalar(dep).astype(jnp.float32) * 0.0
  leaves = [
      v for v in jax.tree.leaves(dep) if v is not None and hasattr(v, "ndim")
  ]
  acc = jnp.zeros((), dtype=jnp.float32)
  for leaf in leaves:
    acc = acc + first_scalar(leaf).astype(jnp.float32) * 0.0
  return acc


def order_after(x: Any, dep: Any) -> Any:
  """Forces consumers of `x` after `dep` in forward; zero-cotangent identity in backward."""
  z_f32 = jax.lax.stop_gradient(_collect_dep_scalars(dep))
  z_f32 = jax.ad_checkpoint.checkpoint_name(z_f32, "fwd_tie_scalar")
  if isinstance(x, Fp8DispatchedTokens):
    return Fp8DispatchedTokens(
        ste_carrier=x.ste_carrier,
        codes_ep=x.codes_ep,
        scale=x.scale + z_f32.astype(x.scale.dtype),
    )
  if isinstance(x, Fp8DenseWeight):
    return Fp8DenseWeight(
        w_ste=x.w_ste,
        codes=x.codes,
        scale=x.scale + z_f32.astype(x.scale.dtype),
    )
  if isinstance(x, MoePartialTokens):
    return MoePartialTokens(
        y_ep=x.y_ep + z_f32.astype(x.y_ep.dtype),
        dy_codes=x.dy_codes,
        dy_scale=x.dy_scale,
    )
  return x + z_f32.astype(x.dtype)


# Alias kept for backward compatibility
barrier_tie = order_after


def _is_tieable_grad_leaf(v: Any) -> bool:
  return (
      hasattr(v, "dtype")
      and jnp.issubdtype(v.dtype, jnp.floating)
      and v.dtype not in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  )


def _is_custom_token_carrier(v: Any) -> bool:
  return isinstance(v, (Fp8DispatchedTokens, MoePartialTokens, Fp8DenseWeight))


@jax.custom_vjp
def backward_barrier_tie(a_tree, b_tree):
  """No-op in forward; in backward, forces `g_a` after `g_b` via zero-multiply data edge."""
  return a_tree, b_tree


def _bwd_barrier_tie_fwd(a_tree, b_tree):
  return (a_tree, b_tree), None


def _bwd_barrier_tie_bwd(_, cotangents):
  """Backward pass tying cotangent `g_a` sequentially after `g_b` scalar."""
  g_a, g_b = cotangents
  b_terms: list[jax.Array] = []
  for leaf in jax.tree_util.tree_leaves(g_b, is_leaf=_is_custom_token_carrier):
    if isinstance(leaf, Fp8DispatchedTokens):
      if _is_tieable_grad_leaf(leaf.ste_carrier):
        b_terms.append(
            (first_scalar(leaf.ste_carrier) * 0.0).astype(jnp.float32)
        )
    elif isinstance(leaf, Fp8DenseWeight):
      if _is_tieable_grad_leaf(leaf.w_ste):
        b_terms.append((first_scalar(leaf.w_ste) * 0.0).astype(jnp.float32))
    elif isinstance(leaf, MoePartialTokens):
      b_terms.append(
          (first_scalar(leaf.dy_codes).astype(jnp.float32) * 0.0)
          + (first_scalar(leaf.dy_scale).astype(jnp.float32) * 0.0)
      )
    elif _is_tieable_grad_leaf(leaf):
      b_terms.append((first_scalar(leaf) * 0.0).astype(jnp.float32))
  if not b_terms:
    return g_a, g_b
  zero_dep = sum(b_terms)

  def _tie_a_leaf(v: Any) -> Any:
    if isinstance(v, Fp8DispatchedTokens):
      return Fp8DispatchedTokens(
          ste_carrier=v.ste_carrier + zero_dep.astype(v.ste_carrier.dtype),
          codes_ep=v.codes_ep,
          scale=v.scale,
      )
    if isinstance(v, Fp8DenseWeight):
      return Fp8DenseWeight(
          w_ste=v.w_ste + zero_dep.astype(v.w_ste.dtype),
          codes=v.codes,
          scale=v.scale,
      )
    if isinstance(v, MoePartialTokens):
      return MoePartialTokens(
          y_ep=v.y_ep,
          dy_codes=v.dy_codes,
          dy_scale=v.dy_scale + zero_dep.astype(v.dy_scale.dtype),
      )
    if _is_tieable_grad_leaf(v):
      return v + zero_dep.astype(v.dtype)
    return v

  g_a_tied = jax.tree_util.tree_map(
      _tie_a_leaf, g_a, is_leaf=_is_custom_token_carrier
  )
  return g_a_tied, g_b


backward_barrier_tie.defvjp(_bwd_barrier_tie_fwd, _bwd_barrier_tie_bwd)


@jax.custom_vjp
def tie_bwd_after_partial_dy_scale(a: jax.Array, partial_in: Any):
  """No-op in forward; in backward, orders `g_a` (`B1(j)`) after `g_partial.dy_scale` (`AR_1`)."""
  return a, partial_in


def _tie_bwd_dy_scale_fwd(a: jax.Array, partial_in: Any):
  return (a, partial_in), None


def _tie_bwd_dy_scale_bwd(_, cotangents):
  g_a, g_partial = cotangents
  if isinstance(g_partial, MoePartialTokens) and _is_tieable_grad_leaf(g_a):
    z = (first_scalar(g_partial.dy_scale).astype(jnp.float32) * 0.0).astype(
        g_a.dtype
    )
    return g_a + z, g_partial
  return g_a, g_partial


tie_bwd_after_partial_dy_scale.defvjp(
    _tie_bwd_dy_scale_fwd, _tie_bwd_dy_scale_bwd
)


# ---------------------------------------------------------------------------
# FP8 quantization & SC0-pinned Core-Minor [ICI, core] EP collectives
# ---------------------------------------------------------------------------


def quantize_rows_fp8(
    x: jax.Array, qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
) -> tuple[jax.Array, jax.Array]:
  """Per-token-row dynamic FP8 quantization for 2D `[T, D]` or 3D `[T, D0, 128]`."""
  qdt = jnp.dtype(qtype)
  if qdt not in ALLOWED_FP8_DTYPES:
    raise ValueError(f"Unsupported FP8 dtype: {qdt}")
  qmax = float(jnp.finfo(qdt).max)
  red_axes = tuple(range(1, x.ndim))
  absmax = jnp.max(jnp.abs(x), axis=red_axes, keepdims=True).astype(jnp.float32)
  scale = jnp.where(absmax > 0.0, absmax / qmax, jnp.ones_like(absmax))
  codes = jnp.clip(x.astype(jnp.float32) / scale, -qmax, qmax).astype(qdt)
  return codes, scale


def quantize_cols_fp8_for_tgmm(
    x: jax.Array, qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
) -> tuple[jax.Array, jax.Array]:
  """Per-output-channel (`N`) FP8 quantization for 2D `[M, N]` or 3D `[M, N0, 128]` `tgmm_v2` RHS."""
  qdt = jnp.dtype(qtype)
  if qdt not in ALLOWED_FP8_DTYPES:
    raise ValueError(f"Unsupported FP8 dtype: {qdt}")
  qmax = float(jnp.finfo(qdt).max)
  x_f32 = x.astype(jnp.float32)
  absmax = jnp.max(jnp.abs(x_f32), axis=0)
  scale = jnp.where(absmax > 0.0, absmax / qmax, jnp.ones_like(absmax))
  codes = jnp.clip(x_f32 / scale[None, ...], -qmax, qmax).astype(qdt)
  return codes, scale.reshape(1, 1, -1)


def dequantize_rows_fp8(
    codes: jax.Array, scale: jax.Array, dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Inverse of `quantize_rows_fp8`."""
  return (codes.astype(jnp.float32) * scale).astype(dtype)


def quantize_subchannel_fp8(
    x: jax.Array,
    *,
    block_size: int = 128,
    axis: int = -1,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
  """Subchannel (`block_size=128`) FP8 quantization along contraction `axis`."""
  qdt = jnp.dtype(qtype)
  if qdt not in ALLOWED_FP8_DTYPES:
    raise ValueError(f"Unsupported FP8 dtype: {qdt}")
  norm_axis = axis % x.ndim
  dim = x.shape[norm_axis]
  if dim % block_size != 0:
    raise ValueError(
        f"Dimension {dim} at axis={axis} must be divisible by"
        f" block_size={block_size}."
    )
  num_blocks = dim // block_size
  blocked_shape = (
      *x.shape[:norm_axis],
      num_blocks,
      block_size,
      *x.shape[norm_axis + 1 :],
  )
  x_blocked = x.reshape(blocked_shape).astype(jnp.float32)
  qmax = float(jnp.finfo(qdt).max)
  amax = jnp.max(jnp.abs(x_blocked), axis=norm_axis + 1, keepdims=True)
  scale_blocked = jnp.where(amax > 0.0, amax / qmax, jnp.ones_like(amax))
  codes_blocked = jnp.clip(x_blocked / scale_blocked, -qmax, qmax).astype(qdt)
  scale = jnp.squeeze(scale_blocked, axis=norm_axis + 1)
  return codes_blocked.reshape(x.shape), scale


def ep_axis_index(cfg: QwenConfig) -> jax.Array:
  """Returns flat EP rank `0 .. cfg.ep_size - 1` matching `ep_axis_tuple`."""
  if cfg.decouple_ep_d2d:
    if cfg.ep_core_minor:
      return jax.lax.axis_index("expert") * 2 + jax.lax.axis_index("ep_d2d")
    return jax.lax.axis_index("ep_d2d") * cfg.ep_ici_size + jax.lax.axis_index(
        "expert"
    )
  return jax.lax.axis_index("expert")


def device_axis_index(cfg: QwenConfig) -> jax.Array:
  """Returns flat device index `0 .. cfg.num_devices - 1` along `all_sharded_axes`."""
  if cfg.num_devices <= 1:
    return jnp.asarray(0, dtype=jnp.int32)
  idx = jnp.asarray(0, dtype=jnp.int32)
  for ax in cfg.all_sharded_axes:
    try:
      ax_sz = jax.lax.axis_size(ax)
      ax_i = jax.lax.axis_index(ax)
    except (NameError, ValueError):
      return jnp.asarray(0, dtype=jnp.int32)
    idx = idx * jnp.asarray(ax_sz, dtype=jnp.int32) + ax_i
  return idx


def sample_random_topk_ids(
    leading_shape: tuple[int, ...],
    cfg: QwenConfig,
    *,
    layer_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    batch_offset: int = 0,
    router_logits_f32: jax.Array | None = None,
    noise_scale: float = 1e3,
) -> jax.Array:
  """Samples uniform without-replacement top-k expert IDs `[T, K]` (`int32`).

  Folds `routing_seed` with `(step, layer_idx, global_seq_idx)` where
  `global_seq_idx = dev_id * per_device_batch_size + batch_offset + b` so
  assignments vary across steps, layers, devices, and tokens, are deterministic
  for a given seed, and are bit-identical between `unsplit` and `batchsplit` /
  `lineage_schedule`. When `router_logits_f32` is provided, adds
  `stop_gradient(router_logits_f32) + noise_scale * noise` before `top_k` so
  XLA keeps routing/top_k/sort data-dependent inside the scanned layer body
  (matching real router dataflow) while preserving uniform load balance.
  """
  base_key = jax.random.PRNGKey(cfg.routing_seed)
  base_key = jax.random.fold_in(base_key, jnp.asarray(step, dtype=jnp.int32))
  base_key = jax.random.fold_in(
      base_key, jnp.asarray(layer_idx, dtype=jnp.int32)
  )
  dev_id = device_axis_index(cfg)
  base_seq = dev_id * jnp.asarray(
      cfg.per_device_batch_size, dtype=jnp.int32
  ) + jnp.asarray(batch_offset, dtype=jnp.int32)
  scale_f32 = jnp.asarray(noise_scale, dtype=jnp.float32)

  if len(leading_shape) == 2:
    bsz, seq_len = leading_shape
    seq_indices = base_seq + jnp.arange(bsz, dtype=jnp.int32)

    if router_logits_f32 is not None:
      logits_3d = jax.lax.stop_gradient(
          router_logits_f32.astype(jnp.float32).reshape(
              bsz, seq_len, cfg.num_experts
          )
      )

      def _sample_seq_dd(
          seq_idx: jax.Array, logits_seq: jax.Array
      ) -> jax.Array:
        k = jax.random.fold_in(base_key, seq_idx)
        noise = jax.random.uniform(
            k, (seq_len, cfg.num_experts), dtype=jnp.float32
        )
        scores = logits_seq + scale_f32 * noise
        _, ids = jax.lax.top_k(scores, cfg.num_experts_per_tok)
        return ids.astype(jnp.int32)

      ids_out = jax.vmap(_sample_seq_dd)(seq_indices, logits_3d).reshape(
          bsz * seq_len, cfg.num_experts_per_tok
      )
    else:

      def _sample_seq(seq_idx: jax.Array) -> jax.Array:
        k = jax.random.fold_in(base_key, seq_idx)
        noise = jax.random.uniform(
            k, (seq_len, cfg.num_experts), dtype=jnp.float32
        )
        _, ids = jax.lax.top_k(noise, cfg.num_experts_per_tok)
        return ids.astype(jnp.int32)

      ids_out = jax.vmap(_sample_seq)(seq_indices).reshape(
          bsz * seq_len, cfg.num_experts_per_tok
      )
  else:
    total_toks = functools.reduce(lambda a, b: a * b, leading_shape, 1)
    k = jax.random.fold_in(base_key, base_seq)
    noise = jax.random.uniform(
        k, (total_toks, cfg.num_experts), dtype=jnp.float32
    )
    if router_logits_f32 is not None:
      scores = (
          jax.lax.stop_gradient(
              router_logits_f32.astype(jnp.float32).reshape(
                  total_toks, cfg.num_experts
              )
          )
          + scale_f32 * noise
      )
    else:
      scores = noise
    _, ids_out = jax.lax.top_k(scores, cfg.num_experts_per_tok)
    ids_out = ids_out.astype(jnp.int32)

  return jax.lax.stop_gradient(ids_out)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def dense_select_topk_probs(
    probs: jax.Array, topk_ids: jax.Array, num_experts: int
) -> jax.Array:
  """Selects `probs[t, topk_ids[t, k]]` via dense compare-select (zero gather/scatter)."""
  e_idx = jnp.arange(num_experts, dtype=topk_ids.dtype)[None, None, :]
  mask = topk_ids[:, :, None] == e_idx
  return jnp.sum(
      jnp.where(mask, probs[:, None, :], jnp.asarray(0.0, dtype=probs.dtype)),
      axis=-1,
  )


def _dense_select_topk_probs_fwd(
    probs: jax.Array, topk_ids: jax.Array, num_experts: int
):
  out = dense_select_topk_probs(probs, topk_ids, num_experts)
  return out, (topk_ids,)


def _dense_select_topk_probs_bwd(
    num_experts: int, res: tuple[jax.Array], g_out: jax.Array
):
  (topk_ids,) = res
  e_idx = jnp.arange(num_experts, dtype=topk_ids.dtype)[None, None, :]
  mask = topk_ids[:, :, None] == e_idx
  g_probs = jnp.sum(
      jnp.where(mask, g_out[:, :, None], jnp.asarray(0.0, dtype=g_out.dtype)),
      axis=1,
  )
  return g_probs, None


dense_select_topk_probs.defvjp(
    _dense_select_topk_probs_fwd, _dense_select_topk_probs_bwd
)


def ep_token_dispatch(tokens_local: jax.Array, cfg: QwenConfig) -> jax.Array:
  """SC0-pinned EP token all-gather (FP8 forward option, bf16 SC0 RS backward)."""
  core_id = _ep_core_id(cfg)
  use_fp8 = cfg.moe_fp8_token_dispatch
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes

  @jax.custom_vjp
  def _dispatch(z: jax.Array) -> jax.Array:
    if use_fp8:
      qval, scale = quantize_rows_fp8(z, jnp.float8_e4m3fn)
      qval_ep = sc_all_gather(
          qval, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
      )
      scale_ep = jax.lax.all_gather(scale, axis_name=ep_ax, axis=0, tiled=True)
      return dequantize_rows_fp8(qval_ep, scale_ep, z.dtype)
    return sc_all_gather(
        z, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
    )

  def _dispatch_fwd(z: jax.Array):
    return _dispatch(z), None

  def _dispatch_bwd(_, g_ep: jax.Array):
    return (
        sc_psum_scatter(
            g_ep,
            ep_ax,
            core_id=core_id,
            scatter_dimension=0,
            min_collective_bytes=min_b,
        ),
    )

  _dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)
  return _dispatch(tokens_local)


def _ep_scalar_pmax(amax_local: jax.Array, cfg: QwenConfig) -> jax.Array:
  """Computes `pmax` over EP axes without cross-microbatch/phase combining (BF16+ enforced)."""
  assert_bf16_or_fp32_reduction(amax_local, "_ep_scalar_pmax")
  ep_ax = cfg.ep_axis_names
  if cfg.fp8_amax_no_combine:
    with set_xla_metadata(_scheduling_group_id="noop"):
      return jax.lax.pmax(amax_local, axis_name=ep_ax)
  return jax.lax.pmax(amax_local, axis_name=ep_ax)


def ep_token_dispatch_3d_fp8(
    tokens_local: jax.Array, cfg: QwenConfig
) -> Fp8DispatchedTokens:
  """SC0-pinned 3D FP8 EP token all-gather keeping tokens in `float8_e4m3fn` through `wi`!"""
  core_id = _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes
  ep_shape = (tokens_local.shape[0] * cfg.ep_size, *tokens_local.shape[1:])
  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)

  @jax.custom_vjp
  def _dispatch_fp8(z: jax.Array) -> Fp8DispatchedTokens:
    amax_local = jax.lax.stop_gradient(jnp.max(jnp.abs(z.astype(jnp.float32))))
    amax_ep = _ep_scalar_pmax(amax_local, cfg).reshape(1, 1)
    scale = jax.lax.stop_gradient(
        jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep))
    )
    inv_scale = (1.0 / scale).reshape(1, *([1] * (z.ndim - 1)))
    codes_l = jax.lax.stop_gradient(
        jnp.clip(z.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(
            jnp.float8_e4m3fn
        )
    )
    codes_ep = sc_all_gather(
        codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
    )
    ste = jax.lax.broadcast(jnp.zeros((), dtype=z.dtype), ep_shape)
    return Fp8DispatchedTokens(ste_carrier=ste, codes_ep=codes_ep, scale=scale)

  def _dispatch_fp8_fwd(z: jax.Array):
    return _dispatch_fp8(z), None

  def _dispatch_fp8_bwd(_, g_disp: Fp8DispatchedTokens):
    return (
        sc_psum_scatter(
            g_disp.ste_carrier,
            ep_ax,
            core_id=core_id,
            scatter_dimension=0,
            min_collective_bytes=min_b,
        ),
    )

  _dispatch_fp8.defvjp(_dispatch_fp8_fwd, _dispatch_fp8_bwd)
  return _dispatch_fp8(tokens_local)


def ep_routing_weights_dispatch(
    topk_weights_local: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """EP all-gather (TC) + SC0 psum_scatter of `topk_weights` (`1.31 MB` -> `80 KB`)."""
  ep_ax = cfg.ep_axis_names

  @jax.custom_vjp
  def _ag_w(w: jax.Array) -> jax.Array:
    return sc_all_gather(
        w,
        ep_ax,
        core_id=_ep_core_id(cfg),
        axis=0,
        min_collective_bytes=0,
    )

  def _ag_w_fwd(w: jax.Array):
    return _ag_w(w), None

  def _ag_w_bwd(_, gw_ep: jax.Array):
    return (
        sc_psum_scatter(
            gw_ep,
            ep_ax,
            core_id=_ep_core_id(cfg),
            scatter_dimension=0,
            min_collective_bytes=min(cfg.sc_min_collective_bytes, 250_000),
        ),
    )

  _ag_w.defvjp(_ag_w_fwd, _ag_w_bwd)
  return _ag_w(topk_weights_local)


def ep_routing_ids_dispatch(
    topk_ids_local: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """SC0-pinned EP all-gather of integer `topk_ids` (`1.31-2.62 MB`)."""
  ids_i32 = jax.lax.stop_gradient(topk_ids_local.astype(jnp.int32))
  return sc_all_gather(
      ids_i32,
      cfg.ep_axis_names,
      core_id=_ep_core_id(cfg),
      axis=0,
      min_collective_bytes=0,
  )


def ep_token_combine_bwd_only(
    g_local_shaped: jax.Array,
    cfg: QwenConfig,
    *,
    as_partial_tokens: bool = False,
    precomputed_amax_ep: jax.Array | None = None,
) -> jax.Array | MoePartialTokens:
  """Exact transpose of `ep_token_combine` (`[B_mb, S, D]` or 3D -> `[T_ep, D]` or 3D) pinned to SC0."""
  core_id = _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes
  emb_dim = cfg.emb_dim
  use_3d = (
      cfg.moe_use_tc_ragged and cfg.moe_3d_ep_tokens and (emb_dim % 128 == 0)
  )
  g_in = (
      g_local_shaped.reshape(-1, emb_dim // 128, 128)
      if use_3d
      else g_local_shaped.reshape(-1, emb_dim)
  )
  if as_partial_tokens:
    qtype = jnp.float8_e5m2
    qmax = float(jnp.finfo(qtype).max)
    if precomputed_amax_ep is not None:
      amax_ep = precomputed_amax_ep.reshape(1, 1)
    else:
      amax_l = jax.lax.stop_gradient(jnp.max(jnp.abs(g_in.astype(jnp.float32))))
      if cfg.sc0_bwd_dy_pmax and core_id is not None:
        amax_ep = sc_pmax(amax_l, ep_ax, core_id=core_id).reshape(1, 1)
      else:
        amax_ep = _ep_scalar_pmax(amax_l, cfg).reshape(1, 1)
    dy_scale = jax.lax.stop_gradient(
        jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep))
    )
    inv_scale = (1.0 / dy_scale).reshape(1, *([1] * (g_in.ndim - 1)))
    dy_codes_l = jnp.clip(
        g_in.astype(jnp.float32) * inv_scale, -qmax, qmax
    ).astype(qtype)
    dy_codes_ep = sc_all_gather(
        dy_codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
    )
    ep_shape = (g_in.shape[0] * cfg.ep_size, *g_in.shape[1:])
    y_zero = jax.lax.broadcast(jnp.zeros((), dtype=g_in.dtype), ep_shape)
    return MoePartialTokens(
        y_ep=y_zero, dy_codes=dy_codes_ep, dy_scale=dy_scale
    )

  if cfg.moe_fp8_bwd_dispatch:
    if cfg.moe_use_forked_3d_gmmv2 and cfg.moe_bwd_dy_fp8_e5m2:
      qtype = jnp.float8_e5m2
      qmax = float(jnp.finfo(qtype).max)
      if precomputed_amax_ep is not None:
        amax_ep = precomputed_amax_ep.reshape(1, 1)
      else:
        amax_l = jax.lax.stop_gradient(
            jnp.max(jnp.abs(g_in.astype(jnp.float32)))
        )
        if cfg.sc0_bwd_dy_pmax and core_id is not None:
          amax_ep = sc_pmax(amax_l, ep_ax, core_id=core_id).reshape(1, 1)
        else:
          amax_ep = _ep_scalar_pmax(amax_l, cfg).reshape(1, 1)
      dy_scale = jax.lax.stop_gradient(
          jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep))
      )
      inv_scale = (1.0 / dy_scale).reshape(1, *([1] * (g_in.ndim - 1)))
      dy_codes_l = jnp.clip(
          g_in.astype(jnp.float32) * inv_scale, -qmax, qmax
      ).astype(qtype)
      dy_codes_ep = sc_all_gather(
          dy_codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
      )
      return (
          dy_codes_ep.astype(jnp.float32)
          * dy_scale.reshape(1, *([1] * (g_in.ndim - 1)))
      ).astype(g_in.dtype)
    qval, scale = quantize_rows_fp8(g_in, jnp.float8_e4m3fn)
    qval_ep = sc_all_gather(
        qval, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
    )
    scale_ep = jax.lax.all_gather(scale, axis_name=ep_ax, axis=0, tiled=True)
    return dequantize_rows_fp8(qval_ep, scale_ep, g_in.dtype)
  return sc_all_gather(
      g_in, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
  )


def combine_bwd_bridge(
    cfg: QwenConfig,
    unperm_prev: jax.Array | MoePartialTokens,
    base_prev: jax.Array,
    x_mb1_val: jax.Array,
) -> jax.Array:
  """Zero-residual cross-layer combine bridge routing `dx_mb1` -> `(combine_bwd(dx_mb1), dx_mb1, 0)`."""
  as_pt = isinstance(unperm_prev, MoePartialTokens)

  @jax.custom_vjp
  def _bridge(u_prev, b_prev, x_val: jax.Array) -> jax.Array:
    del u_prev, b_prev
    return x_val

  def _bridge_fwd(u_prev, b_prev, x_val: jax.Array):
    del u_prev, b_prev
    return x_val, ()

  def _bridge_bwd(_, g_x_mb1: jax.Array):
    return (
        ep_token_combine_bwd_only(g_x_mb1, cfg, as_partial_tokens=as_pt),
        g_x_mb1,
        jnp.zeros_like(g_x_mb1),
    )

  _bridge.defvjp(_bridge_fwd, _bridge_bwd)
  return _bridge(unperm_prev, base_prev, x_mb1_val)


def ep_token_combine(
    partial_tokens_ep: jax.Array | MoePartialTokens, cfg: QwenConfig
) -> jax.Array:
  """SC0-pinned EP token combine `psum_scatter` (BF16+ enforced; NEVER FP8)."""
  if cfg.moe_fp8_combine_sc0:
    raise ValueError(
        "moe_fp8_combine_sc0=True violates the Global Mixed-Precision Contract:"
        " all reductions (EP Combine ReduceScatter) MUST run in bfloat16 or"
        " float32."
    )
  fwd_core_id = None if cfg.ep_combine_fwd_unpinned else _ep_core_id(cfg)
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes
  as_pt = isinstance(partial_tokens_ep, MoePartialTokens)

  @jax.custom_vjp
  def _combine(p_ep: jax.Array | MoePartialTokens) -> jax.Array:
    y_ep = p_ep.y_ep if isinstance(p_ep, MoePartialTokens) else p_ep
    return sc_psum_scatter(
        y_ep,
        ep_ax,
        core_id=fwd_core_id,
        scatter_dimension=0,
        min_collective_bytes=min_b,
    )

  def _combine_fwd(p_ep: jax.Array | MoePartialTokens):
    return _combine(p_ep), None

  def _combine_bwd(_, g_local: jax.Array):
    return (ep_token_combine_bwd_only(g_local, cfg, as_partial_tokens=as_pt),)

  _combine.defvjp(_combine_fwd, _combine_bwd)
  return _combine(partial_tokens_ep)


# ---------------------------------------------------------------------------
# SC1 FSDP dense gather (bf16/fp32 & P1 gather8 FP8) + MXU matmul
# ---------------------------------------------------------------------------


def _dense_scalar_pmax(
    amax_local: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    no_combine: bool = True,
) -> jax.Array:
  """Computes `float32` `pmax` for dense weight scale reduction."""
  assert_bf16_or_fp32_reduction(amax_local, "_dense_scalar_pmax")
  if no_combine:
    with set_xla_metadata(_scheduling_group_id="noop"):
      return jax.lax.pmax(amax_local, axis_name=axis_name)
  return jax.lax.pmax(amax_local, axis_name=axis_name)


def compute_hoisted_dense_absmax(
    w_stacked_local: jax.Array,
    cfg: QwenConfig,
    *,
    is_fsdp_expert: bool = True,
) -> jax.Array:
  """Precomputes `[num_superblocks, 1, 1]` `float32` dense weight `absmax` before `scan`."""
  red_axes = tuple(range(1, w_stacked_local.ndim))
  amax_l = jax.lax.stop_gradient(
      jnp.max(jnp.abs(w_stacked_local), axis=red_axes).astype(jnp.float32)
  )
  pmax_ax = (
      cfg.all_sharded_axes if is_fsdp_expert else cfg.dense_stage2_axis_names
  )
  amax_g = _dense_scalar_pmax(
      amax_l, pmax_ax, no_combine=cfg.fp8_amax_no_combine
  )
  return jax.lax.stop_gradient(amax_g[:, None, None])


def gather_dense_weight_fp8(
    w_sharded: jax.Array,
    cfg: QwenConfig,
    *,
    absmax: jax.Array | None = None,
    sharded_axis: int = 0,
    is_fsdp_expert: bool = False,
    start_after: Any = None,
) -> Fp8DenseWeight:
  """Gathers a dense weight in `float8_e4m3fn` (`gather8`) on SC1 with BF16+ `ReduceScatter` bwd."""
  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
  w_in = (
      order_after(w_sharded, start_after)
      if start_after is not None
      else w_sharded
  )
  if cfg.num_devices == 1:

    @jax.custom_vjp
    def _quant_single(w: jax.Array, amax_opt: Any) -> Fp8DenseWeight:
      amax_v = (
          amax_opt.astype(jnp.float32).reshape(())
          if amax_opt is not None
          else jnp.max(jnp.abs(w.astype(jnp.float32)))
      )
      amax_v = jax.lax.stop_gradient(amax_v)
      scale_s = jax.lax.stop_gradient(
          jnp.where(amax_v > 0.0, amax_v / qmax, jnp.ones_like(amax_v))
      )
      codes_l = jax.lax.stop_gradient(
          jnp.clip(w.astype(jnp.float32) / scale_s, -qmax, qmax).astype(
              jnp.float8_e4m3fn
          )
      )
      scale_out = scale_s.reshape((1,) * codes_l.ndim)
      w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w.dtype), codes_l.shape)
      return Fp8DenseWeight(w_ste=w_ste, codes=codes_l, scale=scale_out)

    def _quant_single_fwd(w: jax.Array, amax_opt: Any):
      return _quant_single(w, amax_opt), None

    def _quant_single_bwd(_, g_carrier: Fp8DenseWeight):
      dw_full = g_carrier.w_ste
      assert_bf16_or_fp32_reduction(dw_full, "gather_dense_weight_fp8_bwd")
      return dw_full, None

    _quant_single.defvjp(_quant_single_fwd, _quant_single_bwd)
    return _quant_single(w_in, absmax)

  ag_core = _fsdp_ag_core_id(cfg, is_bwd_remat=False)
  rs_core = _dense_rs_core_id(cfg)
  s2_ax = cfg.dense_stage2_axis_names
  min_b = cfg.sc_min_collective_bytes
  pmax_ax = cfg.all_sharded_axes if is_fsdp_expert else s2_ax

  def _do_gather_codes(codes: jax.Array) -> jax.Array:
    if is_fsdp_expert:
      if cfg.dense_1stage_3d_fsdp:
        return sc_3d_all_gather(
            codes,
            cfg.all_sharded_axes,
            core_id=ag_core,
            axis=sharded_axis,
            min_collective_bytes=min_b,
        )
      c_ep = sc_all_gather(
          codes,
          "expert",
          core_id=ag_core,
          axis=sharded_axis,
          min_collective_bytes=min_b,
      )
      return sc_all_gather(
          c_ep,
          s2_ax,
          core_id=ag_core,
          axis=sharded_axis,
          min_collective_bytes=min_b,
      )
    return sc_all_gather(
        codes,
        s2_ax,
        core_id=ag_core,
        axis=sharded_axis,
        min_collective_bytes=min_b,
    )

  if absmax is not None:

    @jax.custom_vjp
    def _gather_hoisted(w: jax.Array, amax_in: jax.Array) -> Fp8DenseWeight:
      amax_f32 = jax.lax.stop_gradient(amax_in.astype(jnp.float32).reshape(()))
      scale_s = jax.lax.stop_gradient(
          jnp.where(amax_f32 > 0.0, amax_f32 / qmax, jnp.ones_like(amax_f32))
      )
      codes_l = jax.lax.stop_gradient(
          jnp.clip(w.astype(jnp.float32) / scale_s, -qmax, qmax).astype(
              jnp.float8_e4m3fn
          )
      )
      codes_f = _do_gather_codes(codes_l)
      scale_out = scale_s.reshape((1,) * codes_f.ndim)
      w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w.dtype), codes_f.shape)
      return Fp8DenseWeight(w_ste=w_ste, codes=codes_f, scale=scale_out)

    def _gather_hoisted_fwd(w: jax.Array, amax_in: jax.Array):
      return _gather_hoisted(w, amax_in), None

    def _gather_hoisted_bwd(_, g_carrier: Fp8DenseWeight):
      dw_full = g_carrier.w_ste
      assert_bf16_or_fp32_reduction(dw_full, "gather_dense_weight_fp8_bwd")
      if is_fsdp_expert and cfg.dense_1stage_3d_fsdp:
        return (
            sc_3d_psum_scatter(
                dw_full,
                cfg.all_sharded_axes,
                core_id=rs_core,
                scatter_dimension=sharded_axis,
                min_collective_bytes=min_b,
            ),
            None,
        )
      dw_fsdp = sc_psum_scatter(
          dw_full,
          s2_ax,
          core_id=rs_core,
          scatter_dimension=sharded_axis,
          min_collective_bytes=min_b,
      )
      if is_fsdp_expert:
        dw_sharded = sc_psum_scatter(
            dw_fsdp,
            "expert",
            core_id=rs_core,
            scatter_dimension=sharded_axis,
            min_collective_bytes=min_b,
        )
        return dw_sharded, None
      return psum_bf16_or_fp32(dw_fsdp, "expert"), None

    _gather_hoisted.defvjp(_gather_hoisted_fwd, _gather_hoisted_bwd)
    return _gather_hoisted(w_in, absmax)

  @jax.custom_vjp
  def _gather_dyn(w: jax.Array) -> Fp8DenseWeight:
    amax_l = jax.lax.stop_gradient(jnp.max(jnp.abs(w.astype(jnp.float32))))
    amax_g = _dense_scalar_pmax(
        amax_l, pmax_ax, no_combine=cfg.fp8_amax_no_combine
    )
    scale_s = jax.lax.stop_gradient(
        jnp.where(amax_g > 0.0, amax_g / qmax, jnp.ones_like(amax_g))
    )
    codes_l = jax.lax.stop_gradient(
        jnp.clip(w.astype(jnp.float32) / scale_s, -qmax, qmax).astype(
            jnp.float8_e4m3fn
        )
    )
    codes_f = _do_gather_codes(codes_l)
    scale_out = scale_s.reshape((1,) * codes_f.ndim)
    w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w.dtype), codes_f.shape)
    return Fp8DenseWeight(w_ste=w_ste, codes=codes_f, scale=scale_out)

  def _gather_dyn_fwd(w: jax.Array):
    return _gather_dyn(w), None

  def _gather_dyn_bwd(_, g_carrier: Fp8DenseWeight):
    dw_full = g_carrier.w_ste
    assert_bf16_or_fp32_reduction(dw_full, "gather_dense_weight_fp8_bwd")
    if is_fsdp_expert and cfg.dense_1stage_3d_fsdp:
      return (
          sc_3d_psum_scatter(
              dw_full,
              cfg.all_sharded_axes,
              core_id=rs_core,
              scatter_dimension=sharded_axis,
              min_collective_bytes=min_b,
          ),
      )
    dw_fsdp = sc_psum_scatter(
        dw_full,
        s2_ax,
        core_id=rs_core,
        scatter_dimension=sharded_axis,
        min_collective_bytes=min_b,
    )
    if is_fsdp_expert:
      dw_sharded = sc_psum_scatter(
          dw_fsdp,
          "expert",
          core_id=rs_core,
          scatter_dimension=sharded_axis,
          min_collective_bytes=min_b,
      )
      return (dw_sharded,)
    return (psum_bf16_or_fp32(dw_fsdp, "expert"),)

  _gather_dyn.defvjp(_gather_dyn_fwd, _gather_dyn_bwd)
  return _gather_dyn(w_in)


def gather_dense_weight(
    w_sharded: jax.Array,
    cfg: QwenConfig,
    *,
    sharded_axis: int = 0,
    is_fsdp_expert: bool = False,
    start_after: Any = None,
    use_fp8: bool = False,
    absmax: jax.Array | None = None,
) -> jax.Array | Fp8DenseWeight:
  """Gathers a dense weight on SC1 (`fsdp`) with BF16+ SC1 `psum_scatter` backward."""
  if use_fp8:
    return gather_dense_weight_fp8(
        w_sharded,
        cfg,
        absmax=absmax,
        sharded_axis=sharded_axis,
        is_fsdp_expert=is_fsdp_expert,
        start_after=start_after,
    )
  if cfg.num_devices == 1:
    return (
        order_after(w_sharded, start_after)
        if start_after is not None
        else w_sharded
    )

  ag_core = _fsdp_ag_core_id(cfg, is_bwd_remat=False)
  rs_core = _dense_rs_core_id(cfg)
  s2_ax = cfg.dense_stage2_axis_names
  min_b = cfg.sc_min_collective_bytes
  w_in = (
      order_after(w_sharded, start_after)
      if start_after is not None
      else w_sharded
  )

  def _do_gather(w: jax.Array, sc1_core: int | None) -> jax.Array:
    if is_fsdp_expert:
      if cfg.dense_1stage_3d_fsdp:
        return sc_3d_all_gather(
            w,
            cfg.all_sharded_axes,
            core_id=sc1_core,
            axis=sharded_axis,
            min_collective_bytes=min_b,
        )
      w_ep = sc_all_gather(
          w,
          "expert",
          core_id=sc1_core,
          axis=sharded_axis,
          min_collective_bytes=min_b,
      )
      return sc_all_gather(
          w_ep,
          s2_ax,
          core_id=sc1_core,
          axis=sharded_axis,
          min_collective_bytes=min_b,
      )
    return sc_all_gather(
        w,
        s2_ax,
        core_id=sc1_core,
        axis=sharded_axis,
        min_collective_bytes=min_b,
    )

  @jax.custom_vjp
  def _gather_fn(w: jax.Array) -> jax.Array:
    return _do_gather(w, ag_core)

  def _gather_fwd(w: jax.Array):
    return _do_gather(w, ag_core), None

  def _gather_bwd(_, gw_full: jax.Array):
    assert_bf16_or_fp32_reduction(gw_full, "gather_dense_weight_bwd")
    if is_fsdp_expert and cfg.dense_1stage_3d_fsdp:
      return (
          sc_3d_psum_scatter(
              gw_full,
              cfg.all_sharded_axes,
              core_id=rs_core,
              scatter_dimension=sharded_axis,
              min_collective_bytes=min_b,
          ),
      )
    gw_fsdp = sc_psum_scatter(
        gw_full,
        s2_ax,
        core_id=rs_core,
        scatter_dimension=sharded_axis,
        min_collective_bytes=min_b,
    )
    if is_fsdp_expert:
      gw_sharded = sc_psum_scatter(
          gw_fsdp,
          "expert",
          core_id=rs_core,
          scatter_dimension=sharded_axis,
          min_collective_bytes=min_b,
      )
      return (gw_sharded,)
    gw_sharded = psum_bf16_or_fp32(gw_fsdp, "expert")
    return (gw_sharded,)

  _gather_fn.defvjp(_gather_fwd, _gather_bwd)
  return _gather_fn(w_in)


def _concat_axis1_2d(
    weights: Sequence[jax.Array],
) -> tuple[jax.Array, tuple[int, ...]]:
  if len(weights) == 1:
    return weights[0], (weights[0].shape[1],)
  cols = tuple(w.shape[1] for w in weights)
  return jnp.concatenate(weights, axis=1), cols


def _split_axis1_2d(
    bundled: jax.Array, cols: tuple[int, ...]
) -> tuple[jax.Array, ...]:
  if len(cols) == 1:
    return (bundled,)
  out = []
  offset = 0
  for c in cols:
    out.append(bundled[:, offset : offset + c])
    offset += c
  return tuple(out)


def _run_2d_bundle_collectives(
    weights: Sequence[jax.Array],
    cfg: QwenConfig,
    *,
    stages: tuple[str, ...],
    is_fsdp_expert: bool,
    start_after: Any = None,
    min_collective_bytes: int | None = None,
) -> tuple[jax.Array, ...]:
  """Gathers 2D `[d_local, C_i]` dense shards via concatenated SC1 collectives."""
  if not weights:
    return ()
  if cfg.num_devices == 1 or (stages == ("stage1",) and not is_fsdp_expert):
    if start_after is not None:
      return tuple(order_after(w, start_after) for w in weights)
    return tuple(weights)

  ag_core = _fsdp_ag_core_id(cfg, is_bwd_remat=False)
  rs_core = _dense_rs_core_id(cfg)
  s2_ax = cfg.dense_stage2_axis_names
  min_b = (
      cfg.sc_min_collective_bytes
      if min_collective_bytes is None
      else min_collective_bytes
  )
  cat_w, cols = _concat_axis1_2d(weights)
  cat_in = order_after(cat_w, start_after) if start_after is not None else cat_w

  @jax.custom_vjp
  def _gather_cat(x: jax.Array) -> jax.Array:
    if (
        stages == ("stage1", "stage2")
        and is_fsdp_expert
        and cfg.dense_1stage_3d_fsdp
    ):
      return sc_3d_all_gather(
          x,
          cfg.all_sharded_axes,
          core_id=ag_core,
          axis=0,
          min_collective_bytes=min_b,
      )
    cur = x
    for st in stages:
      if st == "stage1" and is_fsdp_expert:
        cur = sc_all_gather(
            cur, "expert", core_id=ag_core, axis=0, min_collective_bytes=min_b
        )
      elif st == "stage2":
        cur = sc_all_gather(
            cur, s2_ax, core_id=ag_core, axis=0, min_collective_bytes=min_b
        )
    return cur

  def _fwd(x: jax.Array):
    return _gather_cat(x), None

  def _bwd(_, g: jax.Array):
    assert_bf16_or_fp32_reduction(g, "gather_dense_bundled_bwd")
    if (
        stages == ("stage1", "stage2")
        and is_fsdp_expert
        and cfg.dense_1stage_3d_fsdp
    ):
      return (
          sc_3d_psum_scatter(
              g,
              cfg.all_sharded_axes,
              core_id=rs_core,
              scatter_dimension=0,
              min_collective_bytes=min_b,
          ),
      )
    cur = g
    for st in reversed(stages):
      if st == "stage2":
        cur = sc_psum_scatter(
            cur,
            s2_ax,
            core_id=rs_core,
            scatter_dimension=0,
            min_collective_bytes=min_b,
        )
        if not is_fsdp_expert:
          cur = psum_bf16_or_fp32(cur, "expert")
      elif st == "stage1" and is_fsdp_expert:
        cur = sc_psum_scatter(
            cur,
            "expert",
            core_id=rs_core,
            scatter_dimension=0,
            min_collective_bytes=min_b,
        )
    return (cur,)

  _gather_cat.defvjp(_fwd, _bwd)
  return _split_axis1_2d(_gather_cat(cat_in), cols)


def gather_dense_stage1_bundled(
    weights: Sequence[jax.Array],
    cfg: QwenConfig,
    *,
    is_fsdp_expert: bool = True,
    start_after: Any = None,
    min_collective_bytes: int = 0,
) -> tuple[jax.Array, ...]:
  """Bundles Stage-1 `'expert'` gather/psum_scatter on SC1 for `[d_local, C_i]` shards."""
  return _run_2d_bundle_collectives(
      weights,
      cfg,
      stages=("stage1",),
      is_fsdp_expert=is_fsdp_expert,
      start_after=start_after,
      min_collective_bytes=min_collective_bytes,
  )


def gather_dense_stage2_bundled(
    weights: Sequence[jax.Array],
    cfg: QwenConfig,
    *,
    is_fsdp_expert: bool = True,
    start_after: Any = None,
    min_collective_bytes: int | None = None,
) -> tuple[jax.Array, ...]:
  """Bundles Stage-2 `dense_stage2_axis_names` gather/psum_scatter for `[d_ep, C_i]` shards."""
  return _run_2d_bundle_collectives(
      weights,
      cfg,
      stages=("stage2",),
      is_fsdp_expert=is_fsdp_expert,
      start_after=start_after,
      min_collective_bytes=min_collective_bytes,
  )


def gather_dense_weights_bundled(
    weights: Sequence[jax.Array],
    cfg: QwenConfig,
    *,
    is_fsdp_expert: bool = True,
    start_after: Any = None,
    min_collective_bytes: int | None = None,
) -> tuple[jax.Array, ...]:
  """Bundles both Stage-1 and Stage-2 dense gathers for `[d_local, C_i]` shards."""
  return _run_2d_bundle_collectives(
      weights,
      cfg,
      stages=("stage1", "stage2"),
      is_fsdp_expert=is_fsdp_expert,
      start_after=start_after,
      min_collective_bytes=min_collective_bytes,
  )


def _quantize_cols_2d_fp8(
    x_2d: jax.Array, qtype: jax.typing.DTypeLike
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x_2d` `[M, C]` per-column (`axis=0`), returning `(codes [M, C], scale [1, C])`."""
  qdt = jnp.dtype(qtype)
  qmax = float(jnp.finfo(qdt).max)
  x_f32 = x_2d.astype(jnp.float32)
  amax = jnp.max(jnp.abs(x_f32), axis=0, keepdims=True)
  scale = jnp.where(amax > 0.0, amax / qmax, jnp.ones_like(amax))
  codes = jnp.clip(x_f32 / scale, -qmax, qmax).astype(qdt)
  return codes, scale


@jax.custom_vjp
def _dense_fp8_matmul_carrier(x: jax.Array, w: Fp8DenseWeight) -> jax.Array:
  """Native MXU `fp8_e4m3fn x fp8_e4m3fn` 2D matmul (`convolution fusion`) with BF16+ `dw`."""
  k_dim, n_dim = w.codes.shape
  x_2d = x.reshape(-1, k_dim)
  x_codes, x_scale = quantize_rows_fp8(x_2d, jnp.float8_e4m3fn)
  y_f32 = jax.lax.dot_general(
      x_codes,
      w.codes,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  y_2d = (y_f32 * x_scale * w.scale.reshape(1, -1)).astype(x.dtype)
  return y_2d.reshape(*x.shape[:-1], n_dim)


def _dense_fp8_matmul_carrier_fwd(x: jax.Array, w: Fp8DenseWeight):
  y = _dense_fp8_matmul_carrier(x, w)
  return y, (x, w.codes, w.scale, jnp.zeros((), dtype=w.dtype))


def _dense_fp8_matmul_carrier_bwd(res, dy: jax.Array):
  """Backward MXU FP8 dgrad (`e5m2 x e4m3fn`) and BF16+ wgrad (`e4m3fn x e5m2`)."""
  x, w_codes, w_scale, w_dtype_carrier = res
  w_dtype = w_dtype_carrier.dtype
  k_dim, n_dim = w_codes.shape
  dy_2d = dy.reshape(-1, n_dim)
  w_s = w_scale.reshape(1, -1)

  # dgrad: dx = (dy * w_scale) @ w_codes^T using fp8_e5m2 x fp8_e4m3fn MXU.
  dy_scaled = dy_2d.astype(jnp.float32) * w_s
  dy_row_codes, dy_row_scale = quantize_rows_fp8(dy_scaled, jnp.float8_e5m2)
  dx_f32 = jax.lax.dot_general(
      dy_row_codes,
      w_codes,
      (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  dx = (dx_f32 * dy_row_scale).astype(x.dtype).reshape(x.shape)

  # wgrad: dw = x^T @ dy using fp8_e4m3fn x fp8_e5m2 MXU -> BF16/FP32 output.
  x_2d = x.reshape(-1, k_dim)
  x_col_codes, x_col_scale = _quantize_cols_2d_fp8(x_2d, jnp.float8_e4m3fn)
  dy_col_codes, dy_col_scale = _quantize_cols_2d_fp8(dy_2d, jnp.float8_e5m2)
  dw_f32 = jax.lax.dot_general(
      x_col_codes,
      dy_col_codes,
      (((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  dw = (dw_f32 * (jnp.swapaxes(x_col_scale, 0, 1) * dy_col_scale)).astype(
      w_dtype
  )
  assert_bf16_or_fp32_reduction(dw, "dense_fp8_matmul_wgrad")
  z_codes = jax.lax.broadcast(jnp.zeros((), dtype=w_codes.dtype), w_codes.shape)
  z_scale = jnp.zeros_like(w_scale)
  return dx, Fp8DenseWeight(w_ste=dw, codes=z_codes, scale=z_scale)


_dense_fp8_matmul_carrier.defvjp(
    _dense_fp8_matmul_carrier_fwd, _dense_fp8_matmul_carrier_bwd
)


def dense_fp8_matmul(
    x: jax.Array,
    w: jax.Array | Fp8DenseWeight,
    *,
    quantize_bf16_weight: bool = False,
) -> jax.Array:
  """Executes `x @ w`, using native `fp8_e4m3fn x fp8_e4m3fn` MXU when `w` is `Fp8DenseWeight`."""
  if isinstance(w, Fp8DenseWeight):
    return _dense_fp8_matmul_carrier(x, w)
  if quantize_bf16_weight:
    qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jax.lax.stop_gradient(
        jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0, keepdims=True)
    )
    w_scale = jax.lax.stop_gradient(
        jnp.where(amax > 0.0, amax / qmax, jnp.ones_like(amax))
    )
    w_codes = jax.lax.stop_gradient(
        jnp.clip(w.astype(jnp.float32) / w_scale, -qmax, qmax).astype(
            jnp.float8_e4m3fn
        )
    )

    @jax.custom_vjp
    def _unsharded_fp8_mm(x_in: jax.Array, w_in: jax.Array) -> jax.Array:
      carrier = Fp8DenseWeight(
          w_ste=jax.lax.broadcast(jnp.zeros((), dtype=w_in.dtype), w_in.shape),
          codes=w_codes,
          scale=w_scale,
      )
      return _dense_fp8_matmul_carrier(x_in, carrier)

    def _unsharded_fwd(x_in: jax.Array, w_in: jax.Array):
      carrier = Fp8DenseWeight(
          w_ste=jax.lax.broadcast(jnp.zeros((), dtype=w_in.dtype), w_in.shape),
          codes=w_codes,
          scale=w_scale,
      )
      y, res = _dense_fp8_matmul_carrier_fwd(x_in, carrier)
      return y, res

    def _unsharded_bwd(res, dy: jax.Array):
      dx, g_carrier = _dense_fp8_matmul_carrier_bwd(res, dy)
      return dx, g_carrier.w_ste

    _unsharded_fp8_mm.defvjp(_unsharded_fwd, _unsharded_bwd)
    return _unsharded_fp8_mm(x, w)
  return jnp.matmul(x, w)


@jax.custom_vjp
def _dense_fp8_einsum_bshc_carrier(
    x: jax.Array, w: Fp8DenseWeight
) -> jax.Array:
  """Native MXU `fp8_e4m3fn x fp8_e4m3fn` 4D `einsum('bsd,dhc->bshc')` preserving `{3,2,1,0}` layout."""
  bsz, seq_len, d_dim = x.shape
  x_codes_2d, x_scale_2d = quantize_rows_fp8(
      x.reshape(bsz * seq_len, d_dim), jnp.float8_e4m3fn
  )
  x_codes = x_codes_2d.reshape(bsz, seq_len, d_dim)
  x_scale = x_scale_2d.reshape(bsz, seq_len, 1, 1)
  y_f32 = jax.lax.dot_general(
      x_codes,
      w.codes,
      (((2,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  w_s = (
      w.scale.reshape(1, 1, *w.scale.shape[1:])
      if w.scale.ndim == 3
      else w.scale.reshape(1, 1, 1, 1)
  )
  return (y_f32 * x_scale * w_s).astype(x.dtype)


def _dense_fp8_einsum_bshc_fwd(x: jax.Array, w: Fp8DenseWeight):
  y = _dense_fp8_einsum_bshc_carrier(x, w)
  return y, (x, w.codes, w.scale, jnp.zeros((), dtype=w.dtype))


def _dense_fp8_einsum_bshc_bwd(res, dy: jax.Array):
  """Backward MXU FP8 dgrad & wgrad for 4D head-minor `einsum('bsd,dhc->bshc')`."""
  x, w_codes, w_scale, w_dtype_carrier = res
  w_dtype = w_dtype_carrier.dtype
  bsz, seq_len, d_dim = x.shape
  _, h_dim, c_dim = w_codes.shape
  w_s = (
      w_scale.reshape(1, 1, *w_scale.shape[1:])
      if w_scale.ndim == 3
      else w_scale.reshape(1, 1, 1, 1)
  )

  dy_scaled = (dy.astype(jnp.float32) * w_s).reshape(
      bsz * seq_len, h_dim * c_dim
  )
  dy_row_codes_2d, dy_row_scale_2d = quantize_rows_fp8(
      dy_scaled, jnp.float8_e5m2
  )
  dy_row_codes = dy_row_codes_2d.reshape(bsz, seq_len, h_dim, c_dim)
  dx_f32 = jax.lax.dot_general(
      dy_row_codes,
      w_codes,
      (((2, 3), (1, 2)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  dx = (dx_f32 * dy_row_scale_2d.reshape(bsz, seq_len, 1)).astype(x.dtype)

  x_col_codes_2d, x_col_scale_2d = _quantize_cols_2d_fp8(
      x.reshape(bsz * seq_len, d_dim), jnp.float8_e4m3fn
  )
  dy_col_codes_2d, dy_col_scale_2d = _quantize_cols_2d_fp8(
      dy.reshape(bsz * seq_len, h_dim * c_dim), jnp.float8_e5m2
  )
  dw_f32 = jax.lax.dot_general(
      x_col_codes_2d.reshape(bsz, seq_len, d_dim),
      dy_col_codes_2d.reshape(bsz, seq_len, h_dim, c_dim),
      (((0, 1), (0, 1)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  dw_scale = x_col_scale_2d.reshape(d_dim, 1, 1) * dy_col_scale_2d.reshape(
      1, h_dim, c_dim
  )
  dw = (dw_f32 * dw_scale).astype(w_dtype)
  assert_bf16_or_fp32_reduction(dw, "dense_fp8_einsum_bshc_wgrad")
  z_codes = jax.lax.broadcast(jnp.zeros((), dtype=w_codes.dtype), w_codes.shape)
  z_scale = jnp.zeros_like(w_scale)
  return dx, Fp8DenseWeight(w_ste=dw, codes=z_codes, scale=z_scale)


_dense_fp8_einsum_bshc_carrier.defvjp(
    _dense_fp8_einsum_bshc_fwd, _dense_fp8_einsum_bshc_bwd
)


def dense_fp8_einsum_bshc(
    x: jax.Array, w: jax.Array | Fp8DenseWeight
) -> jax.Array:
  """Executes `einsum('bsd,dhc->bshc', x, w)` in BF16/FP32 or native MXU FP8."""
  if isinstance(w, Fp8DenseWeight):
    return _dense_fp8_einsum_bshc_carrier(x, w)
  return jnp.einsum("bsd,dhc->bshc", x, w)


def dense_fp8_gdn(x: jax.Array, w: jax.Array | Fp8DenseWeight) -> jax.Array:
  """GDN `in_proj_qkvz` / `out_proj` matmul (native FP8 MXU when `w` is `Fp8DenseWeight`)."""
  return dense_fp8_matmul(x, w)


def dense_fp8_attn(
    x: jax.Array,
    w: jax.Array | Fp8DenseWeight,
    *,
    is_4d_head_proj: bool = False,
) -> jax.Array:
  """Full-Attention QKV (4D head-minor `einsum`) or `out_proj` (2D matmul) with optional FP8."""
  if is_4d_head_proj:
    return dense_fp8_einsum_bshc(x, w)
  return dense_fp8_matmul(x, w)


def dense_fp8_shared_expert(
    normed_post_mixer: jax.Array,
    shared_wi_0: jax.Array | Fp8DenseWeight,
    shared_wi_1: jax.Array | Fp8DenseWeight,
    shared_wo: jax.Array | Fp8DenseWeight,
) -> jax.Array:
  """Shared-expert SwiGLU MLP supporting native FP8 MXU when weights are `Fp8DenseWeight`."""
  g0 = dense_fp8_matmul(normed_post_mixer, shared_wi_0)
  u0 = dense_fp8_matmul(normed_post_mixer, shared_wi_1)
  act = (jax.nn.silu(g0.astype(jnp.float32)) * u0.astype(jnp.float32)).astype(
      normed_post_mixer.dtype
  )
  return dense_fp8_matmul(act, shared_wo)


def replicate_small_param(
    w_rep: jax.Array, cfg: QwenConfig | None = None
) -> jax.Array:
  """Identity in forward; all-reduces replicated parameter gradient over core-minor axes in BF16+."""
  if cfg is None:
    axes = ("fsdp", "expert")
  elif cfg.decouple_ep_d2d and cfg.ep_core_minor:
    axes = ("fsdp", "expert", "ep_d2d")
  else:
    axes = cfg.all_sharded_axes

  @jax.custom_vjp
  def _rep(w: jax.Array) -> jax.Array:
    return w

  def _rep_fwd(w: jax.Array):
    return w, None

  def _rep_bwd(_, gw: jax.Array):
    gw_sum = psum_bf16_or_fp32(gw, axes)
    return (gw_sum.astype(gw.dtype),)

  _rep.defvjp(_rep_fwd, _rep_bwd)
  return _rep(w_rep)


# ---------------------------------------------------------------------------
# Qwen3.5 / Qwen MAX Layer Normalizations (Pure JAX)
# ---------------------------------------------------------------------------


def rms_norm_zero_centered(
    x: jax.Array, scale: jax.Array, eps: float = 1e-6
) -> jax.Array:
  """Qwen3NextRMSNorm: `(x / rms(x)) * (1.0 + scale)` accumulated in fp32."""
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps
  )
  normed = x_f32 * inv_rms * (1.0 + scale.astype(jnp.float32))
  return normed.astype(orig_dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def rms_norm_gated(
    x: jax.Array, gate_z: jax.Array, scale: jax.Array, eps: float = 1e-6
) -> jax.Array:
  """Qwen3NextRMSNormGated: `rms_norm(x, scale) * silu(gate_z)` in fp32."""
  orig_dtype = x.dtype
  x_var = jax.lax.optimization_barrier(x).astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(jnp.square(x_var), axis=-1, keepdims=True) + eps
  )
  normed = x.astype(jnp.float32) * inv_rms * scale.astype(jnp.float32)
  gated = normed * jax.nn.silu(gate_z.astype(jnp.float32))
  return gated.astype(orig_dtype)


def _rms_norm_gated_fwd(
    x: jax.Array, gate_z: jax.Array, scale: jax.Array, eps: float
):
  orig_dtype = x.dtype
  x_var = jax.lax.optimization_barrier(x).astype(jnp.float32)
  var_eps = jnp.mean(jnp.square(x_var), axis=-1, keepdims=True) + eps
  inv_rms = jax.lax.rsqrt(var_eps)
  normed = x.astype(jnp.float32) * inv_rms * scale.astype(jnp.float32)
  gated = (normed * jax.nn.silu(gate_z.astype(jnp.float32))).astype(orig_dtype)
  return gated, (x, gate_z, scale, inv_rms, var_eps)


def _rms_norm_gated_bwd(eps: float, res, g_out: jax.Array):
  del eps
  x, gate_z, scale, inv_rms, var_eps = res
  w_f32 = scale.astype(jnp.float32)

  # Stage 1+2: reductions from BF16 inputs to tiny `d_sq` (`[B,S,H,1]`) and
  # `g_scale` (`scale.shape`), avoiding any materialized `f32[B,S,H,D]` buffer
  # while matching JAX's exact rsqrt/square VJP arithmetic order (0-ULP parity).
  x_f32_r = x.astype(jnp.float32)
  x_hat_r = x_f32_r * inv_rms
  g_normed_r = g_out.astype(jnp.float32) * jax.nn.silu(
      gate_z.astype(jnp.float32)
  )
  d_inv_rms = jnp.sum((g_normed_r * w_f32) * x_f32_r, axis=-1, keepdims=True)
  d_var = d_inv_rms * (jnp.float32(-0.5) * (inv_rms / var_eps))
  d_sq = d_var / jnp.float32(x.shape[-1])
  scale_ndim = scale.ndim
  lead_axes = tuple(range(x.ndim - scale_ndim))
  bcast_axes = tuple(
      x.ndim - scale_ndim + i
      for i, s_dim in enumerate(scale.shape)
      if s_dim == 1 and x.shape[x.ndim - scale_ndim + i] != 1
  )
  reduce_axes = lead_axes + bcast_axes
  g_scale = (
      jnp.sum(g_normed_r * x_hat_r, axis=reduce_axes)
      .reshape(scale.shape)
      .astype(scale.dtype)
  )

  # Stage 3: barrier-separated elementwise backward directly into BF16 `(g_x, g_z)`.
  x_b, z_b, g_b = jax.lax.optimization_barrier((x, gate_z, g_out))
  x_f32 = x_b.astype(jnp.float32)
  z_f32 = z_b.astype(jnp.float32)
  sig_z = jax.nn.sigmoid(z_f32)
  silu_z = z_f32 * sig_z
  g_f32 = g_b.astype(jnp.float32)
  g_x_hat = (g_f32 * silu_z) * w_f32
  g_x = (g_x_hat * inv_rms + d_sq * (jnp.float32(2.0) * x_f32)).astype(x.dtype)
  g_silu = g_f32 * ((x_f32 * inv_rms) * w_f32)
  g_z = (
      g_silu * sig_z + (g_silu * z_f32) * (sig_z * (jnp.float32(1.0) - sig_z))
  ).astype(gate_z.dtype)
  return g_x, g_z, g_scale


rms_norm_gated.defvjp(_rms_norm_gated_fwd, _rms_norm_gated_bwd)
