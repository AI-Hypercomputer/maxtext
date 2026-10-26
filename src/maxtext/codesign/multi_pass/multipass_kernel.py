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

"""Dtype-agnostic Multi-Pass Tensor Slicing and GEMM Emulation Kernels.

Implements:
1. `quantize_to_dtype_spec`: Quantizes any JAX or NumPy array to any `DtypeSpec`
   (standard `bf16`, `fp16`, `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1`, `int8`, `int4`,
   or arbitrary parametric `eXmY` / `intB` / `uintB`) with configurable rounding
   (`rne`, `stochastic`, `truncate`) and subnormal handling.
2. `decompose_tensor_into_passes`: Sequentially decomposes a source tensor of
   dtype `A` (`src_dtype`) into `K` residual passes of arbitrary target dtypes
   `[B, C, D, ...]` (`pass_dtypes`) with configurable scaling (`per_tensor_pow2`,
   `per_tensor_amax`, `per_channel_pow2`, `per_channel_amax`, `per_block_pow2`,
   `per_block_amax`, `bit_shift`, `none`).
3. `reconstruct_multipass_tensor`: Reconstructs the effective multi-pass tensor
   `sum_{k=0}^{K-1} p_k` (with optional Straight-Through Estimator in backward).
4. `multipass_dot_general`: Executes multi-pass `dot_general` over configurable
   cross-term pass pairs (`triangular` / `X3`, `full` / `X4`, `asymmetric_lhs` /
   `X2`, `asymmetric_rhs` / `X2`, `karatsuba` / `X3`, or `custom` `(i,j)` pairs),
   accumulated in `acc_dtype`, with a custom VJP supporting independent `fwd`,
   `dlhs` (`dgrad`), and `drhs` (`wgrad`) multi-pass specifications.
5. `multipass_einsum` & `multipass_gmm_operands`: Multi-pass wrappers for
   2-operand `einsum` and MoE grouped/ragged matmuls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None  # type: ignore

try:
  import jax
  import jax.numpy as jnp

  HAS_JAX = True
except ImportError:  # pragma: no cover
  jax = None  # type: ignore
  jnp = np  # type: ignore
  HAS_JAX = False

from maxtext.codesign.multi_pass.multipass_config import (
    DtypeSpec,
    MultipassRuleConfig,
    OperandMultipassSpec,
    get_registered_pass_slicer,
    resolve_cross_term_pairs,
)


def _is_numpy_array(x: Any) -> bool:
  return np is not None and isinstance(x, np.ndarray)


def _xp_for(x: Any) -> Any:
  if _is_numpy_array(x) or not HAS_JAX:
    return np
  return jnp


def resolve_axis(
    ndim: int,
    axis_spec: str | int = "contracting",
    contracting_axes: Sequence[int] = (0,),
    batch_axes: Sequence[int] = (),
) -> int:
  """Resolves a symbolic or integer axis specification into a valid axis in `[0, ndim)`."""
  if ndim <= 0:
    return 0
  if isinstance(axis_spec, int):
    ax = axis_spec if axis_spec >= 0 else ndim + axis_spec
    return max(0, min(ndim - 1, ax))

  s = str(axis_spec).strip().lower()
  if s in ("contracting", "reduction", "k"):
    if contracting_axes:
      c0 = int(contracting_axes[0])
      ax = c0 if c0 >= 0 else ndim + c0
      return max(0, min(ndim - 1, ax))
    return max(0, ndim - 1)
  if s in ("non_contracting", "free", "channel", "out"):
    c_set = {c if c >= 0 else ndim + c for c in contracting_axes}
    b_set = {b if b >= 0 else ndim + b for b in batch_axes}
    free_axes = [i for i in range(ndim) if i not in c_set and i not in b_set]
    if free_axes:
      return free_axes[-1]
    return max(0, ndim - 1)
  if s in ("last", "-1"):
    return ndim - 1
  if s in ("first", "0"):
    return 0
  try:
    val = int(s)
    ax = val if val >= 0 else ndim + val
    return max(0, min(ndim - 1, ax))
  except ValueError:
    return max(0, ndim - 1)


def _apply_rounding(
    scaled_abs: Any,
    rounding_mode: str = "rne",
    seed: int = 0,
    xp: Any = None,
) -> Any:
  """Applies rounding (`rne`, `truncate`/`rtz`, or `stochastic`) to non-negative `scaled_abs`."""
  if xp is None:
    xp = _xp_for(scaled_abs)
  mode = rounding_mode.lower()
  if mode in ("truncate", "rtz"):
    return xp.floor(scaled_abs)
  if mode == "stochastic":
    if xp is np:
      rng = np.random.default_rng(seed)
      noise = rng.random(size=scaled_abs.shape, dtype=np.float32)
    else:
      key = jax.random.PRNGKey(int(seed))
      noise = jax.random.uniform(key, shape=scaled_abs.shape, dtype=jnp.float32)
    return xp.floor(scaled_abs + noise)
  # Default: round-to-nearest-even (RNE)
  return xp.round(scaled_abs)


def quantize_to_dtype_spec(
    x: Any,
    dtype_spec: DtypeSpec | str,
    *,
    rounding_mode: str = "rne",
    flush_subnormals: bool = False,
    subnormals: bool | None = None,
    seed: int = 0,
) -> Any:
  """Quantizes tensor `x` into the numerical grid of `dtype_spec` (returned as `float32`).

  Supports:
    - Any floating-point `DtypeSpec` (`fp32`, `bf16`, `fp16`, `fp8_e4m3`,
      `fp8_e5m2`, `fp6_e3m2`, `fp4_e2m1`, or arbitrary `eXmY`).
    - Any signed or unsigned integer `DtypeSpec` (`int8`, `int4`, `uint4`,
      `int2`, `int16`, `int32`, or arbitrary `intB` / `uintB`).
    - `"auto"` (returns `x` cast to `float32` without quantization).
  """
  if subnormals is not None:
    flush_subnormals = not bool(subnormals)
  spec = DtypeSpec.parse(dtype_spec) if not isinstance(dtype_spec, DtypeSpec) else dtype_spec
  xp = _xp_for(x)
  x_f32 = x.astype(xp.float32)

  if spec.kind == "auto":
    return x_f32

  if spec.kind == "float" and spec.exp_bits == 8 and spec.mantissa_bits >= 23:
    if rounding_mode == "rne" and not flush_subnormals:
      return x_f32

  # Fast path for native JAX BF16 RNE without subnormal flushing
  if (
      xp is not np
      and spec.kind == "float"
      and spec.exp_bits == 8
      and spec.mantissa_bits == 7
      and rounding_mode == "rne"
      and not flush_subnormals
  ):
    return x_f32.astype(jnp.bfloat16).astype(jnp.float32)

  if spec.kind in ("int", "uint"):
    sign = xp.where(x_f32 < 0, -1.0, 1.0)
    abs_x = xp.abs(x_f32)
    rounded_mag = _apply_rounding(abs_x, rounding_mode=rounding_mode, seed=seed, xp=xp)
    signed_val = sign * rounded_mag
    if spec.signed:
      if spec.int_bits == 1:
        min_v, max_v = -1.0, 0.0
      elif spec.int_bits >= 31:
        min_v, max_v = -float(1 << 30), float((1 << 30) - 1)
      else:
        min_v = -float(1 << (spec.int_bits - 1))
        max_v = float((1 << (spec.int_bits - 1)) - 1)
    else:
      min_v = 0.0
      max_v = float((1 << min(spec.int_bits, 30)) - 1)
    return xp.clip(signed_val, min_v, max_v)

  # Parametric floating-point eXmY quantization
  sign = xp.sign(x_f32)
  abs_x = xp.abs(x_f32)
  is_zero = abs_x == 0.0

  # Clamp away from 0 before log2
  safe_abs = xp.where(is_zero, xp.ones_like(abs_x), abs_x)
  raw_exp = xp.floor(xp.log2(safe_abs))

  min_norm_exp = float(spec.min_normal_exponent)
  eff_exp = xp.maximum(raw_exp, min_norm_exp)
  # Keep eff_exp within normal float32 exponent range [-126, 127] so exp2(eff_exp) never underflows
  clamped_exp = xp.clip(eff_exp, -126.0, 127.0)
  exp_scale = xp.exp2(clamped_exp)
  man_scale = float(2.0**spec.mantissa_bits)

  # Normalize into [1, 2) * 2^M for normals or [0, 1) * 2^M for subnormals
  scaled_sig = (abs_x / exp_scale) * man_scale
  rounded_sig = _apply_rounding(scaled_sig, rounding_mode=rounding_mode, seed=seed, xp=xp)
  dequant_abs = (rounded_sig / man_scale) * exp_scale

  if flush_subnormals:
    min_normal_val = float(2.0**min_norm_exp)
    dequant_abs = xp.where(dequant_abs < min_normal_val, 0.0, dequant_abs)

  dequant_abs = xp.clip(dequant_abs, 0.0, float(spec.max_val))
  result = xp.where(is_zero, 0.0, sign * dequant_abs)
  # Preserve NaNs if present in input
  result = xp.where(xp.isnan(x_f32), x_f32, result)
  return result


@dataclass(frozen=True)
class MultipassSlice:
  r"""Represents a single pass $k$ from decomposing a tensor $x$.

  Attributes:
    pass_idx: 0-based pass index $k \in [0, K)$.
    dtype_spec: The target `DtypeSpec` $\tau_k$ for this pass.
    quantized: Quantized value $q_k \in \tau_k$ (in `float32`).
    scale: Scale factor $s_k$ (broadcastable to `quantized.shape`, in `float32`).
    dequantized: Reconstructed slice value $p_k = q_k \cdot s_k$ (in `float32`).
  """

  pass_idx: int
  dtype_spec: DtypeSpec
  quantized: Any
  scale: Any
  dequantized: Any


def _compute_pass_scale(
    residual: Any,
    target_dtype: DtypeSpec,
    scale_mode: str,
    *,
    block_size: int = 128,
    contracting_axes: Sequence[int] = (0,),
    batch_axes: Sequence[int] = (),
) -> Any:
  """Computes the scaling factor $s_k$ for a residual tensor prior to quantization."""
  xp = _xp_for(residual)
  mode = scale_mode.lower()
  if mode == "none":
    return xp.ones((), dtype=xp.float32)

  q_max = float(max(target_dtype.max_val, 1e-12))
  abs_r = xp.abs(residual)
  is_pow2 = mode.endswith("_pow2")

  def _amax_to_scale(amax_val: Any) -> Any:
    raw_scale = amax_val / q_max
    nonzero = raw_scale > 0.0
    safe_scale = xp.where(nonzero, raw_scale, xp.ones_like(raw_scale))
    if is_pow2:
      # Use ceil(log2(amax / q_max)) so |r| / scale <= q_max (zero clipping)
      exp_val = xp.clip(xp.ceil(xp.log2(safe_scale)), -126.0, 127.0)
      s = xp.exp2(exp_val)
    else:
      s = safe_scale
    return xp.where(nonzero, s, xp.ones_like(s))

  if mode in ("per_tensor_amax", "per_tensor_pow2"):
    amax = xp.max(abs_r)
    return _amax_to_scale(amax)

  if mode in ("per_channel_amax", "per_channel_pow2"):
    if residual.ndim == 0:
      return _amax_to_scale(abs_r)
    red_axes = tuple(
        sorted(
            {
                resolve_axis(residual.ndim, c, contracting_axes, batch_axes)
                for c in (contracting_axes or (residual.ndim - 1,))
            }
        )
    )
    amax = xp.max(abs_r, axis=red_axes, keepdims=True)
    return _amax_to_scale(amax)

  if mode in ("per_block_amax", "per_block_pow2"):
    if residual.ndim == 0:
      return _amax_to_scale(abs_r)
    ax = resolve_axis(residual.ndim, "contracting", contracting_axes, batch_axes)
    # Move contracting axis to the last dimension for block reduction
    if ax != residual.ndim - 1:
      perm = [i for i in range(residual.ndim) if i != ax] + [ax]
      inv_perm = [0] * residual.ndim
      for i, p in enumerate(perm):
        inv_perm[p] = i
      work = xp.transpose(abs_r, perm)
    else:
      perm = None
      inv_perm = None
      work = abs_r

    k_len = work.shape[-1]
    blk = max(1, int(block_size))
    pad_len = (blk - (k_len % blk)) % blk
    if pad_len > 0:
      pad_widths = [(0, 0)] * (work.ndim - 1) + [(0, pad_len)]
      work_padded = xp.pad(work, pad_widths, mode="constant", constant_values=0.0)
    else:
      work_padded = work

    num_blocks = work_padded.shape[-1] // blk
    blocked_shape = work_padded.shape[:-1] + (num_blocks, blk)
    blocked = work_padded.reshape(blocked_shape)
    block_amax = xp.max(blocked, axis=-1, keepdims=True)
    block_scale = _amax_to_scale(block_amax)
    expanded_scale = xp.broadcast_to(block_scale, blocked_shape).reshape(work_padded.shape)
    if pad_len > 0:
      expanded_scale = expanded_scale[..., :k_len]
    if inv_perm is not None:
      expanded_scale = xp.transpose(expanded_scale, inv_perm)
    return expanded_scale

  return xp.ones((), dtype=xp.float32)


def decompose_tensor_into_passes(
    x: Any,
    spec: OperandMultipassSpec,
    *,
    contracting_axes: Sequence[int] = (0,),
    batch_axes: Sequence[int] = (),
) -> list[MultipassSlice]:
  """Decomposes tensor `x` of `spec.src_dtype` into `spec.num_passes` `MultipassSlice`s.

  Algorithm:
    1. Cast/quantize input `x` to `spec.src_dtype`:
       `r_0 = quantize_to_dtype_spec(x, spec.src_dtype)`
    2. For each pass `k = 0 .. K-1` with target dtype `tau_k = spec.pass_dtypes[k]`:
       - Compute scale `s_k` according to `spec.scale_mode`.
       - Quantize `q_k = quantize_to_dtype_spec(r_k / s_k, tau_k)`.
       - Reconstruct pass `p_k = q_k * s_k`.
       - Update residual `r_{k+1} = r_k - p_k`.
  """
  xp = _xp_for(x)
  x_f32 = x.astype(xp.float32)

  if not spec.enabled or spec.num_passes <= 0:
    one = xp.ones((), dtype=xp.float32)
    return [
        MultipassSlice(
            pass_idx=0,
            dtype_spec=DtypeSpec.parse("auto"),
            quantized=x_f32,
            scale=one,
            dequantized=x_f32,
        )
    ]

  if spec.custom_slicer:
    custom_fn = get_registered_pass_slicer(spec.custom_slicer)
    if custom_fn is None:
      raise ValueError(f"Unknown custom_slicer={spec.custom_slicer!r}")
    raw_slices = custom_fn(
        x_f32,
        spec,
        contracting_axes=contracting_axes,
        batch_axes=batch_axes,
    )
    out_slices: list[MultipassSlice] = []
    for idx, item in enumerate(raw_slices):
      if isinstance(item, MultipassSlice):
        out_slices.append(item)
      elif isinstance(item, tuple) and len(item) == 2:
        q_i, s_i = item
        dt_i = spec.pass_dtypes[idx] if idx < len(spec.pass_dtypes) else spec.pass_dtypes[-1]
        out_slices.append(
            MultipassSlice(
                pass_idx=idx,
                dtype_spec=dt_i,
                quantized=q_i,
                scale=s_i,
                dequantized=q_i * s_i,
            )
        )
      else:
        one = xp.ones((), dtype=xp.float32)
        dt_i = spec.pass_dtypes[idx] if idx < len(spec.pass_dtypes) else spec.pass_dtypes[-1]
        out_slices.append(
            MultipassSlice(
                pass_idx=idx,
                dtype_spec=dt_i,
                quantized=item,
                scale=one,
                dequantized=item,
            )
        )
    return out_slices

  # Step 1: Cast/quantize to source dtype A (with optional src_scale_mode)
  src_scale_mode = getattr(spec, "src_scale_mode", "none") or "none"
  if spec.src_dtype.kind == "auto":
    src_scale = xp.ones((), dtype=xp.float32)
    r_k = x_f32
  else:
    if src_scale_mode not in ("none", "unit", "bit_shift"):
      src_scale = _compute_pass_scale(
          x_f32,
          spec.src_dtype,
          src_scale_mode,
          block_size=spec.block_size,
          contracting_axes=contracting_axes,
          batch_axes=batch_axes,
      )
    else:
      src_scale = xp.ones((), dtype=xp.float32)
    r_k = quantize_to_dtype_spec(
        x_f32 / src_scale,
        spec.src_dtype,
        rounding_mode=spec.rounding_mode,
        flush_subnormals=spec.flush_subnormals,
        seed=spec.seed,
    )

  # Step 2: Bit-shift decomposition for integer multi-pass (e.g. INT8 -> [INT4, UINT4])
  if spec.scale_mode == "bit_shift":
    slices: list[MultipassSlice] = []
    k_total = spec.num_passes
    # Calculate bit-shifts from pass bitwidths
    bit_widths = [(d.int_bits if d.kind in ("int", "uint") else max(1, d.mantissa_bits)) for d in spec.pass_dtypes]
    rem = r_k
    for idx in range(k_total):
      dt_k = spec.pass_dtypes[idx]
      # Remaining bits below pass idx
      lower_bits = sum(
          (
              spec.pass_dtypes[j].int_bits
              if spec.pass_dtypes[j].kind in ("int", "uint")
              else max(1, spec.pass_dtypes[j].mantissa_bits)
          )
          for j in range(idx + 1, k_total)
      )
      # If the next pass is signed intB, use shift 2^(B-1) with RNE so residual fits in signed intB;
      # if the next pass is unsigned uintB, use floor division by 2^B so residual is in [0, 2^B - 1].
      if idx < k_total - 1:
        next_dt = spec.pass_dtypes[idx + 1]
        if next_dt.kind == "uint":
          shift_val = float(2.0**lower_bits)
          q_k = xp.floor(rem / shift_val)
          q_k = quantize_to_dtype_spec(q_k, dt_k, rounding_mode="truncate")
        else:
          eff_shift_bits = sum(
              max(1, bit_widths[j] - (1 if spec.pass_dtypes[j].signed else 0)) for j in range(idx + 1, k_total)
          )
          shift_val = float(2.0**eff_shift_bits)
          q_k = quantize_to_dtype_spec(rem / shift_val, dt_k, rounding_mode="rne")
      else:
        shift_val = 1.0
        q_k = quantize_to_dtype_spec(rem, dt_k, rounding_mode=spec.rounding_mode)

      shift_arr = xp.asarray(shift_val, dtype=xp.float32)
      rem = rem - q_k * shift_arr
      s_k = src_scale * shift_arr
      p_k = q_k * s_k
      slices.append(
          MultipassSlice(
              pass_idx=idx,
              dtype_spec=dt_k,
              quantized=q_k,
              scale=s_k,
              dequantized=p_k,
          )
      )
    return slices

  # Step 3: General floating-point / integer residual multi-pass loop
  r_k = r_k * src_scale
  slices = []
  for idx in range(spec.num_passes):
    dt_k = spec.pass_dtypes[idx]
    s_k = _compute_pass_scale(
        r_k,
        dt_k,
        spec.scale_mode,
        block_size=spec.block_size,
        contracting_axes=contracting_axes,
        batch_axes=batch_axes,
    )
    q_k = quantize_to_dtype_spec(
        r_k / s_k,
        dt_k,
        rounding_mode=spec.rounding_mode,
        flush_subnormals=spec.flush_subnormals,
        seed=spec.seed + idx,
    )
    p_k = q_k * s_k
    r_k = r_k - p_k
    slices.append(
        MultipassSlice(
            pass_idx=idx,
            dtype_spec=dt_k,
            quantized=q_k,
            scale=s_k,
            dequantized=p_k,
        )
    )
  return slices


def reconstruct_multipass_tensor(
    x: Any,
    spec: OperandMultipassSpec,
    *,
    contracting_axes: Sequence[int] = (0,),
    batch_axes: Sequence[int] = (),
    use_ste: bool = True,
    active_cond: Any = None,
) -> Any:
  """Decomposes `x` into `spec.num_passes` slices and returns their sum `sum_k p_k`."""
  if not spec.enabled:
    return x
  xp = _xp_for(x)

  def _compute_reconstructed(operand: Any) -> Any:
    slices = decompose_tensor_into_passes(
        operand,
        spec,
        contracting_axes=contracting_axes,
        batch_axes=batch_axes,
    )
    total = xp.zeros_like(operand, dtype=xp.float32)
    for sl in slices:
      total = total + sl.dequantized
    out = total.astype(operand.dtype)
    if use_ste and xp is not np:
      return operand + jax.lax.stop_gradient(out - operand)
    return out

  if active_cond is not None and xp is not np:
    return jax.lax.cond(
        jnp.asarray(active_cond, dtype=jnp.bool_),
        _compute_reconstructed,
        lambda op: op,
        x,
    )
  return _compute_reconstructed(x)


def reconstruct_multipass_eager(
    x: Any,
    spec: OperandMultipassSpec,
    *,
    contracting_axes: Sequence[int] = (0,),
    batch_axes: Sequence[int] = (),
) -> Any:
  """Eagerly reconstructs `sum_k p_k` without STE (for checkpoint materialization/inspection)."""
  return reconstruct_multipass_tensor(
      x,
      spec,
      contracting_axes=contracting_axes,
      batch_axes=batch_axes,
      use_ste=False,
      active_cond=None,
  )


def _execute_multipass_bilinear(
    lhs: Any,
    rhs: Any,
    bilinear_fn: Callable[[Any, Any], Any],
    *,
    lhs_spec: OperandMultipassSpec | None,
    rhs_spec: OperandMultipassSpec | None,
    lhs_contracting_axes: Sequence[int],
    rhs_contracting_axes: Sequence[int],
    lhs_batch_axes: Sequence[int],
    rhs_batch_axes: Sequence[int],
    cross_term_mode: str = "triangular",
    max_pass_degree: int | None = None,
    active_passes: tuple[tuple[int, int], ...] | None = None,
    acc_dtype: DtypeSpec | None = None,
    out_dtype: Any = None,
) -> Any:
  """Evaluates a bilinear contraction (`dot_general` or VJP transpose) across multi-pass slices."""
  xp = _xp_for(lhs)
  acc_spec = acc_dtype if acc_dtype is not None else DtypeSpec.parse("fp32")

  lhs_active = lhs_spec is not None and lhs_spec.enabled
  rhs_active = rhs_spec is not None and rhs_spec.enabled

  if not lhs_active and not rhs_active:
    res = bilinear_fn(lhs, rhs)
    if out_dtype is not None:
      res = res.astype(out_dtype)
    return res

  mode = cross_term_mode.strip().lower()

  # Determine effective number of passes per operand for asymmetric modes
  eff_lhs_spec = lhs_spec
  eff_rhs_spec = rhs_spec
  if mode == "asymmetric_lhs" and rhs_active and eff_rhs_spec is not None:
    if eff_rhs_spec.num_passes > 1:
      eff_rhs_spec = OperandMultipassSpec(
          enabled=True,
          src_dtype=eff_rhs_spec.src_dtype,
          src_scale_mode=eff_rhs_spec.src_scale_mode,
          num_passes=1,
          pass_dtypes=(eff_rhs_spec.pass_dtypes[0],),
          scale_mode=eff_rhs_spec.scale_mode,
          block_size=eff_rhs_spec.block_size,
          rounding_mode=eff_rhs_spec.rounding_mode,
          flush_subnormals=eff_rhs_spec.flush_subnormals,
          custom_slicer=eff_rhs_spec.custom_slicer,
          seed=eff_rhs_spec.seed,
      )
  elif mode == "asymmetric_rhs" and lhs_active and eff_lhs_spec is not None:
    if eff_lhs_spec.num_passes > 1:
      eff_lhs_spec = OperandMultipassSpec(
          enabled=True,
          src_dtype=eff_lhs_spec.src_dtype,
          src_scale_mode=eff_lhs_spec.src_scale_mode,
          num_passes=1,
          pass_dtypes=(eff_lhs_spec.pass_dtypes[0],),
          scale_mode=eff_lhs_spec.scale_mode,
          block_size=eff_lhs_spec.block_size,
          rounding_mode=eff_lhs_spec.rounding_mode,
          flush_subnormals=eff_lhs_spec.flush_subnormals,
          custom_slicer=eff_lhs_spec.custom_slicer,
          seed=eff_lhs_spec.seed,
      )

  lhs_slices = (
      decompose_tensor_into_passes(
          lhs,
          eff_lhs_spec,
          contracting_axes=lhs_contracting_axes,
          batch_axes=lhs_batch_axes,
      )
      if (lhs_active and eff_lhs_spec is not None)
      else [
          MultipassSlice(
              0,
              DtypeSpec.parse("auto"),
              lhs.astype(xp.float32),
              xp.ones((), dtype=xp.float32),
              lhs.astype(xp.float32),
          )
      ]
  )
  rhs_slices = (
      decompose_tensor_into_passes(
          rhs,
          eff_rhs_spec,
          contracting_axes=rhs_contracting_axes,
          batch_axes=rhs_batch_axes,
      )
      if (rhs_active and eff_rhs_spec is not None)
      else [
          MultipassSlice(
              0,
              DtypeSpec.parse("auto"),
              rhs.astype(xp.float32),
              xp.ones((), dtype=xp.float32),
              rhs.astype(xp.float32),
          )
      ]
  )

  k_a = len(lhs_slices)
  k_b = len(rhs_slices)

  def _cast_acc(val: Any) -> Any:
    if acc_spec.kind == "auto" or (acc_spec.kind == "float" and acc_spec.exp_bits == 8 and acc_spec.mantissa_bits >= 23):
      return val.astype(xp.float32)
    if acc_spec.kind in ("int", "uint"):
      # Slices are dequantized with their float scales (src_scale * shift_val),
      # so clamp to the integer accumulator dynamic range without rounding away fractional scales.
      return xp.clip(val.astype(xp.float32), acc_spec.min_val, acc_spec.max_val)
    return quantize_to_dtype_spec(val, acc_spec)

  # Karatsuba 3-pass evaluation for 2-pass operands
  if mode == "karatsuba" and k_a >= 2 and k_b >= 2:
    a0, a1 = lhs_slices[0], lhs_slices[1]
    b0, b1 = rhs_slices[0], rhs_slices[1]
    lhs_unscaled_int = (
        eff_lhs_spec is not None
        and eff_lhs_spec.scale_mode == "bit_shift"
        and eff_lhs_spec.src_scale_mode in ("none", "unit")
    )
    rhs_unscaled_int = (
        eff_rhs_spec is not None
        and eff_rhs_spec.scale_mode == "bit_shift"
        and eff_rhs_spec.src_scale_mode in ("none", "unit")
    )
    if lhs_unscaled_int and rhs_unscaled_int:
      p0 = _cast_acc(bilinear_fn(a0.quantized, b0.quantized))
      p1 = _cast_acc(bilinear_fn(a1.quantized, b1.quantized))
      p2_raw = _cast_acc(bilinear_fn(a0.quantized + a1.quantized, b0.quantized + b1.quantized))
      p2 = _cast_acc(p2_raw - p0 - p1)
      scale_hh = a0.scale * b0.scale
      scale_cross = a0.scale * b1.scale
      scale_ll = a1.scale * b1.scale
      acc = _cast_acc(scale_hh * p0 + scale_cross * p2 + scale_ll * p1)
    else:
      p0 = _cast_acc(bilinear_fn(a0.dequantized, b0.dequantized))
      p1 = _cast_acc(bilinear_fn(a1.dequantized, b1.dequantized))
      p2_raw = _cast_acc(bilinear_fn(a0.dequantized + a1.dequantized, b0.dequantized + b1.dequantized))
      p2 = _cast_acc(p2_raw - p0 - p1)
      acc = _cast_acc(p0 + p2 + p1)
    if out_dtype is not None:
      return acc.astype(out_dtype)
    return acc

  pairs = resolve_cross_term_pairs(
      k_a,
      k_b,
      cross_term_mode=mode,
      max_pass_degree=max_pass_degree,
      active_passes=active_passes,
  )

  acc = None
  for i, j in pairs:
    term = bilinear_fn(lhs_slices[i].dequantized, rhs_slices[j].dequantized)
    term_acc = _cast_acc(term)
    if acc is None:
      acc = term_acc
    else:
      acc = _cast_acc(acc + term_acc)

  if acc is None:
    acc = _cast_acc(bilinear_fn(lhs.astype(xp.float32), rhs.astype(xp.float32)))

  if out_dtype is not None:
    return acc.astype(out_dtype)
  return acc


def _numpy_dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: Any,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
  """Pure-NumPy implementation of `jax.lax.dot_general` for arbitrary contracting and batch axes."""
  del precision
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_c = [c if c >= 0 else lhs.ndim + c for c in lhs_contract]
  rhs_c = [c if c >= 0 else rhs.ndim + c for c in rhs_contract]
  lhs_b = [b if b >= 0 else lhs.ndim + b for b in lhs_batch]
  rhs_b = [b if b >= 0 else rhs.ndim + b for b in rhs_batch]

  lhs_free = [i for i in range(lhs.ndim) if i not in lhs_c and i not in lhs_b]
  rhs_free = [i for i in range(rhs.ndim) if i not in rhs_c and i not in rhs_b]

  # Build einsum subscripts
  alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  idx_pos = 0
  batch_letters = [alphabet[idx_pos + k] for k in range(len(lhs_b))]
  idx_pos += len(lhs_b)
  contract_letters = [alphabet[idx_pos + k] for k in range(len(lhs_c))]
  idx_pos += len(lhs_c)
  lhs_free_letters = [alphabet[idx_pos + k] for k in range(len(lhs_free))]
  idx_pos += len(lhs_free)
  rhs_free_letters = [alphabet[idx_pos + k] for k in range(len(rhs_free))]

  lhs_chars = [""] * lhs.ndim
  for ax, ch in zip(lhs_b, batch_letters):
    lhs_chars[ax] = ch
  for ax, ch in zip(lhs_c, contract_letters):
    lhs_chars[ax] = ch
  for ax, ch in zip(lhs_free, lhs_free_letters):
    lhs_chars[ax] = ch

  rhs_chars = [""] * rhs.ndim
  for ax, ch in zip(rhs_b, batch_letters):
    rhs_chars[ax] = ch
  for ax, ch in zip(rhs_c, contract_letters):
    rhs_chars[ax] = ch
  for ax, ch in zip(rhs_free, rhs_free_letters):
    rhs_chars[ax] = ch

  out_sub = "".join(batch_letters + lhs_free_letters + rhs_free_letters)
  subscripts = f"{''.join(lhs_chars)},{''.join(rhs_chars)}->{out_sub}"
  res = np.einsum(
      subscripts,
      np.asarray(lhs, dtype=np.float32),
      np.asarray(rhs, dtype=np.float32),
  )
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res.astype(np.result_type(lhs.dtype, rhs.dtype))


def multipass_dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: Any,
    rule: MultipassRuleConfig,
    *,
    base_dot_general_fn: Callable[..., Any] | None = None,
    precision: Any = None,
    preferred_element_type: Any = None,
    active_cond: Any = None,
) -> Any:
  """Executes `dot_general` with multi-pass decomposition and custom VJP backward support."""
  xp = _xp_for(lhs)
  if base_dot_general_fn is None:
    base_dot_general_fn = _numpy_dot_general if xp is np else jax.lax.dot_general

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_c = tuple(int(c) if int(c) >= 0 else lhs.ndim + int(c) for c in lhs_contract)
  rhs_c = tuple(int(c) if int(c) >= 0 else rhs.ndim + int(c) for c in rhs_contract)
  lhs_b = tuple(int(b) if int(b) >= 0 else lhs.ndim + int(b) for b in lhs_batch)
  rhs_b = tuple(int(b) if int(b) >= 0 else rhs.ndim + int(b) for b in rhs_batch)

  orig_out_dtype = preferred_element_type if preferred_element_type is not None else xp.result_type(lhs.dtype, rhs.dtype)

  def _raw_fp32_dot(a: Any, b: Any) -> Any:
    return base_dot_general_fn(
        a.astype(xp.float32),
        b.astype(xp.float32),
        dimension_numbers,
        precision=precision,
        preferred_element_type=xp.float32,
    )

  def _finalize_out(val: Any) -> Any:
    if rule.out_dtype.kind != "auto":
      val = quantize_to_dtype_spec(val, rule.out_dtype)
    return val.astype(orig_out_dtype)

  def _run_fwd_multipass(a: Any, b: Any) -> Any:
    if not rule.emulate_fwd:
      return base_dot_general_fn(
          a,
          b,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    out_f32 = _execute_multipass_bilinear(
        a,
        b,
        _raw_fp32_dot,
        lhs_spec=rule.lhs,
        rhs_spec=rule.rhs,
        lhs_contracting_axes=lhs_c,
        rhs_contracting_axes=rhs_c,
        lhs_batch_axes=lhs_b,
        rhs_batch_axes=rhs_b,
        cross_term_mode=rule.cross_term_mode,
        max_pass_degree=rule.max_pass_degree,
        active_passes=rule.active_passes,
        acc_dtype=rule.acc_dtype,
        out_dtype=None,
    )
    return _finalize_out(out_f32)

  def _execute_emulated(a_in: Any, b_in: Any) -> Any:
    if xp is np:
      return _run_fwd_multipass(a_in, b_in)

    # Use Straight-Through Estimator directly when backward multi-pass is not active
    if rule.bwd_mode == "ste" or (not rule.emulate_dlhs and not rule.emulate_drhs):
      base_out = base_dot_general_fn(
          a_in,
          b_in,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
      emul_out = _run_fwd_multipass(a_in, b_in)
      return base_out + jax.lax.stop_gradient(emul_out - base_out)

    num_batch = len(lhs_b)
    lhs_free = tuple(i for i in range(a_in.ndim) if i not in lhs_c and i not in lhs_b)
    rhs_free = tuple(i for i in range(b_in.ndim) if i not in rhs_c and i not in rhs_b)
    dy_batch_axes = tuple(range(num_batch))
    dy_lhs_free_axes = tuple(range(num_batch, num_batch + len(lhs_free)))
    dy_rhs_free_axes = tuple(
        range(
            num_batch + len(lhs_free),
            num_batch + len(lhs_free) + len(rhs_free),
        )
    )
    order_lhs = list(lhs_b) + list(lhs_free) + list(lhs_c)
    inv_perm_lhs = tuple(order_lhs.index(ax) for ax in range(a_in.ndim))
    order_rhs = list(rhs_b) + list(rhs_c) + list(rhs_free)
    inv_perm_rhs = tuple(order_rhs.index(ax) for ax in range(b_in.ndim))
    dlhs_dnums = ((dy_rhs_free_axes, rhs_free), (dy_batch_axes, rhs_b))
    drhs_dnums = ((lhs_free, dy_lhs_free_axes), (lhs_b, dy_batch_axes))

    def _dlhs_bilinear(dy_slice: Any, rhs_slice: Any) -> Any:
      raw_dlhs = base_dot_general_fn(
          dy_slice.astype(jnp.float32),
          rhs_slice.astype(jnp.float32),
          dlhs_dnums,
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      return jnp.transpose(raw_dlhs, inv_perm_lhs)

    def _drhs_bilinear(lhs_slice: Any, dy_slice: Any) -> Any:
      raw_drhs = base_dot_general_fn(
          lhs_slice.astype(jnp.float32),
          dy_slice.astype(jnp.float32),
          drhs_dnums,
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      return jnp.transpose(raw_drhs, inv_perm_rhs)

    @jax.custom_vjp
    def _mp_dot(a: Any, b: Any) -> Any:
      return _run_fwd_multipass(a, b)

    def _fwd(a: Any, b: Any):
      y = _run_fwd_multipass(a, b)
      return y, (a, b)

    def _bwd(residuals, dy):
      a, b = residuals
      dy_f32 = dy.astype(jnp.float32)

      if rule.bwd_mode == "multipass_bwd" and rule.emulate_dlhs:
        dlhs_ct = rule.dlhs_cross_term_mode or rule.cross_term_mode
        dlhs = _execute_multipass_bilinear(
            dy_f32,
            b,
            _dlhs_bilinear,
            lhs_spec=rule.dlhs_grad or rule.lhs,
            rhs_spec=rule.dlhs_rhs or rule.rhs,
            lhs_contracting_axes=dy_rhs_free_axes,
            rhs_contracting_axes=rhs_free,
            lhs_batch_axes=dy_batch_axes,
            rhs_batch_axes=rhs_b,
            cross_term_mode=dlhs_ct,
            max_pass_degree=rule.max_pass_degree,
            active_passes=rule.active_passes if dlhs_ct == "custom" else None,
            acc_dtype=rule.acc_dtype,
            out_dtype=a.dtype,
        )
      else:
        dlhs = _dlhs_bilinear(dy_f32, b.astype(jnp.float32)).astype(a.dtype)

      if rule.bwd_mode == "multipass_bwd" and rule.emulate_drhs:
        drhs_ct = rule.drhs_cross_term_mode or rule.cross_term_mode
        drhs = _execute_multipass_bilinear(
            a,
            dy_f32,
            _drhs_bilinear,
            lhs_spec=rule.drhs_lhs or rule.lhs,
            rhs_spec=rule.drhs_grad or rule.rhs,
            lhs_contracting_axes=lhs_free,
            rhs_contracting_axes=dy_lhs_free_axes,
            lhs_batch_axes=lhs_b,
            rhs_batch_axes=dy_batch_axes,
            cross_term_mode=drhs_ct,
            max_pass_degree=rule.max_pass_degree,
            active_passes=rule.active_passes if drhs_ct == "custom" else None,
            acc_dtype=rule.acc_dtype,
            out_dtype=b.dtype,
        )
      else:
        drhs = _drhs_bilinear(a.astype(jnp.float32), dy_f32).astype(b.dtype)

      return dlhs, drhs

    _mp_dot.defvjp(_fwd, _bwd)
    return _mp_dot(a_in, b_in)

  if active_cond is not None:
    if xp is np:
      return (
          _execute_emulated(lhs, rhs)
          if bool(active_cond)
          else base_dot_general_fn(
              lhs,
              rhs,
              dimension_numbers,
              precision=precision,
              preferred_element_type=preferred_element_type,
          )
      )
    return jax.lax.cond(
        jnp.asarray(active_cond, dtype=jnp.bool_),
        lambda ops: _execute_emulated(ops[0], ops[1]),
        lambda ops: base_dot_general_fn(
            ops[0],
            ops[1],
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        (lhs, rhs),
    )

  return _execute_emulated(lhs, rhs)


def _infer_einsum_axes(
    subscripts: str, lhs_ndim: int, rhs_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Parses a 2-operand einsum subscript string to find contracting and batch axes."""
  clean = subscripts.replace(" ", "")
  if "->" in clean:
    inputs_str, out_str = clean.split("->", 1)
  else:
    inputs_str = clean
    out_str = ""
  parts = inputs_str.split(",")
  if len(parts) != 2 or "..." in clean:
    return ((max(0, lhs_ndim - 1),), (0,), (), ())

  lhs_sub, rhs_sub = parts
  if not out_str:
    all_chars = lhs_sub + rhs_sub
    out_str = "".join(sorted(c for c in set(all_chars) if all_chars.count(c) == 1))

  lhs_chars = list(lhs_sub)
  rhs_chars = list(rhs_sub)
  out_set = set(out_str)

  shared = set(lhs_chars).intersection(rhs_chars)
  contract_chars = shared - out_set
  batch_chars = shared.intersection(out_set)

  lhs_contract = tuple(i for i, c in enumerate(lhs_chars) if c in contract_chars)
  rhs_contract = tuple(i for i, c in enumerate(rhs_chars) if c in contract_chars)
  lhs_batch = tuple(i for i, c in enumerate(lhs_chars) if c in batch_chars)
  rhs_batch = tuple(i for i, c in enumerate(rhs_chars) if c in batch_chars)
  return lhs_contract, rhs_contract, lhs_batch, rhs_batch


def multipass_einsum(
    subscripts: str,
    lhs: Any,
    rhs: Any,
    rule: MultipassRuleConfig,
    *,
    base_einsum_fn: Callable[..., Any] | None = None,
    active_cond: Any = None,
    **kwargs: Any,
) -> Any:
  """Executes a 2-operand `einsum` using multi-pass cross-term evaluation."""
  xp = _xp_for(lhs)
  if base_einsum_fn is None:
    base_einsum_fn = jnp.einsum if xp is not np else np.einsum

  lhs_c, rhs_c, lhs_b, rhs_b = _infer_einsum_axes(subscripts, lhs.ndim, rhs.ndim)
  orig_dtype = xp.result_type(lhs.dtype, rhs.dtype)

  def _raw_einsum_f32(a: Any, b: Any) -> Any:
    kw = dict(kwargs)
    if xp is not np:
      kw["preferred_element_type"] = jnp.float32
    return base_einsum_fn(subscripts, a.astype(xp.float32), b.astype(xp.float32), **kw)

  def _run_emulated(a: Any, b: Any) -> Any:
    if not rule.emulate_fwd:
      return base_einsum_fn(subscripts, a, b, **kwargs)
    res_f32 = _execute_multipass_bilinear(
        a,
        b,
        _raw_einsum_f32,
        lhs_spec=rule.lhs,
        rhs_spec=rule.rhs,
        lhs_contracting_axes=lhs_c,
        rhs_contracting_axes=rhs_c,
        lhs_batch_axes=lhs_b,
        rhs_batch_axes=rhs_b,
        cross_term_mode=rule.cross_term_mode,
        max_pass_degree=rule.max_pass_degree,
        active_passes=rule.active_passes,
        acc_dtype=rule.acc_dtype,
        out_dtype=None,
    )
    if rule.out_dtype.kind != "auto":
      res_f32 = quantize_to_dtype_spec(res_f32, rule.out_dtype)
    out = res_f32.astype(orig_dtype)
    if xp is not np:
      base_out = base_einsum_fn(subscripts, a, b, **kwargs)
      return base_out + jax.lax.stop_gradient(out - base_out)
    return out

  if active_cond is not None:
    if xp is np:
      return _run_emulated(lhs, rhs) if bool(active_cond) else base_einsum_fn(subscripts, lhs, rhs, **kwargs)
    return jax.lax.cond(
        jnp.asarray(active_cond, dtype=jnp.bool_),
        lambda ops: _run_emulated(ops[0], ops[1]),
        lambda ops: base_einsum_fn(subscripts, ops[0], ops[1], **kwargs),
        (lhs, rhs),
    )

  return _run_emulated(lhs, rhs)


def multipass_gmm_operands(
    lhs: Any,
    rhs: Any,
    rule: MultipassRuleConfig,
    *,
    lhs_contracting_axes: Sequence[int] = (1,),
    rhs_contracting_axes: Sequence[int] = (1,),
    lhs_batch_axes: Sequence[int] = (),
    rhs_batch_axes: Sequence[int] = (0,),
    active_cond: Any = None,
) -> tuple[Any, Any]:
  """Applies multi-pass reconstruction to MoE `gmm` / `ragged_dot` `(lhs, rhs)` operands."""
  lhs_out = lhs
  rhs_out = rhs
  if rule.lhs is not None and rule.lhs.enabled:
    lhs_out = reconstruct_multipass_tensor(
        lhs,
        rule.lhs,
        contracting_axes=lhs_contracting_axes,
        batch_axes=lhs_batch_axes,
        use_ste=True,
        active_cond=active_cond,
    )
  if rule.rhs is not None and rule.rhs.enabled:
    rhs_out = reconstruct_multipass_tensor(
        rhs,
        rule.rhs,
        contracting_axes=rhs_contracting_axes,
        batch_axes=rhs_batch_axes,
        use_ste=True,
        active_cond=active_cond,
    )
  return lhs_out, rhs_out


# Backward-compatible aliases for shared interception / checkpoint utilities
apply_mn_sparsity = reconstruct_multipass_tensor
apply_mn_sparsity_eager = reconstruct_multipass_eager
sparse_dot_general = multipass_dot_general
sparse_einsum = multipass_einsum
sparse_gmm_operands = multipass_gmm_operands
