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

"""Pure-JAX RMSNorm, split-half RoPE, FSDP gather, and CP permutation ops."""

from __future__ import annotations

import functools

import jax
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
from jax.sharding import PartitionSpec
import numpy as np

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_types

_SC_MIN_COLLECTIVE_BYTES: int = 2_000_000


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def flint_rms_norm(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
  """Applies RMSNorm with FP32 scale/variance and compact residuals."""
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  return (x_f32 * inv_rms * weight.astype(jnp.float32)).astype(x.dtype)


def _flint_rms_norm_fwd(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Forward rule for `flint_rms_norm` saving compact `(x, weight)`."""
  return flint_rms_norm(x, weight, eps), (x, weight)


def _flint_rms_norm_bwd(
    eps: float,
    res: tuple[jax.Array, jax.Array],
    dy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Backward rule for `flint_rms_norm` recomputing `inv_rms` in FP32."""
  x, weight = res
  x_f32 = x.astype(jnp.float32)
  w_f32 = weight.astype(jnp.float32)
  dy_f32 = dy.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  x_hat = x_f32 * inv_rms
  reduce_axes = tuple(range(x.ndim - 1))
  dweight = jnp.sum(dy_f32 * x_hat, axis=reduce_axes).astype(weight.dtype)
  wdy = dy_f32 * w_f32
  dx = ((wdy - x_hat * jnp.mean(wdy * x_hat, axis=-1, keepdims=True)) * inv_rms).astype(x.dtype)
  return dx, dweight


flint_rms_norm.defvjp(_flint_rms_norm_fwd, _flint_rms_norm_bwd)


def compute_split_half_rope_cos_sin(
    positions: jax.Array,
    head_dim: int,
    rope_theta: float = 10000.0,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:
  """Computes split-half (NeoX-style) `(cos, sin)` tables `[..., 1, head_dim]`."""
  if head_dim <= 0 or head_dim % 2 != 0:
    raise ValueError(f"head_dim must be positive and even, got {head_dim}.")
  exponents = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)
  inv_freq = 1.0 / (jnp.asarray(rope_theta, dtype=jnp.float32) ** exponents)
  freqs = positions.astype(jnp.float32)[..., None] * inv_freq
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  out_dtype = jnp.dtype(dtype)
  cos = jnp.cos(emb).astype(out_dtype)[..., None, :]
  sin = jnp.sin(emb).astype(out_dtype)[..., None, :]
  return cos, sin


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates the split-half last axis `concat([-x2, x1], axis=-1)`."""
  if x.shape[-1] % 2 != 0:
    raise ValueError(f"Expected even last dimension, got {x.shape}.")
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate([-x2, x1], axis=-1)


def apply_precomputed_split_half_rope(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Applies split-half RoPE to `q, k` using precomputed `(cos, sin)` tables."""
  q_cos = cos.astype(q.dtype)
  q_sin = sin.astype(q.dtype)
  q_out = q * q_cos + rotate_half(q) * q_sin
  k_cos = cos.astype(k.dtype)
  k_sin = sin.astype(k.dtype)
  k_out = k * k_cos + rotate_half(k) * k_sin
  return q_out, k_out


def apply_split_half_rope(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_theta: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
  """Applies split-half RoPE to `q, k` (`[B, S, H, head_dim]`) in `q.dtype`."""
  cos, sin = compute_split_half_rope_cos_sin(
      positions,
      head_dim,
      rope_theta=rope_theta,
      dtype=q.dtype,
  )
  return apply_precomputed_split_half_rope(q, k, cos, sin)


@jax.custom_vjp
def bijective_permute(
    x: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
) -> jax.Array:
  """Permutes `x` on axis 0 via `perm` with gather-based VJP `dy[inv_perm]`."""
  del inv_perm
  return x[perm]


def _bijective_permute_fwd(
    x: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:
  return x[perm], (inv_perm,)


def _bijective_permute_bwd(
    residuals: tuple[jax.Array],
    dy: jax.Array,
) -> tuple[jax.Array, None, None]:
  (inv_perm,) = residuals
  return dy[inv_perm], None, None


bijective_permute.defvjp(_bijective_permute_fwd, _bijective_permute_bwd)


def sc_all_gather(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    core_id: int | None,
    axis: int = 0,
    min_sc_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """All-gathers `x` pinned to SparseCore `core_id` (`0` or `1`) when large."""
  if core_id not in (None, 0, 1):
    raise ValueError(f"core_id must be None, 0, or 1, got {core_id!r}.")
  axis_size = (
      jax.lax.axis_size(axis_name)
      if isinstance(axis_name, str)
      else functools.reduce(lambda a, b: a * b, (jax.lax.axis_size(n) for n in axis_name), 1)
  )
  gathered_bytes = x.size * x.dtype.itemsize * axis_size
  if core_id is None or gathered_bytes < min_sc_bytes:
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
    min_sc_bytes: int = _SC_MIN_COLLECTIVE_BYTES,
) -> jax.Array:
  """Reduce-scatters `x` pinned to SparseCore `core_id` (`0` or `1`)."""
  if core_id not in (None, 0, 1):
    raise ValueError(f"core_id must be None, 0, or 1, got {core_id!r}.")
  input_bytes = x.size * x.dtype.itemsize
  if core_id is None or input_bytes < min_sc_bytes:
    return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=True)

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


def routed_w_to_tiles(w: jax.Array) -> jax.Array:
  """Reshapes `[..., E_loc, rows, cols]` to `[..., E_loc, rows // 8, 8, cols]`."""
  if w.ndim < 3 or w.shape[-2] % 8 != 0:
    raise ValueError("routed_w_to_tiles requires ndim >= 3 and shape[-2] % 8 == 0, got" f" {w.shape}.")
  *prefix, rows, cols = w.shape
  return w.reshape(*prefix, rows // 8, 8, cols)


def routed_w_from_tiles(w: jax.Array) -> jax.Array:
  """Restores `[..., E_loc, r_tiles, 8, cols]` -> `[..., E_loc, rows, cols]`."""
  if w.ndim < 4 or w.shape[-2] != 8:
    raise ValueError("routed_w_from_tiles requires ndim >= 4 and shape[-2] == 8, got" f" {w.shape}.")
  *prefix, r_tiles, sub, cols = w.shape
  return w.reshape(*prefix, r_tiles * sub, cols)


def gather_dense_weight(
    w_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    dtype: jax.typing.DTypeLike | None = None,
    bwd_use_sc: bool = True,
) -> jax.Array:
  """Casts and all-gathers a dense parameter shard across `('fsdp','cp','ep')`.

  Pass `dtype=jnp.float32` for `embed` (keeping embedding scatter-add updates in
  FP32) and `router.gate`. When `sc_collective_mode == 'dual_sc'`, uses a
  zero-tensor-residual custom VJP pinning forward `all_gather` to `SC0` and
  backward `psum_scatter` to `SC1` (unless `bwd_use_sc=False`, which keeps
  backward `psum_scatter` on TensorCore ICI for odd-vocab `embed`/`lm_head`).

  Args:
    w_shard: Local sharded weight slice.
    cfg: Validated `FlintConfig`.
    sharded_axis: Axis along which `w_shard` is sharded across all mesh axes.
    dtype: Optional explicit cast dtype (defaults to `cfg.train.compute_dtype`).
    bwd_use_sc: When `False`, executes backward `psum_scatter` on TensorCore ICI
      even when `sc_collective_mode == 'dual_sc'`.

  Returns:
    All-gathered full weight array cast to `dtype` (or `compute_jnp_dtype`).
  """
  target_dtype = jnp.dtype(dtype) if dtype is not None else cfg.train.compute_jnp_dtype
  if cfg.sharding.num_devices == 1:
    return w_shard.astype(target_dtype)
  if cfg.kernels.sc_collective_mode == "off":
    return jax.lax.all_gather(
        w_shard.astype(target_dtype),
        ("fsdp", "cp", "ep"),
        axis=sharded_axis,
        tiled=True,
    )

  orig_shard_dtype = w_shard.dtype
  min_sc_bytes = cfg.kernels.sc_min_collective_bytes

  @jax.custom_vjp
  def _gather_dense_dual_sc(w_s: jax.Array) -> jax.Array:
    return sc_all_gather(
        w_s.astype(target_dtype),
        ("fsdp", "cp", "ep"),
        core_id=0,
        axis=sharded_axis,
        min_sc_bytes=min_sc_bytes,
    )

  def _fwd(w_s: jax.Array) -> tuple[jax.Array, tuple[()]]:
    w_full = sc_all_gather(
        w_s.astype(target_dtype),
        ("fsdp", "cp", "ep"),
        core_id=0,
        axis=sharded_axis,
        min_sc_bytes=min_sc_bytes,
    )
    return w_full, ()

  def _bwd(
      residuals: tuple[()],
      dw_full: jax.Array,
  ) -> tuple[jax.Array]:
    del residuals
    if bwd_use_sc:
      dw_shard = sc_psum_scatter(
          dw_full.astype(target_dtype),
          ("fsdp", "cp", "ep"),
          core_id=1,
          scatter_dimension=sharded_axis,
          min_sc_bytes=min_sc_bytes,
      )
    else:
      dw_shard = jax.lax.psum_scatter(
          dw_full.astype(target_dtype),
          ("fsdp", "cp", "ep"),
          scatter_dimension=sharded_axis,
          tiled=True,
      )
    return (dw_shard.astype(orig_shard_dtype),)

  _gather_dense_dual_sc.defvjp(_fwd, _bwd)
  return _gather_dense_dual_sc(w_shard)


_FP8_E4M3_MAX: float = 448.0
_FP8_E5M2_MAX: float = 57344.0
_FP8_SCALE_FLOOR: float = 1e-12


def _fp8_max(qtype: jnp.dtype) -> float:
  if qtype == jnp.dtype(jnp.float8_e4m3fn):
    return _FP8_E4M3_MAX
  if qtype == jnp.dtype(jnp.float8_e5m2):
    return _FP8_E5M2_MAX
  raise ValueError(f"Unsupported FP8 qtype {qtype!r}; expected float8_e4m3fn or float8_e5m2.")


def quantize_fp8_axis(
    x: jax.Array,
    reduce_axis: int | tuple[int, ...],
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    *,
    pmax_axes: str | tuple[str, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` along `reduce_axis` to `qtype` with zero-safe FP32 scale.

  Args:
    x: Input tensor to quantize.
    reduce_axis: Axis or axes over which per-channel absmax is reduced.
    qtype: Target FP8 dtype (`float8_e4m3fn` by default).
    pmax_axes: Optional mesh axis name(s) for cross-shard `lax.pmax` reduction
      when `reduce_axis` is sharded (`None` skips `pmax`).

  Returns:
    Tuple `(q_x, scale)` of quantized `qtype` tensor and `keepdims=True` FP32
    dequantization scale.
  """
  q_dt = jnp.dtype(qtype)
  max_val = _fp8_max(q_dt)
  x_sg = jax.lax.stop_gradient(x)
  amax = jnp.max(jnp.abs(x_sg), axis=reduce_axis, keepdims=True).astype(jnp.float32)
  if pmax_axes is not None:
    amax = jax.lax.pmax(amax, pmax_axes)
  has_nonzero = amax > 0.0
  safe_amax = jnp.where(has_nonzero, amax, jnp.float32(1.0))
  scale = jnp.where(has_nonzero, safe_amax / max_val, jnp.float32(1.0))
  scale_inv = jnp.where(has_nonzero, max_val / safe_amax, jnp.float32(0.0))
  q_x = jnp.clip(x_sg.astype(jnp.float32) * scale_inv, -max_val, max_val).astype(q_dt)
  return q_x, scale


def _fp8_dot_general(
    lhs_q: jax.Array,
    rhs_q: jax.Array,
    *,
    lhs_contract: tuple[int, ...],
    rhs_contract: tuple[int, ...],
) -> jax.Array:
  """Contracts FP8 operands into `float32` on MXU (or exact `f32` on CPU)."""
  if jax.default_backend() == "cpu":
    return jax.lax.dot_general(
        lhs_q.astype(jnp.float32),
        rhs_q.astype(jnp.float32),
        ((lhs_contract, rhs_contract), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    )
  return jax.lax.dot_general(
      lhs_q,
      rhs_q,
      ((lhs_contract, rhs_contract), ((), ())),
      preferred_element_type=jnp.float32,
  )


def prefetch_dense_weight_fp8(
    w_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    rhs_is_transposed: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes local dense weight shard to `float8_e4m3fn` and all-gathers.

  Args:
    w_shard: Local 2D weight shard tensor.
    cfg: Model configuration controlling collective offload.
    sharded_axis: Weight dimension (`0` or `1`) sharded across `('fsdp', 'cp',
      'ep')`.
    rhs_is_transposed: Whether `w_shard` is laid out as `[out_dim, in_dim]`.

  Returns:
    Stop-gradient `(q_w, w_scale)` pair of gathered FP8 weights and FP32 scales.
  """
  if w_shard.ndim != 2 or sharded_axis not in (0, 1):
    raise ValueError(
        "prefetch_dense_weight_fp8 requires 2D w_shard and sharded_axis in"
        f" (0, 1), got shape={w_shard.shape}, sharded_axis={sharded_axis}."
    )
  num_dev = cfg.sharding.num_devices
  contract_axis = 1 if rhs_is_transposed else 0
  pmax_axes = ("fsdp", "cp", "ep") if (sharded_axis == contract_axis and num_dev > 1) else None
  q_w_loc, sc_loc = quantize_fp8_axis(
      w_shard,
      reduce_axis=contract_axis,
      qtype=jnp.float8_e4m3fn,
      pmax_axes=pmax_axes,
  )
  if num_dev == 1:
    q_w = q_w_loc
    sc_full = sc_loc
  else:
    ag_core_id = 0 if cfg.kernels.sc_collective_mode == "dual_sc" else None
    q_w = sc_all_gather(
        q_w_loc,
        ("fsdp", "cp", "ep"),
        core_id=ag_core_id,
        axis=sharded_axis,
        min_sc_bytes=cfg.kernels.sc_min_collective_bytes,
    )
    if sharded_axis == contract_axis:
      sc_full = sc_loc
    else:
      sc_full = jax.lax.all_gather(sc_loc, ("fsdp", "cp", "ep"), axis=sharded_axis, tiled=True)
  w_scale = jnp.swapaxes(sc_full, 0, 1) if rhs_is_transposed else sc_full
  return jax.lax.stop_gradient(q_w), jax.lax.stop_gradient(w_scale)


def fp8_dense_dot_sharded(
    lhs: jax.Array,
    w_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    rhs_is_transposed: bool = False,
    pre_gathered: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Runs N-D FP8 dense matmul with `SC0` gather and BF16 `SC1` grad RS.

  Args:
    lhs: Input activation tensor with shape `[..., in_dim]`.
    w_shard: Local sharded 2D weight tensor.
    cfg: Model configuration controlling dtypes and collective offload.
    sharded_axis: Weight axis along which `w_shard` is sharded.
    rhs_is_transposed: Whether `w_shard` has shape `[out_dim, in_dim]`.
    pre_gathered: Optional pre-gathered `(q_w, w_scale)` pair. When `None`,
      `w_shard` is quantized and all-gathered dynamically.

  Returns:
    Contracted output tensor in `cfg.train.compute_jnp_dtype`.
  """
  if lhs.ndim < 2:
    raise ValueError(f"fp8_dense_dot_sharded requires lhs.ndim >= 2, got {lhs.shape}.")
  if lhs.ndim > 2:
    lead_shape = lhs.shape[:-1]
    out_2d = fp8_dense_dot_sharded(
        lhs.reshape(-1, lhs.shape[-1]),
        w_shard,
        cfg,
        sharded_axis=sharded_axis,
        rhs_is_transposed=rhs_is_transposed,
        pre_gathered=pre_gathered,
    )
    return out_2d.reshape(*lead_shape, out_2d.shape[-1])
  if pre_gathered is None:
    q_w_in, w_sc_in = prefetch_dense_weight_fp8(
        w_shard,
        cfg,
        sharded_axis=sharded_axis,
        rhs_is_transposed=rhs_is_transposed,
    )
  else:
    q_w_in = jax.lax.stop_gradient(pre_gathered[0])
    w_sc_in = jax.lax.stop_gradient(pre_gathered[1])

  comp_dt = cfg.train.compute_jnp_dtype
  orig_lhs_dtype = lhs.dtype
  orig_shard_dtype = w_shard.dtype
  min_sc_bytes = cfg.kernels.sc_min_collective_bytes
  num_dev = cfg.sharding.num_devices
  sc_mode = cfg.kernels.sc_collective_mode
  last_ax = lhs.ndim - 1
  batch_axes = tuple(range(last_ax))

  def _eval_fwd(lhs_arr: jax.Array, q_w_g: jax.Array, w_sc_g: jax.Array) -> tuple[jax.Array, jax.Array]:
    lhs_c = lhs_arr.astype(comp_dt)
    q_lhs, lhs_scale = quantize_fp8_axis(lhs_c, reduce_axis=last_ax, qtype=jnp.float8_e4m3fn)
    raw = _fp8_dot_general(
        q_lhs,
        q_w_g,
        lhs_contract=(last_ax,),
        rhs_contract=(1 if rhs_is_transposed else 0,),
    )
    w_sc_1d = jnp.squeeze(w_sc_g, axis=0)
    out = ((raw * lhs_scale) * w_sc_1d).astype(comp_dt)
    return out, lhs_c

  @jax.custom_vjp
  def _op(
      lhs_arr: jax.Array,
      w_s: jax.Array,
      q_w_g: jax.Array,
      w_sc_g: jax.Array,
  ) -> jax.Array:
    del w_s
    out, _ = _eval_fwd(lhs_arr, q_w_g, w_sc_g)
    return out

  def _fwd(
      lhs_arr: jax.Array,
      w_s: jax.Array,
      q_w_g: jax.Array,
      w_sc_g: jax.Array,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    del w_s
    out, lhs_c = _eval_fwd(lhs_arr, q_w_g, w_sc_g)
    return out, (lhs_c, q_w_g, w_sc_g)

  def _bwd(
      residuals: tuple[jax.Array, jax.Array, jax.Array],
      do: jax.Array,
  ) -> tuple[jax.Array, jax.Array, None, None]:
    lhs_c, q_w_g, w_sc_g = residuals
    do_c = do.astype(comp_dt)
    w_sc_1d = jnp.squeeze(w_sc_g, axis=0)

    do_scaled = (do_c.astype(jnp.float32) * w_sc_1d).astype(comp_dt)
    q_do_dlhs, do_dlhs_scale = quantize_fp8_axis(do_scaled, reduce_axis=last_ax, qtype=jnp.float8_e4m3fn)
    raw_dlhs = _fp8_dot_general(
        q_do_dlhs,
        q_w_g,
        lhs_contract=(last_ax,),
        rhs_contract=(0 if rhs_is_transposed else 1,),
    )
    dlhs = (raw_dlhs * do_dlhs_scale).astype(orig_lhs_dtype)

    wgrad_prec = jax.lax.Precision.HIGHEST if jax.default_backend() == "cpu" else None
    if rhs_is_transposed:
      dw_full = jax.lax.dot_general(
          do_c,
          lhs_c,
          ((batch_axes, batch_axes), ((), ())),
          precision=wgrad_prec,
          preferred_element_type=jnp.float32,
      ).astype(comp_dt)
    else:
      dw_full = jax.lax.dot_general(
          lhs_c,
          do_c,
          ((batch_axes, batch_axes), ((), ())),
          precision=wgrad_prec,
          preferred_element_type=jnp.float32,
      ).astype(comp_dt)

    if num_dev == 1:
      dw_shard = dw_full
    elif sc_mode == "dual_sc":
      dw_shard = sc_psum_scatter(
          dw_full,
          ("fsdp", "cp", "ep"),
          core_id=1,
          scatter_dimension=sharded_axis,
          min_sc_bytes=min_sc_bytes,
      )
    else:
      dw_shard = jax.lax.psum_scatter(
          dw_full,
          ("fsdp", "cp", "ep"),
          scatter_dimension=sharded_axis,
          tiled=True,
      )
    return dlhs, dw_shard.astype(orig_shard_dtype), None, None

  _op.defvjp(_fwd, _bwd)
  return _op(lhs, w_shard, q_w_in, w_sc_in)


def prefetch_routed_weight_fp8(
    w_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    role: str,
    rhs_is_transposed: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes local 3D routed weight shard to `float8_e4m3fn` and gathers.

  Args:
    w_shard: Local 3D expert weight shard `[E_local, ...]`.
    cfg: Model configuration controlling Dual-SC collective offload.
    sharded_axis: Axis sharded across `('fsdp', 'cp')`.
    role: Expert projection role (`'gate_up'` or `'down'`).
    rhs_is_transposed: Whether the contracting dimension is axis `2`.

  Returns:
    Stop-gradient `(q_w, w_scale)` pair of gathered FP8 weights and FP32 scales.
  """
  if role not in ("gate_up", "down"):
    raise ValueError(f"role must be 'gate_up' or 'down', got {role!r}.")
  if w_shard.ndim != 3:
    raise ValueError(f"Expected 3D routed w_shard, got {w_shard.shape}.")
  norm_axis = sharded_axis % 3
  contract_axis = 2 if rhs_is_transposed else 1
  routed_ways = cfg.sharding.fsdp_size * cfg.sharding.cp_size
  use_tiles = cfg.kernels.routed_w_tiled
  sc_mode = cfg.kernels.sc_collective_mode
  min_sc_bytes = cfg.kernels.sc_min_collective_bytes

  pmax_axes = ("fsdp", "cp") if (norm_axis == contract_axis and routed_ways > 1) else None
  q_w_loc, sc_loc = quantize_fp8_axis(
      w_shard,
      reduce_axis=contract_axis,
      qtype=jnp.float8_e4m3fn,
      pmax_axes=pmax_axes,
  )

  if routed_ways == 1:
    q_w = routed_w_from_tiles(routed_w_to_tiles(q_w_loc)) if use_tiles else q_w_loc
    w_scale = jnp.swapaxes(sc_loc, 1, 2) if rhs_is_transposed else sc_loc
    return jax.lax.stop_gradient(q_w), jax.lax.stop_gradient(w_scale)

  ag_core_id: int | None = (1 if role == "gate_up" else 0) if sc_mode == "dual_sc" else None
  if norm_axis == 0:
    e_shard, rows, cols = q_w_loc.shape
    q_w = sc_all_gather(
        q_w_loc.reshape(e_shard * rows, cols),
        ("fsdp", "cp"),
        core_id=ag_core_id,
        axis=0,
        min_sc_bytes=min_sc_bytes,
    ).reshape(-1, rows, cols)
  else:
    eff_axis = norm_axis + 1 if (use_tiles and norm_axis == 2) else norm_axis
    work_q = routed_w_to_tiles(q_w_loc) if use_tiles else q_w_loc
    q_w = sc_all_gather(
        work_q,
        ("fsdp", "cp"),
        core_id=ag_core_id,
        axis=eff_axis,
        min_sc_bytes=min_sc_bytes,
    )
    if use_tiles:
      q_w = routed_w_from_tiles(q_w)

  if norm_axis == contract_axis:
    sc_full = sc_loc
  else:
    sc_full = jax.lax.all_gather(sc_loc, ("fsdp", "cp"), axis=norm_axis, tiled=True)
  w_scale = jnp.swapaxes(sc_full, 1, 2) if rhs_is_transposed else sc_full
  return jax.lax.stop_gradient(q_w), jax.lax.stop_gradient(w_scale)


def reduce_scatter_routed_grad(
    dw_full: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    role: str,
    out_dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """Reduce-scatters 3D routed expert gradients across `('fsdp', 'cp')`.

  Args:
    dw_full: Unsharded local-gradient tensor `[E_local, ...]` across `('fsdp',
      'cp')`.
    cfg: Model configuration controlling compute dtype and Dual-SC pinning.
    sharded_axis: Axis along which the parameter shard is partitioned.
    role: Expert projection role (`'gate_up'` or `'down'`).
    out_dtype: Output gradient dtype matching the parameter shard dtype.

  Returns:
    Reduce-scattered expert weight gradient shard cast to `out_dtype`.
  """
  if role not in ("gate_up", "down"):
    raise ValueError(f"role must be 'gate_up' or 'down', got {role!r}.")
  comp_dt = cfg.train.compute_jnp_dtype
  routed_ways = cfg.sharding.fsdp_size * cfg.sharding.cp_size
  use_tiles = cfg.kernels.routed_w_tiled
  if routed_ways == 1:
    work = dw_full.astype(comp_dt)
    if use_tiles:
      work = routed_w_from_tiles(routed_w_to_tiles(work))
    return work.astype(out_dtype)

  norm_axis = sharded_axis % dw_full.ndim
  rs_core_id: int | None = (0 if role == "gate_up" else 1) if cfg.kernels.sc_collective_mode == "dual_sc" else None
  min_sc_bytes = cfg.kernels.sc_min_collective_bytes
  if norm_axis == 0 and dw_full.ndim == 3:
    e_full, rows, cols = dw_full.shape
    dw_2d = sc_psum_scatter(
        dw_full.astype(comp_dt).reshape(e_full * rows, cols),
        ("fsdp", "cp"),
        core_id=rs_core_id,
        scatter_dimension=0,
        min_sc_bytes=min_sc_bytes,
    )
    return dw_2d.reshape(-1, rows, cols).astype(out_dtype)

  eff_axis = norm_axis + 1 if (use_tiles and norm_axis == dw_full.ndim - 1) else norm_axis
  work = dw_full.astype(comp_dt)
  if use_tiles:
    work = routed_w_to_tiles(work)
  dw_shard = sc_psum_scatter(
      work,
      ("fsdp", "cp"),
      core_id=rs_core_id,
      scatter_dimension=eff_axis,
      min_sc_bytes=min_sc_bytes,
  )
  if use_tiles:
    dw_shard = routed_w_from_tiles(dw_shard)
  return dw_shard.astype(out_dtype)


def _quantize_routed_fp8(
    w: jax.Array,
    reduce_axis: int,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes 3D routed weights to `float8_e4m3fn` with FP32 scales."""
  w_f32 = w.astype(jnp.float32)
  amax = jnp.max(jnp.abs(w_f32), axis=reduce_axis, keepdims=True)
  scale = jnp.maximum(amax / _FP8_E4M3_MAX, jnp.float32(_FP8_SCALE_FLOOR))
  q_fp8 = jnp.clip(w_f32 / scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
  return q_fp8, scale


def gather_routed_weight(
    w_shard: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    dtype: jax.typing.DTypeLike | None = None,
    role: str = "down",
) -> jax.Array:
  """Casts and all-gathers a routed weight shard across `('fsdp', 'cp')`."""
  if role not in ("gate_up", "down"):
    raise ValueError(f"role must be 'gate_up' or 'down', got {role!r}.")
  target_dtype = jnp.dtype(dtype) if dtype is not None else cfg.train.compute_jnp_dtype
  use_tiles = cfg.kernels.routed_w_tiled
  use_fp8_ag = cfg.kernels.fp8_routed_all_gather
  sc_mode = cfg.kernels.sc_collective_mode
  norm_axis = sharded_axis % w_shard.ndim
  flatten_expert_2d = norm_axis == 0 and w_shard.ndim == 3

  if sc_mode == "off" and not use_tiles and not use_fp8_ag and not flatten_expert_2d:
    w_cast = w_shard.astype(target_dtype)
    if cfg.sharding.fsdp_size * cfg.sharding.cp_size == 1:
      return w_cast
    return jax.lax.all_gather(
        w_cast,
        ("fsdp", "cp"),
        axis=sharded_axis,
        tiled=True,
    )

  if cfg.sharding.fsdp_size * cfg.sharding.cp_size == 1:
    if use_fp8_ag:
      red_ax = 1 if norm_axis == 0 else (2 if norm_axis == 1 else 1)
      q_fp8, sc = _quantize_routed_fp8(w_shard, reduce_axis=red_ax)
      w_cast = (q_fp8.astype(jnp.float32) * sc).astype(target_dtype)
    else:
      w_cast = w_shard.astype(target_dtype)
    if use_tiles:
      return routed_w_from_tiles(routed_w_to_tiles(w_cast))
    return w_cast

  orig_shard_dtype = w_shard.dtype
  min_sc_bytes = cfg.kernels.sc_min_collective_bytes
  eff_axis = norm_axis + 1 if (use_tiles and norm_axis == w_shard.ndim - 1) else norm_axis
  if sc_mode == "dual_sc":
    ag_core_id: int | None = 1 if role == "gate_up" else 0
    rs_core_id: int | None = 0 if role == "gate_up" else 1
  else:
    ag_core_id = None
    rs_core_id = None

  def _run_ag(w_s: jax.Array) -> jax.Array:
    if flatten_expert_2d:
      e_shard, rows, cols = w_s.shape
      if use_fp8_ag:
        q_fp8, sc_l = _quantize_routed_fp8(w_s, reduce_axis=1)
        q_2d = sc_all_gather(
            q_fp8.reshape(e_shard * rows, cols),
            ("fsdp", "cp"),
            core_id=ag_core_id,
            axis=0,
            min_sc_bytes=min_sc_bytes,
        ).reshape(-1, rows, cols)
        sc_full = jax.lax.all_gather(sc_l, ("fsdp", "cp"), axis=0, tiled=True)
        return (q_2d.astype(jnp.float32) * sc_full).astype(target_dtype)
      work_2d = w_s.astype(target_dtype).reshape(e_shard * rows, cols)
      gathered_2d = sc_all_gather(
          work_2d,
          ("fsdp", "cp"),
          core_id=ag_core_id,
          axis=0,
          min_sc_bytes=min_sc_bytes,
      )
      return gathered_2d.reshape(-1, rows, cols)

    if use_fp8_ag:
      red_ax = 2 if norm_axis == 1 else 1
      q_fp8, sc_l = _quantize_routed_fp8(w_s, reduce_axis=red_ax)
      work_q = routed_w_to_tiles(q_fp8) if use_tiles else q_fp8
      q_full = sc_all_gather(
          work_q,
          ("fsdp", "cp"),
          core_id=ag_core_id,
          axis=eff_axis,
          min_sc_bytes=min_sc_bytes,
      )
      if use_tiles:
        q_full = routed_w_from_tiles(q_full)
      sc_full = jax.lax.all_gather(sc_l, ("fsdp", "cp"), axis=norm_axis, tiled=True)
      return (q_full.astype(jnp.float32) * sc_full).astype(target_dtype)

    work = w_s.astype(target_dtype)
    if use_tiles:
      work = routed_w_to_tiles(work)
    gathered = sc_all_gather(
        work,
        ("fsdp", "cp"),
        core_id=ag_core_id,
        axis=eff_axis,
        min_sc_bytes=min_sc_bytes,
    )
    if use_tiles:
      gathered = routed_w_from_tiles(gathered)
    return gathered

  @jax.custom_vjp
  def _gather_routed_custom(w_s: jax.Array) -> jax.Array:
    return _run_ag(w_s)

  def _fwd(w_s: jax.Array) -> tuple[jax.Array, tuple[()]]:
    return _run_ag(w_s), ()

  def _bwd(
      residuals: tuple[()],
      dw_full: jax.Array,
  ) -> tuple[jax.Array]:
    del residuals
    if flatten_expert_2d:
      e_full, rows, cols = dw_full.shape
      work_2d = dw_full.astype(target_dtype).reshape(e_full * rows, cols)
      dw_2d = sc_psum_scatter(
          work_2d,
          ("fsdp", "cp"),
          core_id=rs_core_id,
          scatter_dimension=0,
          min_sc_bytes=min_sc_bytes,
      )
      return (dw_2d.reshape(-1, rows, cols).astype(orig_shard_dtype),)
    work = dw_full.astype(target_dtype)
    if use_tiles:
      work = routed_w_to_tiles(work)
    dw_shard = sc_psum_scatter(
        work,
        ("fsdp", "cp"),
        core_id=rs_core_id,
        scatter_dimension=eff_axis,
        min_sc_bytes=min_sc_bytes,
    )
    if use_tiles:
      dw_shard = routed_w_from_tiles(dw_shard)
    return (dw_shard.astype(orig_shard_dtype),)

  _gather_routed_custom.defvjp(_fwd, _bwd)
  return _gather_routed_custom(w_shard)


def psum_replicated_grads(
    grads: flint_types.FlintParams[jax.Array],
    pspecs: flint_types.FlintParams[PartitionSpec],
    sharding_cfg: flint_config.FlintShardingConfig,
) -> flint_types.FlintParams[jax.Array]:
  """Applies a single `psum` over `('fsdp','cp','ep')` to `P()` grad leaves."""
  if sharding_cfg.num_devices == 1:
    return grads
  empty_spec = PartitionSpec()
  return jax.tree.map(
      lambda g, spec: (jax.lax.psum(g, ("fsdp", "cp", "ep")) if spec == empty_spec else g),
      grads,
      pspecs,
  )


def lb_chunk_dest(chunk_idx: int, cp_size: int) -> int:
  """Returns the CP rank holding natural chunk `chunk_idx` (of `2 * cp_size`)."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if chunk_idx < 0 or chunk_idx >= 2 * cp_size:
    raise ValueError(f"chunk_idx={chunk_idx} out of range [0, {2 * cp_size}).")
  return chunk_idx if chunk_idx < cp_size else 2 * cp_size - 1 - chunk_idx


def cp_natural_to_lb(
    x: jax.Array,
    cp_size: int,
    axis: int = 1,
    axis_name: str = "cp",
) -> jax.Array:
  """Permutes a natural-order `[2r, 2r+1]` CP shard into `[d, 2n-1-d]` LB order."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    return x
  norm_axis = axis % x.ndim
  if x.shape[norm_axis] % 2 != 0:
    raise ValueError(f"Local CP sequence length {x.shape[norm_axis]} on axis {axis} must be" " even.")
  half = x.shape[norm_axis] // 2
  h0 = jax.lax.slice_in_dim(x, 0, half, axis=norm_axis)
  h1 = jax.lax.slice_in_dim(x, half, 2 * half, axis=norm_axis)
  recv0 = jax.lax.ppermute(
      h0,
      axis_name,
      [(r, lb_chunk_dest(2 * r, cp_size)) for r in range(cp_size)],
  )
  recv1 = jax.lax.ppermute(
      h1,
      axis_name,
      [(r, lb_chunk_dest(2 * r + 1, cp_size)) for r in range(cp_size)],
  )
  even = jax.lax.axis_index(axis_name) % 2 == 0
  first = jnp.where(even, recv0, recv1)
  second = jnp.where(even, recv1, recv0)
  return jnp.concatenate([first, second], axis=norm_axis)


def cp_lb_to_natural(
    x: jax.Array,
    cp_size: int,
    axis: int = 1,
    axis_name: str = "cp",
) -> jax.Array:
  """Permutes a load-balanced `[d, 2n-1-d]` CP shard back to `[2r, 2r+1]`."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    return x
  norm_axis = axis % x.ndim
  if x.shape[norm_axis] % 2 != 0:
    raise ValueError(f"Local CP sequence length {x.shape[norm_axis]} on axis {axis} must be" " even.")
  half = x.shape[norm_axis] // 2
  slot0 = jax.lax.slice_in_dim(x, 0, half, axis=norm_axis)
  slot1 = jax.lax.slice_in_dim(x, half, 2 * half, axis=norm_axis)
  even = jax.lax.axis_index(axis_name) % 2 == 0
  even_chunk = jnp.where(even, slot0, slot1)
  odd_chunk = jnp.where(even, slot1, slot0)
  recv_even = jax.lax.ppermute(
      even_chunk,
      axis_name,
      [(lb_chunk_dest(2 * r, cp_size), r) for r in range(cp_size)],
  )
  recv_odd = jax.lax.ppermute(
      odd_chunk,
      axis_name,
      [(lb_chunk_dest(2 * r + 1, cp_size), r) for r in range(cp_size)],
  )
  return jnp.concatenate([recv_even, recv_odd], axis=norm_axis)


def cp_all_gather_natural(
    x_nat: jax.Array,
    cp_size: int,
    axis: int = 1,
    axis_name: str = "cp",
) -> jax.Array:
  """All-gathers a natural-order CP sequence shard into the full sequence."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    return x_nat
  norm_axis = axis % x_nat.ndim
  return jax.lax.all_gather(x_nat, axis_name, axis=norm_axis, tiled=True)


def cp_all_gather_lb_to_natural(
    x_lb: jax.Array,
    cp_size: int,
    axis: int = 1,
    axis_name: str = "cp",
) -> jax.Array:
  """All-gathers a load-balanced CP shard into the full natural sequence."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    return x_lb
  norm_axis = axis % x_lb.ndim
  if x_lb.shape[norm_axis] % 2 != 0:
    raise ValueError(f"Local CP sequence length {x_lb.shape[norm_axis]} on axis {axis} must" " be even.")
  half = x_lb.shape[norm_axis] // 2
  gathered = jax.lax.all_gather(x_lb, axis_name, axis=0, tiled=False)
  g0 = jax.lax.slice_in_dim(gathered, 0, half, axis=norm_axis + 1)
  g1 = jax.lax.slice_in_dim(gathered, half, 2 * half, axis=norm_axis + 1)
  chunks = jnp.concatenate([g0, jnp.flip(g1, axis=0)], axis=0)
  chunks = jnp.moveaxis(chunks, 0, norm_axis)
  out_shape = list(x_lb.shape)
  out_shape[norm_axis] = 2 * cp_size * half
  return chunks.reshape(out_shape)


def cp_lb_global_positions(
    seq_len: int,
    cp_size: int,
    axis_name: str = "cp",
) -> jax.Array:
  """Returns `int32[seq_len // cp_size]` global indices for this LB CP rank."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    if seq_len < 1:
      raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
    return jnp.arange(seq_len, dtype=jnp.int32)
  if seq_len % (2 * cp_size) != 0:
    raise ValueError(f"seq_len={seq_len} must be divisible by 2 * cp_size = {2 * cp_size}.")
  half = seq_len // (2 * cp_size)
  rank = jax.lax.axis_index(axis_name)
  ar = jnp.arange(half, dtype=jnp.int32)
  return jnp.concatenate([rank * half + ar, (2 * cp_size - 1 - rank) * half + ar], axis=0)


def lb_q_sequence_np(seq_len: int, cp_size: int) -> np.ndarray:
  """Returns the global `int32[seq_len]` query index map across all LB CP ranks."""
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size == 1:
    if seq_len < 1:
      raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
    return np.arange(seq_len, dtype=np.int32)
  if seq_len % (2 * cp_size) != 0:
    raise ValueError(f"seq_len={seq_len} must be divisible by 2 * cp_size = {2 * cp_size}.")
  half = seq_len // (2 * cp_size)
  ar = np.arange(half, dtype=np.int32)
  parts = [np.concatenate([rank * half + ar, (2 * cp_size - 1 - rank) * half + ar]) for rank in range(cp_size)]
  return np.concatenate(parts, axis=0)
