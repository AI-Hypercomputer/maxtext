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

"""VMEM-fused Pallas Mosaic TPU kernel for Qwen3.8 GDN 4-tap causal `Conv1D + SiLU + Q/K L2Norm`.

Eliminates:
  1. `jnp.concatenate([left_halo, x_l], axis=1)` (`335.5 MB` HBM copy to prepend `120 KiB` halo),
  2. Unfused `jit(silu)` (`335.5 MB` HBM round-trip outside `shard_map`), and
  3. Separate `l2norm(query)` / `l2norm(key)` HBM passes across both `CP > 1` (`64K`) and `CP == 1` (`8K`).
"""

import contextlib
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

LANES = 128
_CONV_W = 4
_HALO_PAD = 8
_TILE_S = 128
_BWD_TILE_S = 64
_VMEM_LIMIT_BYTES = 58 * 1024 * 1024
_HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)


@contextlib.contextmanager
def _tpu_mesh_context():
  """Provides a TPU7x AbstractMesh when running Pallas pipelines under CPU interpret mode."""
  needs_mesh = False
  try:
    if not pltpu.is_tpu_device():
      needs_mesh = True
  except (RuntimeError, ValueError, AttributeError, TypeError):
    needs_mesh = True
  if needs_mesh:
    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def can_use_gdn_conv_pallas(
    *,
    s_local: int,
    conv_dim: int,
    kernel_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    channel_sharded: bool,
    is_tpu: bool,
    interpret: bool = False,
) -> bool:
  """Returns True when local GDN shapes satisfy 4-tap Pallas VMEM invariants."""
  if not (is_tpu or interpret) or channel_sharded:
    return False
  if kernel_size != _CONV_W or s_local < 128 or s_local % 128 != 0:
    return False
  if head_k_dim % LANES != 0 or head_v_dim % LANES != 0:
    return False
  expected_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
  return conv_dim == expected_dim and conv_dim <= 16384


def _choose_tile_s(s_local: int, preferred: int) -> int:
  for cand in (preferred, 128, 64):
    if s_local >= cand and s_local % cand == 0:
      return cand
  return 64


def _build_tile_halos(x_l: jax.Array, left_halo: jax.Array, *, tile_s: int) -> jax.Array:
  """Builds tiny `(b_loc, num_s_tiles, 8, conv_dim)` per-tile halo table (`< 0.5%` of `x_l`, zero `x_l` copy)."""
  b_loc, s_loc, c_dim = x_l.shape
  num_tiles = s_loc // tile_s
  h_len = _CONV_W - 1  # 3
  halo_0 = left_halo[:, None, :, :]  # [b, 1, 3, c]
  if num_tiles > 1:
    prev_tails = x_l.reshape(b_loc, num_tiles, tile_s, c_dim)[:, :-1, -h_len:, :]  # [b, num_tiles - 1, 3, c]
    halos_3 = jnp.concatenate([halo_0, prev_tails], axis=1)
  else:
    halos_3 = halo_0
  # Pad sublane axis 3 -> 8 so every `(1, 1, 8, c_dim)` VMEM block is sublane-aligned.
  return jnp.pad(halos_3, ((0, 0), (0, 0), (_HALO_PAD - h_len, 0), (0, 0)))


def _conv_silu_l2norm_tile(
    x_tile: jax.Array,
    halo_tile: jax.Array,
    w_pad: jax.Array,
    *,
    num_k_heads: int,
    head_k_dim: int,
    eps: float,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes 4-tap causal `conv1d + silu + q/k l2norm` for one `(tile_s, conv_dim)` tile in VMEM."""
  tile_s, _ = x_tile.shape
  kd = num_k_heads * head_k_dim
  h_3 = halo_tile[_HALO_PAD - 3 :, :]  # [3, c_dim]
  win = jnp.concatenate([h_3, x_tile], axis=0).astype(jnp.float32)  # [tile_s + 3, c_dim] in VMEM!
  w_4 = w_pad[:_CONV_W, :].astype(jnp.float32)

  acc = win[0:tile_s, :] * w_4[0:1, :]
  for j in range(1, _CONV_W):
    acc = acc + win[j : j + tile_s, :] * w_4[j : j + 1, :]
  mixed = jax.nn.silu(acc).astype(out_dtype)

  q_raw = mixed[:, :kd].reshape(tile_s, num_k_heads, head_k_dim).astype(jnp.float32)
  k_raw = mixed[:, kd : 2 * kd].reshape(tile_s, num_k_heads, head_k_dim).astype(jnp.float32)
  q_norm = (q_raw * lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + jnp.float32(eps))).astype(out_dtype)
  k_norm = (k_raw * lax.rsqrt(jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + jnp.float32(eps))).astype(out_dtype)
  v_act = mixed[:, 2 * kd :]
  return q_norm.reshape(tile_s, kd), k_norm.reshape(tile_s, kd), v_act


def _conv_silu_l2norm_tile_bwd(
    x_tile: jax.Array,
    halo_tile: jax.Array,
    w_pad: jax.Array,
    dq_tile: jax.Array,
    dk_tile: jax.Array,
    dv_tile: jax.Array,
    *,
    num_k_heads: int,
    head_k_dim: int,
    eps: float,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Analytical VJP of `_conv_silu_l2norm_tile` without `dynamic_update_slice` VREG spills (`~21 MiB` VMEM)."""
  tile_s, _ = x_tile.shape
  kd = num_k_heads * head_k_dim
  h_3 = halo_tile[_HALO_PAD - 3 :, :]
  win = jnp.concatenate([h_3, x_tile], axis=0).astype(jnp.float32)
  w_4 = w_pad[:_CONV_W, :].astype(jnp.float32)

  acc = win[0:tile_s, :] * w_4[0:1, :]
  for j in range(1, _CONV_W):
    acc = acc + win[j : j + tile_s, :] * w_4[j : j + 1, :]
  sig = jax.nn.sigmoid(acc)
  mixed = (acc * sig).astype(out_dtype)

  q_raw = mixed[:, :kd].reshape(tile_s, num_k_heads, head_k_dim).astype(jnp.float32)
  k_raw = mixed[:, kd : 2 * kd].reshape(tile_s, num_k_heads, head_k_dim).astype(jnp.float32)
  rq = lax.rsqrt(jnp.sum(q_raw * q_raw, axis=-1, keepdims=True) + jnp.float32(eps))
  rk = lax.rsqrt(jnp.sum(k_raw * k_raw, axis=-1, keepdims=True) + jnp.float32(eps))

  dq_f32 = dq_tile.astype(out_dtype).astype(jnp.float32).reshape(tile_s, num_k_heads, head_k_dim)
  dk_f32 = dk_tile.astype(out_dtype).astype(jnp.float32).reshape(tile_s, num_k_heads, head_k_dim)
  dq_raw = rq * (dq_f32 - (rq * rq * jnp.sum(dq_f32 * q_raw, axis=-1, keepdims=True)) * q_raw)
  dk_raw = rk * (dk_f32 - (rk * rk * jnp.sum(dk_f32 * k_raw, axis=-1, keepdims=True)) * k_raw)

  dmixed = jnp.concatenate(
      [
          dq_raw.reshape(tile_s, kd).astype(out_dtype).astype(jnp.float32),
          dk_raw.reshape(tile_s, kd).astype(out_dtype).astype(jnp.float32),
          dv_tile.astype(out_dtype).astype(jnp.float32),
      ],
      axis=-1,
  )
  dacc = dmixed * sig * (1.0 + acc * (1.0 - sig))

  dw_rows = [jnp.sum(dacc * win[j : j + tile_s, :], axis=0, keepdims=True) for j in range(_CONV_W)]
  dw_t = jnp.pad(jnp.concatenate(dw_rows, axis=0), ((0, 8 - _CONV_W), (0, 0)))

  dx_t = (
      dacc * w_4[3:4, :]
      + jnp.pad(dacc[1:, :] * w_4[2:3, :], ((0, 1), (0, 0)))
      + jnp.pad(dacc[2:, :] * w_4[1:2, :], ((0, 2), (0, 0)))
      + jnp.pad(dacc[3:, :] * w_4[0:1, :], ((0, 3), (0, 0)))
  )
  dh_0 = dacc[0:1, :] * w_4[0:1, :]
  dh_1 = dacc[1:2, :] * w_4[0:1, :] + dacc[0:1, :] * w_4[1:2, :]
  dh_2 = dacc[2:3, :] * w_4[0:1, :] + dacc[1:2, :] * w_4[1:2, :] + dacc[0:1, :] * w_4[2:3, :]
  dh_t = jnp.pad(jnp.concatenate([dh_0, dh_1, dh_2], axis=0), ((_HALO_PAD - 3, 0), (0, 0)))
  return dx_t, dh_t, dw_t


@functools.partial(
    jax.jit,
    static_argnames=("num_k_heads", "head_k_dim", "eps", "tile_s", "interpret"),
)
def _gdn_conv_pallas_fwd(
    x_4d: jax.Array,
    halos_4d: jax.Array,
    w_pad: jax.Array,
    *,
    num_k_heads: int,
    head_k_dim: int,
    eps: float,
    tile_s: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Forward Pallas pipeline over `(b_loc, num_s_tiles)`."""
  b_loc, num_tiles, _, c_dim = x_4d.shape
  kd = num_k_heads * head_k_dim
  vd = c_dim - 2 * kd
  out_dt = x_4d.dtype

  def body(step: pltpu.PipelineStep, x_ref, h_ref, w_ref, q_ref, k_ref, v_ref):
    del step
    q_t, k_t, v_t = _conv_silu_l2norm_tile(
        x_ref[0, 0],
        h_ref[0, 0],
        w_ref[...],
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        eps=eps,
        out_dtype=out_dt,
    )
    q_ref[0, 0] = q_t
    k_ref[0, 0] = k_t
    v_ref[0, 0] = v_t

  spec_x = pl.BlockSpec((1, 1, tile_s, c_dim), lambda b, t: (b, t, 0, 0))
  spec_h = pl.BlockSpec((1, 1, _HALO_PAD, c_dim), lambda b, t: (b, t, 0, 0))
  spec_w = pl.BlockSpec((8, c_dim), lambda b, t: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_q = pl.BlockSpec((1, 1, tile_s, kd), lambda b, t: (b, t, 0, 0))
  spec_v = pl.BlockSpec((1, 1, tile_s, vd), lambda b, t: (b, t, 0, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(b_loc, num_tiles),
      in_specs=(spec_x, spec_h, spec_w),
      out_specs=(spec_q, spec_q, spec_v),
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
      _explicit_indices=True,
  )
  tot_tok = b_loc * num_tiles * tile_s
  flops = int(14 * tot_tok * c_dim)
  bytes_acc = int(2 * tot_tok * c_dim * out_dt.itemsize)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((b_loc, num_tiles, tile_s, kd), out_dt),
            jax.ShapeDtypeStruct((b_loc, num_tiles, tile_s, kd), out_dt),
            jax.ShapeDtypeStruct((b_loc, num_tiles, tile_s, vd), out_dt),
        ),
        in_specs=(_HBM_SPEC,) * 3,
        out_specs=(_HBM_SPEC,) * 3,
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=tot_tok * c_dim, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"gdn_conv_silu_l2norm_fwd_ts{tile_s}",
    )(x_4d, halos_4d, w_pad)


@functools.partial(
    jax.jit,
    static_argnames=("num_k_heads", "head_k_dim", "eps", "tile_s", "interpret"),
)
def _gdn_conv_pallas_bwd(
    x_4d: jax.Array,
    halos_4d: jax.Array,
    w_pad: jax.Array,
    dq_4d: jax.Array,
    dk_4d: jax.Array,
    dv_4d: jax.Array,
    *,
    num_k_heads: int,
    head_k_dim: int,
    eps: float,
    tile_s: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Backward Pallas pipeline producing `(dx_interior, dhalos_4d, dw_pad)` in `~21 MiB` VMEM."""
  b_loc, num_tiles, _, c_dim = x_4d.shape
  kd = num_k_heads * head_k_dim
  vd = c_dim - 2 * kd
  out_dt = x_4d.dtype

  def body(
      step: pltpu.PipelineStep,
      x_ref,
      h_ref,
      w_ref,
      dq_ref,
      dk_ref,
      dv_ref,
      dx_ref,
      dh_ref,
      dw_ref,
  ):
    b_i = step.index[0]
    t_i = step.index[1]
    dx_t, dh_t, dw_t = _conv_silu_l2norm_tile_bwd(
        x_ref[0, 0],
        h_ref[0, 0],
        w_ref[...],
        dq_ref[0, 0],
        dk_ref[0, 0],
        dv_ref[0, 0],
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        eps=eps,
        out_dtype=out_dt,
    )
    dx_ref[0, 0] = dx_t.astype(jnp.float32)
    dh_ref[0, 0] = dh_t.astype(jnp.float32)

    @pl.when((b_i == 0) & (t_i == 0))
    def _init_dw():
      dw_ref[...] = jnp.zeros_like(dw_ref)

    dw_ref[...] += dw_t.astype(jnp.float32)

  spec_x = pl.BlockSpec((1, 1, tile_s, c_dim), lambda b, t: (b, t, 0, 0))
  spec_h = pl.BlockSpec((1, 1, _HALO_PAD, c_dim), lambda b, t: (b, t, 0, 0))
  spec_w_in = pl.BlockSpec((8, c_dim), lambda b, t: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_q = pl.BlockSpec((1, 1, tile_s, kd), lambda b, t: (b, t, 0, 0))
  spec_v = pl.BlockSpec((1, 1, tile_s, vd), lambda b, t: (b, t, 0, 0))
  spec_w_out = pl.BlockSpec((8, c_dim), lambda b, t: (0, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(b_loc, num_tiles),
      in_specs=(spec_x, spec_h, spec_w_in, spec_q, spec_q, spec_v),
      out_specs=(spec_x, spec_h, spec_w_out),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.ARBITRARY),
      _explicit_indices=True,
  )
  tot_tok = b_loc * num_tiles * tile_s
  flops = int(32 * tot_tok * c_dim)
  bytes_acc = int(tot_tok * c_dim * (2 * out_dt.itemsize + 4))
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((b_loc, num_tiles, tile_s, c_dim), jnp.float32),
            jax.ShapeDtypeStruct((b_loc, num_tiles, _HALO_PAD, c_dim), jnp.float32),
            jax.ShapeDtypeStruct((8, c_dim), jnp.float32),
        ),
        in_specs=(_HBM_SPEC,) * 6,
        out_specs=(_HBM_SPEC,) * 3,
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=2 * tot_tok * c_dim, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"gdn_conv_silu_l2norm_bwd_ts{tile_s}",
    )(x_4d, halos_4d, w_pad, dq_4d, dk_4d, dv_4d)


def gdn_causal_conv1d_silu_l2norm_local(
    x_l: jax.Array,
    left_halo: jax.Array,
    conv_kernel: jax.Array,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Local per-shard VMEM-fused 4-tap causal `Conv1D + SiLU + Q/K L2Norm` with custom VJP."""
  b_loc, s_loc, c_dim = x_l.shape
  fwd_ts = _choose_tile_s(s_loc, _TILE_S)
  bwd_ts = _choose_tile_s(s_loc, _BWD_TILE_S)
  kd = num_k_heads * head_k_dim
  vd = num_v_heads * head_v_dim

  @jax.custom_vjp
  def _op(x_in: jax.Array, halo_in: jax.Array, w_in: jax.Array):
    num_tiles = s_loc // fwd_ts
    halos_4d = _build_tile_halos(x_in, halo_in, tile_s=fwd_ts)
    w_pad = jnp.pad(w_in[:, 0, :].astype(jnp.float32), ((0, 8 - _CONV_W), (0, 0)))
    q_4d, k_4d, v_4d = _gdn_conv_pallas_fwd(
        x_in.reshape(b_loc, num_tiles, fwd_ts, c_dim),
        halos_4d,
        w_pad,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        eps=eps,
        tile_s=fwd_ts,
        interpret=interpret,
    )
    return (
        q_4d.reshape(b_loc, s_loc, num_k_heads, head_k_dim),
        k_4d.reshape(b_loc, s_loc, num_k_heads, head_k_dim),
        v_4d.reshape(b_loc, s_loc, num_v_heads, head_v_dim),
    )

  def _fwd(x_in: jax.Array, halo_in: jax.Array, w_in: jax.Array):
    return _op(x_in, halo_in, w_in), (x_in, halo_in, w_in)

  def _bwd(res, cotangents):
    x_in, halo_in, w_in = res
    dq, dk, dv = cotangents
    num_tiles = s_loc // bwd_ts
    halos_4d = _build_tile_halos(x_in, halo_in, tile_s=bwd_ts)
    w_pad = jnp.pad(w_in[:, 0, :].astype(jnp.float32), ((0, 8 - _CONV_W), (0, 0)))
    dx_4d, dhalos_4d, dw_pad = _gdn_conv_pallas_bwd(
        x_in.reshape(b_loc, num_tiles, bwd_ts, c_dim),
        halos_4d,
        w_pad,
        dq.reshape(b_loc, num_tiles, bwd_ts, kd).astype(x_in.dtype),
        dk.reshape(b_loc, num_tiles, bwd_ts, kd).astype(x_in.dtype),
        dv.reshape(b_loc, num_tiles, bwd_ts, vd).astype(x_in.dtype),
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        eps=eps,
        tile_s=bwd_ts,
        interpret=interpret,
    )
    dhalos_3 = dhalos_4d[:, :, _HALO_PAD - 3 :, :]  # [b, num_tiles, 3, c]
    d_left_halo = dhalos_3[:, 0, :, :].astype(halo_in.dtype)
    if num_tiles > 1:
      tail_add = jnp.pad(dhalos_3[:, 1:, :, :], ((0, 0), (0, 1), (bwd_ts - 3, 0), (0, 0)))
      dx_total = (dx_4d + tail_add).reshape(b_loc, s_loc, c_dim).astype(x_in.dtype)
    else:
      dx_total = dx_4d.reshape(b_loc, s_loc, c_dim).astype(x_in.dtype)
    dw = dw_pad[:_CONV_W, None, :].astype(w_in.dtype)
    return dx_total, d_left_halo, dw

  _op.defvjp(_fwd, _bwd)
  return _op(x_l, left_halo, conv_kernel)
