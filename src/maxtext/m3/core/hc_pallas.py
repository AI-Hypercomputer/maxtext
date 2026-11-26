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

"""VMEM-fused Pallas Mosaic TPU kernels for Qwen3.8 4-stream `HyperConnection` (`Qwen3_8GatedResidual`).

Eliminates HBM materialization of `[T, hc_count * emb_dim]` (`335.5 MB` at `64K` per sublayer)
`normed` and `w_up` intermediates across Forward, Remat, and Backward passes while keeping
every minor tensor dimension `128`-lane aligned, per-kernel VMEM `< 36 MiB`, and backward MXU
matmuls in prescaled `bfloat16` (`8x` faster than `float32` MXU on TPU7x).
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
_FWD_TILE_T = 256
_BWD_TILE_T = 128
_DW_TILE_T = 512
_COMBINE_TILE_T = 512
_MAX_DW_SUB_C = 1280
_VMEM_LIMIT_BYTES = 58 * 1024 * 1024
_HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)


def _align_up(val: int, mult: int) -> int:
  return max(mult, ((int(val) + mult - 1) // mult) * mult)


def _choose_sub_c(emb_dim: int) -> int:
  """Selects largest `128`-aligned channel divisor `<= _MAX_DW_SUB_C` so `dw` fits in `< 4 MiB` VMEM."""
  if emb_dim <= _MAX_DW_SUB_C:
    return emb_dim
  for cand in range(_MAX_DW_SUB_C, LANES - 1, -LANES):
    if emb_dim % cand == 0:
      return cand
  return LANES


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


def can_use_hc_pallas(
    *,
    emb_dim: int,
    hc_count: int,
    hc_lowrank: int,
    is_tpu: bool,
    dtype: Any = jnp.bfloat16,
    interpret: bool = False,
) -> bool:
  """Returns True when shapes fit Pallas TPU lane/sublane and VMEM invariants."""
  if not (is_tpu or interpret):
    return False
  dt = jnp.dtype(dtype)
  if dt not in (jnp.bfloat16, jnp.float32):
    return False
  if emb_dim % LANES != 0 or hc_count < 1 or hc_count > 8:
    return False
  hc_dim = hc_count * emb_dim
  max_hc_dim = 16384 if dt == jnp.bfloat16 else 4096
  if hc_dim > max_hc_dim or hc_lowrank < 8 or hc_lowrank % 8 != 0 or hc_lowrank > 1024:
    return False
  return True


def _choose_tile_t(tokens: int, preferred: int) -> int:
  for cand in (preferred, 256, 128, 64, 32, 8):
    if 8 <= cand <= min(preferred, tokens):
      return cand
  return 8


def _prescaled_dot(
    lhs: jax.Array,
    rhs_f32: jax.Array,
    dim_nums: Any,
    *,
    x_dt: Any,
) -> jax.Array:
  """Runs `(lhs, rhs_f32)` on native `bf16` MXU with per-tile `amax` prescaling to preserve dynamic range."""
  if x_dt == jnp.bfloat16:
    amax = jnp.max(jnp.abs(rhs_f32)) + jnp.float32(1e-30)
    rhs_bf16 = (rhs_f32 / amax).astype(jnp.bfloat16)
    return lax.dot_general(lhs.astype(jnp.bfloat16), rhs_bf16, dim_nums, preferred_element_type=jnp.float32) * amax
  return lax.dot_general(lhs.astype(jnp.float32), rhs_f32, dim_nums, preferred_element_type=jnp.float32)


def _hc_core_math_streamed(
    x_tile: jax.Array,
    scale_2d: jax.Array,
    w_di_pad: jax.Array,
    w_up_pad: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    lr_pad: int,
    eps: float,
    has_inject: bool,
) -> tuple[jax.Array, jax.Array]:
  """Stream-looped `128`-lane-aligned VMEM forward math avoiding `[tile_t, hc_dim]` VREG temporaries."""
  x_dt = x_tile.dtype
  w_di_dt = w_di_pad.astype(x_dt)
  w_up_dt = w_up_pad.astype(x_dt)
  scale_1d = scale_2d[0].astype(jnp.float32)
  di_cols = w_di_pad.shape[1]

  rstds = []
  raw_di = jnp.zeros((x_tile.shape[0], di_cols), dtype=jnp.float32)
  for s in range(hc_count):
    x_s = x_tile[:, s * emb_dim : (s + 1) * emb_dim].astype(jnp.float32)
    rstd_s = lax.rsqrt(jnp.mean(x_s * x_s, axis=-1, keepdims=True) + jnp.float32(eps))
    rstds.append(rstd_s)
    sc_s = 1.0 + scale_1d[s * emb_dim : (s + 1) * emb_dim]
    normed_s = (x_s * rstd_s * sc_s).astype(x_dt)
    raw_di = raw_di + lax.dot_general(
        normed_s,
        w_di_dt[s * emb_dim : (s + 1) * emb_dim, :],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

  w_down_pad = jax.nn.silu(raw_di[:, :lr_pad] / jnp.float32(hc_count)).astype(x_dt)

  acc = jnp.zeros((x_tile.shape[0], emb_dim), dtype=jnp.float32)
  for s in range(hc_count):
    x_s = x_tile[:, s * emb_dim : (s + 1) * emb_dim].astype(jnp.float32)
    sc_s = 1.0 + scale_1d[s * emb_dim : (s + 1) * emb_dim]
    normed_s = (x_s * rstds[s] * sc_s).astype(x_dt)
    raw_up_s = lax.dot_general(
        w_down_pad,
        w_up_dt[:, s * emb_dim : (s + 1) * emb_dim],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    u_s = jax.nn.sigmoid(raw_up_s).astype(x_dt).astype(jnp.float32)
    acc = acc + u_s * normed_s.astype(jnp.float32)

  mixed = (acc / jnp.float32(hc_count)).astype(x_dt)
  if has_inject:
    raw_inj = raw_di[:, lr_pad : lr_pad + LANES]
    inject_pad = (2.0 * jax.nn.sigmoid(raw_inj / jnp.float32(hc_count))).astype(x_dt)
  else:
    inject_pad = jnp.zeros((x_tile.shape[0], LANES), dtype=x_dt)
  return mixed, inject_pad


@functools.partial(
    jax.jit,
    static_argnames=("emb_dim", "hc_count", "lr_pad", "eps", "has_inject", "tile_t", "interpret"),
)
def _hc_before_block_pallas_fwd(
    x_pad: jax.Array,
    scale_2d: jax.Array,
    w_di_pad: jax.Array,
    w_up_pad: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    lr_pad: int,
    eps: float,
    has_inject: bool,
    tile_t: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
  """Forward Pallas pipeline streaming `x_pad` with single-buffered stationary weights."""
  t_pad, hc_dim = x_pad.shape
  di_cols = w_di_pad.shape[1]

  def body(step: pltpu.PipelineStep, x_ref, sc_ref, wdi_ref, wup_ref, mix_ref, inj_ref):
    del step
    mixed, inject_pad = _hc_core_math_streamed(
        x_ref[...],
        sc_ref[...],
        wdi_ref[...],
        wup_ref[...],
        emb_dim=emb_dim,
        hc_count=hc_count,
        lr_pad=lr_pad,
        eps=eps,
        has_inject=has_inject,
    )
    mix_ref[...] = mixed
    inj_ref[...] = inject_pad

  spec_x = pl.BlockSpec((tile_t, hc_dim), lambda i: (i, 0))
  spec_sc = pl.BlockSpec((8, hc_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_wdi = pl.BlockSpec((hc_dim, di_cols), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_wup = pl.BlockSpec((lr_pad, hc_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_mix = pl.BlockSpec((tile_t, emb_dim), lambda i: (i, 0))
  spec_inj = pl.BlockSpec((tile_t, LANES), lambda i: (i, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(t_pad // tile_t,),
      in_specs=(spec_x, spec_sc, spec_wdi, spec_wup),
      out_specs=(spec_mix, spec_inj),
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  flops = int(2 * t_pad * hc_dim * (di_cols + lr_pad) + 8 * t_pad * hc_dim)
  bytes_acc = int(t_pad * (hc_dim + emb_dim) * x_pad.dtype.itemsize + (hc_dim * di_cols + lr_pad * hc_dim) * 2)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, emb_dim), x_pad.dtype),
            jax.ShapeDtypeStruct((t_pad, LANES), x_pad.dtype),
        ),
        in_specs=(_HBM_SPEC,) * 4,
        out_specs=(_HBM_SPEC,) * 2,
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=t_pad * hc_dim, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"hc_before_block_fwd_tm{tile_t}",
    )(x_pad, scale_2d, w_di_pad, w_up_pad)


@functools.partial(
    jax.jit,
    static_argnames=("emb_dim", "hc_count", "lr_pad", "eps", "has_inject", "tile_t", "interpret"),
)
def _hc_before_block_pallas_bwd_dx(
    x_pad: jax.Array,
    scale_2d: jax.Array,
    w_di_pad: jax.Array,
    w_up_pad: jax.Array,
    dmixed_pad: jax.Array,
    dinject_pad: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    lr_pad: int,
    eps: float,
    has_inject: bool,
    tile_t: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Pass 1 Backward Pallas kernel: analytical `bf16`-MXU `(dx, dscale, w_down_pad, draw_di, rstd_rep)`."""
  t_pad, hc_dim = x_pad.shape
  di_cols = w_di_pad.shape[1]
  x_dt = x_pad.dtype
  rstd_cols = hc_count * LANES

  def body(
      step: pltpu.PipelineStep,
      x_ref,
      sc_ref,
      wdi_ref,
      wup_ref,
      dmix_ref,
      dinj_ref,
      dx_ref,
      dsc_ref,
      wdown_ref,
      drawdi_ref,
      rstd_ref,
  ):
    tid = step.index[0]
    x_t = x_ref[...]
    sc_1d = sc_ref[0].astype(jnp.float32)
    wdi_dt = wdi_ref[...].astype(x_dt)
    wup_dt = wup_ref[...].astype(x_dt)
    d_acc = dmix_ref[...].astype(jnp.float32) / jnp.float32(hc_count)
    inv_hc = jnp.float32(1.0 / hc_count)

    # 1. Per-stream RMSNorm + `raw_di` (`[tile_t, emb_dim]` VREGs, zero cross-lane rstd packing).
    rstds = []
    rstd_blocks = []
    raw_di = jnp.zeros((tile_t, di_cols), dtype=jnp.float32)
    for s in range(hc_count):
      x_s = x_t[:, s * emb_dim : (s + 1) * emb_dim].astype(jnp.float32)
      rstd_s = lax.rsqrt(jnp.mean(x_s * x_s, axis=-1, keepdims=True) + jnp.float32(eps))
      rstds.append(rstd_s)
      rstd_blocks.append(jnp.broadcast_to(rstd_s, (tile_t, LANES)))
      sc_s = 1.0 + sc_1d[s * emb_dim : (s + 1) * emb_dim]
      normed_s = (x_s * rstd_s * sc_s).astype(x_dt)
      raw_di = raw_di + lax.dot_general(
          normed_s,
          wdi_dt[s * emb_dim : (s + 1) * emb_dim, :],
          (((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    rstd_ref[...] = jnp.concatenate(rstd_blocks, axis=-1)

    # 2. Low-rank forward + analytical derivative factors (`[tile_t, lr_pad]` = `192 KiB`).
    z_dn = raw_di[:, :lr_pad] * inv_hc
    sig_dn = jax.nn.sigmoid(z_dn)
    w_down_pad = (z_dn * sig_dn).astype(x_dt)
    wdown_ref[...] = w_down_pad

    # 3. Single `w_down @ w_up` pass saving `u_s` (`2.5 MiB` total) while accumulating `dw_down_acc` on `bf16` MXU!
    u_list = []
    dw_down_acc = jnp.zeros((tile_t, lr_pad), dtype=jnp.float32)
    for s in range(hc_count):
      x_s = x_t[:, s * emb_dim : (s + 1) * emb_dim].astype(jnp.float32)
      sc_s = 1.0 + sc_1d[s * emb_dim : (s + 1) * emb_dim]
      normed_s_f32 = (x_s * rstds[s] * sc_s).astype(x_dt).astype(jnp.float32)
      wup_s = wup_dt[:, s * emb_dim : (s + 1) * emb_dim]
      ru_s = lax.dot_general(w_down_pad, wup_s, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
      sig_up = jax.nn.sigmoid(ru_s)
      u_s = sig_up.astype(x_dt).astype(jnp.float32)
      u_list.append(u_s)
      du_s = (d_acc * normed_s_f32).astype(x_dt).astype(jnp.float32)
      dru_s = du_s * sig_up * (1.0 - sig_up)
      # `dru_s [tile_t, emb_dim] @ wup_s.T [emb_dim, lr_pad]` on prescaled `bf16` MXU:
      dwdn_s = _prescaled_dot(wup_s, dru_s, (((1,), (1,)), ((), ())), x_dt=x_dt).T
      dw_down_acc = dw_down_acc + dwdn_s

    dw_dn_cast = dw_down_acc.astype(x_dt).astype(jnp.float32)
    draw_dn = dw_dn_cast * sig_dn * (1.0 + z_dn * (1.0 - sig_dn)) * inv_hc
    if has_inject:
      z_inj = raw_di[:, lr_pad : lr_pad + LANES] * inv_hc
      sig_inj = jax.nn.sigmoid(z_inj)
      dinj_f32 = dinj_ref[...].astype(x_dt).astype(jnp.float32)
      draw_inj = dinj_f32 * (2.0 * inv_hc) * sig_inj * (1.0 - sig_inj)
      draw_di_f32 = jnp.concatenate([draw_dn, draw_inj], axis=-1)
    else:
      draw_di_f32 = draw_dn
    drawdi_ref[...] = draw_di_f32

    # 4. Analytical `(dx_s, dsc_s)` reusing `rstds[s]` and `u_list[s]` with prescaled `bf16` MXU!
    dx_streams = []
    dsc_streams = []
    for s in range(hc_count):
      x_s = x_t[:, s * emb_dim : (s + 1) * emb_dim].astype(jnp.float32)
      rstd_s = rstds[s]
      sc_raw = sc_1d[s * emb_dim : (s + 1) * emb_dim]
      sc_s = 1.0 + sc_raw
      wdi_s = wdi_dt[s * emb_dim : (s + 1) * emb_dim, :]
      # `draw_di_f32 [tile_t, di_cols] @ wdi_s.T [di_cols, emb_dim]` on prescaled `bf16` MXU:
      dnorm_di = _prescaled_dot(wdi_s, draw_di_f32, (((1,), (1,)), ((), ())), x_dt=x_dt).T
      dnorm_s = (dnorm_di + d_acc * u_list[s]).astype(x_dt).astype(jnp.float32)
      dsc_s = jnp.sum(dnorm_s * (x_s * rstd_s), axis=0)
      dy_s = dnorm_s * sc_s
      dx_s = rstd_s * (dy_s - (rstd_s * rstd_s * jnp.mean(dy_s * x_s, axis=-1, keepdims=True)) * x_s)
      dx_streams.append(dx_s.astype(x_dt))
      dsc_streams.append(jnp.broadcast_to(dsc_s[None, :], (8, emb_dim)))

    dx_ref[...] = jnp.concatenate(dx_streams, axis=-1)
    dsc_full = jnp.concatenate(dsc_streams, axis=-1)

    @pl.when(tid == 0)
    def _init_dsc():
      dsc_ref[...] = jnp.zeros_like(dsc_ref)

    dsc_ref[...] += dsc_full

  spec_x = pl.BlockSpec((tile_t, hc_dim), lambda i: (i, 0))
  spec_sc_in = pl.BlockSpec((8, hc_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_wdi_in = pl.BlockSpec((hc_dim, di_cols), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_wup_in = pl.BlockSpec((lr_pad, hc_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_mix = pl.BlockSpec((tile_t, emb_dim), lambda i: (i, 0))
  spec_inj = pl.BlockSpec((tile_t, LANES), lambda i: (i, 0))

  spec_sc_out = pl.BlockSpec((8, hc_dim), lambda i: (0, 0))
  spec_wdn_out = pl.BlockSpec((tile_t, lr_pad), lambda i: (i, 0))
  spec_drdi_out = pl.BlockSpec((tile_t, di_cols), lambda i: (i, 0))
  spec_rstd_out = pl.BlockSpec((tile_t, rstd_cols), lambda i: (i, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(t_pad // tile_t,),
      in_specs=(spec_x, spec_sc_in, spec_wdi_in, spec_wup_in, spec_mix, spec_inj),
      out_specs=(spec_x, spec_sc_out, spec_wdn_out, spec_drdi_out, spec_rstd_out),
      dimension_semantics=(pltpu.ARBITRARY,),
      _explicit_indices=True,
  )
  flops = int(4 * t_pad * hc_dim * (di_cols + lr_pad))
  bytes_acc = int(t_pad * (2 * hc_dim + emb_dim) * x_dt.itemsize + (hc_dim * di_cols + lr_pad * hc_dim) * 2)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, hc_dim), x_dt),
            jax.ShapeDtypeStruct((8, hc_dim), jnp.float32),
            jax.ShapeDtypeStruct((t_pad, lr_pad), x_dt),
            jax.ShapeDtypeStruct((t_pad, di_cols), jnp.float32),
            jax.ShapeDtypeStruct((t_pad, rstd_cols), jnp.float32),
        ),
        in_specs=(_HBM_SPEC,) * 6,
        out_specs=(_HBM_SPEC,) * 5,
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=2 * t_pad * hc_dim, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"hc_before_block_bwd_dx_tm{tile_t}",
    )(x_pad, scale_2d, w_di_pad, w_up_pad, dmixed_pad, dinject_pad)


@functools.partial(
    jax.jit,
    static_argnames=("hc_count", "num_sub", "sub_c", "lr_pad", "tile_t", "interpret"),
)
def _hc_before_block_pallas_bwd_dw(
    x_pad: jax.Array,
    sc_2d: jax.Array,
    w_up_pad: jax.Array,
    dmix_pad: jax.Array,
    w_down_pad: jax.Array,
    draw_di_pad: jax.Array,
    rstd_rep: jax.Array,
    *,
    hc_count: int,
    num_sub: int,
    sub_c: int,
    lr_pad: int,
    tile_t: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
  """Pass 2 Backward Pallas kernel: 2D channel-tiled `bf16`-MXU `(dw_di, dw_up)` in `~20 MiB` VMEM."""
  t_pad, hc_dim = x_pad.shape
  di_cols = draw_di_pad.shape[1]
  x_dt = x_pad.dtype
  num_t = t_pad // tile_t

  def body(
      step: pltpu.PipelineStep,
      x_ref,
      sc_ref,
      wup_ref,
      dmix_ref,
      wdn_ref,
      drdi_ref,
      rstd_ref,
      dwdi_ref,
      dwup_ref,
  ):
    t_i = step.index[2]

    x_sub = x_ref[...].astype(jnp.float32)  # [tile_t, sub_c]
    rstd_s = jnp.tile(rstd_ref[...], (1, sub_c // LANES))  # [tile_t, sub_c], zero cross-lane ops!
    sc_sub = 1.0 + sc_ref[0, :].astype(jnp.float32)  # [sub_c]
    norm_sub = (x_sub * rstd_s * sc_sub).astype(x_dt)  # [tile_t, sub_c]
    norm_sub_f32 = norm_sub.astype(jnp.float32)

    drdi_t = drdi_ref[...]  # [tile_t, di_cols] f32
    dwdi_tile = _prescaled_dot(norm_sub, drdi_t, (((0,), (0,)), ((), ())), x_dt=x_dt)

    wdn_t = wdn_ref[...]  # [tile_t, lr_pad] x_dt
    wup_sub = wup_ref[...].astype(x_dt)  # [lr_pad, sub_c]
    d_acc_sub = dmix_ref[...].astype(jnp.float32) / jnp.float32(hc_count)

    ru = lax.dot_general(wdn_t, wup_sub, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    sig_u = jax.nn.sigmoid(ru)
    du_sub = (d_acc_sub * norm_sub_f32).astype(x_dt).astype(jnp.float32)
    dru_sub = du_sub * sig_u * (1.0 - sig_u)
    dwup_tile = _prescaled_dot(wdn_t, dru_sub, (((0,), (0,)), ((), ())), x_dt=x_dt)

    @pl.when(t_i == 0)
    def _init_dw():
      dwdi_ref[...] = jnp.zeros_like(dwdi_ref)
      dwup_ref[...] = jnp.zeros_like(dwup_ref)

    dwdi_ref[...] = dwdi_ref[...] + dwdi_tile.astype(jnp.float32)
    dwup_ref[...] = dwup_ref[...] + dwup_tile.astype(jnp.float32)

  spec_x = pl.BlockSpec((tile_t, sub_c), lambda s, sub, t: (t, s * num_sub + sub))
  spec_sc = pl.BlockSpec((8, sub_c), lambda s, sub, t: (0, s * num_sub + sub), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_wup = pl.BlockSpec(
      (lr_pad, sub_c), lambda s, sub, t: (0, s * num_sub + sub), pipeline_mode=pl.Buffered(buffer_count=1)
  )
  spec_dmix = pl.BlockSpec((tile_t, sub_c), lambda s, sub, t: (t, sub))
  spec_wdn = pl.BlockSpec((tile_t, lr_pad), lambda s, sub, t: (t, 0))
  spec_drdi = pl.BlockSpec((tile_t, di_cols), lambda s, sub, t: (t, 0))
  spec_rstd = pl.BlockSpec((tile_t, LANES), lambda s, sub, t: (t, s))

  spec_dwdi = pl.BlockSpec((sub_c, di_cols), lambda s, sub, t: (s * num_sub + sub, 0))
  spec_dwup = pl.BlockSpec((lr_pad, sub_c), lambda s, sub, t: (0, s * num_sub + sub))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(hc_count, num_sub, num_t),
      in_specs=(spec_x, spec_sc, spec_wup, spec_dmix, spec_wdn, spec_drdi, spec_rstd),
      out_specs=(spec_dwdi, spec_dwup),
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
      _explicit_indices=True,
  )
  flops = int(4 * t_pad * hc_dim * (di_cols + lr_pad))
  bytes_acc = int(t_pad * hc_dim * x_dt.itemsize + (hc_dim * di_cols + lr_pad * hc_dim) * 4)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((hc_dim, di_cols), jnp.float32),
            jax.ShapeDtypeStruct((lr_pad, hc_dim), jnp.float32),
        ),
        in_specs=(_HBM_SPEC,) * 7,
        out_specs=(_HBM_SPEC,) * 2,
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=t_pad * hc_dim, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"hc_before_block_bwd_dw_tm{tile_t}_c{sub_c}",
    )(x_pad, sc_2d, w_up_pad, dmix_pad, w_down_pad, draw_di_pad, rstd_rep)


def hc_before_block_local(
    hyper_input: jax.Array,
    scale: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    w_inject: jax.Array | None,
    *,
    emb_dim: int,
    hc_count: int,
    hc_lowrank: int,
    eps: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Local per-shard VMEM-fused `before_block` / `mix_input` with custom VJP."""
  orig_shape = hyper_input.shape
  hc_dim = hc_count * emb_dim
  tokens = hyper_input.size // hc_dim
  has_inject = w_inject is not None
  lr_pad = _align_up(hc_lowrank, LANES)
  sub_c = _choose_sub_c(emb_dim)
  num_sub = emb_dim // sub_c

  try:
    is_pre_v7 = not interpret and pltpu.is_tpu_device() and pltpu.get_tpu_info().generation < 7
  except (RuntimeError, ValueError, AttributeError, TypeError):
    is_pre_v7 = False
  eff_bwd_tile_t = 64 if (is_pre_v7 and hc_dim >= 8192) else _BWD_TILE_T

  fwd_tile = _choose_tile_t(tokens, _FWD_TILE_T)
  bwd_tile = _choose_tile_t(tokens, eff_bwd_tile_t)
  dw_tile = _choose_tile_t(tokens, _DW_TILE_T)
  tile_lcm = max(fwd_tile, bwd_tile, dw_tile)
  t_pad = _align_up(tokens, tile_lcm)
  pad_rows = t_pad - tokens
  lr_extra = lr_pad - hc_lowrank

  def _pack_weights(sc: jax.Array, wd: jax.Array, wu: jax.Array, wi: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    sc_2d = jnp.broadcast_to(sc.astype(jnp.float32)[None, :], (8, hc_dim))
    wd_p = jnp.pad(wd, ((0, 0), (0, lr_extra))) if lr_extra > 0 else wd
    if has_inject:
      wi_p = jnp.pad(wi, ((0, 0), (0, LANES - hc_count)))
      w_di = jnp.concatenate([wd_p, wi_p], axis=-1)
    else:
      w_di = wd_p
    wu_p = jnp.pad(wu, ((0, lr_extra), (0, 0))) if lr_extra > 0 else wu
    return sc_2d, w_di, wu_p

  w_inj_arg = w_inject if w_inject is not None else jnp.zeros((hc_dim, 0), dtype=w_down.dtype)

  @jax.custom_vjp
  def _op(x_2d: jax.Array, sc: jax.Array, wd: jax.Array, wu: jax.Array, wi: jax.Array):
    sc_2d, w_di, wu_p = _pack_weights(sc, wd, wu, wi)
    x_p = jnp.pad(x_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else x_2d
    mix_p, inj_128 = _hc_before_block_pallas_fwd(
        x_p,
        sc_2d,
        w_di,
        wu_p,
        emb_dim=emb_dim,
        hc_count=hc_count,
        lr_pad=lr_pad,
        eps=eps,
        has_inject=has_inject,
        tile_t=fwd_tile,
        interpret=interpret,
    )
    return (mix_p[:tokens] if pad_rows > 0 else mix_p), inj_128[:tokens, :hc_count]

  def _fwd(x_2d: jax.Array, sc: jax.Array, wd: jax.Array, wu: jax.Array, wi: jax.Array):
    out = _op(x_2d, sc, wd, wu, wi)
    return out, (x_2d, sc, wd, wu, wi)

  def _bwd(res, cotangents):
    x_2d, sc, wd, wu, wi = res
    dmix, dinj = cotangents
    sc_2d, w_di, wu_p = _pack_weights(sc, wd, wu, wi)
    x_p = jnp.pad(x_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else x_2d
    dmix_p = jnp.pad(dmix, ((0, pad_rows), (0, 0))) if pad_rows > 0 else dmix
    dinj_p = jnp.pad(dinj, ((0, pad_rows), (0, LANES - hc_count)))
    dx_p, dsc_2d, w_dn_p, drdi_p, rstd_rep = _hc_before_block_pallas_bwd_dx(
        x_p,
        sc_2d,
        w_di,
        wu_p,
        dmix_p,
        dinj_p,
        emb_dim=emb_dim,
        hc_count=hc_count,
        lr_pad=lr_pad,
        eps=eps,
        has_inject=has_inject,
        tile_t=bwd_tile,
        interpret=interpret,
    )
    dw_di, dw_up_p = _hc_before_block_pallas_bwd_dw(
        x_p,
        sc_2d,
        wu_p,
        dmix_p,
        w_dn_p,
        drdi_p,
        rstd_rep,
        hc_count=hc_count,
        num_sub=num_sub,
        sub_c=sub_c,
        lr_pad=lr_pad,
        tile_t=dw_tile,
        interpret=interpret,
    )
    dx = dx_p[:tokens] if pad_rows > 0 else dx_p
    dsc = dsc_2d[0].astype(sc.dtype)
    dwd = dw_di[:, :hc_lowrank].astype(wd.dtype)
    if has_inject:
      dwi = dw_di[:, lr_pad : lr_pad + hc_count].astype(wi.dtype)
    else:
      dwi = jnp.zeros_like(wi)
    return dx, dsc, dwd, dw_up_p[:hc_lowrank, :].astype(wu.dtype), dwi

  _op.defvjp(_fwd, _bwd)
  mix_2d, inj_2d = _op(jnp.reshape(hyper_input, (tokens, hc_dim)), scale, w_down, w_up, w_inj_arg)
  return jnp.reshape(mix_2d, (*orig_shape[:-1], emb_dim)), jnp.reshape(inj_2d, (*orig_shape[:-1], hc_count))


# ==============================================================================
# VMEM-fused 4-stream `combine`: hyper_input + block_output * injection_weights
# ==============================================================================


@functools.partial(jax.jit, static_argnames=("emb_dim", "hc_count", "tile_t", "interpret"))
def _hc_combine_pallas_fwd(
    x_pad: jax.Array,
    blk_pad: jax.Array,
    inj_rep: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    tile_t: int,
    interpret: bool,
) -> jax.Array:
  """Streams `hyper_input`, `block_output`, and `128`-lane-replicated `inject` with zero cross-lane reductions."""
  t_pad, hc_dim = x_pad.shape
  inj_cols = hc_count * LANES
  n_chunks = emb_dim // LANES

  def body(step: pltpu.PipelineStep, x_ref, b_ref, inj_ref, out_ref):
    del step
    x_t = x_ref[...]
    b_t = b_ref[...]
    inj_t = inj_ref[...]
    streams = []
    for s in range(hc_count):
      inj_s = jnp.tile(inj_t[:, s * LANES : (s + 1) * LANES], (1, n_chunks))
      streams.append((x_t[:, s * emb_dim : (s + 1) * emb_dim] + b_t * inj_s).astype(x_pad.dtype))
    out_ref[...] = jnp.concatenate(streams, axis=-1)

  spec_x = pl.BlockSpec((tile_t, hc_dim), lambda i: (i, 0))
  spec_b = pl.BlockSpec((tile_t, emb_dim), lambda i: (i, 0))
  spec_inj = pl.BlockSpec((tile_t, inj_cols), lambda i: (i, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(t_pad // tile_t,),
      in_specs=(spec_x, spec_b, spec_inj),
      out_specs=spec_x,
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  bytes_acc = int(t_pad * (2 * hc_dim + emb_dim) * x_pad.dtype.itemsize)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=jax.ShapeDtypeStruct((t_pad, hc_dim), x_pad.dtype),
        in_specs=(_HBM_SPEC,) * 3,
        out_specs=_HBM_SPEC,
        cost_estimate=pl.CostEstimate(flops=int(2 * t_pad * hc_dim), transcendentals=0, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"hc_combine_fwd_tm{tile_t}",
    )(x_pad, blk_pad, inj_rep)


@functools.partial(jax.jit, static_argnames=("emb_dim", "hc_count", "tile_t", "interpret"))
def _hc_combine_pallas_bwd(
    blk_pad: jax.Array,
    inj_rep: jax.Array,
    dout_pad: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    tile_t: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
  """Fuses `d_block_output` and lane-local `d_injection_weights` partial sums with zero cross-lane reductions."""
  t_pad, hc_dim = dout_pad.shape
  inj_cols = hc_count * LANES
  n_chunks = emb_dim // LANES

  def body(step: pltpu.PipelineStep, b_ref, inj_ref, dout_ref, db_ref, dinj_ref):
    del step
    b_f32 = b_ref[...].astype(jnp.float32)
    inj_t = inj_ref[...]
    dout_t = dout_ref[...]
    db_acc = jnp.zeros((tile_t, emb_dim), dtype=jnp.float32)
    dinj_blocks = []
    for s in range(hc_count):
      do_s = dout_t[:, s * emb_dim : (s + 1) * emb_dim]
      inj_s = jnp.tile(inj_t[:, s * LANES : (s + 1) * LANES], (1, n_chunks))
      db_acc = db_acc + (do_s * inj_s).astype(jnp.float32)
      prod_f32 = do_s.astype(jnp.float32) * b_f32
      # Lane-parallel reduction across `n_chunks` `128`-wide tiles (`[tile_t, 128]`), avoiding cross-lane sum!
      part_128 = prod_f32[:, :LANES]
      for k in range(1, n_chunks):
        part_128 = part_128 + prod_f32[:, k * LANES : (k + 1) * LANES]
      dinj_blocks.append(part_128)
    db_ref[...] = db_acc.astype(blk_pad.dtype)
    dinj_ref[...] = jnp.concatenate(dinj_blocks, axis=-1)

  spec_x = pl.BlockSpec((tile_t, hc_dim), lambda i: (i, 0))
  spec_b = pl.BlockSpec((tile_t, emb_dim), lambda i: (i, 0))
  spec_inj = pl.BlockSpec((tile_t, inj_cols), lambda i: (i, 0))

  kernel_main = pltpu.emit_pipeline(
      body,
      grid=(t_pad // tile_t,),
      in_specs=(spec_b, spec_inj, spec_x),
      out_specs=(spec_b, spec_inj),
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  bytes_acc = int(t_pad * (hc_dim + 2 * emb_dim) * dout_pad.dtype.itemsize)
  with _tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, emb_dim), blk_pad.dtype),
            jax.ShapeDtypeStruct((t_pad, inj_cols), jnp.float32),
        ),
        in_specs=(_HBM_SPEC,) * 3,
        out_specs=(_HBM_SPEC,) * 2,
        cost_estimate=pl.CostEstimate(flops=int(4 * t_pad * hc_dim), transcendentals=0, bytes_accessed=bytes_acc),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"hc_combine_bwd_tm{tile_t}",
    )(blk_pad, inj_rep, dout_pad)


def hc_combine_local(
    hyper_input: jax.Array,
    block_output: jax.Array,
    injection_weights: jax.Array,
    *,
    emb_dim: int,
    hc_count: int,
    interpret: bool = False,
) -> jax.Array:
  """Local per-shard VMEM-fused 4-stream `combine` with custom VJP."""
  orig_shape = hyper_input.shape
  x_dtype = hyper_input.dtype
  hc_dim = hc_count * emb_dim
  tokens = hyper_input.size // hc_dim
  tile_t = _choose_tile_t(tokens, _COMBINE_TILE_T)
  t_pad = _align_up(tokens, tile_t)
  pad_rows = t_pad - tokens

  @jax.custom_vjp
  def _op(x_2d: jax.Array, b_2d: jax.Array, inj_2d: jax.Array) -> jax.Array:
    x_p = jnp.pad(x_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else x_2d
    b_p = jnp.pad(b_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else b_2d
    i_p = jnp.pad(inj_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else inj_2d
    inj_rep = jnp.repeat(i_p.astype(b_2d.dtype), LANES, axis=1)
    out_p = _hc_combine_pallas_fwd(
        x_p,
        b_p,
        inj_rep,
        emb_dim=emb_dim,
        hc_count=hc_count,
        tile_t=tile_t,
        interpret=interpret,
    )
    return out_p[:tokens] if pad_rows > 0 else out_p

  def _fwd(x_2d: jax.Array, b_2d: jax.Array, inj_2d: jax.Array):
    return _op(x_2d, b_2d, inj_2d), (b_2d, inj_2d)

  def _bwd(res, dout_2d: jax.Array):
    b_2d, inj_2d = res
    dx_2d = dout_2d.astype(x_dtype)
    do_b = dout_2d.astype(b_2d.dtype)
    b_p = jnp.pad(b_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else b_2d
    i_p = jnp.pad(inj_2d, ((0, pad_rows), (0, 0))) if pad_rows > 0 else inj_2d
    inj_rep = jnp.repeat(i_p.astype(b_2d.dtype), LANES, axis=1)
    do_p = jnp.pad(do_b, ((0, pad_rows), (0, 0))) if pad_rows > 0 else do_b
    db_p, dinj_rep = _hc_combine_pallas_bwd(
        b_p,
        inj_rep,
        do_p,
        emb_dim=emb_dim,
        hc_count=hc_count,
        tile_t=tile_t,
        interpret=interpret,
    )
    dinj = jnp.sum(dinj_rep[:tokens].reshape(tokens, hc_count, LANES), axis=-1).astype(inj_2d.dtype)
    return (
        dx_2d,
        db_p[:tokens] if pad_rows > 0 else db_p,
        dinj,
    )

  _op.defvjp(_fwd, _bwd)
  out_2d = _op(
      jnp.reshape(hyper_input, (tokens, hc_dim)),
      jnp.reshape(block_output, (tokens, emb_dim)),
      jnp.reshape(injection_weights, (tokens, hc_count)),
  )
  return jnp.reshape(out_2d, orig_shape)
