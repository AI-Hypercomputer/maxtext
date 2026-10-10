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

"""Fused LatentMoE kernels for fine-grained MoE models (OLMo 3.5 / OLMoE3).

Provides two levels of fusion for LatentMoE (`d_latent = 768, d_ff = 768, E = 512`):
1. `fused_gate_up_gmm`: Concatenates `wi_0` and `wi_1` along `d_ff` into
   `[E, d_latent, 2 * d_ff]` so `wi_0` and `wi_1` execute as a single ragged GEMM
   (2 ragged GEMMs in fwd and 4 in bwd, instead of 3 in fwd and 6 in bwd).
2. `fused_latent_moe_pallas`: A Pallas Mosaic TPU kernel (for both TPU v4
   TPU v4 and TPU v7x) that fuses `wi_0 + wi_1 + SwiGLU + wo` into
   a single kernel invocation, keeping the intermediate activation `[tile_m, d_ff]`
   entirely in VMEM without materializing `[M, d_ff]` to HBM.
"""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels import megablox as mblx
from maxtext.kernels.megablox import backend as mblx_backend


def fused_gate_up_gmm(
    inputs: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    *,
    tile_m: int = 256,
    use_gmm_v2: bool = False,
    dlhs_transpose_in_kernel: bool = False,
) -> jax.Array:
  """Executes LatentMoE with `wi_0` and `wi_1` fused into a single `[E, D, 2*F]` GMM."""
  dtype = inputs.dtype
  d_ff = wi_0.shape[-1]
  wi_01 = jnp.concatenate([wi_0, wi_1], axis=-1)
  gs_i32 = group_sizes.astype(jnp.int32)

  if use_gmm_v2:
    tiling_wi = (tile_m, 512, 2 * d_ff, tile_m, 2 * d_ff, 512, 512, tile_m, 2 * d_ff)
    tiling_wo = (tile_m, 512, d_ff, tile_m, d_ff, 512, 512, tile_m, d_ff)
    gate_up = mblx.gmm(
        lhs=inputs,
        rhs=wi_01,
        tiling=tiling_wi,
        group_sizes=gs_i32,
        preferred_element_type=dtype,
        use_tokamax_backend=True,
        use_gmm_v2=True,
        dlhs_transpose_in_kernel=dlhs_transpose_in_kernel,
    )
    g0, g1 = jnp.split(gate_up, 2, axis=-1)
    act = (jax.nn.silu(g0.astype(jnp.float32)) * g1.astype(jnp.float32)).astype(dtype)
    return mblx.gmm(
        lhs=act,
        rhs=wo,
        tiling=tiling_wo,
        group_sizes=gs_i32,
        preferred_element_type=dtype,
        use_tokamax_backend=True,
        use_gmm_v2=True,
        dlhs_transpose_in_kernel=dlhs_transpose_in_kernel,
    )

  gate_up = mblx.tokamax_ragged_dot_v1(
      lhs=inputs,
      rhs=wi_01,
      group_sizes=gs_i32,
      preferred_element_type=dtype,
      tile_m=tile_m,
  )
  g0, g1 = jnp.split(gate_up, 2, axis=-1)
  act = (jax.nn.silu(g0.astype(jnp.float32)) * g1.astype(jnp.float32)).astype(dtype)
  return mblx.tokamax_ragged_dot_v1(
      lhs=act,
      rhs=wo,
      group_sizes=gs_i32,
      preferred_element_type=dtype,
      tile_m=tile_m,
  )


def _fused_latent_moe_fwd_kernel(
    x_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    out_ref,
    *,
    num_ff_tiles: int,
    cast_rhs_f32: bool = False,
):
  """Pallas kernel: out[e, bm, D] = sum_f (silu(x @ wi0) * (x @ wi1)) @ wo in VMEM."""
  x = x_ref[0, :, :]  # [bm, D]
  x_mm = x.astype(jnp.float32) if cast_rhs_f32 else x
  acc = jnp.zeros((x.shape[0], wo_ref.shape[-1]), dtype=jnp.float32)

  def body(f_idx, acc_in):
    w0 = wi0_ref[0, f_idx, :, :]  # [D, bf]
    w1 = wi1_ref[0, f_idx, :, :]  # [D, bf]
    wo_t = wo_ref[0, f_idx, :, :]  # [bf, D]
    if cast_rhs_f32:
      w0 = w0.astype(jnp.float32)
      w1 = w1.astype(jnp.float32)
      wo_t = wo_t.astype(jnp.float32)
    h0 = jnp.dot(x_mm, w0, preferred_element_type=jnp.float32)
    h1 = jnp.dot(x_mm, w1, preferred_element_type=jnp.float32)
    act = jax.nn.silu(h0) * h1
    act_mm = act if cast_rhs_f32 else act.astype(x.dtype)
    return acc_in + jnp.dot(act_mm, wo_t, preferred_element_type=jnp.float32)

  acc = jax.lax.fori_loop(0, num_ff_tiles, body, acc)
  out_ref[0, :, :] = acc.astype(out_ref.dtype)


def _fused_latent_moe_fwd_v3_kernel(
    x_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    out_init_ref,
    out_ref,
    *,
    cast_rhs_f32: bool = False,
):
  """Memory-lean 3D-grid Pallas forward kernel for 16 MiB VMEM targets (TPU v3 and TPU v4)."""
  f_idx = pl.program_id(2)
  x = x_ref[0, :, :]
  w0 = wi0_ref[0, 0, :, :]
  w1 = wi1_ref[0, 0, :, :]
  wo_t = wo_ref[0, 0, :, :]
  if cast_rhs_f32:
    x = x.astype(jnp.float32)
    w0 = w0.astype(jnp.float32)
    w1 = w1.astype(jnp.float32)
    wo_t = wo_t.astype(jnp.float32)
  h0 = jnp.dot(x, w0, preferred_element_type=jnp.float32)
  h1 = jnp.dot(x, w1, preferred_element_type=jnp.float32)
  act = jax.nn.silu(h0) * h1
  act_mm = act if cast_rhs_f32 else act.astype(x_ref.dtype)
  part = jnp.dot(act_mm, wo_t, preferred_element_type=jnp.float32)
  prev = jnp.where(f_idx == 0, 0.0, out_init_ref[0, :, :].astype(jnp.float32))
  out_ref[0, :, :] = (prev + part).astype(out_ref.dtype)


def _fused_latent_moe_bwd_kernel(
    x_ref,
    dy_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    dx_init_ref,
    dx_ref,
    dwi0_ref,
    dwi1_ref,
    dwo_ref,
    *,
    num_m_tiles: int,
    cast_rhs_f32: bool = False,
):
  """Pallas backward kernel: recomputes SwiGLU in VMEM and accumulates dx, dwi0, dwi1, dwo."""
  w0 = wi0_ref[0, 0, :, :]  # [D, bf]
  w1 = wi1_ref[0, 0, :, :]  # [D, bf]
  wo_t = wo_ref[0, 0, :, :]  # [bf, D]
  w0_rhs = w0.astype(jnp.float32) if cast_rhs_f32 else w0
  w1_rhs = w1.astype(jnp.float32) if cast_rhs_f32 else w1
  wo_t_rhs = wo_t.astype(jnp.float32) if cast_rhs_f32 else wo_t
  dw0_acc = jnp.zeros_like(w0, dtype=jnp.float32)
  dw1_acc = jnp.zeros_like(w1, dtype=jnp.float32)
  dwo_acc = jnp.zeros_like(wo_t, dtype=jnp.float32)

  for m_idx in range(num_m_tiles):
    x = x_ref[0, m_idx, :, :]  # [bm, D]
    dy = dy_ref[0, m_idx, :, :]  # [bm, D]
    x_mm = x.astype(jnp.float32) if cast_rhs_f32 else x
    dy_mm = dy.astype(jnp.float32) if cast_rhs_f32 else dy
    h0 = jnp.dot(x_mm, w0_rhs, preferred_element_type=jnp.float32)
    h1 = jnp.dot(x_mm, w1_rhs, preferred_element_type=jnp.float32)
    sig0 = jax.nn.sigmoid(h0)
    silu0 = h0 * sig0
    act = (silu0 * h1) if cast_rhs_f32 else (silu0 * h1).astype(x.dtype)

    dwo_acc = dwo_acc + jnp.dot(act.T, dy_mm, preferred_element_type=jnp.float32)
    dact = jnp.dot(dy_mm, wo_t_rhs.T, preferred_element_type=jnp.float32)
    dh1 = (dact * silu0) if cast_rhs_f32 else (dact * silu0).astype(x.dtype)
    dsilu = dact * h1 * (sig0 * (1.0 + h0 * (1.0 - sig0)))
    dh0 = dsilu if cast_rhs_f32 else dsilu.astype(x.dtype)

    dw0_acc = dw0_acc + jnp.dot(x_mm.T, dh0, preferred_element_type=jnp.float32)
    dw1_acc = dw1_acc + jnp.dot(x_mm.T, dh1, preferred_element_type=jnp.float32)
    dx_part = jnp.dot(dh0, w0_rhs.T, preferred_element_type=jnp.float32) + jnp.dot(
        dh1, w1_rhs.T, preferred_element_type=jnp.float32
    )
    prev_dx = dx_init_ref[0, m_idx, :, :].astype(jnp.float32)
    dx_ref[0, m_idx, :, :] = (prev_dx + dx_part).astype(dx_ref.dtype)

  dwi0_ref[0, 0, :, :] = dw0_acc.astype(dwi0_ref.dtype)
  dwi1_ref[0, 0, :, :] = dw1_acc.astype(dwi1_ref.dtype)
  dwo_ref[0, 0, :, :] = dwo_acc.astype(dwo_ref.dtype)


def _fused_latent_moe_bwd_3d_kernel(
    x_ref,
    dy_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    dx_init_ref,
    dx_ref,
    dwi0_ref,
    dwi1_ref,
    dwo_ref,
    dw0_scratch,
    dw1_scratch,
    dwo_scratch,
    *,
    num_m_tiles: int,
    cast_rhs_f32: bool = False,
):
  """Memory-lean 3D-grid Pallas backward kernel with FP32 VMEM scratch accumulation."""
  f_idx = pl.program_id(1)
  m_idx = pl.program_id(2)

  @pl.when(m_idx == 0)
  def _zero_dw():
    dw0_scratch[...] = jnp.zeros_like(dw0_scratch)
    dw1_scratch[...] = jnp.zeros_like(dw1_scratch)
    dwo_scratch[...] = jnp.zeros_like(dwo_scratch)

  w0 = wi0_ref[0, 0, :, :]  # [D, bf]
  w1 = wi1_ref[0, 0, :, :]  # [D, bf]
  wo_t = wo_ref[0, 0, :, :]  # [bf, D]
  w0_rhs = w0.astype(jnp.float32) if cast_rhs_f32 else w0
  w1_rhs = w1.astype(jnp.float32) if cast_rhs_f32 else w1
  wo_t_rhs = wo_t.astype(jnp.float32) if cast_rhs_f32 else wo_t

  x = x_ref[0, 0, :, :]  # [bm, D]
  dy = dy_ref[0, 0, :, :]  # [bm, D]
  x_mm = x.astype(jnp.float32) if cast_rhs_f32 else x
  dy_mm = dy.astype(jnp.float32) if cast_rhs_f32 else dy
  h0 = jnp.dot(x_mm, w0_rhs, preferred_element_type=jnp.float32)
  h1 = jnp.dot(x_mm, w1_rhs, preferred_element_type=jnp.float32)
  sig0 = jax.nn.sigmoid(h0)
  silu0 = h0 * sig0
  act = (silu0 * h1) if cast_rhs_f32 else (silu0 * h1).astype(x.dtype)

  dwo_part = jnp.dot(act.T, dy_mm, preferred_element_type=jnp.float32)
  dact = jnp.dot(dy_mm, wo_t_rhs.T, preferred_element_type=jnp.float32)
  dh1 = (dact * silu0) if cast_rhs_f32 else (dact * silu0).astype(x.dtype)
  dsilu = dact * h1 * (sig0 * (1.0 + h0 * (1.0 - sig0)))
  dh0 = dsilu if cast_rhs_f32 else dsilu.astype(x.dtype)

  dw0_part = jnp.dot(x_mm.T, dh0, preferred_element_type=jnp.float32)
  dw1_part = jnp.dot(x_mm.T, dh1, preferred_element_type=jnp.float32)
  dx_part = jnp.dot(dh0, w0_rhs.T, preferred_element_type=jnp.float32) + jnp.dot(
      dh1, w1_rhs.T, preferred_element_type=jnp.float32
  )

  prev_dx = jnp.where(f_idx == 0, 0.0, dx_init_ref[0, 0, :, :].astype(jnp.float32))
  dx_ref[0, 0, :, :] = (prev_dx + dx_part).astype(dx_ref.dtype)

  dw0_scratch[...] += dw0_part
  dw1_scratch[...] += dw1_part
  dwo_scratch[...] += dwo_part

  @pl.when(m_idx == num_m_tiles - 1)
  def _store_dw():
    dwi0_ref[0, 0, :, :] = dw0_scratch[...].astype(dwi0_ref.dtype)
    dwi1_ref[0, 0, :, :] = dw1_scratch[...].astype(dwi1_ref.dtype)
    dwo_ref[0, 0, :, :] = dwo_scratch[...].astype(dwo_ref.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def fused_latent_moe_pallas(
    x_grouped: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    block_m: int = 256,
    block_ff: int = 256,
) -> jax.Array:
  """VMEM-fused `wi_0 + wi_1 + SwiGLU + wo` Pallas kernel for grouped LatentMoE inputs.

  Args:
    x_grouped: Routed tokens of shape `[E, M_per_expert, d_latent]` (or reshaped
      from `[M, d_latent]` when tiles align with `block_m`).
    wi_0: Gate weights `[E, d_latent, d_ff]`.
    wi_1: Up weights `[E, d_latent, d_ff]`.
    wo: Down weights `[E, d_ff, d_latent]`.
    block_m: Token tile size per expert (`128` or `256`).
    block_ff: Intermediate FFN tile size (`256` fits in 16 MiB VMEM on TPU v4;
      `256` or `768` fits in 64 MiB VMEM on TPU v7x).

  Returns:
    Output tokens of shape `[E, M_per_expert, d_latent]`.
  """
  e_dim, m_per_e, d_latent = x_grouped.shape
  d_ff = wi_0.shape[-1]
  num_m_tiles = m_per_e // block_m
  num_ff_tiles = d_ff // block_ff
  dev_kind = jax.devices()[0].device_kind
  cast_rhs_f32 = "v3" in dev_kind
  use_3d_grid = any(k in dev_kind for k in ("v3", "v4"))

  wi0_t = jnp.swapaxes(wi_0.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wi1_t = jnp.swapaxes(wi_1.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wo_t = wo.reshape(e_dim, num_ff_tiles, block_ff, d_latent)

  if use_3d_grid:
    out_init = jnp.zeros_like(x_grouped)
    return pl.pallas_call(
        functools.partial(_fused_latent_moe_fwd_v3_kernel, cast_rhs_f32=cast_rhs_f32),
        out_shape=jax.ShapeDtypeStruct(x_grouped.shape, x_grouped.dtype),
        grid=(e_dim, num_m_tiles, num_ff_tiles),
        in_specs=[
            pl.BlockSpec((1, block_m, d_latent), lambda e, m, f: (e, m, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, m, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, m, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, block_ff, d_latent), lambda e, m, f: (e, f, 0, 0)),
            pl.BlockSpec((1, block_m, d_latent), lambda e, m, f: (e, m, 0)),
        ],
        out_specs=pl.BlockSpec((1, block_m, d_latent), lambda e, m, f: (e, m, 0)),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x_grouped, wi0_t, wi1_t, wo_t, out_init)

  return pl.pallas_call(
      functools.partial(
          _fused_latent_moe_fwd_kernel,
          num_ff_tiles=num_ff_tiles,
          cast_rhs_f32=cast_rhs_f32,
      ),
      out_shape=jax.ShapeDtypeStruct(x_grouped.shape, x_grouped.dtype),
      grid=(e_dim, num_m_tiles),
      in_specs=[
          pl.BlockSpec((1, block_m, d_latent), lambda e, m: (e, m, 0)),
          pl.BlockSpec((1, num_ff_tiles, d_latent, block_ff), lambda e, m: (e, 0, 0, 0)),
          pl.BlockSpec((1, num_ff_tiles, d_latent, block_ff), lambda e, m: (e, 0, 0, 0)),
          pl.BlockSpec((1, num_ff_tiles, block_ff, d_latent), lambda e, m: (e, 0, 0, 0)),
      ],
      out_specs=pl.BlockSpec((1, block_m, d_latent), lambda e, m: (e, m, 0)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
  )(x_grouped, wi0_t, wi1_t, wo_t)


def _fused_latent_moe_pallas_fwd(x_grouped, wi_0, wi_1, wo, block_m: int, block_ff: int):
  out = fused_latent_moe_pallas(x_grouped, wi_0, wi_1, wo, block_m, block_ff)
  return out, (x_grouped, wi_0, wi_1, wo)


def _fused_latent_moe_pallas_bwd(block_m: int, block_ff: int, res, dy):
  x_grouped, wi_0, wi_1, wo = res
  e_dim, m_per_e, d_latent = x_grouped.shape
  d_ff = wi_0.shape[-1]
  num_m_tiles = m_per_e // block_m
  num_ff_tiles = d_ff // block_ff
  dev_kind = jax.devices()[0].device_kind
  cast_rhs_f32 = "v3" in dev_kind
  use_3d_bwd = any(k in dev_kind for k in ("v3", "v4")) and num_m_tiles > 1

  x_t = x_grouped.reshape(e_dim, num_m_tiles, block_m, d_latent)
  dy_t = dy.reshape(e_dim, num_m_tiles, block_m, d_latent)
  wi0_t = jnp.swapaxes(wi_0.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wi1_t = jnp.swapaxes(wi_1.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wo_t = wo.reshape(e_dim, num_ff_tiles, block_ff, d_latent)
  dx_init = jnp.zeros_like(x_t)

  if use_3d_bwd:
    dx_t, dwi0_t, dwi1_t, dwo_t = pl.pallas_call(
        functools.partial(
            _fused_latent_moe_bwd_3d_kernel,
            num_m_tiles=num_m_tiles,
            cast_rhs_f32=cast_rhs_f32,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(x_t.shape, x_t.dtype),
            jax.ShapeDtypeStruct(wi0_t.shape, wi_0.dtype),
            jax.ShapeDtypeStruct(wi1_t.shape, wi_1.dtype),
            jax.ShapeDtypeStruct(wo_t.shape, wo.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(e_dim, num_ff_tiles, num_m_tiles),
            in_specs=[
                pl.BlockSpec((1, 1, block_m, d_latent), lambda e, f, m: (e, m, 0, 0)),
                pl.BlockSpec((1, 1, block_m, d_latent), lambda e, f, m: (e, m, 0, 0)),
                pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f, m: (e, f, 0, 0)),
                pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f, m: (e, f, 0, 0)),
                pl.BlockSpec((1, 1, block_ff, d_latent), lambda e, f, m: (e, f, 0, 0)),
                pl.BlockSpec((1, 1, block_m, d_latent), lambda e, f, m: (e, m, 0, 0)),
            ],
            out_specs=(
                pl.BlockSpec((1, 1, block_m, d_latent), lambda e, f, m: (e, m, 0, 0)),
                pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f, m: (e, f, 0, 0)),
                pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f, m: (e, f, 0, 0)),
                pl.BlockSpec((1, 1, block_ff, d_latent), lambda e, f, m: (e, f, 0, 0)),
            ),
            scratch_shapes=[
                pltpu.VMEM((d_latent, block_ff), jnp.float32),
                pltpu.VMEM((d_latent, block_ff), jnp.float32),
                pltpu.VMEM((block_ff, d_latent), jnp.float32),
            ],
        ),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
    )(x_t, dy_t, wi0_t, wi1_t, wo_t, dx_init)
  else:
    dx_t, dwi0_t, dwi1_t, dwo_t = pl.pallas_call(
        functools.partial(
            _fused_latent_moe_bwd_kernel,
            num_m_tiles=num_m_tiles,
            cast_rhs_f32=cast_rhs_f32,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(x_t.shape, x_t.dtype),
            jax.ShapeDtypeStruct(wi0_t.shape, wi_0.dtype),
            jax.ShapeDtypeStruct(wi1_t.shape, wi_1.dtype),
            jax.ShapeDtypeStruct(wo_t.shape, wo.dtype),
        ),
        grid=(e_dim, num_ff_tiles),
        in_specs=[
            pl.BlockSpec((1, num_m_tiles, block_m, d_latent), lambda e, f: (e, 0, 0, 0)),
            pl.BlockSpec((1, num_m_tiles, block_m, d_latent), lambda e, f: (e, 0, 0, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, block_ff, d_latent), lambda e, f: (e, f, 0, 0)),
            pl.BlockSpec((1, num_m_tiles, block_m, d_latent), lambda e, f: (e, 0, 0, 0)),
        ],
        out_specs=(
            pl.BlockSpec((1, num_m_tiles, block_m, d_latent), lambda e, f: (e, 0, 0, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, d_latent, block_ff), lambda e, f: (e, f, 0, 0)),
            pl.BlockSpec((1, 1, block_ff, d_latent), lambda e, f: (e, f, 0, 0)),
        ),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )(x_t, dy_t, wi0_t, wi1_t, wo_t, dx_init)

  return (
      dx_t.reshape(x_grouped.shape),
      jnp.swapaxes(dwi0_t, 1, 2).reshape(wi_0.shape),
      jnp.swapaxes(dwi1_t, 1, 2).reshape(wi_1.shape),
      dwo_t.reshape(wo.shape),
  )


fused_latent_moe_pallas.defvjp(_fused_latent_moe_pallas_fwd, _fused_latent_moe_pallas_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def hybrid_latent_moe_xla_fwd_pallas_bwd(
    x_grouped: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    block_m: int = 128,
    block_ff: int = 256,
) -> jax.Array:
  """Hybrid LatentMoE: XLA batched/ragged dot forward + VMEM-fused Pallas backward (zero HBM intermediate save)."""
  h0 = jnp.einsum("emd,edf->emf", x_grouped, wi_0, preferred_element_type=jnp.float32)
  h1 = jnp.einsum("emd,edf->emf", x_grouped, wi_1, preferred_element_type=jnp.float32)
  act = (jax.nn.silu(h0) * h1).astype(x_grouped.dtype)
  return jnp.einsum("emf,efd->emd", act, wo, preferred_element_type=jnp.float32).astype(x_grouped.dtype)


def _hybrid_latent_moe_fwd(x_grouped, wi_0, wi_1, wo, block_m: int, block_ff: int):
  out = hybrid_latent_moe_xla_fwd_pallas_bwd(x_grouped, wi_0, wi_1, wo, block_m, block_ff)
  return out, (x_grouped, wi_0, wi_1, wo)


hybrid_latent_moe_xla_fwd_pallas_bwd.defvjp(_hybrid_latent_moe_fwd, _fused_latent_moe_pallas_bwd)


def _fused_ragged_latent_moe_fwd_kernel(
    group_metadata,
    x_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    out_init_ref,
    out_ref,
    out_scratch,
    *,
    block_m: int,
    cast_rhs_f32: bool = False,
):
  """VMEM-fused forward Pallas kernel for dynamic ragged group_sizes[E_local]."""
  grid_id = pl.program_id(1)
  _, _, m_tile_ids = group_metadata
  m_tile = m_tile_ids[grid_id]
  prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
  m_tile_has_changed = jnp.logical_or(grid_id == 0, m_tile_ids[prev_grid_id] != m_tile)

  @pl.when(m_tile_has_changed)
  def _load_out():
    out_scratch[...] = out_init_ref[...].astype(jnp.float32)

  row_mask = mblx_backend._get_store_mask(
      grid_id=grid_id,
      group_metadata=group_metadata,
      tm=block_m,
      tn=1,
  )
  x = jnp.where(row_mask, x_ref[...], 0.0)
  w0 = wi0_ref[0, 0, :, :]
  w1 = wi1_ref[0, 0, :, :]
  wo_t = wo_ref[0, 0, :, :]
  if cast_rhs_f32:
    x = x.astype(jnp.float32)
    w0 = w0.astype(jnp.float32)
    w1 = w1.astype(jnp.float32)
    wo_t = wo_t.astype(jnp.float32)
  h0 = jnp.dot(x, w0, preferred_element_type=jnp.float32)
  h1 = jnp.dot(x, w1, preferred_element_type=jnp.float32)
  act = jax.nn.silu(h0) * h1
  act_mm = act if cast_rhs_f32 else act.astype(x_ref.dtype)
  part = jnp.dot(act_mm, wo_t, preferred_element_type=jnp.float32)
  out_scratch[...] += part

  is_end_of_grid = grid_id == (pl.num_programs(1) - 1)
  next_grid_id = jnp.where(is_end_of_grid, grid_id, grid_id + 1)
  m_tile_is_changing = jnp.logical_or(is_end_of_grid, m_tile != m_tile_ids[next_grid_id])

  @pl.when(m_tile_is_changing)
  def _store_out():
    out_ref[...] = out_scratch[...].astype(out_ref.dtype)


def _fused_ragged_latent_moe_bwd_kernel(
    group_metadata,
    x_ref,
    dy_ref,
    wi0_ref,
    wi1_ref,
    wo_ref,
    dx_init_ref,
    dx_ref,
    dwi0_ref,
    dwi1_ref,
    dwo_ref,
    dx_scratch,
    dw0_scratch,
    dw1_scratch,
    dwo_scratch,
    *,
    block_m: int,
    cast_rhs_f32: bool = False,
):
  """VMEM-fused backward Pallas kernel for dynamic ragged group_sizes[E_local]."""
  grid_id = pl.program_id(1)
  _, group_ids, m_tile_ids = group_metadata
  group = group_ids[grid_id]
  m_tile = m_tile_ids[grid_id]
  prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
  group_has_changed = jnp.logical_or(grid_id == 0, group_ids[prev_grid_id] != group)
  m_tile_has_changed = jnp.logical_or(grid_id == 0, m_tile_ids[prev_grid_id] != m_tile)

  @pl.when(group_has_changed)
  def _zero_dw():
    dw0_scratch[...] = jnp.zeros_like(dw0_scratch)
    dw1_scratch[...] = jnp.zeros_like(dw1_scratch)
    dwo_scratch[...] = jnp.zeros_like(dwo_scratch)

  @pl.when(m_tile_has_changed)
  def _load_dx():
    dx_scratch[...] = dx_init_ref[...].astype(jnp.float32)

  row_mask = mblx_backend._get_store_mask(
      grid_id=grid_id,
      group_metadata=group_metadata,
      tm=block_m,
      tn=1,
  )
  x = jnp.where(row_mask, x_ref[...], 0.0)
  dy = jnp.where(row_mask, dy_ref[...], 0.0)
  w0 = wi0_ref[0, 0, :, :]
  w1 = wi1_ref[0, 0, :, :]
  wo_t = wo_ref[0, 0, :, :]
  w0_rhs = w0.astype(jnp.float32) if cast_rhs_f32 else w0
  w1_rhs = w1.astype(jnp.float32) if cast_rhs_f32 else w1
  wo_t_rhs = wo_t.astype(jnp.float32) if cast_rhs_f32 else wo_t
  x_mm = x.astype(jnp.float32) if cast_rhs_f32 else x
  dy_mm = dy.astype(jnp.float32) if cast_rhs_f32 else dy

  h0 = jnp.dot(x_mm, w0_rhs, preferred_element_type=jnp.float32)
  h1 = jnp.dot(x_mm, w1_rhs, preferred_element_type=jnp.float32)
  sig0 = jax.nn.sigmoid(h0)
  silu0 = h0 * sig0
  act = (silu0 * h1) if cast_rhs_f32 else (silu0 * h1).astype(x.dtype)

  dwo_part = jnp.dot(act.T, dy_mm, preferred_element_type=jnp.float32)
  dact = jnp.dot(dy_mm, wo_t_rhs.T, preferred_element_type=jnp.float32)
  dh1 = (dact * silu0) if cast_rhs_f32 else (dact * silu0).astype(x.dtype)
  dsilu = dact * h1 * (sig0 * (1.0 + h0 * (1.0 - sig0)))
  dh0 = dsilu if cast_rhs_f32 else dsilu.astype(x.dtype)

  dw0_part = jnp.dot(x_mm.T, dh0, preferred_element_type=jnp.float32)
  dw1_part = jnp.dot(x_mm.T, dh1, preferred_element_type=jnp.float32)
  dx_part = jnp.dot(dh0, w0_rhs.T, preferred_element_type=jnp.float32) + jnp.dot(
      dh1, w1_rhs.T, preferred_element_type=jnp.float32
  )

  dx_scratch[...] += dx_part
  dw0_scratch[...] += dw0_part
  dw1_scratch[...] += dw1_part
  dwo_scratch[...] += dwo_part

  is_end_of_grid = grid_id == (pl.num_programs(1) - 1)
  next_grid_id = jnp.where(is_end_of_grid, grid_id, grid_id + 1)
  group_is_changing = jnp.logical_or(is_end_of_grid, group != group_ids[next_grid_id])
  m_tile_is_changing = jnp.logical_or(is_end_of_grid, m_tile != m_tile_ids[next_grid_id])

  @pl.when(m_tile_is_changing)
  def _store_dx():
    dx_ref[...] = dx_scratch[...].astype(dx_ref.dtype)

  @pl.when(group_is_changing)
  def _store_dw():
    dwi0_ref[0, 0, :, :] = dw0_scratch[...].astype(dwi0_ref.dtype)
    dwi1_ref[0, 0, :, :] = dw1_scratch[...].astype(dwi1_ref.dtype)
    dwo_ref[0, 0, :, :] = dwo_scratch[...].astype(dwo_ref.dtype)


def _extract_vma(tensor: jax.Array) -> tuple[str, ...]:
  type_str = str(jax.typeof(tensor))
  if "{V:" in type_str:
    start = type_str.index("{V:") + 3
    end = type_str.index("}", start)
    vma_content = type_str[start:end].strip("()")
    return tuple(sorted(a.strip() for a in vma_content.split(",") if a.strip()))
  return tuple()


def _make_out_sds(shape, dtype, *tensors) -> jax.ShapeDtypeStruct:
  vma = frozenset(ax for t in tensors for ax in _extract_vma(t))
  if vma:
    return jax.ShapeDtypeStruct(
        shape, dtype, manual_axis_type=jax.sharding.ManualAxisType(varying=vma)
    )
  return jax.ShapeDtypeStruct(shape, dtype)


def _run_ragged_latent_moe_pallas_fwd(
    x: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    block_m: int = 128,
    block_ff: int = 256,
) -> jax.Array:
  """Runs the VMEM-fused forward Pallas kernel for dynamic group_sizes[E_local]."""
  m_orig, d_latent = x.shape
  pad_m = (block_m - (m_orig % block_m)) % block_m
  x_pad = jnp.pad(x, ((0, pad_m), (0, 0))) if pad_m > 0 else x
  m_padded = x_pad.shape[0]
  e_dim, _, d_ff = wi_0.shape
  num_ff_tiles = d_ff // block_ff
  dev_kind = jax.devices()[0].device_kind
  cast_rhs_f32 = "v3" in dev_kind

  group_metadata, num_active_tiles = mblx_backend.make_group_metadata(
      group_sizes=group_sizes.astype(jnp.int32),
      m=m_padded,
      tm=block_m,
      start_group=jnp.array(0, dtype=jnp.int32),
      num_nonzero_groups=e_dim,
      visit_empty_groups=True,
  )
  wi0_t = jnp.swapaxes(wi_0.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wi1_t = jnp.swapaxes(wi_1.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wo_t = wo.reshape(e_dim, num_ff_tiles, block_ff, d_latent)
  out_init = jnp.zeros_like(x_pad)

  out_pad = pl.pallas_call(
      functools.partial(
          _fused_ragged_latent_moe_fwd_kernel,
          block_m=block_m,
          cast_rhs_f32=cast_rhs_f32,
      ),
      out_shape=_make_out_sds(x_pad.shape, x_pad.dtype, x_pad, wi_0, wi_1, wo),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(num_ff_tiles, num_active_tiles),
          in_specs=[
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, block_ff, d_latent), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
          ],
          out_specs=pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
          scratch_shapes=[pltpu.VMEM((block_m, d_latent), jnp.float32)],
      ),
      input_output_aliases={7: 0},
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
  )(group_metadata, x_pad, wi0_t, wi1_t, wo_t, out_init)
  return out_pad[:m_orig] if pad_m > 0 else out_pad


def _run_ragged_latent_moe_pallas_bwd(
    block_m: int,
    block_ff: int,
    res,
    dy: jax.Array,
):
  """Runs the VMEM-fused backward Pallas kernel for dynamic group_sizes[E_local]."""
  x, wi_0, wi_1, wo, group_sizes = res
  m_orig, d_latent = x.shape
  pad_m = (block_m - (m_orig % block_m)) % block_m
  x_pad = jnp.pad(x, ((0, pad_m), (0, 0))) if pad_m > 0 else x
  dy_pad = jnp.pad(dy, ((0, pad_m), (0, 0))) if pad_m > 0 else dy
  m_padded = x_pad.shape[0]
  e_dim, _, d_ff = wi_0.shape
  num_ff_tiles = d_ff // block_ff
  dev_kind = jax.devices()[0].device_kind
  cast_rhs_f32 = "v3" in dev_kind

  group_metadata, num_active_tiles = mblx_backend.make_group_metadata(
      group_sizes=group_sizes.astype(jnp.int32),
      m=m_padded,
      tm=block_m,
      start_group=jnp.array(0, dtype=jnp.int32),
      num_nonzero_groups=e_dim,
      visit_empty_groups=True,
  )
  wi0_t = jnp.swapaxes(wi_0.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wi1_t = jnp.swapaxes(wi_1.reshape(e_dim, d_latent, num_ff_tiles, block_ff), 1, 2)
  wo_t = wo.reshape(e_dim, num_ff_tiles, block_ff, d_latent)
  dx_init = jnp.zeros_like(x_pad)

  dx_pad, dwi0_t, dwi1_t, dwo_t = pl.pallas_call(
      functools.partial(
          _fused_ragged_latent_moe_bwd_kernel,
          block_m=block_m,
          cast_rhs_f32=cast_rhs_f32,
      ),
      out_shape=(
          _make_out_sds(x_pad.shape, x_pad.dtype, x_pad, dy_pad, wi_0),
          _make_out_sds(wi0_t.shape, wi_0.dtype, x_pad, dy_pad, wi_0),
          _make_out_sds(wi1_t.shape, wi_1.dtype, x_pad, dy_pad, wi_1),
          _make_out_sds(wo_t.shape, wo.dtype, x_pad, dy_pad, wo),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(num_ff_tiles, num_active_tiles),
          in_specs=[
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, block_ff, d_latent), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
          ],
          out_specs=(
              pl.BlockSpec((block_m, d_latent), lambda f, g, gm: (gm[2][g], 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, d_latent, block_ff), lambda f, g, gm: (gm[1][g], f, 0, 0)),
              pl.BlockSpec((1, 1, block_ff, d_latent), lambda f, g, gm: (gm[1][g], f, 0, 0)),
          ),
          scratch_shapes=[
              pltpu.VMEM((block_m, d_latent), jnp.float32),
              pltpu.VMEM((d_latent, block_ff), jnp.float32),
              pltpu.VMEM((d_latent, block_ff), jnp.float32),
              pltpu.VMEM((block_ff, d_latent), jnp.float32),
          ],
      ),
      input_output_aliases={8: 0},
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
  )(group_metadata, x_pad, dy_pad, wi0_t, wi1_t, wo_t, dx_init)

  dx = dx_pad[:m_orig] if pad_m > 0 else dx_pad
  return (
      dx,
      jnp.swapaxes(dwi0_t, 1, 2).reshape(wi_0.shape),
      jnp.swapaxes(dwi1_t, 1, 2).reshape(wi_1.shape),
      dwo_t.reshape(wo.shape),
      None,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def fused_ragged_latent_moe_pallas(
    x: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    block_m: int = 128,
    block_ff: int = 256,
) -> jax.Array:
  """1-Kernel VMEM-fused forward + backward Pallas LatentMoE for dynamic group_sizes[E_local]."""
  return _run_ragged_latent_moe_pallas_fwd(x, wi_0, wi_1, wo, group_sizes, block_m, block_ff)


def _fused_ragged_latent_moe_pallas_fwd(x, wi_0, wi_1, wo, group_sizes, block_m: int, block_ff: int):
  out = _run_ragged_latent_moe_pallas_fwd(x, wi_0, wi_1, wo, group_sizes, block_m, block_ff)
  return out, (x, wi_0, wi_1, wo, group_sizes)


fused_ragged_latent_moe_pallas.defvjp(_fused_ragged_latent_moe_pallas_fwd, _run_ragged_latent_moe_pallas_bwd)



def _ragged_dot_fwd_op(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile: tuple[int, int, int],
    preferred_element_type: jnp.dtype,
) -> jax.Array:
  """Single forward ragged_dot_general with explicit ragged_dot_tiling metadata."""
  dims_fwd = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((1,), (1,)), ((), ())),
      lhs_ragged_dimensions=(0,),
      rhs_group_dimensions=(0,),
  )
  with jax.experimental.xla_metadata.set_xla_metadata(
      ragged_dot_tiling=",".join(str(t) for t in tile),
  ):
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dims_fwd,
        preferred_element_type=preferred_element_type,
    )


def _ragged_dot_dlhs_op(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile: tuple[int, int, int],
    preferred_element_type: jnp.dtype,
) -> jax.Array:
  """Single backward dlhs ragged_dot_general with explicit ragged_dot_tiling metadata."""
  dims_dlhs = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((1,), (1,)), ((), ())),
      lhs_ragged_dimensions=(0,),
      rhs_group_dimensions=(0,),
  )
  with jax.experimental.xla_metadata.set_xla_metadata(
      ragged_dot_tiling=",".join(str(t) for t in tile),
  ):
    return jax.lax.ragged_dot_general(
        lhs=dout,
        rhs=jnp.swapaxes(rhs, -1, -2),
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dims_dlhs,
        preferred_element_type=preferred_element_type,
    )


def _ragged_dot_drhs_op(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    tile: tuple[int, int, int],
    preferred_element_type: jnp.dtype,
) -> jax.Array:
  """Single backward drhs ragged_dot_general with explicit ragged_dot_tiling metadata."""
  dims_drhs = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((0,), (0,)), ((), ())),
      lhs_ragged_dimensions=(0,),
      rhs_group_dimensions=(),
  )
  with jax.experimental.xla_metadata.set_xla_metadata(
      ragged_dot_tiling=",".join(str(t) for t in tile),
  ):
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dims_drhs,
        preferred_element_type=preferred_element_type,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def fused_ragged_3gmm_swiglu(
    x: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    wi_tiling: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = (
        (128, 768, 1792),
        (128, 1792, 768),
        (128, 768, 896),
    ),
    wo_tiling: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = (
        (128, 1792, 768),
        (128, 768, 1792),
        (128, 896, 768),
    ),
    recompute_up: bool = False,
) -> jax.Array:
  """Custom-VJP 3-GMM SwiGLU for dynamic group_sizes that never caches `act` (`[M, d_ff]`)."""
  del recompute_up
  dtype = x.dtype
  l0 = _ragged_dot_fwd_op(x, wi_0, group_sizes, wi_tiling[0], dtype)
  l1 = _ragged_dot_fwd_op(x, wi_1, group_sizes, wi_tiling[0], dtype)
  act = jnp.asarray(jax.nn.silu(l0) * l1, dtype=dtype)
  return _ragged_dot_fwd_op(act, wo, group_sizes, wo_tiling[0], dtype)


def _fused_ragged_3gmm_swiglu_fwd(x, wi_0, wi_1, wo, group_sizes, wi_tiling, wo_tiling, recompute_up: bool):
  dtype = x.dtype
  l0 = _ragged_dot_fwd_op(x, wi_0, group_sizes, wi_tiling[0], dtype)
  l1 = _ragged_dot_fwd_op(x, wi_1, group_sizes, wi_tiling[0], dtype)
  act = jnp.asarray(jax.nn.silu(l0) * l1, dtype=dtype)
  out = _ragged_dot_fwd_op(act, wo, group_sizes, wo_tiling[0], dtype)
  if recompute_up:
    return out, (x, None, None, wi_0, wi_1, wo, group_sizes)
  return out, (x, l0, l1, wi_0, wi_1, wo, group_sizes)


def _fused_ragged_3gmm_swiglu_bwd(wi_tiling, wo_tiling, recompute_up: bool, res, dout):
  x, l0, l1, wi_0, wi_1, wo, group_sizes = res
  dtype = x.dtype
  dout = dout.astype(dtype)
  if recompute_up:
    l0 = _ragged_dot_fwd_op(x, wi_0, group_sizes, wi_tiling[0], dtype)
    l1 = _ragged_dot_fwd_op(x, wi_1, group_sizes, wi_tiling[0], dtype)

  l0_f32 = l0.astype(jnp.float32)
  l1_f32 = l1.astype(jnp.float32)
  sig0 = jax.nn.sigmoid(l0_f32)
  silu0 = l0_f32 * sig0
  act = (silu0 * l1_f32).astype(dtype)

  d_act = _ragged_dot_dlhs_op(dout, wo, group_sizes, wo_tiling[1], dtype)
  dwo = _ragged_dot_drhs_op(act, dout, group_sizes, wo_tiling[2], wo.dtype)

  d_act_f32 = d_act.astype(jnp.float32)
  dl1 = (d_act_f32 * silu0).astype(dtype)
  dl0 = (d_act_f32 * l1_f32 * sig0 * (1.0 + l0_f32 * (1.0 - sig0))).astype(dtype)

  dwi_0 = _ragged_dot_drhs_op(x, dl0, group_sizes, wi_tiling[2], wi_0.dtype)
  dwi_1 = _ragged_dot_drhs_op(x, dl1, group_sizes, wi_tiling[2], wi_1.dtype)
  dx = _ragged_dot_dlhs_op(dl0, wi_0, group_sizes, wi_tiling[1], dtype) + _ragged_dot_dlhs_op(
      dl1, wi_1, group_sizes, wi_tiling[1], dtype
  )
  return dx, dwi_0, dwi_1, dwo, None


fused_ragged_3gmm_swiglu.defvjp(_fused_ragged_3gmm_swiglu_fwd, _fused_ragged_3gmm_swiglu_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def hybrid_ragged_latent_moe_xla_fwd_pallas_bwd(
    x: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    group_sizes: jax.Array,
    wi_fwd_tile: tuple[int, int, int] = (256, 768, 1792),
    wo_fwd_tile: tuple[int, int, int] = (256, 1792, 768),
    block_m: int = 128,
    block_ff: int = 256,
) -> jax.Array:
  """Hybrid Ragged LatentMoE for dynamic group_sizes: 3-GMM XLA ragged_dot forward + 1-Kernel VMEM-Fused Pallas backward (zero HBM intermediate save)."""
  del block_m, block_ff
  dtype = x.dtype
  l0 = _ragged_dot_fwd_op(x, wi_0, group_sizes, wi_fwd_tile, dtype)
  l1 = _ragged_dot_fwd_op(x, wi_1, group_sizes, wi_fwd_tile, dtype)
  act = jnp.asarray(jax.nn.silu(l0) * l1, dtype=dtype)
  return _ragged_dot_fwd_op(act, wo, group_sizes, wo_fwd_tile, dtype)


def _hybrid_ragged_latent_moe_fwd(
    x,
    wi_0,
    wi_1,
    wo,
    group_sizes,
    wi_fwd_tile: tuple[int, int, int],
    wo_fwd_tile: tuple[int, int, int],
    block_m: int,
    block_ff: int,
):
  out = hybrid_ragged_latent_moe_xla_fwd_pallas_bwd(
      x, wi_0, wi_1, wo, group_sizes, wi_fwd_tile, wo_fwd_tile, block_m, block_ff
  )
  return out, (x, wi_0, wi_1, wo, group_sizes)


def _hybrid_ragged_latent_moe_bwd(
    wi_fwd_tile: tuple[int, int, int],
    wo_fwd_tile: tuple[int, int, int],
    block_m: int,
    block_ff: int,
    res,
    dy: jax.Array,
):
  del wi_fwd_tile, wo_fwd_tile
  return _run_ragged_latent_moe_pallas_bwd(block_m, block_ff, res, dy)


hybrid_ragged_latent_moe_xla_fwd_pallas_bwd.defvjp(
    _hybrid_ragged_latent_moe_fwd, _hybrid_ragged_latent_moe_bwd
)


