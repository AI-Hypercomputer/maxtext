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

"""Manifold-Constrained Hyper Connections (mHC) Pallas kernels."""

import functools
from typing import Callable, Any

from flax import nnx
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from maxtext.common.common_types import Array, HyperConnectionType
from maxtext.utils.sharding import logical_to_mesh_axes

bf16 = jnp.bfloat16
f32 = jnp.float32

VMEM_LIMIT_BYTES = 63 * 1024 * 1024  # 63MB VMEM limit to fit in TPU v5e usable VMEM
EPS = 1e-6  # epsilon for RMSNorm


def _whole(shape):
  """full-array BlockSpec (index_map -> 0): weights load once, stay VMEM-resident."""
  return pl.BlockSpec(shape, lambda i: tuple(0 for _ in shape))


def _grid_sharded(shape):
  """Shards the first dimension of shape along the grid index."""
  return pl.BlockSpec((1,) + shape[1:], lambda i: (i,) + tuple(0 for _ in shape[1:]))


# ======================================================================================
# Helper mathematical functions (JAX-traceable, called inside/outside kernels)
# ======================================================================================
def fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha):
  concat_alpha = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)
  phi = norm_scale.astype(f32)[:, None] * concat_alpha.astype(f32)
  return phi.T


@jax.custom_vjp
def mhc_coeffs(
    x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm
):
  T, k, d = x.shape
  m = k * d
  P = perm.shape[0]
  xf = x.reshape(T, m)

  # Normalize first (Formulation 1)
  xf32 = xf.astype(f32)
  mean2 = jnp.mean(xf32 * xf32, axis=-1, keepdims=True)
  r_inv = jax.lax.rsqrt(mean2 + EPS).astype(x.dtype)
  xf_norm = (xf * r_inv) * norm_scale.astype(x.dtype)

  if phi.shape[0] == 2 * k + P or phi.shape[0] == 32:
    q = jax.lax.dot_general(
        xf_norm,
        phi.astype(x.dtype),
        (((1,), (1,)), ((), (()))),
        preferred_element_type=f32,
    ).astype(x.dtype)
  else:
    q = jnp.dot(xf_norm, phi.astype(x.dtype), preferred_element_type=f32).astype(x.dtype)

  h = q
  h_pre, h_post, h_res = h[:, :k], h[:, k : 2 * k], h[:, 2 * k :]
  H_pre = jax.nn.sigmoid(pre_s.astype(x.dtype) * h_pre + pre_beta.astype(x.dtype))
  H_post = 2.0 * jax.nn.sigmoid(
      post_s.astype(x.dtype) * h_post + post_beta.astype(x.dtype)
  )
  weights_input = res_s.astype(x.dtype) * h_res + res_beta.astype(x.dtype)
  weights = jax.nn.softmax(weights_input.astype(f32), axis=-1).astype(x.dtype)
  res_M = jnp.dot(
      weights,
      perm.reshape(P, k * k).astype(x.dtype),
      preferred_element_type=f32,
  ).astype(x.dtype).reshape(T, k, k)
  return H_pre.astype(x.dtype), H_post.astype(x.dtype), res_M.astype(x.dtype)


def mhc_coeffs_fwd(
    x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm
):
  H_pre, H_post, res_M = mhc_coeffs(
      x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm
  )
  return (H_pre, H_post, res_M), (
      x,
      phi,
      norm_scale,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      perm,
  )


def mhc_coeffs_bwd(res, cotangents):
  print("MHC_TRACE: mhc_coeffs_bwd called!")
  x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm = res
  dy_pre, dy_post, dy_resM = cotangents

  T, k, d = x.shape
  m = k * d
  P = perm.shape[0]
  xf = x.reshape(T, m)

  # Recompute intermediates
  xf32 = xf.astype(f32)
  mean2 = jnp.mean(xf32 * xf32, axis=-1, keepdims=True)
  r_inv = jax.lax.rsqrt(mean2 + EPS).astype(xf.dtype)
  xf_r_inv = xf * r_inv
  xf_norm = xf_r_inv * norm_scale.astype(xf.dtype)

  h = jnp.dot(xf_norm, phi.astype(xf.dtype).T, preferred_element_type=f32).astype(xf.dtype)
  h_pre, h_post, h_res = h[:, :k], h[:, k : 2 * k], h[:, 2 * k :]

  # H_pre branch
  H_pre = jax.nn.sigmoid(pre_s.astype(xf.dtype) * h_pre + pre_beta.astype(xf.dtype))
  dy_pre_type = dy_pre.astype(xf.dtype)
  d_pre_input = dy_pre_type * H_pre * (1.0 - H_pre)

  dpre_s_contrib = d_pre_input.astype(f32) * h_pre.astype(f32)
  dpre_s = jnp.sum(dpre_s_contrib).reshape((1,))

  dpre_beta_contrib = d_pre_input.astype(f32)
  dpre_beta = jnp.sum(dpre_beta_contrib, axis=0)

  d_h_pre = d_pre_input * pre_s.astype(xf.dtype)

  # H_post branch
  H_post_half = jax.nn.sigmoid(
      post_s.astype(xf.dtype) * h_post + post_beta.astype(xf.dtype)
  )
  dy_post_type = dy_post.astype(xf.dtype)
  d_post_input = dy_post_type * 2.0 * H_post_half * (1.0 - H_post_half)

  dpost_s_contrib = d_post_input.astype(f32) * h_post.astype(f32)
  dpost_s = jnp.sum(dpost_s_contrib).reshape((1,))

  dpost_beta_contrib = d_post_input.astype(f32)
  dpost_beta = jnp.sum(dpost_beta_contrib, axis=0)

  d_h_post = d_post_input * post_s.astype(xf.dtype)

  # res_M branch
  dy_resM_type = dy_resM.astype(xf.dtype)
  weights_input = res_s.astype(xf.dtype) * h_res + res_beta.astype(xf.dtype)
  weights = jax.nn.softmax(weights_input.astype(f32), axis=-1).astype(xf.dtype)

  dy_resM_flat = dy_resM_type.reshape(T, k * k)
  perm_flat = perm.reshape(P, k * k).astype(xf.dtype)
  d_weights = jnp.dot(
      dy_resM_flat, perm_flat.T, preferred_element_type=f32
  ).astype(xf.dtype)

  d_weights_f32 = d_weights.astype(f32)
  weights_f32 = weights.astype(f32)
  d_weights_input = weights_f32 * (
      d_weights_f32
      - jnp.sum(d_weights_f32 * weights_f32, axis=-1, keepdims=True)
  )

  dres_s_contrib = d_weights_input * h_res.astype(f32)
  dres_s = jnp.sum(dres_s_contrib).reshape((1,))

  dres_beta_contrib = d_weights_input
  dres_beta = jnp.sum(dres_beta_contrib, axis=0)

  d_h_res = d_weights_input.astype(xf.dtype) * res_s.astype(xf.dtype)

  # Combine d_h
  d_h = jnp.concatenate([d_h_pre, d_h_post, d_h_res], axis=-1)

  # q = xf_norm @ phi.T
  d_xf_norm = jnp.dot(d_h, phi.astype(xf.dtype), preferred_element_type=f32).astype(xf.dtype)
  d_phi = jnp.dot(d_h.T, xf_norm, preferred_element_type=f32)

  # xf_norm = xf_r_inv * norm_scale
  d_xf_r_inv = d_xf_norm * norm_scale.astype(xf.dtype)
  d_norm_scale = jnp.sum((d_xf_norm.astype(f32) * xf_r_inv.astype(f32)), axis=0)

  # xf_r_inv = xf * r_inv
  d_xf_from_r_inv = d_xf_r_inv * r_inv
  d_r_inv = jnp.sum((d_xf_r_inv.astype(f32) * xf32), axis=-1, keepdims=True)

  # r_inv = rsqrt(mean(xf^2) + EPS)
  d_r_inv_f32 = d_r_inv.astype(f32)
  r_inv_f32 = r_inv.astype(f32)
  d_mean = d_r_inv_f32 * (-0.5 * r_inv_f32 * r_inv_f32 * r_inv_f32)
  d_xf_from_mean = d_mean * 2.0 * xf32 / m

  d_xf = d_xf_from_r_inv.astype(f32) + d_xf_from_mean
  dx = d_xf.reshape(T, k, d)

  d_perm = jnp.zeros_like(perm)

  return (
      dx,
      d_phi,
      d_norm_scale,
      dpre_s,
      dpre_beta,
      dpost_s,
      dpost_beta,
      dres_s,
      dres_beta,
      d_perm,
  )


mhc_coeffs.defvjp(mhc_coeffs_fwd, mhc_coeffs_bwd)


def mhc_pre_apply(x, H_pre):
  T, k, d = x.shape
  Hf = H_pre.astype(f32)
  layer_in = sum(
      Hf[:, kk : kk + 1] * x[:, kk, :].astype(f32) for kk in range(k)
  )
  return layer_in.astype(x.dtype)


def mhc_post_apply(x, layer_out, H_post, res_M):
  res_mix = jnp.einsum(
      "tkj,tkd->tjd", res_M.astype(x.dtype), x, preferred_element_type=f32
  )
  post_mix = H_post.astype(f32)[:, :, None] * layer_out.astype(f32)[:, None, :]
  return (res_mix + post_mix).astype(x.dtype)


# ======================================================================================
# FORWARD kernels
# ======================================================================================
def _coeff_fwd(
    xT,
    phi,
    norm_scale,
    pre_s,
    pre_beta,
    post_s,
    post_beta,
    res_s,
    res_beta,
    perm,
    *,
    bt,
    vmem,
    interpret,
):
  T, k, d = xT.shape
  m = k * d
  P = perm.shape[0]

  def kernel(
      x_ref,
      phi_ref,
      norm_scale_ref,
      ps_ref,
      pb_ref,
      qs_ref,
      qb_ref,
      rs_ref,
      rb_ref,
      perm_ref,
      hpre_ref,
      hpost_ref,
      resm_ref,
  ):
    hpre, hpost, resm = mhc_coeffs(
        x_ref[...],
        phi_ref[...],
        norm_scale_ref[...],
        ps_ref[...],
        pb_ref[...],
        qs_ref[...],
        qb_ref[...],
        rs_ref[...],
        rb_ref[...],
        perm_ref[...],
    )
    hpre_ref[...] = hpre
    hpost_ref[...] = hpost
    resm_ref[...] = resm

  cost = pl.CostEstimate(
      flops=int(2 * T * m * (2 * k + P) + 2 * T * P * k * k),
      transcendentals=int(T * (k + P)),
      bytes_accessed=int(T * k * d * 2 + m * (2 * k + P) * 4),
  )
  return pl.pallas_call(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct((T, k), xT.dtype),
          jax.ShapeDtypeStruct((T, k), xT.dtype),
          jax.ShapeDtypeStruct((T, k, k), xT.dtype),
      ],
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),  # x
          _whole((2 * k + P, m)),  # phi
          _whole((m,)),  # norm_scale
          _whole((1,)),
          _whole((k,)),  # pre_s, pre_beta
          _whole((1,)),
          _whole((k,)),  # post_s, post_beta
          _whole((1,)),
          _whole((P,)),  # res_s, res_beta
          _whole((P, k, k)),
      ],  # perm (constant)
      out_specs=[
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
          pl.BlockSpec((bt, k, k), lambda i: (i, 0, 0)),
      ],
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm)


def _pre_apply_fwd(xT, H_pre, *, bt, vmem, interpret):
  T, k, d = xT.shape

  def kernel(x_ref, hpre_ref, o_ref):
    o_ref[...] = mhc_pre_apply(x_ref[...], hpre_ref[...])

  cost = pl.CostEstimate(
      flops=int(2 * T * k * d),
      transcendentals=0,
      bytes_accessed=int(T * k * d * 2 + T * d * 2),
  )
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((T, d), xT.dtype),
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
      ],
      out_specs=pl.BlockSpec((bt, d), lambda i: (i, 0)),
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, H_pre)


def _post_apply_fwd(xT, lo, H_post, res_M, *, bt, vmem, interpret):
  T, k, d = xT.shape

  def kernel(x_ref, lo_ref, hpost_ref, resm_ref, o_ref):
    o_ref[...] = mhc_post_apply(
        x_ref[...], lo_ref[...], hpost_ref[...], resm_ref[...]
    )

  cost = pl.CostEstimate(
      flops=int(2 * T * k * k * d + T * k * d),
      transcendentals=0,
      bytes_accessed=int(2 * T * k * d * 2 + T * d * 4),
  )
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((T, k, d), xT.dtype),
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),  # x
          pl.BlockSpec((bt, d), lambda i: (i, 0)),  # layer_out
          pl.BlockSpec((bt, k), lambda i: (i, 0)),  # H_post
          pl.BlockSpec((bt, k, k), lambda i: (i, 0, 0)),
      ],  # res_M
      out_specs=pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, lo, H_post, res_M)


# ======================================================================================
# BACKWARD kernels
# ======================================================================================
def _coeff_bwd(
    xT,
    phi,
    norm_scale,
    pre_s,
    pre_beta,
    post_s,
    post_beta,
    res_s,
    res_beta,
    perm,
    dH_pre,
    dH_post,
    dres_M,
    dx_acc,
    *,
    bt,
    vmem,
    interpret,
):
  T, k, d = xT.shape
  m = k * d
  P = perm.shape[0]
  grid_size = T // bt

  def kernel(
      x_ref,
      phi_ref,
      norm_scale_ref,
      ps_ref,
      pb_ref,
      qs_ref,
      qb_ref,
      rs_ref,
      rb_ref,
      perm_ref,
      dhpre_ref,
      dhpost_ref,
      dresm_ref,
      dxacc_ref,
      dx_ref,
      dphi_ref,
      dns_ref,
      dps_ref,
      dpb_ref,
      dqs_ref,
      dqb_ref,
      drs_ref,
      drb_ref,
  ):
    i = pl.program_id(0)
    perm_c = perm_ref[...]

    x_val = x_ref[...]
    phi_val = phi_ref[...]
    norm_scale_val = norm_scale_ref[...]
    ps_val = ps_ref[...]
    pb_val = pb_ref[...]
    qs_val = qs_ref[...]
    qb_val = qb_ref[...]
    rs_val = rs_ref[...]
    rb_val = rb_ref[...]

    def f(xb, phib, ns_, ps_, pb_, qs_, qb_, rs_, rb_):
      return mhc_coeffs(xb, phib, ns_, ps_, pb_, qs_, qb_, rs_, rb_, perm_c)

    _, vjp = jax.vjp(
        f,
        x_val,
        phi_val,
        norm_scale_val,
        ps_val,
        pb_val,
        qs_val,
        qb_val,
        rs_val,
        rb_val,
    )
    vjp_args = (
        dhpre_ref[...],
        dhpost_ref[...],
        dresm_ref[...],
    )
    dxb, dphi, dns, dps, dpb, dqs, dqb, drs, drb = vjp(vjp_args)
    dx_ref[...] = (
        dxb.astype(dx_ref.dtype) + dxacc_ref[...].astype(dx_ref.dtype)
    )

    @pl.when(i == 0)
    def _init():
      dphi_ref[...] = jnp.zeros_like(dphi_ref)
      dns_ref[...] = jnp.zeros_like(dns_ref)
      dps_ref[...] = jnp.zeros_like(dps_ref)
      dpb_ref[...] = jnp.zeros_like(dpb_ref)
      dqs_ref[...] = jnp.zeros_like(dqs_ref)
      dqb_ref[...] = jnp.zeros_like(dqb_ref)
      drs_ref[...] = jnp.zeros_like(drs_ref)
      drb_ref[...] = jnp.zeros_like(drb_ref)

    dphi_ref[...] += dphi.astype(f32)
    dns_ref[...] += dns.astype(f32)
    dps_ref[...] += dps.astype(f32)
    dpb_ref[...] += dpb.astype(f32)
    dqs_ref[...] += dqs.astype(f32)
    dqb_ref[...] += dqb.astype(f32)
    drs_ref[...] += drs.astype(f32)
    drb_ref[...] += drb.astype(f32)

  cost = pl.CostEstimate(
      flops=int(2 * (2 * T * m * (2 * k + P) + 2 * T * P * k * k)),
      transcendentals=int(T * (k + P)),
      bytes_accessed=int(3 * T * k * d * 2 + 2 * m * (2 * k + P) * 4),
  )
  return pl.pallas_call(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct((T, k, d), xT.dtype),  # dx
          jax.ShapeDtypeStruct((2 * k + P, m), f32),  # dphi
          jax.ShapeDtypeStruct((m,), f32),  # d_norm_scale
          jax.ShapeDtypeStruct((1,), f32),  # d pre_s
          jax.ShapeDtypeStruct((k,), f32),  # d pre_beta
          jax.ShapeDtypeStruct((1,), f32),  # d post_s
          jax.ShapeDtypeStruct((k,), f32),  # d post_beta
          jax.ShapeDtypeStruct((1,), f32),  # d res_s
          jax.ShapeDtypeStruct((P,), f32),  # d res_beta
      ],
      grid=(grid_size,),
      in_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),  # x
          _whole((2 * k + P, m)),  # phi
          _whole((m,)),  # norm_scale
          _whole((1,)),
          _whole((k,)),  # pre_s, pre_beta
          _whole((1,)),
          _whole((k,)),  # post_s, post_beta
          _whole((1,)),
          _whole((P,)),  # res_s, res_beta
          _whole((P, k, k)),  # perm (constant)
          pl.BlockSpec((bt, k), lambda i: (i, 0)),  # dH_pre cotangent
          pl.BlockSpec((bt, k), lambda i: (i, 0)),  # dH_post cotangent
          pl.BlockSpec((bt, k, k), lambda i: (i, 0, 0)),  # dres_M cotangent
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),  # dx_acc
      ],
      out_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
          _whole((2 * k + P, m)),
          _whole((m,)),
          _whole((1,)),
          _whole((k,)),
          _whole((1,)),
          _whole((k,)),
          _whole((1,)),
          _whole((P,)),
      ],
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(
      xT,
      phi,
      norm_scale,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      perm,
      dH_pre,
      dH_post,
      dres_M,
      dx_acc,
  )


def _pre_apply_bwd(xT, H_pre, d_li, dx_acc, *, bt, vmem, interpret):
  T, k, d = xT.shape

  def kernel(x_ref, hpre_ref, dli_ref, dxacc_ref, dx_ref, dhpre_ref):
    x_val = x_ref[...]
    hpre_val = hpre_ref[...]
    dli_val = dli_ref[...]

    def f(xb, hpre_b):
      Hf = hpre_b.astype(jnp.float32)
      layer_in = sum(
          Hf[:, kk : kk + 1] * xb[:, kk, :].astype(jnp.float32) for kk in range(k)
      )
      return layer_in

    _, vjp = jax.vjp(f, x_val.astype(jnp.float32), hpre_val.astype(jnp.float32))
    dxb, dhpre = vjp(dli_val.astype(jnp.float32))

    dx_ref[...] = (
        dxb.astype(jnp.float32) + dxacc_ref[...].astype(jnp.float32)
    ).astype(dx_ref.dtype)
    dhpre_ref[...] = dhpre.astype(dhpre_ref.dtype)

  cost = pl.CostEstimate(
      flops=int(2 * 2 * T * k * d),
      transcendentals=0,
      bytes_accessed=int(3 * T * k * d * 2 + T * d * 2),
  )
  return pl.pallas_call(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct((T, k, d), xT.dtype),
          jax.ShapeDtypeStruct((T, k), H_pre.dtype),
      ],
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec(
              (bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)
          ),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
          pl.BlockSpec(
              (bt, d), lambda i: (i, 0), pipeline_mode=pl.Buffered(1)
          ),  # d_layer_in cotangent
          pl.BlockSpec(
              (bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)
          ),
      ],  # dx_acc
      out_specs=[
          pl.BlockSpec(
              (bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)
          ),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
      ],
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, H_pre, d_li, dx_acc)


def _post_apply_bwd_body_act(resm_ref, hpost_ref, do_ref, dx_ref, dlo_ref):
  res_M_val = resm_ref[...]
  H_post_val = hpost_ref[...]
  d_out_val = do_ref[...]
  dx_ref[...] = jnp.einsum(
      "tkj,tjd->tkd",
      res_M_val.astype(d_out_val.dtype),
      d_out_val,
      preferred_element_type=f32,
  ).astype(dx_ref.dtype)
  dlo_ref[...] = jnp.sum(
      d_out_val.astype(f32) * H_post_val.astype(f32)[:, :, None], axis=1
  ).astype(dlo_ref.dtype)


def _post_apply_bwd_body_weight(x_ref, lo_ref, do_ref, dhpost_ref, dresm_ref):
  x_val = x_ref[...]
  lo_val = lo_ref[...]
  d_out_val = do_ref[...]
  dhpost_ref[...] = jnp.sum(
      d_out_val.astype(f32) * lo_val.astype(f32)[:, None, :], axis=-1
  ).astype(dhpost_ref.dtype)
  dresm_ref[...] = jnp.einsum(
      "tjd,tkd->tkj",
      d_out_val.astype(x_val.dtype),
      x_val.astype(x_val.dtype),
      preferred_element_type=f32,
  ).astype(dresm_ref.dtype)


def _post_apply_bwd_act(res_M, H_post, d_out, *, bt, vmem, interpret, dlo_dtype=None):
  if dlo_dtype is None:
    dlo_dtype = d_out.dtype
  T, k, d = d_out.shape
  return pl.pallas_call(
      _post_apply_bwd_body_act,
      out_shape=[
          jax.ShapeDtypeStruct((T, k, d), d_out.dtype),  # dx
          jax.ShapeDtypeStruct((T, d), dlo_dtype),
      ],  # d layer_out
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec((bt, k, k), lambda i: (i, 0, 0)),  # res_M
          pl.BlockSpec((bt, k), lambda i: (i, 0)),  # H_post
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
      ],  # d_out
      out_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
          pl.BlockSpec((bt, d), lambda i: (i, 0)),
      ],
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(res_M, H_post, d_out)


def _post_apply_bwd_weight(xT, lo, d_out, *, bt, vmem, interpret):
  T, k, d = xT.shape
  return pl.pallas_call(
      _post_apply_bwd_body_weight,
      out_shape=[
          jax.ShapeDtypeStruct((T, k), xT.dtype),  # d H_post
          jax.ShapeDtypeStruct((T, k, k), xT.dtype),
      ],  # d res_M
      grid=(T // bt,),
      in_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),  # x
          pl.BlockSpec((bt, d), lambda i: (i, 0)),  # layer_out
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0)),
      ],  # d_out
      out_specs=[
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
          pl.BlockSpec((bt, k, k), lambda i: (i, 0, 0)),
      ],
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, lo, d_out)


# ======================================================================================
# Shard-mapped wrappers for Pallas kernels to handle SPMD partitioning
# ======================================================================================
def coeff_fwd_sharded(
    x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm,
    *, bt, vmem, interpret, mesh, rules
):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)
  print("MHC_SHARD: rules =", rules)
  print("MHC_SHARD: mesh =", mesh)
  print("MHC_SHARD: x_spec =", x_spec)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )
  res_M_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None, None), mesh, rules
  )

  def local_fn(
      x_local,
      phi_local,
      norm_scale_local,
      ps_local,
      pb_local,
      qs_local,
      qb_local,
      rs_local,
      rb_local,
      perm_local,
  ):
    print("MHC_SHARD: x_local shape =", x_local.shape)
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)

    H_pre_l, H_post_l, res_M_l = _coeff_fwd(
        xT_local,
        phi_local,
        norm_scale_local,
        ps_local,
        pb_local,
        qs_local,
        qb_local,
        rs_local,
        rb_local,
        perm_local,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
    )

    H_pre_l = H_pre_l.reshape(b_l, s_l, k)
    H_post_l = H_post_l.reshape(b_l, s_l, k)
    res_M_l = res_M_l.reshape(b_l, s_l, k, k)
    return H_pre_l, H_post_l, res_M_l

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, None, None, None, None, None, None, None, None, None),
      out_specs=(H_spec, H_spec, res_M_spec),
      check_vma=False,
  )(x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm)


def pre_apply_fwd_sharded(x, H_pre, *, bt, vmem, interpret, mesh, rules):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)
  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )

  layer_in_logical_spec = ("activation_batch", "activation_length", None)
  layer_in_spec = logical_to_mesh_axes(layer_in_logical_spec, mesh, rules)

  def local_fn(x_local, H_pre_local):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    H_pre_local_flat = H_pre_local.reshape(T_l, k)

    layer_in_local = _pre_apply_fwd(
        xT_local, H_pre_local_flat, bt=bt, vmem=vmem, interpret=interpret
    )
    return layer_in_local.reshape(b_l, s_l, d)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, H_spec),
      out_specs=layer_in_spec,
      check_vma=False,
  )(x, H_pre)


def post_apply_fwd_sharded(
    x, lo, H_post, res_M, *, bt, vmem, interpret, mesh, rules
):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )
  res_M_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None, None), mesh, rules
  )

  def local_fn(x_local, lo_local, H_post_local, res_M_local):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    lo_local_flat = lo_local.reshape(T_l, d)
    H_post_local_flat = H_post_local.reshape(T_l, k)
    res_M_local_flat = res_M_local.reshape(T_l, k, k)

    out_local = _post_apply_fwd(
        xT_local,
        lo_local_flat,
        H_post_local_flat,
        res_M_local_flat,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
    )
    return out_local.reshape(b_l, s_l, k, d)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, lo_spec, H_spec, res_M_spec),
      out_specs=x_spec,
      check_vma=False,
  )(x, lo, H_post, res_M)


def post_apply_bwd_act_sharded(
    res_M, H_post, dO, *, bt, vmem, interpret, mesh, rules, dlo_dtype=None
):
  if dlo_dtype is None:
    dlo_dtype = dO.dtype
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )
  res_M_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None, None), mesh, rules
  )

  def local_fn(res_M_local, H_post_local, dO_local):
    b_l, s_l, k, d = dO_local.shape
    T_l = b_l * s_l
    res_M_local_flat = res_M_local.reshape(T_l, k, k)
    H_post_local_flat = H_post_local.reshape(T_l, k)
    dO_local_flat = dO_local.reshape(T_l, k, d)

    dx_local, dlo_local = _post_apply_bwd_act(
        res_M_local_flat,
        H_post_local_flat,
        dO_local_flat,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
        dlo_dtype=dlo_dtype,
    )
    return dx_local.reshape(b_l, s_l, k, d), dlo_local.reshape(b_l, s_l, d)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(res_M_spec, H_spec, None if x_spec is None else x_spec),
      out_specs=(x_spec, lo_spec),
      check_vma=False,
  )(res_M, H_post, dO)


def pre_apply_bwd_sharded(
    x, H_pre, d_layer_in, dx_acc, *, bt, vmem, interpret, mesh, rules
):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )

  def local_fn(x_local, H_pre_local, d_layer_in_local, dx_acc_local):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    H_pre_local_flat = H_pre_local.reshape(T_l, k)
    d_layer_in_local_flat = d_layer_in_local.reshape(T_l, d)
    dx_acc_local_flat = dx_acc_local.reshape(T_l, k, d)

    dx_acc_out_local, dH_pre_local = _pre_apply_bwd(
        xT_local,
        H_pre_local_flat,
        d_layer_in_local_flat,
        dx_acc_local_flat,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
    )
    return dx_acc_out_local.reshape(b_l, s_l, k, d), dH_pre_local.reshape(
        b_l, s_l, k
    )

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, H_spec, lo_spec, None if x_spec is None else x_spec),
      out_specs=(x_spec, H_spec),
      check_vma=False,
  )(x, H_pre, d_layer_in, dx_acc)


def post_apply_bwd_weight_sharded(
    x, layer_out, dO, *, bt, vmem, interpret, mesh, rules
):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )
  res_M_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None, None), mesh, rules
  )

  def local_fn(x_local, layer_out_local, dO_local):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    layer_out_local_flat = layer_out_local.reshape(T_l, d)
    dO_local_flat = dO_local.reshape(T_l, k, d)

    dH_post_local, dres_M_local = _post_apply_bwd_weight(
        xT_local,
        layer_out_local_flat,
        dO_local_flat,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
    )
    return dH_post_local.reshape(b_l, s_l, k), dres_M_local.reshape(
        b_l, s_l, k, k
    )

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, lo_spec, None if x_spec is None else x_spec),
      out_specs=(H_spec, res_M_spec),
      check_vma=False,
  )(x, layer_out, dO)
def coeff_bwd_sharded(
    x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm,
    dH_pre, dH_post, dres_M, dx_acc,
    *, bt, vmem, interpret, mesh, rules
):
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None), mesh, rules
  )
  res_M_spec = logical_to_mesh_axes(
      ("activation_batch", "activation_length", None, None), mesh, rules
  )

  # Extract mapped axes for reduction
  mapped_axes = tuple(ax for ax in x_spec if ax is not None) if x_spec else ()

  def local_fn(
      x_local,
      phi_local,
      norm_scale_local,
      ps_local,
      pb_local,
      qs_local,
      qb_local,
      rs_local,
      rb_local,
      perm_local,
      dH_pre_local,
      dH_post_local,
      dres_M_local,
      dx_acc_local,
  ):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    dH_pre_local_flat = dH_pre_local.reshape(T_l, k)
    dH_post_local_flat = dH_post_local.reshape(T_l, k)
    dres_M_local_flat = dres_M_local.reshape(T_l, k, k)
    dx_acc_local_flat = dx_acc_local.reshape(T_l, k, d)

    (
        dx_local,
        dphi_local,
        dns_local,
        dps_local,
        dpb_local,
        dqs_local,
        dqb_local,
        drs_local,
        drb_local,
    ) = _coeff_bwd(
        xT_local,
        phi_local,
        norm_scale_local,
        ps_local,
        pb_local,
        qs_local,
        qb_local,
        rs_local,
        rb_local,
        perm_local,
        dH_pre_local_flat,
        dH_post_local_flat,
        dres_M_local_flat,
        dx_acc_local_flat,
        bt=bt,
        vmem=vmem,
        interpret=interpret,
    )

    # Reshape sharded output
    dx_local = dx_local.reshape(b_l, s_l, k, d)

    # Grid dimension has already been accumulated in the kernel
    dphi_sum = dphi_local
    dns_sum = dns_local
    dps_sum = dps_local
    dpb_sum = dpb_local
    dqs_sum = dqs_local
    dqb_sum = dqb_local
    drs_sum = drs_local
    drb_sum = drb_local

    # Reduce global outputs across mapped axes
    if mapped_axes:
      ax_idx = jax.lax.axis_index(mapped_axes[0])
      jax.debug.print("DEV {idx}: dpb_sum = {x}", idx=ax_idx, x=dpb_sum)
      dphi_global = jax.lax.psum(dphi_sum, axis_name=mapped_axes)
      dns_global = jax.lax.psum(dns_sum, axis_name=mapped_axes)
      dps_global = jax.lax.psum(dps_sum, axis_name=mapped_axes)
      dpb_global = jax.lax.psum(dpb_sum, axis_name=mapped_axes)
      dqs_global = jax.lax.psum(dqs_sum, axis_name=mapped_axes)
      dqb_global = jax.lax.psum(dqb_sum, axis_name=mapped_axes)
      drs_global = jax.lax.psum(drs_sum, axis_name=mapped_axes)
      drb_global = jax.lax.psum(drb_sum, axis_name=mapped_axes)
      jax.debug.print("DEV {idx}: dpb_global = {x}", idx=ax_idx, x=dpb_global)
    else:
      jax.debug.print("DEV 0: dpb_sum = {x}", x=dpb_sum)
      dphi_global = dphi_sum
      dns_global = dns_sum
      dps_global = dps_sum
      dpb_global = dpb_sum
      dqs_global = dqs_sum
      dqb_global = dqb_sum
      drs_global = drs_sum
      drb_global = drb_sum

    return (
        dx_local,
        dphi_global,
        dns_global,
        dps_global,
        dpb_global,
        dqs_global,
        dqb_global,
        drs_global,
        drb_global,
    )

  in_specs = (
      None if x_spec is None else x_spec,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      H_spec,
      H_spec,
      res_M_spec,
      None if x_spec is None else x_spec,
  )

  out_specs = (
      PartitionSpec() if x_spec is None else x_spec,
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
      PartitionSpec(),
  )

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_vma=False,
  )(
      x,
      phi,
      norm_scale,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      perm,
      dH_pre,
      dH_post,
      dres_M,
      dx_acc,
  )


# ======================================================================================
# Generalized custom VJP wrapper for NNX Modules
# ======================================================================================
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))
def mhc_pallas_vjp_fn(
    mhc_graphdef,
    norm_graphdef,
    branch_graphdef,
    mhc_type,
    block_t,
    deterministic,
    model_mode,
    out_sharding,
    slot,
    # Differentiable/dynamic arguments (PyTree leaves):
    mhc_params,
    mhc_other,
    norm_params,
    norm_other,
    branch_params,
    branch_other,
    x,
    perm,
    decoder_segment_ids,
    inputs_positions,
    previous_chunk,
):
  out_tuple, _ = _mhc_pallas_vjp_fwd_impl(
      mhc_graphdef,
      norm_graphdef,
      branch_graphdef,
      mhc_type,
      block_t,
      deterministic,
      model_mode,
      out_sharding,
      slot,
      mhc_params,
      mhc_other,
      norm_params,
      norm_other,
      branch_params,
      branch_other,
      x,
      perm,
      decoder_segment_ids,
      inputs_positions,
      previous_chunk,
  )
  return out_tuple


def _mhc_pallas_vjp_fwd_impl(
    mhc_graphdef,
    norm_graphdef,
    branch_graphdef,
    mhc_type,
    block_t,
    deterministic,
    model_mode,
    out_sharding,
    slot,
    mhc_params,
    mhc_other,
    norm_params,
    norm_other,
    branch_params,
    branch_other,
    x,
    perm,
    decoder_segment_ids,
    inputs_positions,
    previous_chunk,
):
  # 1. Reconstruct modules
  mhc_local = nnx.merge(mhc_graphdef, mhc_params, mhc_other)
  norm_local = nnx.merge(norm_graphdef, norm_params, norm_other)
  branch_local = nnx.merge(branch_graphdef, branch_params, branch_other)

  # 2. Extract weights
  norm_scale = mhc_local.mhc_norm.scale.value.astype(mhc_local.dtype)
  pre_alpha = mhc_local.pre_alpha.value.astype(mhc_local.dtype)
  pre_beta = mhc_local.pre_beta.value.astype(mhc_local.dtype)
  pre_s = mhc_local.pre_alpha_scale.value.astype(mhc_local.dtype)
  post_alpha = mhc_local.post_alpha.value.astype(mhc_local.dtype)
  post_beta = mhc_local.post_beta.value.astype(mhc_local.dtype)
  post_s = mhc_local.post_alpha_scale.value.astype(mhc_local.dtype)
  res_alpha = mhc_local.res_alpha.value.astype(mhc_local.dtype)
  res_beta = mhc_local.res_beta.value.astype(mhc_local.dtype)
  res_s = mhc_local.res_alpha_scale.value.astype(mhc_local.dtype)

  print("MHC_VJP_DEBUG: x.dtype =", x.dtype)
  print("MHC_VJP_DEBUG: mhc_local.dtype =", mhc_local.dtype)
  print("MHC_VJP_DEBUG: phi.dtype =", (jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1).T).dtype)

  b, s, k, d = x.shape
  phi = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1).T
  print("MHC_SHAPE_DEBUG: norm_scale shape =", norm_scale.shape)
  print("MHC_SHAPE_DEBUG: pre_alpha shape =", pre_alpha.shape)
  print("MHC_SHAPE_DEBUG: phi shape =", phi.shape)

  mesh = mhc_local.mesh
  rules = mhc_local.config.logical_axis_rules

  # 3. Run Pallas Fwd
  H_pre, H_post, res_M = coeff_fwd_sharded(
      x,
      phi,
      norm_scale,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      perm,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )

  layer_in = pre_apply_fwd_sharded(
      x,
      H_pre,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )

  # 4. Norm + Branch
  layer_in_norm = norm_local(layer_in)

  load_balance_loss = 0.0
  moe_bias_updates = None

  if mhc_type == HyperConnectionType.ATTENTION:
    layer_out, _ = branch_local(
        inputs_q=layer_in_norm,
        inputs_kv=layer_in_norm,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=inputs_positions,
        deterministic=deterministic,
        model_mode=model_mode,
        out_sharding=out_sharding,
        previous_chunk=previous_chunk,
        slot=slot,
    )
  elif mhc_type == HyperConnectionType.MLP_DENSE:
    layer_out = branch_local(
        inputs=layer_in_norm,
        deterministic=deterministic,
        out_sharding=out_sharding,
    )
  elif mhc_type == HyperConnectionType.MLP_MOE:
    layer_out, load_balance_loss, moe_bias_updates = branch_local(
        inputs=layer_in_norm,
        out_sharding=out_sharding,
    )
  else:
    raise ValueError(f"Unsupported mhc_type for Pallas: {mhc_type}")


  # 5. Post Pallas Fwd
  out = post_apply_fwd_sharded(
      x,
      layer_out,
      H_post,
      res_M,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )

  # Extract updated mutable states (Cache, etc.)
  _, _, updated_norm_other = nnx.split(norm_local, nnx.Param, ...)
  _, _, updated_branch_other = nnx.split(branch_local, nnx.Param, ...)

  residuals = (
      mhc_params,
      mhc_other,
      norm_params,
      norm_other,
      branch_params,
      branch_other,
      x,
      phi,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      H_pre,
      H_post,
      res_M,
      layer_in,
      layer_in_norm,
      layer_out,
      b,
      s,
      k,
      d,
      perm,
      decoder_segment_ids,
      inputs_positions,
      previous_chunk,
  )
  return (
      out,
      load_balance_loss,
      updated_norm_other,
      updated_branch_other,
      moe_bias_updates,
  ), residuals


def _mhc_pallas_vjp_bwd_impl(
    mhc_graphdef,
    norm_graphdef,
    branch_graphdef,
    mhc_type,
    block_t,
    deterministic,
    model_mode,
    out_sharding,
    slot,
    residuals,
    cotangents,
):
  d_out, d_load_balance_loss, _, _, _ = cotangents

  (
      mhc_params,
      mhc_other,
      norm_params,
      norm_other,
      branch_params,
      branch_other,
      x,
      phi,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      H_pre,
      H_post,
      res_M,
      layer_in,
      layer_in_norm,
      layer_out,
      b,
      s,
      k,
      d,
      perm,
      decoder_segment_ids,
      inputs_positions,
      previous_chunk,
  ) = residuals

  mhc_local = nnx.merge(mhc_graphdef, mhc_params, mhc_other)
  mesh = mhc_local.mesh
  rules = mhc_local.config.logical_axis_rules

  # 1. Post Apply Bwd Act
  dx_acc, d_layer_out = post_apply_bwd_act_sharded(
      res_M,
      H_post,
      d_out,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
      dlo_dtype=layer_out.dtype,
  )
  print("MHC_BWD_DEBUG: d_out shape =", d_out.shape, "sharding =", d_out.sharding, "min/max/zeros =", jnp.min(jnp.abs(d_out)), jnp.max(jnp.abs(d_out)), jnp.sum(d_out == 0.0))
  print("MHC_BWD_DEBUG: d_layer_out shape =", d_layer_out.shape, "sharding =", d_layer_out.sharding, "min/max/zeros =", jnp.min(jnp.abs(d_layer_out)), jnp.max(jnp.abs(d_layer_out)), jnp.sum(d_layer_out == 0.0))
  print("MHC_BWD_DEBUG: dx_acc shape =", dx_acc.shape, "sharding =", dx_acc.sharding, "min/max/zeros =", jnp.min(jnp.abs(dx_acc)), jnp.max(jnp.abs(dx_acc)), jnp.sum(dx_acc == 0.0))



  # 2. Branch Bwd (Stateless JAX VJP)
  def branch_fwd_stateless(params_in, inputs_in):
    branch_local = nnx.merge(branch_graphdef, params_in, branch_other)
    if mhc_type == HyperConnectionType.ATTENTION:
      layer_out_local, _ = branch_local(
          inputs_q=inputs_in,
          inputs_kv=inputs_in,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=inputs_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          out_sharding=out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
      )
      return layer_out_local
    elif mhc_type == HyperConnectionType.MLP_DENSE:
      layer_out_local = branch_local(
          inputs=inputs_in,
          deterministic=deterministic,
          out_sharding=out_sharding,
      )
      return layer_out_local
    elif mhc_type == HyperConnectionType.MLP_MOE:
      layer_out_local, lbl, _ = branch_local(
          inputs=inputs_in,
          out_sharding=out_sharding,
      )
      return layer_out_local, lbl

  _, branch_vjp = jax.vjp(branch_fwd_stateless, branch_params, layer_in_norm)

  if mhc_type == HyperConnectionType.MLP_MOE:
    d_branch_params, d_layer_in_norm = branch_vjp(
        (d_layer_out, d_load_balance_loss)
    )
  else:
    d_branch_params, d_layer_in_norm = branch_vjp(d_layer_out)
  print("MHC_BWD_DEBUG: d_layer_in_norm shape =", d_layer_in_norm.shape, "sharding =", d_layer_in_norm.sharding, "min/max/zeros =", jnp.min(jnp.abs(d_layer_in_norm)), jnp.max(jnp.abs(d_layer_in_norm)), jnp.sum(d_layer_in_norm == 0.0))
  for path, val in d_branch_params.flat_state():
    print(f"MHC_BWD_DEBUG: d_branch_params {path} shape = {val.shape} sharding = {val.sharding} min/max/zeros = {jnp.min(jnp.abs(val))} {jnp.max(jnp.abs(val))} {jnp.sum(val == 0.0)}")






  # 3. Norm Bwd (Stateless JAX VJP)
  def norm_fwd_stateless(params_in, inputs_in):
    norm_local = nnx.merge(norm_graphdef, params_in, norm_other)
    return norm_local(inputs_in)

  _, norm_vjp = jax.vjp(norm_fwd_stateless, norm_params, layer_in)
  d_norm_params, d_layer_in = norm_vjp(d_layer_in_norm)
  print("MHC_BWD_DEBUG: d_layer_in shape =", d_layer_in.shape, "sharding =", d_layer_in.sharding, "min/max/zeros =", jnp.min(jnp.abs(d_layer_in)), jnp.max(jnp.abs(d_layer_in)), jnp.sum(d_layer_in == 0.0))
  for path, val in d_norm_params.flat_state():
    print(f"MHC_BWD_DEBUG: d_norm_params {path} shape = {val.shape} sharding = {val.sharding} min/max/zeros = {jnp.min(jnp.abs(val))} {jnp.max(jnp.abs(val))} {jnp.sum(val == 0.0)}")






  # 4. Pre Apply Bwd
  dx_acc, dH_pre = pre_apply_bwd_sharded(
      x,
      H_pre,
      d_layer_in,
      dx_acc,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )
  print("MHC_BWD_DEBUG: dH_pre shape =", dH_pre.shape, "sharding =", dH_pre.sharding, "min/max/zeros =", jnp.min(jnp.abs(dH_pre)), jnp.max(jnp.abs(dH_pre)), jnp.sum(dH_pre == 0.0))
  print("MHC_BWD_DEBUG: dx_acc (after pre) shape =", dx_acc.shape, "sharding =", dx_acc.sharding, "min/max/zeros =", jnp.min(jnp.abs(dx_acc)), jnp.max(jnp.abs(dx_acc)), jnp.sum(dx_acc == 0.0))



  # 5. Post Apply Bwd Weight
  dH_post, dres_M = post_apply_bwd_weight_sharded(
      x,
      layer_out,
      d_out,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )
  print("MHC_BWD_DEBUG: dH_post shape =", dH_post.shape, "sharding =", dH_post.sharding, "min/max/zeros =", jnp.min(jnp.abs(dH_post)), jnp.max(jnp.abs(dH_post)), jnp.sum(dH_post == 0.0))
  print("MHC_BWD_DEBUG: dres_M shape =", dres_M.shape, "sharding =", dres_M.sharding, "min/max/zeros =", jnp.min(jnp.abs(dres_M)), jnp.max(jnp.abs(dres_M)), jnp.sum(dres_M == 0.0))



  # 6. Coeff Bwd
  norm_scale = mhc_local.mhc_norm.scale.value.astype(mhc_local.dtype)

  dx, dphi, dns, dps, dpb, dqs, dqb, drs, drb = coeff_bwd_sharded(
      x,
      phi,
      norm_scale,
      pre_s,
      pre_beta,
      post_s,
      post_beta,
      res_s,
      res_beta,
      perm,
      dH_pre,
      dH_post,
      dres_M,
      dx_acc,
      bt=block_t,
      vmem=VMEM_LIMIT_BYTES,
      interpret=False,
      mesh=mesh,
      rules=rules,
  )
  print("MHC_BWD_DEBUG: dx shape =", dx.shape, "sharding =", dx.sharding, "min/max/zeros =", jnp.min(jnp.abs(dx)), jnp.max(jnp.abs(dx)), jnp.sum(dx == 0.0))
  print("MHC_BWD_DEBUG: dphi shape =", dphi.shape, "sharding =", dphi.sharding, "min/max/zeros =", jnp.min(jnp.abs(dphi)), jnp.max(jnp.abs(dphi)), jnp.sum(dphi == 0.0))
  print("MHC_BWD_DEBUG: dns shape =", dns.shape, "sharding =", dns.sharding, "min/max/zeros =", jnp.min(jnp.abs(dns)), jnp.max(jnp.abs(dns)), jnp.sum(dns == 0.0))




  # 7. Split dphi to get alpha grads (no fold_vjp needed!)
  # dphi shape: (2*k+P, m) -> dphi.T shape: (m, 2*k+P)
  dphi_T = dphi.T
  d_pre_alpha = dphi_T[:, :k]
  d_post_alpha = dphi_T[:, k : 2 * k]
  d_res_alpha = dphi_T[:, 2 * k :]
  d_norm_scale = dns

  # 8. Rebuild d_mhc_params State
  grad_dict = {
      path: jnp.zeros_like(val) for path, val in mhc_params.flat_state()
  }

  grad_dict[("mhc_norm", "scale")] = d_norm_scale
  grad_dict[("pre_alpha",)] = d_pre_alpha
  grad_dict[("pre_beta",)] = dpb
  grad_dict[("pre_alpha_scale",)] = dps
  grad_dict[("post_alpha",)] = d_post_alpha
  grad_dict[("post_beta",)] = dqb
  grad_dict[("post_alpha_scale",)] = dqs
  grad_dict[("res_alpha",)] = d_res_alpha
  grad_dict[("res_beta",)] = drb
  grad_dict[("res_alpha_scale",)] = drs

  d_mhc_params = nnx.State.from_flat_path(grad_dict.items())

  # Zeros for other outputs
  d_mhc_other = jax.tree_util.tree_map(jnp.zeros_like, mhc_other)
  d_norm_other = jax.tree_util.tree_map(jnp.zeros_like, norm_other)
  d_branch_other = jax.tree_util.tree_map(jnp.zeros_like, branch_other)

  return (
      d_mhc_params,
      d_mhc_other,
      d_norm_params,
      d_norm_other,
      d_branch_params,
      d_branch_other,
      dx,
      jnp.zeros_like(perm),
      jnp.zeros_like(decoder_segment_ids)
      if decoder_segment_ids is not None
      else None,
      jnp.zeros_like(inputs_positions) if inputs_positions is not None else None,
      jnp.zeros_like(previous_chunk) if previous_chunk is not None else None,
  )


mhc_pallas_vjp_fn.defvjp(_mhc_pallas_vjp_fwd_impl, _mhc_pallas_vjp_bwd_impl)


# ======================================================================================
# Main entry point called by the Layer
# ======================================================================================
def run_mhc_pallas(
    mhc_layer,
    norm_fn: Callable,
    branch_fn: Callable,
    x: Array,
    mhc_type: HyperConnectionType,
    **kwargs,
) -> tuple[Array, dict]:
  """Runs the mHC Pallas kernel with custom VJP.

  Reconstructs modules functionally to avoid side effects in custom VJP.
  """
  # 1. Split states
  _, mhc_params, mhc_other = nnx.split(mhc_layer, nnx.Param, ...)
  _, norm_params, norm_other = nnx.split(norm_fn, nnx.Param, ...)
  _, branch_params, branch_other = nnx.split(branch_fn, nnx.Param, ...)

  mhc_graphdef = nnx.split(mhc_layer)[0]
  norm_graphdef = nnx.split(norm_fn)[0]
  branch_graphdef = nnx.split(branch_fn)[0]

  # 2. Extract static/hashable configs
  deterministic = kwargs.get("deterministic", True)
  model_mode = kwargs.get("model_mode", "train")
  out_sharding = kwargs.get("out_sharding", None)
  slot = kwargs.get("slot", None)

  # 3. Extract dynamic (differentiable) tensors
  decoder_segment_ids = kwargs.get("decoder_segment_ids", None)
  inputs_positions = kwargs.get("inputs_positions", None)
  previous_chunk = kwargs.get("previous_chunk", None)

  # Block size and permutations
  block_t = mhc_layer.config.mhc_pallas_block_t
  perm = mhc_layer.permutation_matrices

  # 4. Call VJP function
  # We return updated other states to propagate mutable variables (like Cache)
  (
      out,
      load_balance_loss,
      updated_norm_other,
      updated_branch_other,
      moe_bias_updates,
  ) = mhc_pallas_vjp_fn(
      mhc_graphdef,
      norm_graphdef,
      branch_graphdef,
      mhc_type,
      block_t,
      deterministic,
      model_mode,
      out_sharding,
      slot,
      mhc_params,
      mhc_other,
      norm_params,
      norm_other,
      branch_params,
      branch_other,
      x,
      perm,
      decoder_segment_ids,
      inputs_positions,
      previous_chunk,
  )

  nnx.update(norm_fn, updated_norm_other)
  nnx.update(branch_fn, updated_branch_other)

  # Reconstruct metadata dict
  metadata = {}
  if mhc_type == HyperConnectionType.MLP_MOE:
    metadata["load_balance_loss"] = load_balance_loss
    metadata["moe_bias_updates"] = moe_bias_updates

  return out, metadata
