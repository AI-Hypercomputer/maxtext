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

"""Common helper functions and constants for mHC Pallas kernels."""

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

bf16 = jnp.bfloat16
f32 = jnp.float32

VMEM_LIMIT_BYTES = 63 * 1024 * 1024  # 63MB VMEM limit to fit in TPU v5e usable VMEM
EPS = 1e-6  # epsilon for RMSNorm


def _whole(shape):
  """full-array BlockSpec (index_map -> 0): weights load once, stay VMEM-resident."""
  return pl.BlockSpec(shape, lambda i: tuple(0 for _ in shape))


def fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha):
  """Folds RMSNorm scale weights into alpha parameters."""
  concat_alpha = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)
  phi = norm_scale.astype(f32)[:, None] * concat_alpha.astype(f32)
  return phi.T


@jax.custom_vjp
def mhc_coeffs(x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm):
  """Computes coefficients for the manifold-constrained hyper connections.

  Args:
    x: Input activations of shape (T, k, d).
    phi: Concatenated pre, post, and res mapping weight matrices, of shape (2*k + P, k * d) or (k * d, 2*k + P).
    norm_scale: RMSNorm scale weights, shape (k * d,).
    pre_s: Sigmoid scale parameter for pre mapping, shape (1,).
    pre_beta: Bias parameter for pre mapping, shape (k,).
    post_s: Sigmoid scale parameter for post mapping, shape (1,).
    post_beta: Bias parameter for post mapping, shape (k,).
    res_s: Softmax scale parameter for res mapping, shape (1,).
    res_beta: Bias parameter for res mapping, shape (P,).
    perm: Permutation matrices of shape (P, k, k).

  Returns:
    H_pre: Pre-mapping coefficients of shape (T, k).
    H_post: Post-mapping coefficients of shape (T, k).
    res_M: Blended residual mapping matrices of shape (T, k, k).
  """
  T, k, d = x.shape
  m = k * d
  P = perm.shape[0]
  xf = x.reshape(T, m)

  # Normalize first (Formulation 1)
  xf32 = xf.astype(f32)
  mean2 = jnp.mean(xf32 * xf32, axis=-1, keepdims=True)
  r_inv = jax.lax.rsqrt(mean2 + EPS).astype(x.dtype)
  xf_norm = (xf * r_inv) * norm_scale.astype(x.dtype)

  # Project normalized input: (T, m) @ (2*k + P, m)^T -> (T, 2*k + P)
  # We contract the 'm' dimension (axis 1 of xf_norm and axis 1 of phi).
  q = jax.lax.dot_general(
      xf_norm,
      phi.astype(x.dtype),
      (((1,), (1,)), ((), (()))),
      preferred_element_type=f32,
  ).astype(x.dtype)

  h = q
  h_pre, h_post, h_res = h[:, :k], h[:, k : 2 * k], h[:, 2 * k :]
  H_pre = jax.nn.sigmoid(pre_s.astype(x.dtype) * h_pre + pre_beta.astype(x.dtype))
  H_post = 2.0 * jax.nn.sigmoid(post_s.astype(x.dtype) * h_post + post_beta.astype(x.dtype))
  weights_input = res_s.astype(x.dtype) * h_res + res_beta.astype(x.dtype)
  weights = jax.nn.softmax(weights_input.astype(f32), axis=-1).astype(x.dtype)
  res_M = (
      jnp.dot(
          weights,
          perm.reshape(P, k * k).astype(x.dtype),
          preferred_element_type=f32,
      )
      .astype(x.dtype)
      .reshape(T, k, k)
  )
  return H_pre.astype(x.dtype), H_post.astype(x.dtype), res_M.astype(x.dtype)


def mhc_coeffs_fwd(x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm):
  """Forward VJP handler for mhc_coeffs."""
  H_pre, H_post, res_M = mhc_coeffs(x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm)
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
  """Backward VJP handler for mhc_coeffs."""
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
  H_post_half = jax.nn.sigmoid(post_s.astype(xf.dtype) * h_post + post_beta.astype(xf.dtype))
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
  d_weights = jnp.dot(dy_resM_flat, perm_flat.T, preferred_element_type=f32).astype(xf.dtype)

  d_weights_f32 = d_weights.astype(f32)
  weights_f32 = weights.astype(f32)
  d_weights_input = weights_f32 * (d_weights_f32 - jnp.sum(d_weights_f32 * weights_f32, axis=-1, keepdims=True))

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
  """Applies pre-mapping to activations by contracting the expansion dimension."""
  Hf = H_pre.astype(f32)
  layer_in = jnp.einsum("tk,tkd->td", Hf, x.astype(f32), preferred_element_type=f32)
  return layer_in.astype(x.dtype)


def mhc_post_apply(x, layer_out, H_post, res_M):
  """Applies post-mapping to activations, blending residual and post-layer connections."""
  res_mix = jnp.einsum("tkj,tkd->tjd", res_M.astype(x.dtype), x, preferred_element_type=f32)
  post_mix = H_post.astype(f32)[:, :, None] * layer_out.astype(f32)[:, None, :]
  return (res_mix + post_mix).astype(x.dtype)
