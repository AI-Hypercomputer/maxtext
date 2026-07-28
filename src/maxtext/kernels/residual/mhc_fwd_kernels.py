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
from typing import Callable

from flax import nnx
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.common.common_types import Array, HyperConnectionType
from maxtext.utils.sharding import logical_to_mesh_axes

from maxtext.kernels.residual.mhc_common import (
    _whole,
    mhc_coeffs,
    mhc_pre_apply,
    mhc_post_apply,
    VMEM_LIMIT_BYTES,
)

# pylint: disable=unused-import
from maxtext.kernels.residual.mhc_bwd_kernels import (
    _mhc_pallas_vjp_bwd_impl,
    pre_apply_bwd_sharded,
    coeff_bwd_sharded,
    post_apply_bwd_act_sharded,
    post_apply_bwd_weight_sharded,
)
# pylint: enable=unused-import


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
  """Pallas kernel for computing MHC coefficients."""
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
  """Pallas kernel for computing pre-mapping."""
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
  """Pallas kernel for computing post-mapping."""
  T, k, d = xT.shape

  def kernel(x_ref, lo_ref, hpost_ref, resm_ref, o_ref):
    o_ref[...] = mhc_post_apply(x_ref[...], lo_ref[...], hpost_ref[...], resm_ref[...])

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


def coeff_fwd_sharded(
    x, phi, norm_scale, pre_s, pre_beta, post_s, post_beta, res_s, res_beta, perm, *, bt, vmem, interpret, mesh, rules
):
  """Shard-mapped wrapper for _coeff_fwd."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)
  res_M_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None, None), mesh, rules)

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
  """Shard-mapped wrapper for _pre_apply_fwd."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)
  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)

  layer_in_logical_spec = ("activation_batch", "activation_length", None)
  layer_in_spec = logical_to_mesh_axes(layer_in_logical_spec, mesh, rules)

  def local_fn(x_local, H_pre_local):
    b_l, s_l, k, d = x_local.shape
    T_l = b_l * s_l
    xT_local = x_local.reshape(T_l, k, d)
    H_pre_local_flat = H_pre_local.reshape(T_l, k)

    layer_in_local = _pre_apply_fwd(xT_local, H_pre_local_flat, bt=bt, vmem=vmem, interpret=interpret)
    return layer_in_local.reshape(b_l, s_l, d)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, H_spec),
      out_specs=layer_in_spec,
      check_vma=False,
  )(x, H_pre)


def post_apply_fwd_sharded(x, lo, H_post, res_M, *, bt, vmem, interpret, mesh, rules):
  """Shard-mapped wrapper for _post_apply_fwd."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)
  res_M_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None, None), mesh, rules)

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
  """Custom VJP wrapper function for computing gradients with NNX Modules."""
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
  """Forward implementation for the custom VJP."""
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

  b, s, k, d = x.shape
  phi = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1).T

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
