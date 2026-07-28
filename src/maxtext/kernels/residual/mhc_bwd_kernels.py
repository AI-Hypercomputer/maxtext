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

"""Backward Pallas kernels and VJP implementation for mHC."""

from flax import nnx
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from maxtext.common.common_types import HyperConnectionType
from maxtext.utils.sharding import logical_to_mesh_axes

from maxtext.kernels.residual.mhc_common import (
    f32,
    _whole,
    mhc_coeffs,
    mhc_pre_apply,
    VMEM_LIMIT_BYTES,
)


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
  """Pallas kernel for computing MHC coefficient gradients."""
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
    dx_ref[...] = dxb.astype(dx_ref.dtype) + dxacc_ref[...].astype(dx_ref.dtype)

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
  """Pallas kernel for computing pre-mapping gradients."""
  T, k, d = xT.shape

  def kernel(x_ref, hpre_ref, dli_ref, dxacc_ref, dx_ref, dhpre_ref):
    x_val = x_ref[...]
    hpre_val = hpre_ref[...]
    dli_val = dli_ref[...]

    def f(xb, hpre_b):
      return mhc_pre_apply(xb, hpre_b)

    _, vjp = jax.vjp(f, x_val.astype(jnp.float32), hpre_val.astype(jnp.float32))
    dxb, dhpre = vjp(dli_val.astype(jnp.float32))

    dx_ref[...] = (dxb.astype(jnp.float32) + dxacc_ref[...].astype(jnp.float32)).astype(dx_ref.dtype)
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
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
          pl.BlockSpec((bt, d), lambda i: (i, 0), pipeline_mode=pl.Buffered(1)),  # d_layer_in cotangent
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)),
      ],  # dx_acc
      out_specs=[
          pl.BlockSpec((bt, k, d), lambda i: (i, 0, 0), pipeline_mode=pl.Buffered(1)),
          pl.BlockSpec((bt, k), lambda i: (i, 0)),
      ],
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
      interpret=interpret,
  )(xT, H_pre, d_li, dx_acc)


def _post_apply_bwd_body_act(resm_ref, hpost_ref, do_ref, dx_ref, dlo_ref):
  """Helper kernel body for post-apply backward activation gradients."""
  res_M_val = resm_ref[...]
  H_post_val = hpost_ref[...]
  d_out_val = do_ref[...]
  dx_ref[...] = jnp.einsum(
      "tkj,tjd->tkd",
      res_M_val.astype(d_out_val.dtype),
      d_out_val,
      preferred_element_type=f32,
  ).astype(dx_ref.dtype)
  dlo_ref[...] = jnp.sum(d_out_val.astype(f32) * H_post_val.astype(f32)[:, :, None], axis=1).astype(dlo_ref.dtype)


def _post_apply_bwd_body_weight(x_ref, lo_ref, do_ref, dhpost_ref, dresm_ref):
  """Helper kernel body for post-apply backward parameter gradients."""
  x_val = x_ref[...]
  lo_val = lo_ref[...]
  d_out_val = do_ref[...]
  dhpost_ref[...] = jnp.sum(d_out_val.astype(f32) * lo_val.astype(f32)[:, None, :], axis=-1).astype(dhpost_ref.dtype)
  dresm_ref[...] = jnp.einsum(
      "tjd,tkd->tkj",
      d_out_val.astype(x_val.dtype),
      x_val.astype(x_val.dtype),
      preferred_element_type=f32,
  ).astype(dresm_ref.dtype)


def _post_apply_bwd_act(res_M, H_post, d_out, *, bt, vmem, interpret, dlo_dtype=None):
  """Pallas kernel for post-apply backward activation gradients."""
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
  """Pallas kernel for post-apply backward parameter gradients."""
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


def post_apply_bwd_act_sharded(res_M, H_post, dO, *, bt, vmem, interpret, mesh, rules, dlo_dtype=None):
  """Shard-mapped wrapper for _post_apply_bwd_act."""
  if dlo_dtype is None:
    dlo_dtype = dO.dtype
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)
  res_M_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None, None), mesh, rules)

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


def pre_apply_bwd_sharded(x, H_pre, d_layer_in, dx_acc, *, bt, vmem, interpret, mesh, rules):
  """Shard-mapped wrapper for _pre_apply_bwd."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)

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
    return dx_acc_out_local.reshape(b_l, s_l, k, d), dH_pre_local.reshape(b_l, s_l, k)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, H_spec, lo_spec, None if x_spec is None else x_spec),
      out_specs=(x_spec, H_spec),
      check_vma=False,
  )(x, H_pre, d_layer_in, dx_acc)


def post_apply_bwd_weight_sharded(x, layer_out, dO, *, bt, vmem, interpret, mesh, rules):
  """Shard-mapped wrapper for _post_apply_bwd_weight."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  lo_logical_spec = ("activation_batch", "activation_length", None)
  lo_spec = logical_to_mesh_axes(lo_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)
  res_M_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None, None), mesh, rules)

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
    return dH_post_local.reshape(b_l, s_l, k), dres_M_local.reshape(b_l, s_l, k, k)

  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(None if x_spec is None else x_spec, lo_spec, None if x_spec is None else x_spec),
      out_specs=(H_spec, res_M_spec),
      check_vma=False,
  )(x, layer_out, dO)


def coeff_bwd_sharded(
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
    *,
    bt,
    vmem,
    interpret,
    mesh,
    rules,
):
  """Shard-mapped wrapper for _coeff_bwd."""
  x_logical_spec = ("activation_batch", "activation_length", None, None)
  x_spec = logical_to_mesh_axes(x_logical_spec, mesh, rules)

  H_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None), mesh, rules)
  res_M_spec = logical_to_mesh_axes(("activation_batch", "activation_length", None, None), mesh, rules)

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
      dphi_global = jax.lax.psum(dphi_sum, axis_name=mapped_axes)
      dns_global = jax.lax.psum(dns_sum, axis_name=mapped_axes)
      dps_global = jax.lax.psum(dps_sum, axis_name=mapped_axes)
      dpb_global = jax.lax.psum(dpb_sum, axis_name=mapped_axes)
      dqs_global = jax.lax.psum(dqs_sum, axis_name=mapped_axes)
      dqb_global = jax.lax.psum(dqb_sum, axis_name=mapped_axes)
      drs_global = jax.lax.psum(drs_sum, axis_name=mapped_axes)
      drb_global = jax.lax.psum(drb_sum, axis_name=mapped_axes)
    else:
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
  """Backward implementation for the custom VJP."""
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
      _,  # b
      _,  # s
      k,
      _,  # d
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
    else:
      raise ValueError(f"Unsupported mhc_type: {mhc_type}")

  _, branch_vjp = jax.vjp(branch_fwd_stateless, branch_params, layer_in_norm)

  if mhc_type == HyperConnectionType.MLP_MOE:
    d_branch_params, d_layer_in_norm = branch_vjp((d_layer_out, d_load_balance_loss))
  else:
    d_branch_params, d_layer_in_norm = branch_vjp(d_layer_out)

  # 3. Norm Bwd (Stateless JAX VJP)
  def norm_fwd_stateless(params_in, inputs_in):
    norm_local = nnx.merge(norm_graphdef, params_in, norm_other)
    return norm_local(inputs_in)

  _, norm_vjp = jax.vjp(norm_fwd_stateless, norm_params, layer_in)
  d_norm_params, d_layer_in = norm_vjp(d_layer_in_norm)

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

  # 7. Split dphi to get alpha grads (no fold_vjp needed!)
  # dphi shape: (2*k+P, m) -> dphi.T shape: (m, 2*k+P)
  dphi_T = dphi.T
  d_pre_alpha = dphi_T[:, :k]
  d_post_alpha = dphi_T[:, k : 2 * k]
  d_res_alpha = dphi_T[:, 2 * k :]
  d_norm_scale = dns

  # 8. Rebuild d_mhc_params State
  grad_dict = {path: jnp.zeros_like(val) for path, val in mhc_params.flat_state()}

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
      jnp.zeros_like(decoder_segment_ids) if decoder_segment_ids is not None else None,
      jnp.zeros_like(inputs_positions) if inputs_positions is not None else None,
      jnp.zeros_like(previous_chunk) if previous_chunk is not None else None,
  )
