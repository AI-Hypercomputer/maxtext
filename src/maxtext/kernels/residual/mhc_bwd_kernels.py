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

"""Backward Pallas kernels and custom-VJP rules for mHC-lite."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels.residual import mhc_common


def _post_apply_bwd(
    x,
    layer_output,
    h_post,
    residual,
    d_output,
    *,
    block_size,
    feature_block_size,
    vmem_limit_bytes,
    interpret,
):
  """Builds the feature-tiled Pallas call for the post-branch backward pass."""
  tokens, streams, embedding = x.shape
  feature_blocks = embedding // feature_block_size

  def kernel(
      x_ref,
      layer_output_ref,
      h_post_ref,
      residual_ref,
      d_output_ref,
      d_x_ref,
      d_layer_output_ref,
      d_h_post_ref,
      d_residual_ref,
  ):
    feature_block = pl.program_id(1)
    d_output_f32 = d_output_ref[...].astype(jnp.float32)

    # These gradients are independent across feature tiles.
    d_x = jnp.einsum(
        "tkj,tjd->tkd",
        residual_ref[...].astype(jnp.bfloat16),
        d_output_f32,
        preferred_element_type=jnp.float32,
    )
    d_layer_output = jnp.sum(h_post_ref[...][:, :, None] * d_output_f32, axis=1)
    d_x_ref[...] = d_x.astype(d_x_ref.dtype)
    d_layer_output_ref[...] = d_layer_output

    # The coefficient gradients reduce over features. Consecutive programs revisit the same
    # small output windows, so only one feature tile is live at a time.
    d_h_post = jnp.sum(layer_output_ref[...][:, None, :] * d_output_f32, axis=-1)
    d_residual = jnp.einsum(
        "tjd,tkd->tjk",
        d_output_f32,
        x_ref[...],
        preferred_element_type=jnp.float32,
    ).transpose(0, 2, 1)

    @pl.when(feature_block == 0)
    def initialize_reductions():
      d_h_post_ref[...] = jnp.zeros_like(d_h_post_ref)
      d_residual_ref[...] = jnp.zeros_like(d_residual_ref)

    d_h_post_ref[...] += d_h_post
    d_residual_ref[...] += d_residual

    # post_apply casts residual to bf16. Its VJP therefore rounds d_residual once after the
    # complete feature reduction; rounding each partial separately would change the result.
    @pl.when(feature_block == feature_blocks - 1)
    def round_d_residual():
      d_residual_ref[...] = d_residual_ref[...].astype(jnp.bfloat16).astype(d_residual_ref.dtype)

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
          jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
          jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
          jax.ShapeDtypeStruct((tokens, streams, streams), residual.dtype),
      ),
      grid=(tokens // block_size, feature_blocks),
      in_specs=(
          pl.BlockSpec((block_size, streams, feature_block_size), lambda token, feature: (token, 0, feature)),
          pl.BlockSpec((block_size, feature_block_size), lambda token, feature: (token, feature)),
          pl.BlockSpec((block_size, streams), lambda token, feature: (token, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda token, feature: (token, 0, 0)),
          pl.BlockSpec((block_size, streams, feature_block_size), lambda token, feature: (token, 0, feature)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams, feature_block_size), lambda token, feature: (token, 0, feature)),
          pl.BlockSpec((block_size, feature_block_size), lambda token, feature: (token, feature)),
          pl.BlockSpec((block_size, streams), lambda token, feature: (token, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda token, feature: (token, 0, 0)),
      ),
      cost_estimate=pl.CostEstimate(
          flops=int(2 * (2 * tokens * streams * streams * embedding + tokens * streams * embedding)),
          transcendentals=0,
          bytes_accessed=int(3 * tokens * streams * embedding * 2 + 2 * tokens * embedding * 4),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=mhc_common.SEQUENTIAL_2D_DIMENSION_SEMANTICS,
      ),
      interpret=interpret,
  )(x, layer_output, h_post, residual, d_output)


def _pre_apply_bwd(x, h_pre, d_layer_input, d_x_acc, *, block_size, vmem_limit_bytes, interpret):
  """Builds the Pallas call for the pre-branch backward pass."""
  tokens, streams, embedding = x.shape

  def kernel(x_ref, h_pre_ref, d_layer_input_ref, d_x_acc_ref, d_x_ref, d_h_pre_ref):
    _, vjp = jax.vjp(mhc_common.pre_apply, x_ref[...], h_pre_ref[...])
    d_x, d_h_pre = vjp(d_layer_input_ref[...])
    d_x_ref[...] = (d_x.astype(jnp.float32) + d_x_acc_ref[...].astype(jnp.float32)).astype(d_x_ref.dtype)
    d_h_pre_ref[...] = d_h_pre

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
          jax.ShapeDtypeStruct((tokens, streams), h_pre.dtype),
      ),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, embedding), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
      ),
      cost_estimate=pl.CostEstimate(
          flops=int(4 * tokens * streams * embedding),
          transcendentals=0,
          bytes_accessed=int(3 * tokens * streams * embedding * 2 + tokens * embedding * 2),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=mhc_common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=interpret,
  )(x, h_pre, d_layer_input, d_x_acc)


def _coeff_bwd(
    x,
    phi,
    pre_scale,
    pre_bias,
    post_scale,
    post_bias,
    res_scale,
    res_bias,
    permutations,
    d_h_pre,
    d_h_post,
    d_residual,
    d_x_acc,
    *,
    block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
):
  """Builds the Pallas call for coefficients and parameter gradients."""
  tokens, streams, embedding = x.shape
  flattened_size = streams * embedding
  permutation_count = permutations.shape[0]

  def kernel(
      x_ref,
      phi_ref,
      pre_scale_ref,
      pre_bias_ref,
      post_scale_ref,
      post_bias_ref,
      res_scale_ref,
      res_bias_ref,
      permutations_ref,
      d_h_pre_ref,
      d_h_post_ref,
      d_residual_ref,
      d_x_acc_ref,
      d_x_ref,
      d_phi_ref,
      d_pre_scale_ref,
      d_pre_bias_ref,
      d_post_scale_ref,
      d_post_bias_ref,
      d_res_scale_ref,
      d_res_bias_ref,
  ):
    program_id = pl.program_id(0)
    permutations_value = permutations_ref[...]

    def mhc_coeffs_fn(
        x_value,
        phi_value,
        pre_scale_value,
        pre_bias_value,
        post_scale_value,
        post_bias_value,
        res_scale_value,
        res_bias_value,
    ):
      return mhc_common.mhc_coeffs(
          x_value,
          phi_value,
          pre_scale_value,
          pre_bias_value,
          post_scale_value,
          post_bias_value,
          res_scale_value,
          res_bias_value,
          permutations_value,
          rms_epsilon=rms_epsilon,
          pre_mapping_epsilon=pre_mapping_epsilon,
      )

    _, vjp = jax.vjp(
        mhc_coeffs_fn,
        x_ref[...],
        phi_ref[...],
        pre_scale_ref[...],
        pre_bias_ref[...],
        post_scale_ref[...],
        post_bias_ref[...],
        res_scale_ref[...],
        res_bias_ref[...],
    )
    (
        d_x,
        d_phi,
        d_pre_scale,
        d_pre_bias,
        d_post_scale,
        d_post_bias,
        d_res_scale,
        d_res_bias,
    ) = vjp((d_h_pre_ref[...], d_h_post_ref[...], d_residual_ref[...]))
    d_x_ref[...] = (d_x.astype(jnp.float32) + d_x_acc_ref[...].astype(jnp.float32)).astype(d_x_ref.dtype)

    @pl.when(program_id == 0)
    def initialize_reductions():
      d_phi_ref[...] = jnp.zeros_like(d_phi_ref)
      d_pre_scale_ref[...] = jnp.zeros_like(d_pre_scale_ref)
      d_pre_bias_ref[...] = jnp.zeros_like(d_pre_bias_ref)
      d_post_scale_ref[...] = jnp.zeros_like(d_post_scale_ref)
      d_post_bias_ref[...] = jnp.zeros_like(d_post_bias_ref)
      d_res_scale_ref[...] = jnp.zeros_like(d_res_scale_ref)
      d_res_bias_ref[...] = jnp.zeros_like(d_res_bias_ref)

    d_phi_ref[...] += d_phi
    d_pre_scale_ref[...] += d_pre_scale
    d_pre_bias_ref[...] += d_pre_bias
    d_post_scale_ref[...] += d_post_scale
    d_post_bias_ref[...] += d_post_bias
    d_res_scale_ref[...] += d_res_scale
    d_res_bias_ref[...] += d_res_bias

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
          jax.ShapeDtypeStruct((flattened_size, 2 * streams + permutation_count), phi.dtype),
          jax.ShapeDtypeStruct((1,), pre_scale.dtype),
          jax.ShapeDtypeStruct((streams,), pre_bias.dtype),
          jax.ShapeDtypeStruct((1,), post_scale.dtype),
          jax.ShapeDtypeStruct((streams,), post_bias.dtype),
          jax.ShapeDtypeStruct((1,), res_scale.dtype),
          jax.ShapeDtypeStruct((permutation_count,), res_bias.dtype),
      ),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          mhc_common.whole((flattened_size, 2 * streams + permutation_count)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((permutation_count,)),
          mhc_common.whole((permutation_count, streams, streams)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          mhc_common.whole((flattened_size, 2 * streams + permutation_count)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((permutation_count,)),
      ),
      cost_estimate=pl.CostEstimate(
          flops=int(
              2
              * (
                  2 * tokens * flattened_size * (2 * streams + permutation_count)
                  + 2 * tokens * permutation_count * streams * streams
              )
          ),
          transcendentals=int(tokens * (streams + permutation_count)),
          bytes_accessed=int(
              3 * tokens * streams * embedding * 2 + 2 * flattened_size * (2 * streams + permutation_count) * 4
          ),
      ),
      # This grid carries parameter-gradient reductions between programs.
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=mhc_common.SEQUENTIAL_DIMENSION_SEMANTICS,
      ),
      interpret=interpret,
  )(
      x,
      phi,
      pre_scale,
      pre_bias,
      post_scale,
      post_bias,
      res_scale,
      res_bias,
      permutations,
      d_h_pre,
      d_h_post,
      d_residual,
      d_x_acc,
  )


def _mhc_pallas_vjp_bwd_impl(
    _block_size,
    bwd_block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
    residuals,
    cotangents,
):
  """Custom-VJP backward rule with in-kernel input-gradient accumulation."""
  (
      x,
      phi,
      pre_scale,
      pre_bias,
      post_scale,
      post_bias,
      res_scale,
      res_bias,
      permutations,
      h_pre,
      norm_scale,
      pre_alpha,
      post_alpha,
      res_alpha,
      batch,
      sequence,
      streams,
      embedding,
  ) = residuals
  d_layer_input, (d_x_acc, d_h_post, d_residual) = cotangents
  d_x_acc = d_x_acc.reshape(batch * sequence, streams, embedding)
  d_h_post = d_h_post.reshape(batch * sequence, streams)
  d_residual = d_residual.reshape(batch * sequence, streams, streams)
  d_layer_input = d_layer_input.reshape(batch * sequence, embedding)

  d_x_acc, d_h_pre = _pre_apply_bwd(
      x,
      h_pre,
      d_layer_input,
      d_x_acc,
      block_size=bwd_block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  (
      d_x,
      d_phi,
      d_pre_scale,
      d_pre_bias,
      d_post_scale,
      d_post_bias,
      d_res_scale,
      d_res_bias,
  ) = _coeff_bwd(
      x,
      phi,
      pre_scale,
      pre_bias,
      post_scale,
      post_bias,
      res_scale,
      res_bias,
      permutations,
      d_h_pre,
      d_h_post,
      d_residual,
      d_x_acc,
      block_size=bwd_block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
  )
  _, phi_vjp = jax.vjp(mhc_common.fold_norm_scale, norm_scale, pre_alpha, post_alpha, res_alpha)
  d_norm_scale, d_pre_alpha, d_post_alpha, d_res_alpha = phi_vjp(d_phi)
  d_x = d_x.reshape(batch, sequence, streams, embedding)
  return (
      d_x,
      d_norm_scale,
      d_pre_alpha,
      d_pre_bias,
      d_pre_scale,
      d_post_alpha,
      d_post_bias,
      d_post_scale,
      d_res_alpha,
      d_res_bias,
      d_res_scale,
      jnp.zeros_like(permutations),
  )


def post_op_bwd(
    _block_size,
    bwd_block_size,
    bwd_feature_block_size,
    vmem_limit_bytes,
    interpret,
    saved,
    d_output,
):
  """Custom-VJP backward rule for the post-branch operation."""
  x, layer_output, h_post, residual = saved
  batch, sequence, streams, embedding = x.shape
  d_x, d_layer_output, d_h_post, d_residual = _post_apply_bwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      d_output.reshape(batch * sequence, streams, embedding),
      block_size=bwd_block_size,
      feature_block_size=bwd_feature_block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return (
      d_x.reshape(batch, sequence, streams, embedding),
      d_layer_output.reshape(batch, sequence, embedding),
      d_h_post.reshape(batch, sequence, streams),
      d_residual.reshape(batch, sequence, streams, streams),
  )
