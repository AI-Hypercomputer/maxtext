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
"""Low-level Pallas backward kernels and custom VJP rules for mHC-lite."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.src.maxtext.kernels.mhc import common


def _post_apply_bwd(
    x: jax.Array,
    layer_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    d_output: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds the feature-tiled Pallas pipeline for the post-branch backward pass."""
  tokens, streams, embedding = x.shape
  feature_block_size = min(embedding, config.bwd_feature_block_size)
  feature_blocks = embedding // feature_block_size
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  def pipeline_body(
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

    d_x, d_layer_output = common.post_apply_bwd_pointwise(
        d_output_ref[...],
        h_post_ref[...],
        residual_ref[...],
    )
    d_x_ref[...] = d_x.astype(d_x_ref.dtype)
    d_layer_output_ref[...] = d_layer_output.astype(d_layer_output_ref.dtype)

    d_h_post, d_residual = common.post_apply_bwd_reductions(
        d_output_ref[...],
        layer_output_ref[...],
        x_ref[...],
    )

    @pl.when(feature_block == 0)
    def initialize_reductions():
      d_h_post_ref[...] = jnp.zeros_like(d_h_post_ref)
      d_residual_ref[...] = jnp.zeros_like(d_residual_ref)

    d_h_post_ref[...] += d_h_post
    d_residual_ref[...] += d_residual

    @pl.when(feature_block == feature_blocks - 1)
    def round_d_residual():
      d_residual_ref[...] = d_residual_ref[...].astype(jnp.bfloat16).astype(d_residual_ref.dtype)

  spec_x = common.feature_tiled_block_spec(
      (tokens, streams, embedding),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=True,
  )
  spec_layer_output = common.feature_tiled_block_spec(
      (tokens, embedding),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=True,
  )
  spec_h_post = common.feature_tiled_block_spec(
      (tokens, streams),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=False,
  )
  spec_residual = common.feature_tiled_block_spec(
      (tokens, streams, streams),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=False,
  )

  kernel_main = pltpu.emit_pipeline(
      pipeline_body,
      grid=(tokens // config.bwd_block_size, feature_blocks),
      in_specs=(
          spec_x,
          spec_layer_output,
          spec_h_post,
          spec_residual,
          spec_x,
      ),
      out_specs=(
          spec_x,
          spec_layer_output,
          spec_h_post,
          spec_residual,
      ),
      dimension_semantics=common.POST_BWD_DIMENSION_SEMANTICS,
  )

  with common.tpu_mesh_context():
    d_x, d_layer_output, d_h_post, d_residual = pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
            jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
            jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
            jax.ShapeDtypeStruct((tokens, streams, streams), residual.dtype),
        ),
        in_specs=common.hbm_specs(5),
        out_specs=common.hbm_specs(4),
        cost_estimate=dims.post_apply_bwd_cost(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=config.interpret,
    )(x, layer_output, h_post, residual, d_output)
  return d_x, d_layer_output, d_h_post, d_residual


def _pre_apply_bwd(
    x: jax.Array,
    h_pre: jax.Array,
    d_layer_input: jax.Array,
    d_x_acc: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array]:
  """Builds the Pallas pipeline for the pre-branch backward pass."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  def pipeline_body(x_ref, h_pre_ref, d_layer_input_ref, d_x_acc_ref, d_x_ref, d_h_pre_ref):
    _, vjp = jax.vjp(common.pre_apply, x_ref[...], h_pre_ref[...])
    d_x, d_h_pre = vjp(d_layer_input_ref[...])
    d_x_ref[...] = (d_x.astype(jnp.float32) + d_x_acc_ref[...].astype(jnp.float32)).astype(d_x_ref.dtype)
    d_h_pre_ref[...] = d_h_pre

  spec_x = common.token_block_spec((tokens, streams, embedding), config.bwd_block_size)
  spec_h_pre = common.token_block_spec((tokens, streams), config.bwd_block_size)
  spec_d_layer_input = common.token_block_spec((tokens, embedding), config.bwd_block_size)

  kernel_main = pltpu.emit_pipeline(
      pipeline_body,
      grid=(tokens // config.bwd_block_size,),
      in_specs=(
          spec_x,
          spec_h_pre,
          spec_d_layer_input,
          spec_x,
      ),
      out_specs=(
          spec_x,
          spec_h_pre,
      ),
      dimension_semantics=common.PARALLEL_DIMENSION_SEMANTICS,
  )

  with common.tpu_mesh_context():
    d_x_acc_out, d_h_pre = pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
            jax.ShapeDtypeStruct((tokens, streams), h_pre.dtype),
        ),
        in_specs=common.hbm_specs(4),
        out_specs=common.hbm_specs(2),
        cost_estimate=dims.pre_apply_bwd_cost(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=config.interpret,
    )(x, h_pre, d_layer_input, d_x_acc)
  return d_x_acc_out, d_h_pre


def _coeff_bwd(
    x: jax.Array,
    coeff_params: common.MhcCoeffParams,
    permutations: jax.Array,
    d_outputs: common.MhcCoeffOutputs,
    d_x_acc: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, common.MhcCoeffGradients]:
  """Builds the Pallas pipeline for coefficients and parameter gradients."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      num_permutations=permutations.shape[0],
  )

  num_params = len(jax.tree.leaves(coeff_params))
  num_outputs = len(jax.tree.leaves(d_outputs))

  def pipeline_body(x_ref, *refs):
    program_id = pl.program_id(0)

    ref_iter = iter(refs)
    param_refs = common.MhcCoeffParams(*(next(ref_iter) for _ in range(num_params)))
    permutations_ref = next(ref_iter)
    cotangent_refs = common.MhcCoeffOutputs(*(next(ref_iter) for _ in range(num_outputs)))
    d_x_acc_ref = next(ref_iter)
    d_x_ref = next(ref_iter)
    d_param_refs = common.MhcCoeffParams(*ref_iter)

    perms = permutations_ref[...]

    def mhc_coeffs_fn(x_val, params_val):
      return common.mhc_coeffs(
          x_val,
          params_val,
          perms,
          rms_epsilon=config.rms_epsilon,
          pre_mapping_epsilon=config.pre_mapping_epsilon,
      )

    params_in = jax.tree.map(lambda ref: ref[...], param_refs)
    cotangents = jax.tree.map(lambda ref: ref[...], cotangent_refs)

    _, vjp = jax.vjp(mhc_coeffs_fn, x_ref[...], params_in)
    d_x, d_params = vjp(cotangents)

    d_x_ref[...] = (d_x.astype(jnp.float32) + d_x_acc_ref[...].astype(jnp.float32)).astype(d_x_ref.dtype)

    @pl.when(program_id == 0)
    def initialize_reductions():
      def _zero(ref):
        ref[...] = jnp.zeros_like(ref)

      jax.tree.map(_zero, d_param_refs)

    def _accumulate(ref, val):
      ref[...] += val.astype(jnp.float32)

    jax.tree.map(_accumulate, d_param_refs, d_params)

  spec_x = common.token_block_spec((tokens, streams, embedding), config.bwd_block_size)
  param_specs = jax.tree.map(lambda p: common.whole(p.shape), coeff_params)
  param_out_shapes = jax.tree.map(lambda p: jax.ShapeDtypeStruct(p.shape, jnp.float32), coeff_params)
  output_specs = jax.tree.map(
      lambda out: common.token_block_spec(out.shape, config.bwd_block_size),
      d_outputs,
  )

  kernel_main = pltpu.emit_pipeline(
      pipeline_body,
      grid=(tokens // config.bwd_block_size,),
      in_specs=(
          spec_x,
          *jax.tree.leaves(param_specs),
          common.whole(permutations.shape),
          *jax.tree.leaves(output_specs),
          spec_x,
      ),
      out_specs=(
          spec_x,
          *jax.tree.leaves(param_specs),
      ),
      dimension_semantics=common.SEQUENTIAL_DIMENSION_SEMANTICS,
  )

  in_specs = common.hbm_specs(1 + num_params + 1 + num_outputs + 1)
  out_specs = common.hbm_specs(1 + num_params)

  with common.tpu_mesh_context():
    d_x, *d_param_grads = pl.pallas_call(
        kernel_main,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
            *jax.tree.leaves(param_out_shapes),
        ),
        in_specs=in_specs,
        out_specs=out_specs,
        cost_estimate=dims.coeff_bwd_cost(),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=config.interpret,
    )(
        x,
        *jax.tree.leaves(coeff_params),
        permutations,
        *jax.tree.leaves(d_outputs),
        d_x_acc,
    )
  d_coeff_grads = common.MhcCoeffGradients(*d_param_grads)
  return d_x, d_coeff_grads


def pre_bwd(
    residuals: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, common.MhcWeights]:
  """Computes pre-branch gradients with in-kernel input-gradient accumulation."""
  phi, h_pre = residuals
  d_layer_input, d_x_acc, d_h_post, d_residual = cotangents
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence

  x_flat = x.reshape(tokens, streams, embedding)
  d_x_acc_flat = d_x_acc.reshape(tokens, streams, embedding)
  d_h_post_flat = d_h_post.reshape(tokens, streams)
  d_residual_flat = d_residual.reshape(tokens, streams, streams)
  d_layer_input_flat = d_layer_input.reshape(tokens, embedding)

  d_x_acc_flat, d_h_pre = _pre_apply_bwd(
      x_flat,
      h_pre,
      d_layer_input_flat,
      d_x_acc_flat,
      config=config,
  )
  coeff_params = common.MhcCoeffParams(
      phi=phi,
      pre_scale=weights.pre_scale,
      pre_bias=weights.pre_bias,
      post_scale=weights.post_scale,
      post_bias=weights.post_bias,
      res_scale=weights.res_scale,
      res_bias=weights.res_bias,
  )
  d_outputs = common.MhcCoeffOutputs(
      h_pre=d_h_pre,
      h_post=d_h_post_flat,
      residual=d_residual_flat,
  )
  d_x, d_coeff_grads = _coeff_bwd(
      x_flat,
      coeff_params,
      permutations,
      d_outputs,
      d_x_acc_flat,
      config=config,
  )
  _, phi_vjp = jax.vjp(
      common.fold_norm_scale,
      weights.norm_scale,
      weights.pre_alpha,
      weights.post_alpha,
      weights.res_alpha,
  )
  d_norm_scale, d_pre_alpha, d_post_alpha, d_res_alpha = phi_vjp(d_coeff_grads.phi)
  d_weights = common.MhcWeights(
      norm_scale=d_norm_scale,
      pre_alpha=d_pre_alpha,
      pre_bias=d_coeff_grads.pre_bias.astype(weights.pre_bias.dtype),
      pre_scale=d_coeff_grads.pre_scale.astype(weights.pre_scale.dtype),
      post_alpha=d_post_alpha,
      post_bias=d_coeff_grads.post_bias.astype(weights.post_bias.dtype),
      post_scale=d_coeff_grads.post_scale.astype(weights.post_scale.dtype),
      res_alpha=d_res_alpha,
      res_bias=d_coeff_grads.res_bias.astype(weights.res_bias.dtype),
      res_scale=d_coeff_grads.res_scale.astype(weights.res_scale.dtype),
  )
  return d_x.reshape(batch, sequence, streams, embedding), d_weights


def post_bwd(
    cotangent: jax.Array,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes post-branch gradients."""
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  d_x, d_layer_output, d_h_post, d_residual = _post_apply_bwd(
      x.reshape(tokens, streams, embedding),
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams, streams),
      cotangent.reshape(tokens, streams, embedding),
      config=config,
  )
  return (
      d_layer_output.reshape(batch, sequence, embedding),
      d_x.reshape(batch, sequence, streams, embedding),
      d_h_post.reshape(batch, sequence, streams),
      d_residual.reshape(batch, sequence, streams, streams),
  )


def pre_op_bwd(
    config: common.MhcKernelConfig,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, common.MhcWeights],
    ],
    cotangents: tuple[jax.Array, common.KernelContext],
) -> tuple[jax.Array, None, common.MhcWeights]:
  """Custom-VJP backward rule for the low-level pre-branch entry point."""
  saved, (x, permutations, weights) = residuals
  d_layer_input, (d_x, d_h_post, d_residual) = cotangents
  d_x_out, d_weights = pre_bwd(
      saved,
      (d_layer_input, d_x, d_h_post, d_residual),
      x,
      weights,
      permutations,
      config=config,
  )
  return d_x_out, None, d_weights


def post_op_bwd(
    config: common.MhcKernelConfig,
    saved: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    d_output: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Custom-VJP backward rule for the low-level post-branch entry point."""
  layer_output, x, h_post, residual = saved
  d_layer_output, d_x, d_h_post, d_residual = post_bwd(
      d_output,
      layer_output,
      x,
      h_post,
      residual,
      config=config,
  )
  return d_layer_output, d_x, d_h_post, d_residual
