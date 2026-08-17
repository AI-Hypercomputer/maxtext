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
"""Low-level Pallas forward kernels and custom VJP functions for mHC-lite."""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.mhc import common
from maxtext.kernels.mhc import mhc_kernels_bwd


def _coeff_fwd(
    x: jax.Array,
    coeff_params: common.MhcCoeffParams,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> common.MhcCoeffOutputs:
  """Builds the Pallas call that computes the shared mHC coefficients."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      num_permutations=permutations.shape[0],
  )

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
      h_pre_ref,
      h_post_ref,
      residual_ref,
  ):
    param_refs = common.MhcCoeffParams(
        phi=phi_ref,
        pre_scale=pre_scale_ref,
        pre_bias=pre_bias_ref,
        post_scale=post_scale_ref,
        post_bias=post_bias_ref,
        res_scale=res_scale_ref,
        res_bias=res_bias_ref,
    )
    params = jax.tree.map(lambda ref: ref[...], param_refs)
    outputs = common.mhc_coeffs(
        x_ref[...],
        params,
        permutations_ref[...],
        rms_epsilon=config.rms_epsilon,
        pre_mapping_epsilon=config.pre_mapping_epsilon,
    )
    output_refs = common.MhcCoeffOutputs(
        h_pre=h_pre_ref,
        h_post=h_post_ref,
        residual=residual_ref,
    )

    def _write_output(ref, val):
      ref[...] = val

    jax.tree.map(_write_output, output_refs, outputs)

  param_specs = jax.tree.map(lambda p: common.whole(p.shape), coeff_params)
  h_pre, h_post, residual = pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
          jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
          jax.ShapeDtypeStruct((tokens, streams, streams), jnp.float32),
      ),
      grid=(tokens // config.block_size,),
      in_specs=(
          common.token_block_spec((tokens, streams, embedding), config.block_size),
          *jax.tree.leaves(param_specs),
          common.whole(permutations.shape),
      ),
      out_specs=(
          common.token_block_spec((tokens, streams), config.block_size),
          common.token_block_spec((tokens, streams), config.block_size),
          common.token_block_spec((tokens, streams, streams), config.block_size),
      ),
      cost_estimate=dims.coeff_fwd_cost(),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=config.vmem_limit_bytes,
          dimension_semantics=common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=config.interpret,
  )(
      x,
      *jax.tree.leaves(coeff_params),
      permutations,
  )
  return common.MhcCoeffOutputs(h_pre=h_pre, h_post=h_post, residual=residual)


def _pre_apply_fwd(
    x: jax.Array,
    h_pre: jax.Array,
    config: common.MhcKernelConfig,
) -> jax.Array:
  """Builds the Pallas call for the pre-branch forward pass."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  def kernel(x_ref, h_pre_ref, output_ref):
    output_ref[...] = common.pre_apply(x_ref[...], h_pre_ref[...])

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((tokens, embedding), jnp.bfloat16),
      grid=(tokens // config.block_size,),
      in_specs=(
          common.token_block_spec((tokens, streams, embedding), config.block_size),
          common.token_block_spec((tokens, streams), config.block_size),
      ),
      out_specs=common.token_block_spec((tokens, embedding), config.block_size),
      cost_estimate=dims.pre_apply_fwd_cost(),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=config.vmem_limit_bytes,
          dimension_semantics=common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=config.interpret,
  )(x, h_pre)


def _post_apply_fwd(
    x: jax.Array,
    layer_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> jax.Array:
  """Builds the Pallas call for the post-branch forward pass."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  def kernel(x_ref, layer_output_ref, h_post_ref, residual_ref, output_ref):
    output_ref[...] = common.post_apply(
        x_ref[...],
        layer_output_ref[...],
        h_post_ref[...],
        residual_ref[...],
    )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((tokens, streams, embedding), jnp.bfloat16),
      grid=(tokens // config.block_size,),
      in_specs=(
          common.token_block_spec((tokens, streams, embedding), config.block_size),
          common.token_block_spec((tokens, embedding), config.block_size),
          common.token_block_spec((tokens, streams), config.block_size),
          common.token_block_spec((tokens, streams, streams), config.block_size),
      ),
      out_specs=common.token_block_spec((tokens, streams, embedding), config.block_size),
      cost_estimate=dims.post_apply_fwd_cost(),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=config.vmem_limit_bytes,
          dimension_semantics=common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=config.interpret,
  )(x, layer_output, h_post, residual)


def pre_fwd(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[tuple[jax.Array, common.KernelContext], tuple[jax.Array, jax.Array]]:
  """Runs coefficient and pre-application forward kernels."""
  common.validate_inputs(x, config.block_size, permutations.shape)
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  x_flat = x.reshape(tokens, streams, embedding)

  coeff_params = weights.to_coeff_params()
  outputs = _coeff_fwd(x_flat, coeff_params, permutations, config)
  layer_input = _pre_apply_fwd(x_flat, outputs.h_pre, config)

  context: common.KernelContext = (
      x,
      outputs.h_post.reshape(batch, sequence, streams),
      outputs.residual.reshape(batch, sequence, streams, streams),
  )
  primals_out = (layer_input.reshape(batch, sequence, embedding), context)
  saved_residuals = (coeff_params.phi, outputs.h_pre)
  return primals_out, saved_residuals


def post_fwd(
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing forward kernel."""
  common.validate_inputs(x, config.block_size)
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  output = _post_apply_fwd(
      x.reshape(tokens, streams, embedding),
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams, streams),
      config,
  )
  return output.reshape(batch, sequence, streams, embedding)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 2))
def _pre_op(
    config: common.MhcKernelConfig,
    x: jax.Array,
    permutations: jax.Array,
    weights: common.MhcWeights,
) -> tuple[jax.Array, common.KernelContext]:
  """Differentiable pre-branch mHC operation."""
  (layer_input, context), _ = pre_fwd(x, weights, permutations, config)
  return layer_input, context


def _pre_op_fwd(
    config: common.MhcKernelConfig,
    x: jax.Array,
    permutations: jax.Array,
    weights: common.MhcWeights,
):
  """Custom-VJP forward rule for the pre-branch operation."""
  primals_out, saved = pre_fwd(x, weights, permutations, config)
  return primals_out, (saved, (x, weights))


_pre_op.defvjp(_pre_op_fwd, mhc_kernels_bwd.pre_op_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _post_op(
    config: common.MhcKernelConfig,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> jax.Array:
  """Differentiable post-branch mHC operation."""
  return post_fwd(layer_output, x, h_post, residual, config)


def _post_op_fwd(
    config: common.MhcKernelConfig,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
):
  """Custom-VJP forward rule for the post-branch operation."""
  output = post_fwd(layer_output, x, h_post, residual, config)
  return output, (layer_output, x, h_post, residual)


_post_op.defvjp(_post_op_fwd, mhc_kernels_bwd.post_op_bwd)


def pre(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.KernelContext]:
  """Runs the coefficient and pre-application kernels.

  Args:
    x: Input streams with shape `[batch, sequence, streams, embedding]`.
    weights: Structured weights container.
    permutations: Permutation matrices with shape `[streams!, streams,
      streams]`.
    config: Kernel tuning and compiler configuration.

  Returns:
    A pair containing the branch input and opaque kernel context.
  """
  common.validate_inputs(x, config.block_size, permutations.shape)
  common.validate_token_block_size(x.shape[0] * x.shape[1], config.bwd_block_size, name="bwd_block_size")
  return _pre_op(config, x, permutations, weights)


def post(
    layer_output: jax.Array,
    context: common.KernelContext,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing kernel.

  Args:
    layer_output: Wrapped branch output with shape `[batch, sequence,
      embedding]`.
    context: Opaque context returned by `pre`.
    config: Kernel tuning and compiler configuration.

  Returns:
    Mixed output streams with shape `[batch, sequence, streams, embedding]`.
  """
  x, h_post, residual = context
  feature_block_size = min(x.shape[-1], config.bwd_feature_block_size)
  common.validate_inputs(x, config.block_size)
  common.validate_token_block_size(x.shape[0] * x.shape[1], config.bwd_block_size, name="bwd_block_size")
  common.validate_feature_block_size(x.shape[-1], feature_block_size)
  return _post_op(config, layer_output, x, h_post, residual)
