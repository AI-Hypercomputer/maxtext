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

"""Pallas TPU kernels for mHC-lite connectivity.

The implementation follows the normalize-late formulation from the mHC paper.
It folds the RMSNorm scale into the three projection matrices and divides the
small projected tensor by the per-token RMS value. This avoids materializing a
normalized ``(tokens, streams * embedding)`` tensor and computes the pre, post,
and residual coefficients in one pass over the input.

The pre operation returns the input as opaque context for the post operation.
During autodiff, the post operation's input cotangent therefore flows into the
pre operation's custom VJP. The backward kernels accumulate that cotangent
in-place while writing their own input gradients, avoiding a separate HBM
fan-in sum.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


DEFAULT_BLOCK_SIZE = 16
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
_PARALLEL = ("parallel",)
_SEQUENTIAL = ("arbitrary",)


def _whole(shape):
  """A full-array BlockSpec for values that stay VMEM-resident."""
  return pl.BlockSpec(shape, lambda _: tuple(0 for _ in shape))


def _fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha):
  """Folds the RMSNorm channel scale into the three projections."""
  alpha = jnp.concatenate((pre_alpha, post_alpha, res_alpha), axis=-1)
  return norm_scale.astype(jnp.float32)[:, None] * alpha.astype(jnp.float32)


def _coefficients(
    x,
    phi,
    pre_scale,
    pre_bias,
    post_scale,
    post_bias,
    res_scale,
    res_bias,
    permutations,
    *,
    rms_epsilon,
    pre_mapping_epsilon,
):
  """Computes all mHC-lite coefficients without materializing normalized x."""
  tokens, streams, embedding = x.shape
  flattened = x.reshape(tokens, streams * embedding)
  projected = jnp.dot(flattened, phi.astype(jnp.bfloat16), preferred_element_type=jnp.float32)

  flattened_f32 = flattened.astype(jnp.float32)
  mean_square = jnp.mean(flattened_f32 * flattened_f32, axis=-1, keepdims=True)
  projected = projected * jax.lax.rsqrt(mean_square + rms_epsilon)

  pre_logits = projected[:, :streams]
  post_logits = projected[:, streams : 2 * streams]
  res_logits = projected[:, 2 * streams :]
  h_pre = jax.nn.sigmoid(pre_scale.astype(jnp.float32) * pre_logits + pre_bias.astype(jnp.float32)) + pre_mapping_epsilon
  h_post = 2.0 * jax.nn.sigmoid(post_scale.astype(jnp.float32) * post_logits + post_bias.astype(jnp.float32))
  weights = jax.nn.softmax(
      res_scale.astype(jnp.float32) * res_logits + res_bias.astype(jnp.float32),
      axis=-1,
  )
  permutation_count = permutations.shape[0]
  residual = jnp.dot(
      weights,
      permutations.reshape(permutation_count, streams * streams).astype(jnp.float32),
  ).reshape(tokens, streams, streams)
  return h_pre, h_post, residual


def _apply_pre(x, h_pre):
  """Collapses the stream dimension before the wrapped model branch."""
  h_pre_f32 = h_pre.astype(jnp.float32)
  return sum(
      h_pre_f32[:, stream : stream + 1] * x[:, stream, :].astype(jnp.float32) for stream in range(x.shape[1])
  ).astype(jnp.bfloat16)


def _apply_post(x, layer_output, h_post, residual):
  """Broadcasts the branch output and applies the residual stream mixing."""
  residual_mix = jnp.einsum(
      "tkj,tkd->tjd",
      residual.astype(jnp.bfloat16),
      x,
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post.astype(jnp.float32)[:, :, None] * layer_output.astype(jnp.float32)[:, None, :]
  return (residual_mix + post_mix).astype(jnp.bfloat16)


def _coefficients_fwd(
    x,
    phi,
    pre_scale,
    pre_bias,
    post_scale,
    post_bias,
    res_scale,
    res_bias,
    permutations,
    *,
    block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
):
  """Builds the Pallas call that computes the shared mHC coefficients."""
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
      h_pre_ref,
      h_post_ref,
      residual_ref,
  ):
    h_pre, h_post, residual = _coefficients(
        x_ref[...],
        phi_ref[...],
        pre_scale_ref[...],
        pre_bias_ref[...],
        post_scale_ref[...],
        post_bias_ref[...],
        res_scale_ref[...],
        res_bias_ref[...],
        permutations_ref[...],
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
    )
    h_pre_ref[...] = h_pre
    h_post_ref[...] = h_post
    residual_ref[...] = residual

  cost = pl.CostEstimate(
      flops=int(
          2 * tokens * flattened_size * (2 * streams + permutation_count)
          + 2 * tokens * permutation_count * streams * streams
      ),
      transcendentals=int(tokens * (streams + permutation_count)),
      bytes_accessed=int(tokens * streams * embedding * 2 + flattened_size * (2 * streams + permutation_count) * 4),
  )
  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
          jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
          jax.ShapeDtypeStruct((tokens, streams, streams), jnp.float32),
      ),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          _whole((flattened_size, 2 * streams + permutation_count)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((permutation_count,)),
          _whole((permutation_count, streams, streams)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
      ),
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=_PARALLEL,
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
  )


def _apply_pre_fwd(x, h_pre, *, block_size, vmem_limit_bytes, interpret):
  tokens, streams, embedding = x.shape

  def kernel(x_ref, h_pre_ref, output_ref):
    output_ref[...] = _apply_pre(x_ref[...], h_pre_ref[...])

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((tokens, embedding), jnp.bfloat16),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
      ),
      out_specs=pl.BlockSpec((block_size, embedding), lambda i: (i, 0)),
      cost_estimate=pl.CostEstimate(
          flops=int(2 * tokens * streams * embedding),
          transcendentals=0,
          bytes_accessed=int(tokens * streams * embedding * 2 + tokens * embedding * 2),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=_PARALLEL,
      ),
      interpret=interpret,
  )(x, h_pre)


def _apply_post_fwd(x, layer_output, h_post, residual, *, block_size, vmem_limit_bytes, interpret):
  """Builds the Pallas call for the post-branch forward pass."""
  tokens, streams, embedding = x.shape

  def kernel(x_ref, layer_output_ref, h_post_ref, residual_ref, output_ref):
    output_ref[...] = _apply_post(
        x_ref[...],
        layer_output_ref[...],
        h_post_ref[...],
        residual_ref[...],
    )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((tokens, streams, embedding), jnp.bfloat16),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, embedding), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
      ),
      out_specs=pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
      cost_estimate=pl.CostEstimate(
          flops=int(2 * tokens * streams * streams * embedding + tokens * streams * embedding),
          transcendentals=0,
          bytes_accessed=int(2 * tokens * streams * embedding * 2 + tokens * embedding * 4),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=_PARALLEL,
      ),
      interpret=interpret,
  )(x, layer_output, h_post, residual)


def _apply_post_bwd(x, layer_output, h_post, residual, d_output, *, block_size, vmem_limit_bytes, interpret):
  """Builds the Pallas call for the post-branch backward pass."""
  tokens, streams, embedding = x.shape

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
    _, vjp = jax.vjp(
        _apply_post,
        x_ref[...],
        layer_output_ref[...],
        h_post_ref[...],
        residual_ref[...],
    )
    d_x, d_layer_output, d_h_post, d_residual = vjp(d_output_ref[...])
    d_x_ref[...] = d_x
    d_layer_output_ref[...] = d_layer_output
    d_h_post_ref[...] = d_h_post
    d_residual_ref[...] = d_residual

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
          jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
          jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
          jax.ShapeDtypeStruct((tokens, streams, streams), residual.dtype),
      ),
      grid=(tokens // block_size,),
      in_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, embedding), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, embedding), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
      ),
      cost_estimate=pl.CostEstimate(
          flops=int(2 * (2 * tokens * streams * streams * embedding + tokens * streams * embedding)),
          transcendentals=0,
          bytes_accessed=int(3 * tokens * streams * embedding * 2 + 2 * tokens * embedding * 4),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=_PARALLEL,
      ),
      interpret=interpret,
  )(x, layer_output, h_post, residual, d_output)


def _apply_pre_bwd(x, h_pre, d_layer_input, d_x_acc, *, block_size, vmem_limit_bytes, interpret):
  tokens, streams, embedding = x.shape

  def kernel(x_ref, h_pre_ref, d_layer_input_ref, d_x_acc_ref, d_x_ref, d_h_pre_ref):
    _, vjp = jax.vjp(_apply_pre, x_ref[...], h_pre_ref[...])
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
          dimension_semantics=_PARALLEL,
      ),
      interpret=interpret,
  )(x, h_pre, d_layer_input, d_x_acc)


def _coefficients_bwd(
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

    def coefficients_fn(
        x_value,
        phi_value,
        pre_scale_value,
        pre_bias_value,
        post_scale_value,
        post_bias_value,
        res_scale_value,
        res_bias_value,
    ):
      return _coefficients(
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
        coefficients_fn,
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
          _whole((flattened_size, 2 * streams + permutation_count)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((permutation_count,)),
          _whole((permutation_count, streams, streams)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams, embedding), lambda i: (i, 0, 0)),
          _whole((flattened_size, 2 * streams + permutation_count)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((streams,)),
          _whole((1,)),
          _whole((permutation_count,)),
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
          dimension_semantics=_SEQUENTIAL,
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


def _pre_forward(
    block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
    x,
    norm_scale,
    pre_alpha,
    pre_bias,
    pre_scale,
    post_alpha,
    post_bias,
    post_scale,
    res_alpha,
    res_bias,
    res_scale,
    permutations,
):
  """Runs the pre-branch forward path and saves its custom-VJP state."""
  batch, sequence, streams, embedding = x.shape
  x_flat = x.reshape(batch * sequence, streams, embedding)
  phi = _fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha)
  h_pre, h_post, residual = _coefficients_fwd(
      x_flat,
      phi,
      pre_scale,
      pre_bias,
      post_scale,
      post_bias,
      res_scale,
      res_bias,
      permutations,
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
  )
  layer_input = _apply_pre_fwd(
      x_flat,
      h_pre,
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  output = (
      layer_input.reshape(batch, sequence, embedding),
      (
          x,
          h_post.reshape(batch, sequence, streams),
          residual.reshape(batch, sequence, streams, streams),
      ),
  )
  residuals = (
      x_flat,
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
  )
  return output, residuals


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _pre_op(
    block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
    x,
    norm_scale,
    pre_alpha,
    pre_bias,
    pre_scale,
    post_alpha,
    post_bias,
    post_scale,
    res_alpha,
    res_bias,
    res_scale,
    permutations,
):
  """Differentiable pre-branch mHC operation."""
  output, _ = _pre_forward(
      block_size,
      vmem_limit_bytes,
      interpret,
      rms_epsilon,
      pre_mapping_epsilon,
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      permutations,
  )
  return output


def _pre_op_fwd(
    block_size,
    vmem_limit_bytes,
    interpret,
    rms_epsilon,
    pre_mapping_epsilon,
    x,
    norm_scale,
    pre_alpha,
    pre_bias,
    pre_scale,
    post_alpha,
    post_bias,
    post_scale,
    res_alpha,
    res_bias,
    res_scale,
    permutations,
):
  """Custom-VJP forward rule for the pre-branch operation."""
  return _pre_forward(
      block_size,
      vmem_limit_bytes,
      interpret,
      rms_epsilon,
      pre_mapping_epsilon,
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      permutations,
  )


def _pre_op_bwd(block_size, vmem_limit_bytes, interpret, rms_epsilon, pre_mapping_epsilon, residuals, cotangents):
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

  d_x_acc, d_h_pre = _apply_pre_bwd(
      x,
      h_pre,
      d_layer_input,
      d_x_acc,
      block_size=block_size,
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
  ) = _coefficients_bwd(
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
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
  )
  _, phi_vjp = jax.vjp(_fold_norm_scale, norm_scale, pre_alpha, post_alpha, res_alpha)
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


_pre_op.defvjp(_pre_op_fwd, _pre_op_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _post_op(block_size, vmem_limit_bytes, interpret, x, layer_output, h_post, residual):
  """Differentiable post-branch mHC operation."""
  batch, sequence, streams, embedding = x.shape
  output = _apply_post_fwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return output.reshape(batch, sequence, streams, embedding)


def _post_op_fwd(block_size, vmem_limit_bytes, interpret, x, layer_output, h_post, residual):
  """Custom-VJP forward rule for the post-branch operation."""
  batch, sequence, streams, embedding = x.shape
  output = _apply_post_fwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return output.reshape(batch, sequence, streams, embedding), (x, layer_output, h_post, residual)


def _post_op_bwd(block_size, vmem_limit_bytes, interpret, saved, d_output):
  """Custom-VJP backward rule for the post-branch operation."""
  x, layer_output, h_post, residual = saved
  batch, sequence, streams, embedding = x.shape
  d_x, d_layer_output, d_h_post, d_residual = _apply_post_bwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      d_output.reshape(batch * sequence, streams, embedding),
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return (
      d_x.reshape(batch, sequence, streams, embedding),
      d_layer_output.reshape(batch, sequence, embedding),
      d_h_post.reshape(batch, sequence, streams),
      d_residual.reshape(batch, sequence, streams, streams),
  )


_post_op.defvjp(_post_op_fwd, _post_op_bwd)


def _validate_inputs(x, block_size, permutations_shape=None):
  """Validates the shape and dtype constraints of the tuned kernel."""
  if x.dtype != jnp.bfloat16:
    raise ValueError(f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}.")
  if x.ndim != 4:
    raise ValueError(f"Expected x to have shape (batch, sequence, streams, embedding); got {x.shape}.")
  batch, sequence, streams, embedding = x.shape
  if streams != 4 or (permutations_shape is not None and permutations_shape != (24, 4, 4)):
    raise ValueError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with expansion rate 4 only; "
        f"got x.shape={x.shape} and permutations.shape={permutations_shape}."
    )
  if embedding % 128:
    raise ValueError(f"The embedding dimension must be divisible by 128; got {embedding}.")
  tokens = batch * sequence
  if block_size < 8 or block_size % 8:
    raise ValueError(f"block_size must be a positive multiple of 8; got {block_size}.")
  if tokens % block_size:
    raise ValueError(f"The per-device token count ({tokens}) must be divisible by block_size ({block_size}).")


def pre(
    x,
    norm_scale,
    pre_alpha,
    pre_bias,
    pre_scale,
    post_alpha,
    post_bias,
    post_scale,
    res_alpha,
    res_bias,
    res_scale,
    permutations,
    *,
    rms_epsilon,
    pre_mapping_epsilon=1e-6,
    block_size=DEFAULT_BLOCK_SIZE,
    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
    interpret=False,
):
  """Runs the coefficient and pre-application kernels.

  Returns ``(layer_input, context)``. Pass ``context`` unchanged to :func:`post`
  after running the attention or feed-forward branch on ``layer_input``.
  """
  _validate_inputs(x, block_size, permutations.shape)
  return _pre_op(
      block_size,
      vmem_limit_bytes,
      interpret,
      rms_epsilon,
      pre_mapping_epsilon,
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      permutations,
  )


def post(
    layer_output,
    context,
    *,
    block_size=DEFAULT_BLOCK_SIZE,
    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
    interpret=False,
):
  """Runs the fused post-gate and residual-mixing kernel."""
  x, h_post, residual = context
  _validate_inputs(x, block_size)
  return _post_op(
      block_size,
      vmem_limit_bytes,
      interpret,
      x,
      layer_output,
      h_post,
      residual,
  )
