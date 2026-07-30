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

"""Forward Pallas kernels and custom-VJP wrappers for mHC-lite.

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

from maxtext.kernels.residual.mhc_bwd_kernels import _mhc_pallas_vjp_bwd_impl
from maxtext.kernels.residual import mhc_bwd_kernels
from maxtext.kernels.residual import mhc_common


DEFAULT_BLOCK_SIZE = mhc_common.DEFAULT_BLOCK_SIZE
DEFAULT_BWD_BLOCK_SIZE = mhc_common.DEFAULT_BWD_BLOCK_SIZE
DEFAULT_POST_BWD_BLOCK_SIZE = mhc_common.DEFAULT_POST_BWD_BLOCK_SIZE
DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE = mhc_common.DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE
DEFAULT_VMEM_LIMIT_BYTES = mhc_common.DEFAULT_VMEM_LIMIT_BYTES


def _coeff_fwd(
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
    h_pre, h_post, residual = mhc_common.mhc_coeffs(
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
          mhc_common.whole((flattened_size, 2 * streams + permutation_count)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((streams,)),
          mhc_common.whole((1,)),
          mhc_common.whole((permutation_count,)),
          mhc_common.whole((permutation_count, streams, streams)),
      ),
      out_specs=(
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams), lambda i: (i, 0)),
          pl.BlockSpec((block_size, streams, streams), lambda i: (i, 0, 0)),
      ),
      cost_estimate=cost,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          dimension_semantics=mhc_common.PARALLEL_DIMENSION_SEMANTICS,
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


def _pre_apply_fwd(x, h_pre, *, block_size, vmem_limit_bytes, interpret):
  tokens, streams, embedding = x.shape

  def kernel(x_ref, h_pre_ref, output_ref):
    output_ref[...] = mhc_common.pre_apply(x_ref[...], h_pre_ref[...])

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
          dimension_semantics=mhc_common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=interpret,
  )(x, h_pre)


def _post_apply_fwd(x, layer_output, h_post, residual, *, block_size, vmem_limit_bytes, interpret):
  """Builds the Pallas call for the post-branch forward pass."""
  tokens, streams, embedding = x.shape

  def kernel(x_ref, layer_output_ref, h_post_ref, residual_ref, output_ref):
    output_ref[...] = mhc_common.post_apply(
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
          dimension_semantics=mhc_common.PARALLEL_DIMENSION_SEMANTICS,
      ),
      interpret=interpret,
  )(x, layer_output, h_post, residual)


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
  phi = mhc_common.fold_norm_scale(norm_scale, pre_alpha, post_alpha, res_alpha)
  h_pre, h_post, residual = _coeff_fwd(
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
  layer_input = _pre_apply_fwd(
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _pre_op(
    block_size,
    _bwd_block_size,
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
    _bwd_block_size,
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


_pre_op.defvjp(_pre_op_fwd, _mhc_pallas_vjp_bwd_impl)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _post_op(
    block_size,
    _bwd_block_size,
    _bwd_feature_block_size,
    vmem_limit_bytes,
    interpret,
    x,
    layer_output,
    h_post,
    residual,
):
  """Differentiable post-branch mHC operation."""
  batch, sequence, streams, embedding = x.shape
  output = _post_apply_fwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return output.reshape(batch, sequence, streams, embedding)


def _post_op_fwd(
    block_size,
    _bwd_block_size,
    _bwd_feature_block_size,
    vmem_limit_bytes,
    interpret,
    x,
    layer_output,
    h_post,
    residual,
):
  """Custom-VJP forward rule for the post-branch operation."""
  batch, sequence, streams, embedding = x.shape
  output = _post_apply_fwd(
      x.reshape(batch * sequence, streams, embedding),
      layer_output.reshape(batch * sequence, embedding),
      h_post.reshape(batch * sequence, streams),
      residual.reshape(batch * sequence, streams, streams),
      block_size=block_size,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return output.reshape(batch, sequence, streams, embedding), (x, layer_output, h_post, residual)


_post_op.defvjp(_post_op_fwd, mhc_bwd_kernels.post_op_bwd)


def pre(
    x: jax.Array,
    norm_scale: jax.Array,
    pre_alpha: jax.Array,
    pre_bias: jax.Array,
    pre_scale: jax.Array,
    post_alpha: jax.Array,
    post_bias: jax.Array,
    post_scale: jax.Array,
    res_alpha: jax.Array,
    res_bias: jax.Array,
    res_scale: jax.Array,
    permutations: jax.Array,
    *,
    rms_epsilon: float,
    pre_mapping_epsilon: float = 1e-6,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bwd_block_size: int = DEFAULT_BWD_BLOCK_SIZE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> tuple[jax.Array, mhc_common.MHCContext]:
  """Runs the coefficient and pre-application kernels.

  Args:
    x: Input streams with shape [batch, sequence, streams, embedding].
    norm_scale: RMSNorm scale with shape [streams * embedding].
    pre_alpha: Pre-gate projection with shape [streams * embedding, streams].
    pre_bias: Pre-gate bias with shape [streams].
    pre_scale: Pre-gate scalar scale with shape [1].
    post_alpha: Post-gate projection with shape [streams * embedding, streams].
    post_bias: Post-gate bias with shape [streams].
    post_scale: Post-gate scalar scale with shape [1].
    res_alpha: Residual projection with shape [streams * embedding, streams!].
    res_bias: Residual bias with shape [streams!].
    res_scale: Residual scalar scale with shape [1].
    permutations: Permutation matrices with shape [streams!, streams, streams].
    rms_epsilon: Epsilon used by RMSNorm.
    pre_mapping_epsilon: Epsilon added to the pre-gate output.
    block_size: Token-axis Pallas block size for the forward kernels.
    bwd_block_size: Token-axis block size for the coefficient and pre-application backward kernels.
    vmem_limit_bytes: Scoped VMEM limit passed to the Mosaic compiler.
    interpret: Whether to run the Pallas calls in interpret mode.

  Returns:
    A pair containing the branch input and opaque context. Pass the context
    unchanged to :func:`post` after running the attention or feed-forward
    branch on the branch input.
  """
  mhc_common.validate_inputs(x, block_size, permutations.shape)
  mhc_common.validate_token_block_size(x.shape[0] * x.shape[1], bwd_block_size, name="bwd_block_size")
  return _pre_op(
      block_size,
      bwd_block_size,
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
    layer_output: jax.Array,
    context: mhc_common.MHCContext,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bwd_block_size: int = DEFAULT_POST_BWD_BLOCK_SIZE,
    bwd_feature_block_size: int = DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool = False,
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing kernel.

  Args:
    layer_output: Wrapped branch output with shape [batch, sequence, embedding].
    context: Opaque context returned by :func:`pre`.
    block_size: Token-axis Pallas block size for the forward kernel.
    bwd_block_size: Token-axis block size for the post-application backward kernel.
    bwd_feature_block_size: Feature-axis block size for the post-application backward kernel.
    vmem_limit_bytes: Scoped VMEM limit passed to the Mosaic compiler.
    interpret: Whether to run the Pallas call in interpret mode.

  Returns:
    Mixed output streams with shape [batch, sequence, streams, embedding].
  """
  x, h_post, residual = context
  mhc_common.validate_inputs(x, block_size)
  mhc_common.validate_token_block_size(x.shape[0] * x.shape[1], bwd_block_size, name="bwd_block_size")
  mhc_common.validate_feature_block_size(x.shape[-1], bwd_feature_block_size)
  return _post_op(
      block_size,
      bwd_block_size,
      bwd_feature_block_size,
      vmem_limit_bytes,
      interpret,
      x,
      layer_output,
      h_post,
      residual,
  )
