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
"""Public API entrypoints for mHC-lite Pallas TPU kernel."""

from typing import Literal, Sequence
import jax
from maxtext.kernels.mhc import common
from maxtext.kernels.mhc import mhc_kernels_fwd

type Implementation = Literal["mosaic", "mosaic_tpu", "xla"]
MhcContext = common.MHCContext
MhcWeights = common.MhcWeights
MhcKernelConfig = common.MhcKernelConfig
MhcDims = common.MhcDims
MhcCoeffParams = common.MhcCoeffParams
MhcCoeffOutputs = common.MhcCoeffOutputs
MhcCoeffGradients = common.MhcCoeffGradients
hbm_specs = common.hbm_specs


def _validate_implementation(
    implementation: Implementation | Sequence[Implementation] | None,
) -> None:
  """Validates that the requested implementation is supported."""
  if implementation is None:
    return
  valid = ("mosaic", "mosaic_tpu", "xla")
  if isinstance(implementation, str):
    if implementation not in valid:
      raise ValueError(f"Unsupported implementation: '{implementation}'")
    return
  if not any(imp in valid for imp in implementation):
    raise ValueError(f"Unsupported implementation: {implementation}")


def pre(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[jax.Array, MhcContext]:
  """Computes the branch input and opaque context for an mHC-wrapped branch.

  Uses the Pallas TPU kernel when running on TPU and the shape/dtype
  contract is supported.

  Args:
    x: Input streams of shape `(batch, sequence, streams, embedding)`.
    weights: Structured `MhcWeights` container with all layer parameters.
    permutations: All permutation matrices of shape `(num_permutations, streams,
      streams)`.
    config: Structured `MhcKernelConfig` tuning and compiler configuration.
    implementation: Preferred implementation (`"mosaic"` or `"mosaic_tpu"`).

  Returns:
    A tuple `(layer_input, context)` where `layer_input` feeds the wrapped
    model branch, and `context` is passed unchanged to `post`.
  """
  permutations = jax.lax.stop_gradient(permutations)
  _validate_implementation(implementation)
  layer_input, kernel_context = mhc_kernels_fwd.pre(
      x,
      weights,
      permutations,
      config=config,
  )
  x_context, h_post, residual = kernel_context
  return layer_input, MhcContext(
      x=x_context,
      h_post=h_post,
      residual=residual,
      implementation="mosaic",
  )


def post(
    layer_output: jax.Array,
    context: MhcContext,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> jax.Array:
  """Runs the post-gate and residual stream mixing.

  Args:
    layer_output: Output from the wrapped branch of shape `(batch, sequence,
      embedding)`.
    context: Opaque `MhcContext` returned by `pre`.
    config: Structured `MhcKernelConfig` tuning and compiler configuration.

  Returns:
    Mixed output streams of shape `(batch, sequence, streams, embedding)`.
  """
  if context.implementation not in ("mosaic", "mosaic_tpu"):
    raise ValueError(
        f"Unsupported implementation in MhcContext: '{context.implementation}'"
    )
  kernel_context = (context.x, context.h_post, context.residual)
  return mhc_kernels_fwd.post(
      layer_output,
      kernel_context,
      config=config,
  )
