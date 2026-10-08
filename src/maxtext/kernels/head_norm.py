# Copyright 2025-2026 Google LLC
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

"""Per-head normalization of head-flattened `[N, H * d]` activations (Pallas TPU).

`y[:, h*d:(h+1)*d] = scale * x_h * rsqrt(c * sum(x_h**2) + eps)` with f32 math, for
c = 1 (L2 norm, e.g. KDA q/k) or c = 1/d (RMS norm without gain, e.g. the KDA
output norm). On TPU a `[N, H, d]` array is tiled on (H, d) while `[N, H * d]`
is tiled on (N, H * d), so doing the per-head reduction in XLA via a reshape
materializes relayout copies (forward and backward). Here each grid step reads a
`[block_n, d]` column block of the flat array directly, so there is no reshape
at all; forward and backward are one memory-bound pass each.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _fwd_kernel(x_ref, y_ref, *, c, eps, scale):
  x = x_ref[...].astype(jnp.float32)
  r = jax.lax.rsqrt(c * jnp.sum(x * x, axis=-1, keepdims=True) + eps)
  y = x * r
  if scale != 1.0:
    y = y * scale
  y_ref[...] = y.astype(y_ref.dtype)


def _bwd_kernel(x_ref, dy_ref, dx_ref, *, c, eps, scale):
  x = x_ref[...].astype(jnp.float32)
  dy = dy_ref[...].astype(jnp.float32)
  if scale != 1.0:
    dy = dy * scale
  r = jax.lax.rsqrt(c * jnp.sum(x * x, axis=-1, keepdims=True) + eps)
  dot = jnp.sum(dy * x, axis=-1, keepdims=True)
  dx_ref[...] = (r * dy - x * (c * r * r * r * dot)).astype(dx_ref.dtype)


def _call(kernel, args, out_dtype, heads, block_n, interpret):
  n, width = args[0].shape
  d = width // heads
  bn = min(block_n, n)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((n, width), out_dtype),
      grid=(pl.cdiv(n, bn), heads),
      in_specs=[pl.BlockSpec((bn, d), lambda i, h: (i, h)) for _ in args],
      out_specs=pl.BlockSpec((bn, d), lambda i, h: (i, h)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
      name="head_norm_flat",
  )(*args)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def _head_norm_2d(x, heads, mean, eps, scale, out_dtype, block_n, interpret):
  c = 1.0 / (x.shape[-1] // heads) if mean else 1.0
  kernel = functools.partial(_fwd_kernel, c=c, eps=eps, scale=scale)
  return _call(kernel, (x,), out_dtype, heads, block_n, interpret)


def _head_norm_2d_fwd(x, heads, mean, eps, scale, out_dtype, block_n, interpret):
  return _head_norm_2d(x, heads, mean, eps, scale, out_dtype, block_n, interpret), x


def _head_norm_2d_bwd(heads, mean, eps, scale, out_dtype, block_n, interpret, x, dy):
  del out_dtype
  c = 1.0 / (x.shape[-1] // heads) if mean else 1.0
  kernel = functools.partial(_bwd_kernel, c=c, eps=eps, scale=scale)
  dx = _call(kernel, (x, dy), x.dtype, heads, block_n, interpret)
  return (dx,)


_head_norm_2d.defvjp(_head_norm_2d_fwd, _head_norm_2d_bwd)


def head_norm_flat(
    x: jax.Array,
    heads: int,
    *,
    mean: bool,
    eps: float,
    scale: float = 1.0,
    out_dtype=None,
    block_n: int = 512,
    interpret: bool | None = None,
) -> jax.Array:
  """Per-head L2 (`mean=False`) or RMS (`mean=True`) normalization of `[..., H * d]`, no gain.

  Args:
    x: `[..., H * d]`; `d` must be a multiple of 128 on TPU.
    heads: number of heads H.
    mean: divide the sum of squares by d (RMS norm) instead of using it directly (L2 norm).
    eps: added inside the rsqrt.
    scale: constant multiplier folded into the output (e.g. the KDA query scale d**-0.5).
    out_dtype: output dtype (default `x.dtype`); math is f32.
    block_n: rows per grid step.
    interpret: Pallas interpret mode (default: off TPU).

  Returns:
    Normalized `[..., H * d]` in `out_dtype`; the VJP returns `dx` in `x.dtype`.
  """
  if interpret is None:
    interpret = jax.devices()[0].platform != "tpu"
  out_dtype = jnp.dtype(out_dtype or x.dtype)
  lead = x.shape[:-1]
  y = _head_norm_2d(
      x.reshape(-1, x.shape[-1]), heads, bool(mean), float(eps), float(scale), out_dtype, block_n, interpret
  )
  return y.reshape(*lead, x.shape[-1])
