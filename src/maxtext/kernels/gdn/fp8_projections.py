# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""FP8 GatedDeltaNet projections (config gdn_fp8_projections): `in_proj_qkvz` and `out_proj` as FP8 matmuls.

`fp8_projection(x, kernel, splits, layout)` computes `x @ kernel` (one output per column segment) with dynamic absmax
scaling, E4M3 activations / weights and E5M2 gradients, every scale on a non-contracting dimension and applied once
to the f32 accumulator. The token dims of x are flattened into one ([M, K]) around the custom VJP, so every GEMM is a
2-D matmul and the wgrad contracts one token dim:

  forward   y_k  = x8 @ w8_k                       x per token, kernel per output channel (once, reused below)
  dgrad     dx   = sum_k g8_k @ w8_k^T             g8_k = quantize_per_token(dy_k * sw_k): the kernel's output-channel
                                                   scale is folded into dy, so the forward w8_k is reused as is
  wgrad     narrow input (K <= N, in_proj_qkvz):   dW_k = (x * s_k)^T @ g8_k / sw_k, i.e. the per-token scale s_k of
                                                   the dgrad operand folded into x, which is re-quantized per input
                                                   channel; the dgrad operand g8_k is reused as is
            wide input (K > N, out_proj):          dW^T = (dy * sx)^T @ x8, i.e. the forward per-token scale sx folded
                                                   into dy, which is re-quantized per output channel; the forward
                                                   operand x8 is reused as is

So no per-channel pass ever runs over the wide tensor (the 20480-wide in_proj_qkvz gradient or the 8192-wide out_proj
input), the kernel is quantized once per output channel and never per input channel, and with several column
segments (the [qkv | z] outputs of gdn_in_proj_kernel_order) each segment's gradient is quantized per token on its
own: no concatenated gradient is materialized and dx is the f32 sum of the segment dgrads.

The wgrad's re-quantized operand is written transposed, [K or N, M], by `transpose_quant`: a Pallas TPU kernel (one
read of the source, one f32 transpose per block, one output per segment) that gives the FP8 wgrad GEMM its
contracting-dim-minor layout; its custom-call layouts are fixed, so the transposed layout never propagates into the
forward GEMMs (an XLA layout constraint does). Off TPU, or with dims that do not tile, it falls back to the same math
in XLA. Under GSPMD the Pallas call runs inside `shard_map` over the token dim (`layout.token_spec`, the per-dim specs of
the token dims merged major to minor); per-token scales never cross devices and the per-channel amaxes are jnp
reductions over the tokens, so GSPMD reduces them (a 1-D f32 all-reduce); the wgrad partial sums over tokens are
reduced by the partitioner as for the bf16 projection.

The kernel is gathered to `layout.gathered_sharding` by one sharding constraint in the forward and once more in the
backward (quantized after each gather, so the gathers move the compute dtype, as the bf16 projection does) and its
gradient is returned without a sharding constraint, so the partitioner reduces it to the kernel's sharding.
"""

import functools
from typing import NamedTuple, Optional, Sequence

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.kernels.gdn.gdn_bwd.runtime_utils import ensure_cpu_interpret_registered
from maxtext.kernels.gdn.gdn_bwd.runtime_utils import target_platform

E4M3 = jnp.float8_e4m3fn  # activations and kernels
E5M2 = jnp.float8_e5m2  # gradients
_TINY = 1e-30
# transpose_quant block: tokens x channels of the source. 1024 x 2048 measured fastest on one TPU 8i TensorCore for
# [16384, 4096] bf16 -> 2 x FP8 [4096, 16384] (0.114 ms, 2.4 TB/s).
_TQ_BLOCK_TOKENS = 1024
_TQ_BLOCK_CHANNELS = 2048
_LANES = 128


class ProjectionLayout(NamedTuple):
  """Static sharding facts of one FP8 projection call (hashable: a custom_vjp nondiff argument).

  Attributes:
    gathered_sharding: Sharding the kernel is gathered to before it is quantized (None: no mesh).
    mesh: Mesh of `token_spec` (None: no shard_map around the Pallas transpose-quantize).
    token_spec: PartitionSpec of the leading (token) dims of x, e.g. P(batch_axes, length_axes).
  """

  gathered_sharding: Optional[NamedSharding] = None
  mesh: Optional[Mesh] = None
  token_spec: Optional[P] = None


def quantize(x: jax.Array, axes: Sequence[int], dtype) -> tuple[jax.Array, jax.Array]:
  """Dynamic absmax quantization of x to the FP8 dtype, one f32 scale per slice reduced over `axes`.

  Returns (x8, scale) with x ~= x8 * scale; scale keeps x's rank (size 1 on `axes`). Under GSPMD the amax is reduced
  over every device that holds part of a slice.
  """
  xf = x.astype(jnp.float32)
  amax = jnp.max(jnp.abs(xf), axis=tuple(axes), keepdims=True)
  scale = jnp.maximum(amax, _TINY) * (1.0 / float(jnp.finfo(dtype).max))
  return (xf * (1.0 / scale)).astype(dtype), scale


def _scale_from_amax(amax: jax.Array, dtype) -> jax.Array:
  return jnp.maximum(amax, _TINY) * (1.0 / float(jnp.finfo(dtype).max))


def _fp8_dot(a8: jax.Array, b8: jax.Array, contract: tuple[Sequence[int], Sequence[int]]) -> jax.Array:
  """FP8 dot_general without batch dims, accumulated in f32."""
  return jax.lax.dot_general(
      a8, b8, ((tuple(contract[0]), tuple(contract[1])), ((), ())), preferred_element_type=jnp.float32
  )


# ------------------------------------------------------------------------------------------- transpose-quantize
def _tq_kernel(a_ref, *refs):
  """One [tokens, channels] block of the source -> each output's [channels, tokens] block (one shared f32 transpose)."""
  n = len(refs) // 3
  rows, cols, outs = refs[:n], refs[n : 2 * n], refs[2 * n :]
  at = a_ref[...].astype(jnp.float32).T
  for r_ref, c_ref, o_ref in zip(rows, cols, outs):
    fmax = float(jnp.finfo(o_ref.dtype).max)
    o_ref[...] = jnp.clip(at * c_ref[...] * r_ref[...], -fmax, fmax).astype(o_ref.dtype)


def _tq_block(dim: int, target: int) -> Optional[int]:
  """Largest multiple of 128 that divides `dim` and is at most `target` (None if there is none)."""
  block = min(dim, target) // _LANES * _LANES
  while block >= _LANES:
    if dim % block == 0:
      return block
    block -= _LANES
  return None


def _tq_pallas_2d(a, row_scales, col_scales, dtypes, block_tokens, block_channels, interpret):
  """Pallas transpose-quantize of a [M, K] source: outputs [K, M] (row scales [M], column scales [K])."""
  m, k = a.shape
  n = len(dtypes)
  in_specs = (
      [pl.BlockSpec((block_tokens, block_channels), lambda i, j: (i, j))]
      + [pl.BlockSpec((1, block_tokens), lambda i, j: (0, i))] * n  # token factors, in the transposed orientation
      + [pl.BlockSpec((block_channels, 1), lambda i, j: (j, 0))] * n  # channel factors
  )
  out_specs = [pl.BlockSpec((block_channels, block_tokens), lambda i, j: (j, i))] * n
  kwargs = {}
  if interpret:
    ensure_cpu_interpret_registered()
  else:
    kwargs["compiler_params"] = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel"),
        vmem_limit_bytes=int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes),
    )
  outs = pl.pallas_call(
      _tq_kernel,
      grid=(m // block_tokens, k // block_channels),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=[jax.ShapeDtypeStruct((k, m), d) for d in dtypes],
      interpret=interpret,
      name="gdn_fp8_transpose_quant",
      **kwargs,
  )(
      a,
      *[r.astype(jnp.float32).reshape(1, m) for r in row_scales],
      *[c.astype(jnp.float32).reshape(k, 1) for c in col_scales],
  )
  return list(outs)


def _tq_xla(a, row_scales, col_scales, dtypes):
  """XLA form of transpose_quant: the same values, the channel dim moved first."""
  af = a.astype(jnp.float32)

  def one(r, c, dtype):
    fmax = float(jnp.finfo(dtype).max)
    # Same multiplication order as the Pallas kernel (channel factor first), so both forms give the same bits.
    q = jnp.clip(af * c.astype(jnp.float32) * r.astype(jnp.float32), -fmax, fmax).astype(dtype)
    return jnp.moveaxis(q, -1, 0)

  return [one(r, c, dtype) for r, c, dtype in zip(row_scales, col_scales, dtypes)]


def transpose_quant(
    a: jax.Array,
    row_scales: Sequence[jax.Array],
    col_scales: Sequence[jax.Array],
    dtypes: Sequence,
    layout: ProjectionLayout = ProjectionLayout(),
    *,
    use_pallas: Optional[bool] = None,
    interpret: bool = False,
) -> list[jax.Array]:
  """outputs[i] = clip(a * row_scales[i] * col_scales[i]) as dtypes[i], with the channel dim moved first.

  Args:
    a: [*T, K] source (T: token dims).
    row_scales: per output, [*T, 1] per-token factors.
    col_scales: per output, [K] (or broadcastable) per-channel factors.
    dtypes: per output, the FP8 dtype.
    layout: mesh and token PartitionSpec for the shard_map around the Pallas kernel.
    use_pallas: None = the Pallas TPU kernel when compiling for TPU and the local dims tile (multiples of 128),
      else the XLA form; True / False force it (True with interpret=True runs the kernel in Pallas interpret mode).
    interpret: Pallas interpret mode (tests on CPU).

  Returns:
    One [K, *T] array per output.
  """
  token_shape, k = a.shape[:-1], a.shape[-1]
  cols = [jnp.broadcast_to(jnp.reshape(c, (-1,)), (k,)) for c in col_scales]
  rows = [jnp.broadcast_to(r, token_shape + (1,)) for r in row_scales]
  if use_pallas is None:
    use_pallas = target_platform(a) == "tpu"
  if not use_pallas:
    return _tq_xla(a, rows, cols, dtypes)

  sharded = layout.mesh is not None and layout.token_spec is not None
  local_tokens = 1
  for i, size in enumerate(token_shape):
    shards = 1
    if sharded and i < len(layout.token_spec) and layout.token_spec[i] is not None:
      axes = layout.token_spec[i] if isinstance(layout.token_spec[i], tuple) else (layout.token_spec[i],)
      for axis in axes:
        shards *= layout.mesh.shape[axis]
    if size % shards:
      return _tq_xla(a, rows, cols, dtypes)
    local_tokens *= size // shards
  block_tokens = _tq_block(local_tokens, _TQ_BLOCK_TOKENS)
  block_channels = _tq_block(k, _TQ_BLOCK_CHANNELS)
  if block_tokens is None or block_channels is None:
    return _tq_xla(a, rows, cols, dtypes)

  def local(a_local, *scales):
    n = len(dtypes)
    local_token_shape = a_local.shape[:-1]
    m = 1
    for size in local_token_shape:
      m *= size
    outs = _tq_pallas_2d(
        a_local.reshape(m, k),
        [s.reshape(m) for s in scales[:n]],
        list(scales[n:]),
        dtypes,
        block_tokens,
        block_channels,
        interpret,
    )
    return tuple(o.reshape((k,) + local_token_shape) for o in outs)

  if not sharded:
    return list(local(a, *rows, *cols))
  spec = tuple(layout.token_spec) + (None,) * (len(token_shape) - len(layout.token_spec))
  token_spec = P(*spec, None)
  out_spec = P(None, *spec)
  fn = jax.shard_map(
      local,
      mesh=layout.mesh,
      in_specs=(token_spec,) + (token_spec,) * len(dtypes) + (P(None),) * len(dtypes),
      out_specs=(out_spec,) * len(dtypes),
      check_vma=False,
  )
  return list(fn(a, *rows, *cols))


# ------------------------------------------------------------------------------------------- FP8 projection
def _gather(kernel: jax.Array, layout: ProjectionLayout) -> jax.Array:
  if layout.gathered_sharding is None:
    return kernel
  return jax.lax.with_sharding_constraint(kernel, layout.gathered_sharding)


def _segment_bounds(num_cols: int, splits: tuple[int, ...]) -> list[tuple[int, int]]:
  bounds = (0,) + tuple(splits) + (num_cols,)
  return list(zip(bounds[:-1], bounds[1:]))


def _quantized_segments(kernel, splits, layout):
  """The gathered kernel quantized per output channel, one (w8, sw [1, N_k]) per column segment."""
  gathered = _gather(kernel, layout)
  return [quantize(gathered[:, lo:hi], (0,), E4M3) for lo, hi in _segment_bounds(kernel.shape[1], splits)]


def _merged_token_spec(token_spec: P) -> P:
  """The PartitionSpec of the token dims merged into one, major to minor.

  For example P("fsdp", "context") -> P(("fsdp", "context")), the sharding of x.reshape(-1, K) for that x.
  """
  axes = []
  for entry in token_spec:
    if entry is not None:
      axes.extend(entry if isinstance(entry, tuple) else (entry,))
  return P(tuple(axes) if axes else None)


def fp8_projection(
    x: jax.Array, kernel: jax.Array, splits: tuple[int, ...] = (), layout: ProjectionLayout = ProjectionLayout()
) -> tuple[jax.Array, ...]:
  """FP8 `x @ kernel[:, segment]` for every column segment (see the module docstring for the recipe).

  Args:
    x: [*T, K] input in the compute dtype (T: token dims, flattened into one for the matmuls).
    kernel: [K, N] kernel in the compute dtype (possibly sharded; see `layout`).
    splits: column boundaries of the output segments, e.g. (12288,) for [qkv | z]; () for one output.
    layout: kernel gather sharding and token sharding (ProjectionLayout).

  Returns:
    One [*T, N_k] output per segment, in x's dtype.
  """
  token_shape = x.shape[:-1]
  if layout.token_spec is not None:
    layout = layout._replace(token_spec=_merged_token_spec(layout.token_spec))
  outputs = _fp8_projection_2d(jnp.reshape(x, (-1, x.shape[-1])), kernel, tuple(splits), layout)
  return tuple(jnp.reshape(y, token_shape + y.shape[-1:]) for y in outputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _fp8_projection_2d(x, kernel, splits, layout):
  """fp8_projection on a [M, K] input (the custom VJP)."""
  return _fp8_projection_fwd(x, kernel, splits, layout)[0]


def _fp8_projection_fwd(x, kernel, splits, layout):
  """Forward rule of _fp8_projection_2d: (outputs, residuals)."""
  x8, sx = quantize(x, (x.ndim - 1,), E4M3)
  outputs = tuple(
      (_fp8_dot(x8, w8, ((x.ndim - 1,), (0,))) * sx * sw).astype(x.dtype)
      for w8, sw in _quantized_segments(kernel, splits, layout)
  )
  # Narrow input: the backward re-quantizes x (with the dgrad scale folded in); wide input: it reuses x8 and sx.
  if kernel.shape[0] <= kernel.shape[1]:
    residuals = (x, None, None, kernel)
  else:
    residuals = (None, x8, sx, kernel)
  return outputs, residuals


def _fp8_projection_bwd(splits, layout, residuals, cotangents):
  """Backward rule of _fp8_projection_2d: (d_x, d_kernel), from one output cotangent per column segment."""
  x, x8, sx, kernel = residuals
  ndim = cotangents[0].ndim
  token_axes = tuple(range(ndim - 1))
  out_dtype = cotangents[0].dtype
  segments = _quantized_segments(kernel, splits, layout)

  # dgrad: dx = sum_k g8_k @ w8_k^T * s_k, the kernel's output-channel scale folded into dy before its per-token
  # quantization; one f32 sum over the segments.
  g8s, g_scales = [], []
  d_x = None
  for (w8, sw), g in zip(segments, cotangents):
    g8, s_g = quantize(g.astype(jnp.float32) * sw, (ndim - 1,), E5M2)
    g8s.append(g8)
    g_scales.append(s_g)
    part = _fp8_dot(g8, w8, ((ndim - 1,), (1,))) * s_g
    d_x = part if d_x is None else d_x + part
  d_x = d_x.astype(out_dtype)

  d_kernels = []
  if x is not None:
    # Narrow input: dW_k = (x * s_k)^T @ g8_k / sw_k, x * s_k quantized per input channel, written [K, *T].
    xf = x.astype(jnp.float32)
    col_scales = [_scale_from_amax(jnp.max(jnp.abs(xf * s_g), axis=token_axes, keepdims=True), E4M3) for s_g in g_scales]
    a8ts = transpose_quant(x, g_scales, [1.0 / c for c in col_scales], [E4M3] * len(g8s), layout)
    contract = (tuple(range(1, ndim)), token_axes)
    for a8t, c, g8, (_, sw) in zip(a8ts, col_scales, g8s, segments):
      acc = _fp8_dot(a8t, g8, contract) * jnp.reshape(c, (-1, 1))
      d_kernels.append(acc * (1.0 / sw))
  else:
    # Wide input: dW^T = (dy * sx)^T @ x8, dy * sx quantized per output channel, written [N, *T].
    contract = (tuple(range(1, ndim)), token_axes)
    for g in cotangents:
      gf = g.astype(jnp.float32)
      c = _scale_from_amax(jnp.max(jnp.abs(gf * sx), axis=token_axes, keepdims=True), E5M2)
      g8t = transpose_quant(g, [sx], [1.0 / c], [E5M2], layout)[0]
      d_kernel_t = (_fp8_dot(g8t, x8, contract) * jnp.reshape(c, (-1, 1))).astype(kernel.dtype)
      # Keep dW^T = g8t @ x8 as computed: XLA would otherwise rewrite dot(g8t, x8).T as dot(x8^T, g8t^T) and copy x8
      # transposed.
      d_kernels.append(jax.lax.optimization_barrier(d_kernel_t).T)
  d_kernel = d_kernels[0] if len(d_kernels) == 1 else jnp.concatenate(d_kernels, axis=1)
  return d_x, d_kernel.astype(kernel.dtype)


_fp8_projection_2d.defvjp(_fp8_projection_fwd, _fp8_projection_bwd)
