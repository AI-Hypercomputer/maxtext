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

"""Shared routed-expert execution, load-balancing statistics, and overlap hooks.

Model modules own router score computation (softmax, sigmoid, group-limited
top-k, routing bias, shared experts) and pass selected `(weights, indices)` to
a frozen `MoEExecutor` resolved from `(config, sharding)` at construction:

    self.executor = moe.moe_from_config(config, self.sharding)
    ...
    out = self.executor(x, weights, indices, wi_0, wi_1, wo)
    # or for fused (E, D, 2*M) / (E, D, 2, M) gate_up projections:
    out = self.executor.fused(x, weights, indices, gate_up_proj, down_proj)

Resolving `PartitionSpec` axes from `Sharding` at construction avoids reading
`jax.typeof(x).sharding` at trace time, which reports replicated `P(None, ...)`
on `Auto` mesh axes inside `jax.jit` and `ScannedLayers`. Because `MoEExecutor`
is a frozen dataclass of hashable primitives, Flax NNX treats it as a static
attribute across `nnx.split`, `nnx.merge`, `jax.jit`, and layer scan transforms.
"""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Any, Literal

# pylint: disable=too-many-positional-arguments

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import qwix
from qwix import pallas as qpl

from maxtext.m3.core import kernels
from maxtext.m3.core.sharding import MeshAxes, Sharding, TensorType, sharding_from_config

ExecutorKind = Literal["dense", "ragged_dot", "gmm_v2", "tokamax"]
SHARDED_EXECUTOR_KINDS: tuple[str, ...] = ("gmm_v2", "tokamax")

_COMBINE_DTYPE = jnp.float32
_TOKAMAX_M_ALIGN = 256
_DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(),
)


def _align_up(value: int, multiple: int) -> int:
  """Rounds `value` up to a positive multiple of `multiple`."""
  return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def on_tensor_core(fn: Callable[..., Any], enabled: bool = True) -> Callable[..., Any]:
  """Pins `fn` to TensorCore (`compute_type='device'`) so XLA does not auto-offload tiny ops to SparseCore."""
  if not enabled:
    return fn
  from jax.experimental.compute_on import compute_on  # pylint: disable=import-outside-toplevel

  return compute_on(
      fn,
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )


def on_any_sparse_core(fn: Callable[..., Any], enabled: bool = True) -> Callable[..., Any]:
  """Offloads `fn` to TPU SparseCore (`compute_type='tpu_sparsecore'`)."""
  if not enabled:
    return fn
  from jax.experimental.compute_on import compute_on  # pylint: disable=import-outside-toplevel

  return compute_on(
      fn,
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
  )


@jax.custom_vjp
def _backward_wgrad_tie(
    x: jax.Array,
    weights: tuple[jax.Array, ...],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
  """Ties token input cotangents to expert weight cotangents in backward.

  Fencing `dx` with local expert weight gradients (`dw`) inside `shard_map`
  forces expert TGMM kernels to run immediately after `dx` grouped matmuls, so
  FSDP weight-gradient reduce-scatters overlap with the preceding sublayer's
  attention or linear-attention backward pass instead of queuing at block end.
  """
  return x, weights


def _backward_wgrad_tie_fwd(
    x: jax.Array,
    weights: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, tuple[jax.Array, ...]], None]:
  return (x, weights), None


def _backward_wgrad_tie_bwd(
    _: None,
    cotangents: tuple[jax.Array, tuple[jax.Array, ...]],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
  dx, dweights = jax.lax.optimization_barrier(cotangents)
  return dx, dweights


_backward_wgrad_tie.defvjp(_backward_wgrad_tie_fwd, _backward_wgrad_tie_bwd)


# ==============================================================================
# Permutation & grouped-matmul primitives
# ==============================================================================


@jax.custom_vjp
def _permute_rows(
    inputs: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array | None = None,
) -> jax.Array:
  """Permutes axis 0 by bijection `perm` with gather-only inverse VJP."""
  del inv_perm
  return inputs[perm]


def _permute_rows_fwd(
    inputs: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array | None, jax.Array | None]]:
  if inv_perm is not None:
    return inputs[perm], (None, inv_perm)
  return inputs[perm], (perm, None)


def _permute_rows_bwd(
    res: tuple[jax.Array | None, jax.Array | None],
    grad: jax.Array,
) -> tuple[jax.Array, None, None]:
  perm, inv_perm = res
  inv = inv_perm if inv_perm is not None else jnp.argsort(perm)
  return grad[inv], None, None


_permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _gather_token_slots(
    flat_x: jax.Array,
    perm: jax.Array,
    inv_perm_bwd: jax.Array,
    top_k: int,
    slot_major_bwd: bool,
) -> jax.Array:
  """Gathers token rows directly into expert-sorted slot order with scatter-free inverse VJP."""
  del inv_perm_bwd, slot_major_bwd
  return flat_x[perm // top_k]


def _gather_token_slots_fwd(
    flat_x: jax.Array,
    perm: jax.Array,
    inv_perm_bwd: jax.Array,
    top_k: int,
    slot_major_bwd: bool,
) -> tuple[jax.Array, tuple[jax.Array, int]]:
  """Forward rule for `_gather_token_slots`."""
  del slot_major_bwd
  return flat_x[perm // top_k], (inv_perm_bwd, flat_x.shape[0])


def _gather_token_slots_bwd(
    top_k: int,
    slot_major_bwd: bool,
    res: tuple[jax.Array, int],
    grad: jax.Array,
) -> tuple[jax.Array, None, None]:
  """Backward rule for `_gather_token_slots` using a single gather + sum."""
  inv_perm_bwd, num_tokens = res
  d_dim = grad.shape[1]
  d_slots = grad[inv_perm_bwd]
  if slot_major_bwd:
    d_flat_x = jnp.sum(jnp.reshape(d_slots, (top_k, num_tokens, d_dim)), axis=0)
  else:
    d_flat_x = jnp.sum(jnp.reshape(d_slots, (num_tokens, top_k, d_dim)), axis=1)
  return d_flat_x, None, None


_gather_token_slots.defvjp(_gather_token_slots_fwd, _gather_token_slots_bwd)


def _chunk_indices_from_ends(chunk_ends: jax.Array, out_len: int) -> jax.Array:
  """Computes `jnp.searchsorted(chunk_ends, jnp.arange(out_len), side='right')` in O(out_len) prefix sum."""
  if out_len <= 0:
    return jnp.zeros((0,), dtype=jnp.int32)
  clamped_ends = jnp.clip(chunk_ends.astype(jnp.int32), 0, out_len)
  delta = jnp.zeros((out_len + 1,), dtype=jnp.int32).at[clamped_ends].add(1)
  return jnp.cumsum(delta[:-1])


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _ste_quantize_tokens(
    x_f32: jax.Array,
    qtype: Any,
    channelwise_axes: tuple[int, ...],
    calibration_method: str,
) -> qpl.QArray:
  """Quantizes `float32` token activations to `qpl.QArray` with exact STE backward (`g_qa.qvalue / scale`)."""
  return qpl.quantize(
      x_f32,
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      calibration_method=calibration_method,
  )


def _ste_quantize_tokens_fwd(
    x_f32: jax.Array,
    qtype: Any,
    channelwise_axes: tuple[int, ...],
    calibration_method: str,
) -> tuple[qpl.QArray, jax.Array]:
  """Forward rule for `_ste_quantize_tokens`."""
  qa = _ste_quantize_tokens(x_f32, qtype, channelwise_axes, calibration_method)
  return qa, checkpoint_name(qa.scale, "moe_routing")


def _ste_quantize_tokens_bwd(
    qtype: Any,
    channelwise_axes: tuple[int, ...],
    calibration_method: str,
    scale: jax.Array,
    g_qa: qpl.QArray,
) -> tuple[jax.Array]:
  """STE backward rule dividing quantized cotangents by `scale`."""
  del qtype, channelwise_axes, calibration_method
  scale_f32 = scale.astype(jnp.float32)
  safe_scale = jnp.where(scale_f32 != 0.0, scale_f32, 1.0)
  g_x = jnp.where(scale_f32 != 0.0, g_qa.qvalue.astype(jnp.float32) / safe_scale, 0.0)
  return (g_x,)


_ste_quantize_tokens.defvjp(_ste_quantize_tokens_fwd, _ste_quantize_tokens_bwd)


def _window_token_scale(
    scale_1d: jax.Array,
    token_ids_sorted: jax.Array,
    count: jax.Array,
    cap: int,
    start: jax.Array | None = None,
) -> jax.Array:
  """Gathers per-token scales into a truncated `[cap]` ragged buffer (`0` past `count`)."""
  if start is not None:
    token_ids_sorted = jax.lax.dynamic_slice_in_dim(jnp.pad(token_ids_sorted, (0, cap)), start, cap, axis=0)
  valid = jnp.arange(cap, dtype=jnp.int32) < count
  safe_ids = jnp.where(valid, jnp.clip(token_ids_sorted, 0, scale_1d.shape[0] - 1), 0)
  return jnp.where(valid, scale_1d[safe_ids], jnp.zeros((), scale_1d.dtype))


@jax.custom_vjp
def _ste_quantize_routed_weight(w_local: jax.Array) -> qpl.QArray:
  """Quantizes local `[e_loc, K, N]` weights to `float8_e4m3fn` via per-`(expert, out_channel)` STE."""
  qa = qpl.quantize(
      w_local.astype(jnp.float32),
      qtype=jnp.float8_e4m3fn,
      channelwise_axes=(0, 2),
      calibration_method="absmax",
  )
  return qpl.QArray(qvalue=qa.qvalue, scale=jax.lax.stop_gradient(qa.scale))


def _ste_quantize_routed_weight_fwd(w_local: jax.Array) -> tuple[qpl.QArray, jax.Array]:
  """Forward rule for `_ste_quantize_routed_weight`."""
  return _ste_quantize_routed_weight(w_local), jnp.zeros((), dtype=w_local.dtype)


def _ste_quantize_routed_weight_bwd(dtype_proto: jax.Array, g_qa: qpl.QArray) -> tuple[jax.Array]:
  """Backward rule passing through unscaled `drhs` from `megablox.ops._gmm_bwd`."""
  return (g_qa.qvalue.astype(dtype_proto.dtype),)


_ste_quantize_routed_weight.defvjp(_ste_quantize_routed_weight_fwd, _ste_quantize_routed_weight_bwd)


def _ep_all_gather_aligned128(arr: jax.Array, ep_axis: str, *, is_tpu: bool = False) -> jax.Array:
  """All-gathers `arr` along axis 0 over `ep_axis` on TensorCore ICI in 128-lane minor tiles."""

  def _impl(x_in: jax.Array) -> jax.Array:
    flat = jnp.reshape(x_in, (-1,))
    n = flat.shape[0]
    n_pad = _align_up(n, 128)
    flat_pad = jnp.pad(flat, (0, n_pad - n)) if n_pad > n else flat
    ag_3d = jax.lax.all_gather(jnp.reshape(flat_pad, (n_pad // 128, 128)), ep_axis, axis=0, tiled=False)
    ep_count = ag_3d.shape[0]
    ag_2d = jnp.reshape(ag_3d, (ep_count, n_pad))
    if n_pad > n:
      ag_2d = ag_2d[:, :n]
    return jnp.reshape(ag_2d, (ep_count * x_in.shape[0], *x_in.shape[1:]))

  return on_tensor_core(_impl, is_tpu)(arr)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _ep_all_gather_weights_aligned128(
    arr: jax.Array,
    ep_axis: str,
    is_tpu: bool = False,
) -> jax.Array:
  """All-gathers routing weights on TensorCore in forward; `psum_scatter`s cotangents in 128-lane tiles in backward."""
  return _ep_all_gather_aligned128(arr, ep_axis, is_tpu=is_tpu)


def _ep_all_gather_weights_aligned128_fwd(
    arr: jax.Array,
    ep_axis: str,
    is_tpu: bool = False,
) -> tuple[jax.Array, tuple[tuple[int, ...], jax.Array]]:
  """Forward rule for `_ep_all_gather_weights_aligned128`."""
  return _ep_all_gather_weights_aligned128(arr, ep_axis, is_tpu), (arr.shape, jnp.zeros((), dtype=arr.dtype))


def _ep_all_gather_weights_aligned128_bwd(
    ep_axis: str,
    is_tpu: bool,
    res: tuple[tuple[int, ...], jax.Array],
    g_ag: jax.Array,
) -> tuple[jax.Array]:
  """Backward rule `psum_scatter`ing weight cotangents in 128-lane tiles."""
  del is_tpu
  orig_shape, dtype_proto = res
  orig_dtype = dtype_proto.dtype
  n_loc = math.prod(orig_shape)
  if n_loc == 0:
    return (jnp.zeros(orig_shape, dtype=orig_dtype),)
  ep_count = g_ag.size // n_loc
  n_pad = _align_up(n_loc, 1024)
  rs_dtype = g_ag.dtype if g_ag.dtype in (jnp.bfloat16, jnp.float32) else jnp.float32
  g_2d = jnp.reshape(g_ag.astype(rs_dtype), (ep_count, n_loc))
  g_pad = jnp.pad(g_2d, ((0, 0), (0, n_pad - n_loc))) if n_pad > n_loc else g_2d
  g_aligned = jnp.reshape(g_pad, (ep_count * (n_pad // 128), 128))
  g_loc_2d = jax.lax.psum_scatter(g_aligned, ep_axis, scatter_dimension=0, tiled=True)
  g_loc_flat = jnp.reshape(g_loc_2d, (n_pad,))[:n_loc]
  return (jnp.reshape(g_loc_flat, orig_shape).astype(orig_dtype),)


_ep_all_gather_weights_aligned128.defvjp(
    _ep_all_gather_weights_aligned128_fwd,
    _ep_all_gather_weights_aligned128_bwd,
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _ep_gather_tokens(
    x_loc: jax.Array,
    ep_axis: str,
    bwd_dtype: Any,
    bwd_on_tc: bool = False,
    after: Any = None,
) -> jax.Array:
  """All-gathers token payload (FP8 or BF16) over `ep_axis` in forward; `psum_scatter`s in `bwd_dtype` in backward."""
  del bwd_dtype, bwd_on_tc
  if after is not None:
    x_loc, _ = jax.lax.optimization_barrier((x_loc, jax.lax.stop_gradient(after)))
  return jax.lax.all_gather(x_loc, ep_axis, axis=0, tiled=True)


def _ep_gather_tokens_fwd(
    x_loc: jax.Array,
    ep_axis: str,
    bwd_dtype: Any,
    bwd_on_tc: bool = False,
    after: Any = None,
) -> tuple[jax.Array, None]:
  """Forward rule for `_ep_gather_tokens`."""
  return _ep_gather_tokens(x_loc, ep_axis, bwd_dtype, bwd_on_tc, after), None


def _ep_gather_tokens_bwd(
    ep_axis: str,
    bwd_dtype: Any,
    bwd_on_tc: bool,
    _: None,
    g_ag: jax.Array,
) -> tuple[jax.Array, None]:
  """Backward rule `psum_scatter`ing gathered token cotangents."""
  g_cast = g_ag.astype(bwd_dtype)

  def _rs_fn(arr: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(arr, ep_axis, scatter_dimension=0, tiled=True)

  if bwd_on_tc:
    return on_tensor_core(_rs_fn, True)(g_cast), None
  return _rs_fn(g_cast), None


_ep_gather_tokens.defvjp(_ep_gather_tokens_fwd, _ep_gather_tokens_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _ep_combine_tokens(
    y_ag: jax.Array,
    ep_axis: str,
    use_fp8_bwd: bool,
    is_tpu: bool = False,
    fwd_on_tc: bool = False,
) -> jax.Array:
  """Reduce-scatters EP partial token sums in forward; rowwise-`e5m2` (or BF16) `all_gather`s cotangents in backward."""
  del use_fp8_bwd

  def _rs_fn(arr: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(arr, ep_axis, scatter_dimension=0, tiled=True)

  if fwd_on_tc and is_tpu:
    return on_tensor_core(_rs_fn, True)(y_ag)
  return _rs_fn(y_ag)


def _ep_combine_tokens_fwd(
    y_ag: jax.Array,
    ep_axis: str,
    use_fp8_bwd: bool,
    is_tpu: bool = False,
    fwd_on_tc: bool = False,
) -> tuple[jax.Array, None]:
  """Forward rule for `_ep_combine_tokens`."""
  return _ep_combine_tokens(y_ag, ep_axis, use_fp8_bwd, is_tpu, fwd_on_tc), None


def _ep_combine_tokens_bwd(
    ep_axis: str,
    use_fp8_bwd: bool,
    is_tpu: bool,
    fwd_on_tc: bool,
    _: None,
    g_loc: jax.Array,
) -> tuple[jax.Array]:
  """Backward rule all-gathering combined token cotangents in FP8 `e5m2` or BF16."""
  del fwd_on_tc
  if not use_fp8_bwd:
    return (jax.lax.all_gather(g_loc, ep_axis, axis=0, tiled=True),)
  g_qa = qpl.quantize(
      g_loc.astype(jnp.float32),
      qtype=jnp.float8_e5m2,
      channelwise_axes=(0,),
      calibration_method="absmax",
  )
  qval_ag = jax.lax.all_gather(g_qa.qvalue, ep_axis, axis=0, tiled=True)
  scale_ag_1d = _ep_all_gather_aligned128(jnp.reshape(g_qa.scale, (-1,)), ep_axis, is_tpu=is_tpu)
  scale_ag = jnp.reshape(scale_ag_1d, (qval_ag.shape[0],) + (1,) * (qval_ag.ndim - 1))
  return ((qval_ag.astype(jnp.float32) * scale_ag).astype(g_loc.dtype),)


_ep_combine_tokens.defvjp(_ep_combine_tokens_fwd, _ep_combine_tokens_bwd)


def _cpu_qarray_gmm(
    lhs: jax.Array | qpl.QArray,
    rhs: jax.Array | qpl.QArray,
    group_sizes: jax.Array,
    preferred_element_type: Any,
    *,
    moe_fp8: bool,
    out_is_3d: bool,
) -> jax.Array:
  """CPU reference mirroring `megablox.ops.gmm` (`use_gmm_v2=True`) QArray scale & `e5m2` backward semantics."""
  lhs_is_qa = isinstance(lhs, qpl.QArray)
  rhs_is_qa = isinstance(rhs, qpl.QArray)

  @jax.custom_vjp
  def _impl(l_in: Any, r_in: Any, gs: jax.Array) -> jax.Array:
    lv = l_in.qvalue if lhs_is_qa else l_in
    rv = r_in.qvalue if rhs_is_qa else r_in
    m = lv.shape[0]
    k = rv.shape[1]
    n = rv.shape[2]
    l_mat = jnp.reshape(lv, (m, k)).astype(jnp.float32)
    l_scale_dyn = None
    if not lhs_is_qa and moe_fp8:
      lq_fwd = qpl.quantize(l_mat, qtype=jnp.float8_e4m3fn, channelwise_axes=(0,), calibration_method="absmax")
      l_mat = lq_fwd.qvalue.astype(jnp.float32)
      l_scale_dyn = lq_fwd.scale.astype(jnp.float32)
    out = jax.lax.ragged_dot(
        l_mat,
        rv.astype(jnp.float32),
        gs.astype(jnp.int32),
    )
    if rhs_is_qa:
      rs_row = jnp.repeat(jnp.squeeze(r_in.scale, axis=1).astype(jnp.float32), gs, axis=0, total_repeat_length=m)
      out = out * rs_row
    elif moe_fp8:
      rq = qpl.quantize(
          rv.astype(jnp.float32), qtype=jnp.float8_e4m3fn, channelwise_axes=(0, 2), calibration_method="absmax"
      )
      out = jax.lax.ragged_dot(l_mat, rq.qvalue.astype(jnp.float32), gs.astype(jnp.int32))
      out = out * jnp.repeat(jnp.squeeze(rq.scale, axis=1).astype(jnp.float32), gs, axis=0, total_repeat_length=m)
    if lhs_is_qa:
      out = out * jnp.reshape(l_in.scale.astype(jnp.float32), (m, 1))
    elif l_scale_dyn is not None:
      out = out * l_scale_dyn
    out_cast = out.astype(preferred_element_type)
    if out_is_3d:
      return jnp.reshape(out_cast, (m, n // 128, 128))
    return out_cast

  def _fwd(l_in: Any, r_in: Any, gs: jax.Array):
    return _impl(l_in, r_in, gs), (l_in, r_in, gs)

  def _bwd(res: Any, grad: jax.Array):
    l_in, r_in, gs = res
    lv = l_in.qvalue if lhs_is_qa else l_in
    rv = r_in.qvalue if rhs_is_qa else r_in
    m = lv.shape[0]
    k = rv.shape[1]
    n = rv.shape[2]
    g2d = jnp.reshape(grad, (m, n)).astype(jnp.float32)
    dlhs_dout = g2d
    drhs_dout = g2d
    if rhs_is_qa:
      rs_row = jnp.repeat(jnp.squeeze(r_in.scale, axis=1).astype(jnp.float32), gs, axis=0, total_repeat_length=m)
      dlhs_dout = dlhs_dout * rs_row
    if lhs_is_qa:
      drhs_dout = drhs_dout * jnp.reshape(l_in.scale.astype(jnp.float32), (m, 1))
    elif moe_fp8:
      lq = qpl.quantize(
          jnp.reshape(lv, (m, k)).astype(jnp.float32),
          qtype=jnp.float8_e4m3fn,
          channelwise_axes=(0,),
          calibration_method="absmax",
      )
      lv = jnp.reshape(lq.qvalue, lv.shape)
      drhs_dout = drhs_dout * lq.scale
    if moe_fp8:
      dlq = qpl.quantize(dlhs_dout, qtype=jnp.float8_e5m2, channelwise_axes=(0,), calibration_method="absmax")
      drq = qpl.quantize(drhs_dout, qtype=jnp.float8_e5m2, channelwise_axes=(1,), calibration_method="absmax")
      dlhs_dout = dlq.qvalue.astype(jnp.float32) * dlq.scale
      drhs_dout = drq.qvalue.astype(jnp.float32) * drq.scale
    dlhs_2d = jax.lax.ragged_dot(dlhs_dout, jnp.swapaxes(rv.astype(jnp.float32), 1, 2), gs.astype(jnp.int32))
    dlhs = jnp.reshape(dlhs_2d.astype(preferred_element_type), lv.shape)
    drhs = jax.lax.ragged_dot_general(
        jnp.reshape(lv, (m, k)).astype(jnp.float32),
        drhs_dout,
        gs.astype(jnp.int32),
        _DRHS_RAGGED_DOT_DIM_NUMS,
    ).astype(preferred_element_type)
    if lhs_is_qa:
      dlhs = dlhs * jnp.reshape(l_in.scale, (m, *([1] * (dlhs.ndim - 1)))).astype(dlhs.dtype)
      dlhs_out: Any = qpl.QArray(
          qvalue=dlhs,
          scale=jnp.zeros_like(l_in.scale),
          zero_point=jnp.zeros_like(l_in.zero_point) if l_in.zero_point is not None else None,
          qtype=l_in.qtype,
      )
    else:
      dlhs_out = dlhs
    if rhs_is_qa:
      drhs_out: Any = qpl.QArray(
          qvalue=drhs,
          scale=jnp.zeros_like(r_in.scale),
          zero_point=jnp.zeros_like(r_in.zero_point) if r_in.zero_point is not None else None,
          qtype=r_in.qtype,
      )
    else:
      drhs_out = drhs
    return dlhs_out, drhs_out, None

  _impl.defvjp(_fwd, _bwd)
  return _impl(lhs, rhs, group_sizes)


def _gmm_fp8_or_ref(
    lhs: jax.Array | qpl.QArray,
    rhs: jax.Array | qpl.QArray,
    group_sizes: jax.Array,
    preferred_element_type: Any,
    *,
    moe_fp8: bool,
    is_tpu: bool,
    out_is_3d: bool = False,
    use_fused_bwd_quant: bool = False,
) -> jax.Array:
  """Dispatches pre-quantized `QArray` (or BF16) grouped matmul to `megablox.ops.gmm` (`gmm_v2`) on TPU or CPU ref."""
  if not is_tpu:
    return _cpu_qarray_gmm(lhs, rhs, group_sizes, preferred_element_type, moe_fp8=moe_fp8, out_is_3d=out_is_3d)
  from maxtext.kernels.megablox import ops as megablox_ops  # pylint: disable=import-outside-toplevel

  orig_m = (lhs.qvalue if isinstance(lhs, qpl.QArray) else lhs).shape[0]
  padded_m = _align_up(orig_m, _TOKAMAX_M_ALIGN)
  pad_rows = padded_m - orig_m
  if pad_rows > 0:
    if isinstance(lhs, qpl.QArray):
      pq = jnp.pad(lhs.qvalue, ((0, pad_rows), *([(0, 0)] * (lhs.qvalue.ndim - 1))))
      ps = jnp.pad(lhs.scale, ((0, pad_rows), *([(0, 0)] * (lhs.scale.ndim - 1)))) if lhs.scale.size > 1 else lhs.scale
      lhs_in: jax.Array | qpl.QArray = dataclasses.replace(lhs, qvalue=pq, scale=ps)
    else:
      lhs_in = jnp.pad(lhs, ((0, pad_rows), *([(0, 0)] * (lhs.ndim - 1))))
  else:
    lhs_in = lhs

  quant_rule = (
      qwix.QtRule(
          module_path=".*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method="absmax",
          act_calibration_method="absmax",
          bwd_calibration_method="absmax",
          disable_channelwise_axes=False,
          op_names=("gmm",),
      )
      if moe_fp8
      else None
  )
  out = megablox_ops.gmm(
      lhs_in,
      rhs,
      group_sizes.astype(jnp.int32),
      preferred_element_type=preferred_element_type,
      use_qwix_quantization=moe_fp8,
      qwix_rule=quant_rule,
      use_tokamax_backend=True,
      use_gmm_v2=True,
      use_gmm_v2_heuristic_tiling=True,
      use_dlhs_transpose_rhs=True,
      out_is_3d=out_is_3d,
      use_fused_bwd_quant=use_fused_bwd_quant and moe_fp8,
  )
  return out[:orig_m] if pad_rows > 0 else out


@jax.custom_vjp
def _window_gather_rows(
    x_ag: jax.Array,
    token_ids_win: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
  """Gathers `x_ag[token_ids_win]` on valid window rows, propagating full-precision cotangents for FP8 primals."""
  safe_tok = jnp.where(valid_mask, token_ids_win, 0)
  bcast = valid_mask.reshape(valid_mask.shape[0], *([1] * (x_ag.ndim - 1)))
  return jnp.where(bcast, x_ag[safe_tok], jnp.zeros((), x_ag.dtype))


def _window_gather_rows_fwd(
    x_ag: jax.Array,
    token_ids_win: jax.Array,
    valid_mask: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, int]]:
  return _window_gather_rows(x_ag, token_ids_win, valid_mask), (token_ids_win, valid_mask, x_ag.shape[0])


def _window_gather_rows_bwd(
    res: tuple[jax.Array, jax.Array, int],
    g_sorted: jax.Array,
) -> tuple[jax.Array, None, None]:
  token_ids_win, valid_mask, n_tok = res
  safe_tok = jnp.where(valid_mask, token_ids_win, 0)
  bcast = valid_mask.reshape(valid_mask.shape[0], *([1] * (g_sorted.ndim - 1)))
  g_valid = jnp.where(bcast, g_sorted, jnp.zeros((), g_sorted.dtype))
  dx_ag = jnp.zeros((n_tok, *g_sorted.shape[1:]), dtype=g_sorted.dtype).at[safe_tok].add(g_valid)
  return dx_ag, None, None


_window_gather_rows.defvjp(_window_gather_rows_fwd, _window_gather_rows_bwd)


def _clamp_group_sizes_to_cap(gs: jax.Array, cap: int) -> jax.Array:
  """Clamps cumulative per-expert counts to buffer capacity `cap`."""
  cumsum = jnp.cumsum(gs.astype(jnp.int32))
  clamped_end = jnp.minimum(cumsum, cap)
  clamped_start = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), clamped_end[:-1]])
  return (clamped_end - clamped_start).astype(jnp.int32)


def _tag_disp_ep(arr: jax.Array, disp_ep_tags: tuple[str, ...]) -> jax.Array:
  """Applies each configured post-dispatch checkpoint tag in `disp_ep_tags`."""
  for tag in disp_ep_tags:
    arr = checkpoint_name(arr, tag)
  return arr


def _matmul_precision(dtype: Any) -> jax.lax.Precision:
  """Returns single-pass MXU `DEFAULT` for `bfloat16` and `HIGHEST` otherwise."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16):
    return jax.lax.Precision.DEFAULT
  return jax.lax.Precision.HIGHEST


_pallas_grouped_matmul = kernels._vendored_pallas_grouped_matmul  # pylint: disable=protected-access
grouped_matmul = kernels.grouped_matmul


# ==============================================================================
# Executor descriptor & config resolution
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class MoEExecutor:
  """Static routed-expert execution descriptor resolved from `(config, sharding)`."""

  kind: ExecutorKind = "dense"
  batch: MeshAxes = None
  experts: MeshAxes = None
  mlp: MeshAxes = None
  capacity_factor: float = -1.0
  ragged_buffer_factor: float = -1.0
  use_ragged_sort: bool = False
  length: MeshAxes = None
  disp_ep_tags: tuple[str, ...] = ("moe_disp_ep",)
  bwd_wgrad_overlap: bool = False
  ep_allgather_tc: bool = False
  moe_fp8: bool = False
  ep_combine_mode: str = "auto"
  row_scale_placement: str = "tc"
  tc_3d_gmm: bool = False
  fused_bwd_quant: bool = False
  tc_ragged_flatten_block_size: int = 0

  def __call__(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array,
      wi_0: jax.Array,
      wi_1: jax.Array,
      wo: jax.Array,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
  ) -> jax.Array:
    """Executes routed experts with separate `(wi_0, wi_1, wo)` projections."""
    return routed_experts(
        x,
        weights,
        indices,
        wi_0,
        wi_1,
        wo,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

  def fused(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array,
      gate_up_proj: jax.Array,
      down_proj: jax.Array,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
  ) -> jax.Array:
    """Executes routed experts with 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
    return routed_experts_fused(
        x,
        weights,
        indices,
        gate_up_proj,
        down_proj,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


DENSE = MoEExecutor("dense")
RAGGED_DOT = MoEExecutor("ragged_dot")


_EP_PARALLELISM_ATTRS: tuple[str, ...] = ("ici_expert_parallelism", "dcn_expert_parallelism")
_TP_PARALLELISM_ATTRS: tuple[str, ...] = (
    "ici_tensor_parallelism",
    "dcn_tensor_parallelism",
    "ici_tensor_sequence_parallelism",
    "dcn_tensor_sequence_parallelism",
)

_UNSUPPORTED_BOOL_OFF_MOE_FIELDS: tuple[str, ...] = (
    "use_random_routing",
    "use_te_grouped_gemm",
    "te_moe_block",
    "te_moe_disable_cudnn_gmm_fusion",
    "te_moe_quantize_before_fsdp_all_gather",
    "fp8_moe",
    "fp8_moe_fake_quant",
    "moe_flat_fsdp_weights",
    "shard_exp_on_fsdp",
    "shard_embed_moe_on_fsdp",
    "use_2d_fsdp_sharding",
    "moe_fsdp_use_two_stage_all_gather",
    "moe_wo_fsdp_on_mlp",
    "moe_use_fused_gmm3d",
    "moe_dispatch_no_expert_sharding",
    "moe_use_megatron_seq_aux_loss",
    "moe_exclude_padding_from_routing",
    "float32_gate_logits",
    "merge_gating_gmm",
    "moe_pin_sparse_core_all_gathers",
    "moe_pin_sparse_core_ep_all_gathers",
    "moe_pin_sparse_core_fsdp_all_gathers",
    "moe_pin_sparse_core_fsdp_all_gathers_fwd_only",
    "moe_quantize_token_all_gather",
    "moe_accumulate_chunk_wgrad",
    "moe_accumulate_wi_dlhs",
    "moe_gmm_v2_dlhs_transpose_rhs",
    "moe_gmm_v2_tail_chunks",
    "moe_bwd_prequant_before_unsort",
    "moe_combine_bwd_direct_qarray",
    "moe_3d_dlhs_partial_sum",
    "moe_chunk_barrier",
    "moe_log_max_load_ratio",
    "log_required_ragged_buffer_factor",
    "router_topk_matmul_vjp",
    "moe_topk_before_ep_all_gather",
    "moe_tc_ragged_sort",
    "moe_tc_ragged_weights_on_activation",
    "moe_tc_ragged_3d_gmm",
    "moe_tc_ragged_3d_dispatch",
    "moe_tc_window_routing",
    "moe_use_direct_token_gather",
    "moe_packed_routing_sort",
    "moe_tc_routing_offload_free",
    "moe_unsliced_prefused_wi",
    "prefuse_moe_weights",
    "moe_tc_combine",
    "moe_tc_permute",
    "use_gather_mosaic_kernel",
    "ragged_sort_use_single_sparsecore",
    "ragged_unsort_spread_masked_indices",
    "ragged_gather_fallback",
    "ragged_gather_reduce_fallback",
    "use_gmm_v2_heuristic_tiling",
    "defer_small_all_reduces",
    "use_batch_split_schedule",
    "use_manual_quantization",
    "moe_fused_gmm3d_prequant_dy",
    "moe_fused_gmm3d_early_dwo_rs",
    "moe_fused_gmm3d_sc_routed_gather",
    "moe_fused_gmm3d_exact_topk_combine",
    "moe_fused_gmm3d_exact_topk_out_2d",
    "moe_fused_gmm3d_inv_perm_combine",
    "moe_fused_gmm3d_relayout_x_2d_to_3d",
    "moe_fused_gmm3d_fused_quant_relayout",
    "moe_fused_gmm3d_overlap_dcoeffs_scatter",
)

_UNSUPPORTED_BOOL_ON_MOE_FIELDS: tuple[str, ...] = (
    "float32_weight_sum",
    "use_custom_sort_vjp",
    "moe_tc_ragged_mask_padding",
    "moe_fused_gmm3d_gmm_combine",
    "moe_fused_gmm3d_split_gate_bwd",
    "moe_fused_gmm3d_split_dw_gate",
)

_UNSUPPORTED_EMPTY_STR_MOE_FIELDS: tuple[str, ...] = (
    "te_gmm_quantization",
    "moe_quantize_combine_bwd_method",
    "moe_quantize_dispatch_fwd_method",
    "moe_quantize_combine_fwd_method",
    "moe_quantize_dispatch_bwd_method",
)

_UNSUPPORTED_ENUM_STR_MOE_FIELDS: dict[str, str] = {
    "moe_weight_quantization_granularity": "per_channel",
    "moe_routing_index_math": "sort",
}

_UNSUPPORTED_FLOAT_MOE_FIELDS: dict[str, float] = {
    "eval_ragged_buffer_factor": -1.0,
    "first_phase_ragged_buffer_factor": 0.0,
}

_UNSUPPORTED_INT_MOE_FIELDS: dict[str, int] = {
    "te_ep_overflow_check_every_n_steps": 20,
    "num_moe_token_chunks": 1,
    "num_moe_emb_chunks": 0,
    "expert_replicas_per_device": 0,
    "expert_replica_max_send": 4,
    "retry_dropless_first_steps": 0,
    "batch_split_factor": 1,
    "moe_wi_ag_latency_ns": 0,
    "moe_fsdp_all_gather_sparse_core_id": 0,
    "moe_ep_all_gather_sparse_core_id": 1,
    "moe_tc_ragged_gather_block_size": 1024,
    "moe_tc_ragged_reduce_block_size": 896,
    "moe_tc_combine_block_tokens": 256,
    "moe_tc_combine_expert_chunks": 1,
    "moe_tc_combine_col_chunks": 1,
    "moe_tc_permute_block_tokens": 512,
    "moe_tc_permute_expert_chunks": 1,
    "moe_tc_permute_col_chunks": 1,
    "moe_fused_gmm3d_gate_tile_m": 768,
    "moe_fused_gmm3d_gate_tile_k": 1024,
    "moe_fused_gmm3d_dgrad_tile_rows": 256,
    "moe_fused_gmm3d_dgrad_bucket_rows": 128,
    "moe_fused_gmm3d_dgrad_tile_n": 512,
    "moe_fused_gmm3d_dx_tile_m": 512,
    "moe_fused_gmm3d_dx_tile_k": 3072,
    "moe_fused_gmm3d_dx_tile_n": 2048,
    "moe_fused_gmm3d_dw_tile_k": 1024,
    "moe_fused_gmm3d_dgrad_tile_k": 1024,
    "moe_fused_gmm3d_wgrad_schedule_mode": 0,
    "ragged_gather_cost_estimate_flops": -1,
    "ragged_gather_reduce_cost_estimate_flops": -1,
    "ragged_gather_cost_estimate_bytes_accessed": -1,
    "ragged_gather_reduce_cost_estimate_bytes_accessed": -1,
    "wi_tile_fwd_batch_seq": 512,
    "wi_tile_fwd_embed_dim": 1024,
    "wi_tile_fwd_mlp_dim": 1024,
    "eval_wi_tile_fwd_batch_seq": 512,
    "eval_wi_tile_fwd_embed_dim": 1024,
    "eval_wi_tile_fwd_mlp_dim": 1024,
    "wi_tile_dlhs_batch_seq": 512,
    "wi_tile_dlhs_embed_dim": 1024,
    "wi_tile_dlhs_mlp_dim": 1024,
    "wi_tile_drhs_batch_seq": 512,
    "wi_tile_drhs_embed_dim": 1024,
    "wi_tile_drhs_mlp_dim": 1024,
    "wo_tile_fwd_batch_seq": 512,
    "wo_tile_fwd_embed_dim": 1024,
    "wo_tile_fwd_mlp_dim": 1024,
    "eval_wo_tile_fwd_batch_seq": 512,
    "eval_wo_tile_fwd_embed_dim": 1024,
    "eval_wo_tile_fwd_mlp_dim": 1024,
    "wo_tile_dlhs_batch_seq": 512,
    "wo_tile_dlhs_embed_dim": 1024,
    "wo_tile_dlhs_mlp_dim": 1024,
    "wo_tile_drhs_batch_seq": 512,
    "wo_tile_drhs_embed_dim": 1024,
    "wo_tile_drhs_mlp_dim": 1024,
}


def _parallelism_product(config: Any, *attrs: str) -> int:
  """Returns the product of positive (> 1) parallelism attributes on `config`."""
  size = 1
  for attr in attrs:
    val = int(getattr(config, attr, 1) or 1)
    if val > 1:
      size *= val
  return size


def _has_auto_parallelism(config: Any, *attrs: str) -> bool:
  """Returns whether any parallelism attribute in `attrs` uses a negative auto-fill sentinel (`-1`)."""
  for attr in attrs:
    raw = getattr(config, attr, 1)
    if raw is not None and int(raw) < 0:
      return True
  return False


def _active_mesh_or_none(mesh: Any = None) -> Any | None:
  """Returns `mesh` if provided and non-empty, else the active abstract or concrete JAX mesh."""
  if mesh is not None and not getattr(mesh, "empty", False):
    return mesh
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty:
    return abstract_mesh
  try:
    concrete_mesh = jax.sharding.get_mesh()
  except ValueError:
    concrete_mesh = None
  if concrete_mesh is not None and not getattr(concrete_mesh, "empty", False):
    return concrete_mesh
  return None


def _is_moe_dropless_fallback_set(val: Any) -> bool:
  """Returns whether `moe_dropless_fallback` is set to a non-default active mode."""
  if val is None:
    return False
  if isinstance(val, str):
    return val.strip().lower() not in ("", "none")
  return True


def unsupported_moe_settings(config: Any) -> dict[str, Any]:
  """Returns legacy MoE knobs set to non-default values on `config` that m3 does not implement."""
  unsupported: dict[str, Any] = {}
  for field in _UNSUPPORTED_BOOL_OFF_MOE_FIELDS:
    if bool(getattr(config, field, False)):
      unsupported[field] = getattr(config, field)
  for field in _UNSUPPORTED_BOOL_ON_MOE_FIELDS:
    val = getattr(config, field, True)
    if val is not None and not bool(val):
      unsupported[field] = val
  for field in _UNSUPPORTED_EMPTY_STR_MOE_FIELDS:
    val = getattr(config, field, "")
    if _is_moe_dropless_fallback_set(val):
      unsupported[field] = val
  for field, default_str in _UNSUPPORTED_ENUM_STR_MOE_FIELDS.items():
    val = getattr(config, field, default_str)
    if val is not None and (not isinstance(val, str) or val.strip().lower() != default_str):
      unsupported[field] = val
  for field, default_flt in _UNSUPPORTED_FLOAT_MOE_FIELDS.items():
    val = getattr(config, field, default_flt)
    if val is not None and (isinstance(val, bool) or float(val) != default_flt):
      unsupported[field] = val
  for field, default_int in _UNSUPPORTED_INT_MOE_FIELDS.items():
    val = getattr(config, field, default_int)
    if val is not None and (isinstance(val, bool) or val != default_int):
      unsupported[field] = val
  fallback = getattr(config, "moe_dropless_fallback", None)
  if _is_moe_dropless_fallback_set(fallback):
    unsupported["moe_dropless_fallback"] = fallback
  return unsupported


def _resolve_executor_kind(config: Any) -> ExecutorKind:
  """Resolves `MoEExecutor.kind` following legacy MoE backend flag precedence."""
  # Legacy precedence (`layers/moe.py:3098-3189`, `megablox/ops.py:320-346`, `configs/types.py:7956`):
  # `sparse_matmul=False` runs dense einsum; `use_gmm_v2` requires `use_tokamax_gmm`; `use_tokamax_gmm=True`
  # selects `"gmm_v2"` when `use_gmm_v2=True` else `"tokamax"` (even if `megablox=True`); `megablox=True` without
  # `use_tokamax_gmm` selects the legacy Megablox kernel, which m3 refuses rather than remapping to `gmm_v2`.
  sparse_matmul = bool(getattr(config, "sparse_matmul", True))
  use_tokamax_gmm = bool(getattr(config, "use_tokamax_gmm", False))
  use_gmm_v2 = bool(getattr(config, "use_gmm_v2", False))
  megablox = bool(getattr(config, "megablox", False))

  if use_gmm_v2 and not use_tokamax_gmm:
    raise ValueError("use_gmm_v2=True requires use_tokamax_gmm=True.")
  if not sparse_matmul:
    return "dense"
  if use_tokamax_gmm and use_gmm_v2:
    return "gmm_v2"
  if use_tokamax_gmm:
    return "tokamax"
  if megablox:
    raise ValueError(
        "m3 MoE does not implement the legacy Megablox kernel (megablox=True with use_tokamax_gmm=False). "
        "Set use_tokamax_gmm=True and use_gmm_v2=True for Tokamax GMM v2 (sharded EP/TP on TPU), "
        "use_tokamax_gmm=True and use_gmm_v2=False for Tokamax v1 ragged_dot, or "
        "megablox=False for jax.lax.ragged_dot (EP=1, TP=1)."
    )
  return "ragged_dot"


def _require_sharded_gmm_for_ep_or_tp(
    config: Any,
    *,
    mesh: Any = None,
    sharding: Sharding | None = None,
) -> None:
  """Rejects unsharded `ragged_dot` when EP or TP exceeds 1, uses `-1` auto-fill, or shards in the active mesh."""
  ep_cfg = _parallelism_product(config, *_EP_PARALLELISM_ATTRS)
  tp_cfg = _parallelism_product(config, *_TP_PARALLELISM_ATTRS)
  ep_auto = _has_auto_parallelism(config, *_EP_PARALLELISM_ATTRS)
  tp_auto = _has_auto_parallelism(config, *_TP_PARALLELISM_ATTRS)

  mesh_ep = 1
  mesh_tp = 1
  active_mesh = _active_mesh_or_none(mesh)
  if active_mesh is not None:
    shape = getattr(active_mesh, "shape", {})
    mesh_ep = max(mesh_ep, int(shape.get("expert", 1) or 1))
    mesh_tp = max(
        mesh_tp,
        int(shape.get("tensor", 1) or 1) * int(shape.get("tensor_sequence", 1) or 1),
    )
    if sharding is not None:
      wi_spec = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
      mesh_ep = max(mesh_ep, kernels.mesh_axis_size(active_mesh, wi_spec[0]))
      mesh_tp = max(mesh_tp, kernels.mesh_axis_size(active_mesh, wi_spec[2]))

  if ep_cfg > 1 or tp_cfg > 1 or ep_auto or tp_auto or mesh_ep > 1 or mesh_tp > 1:
    raise ValueError(
        "Expert parallelism (expert > 1 or -1 auto-fill) or tensor parallelism (tensor > 1 or -1 auto-fill) "
        "with sparse_matmul=True requires a sharded GMM backend (gmm_v2 or tokamax) so tokens dispatch via "
        "shard_map without all-gathering expert weights."
    )


def _validate_executor_backend_config(config: Any, *, mesh: Any = None) -> ExecutorKind:
  """Validates resolved MoE backend flags against `use_ragged_sort`, `moe_fp8`, `moe_ep_allgather_tc`, and EP/TP."""
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  bwd_wgrad_overlap = bool(getattr(config, "moe_bwd_wgrad_overlap", False))
  ep_allgather_tc = bool(getattr(config, "moe_ep_allgather_tc", False))
  moe_fp8 = bool(getattr(config, "moe_fp8", False))
  kind = _resolve_executor_kind(config)
  if (ep_allgather_tc or moe_fp8) and kind not in SHARDED_EXECUTOR_KINDS:
    raise ValueError(
        "moe_ep_allgather_tc / moe_fp8 requires sparse_matmul=True with a sharded GMM backend (gmm_v2 or tokamax)."
    )
  if use_ragged_sort and kind not in SHARDED_EXECUTOR_KINDS:
    raise ValueError("use_ragged_sort=True requires sparse_matmul=True with a sharded GMM backend (gmm_v2 or tokamax).")
  if bwd_wgrad_overlap and kind == "dense":
    raise ValueError("moe_bwd_wgrad_overlap=True requires sparse_matmul=True.")
  if kind == "ragged_dot":
    _require_sharded_gmm_for_ep_or_tp(config, mesh=mesh)
  return kind


def validate_moe_config(
    config: Any,
    *,
    check_executor_backend: bool = True,
    require_moe: bool = False,
    mesh: Any = None,
) -> None:
  """Validates m3 MoE configuration fields, failing closed on unsupported modes."""
  num_experts = int(getattr(config, "num_experts", 1) or 1)
  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  use_ring_of_experts = bool(getattr(config, "use_ring_of_experts", False))
  bwd_wgrad_overlap = bool(getattr(config, "moe_bwd_wgrad_overlap", False))
  ep_allgather_tc = bool(getattr(config, "moe_ep_allgather_tc", False))
  moe_fp8 = bool(getattr(config, "moe_fp8", False))
  ep_combine_mode = str(getattr(config, "moe_ep_combine_mode", "auto") or "auto")
  row_scale_placement = str(getattr(config, "moe_row_scale_placement", "tc") or "tc")
  tc_flat_blk = int(getattr(config, "moe_tc_ragged_flatten_block_size", 0) or 0)

  if ep_combine_mode not in ("auto", "tensorcore"):
    raise ValueError(f"moe_ep_combine_mode must be 'auto' or 'tensorcore'; got {ep_combine_mode!r}.")
  if row_scale_placement not in ("tc", "auto", "sc"):
    raise ValueError(f"moe_row_scale_placement must be 'tc', 'auto', or 'sc'; got {row_scale_placement!r}.")
  if tc_flat_blk < 0:
    raise ValueError(f"moe_tc_ragged_flatten_block_size must be >= 0; got {tc_flat_blk}.")

  if capacity_factor == 0.0:
    raise ValueError("capacity_factor cannot be 0.0; set < 0 for dropless routing or > 0 for capacity dropping.")
  if ragged_buffer_factor == 0.0:
    raise ValueError("ragged_buffer_factor cannot be 0.0; set < 0 for worst-case buffer or > 0 to bound buffer size.")

  unsupported = unsupported_moe_settings(config)
  if unsupported:
    raise ValueError(f"m3 MoE does not support {unsupported} (see m3/README.md).")

  if bool(getattr(config, "use_gmm_v2", False)) and not bool(getattr(config, "use_tokamax_gmm", False)):
    raise ValueError("use_gmm_v2=True requires use_tokamax_gmm=True.")

  if (
      not require_moe
      and num_experts <= 1
      and not bwd_wgrad_overlap
      and not use_ragged_sort
      and not use_ring_of_experts
      and not ep_allgather_tc
      and not moe_fp8
  ):
    return

  if use_ring_of_experts:
    raise NotImplementedError(
        "m3 MoE uses ragged_all_to_all expert parallelism; use_ring_of_experts=True is not supported."
    )
  if not check_executor_backend:
    unwired = {
        field: getattr(config, field)
        for field in (
            "use_tokamax_gmm",
            "use_gmm_v2",
            "use_ragged_sort",
            "moe_bwd_wgrad_overlap",
            "moe_ep_allgather_tc",
            "moe_fp8",
        )
        if bool(getattr(config, field, False))
    }
    if unwired:
      raise ValueError(
          f"Model {getattr(config, 'model_name', '<unknown>')!r} does not use MoEExecutor yet; "
          f"m3 cannot apply {unwired}."
      )
    return

  _validate_executor_backend_config(config, mesh=mesh)


def moe_from_config(config: Any, sharding: Sharding | None = None) -> MoEExecutor:
  """Resolves a static `MoEExecutor` from `config` and `sharding`."""
  active_mesh = _active_mesh_or_none()
  validate_moe_config(config, require_moe=True, mesh=active_mesh)
  sharding = sharding or sharding_from_config(config)

  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  bwd_wgrad_overlap = bool(getattr(config, "moe_bwd_wgrad_overlap", False))
  ep_allgather_tc = bool(getattr(config, "moe_ep_allgather_tc", False))
  moe_fp8 = bool(getattr(config, "moe_fp8", False))
  ep_combine_mode = str(getattr(config, "moe_ep_combine_mode", "auto") or "auto")
  row_scale_placement = str(getattr(config, "moe_row_scale_placement", "tc") or "tc")
  tc_3d_gmm = bool(getattr(config, "moe_tc_3d_gmm", False))
  fused_bwd_quant = bool(getattr(config, "moe_fused_bwd_quant", False))
  tc_ragged_flatten_block_size = max(0, int(getattr(config, "moe_tc_ragged_flatten_block_size", 0) or 0))

  cap = capacity_factor if capacity_factor > 0.0 else -1.0
  rbf = ragged_buffer_factor if ragged_buffer_factor > 0.0 else -1.0

  kind = _resolve_executor_kind(config)
  if kind == "dense":
    return MoEExecutor(
        kind="dense",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
    )

  if kind == "ragged_dot":
    _require_sharded_gmm_for_ep_or_tp(config, mesh=active_mesh, sharding=sharding)
    return MoEExecutor(
        kind="ragged_dot",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
        bwd_wgrad_overlap=bwd_wgrad_overlap,
    )

  tokens_spec = sharding("moe_tokens", ("batch", "length", "embed"), TensorType.ACTIVATION)
  wi_spec = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
  return MoEExecutor(
      kind=kind,
      batch=tokens_spec[0],
      experts=wi_spec[0],
      mlp=wi_spec[2],
      capacity_factor=cap,
      ragged_buffer_factor=rbf,
      use_ragged_sort=use_ragged_sort,
      length=tokens_spec[1],
      bwd_wgrad_overlap=bwd_wgrad_overlap,
      ep_allgather_tc=ep_allgather_tc,
      moe_fp8=moe_fp8,
      ep_combine_mode=ep_combine_mode,
      row_scale_placement=row_scale_placement,
      tc_3d_gmm=tc_3d_gmm,
      fused_bwd_quant=fused_bwd_quant,
      tc_ragged_flatten_block_size=tc_ragged_flatten_block_size,
  )


# ==============================================================================
# Load-balancing loss & dense combine helpers
# ==============================================================================


def combine_weights(weights: jax.Array, indices: jax.Array, num_experts: int) -> jax.Array:
  """Projects per-token top-k weights into a dense `(batch, length, num_experts)` table."""
  valid = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid, indices, 0)
  safe_weights = jnp.where(valid, weights, jnp.zeros_like(weights))
  return jnp.sum(jax.nn.one_hot(safe_indices, num_experts, dtype=weights.dtype) * safe_weights[..., None], axis=-2)


def load_balance_loss(
    indices: jax.Array,
    probs: jax.Array,
    *,
    weight: float = 1.0,
    mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Switch-Transformer load-balance loss masking padding and binarizing `segment_ids`."""
  if probs.ndim != 3 or indices.ndim != 3 or probs.shape[:2] != indices.shape[:2]:
    raise ValueError(
        f"Expected 3D indices (B, S, K) and probs (B, S, E) with matching (B, S); "
        f"got indices={indices.shape}, probs={probs.shape}."
    )
  batch, length, num_experts = probs.shape
  top_k = indices.shape[-1]
  probs_f32 = probs.astype(jnp.float32)

  valid_slots = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid_slots, indices, 0)
  expert_mask = jax.nn.one_hot(safe_indices, num_experts, dtype=jnp.float32) * valid_slots[..., None].astype(jnp.float32)

  token_valid: jax.Array | None = None
  if mask is not None:
    if mask.shape != (batch, length):
      raise ValueError(f"mask shape {mask.shape} must match (batch, length)=({batch}, {length}).")
    token_valid = mask != 0
  if segment_ids is not None:
    if segment_ids.shape != (batch, length):
      raise ValueError(f"segment_ids shape {segment_ids.shape} must match (batch, length)=({batch}, {length}).")
    seg_valid = segment_ids != 0
    token_valid = seg_valid if token_valid is None else (token_valid & seg_valid)

  if token_valid is None:
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / top_k
    density_prob = jnp.mean(probs_f32, axis=1)
    return (jnp.mean(density * density_prob) * (num_experts**2) * float(weight)).astype(jnp.float32)

  m = token_valid.astype(jnp.float32)
  seq_counts = jnp.sum(m, axis=1, keepdims=True)
  safe_counts = jnp.maximum(seq_counts, 1.0)
  seq_active = (seq_counts[:, 0] > 0).astype(jnp.float32)

  masked_expert_mask = expert_mask * m[:, :, None, None]
  density = jnp.sum(masked_expert_mask, axis=(1, 2)) / (safe_counts * top_k)
  density_prob = jnp.sum(probs_f32 * m[:, :, None], axis=1) / safe_counts
  per_seq = jnp.mean(density * density_prob, axis=-1)
  active_denom = jnp.maximum(jnp.sum(seq_active), 1.0)
  return ((jnp.sum(per_seq * seq_active) / active_denom) * (num_experts**2) * float(weight)).astype(jnp.float32)


# ==============================================================================
# Expert MLP & EP dispatch/combine execution
# ==============================================================================


def _apply_gated_activation(
    gate: jax.Array,
    up: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any,
) -> jax.Array:
  """Computes `activation(gate) * up` with optional symmetric/upper SwiGLU clipping."""
  if swiglu_limit is not None and swiglu_limit > 0:
    limit = jnp.asarray(swiglu_limit, dtype=gate.dtype)
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
  return (activation(gate) * up).astype(out_dtype)


def _truncate_slots_by_capacity(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    expert_cap: int,
    prior_expert_counts: jax.Array | None = None,
) -> jax.Array:
  """Drops slots whose per-expert token-major rank exceeds `expert_cap` in O(M) memory."""
  num_slots = safe_idx.shape[0]
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  counts = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
  starts = jnp.cumsum(counts) - counts
  sorted_exp = safe_idx[order]
  rank_in_exp = jnp.arange(num_slots, dtype=jnp.int32) - starts[sorted_exp]
  if prior_expert_counts is not None:
    rank_in_exp = rank_in_exp + prior_expert_counts[sorted_exp]
  keep_sorted = slot_valid[order] & (rank_in_exp < int(expert_cap))
  return keep_sorted[jnp.argsort(order)]


def _truncate_slots_by_buffer(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    buf_cap: int,
) -> jax.Array:
  """Keeps at most `buf_cap` valid slots in expert-sorted order."""
  num_slots = safe_idx.shape[0]
  if buf_cap >= num_slots:
    return slot_valid
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  keep_sorted = slot_valid[order] & (jnp.arange(num_slots, dtype=jnp.int32) < int(buf_cap))
  return keep_sorted[jnp.argsort(order)]


def _ep_ragged_exchange(
    operand: jax.Array,
    out_len: int,
    counts_matrix: jax.Array,
    ep_id: jax.Array,
    ep_axis: str,
    *,
    is_tpu: bool,
) -> jax.Array:
  """Exchanges variable-length token slices across `ep_axis`.

  `counts_matrix[src, dst]` gives the number of rows sent from `src` to `dst`.
  On TPU, dispatches directly to `jax.lax.ragged_all_to_all` using sender-side
  destination offsets. On CPU (where XLA's CPU thunk emitter lacks
  `ragged-all-to-all`), computes the bit-identical `[out_len, d_dim]` buffer via
  token-only `jax.lax.all_to_all`.
  """
  ep_size = counts_matrix.shape[0]
  d_dim = operand.shape[1]
  send_sz = counts_matrix[ep_id]
  in_off = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(send_sz)[:-1]], axis=0)
  out_off_all = jnp.cumsum(
      jnp.concatenate([jnp.zeros((1, ep_size), dtype=jnp.int32), counts_matrix], axis=0),
      axis=0,
  )[:-1]
  out_off = out_off_all[ep_id]
  recv_sz = counts_matrix[:, ep_id]

  if is_tpu:
    out_buf = jnp.zeros((out_len, d_dim), dtype=operand.dtype)
    return jax.lax.ragged_all_to_all(
        operand,
        out_buf,
        in_off,
        send_sz,
        out_off,
        recv_sz,
        axis_name=ep_axis,
    )

  in_len = operand.shape[0]
  cap = max(1, min(int(in_len), int(out_len)))
  col = jnp.arange(cap, dtype=jnp.int32)[None, :]
  gather_idx = in_off[:, None] + col
  valid_send = col < send_sz[:, None]
  safe_gather = jnp.clip(gather_idx, 0, max(int(in_len) - 1, 0))
  staged = jnp.where(valid_send[:, :, None], operand[safe_gather], jnp.zeros((), dtype=operand.dtype))
  recvd = jax.lax.all_to_all(staged, ep_axis, split_axis=0, concat_axis=0)

  recv_ends = jnp.cumsum(recv_sz)
  recv_starts = recv_ends - recv_sz
  row_idx = jnp.arange(out_len, dtype=jnp.int32)
  src_idx = _chunk_indices_from_ends(recv_ends, out_len)
  valid_row = row_idx < recv_ends[-1]
  safe_src = jnp.minimum(src_idx, ep_size - 1)
  pos_in_src = jnp.clip(row_idx - recv_starts[safe_src], 0, cap - 1)
  return jnp.where(valid_row[:, None], recvd[safe_src, pos_in_src], jnp.zeros((), dtype=operand.dtype))


def _dense_separate(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Runs every expert on every token and combines selected expert outputs."""
  if capacity_factor > 0.0:
    batch, length, _ = x.shape
    top_k = indices.shape[-1]
    num_tokens = batch * length
    flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)
    cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
    slot_valid = checkpoint_name(slot_valid.astype(jnp.int32), "moe_routing") != 0
    eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
    eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
  else:
    eff_indices = indices
    eff_weights = weights
  table = checkpoint_name(combine_weights(eff_weights, eff_indices, num_experts), "moe_routing")
  prec = _matmul_precision(x.dtype)
  gate = checkpoint_name(jnp.einsum("bse,neh->bsnh", x, wi_0, precision=prec), "moe_gate_up")
  up = checkpoint_name(jnp.einsum("bse,neh->bsnh", x, wi_1, precision=prec), "moe_gate_up")
  hidden = _apply_gated_activation(gate, up, activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype)
  weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(wo.dtype)
  out = jnp.einsum("bsnh,nhe->bse", weighted_hidden, wo, precision=prec, preferred_element_type=_COMBINE_DTYPE)
  return checkpoint_name(out, "moe_routed_output")


def _dense_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Dense reference execution for 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim == 4:
    if capacity_factor > 0.0:
      batch, length, _ = x.shape
      top_k = indices.shape[-1]
      num_tokens = batch * length
      flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
      slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
      safe_idx = jnp.where(slot_valid, flat_idx, 0)
      cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
      slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
      slot_valid = checkpoint_name(slot_valid.astype(jnp.int32), "moe_routing") != 0
      eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
      eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
    else:
      eff_indices = indices
      eff_weights = weights
    table = checkpoint_name(combine_weights(eff_weights, eff_indices, num_experts), "moe_routing")
    prec = _matmul_precision(x.dtype)
    gu = checkpoint_name(jnp.einsum("bse,negh->bsngh", x, gate_up_proj, precision=prec), "moe_gate_up")
    hidden = _apply_gated_activation(
        gu[..., 0, :], gu[..., 1, :], activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype
    )
    weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(down_proj.dtype)
    out = jnp.einsum("bsnh,nhe->bse", weighted_hidden, down_proj, precision=prec, preferred_element_type=_COMBINE_DTYPE)
    return checkpoint_name(out, "moe_routed_output")
  wi_0, wi_1 = jnp.split(gate_up_proj, 2, axis=-1)
  return _dense_separate(
      x,
      weights,
      indices,
      wi_0,
      wi_1,
      down_proj,
      num_experts=num_experts,
      activation=activation,
      swiglu_limit=swiglu_limit,
      capacity_factor=capacity_factor,
  )


def _run_local_expert_mlp(
    sorted_x: jax.Array,
    projections: tuple[jax.Array, ...],
    group_sizes: jax.Array,
    *,
    fused: bool,
    matmul_fn: Callable[..., jax.Array],
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any = None,
) -> jax.Array:
  """Runs gate/up + activation + down grouped matmuls on sorted rows."""
  if fused:
    w_gu, w_dn = projections
    gate_up = checkpoint_name(matmul_fn(sorted_x, w_gu, group_sizes), "moe_gate_up")
    gate, up = jnp.split(gate_up, 2, axis=-1)
    hidden = _apply_gated_activation(
        gate,
        up,
        activation=activation,
        swiglu_limit=swiglu_limit,
        out_dtype=sorted_x.dtype,
    )
    return matmul_fn(hidden, w_dn, group_sizes, preferred_element_type=out_dtype)

  wi_0, wi_1, wo = projections
  gate = checkpoint_name(matmul_fn(sorted_x, wi_0, group_sizes), "moe_gate_up")
  up = checkpoint_name(matmul_fn(sorted_x, wi_1, group_sizes), "moe_gate_up")
  hidden = _apply_gated_activation(
      gate,
      up,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=sorted_x.dtype,
  )
  return matmul_fn(hidden, wo, group_sizes, preferred_element_type=out_dtype)


def _grouped_unsharded(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
) -> jax.Array:
  """Portable `ragged_dot` grouped execution without `shard_map`."""
  if executor.bwd_wgrad_overlap:
    x, projections = _backward_wgrad_tie(x, projections)
  if fused and projections[0].ndim == 4:
    w_gu_4d, w_dn = projections
    projections = (
        jnp.reshape(w_gu_4d, (w_gu_4d.shape[0], w_gu_4d.shape[1], 2 * w_gu_4d.shape[3])),
        w_dn,
    )

  batch, length, embed = x.shape
  top_k = indices.shape[-1]
  num_tokens = batch * length
  num_slots = num_tokens * top_k

  flat_idx = jnp.reshape(indices, (num_slots,)).astype(jnp.int32)
  flat_w = jnp.reshape(weights, (num_slots,))
  slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
  safe_idx = jnp.where(slot_valid, flat_idx, 0)

  if executor.capacity_factor > 0.0:
    cap = max(1, int(math.ceil((num_slots / num_experts) * executor.capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
  if executor.ragged_buffer_factor > 0.0:
    buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
    slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)
  if executor.capacity_factor > 0.0 or executor.ragged_buffer_factor > 0.0:
    slot_valid = checkpoint_name(slot_valid.astype(jnp.int32), "moe_routing") != 0

  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  perm = checkpoint_name(jnp.argsort(sort_key, stable=True), "moe_routing")
  inv_perm = checkpoint_name(jnp.argsort(perm), "moe_routing")
  group_sizes = checkpoint_name(
      jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32),
      "moe_routing",
  )
  valid_count = jnp.sum(group_sizes)

  slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
  slot_rows = jnp.reshape(x, (num_tokens, embed))[slot_token_ids]
  sorted_x = _permute_rows(slot_rows, perm, inv_perm)
  sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)
  sorted_x = _tag_disp_ep(sorted_x, executor.disp_ep_tags)

  matmul_fn = functools.partial(grouped_matmul, backend="ragged_dot", is_tpu=False)
  expert_out = _run_local_expert_mlp(
      sorted_x,
      projections,
      group_sizes,
      fused=fused,
      matmul_fn=matmul_fn,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=_COMBINE_DTYPE,
  )
  unsorted_out = checkpoint_name(_permute_rows(expert_out, inv_perm, perm), "moe_routed_output")
  eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
  safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
  weighted = safe_out * eff_w[:, None]
  combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, embed)), axis=1)
  return jnp.reshape(combined, (batch, length, embed))


def _can_use_sparsecore_ragged_sort(num_rows: int, hidden_size: int, dtype: Any) -> bool:
  """Returns True when SparseCore `ragged_sort` kernels support `(num_rows, hidden_size, dtype)` on the active TPU."""
  dt = jnp.dtype(dtype)
  if dt not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
    return False
  if num_rows <= 0 or hidden_size <= 0:
    return False
  if (num_rows * jax.dtypes.itemsize_bits(dt)) % 32 != 0:
    return False
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=import-outside-toplevel

  if not pltpu.is_tpu_device():
    return False
  try:
    tpu_info = pltpu.get_tpu_info()
  except ValueError:
    return False
  sc_info = tpu_info.sparse_core
  if sc_info is None:
    return False
  num_lanes = int(tpu_info.num_lanes)
  if hidden_size % num_lanes != 0:
    return False
  num_cores = int(sc_info.num_cores) * int(sc_info.num_subcores)
  num_simd_lanes = int(sc_info.num_lanes)
  preferred_num_stages = 4
  num_col_parts = 1
  while num_cores // num_col_parts > num_simd_lanes:
    next_col_parts = num_col_parts * 2
    if num_cores % next_col_parts != 0:
      break
    if hidden_size % (num_lanes * next_col_parts) != 0:
      break
    if hidden_size // (num_lanes * next_col_parts) < preferred_num_stages:
      break
    num_col_parts = next_col_parts
  return (num_cores // num_col_parts) <= num_simd_lanes


def _sharded_grouped(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
) -> jax.Array:
  """Distributed `shard_map` execution supporting `EP >= 1`, `TP >= 1`, and `FSDP >= 1`."""
  mesh = kernels.mesh_in_scope()
  is_tpu = kernels.is_tpu_mesh(mesh)

  batch_ax = kernels.resolve_mesh_axes(mesh, executor.batch)
  experts_ax = kernels.resolve_mesh_axes(mesh, executor.experts)
  mlp_ax = kernels.resolve_mesh_axes(mesh, executor.mlp)

  batch_shards = kernels.mesh_axis_size(mesh, batch_ax)
  ep_size = kernels.mesh_axis_size(mesh, experts_ax)
  tp_size = kernels.mesh_axis_size(mesh, mlp_ax)
  ep_axes = kernels.mesh_axis_names(experts_ax)
  tp_axes = kernels.mesh_axis_names(mlp_ax)

  if len(ep_axes) > 1:
    raise NotImplementedError(f"Expert sharding supports at most one mesh axis; got {ep_axes}.")
  if x.shape[0] % batch_shards != 0:
    raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {batch_ax!r}.")
  if num_experts % ep_size != 0:
    raise ValueError(
        f"num_experts ({num_experts}) must be divisible by expert parallelism size ({ep_size}) on {experts_ax!r}."
    )

  if fused:
    w_gu_global, w_dn_global = projections
    is_4d_fused = w_gu_global.ndim == 4
    if is_4d_fused:
      mlp_dim = w_gu_global.shape[3]
      if w_dn_global.shape[1] != mlp_dim:
        raise ValueError(f"down_proj MLP dim ({w_dn_global.shape[1]}) must match gate_up MLP dim ({mlp_dim}).")
      axis_types = tuple(getattr(mesh, "axis_types", ()))
      is_explicit = bool(axis_types) and all(t == jax.sharding.AxisType.Explicit for t in axis_types)
      cur_spec = tuple(getattr(jax.typeof(w_gu_global).sharding, "spec", ())) if is_explicit else ()
      dim2_shards = kernels.mesh_axis_size(mesh, cur_spec[2]) if len(cur_spec) > 2 else 1
      if tp_size == 1 and dim2_shards == 1:
        if is_explicit:
          ax0 = cur_spec[0] if len(cur_spec) > 0 else None
          ax1 = cur_spec[1] if len(cur_spec) > 1 else None
          w_gu_unsharded_tp = kernels.ensure_explicit_spec(w_gu_global, P(ax0, ax1, None, None), mesh)
          assert w_gu_unsharded_tp is not None
          w_gu_global = jax.lax.reshape(
              w_gu_unsharded_tp,
              (num_experts, w_gu_global.shape[1], 2 * mlp_dim),
              out_sharding=P(ax0, ax1, None),
          )
        else:
          w_gu_global = jnp.reshape(w_gu_global, (num_experts, w_gu_global.shape[1], 2 * mlp_dim))
        projections = (w_gu_global, w_dn_global)
        is_4d_fused = False
        proj_specs: tuple[P, ...] = (P(experts_ax, None, None), P(experts_ax, mlp_ax, None))
      else:
        proj_specs = (P(experts_ax, None, None, mlp_ax), P(experts_ax, mlp_ax, None))
    else:
      if tp_size > 1:
        raise ValueError(
            "3D fused gate_up_proj cannot be tensor-parallel sharded across tp_size > 1 without replicating expert weights; "
            "store gate_up_proj as 4D (num_experts, emb_dim, 2, moe_mlp_dim)."
        )
      mlp_dim = w_gu_global.shape[2] // 2
      proj_specs = (P(experts_ax, None, None), P(experts_ax, mlp_ax, None))
      if w_dn_global.shape[1] != mlp_dim:
        raise ValueError(f"down_proj MLP dim ({w_dn_global.shape[1]}) must match gate_up MLP dim ({mlp_dim}).")
  else:
    wi_0_global, wi_1_global, wo_global = projections
    is_4d_fused = False
    mlp_dim = wi_0_global.shape[2]
    if wi_1_global.shape != wi_0_global.shape or wo_global.shape[1] != mlp_dim:
      raise ValueError(
          f"Mismatched expert weight shapes: wi_0={wi_0_global.shape}, wi_1={wi_1_global.shape}, wo={wo_global.shape}."
      )
    proj_specs = (P(experts_ax, None, mlp_ax), P(experts_ax, None, mlp_ax), P(experts_ax, mlp_ax, None))

  if mlp_dim % tp_size != 0:
    raise ValueError(f"Expert MLP dim ({mlp_dim}) must be divisible by tensor parallelism size ({tp_size}).")

  raw_length_ax = kernels.resolve_mesh_axes(mesh, executor.length)
  # Tokens stay on the sequence shard that holds them under EP=CP; exclude only batch_ax (already on dim 0).
  claimed_axes = set(kernels.mesh_axis_names(batch_ax))
  cand_length_axes = [
      ax for ax in kernels.mesh_axis_names(raw_length_ax) if ax not in claimed_axes and int(mesh.shape[ax]) > 1
  ]
  cand_length_ax: MeshAxes = (
      cand_length_axes[0] if len(cand_length_axes) == 1 else (tuple(cand_length_axes) if cand_length_axes else None)
  )
  length_shards = kernels.mesh_axis_size(mesh, cand_length_ax)
  length_ax = cand_length_ax if (length_shards > 1 and x.shape[1] > 1 and x.shape[1] % length_shards == 0) else None
  tokens_spec = P(batch_ax, length_ax, None)

  d_embed = x.shape[2]
  e_loc = num_experts // ep_size
  m_tp = mlp_dim // tp_size
  if executor.moe_fp8:
    if tp_size > 1:
      raise NotImplementedError("moe_fp8 currently requires expert tensor parallelism == 1.")
    use_fp8_tc_gmm = is_tpu and (d_embed % 128 == 0) and (m_tp % 128 == 0)

    def _fp8_matmul_fn(lhs: Any, rhs: Any, gs: jax.Array, *, preferred_element_type: Any = None) -> jax.Array:
      out = _gmm_fp8_or_ref(
          lhs,
          rhs,
          gs,
          x.dtype,
          moe_fp8=True,
          is_tpu=use_fp8_tc_gmm,
          out_is_3d=False,
          use_fused_bwd_quant=executor.fused_bwd_quant,
      )
      return out if preferred_element_type is None else out.astype(preferred_element_type)

    matmul_fn: Callable[..., jax.Array] = _fp8_matmul_fn
  else:
    matmul_fn = functools.partial(grouped_matmul, backend=executor.kind, is_tpu=is_tpu)

  def _slice_local_projections(proj_local: tuple[jax.Array, ...]) -> tuple[Any, ...]:
    if not fused:
      wi_0_loc, wi_1_loc, wo_loc = proj_local
      if executor.ep_allgather_tc or executor.moe_fp8:
        out_projs: tuple[Any, ...] = (jnp.concatenate([wi_0_loc, wi_1_loc], axis=-1), wo_loc)
      else:
        out_projs = proj_local
    else:
      w_gu, w_dn = proj_local
      if is_4d_fused:
        out_projs = (jnp.reshape(w_gu, (e_loc, w_gu.shape[1], 2 * m_tp)), w_dn)
      else:
        out_projs = (w_gu, w_dn)
    if executor.moe_fp8:
      return tuple(_ste_quantize_routed_weight(p) for p in out_projs)
    return out_projs

  def per_device(x_sh: jax.Array, weights_sh: jax.Array, indices_sh: jax.Array, *proj_sh: jax.Array) -> jax.Array:
    proj_tuple: tuple[jax.Array, ...] = tuple(proj_sh)
    if executor.bwd_wgrad_overlap:
      x_sh, proj_tuple = _backward_wgrad_tie(x_sh, proj_tuple)
    local_projs = _slice_local_projections(proj_tuple)

    b_loc, seq_len, d_dim = x_sh.shape
    top_k = indices_sh.shape[-1]
    num_tokens = b_loc * seq_len
    num_slots = num_tokens * top_k

    flat_x = jnp.reshape(x_sh, (num_tokens, d_dim))
    flat_idx = jnp.reshape(indices_sh, (num_slots,)).astype(jnp.int32)
    flat_w = jnp.reshape(weights_sh, (num_slots,))
    use_slot_major = (top_k > 1) and (num_tokens % 8 == 0)

    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)

    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      prior_exp = None
      if ep_size > 1:
        local_counts_pre = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(
            jnp.int32
        )
        all_counts_pre = jax.lax.all_gather(local_counts_pre, ep_axes[0], axis=0, tiled=False)
        ep_id_pre = jax.lax.axis_index(ep_axes[0])
        prior_exp = (jnp.cumsum(all_counts_pre, axis=0) - all_counts_pre)[ep_id_pre]
      slot_valid = _truncate_slots_by_capacity(
          safe_idx,
          slot_valid,
          num_experts=num_experts,
          expert_cap=expert_cap,
          prior_expert_counts=prior_exp,
      )

    if executor.ep_allgather_tc:
      if tp_size > 1:
        raise NotImplementedError("moe_ep_allgather_tc / moe_fp8 currently requires expert tensor parallelism == 1.")
      w_gu_use, w_dn_use = local_projs
      use_tc_kernels = is_tpu and (d_dim % 128 == 0) and (m_tp % 128 == 0)
      total_ag_tokens = ep_size * num_tokens
      total_ag_slots = total_ag_tokens * top_k
      cap = total_ag_slots
      if executor.capacity_factor > 0.0:
        expert_cap_tc = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
        cap = min(cap, e_loc * expert_cap_tc)
      if executor.ragged_buffer_factor > 0.0:
        rbf_cap_tc = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
        cap = min(cap, rbf_cap_tc)
      if use_tc_kernels and cap < total_ag_slots:
        cap = min(total_ag_slots, _align_up(cap, _TOKAMAX_M_ALIGN))

      if use_tc_kernels and executor.tc_3d_gmm:
        from maxtext.kernels.megablox import pallas_fused_quant  # pylint: disable=import-outside-toplevel

        use_3d_gmm = pallas_fused_quant.can_use_3d_gmm_tpu(
            num_rows=cap,
            num_local_experts=e_loc,
            d_dim=d_dim,
            m_tp=m_tp,
            moe_fp8=executor.moe_fp8,
            act_dtype=x_sh.dtype,
        )
        kernels.log_kernel_dispatch(
            f"moe_tc_3d_gmm_{use_3d_gmm}_{cap}_{e_loc}_{d_dim}_{m_tp}",
            f"m3.moe: moe_tc_3d_gmm={use_3d_gmm} (cap={cap}, e_loc={e_loc}, d={d_dim}, m_tp={m_tp})",
        )
      else:
        gmm_3d_align = 4096 if executor.moe_fp8 else 2048
        use_3d_gmm = use_tc_kernels and (d_dim % gmm_3d_align == 0)
      x_tok = jnp.reshape(x_sh, (num_tokens, d_dim // 128, 128) if use_3d_gmm else (num_tokens, d_dim))

      dummy_idx = jnp.arange(num_slots, dtype=jnp.int32) % num_experts
      route_idx_loc = jnp.reshape(jnp.where(slot_valid, safe_idx, dummy_idx), (num_tokens, top_k))
      route_w_loc = jnp.reshape(jnp.where(slot_valid, flat_w, jnp.zeros_like(flat_w)), (num_tokens, top_k))
      tc_combine_bwd = is_tpu and executor.ep_combine_mode == "tensorcore"

      if executor.moe_fp8:
        qa_loc = _ste_quantize_tokens(x_tok.astype(jnp.float32), jnp.float8_e4m3fn, (0,), "absmax")
        scale_loc_1d = jax.lax.stop_gradient(jnp.reshape(qa_loc.scale, (num_tokens,)))
        if ep_size > 1:
          ep_axis_tc = ep_axes[0]
          packed_meta_1d = jnp.concatenate(
              [jax.lax.bitcast_convert_type(scale_loc_1d, jnp.int32), jnp.ravel(route_idx_loc)],
              axis=0,
          )
          packed_meta_ag = jax.lax.stop_gradient(
              jnp.reshape(
                  _ep_all_gather_aligned128(packed_meta_1d, ep_axis_tc, is_tpu=is_tpu),
                  (ep_size, num_tokens + num_slots),
              )
          )
          route_w_ag = _ep_all_gather_weights_aligned128(route_w_loc, ep_axis_tc, is_tpu)
          qval_ag = _ep_gather_tokens(
              qa_loc.qvalue,
              ep_axis_tc,
              x_sh.dtype,
              tc_combine_bwd,
              after=(jax.lax.bitcast_convert_type(packed_meta_ag, jnp.float32), route_w_ag),
          )
          scale_ag_1d: jax.Array | None = jax.lax.bitcast_convert_type(
              jnp.reshape(packed_meta_ag[:, :num_tokens], (ep_size * num_tokens,)),
              jnp.float32,
          )
          route_idx_ag = jnp.reshape(packed_meta_ag[:, num_tokens:], (ep_size * num_tokens, top_k))
        else:
          ep_axis_tc = ""
          qval_ag = qa_loc.qvalue
          scale_ag_1d = scale_loc_1d
          route_idx_ag = route_idx_loc
          route_w_ag = route_w_loc
      else:
        scale_ag_1d = None
        if ep_size > 1:
          ep_axis_tc = ep_axes[0]
          route_idx_ag = jax.lax.stop_gradient(_ep_all_gather_aligned128(route_idx_loc, ep_axis_tc, is_tpu=is_tpu))
          route_w_ag = _ep_all_gather_weights_aligned128(route_w_loc, ep_axis_tc, is_tpu)
          qval_ag = _ep_gather_tokens(
              x_tok,
              ep_axis_tc,
              x_sh.dtype,
              tc_combine_bwd,
              after=(jax.lax.bitcast_convert_type(route_idx_ag, jnp.float32), route_w_ag),
          )
        else:
          ep_axis_tc = ""
          qval_ag = x_tok
          route_idx_ag = route_idx_loc
          route_w_ag = route_w_loc

      raw_flat_blk = 0 if use_3d_gmm else max(0, int(executor.tc_ragged_flatten_block_size))
      eff_flat_blk = math.gcd(raw_flat_blk, math.gcd(total_ag_tokens, cap)) if raw_flat_blk > 0 else 0
      if eff_flat_blk < 128 or (eff_flat_blk * d_dim * 2 * 4 > 56 * 1024 * 1024):
        eff_flat_blk = 0
      if raw_flat_blk > 0 and use_tc_kernels:
        kernels.log_kernel_dispatch(
            f"moe_tc_flat_blk_{raw_flat_blk}_{eff_flat_blk}_{total_ag_tokens}_{cap}",
            f"m3.moe: ring_ragged_sort_tc flatten_block_size={eff_flat_blk} "
            f"(requested={raw_flat_blk}, total_ag_tokens={total_ag_tokens}, cap={cap})",
        )

      shard_idx = jax.lax.axis_index(ep_axis_tc) if ep_size > 1 else jnp.int32(0)

      def _tag_routing(t: jax.Array) -> jax.Array:
        return checkpoint_name(t, "moe_routing")

      if use_tc_kernels:
        from maxtext.kernels.ragged import ragged_sort_tc  # pylint: disable=import-outside-toplevel

        sorted_qval, global_gs, _, tc_routing = ragged_sort_tc.ring_ragged_sort_tc(
            qval_ag,
            route_idx_ag,
            num_experts,
            top_k,
            ep_axis_tc,
            ep_size,
            cap,
            gather_block_size=1024,
            reduce_block_size=896,
            mask_padding=True,
            flatten_block_size=eff_flat_blk,
            keep_3d=use_3d_gmm,
            window_routing=True,
            tag_routing_fn=_tag_routing,
        )
        raw_local_gs = jax.lax.dynamic_slice_in_dim(global_gs, shard_idx * e_loc, e_loc, axis=0)
        local_gs = _tag_routing(_clamp_group_sizes_to_cap(raw_local_gs, cap))
        pin_tc_scales = use_tc_kernels and executor.row_scale_placement == "tc"
        pin_sc_scales = use_tc_kernels and executor.row_scale_placement == "sc"

        def _wrap_row_scale(fn: Any) -> Any:
          if pin_sc_scales:
            return on_any_sparse_core(fn, True)
          return on_tensor_core(fn, pin_tc_scales)

        w_rows = _tag_routing(
            _wrap_row_scale(lambda rw: ragged_sort_tc.tc_buffer_row_weights(tc_routing, jnp.ravel(rw), cap))(route_w_ag)
        )
        if scale_ag_1d is not None:
          sorted_scale_1d = _wrap_row_scale(
              lambda s_1d, tids, cnt: _window_token_scale(
                  s_1d,
                  tids,
                  cnt,
                  cap,
                  start=None if tc_routing.windowed else tc_routing.start,
              )
          )(scale_ag_1d, tc_routing.token_ids_sorted, tc_routing.count)
          scale_shape = (cap,) + (1,) * (sorted_qval.ndim - 1)
          sorted_lhs: jax.Array | qpl.QArray = qpl.QArray(
              qvalue=_tag_disp_ep(sorted_qval, executor.disp_ep_tags),
              scale=_tag_routing(jax.lax.stop_gradient(jnp.reshape(sorted_scale_1d, scale_shape))),
          )
        else:
          sorted_lhs = _tag_disp_ep(sorted_qval, executor.disp_ep_tags)
        token_ids_win = jnp.zeros((1,), dtype=jnp.int32)
        valid_win = jnp.zeros((1,), dtype=jnp.bool_)
      else:
        flat_idx_ag = jnp.ravel(route_idx_ag).astype(jnp.int32)
        order_ag = _tag_routing(jnp.argsort(flat_idx_ag, stable=True).astype(jnp.int32))
        global_gs = _tag_routing(jnp.bincount(flat_idx_ag, length=num_experts).astype(jnp.int32))
        offsets_ag = jnp.cumulative_sum(global_gs, include_initial=True)
        win_start = offsets_ag[shard_idx * e_loc].astype(jnp.int32)
        win_end = offsets_ag[(shard_idx + 1) * e_loc].astype(jnp.int32)
        win_count = jnp.minimum(win_end - win_start, cap).astype(jnp.int32)
        slot_win = jax.lax.dynamic_slice_in_dim(jnp.pad(order_ag, (0, cap)), win_start, cap, axis=0)
        token_ids_win = _tag_routing(jax.lax.stop_gradient(slot_win // top_k))
        valid_win = jnp.arange(cap, dtype=jnp.int32) < win_count
        raw_local_gs = jax.lax.dynamic_slice_in_dim(global_gs, shard_idx * e_loc, e_loc, axis=0)
        local_gs = _tag_routing(_clamp_group_sizes_to_cap(raw_local_gs, cap))
        sorted_qval = _window_gather_rows(qval_ag, token_ids_win, valid_win)
        w_flat_ag = jnp.ravel(route_w_ag).astype(jnp.float32)
        safe_slot_win = jnp.where(valid_win, slot_win, total_ag_slots + jnp.arange(cap, dtype=jnp.int32))
        w_rows = _tag_routing(w_flat_ag.at[safe_slot_win].get(mode="fill", fill_value=0.0, unique_indices=True))
        if scale_ag_1d is not None:
          sorted_scale_1d = _window_token_scale(scale_ag_1d, token_ids_win, win_count, cap, start=None)
          sorted_lhs = qpl.QArray(
              qvalue=_tag_disp_ep(sorted_qval, executor.disp_ep_tags),
              scale=_tag_routing(jax.lax.stop_gradient(jnp.reshape(sorted_scale_1d, (cap, 1)))),
          )
        else:
          sorted_lhs = _tag_disp_ep(sorted_qval, executor.disp_ep_tags)
        tc_routing = None

      gate_up_out = checkpoint_name(
          _gmm_fp8_or_ref(
              sorted_lhs,
              w_gu_use,
              local_gs,
              x_sh.dtype,
              moe_fp8=executor.moe_fp8,
              is_tpu=use_tc_kernels,
              out_is_3d=False,
              use_fused_bwd_quant=executor.fused_bwd_quant,
          ),
          "moe_gate_up",
      )
      gate_part, up_part = jnp.split(gate_up_out, 2, axis=-1)
      act_hidden = _apply_gated_activation(
          gate_part,
          up_part,
          activation=activation,
          swiglu_limit=swiglu_limit,
          out_dtype=x_sh.dtype,
      )
      act_prescaled = (act_hidden.astype(jnp.float32) * w_rows[:, None]).astype(x_sh.dtype)
      expert_out_tc = _gmm_fp8_or_ref(
          act_prescaled,
          w_dn_use,
          local_gs,
          x_sh.dtype,
          moe_fp8=executor.moe_fp8,
          is_tpu=use_tc_kernels,
          out_is_3d=use_3d_gmm,
          use_fused_bwd_quant=executor.fused_bwd_quant,
      )

      if use_tc_kernels:
        from maxtext.kernels.ragged import ragged_sort_tc  # pylint: disable=import-outside-toplevel

        combined_ag = ragged_sort_tc.ring_ragged_unsort_tc(
            expert_out_tc,
            tc_routing,
            top_k,
            jnp.ravel(route_w_ag).astype(jnp.float32),
            gather_block_size=1024,
            mask_padding=True,
            flatten_block_size=eff_flat_blk,
            prescaled=True,
            out_3d=use_3d_gmm,
        ).astype(x_sh.dtype)
      else:
        safe_tok_out = jnp.where(valid_win, token_ids_win, 0)
        valid_out = jnp.where(valid_win[:, None], expert_out_tc.astype(_COMBINE_DTYPE), 0.0)
        combined_ag = (
            jnp.zeros((total_ag_tokens, d_dim), dtype=_COMBINE_DTYPE).at[safe_tok_out].add(valid_out).astype(x_sh.dtype)
        )

      if ep_size > 1:
        tc_combine = is_tpu and executor.ep_combine_mode == "tensorcore"
        combined_loc = _ep_combine_tokens(combined_ag, ep_axis_tc, executor.moe_fp8, is_tpu, tc_combine)
      else:
        combined_loc = combined_ag
      return jnp.reshape(combined_loc, (b_loc, seq_len, d_dim))

    if ep_size == 1:
      if executor.ragged_buffer_factor > 0.0:
        buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
        slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)
      if executor.capacity_factor > 0.0 or executor.ragged_buffer_factor > 0.0:
        slot_valid = checkpoint_name(slot_valid.astype(jnp.int32), "moe_routing") != 0

      sort_key = jnp.where(slot_valid, safe_idx, num_experts)
      perm = checkpoint_name(jnp.argsort(sort_key, stable=True), "moe_routing")
      raw_inv_perm = jnp.argsort(perm)
      group_sizes = checkpoint_name(
          jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32),
          "moe_routing",
      )
      valid_count = jnp.sum(group_sizes)
      use_sc_rsort = (
          executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(num_slots, d_dim, flat_x.dtype)
      )
      inv_perm = raw_inv_perm
      inv_perm_bwd = raw_inv_perm

      if use_sc_rsort:
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        inv_perm = checkpoint_name(raw_inv_perm, "moe_routing")
        slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
        slot_rows = flat_x[slot_token_ids]
        sorted_x = ragged_sort.a2a_ragged_sort(slot_rows, perm, valid_count)
      else:
        inv_perm_bwd = (
            jnp.reshape(jnp.swapaxes(jnp.reshape(raw_inv_perm, (num_tokens, top_k)), 0, 1), (num_slots,))
            if use_slot_major
            else raw_inv_perm
        )
        inv_perm_bwd = checkpoint_name(inv_perm_bwd, "moe_routing")
        sorted_x = _gather_token_slots(flat_x, perm, inv_perm_bwd, top_k, use_slot_major)
        sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)
      sorted_x = _tag_disp_ep(sorted_x, executor.disp_ep_tags)

      expert_out = _run_local_expert_mlp(
          sorted_x,
          local_projs,
          group_sizes,
          fused=fused or executor.moe_fp8,
          matmul_fn=matmul_fn,
          activation=activation,
          swiglu_limit=swiglu_limit,
          out_dtype=_COMBINE_DTYPE,
      )

      if use_sc_rsort and _can_use_sparsecore_ragged_sort(num_slots, d_dim, expert_out.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        unsorted_out = checkpoint_name(
            ragged_sort.a2a_ragged_unsort(expert_out, inv_perm, num_slots),
            "moe_routed_output",
        )
        eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
        safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
        weighted = safe_out * eff_w[:, None]
        combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
      elif use_sc_rsort:
        unsorted_out = checkpoint_name(_permute_rows(expert_out, inv_perm, perm), "moe_routed_output")
        eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
        safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
        weighted = safe_out * eff_w[:, None]
        combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
      elif use_slot_major:
        perm_kt = (perm % top_k) * num_tokens + (perm // top_k)
        unsorted_kt = checkpoint_name(_permute_rows(expert_out, inv_perm_bwd, perm_kt), "moe_routed_output")
        valid_kt = jnp.reshape(jnp.swapaxes(jnp.reshape(slot_valid, (num_tokens, top_k)), 0, 1), (num_slots,))
        w_kt = jnp.reshape(jnp.swapaxes(jnp.reshape(flat_w, (num_tokens, top_k)), 0, 1), (num_slots,))
        eff_w_kt = jnp.where(valid_kt, w_kt, 0).astype(_COMBINE_DTYPE)
        safe_kt = jnp.where(valid_kt[:, None], unsorted_kt.astype(_COMBINE_DTYPE), 0.0)
        weighted_kt = safe_kt * eff_w_kt[:, None]
        combined = jnp.sum(jnp.reshape(weighted_kt, (top_k, num_tokens, d_dim)), axis=0)
      else:
        unsorted_out = checkpoint_name(_permute_rows(expert_out, inv_perm_bwd, perm), "moe_routed_output")
        eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
        safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
        weighted = safe_out * eff_w[:, None]
        combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
      if tp_axes:
        combined = jax.lax.psum(combined, tp_axes)
      return jnp.reshape(combined, (b_loc, seq_len, d_dim))

    # Multi-device Expert Parallelism (EP > 1) via ragged_all_to_all.
    ep_axis = ep_axes[0]
    ep_id = jax.lax.axis_index(ep_axis)

    recv_buf_len = ep_size * num_slots
    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      recv_buf_len = min(recv_buf_len, e_loc * expert_cap)
    if executor.ragged_buffer_factor > 0.0:
      rbf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
      recv_buf_len = min(recv_buf_len, rbf_cap)
    if executor.moe_fp8 and is_tpu and recv_buf_len < ep_size * num_slots:
      recv_buf_len = min(ep_size * num_slots, _align_up(recv_buf_len, _TOKAMAX_M_ALIGN))

    local_exp_counts = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
    all_exp_counts = jax.lax.all_gather(local_exp_counts, ep_axis, axis=0, tiled=False)

    ep_on_tokens = ep_axis in (kernels.mesh_axis_names(batch_ax) + kernels.mesh_axis_names(length_ax))
    need_recv_clamp = (recv_buf_len < ep_size * num_slots) and not (
        ep_on_tokens and executor.ragged_buffer_factor <= 0.0 < executor.capacity_factor
    )
    if need_recv_clamp:
      by_dst = jnp.transpose(jnp.reshape(all_exp_counts, (ep_size, ep_size, e_loc)), (1, 0, 2))
      flat_by_dst = jnp.reshape(by_dst, (ep_size, ep_size * e_loc))
      cum_end = jnp.cumsum(flat_by_dst, axis=1)
      cum_start = cum_end - flat_by_dst
      clamped_flat = jnp.clip(recv_buf_len - cum_start, 0, flat_by_dst)
      all_exp_counts = jnp.reshape(
          jnp.transpose(jnp.reshape(clamped_flat, (ep_size, ep_size, e_loc)), (1, 0, 2)),
          (ep_size, num_experts),
      )
      allowed_local = all_exp_counts[ep_id]
      pre_key = jnp.where(slot_valid, safe_idx, num_experts)
      pre_order = jnp.argsort(pre_key, stable=True)
      pre_starts = jnp.cumsum(local_exp_counts) - local_exp_counts
      pre_exp = safe_idx[pre_order]
      pre_rank = jnp.arange(num_slots, dtype=jnp.int32) - pre_starts[pre_exp]
      keep_pre = slot_valid[pre_order] & (pre_rank < allowed_local[pre_exp])
      slot_valid = keep_pre[jnp.argsort(pre_order)]
    if executor.capacity_factor > 0.0 or recv_buf_len < ep_size * num_slots:
      slot_valid = checkpoint_name(slot_valid.astype(jnp.int32), "moe_routing") != 0

    send_key = jnp.where(slot_valid, safe_idx, num_experts)
    send_perm = checkpoint_name(jnp.argsort(send_key, stable=True), "moe_routing")
    raw_inv_send_perm = jnp.argsort(send_perm)
    inv_send_bwd = (
        jnp.reshape(jnp.swapaxes(jnp.reshape(raw_inv_send_perm, (num_tokens, top_k)), 0, 1), (num_slots,))
        if use_slot_major
        else raw_inv_send_perm
    )
    inv_send_bwd = checkpoint_name(inv_send_bwd, "moe_routing")
    send_x = _gather_token_slots(flat_x, send_perm, inv_send_bwd, top_k, use_slot_major)

    counts_matrix = checkpoint_name(
        jnp.sum(jnp.reshape(all_exp_counts, (ep_size, ep_size, e_loc)), axis=2),
        "moe_routing",
    )
    recv_x = _ep_ragged_exchange(send_x, recv_buf_len, counts_matrix, ep_id, ep_axis, is_tpu=is_tpu)

    recv_exp_by_src = jax.lax.dynamic_slice_in_dim(all_exp_counts, ep_id * e_loc, e_loc, axis=1)
    local_group_sizes = checkpoint_name(jnp.sum(recv_exp_by_src, axis=0), "moe_routing")
    recv_valid_end = jnp.sum(local_group_sizes)
    chunk_sizes = jnp.reshape(recv_exp_by_src, (ep_size * e_loc,))
    chunk_ends = jnp.cumsum(chunk_sizes)
    row_idx = jnp.arange(recv_buf_len, dtype=jnp.int32)
    chunk_idx = _chunk_indices_from_ends(chunk_ends, recv_buf_len)
    recv_local_exp = jnp.where(row_idx < recv_valid_end, chunk_idx % e_loc, e_loc)

    local_perm = checkpoint_name(jnp.argsort(recv_local_exp, stable=True), "moe_routing")
    inv_local_perm = checkpoint_name(jnp.argsort(local_perm), "moe_routing")

    if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(recv_buf_len, d_dim, recv_x.dtype):
      from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

      sorted_recv_x = ragged_sort.a2a_ragged_sort(recv_x, local_perm, recv_valid_end)
    else:
      sorted_recv_x = _permute_rows(recv_x, local_perm, inv_local_perm)
      sorted_recv_x = jnp.where((row_idx < recv_valid_end)[:, None], sorted_recv_x, 0)
    sorted_recv_x = _tag_disp_ep(sorted_recv_x, executor.disp_ep_tags)

    expert_out = _run_local_expert_mlp(
        sorted_recv_x,
        local_projs,
        local_group_sizes,
        fused=fused or executor.moe_fp8,
        matmul_fn=matmul_fn,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(recv_buf_len, d_dim, expert_out.dtype):
      from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

      recv_unsorted = ragged_sort.a2a_ragged_unsort(expert_out, inv_local_perm, recv_valid_end)
    else:
      recv_unsorted = _permute_rows(expert_out, inv_local_perm, local_perm)
      recv_unsorted = jnp.where((row_idx < recv_valid_end)[:, None], recv_unsorted, 0)

    returned_sorted_slots = _ep_ragged_exchange(recv_unsorted, num_slots, counts_matrix.T, ep_id, ep_axis, is_tpu=is_tpu)
    if use_slot_major:
      send_perm_kt = (send_perm % top_k) * num_tokens + (send_perm // top_k)
      unsorted_kt = checkpoint_name(_permute_rows(returned_sorted_slots, inv_send_bwd, send_perm_kt), "moe_routed_output")
      valid_kt = jnp.reshape(jnp.swapaxes(jnp.reshape(slot_valid, (num_tokens, top_k)), 0, 1), (num_slots,))
      w_kt = jnp.reshape(jnp.swapaxes(jnp.reshape(flat_w, (num_tokens, top_k)), 0, 1), (num_slots,))
      eff_w_kt = jnp.where(valid_kt, w_kt, 0).astype(_COMBINE_DTYPE)
      safe_kt = jnp.where(valid_kt[:, None], unsorted_kt.astype(_COMBINE_DTYPE), 0.0)
      weighted_kt = safe_kt * eff_w_kt[:, None]
      combined = jnp.sum(jnp.reshape(weighted_kt, (top_k, num_tokens, d_dim)), axis=0)
    else:
      unsorted_slots = checkpoint_name(_permute_rows(returned_sorted_slots, inv_send_bwd, send_perm), "moe_routed_output")
      eff_w = jnp.where(slot_valid, flat_w, 0).astype(_COMBINE_DTYPE)
      safe_slots = jnp.where(slot_valid[:, None], unsorted_slots.astype(_COMBINE_DTYPE), 0.0)
      weighted = safe_slots * eff_w[:, None]
      combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
    if tp_axes:
      combined = jax.lax.psum(combined, tp_axes)
    return jnp.reshape(combined, (b_loc, seq_len, d_dim))

  sharded_fn = jax.shard_map(
      per_device,
      mesh=mesh,
      in_specs=(tokens_spec, tokens_spec, tokens_spec, *proj_specs),
      out_specs=tokens_spec,
      check_vma=False,
  )
  x_in = kernels.ensure_explicit_spec(x, tokens_spec, mesh)
  weights_in = kernels.ensure_explicit_spec(weights, tokens_spec, mesh)
  indices_in = kernels.ensure_explicit_spec(indices, tokens_spec, mesh)
  proj_in = tuple(kernels.ensure_explicit_spec(p, spec, mesh) for p, spec in zip(projections, proj_specs))
  return sharded_fn(x_in, weights_in, indices_in, *proj_in)


# ==============================================================================
# Public entrypoints
# ==============================================================================


def _validate_routed_inputs(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    num_experts: int,
) -> None:
  """Validates 3D token, routing weight, and routing index tensor shapes."""
  if x.ndim != 3 or weights.ndim != 3 or indices.ndim != 3:
    raise ValueError(
        f"Expected 3D x (B, S, D), weights (B, S, K), and indices (B, S, K); "
        f"got x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if x.shape[:2] != weights.shape[:2] or weights.shape != indices.shape:
    raise ValueError(
        f"Mismatched batch/length/top_k shapes: x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if num_experts < 1:
    raise ValueError(f"num_experts must be >= 1; got {num_experts}.")


def routed_experts(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
) -> jax.Array:
  """Executes selected experts with separate `(wi_0, wi_1, wo)` projections."""
  if wi_0.ndim != 3 or wi_1.ndim != 3 or wo.ndim != 3:
    raise ValueError(f"Expected 3D wi_0, wi_1, wo; got {wi_0.shape}, {wi_1.shape}, {wo.shape}.")
  num_experts = wi_0.shape[0]
  _validate_routed_inputs(x, weights, indices, num_experts)
  if wi_0.shape[1] != x.shape[2] or wo.shape[0] != num_experts or wo.shape[2] != x.shape[2]:
    raise ValueError(f"Incompatible expert weight shapes for x={x.shape}: wi_0={wi_0.shape}, wo={wo.shape}.")

  projections = (wi_0.astype(x.dtype), wi_1.astype(x.dtype), wo.astype(x.dtype))
  match executor.kind:
    case "dense":
      out = _dense_separate(
          x,
          weights,
          indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      out = _grouped_unsharded(
          x,
          weights,
          indices,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights,
          indices,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")


def routed_experts_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
) -> jax.Array:
  """Executes selected experts with fused 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim not in (3, 4) or down_proj.ndim != 3:
    raise ValueError(
        f"Expected 3D or 4D gate_up_proj and 3D down_proj; got gate_up_proj={gate_up_proj.shape}, "
        f"down_proj={down_proj.shape}."
    )
  if gate_up_proj.ndim == 3 and gate_up_proj.shape[2] % 2 != 0:
    raise ValueError(f"3D gate_up_proj last dim must be even (2 * mlp_dim); got {gate_up_proj.shape}.")
  if gate_up_proj.ndim == 4 and gate_up_proj.shape[2] != 2:
    raise ValueError(f"4D gate_up_proj axis 2 must have size 2 (gate, up); got {gate_up_proj.shape}.")

  num_experts = gate_up_proj.shape[0]
  _validate_routed_inputs(x, weights, indices, num_experts)
  if gate_up_proj.shape[1] != x.shape[2] or down_proj.shape[0] != num_experts or down_proj.shape[2] != x.shape[2]:
    raise ValueError(
        f"Incompatible fused expert weight shapes for x={x.shape}: "
        f"gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}."
    )

  projections = (gate_up_proj.astype(x.dtype), down_proj.astype(x.dtype))
  match executor.kind:
    case "dense":
      out = _dense_fused(
          x,
          weights,
          indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      out = _grouped_unsharded(
          x,
          weights,
          indices,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights,
          indices,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")
