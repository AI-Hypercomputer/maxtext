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

"""DeepSeek MoE router implementation."""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import math
from typing import Any, Generic, TypeVar

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops

compute_on = jax.experimental.compute_on.compute_on
ArrayType = jax.Array
ShardingType = jax.sharding.PartitionSpec
T = TypeVar("T", ArrayType, ShardingType, Any)
# MoE chunk capacities are rounded up to a multiple of this many slots, since
# the chunked MoE path produces garbage for unaligned capacities.
_CAPACITY_ALIGN = 256
# Matches `--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes`; XLA's
# SparseCore collective offloader rejects `tpu_sparsecore`-annotated all-gathers
# whose gathered output is smaller than this threshold.
_SC_ALL_GATHER_MIN_BYTES = 500_000


def _get_axis_size(axis_name: str | tuple[str, ...]) -> int:
  """Returns the size of the given axis name or combination of axis names."""
  if isinstance(axis_name, str):
    return jax.lax.axis_size(axis_name)
  else:
    return math.prod(jax.lax.axis_size(name) for name in axis_name)


def _get_axis_index(axis_name: str | tuple[str, ...]) -> jax.Array:
  """Returns the index of the given axis name or combination of axis names."""
  if isinstance(axis_name, str):
    return jax.lax.axis_index(axis_name)
  else:
    idx = jnp.array(0, dtype=jnp.int32)
    for name in axis_name:
      idx = idx * jax.lax.axis_size(name) + jax.lax.axis_index(name)
    return idx


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_get_token_axis(
    x_pspec: jax.sharding.PartitionSpec,
    num_model_dims: int = 1,
) -> str | tuple[str, ...] | None:
  """Extracts the token axis from a PartitionSpec for leading token dimensions."""
  token_axes = []
  for ax in x_pspec[:-num_model_dims] if num_model_dims > 0 else x_pspec:
    if ax is not None:
      if isinstance(ax, tuple):
        token_axes.extend(ax)
      else:
        token_axes.append(ax)
  if not token_axes:
    return None
  elif len(token_axes) == 1:
    return token_axes[0]
  else:
    return tuple(token_axes)


MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD: int = 4096


def _should_tc_fallback(
    axis_name: str | tuple[str, ...],
    num_tokens: int,
) -> bool:
  """Whether a routing-metadata collective should skip the SparseCore 0 block.

  Only the routing-metadata collectives (`_all_gather_routing_metadata` and
  `_psum_scatter_routing_metadata`) consult this. Token dispatch and combine
  always run as aggregated SparseCore 0 `compute_on` blocks: emitting them as
  plain collectives splits them into separately scheduled SparseCore offloads
  (cl/999064204 applied the fallback to them too; pdb=2 8x16x16 profiles showed
  the dispatch all-gathers landing on SparseCore 0 un-aggregated).

  On small per-device token counts (< MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD,
  e.g. pdb=1 with 2048 tokens/core), the SparseCore 0 block overlaps
  asynchronously with TensorCore compute without contention. At larger token
  counts (>= MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD, e.g. pdb >= 2 with 4096
  tokens/core) on multi-axis meshes, the metadata collectives are emitted as
  plain `jax.lax` collectives for XLA to schedule instead.

  Args:
    axis_name: Mesh axis or tuple of mesh axes along which to communicate.
    num_tokens: Explicit per-shard token count.

  Returns:
    True if the collective should be emitted as a plain `jax.lax` collective,
    False if it should run in a SparseCore 0 `compute_on` block.
  """
  if not (isinstance(axis_name, (tuple, list)) and len(axis_name) > 1):
    return False
  return num_tokens >= MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD


def _all_gather_sc0(
    *operands: jax.Array,
    axis_name: str | tuple[str, ...],
) -> tuple[jax.Array, ...]:
  """Executes all_gather on SparseCore 0.

  Falls back to TensorCore when any gathered output is below XLA's SparseCore
  offload threshold (`_SC_ALL_GATHER_MIN_BYTES`), e.g. a fractional eval batch.

  Args:
    *operands: Arrays to all-gather along axis 0.
    axis_name: Mesh axis or tuple of mesh axes along which to gather.

  Returns:
    A tuple of the gathered arrays.
  """
  axis_size = _get_axis_size(axis_name)
  if any(x.size * x.dtype.itemsize * axis_size < _SC_ALL_GATHER_MIN_BYTES for x in operands):
    return tuple(jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True) for x in operands)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _ag(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True) for x in xs)

  return _ag(*operands)


def _psum_scatter_sc0(
    *operands: jax.Array,
    axis_name: str | tuple[str, ...],
) -> tuple[jax.Array, ...]:
  """Executes psum_scatter on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _rs(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True) for x in xs)

  return _rs(*operands)


def _quantized_reduce_scatter_sc0(
    y_ag: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
    qtype: Any,
    per_hop_scaling: bool = False,
) -> jax.Array:
  """Token reduce-scatter of `y_ag` with `qtype` ring sends on SparseCore 0.

  `ops.reduce_scatter_sc0` keeps the local, D2D and output tokens in bf16 and
  only quantizes the running sums it sends along the ICI ring to `qtype`. It
  sums those raw `qtype` values, so every expert device must scale a token
  identically and no partial sum may overflow. The kernel therefore scales each
  token by (a power of two bounding) the sum over the expert devices of its
  local absmax over the `qtype` max and rescales the reduced tokens; only the
  local per-token absmax is computed here.

  Args:
    y_ag: Partial top-k reduced expert outputs of shape (NBT, D0, D1).
    axis_name: Physical expert axes, `(ring_axis, core_axis)`.
    qtype: Dtype of the ring sends; `jnp.float8_e4m3fn`.
    per_hop_scaling: Scale every ring hop by its own per-token bound instead of
      one all-device scale per token (see
      `ops.reduce_scatter_sc0(per_hop_scaling=...)`).

  Returns:
    This device's reduced tokens, shape (NBT / num_expert_devices, D0, D1), in
    `y_ag`'s dtype.

  Raises:
    ValueError: If `axis_name` is not a `(ring_axis, core_axis)` pair or `qtype`
      is not `jnp.float8_e4m3fn`.
  """
  if not (isinstance(axis_name, (tuple, list)) and len(axis_name) == 2):
    raise ValueError(
        "The quantized combine reduce-scatter needs a (ring_axis, core_axis)" f" expert axis mapping, got {axis_name!r}."
    )
  if qtype != jnp.float8_e4m3fn:
    raise ValueError(f"ops.reduce_scatter_sc0 only sends jnp.float8_e4m3fn, got {qtype!r}.")
  ring_axis, core_axis = axis_name
  # With `per_hop_scaling=True`, `ops.reduce_scatter_sc0` fuses the per-token
  # absmax of `local + D2D` directly into the SparseCore 0 vector subcores,
  # avoiding the standalone TensorCore `ops.token_absmax` pass.
  token_amax = None if per_hop_scaling else ops.token_absmax(y_ag)
  return ops.reduce_scatter_sc0(
      y_ag,
      token_amax=token_amax,
      per_hop_scaling=per_hop_scaling,
      ring_axis=ring_axis,
      core_axis=core_axis,
  )


def _psum_scatter_and_reduce_sc0(
    *operands: jax.Array,
    axis_name: str | tuple[str, ...],
    reduce_fn: Callable[[Any], Any],
    reduce_operand: Any,
) -> tuple[tuple[jax.Array, ...], Any]:
  """Executes psum_scatter, then `reduce_fn(reduce_operand)`, on SparseCore 0.

  Both run in one SparseCore block, so `reduce_fn`'s collectives start as soon
  as the psum_scatters are done.

  Args:
    *operands: Arrays to psum_scatter along dimension 0.
    axis_name: Mesh axis name(s) to psum_scatter over.
    reduce_fn: Function applied to `reduce_operand` after the psum_scatters,
      typically issuing further collectives.
    reduce_operand: Pytree passed to `reduce_fn`.

  Returns:
    A tuple of the scattered operands and `reduce_fn(reduce_operand)`.
  """

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _rs_then_reduce(xs: tuple[jax.Array, ...], reduce_operand: Any) -> tuple[tuple[jax.Array, ...], Any]:
    scattered = tuple(jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True) for x in xs)
    return scattered, reduce_fn(reduce_operand)

  return _rs_then_reduce(operands, reduce_operand)


# Whether the EP token reduce-scatters (the forward combine `dsv3_combine` and
# the dispatch transpose `dsv3_combine_with_coeff_grads*`) are replaced by
# `_a2a_reduce_scatter`. See `set_ep_combine_a2a`.
_EP_COMBINE_A2A_FWD = False
_EP_COMBINE_A2A_BWD = False
# Whether `_a2a_reduce_scatter`'s all-to-all runs on SparseCore 0.
_EP_COMBINE_A2A_SC = False
# XLA flag that, when unset ("auto"), offloads exactly the `compute_on`
# SparseCore-annotated dense all-to-alls to SparseCore.
_SC_DENSE_A2A_FLAG = "xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"
_EP_COMBINE_HIER_ORDERS = (
    None,
    "",
    "a2a_first",
    "rs_first",
    "rs_first_fused",
    "rs_first_ppermute",
)
# Whether the EP combine uses the `ops.reduce_scatter_sc0` kernel.
_EP_COMBINE_KERNEL = False
_EP_COMBINE_KERNEL_MB = 16
# Order of the hierarchical EP combine (`"a2a_first"` / `"rs_first"`), or None.
_EP_COMBINE_HIER: str | None = None
# Dtype of the partials `_a2a_reduce_scatter` sends.
_EP_COMBINE_A2A_DTYPE = jnp.bfloat16


def set_ep_combine_a2a(
    fwd: bool,
    bwd: bool | None = None,
    *,
    sparse_core: bool = False,
    hierarchical: str | None = None,
    kernel: bool = False,
    kernel_microbatches: int = 16,
) -> None:
  """Whether to reduce-scatter EP tokens as a bf16 all-to-all and a local sum.

  Takes effect for functions traced afterwards. Instead of a SparseCore 0
  reduce-scatter of the (f32 with `QuantConfig.ep_reduce_dtype`) top-k reduced
  partials, `_a2a_reduce_scatter` sends every expert device's partial of each
  token block to the block's owner in bf16 (an all-to-all) and sums the
  received partials locally in f32, which moves half the bytes of an f32
  reduce-scatter. Only the token reduce-scatters change; the coefficient
  gradients and `combine_qtype` combines keep their reduce-scatters. The
  backward of either is unchanged: the transpose of the all-to-all and local
  sum is a broadcast and the inverse all-to-all, i.e. the same all-gather as
  the transpose of the reduce-scatter (`dsv3_aggregated_dispatches`).

  Args:
    fwd: Whether to use it for the forward combine (`dsv3_combine`).
    bwd: Whether to use it for the dispatch transpose of the token gradients
      (`dsv3_combine_with_coeff_grads*`); defaults to `fwd`.
    sparse_core: Whether to run the all-to-all in a SparseCore 0 `compute_on`
      block instead of on the TensorCore. The compile options must then leave
      SparseCore dense all-to-all offload at "auto" (see
      `adjust_compiler_options`).
    hierarchical: If set, split the reduce-scatter over the expert axes
      `(*outer, inner)` (e.g. `("z", "core")`) into an all-to-all over the
      outer axes and an f32 reduce-scatter over the inner one, both on
      SparseCore 0 (implies `sparse_core`), in the given order:
      `"a2a_first"`: bf16 all-to-all over outer, local f32 sum, f32
        reduce-scatter over inner;
      `"rs_first"`: f32 reduce-scatter over inner, bf16 all-to-all over outer,
        local f32 sum;
      `"rs_first_fused"`: `"rs_first"` with the reduce-scatter, bf16 cast and
        all-to-all in one SparseCore block (no TensorCore cast in between);
      `"rs_first_ppermute"`: `"rs_first"` with the all-to-all done as
        num_outer - 1 (TensorCore) collective permutes over outer.
      See `_hier_reduce_scatter`.
    kernel: Instead, reduce-scatter f32 partials with the SparseCore Pallas
      kernel `ops.reduce_scatter_sc0` (f32 D2D and accumulation, bf16 ring
      sends, one bf16 rounding per hop). Needs `(ring_axis, core_axis)` expert
      axes and f32 partials (`QuantConfig.ep_reduce_dtype`); falls back to the
      XLA reduce-scatter otherwise.
    kernel_microbatches: `num_microbatches` of the kernel. On v7x, 16
      microbatches came within 1% of the best time, about 2.3x faster than the
      XLA SparseCore f32 reduce-scatter.
  """
  global _EP_COMBINE_A2A_FWD, _EP_COMBINE_A2A_BWD, _EP_COMBINE_A2A_SC
  global _EP_COMBINE_HIER, _EP_COMBINE_KERNEL, _EP_COMBINE_KERNEL_MB
  if hierarchical not in _EP_COMBINE_HIER_ORDERS:
    raise ValueError(f"Unknown hierarchical EP combine {hierarchical!r}.")
  _EP_COMBINE_A2A_FWD = fwd
  _EP_COMBINE_A2A_BWD = fwd if bwd is None else bwd
  _EP_COMBINE_HIER = hierarchical or None
  _EP_COMBINE_KERNEL = kernel
  _EP_COMBINE_KERNEL_MB = kernel_microbatches
  _EP_COMBINE_A2A_SC = (
      not kernel and (sparse_core or _EP_COMBINE_HIER is not None) and (_EP_COMBINE_A2A_FWD or _EP_COMBINE_A2A_BWD)
  )


# Whether another module (e.g. `dsv3_sparse_layer.set_fsdp_grad_a2a`) emits
# SparseCore `compute_on` dense all-to-alls. See
# `request_sparse_core_dense_a2a`.
_SC_DENSE_A2A_REQUESTED = False


def request_sparse_core_dense_a2a(enabled: bool) -> None:
  """Whether `adjust_compiler_options` must allow SparseCore dense all-to-alls.

  For all-to-alls in SparseCore `compute_on` blocks outside this module (e.g.
  the routed expert weight-gradient all-to-all of
  `dsv3_sparse_layer.set_fsdp_grad_a2a`).

  Args:
    enabled: Whether such all-to-alls are emitted.
  """
  global _SC_DENSE_A2A_REQUESTED
  _SC_DENSE_A2A_REQUESTED = enabled


def adjust_compiler_options(options: dict[str, Any]) -> dict[str, Any]:
  """Returns XLA compile options adjusted for `set_ep_combine_a2a`.

  Also for `request_sparse_core_dense_a2a`. With a SparseCore all-to-all, drops
  a forced value of the SparseCore dense all-to-all offload flag (the lineage
  config sets it to false), so it resolves to "auto": only the `compute_on`
  SparseCore-annotated all-to-alls (e.g. the EP combine) are offloaded and
  every other all-to-all stays on the TensorCore.

  Args:
    options: XLA compile options, e.g. parsed from `compile_xla_flags`.

  Returns:
    The adjusted options (a copy if changed).
  """
  if _EP_RAGGED_A2A_SC:
    options = {**options, _SC_RAGGED_A2A_FLAG: True}
  if not (_EP_COMBINE_A2A_SC or _SC_DENSE_A2A_REQUESTED) or _SC_DENSE_A2A_FLAG not in options:
    return options
  return {k: v for k, v in options.items() if k != _SC_DENSE_A2A_FLAG}


def _a2a_reduce_scatter(
    x: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
) -> jax.Array:
  """Tiled psum_scatter of x along dim 0 as an all-to-all and a local sum.

  Every device's partial of each 1/num_shards block of dim 0 is sent in
  `_EP_COMBINE_A2A_DTYPE` to the device owning the block, which sums the
  num_shards received partials in f32. The all-to-all runs on the TensorCore
  or, with `set_ep_combine_a2a(sparse_core=True)`, on SparseCore 0.

  Args:
    x: Partials of shape (num_shards * M, ...).
    axis_name: Mesh axis name(s) to reduce-scatter over.

  Returns:
    This device's reduced block of shape (M, ...), in x's dtype.
  """
  if _EP_COMBINE_KERNEL:
    if isinstance(axis_name, (tuple, list)) and len(axis_name) == 2 and x.dtype == jnp.float32 and x.ndim == 3:
      return ops.reduce_scatter_sc0(
          x,
          ring_axis=axis_name[0],
          core_axis=axis_name[1],
          num_microbatches=_EP_COMBINE_KERNEL_MB,
      )
    (out,) = _psum_scatter_sc0(x, axis_name=axis_name)
    return out
  if _EP_COMBINE_HIER is not None and not isinstance(axis_name, str):
    if len(axis_name) > 1:
      return _hier_reduce_scatter(x, axis_name=tuple(axis_name), order=_EP_COMBINE_HIER)
  num_shards = _get_axis_size(axis_name)

  def _a2a(y: jax.Array) -> jax.Array:
    return jax.lax.all_to_all(y, axis_name, split_axis=0, concat_axis=0, tiled=True)

  a2a_fn = _a2a
  if _EP_COMBINE_A2A_SC:
    a2a_fn = compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )(_a2a)
  parts = a2a_fn(x.astype(_EP_COMBINE_A2A_DTYPE))
  parts = jnp.reshape(parts, (num_shards, x.shape[0] // num_shards, *x.shape[1:]))
  return jnp.sum(parts.astype(jnp.float32), axis=0).astype(x.dtype)


def _sc0(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps fn in a SparseCore 0 `compute_on` block."""
  return compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )(fn)


def _hier_reduce_scatter(
    x: jax.Array,
    *,
    axis_name: tuple[str, ...],
    order: str,
) -> jax.Array:
  """Tiled psum_scatter of x along dim 0 over `(*outer, inner)`, in two stages.

  With `axis_name = (*outer, inner)`, the device with outer index o and inner
  index i owns block `o * num_inner + i` of dim 0 (as `psum_scatter` over the
  tuple). The two stages run on SparseCore 0 with the casts and local sums on
  the TensorCore in between:

  - `"a2a_first"`: each device sends, in `_EP_COMBINE_A2A_DTYPE`, its partials
    of outer group o' (blocks `o' * num_inner + [0, num_inner)`) to the device
    (o', i) (all-to-all over outer), sums the num_outer received partials in
    f32 and reduce-scatters the result over inner in f32.
  - `"rs_first"`: each device reduce-scatters, in f32 over inner, the blocks
    with inner index i (`[0, num_outer) * num_inner + i`), then sends the
    reduced block o' in `_EP_COMBINE_A2A_DTYPE` to (o', i) (all-to-all over
    outer) and sums the num_outer received partials in f32.

  Either is linear, so its transpose is the inverse all-to-all plus
  all-gather, i.e. the all-gather the backward pass already uses.

  Args:
    x: Partials of shape (num_shards * M, ...).
    axis_name: Mesh axes `(*outer, inner)` to reduce-scatter over.
    order: `"a2a_first"` or `"rs_first"`.

  Returns:
    This device's reduced block of shape (M, ...), in x's dtype.
  """
  outer, inner = axis_name[:-1], axis_name[-1]
  num_outer = _get_axis_size(outer)
  num_inner = _get_axis_size(inner)
  m = x.shape[0] // (num_outer * num_inner)
  rest = x.shape[1:]
  f32 = jnp.float32

  def _a2a_outer(y: jax.Array) -> jax.Array:
    return jax.lax.all_to_all(y, outer, split_axis=0, concat_axis=0, tiled=True)

  def _rs_inner(y: jax.Array, dim: int) -> jax.Array:
    return jax.lax.psum_scatter(y, axis_name=inner, scatter_dimension=dim, tiled=True)

  if order == "a2a_first":
    parts = _sc0(_a2a_outer)(x.astype(_EP_COMBINE_A2A_DTYPE))
    parts = jnp.reshape(parts, (num_outer, num_inner * m, *rest))
    summed = jnp.sum(parts.astype(f32), axis=0)
    out = _sc0(functools.partial(_rs_inner, dim=0))(summed)
  elif order in ("rs_first", "rs_first_fused", "rs_first_ppermute"):
    # (outer block, inner block, M, ...): reduce-scatter the inner block dim.
    blocks = jnp.reshape(x.astype(f32), (num_outer, num_inner, m, *rest))

    # Shapes stay (num_outer, 1, M, ...) on SparseCore: reshaping there makes
    # XLA insert relayout copies that SparseCore does not support.
    a2a_dtype = _EP_COMBINE_A2A_DTYPE

    def _rs(y: jax.Array) -> jax.Array:
      return _rs_inner(y, dim=1)

    if order == "rs_first":
      reduced = _sc0(_rs)(blocks).astype(a2a_dtype)
      parts = (_sc0(_a2a_outer)(reduced),)
    elif order == "rs_first_ppermute":
      outer_index = _get_axis_index(outer)

      def _permute(reduced: jax.Array) -> tuple[jax.Array, ...]:
        def _block(k):
          return jax.lax.dynamic_slice_in_dim(reduced, k % num_outer, 1, axis=0)

        # Shift s sends block o + s to device o + s, i.e. every device
        # receives the partial of its own block from device o - s.
        received = [_block(outer_index)]
        for shift in range(1, num_outer):
          received.append(
              jax.lax.ppermute(
                  _block(outer_index + shift),
                  outer,
                  perm=[(j, (j + shift) % num_outer) for j in range(num_outer)],
              )
          )
        return tuple(received)

      # Collective permutes in SparseCore blocks crash the TPU latency hiding
      # scheduler; they run (asynchronously) on the TensorCore.
      parts = _permute(_sc0(_rs)(blocks).astype(a2a_dtype))
    else:
      parts = (_sc0(lambda y: _a2a_outer(_rs(y).astype(a2a_dtype)))(blocks),)
    parts = jnp.concatenate([jnp.reshape(p, (-1, m, *rest)) for p in parts], axis=0)
    out = jnp.sum(parts.astype(f32), axis=0)
  else:
    raise ValueError(f"Unknown hierarchical EP combine order {order!r}.")
  return out.astype(x.dtype)


def _all_gather_routing_metadata(
    x: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
    num_tokens: int,
) -> jax.Array:
  """All-gathers routing metadata along axis 0.

  Runs on SparseCore 0 unless `_should_tc_fallback` says to emit a plain
  collective for XLA to schedule.

  Args:
    x: Array to all-gather along axis 0.
    axis_name: Mesh axis or tuple of mesh axes along which to gather.
    num_tokens: Per-shard token count, see `_should_tc_fallback`.

  Returns:
    The gathered array.
  """
  if _should_tc_fallback(axis_name, num_tokens):
    return jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
  (x_ag,) = _all_gather_sc0(x, axis_name=axis_name)
  return x_ag


def _psum_scatter_routing_metadata(
    x: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
    num_tokens: int,
) -> jax.Array:
  """Reduce-scatters routing metadata along dimension 0.

  Runs on SparseCore 0 unless `_should_tc_fallback` says to emit a plain
  collective for XLA to schedule.

  Args:
    x: Array to psum_scatter along dimension 0.
    axis_name: Mesh axis or tuple of mesh axes along which to scatter.
    num_tokens: Per-shard token count, see `_should_tc_fallback`.

  Returns:
    The scattered array.
  """
  if _should_tc_fallback(axis_name, num_tokens):
    return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True)
  (x_rs,) = _psum_scatter_sc0(x, axis_name=axis_name)
  return x_rs


def _gather_sc0(
    x: jax.Array,
    indices: jax.Array,
) -> jax.Array:
  """Executes a gather on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _gather(
      flat_x: jax.Array,
      flat_indices: jax.Array,
  ) -> jax.Array:
    return flat_x.at[flat_indices].get(mode="promise_in_bounds", wrap_negative_indices=False)

  out = _gather(jnp.ravel(x), jnp.ravel(indices))
  return jnp.reshape(out, indices.shape + x.shape[1:])


def _scatter_sc0(
    x: jax.Array,
    indices: jax.Array,
) -> jax.Array:
  """Executes a 1D permutation scatter on SparseCore 0.

  SparseCore scatter only supports add, and a float32 scatter-add is far slower
  than an int32 one, so the 32-bit patterns of `x` are scattered as int32 with
  unique indices into zeros, which reproduces them exactly.

  Args:
    x: Values to scatter.
    indices: Destination indices, a permutation of `[0, x.size)` (not checked at
      runtime).

  Returns:
    Array of shape `x.shape` with `out.ravel()[indices.ravel()[i]] =
    x.ravel()[i]`.
  """

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _scatter(
      zeros: jax.Array,
      flat_x: jax.Array,
      flat_indices: jax.Array,
  ) -> jax.Array:
    return zeros.at[flat_indices].add(
        flat_x,
        mode="promise_in_bounds",
        wrap_negative_indices=False,
        unique_indices=True,
    )

  sc_dtype = x.dtype if x.dtype in (jnp.float32, jnp.int32) else jnp.float32
  flat_x = jax.lax.bitcast_convert_type(jnp.ravel(x).astype(sc_dtype), jnp.int32)
  zeros = jnp.zeros_like(flat_x)
  out = _scatter(zeros, flat_x, jnp.ravel(indices))
  out = jax.lax.bitcast_convert_type(out, sc_dtype).astype(x.dtype)
  return jnp.reshape(out, x.shape)


# Whether the routing metadata includes a `RaggedExchange`, which switches the
# EP dispatch and combine to ragged all-to-alls. See `set_ep_ragged_a2a`.
_EP_RAGGED_A2A = False
# Whether the EP ragged all-to-alls are offloaded to SparseCore.
_EP_RAGGED_A2A_SC = False
_SC_RAGGED_A2A_FLAG = "xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all"


def set_ep_ragged_a2a(enabled: bool, *, sparse_core: bool = False) -> None:
  """Whether to dispatch and combine EP tokens with ragged all-to-alls.

  Takes effect for functions traced afterwards. `dsv3_routing_metadata` then
  also returns a `RaggedExchange` (`RouterMetadata.ragged`), and the MoE layer
  sends every (token, k) slot's row only to its expert's device with a
  `jax.lax.ragged_all_to_all`, instead of all-gathering the tokens over the
  expert axis, and sends the expert outputs back with the reverse ragged
  all-to-all, instead of reduce-scattering expert-side top-k partial sums.

  Args:
    enabled: Whether to use the ragged all-to-alls.
    sparse_core: Whether XLA offloads them to SparseCore instead of running
      them on the TensorCore (`adjust_compiler_options` then sets the
      SparseCore ragged all-to-all offload flag).
  """
  global _EP_RAGGED_A2A, _EP_RAGGED_A2A_SC
  _EP_RAGGED_A2A = enabled
  _EP_RAGGED_A2A_SC = enabled and sparse_core


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RaggedExchange(Generic[T]):
  """Per-device routing of the EP ragged all-to-alls (`set_ep_ragged_a2a`).

  Each device sorts its `kBT` (token, k) slots by (physical) expert, which
  orders them by destination device and then by local expert, and sends one
  slice per (destination, local expert) with `jax.lax.ragged_all_to_all`.
  Receivers lay their slices out expert-major and then source-major, i.e.
  exactly in the order of `RouterMetadata.sort_indices`, so the first
  `capacity` rows of `RouterMetadata.sorted_coeffs` and the first chunk of
  `RouterMetadata.group_sizes` describe the received buffer. Slots past a
  receiver's `capacity` rows are dropped (not sent, and contribute zero).

  The offsets and sizes are in `jax.lax.ragged_all_to_all`'s convention, with
  `num_devices * local_num_experts` slices (destination-major); the `rev_*`
  ones send the expert outputs back.
  """

  # (kBT,) sorted send position -> local slot (token * k + j).
  send_order: T
  # (kBT,) local slot -> sorted send position.
  slot_pos: T
  # (kBT,) whether the slot is sent, i.e. not dropped by the capacity.
  slot_sent: T
  input_offsets: T
  send_sizes: T
  output_offsets: T
  recv_sizes: T
  rev_input_offsets: T
  rev_send_sizes: T
  rev_output_offsets: T
  rev_recv_sizes: T


def _ragged_exchange_impl(
    selected_experts_flat: jt.Int[jax.Array, "kBT"],
    all_group_sizes: jt.Int[jax.Array, "num_devices E"],
    *,
    device_index: jax.Array,
    num_devices: int,
    local_num_experts: int,
    capacity: int,
) -> RaggedExchange[jax.Array]:
  """Computes this device's `RaggedExchange` from the slot count matrix.

  Args:
    selected_experts_flat: Physical expert of each local slot.
    all_group_sizes: Slot counts per (source device, physical expert).
    device_index: This device's index along the expert axes.
    num_devices: Number of devices along the expert axes.
    local_num_experts: Number of experts per device.
    capacity: Rows of each receiver's buffer.

  Returns:
    The ragged exchange routing.
  """
  n, el = num_devices, local_num_experts
  counts = jnp.reshape(all_group_sizes, (n, n, el)).astype(jnp.int32)
  # Receiver-side layout: expert-major, then source-major.
  per_dst_expert = jnp.sum(counts, axis=0)
  expert_base = jnp.cumsum(per_dst_expert, axis=1) - per_dst_expert
  src_base = jnp.cumsum(counts, axis=0) - counts
  start = expert_base[None] + src_base
  # Rows past the receiver's capacity are dropped.
  sizes = jnp.clip(capacity - start, 0, counts)
  start = jnp.minimum(start, capacity)
  # Sender-side layout: sorted by physical expert (destination-major).
  flat_counts = jnp.reshape(counts, (n, n * el))
  send_base = jnp.cumsum(flat_counts, axis=1) - flat_counts
  me = device_index
  input_offsets = send_base[me]
  send_sizes = jnp.reshape(sizes[me], (-1,))
  recv_sizes = jnp.reshape(sizes[:, me, :], (-1,))

  num_slots = selected_experts_flat.shape[0]
  send_order = _counting_argsort(selected_experts_flat, n * el)
  slot_pos = _scatter_sc0(jnp.arange(num_slots, dtype=jnp.int32), send_order)
  slot_sent = (slot_pos - input_offsets[selected_experts_flat]) < send_sizes[selected_experts_flat]
  return RaggedExchange[jax.Array](
      send_order=send_order,
      slot_pos=slot_pos,
      slot_sent=slot_sent,
      input_offsets=input_offsets,
      send_sizes=send_sizes,
      output_offsets=jnp.reshape(start[me], (-1,)),
      recv_sizes=recv_sizes,
      rev_input_offsets=jnp.reshape(start[:, me, :], (-1,)),
      rev_send_sizes=recv_sizes,
      rev_output_offsets=jnp.reshape(jnp.reshape(send_base, (n, n, el))[:, me, :], (-1,)),
      rev_recv_sizes=send_sizes,
  )


def _ragged_a2a(
    rows: jax.Array,
    num_out_rows: int,
    ragged: RaggedExchange[jax.Array],
    *,
    axis_name: str | tuple[str, ...],
    reverse: bool = False,
    zero_init: bool = False,
) -> jax.Array:
  """Sends slot rows to the expert devices (or back, with `reverse`)."""
  if reverse:
    offsets = (
        ragged.rev_input_offsets,
        ragged.rev_send_sizes,
        ragged.rev_output_offsets,
        ragged.rev_recv_sizes,
    )
  else:
    offsets = (
        ragged.input_offsets,
        ragged.send_sizes,
        ragged.output_offsets,
        ragged.recv_sizes,
    )
  shape = (num_out_rows, *rows.shape[1:])
  out = jnp.zeros(shape, rows.dtype) if zero_init else jax.lax.empty(shape, rows.dtype)

  # With `set_ep_ragged_a2a(sparse_core=True)`, XLA offloads it to SparseCore
  # (see `adjust_compiler_options`); the SparseCore lowering rejects ragged
  # all-to-alls in `compute_on` blocks.
  return jax.lax.ragged_all_to_all(rows, out, *offsets, axis_name=axis_name)


def _ragged_send_rows(x: jt.Num[jax.Array, "B T D"], ragged: RaggedExchange[jax.Array]) -> jt.Num[jax.Array, "kBT D0 D1"]:
  """Gathers the local tokens' rows in sorted send order."""
  x_flat = jnp.reshape(x, (-1, x.shape[-1] // 128, 128))
  num_slots = ragged.send_order.shape[0]
  k = num_slots // x_flat.shape[0]
  # Clamped so that garbage routing (e.g. HybridSim's random inputs) cannot
  # make the gather's DMAs go out of bounds; a no-op for valid routing.
  token_indices = jnp.clip(ragged.send_order // k, 0, x_flat.shape[0] - 1)
  return ops.ragged_gather_tc(x_flat, token_indices, 0, num_slots, num_slots)


def _ragged_combine_local(
    rows: jt.Num[jax.Array, "C D0 D1"],
    ragged: RaggedExchange[jax.Array],
    *,
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
) -> jt.Float[jax.Array, "BT D0 D1"]:
  """Sends expert-side rows back to their tokens and sums them over k in f32.

  Dropped slots (`ragged.slot_sent` False) contribute zero.

  Args:
    rows: Expert-side rows, laid out like the received dispatch buffer.
    ragged: Ragged exchange routing of this device.
    num_experts_per_tok: Number of slots per token.
    axis_name: Physical expert axes.

  Returns:
    Per-token sums over the token's k slots, in f32.
  """
  num_slots = ragged.send_order.shape[0]
  back = _ragged_a2a(rows, num_slots, ragged, axis_name=axis_name, reverse=True)
  gathered = jnp.take(back, ragged.slot_pos, axis=0)
  gathered = jnp.where(ragged.slot_sent[:, None, None], gathered.astype(jnp.float32), 0.0)
  return jnp.sum(
      jnp.reshape(gathered, (-1, num_experts_per_tok, *rows.shape[1:])),
      axis=1,
  )


@jax.named_call
def dsv3_ragged_dispatch(
    moe_in: jt.Num[jax.Array, "B T D"],
    metadata: Any,
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    grad_out: jt.Num[jax.Array, "B T D"] | None = None,
) -> Any:
  """Sends every slot's token row to its expert's device (ragged all-to-all).

  Args:
    moe_in: Local tokens of shape (B, T, D).
    metadata: `RouterMetadata` with a `RaggedExchange`.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    grad_out: Optional output cotangent (B, T, D) to send the same way, e.g. to
      transpose the combine in the backward pass.

  Returns:
    The received rows, (capacity, D0, D1) per device in the order of
    `metadata.sort_indices`, or a tuple of them and the received `grad_out`
    rows if `grad_out` is given.
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  token_axis = dsv3_get_token_axis(jax.typeof(moe_in).sharding.spec, num_model_dims=1)
  out_spec = ops.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  capacity = metadata.capacity

  def _send(x, ragged):
    return _ragged_a2a(
        _ragged_send_rows(x, ragged),
        capacity,
        ragged,
        axis_name=physical_expert_axis_name,
    )

  if grad_out is None:
    return jax.shard_map(_send, mesh=mesh, out_specs=out_spec, check_vma=False)(moe_in, metadata.ragged)

  def _send_both(x, g, ragged):
    return _send(x, ragged), _send(g, ragged)

  return jax.shard_map(
      _send_both,
      mesh=mesh,
      out_specs=(out_spec, out_spec),
      check_vma=False,
  )(moe_in, grad_out, metadata.ragged)


@jax.named_call
def dsv3_ragged_combine(
    y_rows: jt.Num[jax.Array, "C D0 D1"],
    ragged: RaggedExchange[jax.Array],
    *,
    num_experts_per_tok: int,
    local_seq_length: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Sends expert outputs back to their tokens and sums them over k.

  Args:
    y_rows: Expert outputs, laid out like the `dsv3_ragged_dispatch` rows.
    ragged: `RouterMetadata.ragged`.
    num_experts_per_tok: Number of experts selected per token.
    local_seq_length: Sequence length per shard.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for (B, T, D).
    out_dtype: Dtype of the output; defaults to y_rows' dtype.

  Returns:
    Combined output tokens of shape (B, T, D).
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  out_dtype = y_rows.dtype if out_dtype is None else out_dtype

  def _impl(rows, rg):
    y = _ragged_combine_local(
        rows,
        rg,
        num_experts_per_tok=num_experts_per_tok,
        axis_name=physical_expert_axis_name,
    ).astype(out_dtype)
    return jnp.reshape(y, (-1, local_seq_length, y.shape[-2] * y.shape[-1]))

  return jax.shard_map(
      _impl,
      mesh=mesh,
      out_specs=ops.physical_pspec(out_specs, axis_mapping),
      check_vma=False,
  )(y_rows, ragged)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RouterMetadata(Generic[T]):
  sort_indices: T
  group_sizes: T
  sorted_coeffs: T
  # Routing of the ragged gather reduce of chunk 0 in unpermute fwd and permute
  # bwd. A single PartitionSpec (a pytree prefix) in the spec tree.
  reduce_metadata: ops.RaggedGatherReduceMetadata | T
  # Forward routing decision of the local tokens, shape (BT, k). The backward
  # pass replays it (`dsv3_routing_replay`) instead of recomputing top-k.
  selected_experts: T
  # Forward pre-bias router scores sigmoid(x @ kernel) of the local tokens in
  # the router compute dtype, shape (BT, E). The backward pass differentiates
  # through these saved scores instead of recomputing the router projection.
  probs: T
  # Number of sorted local-expert slots processed per chunk on each device.
  capacity: int = jax.tree.static()
  # Routing of the EP ragged all-to-alls, with `set_ep_ragged_a2a`. A single
  # PartitionSpec (a pytree prefix) in the spec tree.
  ragged: RaggedExchange[T] | T | None = None


def chunk_num_tokens(
    num_local_tokens: jt.Int[jax.Array, ""],
    start: jt.Int[jax.Array, ""] | int,
    capacity: int,
) -> jt.Int[jax.Array, ""]:
  """Returns the number of local-expert slots in the chunk starting at `start`."""
  remaining = num_local_tokens - start
  return jnp.where(remaining >= capacity, capacity, remaining)


def chunk_group_sizes(
    group_sizes: jt.Int[jax.Array, "E_local"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
) -> jt.Int[jax.Array, "E_local"]:
  """Returns the per-expert sizes of sorted slots [start, start + num_tokens)."""
  ends = jnp.cumsum(group_sizes)
  begins = ends - group_sizes
  stop = start + num_tokens
  return (jnp.clip(ends, start, stop) - jnp.clip(begins, start, stop)).astype(group_sizes.dtype)


# Default `ragged_gather_reduce_tc` block size (VMEM rows per block).
_REDUCE_BLOCK_SIZE = 896


def reduce_block_size(token_bytes: int | None) -> int:
  """Returns the gather-reduce block size whose VMEM buffers fit the kernel.

  Args:
    token_bytes: Bytes of one reduced token row (e.g. `emb_dim * 4` for f32
      reductions), or None for the default.

  Returns:
    `_REDUCE_BLOCK_SIZE`, lowered to a multiple of 64 rows if its block buffers
    of `token_bytes` rows would not fit `ops.ragged_gather_reduce_tc`'s VMEM.
  """
  if token_bytes is None:
    return _REDUCE_BLOCK_SIZE
  max_rows = ops.ragged_gather_reduce_tc_max_block_size(token_bytes)
  return max(64, min(_REDUCE_BLOCK_SIZE, max_rows // 64 * 64))


def ep_reduce_token_bytes(emb_dim: int, ep_reduce_dtype: jax.typing.DTypeLike | None) -> int | None:
  """Bytes of one token row reduced in `ep_reduce_dtype`, or None if unset."""
  if ep_reduce_dtype is None:
    return None
  return emb_dim * jnp.dtype(ep_reduce_dtype).itemsize


def chunk_reduce_metadata(
    sort_indices: jt.Int[jax.Array, "kNBT"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
    *,
    capacity: int,
    num_experts_per_tok: int,
    token_bytes: int | None = None,
) -> ops.RaggedGatherReduceMetadata:
  """Derives the ragged gather reduce routing of one chunk of sorted slots.

  Args:
    sort_indices: Sorted slot order of the all-gathered tokens.
    start: First sorted slot of the chunk.
    num_tokens: Number of slots in the chunk.
    capacity: Maximum slots per chunk.
    num_experts_per_tok: Number of experts selected per token.
    token_bytes: Optional bytes of one reduced token row, which sizes the
      reduction blocks (see `reduce_block_size`); None for bf16-sized blocks.

  Returns:
    The chunk's ragged gather reduce routing.
  """
  return ops.ragged_gather_reduce_tc_metadata(
      sort_indices // num_experts_per_tok,
      start=start,
      num_tokens=num_tokens,
      top_k=num_experts_per_tok,
      block_size=reduce_block_size(token_bytes),
      max_num_tokens=capacity,
  )


def permute_chunk(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
    *,
    capacity: int,
    num_experts_per_tok: int,
    with_absmax: bool = False,
) -> Any:
  """Gathers sorted slots [start, start + num_tokens) into a capacity buffer."""
  return ops.ragged_gather_tc(
      x_ag,
      sort_indices // num_experts_per_tok,
      start=start,
      num_tokens=num_tokens,
      max_out_tokens=capacity,
      with_absmax=with_absmax,
  )


def unpermute_chunk(
    x: jt.Num[jax.Array, "C D0 D1"],
    reduce_metadata: ops.RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    num_experts_per_tok: int,
    out: jt.Num[jax.Array, "NBT D0 D1"] | None = None,
    zero_initialized: bool | None = None,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Sends a chunk's slots back to their tokens, summing over top-k."""
  return ops.ragged_gather_reduce_tc(
      x,
      reduce_metadata,
      num_out_tokens=num_out_tokens,
      top_k=num_experts_per_tok,
      out=out,
      zero_initialized=zero_initialized,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _sort_coeffs(
    coeffs: jt.Num[jax.Array, "BT k"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
) -> jt.Num[jax.Array, "kNBT 1"]:
  """All-gathers coeffs and sorts them into expert order with local experts first."""
  return _sort_coeffs_fwd(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      axis_name,
  )[0]


def _sort_coeffs_fwd(
    coeffs: jt.Num[jax.Array, "BT k"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
) -> tuple[
    jt.Num[jax.Array, "kNBT 1"],
    tuple[jt.Int[jax.Array, "kNBT"], jt.Int[jax.Array, "E_local"]],
]:
  del num_experts_per_tok
  coeffs_ag = _all_gather_routing_metadata(coeffs, axis_name=axis_name, num_tokens=coeffs.shape[0])
  sorted_coeffs = _gather_sc0(jnp.reshape(coeffs_ag, (-1, 1)), sort_indices)
  return sorted_coeffs, (sort_indices, group_sizes)


def _sort_coeffs_bwd(
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
    res: tuple[jt.Int[jax.Array, "kNBT"], jt.Int[jax.Array, "E_local"]],
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT 1"],
) -> tuple[jt.Num[jax.Array, "BT k"], None, None]:
  """Masks non-local slots, unsorts via scatter, and reduce-scatters coeff gradients."""
  sort_indices, group_sizes = res
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked_grad = jnp.where(valid[:, None], grad_sorted_coeffs, 0)
  unsorted = _scatter_sc0(masked_grad, sort_indices)
  grad_coeffs_ag = jnp.reshape(unsorted, (-1, num_experts_per_tok))
  num_tokens = grad_coeffs_ag.shape[0] // _get_axis_size(axis_name)
  grad_coeffs = _psum_scatter_routing_metadata(grad_coeffs_ag, axis_name=axis_name, num_tokens=num_tokens)
  return grad_coeffs, None, None


_sort_coeffs.defvjp(_sort_coeffs_fwd, _sort_coeffs_bwd)


def _cumsum_small_ints(
    x: jt.Shaped[jax.Array, "*B n"],
) -> jt.Int[jax.Array, "*B n"]:
  """Returns the inclusive cumsum over the last axis of small ints on the MXU.

  Exact, as bf16 represents the inputs and f32 the sums.

  Args:
    x: Bools or integers in [0, 256] whose sums are below 2**24.
  """
  n = x.shape[-1]
  upper = jnp.arange(n)[:, None] <= jnp.arange(n)[None, :]
  return jnp.matmul(
      x.astype(jnp.bfloat16),
      upper.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,
  ).astype(jnp.int32)


def _counting_argsort(keys: jt.Int[jax.Array, "N"], num_keys: int) -> jt.Int[jax.Array, "N"]:
  """Returns `jnp.argsort(keys, stable=True)` for keys in [0, num_keys).

  A counting sort, much faster on TPU than a sort for few distinct keys: a
  slot's destination is the number of slots with a smaller key plus the number
  of earlier slots with its key. The slots are laid out in blocks of 128 rows
  of 128 slots, the ranks are counted within rows and then within blocks on the
  MXU and across blocks by a short cumulative sum, and the destinations are
  inverted by a SparseCore scatter.

  Args:
    keys: Keys to sort, with values in [0, num_keys).
    num_keys: Number of distinct keys.
  """
  (n,) = keys.shape
  row = 128
  # Padded slots take the last key after every slot, so they sort last.
  rows = jnp.reshape(
      jnp.pad(keys, (0, -n % (row * row)), constant_values=num_keys - 1),
      (-1, row),
  )
  one_hot = rows[None] == jnp.arange(num_keys, dtype=keys.dtype)[:, None, None]
  # [c, r, i]: slots of key c in row r up to slot i.
  rank_in_row = _cumsum_small_ints(one_hot)
  row_counts = jnp.reshape(rank_in_row[..., -1], (num_keys, -1, row))
  # [c, b, r]: slots of key c in block b of 128 rows up to row r.
  rows_in_block = _cumsum_small_ints(row_counts)
  block_counts = rows_in_block[..., -1]
  key_counts = jnp.sum(block_counts, axis=1)
  # Slots with a smaller key, or with the same key in an earlier block or row.
  row_offsets = jnp.reshape(
      (jnp.cumsum(key_counts) - key_counts)[:, None, None]
      + (jnp.cumsum(block_counts, axis=1) - block_counts)[..., None]
      + rows_in_block
      - row_counts,
      (num_keys, -1),
  )
  dest = jnp.sum(jnp.where(one_hot, rank_in_row + row_offsets[..., None], 0), 0) - 1
  dest = jnp.ravel(dest)[:n]
  return _scatter_sc0(jnp.arange(n, dtype=jnp.int32), dest)


def _routing_metadata_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    capacity: int,
    router_dtype: jax.typing.DTypeLike | None = None,
    ep_reduce_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    RouterMetadata[jax.Array],
    jt.Num[jax.Array, "BT k"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Computes routing metadata and coefficients for local tokens."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))

  (
      selected_experts,
      coeffs,
      local_group_sizes,
      logits,
      probs,
  ) = _expert_selection_impl(
      x_flat,
      w,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      router_dtype=router_dtype,
  )

  # Routing metadata (sort order, local group sizes) is computed on physical
  # expert positions, which differ from logical expert ids when the routed
  # expert weights are shuffled. Aux outputs stay in logical expert order.
  if expert_permutations is not None:
    physical_selected_experts = dsv3_expert_shuffle.remap_selected_experts(selected_experts, expert_permutations)
    physical_group_sizes = dsv3_expert_shuffle.remap_group_sizes(local_group_sizes, expert_permutations)
  else:
    physical_selected_experts = selected_experts
    physical_group_sizes = local_group_sizes

  # Metadata computation across all devices along the expert axis.
  num_devices = _get_axis_size(physical_expert_axis_name)
  local_num_experts = num_experts // num_devices
  device_index = _get_axis_index(physical_expert_axis_name)

  selected_experts_flat = jnp.ravel(physical_selected_experts)
  all_selected_experts = _all_gather_routing_metadata(
      selected_experts_flat,
      axis_name=physical_expert_axis_name,
      num_tokens=physical_selected_experts.shape[0],
  )

  # Place tokens routed to local experts at the start of the buffer in expert
  # order; non-local tokens are placed at the end.
  local_expert = all_selected_experts - device_index * local_num_experts
  is_local = (local_expert >= 0) & (local_expert < local_num_experts)
  sort_key = jnp.where(is_local, local_expert, local_num_experts)
  sort_indices = _counting_argsort(sort_key, local_num_experts + 1)

  # Sizes
  all_group_sizes = jax.lax.all_gather(physical_group_sizes, axis_name=physical_expert_axis_name, axis=0)
  all_send_sizes_3d = jnp.reshape(all_group_sizes, (num_devices, num_devices, local_num_experts))
  group_sizes = jnp.sum(all_send_sizes_3d, axis=0)[device_index]

  sorted_coeffs = _sort_coeffs(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      physical_expert_axis_name,
  )

  # Permute fwd gathers chunk 0, the first min(sum(group_sizes), capacity)
  # local-expert slots of sort_indices // k; unpermute fwd and permute bwd
  # reduce them. Later chunks derive their routing in the MoE loop.
  reduce_metadata = chunk_reduce_metadata(
      sort_indices,
      0,
      chunk_num_tokens(jnp.sum(group_sizes, dtype=jnp.int32), 0, capacity),
      capacity=capacity,
      num_experts_per_tok=num_experts_per_tok,
      token_bytes=ep_reduce_token_bytes(x.shape[-1], ep_reduce_dtype),
  )

  metadata = RouterMetadata[jax.Array](
      sort_indices=sort_indices,
      group_sizes=group_sizes,
      sorted_coeffs=sorted_coeffs,
      reduce_metadata=reduce_metadata,
      selected_experts=selected_experts,
      probs=jax.lax.stop_gradient(probs),
      capacity=capacity,
      ragged=_ragged_exchange_impl(
          selected_experts_flat,
          all_group_sizes,
          device_index=device_index,
          num_devices=num_devices,
          local_num_experts=local_num_experts,
          capacity=capacity,
      )
      if _EP_RAGGED_A2A
      else None,
  )

  selected_experts = jnp.reshape(selected_experts, x.shape[:-1] + (num_experts_per_tok,))
  logits = jnp.reshape(logits, x.shape[:-1] + (num_experts,))
  aux = dsv3_types.DSv3RouterAux[jax.Array](
      group_sizes=local_group_sizes[None, :],
      selected_experts=selected_experts,
      logits=logits,
  )
  return metadata, coeffs, aux


def chunk_capacity(num_local_slots: int, num_expert_shards: int, capacity_factor: float) -> int:
  """Returns the number of sorted local-expert slots in each MoE chunk.

  Args:
    num_local_slots: Number of local tokens times `num_experts_per_tok`.
    num_expert_shards: Number of devices the experts are sharded over.
    capacity_factor: A chunk holds `capacity_factor * num_local_slots` slots,
      rounded up to a multiple of `_CAPACITY_ALIGN`, but never more than all the
      slots that can be routed to the device.

  Returns:
    The chunk capacity.
  """
  return min(
      num_local_slots * num_expert_shards,
      math.ceil(capacity_factor * num_local_slots / _CAPACITY_ALIGN) * _CAPACITY_ALIGN,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_metadata(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    capacity_factor: float,
    router_dtype: jax.typing.DTypeLike | None = None,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
    ep_reduce_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    RouterMetadata[jax.Array],
    jt.Num[jax.Array, "BT k"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Computes routing metadata and routing coefficients for all tokens.

  Args:
    x: Input tokens of shape (B, T, D).
    w: MoE router weights.
    num_experts: Total number of routed experts.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    capacity_factor: Each MoE chunk processes `capacity_factor` times the
      average number of slots routed to each device's local experts, i.e. the
      number of local tokens times `num_experts_per_tok`, rounded up to a
      multiple of `_CAPACITY_ALIGN` slots.
    router_dtype: Optional dtype for expert selection; see `expert_selection`.
    expert_permutations: Optional physical position of each logical expert (see
      `dsv3_expert_shuffle`). Routing decisions, coefficients, and aux outputs
      are unaffected; only the metadata (sort order and local group sizes) is
      expressed in physical expert positions so that it matches shuffled routed
      expert weights.
    ep_reduce_dtype: Optional dtype of the expert-parallel token reductions (see
      `quantization.QuantConfig.ep_reduce_dtype`), which sizes the gather-reduce
      blocks of `RouterMetadata.reduce_metadata`.

  Returns:
    A tuple of (RouterMetadata, coeffs, aux).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)

  coeffs_spec = jax.sharding.PartitionSpec(token_axis, None)
  token_spec_1d = jax.sharding.PartitionSpec(token_axis)
  token_pspec = x_pspec[:-1]
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)

  def _num_shards(axis: str | tuple[str, ...] | None) -> int:
    if axis is None:
      return 1
    if isinstance(axis, str):
      return mesh.shape[axis]
    return math.prod(mesh.shape[a] for a in axis)

  num_local_slots = math.prod(x.shape[:-1]) // _num_shards(token_axis) * num_experts_per_tok
  capacity = chunk_capacity(num_local_slots, _num_shards(physical_expert_axis_name), capacity_factor)

  metadata_out_spec = RouterMetadata[jax.sharding.PartitionSpec](
      sort_indices=token_spec_1d,
      group_sizes=jax.sharding.PartitionSpec(physical_expert_axis_name),
      sorted_coeffs=coeffs_spec,
      reduce_metadata=token_spec_1d,
      selected_experts=coeffs_spec,
      probs=coeffs_spec,
      capacity=capacity,
      ragged=token_spec_1d if _EP_RAGGED_A2A else None,
  )
  aux_out_spec = dsv3_types.DSv3RouterAux[jax.sharding.PartitionSpec](
      group_sizes=jax.sharding.PartitionSpec(token_axis, None),
      selected_experts=jax.sharding.PartitionSpec(*token_pspec, None),
      logits=jax.sharding.PartitionSpec(*token_pspec, None),
  )
  out_specs = (metadata_out_spec, coeffs_spec, aux_out_spec)

  if expert_permutations is not None:
    expert_permutations = jax.reshard(
        expert_permutations,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
  return jax.shard_map(
      functools.partial(
          _routing_metadata_impl,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          capacity=capacity,
          router_dtype=router_dtype,
          ep_reduce_dtype=ep_reduce_dtype,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(x, w, expert_permutations)


@jax.custom_vjp
def _saved_router_probs(
    x: jt.Num[jax.Array, "BT D"],
    kernel: jt.Num[jax.Array, "D E"],
    probs: jt.Num[jax.Array, "BT E"],
) -> jt.Num[jax.Array, "BT E"]:
  """Returns the saved `probs = sigmoid(x @ kernel)` without recomputing them.

  The VJP is that of `sigmoid(x @ kernel)` evaluated at the saved `probs`. The
  sigmoid derivative runs in fp32; the projection transposes take bf16 (the
  dtype of `x` and `kernel`) operands with fp32 accumulation and return the
  gradients in the dtypes of `x` and `kernel`.

  Args:
    x: Flattened local input tokens.
    kernel: Router kernel.
    probs: Saved forward `sigmoid(x @ kernel)`.

  Returns:
    `probs`.
  """
  del x, kernel
  return probs


def _saved_router_probs_fwd(
    x: jax.Array, kernel: jax.Array, probs: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  return probs, (x, kernel, probs)


def _saved_router_probs_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array],
    grad_probs: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  x, kernel, probs = res
  input_dtype = jnp.result_type(x, kernel)
  probs_f32 = probs.astype(jnp.float32)
  grad_z = (grad_probs.astype(jnp.float32) * probs_f32 * (1.0 - probs_f32)).astype(input_dtype)
  grad_x = jnp.dot(
      grad_z,
      kernel.astype(input_dtype).T,
      preferred_element_type=jnp.float32,
  ).astype(x.dtype)
  grad_kernel = jnp.dot(
      x.astype(input_dtype).T,
      grad_z,
      preferred_element_type=jnp.float32,
  ).astype(kernel.dtype)
  return grad_x, grad_kernel, None


_saved_router_probs.defvjp(_saved_router_probs_fwd, _saved_router_probs_bwd)


def _routing_replay_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  """Rebuilds the local sorted coefficients and logits from saved routing."""
  assert w.kernel is not None
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = selected_coeffs(
      pre_bias_logits,
      selected_experts,
      routed_scaling_factor=routed_scaling_factor,
  ).astype(x.dtype)
  sorted_coeffs = _sort_coeffs(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      physical_expert_axis_name,
  )
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return sorted_coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_replay(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  """Replays the forward routing of `dsv3_routing_metadata` for its backward.

  Returns the `(metadata.sorted_coeffs, aux.logits)` that
  `dsv3_routing_metadata` produced for `metadata`, as differentiable functions
  of `x` and `w`, without recomputing the routing: expert selection, the sort
  into expert order and the group sizes come from `metadata`, and the pre-bias
  router scores are the saved `metadata.probs`. Differentiating the replay
  therefore uses exactly the forward routing decisions and scores, even when a
  recomputation would round differently. The gradients match those of
  `dsv3_routing_metadata` with the same routing; they are zero for the bias,
  which only enters the routing.

  Args:
    x: Input tokens of shape (B, T, D), as passed to `dsv3_routing_metadata`.
    w: MoE router weights, as passed to `dsv3_routing_metadata`.
    metadata: Routing metadata that `dsv3_routing_metadata` returned for `x` and
      `w`.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of (sorted_coeffs, logits).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)
  out_specs = (
      jax.sharding.PartitionSpec(token_axis, None),
      jax.sharding.PartitionSpec(*x_pspec[:-1], None),
  )
  return jax.shard_map(
      functools.partial(
          _routing_replay_impl,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(
      x,
      w,
      metadata.selected_experts,
      metadata.probs,
      metadata.sort_indices,
      metadata.group_sizes,
  )


def _unsort_coeff_grads_impl(
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    *,
    num_shards: int,
) -> jt.Num[jax.Array, "R C"]:
  """Unsorts this device's coefficient gradients into all-gathered slot order.

  Slots past the local experts' tokens are zeroed, so every slot is nonzero on
  at most one device (the one holding its expert) and a psum_scatter over the
  expert axis sends each local token exactly its coefficient gradients. The
  result is laid out lane-dense, (slots / 128, 128) when that splits evenly
  across `num_shards`, with each shard's block contiguous along dim 0: a
  (slots, 1) or (tokens, k) layout is padded to 128 lanes and makes the
  SparseCore reduce-scatter more than an order of magnitude slower.

  Args:
    grad_sorted_coeffs: Cotangent of the sorted coefficients on this device.
    sort_indices: Sorted slot -> all-gathered slot (token * k + j) permutation.
    group_sizes: Local expert group sizes.
    num_shards: Size of the expert axis.

  Returns:
    The unsorted coefficient gradients of all all-gathered slots, reshaped for
    a tiled psum_scatter along dim 0.
  """
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked = jnp.where(valid, grad_sorted_coeffs, 0).astype(grad_sorted_coeffs.dtype)
  unsorted = _scatter_sc0(masked, sort_indices)
  num_slots = unsorted.shape[0]
  if num_slots % (num_shards * 128) == 0:
    return jnp.reshape(unsorted, (num_slots // 128, 128))
  return jnp.reshape(unsorted, (num_shards, num_slots // num_shards))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unsort_coeff_grads(
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT"],
    metadata: RouterMetadata[jax.Array],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "R C"]:
  """Locally unsorts coefficient gradients for `dsv3_combine_with_coeff_grads`.

  The unsort is a SparseCore scatter that the combine's reduce-scatter depends
  on. Callers should make sure it completes before the compute that the
  combine is meant to overlap (e.g. with an optimization barrier); otherwise
  the scheduler defers its completion behind that compute and the
  reduce-scatter starts late.

  Args:
    grad_sorted_coeffs: Cotangent of `metadata.sorted_coeffs`, flattened.
    metadata: Router metadata of the microbatch.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Per-device unsorted, lane-dense coefficient gradients of all all-gathered
    slots, sharded along dim 0 like `grad_sorted_coeffs`.
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  token_axis = jax.typeof(grad_sorted_coeffs).sharding.spec[0]

  def _impl(g, sort_indices, group_sizes):
    return _unsort_coeff_grads_impl(
        g,
        sort_indices,
        group_sizes,
        num_shards=_get_axis_size(physical_expert_axis_name),
    )

  return jax.shard_map(
      _impl,
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None),
      check_vma=False,
  )(grad_sorted_coeffs, metadata.sort_indices, metadata.group_sizes)


def _reshape_combined_grads(
    grad_x_local: jax.Array,
    grad_coeffs_local: jax.Array,
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  """Reshapes (and optionally casts) the local psum_scatter outputs."""
  if out_dtype is not None:
    grad_x_local = grad_x_local.astype(out_dtype)
  grad_x = jnp.reshape(
      grad_x_local,
      (-1, local_seq_length, grad_x_local.shape[-2] * grad_x_local.shape[-1]),
  )
  return grad_x, jnp.reshape(grad_coeffs_local, (-1, num_experts_per_tok))


def _combine_with_coeff_grads_impl(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    ragged: RaggedExchange[jax.Array] | None = None,
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  """Per-shard body of `dsv3_combine_with_coeff_grads`."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  if ragged is not None:
    # grad_x_ag holds the expert-side token gradient rows (see
    # `set_ep_ragged_a2a`).
    grad_x_local = _ragged_combine_local(
        grad_x_ag,
        ragged,
        num_experts_per_tok=num_experts_per_tok,
        axis_name=physical_expert_axis_name,
    )
    if out_dtype is None:
      out_dtype = grad_x_ag.dtype
    (grad_coeffs_local,) = _psum_scatter_sc0(grad_coeffs_ag, axis_name=physical_expert_axis_name)
  elif _EP_COMBINE_A2A_BWD:
    grad_x_local = _a2a_reduce_scatter(grad_x_ag, axis_name=physical_expert_axis_name)
    (grad_coeffs_local,) = _psum_scatter_sc0(grad_coeffs_ag, axis_name=physical_expert_axis_name)
  else:
    grad_x_local, grad_coeffs_local = _psum_scatter_sc0(grad_x_ag, grad_coeffs_ag, axis_name=physical_expert_axis_name)
  return _reshape_combined_grads(
      grad_x_local,
      grad_coeffs_local,
      local_seq_length=local_seq_length,
      num_experts_per_tok=num_experts_per_tok,
      out_dtype=out_dtype,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_with_coeff_grads(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    metadata: RouterMetadata[jax.Array],
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  """Transposes the dispatch and the coefficient sort in one psum_scatter.

  The token gradients are reduce-scattered in grad_x_ag's dtype (e.g. f32 with
  `quantization.QuantConfig.ep_reduce_dtype`) and cast to `out_dtype`, if set.

  The token gradients `grad_x_ag` go through the same psum_scatter as
  `dsv3_combine`. The coefficient gradients, already unsorted into
  all-gathered slot order by `dsv3_unsort_coeff_grads`, are reduce-scattered
  in the same SparseCore block, so the router backward needs no collective of
  its own. The coefficient gradients are exactly those of `_sort_coeffs`'s VJP.

  Args:
    grad_x_ag: Cotangent of the all-gathered tokens, shape (NBT, D0, D1).
    grad_coeffs_ag: Output of `dsv3_unsort_coeff_grads`.
    metadata: Router metadata of the microbatch.
    local_seq_length: Sequence length per shard.
    num_experts_per_tok: Number of experts selected per token.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for the (B, T, D) token gradients.
    out_dtype: Optional dtype the reduce-scattered token gradients are cast to.

  Returns:
    A tuple of the local token gradients (B, T, D) and the local token
    coefficient gradients (BT, k), sharded like `metadata.selected_experts`.
  """
  out_specs_phys = ops.physical_pspec(out_specs, axis_mapping)
  coeffs_spec = jax.typeof(metadata.selected_experts).sharding.spec
  ragged = metadata.ragged
  assert ragged is None or isinstance(ragged, RaggedExchange)
  return jax.shard_map(
      functools.partial(
          _combine_with_coeff_grads_impl,
          local_seq_length=local_seq_length,
          num_experts_per_tok=num_experts_per_tok,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          out_dtype=out_dtype,
      ),
      mesh=mesh,
      out_specs=(out_specs_phys, coeffs_spec),
      check_vma=False,
  )(grad_x_ag, grad_coeffs_ag, ragged)


def _combine_with_coeff_grads_and_reduce_impl(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    reduce_operand: list[jax.Array],
    ragged: RaggedExchange[jax.Array] | None = None,
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    reduce_fn: Callable[[list[jax.Array]], list[jax.Array]],
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"], list[jax.Array]]:
  """Per-shard body of `dsv3_combine_with_coeff_grads_and_reduce`."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  if ragged is not None:
    grad_x_local = _ragged_combine_local(
        grad_x_ag,
        ragged,
        num_experts_per_tok=num_experts_per_tok,
        axis_name=physical_expert_axis_name,
    )
    if out_dtype is None:
      out_dtype = grad_x_ag.dtype
    # The coefficient gradients (needed right away by the router backward) do
    # not wait for the deferred reduction, which runs in its own block.
    (grad_coeffs_local,) = _psum_scatter_sc0(grad_coeffs_ag, axis_name=physical_expert_axis_name)
    _, reduced = _psum_scatter_and_reduce_sc0(
        axis_name=physical_expert_axis_name,
        reduce_fn=reduce_fn,
        reduce_operand=reduce_operand,
    )
  elif _EP_COMBINE_A2A_BWD:
    grad_x_local = _a2a_reduce_scatter(grad_x_ag, axis_name=physical_expert_axis_name)
    (grad_coeffs_local,), reduced = _psum_scatter_and_reduce_sc0(
        grad_coeffs_ag,
        axis_name=physical_expert_axis_name,
        reduce_fn=reduce_fn,
        reduce_operand=reduce_operand,
    )
  else:
    (grad_x_local, grad_coeffs_local), reduced = _psum_scatter_and_reduce_sc0(
        grad_x_ag,
        grad_coeffs_ag,
        axis_name=physical_expert_axis_name,
        reduce_fn=reduce_fn,
        reduce_operand=reduce_operand,
    )
  grad_x, grad_coeffs = _reshape_combined_grads(
      grad_x_local,
      grad_coeffs_local,
      local_seq_length=local_seq_length,
      num_experts_per_tok=num_experts_per_tok,
      out_dtype=out_dtype,
  )
  return grad_x, grad_coeffs, reduced


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_with_coeff_grads_and_reduce(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    metadata: RouterMetadata[jax.Array],
    deferred_reduce: ops.DeferredReduce | None,
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"], Any]:
  """`dsv3_combine_with_coeff_grads` that also runs a `DeferredReduce`.

  The reduction runs in the same SparseCore block as the psum_scatter, right
  after it, so it starts as soon as the psum_scatter is done rather than when
  the TensorCore compute that it overlaps with reaches a point where it could
  be issued. The token and coefficient gradients are then available only once
  the reduction is done too.

  Args:
    grad_x_ag: Cotangent of the all-gathered tokens, shape (NBT, D0, D1).
    grad_coeffs_ag: Output of `dsv3_unsort_coeff_grads`.
    metadata: Router metadata of the microbatch.
    deferred_reduce: Reduction to run after the psum_scatter, or `None`.
    local_seq_length: Sequence length per shard.
    num_experts_per_tok: Number of experts selected per token.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for the (B, T, D) token gradients.
    out_dtype: Optional dtype the reduce-scattered token gradients are cast to.

  Returns:
    The outputs of `dsv3_combine_with_coeff_grads`, and the output of
    `deferred_reduce.finish` (`None` if `deferred_reduce` is `None`).
  """
  if deferred_reduce is None:
    grad_x, grad_coeffs = dsv3_combine_with_coeff_grads(
        grad_x_ag,
        grad_coeffs_ag,
        metadata,
        local_seq_length=local_seq_length,
        num_experts_per_tok=num_experts_per_tok,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
        out_specs=out_specs,
        out_dtype=out_dtype,
    )
    return grad_x, grad_coeffs, None
  out_specs_phys = ops.physical_pspec(out_specs, axis_mapping)
  coeffs_spec = jax.typeof(metadata.selected_experts).sharding.spec
  ragged = metadata.ragged
  assert ragged is None or isinstance(ragged, RaggedExchange)
  grad_x, grad_coeffs, reduced = jax.shard_map(
      functools.partial(
          _combine_with_coeff_grads_and_reduce_impl,
          local_seq_length=local_seq_length,
          num_experts_per_tok=num_experts_per_tok,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          reduce_fn=deferred_reduce.local_fn,
          out_dtype=out_dtype,
      ),
      mesh=mesh,
      out_specs=(out_specs_phys, coeffs_spec, deferred_reduce.out_specs),
      check_vma=False,
  )(grad_x_ag, grad_coeffs_ag, deferred_reduce.operand, ragged)
  return grad_x, grad_coeffs, deferred_reduce.finish(reduced)


def _dense_selected_coeffs(
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    indices: jt.Int[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """`selected_coeffs` with a one-hot select instead of a SparseCore gather.

  Each selected score is the only nonzero term of its sum, so the gathered
  scores, and the scatter-add transpose over the k distinct experts of a
  token, are bitwise those of the gather. The (BT, k, E) select fuses into a
  TensorCore reduction, avoiding the SparseCore sort and scatter-add that the
  gather's transpose needs.

  Args:
    pre_bias_logits: Router scores of each token, shape (BT, E).
    indices: Selected experts of each token, shape (BT, k).
    routed_scaling_factor: Scaling applied to the normalized coefficients.

  Returns:
    The normalized, scaled coefficients of the selected experts, (BT, k).
  """
  num_experts = pre_bias_logits.shape[-1]
  one_hot = indices[..., None] == jnp.arange(num_experts, dtype=indices.dtype)
  coeffs = jnp.sum(
      jnp.where(one_hot, pre_bias_logits[:, None, :], 0),
      axis=-1,
      dtype=pre_bias_logits.dtype,
  )
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _local_routing_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    *,
    routed_scaling_factor: float,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "B T E"]]:
  """Per-shard body of `dsv3_local_routing`."""
  assert w.kernel is not None
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = _dense_selected_coeffs(
      pre_bias_logits,
      selected_experts,
      routed_scaling_factor=routed_scaling_factor,
  ).astype(x.dtype)
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_local_routing(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    *,
    routed_scaling_factor: float,
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "B T E"]]:
  """Computes local token coefficients and logits from saved router scores."""
  x_pspec = jax.typeof(x).sharding.spec
  out_specs = (
      jax.typeof(selected_experts).sharding.spec,
      jax.sharding.PartitionSpec(*x_pspec[:-1], None),
  )
  return jax.shard_map(
      functools.partial(
          _local_routing_impl,
          routed_scaling_factor=routed_scaling_factor,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(x, w, selected_experts, probs)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_bwd(
    grad_coeffs: jt.Num[jax.Array, "BT k"],
    grad_logits: jt.Num[jax.Array, "B T E"] | None,
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    metadata: RouterMetadata[jax.Array],
    *,
    routed_scaling_factor: float,
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3MoERouterWeightsPytree]:
  """Computes the router input and weight gradients without communication.

  Differentiates the local routing (coefficients and logits of the local
  tokens) at the saved routing decision and scores. `grad_coeffs` are the
  unsorted, local coefficient gradients from `dsv3_combine_with_coeff_grads`,
  which already did the cross-device part of `_sort_coeffs`'s transpose.

  Args:
    grad_coeffs: Coefficient gradients of the local tokens, shape (BT, k).
    grad_logits: Cotangent of the aux logits, or None.
    x: Router input tokens, shape (B, T, D).
    w: MoE router weights.
    metadata: Router metadata of the microbatch.
    routed_scaling_factor: Scaling factor for routed token scores.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of (grad_x, grad_w).
  """

  def _fwd(x, w):
    return dsv3_local_routing(
        x,
        w,
        metadata.selected_experts,
        metadata.probs,
        routed_scaling_factor=routed_scaling_factor,
        mesh=mesh,
    )

  (_, logits), vjp_fn = jax.vjp(_fwd, x, w)
  if grad_logits is None:
    grad_logits = jnp.zeros_like(logits)
  elif jax.typeof(grad_logits).sharding != jax.typeof(logits).sharding:
    grad_logits = jax.reshard(grad_logits, jax.typeof(logits).sharding)
  grad_x, grad_w = vjp_fn((grad_coeffs, grad_logits))
  return grad_x, grad_w


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _permute_impl(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> jt.Num[jax.Array, "C D0 D1"]:
  return _permute_impl_fwd(x_ag, metadata, num_experts_per_tok)[0]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_fwd(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> tuple[
    jt.Num[jax.Array, "C D0 D1"],
    RouterMetadata,
]:
  """Sorts chunk 0 of all-gathered tokens into local expert order."""
  num_local_tokens = jnp.sum(metadata.group_sizes, dtype=jnp.int32)
  out = permute_chunk(
      x_ag,
      metadata.sort_indices,
      0,
      chunk_num_tokens(num_local_tokens, 0, metadata.capacity),
      capacity=metadata.capacity,
      num_experts_per_tok=num_experts_per_tok,
  )
  return out, metadata


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_bwd(
    num_experts_per_tok: int,
    metadata: RouterMetadata,
    grads: jt.Num[jax.Array, "C D0 D1"],
) -> tuple[
    jt.Num[jax.Array, "NBT D0 D1"],
    None,
]:
  """Backward pass for permute: sums chunk 0 slots of each token over k."""
  grad_x_ag = unpermute_chunk(
      grads,
      metadata.reduce_metadata,
      num_out_tokens=metadata.sort_indices.shape[0] // num_experts_per_tok,
      num_experts_per_tok=num_experts_per_tok,
  )
  return grad_x_ag, None


_permute_impl.defvjp(_permute_impl_fwd, _permute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  """Permutes chunk 0 of all-gathered tokens into sorted expert order.

  Args:
    x_ag: All-gathered tokens of shape (NBT, D0, D1).
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Sorted token buffer of shape (capacity, D0, D1) per device holding the first
    min(sum(group_sizes), capacity) local-expert slots.
  """
  x_pspec = jax.typeof(x_ag).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)

  return jax.shard_map(
      functools.partial(
          _permute_impl,
          num_experts_per_tok=num_experts_per_tok,
      ),
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(x_ag, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute_bwd(
    grad_x_sorted: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Backward pass for permute returning gradients wrt all-gathered tokens."""
  x_pspec = jax.typeof(grad_x_sorted).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)

  def _bwd(grads, md):
    grad_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, md, grads)
    return grad_x_ag

  return jax.shard_map(
      _bwd,
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(grad_x_sorted, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch_impl(
    x: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Dispatches local tokens via all-gather on SC0."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1] // 128, 128))
  (x_ag,) = _all_gather_sc0(x_flat, axis_name=physical_expert_axis_name)
  return x_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch(
    x: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Dispatches tokens across the expert axis via all-gather.

  Args:
    x: Input tokens of shape (B, T, D).
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    All-gathered tokens x_ag of shape (NBT, D0, D1).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=1)
  x_ag_spec = ops.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_dispatch_impl,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=x_ag_spec,
  )(x)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches_impl(
    moe_in: jt.Num[jax.Array, "B T D"],
    grad_out: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  """All-gathers moe_in and grad_out in a single SC0 compute_on block."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  moe_in_flat = jnp.reshape(moe_in, (-1, moe_in.shape[-1] // 128, 128))
  grad_out_flat = jnp.reshape(grad_out, (-1, grad_out.shape[-1] // 128, 128))
  x_ag, grad_y_ag = _all_gather_sc0(moe_in_flat, grad_out_flat, axis_name=physical_expert_axis_name)
  return x_ag, grad_y_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches(
    moe_in: jt.Num[jax.Array, "B T D"],
    grad_out: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  """Rematerializes dispatch of moe_in and transposes combine of grad_out.

  Args:
    moe_in: Input tokens of shape (B, T, D).
    grad_out: Upstream output cotangent of shape (B, T, D).
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Tuple of (x_ag, grad_y_ag), each of shape (NBT, D0, D1).
  """
  x1_pspec = jax.typeof(moe_in).sharding.spec
  x2_pspec = jax.typeof(grad_out).sharding.spec
  token_axis = dsv3_get_token_axis(x1_pspec, num_model_dims=1) or dsv3_get_token_axis(x2_pspec, num_model_dims=1)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  out_specs = (token_spec, token_spec)
  return jax.shard_map(
      functools.partial(
          dsv3_aggregated_dispatches_impl,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(moe_in, grad_out)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_impl(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    *,
    local_seq_length: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    combine_qtype: Any | None = None,
    combine_per_hop_scaling: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Combines reduced expert output tokens across devices via psum_scatter."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  if combine_qtype is not None:
    y_local = _quantized_reduce_scatter_sc0(
        y_ag,
        axis_name=physical_expert_axis_name,
        qtype=combine_qtype,
        per_hop_scaling=combine_per_hop_scaling,
    )
  elif _EP_COMBINE_A2A_FWD:
    y_local = _a2a_reduce_scatter(y_ag, axis_name=physical_expert_axis_name)
  else:
    (y_local,) = _psum_scatter_sc0(y_ag, axis_name=physical_expert_axis_name)
  if out_dtype is not None:
    y_local = y_local.astype(out_dtype)
  return jnp.reshape(
      y_local,
      (-1, local_seq_length, y_local.shape[-2] * y_local.shape[-1]),
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    *,
    local_seq_length: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
    combine_qtype: Any | None = None,
    combine_per_hop_scaling: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Combines reduced expert output tokens across devices via psum_scatter.

  The reduce-scatter runs in y_ag's dtype (e.g. f32 with
  `quantization.QuantConfig.ep_reduce_dtype`), unless `combine_qtype` is set.

  Args:
    y_ag: Partial top-k reduced expert outputs of shape (NBT, D0, D1).
    local_seq_length: Sequence length per shard.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for (B, T, D).
    combine_qtype: If set, reduce-scatters in this dtype with
      `ops.reduce_scatter_sc0` (see `quantization.QuantConfig.combine_qtype`).
    combine_per_hop_scaling: With `combine_qtype`, scale every ring hop by its
      own per-token bound (see
      `quantization.QuantConfig.combine_per_hop_scaling`).
    out_dtype: Optional dtype the reduce-scattered tokens are cast to.

  Returns:
    Combined output tokens of shape (B, T, D).
  """
  out_specs_phys = ops.physical_pspec(out_specs, axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_combine_impl,
          local_seq_length=local_seq_length,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          combine_qtype=combine_qtype,
          combine_per_hop_scaling=combine_per_hop_scaling,
          out_dtype=out_dtype,
      ),
      mesh=mesh,
      out_specs=out_specs_phys,
      # The SparseCore Pallas lowering has no `pvary` rule, which VMA checking
      # inserts when resolving the kernel's remote DMA `device_id`s.
      check_vma=combine_qtype is None and not (_EP_COMBINE_KERNEL and _EP_COMBINE_A2A_FWD),
  )(y_ag)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _unpermute_impl(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Unsorts local expert outputs, masks non-local slots, and reduces over k."""
  return _unpermute_impl_fwd(x, metadata, num_experts_per_tok)[0]


def _unpermute_impl_fwd(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], RouterMetadata]:
  """Forward pass for unpermute, which is the transpose of permute."""
  combined_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, metadata, x)
  return combined_x_ag, metadata


def _unpermute_impl_bwd(
    num_experts_per_tok: int,
    metadata: RouterMetadata,
    cotangent_ag: jt.Num[jax.Array, "NBT D0 D1"],
) -> tuple[jt.Num[jax.Array, "C D0 D1"], None]:
  """Backward pass for unpermute, which is identical to permute."""
  grad_x, _ = _permute_impl_fwd(cotangent_ag, metadata, num_experts_per_tok)
  return grad_x, None


_unpermute_impl.defvjp(_unpermute_impl_fwd, _unpermute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Unsorts chunk 0 local expert tokens and reduces over top-k on each device.

  Args:
    x: Sorted, coefficient-weighted expert output tokens of chunk 0, of shape
      (capacity, D0, D1) per device.
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Partial top-k reduced tokens of shape (NBT, D0, D1).
  """
  token_axis = dsv3_get_token_axis(jax.typeof(x).sharding.spec, num_model_dims=2)
  out_specs = jax.sharding.PartitionSpec(token_axis, None, None)
  return jax.shard_map(
      functools.partial(
          _unpermute_impl,
          num_experts_per_tok=num_experts_per_tok,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute_bwd(
    cotangent_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  """Computes gradient wrt sorted expert outputs; identical to permute."""
  return dsv3_permute(
      cotangent_ag,
      metadata,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_group_mask(
    gate_logits: jt.Num[jax.Array, "BT E"],
    *,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
) -> jt.Num[jax.Array, "BT E"]:
  """Computes expert group mask for node-limited routing.

  Args:
    gate_logits: Gate logits for all experts of shape (BT, E).
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.

  Returns:
    Mask of shape (BT, E) indicating which experts can be selected.
  """
  num_experts = gate_logits.shape[-1]
  # Flatten to 2D `(batch * seq * n_routing_groups, experts_per_group)` before
  # `top_k` so XLA's `TpuTopkRewriter` keeps the batch dimension in the 8-row
  # tile dimension `{0, 1:T(8, 128)}` instead of transposing 3D inputs to
  # `{1, 2, 0:T(8, 128)}`.
  scores_grouped_2d = jnp.reshape(
      gate_logits,
      (-1, num_experts // n_routing_groups),
  )
  topk_in_group_vals, _ = jax.lax.top_k(scores_grouped_2d, k=topk_in_group)
  group_scores = jnp.reshape(
      jnp.sum(jnp.astype(topk_in_group_vals, jnp.float32), axis=-1),
      gate_logits.shape[:-1] + (n_routing_groups,),
  )
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)

  # Mask selected groups so that only those experts are considered.
  group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
  group_mask = jnp.sum(group_mask, axis=-2)

  # Apply masks and get top-k indices.
  score_mask_expanded = jnp.broadcast_to(
      group_mask[..., None],
      group_mask.shape + (num_experts // n_routing_groups,),
  )
  return jnp.reshape(
      score_mask_expanded,
      score_mask_expanded.shape[:-2] + (num_experts,),
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_indices_and_coeffs(
    gate_logits: jt.Num[jax.Array, "BT E"],
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "BT k"]]:
  """Computes expert indices for each token and their corresponding scores.

  Args:
    gate_logits: Gate logits for all experts of shape (BT, E).
    pre_bias_logits: Pre-bias sigmoid logits of shape (BT, E).
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.

  Returns:
    A tuple of (selected_expert_indices, routing_coefficients).
  """
  expert_mask = expert_group_mask(
      gate_logits,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )
  gate_logits = jnp.where(expert_mask > 0, gate_logits, -jnp.inf)

  _, indices = jax.lax.top_k(
      gate_logits,
      k=num_experts_per_tok,
  )
  coeffs = selected_coeffs(pre_bias_logits, indices, routed_scaling_factor=routed_scaling_factor)
  return indices, coeffs


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def selected_coeffs(
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    indices: jt.Int[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """Gathers and normalizes the pre-bias scores of the selected experts.

  Args:
    pre_bias_logits: Pre-bias sigmoid logits of shape (BT, E).
    indices: Selected expert indices of shape (BT, k).
    routed_scaling_factor: Scaling factor for routed token scores.

  Returns:
    Routing coefficients of shape (BT, k).
  """
  num_experts_per_tok = indices.shape[-1]
  bt, num_e = pre_bias_logits.shape
  flat_logits = jnp.ravel(pre_bias_logits)
  flat_indices = jnp.ravel(indices + jnp.arange(bt, dtype=indices.dtype)[:, None] * num_e)

  # We don't pin to SC0 if padding is required because it results in
  # an XLA SC fusion error. This code path is only used for the tiny test.
  if flat_indices.shape[0] % 1024 != 0:
    coeffs = (
        flat_logits.at[flat_indices]
        .get(mode="promise_in_bounds", wrap_negative_indices=False)
        .reshape(bt, num_experts_per_tok)
    )
  else:

    @compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    def _sc_gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return flat_a.at[flat_idx].get(mode="promise_in_bounds", wrap_negative_indices=False)

    @compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    def _sc_scatter_add_f32(zeros_f32: jax.Array, flat_idx: jax.Array, updates_f32: jax.Array) -> jax.Array:
      return zeros_f32.at[flat_idx].add(updates_f32, mode="promise_in_bounds", wrap_negative_indices=False)

    @jax.custom_vjp
    def _gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return _sc_gather(flat_a, flat_idx)

    def _gather_fwd(flat_a: jax.Array, flat_idx: jax.Array):
      return _sc_gather(flat_a, flat_idx), flat_idx

    def _gather_bwd(flat_idx: jax.Array, g: jax.Array):
      zeros_f32 = jnp.zeros(flat_logits.shape, dtype=jnp.float32)
      grad_a = _sc_scatter_add_f32(zeros_f32, flat_idx, g.astype(jnp.float32)).astype(g.dtype)
      return grad_a, None

    _gather.defvjp(_gather_fwd, _gather_bwd)

    coeffs = _gather(flat_logits, flat_indices).reshape(bt, num_experts_per_tok)
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _normalize_coeffs(
    coeffs: jt.Num[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """Normalizes the selected scores of each token and applies the scaling."""
  # Normalize in fp32 and round once, so the forward and the backward
  # (`dsv3_routing_bwd`) produce identical coefficients from identical scores
  # regardless of how XLA fuses a low-precision normalization.
  coeffs_f32 = coeffs.astype(jnp.float32)
  coeffs_f32 = routed_scaling_factor * (coeffs_f32 / coeffs_f32.sum(-1, keepdims=True))
  return coeffs_f32.astype(coeffs.dtype)


_RANDOM_ROUTING = False


def set_random_routing(enabled: bool) -> None:
  """Routes tokens to (pseudo-)random experts, for benchmarking only.

  Takes effect for functions traced afterwards. The routing is a deterministic
  hash of the fp32 router projection, so it is independent per token and
  expert (and therefore load balanced in expectation) and reproducible, and the
  backward replays it like the regular routing.

  Args:
    enabled: Whether to route randomly.
  """
  global _RANDOM_ROUTING
  _RANDOM_ROUTING = enabled


def _hash_uniform(v: jax.Array) -> jax.Array:
  """Uniform [0, 1) fp32 values from a bit hash (lowbias32) of fp32 `v`."""
  b = jax.lax.bitcast_convert_type(v.astype(jnp.float32), jnp.uint32)
  b ^= b >> 16
  b *= jnp.uint32(0x7FEB352D)
  b ^= b >> 15
  b *= jnp.uint32(0x846CA68B)
  b ^= b >> 16
  return (b >> 8).astype(jnp.float32) * jnp.float32(2.0**-24)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_selection(
    x: jt.Num[jax.Array, "BT D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"],
]:
  """Selects experts for each token and calculates group sizes for each expert.

  Expert selection (projection accumulation, sigmoid, bias add, group
  selection, top-k and coefficient normalization) runs in `router_dtype`
  promoted with the dtypes of `x` and the router weights. With bf16 tokens and
  weights and `router_dtype=jnp.float32` (MaxText `float32_gate_logits=True`),
  the bf16 x bf16 projection is accumulated in fp32 and everything after it is
  fp32, while the router weights, and therefore their gradients and weight
  collectives, stay bf16. The coefficients and the returned `logits` are cast
  back to `x.dtype`, so the dispatch, expert and combine kernels and the
  auxiliary routing statistics are unaffected by the router dtype. The returned
  `logits` are the pre-bias scores `sigmoid(x @ kernel)` (MaxText's
  `pre_bias_logits`): the bias only steers the expert selection.

  Args:
    x: Flattened input tokens of shape (BT, D).
    w: MoE router weights.
    num_experts: Total number of routed experts.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.
    router_dtype: Optional minimum dtype for expert selection. None computes in
      the promoted dtype of `x` and the router weights.

  Returns:
    A tuple of (selected_experts, coeffs, group_sizes, logits).
  """
  return _expert_selection_impl(
      x,
      w,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      router_dtype=router_dtype,
  )[:4]


def _expert_selection_impl(
    x: jt.Num[jax.Array, "BT D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"],
    jt.Num[jax.Array, "BT E"],
]:
  """`expert_selection` that also returns the pre-bias scores.

  Returns:
    A tuple of (selected_experts, coeffs, group_sizes, logits, probs), where
    `probs = sigmoid(x @ kernel)` in the router compute dtype and `logits` is
    `probs` cast to `x.dtype`.
  """
  assert w.kernel is not None
  assert w.bias is not None
  input_dtype = jnp.result_type(x, w.kernel)
  compute_dtype = jnp.result_type(input_dtype, w.bias)
  if router_dtype is not None:
    compute_dtype = jnp.promote_types(compute_dtype, router_dtype)
  pre_bias_logits = jax.nn.sigmoid(
      jnp.tensordot(
          x.astype(input_dtype),
          w.kernel.astype(input_dtype),
          axes=1,
          preferred_element_type=compute_dtype,
      )
  )
  if jnp.finfo(compute_dtype).bits < 32:
    # XLA may keep the fused sigmoid in fp32 (excess precision). Round it so
    # the routing and coefficients use exactly the scores saved for the
    # backward replay.
    finfo = jnp.finfo(compute_dtype)
    pre_bias_logits = jax.lax.reduce_precision(pre_bias_logits, exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
  logits = pre_bias_logits + jax.lax.stop_gradient(w.bias.astype(compute_dtype))
  if _RANDOM_ROUTING:
    # Benchmarking only: select experts by scores that are uniform and
    # independent per token and expert, so the load is balanced in
    # expectation whatever the weights. The coefficients still come from the
    # real router scores of the selected experts.
    logits = _hash_uniform(
        jnp.tensordot(
            x.astype(input_dtype),
            w.kernel.astype(input_dtype),
            axes=1,
            preferred_element_type=jnp.float32,
        )
    ).astype(compute_dtype)
    logits = jax.lax.stop_gradient(logits)
  selected_experts, coeffs = expert_indices_and_coeffs(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )
  coeffs = coeffs.astype(x.dtype)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _bincount(x: jax.Array) -> jax.Array:
    return jnp.bincount(x, length=num_experts)

  group_sizes = _bincount(jnp.ravel(selected_experts))
  return (
      selected_experts,
      coeffs,
      group_sizes,
      pre_bias_logits.astype(x.dtype),
      pre_bias_logits,
  )
