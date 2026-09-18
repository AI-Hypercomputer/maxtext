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

"""Common operations shared across models."""

from collections.abc import Callable, Mapping
import functools
import string
from typing import Any, Sequence, Union
import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

AnyJaxArray = jt.Num[jax.Array, "..."]
AnyJaxArrayOrPyTree = AnyJaxArray | jt.PyTree[AnyJaxArray]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def rms_norm(
    x: jt.Num[jax.Array, "*input_dims"],
    scale: jt.Num[jax.Array, "..."] | None = None,
    *,
    epsilon: float,
    axis: Union[int, Sequence[int]] = -1,
    norm_dtype: jt.DTypeLike = jnp.float32,
) -> jt.Num[jax.Array, "*input_dims"]:
  """Performs RMS normalization.

  Args:
    x: Input array.
    scale: Scale array, which scales the last `scale.ndim` dimensions of x. The
      shape of scale must be a suffix of the shape of x.
    epsilon: Epsilon value for RMS normalization.
    axis: Optional, int or sequence of ints, default=-1. Axis or set of axes to
      use for normalization.
    norm_dtype: Dtype used for normalization. Defaults to float32.

  Returns:
    The normalized array.
  """
  assert scale is None or scale.shape == x.shape[-scale.ndim :], (
      f"scale.shape must be a suffix of x.shape, but got {scale.shape} and" f" {x.shape}"
  )

  x_dtype = x.dtype
  x = jnp.astype(x, norm_dtype)
  mean2 = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), x_dtype)
  if scale is None:
    return y
  scale_dims = string.ascii_lowercase[: scale.ndim]
  return jnp.einsum(f"...{scale_dims},{scale_dims}->...{scale_dims}", y, scale)


def physical_pspec(
    logical_pspec: jax.sharding.PartitionSpec,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jax.sharding.PartitionSpec:
  """Translates logical axes in a PartitionSpec to physical mesh axes via axis_mapping."""

  def _map_axis(
      axis: str | tuple[str, ...] | None,
  ) -> str | tuple[str, ...] | None:
    if axis is None:
      return None
    if isinstance(axis, str):
      # Map a single logical axis to physical mesh axis/axes via axis_mapping,
      # deduplicating any repeated physical axes while preserving order if
      # mapped to a tuple.
      mapped = axis_mapping.get(axis, axis)
      if isinstance(mapped, tuple):
        return tuple(dict.fromkeys(mapped))
      return mapped
    out = []
    for sub_axis in axis:
      mapped = axis_mapping.get(sub_axis, sub_axis)
      if isinstance(mapped, tuple):
        out.extend(mapped)
      else:
        out.append(mapped)
    return tuple(dict.fromkeys(out))

  def _map_set(s: frozenset[str] | None) -> frozenset[str]:
    if not s:
      return frozenset()
    out = set()
    for p in s:
      mapped = _map_axis(p)
      if isinstance(mapped, tuple):
        out.update(mapped)
      elif mapped is not None:
        out.add(mapped)
    return frozenset(out)

  physical_partitions = tuple(_map_axis(p) for p in logical_pspec.partitions)
  reduced = _map_set(logical_pspec.reduced)
  unreduced = _map_set(logical_pspec.unreduced)
  return jax.sharding.PartitionSpec(*physical_partitions, reduced=reduced, unreduced=unreduced)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_along_axis(
    xs: jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]],
    axis_name: str | Sequence[str],
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]]:
  """All-gathers each array along the given axis names if it is sharded along them.

  The dual of this function will perform a reduce-scatter along the given axes
  for each array that is sharded along them. When an array is not sharded
  along a given axis, the function is a no-op in the forward pass. However, the
  dual of this function will perform an all-reduce along that axis. This makes
  it useful for specifying when the all-reduces should occur for specific axes
  in the backward pass, such as for pipelining all-reduces across DCN.

  Args:
    xs: Input array or pytree of arrays, which may or may not be sharded along
      the given axis names.
    axis_name: Name of the axis or sequence of axis names to collect along. For
      weights, these should be the axes used for data parallelism for the
      activations that go through them.
    axis_mapping: Mapping from logical to physical mesh axes.

  Returns:
    The array all-gathered along the given axis names if sharded along them.
  """

  def _to_tuple(m: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(m, str):
      return (m,)
    return tuple(m)

  physical_axis_names = set()
  for name in _to_tuple(axis_name):
    for ax in _to_tuple(axis_mapping.get(name, name)):
      physical_axis_names.add(ax)

  def _remove_axis_names(
      partitions: tuple[str | tuple[str, ...] | None, ...],
  ) -> tuple[str | tuple[str, ...] | None, ...]:
    """Removes axis names from the partitions of a PartitionSpec for use in jax.reshard to ensure replication."""
    out = []
    for a in partitions:
      if a is None:
        out.append(None)
      elif isinstance(a, str):
        out.append(a if a not in physical_axis_names else None)
      else:
        out.append(tuple(b for b in a if b not in physical_axis_names))
    return tuple(out)

  def _collect_arr(arr):
    sharding = jax.typeof(arr).sharding
    target_spec = jax.sharding.PartitionSpec(
        *_remove_axis_names(sharding.spec.partitions),
        # Concatenates the existing reduced axes with the new axes.
        reduced=sharding.spec.reduced | frozenset(physical_axis_names),
        unreduced=sharding.spec.unreduced,
    )
    return jax.reshard(
        arr,
        jax.sharding.NamedSharding(sharding.mesh, target_spec),
    )

  return jax.tree.map(_collect_arr, xs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def w_collect_pipelined_scan(
    scan_fn: Callable[
        [AnyJaxArray, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArray, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning over scan_fn.

  Args:
    scan_fn: Function to execute at each scan iteration with signature `(x:
      AnyJaxArray, w: AnyJaxArrayOrPyTree) -> tuple[AnyJaxArray,
      AnyJaxArrayOrPyTree]` which takes current activations and collected
      weights and returns `(output, aux_step)`.
    x: Input array.
    w: Uncollected weights to scan over. Dimension 0 is the dimension to scan
      over and must be present.
    collect_fn: Function to collect weights.

  Returns:
    A tuple of (output, aux) where aux is the stacked auxiliary outputs across
    all scan iterations.
  """
  w_first = collect_fn(jax.tree.map(lambda _w: _w[0], w))
  carry = (x, w_first)

  def body(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn(w_n_sharded)
    x, aux_step = scan_fn(x, w)
    return (x, w_n), aux_step

  (x, w_last), aux_scanned = jax.lax.scan(body, carry, jax.tree.map(lambda _w: _w[1:], w))
  x, aux_last = scan_fn(x, w_last)
  aux = jax.tree.map(
      lambda s, l: jnp.concatenate([s, l[jnp.newaxis]], axis=0),
      aux_scanned,
      aux_last,
  )
  return x, aux


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_fwd(
    scan_fn: Callable[
        [AnyJaxArray, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArray, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[tuple[AnyJaxArray, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan which does not save collected weights."""
  assert jax.tree.leaves(w)[0].shape[0] >= 2, (
      "there must be at least two scan iterations to perform a pipelined scan" " and its transpose"
  )
  # Derive the reduce function (the dual of the collect function)
  w_first, reduce_fn = jax.vjp(collect_fn, jax.tree.map(lambda _w: _w[0], w))
  getattr(reduce_fn, "args_res")[0] = None

  (x, aux_first), scan_fn_vjp_first = jax.vjp(scan_fn, x, w_first)
  getattr(scan_fn_vjp_first, "args_res")[1] = None

  w_second = collect_fn(jax.tree.map(lambda _w: _w[1], w))
  carry = (x, w_second)

  def body_fwd(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn(w_n_sharded)
    (x, aux_step), scan_fn_vjp = jax.vjp(scan_fn, x, w)
    getattr(scan_fn_vjp, "args_res")[1] = None
    return (x, w_n), (scan_fn_vjp, aux_step)

  (x, w_last), (scan_fn_vjps, aux_scanned) = jax.lax.scan(body_fwd, carry, jax.tree.map(lambda _w: _w[2:], w))
  (x, aux_last), scan_fn_vjp_last = jax.vjp(scan_fn, x, w_last)
  getattr(scan_fn_vjp_last, "args_res")[1] = None
  aux = jax.tree.map(
      lambda first, scanned, last: jnp.concatenate([first[jnp.newaxis], scanned, last[jnp.newaxis]], axis=0),
      aux_first,
      aux_scanned,
      aux_last,
  )
  return (x, aux), (
      scan_fn_vjp_first,
      scan_fn_vjps,
      scan_fn_vjp_last,
      w,
      reduce_fn,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_bwd(
    _,
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    res: Any,
    grad_outputs: tuple[AnyJaxArray, Any],
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Backward pass of w_collect_pipelined_scan which re-collects weights."""
  (
      scan_fn_vjp_first,
      scan_fn_vjps,
      scan_fn_vjp_last,
      w,
      reduce_fn,
  ) = res

  x_grad, aux_grad = grad_outputs
  aux_step_last = jax.tree.map(lambda g: g[-1], aux_grad)
  step_grad_last = (x_grad, aux_step_last)

  w_last = collect_fn(jax.tree.map(lambda _w: _w[-1], w))
  getattr(scan_fn_vjp_last, "args_res")[1] = w_last
  x_grad, w_last_grad_unreduced = scan_fn_vjp_last(step_grad_last)

  w_second_last = collect_fn(jax.tree.map(lambda _w: _w[-2], w))
  carry = (x_grad, w_second_last, w_last_grad_unreduced)

  aux_scanned_grad = jax.tree.map(lambda g: g[1:-1], aux_grad)
  scan_xs = (
      scan_fn_vjps,
      jax.tree.map(lambda _w: _w[:-2], w),
      aux_scanned_grad,
  )

  def body_bwd(carry, xs):
    x_grad, w_n, w_prev_grad_unreduced = carry
    scan_fn_vjp, w_next_sharded, aux_step_grad = xs

    w_next = collect_fn(w_next_sharded)
    w_prev_grad_reduced = reduce_fn(w_prev_grad_unreduced)[0]

    getattr(scan_fn_vjp, "args_res")[1] = w_n
    x_grad, w_n_grad_unreduced = scan_fn_vjp((x_grad, aux_step_grad))
    return (x_grad, w_next, w_n_grad_unreduced), w_prev_grad_reduced

  (x_grad, w_first, w_second_grad_unreduced), w_grads_reduced = jax.lax.scan(
      body_bwd,
      carry,
      scan_xs,
      reverse=True,
  )

  w_second_grad_reduced = reduce_fn(w_second_grad_unreduced)[0]
  getattr(scan_fn_vjp_first, "args_res")[1] = w_first
  aux_step_first = jax.tree.map(lambda g: g[0], aux_grad)
  step_grad_first = (x_grad, aux_step_first)
  x_grad, w_first_grad_unreduced = scan_fn_vjp_first(step_grad_first)
  w_first_grad_reduced = reduce_fn(w_first_grad_unreduced)[0]
  # Likely better to initialize w_grad as a tree of zeros and insert into it
  # rather than concatenating at the end.
  w_grad = jax.tree.map(
      lambda x, y, z: jnp.concatenate([x[jnp.newaxis], y[jnp.newaxis], z], axis=0),
      w_first_grad_reduced,
      w_second_grad_reduced,
      w_grads_reduced,
  )
  return x_grad, w_grad


w_collect_pipelined_scan.defvjp(w_collect_pipelined_scan_fwd, w_collect_pipelined_scan_bwd)


def _combine_w_grads(w_mla: AnyJaxArrayOrPyTree, w_expert: AnyJaxArrayOrPyTree) -> AnyJaxArrayOrPyTree:
  """Combines MLA/norm/router gradients and expert gradients for a layer."""

  def _add(a, b):
    if a is None:
      return b
    if b is None:
      return a
    return a + b

  return jax.tree.map(_add, w_mla, w_expert, is_leaf=lambda x: x is None)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 5))
def w_collect_pipelined_scan_prologue_epilogue(
    prologue_fn: Callable[
        [AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
        Any,
    ],
    scan_body_fn: Callable[
        [Any, AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
        tuple[Any, AnyJaxArrayOrPyTree],
    ],
    epilogue_fn: Callable[
        [Any, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArrayOrPyTree,
    w: AnyJaxArrayOrPyTree,
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning with prologue and epilogue.

  Args:
    prologue_fn: Function to execute for the first layer with signature `(x:
      AnyJaxArray, w: AnyJaxArrayOrPyTree) -> carry`.
    scan_body_fn: Function to execute at each scan iteration with signature
      `(carry: Any, w_curr: AnyJaxArrayOrPyTree, w_next: AnyJaxArrayOrPyTree) ->
      tuple[carry, aux_step]`. Starts with permute and ends with mla, routing,
      and chunk metadata.
    epilogue_fn: Function to execute for the last layer with signature `(carry:
      Any, w: AnyJaxArrayOrPyTree) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]`.
      Runs the rest of the ubatch functions.
    x: Input array.
    w: Uncollected weights to scan over. Dimension 0 is the dimension to scan
      over and must be present (num_layers >= 2).
    collect_fn: Function to collect weights.

  Returns:
    A tuple of (output, aux) where aux is the stacked auxiliary outputs across
    all scan iterations.
  """
  w_first = collect_fn(jax.tree.map(lambda _w: _w[0], w))
  carry = prologue_fn(x, w_first)

  w_second = collect_fn(jax.tree.map(lambda _w: _w[1], w))
  carry, aux_second = scan_body_fn(carry, w_first, w_second)

  def body(carry_tuple, w_n_sharded):
    activations, w_curr = carry_tuple
    w_next = collect_fn(w_n_sharded)
    activations, aux_step = scan_body_fn(activations, w_curr, w_next)
    return (activations, w_next), aux_step

  (carry, w_last), aux_scanned = jax.lax.scan(body, (carry, w_second), jax.tree.map(lambda _w: _w[2:], w))
  out, aux_last = epilogue_fn(carry, w_last)
  aux = jax.tree.map(
      lambda second, scanned, last: jnp.concatenate([second[jnp.newaxis], scanned, last[jnp.newaxis]], axis=0),
      aux_second,
      aux_scanned,
      aux_last,
  )
  return out, aux


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_fwd(
    prologue_fn: Callable[
        [AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
        Any,
    ],
    scan_body_fn: Callable[
        [Any, AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
        tuple[Any, AnyJaxArrayOrPyTree],
    ],
    epilogue_fn: Callable[
        [Any, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArrayOrPyTree,
    w: AnyJaxArrayOrPyTree,
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan_prologue_epilogue."""
  assert jax.tree.leaves(w)[0].shape[0] >= 2, (
      "there must be at least two scan iterations to perform a pipelined scan" " and its transpose"
  )
  w_first, reduce_fn = jax.vjp(collect_fn, jax.tree.map(lambda _w: _w[0], w))
  getattr(reduce_fn, "args_res")[0] = None

  carry, prologue_vjp = jax.vjp(prologue_fn, x, w_first)
  getattr(prologue_vjp, "args_res")[1] = None

  w_second = collect_fn(jax.tree.map(lambda _w: _w[1], w))
  (carry, aux_second), scan_body_vjp_0 = jax.vjp(scan_body_fn, carry, w_first, w_second)
  getattr(scan_body_vjp_0, "args_res")[1] = None
  getattr(scan_body_vjp_0, "args_res")[2] = None

  def body_fwd(carry_tuple, w_n_sharded):
    activations, w_curr = carry_tuple
    w_next = collect_fn(w_n_sharded)
    (activations, aux_step), scan_body_vjp = jax.vjp(scan_body_fn, activations, w_curr, w_next)
    getattr(scan_body_vjp, "args_res")[1] = None
    getattr(scan_body_vjp, "args_res")[2] = None
    return (activations, w_next), (scan_body_vjp, aux_step)

  (carry, w_last), (scan_body_vjps, aux_scanned) = jax.lax.scan(
      body_fwd, (carry, w_second), jax.tree.map(lambda _w: _w[2:], w)
  )

  (out, aux_last), epilogue_vjp = jax.vjp(epilogue_fn, carry, w_last)
  getattr(epilogue_vjp, "args_res")[1] = None

  aux = jax.tree.map(
      lambda second, scanned, last: jnp.concatenate([second[jnp.newaxis], scanned, last[jnp.newaxis]], axis=0),
      aux_second,
      aux_scanned,
      aux_last,
  )
  return (out, aux), (
      prologue_vjp,
      scan_body_vjp_0,
      scan_body_vjps,
      epilogue_vjp,
      w,
      reduce_fn,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_bwd(
    prologue_fn: Callable[..., Any],
    scan_body_fn: Callable[..., Any],
    epilogue_fn: Callable[..., Any],
    collect_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    res: Any,
    grad_outputs: tuple[AnyJaxArrayOrPyTree, Any],
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Backward pass of w_collect_pipelined_scan_prologue_epilogue."""
  del prologue_fn, scan_body_fn, epilogue_fn
  (
      prologue_vjp,
      scan_body_vjp_0,
      scan_body_vjps,
      epilogue_vjp,
      w,
      reduce_fn,
  ) = res

  out_grad, aux_grad = grad_outputs
  aux_step_0 = jax.tree.map(lambda g: g[0], aux_grad)
  aux_scanned_grad = jax.tree.map(lambda g: g[1:-1], aux_grad)
  aux_last_grad = jax.tree.map(lambda g: g[-1], aux_grad)

  # 1. Epilogue backward with last layer weights
  w_last = collect_fn(jax.tree.map(lambda _w: _w[-1], w))
  getattr(epilogue_vjp, "args_res")[1] = w_last
  carry_grad, w_last_expert_grad_unreduced = epilogue_vjp((out_grad, aux_last_grad))

  # 2. Reverse scan body backward for iterations n-2 down to 1
  scan_xs = (
      scan_body_vjps,
      jax.tree.map(lambda _w: _w[1:-1], w),
      aux_scanned_grad,
  )

  def body_bwd(carry, xs):
    carry_grad, w_prev_expert_grad_unreduced, w_next = carry
    scan_body_vjp, w_curr_sharded, aux_step_grad = xs

    w_curr = collect_fn(w_curr_sharded)

    getattr(scan_body_vjp, "args_res")[1] = w_curr
    getattr(scan_body_vjp, "args_res")[2] = w_next

    (
        carry_grad,
        w_curr_expert_grad_unreduced,
        w_next_mla_grad_unreduced,
    ) = scan_body_vjp((carry_grad, aux_step_grad))

    w_layer_unreduced = _combine_w_grads(w_next_mla_grad_unreduced, w_prev_expert_grad_unreduced)
    w_layer_reduced = reduce_fn(w_layer_unreduced)[0]

    return (carry_grad, w_curr_expert_grad_unreduced, w_curr), w_layer_reduced

  (
      (carry_grad, w_second_expert_grad_unreduced, w_1),
      w_grads_reduced,
  ) = jax.lax.scan(
      body_bwd,
      (carry_grad, w_last_expert_grad_unreduced, w_last),
      scan_xs,
      reverse=True,
  )

  # 3. Scan body 0 backward with weights of layer 0 and 1
  w_0 = collect_fn(jax.tree.map(lambda _w: _w[0], w))
  getattr(scan_body_vjp_0, "args_res")[1] = w_0
  getattr(scan_body_vjp_0, "args_res")[2] = w_1
  (
      carry_grad,
      w_0_expert_grad_unreduced,
      w_1_mla_grad_unreduced,
  ) = scan_body_vjp_0((carry_grad, aux_step_0))

  w_1_unreduced = _combine_w_grads(w_1_mla_grad_unreduced, w_second_expert_grad_unreduced)
  w_second_grad_reduced = reduce_fn(w_1_unreduced)[0]

  # 4. Prologue backward with weights of layer 0
  getattr(prologue_vjp, "args_res")[1] = w_0
  x_grad, w_0_mla_grad_unreduced = prologue_vjp(carry_grad)

  w_0_unreduced = _combine_w_grads(w_0_mla_grad_unreduced, w_0_expert_grad_unreduced)
  w_first_grad_reduced = reduce_fn(w_0_unreduced)[0]

  w_grad = jax.tree.map(
      lambda x, y, z: jnp.concatenate([x[jnp.newaxis], y[jnp.newaxis], z], axis=0),
      w_first_grad_reduced,
      w_second_grad_reduced,
      w_grads_reduced,
  )
  return x_grad, w_grad


w_collect_pipelined_scan_prologue_epilogue.defvjp(
    w_collect_pipelined_scan_prologue_epilogue_fwd,
    w_collect_pipelined_scan_prologue_epilogue_bwd,
)


def _ragged_silu_mul_kernel(
    x_ref: Any,
    y_ref: Any,
    out_ref: Any,
) -> None:
  """Kernel body for ragged_silu_mul."""
  out_ref[...] = jax.nn.silu(x_ref[...]) * y_ref[...]


def _pallas_ragged_silu_mul(
    x: jt.Num[jax.Array, "max_num_tokens double_hidden_dim"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
    block_size_hidden: int,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Pallas call wrapper for ragged_silu_mul forward pass."""
  max_num_tokens, double_hidden_dim = x.shape
  hidden_dim = double_hidden_dim // 2
  block_size_hidden = min(block_size_hidden, hidden_dim)
  grid = (
      (num_tokens + block_size_tokens - 1) // block_size_tokens,
      (hidden_dim + block_size_hidden - 1) // block_size_hidden,
  )
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j),
      ),
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j + num_hidden_blocks),
      ),
  ]
  out_spec = pl.BlockSpec(
      block_shape=(block_size_tokens, block_size_hidden),
      index_map=lambda i, j: (i, j),
  )
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_silu_mul_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, hidden_dim),
          x.dtype,
          manual_axis_type=manual_axis_type,
      ),
      in_specs=in_specs,
      out_specs=out_spec,
      grid=grid,
  )(x, x)


def _ragged_silu_mul_bwd_kernel(
    g_ref: Any,
    x_ref: Any,
    y_ref: Any,
    dx_ref: Any,
    dy_ref: Any,
) -> None:
  """Kernel body for ragged_silu_mul backward pass."""
  x_val = x_ref[...]
  y_val = y_ref[...]
  g_val = g_ref[...]

  sig_x = jax.nn.sigmoid(x_val)
  silu_x = x_val * sig_x
  dsilu_x = sig_x * (1.0 + x_val * (1.0 - sig_x))

  dx_ref[...] = g_val * y_val * dsilu_x
  dy_ref[...] = g_val * silu_x


def _pallas_ragged_dx_dy_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"],
    x: jt.Num[jax.Array, "max_num_tokens double_hidden_dim"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
    block_size_hidden: int,
) -> tuple[
    jt.Num[jax.Array, "max_num_tokens hidden_dim"],
    jt.Num[jax.Array, "max_num_tokens hidden_dim"],
]:
  """Pallas call wrapper for computing dx and dy gradients in backward pass."""
  max_num_tokens, double_hidden_dim = x.shape
  hidden_dim = double_hidden_dim // 2
  block_size_hidden = min(block_size_hidden, hidden_dim)
  grid = (
      (num_tokens + block_size_tokens - 1) // block_size_tokens,
      (hidden_dim + block_size_hidden - 1) // block_size_hidden,
  )
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j),
      ),
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j),
      ),
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j + num_hidden_blocks),
      ),
  ]
  out_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j),
      ),
      pl.BlockSpec(
          block_shape=(block_size_tokens, block_size_hidden),
          index_map=lambda i, j: (i, j),
      ),
  ]
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_silu_mul_bwd_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              (max_num_tokens, hidden_dim),
              x.dtype,
              manual_axis_type=manual_axis_type,
          ),
          jax.ShapeDtypeStruct(
              (max_num_tokens, hidden_dim),
              x.dtype,
              manual_axis_type=manual_axis_type,
          ),
      ),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
  )(g, x, x)


def _pallas_ragged_silu_mul_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"],
    x: jt.Num[jax.Array, "max_num_tokens double_hidden_dim"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens_bwd: int,
    block_size_hidden_bwd: int,
) -> jt.Num[jax.Array, "max_num_tokens double_hidden_dim"]:
  """Computes backward pass gradient w.r.t. x using Pallas."""
  dx, dy = _pallas_ragged_dx_dy_bwd(g, x, num_tokens, block_size_tokens_bwd, block_size_hidden_bwd)
  return jnp.concatenate([dx, dy], axis=-1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _ragged_silu_mul_vjp(
    x: jt.Num[jax.Array, "max_num_tokens double_hidden_dim"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens_fwd: int,
    block_size_hidden_fwd: int,
    block_size_tokens_bwd: int,
    block_size_hidden_bwd: int,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Custom VJP wrapper function for ragged_silu_mul."""
  return _ragged_silu_mul_fwd(
      x,
      num_tokens,
      block_size_tokens_fwd,
      block_size_hidden_fwd,
      block_size_tokens_bwd,
      block_size_hidden_bwd,
  )[0]


def _ragged_silu_mul_fwd(
    x,
    num_tokens,
    block_size_tokens_fwd,
    block_size_hidden_fwd,
    block_size_tokens_bwd,
    block_size_hidden_bwd,
):
  """Forward pass for custom VJP of ragged_silu_mul."""
  del block_size_tokens_bwd, block_size_hidden_bwd
  out = _pallas_ragged_silu_mul(x, num_tokens, block_size_tokens_fwd, block_size_hidden_fwd)
  return out, (x, num_tokens)


def _ragged_silu_mul_bwd(
    block_size_tokens_fwd,
    block_size_hidden_fwd,
    block_size_tokens_bwd,
    block_size_hidden_bwd,
    res,
    g,
):
  """Backward pass for custom VJP of ragged_silu_mul."""
  del block_size_tokens_fwd, block_size_hidden_fwd
  x, num_tokens = res
  dx = _pallas_ragged_silu_mul_bwd(
      g,
      x,
      num_tokens,
      block_size_tokens_bwd,
      block_size_hidden_bwd,
  )
  return (dx, None)


_ragged_silu_mul_vjp.defvjp(_ragged_silu_mul_fwd, _ragged_silu_mul_bwd)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_silu_mul(
    x: jt.Num[jax.Array, "max_num_tokens double_hidden_dim"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens_fwd: int = 128,
    block_size_hidden_fwd: int = 128,
    block_size_tokens_bwd: int = 128,
    block_size_hidden_bwd: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Computes ragged silu(g0) * g1 using a Pallas kernel, where x is [g0, g1].

  Args:
    x: 2D input array of shape (max_num_tokens, 2 * hidden_dim).
    num_tokens: Number of valid tokens to process. Leftover tokens in the output
      buffer are left uninitialized/unmasked for performance.
    block_size_tokens_fwd: Tile size along token dim for fwd (default 128).
    block_size_hidden_fwd: Tile size along hidden dim for fwd (default 128).
    block_size_tokens_bwd: Tile size along token dim for bwd dx/dy (default
      128).
    block_size_hidden_bwd: Tile size along hidden dim for bwd dx/dy (default
      128).

  Returns:
    Result array of shape (max_num_tokens, hidden_dim).
  """
  return _ragged_silu_mul_vjp(
      x,
      num_tokens,
      block_size_tokens_fwd,
      block_size_hidden_fwd,
      block_size_tokens_bwd,
      block_size_hidden_bwd,
  )


# --- Pallas Kernels for Ragged Flatten / Unflatten ---


def _ragged_flatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for flattening [X, D0, D1] -> [X, D]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, (x_val.shape[0], -1))


def _pallas_ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Pallas call wrapper for flattening [X, D0, D1] -> [X, D]."""
  max_num_tokens, d0, d1 = x.shape
  d = d0 * d1
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, d0, d1),
          index_map=lambda i: (i, 0, 0),
      ),
  ]
  out_spec = pl.BlockSpec(
      block_shape=(block_size_tokens, d),
      index_map=lambda i: (i, 0),
  )
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_flatten_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, d),
          x.dtype,
          manual_axis_type=manual_axis_type,
      ),
      in_specs=in_specs,
      out_specs=out_spec,
      grid=grid,
  )(x)


def _ragged_unflatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for unflattening [X, D] -> [X, D0, D1]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, out_ref.shape)


def _pallas_ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Pallas call wrapper for unflattening [X, D] -> [X, D0, D1]."""
  max_num_tokens, d = x.shape
  d0, d1 = shape
  assert d == d0 * d1, f"Shape mismatch: {d} != {d0} * {d1}"
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, d),
          index_map=lambda i: (i, 0),
      ),
  ]
  out_spec = pl.BlockSpec(
      block_shape=(block_size_tokens, d0, d1),
      index_map=lambda i: (i, 0, 0),
  )
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_unflatten_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, d0, d1),
          x.dtype,
          manual_axis_type=manual_axis_type,
      ),
      in_specs=in_specs,
      out_specs=out_spec,
      grid=grid,
  )(x)


# --- Custom VJPs (Dual of each other) ---


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _ragged_flatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  return _ragged_flatten_fwd(x, num_tokens, block_size_tokens)[0]


def _ragged_flatten_fwd(x, num_tokens, block_size_tokens):
  out = _pallas_ragged_flatten(x, num_tokens, block_size_tokens)
  return out, (num_tokens, x.shape[1:])


def _ragged_flatten_bwd(block_size_tokens, res, g):
  num_tokens, shape = res
  # Dual of Flatten backward pass is Unflatten forward pass!
  dx = _pallas_ragged_unflatten(g, shape, num_tokens, block_size_tokens)
  return dx, None


_ragged_flatten_vjp.defvjp(_ragged_flatten_fwd, _ragged_flatten_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 3))
def _ragged_unflatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  return _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens)[0]


def _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens):
  out = _pallas_ragged_unflatten(x, shape, num_tokens, block_size_tokens)
  return out, (num_tokens,)


def _ragged_unflatten_bwd(shape, block_size_tokens, res, g):
  del shape
  (num_tokens,) = res
  # Dual of Unflatten backward pass is Flatten forward pass!
  dx = _pallas_ragged_flatten(g, num_tokens, block_size_tokens)
  return dx, None


_ragged_unflatten_vjp.defvjp(_ragged_unflatten_fwd, _ragged_unflatten_bwd)


# --- Public APIs ---


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Flattens ragged tensor from [X, D0, D1] to [X, D] for the first num_tokens using Pallas.

  The dual backward pass (custom VJP) uses ragged_unflatten.

  Args:
    x: 3D input array of shape (max_num_tokens, d0, d1).
    num_tokens: Number of valid tokens to process. Leftover tokens in the output
      buffer are left uninitialized/unmasked for performance.
    block_size_tokens: Tile size along token dim (default 128).

  Returns:
    Reshaped 2D array of shape (max_num_tokens, d0 * d1).
  """
  return _ragged_flatten_vjp(x, num_tokens, block_size_tokens)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Unflattens ragged tensor from [X, D] to [X, D0, D1] for the first num_tokens using Pallas.

  The dual backward pass (custom VJP) uses ragged_flatten.

  Args:
    x: 2D input array of shape (max_num_tokens, D).
    shape: Target (D0, D1) shape such that D0 * D1 == D.
    num_tokens: Number of valid tokens to process. Leftover tokens in the output
      buffer are left uninitialized/unmasked for performance.
    block_size_tokens: Tile size along token dim (default 128).

  Returns:
    Reshaped 3D array of shape (max_num_tokens, D0, D1).
  """
  return _ragged_unflatten_vjp(x, shape, num_tokens, block_size_tokens)


# --- Pallas Kernels for Ragged Write ---


def _ragged_write_kernel(
    dst_off_ref: Any,
    src_off_ref: Any,
    dst_in_ref: Any,
    src_ref: Any,
    dst_out_ref: Any,
) -> None:
  """Kernel body for ragged_write."""
  del dst_off_ref, src_off_ref, dst_in_ref
  dst_out_ref[...] = src_ref[...]


def _pallas_ragged_write(
    dst: AnyJaxArray,
    src: AnyJaxArray,
    dst_offset: jt.Num[jax.Array, ""] | int,
    src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int,
    axis: int,
) -> AnyJaxArray:
  """Pallas call wrapper for ragged_write."""
  src_block_shape = list(src.shape)
  src_block_shape[axis] = 1

  dst_block_shape = list(dst.shape)
  dst_block_shape[axis] = 1

  def src_index_map(slice_idx, dst_off_ref, src_off_ref):
    del dst_off_ref
    idx = [0] * src.ndim
    idx[axis] = src_off_ref[()] + slice_idx
    return tuple(idx)

  def dst_index_map(slice_idx, dst_off_ref, src_off_ref):
    del src_off_ref
    idx = [0] * dst.ndim
    idx[axis] = dst_off_ref[()] + slice_idx
    return tuple(idx)

  return pl.pallas_call(
      _ragged_write_kernel,
      out_shape=jax.ShapeDtypeStruct(dst.shape, dst.dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec(tuple(dst_block_shape), dst_index_map),
              pl.BlockSpec(tuple(src_block_shape), src_index_map),
          ],
          out_specs=pl.BlockSpec(tuple(dst_block_shape), dst_index_map),
          grid=(slice_size,),
      ),
      input_output_aliases={2: 0},
  )(dst_offset, src_offset, dst, src)


@jt.jaxtyped(typechecker=typeguard.typechecked)
@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("axis",))
def ragged_write(
    dst: AnyJaxArray,
    src: AnyJaxArray,
    dst_offset: jt.Num[jax.Array, ""] | int,
    src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int,
    *,
    axis: int = -1,
) -> AnyJaxArray:
  """Updates a dynamic slice of an array with a dynamic slice from a source array.

  Conceptually, this operation is equivalent to:
      ```python
      dst_idx = [slice(None)] * dst.ndim
      dst_idx[axis] = slice(dst_offset, dst_offset + slice_size)
      src_idx = [slice(None)] * src.ndim
      src_idx[axis] = slice(src_offset, src_offset + slice_size)

      dst_out = jnp.copy(dst)
      dst_out[tuple(dst_idx)] = src[tuple(src_idx)]
      return dst_out
      ```

  This function safely handles dynamic variables (JAX Tracers) for `dst_offset`,
  `src_offset`, and `slice_size` within JIT-compiled functions.
  It leverages dynamic grid scheduling on TPU. If the requested write operation
  is
  out-of-bounds or `slice_size <= 0`, it gracefully acts as a no-op and returns

  Args:
      dst: Destination array to be updated.
      src: Source array supplying the update data.
      dst_offset: Integer scalar identifying the starting index along `axis` in
        `dst`.
      src_offset: Integer scalar identifying the starting index along `axis` in
        `src`.
      slice_size: Integer scalar indicating the number of elements to copy along
        `axis`.
      axis: The dimension along which the ragged write is applied. Defaults to
        -1.

  Returns:
      An array with identical shape, dtype, and values since we are doing an
      in-place update.
  """
  axis = axis % dst.ndim

  # 1. OOB Checking (No-op if any element is OOB or slice is invalid)
  is_oob = (
      (slice_size <= 0)
      | (dst_offset < 0)
      | (dst_offset + slice_size > dst.shape[axis])
      | (src_offset < 0)
      | (src_offset + slice_size > src.shape[axis])
  )

  def _do_write(dst_operand):
    return _pallas_ragged_write(dst_operand, src, dst_offset, src_offset, slice_size, axis)

  # Conditionally execute the Pallas op to gracefully handle OOB
  return jax.lax.cond(is_oob, lambda x: x, _do_write, dst)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def split_microbatches(
    x: AnyJaxArrayOrPyTree,
    *,
    mesh: jax.sharding.Mesh,
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Splits the batch into two micro-batches locally in a shard map.

  Assumes the first dimension is the batch dimension and splits along it.
  Does not split any other dimensions, including the sequence dimension,
  which should already be split when pdbs < 1.

  Args:
    x: Input array or PyTree of arrays.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of two PyTrees with the same structure as the input, each containing
    half of the data along the first dimension.
  """
  if x is None:
    return None, None

  def _split_leaf(arr):
    s = jax.typeof(arr).sharding
    arr_pspec = getattr(s, "spec", jax.sharding.PartitionSpec())

    def _local_split(local_arr):
      return tuple(jnp.split(local_arr, 2, axis=0))

    return jax.shard_map(
        _local_split,
        mesh=mesh,
        in_specs=arr_pspec,
        out_specs=(arr_pspec, arr_pspec),
    )(arr)

  leaves, treedef = jax.tree.flatten(x)
  splits = [_split_leaf(leaf) for leaf in leaves]
  mb0 = jax.tree.unflatten(treedef, [s[0] for s in splits])
  mb1 = jax.tree.unflatten(treedef, [s[1] for s in splits])
  return mb0, mb1


@jt.jaxtyped(typechecker=typeguard.typechecked)
def merge_microbatches(
    mb0: AnyJaxArrayOrPyTree,
    mb1: AnyJaxArrayOrPyTree,
    *,
    mesh: jax.sharding.Mesh,
) -> AnyJaxArrayOrPyTree:
  """Merges two micro-batches back into a single batch in a shard map."""
  if mb0 is None:
    return None

  def _merge_leaf(arr0, arr1):
    s0 = jax.typeof(arr0).sharding
    s1 = jax.typeof(arr1).sharding
    spec0 = getattr(s0, "spec", jax.sharding.PartitionSpec())
    spec1 = getattr(s1, "spec", jax.sharding.PartitionSpec())
    arr_pspec = spec0 if spec0 is not None else spec1

    def _local_merge(local_arr0, local_arr1):
      return jnp.concatenate([local_arr0, local_arr1], axis=0)

    return jax.shard_map(
        _local_merge,
        mesh=mesh,
        in_specs=(arr_pspec, arr_pspec),
        out_specs=arr_pspec,
    )(arr0, arr1)

  return jax.tree.map(_merge_leaf, mb0, mb1)
