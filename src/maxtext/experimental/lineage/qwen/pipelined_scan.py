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

"""Lineage-pattern FSDP pipelined scan & SC1/SC0 weight collect/reduce for Qwen."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, NamedTuple

import jax
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.experimental.lineage.qwen import config as config_lib
from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen import params as params_lib

QwenConfig = config_lib.QwenConfig


class PipelinedScanLayer(NamedTuple):
  fwd: Callable[..., tuple[Any, Any]]
  bwd: Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class DeferredReduce:
  """An EP-axis reduction deferred to run inside an SC0 `shard_map`/`compute_on` block."""

  operand: list[jax.Array]
  local_fn: Callable[[list[jax.Array]], list[jax.Array]]
  finish: Callable[[list[jax.Array]], Any]


def combine_w_grads(w_attn: Any, w_expert: Any) -> Any:
  """Combines attention/GDN gradients and MoE gradients for a superblock or sublayer."""

  def _add(a, b):
    if a is None:
      return b
    if b is None:
      return a
    return a + b

  return jax.tree.map(_add, w_attn, w_expert, is_leaf=lambda x: x is None)


def _interleave_arrays(a0: Any, a1: Any) -> Any:
  if a0 is None or a1 is None:
    return None
  stacked = jnp.stack([a0, a1], axis=1)
  new_shape = (a0.shape[0] * 2,) + a0.shape[1:]
  return stacked.reshape(new_shape)


def _layer_slice(tree: Any, idx: int | slice) -> Any:
  if tree is None:
    return None
  return jax.tree.map(lambda t: t[idx], tree)


def _layer_at(tree: Any, idx: int | jax.Array | None) -> Any:
  if tree is None or idx is None:
    return None
  if isinstance(idx, int):
    return _layer_slice(tree, idx)
  return jax.tree.map(
      lambda t: jax.lax.dynamic_index_in_dim(t, idx, keepdims=False), tree
  )


def _w_aux_kwargs(w_aux_layer: Any) -> dict[str, Any]:
  return {} if w_aux_layer is None else {"w_aux": w_aux_layer}


class _LayerCollect(NamedTuple):
  collect: Callable[[Any, Any], Any]
  pre: Callable[[Any, Any], Any]
  collect_ici: Callable[[Any, Any], Any]
  pre_carry: Callable[[int | jax.Array | None], Any]


def _layer_collect(
    collect_fn_dcn: Callable[..., Any],
    collect_fn_ici: Callable[..., Any],
    collect_fn_pre: Callable[..., Any] | None,
    w: Any,
    w_aux: Any,
) -> _LayerCollect:
  def pre(w_dcn, w_aux_layer):
    if collect_fn_pre is None:
      return w_dcn
    return collect_fn_pre(w_dcn, **_w_aux_kwargs(w_aux_layer))

  def collect_ici(w_pre, w_aux_layer):
    return collect_fn_ici(w_pre, **_w_aux_kwargs(w_aux_layer))

  def collect(w_layer, w_aux_layer):
    return collect_ici(pre(collect_fn_dcn(w_layer), w_aux_layer), w_aux_layer)

  def pre_carry(idx):
    if collect_fn_pre is None or idx is None:
      return None
    w_aux_layer = _layer_at(w_aux, idx)
    return pre(collect_fn_dcn(_layer_at(w, idx)), w_aux_layer), w_aux_layer

  return _LayerCollect(
      collect=collect, pre=pre, collect_ici=collect_ici, pre_carry=pre_carry
  )


def _shifted_indices(layers: slice | int, shift: int, num_layers: int) -> Any:
  if isinstance(layers, slice):
    idx = jnp.arange(layers.start, layers.stop, layers.step) + shift
    return jnp.clip(idx, 0, num_layers - 1)
  idx = layers + shift
  return idx if 0 <= idx < num_layers else None


def _fwd_step_xs(
    w: Any, w_aux: Any, layers: slice | int, shifted: bool
) -> tuple[Any, Any, Any]:
  if shifted:
    num_layers = jax.tree.leaves(w)[0].shape[0]
    return None, None, _shifted_indices(layers, 1, num_layers)
  return _layer_slice(w, layers), _layer_slice(w_aux, layers), None


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16),
)
def w_collect_pipelined_scan_prologue_epilogue(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    x: Any,
    w: Any,
    collect_fn_dcn: Callable[[Any], Any],
    collect_fn_ici_fwd: Callable[..., Any],
    reduce_fn_dcn: Callable[[Any, Any], Any],
    reduce_fn_ici_first: Callable[..., Any],
    reduce_fn_ici_second: Callable[..., Any],
    collect_fn_ici_bwd: Callable[..., Any] | None = None,
    quantize_fn: Callable[[Any], Any] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., Any] | None = None,
    reduce_fn_pre: Callable[..., Any] | None = None,
) -> tuple[Any, Any]:
  """Lineage pipelined scan with peeled prologue/epilogue (`num_layers >= 3`)."""
  del offload_fn, load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second, reduce_fn_pre
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan with a"
      f" peeled prologue and epilogue, got {num_layers}"
  )
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(
      collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux
  )
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)

  w_aux_0 = _layer_slice(w_aux, 0)
  w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
  carry = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))[0]

  w_aux_1 = _layer_slice(w_aux, 1)
  w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
  pre_carry = fns.pre_carry(2 if shifted else None)
  carry, aux_first = scan_body_fn.fwd(
      carry, w_first, w_second, **_w_aux_kwargs(w_aux_0)
  )[0]

  def _scan_step(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry = carry_tuple
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    activations, aux_step = scan_body_fn.fwd(
        activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr)
    )[0]
    return (activations, w_next, w_aux_n, pre_carry), aux_step

  def body(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, aux_0 = _scan_step(carry_tuple, xs_0)
    (carry_tuple, aux_0), xs_1 = jax.lax.optimization_barrier(
        ((carry_tuple, aux_0), xs_1)
    )
    carry_tuple, aux_1 = _scan_step(carry_tuple, xs_1)
    return carry_tuple, (aux_0, aux_1)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)

  (carry, w_curr, w_aux_curr, pre_carry), (aux_0, aux_1) = jax.lax.scan(
      body,
      (carry, w_second, w_aux_1, pre_carry),
      (step_xs(even), step_xs(odd)),
  )
  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry), aux_rem = _scan_step(
        (carry, w_curr, w_aux_curr, pre_carry), step_xs(rem)
    )
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))

  (carry, w_last, w_aux_last, _), aux_last_sb = _scan_step(
      (carry, w_curr, w_aux_curr, pre_carry), step_xs(num_layers - 1)
  )
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))

  out, aux_last = epilogue_fn.fwd(carry, w_last, **_w_aux_kwargs(w_aux_last))[0]
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))

  if aux_parts[0] is None:
    return out, None
  aux = jax.tree.map(
      lambda *parts: jnp.concatenate(parts, axis=0),
      *aux_parts,
  )
  return out, aux


def _is_pinned_host(x: Any) -> bool:
  sharding = getattr(jax.typeof(x), "sharding", None)
  return getattr(sharding, "memory_kind", None) == "pinned_host"


def w_collect_pipelined_scan_prologue_epilogue_fwd(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    x: Any,
    w: Any,
    collect_fn_dcn: Callable[[Any], Any],
    collect_fn_ici_fwd: Callable[..., Any],
    reduce_fn_dcn: Callable[[Any, Any], Any],
    reduce_fn_ici_first: Callable[..., Any],
    reduce_fn_ici_second: Callable[..., Any],
    collect_fn_ici_bwd: Callable[..., Any] | None = None,
    quantize_fn: Callable[[Any], Any] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., Any] | None = None,
    reduce_fn_pre: Callable[..., Any] | None = None,
) -> tuple[tuple[Any, Any], Any]:
  """Forward pass of `w_collect_pipelined_scan_prologue_epilogue`."""
  del load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second, reduce_fn_pre
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan with a"
      f" peeled prologue and epilogue, got {num_layers}"
  )
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(
      collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux
  )
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)

  def _split_offload(res):
    res_offloaded = offload_fn(res)
    res_host = jax.tree.map(
        lambda r: r if _is_pinned_host(r) else None, res_offloaded
    )
    res_device = jax.tree.map(
        lambda r, h: None if h is not None else r,
        res,
        res_host,
        is_leaf=lambda x: x is None,
    )
    return res_device, res_host

  w_aux_0 = _layer_slice(w_aux, 0)
  w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
  carry, res_prologue = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))
  res_prologue_device, res_prologue_host = _split_offload(res_prologue)
  del res_prologue_device  # Prologue residuals stay on device if not offloaded.
  if jax.tree.leaves(res_prologue_host):
    pass  # Keep res_prologue_host if pinned_host used.
  else:
    res_prologue_host = offload_fn(res_prologue)

  w_aux_1 = _layer_slice(w_aux, 1)
  w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
  pre_carry = fns.pre_carry(2 if shifted else None)
  (carry, aux_first), res_sb_0 = scan_body_fn.fwd(
      carry, w_first, w_second, **_w_aux_kwargs(w_aux_0)
  )
  res_sb_0_device, res_sb_0_host = _split_offload(res_sb_0)

  def _scan_step(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry = carry_tuple
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    (activations, aux_step), res_step = scan_body_fn.fwd(
        activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr)
    )
    res_step_device, res_step_host = _split_offload(res_step)
    return (activations, w_next, w_aux_n, pre_carry), (
        res_step_device,
        res_step_host,
        aux_step,
    )

  def body(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, (res_0_device, res_0_host, aux_0) = _scan_step(
        carry_tuple, xs_0
    )
    (carry_tuple, res_0_device, res_0_host, aux_0), xs_1 = (
        jax.lax.optimization_barrier(
            ((carry_tuple, res_0_device, res_0_host, aux_0), xs_1)
        )
    )
    carry_tuple, (res_1_device, res_1_host, aux_1) = _scan_step(
        carry_tuple, xs_1
    )
    return carry_tuple, (
        (res_0_device, res_1_device),
        (res_0_host, res_1_host),
        (aux_0, aux_1),
    )

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)

  (
      (carry, w_curr, w_aux_curr, pre_carry),
      (
          (res_0_scanned_device, res_1_scanned_device),
          (res_0_scanned_host, res_1_scanned_host),
          (aux_0, aux_1),
      ),
  ) = jax.lax.scan(
      body,
      (carry, w_second, w_aux_1, pre_carry),
      (step_xs(even), step_xs(odd)),
  )
  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)

  res_rem_device, res_rem_host = None, None
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry), (
        res_rem_device,
        res_rem_host,
        aux_rem,
    ) = _scan_step((carry, w_curr, w_aux_curr, pre_carry), step_xs(rem))
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))

  (carry, w_last, w_aux_last, _), (
      res_last_sb_device,
      res_last_sb_host,
      aux_last_sb,
  ) = _scan_step(
      (carry, w_curr, w_aux_curr, pre_carry), step_xs(num_layers - 1)
  )
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))

  (out, aux_last), res_epilogue = epilogue_fn.fwd(
      carry, w_last, **_w_aux_kwargs(w_aux_last)
  )
  res_epilogue_host = offload_fn(res_epilogue)
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))

  if aux_parts[0] is None:
    aux = None
  else:
    aux = jax.tree.map(
        lambda *parts: jnp.concatenate(parts, axis=0),
        *aux_parts,
    )

  if num_scanned % 2 == 1:
    res_penultimate_sb_host = res_rem_host
    if num_pairs > 0:
      res_rem_prev_host = _layer_slice(res_1_scanned_host, -1)
    else:
      res_rem_prev_host = res_sb_0_host
  elif num_pairs > 0:
    res_penultimate_sb_host = _layer_slice(res_1_scanned_host, -1)
    res_rem_prev_host = None
  else:
    res_penultimate_sb_host = res_sb_0_host
    res_rem_prev_host = None

  if num_pairs > 0:
    res_prev_1_scanned_host = _layer_slice(res_0_scanned_host, slice(None))
    res_prev_0_scanned_host = jax.tree.map(
        lambda x, y: jnp.concatenate([x[jnp.newaxis], y[:-1]], axis=0),
        res_sb_0_host,
        res_1_scanned_host,
    )
  else:
    res_prev_0_scanned_host = None
    res_prev_1_scanned_host = None

  return (out, aux), (
      res_prologue_host,
      res_sb_0_host,
      res_sb_0_device,
      res_0_scanned_device,
      res_1_scanned_device,
      res_prev_0_scanned_host,
      res_prev_1_scanned_host,
      res_rem_device,
      res_rem_host,
      res_rem_prev_host,
      res_penultimate_sb_host,
      res_last_sb_host,
      res_last_sb_device,
      res_epilogue_host,
      w,
      w_aux,
  )


def w_collect_pipelined_scan_prologue_epilogue_bwd(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    collect_fn_dcn: Callable[[Any], Any],
    collect_fn_ici_fwd: Callable[..., Any],
    reduce_fn_dcn: Callable[[Any, Any], Any],
    reduce_fn_ici_first: Callable[..., Any],
    reduce_fn_ici_second: Callable[..., Any],
    collect_fn_ici_bwd: Callable[..., Any] | None,
    quantize_fn: Callable[[Any], Any] | None,
    offload_fn: Callable[[Any], Any],
    load_fn: Callable[[Any], Any],
    collect_fn_pre: Callable[..., Any] | None,
    reduce_fn_pre: Callable[..., Any] | None,
    res: Any,
    grad_outputs: tuple[Any, Any],
) -> tuple[Any, Any]:
  """Backward pass of `w_collect_pipelined_scan_prologue_epilogue`."""
  del offload_fn, quantize_fn
  if collect_fn_ici_bwd is None:
    collect_fn_ici_bwd = collect_fn_ici_fwd
  (
      res_prologue_host,
      res_sb_0_host,
      res_sb_0_device,
      res_curr_0_scanned_device,
      res_curr_1_scanned_device,
      res_prev_0_scanned_host,
      res_prev_1_scanned_host,
      res_rem_device,
      res_rem_host,
      res_rem_prev_host,
      res_penultimate_sb_host,
      res_last_sb_host,
      res_last_sb_device,
      res_epilogue_host,
      w,
      w_aux,
  ) = res
  num_layers = jax.tree.leaves(w)[0].shape[0]
  shifted = collect_fn_pre is not None
  fns = _layer_collect(
      collect_fn_dcn, collect_fn_ici_bwd, collect_fn_pre, w, w_aux
  )

  def _load(res_host):
    return None if res_host is None else load_fn(res_host)

  def _merge_offloaded(res_loaded, res_device):
    if res_loaded is None:
      return res_device
    return jax.tree.map(
        lambda l, d: d if l is None else l,
        res_loaded,
        res_device,
        is_leaf=lambda x: x is None,
    )

  out_grad, aux_grad = grad_outputs
  aux_step_0 = _layer_slice(aux_grad, 0)
  aux_scanned_grad = _layer_slice(aux_grad, slice(1, -2))
  aux_last_sb_grad = _layer_slice(aux_grad, -2)
  aux_last_grad = _layer_slice(aux_grad, -1)

  res_epilogue_device = load_fn(res_epilogue_host)
  w_last_sharded = _layer_slice(w, -1)
  w_last_dcn = collect_fn_dcn(w_last_sharded)
  w_aux_last = _layer_slice(w_aux, -1)
  w_last = fns.collect_ici(fns.pre(w_last_dcn, w_aux_last), w_aux_last)

  def _reduce_ici_first(w_unreduced, w_aux_layer):
    return reduce_fn_ici_first(
        w_unreduced, w_last_dcn, **_w_aux_kwargs(w_aux_layer)
    )

  def _reduce_ici_second(w_partial, w_aux_layer, **kwargs):
    return reduce_fn_ici_second(
        w_partial, w_last_dcn, **_w_aux_kwargs(w_aux_layer), **kwargs
    )

  def _reduce_pre(w_partial, w_aux_layer):
    if reduce_fn_pre is None:
      return w_partial
    return reduce_fn_pre(w_partial, **_w_aux_kwargs(w_aux_layer))

  def _reduce_dcn(w_ici):
    return reduce_fn_dcn(w_ici, w_last_sharded)

  with jax.named_scope("epilogue_bwd"):
    res_last_sb_loaded = _load(res_last_sb_host)
    w_aux_penultimate = _layer_slice(w_aux, -2)
    w_curr = fns.collect(_layer_slice(w, -2), w_aux_penultimate)
    pre_carry = fns.pre_carry(num_layers - 3 if shifted else None)
    carry_grad, w_last_expert_grad_unreduced = epilogue_fn.bwd(
        res_epilogue_device,
        (out_grad, aux_last_grad),
        w_last,
        **_w_aux_kwargs(w_aux_last),
    )

  def _scan_step_bwd(carry, xs):
    (
        carry_grad,
        w_prev_expert_grad_unreduced,
        w_curr,
        w_next,
        w_aux_curr,
        w_aux_next,
        res_curr_loaded,
        w_partial_pending,
        pre_carry,
    ) = carry
    (
        res_curr_device,
        res_prev_host,
        w_prev_sharded,
        w_aux_prev,
        aux_step_grad,
        pre_idx,
    ) = xs

    w_partial_pending_grad, w_aux_partial_pending = (
        (None, None) if w_partial_pending is None else w_partial_pending
    )
    if w_partial_pending is not None:
      w_partial_pending_grad = _reduce_pre(
          w_partial_pending_grad, w_aux_partial_pending
      )

    res_prev_loaded = _load(res_prev_host)
    res_curr_full = _merge_offloaded(res_curr_loaded, res_curr_device)

    if shifted:
      w_prev_pre, w_aux_prev = pre_carry
    else:
      w_prev_pre = collect_fn_dcn(w_prev_sharded)
    w_prev_collected = fns.collect_ici(w_prev_pre, w_aux_prev)
    pre_carry = fns.pre_carry(pre_idx)

    def _reduce_ici_first_with_collect(w_unreduced):
      w_partial = _reduce_ici_first(w_unreduced, w_aux_next)
      w_partial, _ = jax.lax.optimization_barrier((w_partial, w_prev_collected))
      return w_partial

    (
        carry_grad,
        w_curr_expert_grad_unreduced,
        w_next_grad_partial,
        w_partial_pending_grad_ici,
    ) = scan_body_fn.bwd(
        res_curr_full,
        (carry_grad, aux_step_grad),
        w_curr,
        w_next,
        w_prev_expert_grad_unreduced,
        _reduce_ici_first_with_collect,
        w_partial_pending_grad,
        functools.partial(
            _reduce_ici_second, w_aux_layer=w_aux_partial_pending
        ),
        **_w_aux_kwargs(w_aux_curr),
    )
    w_pending_reduced = (
        None
        if w_partial_pending is None
        else _reduce_dcn(w_partial_pending_grad_ici)
    )

    return (
        carry_grad,
        w_curr_expert_grad_unreduced,
        w_prev_collected,
        w_curr,
        w_aux_prev,
        w_aux_curr,
        res_prev_loaded,
        (w_next_grad_partial, w_aux_next),
        pre_carry,
    ), w_pending_reduced

  def _step_xs(res_curr_device, res_prev_host, layers, aux_step_grad):
    if shifted:
      pre_idx = _shifted_indices(layers, -1, num_layers)
      return (
          res_curr_device,
          res_prev_host,
          None,
          None,
          aux_step_grad,
          pre_idx,
      )
    return (
        res_curr_device,
        res_prev_host,
        _layer_slice(w, layers),
        _layer_slice(w_aux, layers),
        aux_step_grad,
        None,
    )

  @jax.named_call
  def body_bwd(carry, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_in = carry
    carry, w_1_reduced = jax.named_call(_scan_step_bwd, name="unrolled_1")(
        carry, xs_1
    )
    carry_for_barrier = jax.tree.map(
        lambda x_in, x_out: None if x_out is x_in else x_out,
        carry_in,
        carry,
    )
    res_curr_0_device, *xs_0_rest = xs_0
    (carry_for_barrier, w_1_reduced), xs_0_rest = jax.lax.optimization_barrier(
        ((carry_for_barrier, w_1_reduced), xs_0_rest)
    )
    xs_0 = (res_curr_0_device, *xs_0_rest)
    carry = jax.tree.map(
        lambda x_in, x_bar: x_in if x_bar is None else x_bar,
        carry_in,
        carry_for_barrier,
        is_leaf=lambda x: x is None,
    )
    carry, w_0_reduced = jax.named_call(_scan_step_bwd, name="unrolled_0")(
        carry, xs_0
    )
    return carry, (w_0_reduced, w_1_reduced)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2

  carry_bwd = (
      carry_grad,
      w_last_expert_grad_unreduced,
      w_curr,
      w_last,
      w_aux_penultimate,
      w_aux_last,
      res_last_sb_loaded,
      None,
      pre_carry,
  )

  with jax.named_scope("scan_step_bwd_last"):
    carry_bwd, _ = _scan_step_bwd(
        carry_bwd,
        _step_xs(
            res_last_sb_device,
            res_penultimate_sb_host,
            num_layers - 3,
            aux_last_sb_grad,
        ),
    )

  w_rem_reduced = None
  if num_scanned % 2 == 1:
    xs_rem = _step_xs(
        res_rem_device,
        res_rem_prev_host,
        num_scanned - 1,
        _layer_slice(aux_scanned_grad, -1),
    )
    carry_bwd, w_rem_reduced = _scan_step_bwd(carry_bwd, xs_rem)

  w_grads_reduced = None
  if num_pairs > 0:
    even = slice(0, 2 * num_pairs, 2)
    odd = slice(1, 1 + 2 * num_pairs, 2)
    aux_even_grad = _layer_slice(aux_scanned_grad, slice(0, 2 * num_pairs, 2))
    aux_odd_grad = _layer_slice(aux_scanned_grad, slice(1, 1 + 2 * num_pairs, 2))

    xs_0 = _step_xs(
        res_curr_0_scanned_device, res_prev_0_scanned_host, even, aux_even_grad
    )
    xs_1 = _step_xs(
        res_curr_1_scanned_device, res_prev_1_scanned_host, odd, aux_odd_grad
    )

    carry_bwd, (w_0_reduced_scanned, w_1_reduced_scanned) = jax.lax.scan(
        body_bwd,
        carry_bwd,
        (xs_0, xs_1),
        reverse=True,
    )

    w_grads_reduced = jax.tree.map(
        _interleave_arrays,
        w_0_reduced_scanned,
        w_1_reduced_scanned,
    )

  (
      carry_grad,
      w_second_expert_grad_unreduced,
      w_0,
      w_1,
      w_aux_0,
      w_aux_1,
      res_sb_0_loaded,
      (w_third_grad_partial, w_aux_2),
      _,
  ) = carry_bwd

  with jax.named_scope("scan_step_bwd_0"):
    w_third_grad_partial = _reduce_pre(w_third_grad_partial, w_aux_2)
    res_prologue_device = load_fn(res_prologue_host)
    res_sb_0_full = _merge_offloaded(res_sb_0_loaded, res_sb_0_device)

    (
        carry_grad,
        w_0_expert_grad_unreduced,
        w_second_grad_partial,
        w_third_grad_ici,
    ) = scan_body_fn.bwd(
        res_sb_0_full,
        (carry_grad, aux_step_0),
        w_0,
        w_1,
        w_second_expert_grad_unreduced,
        functools.partial(_reduce_ici_first, w_aux_layer=w_aux_1),
        w_third_grad_partial,
        functools.partial(_reduce_ici_second, w_aux_layer=w_aux_2),
        **_w_aux_kwargs(w_aux_0),
    )
    w_third_grad_reduced = _reduce_dcn(w_third_grad_ici)

  with jax.named_scope("prologue_bwd"):
    x_grad, w_0_mla_grad_unreduced = prologue_fn.bwd(
        res_prologue_device,
        carry_grad,
        w_0,
        **_w_aux_kwargs(w_aux_0),
    )
    w_0_grad_unreduced = combine_w_grads(
        w_0_mla_grad_unreduced, w_0_expert_grad_unreduced
    )
    w_0_grad_partial = _reduce_pre(
        _reduce_ici_first(w_0_grad_unreduced, w_aux_0), w_aux_0
    )
    w_second_grad_partial = _reduce_pre(w_second_grad_partial, w_aux_1)

    w_0_grad_reduced = _reduce_dcn(
        _reduce_ici_second(w_0_grad_partial, w_aux_0)
    )
    w_second_grad_reduced = _reduce_dcn(
        _reduce_ici_second(w_second_grad_partial, w_aux_1)
    )

  grad_parts = [
      jax.tree.map(lambda x: x[jnp.newaxis], w_0_grad_reduced),
      jax.tree.map(lambda x: x[jnp.newaxis], w_second_grad_reduced),
      jax.tree.map(lambda x: x[jnp.newaxis], w_third_grad_reduced),
  ]
  if w_grads_reduced is not None:
    grad_parts.append(w_grads_reduced)
  if w_rem_reduced is not None:
    grad_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], w_rem_reduced))

  w_grad = jax.tree.map(
      lambda *parts: jnp.concatenate(parts, axis=0),
      *grad_parts,
  )
  g_w_aux = (
      None
      if w_aux is None
      else jax.tree.map(lambda v: None if v is None else jnp.zeros_like(v), w_aux)
  )
  return x_grad, w_grad, g_w_aux


w_collect_pipelined_scan_prologue_epilogue.defvjp(
    w_collect_pipelined_scan_prologue_epilogue_fwd,
    w_collect_pipelined_scan_prologue_epilogue_bwd,
)


# ---------------------------------------------------------------------------
# Qwen SC1 / SC0 FSDP Weight Collection & Two-Stage Gradient Reduction
# ---------------------------------------------------------------------------


def _routed_moe_gather_impl(
    w_local: jax.Array,
    absmax_local: jax.Array | None,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
    use_fp8: bool,
) -> Any:
  """Pure forward gather of routed MoE weights (`wi` or `wo`) on SC1."""
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  legacy_k_sharded = k_shards == 0
  k_sharded = flat_k_sharded or legacy_k_sharded
  gather_axis = 1 if legacy_k_sharded else 0
  use_tiles = (
      cfg.use_pallas_kernels
      and cfg.routed_w_tiled_layout
      and (w_local.shape[1] % 8 == 0)
  )
  if flat_k_sharded:
    full_k = w_local.shape[1] * k_shards
    e_full = (w_local.shape[0] * cfg.fsdp_size) // k_shards
  elif legacy_k_sharded:
    full_k = w_local.shape[1] * cfg.fsdp_size
    e_full = w_local.shape[0]
  else:
    full_k = w_local.shape[1]
    e_full = w_local.shape[0] * cfg.fsdp_size
  keep_tiled_carrier = (
      cfg.use_pallas_kernels
      and cfg.moe_use_forked_3d_gmmv2
      and use_tiles
      and (full_k % 8 == 0)
  )
  min_b = cfg.sc_min_collective_bytes
  full_shape = (e_full, full_k, w_local.shape[2])

  if use_fp8 and absmax_local is not None:
    from maxtext.experimental.lineage.qwen import moe_kernels

    # `hoisted_absmax` already computed `pmax(..., axis_name="fsdp")` outside
    # the scan when `forked_ep_scalar` is active; only run `pmax` here for the
    # non-forked `k_sharded` fallback path.
    forked_ep_scalar = (
        cfg.moe_use_forked_3d_gmmv2
        and cfg.moe_use_tc_ragged
        and cfg.moe_3d_ep_tokens
        and cfg.moe_use_ragged_silu_3d_fp8
        and cfg.moe_fp8_token_dispatch
        and cfg.moe_fp8_gmm_compute
    )
    amax = absmax_local
    if k_sharded and cfg.fsdp_size > 1 and not forked_ep_scalar:
      amax = jax.lax.pmax(amax, axis_name="fsdp")
    codes_l, scale_l = moe_kernels.quantize_moe_codes_from_absmax(w_local, amax)
    if use_tiles:
      codes_l = ops.routed_w_to_tiles(codes_l)
    codes_f = (
        ops.sc_all_gather(
            codes_l,
            "fsdp",
            core_id=ag_core,
            axis=gather_axis,
            min_collective_bytes=min_b,
        )
        if cfg.fsdp_size > 1
        else codes_l
    )
    if flat_k_sharded:
      if use_tiles and keep_tiled_carrier:
        codes_f = codes_f.reshape(e_full, full_k // 8, 8, codes_f.shape[-1])
      elif use_tiles:
        codes_f = ops.routed_w_from_tiles(codes_f).reshape(full_shape)
      else:
        codes_f = codes_f.reshape(full_shape)
    elif use_tiles and not keep_tiled_carrier:
      codes_f = ops.routed_w_from_tiles(codes_f)
    if scale_l.ndim == 2 or cfg.fsdp_size <= 1:
      scale_f = scale_l
    elif flat_k_sharded:
      scale_f = jnp.broadcast_to(scale_l, (e_full, 1, scale_l.shape[-1]))
    elif legacy_k_sharded:
      scale_f = scale_l
    else:
      scale_f = ops.sc_all_gather(
          scale_l,
          "fsdp",
          core_id=ag_core,
          axis=0,
          min_collective_bytes=min_b,
      )
    ste_shape = codes_f.shape if keep_tiled_carrier else full_shape
    w_ste = jax.lax.broadcast(jnp.zeros((), dtype=w_local.dtype), ste_shape)
    return moe_kernels.Fp8SteWeight(w_ste=w_ste, codes=codes_f, scale=scale_f)

  w_in = ops.routed_w_to_tiles(w_local) if use_tiles else w_local
  w_f = (
      ops.sc_all_gather(
          w_in,
          "fsdp",
          core_id=ag_core,
          axis=gather_axis,
          min_collective_bytes=min_b,
      )
      if cfg.fsdp_size > 1
      else w_in
  )
  w_out = ops.routed_w_from_tiles(w_f) if use_tiles else w_f
  return w_out.reshape(full_shape) if flat_k_sharded else w_out


def _gather_dense_leaf(
    w_local: jax.Array,
    cfg: QwenConfig,
    *,
    sharded_axis: int,
    is_fsdp_expert: bool,
    core_id: int | None,
) -> jax.Array:
  """Pure forward all-gather of a dense weight leaf on SC1."""
  min_b = cfg.sc_min_collective_bytes
  if not is_fsdp_expert:
    stage2_axes = cfg.dense_stage2_axis_names
    if cfg.dense_stage2_size <= 1:
      return w_local
    return ops.sc_all_gather(
        w_local,
        stage2_axes,
        core_id=core_id,
        axis=sharded_axis,
        min_collective_bytes=min_b,
    )
  if cfg.num_devices <= 1:
    return w_local
  if cfg.dense_1stage_3d_fsdp:
    return ops.sc_3d_all_gather(
        w_local,
        cfg.all_sharded_axes,
        core_id=core_id,
        axis=sharded_axis,
        min_collective_bytes=min_b,
    )
  w_ep = ops.sc_all_gather(
      w_local,
      "expert",
      core_id=core_id,
      axis=sharded_axis,
      min_collective_bytes=min_b,
  )
  return ops.sc_all_gather(
      w_ep,
      cfg.dense_stage2_axis_names,
      core_id=core_id,
      axis=sharded_axis,
      min_collective_bytes=min_b,
  )


def _gather_moe_only(
    m: params_lib.MoeParams,
    sub_a: Any,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
) -> params_lib.MoeParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  router_fe = cfg.dense_proj_fsdp_expert(cfg.num_experts)
  shared_fe = cfg.dense_proj_fsdp_expert(cfg.shared_mlp_dim)
  use_fp8_moe = bool(cfg.use_pallas_kernels and cfg.moe_fp8_weights)
  wi_a = getattr(sub_a, "wi", None) if sub_a is not None else None
  wo_a = getattr(sub_a, "wo", None) if sub_a is not None else None
  return params_lib.MoeParams(
      router_gate=_gather_dense_leaf(
          m.router_gate,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=router_fe,
          core_id=ag_core,
      ),
      wi=_routed_moe_gather_impl(
          m.wi, wi_a, cfg, ag_core=ag_core, use_fp8=use_fp8_moe
      ),
      wo=_routed_moe_gather_impl(
          m.wo, wo_a, cfg, ag_core=ag_core, use_fp8=use_fp8_moe
      ),
      shared_wi_0=_gather_dense_leaf(
          m.shared_wi_0,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          core_id=ag_core,
      ),
      shared_wi_1=_gather_dense_leaf(
          m.shared_wi_1,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          core_id=ag_core,
      ),
      shared_wo=_gather_dense_leaf(
          m.shared_wo,
          cfg,
          sharded_axis=1,
          is_fsdp_expert=shared_fe,
          core_id=ag_core,
      ),
      shared_gate=(
          m.shared_gate
          if cfg.replicate_shared_gate
          else _gather_dense_leaf(
              m.shared_gate,
              cfg,
              sharded_axis=0,
              is_fsdp_expert=is_fe,
              core_id=ag_core,
          )
      ),
  )


def _gather_gdn_mixer_only(
    p: params_lib.GdnSublayerParams,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
) -> params_lib.GdnSublayerParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  gdn_ba_fe = cfg.dense_proj_fsdp_expert(2 * cfg.gdn_num_v_heads)
  return params_lib.GdnSublayerParams(
      pre_attn_norm=p.pre_attn_norm,
      post_attn_norm=None,
      in_proj_qkvz=_gather_dense_leaf(
          p.in_proj_qkvz, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      in_proj_ba=_gather_dense_leaf(
          p.in_proj_ba, cfg, sharded_axis=0, is_fsdp_expert=gdn_ba_fe, core_id=ag_core
      ),
      conv1d=p.conv1d,
      A_log=p.A_log,
      dt_bias=p.dt_bias,
      gdn_norm=p.gdn_norm,
      out_proj=_gather_dense_leaf(
          p.out_proj, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      moe=None,
  )


def _gather_attn_mixer_only(
    p: params_lib.AttnSublayerParams,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
) -> params_lib.AttnSublayerParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  attn_kv_fe = cfg.dense_proj_fsdp_expert(cfg.num_kv_heads * cfg.head_dim)
  return params_lib.AttnSublayerParams(
      pre_attn_norm=p.pre_attn_norm,
      post_attn_norm=None,
      q_proj=_gather_dense_leaf(
          p.q_proj, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      k_proj=_gather_dense_leaf(
          p.k_proj, cfg, sharded_axis=0, is_fsdp_expert=attn_kv_fe, core_id=ag_core
      ),
      v_proj=_gather_dense_leaf(
          p.v_proj, cfg, sharded_axis=0, is_fsdp_expert=attn_kv_fe, core_id=ag_core
      ),
      q_norm=p.q_norm,
      k_norm=p.k_norm,
      out_proj=_gather_dense_leaf(
          p.out_proj, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      moe=None,
  )


def collect_sublayer_mixer(
    sub_p: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
) -> Any:
  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  if is_attn:
    return _gather_attn_mixer_only(sub_p, cfg, ag_core=ag_core)
  return _gather_gdn_mixer_only(sub_p, cfg, ag_core=ag_core)


def collect_sublayer_moe(
    sub_p: Any,
    sub_a: Any,
    cfg: QwenConfig,
) -> Any:
  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  return _gather_moe_only(sub_p.moe, sub_a, cfg, ag_core=ag_core)


def collect_sublayer_full(
    sub_p: Any,
    sub_a: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
) -> Any:
  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  if is_attn:
    return _gather_attn_full(sub_p, sub_a, cfg, ag_core=ag_core)
  return _gather_gdn_full(sub_p, sub_a, cfg, ag_core=ag_core)


def _gather_gdn_full(
    p: params_lib.GdnSublayerParams,
    sub_a: Any,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
) -> params_lib.GdnSublayerParams:
  m = _gather_gdn_mixer_only(p, cfg, ag_core=ag_core)
  return params_lib.GdnSublayerParams(
      pre_attn_norm=m.pre_attn_norm,
      post_attn_norm=p.post_attn_norm,
      in_proj_qkvz=m.in_proj_qkvz,
      in_proj_ba=m.in_proj_ba,
      conv1d=m.conv1d,
      A_log=m.A_log,
      dt_bias=m.dt_bias,
      gdn_norm=m.gdn_norm,
      out_proj=m.out_proj,
      moe=_gather_moe_only(p.moe, sub_a, cfg, ag_core=ag_core),
  )


def _gather_attn_full(
    p: params_lib.AttnSublayerParams,
    sub_a: Any,
    cfg: QwenConfig,
    *,
    ag_core: int | None,
) -> params_lib.AttnSublayerParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  attn_kv_fe = cfg.dense_proj_fsdp_expert(cfg.num_kv_heads * cfg.head_dim)
  return params_lib.AttnSublayerParams(
      pre_attn_norm=p.pre_attn_norm,
      post_attn_norm=p.post_attn_norm,
      q_proj=_gather_dense_leaf(
          p.q_proj, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      k_proj=_gather_dense_leaf(
          p.k_proj, cfg, sharded_axis=0, is_fsdp_expert=attn_kv_fe, core_id=ag_core
      ),
      v_proj=_gather_dense_leaf(
          p.v_proj, cfg, sharded_axis=0, is_fsdp_expert=attn_kv_fe, core_id=ag_core
      ),
      q_norm=p.q_norm,
      k_norm=p.k_norm,
      out_proj=_gather_dense_leaf(
          p.out_proj, cfg, sharded_axis=0, is_fsdp_expert=is_fe, core_id=ag_core
      ),
      moe=_gather_moe_only(p.moe, sub_a, cfg, ag_core=ag_core),
  )


def collect_w_ici(
    w_sb: params_lib.SuperblockParams,
    cfg: QwenConfig,
    *,
    split: bool = True,
    w_aux: Any = None,
) -> params_lib.SuperblockParams:
  """Collects one superblock's weights on SC1, splitting `gdn_0` mixer from rest when `split=True`."""
  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  a0 = getattr(w_aux, "gdn_0", None) if w_aux is not None else None
  a1 = getattr(w_aux, "gdn_1", None) if w_aux is not None else None
  a2 = getattr(w_aux, "gdn_2", None) if w_aux is not None else None
  a3 = getattr(w_aux, "attn_3", None) if w_aux is not None else None

  if split:
    # Block 1 (SC1): `gdn_0` mixer weights needed first in prev superblock's F7.
    gdn0_mixer = _gather_gdn_mixer_only(w_sb.gdn_0, cfg, ag_core=ag_core)
    # Block 2 (SC1): `gdn_0` MoE + remaining sublayers.
    gdn0_moe = _gather_moe_only(w_sb.gdn_0.moe, a0, cfg, ag_core=ag_core)
    gdn0 = params_lib.GdnSublayerParams(
        pre_attn_norm=gdn0_mixer.pre_attn_norm,
        post_attn_norm=w_sb.gdn_0.post_attn_norm,
        in_proj_qkvz=gdn0_mixer.in_proj_qkvz,
        in_proj_ba=gdn0_mixer.in_proj_ba,
        conv1d=gdn0_mixer.conv1d,
        A_log=gdn0_mixer.A_log,
        dt_bias=gdn0_mixer.dt_bias,
        gdn_norm=gdn0_mixer.gdn_norm,
        out_proj=gdn0_mixer.out_proj,
        moe=gdn0_moe,
    )
    gdn1 = _gather_gdn_full(w_sb.gdn_1, a1, cfg, ag_core=ag_core)
    gdn2 = _gather_gdn_full(w_sb.gdn_2, a2, cfg, ag_core=ag_core)
    attn3 = _gather_attn_full(w_sb.attn_3, a3, cfg, ag_core=ag_core)
    return params_lib.SuperblockParams(
        gdn_0=gdn0, gdn_1=gdn1, gdn_2=gdn2, attn_3=attn3
    )

  return params_lib.SuperblockParams(
      gdn_0=_gather_gdn_full(w_sb.gdn_0, a0, cfg, ag_core=ag_core),
      gdn_1=_gather_gdn_full(w_sb.gdn_1, a1, cfg, ag_core=ag_core),
      gdn_2=_gather_gdn_full(w_sb.gdn_2, a2, cfg, ag_core=ag_core),
      attn_3=_gather_attn_full(w_sb.attn_3, a3, cfg, ag_core=ag_core),
  )


def _reduce_routed_sc1(
    dw_full: Any, cfg: QwenConfig
) -> jax.Array:
  """Stage-1 SC1 ReduceScatter of routed MoE gradient `g_wi` or `g_wo`."""
  if hasattr(dw_full, "w_ste"):
    dw_full = dw_full.w_ste
  ops.assert_bf16_or_fp32_reduction(dw_full, "reduce_routed_sc1")
  if cfg.fsdp_size <= 1:
    return dw_full
  rs_core = 1 if cfg.fsdp_rs_sc_pinning == "sc1" else None
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  legacy_k_sharded = k_shards == 0
  gather_axis = 1 if legacy_k_sharded else 0
  min_b = cfg.sc_min_collective_bytes
  use_tiles = (
      cfg.use_pallas_kernels
      and cfg.routed_w_tiled_layout
      and (dw_full.shape[-2] % 8 == 0)
  )
  if dw_full.ndim == 4:
    dw_rs = (
        dw_full.reshape(
            dw_full.shape[0] * k_shards,
            dw_full.shape[1] // k_shards,
            dw_full.shape[2],
            dw_full.shape[3],
        )
        if flat_k_sharded
        else dw_full
    )
    dw_local_tiled = ops.sc_psum_scatter(
        dw_rs,
        "fsdp",
        core_id=rs_core,
        scatter_dimension=gather_axis,
        min_collective_bytes=min_b,
    )
    return ops.routed_w_from_tiles(dw_local_tiled)
  if use_tiles:
    dw_in = (
        dw_full.reshape(
            dw_full.shape[0] * k_shards,
            dw_full.shape[1] // k_shards,
            dw_full.shape[2],
        )
        if flat_k_sharded
        else dw_full
    )
    dw_tiled = ops.routed_w_to_tiles(dw_in)
    dw_local_tiled = ops.sc_psum_scatter(
        dw_tiled,
        "fsdp",
        core_id=rs_core,
        scatter_dimension=gather_axis,
        min_collective_bytes=min_b,
    )
    return ops.routed_w_from_tiles(dw_local_tiled)
  dw_rs = (
      dw_full.reshape(
          dw_full.shape[0] * k_shards,
          dw_full.shape[1] // k_shards,
          dw_full.shape[2],
      )
      if flat_k_sharded
      else dw_full
  )
  return ops.sc_psum_scatter(
      dw_rs,
      "fsdp",
      core_id=rs_core,
      scatter_dimension=gather_axis,
      min_collective_bytes=min_b,
  )


def _reduce_dense_sc1(
    g_full: jax.Array,
    cfg: QwenConfig,
    *,
    scatter_dim: int,
    is_fsdp_expert: bool,
    rs_core: int | None,
) -> jax.Array:
  """Stage-1 SC1 ReduceScatter of a dense projection gradient."""
  ops.assert_bf16_or_fp32_reduction(g_full, "reduce_dense_sc1")
  min_b = cfg.sc_min_collective_bytes
  if not is_fsdp_expert:
    if cfg.dense_stage2_size <= 1:
      return g_full
    return ops.sc_psum_scatter(
        g_full,
        cfg.dense_stage2_axis_names,
        core_id=rs_core,
        scatter_dimension=scatter_dim,
        min_collective_bytes=min_b,
    )
  if cfg.num_devices <= 1:
    return g_full
  if cfg.dense_1stage_3d_fsdp:
    return ops.sc_3d_psum_scatter(
        g_full,
        cfg.all_sharded_axes,
        core_id=rs_core,
        scatter_dimension=scatter_dim,
        min_collective_bytes=min_b,
    )
  g_s2 = ops.sc_psum_scatter(
      g_full,
      cfg.dense_stage2_axis_names,
      core_id=rs_core,
      scatter_dimension=scatter_dim,
      min_collective_bytes=min_b,
  )
  return ops.sc_psum_scatter(
      g_s2,
      "expert",
      core_id=rs_core,
      scatter_dimension=scatter_dim,
      min_collective_bytes=min_b,
  )


def _reduce_replicated_sc1(
    g: jax.Array,
    cfg: QwenConfig,
    *,
    rs_core: int | None,
    defer_replicated_ar: bool = False,
) -> jax.Array:
  """Stage-1 SC1 AllReduce over `dense_stage2_axis_names`, leaving `'expert'` for SC0."""
  ops.assert_bf16_or_fp32_reduction(g, "reduce_replicated_sc1")
  if defer_replicated_ar or cfg.dense_stage2_size <= 1:
    return g
  return ops.sc_all_reduce(
      g,
      cfg.dense_stage2_axis_names,
      core_id=rs_core,
      min_collective_bytes=cfg.sc_min_collective_bytes,
  )


def reduce_routed_wi_wo_sc1(
    g_wi: Any, g_wo: Any, cfg: QwenConfig
) -> tuple[jax.Array, jax.Array]:
  """Stage-1 SC1 ReduceScatter of routed MoE `wi`/`wo` gradients only."""
  return _reduce_routed_sc1(g_wi, cfg), _reduce_routed_sc1(g_wo, cfg)


def _reduce_moe_sc1(
    m: params_lib.MoeParams,
    cfg: QwenConfig,
    *,
    rs_core: int | None,
    defer_replicated_ar: bool = False,
    skip_routed_wi_wo: bool = False,
) -> params_lib.MoeParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  router_fe = cfg.dense_proj_fsdp_expert(cfg.num_experts)
  shared_fe = cfg.dense_proj_fsdp_expert(cfg.shared_mlp_dim)
  return params_lib.MoeParams(
      router_gate=_reduce_dense_sc1(
          m.router_gate,
          cfg,
          scatter_dim=0,
          is_fsdp_expert=router_fe,
          rs_core=rs_core,
      ),
      wi=m.wi if skip_routed_wi_wo else _reduce_routed_sc1(m.wi, cfg),
      wo=m.wo if skip_routed_wi_wo else _reduce_routed_sc1(m.wo, cfg),
      shared_wi_0=_reduce_dense_sc1(
          m.shared_wi_0,
          cfg,
          scatter_dim=0,
          is_fsdp_expert=shared_fe,
          rs_core=rs_core,
      ),
      shared_wi_1=_reduce_dense_sc1(
          m.shared_wi_1,
          cfg,
          scatter_dim=0,
          is_fsdp_expert=shared_fe,
          rs_core=rs_core,
      ),
      shared_wo=_reduce_dense_sc1(
          m.shared_wo,
          cfg,
          scatter_dim=1,
          is_fsdp_expert=shared_fe,
          rs_core=rs_core,
      ),
      shared_gate=(
          _reduce_replicated_sc1(
              m.shared_gate,
              cfg,
              rs_core=rs_core,
              defer_replicated_ar=defer_replicated_ar,
          )
          if cfg.replicate_shared_gate
          else _reduce_dense_sc1(
              m.shared_gate,
              cfg,
              scatter_dim=0,
              is_fsdp_expert=is_fe,
              rs_core=rs_core,
          )
      ),
  )


def _reduce_gdn_sc1(
    g: params_lib.GdnSublayerParams,
    cfg: QwenConfig,
    *,
    rs_core: int | None,
    defer_replicated_ar: bool = False,
    skip_routed_wi_wo: bool = False,
) -> params_lib.GdnSublayerParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  gdn_ba_fe = cfg.dense_proj_fsdp_expert(2 * cfg.gdn_num_v_heads)
  return params_lib.GdnSublayerParams(
      pre_attn_norm=_reduce_replicated_sc1(
          g.pre_attn_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      post_attn_norm=_reduce_replicated_sc1(
          g.post_attn_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      in_proj_qkvz=_reduce_dense_sc1(
          g.in_proj_qkvz, cfg, scatter_dim=0, is_fsdp_expert=is_fe, rs_core=rs_core
      ),
      in_proj_ba=_reduce_dense_sc1(
          g.in_proj_ba,
          cfg,
          scatter_dim=0,
          is_fsdp_expert=gdn_ba_fe,
          rs_core=rs_core,
      ),
      conv1d=_reduce_replicated_sc1(
          g.conv1d,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      A_log=_reduce_replicated_sc1(
          g.A_log,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      dt_bias=_reduce_replicated_sc1(
          g.dt_bias,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      gdn_norm=_reduce_replicated_sc1(
          g.gdn_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      out_proj=_reduce_dense_sc1(
          g.out_proj, cfg, scatter_dim=0, is_fsdp_expert=is_fe, rs_core=rs_core
      ),
      moe=_reduce_moe_sc1(
          g.moe,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
          skip_routed_wi_wo=skip_routed_wi_wo,
      ),
  )


def _reduce_attn_sc1(
    g: params_lib.AttnSublayerParams,
    cfg: QwenConfig,
    *,
    rs_core: int | None,
    defer_replicated_ar: bool = False,
    skip_routed_wi_wo: bool = False,
) -> params_lib.AttnSublayerParams:
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  attn_kv_fe = cfg.dense_proj_fsdp_expert(cfg.num_kv_heads * cfg.head_dim)
  return params_lib.AttnSublayerParams(
      pre_attn_norm=_reduce_replicated_sc1(
          g.pre_attn_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      post_attn_norm=_reduce_replicated_sc1(
          g.post_attn_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      q_proj=_reduce_dense_sc1(
          g.q_proj, cfg, scatter_dim=0, is_fsdp_expert=is_fe, rs_core=rs_core
      ),
      k_proj=_reduce_dense_sc1(
          g.k_proj, cfg, scatter_dim=0, is_fsdp_expert=attn_kv_fe, rs_core=rs_core
      ),
      v_proj=_reduce_dense_sc1(
          g.v_proj, cfg, scatter_dim=0, is_fsdp_expert=attn_kv_fe, rs_core=rs_core
      ),
      q_norm=_reduce_replicated_sc1(
          g.q_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      k_norm=_reduce_replicated_sc1(
          g.k_norm,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
      ),
      out_proj=_reduce_dense_sc1(
          g.out_proj, cfg, scatter_dim=0, is_fsdp_expert=is_fe, rs_core=rs_core
      ),
      moe=_reduce_moe_sc1(
          g.moe,
          cfg,
          rs_core=rs_core,
          defer_replicated_ar=defer_replicated_ar,
          skip_routed_wi_wo=skip_routed_wi_wo,
      ),
  )


def reduce_w_ici_sc1(
    grad_w: params_lib.SuperblockParams,
    like: Any,
    cfg: QwenConfig,
    *,
    w_aux: Any = None,
) -> params_lib.SuperblockParams:
  """Stage-1 ICI reduction on SC1 for ALL superblock grads (dense + MoE `wi`/`wo`)."""
  del like, w_aux
  rs_core = 1 if cfg.fsdp_rs_sc_pinning == "sc1" else None
  return params_lib.SuperblockParams(
      gdn_0=_reduce_gdn_sc1(grad_w.gdn_0, cfg, rs_core=rs_core),
      gdn_1=_reduce_gdn_sc1(grad_w.gdn_1, cfg, rs_core=rs_core),
      gdn_2=_reduce_gdn_sc1(grad_w.gdn_2, cfg, rs_core=rs_core),
      attn_3=_reduce_attn_sc1(grad_w.attn_3, cfg, rs_core=rs_core),
  )


def _is_replicated_param_mask(cfg: QwenConfig) -> params_lib.SuperblockParams:
  """Boolean mask matching `SuperblockParams`: True iff leaf is fully replicated."""
  rep_sg = bool(cfg.replicate_shared_gate)

  def _moe_mask():
    return params_lib.MoeParams(
        router_gate=False,
        wi=False,
        wo=False,
        shared_wi_0=False,
        shared_wi_1=False,
        shared_wo=False,
        shared_gate=rep_sg,
    )

  def _gdn_mask():
    return params_lib.GdnSublayerParams(
        pre_attn_norm=True,
        post_attn_norm=True,
        in_proj_qkvz=False,
        in_proj_ba=False,
        conv1d=True,
        A_log=True,
        dt_bias=True,
        gdn_norm=True,
        out_proj=False,
        moe=_moe_mask(),
    )

  attn_mask = params_lib.AttnSublayerParams(
      pre_attn_norm=True,
      post_attn_norm=True,
      q_proj=False,
      k_proj=False,
      v_proj=False,
      q_norm=True,
      k_norm=True,
      out_proj=False,
      moe=_moe_mask(),
  )
  return params_lib.SuperblockParams(
      gdn_0=_gdn_mask(),
      gdn_1=_gdn_mask(),
      gdn_2=_gdn_mask(),
      attn_3=attn_mask,
  )


def _needs_ep_allreduce_mask(
    cfg: QwenConfig, *, defer_replicated_ar: bool = False
) -> params_lib.SuperblockParams:
  """Boolean mask matching `SuperblockParams`: True iff leaf is replicated on `'expert'`."""
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  router_fe = cfg.dense_proj_fsdp_expert(cfg.num_experts)
  shared_fe = cfg.dense_proj_fsdp_expert(cfg.shared_mlp_dim)
  gdn_ba_fe = cfg.dense_proj_fsdp_expert(2 * cfg.gdn_num_v_heads)
  attn_kv_fe = cfg.dense_proj_fsdp_expert(cfg.num_kv_heads * cfg.head_dim)
  rep_here = not defer_replicated_ar

  def _moe_mask():
    return params_lib.MoeParams(
        router_gate=not router_fe,
        wi=False,
        wo=False,
        shared_wi_0=not shared_fe,
        shared_wi_1=not shared_fe,
        shared_wo=not shared_fe,
        shared_gate=(
            rep_here if cfg.replicate_shared_gate else (not is_fe)
        ),
    )

  def _gdn_mask():
    return params_lib.GdnSublayerParams(
        pre_attn_norm=rep_here,
        post_attn_norm=rep_here,
        in_proj_qkvz=not is_fe,
        in_proj_ba=not gdn_ba_fe,
        conv1d=rep_here,
        A_log=rep_here,
        dt_bias=rep_here,
        gdn_norm=rep_here,
        out_proj=not is_fe,
        moe=_moe_mask(),
    )

  attn_mask = params_lib.AttnSublayerParams(
      pre_attn_norm=rep_here,
      post_attn_norm=rep_here,
      q_proj=not is_fe,
      k_proj=not attn_kv_fe,
      v_proj=not attn_kv_fe,
      q_norm=rep_here,
      k_norm=rep_here,
      out_proj=not is_fe,
      moe=_moe_mask(),
  )
  return params_lib.SuperblockParams(
      gdn_0=_gdn_mask(),
      gdn_1=_gdn_mask(),
      gdn_2=_gdn_mask(),
      attn_3=attn_mask,
  )


def make_deferred_ep_reduce(
    subtree: Any,
    mask_subtree: Any,
    cfg: QwenConfig,
) -> DeferredReduce | None:
  """Builds a `DeferredReduce` for EP-axis `AllReduce` (`'expert'`) of `subtree`."""
  if cfg.ep_ici_size <= 1:
    return None
  masked_tree = jax.tree.map(
      lambda x, m: x if (x is not None and m) else None,
      subtree,
      mask_subtree,
      is_leaf=lambda x: x is None,
  )
  operands, masked_treedef = jax.tree.flatten(masked_tree)
  if not operands:
    return None
  for op in operands:
    ops.assert_bf16_or_fp32_reduction(op, "deferred_ep_allreduce")

  def _local_fn(ops_list: list[jax.Array]) -> list[jax.Array]:
    return [jax.lax.psum(x, axis_name="expert") for x in ops_list]

  def _finish(reduced_list: list[jax.Array]) -> Any:
    reduced_tree = jax.tree.unflatten(masked_treedef, reduced_list)
    return jax.tree.map(
        lambda orig, red: orig if red is None else red,
        subtree,
        reduced_tree,
        is_leaf=lambda x: x is None,
    )

  return DeferredReduce(operand=operands, local_fn=_local_fn, finish=_finish)


def sc_compute_on(fn: Callable[..., Any], *, core_id: int) -> Callable[..., Any]:
  """Pins `fn` onto SparseCore `core_id` (`0` or `1`)."""
  return compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )(fn)


def reduce_w_ici_sc0(
    grad_w_partial: params_lib.SuperblockParams,
    like: Any,
    cfg: QwenConfig,
    *,
    defer: bool = False,
    w_aux: Any = None,
) -> Any:
  """Stage-2 ICI reduction on SC0 (`AllReduce` over `'expert'` for replicated leaves)."""
  del like, w_aux
  mask = _needs_ep_allreduce_mask(cfg)
  if not defer:
    if cfg.ep_ici_size <= 1:
      return grad_w_partial
    dr = make_deferred_ep_reduce(grad_w_partial, mask, cfg)
    if dr is None:
      return grad_w_partial
    sc0_fn = sc_compute_on(dr.local_fn, core_id=0)
    return dr.finish(sc0_fn(dr.operand))

  # Return one DeferredReduce per sublayer (or None if ep_ici_size <= 1).
  return tuple(
      make_deferred_ep_reduce(
          getattr(grad_w_partial, attr), getattr(mask, attr), cfg
      )
      for attr in ("gdn_0", "gdn_1", "gdn_2", "attn_3")
  )


def reduce_sublayer_sc1(
    grad_sub: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    defer_replicated_ar: bool = False,
    skip_routed_wi_wo: bool = False,
) -> Any:
  """Stage-1 ICI reduction on SC1 for one sublayer's grads (dense + optional MoE)."""
  rs_core = 1 if cfg.fsdp_rs_sc_pinning == "sc1" else None
  if is_attn:
    return _reduce_attn_sc1(
        grad_sub,
        cfg,
        rs_core=rs_core,
        defer_replicated_ar=defer_replicated_ar,
        skip_routed_wi_wo=skip_routed_wi_wo,
    )
  return _reduce_gdn_sc1(
      grad_sub,
      cfg,
      rs_core=rs_core,
      defer_replicated_ar=defer_replicated_ar,
      skip_routed_wi_wo=skip_routed_wi_wo,
  )


def reduce_sublayer_sc0(
    grad_sub: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    defer_replicated_ar: bool = False,
) -> DeferredReduce | None:
  """Returns a `DeferredReduce` for EP-axis `'expert'` `AllReduce` on SC0."""
  if cfg.ep_ici_size <= 1:
    return None
  mask = _needs_ep_allreduce_mask(cfg, defer_replicated_ar=defer_replicated_ar)
  sub_mask = mask.attn_3 if is_attn else mask.gdn_0
  return make_deferred_ep_reduce(grad_sub, sub_mask, cfg)


def reduce_hoisted_replicated_grads(
    dw_blocks: params_lib.SuperblockParams,
    cfg: QwenConfig,
) -> params_lib.SuperblockParams:
  """Reduces all stacked replicated 1D/small grads in ONE batched post-scan AR."""
  if cfg.num_devices <= 1:
    return dw_blocks
  mask = _is_replicated_param_mask(cfg)
  masked_tree = jax.tree.map(
      lambda x, m: x if (x is not None and m) else None,
      dw_blocks,
      mask,
      is_leaf=lambda x: x is None,
  )
  operands, masked_treedef = jax.tree.flatten(masked_tree)
  if not operands:
    return dw_blocks
  for op in operands:
    ops.assert_bf16_or_fp32_reduction(op, "hoisted_replicated_ar")

  all_axes = (
      ("fsdp", "expert", "ep_d2d")
      if cfg.decouple_ep_d2d
      else ("fsdp", "expert")
  )

  def _ar_all(ops_list: list[jax.Array]) -> list[jax.Array]:
    return [jax.lax.psum(x, axis_name=all_axes) for x in ops_list]

  if cfg.use_pallas_kernels and cfg.fsdp_rs_sc_pinning == "sc1":
    reduced_list = sc_compute_on(_ar_all, core_id=1)(operands)
  else:
    reduced_list = _ar_all(operands)
  reduced_tree = jax.tree.unflatten(masked_treedef, reduced_list)
  return jax.tree.map(
      lambda orig, red: orig if red is None else red,
      dw_blocks,
      reduced_tree,
      is_leaf=lambda x: x is None,
  )
