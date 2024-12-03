"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import jax
import jax.numpy as jnp
from jax._src.api import TransferToMemoryKind
import optax
import max_utils

from flax import core, struct
from flax import linen as nn
from flax.training import train_state
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

from typing import Any, Callable, NamedTuple


class PiecewiseOptimizerState(NamedTuple):
  """State used for the PiecewiseOptimizer."""

  unstacked_opt_state: Any
  stacked_opt_state: Any


def _add_dummy_stacking(
    params: Any,
) -> Any:
  """Adds a singleton outer dimension to the params."""
  return params
  # return jax.tree.map(lambda x: jnp.expand_dims(x, 0), params)


def _remove_dummy_stacking(
    params: Any,
) -> Any:
  """Removes the singleton outer dimension from the params."""
  return params
  # return jax.tree.map(lambda x: jnp.squeeze(x, 0) if x.ndim > 1 else x, params)


class TrainState(train_state.TrainState):

  step: int | jax.Array
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: PiecewiseOptimizerState = struct.field(pytree_node=True)
  optimizer_memory_host_offload: bool = struct.field(pytree_node=False)


  # def apply_to_slice(self, grads, params, index, stacked_opt_state):
  #   slice_opt_state = jax.tree.map(
  #     lambda x: jax.lax.dynamic_index_in_dim(x, index, axis=1, keepdims=False) if x.ndim > 1 else x,
  #     stacked_opt_state,
  #   )
  #   if self.optimizer_memory_host_offload:
  #     params = jax.device_put(params, TransferToMemoryKind('device'))
  #     slice_opt_state = jax.device_put(slice_opt_state, TransferToMemoryKind('device'))
  #   breakpoint()
  #   updates, new_opt_state = self.tx.update(
  #     _add_dummy_stacking(grads),
  #     slice_opt_state,
  #     _add_dummy_stacking(params),
  #   )
  #   new_params = optax.apply_updates(params, updates)
  #   if self.optimizer_memory_host_offload:
  #     new_params = jax.device_put(new_params, TransferToMemoryKind('pinned_host'))
  #     new_opt_state = jax.device_put(new_opt_state, TransferToMemoryKind('pinned_host'))
  #   return _remove_dummy_stacking(new_params), new_opt_state

  def apply_to_slice(self, grads, params, opt_state):
    updates, new_opt_state = self.tx.update(
      _add_dummy_stacking(grads),
      opt_state,
      _add_dummy_stacking(params),
    )
    new_params = optax.apply_updates(params, updates)
    # if self.optimizer_memory_host_offload:
    #   new_params = jax.device_put(new_params, TransferToMemoryKind('pinned_host'))
    #   new_opt_state = jax.device_put(new_opt_state, TransferToMemoryKind('pinned_host'))
    return _remove_dummy_stacking(new_params), new_opt_state


  def apply_complete(self, unstacked_grads, unstacked_params, new_stacked_params, new_stacked_opt_state):
    unstacked_opt_state = self.opt_state.unstacked_opt_state
    if self.optimizer_memory_host_offload:
      unstacked_params = jax.device_put(unstacked_params, TransferToMemoryKind('device'))
      unstacked_opt_state = jax.device_put(unstacked_opt_state, TransferToMemoryKind('device'))
    updates, new_unstacked_opt_state = self.tx.update(
      unstacked_grads, unstacked_opt_state, unstacked_params
    )
    new_unstacked_params = optax.apply_updates(unstacked_params, updates)
    # if self.optimizer_memory_host_offload:
    #   new_unstacked_params = jax.device_put(new_unstacked_params, TransferToMemoryKind('pinned_host'))
    #   new_unstacked_opt_state = jax.device_put(new_unstacked_opt_state, TransferToMemoryKind('pinned_host'))
    new_params = max_utils.merge_pytrees(new_stacked_params, new_unstacked_params)
    new_opt_state = PiecewiseOptimizerState(
        unstacked_opt_state=new_unstacked_opt_state,
        stacked_opt_state=new_stacked_opt_state,
    )
    return new_params, new_opt_state


  def apply_gradients(self, *, grads, **kwargs):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
    """
    is_stacked = lambda param_name: 'layers' in param_name
    reshape_fn = lambda x: jnp.swapaxes(x, 0, 1) if x.ndim > 1 else x

    if OVERWRITE_WITH_GRADIENT in grads:
      grads_with_opt = grads['params']
      params_with_opt = self.params['params']
    else:
      grads_with_opt = grads
      params_with_opt = self.params

    stacked_params, unstacked_params = max_utils.partition_pytree(params_with_opt, is_stacked)
    stacked_grads, unstacked_grads = max_utils.partition_pytree(grads_with_opt, is_stacked)

    if stacked_params:
      # reshaping params, grad and opt_state for scan
      stacked_params = jax.tree_util.tree_map(reshape_fn, stacked_params)
      stacked_grads = jax.tree_util.tree_map(reshape_fn, stacked_grads)
      stacked_opt_state = self.opt_state.stacked_opt_state
      # stacked_opt_state = jax.tree_util.tree_map(reshape_fn, self.opt_state.stacked_opt_state)
      # stacked_opt_state = _add_dummy_stacking(stacked_opt_state)
      # stacked_opt_state = jax.tree_util.tree_map(reshape_fn, stacked_opt_state)

      # stack_length = jax.tree.flatten(stacked_params)[0][0].shape[0]
      def apply_to_slice(carry, xs):
        del carry
        grads, params, opt_state = xs
        new_params, new_opt_state = self.apply_to_slice(
            grads, params, opt_state
        )
        return (None, (new_params, new_opt_state))
      if self.optimizer_memory_host_offload:
        stacked_params = jax.device_put(stacked_params, TransferToMemoryKind('device'))
        stacked_opt_state = jax.device_put(stacked_opt_state, TransferToMemoryKind('device'))
      new_stacked_params, new_stacked_opt_state = jax.lax.scan(
          apply_to_slice,
          None,
          (stacked_grads, stacked_params, stacked_opt_state),
      )[1]
      # reshaping back to original shapes
      new_stacked_params = jax.tree_util.tree_map(reshape_fn, new_stacked_params)
      # new_stacked_opt_state = jax.tree_util.tree_map(reshape_fn_2, new_stacked_opt_state)
      # new_stacked_opt_state = _remove_dummy_stacking(new_stacked_opt_state)
      # new_stacked_opt_state = jax.tree_util.tree_map(reshape_fn_2, new_stacked_opt_state)
    else:
      new_stacked_params = {}
      new_stacked_opt_state = {}
    new_params_with_opt, new_opt_state = self.apply_complete(
      unstacked_grads,
      unstacked_params,
      new_stacked_params,
      new_stacked_opt_state,
    )
    if OVERWRITE_WITH_GRADIENT in grads:
      new_params = {
        'params': new_params_with_opt,
        OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
      }
    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )


  @classmethod
  def create(cls, *, apply_fn, params, tx, optimizer_memory_host_offload, **kwargs):
    """Creates a new instance with ``step=0`` and initialized ``opt_state``."""

    is_stacked = lambda param_name: 'layers' in param_name
    params_with_opt = (
      params['params'] if OVERWRITE_WITH_GRADIENT in params else params
    )
    stacked_params, unstacked_params = max_utils.partition_pytree(params_with_opt, is_stacked)
    # reshape stacked_params for jax.lax.scan
    reshape_fn = lambda x: jnp.swapaxes(x, 0, 1) if x.ndim > 1 else x
    stacked_params = jax.tree_util.tree_map(reshape_fn, stacked_params)
    unstacked_opt_state = tx.init(unstacked_params)
    if stacked_params:
      def init_slice(carry, params):
        del carry
        return None, tx.init(params)

      stacked_opt_state = jax.lax.scan(init_slice, None, stacked_params)[1]
    else:
      stacked_opt_state = {}

    # reshaping back to original shapes
    stacked_params = jax.tree_util.tree_map(reshape_fn, stacked_params)
    # stacked_opt_state = jax.tree_util.tree_map(reshape_fn, stacked_opt_state)
    # stacked_opt_state = _remove_dummy_stacking(stacked_opt_state)
    # stacked_opt_state = jax.tree_util.tree_map(reshape_fn, stacked_opt_state)
    
    # transfer to host explicitly
    if optimizer_memory_host_offload:
      unstacked_opt_state = jax.device_put(unstacked_opt_state, TransferToMemoryKind('pinned_host'))
      stacked_opt_state = jax.device_put(stacked_opt_state, TransferToMemoryKind('pinned_host'))
      params = jax.device_put(params, TransferToMemoryKind('pinned_host'))

    opt_state = PiecewiseOptimizerState(
        unstacked_opt_state=unstacked_opt_state,
        stacked_opt_state=stacked_opt_state,
    )
    return cls(
      step=0,
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=opt_state,
      optimizer_memory_host_offload=optimizer_memory_host_offload,
      **kwargs,
    )