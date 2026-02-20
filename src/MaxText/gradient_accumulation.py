# Copyright 2025 Google LLC
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

"""Functions for gradient accumulation (GA)"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.common.common_types import ShardMode
from MaxText.sharding import maybe_shard_with_name


def gradient_accumulation_loss_and_grad(
    _loss_fn,
    config,
    model,
    params,
    params_shardings,
    data,
    dropout_rng,
    extra_dpo_args,
):
  """
  Calculates gradients using gradient accumulation.

  This function computes the gradient of `_loss_fn` over multiple microbatches
  and accumulates them before returning a single, averaged gradient. It uses
  `jax.lax.scan` for efficient accumulation on device.

  It also supports a `shard_optimizer_over_data` mode (e.g., ZeRO-1) where
  parameters are cast to bf16 and sharded *before* the accumulation loop
  to perform the all-gather in lower precision.

  Args:
      _loss_fn: The loss function to differentiate. Its signature is expected
          to be: `(model, config, data, dropout_rng, params, *extra_args, is_train=True)`.
      config: Model and training configuration object. Must contain
          `gradient_accumulation_steps` and `shard_optimizer_over_data`.
      model: The model module.
      params: The model parameters (PyTree).
      params_shardings: The sharding constraints for the parameters (PyTree).
      data: A PyTree of batched data. The leading dimension is assumed
          to be the total batch size (microbatch_size * num_accumulations).
      dropout_rng: JAX PRNGKey for dropout.
      extra_dpo_args: A tuple of extra arguments to pass to the loss function.

  Returns:
      A tuple containing:
      - total_loss (Array): The mean loss, averaged over all microbatches.
      - final_aux (PyTree): Auxiliary outputs, summed across microbatches.
      - raw_grads (PyTree): The accumulated and averaged gradients.
  """

  def _maybe_shard_with_name(inputs, sharding_names):
    """Wrapper of maybe_shard_with_name with fixed shard_mode"""
    return maybe_shard_with_name(inputs, sharding_names, config.shard_mode, debug_sharding=config.debug_sharding)

  # For more efficient DP/ZeRO-1 + GA
  if config.shard_mode == ShardMode.EXPLICIT and config.ici_data_parallelism > 1:
    ga_params_shardings = jax.tree.map(update_sharding_for_reduced, params_shardings)
    grad_shardings = jax.tree.map(update_sharding_for_unreduced, params_shardings)
  else:
    ga_params_shardings = grad_shardings = params_shardings
  # When using Zero-1 optimizer sharding, cast params to lower precision and apply sharding constraints
  # so that all-gather is done once in the lower precision before the gradient accumulation loop
  if config.shard_optimizer_over_data:

    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param

    ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
  else:
    ga_params = params

  ga_params = jax.tree.map(_maybe_shard_with_name, ga_params, ga_params_shardings)
  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)

  def accumulate_gradient(acc_grad_and_loss, data):
    ga_params = acc_grad_and_loss["ga_params"]
    (_, aux), cur_batch_gradient = grad_func(model, config, data, dropout_rng, ga_params, *extra_dpo_args, is_train=True)
    acc_grad_and_loss["loss"] += aux["total_loss"]
    acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
    acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
    acc_grad_and_loss["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"])
    acc_grad_and_loss["total_weights"] += aux["total_weights"]
    return acc_grad_and_loss, aux

  def reshape_to_microbatch_accumulations(batch_arr):
    """Reshape global batch to microbatches, assuming batch axis is leading."""
    num_microbatches = config.gradient_accumulation_steps
    microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
    reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
    return jnp.swapaxes(reshaped_batch_arr, 0, 1)

  data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
  init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params)
  init_grad = jax.tree.map(_maybe_shard_with_name, init_grad, grad_shardings)
  init_grad_and_loss = {
      "loss": 0.0,
      "grad": init_grad,
      "total_weights": 0,
      "moe_lb_loss": 0.0,
      "mtp_loss": 0.0,
      "ga_params": ga_params,
  }

  grad_and_loss, aux = jax.lax.scan(
      accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
  )
  loss = (
      grad_and_loss["loss"] / grad_and_loss["total_weights"]
      + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
  )
  raw_grads = grad_and_loss["grad"]
  raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)
  raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], raw_grads)
  aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr

  return loss, aux, raw_grads


# GA helper functions
def update_sharding_for_reduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add reduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(reduced={"data"}))


def update_sharding_for_unreduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add unreduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(unreduced={"data"}))
