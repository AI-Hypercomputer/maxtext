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

# pylint: disable=bare-except, consider-using-generator, too-many-positional-arguments
""" Utils that are only interesting to MaxText. """

import re
import jax
import jax.numpy as jnp

import optax
from optax.contrib._muon import muon
from maxtext.utils.muon_utils import get_muon_weight_dimension_numbers


def get_adamw_mask(config):
  """Create a mask function for AdamW optimizer to exclude certain parameters from weight decay."""
  if not getattr(config, "adamw_mask", None):
    return None

  compiled_patterns = [re.compile(pattern) for pattern in config.adamw_mask]

  def mask_fn(params):
    def _is_decayed(path, _):
      # Join path keys into a single string for pattern matching (e.g., "layer1/bias")
      path_str = "/".join(str(p.key if hasattr(p, "key") else p.idx if hasattr(p, "idx") else p) for p in path)
      # If any pattern in adamw_mask matches the path, exclude from weight decay (return False).
      # Otherwise, apply weight decay (return True).
      return not any(pattern.search(path_str) for pattern in compiled_patterns)

    return jax.tree_util.tree_map_with_path(_is_decayed, params)

  return mask_fn


def get_optimizer(config, learning_rate_schedule, model=None):
  """Create optimizer."""
  if config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    return optax.adamw(
        learning_rate_schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        eps_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mu_dtype=config.mu_dtype,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "adam_pax":
    return adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "sgd":
    return optax.sgd(learning_rate_schedule)
  elif config.opt_type == "muon":
    # extract muon dimension number from model structure
    if model is not None:
      muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config)
    else:
      raise ValueError("Please specify model to extract muon dimension number.")
    muon_kwargs = {
        # Shared parameters: "nesterov" uses default
        "learning_rate": learning_rate_schedule,
        "eps": config.adam_eps,
        "mu_dtype": config.mu_dtype,
        # Muon-specific parameters: "ns_coeffs", "ns_steps", "weight_decay_mask", "adaptive" uses default
        "beta": config.muon_beta,
        "weight_decay": config.muon_weight_decay,
        "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
        "consistent_rms": config.muon_consistent_rms,
        # AdamW-specific parameters
        "adam_b1": config.adam_b1,
        "adam_b2": config.adam_b2,
        "adam_eps_root": config.adam_eps_root,
        "adam_weight_decay": config.adam_weight_decay,
    }
    return muon(**muon_kwargs)
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")


def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    mask=None,
) -> optax.GradientTransformation:
  """Standard Adam optimizer that supports weight decay.

  Follows the implementation in pax/praxis sharded_adam
  https://github.com/google/praxis/blob/545e00ab126b823265d70c715950d39333484f38/praxis/optimizers.py#L621

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `optax.GradientTransformation`.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.0
    return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    count = state.count

    class _slot_opt_state:

      def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu

    def _update_momentum(update, mu, nu):
      # The conversion to the data type of the update ensures that bfloat16 remains
      # bfloat16 in the optimizer state. This conversion has to be done after
      # `bias_corrected_dacay` is calculated as calculating `jnp.power(decay, t)` in low
      # precision can result in it being rounded to 1 and subsequently a
      # "division by zero" error.
      beta1_decay = bias_corrected_decay(count, beta1).astype(update.dtype)
      beta2_decay = bias_corrected_decay(count, beta2).astype(update.dtype)
      mu = (1.0 - beta1_decay) * update + beta1_decay * mu
      nu = (1.0 - beta2_decay) * (update**2) + beta2_decay * nu
      return _slot_opt_state(mu=mu, nu=nu)

    updated_moments = jax.tree_util.tree_map(_update_momentum, updates, state.mu, state.nu)

    mu = jax.tree_util.tree_map(lambda x: x.mu, updated_moments)
    nu = jax.tree_util.tree_map(lambda x: x.nu, updated_moments)

    updates = jax.tree_util.tree_map(lambda mu, nu: mu / (jnp.sqrt(nu + epsilon_root) + epsilon), mu, nu)

    if weight_decay > 0:
      if mask is not None:
        mask_tree = mask(params) if callable(mask) else mask
        updates = jax.tree_util.tree_map(lambda x, v, m: x + weight_decay * v if m else x, updates, params, mask_tree)
      else:
        updates = jax.tree_util.tree_map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
