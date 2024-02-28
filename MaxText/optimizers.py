"""
 Copyright 2023 Google LLC

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

# pylint: disable=bare-except, consider-using-generator, ungrouped-imports
"""Utils that are only interesting to MaxText. """

import jax
import re

import optax
from optax._src import utils
import jax.numpy as jnp

from max_utils import create_learning_rate_schedule

def tree_path_to_string(path, sep=None):
  keys = []
  for key in path:
    if isinstance(key, jax.tree_util.SequenceKey):
      keys.append(str(key.idx))
    elif isinstance(key, jax.tree_util.DictKey):
      keys.append(str(key.key))
    elif isinstance(key, jax.tree_util.GetAttrKey):
      keys.append(str(key.name))
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
      keys.append(str(key.key))
    else:
      keys.append(str(key))
  if sep is None:
    return tuple(keys)
  return sep.join(keys)

def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
  """ An extended version of jax.tree_util.tree_map, where the mapped function
      f takes both the name (path) and the tree leaf as input.
  """
  return jax.tree_util.tree_map_with_path(
    lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
    tree, *rest,
    is_leaf=is_leaf
  )

def get_weight_decay_mask(exclusions):
  """ Return a weight decay mask function that computes the pytree masks
      according to the given exclusion rules.
  """
  def decay(name, _):
    for rule in exclusions:
      if re.search(rule, name) is not None:
        print((name, "not use weight_decay"))
        return False
      
    print((name, "use weight_decay"))
    return True

  def weight_decay_mask(params):
    return named_tree_map(decay, params, sep='/')
  
  return weight_decay_mask


def get_optimizer(config):

  if config.opt_type == "tiger":
    learning_rate_schedule = create_learning_rate_schedule(
      config, 
      step_reduction=1,
      update_step=config.gradient_accumulation_steps,
    )
    optimizer = tiger_pax(
      learning_rate_schedule,
      beta=config.adam_b1,
      weight_decay=config.adam_weight_decay,
      mask=get_weight_decay_mask([
        "norm",
        "scale",
        "bias",
      ]),
    )
    return optimizer, learning_rate_schedule

  """other optimizer"""

  learning_rate_schedule = create_learning_rate_schedule(
    config, 
    step_reduction=config.gradient_accumulation_steps,
    update_step=1,
  )

  if config.opt_type == "sgd":
    optimizer = optax.sgd(
      learning_rate_schedule,
    )
  elif config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    optimizer = optax.adamw(
      learning_rate_schedule,
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
      mask=get_weight_decay_mask([
        "norm",
        "scale",
        "bias",
      ]),
    )
  elif config.opt_type == "lion":
    optimizer = optax.lion(
      learning_rate_schedule,
      b1=config.adam_b1,
      b2=config.adam_b2,
      weight_decay=config.adam_weight_decay,
      mask=get_weight_decay_mask([
        "norm",
        "scale",
        "bias",
      ]),
    )
  elif config.opt_type == "adam_pax":
    optimizer = adam_pax(
      learning_rate_schedule,
      beta1=config.adam_b1,
      beta2=config.adam_b2,
      epsilon=config.adam_eps,
      epsilon_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
    )
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")

  if config.gradient_accumulation_steps > 1:
    optimizer = optax.MultiSteps(
        optimizer, config.gradient_accumulation_steps
    )

  return optimizer, learning_rate_schedule


def tiger_pax(
  learning_rate: optax.Schedule,
  beta: float,
  mu_dtype = None,
  weight_decay: float = 1e-3,
  mask = None,
):
  
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    return optax.ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

  def update_fn(
    updates, 
    state: optax.ScaleByLionState, 
    params=None
  ):
    del params
    mu = optax.update_moment(updates, state.mu, beta, 1)
    mu = utils.cast_tree(mu, mu_dtype)
    updates_new = jax.tree_util.tree_map(lambda m: jnp.sign(m), mu)
    count_inc = optax.safe_int32_increment(state.count)
    return updates_new, optax.ScaleByLionState(count=count_inc, mu=mu)

  return optax.chain(
    optax.GradientTransformation(init_fn, update_fn),
    optax.add_decayed_weights(weight_decay, mask),
    optax.scale_by_learning_rate(learning_rate),
  )
  

def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    ) -> optax.GradientTransformation:
  """Standard Adam optimizer that supports weight decay.

  Follows the implemenation in pax/praxis sharded_adam
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
    mu = jax.tree_util.tree_map(  # First moment
        jnp.zeros_like, params)
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
    t = step.astype(jnp.float32) + 1.
    return decay * (1. - jnp.power(decay, t - 1.)) / (1. - jnp.power(decay, t))

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
      beta1_decay = bias_corrected_decay(count, beta1)
      beta2_decay = bias_corrected_decay(count, beta2)
      mu = (1.0 - beta1_decay) * update + beta1_decay * mu
      nu = (1.0 - beta2_decay) * (update**2) + beta2_decay * nu
      return _slot_opt_state(mu=mu, nu=nu)

    updated_moments = jax.tree_map(_update_momentum, updates, state.mu, state.nu)

    mu = jax.tree_map(lambda x: x.mu, updated_moments)
    nu = jax.tree_map(lambda x: x.nu, updated_moments)

    updates = jax.tree_map(
        lambda mu, nu: mu / (jnp.sqrt(nu + epsilon_root) + epsilon), mu, nu)

    if weight_decay > 0:
      updates = jax.tree_map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
