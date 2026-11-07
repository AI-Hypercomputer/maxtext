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
"""Utils that are only interesting to MaxText."""

import re
import jax
import jax.numpy as jnp

import optax
from optax.contrib._muon import muon as optax_muon
from maxtext.common.common_types import DecoderBlockType
from maxtext.models import olmoe3_roles
from maxtext.optimizers import completedp
from maxtext.optimizers import mup
from maxtext.optimizers.muon import muon as maxtext_muon
from maxtext.utils import max_logging
from maxtext.utils.muon_utils import get_muon_weight_dimension_numbers
from maxtext.utils.maxtext_muon_utils import get_maxtext_muon_weight_dimension_numbers


def _get_path_mask_fn(patterns, match_returns_true=True):
  """Helper to create a mask function from a list of regex patterns."""
  if not patterns:
    return None

  compiled_patterns = [re.compile(pattern) for pattern in patterns]

  def mask_fn(params):
    def _is_masked(path, _):
      # Join path keys into a single string for pattern matching (e.g., "layer1/bias")
      path_str = jax.tree_util.keystr(path, simple=True, separator="/")
      matched = any(pattern.search(path_str) for pattern in compiled_patterns)
      return matched if match_returns_true else not matched

    return jax.tree_util.tree_map_with_path(_is_masked, params)

  return mask_fn


def get_adamw_mask(config):
  """Create a mask function for AdamW optimizer to exclude certain parameters from weight decay."""
  patterns = list(getattr(config, "adamw_mask", None) or [])
  if getattr(config, "use_multimodal", False) and (
      getattr(getattr(config, "vision_encoder_block", None), "value", str(getattr(config, "vision_encoder_block", "")))
      == "olmo35"
  ):
    patterns.extend(["OLMo35VisionEncoder", "OLMo35VisionConnector"])
  if getattr(config, "olmo_trainable_embedding_row_range", None):
    patterns.append("token_embedder")
  return _get_path_mask_fn(patterns, match_returns_true=False)


def _scale_olmo35_vision_updates(vis_scale: float, conn_scale: float) -> optax.GradientTransformation:
  """Scales parameter updates for OLMo35VisionEncoder and OLMo35VisionConnector (both have wd=0)."""

  def init_fn(params):
    del params
    return optax.EmptyState()

  def update_fn(updates, state, params=None):
    del params

    def _scale_leaf(path, u):
      if u is None:
        return None
      p_str = jax.tree_util.keystr(path, simple=True, separator="/")
      if "OLMo35VisionConnector" in p_str:
        return u if conn_scale == 1.0 else (u * jnp.asarray(conn_scale, dtype=u.dtype))
      if "OLMo35VisionEncoder" in p_str:
        return u if vis_scale == 1.0 else (u * jnp.asarray(vis_scale, dtype=u.dtype))
      return u

    return jax.tree_util.tree_map_with_path(_scale_leaf, updates), state

  return optax.GradientTransformation(init_fn, update_fn)


def _compute_rolling_stats(arr: jax.Array, count: jax.Array, interval: int):
  """Computes mean and unbiased std (Bessel's correction) over a rolling window."""
  valid_elements = jnp.minimum(count, interval)
  safe_elements = jnp.maximum(1, valid_elements)
  mask = jnp.arange(interval) < valid_elements

  mean = jnp.sum(jnp.where(mask, arr, 0.0)) / safe_elements
  sq_diff = jnp.where(mask, (arr - mean) ** 2, 0.0)

  # Use Bessel's correction (N - 1) for unbiased variance to align with torch.std
  variance = jnp.sum(sq_diff) / jnp.maximum(1, valid_elements - 1)
  std = jnp.sqrt(variance)
  return mean, std


def skip_step_on_spikes(
    inner_opt: optax.GradientTransformation, interval: int, scaling_factor: float
) -> optax.GradientTransformationExtraArgs:
  """Wrapper that skips updates when loss or grad_norm spike.

  This wrapper calculates a rolling mean and standard deviation (using
  Bessel's correction) over the last `interval` steps for both the loss
  and the gradient norm. If the current step's loss or gradient norm
  exceeds `mean + scaling_factor * std`, the update is zeroed and the
  optimizer state is not advanced, effectively skipping the step.

  Reference implementation:
  https://github.com/allenai/OLMo-core/blob/c757b7c3c15197154c753d883330afbfa4869dcc/src/olmo_core/optim/skip_step_optimizer.py#L12

  Args:
    inner_opt: The inner Optax gradient transformation to wrap.
    interval: The number of recent steps to use for calculating mean and std.
    scaling_factor: The multiplier for standard deviation to set the spike threshold.

  Returns:
    An optax.GradientTransformationExtraArgs that skips spikes.
  """

  def init_fn(params):
    return {
        "inner_state": inner_opt.init(params),
        "losses": jnp.zeros(interval, dtype=jnp.float32),
        "grad_norms": jnp.zeros(interval, dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.int32),
        "is_skipped": jnp.array(False, dtype=jnp.bool_),
    }

  def update_fn(updates, state, params=None, **extra_args):
    # Using `pop()` removes `loss` and `grad_norm` from `extra_args` before they are
    # passed downstream to `inner_opt.update()`. This prevents `TypeError` if the
    # inner optimizer doesn't explicitly accept these as `kwargs`.
    loss = extra_args.pop("loss", None)
    grad_norm = extra_args.pop("grad_norm", None)

    # Fallback to standard update if loss is not provided
    if loss is None:
      inner_updates, new_inner_state = inner_opt.update(updates, state["inner_state"], params, **extra_args)
      return inner_updates, {
          "inner_state": new_inner_state,
          "losses": state["losses"],
          "grad_norms": state["grad_norms"],
          "count": state["count"],
          "is_skipped": jnp.array(False, dtype=jnp.bool_),
      }

    count = state["count"]
    losses = state["losses"]
    grad_norms = state["grad_norms"]

    # Compute rolling stats
    loss_mean, loss_std = _compute_rolling_stats(losses, count, interval)
    grad_norm_mean, grad_norm_std = _compute_rolling_stats(grad_norms, count, interval)

    # Check if the current metrics are finite and within the allowed thresholds.
    # Non-finite loss or grad_norm (e.g. fp32 norm overflow on untrained embedding rows) must
    # ALWAYS be skipped even during warmup, and must never be written into the rolling buffers
    # where `inf`/`nan` would poison `mean`/`std` for the next `interval` steps.
    loss_f32 = jnp.asarray(loss, dtype=jnp.float32)
    is_finite = jnp.isfinite(loss_f32)
    is_loss_ok = (loss_f32 - loss_mean) <= scaling_factor * loss_std
    if grad_norm is not None:
      gn_f32 = jnp.asarray(grad_norm, dtype=jnp.float32)
      is_finite = jnp.logical_and(is_finite, jnp.isfinite(gn_f32))
      is_grad_norm_ok = (gn_f32 - grad_norm_mean) <= scaling_factor * grad_norm_std
      is_ok = jnp.logical_and(is_loss_ok, is_grad_norm_ok)
    else:
      gn_f32 = None
      is_ok = is_loss_ok

    # Only enforce statistical spike skip if we have at least half the interval filled (or 2 elements minimum),
    # but ALWAYS skip non-finite steps even during warmup.
    min_history = max(2, interval // 2)
    is_warmup = count < min_history
    is_ok = jnp.logical_and(is_finite, jnp.logical_or(is_warmup, is_ok))

    # Conditionally execute the inner optimizer to prevent momentum poisoning
    def do_update():
      return inner_opt.update(updates, state["inner_state"], params, **extra_args)

    out_updates_struct = jax.eval_shape(do_update)[0]

    def skip_update():
      # b/500923599: Investigate logging compatible with jax.jit, jax.lax.cond, and Pathway
      # Match `do_update`'s exact leaf dtypes (e.g. float32 master updates when incoming grads are bfloat16).
      inner_updates = jax.tree_util.tree_map(
          lambda u, s: jnp.zeros_like(u, dtype=s.dtype), updates, out_updates_struct
      )
      return inner_updates, state["inner_state"]

    inner_updates, new_inner_state = jax.lax.cond(is_ok, do_update, skip_update)

    # Update rolling buffers only on finite steps (append finite spikes so regime shifts can adapt,
    # while leaving rolling history and valid sample count untouched when `inf`/`nan` occurs).
    idx = count % interval
    safe_loss = jnp.where(is_finite, loss_f32, losses[idx])
    new_losses = losses.at[idx].set(safe_loss)

    new_grad_norms = grad_norms
    if gn_f32 is not None:
      safe_gn = jnp.where(is_finite, gn_f32, grad_norms[idx])
      new_grad_norms = grad_norms.at[idx].set(safe_gn)

    new_state = {
        "inner_state": new_inner_state,
        "losses": new_losses,
        "grad_norms": new_grad_norms,
        "count": jnp.where(is_finite, count + 1, count),
        "is_skipped": jnp.logical_not(is_ok),
    }
    return inner_updates, new_state

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_optimizer(config, learning_rate_schedule, model=None, mesh=None):
  """Create optimizer."""
  if getattr(config, "completedp_base_model_name", ""):
    completedp.check_supported(config)
    base = olmoe3_roles.load_base_config(config.completedp_base_model_name)
    base_opt = completedp.completedp_adamw(learning_rate_schedule, config, base, mask=get_adamw_mask(config))
  elif getattr(config, "mup_base_model_name", ""):
    mup.check_supported(config)
    if config.olmo_init_method != "nanochat":
      max_logging.warning(f"muP learning rates with olmo_init_method={config.olmo_init_method}; muP expects nanochat.")
    base_opt = mup.mup_adamw(learning_rate_schedule, config, mup.load_base_config(config), mask=get_adamw_mask(config))
  elif config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    base_opt = optax.adamw(
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
    base_opt = adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "sgd":
    base_opt = optax.sgd(learning_rate_schedule)
  elif config.opt_type == "muon":
    muon_type = getattr(config, "muon_type", "optax_muon")
    if config.decoder_block == DecoderBlockType.DEEPSEEK4:
      ns_coeffs = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2
      ns_steps = 10
    else:
      ns_coeffs = (3.4445, -4.7750, 2.0315)
      ns_steps = 5

    muon_kwargs = {
        # Shared parameters: "nesterov" uses default
        "learning_rate": learning_rate_schedule,
        "eps": config.adam_eps,
        "mu_dtype": config.mu_dtype,
        # Muon-specific parameters: "weight_decay_mask", "adaptive" uses default
        "beta": config.muon_beta,
        "weight_decay": config.muon_weight_decay,
        "consistent_rms": config.muon_consistent_rms,
        "ns_coeffs": ns_coeffs,
        "ns_steps": ns_steps,
        # AdamW-specific parameters
        "adam_b1": config.adam_b1,
        "adam_b2": config.adam_b2,
        "adam_eps_root": config.adam_eps_root,
        "adam_weight_decay": config.adam_weight_decay,
    }

    if muon_type in ("optax", "optax_muon"):
      assert not getattr(
          config, "muon_use_all_to_all", False
      ), "all-to-all communication in muon is only supported with maxtext_muon, not optax_muon."
      if model is not None:
        muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config)
      else:
        raise ValueError("Please specify model to extract muon dimension number.")
      muon_kwargs = muon_kwargs | {"muon_weight_dimension_numbers": muon_weight_dimension_numbers}
      base_opt = optax_muon(**muon_kwargs)  # pyrefly: ignore[bad-argument-type]
    elif muon_type in ("maxtext", "maxtext_muon"):
      if model is not None:
        muon_weight_dimension_numbers = get_maxtext_muon_weight_dimension_numbers(model, config, mesh=mesh)
      else:
        raise ValueError("Please specify model to extract muon dimension number.")

      use_all_to_all = getattr(config, "muon_use_all_to_all", True)
      muon_kwargs = muon_kwargs | {
          "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
          "use_all_to_all": use_all_to_all,
      }
      base_opt = maxtext_muon(**muon_kwargs)  # pyrefly: ignore[bad-argument-type]
    else:
      raise ValueError(
          f"Unsupported muon_type: {muon_type}. Must be 'optax_muon' (or 'optax') or 'maxtext_muon' (or 'maxtext')."
      )
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")

  is_olmo_mm = getattr(config, "use_multimodal", False) and (
      getattr(getattr(config, "vision_encoder_block", None), "value", str(getattr(config, "vision_encoder_block", "")))
      == "olmo35"
  )
  if is_olmo_mm and not getattr(config, "completedp_base_model_name", ""):
    vis_scale = float(getattr(config, "olmo_vision_lr_scale", 0.1))
    conn_scale = float(getattr(config, "olmo_connector_lr_scale", 2.0))
    if vis_scale != 1.0 or conn_scale != 1.0:
      base_opt = optax.chain(base_opt, _scale_olmo35_vision_updates(vis_scale, conn_scale))

  if getattr(config, "skip_step_on_spikes", False):
    base_opt = skip_step_on_spikes(
        base_opt,
        interval=config.skip_step_interval,
        scaling_factor=config.skip_step_scaling_factor,
    )

  # Parameter freezing: support both whitelist (`trainable_parameters_mask`) and blacklist (`frozen_parameters_mask`,
  # plus `OLMo35VisionEncoder` when `freeze_vision_encoder_params` is True).
  trainable_patterns = list(getattr(config, "trainable_parameters_mask", None) or [])
  frozen_patterns = list(getattr(config, "frozen_parameters_mask", None) or [])
  if is_olmo_mm and getattr(config, "freeze_vision_encoder_params", False):
    frozen_patterns.append("OLMo35VisionEncoder")

  if trainable_patterns or frozen_patterns:
    compiled_trainable = [re.compile(p) for p in trainable_patterns]
    compiled_frozen = [re.compile(p) for p in frozen_patterns]

    def _param_label_tree(params):
      def _label(path, _):
        p_str = jax.tree_util.keystr(path, simple=True, separator="/")
        if compiled_trainable and not any(p.search(p_str) for p in compiled_trainable):
          return "frozen"
        if compiled_frozen and any(p.search(p_str) for p in compiled_frozen):
          return "frozen"
        return "trainable"

      return jax.tree_util.tree_map_with_path(_label, params)

    # Use optax.multi_transform to explicitly map frozen parameters to a stateless set_to_zero() optimizer
    # so frozen parameters do not allocate momentum/variance state in HBM.
    return optax.multi_transform(
        {"trainable": base_opt, "frozen": optax.set_to_zero()},
        _param_label_tree,
    )

  return base_opt


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

    # learning_rate_fn may be a callable schedule or a scalar (e.g. when wrapped
    # by optax.inject_hyperparams, it is passed as a pre-evaluated scalar).
    step_size = -1.0 * (learning_rate_fn(count) if callable(learning_rate_fn) else learning_rate_fn)
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
