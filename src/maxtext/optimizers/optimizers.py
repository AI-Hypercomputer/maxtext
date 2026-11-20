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

from flax import nnx
import optax
from optax.contrib._muon import muon as optax_muon
from maxtext.common.common_types import DecoderBlockType
from maxtext.optimizers.muon import muon as maxtext_muon
from maxtext.utils import max_logging
from maxtext.utils.muon_utils import MOE_BLOCK_NAMES
from maxtext.utils.muon_utils import get_muon_weight_dimension_numbers
from maxtext.utils.muon_utils import get_scan_axes
from maxtext.utils.maxtext_muon_utils import get_maxtext_muon_weight_dimension_numbers


def get_path_mask_fn(patterns, match_returns_true=True):
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
  return get_path_mask_fn(config.adamw_mask, match_returns_true=False)


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

    # Ensure non-finite metrics are never treated as valid
    is_finite = jnp.isfinite(loss)
    if grad_norm is not None:
      is_finite = jnp.logical_and(is_finite, jnp.isfinite(grad_norm))

    # Compute rolling stats
    loss_mean, loss_std = _compute_rolling_stats(losses, count, interval)
    grad_norm_mean, grad_norm_std = _compute_rolling_stats(grad_norms, count, interval)

    # Check if the current metrics are within the allowed thresholds
    is_loss_ok = (loss - loss_mean) <= scaling_factor * loss_std
    if grad_norm is not None:
      is_grad_norm_ok = (grad_norm - grad_norm_mean) <= scaling_factor * grad_norm_std
      is_ok = jnp.logical_and(is_loss_ok, is_grad_norm_ok)
    else:
      is_ok = is_loss_ok

    # Only enforce skip if we have at least half the interval filled (or 2 elements minimum).
    # Warmup permits normal variations before statistics stabilize, but never non-finite values.
    min_history = max(2, interval // 2)
    is_warmup = (count + 1) < min_history
    is_ok = jnp.logical_and(is_finite, jnp.logical_or(is_warmup, is_ok))

    # Conditionally execute the inner optimizer to prevent momentum poisoning
    def do_update():
      return inner_opt.update(updates, state["inner_state"], params, **extra_args)

    def skip_update():
      # b/500923599: Investigate logging compatible with jax.jit, jax.lax.cond, and Pathway
      # Match the dtypes `do_update` returns (e.g. bf16 updates from bf16 params while
      # gradients are float32), since both `jax.lax.cond` branches must agree.
      update_avals = jax.eval_shape(lambda: do_update()[0])
      inner_updates = jax.tree_util.tree_map(lambda u, a: jnp.zeros_like(u, dtype=a.dtype), updates, update_avals)
      return inner_updates, state["inner_state"]

    inner_updates, new_inner_state = jax.lax.cond(is_ok, do_update, skip_update)

    # Update rolling buffers (append only finite values so spikes become the new baseline without NaN poisoning)
    idx = count % interval
    new_losses = jnp.where(is_finite, losses.at[idx].set(loss), losses)

    new_grad_norms = grad_norms
    if grad_norm is not None:
      new_grad_norms = jnp.where(is_finite, grad_norms.at[idx].set(grad_norm), grad_norms)

    new_count = jnp.where(is_finite, count + 1, count)

    new_state = {
        "inner_state": new_inner_state,
        "losses": new_losses,
        "grad_norms": new_grad_norms,
        "count": new_count,
        "is_skipped": jnp.logical_not(is_ok),
    }
    return inner_updates, new_state

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_optimizer(config, learning_rate_schedule, model=None, mesh=None):
  """Create optimizer."""
  if config.opt_type == "adamw":
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
  elif config.opt_type == "adam_pax_low_memory":
    if isinstance(model, nnx.Module):
      param_scan_axes = get_param_scan_axes(model)
    elif config.scan_layers:
      raise ValueError(
          "opt_type='adam_pax_low_memory' with scan_layers=True needs the NNX `model` passed to get_optimizer to"
          " find each parameter's layer-scan axes, so that RMS normalization stays within a layer. Pass `model`"
          " or set scan_layers=False."
      )
    else:
      param_scan_axes = None
    if config.bf16_update_rounding != "nearest" and jnp.dtype(config.weight_dtype) != jnp.bfloat16:
      max_logging.warning(
          f"bf16_update_rounding={config.bf16_update_rounding!r} only affects bfloat16 parameters, but"
          f" weight_dtype={config.weight_dtype}; non-bf16 parameters use plain updates."
      )
    # `mu_dtype` defaults to `weight_dtype` (see types.py); only an explicit, different value overrides the
    # params' dtype for the first moment.
    mu_dtype = config.mu_dtype if jnp.dtype(config.mu_dtype) != jnp.dtype(config.weight_dtype) else None
    base_opt = adam_pax_low_memory(
        learning_rate_schedule,
        beta1=config.adam_b1,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=get_adamw_mask(config),
        update_rounding=config.bf16_update_rounding,
        stochastic_rounding_seed=config.stochastic_rounding_seed,
        moe_expert_lr_multiplier=config.moe_expert_lr_multiplier,
        mu_dtype=mu_dtype,
        param_scan_axes=param_scan_axes,
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

  # Apply `trainable_parameters_mask` before `skip_step_on_spikes` so `skip_step_on_spikes`
  # stays the outermost transformation: callers expect `opt_state` to be a top-level dict
  # exposing `opt_state["is_skipped"]` (whereas `optax.multi_transform` wraps state in
  # `PartitionState`), while `multi_transform` still forwards extra_args (`loss`, `grad_norm`)
  # and avoids allocating optimizer state for frozen parameters (`MaskedNode` placeholders in
  # the trainable branch's state and a single `EmptyState` for the frozen branch).
  base_opt = apply_trainable_parameters_mask(base_opt, config)

  if config.skip_step_on_spikes:
    base_opt = skip_step_on_spikes(
        base_opt,
        interval=config.skip_step_interval,
        scaling_factor=config.skip_step_scaling_factor,
    )

  return base_opt


def apply_trainable_parameters_mask(base_opt, config):
  """Freeze every parameter not matched by `config.trainable_parameters_mask`.

  Returns `base_opt` unchanged when the mask is empty, so callers can apply this
  unconditionally. Wrap only the optimizer itself, not a chain that includes
  gradient clipping: in the `pre_train` loop, clipping sees the whole gradient
  tree before updates are zeroed, whereas `MaxTextTrainingEngine` also stops
  gradients on frozen parameters inside its forward-backward kernel so frozen
  parameters are excluded from both gradient computation and `grad_norm`.
  """
  # When trainable_parameters_mask is empty, freeze_mask_fn is None and all parameters are trained.
  trainable_patterns = config.trainable_parameters_mask
  freeze_mask_fn = get_path_mask_fn(trainable_patterns, match_returns_true=False)
  if freeze_mask_fn is None:
    return base_opt

  has_logged = False

  def _label_params(params):
    """Labels every parameter array, and reports what the mask actually did.

    `trainable_parameters_mask` is a whitelist, so expressing "freeze only X"
    needs a negative pattern -- and a pattern that matches nothing freezes the
    whole model. In the `pre_train` loop that failure is otherwise silent
    (gradients are still computed and `grad_norm` looks normal, only the updates
    are zeroed), while `MaxTextTrainingEngine` stops gradients on frozen leaves
    up front. Log the counts once so the intent is checkable against the run log.
    """
    nonlocal has_logged
    labels = jax.tree_util.tree_map(lambda x: "frozen" if x else "trainable", freeze_mask_fn(params))
    if not has_logged:
      has_logged = True
      leaves = jax.tree_util.tree_flatten_with_path(labels)[0]
      frozen = [p for p, v in leaves if v == "frozen"]
      max_logging.log(
          f"trainable_parameters_mask={trainable_patterns}: freezing {len(frozen)}" f" of {len(leaves)} parameter arrays"
      )
      if not frozen:
        max_logging.warning("trainable_parameters_mask froze nothing -- did the pattern match everything?")
      elif len(frozen) == len(leaves):
        max_logging.warning("trainable_parameters_mask froze EVERY parameter -- training is a no-op")
      max_display = 50
      for path in frozen[:max_display]:
        max_logging.log(f"  frozen: {jax.tree_util.keystr(path, simple=True, separator='/')}")
      if len(frozen) > max_display:
        max_logging.log(f"  ... and {len(frozen) - max_display} more frozen parameter arrays")
    return labels

  # Use optax.multi_transform to explicitly map frozen parameters to a stateless set_to_zero() optimizer.
  # If we simply wrapped base_opt in optax.masked() or chained it, Optax would still allocate
  # massive states (momentum, variance) for the entire model before zeroing the updates.
  # By using multi_transform, only the trainable parameters get states allocated.
  return optax.multi_transform(
      {"trainable": base_opt, "frozen": optax.set_to_zero()},
      _label_params,
  )


def _stochastic_round_update(u, p, step_size, key):
  """Returns a bf16 update d with p + d == SR_bf16(p + step_size * u) whenever |step| <= |p|.

  Stochastic rounding keeps tiny updates (|lr * u| < half a bf16 ulp) from being
  silently rounded away; E[p_new] equals the exact fp32 result. Zero weights
  with a zero update (2:4-masked entries) stay exactly zero.

  The update is returned in bf16 so no param-sized float32 tree is materialized.
  The bf16 delta is exact whenever the old and new weights are within a factor of
  2 of each other (Sterbenz), which covers every sub-ULP or few-ULP step. Only a
  step larger than the weight itself (|p| near 0) can land one bf16 ULP of the new
  value away from the stochastically rounded target.
  """
  if p.dtype != jnp.bfloat16:
    return (step_size * u).astype(u.dtype)
  p32 = p.astype(jnp.float32)
  new32 = p32 + jnp.asarray(step_size, jnp.float32) * u.astype(jnp.float32)
  bits = jax.lax.bitcast_convert_type(new32, jnp.uint32)
  noise = jax.random.bits(key, new32.shape, jnp.uint32) & jnp.uint32(0xFFFF)
  bits = (bits + noise) & jnp.uint32(0xFFFF0000)
  new_bf16 = jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)
  return (new_bf16.astype(jnp.float32) - p32).astype(p.dtype)


def _kahan_update(u, r, p, step_size):
  """Returns (bf16 update, new residual) for a Kahan-compensated bf16 update of `p`.

  The residual is computed from the weight `p + update` actually reaches when the
  update is added in bf16 (as `optax.apply_updates` does), so nothing is lost even
  when the bf16 delta cannot represent the exact step. Weights that are zero with a
  zero update (e.g. N:M-masked entries) stay zero with a zero residual; any other
  weight keeps carrying its residual.
  """
  p32 = p.astype(jnp.float32)
  tot32 = r.astype(jnp.float32) + jnp.asarray(step_size, jnp.float32) * u.astype(jnp.float32)
  update = ((p32 + tot32).astype(jnp.bfloat16).astype(jnp.float32) - p32).astype(p.dtype)
  applied32 = (p + update).astype(p.dtype).astype(jnp.float32) - p32
  new_r = (tot32 - applied32).astype(r.dtype)
  masked = (p == 0) & (u == 0)
  return jnp.where(masked, jnp.zeros_like(update), update), jnp.where(masked, jnp.zeros_like(new_r), new_r)


_ROUTED_MOE_EXPERT_WEIGHT_NAMES = frozenset(("wi", "wi_0", "wi_1", "wo"))
_SHARED_EXPERT_NAMES = frozenset(("shared_experts", "shared_expert"))


def _key_names(path) -> tuple[str, ...]:
  return tuple(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))) for k in path)


def _is_routed_moe_expert_path(path) -> bool:
  """Returns True for routed MoE expert weights (`wi`, `wi_0`, `wi_1`, `wo` in a MoE block, not shared experts)."""
  names = _key_names(path)
  return (
      any(n in MOE_BLOCK_NAMES for n in names)
      and not any(n in _SHARED_EXPERT_NAMES for n in names)
      and any(n in _ROUTED_MOE_EXPERT_WEIGHT_NAMES for n in names)
  )


def _dict_key_path(path) -> tuple[str, ...]:
  """Returns the dict keys of a pytree key path, dropping attribute and sequence entries."""
  return tuple(str(k.key) for k in path if isinstance(k, jax.tree_util.DictKey))


def get_param_scan_axes(model) -> dict[tuple[str, ...], tuple[int, ...]]:
  """Maps the dict-key path of each scanned `nnx.Param` of `model` to its layer-scan axes."""
  _, abstract_params, _ = nnx.split(model, nnx.Param, ...)
  leaves = jax.tree_util.tree_flatten_with_path(abstract_params, is_leaf=lambda x: isinstance(x, nnx.Variable))[0]
  return {_dict_key_path(path): axes for path, variable in leaves if (axes := get_scan_axes(variable))}


def _rms_reduction_axis(ndim: int, scan_axes: tuple[int, ...]):
  """Returns the axis the RMS is taken over: the last non-scan axis.

  Scalars use their magnitude (None reduces over everything), and a stack of
  scalars (only scan axes) normalizes each element by its own magnitude (`()`).
  """
  if ndim == 0:
    return None
  feature_axes = [axis for axis in range(ndim) if axis not in scan_axes]
  return feature_axes[-1] if feature_axes else ()


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


def adam_pax_low_memory(
    learning_rate_fn: optax.ScalarOrSchedule,
    beta1: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    mask=None,
    update_rounding: str = "nearest",
    stochastic_rounding_seed: int = 0,
    moe_expert_lr_multiplier: float = 1.0,
    mu_dtype=None,
    param_scan_axes: dict[tuple[str, ...], tuple[int, ...]] | None = None,
) -> optax.GradientTransformation:
  """Low-memory single-moment variant of `adam_pax` for bf16 parameters.

  Instead of a per-parameter second moment, each gradient is normalized on the
  fly by its RMS over its last feature axis (for MaxText `[in, out]` kernels this
  is the output dimension, i.e. one scale per input row; 1-D params use one scale
  per layer and scalars their magnitude). Layer-scan axes are never reduced over,
  so scanned and unscanned layers are normalized the same way. Only the first
  moment of the normalized gradient is kept. Optimizer state is the first moment
  plus, for `update_rounding="kahan"`, a compensation residual with the bf16
  params' shape and dtype.

  The state is an `optax.ScaleByAdamState` so it checkpoints like `adam_pax`;
  `nu` holds the Kahan residual (or 0-d placeholders where unused) rather than a
  second moment.

  Args:
    learning_rate_fn: the learning rate, either a scalar or a callable that given
      the current training step returns it.
    beta1: decay rate to track the first moment.
    epsilon: Small constant added to the RMS denominator (outside the sqrt).
    epsilon_root: Small constant added inside the square root of the RMS.
    weight_decay: If > 0, weight decay to apply.
    mask: Optional weight decay mask (tree or callable).
    update_rounding: How bf16 parameter updates are rounded: "nearest" (default
      round-to-nearest), "kahan" (deterministic Kahan-compensated summation that
      carries the sub-ULP residual in optimizer state), or "stochastic"
      (unbiased stochastic rounding).
    stochastic_rounding_seed: PRNG seed for `update_rounding="stochastic"`.
    moe_expert_lr_multiplier: Multiplier on the learning rate of routed MoE
      expert weights (`wi`, `wi_0`, `wi_1`, `wo` under a MoE block, excluding
      shared experts), applied before rounding. 1.0 leaves all updates unchanged.
    mu_dtype: Optional dtype for the first moment; defaults to each param's dtype.
    param_scan_axes: Optional map from a param's dict-key path to its layer-scan
      axes (see `get_param_scan_axes`); unlisted params have none.

  Returns:
    A `optax.GradientTransformation`.
  """
  if update_rounding not in ("nearest", "kahan", "stochastic"):
    raise ValueError(f"Unsupported update_rounding: {update_rounding!r}")
  use_kahan = update_rounding == "kahan"

  def init_fn(params):
    mu = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=mu_dtype), params)
    if use_kahan:
      # `nu` holds the Kahan compensation residual for bf16 params (same shape and
      # dtype as the param, so it is sharded like it); other params never use it.
      nu = jax.tree_util.tree_map(
          lambda p: jnp.zeros_like(p) if p.dtype == jnp.bfloat16 else jnp.zeros([], jnp.float32), params
      )
    else:
      nu = jnp.zeros([], jnp.float32)  # 0-d placeholder; no per-param state.
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    t = step.astype(jnp.float32) + 1.0
    return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))

  def update_fn(updates, state, params=None):
    if weight_decay > 0 or update_rounding != "nearest":
      assert params is not None
    count = state.count
    num_scan_matches = 0

    def _update_mu_rms(path, update, mu):
      nonlocal num_scan_matches
      scan_axes = ()
      if param_scan_axes:
        scan_axes = param_scan_axes.get(_dict_key_path(path), ())
        num_scan_matches += bool(scan_axes)
      u_f32 = update.astype(jnp.float32)
      axis = _rms_reduction_axis(u_f32.ndim, scan_axes)
      rms = jnp.sqrt(jnp.mean(jnp.square(u_f32), axis=axis, keepdims=True) + epsilon_root) + epsilon
      # The EMA runs in float32 (a bf16 `beta1` is off by up to ~17% in `1 - beta1`); only `mu` is stored narrow.
      beta1_decay = bias_corrected_decay(count, beta1)
      return ((1.0 - beta1_decay) * (u_f32 / rms) + beta1_decay * mu.astype(jnp.float32)).astype(mu.dtype)

    mu = jax.tree_util.tree_map_with_path(_update_mu_rms, updates, state.mu)
    if param_scan_axes and not num_scan_matches:
      max_logging.warning(
          "adam_pax_low_memory: none of the model's scanned parameter paths (e.g."
          f" {next(iter(param_scan_axes))}) match a trainable update, so no layer-scan axes were applied."
      )
    nu = state.nu
    updates = mu

    if weight_decay > 0:
      if mask is not None:
        mask_tree = mask(params) if callable(mask) else mask
        updates = jax.tree_util.tree_map(
            lambda x, v, m: x + weight_decay * v.astype(x.dtype) if m else x, updates, params, mask_tree
        )
      else:
        updates = jax.tree_util.tree_map(lambda x, v: x + weight_decay * v.astype(x.dtype), updates, params)

    step_size = -1.0 * (learning_rate_fn(count) if callable(learning_rate_fn) else learning_rate_fn)

    u_paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(updates)
    u_leaves = [u for _, u in u_paths_and_leaves]
    if moe_expert_lr_multiplier != 1.0:
      is_expert = [_is_routed_moe_expert_path(path) for path, _ in u_paths_and_leaves]
      if not any(is_expert):
        max_logging.warning(
            f"moe_expert_lr_multiplier={moe_expert_lr_multiplier} matched no routed MoE expert weights; it has no"
            " effect."
        )
      step_sizes = [step_size * moe_expert_lr_multiplier if e else step_size for e in is_expert]
    else:
      step_sizes = [step_size] * len(u_leaves)

    if update_rounding == "stochastic":
      base_key = jax.random.fold_in(jax.random.PRNGKey(stochastic_rounding_seed), count)
      p_leaves = treedef.flatten_up_to(params)
      updates = jax.tree_util.tree_unflatten(
          treedef,
          [
              _stochastic_round_update(u, p, s, jax.random.fold_in(base_key, i))
              for i, (u, p, s) in enumerate(zip(u_leaves, p_leaves, step_sizes))
          ],
      )
    elif use_kahan:
      r_leaves = treedef.flatten_up_to(state.nu)
      p_leaves = treedef.flatten_up_to(params)
      new_u_leaves = []
      new_r_leaves = []
      for u, r, p, s in zip(u_leaves, r_leaves, p_leaves, step_sizes):
        if p.dtype != jnp.bfloat16:
          new_u_leaves.append((s * u).astype(u.dtype))
          new_r_leaves.append(r)
        else:
          new_u, new_r = _kahan_update(u, r, p, s)
          new_u_leaves.append(new_u)
          new_r_leaves.append(new_r)
      updates = jax.tree_util.tree_unflatten(treedef, new_u_leaves)
      nu = jax.tree_util.tree_unflatten(treedef, new_r_leaves)
    else:
      updates = jax.tree_util.tree_unflatten(treedef, [(s * u).astype(u.dtype) for u, s in zip(u_leaves, step_sizes)])

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
