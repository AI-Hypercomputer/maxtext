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

"""Loss, learning-rate schedule, AdamW and the train step -- all in raw JAX.

Everything here reproduces MaxText's `trainers/pre_train/train.py` +
`optimizers/optimizers.py` + `maxtext_utils.create_learning_rate_schedule`
for the DeepSeek-V3 target config, without optax or Flax:

  * the loss is the T5X custom-VJP cross entropy (`max_utils.cross_entropy_with_logits`),
  * the optimizer is `optax.adamw`'s exact update rule, hand-written,
  * gradients are clipped by `optax.clip_by_global_norm(1.0)`'s exact rule.
"""

import dataclasses

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek import model as ds_model
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import sharding_mode as mode

EPS = 1e-8  # MaxText's globals.EPS


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


@jax.custom_vjp
def cross_entropy_with_logits(logits, targets, z_loss=0.0):
  """`max_utils.cross_entropy_with_logits`, the T5X kernel.

  This un-differentiated primal is mathematically the same as `_fwd` below but
  can round differently in the last ulp; under `value_and_grad` -- which is how
  the train step always calls it -- only `_fwd` runs.
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_fwd(logits, targets, z_loss=0.0):
  """`max_utils._cross_entropy_with_logits_fwd` -- max-subtracted, and `loss` includes the z-loss."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp, log_z)


def _cross_entropy_bwd(res, g):
  """`max_utils._cross_entropy_with_logits_bwd` -- the z-loss cotangent is discarded.

  Under explicit sharding the cotangent must be typed exactly like `logits`.
  Everything here is elementwise, so it comes out typed by its operands, and
  those can disagree with `logits`: replicated logits against batch-sharded
  targets give a batch-sharded `deriv`, and jax refuses the rule.  So it is
  retyped from `logits`' own aval (`mode.retype_like`), which costs nothing
  when the types already agree and emits nothing under auto.
  """
  g = g[0]
  logits, targets, z_loss, exp_shifted, sum_exp, log_z = res
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  return mode.retype_like(jnp.asarray(g_logits, logits.dtype), logits), None, None


cross_entropy_with_logits.defvjp(_cross_entropy_fwd, _cross_entropy_bwd)


def loss_from_logits(logits, targets, targets_segmentation, vocab_size, z_loss_multiplier=0.0):
  """The loss of MaxText's `train.loss_fn` (trainers/pre_train/train.py).  Returns `(loss, metrics)`."""
  one_hot_targets = jax.nn.one_hot(targets, vocab_size)  # float32
  xent, z_loss = cross_entropy_with_logits(logits, one_hot_targets, z_loss_multiplier)

  weights = targets_segmentation != 0
  xent = xent * weights
  z_loss = z_loss * weights

  xent_sum = jnp.sum(xent)
  total_weights = jnp.sum(weights)
  loss = xent_sum / (total_weights + EPS)
  metrics = {
      "lm_loss": loss,
      "z_loss": jnp.sum(z_loss) / (total_weights + EPS),
      "total_weights": total_weights,
  }
  return loss, metrics


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TrainConfig:
  """MaxText's optimizer/schedule knobs (base.yml defaults for deepseek3)."""

  steps: int = 1
  learning_rate: float = 3.0e-5
  learning_rate_final_fraction: float = 0.1
  learning_rate_schedule_steps: int = -1  # -1 -> steps
  warmup_steps_fraction: float = 0.1
  lr_schedule_type: str = "cosine"
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1.0e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  mu_dtype: jnp.dtype = jnp.float32
  gradient_clipping_threshold: float = 1.0
  z_loss_multiplier: float = 0.0

  @property
  def schedule_steps(self) -> int:
    return self.steps if self.learning_rate_schedule_steps < 0 else self.learning_rate_schedule_steps


def _linear_schedule(init_value, end_value, transition_steps):
  """`optax.linear_schedule` == `polynomial_schedule(..., power=1)`."""

  def schedule(step):
    count = jnp.clip(step, 0, transition_steps)
    frac = 1 - count / transition_steps
    return (init_value - end_value) * frac + end_value

  return schedule


def _cosine_schedule(init_lr, final_lr, len_steps):
  """`maxtext_utils.create_learning_rate_schedule`'s `make_cos_schedule` -- note the `len_steps - 1` denominator."""

  def schedule(step):
    pct = step / (len_steps - 1) if len_steps > 1 else 1.0
    a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
    return init_lr * a + final_lr * (1 - a)

  return schedule


def learning_rate_schedule(cfg: TrainConfig):
  """`maxtext_utils.create_learning_rate_schedule` + `optax.join_schedules`."""
  lr = cfg.learning_rate
  final_lr = lr * cfg.learning_rate_final_fraction
  sched_steps = cfg.schedule_steps
  warmup_steps = int(sched_steps * cfg.warmup_steps_fraction)
  constant_zero_steps = cfg.steps - sched_steps

  pieces, boundaries = [], []
  if warmup_steps > 0:
    pieces.append(_linear_schedule(0.0, lr, warmup_steps))
    boundaries.append(warmup_steps)

  if cfg.lr_schedule_type == "cosine":
    cos_steps = sched_steps - warmup_steps
    if cos_steps > 0:
      pieces.append(_cosine_schedule(lr, final_lr, cos_steps))
      boundaries.append(warmup_steps + cos_steps)
  else:
    raise ValueError(f"unsupported lr_schedule_type: {cfg.lr_schedule_type}")

  if constant_zero_steps > 0:
    pieces.append(lambda _: jnp.asarray(0.0, jnp.float32))
    boundaries.append(sched_steps)

  def schedule(step):
    out = pieces[0](step)
    for boundary, piece in zip(boundaries, pieces[1:]):
      out = jnp.where(step < boundary, out, piece(step - boundary))
    return out

  return schedule


# --------------------------------------------------------------------------- #
# AdamW (optax.adamw's update rule, written out)
# --------------------------------------------------------------------------- #


def adamw_init(params, cfg: TrainConfig):
  return {
      "count": jnp.zeros([], jnp.int32),  # scale_by_adam
      "sched_count": jnp.zeros([], jnp.int32),  # scale_by_schedule
      "mu": jax.tree.map(lambda p: jnp.zeros_like(p, cfg.mu_dtype), params),
      "nu": jax.tree.map(jnp.zeros_like, params),
  }


def adamw_update(grads, state, params, cfg: TrainConfig, schedule):
  """`chain(scale_by_adam, add_decayed_weights, scale_by_learning_rate)`."""
  b1, b2 = cfg.adam_b1, cfg.adam_b2
  count = state["count"] + 1

  mu = jax.tree.map(lambda g, m: (1 - b1) * g + b1 * m, grads, state["mu"])
  nu = jax.tree.map(lambda g, n: (1 - b2) * (g**2) + b2 * n, grads, state["nu"])

  mu_correction = 1 - b1**count
  nu_correction = 1 - b2**count
  mu_hat = jax.tree.map(lambda t: t / mu_correction.astype(t.dtype), mu)
  nu_hat = jax.tree.map(lambda t: t / nu_correction.astype(t.dtype), nu)

  updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v + cfg.adam_eps_root) + cfg.adam_eps), mu_hat, nu_hat)
  mu = jax.tree.map(lambda t: t.astype(cfg.mu_dtype), mu)

  # add_decayed_weights (decoupled weight decay, applied before the LR scale).
  updates = jax.tree.map(lambda u, p: u + cfg.adam_weight_decay * p, updates, params)

  # scale_by_learning_rate(schedule, flip_sign=True) -- its own count, from 0.
  step_size = -1 * schedule(state["sched_count"])
  updates = jax.tree.map(lambda u: jnp.array(step_size, dtype=u.dtype) * u, updates)

  new_state = {
      "count": count,
      "sched_count": state["sched_count"] + 1,
      "mu": mu,
      "nu": nu,
  }
  return updates, new_state


def global_norm(tree):
  """`optax.tree.norm` / `max_utils.l2norm_pytree`."""
  return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))


def clip_by_global_norm(grads, max_norm):
  """`optax.clip_by_global_norm` -- a `lax.select` identity below the threshold."""
  g_norm = global_norm(grads)
  trigger = jnp.squeeze(g_norm < max_norm)

  def clip(t):
    return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

  return jax.tree.map(clip, grads)


def apply_updates(params, updates):
  return jax.tree.map(lambda p, u: (p + u).astype(p.dtype), params, updates)


# --------------------------------------------------------------------------- #
# Train step
# --------------------------------------------------------------------------- #


def compute_loss(params, aux, batch, config: DeepSeekConfig, train_cfg: TrainConfig, freqs_cis=None):
  # MaxText's loss adds the MTP layers' (multi_token_prediction.py), which this port does not build (ds_ep_pipeline
  # --mtp does): refused, as model.decoder_layer refuses the aux loss and the bias update, not trained without.
  if config.mtp_num_layers > 0:
    raise NotImplementedError(
        f"mtp_num_layers={config.mtp_num_layers} is not supported: MaxText's loss adds the MTP "
        "layers' cross entropy, which this port does not compute"
    )
  logits = ds_model.forward(
      params,
      aux,
      batch["inputs"],
      batch["inputs_position"],
      batch["inputs_segmentation"],
      config,
      freqs_cis=freqs_cis,
  )
  loss, metrics = loss_from_logits(
      logits,
      batch["targets"],
      batch["targets_segmentation"],
      config.vocab_size,
      train_cfg.z_loss_multiplier,
  )
  return loss, metrics


def train_step(params, aux, opt_state, batch, config, train_cfg, schedule, freqs_cis=None):
  """One MaxText training step: loss -> grads -> clip -> AdamW -> apply."""
  grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
  (loss, metrics), raw_grads = grad_fn(params, aux, batch, config, train_cfg, freqs_cis)

  grads = clip_by_global_norm(raw_grads, train_cfg.gradient_clipping_threshold)
  updates, opt_state = adamw_update(grads, opt_state, params, train_cfg, schedule)
  params = apply_updates(params, updates)

  metrics = dict(metrics)
  metrics["loss"] = loss
  metrics["raw_grad_norm"] = global_norm(raw_grads)
  metrics["grad_norm"] = global_norm(grads)
  metrics["param_norm"] = global_norm(params)
  return params, opt_state, metrics


def packed_segment_ids(batch_size, seq_len, max_segments_per_seq):
  """`synthetic_data_processing._make_packed_segment_ids`."""
  if max_segments_per_seq < 2 or seq_len < 2:
    return jnp.ones((batch_size, seq_len), jnp.int32)

  max_segs = min(max_segments_per_seq, seq_len)
  k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
  n_segs = jax.random.randint(k1, (batch_size, 1), minval=2, maxval=max_segs + 1)
  num_splits = n_segs - 1
  scores = jax.random.uniform(k2, (batch_size, seq_len - 1))
  ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
  split_mask = ranks < num_splits
  first_pos = jnp.ones((batch_size, 1), jnp.bool_)
  is_boundary = jnp.concatenate([first_pos, split_mask], axis=-1)
  return jnp.cumsum(is_boundary, axis=-1, dtype=jnp.int32)


def synthetic_batch(vocab_size, batch_size, seq_len, seed=0, packing=True, max_segments_per_seq=-1):
  """`synthetic_data_processing.SyntheticDataIterator.raw_generate_synthetic_data`.

  Note the hardcoded `PRNGKey(0)` and that targets are already next-token
  shifted -- neither the model nor the loss shifts anything.
  """
  tokens = jax.random.randint(jax.random.PRNGKey(seed), (batch_size, seq_len + 1), 0, vocab_size, dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(0, seq_len + 1, dtype=jnp.int32), (batch_size, seq_len + 1))
  if packing:
    segmentation = packed_segment_ids(batch_size, seq_len, max_segments_per_seq)
  else:
    segmentation = jnp.ones((batch_size, seq_len), jnp.int32)
  return {
      "inputs": tokens[:, :-1],
      "inputs_position": positions[:, :-1],
      "inputs_segmentation": segmentation,
      "targets": tokens[:, 1:],
      "targets_position": positions[:, 1:],
      "targets_segmentation": segmentation,
  }
