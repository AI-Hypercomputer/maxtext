"""
Copyright 2025 Google LLC

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

"""DPO (Direct Preference Optimization) utilities."""

import jax
import jax.numpy as jnp

from flax import nnx

from maxtext.utils import maxtext_utils


def _split_dpo_state(state):
  """Split DPO state to separate reference parameters."""
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def dpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A model module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    reference_params: Reference model params for DPO
    is_train: True for train_step and False for eval_step

  Returns:
    loss: DPO preference loss + MoE load balance loss (if applicable)
    aux: a dictionary with:
      - intermediate_outputs: model intermediates from forward pass
      - xent_sum: always 0.0 (DPO has no per-token cross-entropy)
      - dpo_loss: pure preference loss before auxiliary losses
      - total_weights: number of samples in the batch
      - moe_lb_loss: MoE load balance loss (0.0 if num_experts <= 1)
      - reward_accuracy: fraction of examples where chosen is preferred over rejected
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]

  # for DPO we don't support packed sequence (they shouldn't be present in the first place)
  data["chosen_segmentation"] = (data["chosen_segmentation"] == 1).astype(jnp.int32)
  data["rejected_segmentation"] = (data["rejected_segmentation"] == 1).astype(jnp.int32)
  data["chosen_position"] = data["chosen_position"] * (data["chosen_segmentation"] == 1)
  data["rejected_position"] = data["rejected_position"] * (data["rejected_segmentation"] == 1)

  # concatenated model and reference model forward pass
  inputs = jnp.concatenate([data["chosen"], data["rejected"]], 0)
  inputs_position = jnp.concatenate([data["chosen_position"], data["rejected_position"]], 0)
  inputs_segmentation = jnp.concatenate([data["chosen_segmentation"], data["rejected_segmentation"]], 0)

  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  ref_logits = model.apply(
      {"params": reference_params},
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
  )
  ref_logits = jax.lax.stop_gradient(ref_logits)

  # extract token ids, segmentation and logits for chosen and rejected sequences
  chosen_ids = data["chosen"][..., 1:]
  rejected_ids = data["rejected"][..., 1:]
  chosen_segmentation = data["chosen_segmentation"][..., 1:]
  rejected_segmentation = data["rejected_segmentation"][..., 1:]
  n_logits = logits.shape[-3] // 2  # [B, S, E] - [batch, sequence, embedding/vocab]
  chosen_logits, rejected_logits = logits[:n_logits, :, :], logits[n_logits:, :, :]  # [B, S, E], [B, S, E]
  # ^ [B, S, E], [B, S, E]
  chosen_ref_logits, rejected_ref_logits = ref_logits[:n_logits, :, :], ref_logits[n_logits:, :, :]

  # common subsequence and padding mask
  common_prefix_mask = jnp.cumsum(chosen_ids != rejected_ids, axis=-1) == 0  # [B, S]
  valid_seq_mask = (chosen_segmentation != 0) & (rejected_segmentation != 0) & ~common_prefix_mask  # [B, S]

  # compute logratios from the sequence-reduced observed token log-probability
  chosen_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_logps = jnp.sum(chosen_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_ref_logps = jnp.sum(chosen_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

  rejected_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_logps = jnp.sum(rejected_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_ref_logps = jnp.sum(rejected_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

  # DPO loss from chosen and rejected logratios
  LABEL_SMOOTHING, BETA = config.dpo_label_smoothing, config.dpo_beta
  logratios_delta = BETA * (chosen_logratios - rejected_logratios)  # [B]
  losses = (  # [B]
      -jax.nn.log_sigmoid(BETA * logratios_delta) * (1 - LABEL_SMOOTHING)
      - jax.nn.log_sigmoid(-BETA * logratios_delta) * LABEL_SMOOTHING
  )
  total_loss, total_weights = jnp.mean(losses), losses.shape[0]
  # Under manual gradient accumulation, return the unnormalized sum: the accumulator
  # sums per-microbatch grads then divides once by total_weights, so a pre-normalized
  # mean would scale the gradient down by an extra microbatch-size factor. Tunix GA
  # expects a normalized per-step loss. Mirrors loss_fn in train.py.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = jnp.sum(losses)
  else:
    loss = total_loss

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      moe_lb_loss = jnp.mean(jnp.concatenate(moe_lb_losses))
      loss += moe_lb_loss
  reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": 0.0,  # DPO has no per-token cross-entropy sum; set to 0 for train_step compatibility
      "dpo_loss": total_loss,  # pure preference loss before MoE lb, analogous to lm_loss in pre-training
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "reward_accuracy": reward_accuracy,
      "indexer_loss": 0.0,  # for gradient_accumulation aux pytree compatibility
      "mtp_loss": 0.0,  # for gradient_accumulation aux pytree compatibility
  }
  return loss, aux


def _merge_dpo_state(state, reference_params):
  """Merge reference parameters back into DPO state."""
  return state.replace(params=dict(state.params, reference_params=reference_params))


# NNX DPO has no split/merge counterpart: the Linen path overlays
# `reference_params` inside `state.params`, so it must be peeled off and
# reattached around `apply_gradients`. The NNX path holds the reference as a
# sibling field `TrainStateNNX.reference_model`; `apply_gradients` already
# only touches `self.model`, so no split/merge is needed.


def dpo_loss_fn_nnx(policy_model, config, data, dropout_rng, params, reference_model, is_train=True):
  """NNX DPO loss_fn for both train and eval.

  Signature mirrors the Linen `dpo_loss_fn` so it slots into the same
  dispatcher in `gradient_accumulation_loss_and_grad`:
    `(model, config, data, dropout_rng, params, *extra_dpo_args, is_train=True)`

  Differences from the Linen `dpo_loss_fn`:
    * `policy_model` is an `nnx.Module` (carries its own params + RNG state).
    * `dropout_rng` and `params` are unused for NNX (kept positional for
      signature parity; NNX models manage these internally).
    * The 6th arg (the `extra_dpo_args[0]`) is a frozen reference
      `nnx.Module`, not a `reference_params` pytree.
    * Reference forward is wrapped in `jax.lax.stop_gradient`; combined with
      `nnx.value_and_grad(..., argnums=0)` over the policy, no gradient flows
      to the reference's `nnx.Param` leaves.

  Args:
    policy_model: Policy `nnx.Module` (the model being trained).
    config: Config of parameters.
    data: Batch of preference data with `chosen` / `rejected` fields.
    dropout_rng: Unused for NNX (kept for signature parity with Linen).
    params: Unused for NNX (kept for signature parity with Linen).
    reference_model: Frozen reference `nnx.Module` for DPO logratio computation.
    is_train: True for train_step and False for eval_step.

  Returns:
    loss: DPO preference loss + MoE load balance loss (if applicable).
    aux: dict with intermediate_outputs, xent_sum (always 0.0), dpo_loss,
      total_weights, moe_lb_loss, reward_accuracy.
  """
  del dropout_rng, params  # unused for NNX
  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]

  # for DPO we don't support packed sequences (they shouldn't be present in the first place)
  data["chosen_segmentation"] = (data["chosen_segmentation"] == 1).astype(jnp.int32)
  data["rejected_segmentation"] = (data["rejected_segmentation"] == 1).astype(jnp.int32)
  data["chosen_position"] = data["chosen_position"] * (data["chosen_segmentation"] == 1)
  data["rejected_position"] = data["rejected_position"] * (data["rejected_segmentation"] == 1)

  # concatenated policy/reference forward pass
  inputs = jnp.concatenate([data["chosen"], data["rejected"]], 0)
  inputs_position = jnp.concatenate([data["chosen_position"], data["rejected_position"]], 0)
  inputs_segmentation = jnp.concatenate([data["chosen_segmentation"], data["rejected_segmentation"]], 0)

  logits = policy_model(
      decoder_input_tokens=inputs,
      decoder_positions=inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=config.enable_dropout if is_train else False,
  )
  # pop (not snapshot) so sown Intermediates don't persist on the model across
  # microbatches during gradient accumulation; matches loss_fn in train.py.
  intermediates = nnx.pop(policy_model, nnx.Intermediate)
  intermediate_outputs = intermediates.to_pure_dict()

  ref_logits = reference_model(
      decoder_input_tokens=inputs,
      decoder_positions=inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
  )
  ref_logits = jax.lax.stop_gradient(ref_logits)

  # extract token ids, segmentation and logits for chosen and rejected sequences
  chosen_ids = data["chosen"][..., 1:]
  rejected_ids = data["rejected"][..., 1:]
  chosen_segmentation = data["chosen_segmentation"][..., 1:]
  rejected_segmentation = data["rejected_segmentation"][..., 1:]
  n_logits = logits.shape[-3] // 2  # [B, S, E] - [batch, sequence, embedding/vocab]
  chosen_logits, rejected_logits = logits[:n_logits, :, :], logits[n_logits:, :, :]
  chosen_ref_logits, rejected_ref_logits = ref_logits[:n_logits, :, :], ref_logits[n_logits:, :, :]

  # common subsequence and padding mask
  common_prefix_mask = jnp.cumsum(chosen_ids != rejected_ids, axis=-1) == 0  # [B, S]
  valid_seq_mask = (chosen_segmentation != 0) & (rejected_segmentation != 0) & ~common_prefix_mask  # [B, S]

  # compute logratios from the sequence-reduced observed token log-probability
  chosen_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_logps = jnp.sum(chosen_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_ref_logps = jnp.sum(chosen_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

  rejected_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_logps = jnp.sum(rejected_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_ref_logps = jnp.sum(rejected_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

  # DPO loss from chosen and rejected logratios
  LABEL_SMOOTHING, BETA = config.dpo_label_smoothing, config.dpo_beta
  logratios_delta = BETA * (chosen_logratios - rejected_logratios)  # [B]
  losses = (  # [B]
      -jax.nn.log_sigmoid(BETA * logratios_delta) * (1 - LABEL_SMOOTHING)
      - jax.nn.log_sigmoid(-BETA * logratios_delta) * LABEL_SMOOTHING
  )
  total_loss, total_weights = jnp.mean(losses), losses.shape[0]
  # Under manual gradient accumulation, return the unnormalized sum: the accumulator
  # sums per-microbatch grads then divides once by total_weights, so a pre-normalized
  # mean would scale the gradient down by an extra microbatch-size factor. Tunix GA
  # expects a normalized per-step loss. Mirrors loss_fn in train.py.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = jnp.sum(losses)
  else:
    loss = total_loss

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      moe_lb_loss = jnp.mean(jnp.concatenate(moe_lb_losses))
      loss += moe_lb_loss
  reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": 0.0,  # DPO has no per-token cross-entropy sum; set to 0 for train_step compatibility
      "dpo_loss": total_loss,  # pure preference loss before MoE lb, analogous to lm_loss in pre-training
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "reward_accuracy": reward_accuracy,
      "indexer_loss": 0.0,  # for gradient_accumulation aux pytree compatibility
      "mtp_loss": 0.0,  # for gradient_accumulation aux pytree compatibility
  }
  return loss, aux
