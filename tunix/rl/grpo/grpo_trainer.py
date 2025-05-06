# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO trainer."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
import optax
from tunix.rl import common
from tunix.rl.grpo import grpo_helpers
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from typing_extensions import override

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


class RepeatTrainingInputIter:
  """Repeat data to simulate multiple GRPO generations and iterations."""

  def __init__(
      self,
      data: Iterable[_TrainingInputT],
      sample_repeat: int = 1,
      batch_repeat: int = 1,
      gradient_accumulation_steps: int = 1,
  ):
    assert sample_repeat > 0, f"sample repeat must be positive: {sample_repeat}"
    assert batch_repeat > 0, f"batch repeat must be positive: {batch_repeat}"
    assert gradient_accumulation_steps > 0, (
        "gradient accumulation steps must be positive:"
        f" {gradient_accumulation_steps}"
    )
    self._data = data
    self._sample_repeat = sample_repeat
    self._batch_repeat = batch_repeat
    self._gradient_accumulation_steps = gradient_accumulation_steps
    self._data_buffer = [None] * self._gradient_accumulation_steps
    self._itr_cnt = 0
    self._batch_repeat_cnt = 0

  def __iter__(self):
    self._iterator = None
    return self

  def __next__(self):
    if self._iterator is None:
      self._iterator = iter(self._data)
    if self._batch_repeat_cnt % self._batch_repeat == 0:
      self._data_buffer[self._itr_cnt % self._gradient_accumulation_steps] = (
          self._repeat_sample(self._iterator)
      )
    self._itr_cnt += 1
    if self._itr_cnt % self._gradient_accumulation_steps == 0:
      self._batch_repeat_cnt += 1

    return self._data_buffer[
        (self._itr_cnt - 1) % self._gradient_accumulation_steps
    ]

  def _repeat_sample(self, itr: Iterator[_TrainingInputT]):
    data = next(itr)
    data = jax.tree.map(
        lambda x: np.repeat(x, self._sample_repeat, axis=0), data
    )
    return data


@flax.struct.dataclass(frozen=True)
class TrainExample:
  prompt_ids: jax.Array
  prompt_mask: jax.Array
  completion_ids: jax.Array
  completion_mask: jax.Array
  advantages: jax.Array
  ref_per_token_logps: jax.Array | None
  old_per_token_logps: jax.Array | None


@dataclasses.dataclass(slots=True, kw_only=True)
class GrpoTrainingConfig(peft_trainer.TrainingConfig):
  """Configuration for GRPO trainer."""

  total_generation_steps: int = 1
  num_generations: int = 2  # G in GRPO algo 1 https://arxiv.org/pdf/2402.03300
  num_iterations: int = 1  # 𝜇 in GRPO algo 1 https://arxiv.org/pdf/2402.03300
  beta: float = 0.04  # 𝛽 in GRPO KL penalty https://arxiv.org/pdf/2402.03300
  epsilon: float = 0.2  # 𝜀 in GRPO loss https://arxiv.org/pdf/2402.03300
  temperature: float = 0.9  # temperature for sampling
  top_p: float = 1.0  # top-p sampling threshold

  max_prompt_length: int = 256  # max prompt length

  def __post_init__(self):
    assert self.num_generations > 1, (
        "num_generations must be greater than 1. Received: "
        f"{self.num_generations}"
    )


class GrpoTrainer(peft_trainer.PeftTrainer):
  """GRPO trainer."""

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module,
      sampler,
      reward_fns: RewardFn | List[RewardFn],
      optimizer: optax.GradientTransformation,
      training_config: GrpoTrainingConfig,
  ):
    self.model = model
    self.ref_model = ref_model
    self.sampler = sampler
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    super(GrpoTrainer, self).__init__(model, optimizer, training_config)
    self.grpo_config = training_config
    self.loss_fn = grpo_loss_fn
    self.eval_loss_fn = grpo_loss_fn
    self.gen_model_input_fn = lambda x: {
        "train_example": x,
        "beta": self.grpo_config.beta,
        "epsilon": self.grpo_config.epsilon,
    }
    self._data_buffer = [None] * self.grpo_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    self._num_iterations = 0

    self._has_aux = True

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    metrics = ["loss", "perplexity", "rewards/overall"]
    if self.grpo_config.beta != 0.0:
      metrics.append("kl")
    return metrics

  @property
  def _tqdm_eval_metrics(self) -> list[str]:
    metrics = ["loss", "perplexity", "rewards/overall"]
    if self.grpo_config.beta != 0.0:
      metrics.append("kl")
    return metrics

  def _get_metric_logging_steps(self) -> int:
    return (
        self._train_steps
        if self._mode == metrics_logger.Mode.TRAIN
        else self._eval_steps
    )

  @override
  def _prepare_inputs(self, training_input: _TrainingInputT) -> Any:
    if self._mode == metrics_logger.Mode.TRAIN:
      idx = self._train_steps % len(self._data_buffer)
      data = self._data_buffer[idx]
      if data is None or self._num_iterations == 0:
        data = self._generate_and_compute_advantage(training_input)
        self._data_buffer[idx] = data

      grad_acc_steps = self.grpo_config.get_with_default(
          "gradient_accumulation_steps", 1
      )
      if self._train_steps % grad_acc_steps == grad_acc_steps - 1:
        self._num_iterations += 1

      if self._num_iterations == self.grpo_config.num_iterations:
        self._num_iterations = 0
      return self._data_buffer[idx]
    else:
      return self._generate_and_compute_advantage(training_input)

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    self._metrics_logger.log("kl", aux["kl"], self._mode, self._train_steps)

  def _generate_and_compute_advantage(
      self, training_input: _TrainingInputT
  ) -> TrainExample:
    pad_value = self.sampler.vocab.pad_id()
    eos_value = self.sampler.vocab.eos_id()

    # Generate, and pad output.
    completion_output = self.sampler(
        input_strings=training_input["prompts"],
        total_generation_steps=self.grpo_config.total_generation_steps,
        max_prompt_length=self.grpo_config.max_prompt_length,
        echo=False,
        temperature=self.grpo_config.temperature,
        top_p=self.grpo_config.top_p,
    )
    completion_ids = pad_inputs(
        completion_output.tokens,
        target_length=self.grpo_config.total_generation_steps,
        pad_value=pad_value,
        left=False,
    )
    prompt_ids = completion_output.padded_prompt_tokens

    (
        positions,
        prompt_completion_ids,
        completion_mask,
        prompt_completion_mask,
        prompt_completion_causal_mask,
    ) = process_ids(prompt_ids, completion_ids, pad_value, eos_value)

    logits_to_keep = completion_ids.shape[1]
    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = common.get_per_token_logps(
          self.ref_model,
          input_tokens=prompt_completion_ids,
          positions=positions,
          attn_mask=prompt_completion_causal_mask,
          logits_to_keep=logits_to_keep,
      )
    else:
      ref_per_token_logps = None

    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = common.get_per_token_logps(
          self.model,
          input_tokens=prompt_completion_ids,
          positions=positions,
          attn_mask=prompt_completion_causal_mask,
          logits_to_keep=logits_to_keep,
      )
    else:
      old_per_token_logps = None

    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_output.text,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )

    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
    )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    steps = self._get_metric_logging_steps()
    self._metrics_logger.log(
        "completions/mean_length",
        agg_completion_mask.mean(),
        self._mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/max_length",
        agg_completion_mask.max(),
        self._mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/min_length",
        agg_completion_mask.min(),
        self._mode,
        steps,
    )

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_completion_mask[:, : len(prompt_ids[0])],
        completion_ids=completion_ids,
        completion_mask=prompt_completion_mask[:, len(prompt_ids[0]) :],
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  def _compute_rewards(
      self, prompts: List[str], completions: List[str], **kargs
  ):
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    steps = self._get_metric_logging_steps()
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)

      self._metrics_logger.log(
          f"rewards/{reward_fn.__name__}",
          r.mean(),
          self._mode,
          steps,
      )

    rewards = jnp.nansum(rewards, axis=1)

    self._metrics_logger.log(
        "rewards/overall",
        rewards.mean(),
        self._mode,
        steps,
    )

    return rewards

  @override
  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    train_ds = RepeatTrainingInputIter(
        train_ds,
        sample_repeat=self.grpo_config.num_generations,
        batch_repeat=self.grpo_config.num_iterations,
        gradient_accumulation_steps=self.grpo_config.get_with_default(
            "gradient_accumulation_steps", 1
        ),
    )
    if eval_ds:
      eval_ds = RepeatTrainingInputIter(
          eval_ds,
          sample_repeat=self.grpo_config.num_generations,
          batch_repeat=self.grpo_config.num_iterations,
      )
    super().train(train_ds, eval_ds, skip_jit)


def pad_inputs(
    inputs: list[jax.Array],
    target_length: int,
    pad_value: int,
    left: bool,
):
  """Pads provided list of tensors to the same length."""
  padded_inputs = []

  for s in inputs:
    padded_s = common.pad_to_length(
        jnp.array(s),
        target_length=target_length,
        pad_value=pad_value,
        left=left,
        axis=-1,
    )
    padded_s = padded_s[..., -target_length:]
    padded_inputs.append(padded_s)
  return jnp.array(padded_inputs)


@functools.partial(jax.jit, static_argnames=["pad_value", "eos_value"])
def process_ids(
    prompt_ids: jax.Array,
    completion_ids: jax.Array,
    pad_value: int,
    eos_value: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Process prompt and completion ids."""
  prompt_completion_ids = jnp.concat([prompt_ids, completion_ids], axis=1)

  # Compute masks. For prompt, this is just the padding mask. For completion,
  # we do an and of the padding mask and the completion mask (computed using
  # the eos token).
  prompt_mask = (prompt_ids != pad_value).astype("int32")

  completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
      "int32"
  )
  completion_mask = common.make_completion_mask(
      completion_ids, eos_tok=eos_value
  )
  completion_mask = completion_mask * completion_padding_mask

  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )

  # Get positions for the concatenated prompt and completion ids.
  positions = common.build_positions_from_mask(prompt_completion_mask)
  prompt_completion_causal_mask = common.make_causal_attn_mask(
      prompt_completion_mask
  )
  return (
      positions,
      prompt_completion_ids,
      completion_mask,
      prompt_completion_mask,
      prompt_completion_causal_mask,
  )


def grpo_loss_fn(model, train_example, beta, epsilon):
  """GRPO loss function."""
  prompt_ids, prompt_mask = (
      train_example.prompt_ids,
      train_example.prompt_mask,
  )
  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)
  prompt_completion_mask = jnp.concat([prompt_mask, completion_mask], axis=-1)
  attention_mask = common.make_causal_attn_mask(prompt_completion_mask)
  logits_to_keep = completion_ids.shape[1]
  positions = common.build_positions_from_mask(prompt_completion_mask)

  per_token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  advantages = train_example.advantages

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps
  coef_1 = jnp.exp(per_token_logps - old_per_token_logps)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)
  per_token_loss = -jnp.minimum(
      coef_1 * jnp.expand_dims(advantages, 1),
      coef_2 * jnp.expand_dims(advantages, 1),
  )

  loss_denominator = jnp.clip(completion_mask.sum(), min=1)

  aux = {"kl": 0.0}
  if beta != 0.0:
    kl = grpo_helpers.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    per_token_loss = per_token_loss + beta * kl

    # Log mean KL.
    aux["kl"] = (kl * completion_mask).sum() / loss_denominator

  loss = (per_token_loss * completion_mask).sum() / loss_denominator

  return loss, aux
