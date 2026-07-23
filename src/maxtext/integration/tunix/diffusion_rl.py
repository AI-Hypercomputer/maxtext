# Copyright 2026 Google LLC
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

"""Trajectory-aware MaxText adapters for block-diffusion reinforcement learning."""

from collections.abc import Sequence

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.diffusion import scoring
from maxtext.diffusion import denoise
from tunix.diffusion import types as diffusion_types
from tunix.rl import reshard as rl_reshard
from tunix.rl.rollout import base_rollout


def _concrete_numpy(value):
  if isinstance(value, jax.core.Tracer):
    return None
  if isinstance(value, jax.Array) and not value.is_fully_addressable:
    return None
  return np.asarray(value)


def prepare_diffusion_policy_batch(
    *,
    prompt_tokens,
    prompt_mask,
    completion_tokens,
    completion_mask,
    action_steps,
    loss_mask=None,
    inactive_target_id: int = 0,
) -> diffusion_types.DiffusionTokenBatch:
  """Builds one completion-aligned policy batch from a sampled denoising trace.

  ``completion_mask`` describes the full canvas used during sampling, while
  ``loss_mask`` selects actions owned by the RL objective. They intentionally
  differ after an EOS token: exact replay still needs the post-EOS trace as
  context for earlier bidirectional actions, but those later actions carry no
  policy loss.
  """
  expected_shape = tuple(completion_tokens.shape)
  if tuple(completion_mask.shape) != expected_shape or tuple(action_steps.shape) != expected_shape:
    raise ValueError("completion_tokens, completion_mask, and action_steps must have identical shapes")
  if tuple(prompt_tokens.shape) != tuple(prompt_mask.shape):
    raise ValueError("prompt_tokens and prompt_mask must have identical shapes")
  if prompt_tokens.shape[0] != completion_tokens.shape[0]:
    raise ValueError("prompt and completion batches must have the same leading dimension")
  if loss_mask is None:
    loss_mask = completion_mask
  if tuple(loss_mask.shape) != expected_shape:
    raise ValueError("loss_mask must match completion_tokens shape")
  concrete_mask = _concrete_numpy(completion_mask)
  concrete_steps = _concrete_numpy(action_steps)
  concrete_loss_mask = _concrete_numpy(loss_mask)
  if concrete_mask is not None and concrete_steps is not None and concrete_loss_mask is not None:
    active = np.asarray(concrete_mask, dtype=bool)
    weighted = np.asarray(concrete_loss_mask, dtype=bool)
    concrete_steps = np.asarray(concrete_steps)
    if not np.array_equal(concrete_steps >= 0, active):
      raise ValueError("each active completion token must have exactly one nonnegative action step")
    if np.any(weighted & ~active):
      raise ValueError("loss_mask must be a subset of the sampled completion canvas")
    if np.any(concrete_steps[active] >= completion_tokens.shape[1]):
      raise ValueError("active action steps must be smaller than the completion length")
  use_numpy = all(
      isinstance(value, np.ndarray)
      for value in (prompt_tokens, prompt_mask, completion_tokens, completion_mask, action_steps, loss_mask)
  )
  array_module = np if use_numpy else jnp
  completion_tokens = array_module.asarray(completion_tokens)
  loss_mask = array_module.asarray(loss_mask, dtype=array_module.bool_)
  target_ids = array_module.where(
      loss_mask,
      completion_tokens,
      array_module.asarray(inactive_target_id, dtype=completion_tokens.dtype),
  )
  return diffusion_types.DiffusionTokenBatch.create(
      model_inputs={
          "prompt_tokens": array_module.asarray(prompt_tokens),
          "prompt_mask": array_module.asarray(prompt_mask, dtype=array_module.bool_),
          "completion_tokens": completion_tokens,
          "completion_mask": array_module.asarray(completion_mask, dtype=array_module.bool_),
          "action_steps": array_module.asarray(action_steps, dtype=array_module.int32),
      },
      target_ids=target_ids,
      loss_weights=array_module.asarray(loss_mask, dtype=array_module.float32),
  )


def _logical_positions(validity_mask):
  positions = jnp.cumsum(jnp.asarray(validity_mask, dtype=jnp.int32), axis=1) - 1
  return jnp.where(validity_mask, positions, 0)


def make_diffusion_trace_logits_fn(config):
  """Builds a scorer that replays the exact pre-commit canvas for each action."""
  mask_id = int(config.block_diffusion_mask_id)
  alignment = config.block_diffusion_logit_alignment
  vocab_size = int(config.vocab_size)

  def logits_fn(model: nnx.Module, model_inputs: diffusion_types.ModelInputs):
    prompt_tokens = jnp.asarray(model_inputs["prompt_tokens"])
    prompt_mask = jnp.asarray(model_inputs["prompt_mask"], dtype=jnp.bool_)
    completion_tokens = jnp.asarray(model_inputs["completion_tokens"])
    completion_mask = jnp.asarray(model_inputs["completion_mask"], dtype=jnp.bool_)
    action_steps = jnp.asarray(model_inputs["action_steps"], dtype=jnp.int32)
    batch_size, completion_length = completion_tokens.shape
    prompt_length = prompt_tokens.shape[1]
    validity_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
    positions = _logical_positions(validity_mask)
    segment_ids = validity_mask.astype(jnp.int32)
    base_model = getattr(model, "base", model)
    graphdef, params, rest = nnx.split(base_model, nnx.Param, ...)
    output_logits = jnp.zeros((batch_size, completion_length, vocab_size), dtype=jnp.float32)

    def score_action_step(step, accumulated_logits):
      active_actions = completion_mask & (action_steps == step)

      def run_forward(current_logits):
        pre_commit_completion = jnp.where(action_steps >= step, mask_id, completion_tokens)
        canvas = jnp.concatenate([prompt_tokens, pre_commit_completion], axis=1)
        local_model = nnx.merge(graphdef, params, rest, copy=True)
        raw_logits = local_model(
            decoder_input_tokens=canvas,
            decoder_positions=positions,
            decoder_segment_ids=segment_ids,
            enable_dropout=False,
            decoder_target_tokens=canvas,
            decoder_target_mask=segment_ids,
        )
        target_aligned = scoring.align_logits_to_targets(raw_logits, alignment, positions, validity_mask)
        completion_logits = jnp.asarray(target_aligned[:, prompt_length:, :], dtype=jnp.float32)
        completion_logits = completion_logits.at[..., mask_id].set(-jnp.inf)
        return jnp.where(active_actions[..., None], completion_logits, current_logits)

      return jax.lax.cond(jnp.any(active_actions), run_forward, lambda value: value, accumulated_logits)

    output_logits = jax.lax.fori_loop(0, completion_length, score_action_step, output_logits)
    return jnp.where(completion_mask[..., None], output_logits, 0.0)

  return logits_fn


def make_diffusion_rollout_fn(config, temperature: float, *, sample_tokens: bool = True):
  """Builds one compiled stochastic rollout over the current MaxText policy."""
  max_denoise_steps = config.rl.diffusion_max_denoise_steps
  if max_denoise_steps == -1:
    max_denoise_steps = config.block_diffusion_block_size

  @nnx.jit
  def rollout(model, initial_tokens, positions, segment_ids, completion_mask, rng):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    validity_mask = segment_ids != 0

    def canvas_logits(canvas):
      local_model = nnx.merge(graphdef, params, rest, copy=True)
      base_model = getattr(local_model, "base", local_model)
      raw_logits = base_model(  # pylint: disable=not-callable
          decoder_input_tokens=canvas,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          enable_dropout=False,
          decoder_target_tokens=canvas,
          decoder_target_mask=segment_ids,
      )
      return scoring.align_logits_to_targets(
          raw_logits,
          config.block_diffusion_logit_alignment,
          positions,
          validity_mask,
      )

    return denoise.low_confidence_rollout(
        canvas_logits,
        initial_tokens,
        positions,
        validity_mask,
        completion_mask,
        block_size=config.block_diffusion_block_size,
        mask_id=config.block_diffusion_mask_id,
        logit_alignment=config.block_diffusion_logit_alignment,
        canvas_policy=config.block_diffusion_canvas_policy,
        confidence_threshold=config.rl.diffusion_confidence_threshold,
        temperature=temperature,
        max_denoise_steps=max_denoise_steps,
        rng=rng if sample_tokens else None,
    )

  return rollout


def resolve_stop_token_ids(config, tokenizer_eos_id: int | None) -> tuple[int, ...]:
  """Resolves and validates the model's generated completion terminators."""
  configured = tuple(int(token_id) for token_id in config.rl.diffusion_stop_token_ids)
  if configured:
    stop_token_ids = configured
  elif tokenizer_eos_id is not None:
    stop_token_ids = (int(tokenizer_eos_id),)
  else:
    raise ValueError("block-diffusion RL requires an EOS token or explicit diffusion_stop_token_ids")
  if len(set(stop_token_ids)) != len(stop_token_ids) or any(
      token_id < 0 or token_id >= config.vocab_size for token_id in stop_token_ids
  ):
    raise ValueError("diffusion stop-token IDs must be unique and inside the model vocabulary")
  if config.block_diffusion_mask_id in stop_token_ids:
    raise ValueError("the diffusion mask token cannot also be a generated stop token")
  return stop_token_ids


def policy_mask_at_stop(
    tokens: jax.Array,
    action_steps: jax.Array,
    stop_token_ids: Sequence[int],
) -> jax.Array:
  """Selects sampled actions through the first generated stop token."""
  active = action_steps >= 0
  stop_positions = active & jnp.isin(tokens, jnp.asarray(stop_token_ids, tokens.dtype))
  seen_stop = jax.lax.associative_scan(jnp.logical_or, stop_positions, axis=1)
  after_stop = jnp.concatenate([jnp.zeros_like(seen_stop[:, :1]), seen_stop[:, :-1]], axis=1)
  return active & ~after_stop


class MaxTextDiffusionRollout(base_rollout.BaseRollout):
  """Correctness-first in-process rollout that exports exact denoising traces."""

  def __init__(
      self,
      rollout_actor,
      tokenizer,
      mesh,
      rollout_config,
      maxtext_config,
      cache_config_or_size=None,
  ):
    del cache_config_or_size
    self._model = rollout_actor
    self._tokenizer = tokenizer
    self._mesh = mesh
    self._rollout_config = rollout_config
    self._config = maxtext_config
    self._pad_id = tokenizer.pad_id()
    if self._pad_id is None:
      raise ValueError("block-diffusion RL requires a tokenizer pad token")
    self._pad_id = int(self._pad_id)
    if not 0 <= self._pad_id < maxtext_config.vocab_size:
      raise ValueError("the tokenizer pad token must be inside the model vocabulary")
    self._eos_id = tokenizer.eos_id()
    self._stop_token_ids = resolve_stop_token_ids(maxtext_config, self._eos_id)
    self._rollout_fns = {
        (float(rollout_config.temperature), True): make_diffusion_rollout_fn(
            maxtext_config, float(rollout_config.temperature)
        )
    }
    configured_seed = getattr(rollout_config, "seed", None)
    if configured_seed is None:
      self._base_rng = jax.random.PRNGKey(int(getattr(maxtext_config, "data_shuffle_seed", 0)))
    else:
      configured_seed = jnp.asarray(configured_seed)
      if not configured_seed.shape:
        self._base_rng = jax.random.PRNGKey(int(configured_seed))
      elif tuple(configured_seed.shape) == (2,):
        self._base_rng = configured_seed
      else:
        raise ValueError("RolloutConfig.seed must be an integer or one JAX PRNG key")
    training_stream = self._sampling_stream_key(rollout_config)
    self._rng_streams = {training_stream: self._base_rng}
    self._generation_context = None
    self._generation_context_invocation = 0
    self._generation_call = 0

  @property
  def mesh(self):
    return self._mesh

  def _encode_left_padded(self, prompts, max_prompt_length):
    rows = []
    masks = []
    for prompt in prompts:
      token_ids = list(self._tokenizer.encode(prompt))[-max_prompt_length:]
      padding = max_prompt_length - len(token_ids)
      rows.append([self._pad_id] * padding + token_ids)
      masks.append([False] * padding + [True] * len(token_ids))
    return np.asarray(rows, dtype=np.int32), np.asarray(masks, dtype=bool)

  def _rollout_fn_for(self, rollout_config):
    """Returns a cached rollout compiled for the requested sampling mode."""
    top_k = getattr(rollout_config, "top_k", None)
    top_p = getattr(rollout_config, "top_p", 1.0)
    if top_p not in (None, 1.0) or top_k not in (None, -1, 1):
      raise ValueError("block-diffusion rollout supports categorical sampling or greedy top_k=1")
    sample_tokens = top_k != 1
    key = (float(rollout_config.temperature), sample_tokens)
    if key not in self._rollout_fns:
      self._rollout_fns[key] = make_diffusion_rollout_fn(
          self._config,
          float(rollout_config.temperature),
          sample_tokens=sample_tokens,
      )
    return self._rollout_fns[key]

  def _sampling_stream_key(self, rollout_config):
    return (
        float(rollout_config.temperature),
        getattr(rollout_config, "top_k", None),
        getattr(rollout_config, "top_p", 1.0),
        bool(getattr(rollout_config, "return_logprobs", False)),
    )

  def _next_rng(self, rollout_config):
    """Advances the deterministic RNG stream for one generation call."""
    if self._generation_context is not None:
      global_step, mode = self._generation_context
      mode_id = 0 if mode == "train" else 1
      rollout_key = jax.random.fold_in(self._base_rng, global_step)
      rollout_key = jax.random.fold_in(rollout_key, mode_id)
      rollout_key = jax.random.fold_in(rollout_key, self._generation_context_invocation)
      rollout_key = jax.random.fold_in(rollout_key, self._generation_call)
      self._generation_call += 1
      return rollout_key
    stream = self._sampling_stream_key(rollout_config)
    if stream not in self._rng_streams:
      self._rng_streams[stream] = jax.random.fold_in(self._base_rng, len(self._rng_streams))
    self._rng_streams[stream], rollout_key = jax.random.split(self._rng_streams[stream])
    return rollout_key

  def set_generation_context(self, *, global_step: int, mode):
    """Derives resumable rollout keys without coupling train and eval calls."""
    context = (int(global_step), str(mode))
    if context == self._generation_context:
      self._generation_context_invocation += 1
    else:
      self._generation_context = context
      self._generation_context_invocation = 0
    self._generation_call = 0

  def generate(self, prompts, rollout_config, **kwargs):
    del kwargs
    rollout_fn = self._rollout_fn_for(rollout_config)
    prompt_tokens, prompt_mask = self._encode_left_padded(prompts, rollout_config.max_prompt_length)
    batch_size = prompt_tokens.shape[0]
    completion_length = rollout_config.max_tokens_to_generate
    completion_mask = np.ones((batch_size, completion_length), dtype=bool)
    validity_mask = np.concatenate([prompt_mask, completion_mask], axis=1)
    positions = _logical_positions(jnp.asarray(validity_mask))
    segment_ids = jnp.asarray(validity_mask, dtype=jnp.int32)
    initial_tokens = jnp.concatenate(
        [
            jnp.asarray(prompt_tokens),
            jnp.full(
                (batch_size, completion_length),
                self._config.block_diffusion_mask_id,
                dtype=jnp.int32,
            ),
        ],
        axis=1,
    )
    full_completion_mask = jnp.concatenate(
        [
            jnp.zeros(prompt_tokens.shape, dtype=jnp.bool_),
            jnp.ones((batch_size, completion_length), dtype=jnp.bool_),
        ],
        axis=1,
    )
    rollout_key = self._next_rng(rollout_config)
    trace = rollout_fn(
        self._model,
        initial_tokens,
        positions,
        segment_ids,
        full_completion_mask,
        rollout_key,
    )
    completion_tokens = trace.tokens[:, -completion_length:]
    completion_steps = trace.action_steps[:, -completion_length:]
    completion_logps = trace.action_logps[:, -completion_length:]
    output_mask = policy_mask_at_stop(
        completion_tokens,
        completion_steps,
        self._stop_token_ids,
    )
    for name, value in (
        ("completion tokens", completion_tokens),
        ("action steps", completion_steps),
        ("action log probabilities", completion_logps),
        ("output mask", output_mask),
    ):
      if isinstance(value, jax.Array) and not value.is_fully_addressable:
        raise RuntimeError(
            f"block-diffusion RL requires fully addressable {name}; use the supported Pathways single-controller mode"
        )
    host_completion_tokens = np.asarray(jax.device_get(completion_tokens))
    host_completion_steps = np.asarray(jax.device_get(completion_steps))
    host_completion_logps = np.asarray(jax.device_get(completion_logps))
    host_completion_mask = host_completion_steps >= 0
    host_output_mask = np.asarray(jax.device_get(output_mask))
    if not np.all(host_completion_mask) or np.any(host_completion_tokens == self._config.block_diffusion_mask_id):
      raise RuntimeError("block-diffusion rollout ended with unresolved mask tokens")
    if not np.all(np.isfinite(host_completion_logps[host_completion_mask])):
      raise RuntimeError("block-diffusion rollout produced non-finite action log probabilities")
    diffusion_batch = prepare_diffusion_policy_batch(
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        completion_tokens=host_completion_tokens,
        completion_mask=host_completion_mask,
        action_steps=host_completion_steps,
        inactive_target_id=self._pad_id,
    )
    tokens_out = []
    text_out = []
    logps_out = []
    for tokens_row, active_row, logps_row in zip(host_completion_tokens, host_output_mask, host_completion_logps):
      length = int(active_row.sum())
      trimmed_tokens = np.asarray(tokens_row[:length], dtype=np.int32)
      tokens_out.append(trimmed_tokens)
      text_out.append(self._tokenizer.decode(trimmed_tokens.tolist()))
      logps_out.append(np.asarray(logps_row, dtype=np.float32))
    return base_rollout.RolloutOutput(
        text=text_out,
        logits=None,
        tokens=tokens_out,
        left_padded_prompt_tokens=prompt_tokens,
        logprobs=logps_out if rollout_config.return_logprobs else None,
        diffusion_batch=diffusion_batch,
    )

  def update_params(self, params, filter_types=None):
    destination = nnx.state(self._model, filter_types) if filter_types is not None else nnx.state(self._model)
    nnx.update(self._model, rl_reshard.reshard_pytree(params, destination))

  def get_per_token_logps(self, prompt_tokens, completion_tokens):
    del prompt_tokens, completion_tokens
    raise NotImplementedError(
        "block-diffusion policy scores require RolloutOutput.diffusion_batch; final-sequence AR scoring is invalid"
    )

  def pad_id(self):
    return self._pad_id

  def eos_id(self):
    return self._eos_id

  def model(self):
    return self._model
