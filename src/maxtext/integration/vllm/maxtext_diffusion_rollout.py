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

"""Block-diffusion RL rollout for GRPO/DAPO (TraceRL-style).

``MaxTextVllmRollout`` (in ``maxtext_vllm_rollout.py``) drives autoregressive
generation through a KV-cached vLLM engine and returns the vLLM sampler's
next-token logprobs. A block-diffusion policy is NOT autoregressive: it generates
a completion by *denoising* a masked canvas block-by-block, and its per-token
log-probability must come from a bidirectional-within-block forward, not an AR
decode. This module therefore provides a fresh JAX rollout (a plain
``tunix.rl.rollout.base_rollout.BaseRollout``, NOT a ``VllmRollout`` subclass):

  * ``generate`` runs block-diffusion denoising via
    ``diffusion_generate.diffusion_generate``;
  * ``get_per_token_logps`` returns the shared block-diffusion logprob
    (``diffusion_generate.diffusion_per_token_logps``).

THE P0 INVARIANT. The GRPO importance ratio ``exp(new_logp - old_logp)`` is only
unbiased if the *old* logp (this rollout's ``get_per_token_logps``) and the *new*
logp (computed in the tunix loss) come from the SAME function. Both funnel through
``diffusion_per_token_logps`` via ``_diffusion_per_token_logps_from_model`` below:
this rollout calls it directly, and the tunix loss calls it through the pluggable
hook built by ``make_diffusion_per_token_logps_fn`` (installed on the GRPO learner
when the diffusion rollout is active). tunix never imports maxtext — the shared fn
lives here and is injected into tunix as a callback.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tunix.rl.rollout import base_rollout

from maxtext.trainers.post_train.distillation import diffusion_generate


def _diffusion_model_apply(model: Any) -> diffusion_generate.ForwardFn:
  """Builds the cacheless block-diffusion forward ``(tokens, positions) -> logits``.

  ``model`` is the tunix-wrapped MaxText actor (``TunixMaxTextAdapter``), whose
  call signature is ``(input_tokens, positions, cache, attention_mask) -> (logits,
  None)``. The adapter ignores ``attention_mask`` (MaxText applies its own
  block-diffusion mask internally when ``enable_block_diffusion`` is on) and, given
  ``decoder_segment_ids=None``, synthesizes pad-masking segment ids from its
  ``pad_id``. Both RL sites (rollout + loss hook) build ``model_apply`` this way, so
  the forward is identical on both sides of the importance ratio.
  """

  def model_apply(tokens: jax.Array, positions: jax.Array) -> jax.Array:
    out = model(tokens, positions, None, None)
    return out[0] if isinstance(out, tuple) else out

  return model_apply


def _diffusion_per_token_logps_from_model(
    model: Any,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    completion_mask: jax.Array | None = None,
    *,
    bd_size: int,
    pad_id: int,
    temperature: float = 1.0,
    stop_gradient: bool = False,
) -> jax.Array:
  """The one bridge from a (wrapped) MaxText model to the shared logprob fn.

  Both P0 sites call THIS helper so they are byte-for-byte the same code path into
  ``diffusion_generate.diffusion_per_token_logps`` (the single source of truth).
  """
  return diffusion_generate.diffusion_per_token_logps(
      _diffusion_model_apply(model),
      prompt_tokens,
      completion_tokens,
      completion_mask,
      bd_size=bd_size,
      pad_id=pad_id,
      temperature=temperature,
      stop_gradient=stop_gradient,
  )


def make_diffusion_per_token_logps_fn(bd_size: int, mask_id: int | None = None):
  """Builds the tunix ``per_token_logps_fn`` hook for a block-diffusion policy.

  The returned callable is a drop-in for ``tunix.rl.common.compute_per_token_logps``
  (same call signature, called with ``graphdef, state, ...`` from ``grpo_loss_fn``),
  but computes the block-diffusion logprob instead of the AR one. It reconstructs
  the model from ``graphdef``/``state`` and funnels into
  ``_diffusion_per_token_logps_from_model`` — the exact path the rollout uses — so
  the new-policy and old-policy logps are guaranteed identical (the P0 invariant).

  ``mask_id`` is accepted for symmetry with the rollout config; it is only needed
  for generation (seeding the canvas), not for scoring an already-committed
  sequence, so it is unused here.
  """
  del mask_id  # only used by generate(); scoring needs no mask token

  @functools.partial(
      jax.jit,
      static_argnames=(
          "pad_id",
          "eos_id",
          "stop_gradient",
          "return_logits",
          "temperature",
      ),
  )
  def diffusion_per_token_logps_hook(
      graphdef,
      state,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      images: jax.Array | None = None,
      completion_mask: jax.Array | None = None,
      stop_gradient: bool = True,
      return_logits: bool = False,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      temperature: float = 1.0,
  ):
    # Block-diffusion RL scores the plain prompt|completion sequence; packing
    # (segment_ids), images and eos-based masking are AR-path concerns.
    del images, eos_id, segment_ids, segment_positions
    model = nnx.merge(graphdef, state)
    logps = _diffusion_per_token_logps_from_model(
        model,
        prompt_tokens,
        completion_tokens,
        completion_mask,
        bd_size=bd_size,
        pad_id=pad_id,
        temperature=temperature,
        stop_gradient=stop_gradient,
    )
    if return_logits:
      return logps, None
    return logps

  return diffusion_per_token_logps_hook


class MaxTextDiffusionRollout(base_rollout.BaseRollout):
  """Block-diffusion RL rollout (denoising generation + shared diffusion logprob).

  Constructed by ``RLCluster`` via the same custom-class path as
  ``MaxTextVllmRollout`` — i.e. ``rollout_engine=functools.partial(
  MaxTextDiffusionRollout, maxtext_config=trainer_config)`` — which calls it with
  ``rollout_actor=``, ``tokenizer=``, ``mesh=``, ``rollout_config=``.
  """

  def __init__(
      self,
      rollout_actor: Any,
      tokenizer: Any,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
      maxtext_config: Any,
      cache_config_or_size: base_rollout.CacheConfig | int | None = None,
  ):  # pylint: disable=too-many-positional-arguments
    del cache_config_or_size  # block-diffusion rollout is cacheless
    self._model = rollout_actor
    self._tokenizer = tokenizer
    self._mesh = mesh
    self._rollout_config = rollout_config
    self._config = maxtext_config

    if not getattr(maxtext_config, "enable_block_diffusion", False):
      raise ValueError(
          "MaxTextDiffusionRollout requires enable_block_diffusion=True on the "
          "policy model (the rollout denoises a block-diffusion canvas)."
      )
    # bd_size / mask_id / enable_block_diffusion are top-level (flattened
    # Attention config); the diffusion-rollout generation knobs live in the nested
    # `rl` config, so read them off that sub-config.
    self._bd_size = maxtext_config.bd_size
    self._mask_id = maxtext_config.mask_id
    _rl = getattr(maxtext_config, "rl", None)
    self._threshold = getattr(_rl, "rl_diffusion_threshold", 0.9) if _rl is not None else 0.9
    self._max_denoise_steps = getattr(_rl, "rl_diffusion_max_denoise_steps", 0) if _rl is not None else 0
    self._temperature = rollout_config.temperature
    self._pad_id = tokenizer.pad_id()
    self._eos_id = tokenizer.eos_id()

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._mesh

  def _encode_left_padded(self, prompts: list[str], max_prompt_length: int) -> np.ndarray:
    """Tokenizes and left-pads/truncates each prompt to ``[B, max_prompt_length]``."""
    rows = []
    for prompt in prompts:
      ids = list(self._tokenizer.encode(prompt))
      ids = ids[-max_prompt_length:]  # truncate on the left (keep the most recent)
      pad = [self._pad_id] * (max_prompt_length - len(ids))
      rows.append(pad + ids)
    return np.asarray(rows, dtype=np.int32)

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates completions by block-diffusion denoising."""
    del kwargs
    max_prompt_length = rollout_config.max_prompt_length
    max_gen = rollout_config.max_tokens_to_generate

    prompt_ids = self._encode_left_padded(prompts, max_prompt_length)  # [B, P]
    batch_size = prompt_ids.shape[0]

    # Pad the canvas so prompt+completion is a whole number of blocks (the
    # block-diffusion / diffusion_generate contract). Completion tokens beyond
    # max_gen are dropped from the returned output.
    total = max_prompt_length + max_gen
    padded_total = -(-total // self._bd_size) * self._bd_size  # round up to bd_size
    canvas_len = padded_total
    completion_span = canvas_len - max_prompt_length

    # Seed canvas: prompt (left-padded) then mask_id over the completion span.
    canvas = np.full((batch_size, canvas_len), self._mask_id, dtype=np.int32)
    canvas[:, :max_prompt_length] = prompt_ids
    canvas = jnp.asarray(canvas)

    # Generation validity: real prompt tokens + the whole completion span.
    prompt_valid = jnp.asarray(prompt_ids != self._pad_id)
    gen_mask = jnp.concatenate(
        [
            jnp.zeros((batch_size, max_prompt_length), dtype=bool),
            jnp.ones((batch_size, completion_span), dtype=bool),
        ],
        axis=1,
    )
    attend_mask = jnp.concatenate([prompt_valid, jnp.ones((batch_size, completion_span), dtype=bool)], axis=1)
    positions = diffusion_generate._positions_from_mask(attend_mask)  # pylint: disable=protected-access

    gen_tokens, _ = diffusion_generate.diffusion_generate(
        _diffusion_model_apply(self._model),
        canvas,
        positions,
        gen_mask,
        mask_id=self._mask_id,
        bd_size=self._bd_size,
        threshold=self._threshold,
        temperature=self._temperature,
        max_denoise_steps=self._max_denoise_steps,
    )
    completions = np.asarray(gen_tokens[:, max_prompt_length : max_prompt_length + max_gen])

    # Truncate each completion at its first EOS (inclusive) and decode.
    tokens_out: list[np.ndarray] = []
    text_out: list[str] = []
    for row in completions:
      row = np.asarray(row, dtype=np.int32)
      eos_hits = np.flatnonzero(row == self._eos_id)
      end = int(eos_hits[0]) + 1 if eos_hits.size > 0 else row.shape[0]
      trimmed = row[:end]
      tokens_out.append(trimmed)
      text_out.append(self._tokenizer.decode(trimmed.tolist()))

    return base_rollout.RolloutOutput(
        text=text_out,
        logits=None,
        tokens=tokens_out,
        left_padded_prompt_tokens=prompt_ids,
        logprobs=None,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Old-policy block-diffusion logps (P0 site #1 — shared with the loss)."""
    return _diffusion_per_token_logps_from_model(
        self._model,
        prompt_tokens,
        completion_tokens,
        completion_mask,
        bd_size=self._bd_size,
        pad_id=self._pad_id,
        temperature=self._temperature,
        stop_gradient=True,
    )

  def update_params(
      self,
      params: Any,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    del filter_types
    nnx.update(self._model, params)

  def pad_id(self) -> int:
    return self._pad_id

  def eos_id(self) -> int:
    return self._eos_id

  def model(self) -> nnx.Module:
    return self._model
