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

"""Student block-diffusion rollout for on-policy distillation (OPD).

This is the MaxText training-side analog of tpu_inference's standalone
``runner/diffusion_decode.py`` inference loop. The student is a block-diffusion
model (``enable_block_diffusion`` on): attention is bidirectional within a block
of ``bd_size`` tokens and block-causal across blocks. Given a prompt prefix, the
student generates its own completion by block-diffusion *denoising*, so the loss
can be computed on-policy against a frozen teacher (see
``train_distill.MaxTextDistillationTrainer._run_on_policy_rollout``).

Design
------
The module is intentionally free of MaxText / Tunix imports (only ``jax``) so the
whole seed -> forward -> shift -> commit -> advance loop is exercisable on CPU
with a stub ``forward_fn`` (dependency injection), exactly like the reference
inference loop. The real student is injected by the trainer as::

    forward_fn(tokens[B, T] int32, positions[B, T] int32) -> logits[B, T, V]

Unlike the inference reference (which forwards a single ``block_size`` canvas
against a paged KV cache), the MaxText trainer forward is *cacheless* and returns
per-position logits for the whole ``[B, T]`` sequence. We therefore operate on
the full sequence and rely on the block-causal attention mask: when denoising
block ``k`` the model's logits for block ``k`` depend only on blocks ``0..k``, so
the still-masked trailing blocks never corrupt an earlier block and we can fill
strictly left-to-right, one block at a time.

Conventions (matching the reference decode loop)
------------------------------------------------
* **Seed / canvas**: the completion positions (``gen_mask``) are seeded with
  ``mask_id``; the prompt positions keep their real tokens and are never masked.
* **Shifted logits**: the token at position ``i`` is predicted from hidden
  ``i - 1`` (clamped ``>= 0``), i.e. ``shifted[:, i] = logits[:, max(i - 1, 0)]``
  — the AR next-token convention. For the first completion position this reads
  the last prompt hidden, so no separate "first token" seed is needed.
* **Threshold-commit + forced argmax**: each denoise iteration commits the
  masked positions whose top-1 probability exceeds ``threshold`` and always
  force-commits the single highest-confidence still-masked position per row, so
  a block of ``bd_size`` positions fully commits in ``<= bd_size`` iterations.

Loop design
-----------
The **inner** per-block denoise iterations run as a ``jax.lax.while_loop`` and
the **outer** block advance runs as a ``jax.lax.fori_loop`` (sequence length is
fixed in training, so the block count ``T // bd_size`` is static) — the whole
rollout is one jittable, ``stop_gradient``-able program. Generation is OFF the
student gradient path: the trainer runs this outside ``value_and_grad`` and wraps
the outputs in ``jax.lax.stop_gradient`` (the committed tokens are argmax indices,
which are non-differentiable regardless).
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

# forward_fn(tokens[B, T] int32, positions[B, T] int32) -> logits[B, T, V]
ForwardFn = Callable[[jax.Array, jax.Array], jax.Array]


def _shifted_logit_indices(seq_len: int) -> jax.Array:
  """Indices implementing the shifted-logit convention.

  ``shifted[:, i] = logits[:, max(i - 1, 0)]`` so the token at position ``i`` is
  predicted from hidden ``i - 1`` (clamped ``>= 0`` for position 0).
  """
  return jnp.maximum(jnp.arange(seq_len, dtype=jnp.int32) - 1, 0)


def _selective_log_softmax(logits: jax.Array, tokens: jax.Array) -> jax.Array:
  """Log-prob of ``tokens`` under ``logits`` (pure-jax; no tunix dependency).

  ``logits`` is ``(..., L, V)`` and ``tokens`` is ``(..., L)``; returns ``(..., L)``
  where entry ``p`` is ``log softmax(logits[..., p, :])[tokens[..., p]]``. Mirrors
  ``tunix.rl.common.selective_log_softmax`` so the diffusion logprob and the AR
  logprob agree numerically on the extraction step.
  """
  logits = logits.astype(jnp.float32)
  target = jnp.take_along_axis(logits, tokens[..., None], axis=-1).squeeze(-1)
  normalizer = jax.nn.logsumexp(logits, axis=-1)
  return target - normalizer


def _positions_from_mask(mask: jax.Array) -> jax.Array:
  """Absolute positions from a validity mask (RoPE), pure-jax.

  Reimplements ``tunix.sft.utils.build_positions_from_mask``
  (``cumsum(mask) - (cumsum(mask) >= 1)``) so this module stays jax-only: pad
  positions and the first real token share position ``0`` and real tokens then
  count ``0, 1, 2, ...``. Block-diffusion blocks are ``position // bd_size``, so
  using the SAME position convention as the AR path keeps the two logprob paths
  aligned.
  """
  positions = jnp.cumsum(mask.astype(jnp.int32), axis=-1)
  return positions - (positions >= 1).astype(jnp.int32)


def diffusion_commit(
    logits: jax.Array,
    mask: jax.Array,
    threshold: float,
    temperature: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
  """Per-position, threshold-based commit step for block-diffusion denoising.

  MaxText-side reimplementation of ``tpu_inference``'s ``diffusion_commit`` (that
  repo is a separate, read-only checkout, so the pure-jax logic is duplicated
  here rather than imported). Given per-position vocab logits and a boolean mask
  marking positions that are still masked (not yet committed), greedily commit
  the positions whose top-1 probability exceeds ``threshold``. To guarantee
  forward progress, the single highest-confidence still-masked position in each
  row is always committed even if it is below ``threshold`` (unless the row has
  no masked positions left).

  Args:
    logits: ``(..., L, V)`` float logits. Any number of leading batch/row dims is
      supported (e.g. ``(B, T, V)`` or ``(L, V)``).
    mask: ``(..., L)`` boolean array; ``True`` marks positions still masked and
      therefore eligible to be committed this step.
    threshold: scalar confidence threshold in ``[0, 1]``. A masked position
      commits when its top-1 softmax probability is strictly greater than this.
    temperature: optional softmax temperature; when ``> 0`` the logits are
      divided by it before the softmax, when ``0`` (default) the raw logits are
      used.

  Returns:
    ``(committed_token_ids, new_mask)`` where ``committed_token_ids`` is the
    ``(..., L)`` int32 top-1 token id for every position (values at still-masked
    positions should be ignored by the caller), and ``new_mask`` equals ``mask``
    with the newly-committed positions cleared (a strictly shrinking mask).
  """
  if temperature > 0.0:
    logits = logits / temperature
  probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

  top_prob = jnp.max(probs, axis=-1)
  top_tok = jnp.argmax(probs, axis=-1).astype(jnp.int32)

  mask_bool = mask.astype(bool)

  # Threshold commit: only sufficiently-confident masked positions commit.
  commit = (top_prob > threshold) & mask_bool

  # Progress guarantee: force the single highest-confidence still-masked position
  # in each row to commit. Non-masked positions are excluded from the argmax so
  # an already-committed position can never be re-selected, and rows with no
  # masked positions left force nothing.
  neg_inf = jnp.array(-jnp.inf, dtype=top_prob.dtype)
  masked_conf = jnp.where(mask_bool, top_prob, neg_inf)
  forced_idx = jnp.argmax(masked_conf, axis=-1)
  row_has_mask = jnp.any(mask_bool, axis=-1)
  forced_onehot = jax.nn.one_hot(forced_idx, mask_bool.shape[-1], dtype=bool)
  forced_onehot = forced_onehot & jnp.expand_dims(row_has_mask, axis=-1)
  commit = commit | forced_onehot

  new_mask = mask_bool & (~commit)
  return top_tok, new_mask


def diffusion_generate(
    forward_fn: ForwardFn,
    tokens: jax.Array,
    positions: jax.Array,
    gen_mask: jax.Array,
    *,
    mask_id: int,
    bd_size: int,
    threshold: float = 0.9,
    temperature: float = 0.0,
    max_denoise_steps: int = 0,
) -> Tuple[jax.Array, jax.Array]:
  """Generate a completion for a batch by block-diffusion denoising.

  Fills the ``gen_mask`` positions of ``tokens`` block-by-block (blocks aligned
  to ``bd_size``, matching the block-diffusion attention's ``position // bd_size``)
  using the injected student ``forward_fn``. Prompt positions (``gen_mask`` False)
  are held fixed.

  Args:
    forward_fn: injected cacheless student forward. See module docstring for the
      signature ``(tokens, positions) -> logits[B, T, V]``.
    tokens: ``[B, T]`` int32 batch; prompt region holds real tokens, the
      completion region is (re)seeded with ``mask_id`` internally.
    positions: ``[B, T]`` int32 absolute positions (for RoPE), passed through to
      ``forward_fn``.
    gen_mask: ``[B, T]`` boolean; ``True`` where the student should generate.
    mask_id: token id used to mark not-yet-committed canvas positions.
    bd_size: block size; ``T`` must be divisible by it.
    threshold: confidence threshold for committing a position.
    temperature: softmax temperature for the commit step (``0.0`` = argmax).
    max_denoise_steps: per-block denoise iteration cap; ``<= 0`` means use
      ``bd_size`` (which fully commits, since the forced-argmax progress guarantee
      commits ``>= 1`` masked position per iteration).

  Returns:
    ``(gen_tokens, committed_mask)`` where ``gen_tokens`` is the ``[B, T]`` int32
    sequence (prompt ⊕ generated completion, no ``mask_id`` left) and
    ``committed_mask`` is the ``[B, T]`` boolean mask of the positions the student
    generated (equals ``gen_mask`` once every block is filled).
  """
  tokens = jnp.asarray(tokens, dtype=jnp.int32)
  positions = jnp.asarray(positions, dtype=jnp.int32)
  gen_mask = jnp.asarray(gen_mask, dtype=bool)

  batch_size, seq_len = tokens.shape
  if bd_size <= 0:
    raise ValueError(f"bd_size must be > 0, got {bd_size}")
  if seq_len % bd_size != 0:
    raise ValueError(f"sequence length ({seq_len}) must be divisible by bd_size ({bd_size})")
  num_blocks = seq_len // bd_size
  eff_steps = bd_size if max_denoise_steps <= 0 else int(max_denoise_steps)

  shift_idx = _shifted_logit_indices(seq_len)  # [T]
  pos_ids = jnp.arange(seq_len, dtype=jnp.int32)  # [T]

  # Seed canvas: mask_id at generation positions, real tokens (prompt) elsewhere.
  canvas0 = jnp.where(gen_mask, jnp.array(mask_id, dtype=jnp.int32), tokens)  # [B, T]
  committed0 = jnp.zeros((batch_size, seq_len), dtype=bool)

  def block_body(blk, carry):
    canvas, committed = carry
    start = blk * bd_size
    in_block = (pos_ids >= start) & (pos_ids < start + bd_size)  # [T]
    block_gen = gen_mask & in_block[None, :]  # [B, T] positions to denoise this block
    has_gen = jnp.any(block_gen)

    def cond_fn(inner):
      _, mask, step, _ = inner
      under_cap = step < eff_steps
      # Always run step 0 so a degenerate all-committed block still terminates;
      # otherwise stop once every eligible position in the block is committed.
      more_work = jnp.logical_or(step == 0, jnp.any(mask))
      return jnp.logical_and(under_cap, more_work)

    def body_fn(inner):
      cv, mask, step, _ = inner
      logits = forward_fn(cv, positions).astype(jnp.float32)  # [B, T, V]
      shifted = logits[:, shift_idx, :]  # [B, T, V] — token i predicted from hidden i-1
      committed_tok, new_mask = diffusion_commit(shifted, mask, threshold, temperature)
      newly = jnp.logical_and(mask, jnp.logical_not(new_mask))
      cv = jnp.where(newly, committed_tok, cv)
      return (cv, new_mask, step + 1, committed_tok)

    def do_denoise(cv):
      init = (cv, block_gen, jnp.array(0, dtype=jnp.int32), cv)
      out_canvas, mask_final, _, last_tok = jax.lax.while_loop(cond_fn, body_fn, init)
      # If the step cap was hit before full commitment, force-fill any still-masked
      # positions with their argmax so mask_id never leaks into the output.
      return jnp.where(mask_final, last_tok, out_canvas)

    # Skip the (wasted) forward for prompt-only blocks with nothing to generate.
    canvas = jax.lax.cond(has_gen, do_denoise, lambda cv: cv, canvas)
    committed = committed | block_gen
    return (canvas, committed)

  canvas, committed = jax.lax.fori_loop(0, num_blocks, block_body, (canvas0, committed0))
  return canvas, committed


def diffusion_per_token_logps(
    model_apply: ForwardFn,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    completion_mask: jax.Array | None = None,
    *,
    bd_size: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    stop_gradient: bool = False,
) -> jax.Array:
  """Block-diffusion (TraceRL-style) per-token log-probability of a completion.

  THE single source of truth for the GRPO/DAPO importance ratio when the policy
  is a block-diffusion model. It is invoked at BOTH sites of the ratio — the
  rollout's old-policy logps (``MaxTextDiffusionRollout.get_per_token_logps``)
  and the loss's new-policy logps (the tunix ``per_token_logps_fn`` hook) — so the
  two are guaranteed identical and the ratio is unbiased. Do NOT reimplement the
  math at either call site; both must funnel through this function.

  Unlike an autoregressive next-token logprob, the score comes from a single
  block-diffusion forward over the whole committed sequence: each position attends
  bidirectionally within its block of ``bd_size`` tokens and block-causally across
  blocks (the model applies this mask internally when ``enable_block_diffusion`` is
  on; ``model_apply`` just forwards the full sequence). The completed sequence is
  then scored with the same shifted-logit convention the rollout generated it under
  (``_shifted_logit_indices``: ``shifted[:, p] = logits[:, p - 1]``, so the
  post-shift logits at position ``p`` predict token ``p`` — matching
  ``diffusion_generate``'s commit rule, which argmaxes ``logits[p - 1]`` onto token
  ``p``). Scoring under the generation rule is what makes the old-policy logp equal
  the probability the token was drawn with.

  Args:
    model_apply: cacheless block-diffusion forward ``(tokens[B, T], positions[B,
      T]) -> logits[B, T, V]`` (same signature/injection as ``diffusion_generate``,
      so it is exercisable on CPU with a stub).
    prompt_tokens: ``[B, P]`` int prompt ids (may be left-padded with ``pad_id``).
    completion_tokens: ``[B, C]`` int completion ids (may be right-padded).
    completion_mask: optional ``[B, C]`` mask; when provided the returned logps are
      zeroed outside it and it also defines completion validity for positions.
    bd_size: block size; ``P + C`` must be divisible by it (block-diffusion
      contract, matching ``diffusion_generate`` and the config validator).
    pad_id: pad id used to derive the prompt validity mask for positions.
    temperature: divides the logits before the softmax when not ``0.0``/``1.0``
      (matches ``tunix.rl.common.compute_per_token_logps``).
    stop_gradient: when True the result is wrapped in ``jax.lax.stop_gradient``
      (used for the rollout's old-policy logps; the loss passes ``False`` so the
      policy gradient flows).

  Returns:
    ``[B, C]`` per-token log-probabilities for the completion positions.
  """
  prompt_tokens = jnp.asarray(prompt_tokens, dtype=jnp.int32)
  completion_tokens = jnp.asarray(completion_tokens, dtype=jnp.int32)
  if bd_size <= 0:
    raise ValueError(f"bd_size must be > 0, got {bd_size}")

  prompt_len = prompt_tokens.shape[1]
  full_tokens = jnp.concatenate([prompt_tokens, completion_tokens], axis=1)  # [B, T]
  seq_len = full_tokens.shape[1]
  if seq_len % bd_size != 0:
    raise ValueError(f"prompt+completion length ({seq_len}) must be divisible by bd_size ({bd_size})")

  # Validity mask -> RoPE positions (block index = position // bd_size). Prompt
  # validity comes from pad_id; completion validity from completion_mask (or
  # non-pad when absent), matching the AR path so both logprob paths align.
  prompt_mask = prompt_tokens != pad_id
  if completion_mask is None:
    completion_valid = completion_tokens != pad_id
  else:
    completion_valid = completion_mask.astype(bool)
  full_mask = jnp.concatenate([prompt_mask, completion_valid], axis=1)
  positions = _positions_from_mask(full_mask)  # [B, T]

  # Single block-diffusion forward over the committed sequence.
  logits = model_apply(full_tokens, positions).astype(jnp.float32)  # [B, T, V]

  # Shifted-logit convention (reuse M3): post-shift logits at p predict token p.
  shift_idx = _shifted_logit_indices(seq_len)  # [T]
  shifted = logits[:, shift_idx, :]  # [B, T, V]
  if temperature != 0.0 and temperature != 1.0:
    shifted = shifted / temperature

  logps_full = _selective_log_softmax(shifted, full_tokens)  # [B, T]
  completion_logps = logps_full[:, prompt_len:]  # [B, C]

  if completion_mask is not None:
    completion_logps = completion_logps * completion_mask.astype(completion_logps.dtype)
  if stop_gradient:
    completion_logps = jax.lax.stop_gradient(completion_logps)
  return completion_logps
