# Copyright 2023–2025 Google LLC
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

"""Adapter for integrating MaxText Transformer models with Tunix.

This module provides the `TunixMaxTextAdapter` class, which wraps a MaxText
Transformer model to expose a call signature compatible with Tunix Trainers.
It also handles weight mapping for compatibility with Hugging Face models.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=ungrouped-imports
from maxtext.integration.tunix.utils import VllmWeightMapping
from maxtext.models.models import Transformer


# [NAVI patch E] Opt-in microbatch memory/shape logging. Enable with NAVI_MEM_LOG=1.
# Prints tensor shape/dtype/bytes at the trainer forward boundary + device HBM
# in-use, so we can empirically prove the [micro, seq, vocab] logits
# materialization and measure the chunked/offload reduction. Zero cost when off.
_NAVI_MEM_LOG = os.environ.get("NAVI_MEM_LOG", "0") == "1"
_navi_mem_seen: set[str] = set()


def _navi_log_mem(tag: str, **arrays: Any) -> None:
  """Opt-in (NAVI_MEM_LOG=1) logging of tensor shape/dtype/bytes + device HBM
  in-use at the trainer forward boundary. First occurrence per tag only."""
  if not _NAVI_MEM_LOG:
    return
  try:
    # Only log the first occurrence of each tag to avoid trace-time spam.
    if tag in _navi_mem_seen:
      return
    _navi_mem_seen.add(tag)
    parts = []
    for name, a in arrays.items():
      if a is None:
        continue
      shp = getattr(a, "shape", None)
      dt = getattr(a, "dtype", None)
      nbytes = None
      if shp is not None and dt is not None:
        try:
          nbytes = int(jnp.prod(jnp.array(shp))) * jnp.dtype(dt).itemsize
        except Exception:  # pylint: disable=broad-exception-caught
          nbytes = None
      gb = f"{nbytes/1e9:.2f}GB" if nbytes else "?"
      parts.append(f"{name}={shp}:{dt}:{gb}")
    hbm = ""
    try:
      d = jax.devices()[0]
      if hasattr(d, "memory_stats"):
        ms = d.memory_stats() or {}
        used = ms.get("bytes_in_use")
        peak = ms.get("peak_bytes_in_use")
        if used is not None:
          hbm = f" | HBM_in_use={used/1e9:.1f}GB peak={((peak or 0))/1e9:.1f}GB"
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    print(f"[NAVI_MEM_LOG] {tag}: {' '.join(parts)}{hbm}", flush=True)
  except Exception:  # pylint: disable=broad-exception-caught
    pass


class TunixMaxTextAdapter(nnx.Module):
  """Adapter exposing Tunix Trainer call signature over a Transformer model."""

  def __init__(
      self,
      base_model: Transformer,
      use_standalone_mappings: bool = True,
      use_no_op_mappings: bool = False,
      pad_id: Optional[int] = None,
  ):
    super().__init__()
    self.base = base_model
    self._vllm_weight_mapping = VllmWeightMapping(
        self.base.config.model_name,
        HF_MODEL_CONFIGS[self.base.config.model_name].to_dict(),
        use_standalone_mappings,
    )
    self.use_no_op_mappings = use_no_op_mappings
    # When `pad_id` is provided AND tunix passes `decoder_segment_ids=None`,
    # synthesize per-token segment_ids (1 for non-pad, 0 for pad). MaxText's
    # segment-based attention mask then blocks queries at non-pad positions
    # from attending to pad-position keys. Without this, the adapter forwards
    # `decoder_segment_ids=None` and MaxText falls back to causal-only masking,
    # so padded tokens get attended to as if they were real input — silently
    # corrupting trainer log-probs on every batch. (Rollout-side vLLM does the
    # right thing because it batches without padding via its scheduler.)
    self._pad_id = pad_id

  # ------------------------------------------------------------------ #
  # Tunix call signature
  # ------------------------------------------------------------------ #
  def __call__(
      self,
      input_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Optional[Any],  # Tunix currently passes None from Trainers
      attention_mask: Optional[Array],  # [B, L, L] or None
      decoder_segment_ids: Optional[Array] = None,
      output_hidden_states: bool = False,  # ignored
      skip_lm_head: bool = False,  # [NAVI patch C] chunked-logps memory lever
  ) -> Tuple[Array, None]:
    """Forward compatible with Tunix Trainers default loss.
    Returns logits, None.

    When ``skip_lm_head=True`` (used by tunix ``compute_chunked_logps`` when
    ``compute_logps_chunk_size > 0``), the expensive [B, L, vocab] output-head
    projection is deferred; the decoder's final hidden state [B, L, hidden] is
    returned instead and the caller projects it chunk-by-chunk via
    ``compute_final_logits`` under ``jax.lax.scan`` + ``nnx.remat``. This caps
    the LM-head logits peak at [B, chunk, vocab] instead of [B, L, vocab],
    lifting the train_micro-batch HBM ceiling. Math is identical: per-token
    log-softmax is independent along the vocab axis (see smoke proof).
    """
    if decoder_segment_ids is None and self._pad_id is not None:
      decoder_segment_ids = (input_tokens != self._pad_id).astype(jnp.int32)
    if skip_lm_head:
      # Return the decoder hidden state; LM head applied later per-chunk.
      hidden_state = self.base(
          decoder_input_tokens=input_tokens,
          decoder_positions=positions,
          decoder_segment_ids=decoder_segment_ids,
          skip_lm_head=True,
      )
      _navi_log_mem("actor_fwd.skip_lm_head", input_tokens=input_tokens, hidden_state=hidden_state)
      return hidden_state, None
    logits = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    _navi_log_mem("actor_fwd.full_logits", input_tokens=input_tokens, logits=logits)
    return logits, None

  def compute_final_logits(self, hidden_states: Array) -> Array:
    """[NAVI patch C] LM-head projection over precomputed hidden states.

    Required by tunix ``compute_chunked_logps``. Delegates to the MaxText
    Transformer's existing ``logits_from_hidden_states_for_vocab_tiling`` (the
    same output-head used by native vocab-tiling), so no new params/graph.

    NOTE: tunix calls this inside ``jax.lax.scan`` + ``nnx.remat``. The output
    head's ``nn.Dropout`` (a no-op here: dropout_rate=0 + deterministic=True) is
    made RNG-free at its source by the ``skip_lm_head`` deterministic guard in
    apply_output_head (see layers/decoders.py) so it does not mutate an RngCount
    across the scan's trace level (which would raise flax TraceContextError).
    """
    from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
    # Pass a STATIC rng key so the ToNNX decoder wrapper takes its jax.Array branch
    # (_rngs={"params": key}) instead of calling stream() on live Rngs. stream() would
    # mutate an RngCount across the enclosing jax.lax.scan trace level (tunix
    # compute_chunked_logps) and raise flax TraceContextError. The projection is
    # deterministic (dropout guarded off), so this key is never used to draw randomness.
    _static_key = jax.random.PRNGKey(0)
    logits = self.base.logits_from_hidden_states_for_vocab_tiling(
        hidden_states=hidden_states,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs=_static_key,
    )
    _navi_log_mem("actor_fwd.chunk_logits", hidden_states=hidden_states, logits=logits)
    return logits

  def to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_mapping()

  def to_hf_transpose_keys(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.lora_to_hf_mappings()
