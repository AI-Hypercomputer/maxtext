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

import contextlib
import dataclasses
import itertools
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from jax import Array
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=ungrouped-imports
from maxtext.integration.tunix.utils import VllmWeightMapping
from maxtext.layers import moe
from maxtext.models.models import Transformer


import jax

# Compatibility shims for JAX 0.11.0+ strict sharding assertions
_orig_wsc = jax.lax.with_sharding_constraint
_orig_top_k = jax.lax.top_k


def _compat_wsc(x, shardings):
  try:
    return _orig_wsc(x, shardings)
  except Exception:  # pylint: disable=broad-exception-caught
    return jax.sharding.reshard(x, shardings)


def _compat_top_k(operand, k, axis=-1):
  """Compat shim around jax.lax.top_k to reshard sharded reduction operands."""
  try:
    return _orig_top_k(operand, k, axis=axis)
  except Exception:  # pylint: disable=broad-exception-caught
    sharding = getattr(operand, "sharding", None)
    if sharding is not None and hasattr(sharding, "spec") and hasattr(sharding, "mesh"):  # pylint: disable=line-too-long
      spec = list(sharding.spec)
      idx = axis if axis >= 0 else len(spec) + axis
      if 0 <= idx < len(spec):
        spec[idx] = None
        target_sharding = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(*spec))  # pylint: disable=line-too-long
        try:
          operand = _orig_wsc(operand, target_sharding)
        except Exception:  # pylint: disable=broad-exception-caught
          operand = jax.sharding.reshard(operand, target_sharding)
    return _orig_top_k(operand, k, axis=axis)


jax.lax.with_sharding_constraint = _compat_wsc
# pyrefly: ignore[bad-assignment]
jax.lax.top_k = _compat_top_k


def _segment_ids_from_attention_mask(attention_mask: Array, input_tokens: Array) -> Array:
  """Recovers Tunix's per-token non-pad mask from its `[B, L, L]` attention mask.

  MaxText takes no precomputed mask: it derives one per layer from
  `decoder_segment_ids` plus causality, and flash and splash take a
  `SequenceDescriptor` built from those ids rather than a dense array. Using
  Tunix's mask therefore means converting it, not passing it through.

  Nothing is lost. `tunix/sft/utils.py:make_causal_attn_mask` builds
  `input_mask[..., None, :] * tril`, masking the key side only, so the last query
  row carries the full mask. Deriving the ids here rather than from `pad_id` also
  leaves one source of truth about which tokens are padding instead of two.

  Raises:
    ValueError: If `attention_mask` is not `[B, L, L]` against `input_tokens`.
      Shapes are static, so this raises under `jit` as well as eagerly.
  """
  attention_mask = jnp.asarray(attention_mask)
  batch, seq_len = input_tokens.shape
  if attention_mask.shape != (batch, seq_len, seq_len):
    raise ValueError(
        f"attention_mask has shape {attention_mask.shape}, expected {(batch, seq_len, seq_len)} "
        f"for input_tokens of shape {input_tokens.shape}. The last query row is read as Tunix's "
        "per-token non-pad mask, which requires that shape."
    )
  return attention_mask[:, -1, :].astype(jnp.int32)


# Attribute `collect_moe_buffer_probe` stamps on the model copy the loss runs (see there).
MOE_BUFFER_PROBE_TRACE_ID_ATTR = "moe_buffer_probe_trace_id"


@dataclasses.dataclass
class MoeBufferProbeCollector:
  """log_required_ragged_buffer_factor probe values reported by adapter forward passes, reduced in `ref`."""

  trace_id: int
  ref: Any  # jax Ref holding the reduced [len(moe.MOE_BUFFER_PROBE_METRICS)] float32 vector.
  reported: bool = False  # Whether any forward pass reported (known at trace time).


_moe_buffer_probe_collector: Optional[MoeBufferProbeCollector] = None
_moe_buffer_probe_trace_ids = itertools.count(1)


@contextlib.contextmanager
def collect_moe_buffer_probe(model: nnx.Module):
  """Collects the probe values of every forward pass of `model` (or a copy of it) inside the block.

  Tunix losses do not run `model` itself: they split it and run a copy inside their own `jax.jit` (e.g.
  `common.compute_per_token_logps`), so what the MoE layers `sow` never reaches the caller, and a value taken
  out of that inner trace by a Python side channel escapes its scope (UnexpectedTracerError). Instead the
  adapter folds each pass's values into a JAX Ref created here, in the caller's trace; JAX carries Ref writes
  across jit boundaries. `model` is stamped with a per-block trace id: it is part of the graphdef a Tunix loss
  passes to its jitted function as a static argument, so that function is retraced, with the Ref, rather
  than reusing a trace cached by a call outside the block (e.g. old log-probs), which would report nothing.

  Read `collector.ref[...]` (reduced like `moe.reduce_moe_buffer_probe`) when `collector.reported`. A loss
  that runs the model more than once per batch reports each pass, so moe_dropped_rows_total counts each.
  """
  global _moe_buffer_probe_collector
  collector = MoeBufferProbeCollector(
      trace_id=next(_moe_buffer_probe_trace_ids),
      ref=jax.new_ref(jnp.zeros((len(moe.MOE_BUFFER_PROBE_METRICS),), jnp.float32)),
  )
  setattr(model, MOE_BUFFER_PROBE_TRACE_ID_ATTR, collector.trace_id)
  previous, _moe_buffer_probe_collector = _moe_buffer_probe_collector, collector
  try:
    yield collector
  finally:
    _moe_buffer_probe_collector = previous


def _report_moe_buffer_probe(adapter: nnx.Module) -> None:
  """Folds the probe values the last forward pass sowed into the active collector's Ref, if it is ours."""
  collector = _moe_buffer_probe_collector
  if collector is None or getattr(adapter, MOE_BUFFER_PROBE_TRACE_ID_ATTR, None) != collector.trace_id:
    return
  values = []
  for path, value in nnx.state(adapter.base, nnx.Intermediate).flat_state():
    if path and path[-1] == "moe_buffer_probe":
      if hasattr(value, "get_value"):
        value = value.get_value()
      elif hasattr(value, "value"):
        value = value.value
      values.extend(jax.tree.leaves(value))
  if values:
    collector.ref[...] = moe.reduce_moe_buffer_probe([collector.ref[...]] + values)
    collector.reported = True


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
    # Lowest-priority source of segment ids, used only when Tunix supplies
    # neither `segment_ids` nor an `attention_mask` (`__call__` has the order).
    # Synthesizes them per token -- 1 for non-pad, 0 for pad -- so MaxText's
    # segment-based mask stops non-pad queries attending to pad keys. Without it
    # the adapter forwards `decoder_segment_ids=None`, MaxText falls back to
    # causal-only masking, and padding is attended to as real input, silently
    # corrupting trainer log-probs on every batch. Rollout-side vLLM is
    # unaffected: its scheduler batches without padding.
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
      forced_routed_experts: Optional[Array] = None,
      segment_ids: Optional[Array] = None,
      skip_lm_head: bool = False,
  ) -> Tuple[Array, None]:
    """Forward compatible with Tunix Trainers default loss.

    Returns logits (or hidden_states when skip_lm_head=True), None.

    `segment_ids` is the name Tunix uses for packed-sequence segment ids: it
    forwards them only to models whose call signature has a parameter of that
    exact name. They are MaxText's `decoder_segment_ids`; when both are given,
    `segment_ids` wins so packed rows keep per-sequence attention isolation.

    Three sources can supply the segment ids, in descending priority:
    explicit segment ids, Tunix's `attention_mask`, then `pad_id` synthesis.
    """
    if segment_ids is not None:
      decoder_segment_ids = segment_ids
    if decoder_segment_ids is None and attention_mask is not None:
      decoder_segment_ids = _segment_ids_from_attention_mask(attention_mask, input_tokens)
    if decoder_segment_ids is None and self._pad_id is not None:
      decoder_segment_ids = (input_tokens != self._pad_id).astype(jnp.int32)
    outputs = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
        forced_routed_experts=forced_routed_experts,
        skip_lm_head=skip_lm_head,
    )
    _report_moe_buffer_probe(self)
    return outputs, None

  def compute_final_logits(self, hidden_states: Array) -> Array:
    """Projects a chunk of pre-norm hidden states [B, Chunk, Emb] to logits [B, Chunk, Vocab]."""
    return self.base.logits_from_hidden_states_for_vocab_tiling(
        hidden_states,
        deterministic=True,
        model_mode=getattr(self.base, "model_mode", "train"),
    )

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
