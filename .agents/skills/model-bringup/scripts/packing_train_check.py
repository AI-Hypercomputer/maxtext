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

"""Model-agnostic sequence-packing training verification utility and CLI.

Verifies four invariants required before training a model with packed sequences
(`packing=True` / multi-document `decoder_segment_ids` and reset
`decoder_positions`):
  1. Forward logit & masked loss isolation: packed variable-length documents
     separated by `0` padding gaps match standalone single-document padded runs,
     and mutating doc-2 or padding tokens leaves doc-1 logits invariant (and
     vice versa).
  2. Backward per-leaf gradient superposition & cross-doc zero-leakage: total
     packed parameter gradients equal the token-weighted sum of standalone
     document gradients across every parameter leaf, while
     `d(loss_doc2) / d(embed_doc1)` and `d(loss) / d(embed_pad)` are zero.
  3. Multi-step `train_step` + Optax trajectory parity: `K >= 2` optimizer
     update steps on packed batches match accumulated standalone document
     updates across step losses and all updated parameter leaves.
  4. Built-in negative control (mandatory mutant kill): collapsing packed
     non-padding `decoder_segment_ids` into `segment_id = 1` must cause the
     forward, backward, and multi-step training checks to fail while staying
     finite; exits non-zero if the negative control fails to fire.

Usage:
  # Standalone self-check on a pure-JAX causal RoPE Transformer:
  python3 .agents/skills/model-bringup/scripts/packing_train_check.py

  # Against a MaxText mini-config (CPU or TPU):
  PYTHONPATH=$PWD/src python3 .agents/skills/model-bringup/scripts/packing_train_check.py \
    --model-name=qwen3-0.6b
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import inspect
import json
import os
import re
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

_EMBED_PATH_RE = re.compile(r"embed|tok_embeddings|wte", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class PackingTolerance:
  """Numerical thresholds for packed vs. standalone training verification."""

  fwd_atol: float = 1e-5
  fwd_rtol: float = 1e-5
  fwd_frob_rel: float = 1e-5
  invariance_atol: float = 1e-6
  grad_atol: float = 1e-5
  grad_rtol: float = 1e-5
  grad_frob_rel: float = 1e-5
  zero_grad_atol: float = 1e-7
  loss_atol: float = 1e-5
  loss_rtol: float = 1e-5
  param_atol: float = 5e-5
  param_rtol: float = 5e-5
  param_frob_rel: float = 5e-5
  tied_bf16_embed_grad_tol: float | None = None
  tied_bf16_embed_param_tol: float | None = None


@dataclasses.dataclass(frozen=True)
class DocumentSlot:
  """Metadata for one document packed into a sequence."""

  doc_id: int
  start: int
  length: int

  @property
  def end(self) -> int:
    """Exclusive end index of the document span."""
    return self.start + self.length


@dataclasses.dataclass(frozen=True)
class PackedStepBatch:
  """Paired packed batch and standalone single-document padded batches."""

  tokens_packed: jax.Array
  targets_packed: jax.Array
  positions_packed: jax.Array
  segment_ids_packed: jax.Array
  slots: tuple[DocumentSlot, ...]
  standalone_tokens: tuple[jax.Array, ...]
  standalone_targets: tuple[jax.Array, ...]
  standalone_positions: tuple[jax.Array, ...]
  standalone_segment_ids: tuple[jax.Array, ...]


@dataclasses.dataclass
class PackingParityReport:
  """Structured result of `verify_packed_training_parity`."""

  passed: bool
  negative_control_fired: bool
  forward_metrics: dict[str, float]
  backward_metrics: dict[str, float]
  training_metrics: dict[str, Any]
  negative_control_metrics: dict[str, Any]
  failures: list[str]

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary summary."""
    return dataclasses.asdict(self)


def _leaf_path_str(path: tuple[Any, ...]) -> str:
  """Formats a JAX pytree path tuple as a slash-separated string."""
  parts: list[str] = []
  for key in path:
    if hasattr(key, "key"):
      parts.append(str(key.key))
    elif hasattr(key, "idx"):
      parts.append(str(key.idx))
    elif hasattr(key, "name"):
      parts.append(str(key.name))
    else:
      parts.append(str(key))
  return "/".join(parts)


def _compare_arrays(
    cand: np.ndarray,
    ref: np.ndarray,
    *,
    atol: float,
    rtol: float,
    frob_rel: float,
) -> tuple[bool, dict[str, float]]:
  """Compares two arrays in float64 using elementwise and relative Frobenius bounds."""
  if cand.shape != ref.shape:
    raise ValueError(f"Shape mismatch: {cand.shape} vs {ref.shape}")
  c64 = np.asarray(cand, dtype=np.float64)
  r64 = np.asarray(ref, dtype=np.float64)
  if not (np.all(np.isfinite(c64)) and np.all(np.isfinite(r64))):
    return False, {"max_abs": np.nan, "frob_rel": np.nan, "finite": 0.0}
  diff = np.abs(c64 - r64)
  ref_inf = float(np.max(np.abs(r64))) if r64.size else 0.0
  ref_l2 = float(np.linalg.norm(r64))
  max_abs = float(np.max(diff)) if diff.size else 0.0
  rel_frob = float(np.linalg.norm(diff)) / max(1.0, ref_l2)
  elem_ok = bool(np.all(diff <= atol * max(1.0, ref_inf) + rtol * np.abs(r64)))
  ok = elem_ok and (rel_frob <= frob_rel)
  return ok, {"max_abs": max_abs, "frob_rel": rel_frob, "finite": 1.0}


def _masked_token_ce_sum(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
  """Computes unnormalized masked cross-entropy sum in float32."""
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
  return jnp.sum(nll * mask.astype(jnp.float32))


def build_packed_step_batch(
    *,
    doc_lengths: Sequence[int] = (3, 2),
    pad_gaps: Sequence[int] = (1, 2),
    batch_size: int = 1,
    vocab_size: int = 128,
    seed: int = 0,
) -> PackedStepBatch:
  """Constructs a multi-document packed batch with `0` padding gaps and standalone twins."""
  if len(doc_lengths) < 2:
    raise ValueError(f"doc_lengths must pack >= 2 documents, got {doc_lengths}")
  if len(pad_gaps) != len(doc_lengths):
    raise ValueError(f"pad_gaps length {len(pad_gaps)} != doc_lengths length {len(doc_lengths)}")
  if any(length < 1 for length in doc_lengths):
    raise ValueError(f"Every document length must be >= 1, got {doc_lengths}")
  if any(gap < 0 for gap in pad_gaps) or sum(pad_gaps) < 1:
    raise ValueError(f"pad_gaps must be non-negative with at least one 0-padding slot, got {pad_gaps}")
  if vocab_size < 8:
    raise ValueError(f"vocab_size must be >= 8, got {vocab_size}")

  seq_len = int(sum(doc_lengths) + sum(pad_gaps))
  rng = np.random.default_rng(seed)

  tok_packed = np.zeros((batch_size, seq_len), dtype=np.int32)
  tgt_packed = np.zeros((batch_size, seq_len), dtype=np.int32)
  pos_packed = np.zeros((batch_size, seq_len), dtype=np.int32)
  seg_packed = np.zeros((batch_size, seq_len), dtype=np.int32)

  slots: list[DocumentSlot] = []
  solo_tokens: list[jax.Array] = []
  solo_targets: list[jax.Array] = []
  solo_positions: list[jax.Array] = []
  solo_segments: list[jax.Array] = []

  cursor = 0
  for idx, (length, gap) in enumerate(zip(doc_lengths, pad_gaps)):
    doc_id = idx + 1
    doc_tok = rng.integers(1, vocab_size, size=(batch_size, length + 1), dtype=np.int32)
    in_tok = doc_tok[:, :-1]
    out_tok = doc_tok[:, 1:]
    pos_row = np.broadcast_to(np.arange(length, dtype=np.int32), (batch_size, length))

    tok_packed[:, cursor : cursor + length] = in_tok
    tgt_packed[:, cursor : cursor + length] = out_tok
    pos_packed[:, cursor : cursor + length] = pos_row
    seg_packed[:, cursor : cursor + length] = doc_id
    slots.append(DocumentSlot(doc_id=doc_id, start=cursor, length=length))

    s_tok = np.zeros((batch_size, seq_len), dtype=np.int32)
    s_tgt = np.zeros((batch_size, seq_len), dtype=np.int32)
    s_pos = np.zeros((batch_size, seq_len), dtype=np.int32)
    s_seg = np.zeros((batch_size, seq_len), dtype=np.int32)
    s_tok[:, :length] = in_tok
    s_tgt[:, :length] = out_tok
    s_pos[:, :length] = pos_row
    s_seg[:, :length] = 1

    solo_tokens.append(jnp.asarray(s_tok))
    solo_targets.append(jnp.asarray(s_tgt))
    solo_positions.append(jnp.asarray(s_pos))
    solo_segments.append(jnp.asarray(s_seg))
    cursor += length + gap

  return PackedStepBatch(
      tokens_packed=jnp.asarray(tok_packed),
      targets_packed=jnp.asarray(tgt_packed),
      positions_packed=jnp.asarray(pos_packed),
      segment_ids_packed=jnp.asarray(seg_packed),
      slots=tuple(slots),
      standalone_tokens=tuple(solo_tokens),
      standalone_targets=tuple(solo_targets),
      standalone_positions=tuple(solo_positions),
      standalone_segment_ids=tuple(solo_segments),
  )


def _check_forward_isolation(
    forward_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    params: Any,
    batch: PackedStepBatch,
    *,
    vocab_size: int,
    tol: PackingTolerance,
) -> tuple[bool, dict[str, float], list[str]]:
  """Checks packed vs. standalone per-doc logits/losses and mutation invariance."""
  failures: list[str] = []
  logits_packed = np.asarray(
      forward_fn(
          params,
          batch.tokens_packed,
          batch.positions_packed,
          batch.segment_ids_packed,
      ),
      dtype=np.float32,
  )
  if not np.all(np.isfinite(logits_packed)):
    return False, {"finite": 0.0}, ["forward: non-finite packed logits"]

  max_logit_abs = 0.0
  max_logit_frob = 0.0
  max_loss_abs = 0.0
  bsz = int(batch.tokens_packed.shape[0])

  for slot, s_tok, s_tgt, s_pos, s_seg in zip(
      batch.slots,
      batch.standalone_tokens,
      batch.standalone_targets,
      batch.standalone_positions,
      batch.standalone_segment_ids,
  ):
    logits_solo = np.asarray(forward_fn(params, s_tok, s_pos, s_seg), dtype=np.float32)
    if not np.all(np.isfinite(logits_solo)):
      return (
          False,
          {"finite": 0.0},
          [f"forward: non-finite standalone logits for doc {slot.doc_id}"],
      )
    packed_slice = logits_packed[:, slot.start : slot.end, :]
    solo_slice = logits_solo[:, : slot.length, :]
    ok, m = _compare_arrays(
        packed_slice,
        solo_slice,
        atol=tol.fwd_atol,
        rtol=tol.fwd_rtol,
        frob_rel=tol.fwd_frob_rel,
    )
    max_logit_abs = max(max_logit_abs, m["max_abs"])
    max_logit_frob = max(max_logit_frob, m["frob_rel"])
    if not ok:
      failures.append(f"forward doc {slot.doc_id} logit mismatch: {m}")

    doc_mask_packed = (batch.segment_ids_packed == slot.doc_id).astype(jnp.float32)
    doc_mask_solo = (s_seg == 1).astype(jnp.float32)
    denom = float(bsz * slot.length)
    loss_packed_doc = float(
        _masked_token_ce_sum(jnp.asarray(logits_packed), batch.targets_packed, doc_mask_packed) / denom
    )
    loss_solo_doc = float(_masked_token_ce_sum(jnp.asarray(logits_solo), s_tgt, doc_mask_solo) / denom)
    loss_diff = abs(loss_packed_doc - loss_solo_doc)
    max_loss_abs = max(max_loss_abs, loss_diff)
    if loss_diff > tol.loss_atol + tol.loss_rtol * max(1.0, abs(loss_solo_doc)):
      failures.append(f"forward doc {slot.doc_id} loss diff {loss_diff:.3e} exceeds tolerance")

  tok_np = np.asarray(batch.tokens_packed)
  seg_np = np.asarray(batch.segment_ids_packed)

  # Mutate doc 2+ tokens and assert doc 1 logits remain invariant.
  tok_mut_doc2 = tok_np.copy()
  tok_mut_doc2[seg_np >= 2] = (tok_mut_doc2[seg_np >= 2] % (vocab_size - 3)) + 2
  logits_mut_doc2 = np.asarray(
      forward_fn(
          params,
          jnp.asarray(tok_mut_doc2),
          batch.positions_packed,
          batch.segment_ids_packed,
      ),
      dtype=np.float32,
  )
  doc1_mut_diff = float(np.max(np.abs(logits_mut_doc2[seg_np == 1] - logits_packed[seg_np == 1])))
  if not np.isfinite(doc1_mut_diff) or doc1_mut_diff > tol.invariance_atol:
    failures.append(f"forward: mutating doc>=2 leaked into doc 1 (max_abs={doc1_mut_diff:.3e})")

  # Mutate doc 1 tokens and assert doc 2+ logits remain invariant.
  tok_mut_doc1 = tok_np.copy()
  tok_mut_doc1[seg_np == 1] = (tok_mut_doc1[seg_np == 1] % (vocab_size - 3)) + 2
  logits_mut_doc1 = np.asarray(
      forward_fn(
          params,
          jnp.asarray(tok_mut_doc1),
          batch.positions_packed,
          batch.segment_ids_packed,
      ),
      dtype=np.float32,
  )
  doc2_mut_diff = float(np.max(np.abs(logits_mut_doc1[seg_np >= 2] - logits_packed[seg_np >= 2])))
  if not np.isfinite(doc2_mut_diff) or doc2_mut_diff > tol.invariance_atol:
    failures.append(f"forward: mutating doc 1 leaked into doc>=2 (max_abs={doc2_mut_diff:.3e})")

  # Mutate padding tokens (seg == 0) and assert all valid doc logits stay invariant.
  tok_mut_pad = tok_np.copy()
  tok_mut_pad[seg_np == 0] = min(vocab_size - 1, 5)
  logits_mut_pad = np.asarray(
      forward_fn(
          params,
          jnp.asarray(tok_mut_pad),
          batch.positions_packed,
          batch.segment_ids_packed,
      ),
      dtype=np.float32,
  )
  pad_mut_diff = float(np.max(np.abs(logits_mut_pad[seg_np > 0] - logits_packed[seg_np > 0])))
  if not np.isfinite(pad_mut_diff) or pad_mut_diff > tol.invariance_atol:
    failures.append(f"forward: mutating pad tokens leaked into valid docs (max_abs={pad_mut_diff:.3e})")

  metrics = {
      "finite": 1.0,
      "max_logit_abs_diff": max_logit_abs,
      "max_logit_frob_rel": max_logit_frob,
      "max_loss_abs_diff": max_loss_abs,
      "doc1_invariance_under_doc2_mut": doc1_mut_diff,
      "doc2_invariance_under_doc1_mut": doc2_mut_diff,
      "valid_invariance_under_pad_mut": pad_mut_diff,
  }
  return not failures, metrics, failures


def _check_embedding_zero_leakage(
    *,
    forward_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    params: Any,
    batch: PackedStepBatch,
    vocab_size: int,
    tol: PackingTolerance,
    embed_fn: Callable[[Any, jax.Array], jax.Array] | None,
    forward_from_embeds_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array] | None,
) -> tuple[dict[str, float], list[str]]:
  """Checks `d(loss_doc2)/d(embed_doc1) == 0` and `d(loss)/d(embed_pad) == 0`."""
  failures: list[str] = []
  seg_np = np.asarray(batch.segment_ids_packed)
  bsz, seq_len = seg_np.shape
  doc2_mask = (batch.segment_ids_packed == 2).astype(jnp.float32)
  valid_mask = (batch.segment_ids_packed > 0).astype(jnp.float32)

  if embed_fn is not None and forward_from_embeds_fn is not None:
    emb_in = embed_fn(params, batch.tokens_packed)

    def _loss_doc2_emb(emb: jax.Array) -> jax.Array:
      logits = forward_from_embeds_fn(params, emb, batch.positions_packed, batch.segment_ids_packed)
      return _masked_token_ce_sum(logits, batch.targets_packed, doc2_mask)

    def _loss_total_emb(emb: jax.Array) -> jax.Array:
      logits = forward_from_embeds_fn(params, emb, batch.positions_packed, batch.segment_ids_packed)
      return _masked_token_ce_sum(logits, batch.targets_packed, valid_mask)

    pos_grad_doc2 = np.asarray(jax.grad(_loss_doc2_emb)(emb_in), dtype=np.float64)
    pos_grad_total = np.asarray(jax.grad(_loss_total_emb)(emb_in), dtype=np.float64)
  else:
    max_probe_id = bsz * seq_len
    if vocab_size <= max_probe_id + 2:
      raise ValueError(
          f"vocab_size={vocab_size} must exceed batch_size*seq_len+2={max_probe_id + 2} " "when embed_fn is not provided"
      )
    probe_ids_np = np.arange(1, max_probe_id + 1, dtype=np.int32).reshape(bsz, seq_len)
    probe_ids = jnp.asarray(probe_ids_np)
    tail_vocab = vocab_size - (max_probe_id + 1)
    probe_targets = jnp.asarray(np.asarray(batch.targets_packed) % tail_vocab, dtype=jnp.int32)

    def _probe_doc2_loss(p: Any) -> jax.Array:
      logits = forward_fn(p, probe_ids, batch.positions_packed, batch.segment_ids_packed)
      sliced = logits.astype(jnp.float32)[..., max_probe_id + 1 :]
      return _masked_token_ce_sum(sliced, probe_targets, doc2_mask)

    def _probe_total_loss(p: Any) -> jax.Array:
      logits = forward_fn(p, probe_ids, batch.positions_packed, batch.segment_ids_packed)
      sliced = logits.astype(jnp.float32)[..., max_probe_id + 1 :]
      return _masked_token_ce_sum(sliced, probe_targets, valid_mask)

    g_doc2 = jax.grad(_probe_doc2_loss)(params)
    g_tot = jax.grad(_probe_total_loss)(params)
    flat_doc2 = jax.tree_util.tree_flatten_with_path(g_doc2)[0]
    flat_tot = dict(
        (_leaf_path_str(p), np.asarray(v, dtype=np.float64)) for p, v in jax.tree_util.tree_flatten_with_path(g_tot)[0]
    )
    cands = [
        (_leaf_path_str(p), np.asarray(v, dtype=np.float64))
        for p, v in flat_doc2
        if hasattr(v, "ndim") and v.ndim == 2 and v.shape[0] == vocab_size
    ]
    named = [(k, v) for k, v in cands if _EMBED_PATH_RE.search(k)]
    chosen = named if named else cands
    if not chosen:
      raise ValueError(f"Could not find 2-D token embedding leaf of shape ({vocab_size}, D)")
    emb_key, emb_g_doc2 = chosen[0]
    emb_g_tot = flat_tot[emb_key]
    pos_grad_doc2 = emb_g_doc2[probe_ids_np]
    pos_grad_total = emb_g_tot[probe_ids_np]

  if not (np.all(np.isfinite(pos_grad_doc2)) and np.all(np.isfinite(pos_grad_total))):
    return (
        {"embed_grad_finite": 0.0},
        ["backward: non-finite input-embedding gradients"],
    )

  g_doc2_on_doc1 = float(np.max(np.abs(pos_grad_doc2[seg_np == 1])))
  g_doc2_on_pad = float(np.max(np.abs(pos_grad_doc2[seg_np == 0])))
  g_total_on_pad = float(np.max(np.abs(pos_grad_total[seg_np == 0])))
  g_doc2_on_doc2_norm = float(np.linalg.norm(pos_grad_doc2[seg_np == 2]))

  if g_doc2_on_doc1 > tol.zero_grad_atol:
    failures.append(f"backward: non-zero cross-doc gradient d(loss_doc2)/d(embed_doc1)={g_doc2_on_doc1:.3e}")
  if g_doc2_on_pad > tol.zero_grad_atol:
    failures.append(f"backward: non-zero pad gradient d(loss_doc2)/d(embed_pad)={g_doc2_on_pad:.3e}")
  if g_total_on_pad > tol.zero_grad_atol:
    failures.append(f"backward: non-zero total-loss pad gradient d(loss)/d(embed_pad)={g_total_on_pad:.3e}")
  if g_doc2_on_doc2_norm <= 0.0:
    failures.append("backward: zero within-document gradient norm ||d(loss_doc2)/d(embed_doc2)||_2")

  return {
      "embed_grad_finite": 1.0,
      "max_d_doc2_d_embed_doc1": g_doc2_on_doc1,
      "max_d_doc2_d_embed_pad": g_doc2_on_pad,
      "max_d_total_d_embed_pad": g_total_on_pad,
      "norm_d_doc2_d_embed_doc2": g_doc2_on_doc2_norm,
  }, failures


def _check_backward_superposition(
    *,
    forward_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    params: Any,
    batch: PackedStepBatch,
    vocab_size: int,
    tol: PackingTolerance,
    embed_fn: Callable[[Any, jax.Array], jax.Array] | None,
    forward_from_embeds_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array] | None,
) -> tuple[bool, dict[str, float], list[str]]:
  """Checks per-leaf parameter gradient superposition and zero cross-doc embed leakage."""
  failures: list[str] = []
  bsz = int(batch.tokens_packed.shape[0])
  total_valid = float(bsz * sum(s.length for s in batch.slots))

  def _packed_loss(p: Any) -> jax.Array:
    logits = forward_fn(p, batch.tokens_packed, batch.positions_packed, batch.segment_ids_packed)
    mask = (batch.segment_ids_packed > 0).astype(jnp.float32)
    return _masked_token_ce_sum(logits, batch.targets_packed, mask) / total_valid

  grads_packed = jax.grad(_packed_loss)(params)
  solo_grads_weighted: Any = None
  for slot, s_tok, s_tgt, s_pos, s_seg in zip(
      batch.slots,
      batch.standalone_tokens,
      batch.standalone_targets,
      batch.standalone_positions,
      batch.standalone_segment_ids,
  ):
    doc_valid = float(bsz * slot.length)
    weight = doc_valid / total_valid

    def _solo_loss(
        p: Any,
        *,
        tok: jax.Array = s_tok,
        tgt: jax.Array = s_tgt,
        pos: jax.Array = s_pos,
        seg: jax.Array = s_seg,
        denom: float = doc_valid,
    ) -> jax.Array:
      logits = forward_fn(p, tok, pos, seg)
      mask = (seg == 1).astype(jnp.float32)
      return _masked_token_ce_sum(logits, tgt, mask) / denom

    g_doc = jax.grad(_solo_loss)(params)
    scaled = jax.tree_util.tree_map(lambda g, w=weight: g * w, g_doc)
    if solo_grads_weighted is None:
      solo_grads_weighted = scaled
    else:
      solo_grads_weighted = jax.tree_util.tree_map(lambda a, b: a + b, solo_grads_weighted, scaled)

  packed_flat = jax.tree_util.tree_flatten_with_path(grads_packed)[0]
  solo_flat = jax.tree_util.tree_flatten_with_path(solo_grads_weighted)[0]
  max_grad_abs = 0.0
  max_grad_frob = 0.0
  nonzero_leaves = 0

  for (path, g_p), (_, g_s) in zip(packed_flat, solo_flat):
    p_str = _leaf_path_str(path)
    g_p_np = np.asarray(g_p)
    g_s_np = np.asarray(g_s)
    if not (np.all(np.isfinite(g_p_np)) and np.all(np.isfinite(g_s_np))):
      return (
          False,
          {"finite": 0.0},
          [f"backward: non-finite parameter gradient on leaf {p_str!r}"],
      )
    if float(np.linalg.norm(np.asarray(g_p_np, dtype=np.float64))) > 0.0:
      nonzero_leaves += 1
    is_tied_emb = tol.tied_bf16_embed_grad_tol is not None and _EMBED_PATH_RE.search(p_str)
    leaf_grad_tol = (
        float(tol.tied_bf16_embed_grad_tol) if is_tied_emb and tol.tied_bf16_embed_grad_tol is not None else tol.grad_atol
    )
    ok, m = _compare_arrays(
        g_p_np,
        g_s_np,
        atol=leaf_grad_tol,
        rtol=leaf_grad_tol if is_tied_emb else tol.grad_rtol,
        frob_rel=leaf_grad_tol if is_tied_emb else tol.grad_frob_rel,
    )
    max_grad_abs = max(max_grad_abs, m["max_abs"])
    max_grad_frob = max(max_grad_frob, m["frob_rel"])
    if not ok:
      failures.append(f"backward leaf {p_str!r} superposition mismatch: {m}")

  if nonzero_leaves == 0:
    failures.append("backward: all parameter gradients are identically zero")

  emb_metrics, emb_failures = _check_embedding_zero_leakage(
      forward_fn=forward_fn,
      params=params,
      batch=batch,
      vocab_size=vocab_size,
      tol=tol,
      embed_fn=embed_fn,
      forward_from_embeds_fn=forward_from_embeds_fn,
  )
  failures.extend(emb_failures)
  finite_flag = 1.0 if emb_metrics.get("embed_grad_finite", 0.0) == 1.0 else 0.0
  metrics = {
      "finite": finite_flag,
      "num_param_leaves": float(len(packed_flat)),
      "nonzero_grad_leaves": float(nonzero_leaves),
      "max_param_grad_abs_diff": max_grad_abs,
      "max_param_grad_frob_rel": max_grad_frob,
      **emb_metrics,
  }
  return not failures, metrics, failures


def _check_multistep_training(
    *,
    forward_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    params: Any,
    step_batches: Sequence[PackedStepBatch],
    tx: optax.GradientTransformation,
    tol: PackingTolerance,
) -> tuple[bool, dict[str, Any], list[str]]:
  """Runs `K >= 2` Optax update steps on packed vs. accumulated standalone batches."""
  failures: list[str] = []

  @jax.jit
  def _apply_opt_update(p: Any, opt_state: optax.OptState, grads: Any) -> tuple[Any, optax.OptState]:
    updates, new_opt_state = tx.update(grads, opt_state, p)
    return optax.apply_updates(p, updates), new_opt_state

  @jax.jit
  def _packed_train_step(
      p: Any,
      opt_state: optax.OptState,
      packed_arrays: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      denom: jax.Array,
  ) -> tuple[Any, optax.OptState, jax.Array]:
    tokens, targets, positions, segment_ids = packed_arrays

    def _loss_fn(curr_p: Any) -> jax.Array:
      logits = forward_fn(curr_p, tokens, positions, segment_ids)
      mask = (segment_ids > 0).astype(jnp.float32)
      return _masked_token_ce_sum(logits, targets, mask) / denom

    loss, grads = jax.value_and_grad(_loss_fn)(p)
    new_p, new_opt_state = _apply_opt_update(p, opt_state, grads)
    return new_p, new_opt_state, loss

  @jax.jit
  def _solo_loss_and_grad(
      p: Any,
      solo_arrays: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      denom: jax.Array,
  ) -> tuple[jax.Array, Any]:
    tokens, targets, positions, segment_ids = solo_arrays

    def _loss_fn(curr_p: Any) -> jax.Array:
      logits = forward_fn(curr_p, tokens, positions, segment_ids)
      mask = (segment_ids == 1).astype(jnp.float32)
      return _masked_token_ce_sum(logits, targets, mask) / denom

    return jax.value_and_grad(_loss_fn)(p)

  params_init_flat = [
      (_leaf_path_str(p), np.asarray(v, dtype=np.float64)) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]
  ]
  params_packed = jax.tree_util.tree_map(jnp.array, params)
  params_solo = jax.tree_util.tree_map(jnp.array, params)
  opt_packed = tx.init(params_packed)
  opt_solo = tx.init(params_solo)

  packed_losses: list[float] = []
  solo_losses: list[float] = []
  max_step_loss_diff = 0.0

  for step_idx, batch in enumerate(step_batches):
    bsz = int(batch.tokens_packed.shape[0])
    total_valid = float(bsz * sum(s.length for s in batch.slots))
    params_packed, opt_packed, l_pack = _packed_train_step(
        params_packed,
        opt_packed,
        (
            batch.tokens_packed,
            batch.targets_packed,
            batch.positions_packed,
            batch.segment_ids_packed,
        ),
        jnp.asarray(total_valid, dtype=jnp.float32),
    )

    acc_loss = 0.0
    acc_grads: Any = None
    for slot, s_tok, s_tgt, s_pos, s_seg in zip(
        batch.slots,
        batch.standalone_tokens,
        batch.standalone_targets,
        batch.standalone_positions,
        batch.standalone_segment_ids,
    ):
      doc_valid = float(bsz * slot.length)
      weight = doc_valid / total_valid
      l_doc, g_doc = _solo_loss_and_grad(
          params_solo,
          (s_tok, s_tgt, s_pos, s_seg),
          jnp.asarray(doc_valid, dtype=jnp.float32),
      )
      acc_loss += weight * float(l_doc)
      scaled = jax.tree_util.tree_map(lambda g, w=weight: g * w, g_doc)
      acc_grads = scaled if acc_grads is None else jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, scaled)

    params_solo, opt_solo = _apply_opt_update(params_solo, opt_solo, acc_grads)

    lp_val = float(l_pack)
    if not (np.isfinite(lp_val) and np.isfinite(acc_loss)):
      return (
          False,
          {"finite": 0.0, "packed_losses": packed_losses, "solo_losses": solo_losses},
          [f"training: non-finite loss at step {step_idx} ({lp_val} vs {acc_loss})"],
      )
    packed_losses.append(lp_val)
    solo_losses.append(acc_loss)
    step_diff = abs(lp_val - acc_loss)
    max_step_loss_diff = max(max_step_loss_diff, step_diff)
    if step_diff > tol.loss_atol + tol.loss_rtol * max(1.0, abs(acc_loss)):
      failures.append(f"training: step {step_idx} loss diff {step_diff:.3e} exceeds tolerance")

  packed_end = jax.tree_util.tree_flatten_with_path(params_packed)[0]
  solo_end = jax.tree_util.tree_flatten_with_path(params_solo)[0]
  max_param_abs = 0.0
  max_param_frob = 0.0
  total_update_l2 = 0.0

  for (path, p_pack), (_, p_sol), (_, p0_np) in zip(packed_end, solo_end, params_init_flat):
    p_str = _leaf_path_str(path)
    p_pack_np = np.asarray(p_pack)
    p_sol_np = np.asarray(p_sol)
    if not (np.all(np.isfinite(p_pack_np)) and np.all(np.isfinite(p_sol_np))):
      return (
          False,
          {"finite": 0.0},
          [f"training: non-finite updated parameter leaf {p_str!r}"],
      )
    p_pack_64 = np.asarray(p_pack_np, dtype=np.float64)
    total_update_l2 += float(np.linalg.norm(p_pack_64 - p0_np))
    if float(np.linalg.norm(p0_np)) > 0.0 and float(np.linalg.norm(p_pack_64)) == 0.0:
      failures.append(f"training: parameter leaf {p_str!r} collapsed to zero norm")

    is_tied_emb = tol.tied_bf16_embed_param_tol is not None and _EMBED_PATH_RE.search(p_str)
    leaf_param_tol = (
        float(tol.tied_bf16_embed_param_tol)
        if is_tied_emb and tol.tied_bf16_embed_param_tol is not None
        else tol.param_atol
    )
    ok, m = _compare_arrays(
        p_pack_np,
        p_sol_np,
        atol=leaf_param_tol,
        rtol=leaf_param_tol if is_tied_emb else tol.param_rtol,
        frob_rel=leaf_param_tol if is_tied_emb else tol.param_frob_rel,
    )
    max_param_abs = max(max_param_abs, m["max_abs"])
    max_param_frob = max(max_param_frob, m["frob_rel"])
    if not ok:
      failures.append(f"training: updated leaf {p_str!r} mismatch: {m}")

  if total_update_l2 <= 0.0:
    failures.append("training: zero parameter movement across multi-step Optax updates")

  metrics: dict[str, Any] = {
      "finite": 1.0,
      "num_steps": len(step_batches),
      "packed_losses": packed_losses,
      "solo_losses": solo_losses,
      "max_step_loss_abs_diff": max_step_loss_diff,
      "max_param_abs_diff": max_param_abs,
      "max_param_frob_rel": max_param_frob,
      "total_update_l2": total_update_l2,
  }
  return not failures, metrics, failures


def verify_packed_training_parity(
    *,
    forward_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    params: Any,
    vocab_size: int = 128,
    batch_size: int = 1,
    doc_lengths: Sequence[int] = (3, 2),
    pad_gaps: Sequence[int] = (1, 2),
    num_steps: int = 3,
    learning_rate: float = 1e-2,
    optimizer: optax.GradientTransformation | None = None,
    embed_fn: Callable[[Any, jax.Array], jax.Array] | None = None,
    forward_from_embeds_fn: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    tol: PackingTolerance = PackingTolerance(),
    seed: int = 0,
    run_negative_control: bool = True,
) -> PackingParityReport:
  """Verifies forward, backward, multi-step Optax training, and mutant-kill packing parity."""
  if num_steps < 2:
    raise ValueError(f"num_steps must be >= 2 for multi-step parity, got {num_steps}")
  if (embed_fn is None) != (forward_from_embeds_fn is None):
    raise ValueError("Pass both embed_fn and forward_from_embeds_fn, or neither")

  tx = optimizer if optimizer is not None else optax.adamw(learning_rate=learning_rate, eps=1e-4, weight_decay=1e-2)
  step_batches = [
      build_packed_step_batch(
          doc_lengths=doc_lengths,
          pad_gaps=pad_gaps,
          batch_size=batch_size,
          vocab_size=vocab_size,
          seed=seed + 97 * step_i,
      )
      for step_i in range(num_steps)
  ]

  with contextlib.ExitStack() as stack:
    if mesh is not None:
      stack.enter_context(jax.set_mesh(mesh))
    stack.enter_context(jax.default_matmul_precision("highest"))
    fwd_ok, fwd_metrics, fwd_failures = _check_forward_isolation(
        forward_fn, params, step_batches[0], vocab_size=vocab_size, tol=tol
    )
    bwd_ok, bwd_metrics, bwd_failures = _check_backward_superposition(
        forward_fn=forward_fn,
        params=params,
        batch=step_batches[0],
        vocab_size=vocab_size,
        tol=tol,
        embed_fn=embed_fn,
        forward_from_embeds_fn=forward_from_embeds_fn,
    )
    trn_ok, trn_metrics, trn_failures = _check_multistep_training(
        forward_fn=forward_fn,
        params=params,
        step_batches=step_batches,
        tx=tx,
        tol=tol,
    )

    failures = [*fwd_failures, *bwd_failures, *trn_failures]
    neg_fired = False
    neg_metrics: dict[str, Any] = {"ran": run_negative_control}

    if run_negative_control:
      if not (fwd_ok and bwd_ok and trn_ok):
        neg_metrics["status"] = "ERROR_PRIMARY_FAILED"
        failures.append("negative_control: primary uncorrupted run failed, cannot certify mutant kill")
      else:

        def _corrupted_fwd(p: Any, tok: jax.Array, pos: jax.Array, seg: jax.Array) -> jax.Array:
          collapsed_seg = jnp.where(seg > 0, jnp.int32(1), jnp.int32(0))
          return forward_fn(p, tok, pos, collapsed_seg)

        corrupted_emb_fwd = None
        if forward_from_embeds_fn is not None:

          def _corrupted_emb_fwd(p: Any, emb: jax.Array, pos: jax.Array, seg: jax.Array) -> jax.Array:
            collapsed_seg = jnp.where(seg > 0, jnp.int32(1), jnp.int32(0))
            return forward_from_embeds_fn(p, emb, pos, collapsed_seg)

          corrupted_emb_fwd = _corrupted_emb_fwd

        nf_ok, nf_m, _ = _check_forward_isolation(_corrupted_fwd, params, step_batches[0], vocab_size=vocab_size, tol=tol)
        nb_ok, nb_m, _ = _check_backward_superposition(
            forward_fn=_corrupted_fwd,
            params=params,
            batch=step_batches[0],
            vocab_size=vocab_size,
            tol=tol,
            embed_fn=embed_fn,
            forward_from_embeds_fn=corrupted_emb_fwd,
        )
        nt_ok, nt_m, _ = _check_multistep_training(
            forward_fn=_corrupted_fwd,
            params=params,
            step_batches=step_batches,
            tx=tx,
            tol=tol,
        )
        all_finite = nf_m.get("finite", 0.0) == 1.0 and nb_m.get("finite", 0.0) == 1.0 and nt_m.get("finite", 0.0) == 1.0
        neg_fired = bool(all_finite and (not nf_ok) and (not nb_ok) and (not nt_ok))
        neg_metrics.update(
            {
                "status": "FIRED" if neg_fired else ("ERROR_NON_FINITE" if not all_finite else "MISSED"),
                "forward_failed": not nf_ok,
                "backward_failed": not nb_ok,
                "training_failed": not nt_ok,
                "all_finite": all_finite,
                "corrupted_max_logit_abs_diff": nf_m.get("max_logit_abs_diff", 0.0),
                "corrupted_d_doc2_d_embed_doc1": nb_m.get("max_d_doc2_d_embed_doc1", 0.0),
                "corrupted_max_param_frob_rel": nt_m.get("max_param_frob_rel", 0.0),
            }
        )
        if not neg_fired:
          failures.append(f"negative_control: collapsing packed segment_ids to 1 did not fire cleanly ({neg_metrics})")

  passed = (not failures) and (neg_fired if run_negative_control else True)
  return PackingParityReport(
      passed=passed,
      negative_control_fired=neg_fired,
      forward_metrics=fwd_metrics,
      backward_metrics=bwd_metrics,
      training_metrics=trn_metrics,
      negative_control_metrics=neg_metrics,
      failures=failures,
  )


def _rms_norm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
  rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
  return (x / rms) * scale


def _apply_rope(x: jax.Array, positions: jax.Array) -> jax.Array:
  """Applies rotary position embeddings using per-segment reset `positions`."""
  head_dim = x.shape[-1]
  half = head_dim // 2
  inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / float(half)))
  angles = positions.astype(jnp.float32)[:, :, None, None] * inv_freq[None, None, None, :]
  cos = jnp.cos(angles)
  sin = jnp.sin(angles)
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def build_synthetic_causal_transformer(
    *,
    vocab_size: int = 128,
    emb_dim: int = 32,
    num_heads: int = 4,
    mlp_dim: int = 64,
    num_layers: int = 2,
    seed: int = 42,
) -> tuple[
    Any,
    Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    Callable[[Any, jax.Array], jax.Array],
    Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
]:
  """Builds a pure-JAX multi-layer causal RoPE + SwiGLU Transformer with segment masking."""
  rng = np.random.default_rng(seed)
  head_dim = emb_dim // num_heads

  def _rand(shape: tuple[int, ...], scale: float = 0.08) -> jax.Array:
    return jnp.asarray(rng.standard_normal(shape) * scale, dtype=jnp.float32)

  layers = []
  for _ in range(num_layers):
    layers.append(
        {
            "attn_norm": jnp.ones((emb_dim,), dtype=jnp.float32),
            "wq": _rand((emb_dim, num_heads, head_dim)),
            "wk": _rand((emb_dim, num_heads, head_dim)),
            "wv": _rand((emb_dim, num_heads, head_dim)),
            "wo": _rand((num_heads, head_dim, emb_dim)),
            "mlp_norm": jnp.ones((emb_dim,), dtype=jnp.float32),
            "w_gate": _rand((emb_dim, mlp_dim)),
            "w_up": _rand((emb_dim, mlp_dim)),
            "w_down": _rand((mlp_dim, emb_dim)),
        }
    )
  params = {
      "tok_embeddings": _rand((vocab_size, emb_dim)),
      "layers": layers,
      "final_norm": jnp.ones((emb_dim,), dtype=jnp.float32),
      "lm_head": _rand((emb_dim, vocab_size)),
  }

  def embed_fn(p: Any, tokens: jax.Array) -> jax.Array:
    return p["tok_embeddings"][tokens]

  def forward_from_embeds_fn(p: Any, emb: jax.Array, positions: jax.Array, segment_ids: jax.Array) -> jax.Array:
    valid_tok = (segment_ids != 0).astype(jnp.float32)[:, :, None]
    h = emb * valid_tok
    same_seg = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
    non_pad = (segment_ids[:, None, :, None] != 0) & (segment_ids[:, None, None, :] != 0)
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    attn_mask = same_seg & non_pad & causal

    for lyr in p["layers"]:
      normed = _rms_norm(h, lyr["attn_norm"])
      q = _apply_rope(jnp.einsum("bsd,dhk->bshk", normed, lyr["wq"]), positions)
      k = _apply_rope(jnp.einsum("bsd,dhk->bshk", normed, lyr["wk"]), positions)
      v = jnp.einsum("bsd,dhk->bshk", normed, lyr["wv"])
      scores = jnp.einsum("bshk,bthk->bhst", q, k) / jnp.sqrt(float(head_dim))
      masked_scores = jnp.where(attn_mask, scores, -1e30)
      probs = jnp.where(attn_mask, jax.nn.softmax(masked_scores, axis=-1), 0.0)
      ctx = jnp.einsum("bhst,bthk->bshk", probs, v)
      h = (h + jnp.einsum("bshk,hkd->bsd", ctx, lyr["wo"])) * valid_tok

      m_in = _rms_norm(h, lyr["mlp_norm"])
      gate = jax.nn.silu(jnp.einsum("bsd,dm->bsm", m_in, lyr["w_gate"]))
      up = jnp.einsum("bsd,dm->bsm", m_in, lyr["w_up"])
      h = (h + jnp.einsum("bsm,md->bsd", gate * up, lyr["w_down"])) * valid_tok

    h_out = _rms_norm(h, p["final_norm"]) * valid_tok
    return jnp.einsum("bsd,dv->bsv", h_out, p["lm_head"])

  def forward_fn(p: Any, tokens: jax.Array, positions: jax.Array, segment_ids: jax.Array) -> jax.Array:
    return forward_from_embeds_fn(p, embed_fn(p, tokens), positions, segment_ids)

  return params, forward_fn, embed_fn, forward_from_embeds_fn


def build_maxtext_mini_adapter(
    model_name: str,
    extra_overrides: Sequence[str] = (),
    *,
    seq_len: int = 8,
    vocab_size: int = 128,
) -> tuple[Any, Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array], int, jax.sharding.Mesh, bool]:
  """Builds a `params, forward_fn` pair for a MaxText NNX or m3 mini-configuration."""
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import globals as mt_globals  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

  base_yml = os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
  model_yml = os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "models", f"{model_name}.yml")
  model_yml_text = ""
  if os.path.isfile(model_yml):
    with open(model_yml, "r", encoding="utf-8") as f:
      model_yml_text = f.read()

  defaults = {
      "model_name": model_name,
      "override_model_config": "true",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": str(vocab_size),
      "enable_dropout": "false",
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "per_device_batch_size": "1",
      "max_target_length": str(max(seq_len, 16)),
      "scan_layers": "false",
      "skip_jax_distributed_system": "true",
      "enable_checkpointing": "false",
      "base_output_directory": "/tmp/maxtext_packing_check",
      "hardware": "cpu" if jax.default_backend() == "cpu" else "tpu",
  }
  if "base_moe_mlp_dim:" in model_yml_text:
    defaults["base_moe_mlp_dim"] = defaults["base_mlp_dim"]
  if "num_experts:" in model_yml_text:
    defaults["num_experts"] = "4"
    defaults["num_experts_per_tok"] = "2"

  for item in extra_overrides:
    if "=" not in item:
      raise ValueError(f"Expected MaxText config override key=value, got {item!r}")
    k, v = item.split("=", 1)
    defaults[k] = v
  if "base_moe_mlp_dim:" in model_yml_text and "base_moe_mlp_dim" not in dict(
      item.split("=", 1) for item in extra_overrides if "=" in item
  ):
    defaults["base_moe_mlp_dim"] = defaults["base_mlp_dim"]

  mt_argv = ["packing_train_check.py", base_yml] + [f"{k}={v}" for k, v in defaults.items()]
  config = pyconfig.initialize_pydantic(mt_argv)
  mesh = maxtext_utils.get_mesh_from_config(config)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    model = model_creation_utils.get_nnx_create_model_fn(config, mesh)()
    call_params = inspect.signature(model.__call__).parameters
    graphdef, param_state, rest_state = nnx.split(model, nnx.Param, ...)

  uses_decoder_kwargs = "decoder_segment_ids" in call_params or "decoder_input_tokens" in call_params
  accepts_enable_dropout = "enable_dropout" in call_params or any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in call_params.values()
  )

  def forward_fn(
      p_state: Any,
      tokens: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array,
  ) -> jax.Array:
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      fresh_rest = jax.tree_util.tree_map(jnp.array, rest_state)
      merged = nnx.merge(graphdef, p_state, fresh_rest, copy=True)
      if uses_decoder_kwargs:
        call_kwargs: dict[str, Any] = {
            "decoder_input_tokens": tokens,
            "decoder_positions": positions,
            "decoder_segment_ids": segment_ids,
        }
        if accepts_enable_dropout:
          call_kwargs["enable_dropout"] = False
        out = merged(**call_kwargs)
      else:
        out = merged(tokens, positions, segment_ids=segment_ids)
      return out[0] if isinstance(out, tuple) else out

  has_bf16_tied_head = bool(config.logits_via_embedding and not config.use_m3_model)
  return param_state, forward_fn, int(config.global_batch_size_to_train_on), mesh, has_bf16_tied_head


def _parse_int_csv(raw: str) -> tuple[int, ...]:
  return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for sequence-packing training verification."""
  parser = argparse.ArgumentParser(
      description=(
          "Verify packed vs. standalone forward isolation, per-leaf gradient "
          "superposition, multi-step Optax training parity, and mutant-kill "
          "negative control."
      )
  )
  parser.add_argument(
      "--model-name",
      default="",
      help="Optional MaxText model_name (runs synthetic pure-JAX model when omitted).",
  )
  parser.add_argument(
      "--doc-lengths",
      default="3,2",
      help="Comma-separated document token lengths (default: 3,2).",
  )
  parser.add_argument(
      "--pad-gaps",
      default="1,2",
      help="Comma-separated 0-padding gap lengths after each document (default: 1,2).",
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--vocab-size", type=int, default=128)
  parser.add_argument("--num-steps", type=int, default=3)
  parser.add_argument("--learning-rate", type=float, default=1e-2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--json-out",
      default="",
      help="Optional path to write the full JSON verification report.",
  )
  parser.add_argument(
      "overrides",
      nargs="*",
      help="Optional MaxText key=value config overrides when --model-name is set.",
  )
  args = parser.parse_args(argv)

  doc_lengths = _parse_int_csv(args.doc_lengths)
  pad_gaps = _parse_int_csv(args.pad_gaps)
  seq_len = sum(doc_lengths) + sum(pad_gaps)

  batch_size = args.batch_size
  vocab_size = args.vocab_size
  mesh: jax.sharding.Mesh | None = None
  cli_optimizer: optax.GradientTransformation | None = None
  if args.model_name:
    min_vocab = max(vocab_size, max(batch_size, jax.device_count()) * seq_len + 16)
    vocab_size = ((min_vocab + 63) // 64) * 64
    params, forward_fn, min_batch, mesh, has_bf16_tied_head = build_maxtext_mini_adapter(
        args.model_name,
        args.overrides,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    batch_size = max(batch_size, min_batch)
    embed_fn = None
    forward_from_embeds_fn = None
    # Legacy attend_on_embedding casts tied embedding_table to bfloat16 in the
    # output head einsum (~1.4e-3 1-ULP VJP rounding on token_embedder while
    # all decoder layers and cross-doc embed isolation stay fp32); set AdamW
    # eps above 1-ULP bf16 noise so near-zero vocab rows are not amplified to
    # O(lr) step sizes.
    if has_bf16_tied_head:
      cli_optimizer = optax.adamw(learning_rate=args.learning_rate, eps=0.25, weight_decay=1e-2)
    tol = PackingTolerance(
        fwd_atol=5e-5,
        fwd_rtol=5e-5,
        fwd_frob_rel=5e-5,
        invariance_atol=1e-5,
        grad_atol=5e-5,
        grad_rtol=5e-5,
        grad_frob_rel=5e-5,
        zero_grad_atol=1e-6,
        loss_atol=5e-5,
        loss_rtol=5e-5,
        param_atol=1e-4,
        param_rtol=1e-4,
        param_frob_rel=1e-4,
        tied_bf16_embed_grad_tol=5e-3 if has_bf16_tied_head else None,
        tied_bf16_embed_param_tol=5e-4 if has_bf16_tied_head else None,
    )
  else:
    params, forward_fn, embed_fn, forward_from_embeds_fn = build_synthetic_causal_transformer(
        vocab_size=args.vocab_size, seed=args.seed + 17
    )
    tol = PackingTolerance()

  report = verify_packed_training_parity(
      forward_fn=forward_fn,
      params=params,
      vocab_size=vocab_size,
      batch_size=batch_size,
      doc_lengths=doc_lengths,
      pad_gaps=pad_gaps,
      num_steps=args.num_steps,
      learning_rate=args.learning_rate,
      optimizer=cli_optimizer,
      embed_fn=embed_fn,
      forward_from_embeds_fn=forward_from_embeds_fn,
      mesh=mesh,
      tol=tol,
      seed=args.seed,
      run_negative_control=True,
  )
  payload = report.to_dict()
  if args.json_out:
    with open(args.json_out, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")

  status = "PASS" if report.passed else "FAIL"
  print(
      f"PACKING_TRAIN_CHECK [{status}] negctl_fired={report.negative_control_fired}"
      f" fwd_max_abs={report.forward_metrics.get('max_logit_abs_diff', float('nan')):.3e}"
      f" grad_frob={report.backward_metrics.get('max_param_grad_frob_rel', float('nan')):.3e}"
      f" param_frob={report.training_metrics.get('max_param_frob_rel', float('nan')):.3e}"
  )
  if report.failures:
    for item in report.failures:
      print(f"  FAILURE: {item}", file=sys.stderr)
  return 0 if report.passed else 1


if __name__ == "__main__":
  sys.exit(main())
