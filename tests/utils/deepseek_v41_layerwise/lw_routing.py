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

"""Teacher-forced (replayed) MoE and indexer top-k for the layerwise gates, with in-forward natural top-k.

MaxText's native `forced_routed_experts` path cannot be used for DeepSeek-V4.1: `RoutedMoE.get_topk` gathers the
forced weights from `gate_logits` (moe.py:1381-1383), which already carry the routing bias (moe.py:783), while the
natural path (`deepseek_routing`, moe.py:1509-1522) and the reference (deepseek_v41_reference/model.py:808-809)
weight the experts with the raw pre-bias scores; `DeepSeek41DecoderLayer` also never passes the argument
(deepseek41.py:296). `replayed_deepseek_routing` is a copy of `RoutedMoE.deepseek_routing` that differs only in its
`top_k` line: it still computes the natural top-k from the same `gate_logits`, sows it, and then selects the
replayed ids. `ORIGINAL_DEEPSEEK_ROUTING_SHA256` pins the copied source.

`replayed_indexer_call` is likewise a copy of `DeepSeek41Indexer.__call__` (pinned by `ORIGINAL_INDEXER_CALL_SHA256`):
it sows the raw scores (the L4 indexer cotangent multiplies them) and the natural top-k mask, keeps the candidate-
block update natural, and returns the replayed mask.
"""

import contextlib
import dataclasses
import hashlib
import inspect
from collections.abc import Iterator
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers import attention_deepseek41, deepseek41_quant, moe
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config

# sha256 of `inspect.getsource(moe.RoutedMoE.deepseek_routing)` at eda64008d78a7d043006f74edc48cb575b570db0.
ORIGINAL_DEEPSEEK_ROUTING_SHA256 = "dbe894a853fb6bd33d983ba4e08ff3eee2b41516d2f72829e7ed019c3dbc96e6"

NATURAL_MOE_TOPK = "lw_natural_moe_topk"
REPLAYED_MOE_TOPK = "lw_replayed_moe_topk"
NATURAL_INDEXER_MASK = "lw_natural_indexer_mask"
INDEXER_RAW_SCORES = "lw_indexer_raw_scores"
# sha256 of `inspect.getsource(attention_deepseek41.DeepSeek41Indexer.__call__)` at eda64008.
ORIGINAL_INDEXER_CALL_SHA256 = "0323541f4e8520bd3ec4ab4fae7134aed035a49f1baa33f0ec5f0567fa94d5b7"
REPLAYED_INDEXER_MASK = "lw_replayed_indexer_mask"


class ReplayedTopK(nnx.Variable):  # pylint: disable=abstract-method
  """Replayed top-k ids / masks attached to a module; never a parameter, so never differentiated."""


def deepseek_routing_source_sha256() -> str:
  """sha256 of the current `RoutedMoE.deepseek_routing` source."""
  return hashlib.sha256(inspect.getsource(moe.RoutedMoE.deepseek_routing).encode("utf-8")).hexdigest()


def replayed_deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
  """`RoutedMoE.deepseek_routing` with the replayed ids in place of the natural top-k (see the module docstring)."""
  expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
  _, natural_top_k_indices = jax.lax.top_k(
      jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
      k=self.num_experts_per_tok,
  )
  self.sow(nnx.Intermediate, NATURAL_MOE_TOPK, natural_top_k_indices)
  top_k_indices = getattr(self, REPLAYED_MOE_TOPK)[...]
  if top_k_indices.shape != natural_top_k_indices.shape:
    raise ValueError(f"Replayed MoE top-k {top_k_indices.shape} != natural top-k {natural_top_k_indices.shape}")
  # Tag before the weight gather, so the gather's backward reads the saved ids instead of re-running top-k.
  tag_routing_fn = moe.routing_tag_fn(self.config)
  if tag_routing_fn is not None:
    top_k_indices = tag_routing_fn(top_k_indices)
  if getattr(self.config, "router_topk_matmul_vjp", False):
    top_k_weights = moe.take_along_last_axis_dense_vjp(pre_bias_logits, top_k_indices, pre_bias_logits.shape[-1])
  else:
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def indexer_call_source_sha256() -> str:
  """sha256 of the current `DeepSeek41Indexer.__call__` source."""
  return hashlib.sha256(inspect.getsource(attention_deepseek41.DeepSeek41Indexer.__call__).encode("utf-8")).hexdigest()


def replayed_indexer_call(  # pylint: disable=too-many-positional-arguments
    self,
    x: jax.Array,
    qr: jax.Array,
    positions: jax.Array,
    shared: dict,
    decoder_segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, dict]:
  """`DeepSeek41Indexer.__call__` (attention_deepseek41.py:304-364) that sows its raw scores and natural top-k mask
  and returns the replayed mask; every other line, including the candidate-block update, is unchanged."""
  updates = {}
  # Stop gradients flowing backward into upstream backbone leaves
  x_in = jax.lax.stop_gradient(x)
  qr_in = jax.lax.stop_gradient(qr)
  index_k = shared["index_k"]  # [B, C, Di]
  num_compressed = index_k.shape[1]
  q = self.wq_b(deepseek41_quant.fp8_qdq(qr_in) if self.act_quant else qr_in)  # [B, S, Hi, Di]
  q = self.rotary_emb(q, positions, unsqueeze_dim=-2)
  if self.act_quant:
    q = deepseek41_quant.fp4_qdq(q)
  weights = self.weights_proj(x_in).astype(jnp.float32) * (self.head_dim**-0.5 * self.n_heads**-0.5)
  scores = jnp.einsum(
      "bshd,btd->bsht",
      q.astype(jnp.float32),
      index_k.astype(jnp.float32),
      precision=jax.lax.Precision(self.config.matmul_precision),
  )
  scores = jnp.sum(jax.nn.relu(scores) * weights[..., None], axis=2)  # [B, S, C]
  self.sow(nnx.Intermediate, INDEXER_RAW_SCORES, scores)

  ratio = self.compress_ratio
  if decoder_segment_ids is None:
    compress_lens = attention_deepseek41.compressed_visible_len(positions, ratio)  # [B, S]
    reachable = jnp.arange(num_compressed)[None, None, :] < compress_lens[..., None]
  else:
    block_start_seg = decoder_segment_ids[:, : num_compressed * ratio : ratio]
    block_end_seg = decoder_segment_ids[:, ratio - 1 : num_compressed * ratio : ratio]
    block_end_pos = positions[:, ratio - 1 : num_compressed * ratio : ratio]
    reachable = (
        (block_start_seg[:, None, :] == decoder_segment_ids[:, :, None])
        & (block_end_seg[:, None, :] == decoder_segment_ids[:, :, None])
        & (decoder_segment_ids[:, :, None] != 0)
        & (block_end_pos[:, None, :] <= positions[:, :, None])
    )
    compress_lens = jnp.max(
        jnp.where(reachable, jnp.arange(num_compressed)[None, None, :] + 1, 0),
        axis=-1,
    )
  scores = jnp.where(reachable, scores, -jnp.inf)

  if self.is_candidate_source:
    updates["candidates"] = attention_deepseek41.select_candidate_blocks(
        scores, compress_lens, self.config.candidate_topk_blocks, self.config.candidate_block_size
    )
  elif self.uses_candidates:
    scores = jnp.where(shared["candidates"], scores, -jnp.inf)

  k = min(self.index_topk, num_compressed)
  top_idx = jax.lax.top_k(scores, k)[1]
  # Like the reference, a pick is kept iff it is causally reachable.
  keep = (
      top_idx < compress_lens[..., None]
      if decoder_segment_ids is None
      else jnp.take_along_axis(reachable, top_idx, axis=-1)
  )
  natural_mask = attention_deepseek41._scatter_bool(top_idx, keep, num_compressed)  # pylint: disable=protected-access
  self.sow(nnx.Intermediate, NATURAL_INDEXER_MASK, natural_mask)
  replayed = getattr(self, REPLAYED_INDEXER_MASK)[...]
  if decoder_segment_ids is not None:
    replayed = replayed & reachable
  if replayed.shape != natural_mask.shape:
    raise ValueError(f"Replayed indexer mask {replayed.shape} != natural mask {natural_mask.shape}")
  return replayed, updates


@dataclasses.dataclass(frozen=True)
class LayerReplay:
  """Replayed routing of one decoder layer.

  moe_topk: [B, S, K] int32 expert ids the MoE must use.
  indexer_mask: [B, S, C] bool compressed positions the indexer must select; None exactly when the layer has no
    indexer (its spec says `is_index_source=False`), which `attach_replay` checks.
  """

  moe_topk: Any
  indexer_mask: Any


def attach_replay(layer: Any, replay: LayerReplay) -> None:
  """Attaches `replay` to `layer` (a `DeepSeek41DecoderLayer`) as `ReplayedTopK` variables."""
  has_indexer = bool(layer.spec.is_index_source)
  if has_indexer != (replay.indexer_mask is not None):
    raise ValueError(
        f"Layer {layer.spec.layer_id}: is_index_source={has_indexer} but replayed indexer mask is "
        f"{'missing' if replay.indexer_mask is None else 'present'}"
    )
  setattr(layer.mlp.MoeBlock_0, REPLAYED_MOE_TOPK, ReplayedTopK(jnp.asarray(replay.moe_topk, jnp.int32)))
  if has_indexer:
    indexer = layer.self_attention.indexer
    setattr(indexer, REPLAYED_INDEXER_MASK, ReplayedTopK(jnp.asarray(replay.indexer_mask, jnp.bool_)))


_ORIGINAL_INDEXER_INDEX_KEYS = attention_deepseek41.DeepSeek41Indexer.index_keys


def replayed_index_keys(
    self: attention_deepseek41.DeepSeek41Indexer, latent: jax.Array, block_positions: jax.Array
) -> jax.Array:
  """`DeepSeek41Indexer.index_keys` with stopped gradient on latent so indexer gradients do not leak upstream."""
  return _ORIGINAL_INDEXER_INDEX_KEYS(self, jax.lax.stop_gradient(latent), block_positions)


@contextlib.contextmanager
def replayed_routing() -> Iterator[None]:
  """Routes every `RoutedMoE` / `DeepSeek41Indexer` call through the replayed versions for the duration.

  Raises:
    AssertionError: If `RoutedMoE.deepseek_routing` no longer matches the source `replayed_deepseek_routing` copies.
  """
  for name, actual, pinned in (
      ("RoutedMoE.deepseek_routing", deepseek_routing_source_sha256(), ORIGINAL_DEEPSEEK_ROUTING_SHA256),
      ("DeepSeek41Indexer.__call__", indexer_call_source_sha256(), ORIGINAL_INDEXER_CALL_SHA256),
  ):
    if actual != pinned:
      raise AssertionError(f"{name} changed (sha256 {actual} != pinned {pinned}); re-derive its replayed copy first.")
  with (
      mock.patch.object(moe.RoutedMoE, "deepseek_routing", replayed_deepseek_routing),
      mock.patch.object(attention_deepseek41.DeepSeek41Indexer, "__call__", replayed_indexer_call),
      mock.patch.object(attention_deepseek41.DeepSeek41Indexer, "index_keys", replayed_index_keys),
  ):
    yield


def topk_ids_to_mask(topk_ids: np.ndarray, width: int) -> np.ndarray:
  """[B, S, K] position ids (-1 = no pick) -> [B, S, width] bool mask; raises on an id outside [-1, width)."""
  ids = np.asarray(topk_ids)
  if ids.size and (ids.min() < -1 or ids.max() >= width):
    raise ValueError(f"top-k ids outside [-1, {width}): min={ids.min()}, max={ids.max()}")
  mask = np.zeros(ids.shape[:-1] + (width,), dtype=bool)
  b_idx, s_idx, k_idx = np.nonzero(ids >= 0)
  mask[b_idx, s_idx, ids[b_idx, s_idx, k_idx]] = True
  return mask


@dataclasses.dataclass(frozen=True)
class Agreement:
  """Top-k agreement between the candidate's natural choice and the replayed choice at eligible positions."""

  eligible: int
  failures: int
  failure_positions: tuple[tuple[int, int], ...]

  def to_json(self) -> dict[str, Any]:
    return {"eligible": self.eligible, "failures": self.failures, "failure_positions": list(self.failure_positions)}


def production_boundary_ulp(boundary_ulp: np.ndarray, kind: str = "moe") -> np.ndarray:
  """Scales float32 boundary_ulp (2^-23) to bfloat16 score spacing for TIER_PRODUCTION."""
  ulp = np.asarray(boundary_ulp, np.float64) * 65536.0
  lower = float(2.0**-9)
  if kind == "indexer":
    return np.maximum(ulp, lower)
  return np.clip(ulp, lower, float(2.0**-8))


def eligible_by_margin(
    margin: np.ndarray,
    boundary_ulp: np.ndarray,
    ulp_multiplier: float,
    *,
    tier: str = lw_config.TIER_CODE_PARITY,
    kind: str = "moe",
) -> np.ndarray:
  """Positions whose reference top-k margin is at least `ulp_multiplier` ULPs of the boundary score.

  Args:
    margin: [B, S] reference score gap between the k-th and (k+1)-th choice; NaN where there is no competitor.
    boundary_ulp: [B, S] the ULP itself, as the capture records it (`abs(np.spacing(s_k))`, keys
      `layer_<i>_ref_gate_boundary_ulp` / `layer_<i>_ref_indexer_boundary_ulp`).
    ulp_multiplier: Tie margin in ULPs (floors file `router_agreement.ulp_multiplier`, the 8-ULP rule).
    tier: Precision tier (`code_parity` uses unscaled float32 ULP; `production` scales to bfloat16 ULP).
    kind: Routing kind ('moe' or 'indexer'); indexer omits 2^-8 upper clip in Tier B.
  """
  margin = np.asarray(margin, np.float64)
  if tier == lw_config.TIER_PRODUCTION:
    ulp = production_boundary_ulp(boundary_ulp, kind=kind)
  else:
    ulp = np.asarray(boundary_ulp, np.float64)
  if margin.shape != ulp.shape:
    raise ValueError(f"margin {margin.shape} and boundary_ulp {ulp.shape} differ in shape")
  return np.isfinite(margin) & (margin >= ulp_multiplier * ulp)


def set_agreement(natural: np.ndarray, replayed: np.ndarray, eligible: np.ndarray) -> Agreement:
  """Compares two per-position selections ([B, S, K] ids or [B, S, C] masks) at `eligible` [B, S] positions."""
  natural = np.asarray(natural)
  replayed = np.asarray(replayed)
  if natural.shape != replayed.shape or natural.shape[:2] != eligible.shape:
    raise ValueError(f"Shape mismatch: natural {natural.shape}, replayed {replayed.shape}, eligible {eligible.shape}")
  if natural.dtype == np.bool_:
    same = np.all(natural == replayed, axis=-1)
  else:
    same = np.all(np.sort(natural, axis=-1) == np.sort(replayed, axis=-1), axis=-1)
  fail = eligible & ~same
  return Agreement(
      eligible=int(np.sum(eligible)),
      failures=int(np.sum(fail)),
      failure_positions=tuple((int(b), int(s)) for b, s in zip(*np.nonzero(fail))),
  )


def indexer_geometry(positions: np.ndarray, compress_ratio: int, index_topk: int) -> dict[str, int]:
  """How many queries can have a (k+1)-th indexer competitor, from the geometry alone.

  A query at position p reaches (p + 1) // compress_ratio compressed positions (attention_deepseek41.py:100-103);
  only when that exceeds `index_topk` does its top-k make a choice.
  """
  reachable = (np.asarray(positions, np.int64) + 1) // compress_ratio
  return {
      "compress_ratio": compress_ratio,
      "index_topk": index_topk,
      "max_reachable": int(reachable.max()),
      "geometrically_eligible_queries": int(np.sum(reachable > index_topk)),
  }


def candidate_geometry(positions: np.ndarray, compress_ratio: int, topk_blocks: int, block_size: int) -> dict[str, int]:
  """How many queries reach more compressed positions than the candidate blocks can hold (a choice to make).

  `select_candidate_blocks` keeps `topk_blocks` blocks of `block_size` positions (attention_deepseek41.py:106-129);
  a query reaching at most `topk_blocks * block_size` positions keeps all of them.
  """
  reachable = (np.asarray(positions, np.int64) + 1) // compress_ratio
  return {
      "compress_ratio": compress_ratio,
      "candidate_topk_blocks": topk_blocks,
      "candidate_block_size": block_size,
      "max_reachable": int(reachable.max()),
      "geometrically_eligible_queries": int(np.sum(reachable > topk_blocks * block_size)),
  }


def reference_candidate_blocks(
    raw_scores: np.ndarray, positions: np.ndarray, compress_ratio: int, topk_blocks: int, block_size: int
) -> np.ndarray:
  """Reference candidate-block mask [B, S, C] (deepseek_v41_reference/model.py:569-592) from raw indexer scores."""
  scores = np.array(raw_scores, dtype=np.float64, copy=True)
  pos = np.asarray(positions, dtype=np.int64)
  width = scores.shape[-1]
  compress_lens = (pos + 1) // compress_ratio
  scores = np.where(np.arange(width)[None, None, :] < compress_lens[..., None], scores, -np.inf)
  pad = -width % block_size
  if pad:
    scores = np.pad(scores, ((0, 0), (0, 0), (0, pad)), constant_values=-np.inf)
  block_scores = scores.reshape(scores.shape[:-1] + (-1, block_size)).max(axis=-1)
  num_blocks = block_scores.shape[-1]
  last = (compress_lens - 1) // block_size
  block_scores = np.where(np.arange(num_blocks)[None, None, :] == last[..., None], np.inf, block_scores)
  k = min(topk_blocks, num_blocks)
  order = np.argsort(-block_scores, axis=-1, kind="stable")[..., :k]
  top_vals = np.take_along_axis(block_scores, order, axis=-1)
  keep = np.zeros_like(block_scores, dtype=bool)
  b_idx, s_idx, k_idx = np.nonzero(top_vals > -np.inf)
  keep[b_idx, s_idx, order[b_idx, s_idx, k_idx]] = True
  return np.repeat(keep, block_size, axis=-1)[..., :width]


def agreement_status(agreement: Agreement, *, geometrically_eligible: int, allowed_failures: int) -> str:
  """PASS / FAIL / NOT_APPLICABLE_GEOMETRY for one top-k agreement (lead rule: zero eligible where the geometry
  allows a choice is a FAIL)."""
  if geometrically_eligible == 0:
    return lw_checks.STATUS_NOT_APPLICABLE_GEOMETRY
  if agreement.eligible == 0:
    return lw_checks.STATUS_FAIL
  return lw_checks.STATUS_PASS if agreement.failures <= allowed_failures else lw_checks.STATUS_FAIL


def corrupt_replayed_expert(moe_topk: np.ndarray, position: tuple[int, int], num_experts: int) -> np.ndarray:
  """NC_L2a: `moe_topk` with slot 0 at `position` replaced by the lowest expert id that token does not use."""
  ids = np.array(moe_topk, copy=True)
  b, s = position
  unused = sorted(set(range(num_experts)) - set(ids[b, s].tolist()))
  ids[b, s, 0] = unused[0]
  return ids


def corrupt_replayed_indexer(mask: np.ndarray, position: tuple[int, int], reachable: int) -> np.ndarray:
  """Indexer NC: `mask` [B, S, C] with, at query `position`, its lowest selected compressed position swapped for the
  lowest unselected one among the query's `reachable` positions."""
  out = np.array(mask, copy=True)
  b, s = position
  selected = np.flatnonzero(out[b, s, :reachable])
  unselected = np.flatnonzero(~out[b, s, :reachable])
  if selected.size == 0 or unselected.size == 0:
    raise ValueError(f"Query {position}: no selected/unselected pair among {reachable} reachable positions")
  out[b, s, selected[0]] = False
  out[b, s, unselected[0]] = True
  return out


@contextlib.contextmanager
def gate_image_mask_flipped() -> Iterator[None]:
  """NC_L2c: every `GateLogit` call sees the inverted image mask, so `bias_vl` and `bias` swap places for routing
  only (the engram, attention and output gating still see the true mask)."""
  original = moe.GateLogit.__call__

  def flipped(self, inputs, _initializing: bool = False, image_mask=None):
    if image_mask is None:
      raise ValueError("gate_image_mask_flipped needs a call with an image_mask")
    return original(self, inputs, _initializing, image_mask=jnp.logical_not(image_mask))

  with mock.patch.object(moe.GateLogit, "__call__", flipped):
    yield


@contextlib.contextmanager
def gate_logits_in_bf16() -> Iterator[None]:
  """NC_L2f: every `GateLogit` call has its logits truncated to bfloat16, reproducing Defect 1."""
  original = moe.GateLogit.__call__

  def bf16_gate(self, inputs, _initializing: bool = False, image_mask=None):
    out = original(self, inputs, _initializing, image_mask=image_mask)
    if isinstance(out, tuple):
      logits, pre = out
      logits_bf16 = logits.astype(jnp.bfloat16).astype(jnp.float32)
      pre_bf16 = pre.astype(jnp.bfloat16).astype(jnp.float32) if pre is not None else None
      return logits_bf16, pre_bf16
    return out.astype(jnp.bfloat16).astype(jnp.float32)

  with mock.patch.object(moe.GateLogit, "__call__", bf16_gate):
    yield
