#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""DeepSeek-V4.1 attention: sliding window plus shared compressed KV with a two-level indexer.

Every DeepSeek-V4.1 layer attends over a sliding window of its own (uncompressed) KV. Layers with
`compress_ratio > 0` additionally attend over `index_topk` positions of a *compressed* KV sequence, in the same
softmax (with a learned per-head attention sink):

  * Compressed KV is only produced by the `kv_source_layers` (one latent per `compress_ratio` tokens: a softmax
    pooled projection for ratio > 1, a plain projection for ratio 1). Every later layer with the same ratio reuses
    the most recent source's compressed KV and its index keys.
  * Top-k positions are only chosen by the `index_source_layers`; the layers in between reuse the source's choice.
  * The `candidate_source_layer` additionally keeps its `candidate_topk_blocks` best blocks of
    `candidate_block_size` positions; every later index source only picks its top-k inside those blocks.

What a layer shares with later layers is threaded through a small dict (`shared`):
  `compress_kv` [B, C, D] (RoPE'd), `index_k` [B, C, Di], `topk_mask` [B, S, C] (bool), `candidates` [B, S, C].

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
"""

import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.common.common_types import Array, Config
from maxtext.layers import deepseek41_quant
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.initializers import nd_dense_init, NdInitializer
from maxtext.layers.linears import DenseGeneral, DeepSeekV4GroupedLinear
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant


class DeepSeek41RotaryEmbedding(DeepSeekV4RotaryEmbedding):
  """DeepSeek-V4 interleaved partial RoPE with optional YaRN frequency interpolation.

  With `original_seq_len > 0`, dimensions whose wavelength already fits in the training context keep their
  frequency, those far beyond it are divided by `rope_factor`, and the `beta_fast`..`beta_slow` band in between is
  blended with a linear ramp (reference `precompute_freqs_cis`). No attention-temperature (mscale) is applied.
  """

  def __init__(
      self,
      head_dim: int,
      partial_rotary_factor: float,
      rope_theta: float,
      fprop_dtype: Any = jnp.float32,
      original_seq_len: int = 0,
      rope_factor: float = 1.0,
      beta_fast: int = 32,
      beta_slow: int = 1,
      **kwargs,
  ):
    super().__init__(
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=rope_theta,
        fprop_dtype=fprop_dtype,
        **kwargs,
    )
    self.original_seq_len = original_seq_len
    self.rope_factor = rope_factor
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow

  @property
  def inv_freq(self):
    dim, base = self.dim, self.rope_theta
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    if self.original_seq_len > 0:

      def corrected_dim(rotations):
        return dim * math.log(self.original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

      low = max(math.floor(corrected_dim(self.beta_fast)), 0)
      high = min(math.ceil(corrected_dim(self.beta_slow)), dim - 1)
      ramp = jnp.clip((jnp.arange(dim // 2, dtype=jnp.float32) - low) / max(high - low, 1e-3), 0, 1)
      smooth = 1 - ramp
      freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth
    return freqs


def compressed_visible_len(positions: Array, ratio: int) -> Array:
  """Number of compressed positions a query at `positions` may see: a group of `ratio` tokens becomes visible
  once the query has passed its last token. [B, S] -> [B, S]."""
  return (positions + 1) // ratio


def select_candidate_blocks(logits: Array, compress_lens: Array, topk_blocks: int, block_size: int) -> Array:
  """Level one of the two-level top-k: keeps the `topk_blocks` highest-scoring blocks per query.

  Args:
    logits: [B, S, C] index scores with unreachable positions already at -inf.
    compress_lens: [B, S] number of reachable compressed positions per query.
    topk_blocks: number of blocks kept.
    block_size: positions per block.

  Returns:
    [B, S, C] bool mask of the positions inside the kept blocks.
  """
  width = logits.shape[-1]
  pad = -width % block_size
  scores = jnp.pad(logits, ((0, 0), (0, 0), (0, pad)), constant_values=-jnp.inf)
  scores = scores.reshape(scores.shape[:-1] + (-1, block_size)).max(axis=-1)  # [B, S, NB]
  num_blocks = scores.shape[-1]
  # The block holding the query's newest position is only partly filled: always keep it.
  last = (compress_lens - 1) // block_size
  scores = jnp.where(jnp.arange(num_blocks) == last[..., None], jnp.inf, scores)
  top_vals, top_idx = jax.lax.top_k(scores, min(topk_blocks, num_blocks))
  # Leftover picks that came back -inf (fewer reachable blocks than topk_blocks) are dropped.
  keep = _scatter_bool(top_idx, top_vals > -jnp.inf, num_blocks)
  return jnp.repeat(keep, block_size, axis=-1)[..., :width]


def _scatter_bool(indices: Array, values: Array, width: int) -> Array:
  """[B, S, K] indices and bool values -> [B, S, width] bool with `values` scattered at `indices` (OR-combined)."""
  b, s, _ = indices.shape
  bi = jnp.arange(b)[:, None, None]
  si = jnp.arange(s)[None, :, None]
  return jnp.zeros((b, s, width), dtype=jnp.bool_).at[bi, si, indices].max(values)


class DeepSeek41Compressor(nnx.Module):
  """Pools `compress_ratio` consecutive tokens into one KV latent (returned before RoPE).

  ratio == 1: a plain projection plus RMSNorm. ratio > 1: KV and gate projections, softmax over the gate within
  each group of `ratio` tokens (in float32), weighted sum, RMSNorm. A trailing partial group is dropped (it only
  becomes visible to queries past the end of the sequence).
  """

  def __init__(
      self,
      config: Config,
      compress_ratio: int,
      kernel_init: NdInitializer,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_ratio = compress_ratio
    self.head_dim = config.head_dim
    # The pooling runs in float32 (the reference promotes these weights to float32).
    proj_dtype = jnp.float32 if compress_ratio > 1 else config.dtype
    dense_kwargs = {
        "in_features_shape": config.emb_dim,
        "out_features_shape": self.head_dim,
        "axis": -1,
        "kernel_init": kernel_init,
        "kernel_axes": ("embed_attn", "kv_head_dim"),
        "dtype": proj_dtype,
        "weight_dtype": config.weight_dtype,
        "quant": quant,
        "matmul_precision": config.matmul_precision,
        "shard_mode": config.shard_mode,
        "rngs": rngs,
    }
    self.wkv = DenseGeneral(**dense_kwargs)
    if compress_ratio > 1:
      self.wgate = DenseGeneral(**dense_kwargs)
    self.norm = RMSNorm(
        num_features=self.head_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

  def __call__(self, x: Array) -> Array:
    """[B, S, emb] -> [B, S // ratio, head_dim] (pre-RoPE)."""
    ratio = self.compress_ratio
    if ratio == 1:
      return self.norm(self.wkv(x))
    b, s, _ = x.shape
    usable = (s // ratio) * ratio
    xf = x[:, :usable].astype(jnp.float32)
    kv = self.wkv(xf).reshape(b, usable // ratio, ratio, self.head_dim)
    score = self.wgate(xf).reshape(b, usable // ratio, ratio, self.head_dim)
    pooled = jnp.sum(kv * jax.nn.softmax(score, axis=2), axis=2)
    return self.norm(pooled.astype(self.config.dtype))


class DeepSeek41Indexer(nnx.Module):
  """Scores compressed positions per query and keeps the `index_topk` best (DeepSeek-V4.1 indexer).

  Index keys are derived from the KV source's compressed latent (`wk` + `k_norm`, RoPE on the trailing
  `rope_head_dim`), so only an indexer in a `kv_source_layers` layer owns those weights; every other indexer reads
  the keys its source published. Scores are `sum_h relu(q_h . k) * w_h` with `w = weights_proj(x)` scaled by
  `index_head_dim**-0.5 * index_n_heads**-0.5`.
  """

  def __init__(
      self,
      config: Config,
      compress_ratio: int,
      owns_k: bool,
      is_candidate_source: bool,
      uses_candidates: bool,
      rotary_embedding: DeepSeek41RotaryEmbedding,
      kernel_init: NdInitializer,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_ratio = compress_ratio
    self.owns_k = owns_k
    self.is_candidate_source = is_candidate_source
    self.uses_candidates = uses_candidates
    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.index_topk = config.indexer_topk
    self.rotary_emb = rotary_embedding
    self.act_quant = deepseek41_quant.enabled(config)
    common = {
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "quant": quant,
        "matmul_precision": config.matmul_precision,
        "shard_mode": config.shard_mode,
        "kernel_init": kernel_init,
        "rngs": rngs,
    }
    self.wq_b = DenseGeneral(
        in_features_shape=config.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_axes=("q_lora", "q_heads", "kv"),
        **common,
    )
    self.weights_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_axes=("embed_attn", "indexer_weights"),
        **common,
    )
    if owns_k:
      self.wk = DenseGeneral(
          in_features_shape=config.head_dim,
          out_features_shape=self.head_dim,
          axis=-1,
          kernel_axes=("kv_head_dim", "kv"),
          **common,
      )
      self.k_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=rngs,
      )

  def index_keys(self, latent: Array, block_positions: Array) -> Array:
    """Index keys from the (pre-RoPE) compressed latent: [B, C, head_dim] -> [B, C, index_head_dim]."""
    k = self.rotary_emb(self.k_norm(self.wk(latent)), block_positions, unsqueeze_dim=None)
    return deepseek41_quant.fp4_qdq(k) if self.act_quant else k

  def __call__(
      self,
      x: Array,
      qr: Array,
      positions: Array,
      shared: dict,
      decoder_segment_ids: Array | None = None,
  ) -> tuple[Array, dict]:
    """Returns the [B, S, C] bool mask of selected compressed positions, plus shared-state updates."""
    updates = {}
    index_k = shared["index_k"]  # [B, C, Di]
    num_compressed = index_k.shape[1]
    q = self.wq_b(deepseek41_quant.fp8_qdq(qr) if self.act_quant else qr)  # [B, S, Hi, Di]
    q = self.rotary_emb(q, positions, unsqueeze_dim=-2)
    if self.act_quant:
      q = deepseek41_quant.fp4_qdq(q)
    weights = self.weights_proj(x).astype(jnp.float32) * (self.head_dim**-0.5 * self.n_heads**-0.5)
    scores = jnp.einsum(
        "bshd,btd->bsht",
        q.astype(jnp.float32),
        index_k.astype(jnp.float32),
        precision=jax.lax.Precision(self.config.matmul_precision),
    )
    scores = jnp.sum(jax.nn.relu(scores) * weights[..., None], axis=2)  # [B, S, C]

    ratio = self.compress_ratio
    if decoder_segment_ids is None:
      compress_lens = compressed_visible_len(positions, ratio)  # [B, S]
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
      updates["candidates"] = select_candidate_blocks(
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
    return _scatter_bool(top_idx, keep, num_compressed), updates


class DeepSeek41Attention(nnx.Module):
  """DeepSeek-V4.1 attention layer (see module docstring).

  Q: `wq_a` -> RMSNorm -> `wq_b` (per-head, no further norm). KV: a single shared head, `wkv` -> RMSNorm, used as
  both key and value. RoPE on the trailing `qk_rope_head_dim` of q / k (and inverse RoPE on the output). Output:
  block-diagonal grouped projection `wo_a` over `o_groups` followed by `wo_b`.
  """

  def __init__(
      self,
      config: Config,
      compress_ratio: int,
      is_kv_source: bool,
      is_index_source: bool,
      is_candidate_source: bool,
      uses_candidates: bool,
      quant: Optional[Quant] = None,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_ratio = compress_ratio
    self.is_kv_source = is_kv_source
    self.is_index_source = is_index_source
    self.num_heads = config.num_query_heads
    self.head_dim = config.head_dim
    self.window_size = config.sliding_window_size
    self.softmax_scale = self.head_dim**-0.5
    self.o_groups = config.o_groups
    self.o_lora_rank = config.o_lora_rank
    self.act_quant = deepseek41_quant.enabled(config)
    common = {
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "quant": quant,
        "matmul_precision": config.matmul_precision,
        "shard_mode": config.shard_mode,
        "kernel_init": kernel_init,
        "rngs": rngs,
    }
    norm_kwargs = {
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "epsilon": config.normalization_layer_epsilon,
        "kernel_axes": ("norm",),
        "rngs": rngs,
    }

    self.wq_a = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.q_lora_rank,
        axis=-1,
        kernel_axes=("embed_attn", "q_lora"),
        **common,
    )
    self.q_norm = RMSNorm(num_features=config.q_lora_rank, **norm_kwargs)
    self.wq_b = DenseGeneral(
        in_features_shape=config.q_lora_rank,
        out_features_shape=(self.num_heads, self.head_dim),
        axis=-1,
        kernel_axes=("q_lora", "q_heads", "kv"),
        **common,
    )
    self.wkv = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_axes=("embed_attn", "kv_head_dim"),
        **common,
    )
    self.kv_norm = RMSNorm(num_features=self.head_dim, **norm_kwargs)
    self.o_a_proj = DeepSeekV4GroupedLinear(
        in_features_per_group=(self.num_heads * self.head_dim) // self.o_groups,
        out_features=self.o_groups * self.o_lora_rank,
        n_groups=self.o_groups,
        kernel_init=kernel_init,
        kernel_axes=("o_groups", "q_heads", "o_lora_up_proj"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.o_b_proj = DenseGeneral(
        in_features_shape=self.o_groups * self.o_lora_rank,
        out_features_shape=config.emb_dim,
        axis=-1,
        kernel_axes=("o_lora_up_proj", "embed_attn"),
        **common,
    )
    self.sinks = nnx.Param(jnp.zeros((self.num_heads,), dtype=jnp.float32), out_sharding=(None,))

    # Compressed layers rotate at `compressed_rope_max_timescale` with YaRN; sliding-window-only layers use the
    # base theta without YaRN.
    if compress_ratio > 0:
      rope_kwargs = {
          "rope_theta": config.compressed_rope_max_timescale,
          "original_seq_len": config.original_max_position_embeddings if config.ds41_compressed_rope_yarn else 0,
          "rope_factor": config.rope_factor,
          "beta_fast": config.beta_fast,
          "beta_slow": config.beta_slow,
      }
    else:
      rope_kwargs = {"rope_theta": config.rope_max_timescale}
    self.rotary_embedding = DeepSeek41RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=config.qk_rope_head_dim / self.head_dim,
        fprop_dtype=config.dtype,
        **rope_kwargs,
    )

    if is_kv_source:
      self.compressor = DeepSeek41Compressor(config, compress_ratio, kernel_init, quant, rngs=rngs)
    if is_index_source:
      self.indexer = DeepSeek41Indexer(
          config,
          compress_ratio,
          owns_k=is_kv_source,
          is_candidate_source=is_candidate_source,
          uses_candidates=uses_candidates,
          rotary_embedding=self.rotary_embedding,
          kernel_init=kernel_init,
          quant=quant,
          rngs=rngs,
      )

  def _compressed_kv(
      self,
      x: Array,
      qr: Array,
      positions: Array,
      shared: dict,
      decoder_segment_ids: Array | None = None,
  ) -> tuple[Array, Array, dict]:
    """Returns (compress_kv [B, C, D], selected-position mask [B, S, C], shared-state updates)."""
    updates = {}
    ratio = self.compress_ratio
    if self.is_kv_source:
      latent = self.compressor(x)  # [B, C, D], pre-RoPE
      num_compressed = latent.shape[1]
      # A latent stands for the first token of its group.
      block_positions = positions[:, : num_compressed * ratio : ratio]
      if self.is_index_source:
        updates["index_k"] = self.indexer.index_keys(latent, block_positions)
      compress_kv = self.rotary_embedding(latent, block_positions, unsqueeze_dim=None)
      if self.act_quant:
        compress_kv = deepseek41_quant.fp4_qdq(compress_kv, block_size=16, e4m3_scale=True)
      updates["compress_kv"] = compress_kv
    state = {**shared, **updates}
    if self.is_index_source:
      topk_mask, index_updates = self.indexer(x, qr, positions, state, decoder_segment_ids=decoder_segment_ids)
      updates.update(index_updates)
      updates["topk_mask"] = topk_mask
      state = {**state, **updates}
    return state["compress_kv"], state["topk_mask"], updates

  def __call__(
      self,
      x: Array,
      positions: Array,
      shared: dict,
      decoder_segment_ids: Array | None = None,
  ) -> tuple[Array, dict]:
    """x: [B, S, emb]; positions: [B, S] (contiguous from 0). Returns the [B, S, emb] output and the updates this
    layer makes to the shared cross-layer state."""
    b, s, _ = x.shape
    # The reference quantizes the input of every FP8 GEMM and round-trips the window KV through FP8.
    qd = deepseek41_quant.fp8_qdq if self.act_quant else lambda t: t
    xq = qd(x)
    qr = self.q_norm(self.wq_a(xq))
    q = self.rotary_embedding(self.wq_b(qd(qr)), positions, unsqueeze_dim=-2)  # [B, S, H, D]
    kv = qd(self.rotary_embedding(self.kv_norm(self.wkv(xq)), positions, unsqueeze_dim=None))  # [B, S, D]

    # Sliding window over the raw KV: key j is visible to query i iff 0 <= pos_i - pos_j < window_size.
    delta = positions[:, :, None] - positions[:, None, :]
    mask = (delta >= 0) & (delta < self.window_size)  # [B, S, S]
    if decoder_segment_ids is not None:
      same_seg = (decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]) & (
          decoder_segment_ids[:, :, None] != 0
      )
      mask = mask & same_seg
    keys = kv
    updates = {}
    if self.compress_ratio > 0:
      compress_kv, topk_mask, updates = self._compressed_kv(
          x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids
      )
      keys = jnp.concatenate([kv, compress_kv.astype(kv.dtype)], axis=1)
      mask = jnp.concatenate([mask, topk_mask], axis=-1)

    precision = jax.lax.Precision(self.config.matmul_precision)
    qf = q.astype(jnp.float32)
    kf = keys.astype(jnp.float32)
    logits = jnp.einsum("bshd,btd->bhst", qf, kf, precision=precision) * self.softmax_scale
    logits = jnp.where(mask[:, None], logits, -jnp.inf)
    sink = jnp.broadcast_to(self.sinks[...].astype(jnp.float32)[None, :, None, None], (b, self.num_heads, s, 1))
    probs = jax.nn.softmax(jnp.concatenate([logits, sink], axis=-1), axis=-1)[..., :-1]
    out = jnp.einsum("bhst,btd->bshd", probs, kf, precision=precision).astype(self.config.dtype)

    # Undo the query rotation so the output lives in the unrotated value space.
    out = self.rotary_embedding(out, positions, unsqueeze_dim=-2, reverse=True)
    grouped = qd(out.reshape(b, s, self.o_groups, (self.num_heads * self.head_dim) // self.o_groups))
    out = self.o_b_proj(qd(self.o_a_proj(grouped).reshape(b, s, -1)))
    return out, updates
