# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NoPE Multi-Head Latent Attention (MLA) and DSA Indexer for GLM-5.3-Flash."""

import typing
from typing import Literal
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config


class Glm5RMSNorm(nnx.Module):
  """RMSNorm matching Glm5NextTextRMSNorm (T5LayerNorm)."""

  def __init__(
      self,
      d: int,
      *,
      eps: float,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = (x32 * jax.lax.rsqrt(v + self.eps)).astype(in_dt)
    return self.weight[...].astype(in_dt) * normed


class Glm5LayerNorm(nnx.Module):
  """LayerNorm with weight and bias matching nn.LayerNorm."""

  def __init__(
      self,
      d: int,
      *,
      eps: float,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), sharding=spec)
    self.bias = nnx.Param(jnp.zeros((d,), dtype=param_dtype), sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    m = jnp.mean(x32, axis=-1, keepdims=True)
    v = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - m) * jax.lax.rsqrt(v + self.eps)
    w = self.weight[...].astype(jnp.float32)
    b = self.bias[...].astype(jnp.float32)
    return (normed * w + b).astype(in_dt)


def _make_lin(dt: DTypeLike, p_dt: DTypeLike, sharding, rngs: nnx.Rngs):
  """Constructs a Linear layer factory closure with sharding metadata."""

  def _lin(in_f: int, out_f: int, name: str, axes: tuple[str | None, ...]):
    km = {"sharding": sharding(name, axes)} if sharding else {}
    return nnx.Linear(
        in_f,
        out_f,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        # Precision.HIGHEST is required on TPU so MXU matmuls match IEEE fp32
        # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=km,
        rngs=rngs,
    )

  return _lin


class Glm5Indexer(nnx.Module):
  """DeepSeek Sparse Attention (DSA) indexer with k-pool compression."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    h, d = config.hidden_size, config.index_head_dim
    n_heads, q_rank = config.index_n_heads, config.q_lora_rank
    self.head_dim = d
    self.index_topk = config.index_topk
    self.index_kpool = config.index_kpool
    self.index_kpool_always_select_tail = config.index_kpool_always_select_tail

    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)
    lin = _make_lin(dt, p_dt, sharding, rngs)

    self.wq_b = lin(q_rank, n_heads * d, "wq_b", ("q_lora", "heads"))
    self.wk = lin(h, d, "wk", ("embed", None))
    self.k_norm = Glm5LayerNorm(
        d,
        eps=1e-6,
        param_dtype=p_dt,
        sharding=sharding,
    )
    self.weights_proj = lin(h, n_heads, "weights_proj", ("embed", None))

    s_ape = (None, None)
    ape_spec = sharding("index_kpool_compress_ape", s_ape) if sharding else None
    ape_init = jnp.zeros((self.index_kpool, d), dtype=p_dt)
    self.index_kpool_compress_ape = nnx.Param(ape_init, sharding=ape_spec)

    s_gate = (None, "embed")
    g_name = "index_kpool_compress_gate"
    gate_spec = sharding(g_name, s_gate) if sharding else None
    gate_init = jnp.ones((d, h), dtype=p_dt)
    self.index_kpool_compress_gate = nnx.Param(gate_init, sharding=gate_spec)

  def __call__(
      self,
      hidden_states: jax.Array,
      q_resid: jax.Array,
      attention_mask: jax.Array | None = None,
  ) -> jax.Array:
    hidden_states = jax.lax.stop_gradient(hidden_states)
    q_resid = jax.lax.stop_gradient(q_resid)
    b, s = hidden_states.shape[:2]
    if attention_mask is None:
      attention_mask = jnp.ones((b, s), dtype=jnp.bool_)
    else:
      attention_mask = attention_mask.astype(jnp.bool_)

    q = self.wq_b(q_resid).reshape(b, s, -1, self.head_dim)
    k = self.k_norm(self.wk(hidden_states))
    gate_w = self.index_kpool_compress_gate[...].astype(hidden_states.dtype)
    # Precision.HIGHEST is required on TPU so MXU einsums match IEEE fp32
    # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
    gate = jnp.einsum(
        "bsd,hd->bsh",
        hidden_states,
        gate_w,
        precision=jax.lax.Precision.HIGHEST,
    )

    num_pools = (s + self.index_kpool - 1) // self.index_kpool
    any_valid = jnp.any(attention_mask, axis=-1)
    first_key = jnp.where(any_valid, jnp.argmax(attention_mask, axis=-1), s)
    tot_p = num_pools * self.index_kpool
    offsets = jnp.arange(tot_p).reshape(1, num_pools, self.index_kpool)
    raw_pool_indices = first_key[:, None, None] + offsets

    b_idx = jnp.arange(b)[:, None, None]
    safe_idx = jnp.clip(raw_pool_indices, 0, s - 1)
    grp_keys = k[b_idx, safe_idx]
    grp_gate = gate[b_idx, safe_idx]
    grp_valid = attention_mask[b_idx, safe_idx] & (raw_pool_indices < s)
    pool_valid = jnp.all(grp_valid, axis=-1)
    pool_indices = jnp.where(grp_valid, raw_pool_indices, -1)

    ape = self.index_kpool_compress_ape[...].astype(jnp.float32)
    gate_ape = grp_gate.astype(jnp.float32) + ape[None, None]
    logits = jnp.where(grp_valid[..., None], gate_ape, -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits, axis=2), nan=0.0)
    pool_keys = jnp.sum(probs.astype(grp_keys.dtype) * grp_keys, axis=2)

    q_f32, p_f32 = q.astype(jnp.float32), pool_keys.astype(jnp.float32)
    # Precision.HIGHEST is required on TPU so MXU einsums match IEEE fp32
    # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
    raw_scores = jnp.einsum("bshd,bpd->bshp", q_f32, p_f32, precision=jax.lax.Precision.HIGHEST)
    scores = jax.nn.relu(raw_scores * (self.head_dim**-0.5))
    scale_w = q.shape[2] ** -0.5
    w = self.weights_proj(hidden_states).astype(jnp.float32) * scale_w
    index_scores = jnp.einsum("bsh,bshp->bsp", w, scores, precision=jax.lax.Precision.HIGHEST)

    s_idx = jnp.arange(s)[None, :, None]
    p_end = jnp.clip(raw_pool_indices[..., -1], 0, s - 1)[:, None, :]
    valid_cand = (p_end <= s_idx) & pool_valid[:, None, :]
    min_score = jnp.finfo(jnp.float32).min
    index_scores = jnp.where(valid_cand, index_scores, min_score)

    sel_k = min(self.index_topk // self.index_kpool, num_pools)
    _, selected = jax.lax.top_k(index_scores, sel_k)
    sel_valid = valid_cand[b_idx, s_idx, selected]
    sel_indices = pool_indices[b_idx, selected]
    flat_len = sel_k * self.index_kpool
    v_rep = jnp.repeat(sel_valid[..., None], self.index_kpool, axis=-1)
    v_mask = v_rep.reshape(b, s, flat_len)
    topk_indices = jnp.where(v_mask, sel_indices.reshape(b, s, flat_len), -1)

    if self.index_kpool_always_select_tail:
      max_tail = self.index_kpool - 1
      if max_tail > 0:
        vis_cnt = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1)
        tail_cnt = vis_cnt % self.index_kpool
        t_offsets = jnp.arange(max_tail)
        t_start = first_key[:, None] + vis_cnt - tail_cnt
        t_indices = t_start[..., None] + t_offsets[None, None, :]
        t_cnt = tail_cnt[..., None]
        t_valid = (t_offsets[None, None, :] < t_cnt) & (t_indices < s)
        t_safe = jnp.clip(t_indices, 0, s - 1)
        t_vis = (t_safe <= s_idx) & attention_mask[b_idx, t_safe]
        t_masked = jnp.where(t_valid & t_vis, t_indices, -1)
        topk_indices = jnp.concatenate([topk_indices, t_masked], axis=-1)
      out_w = self.index_topk + max_tail
    else:
      out_w = self.index_topk

    diff_w = out_w - topk_indices.shape[-1]
    if diff_w > 0:
      pad = ((0, 0), (0, 0), (0, diff_w))
      topk_indices = jnp.pad(topk_indices, pad, constant_values=-1)
    res = jnp.where(attention_mask[..., None], topk_indices[..., :out_w], -1)
    return res.astype(jnp.int32)


class Glm5DsaMlaAttention(nnx.Module):
  """NoPE Multi-head Latent Attention with DSA Lightning Indexer.

  This dense [B, H, S, S] masked attention is the reference implementation
  for parity verification against HuggingFace Glm5NextTextAttention.
  TODO: Add Splash block-sparse Pallas kernel execution for sparse DSA
  attention.
  """

  def __init__(
      self,
      config: Glm5Config,
      layer_idx: int,
      *,
      indexer_type: str,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if indexer_type != "full":
      raise ValueError(f"Shared indexer unsupported: {indexer_type!r}.")
    self.config, self.layer_idx = config, layer_idx
    self.indexer_type = indexer_type
    self.sharding = sharding
    if config.qk_rope_head_dim != 0 or config.rope_parameters is not None:
      raise ValueError(f"NoPE violation: {config.qk_rope_head_dim=}.")
    ilsf = config.indexer_loss_scaling_factor
    if ilsf != 0.0:
      # TODO: Support non-zero indexer_loss_scaling_factor for auxiliary indexer loss.
      raise ValueError(f"indexer_loss_scaling_factor != 0 is not yet supported: {ilsf}.")

    h, n_heads = config.hidden_size, config.num_attention_heads
    q_rank, kv_rank = config.q_lora_rank, config.kv_lora_rank
    qk_dim, v_dim = config.qk_nope_head_dim, config.v_head_dim
    self.scaling = qk_dim**-0.5
    self.num_heads = n_heads
    self.qk_nope_head_dim = qk_dim
    self.v_head_dim = v_dim

    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)
    lin = _make_lin(dt, p_dt, sharding, rngs)
    eps = config.rms_norm_eps

    def _norm(dim):
      return Glm5RMSNorm(dim, eps=eps, param_dtype=p_dt, sharding=sharding)

    self.q_a_proj = lin(h, q_rank, "q_a_proj", ("embed", None))
    self.q_a_layernorm = _norm(q_rank)
    self.q_b_proj = lin(q_rank, n_heads * qk_dim, "q_b_proj", ("q_lora", "heads"))

    self.kv_a_proj_with_mqa = lin(h, kv_rank, "kv_a_proj_with_mqa", ("embed", None))
    self.kv_a_layernorm = _norm(kv_rank)
    kv_tot = n_heads * (qk_dim + v_dim)
    self.kv_b_proj = lin(kv_rank, kv_tot, "kv_b_proj", ("kv_lora", "heads"))
    self.o_proj = lin(n_heads * v_dim, h, "o_proj", ("heads", "embed"))
    self.indexer = Glm5Indexer(config, sharding=sharding, rngs=rngs) if config.use_indexer else None

  def _build_causal_padding_mask(
      self,
      attn_mask: jax.Array | None,
      b: int,
      s: int,
      dtype: DTypeLike,
  ) -> jax.Array:
    """Builds dense additive attention mask [B, 1, S, S] combining causal and validity masks."""
    pos = jnp.arange(s)
    causal = pos[None, None, None, :] <= pos[None, None, :, None]
    if attn_mask is not None:
      vis = causal & attn_mask[:, None, None, :] & attn_mask[:, None, :, None]
    else:
      vis = jnp.broadcast_to(causal, (b, 1, s, s))
    min_val = jnp.finfo(dtype).min
    return jnp.where(vis, 0.0, min_val).astype(dtype)

  def build_attention_mask_from_topk(
      self,
      topk_indices: jax.Array,
      query_states: jax.Array,
      kv_len: int,
  ) -> jax.Array:
    """Converts top-k indices to additive attention bias mask [B, 1, S, kv_len]."""
    b, s = topk_indices.shape[:2]
    topk_valid = (topk_indices >= 0) & (topk_indices < kv_len)
    target_idx = jnp.where(topk_valid, topk_indices, kv_len)
    b_idx = jnp.arange(b)[:, None, None]
    s_idx = jnp.arange(s)[None, :, None]
    buf = jnp.zeros((b, s, kv_len + 1), dtype=jnp.bool_)
    mask = buf.at[b_idx, s_idx, target_idx].set(True)[:, None, :, :kv_len]
    min_val = jnp.finfo(query_states.dtype).min
    return jnp.where(mask, 0.0, min_val).astype(query_states.dtype)

  def _call_dense_mask(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      mask_or_topk: jax.Array,
      decoder_segment_ids: jax.Array | None,
      s: int | None = None,
  ) -> jax.Array:
    """Computes attention using dense bias mask [B, 1, S, S] with unified zero-padding."""
    if mask_or_topk.ndim == 4:
      mask = mask_or_topk
    else:
      if s is None:
        s = q.shape[2]
      mask = self.build_attention_mask_from_topk(mask_or_topk, q, s)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=jax.lax.Precision.HIGHEST) * self.scaling + mask
    scores_f32 = scores.astype(jnp.float32)
    weights = jax.nn.softmax(scores_f32, axis=-1).astype(q.dtype)
    out = jnp.matmul(weights, v, precision=jax.lax.Precision.HIGHEST)
    if decoder_segment_ids is not None:
      valid_q = (decoder_segment_ids != 0)[:, None, :, None]
      out = jnp.where(valid_q, out, 0.0)
    return out

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: Literal[False] = False,
  ) -> jax.Array:
    ...

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: Literal[True],
  ) -> tuple[jax.Array, jax.Array | None]:
    ...

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    ...

  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    """Applies NoPE MLA attention with DSA top-k indexer masking.

    Supports unpacked sequences only (`decoder_segment_ids` entries in `{0, 1}`
    where `0` is padding and `1` is the single active document; packing is
    rejected by `Glm5Config(packing=False)` (`ValueError` raised in
    `Glm5Config.__post_init__` when `packing=True`); TODO: add sequence packing
    support).

    Args:
      hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
      decoder_segment_ids: Optional segment IDs in {0, 1} for right/left
        padding.
      _force_indexer: Internal flag to force running the indexer path for parity
        testing.
      return_topk_indices: If True, returns (output, topk_idx) tuple.

    Returns:
      Output tensor of shape [batch, seq_len, hidden_size] or tuple.
    """
    b, s = hidden_states.shape[:2]
    attn_mask = None
    if decoder_segment_ids is not None:
      attn_mask = decoder_segment_ids != 0

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_heads, qk_dim = self.num_heads, self.qk_nope_head_dim
    q = self.q_b_proj(q_resid).reshape(b, s, q_heads, qk_dim)
    q = q.transpose(0, 2, 1, 3)

    kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
    kv_dim = qk_dim + self.v_head_dim
    kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
    kv = kv.reshape(b, s, q_heads, kv_dim).transpose(0, 2, 1, 3)
    k, v = jnp.split(kv, [qk_dim], axis=-1)

    if not _force_indexer and (
        not self.config.use_indexer or (s <= self.config.index_topk and self.config.index_kpool_always_select_tail)
    ):
      mask = self._build_causal_padding_mask(attn_mask, b, s, q.dtype)
      topk_idx = (
          self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
          if (return_topk_indices and self.indexer is not None)
          else None
      )
    else:
      if self.indexer is None:
        raise ValueError("indexer is None but indexer path was requested.")
      topk_idx = self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
      mask = self.build_attention_mask_from_topk(topk_idx, q, s)

    if self.config.dsa_attention_implementation == "dense_mask":
      out = self._call_dense_mask(q, k, v, mask, decoder_segment_ids)
    else:
      raise ValueError(
          "Unsupported Glm5Config.dsa_attention_implementation:" f" {self.config.dsa_attention_implementation!r}."
      )

    out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
    out = self.o_proj(out)
    if self.sharding:
      axes = ("batch", "length", "embed")
      out = self.sharding.constrain(out, "mla_out", axes)
    if return_topk_indices:
      return out, topk_idx
    return out
