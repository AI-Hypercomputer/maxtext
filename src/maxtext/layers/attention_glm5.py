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

"""GLM-5.3-Flash DeepSeek Sparse Attention: NoPE MLA with a k-pool DSA indexer.

Differences from DeepSeek-V3.2 DSA (`attention_mla.MLA` + `attention_mla.Indexer`):
  * No RoPE anywhere (`qk_rope_head_dim == 0`): q/k heads are pure "nope" dims.
  * The indexer scores *pools* of `index_kpool` consecutive tokens instead of single tokens.
    Each pool key is a learned softmax-weighted average of its token keys
    (gate = x @ index_kpool_compress_gate^T, plus a per-slot bias `index_kpool_compress_ape`).
    The top `index_topk // index_kpool` visible pools are selected and expanded back into
    tokens; the current incomplete tail pool is always attended
    (`index_kpool_always_select_tail`).

Reference: `Glm5NextTextAttention` / `Glm5NextTextIndexer` in
transformers/models/glm5_next/modeling_glm5_next.py.
"""

from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Array, Config, DType, MODEL_MODE_TRAIN, get_weight_dtype
from maxtext.layers import quantizations
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm

_HIGHEST = jax.lax.Precision.HIGHEST


def _dense(config, name, in_shape, out_shape, axes, dtype, quant, rngs, in_axis=-1, weight_dtype=None):
  return DenseGeneral(
      in_features_shape=in_shape,
      out_features_shape=out_shape,
      axis=in_axis,
      kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=axes,
      dtype=dtype,
      weight_dtype=weight_dtype if weight_dtype is not None else get_weight_dtype(config, name),
      weight_quant=None if weight_dtype is not None else quantizations.get_weight_quant_config(config, name),
      quant=quant,
      matmul_precision=config.matmul_precision,
      shard_mode=config.shard_mode,
      rngs=rngs,
  )


class KPoolIndexer(nnx.Module):
  """DSA indexer with k-pool compression. Produces a boolean [B, T, S] token selection mask."""

  def __init__(
      self,
      config: Config,
      dtype: DType,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.dtype = dtype
    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.topk = config.indexer_topk
    self.kpool = config.indexer_kpool
    self.always_select_tail = config.indexer_kpool_always_select_tail
    self.softmax_scale = self.head_dim**-0.5
    emb = config.emb_dim

    self.wq_b = _dense(config, "indexer_wq_b", config.q_lora_rank, (self.n_heads, self.head_dim),
                       ("q_lora", "q_heads", "kv"), dtype, quant, rngs)
    self.wk = _dense(config, "indexer_wk", emb, self.head_dim, ("embed", "kv"), dtype, quant, rngs)
    self.k_norm = nnx.LayerNorm(
        num_features=self.head_dim, epsilon=1e-6, use_bias=True, dtype=jnp.float32,
        param_dtype=get_weight_dtype(config, "indexer_k_norm"), rngs=rngs,
    )
    self.weights_proj = _dense(config, "indexer_weights_proj", emb, self.n_heads, ("embed", "q_heads"), dtype, quant, rngs)
    # HF index_kpool_compress_gate is [head_dim, emb]; stored here as a kernel [emb, head_dim].
    self.compress_gate = _dense(config, "indexer_compress_gate", emb, self.head_dim, ("embed", "kv"), dtype, quant, rngs)
    self.compress_ape = nnx.Param(
        jnp.zeros((self.kpool, self.head_dim), get_weight_dtype(config, "indexer_compress_ape")), sharding=(None, None)
    )

  def __call__(self, x: Array, q_resid: Array, decoder_segment_ids: Optional[Array] = None) -> Array:
    """Args: x [B, S, E] (normed layer input), q_resid [B, S, q_lora_rank]. Returns bool mask [B, T, S]."""
    x = jax.lax.stop_gradient(x)
    q_resid = jax.lax.stop_gradient(q_resid)
    b, s, _ = x.shape
    p_size = self.kpool
    n_pools = -(-s // p_size)

    q = self.wq_b(q_resid).astype(jnp.float32)  # [B, T, H, D]
    k = self.k_norm(self.wk(x).astype(jnp.float32))  # [B, S, D]
    gate = self.compress_gate(x).astype(jnp.float32)  # [B, S, D]

    if decoder_segment_ids is None:
      valid = jnp.ones((b, s), dtype=bool)
    else:
      valid = decoder_segment_ids != 0

    # Pools start at token 0 (MaxText pads on the right, so the first real token is position 0).
    pad = n_pools * p_size - s
    pad_fn = lambda t, v=0: jnp.pad(t, ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2), constant_values=v)
    k_p = pad_fn(k).reshape(b, n_pools, p_size, -1)
    gate_p = pad_fn(gate).reshape(b, n_pools, p_size, -1)
    valid_p = pad_fn(valid, False).reshape(b, n_pools, p_size)
    pool_valid = valid_p.all(-1)  # [B, P]

    logits = gate_p + self.compress_ape[...].astype(jnp.float32)[None, None]
    logits = jnp.where(valid_p[..., None], logits, -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits, axis=2))
    pool_keys = jnp.sum(probs * k_p, axis=2)  # [B, P, D]

    scores = jnp.einsum("bthd,bpd->bthp", q, pool_keys, precision=_HIGHEST)
    scores = jax.nn.relu(scores * self.softmax_scale)
    weights = self.weights_proj(x).astype(jnp.float32) * (self.n_heads**-0.5)  # [B, T, H]
    index_scores = jnp.einsum("bth,bthp->btp", weights, scores, precision=_HIGHEST)

    q_pos = jnp.arange(s)
    pool_end = jnp.arange(n_pools) * p_size + p_size - 1
    # A pool is selectable only if its last token is visible (causal + valid) and it is complete.
    end_valid = pad_fn(valid, False)[:, jnp.minimum(pool_end, n_pools * p_size - 1)]  # [B, P]
    candidates = (pool_end[None, :] <= q_pos[:, None])[None] & (pool_valid & end_valid)[:, None, :]  # [B, T, P]
    index_scores = jnp.where(candidates, index_scores, jnp.finfo(jnp.float32).min)

    select_k = min(self.topk // p_size, n_pools)
    _, selected = jax.lax.top_k(index_scores, select_k)  # [B, T, K]
    selected_valid = jnp.take_along_axis(candidates, selected, axis=-1)
    pool_mask = jnp.any(
        (jax.nn.one_hot(selected, n_pools, dtype=jnp.int32) * selected_valid[..., None]) > 0, axis=-2
    )  # [B, T, P]
    token_pool = jnp.arange(s) // p_size
    token_mask = pool_mask[:, :, token_pool]  # [B, T, S]

    if self.always_select_tail and p_size > 1:
      # Visible keys for query t are valid tokens <= t; tail = the trailing (count % kpool) of them.
      visible = (jnp.arange(s)[None, :] <= q_pos[:, None])[None] & valid[:, None, :]  # [B, T, S]
      visible_count = visible.sum(-1)  # [B, T]
      tail_start = visible_count - visible_count % p_size  # first key index of the tail
      kv_idx = jnp.arange(s)[None, None, :]
      tail = (kv_idx >= tail_start[..., None]) & visible
      token_mask = token_mask | tail

    return token_mask & valid[:, :, None]


class Glm5SparseMLA(nnx.Module):
  """NoPE Multi-head Latent Attention with k-pool DSA token selection (GLM-5.3-Flash)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      dtype: DType,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.dtype = dtype
    self.num_heads = config.num_query_heads
    self.q_lora_rank = config.q_lora_rank
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_head_dim = config.qk_nope_head_dim
    self.v_head_dim = config.v_head_dim
    self.scaling = self.qk_head_dim**-0.5
    assert config.qk_rope_head_dim == 0, "GLM-5.3 DSA layers are NoPE (qk_rope_head_dim must be 0)."
    emb, h = config.emb_dim, self.num_heads

    self.wq_a = _dense(config, "wq_a", emb, self.q_lora_rank, ("embed", "q_lora"), dtype, quant, rngs)
    self.q_norm = RMSNorm(
        num_features=self.q_lora_rank, dtype=jnp.float32, weight_dtype=get_weight_dtype(config, "norm"),
        kernel_axes=("norm",), epsilon=config.normalization_layer_epsilon, rngs=rngs,
    )
    self.wq_b = _dense(config, "wq_b", self.q_lora_rank, (h, self.qk_head_dim), ("q_lora", "q_heads", "kv"),
                       dtype, quant, rngs)
    self.wkv_a = _dense(config, "wkv_a", emb, self.kv_lora_rank, ("embed", "kv_lora"), dtype, quant, rngs)
    self.kv_norm = RMSNorm(
        num_features=self.kv_lora_rank, dtype=jnp.float32, weight_dtype=get_weight_dtype(config, "norm"),
        kernel_axes=("norm",), epsilon=config.normalization_layer_epsilon, rngs=rngs,
    )
    self.wkv_b = _dense(config, "wkv_b", self.kv_lora_rank, (h, self.qk_head_dim + self.v_head_dim),
                        ("kv_lora", "kv_heads", "kv_head_dim"), dtype, quant, rngs)
    self.out = _dense(config, "out", (h, self.v_head_dim), emb, ("heads", "kv", "embed"), dtype, quant, rngs,
                      in_axis=(-2, -1))
    self.indexer = KPoolIndexer(config, dtype, quant, rngs=rngs)

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Optional[Array] = None,
      decoder_segment_ids: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ):
    """Returns (output [B, S, E], None)."""
    del inputs_kv, kwargs
    if model_mode != MODEL_MODE_TRAIN:
      # TODO(glm5.3): add latent KV + indexer caches for prefill/autoregressive decode.
      raise NotImplementedError("Glm5SparseMLA currently supports only full-sequence (train) mode.")
    x = inputs_q

    q_resid = self.q_norm(self.wq_a(x)).astype(self.dtype)
    q = self.wq_b(q_resid)  # [B, T, H, Dqk]
    kv_latent = self.kv_norm(self.wkv_a(x)).astype(self.dtype)
    kv = self.wkv_b(kv_latent)  # [B, S, H, Dqk + Dv]
    k, v = kv[..., : self.qk_head_dim], kv[..., self.qk_head_dim :]

    mask = self.indexer(x, q_resid, decoder_segment_ids)  # [B, T, S]

    logits = jnp.einsum("bthd,bshd->bhts", q.astype(jnp.float32), k.astype(jnp.float32), precision=_HIGHEST)
    logits = logits * self.scaling
    logits = jnp.where(mask[:, None], logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(logits, axis=-1).astype(self.dtype)
    attn = jnp.einsum("bhts,bshd->bthd", probs, v, precision=_HIGHEST)
    return self.out(attn), None
