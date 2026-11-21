# Copyright 2023–2026 Google LLC
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

"""Ming-Image MLLM text path: the BailingMoeV2 language model as a text-conditioning encoder (Flax NNX).

Port of `BailingMoeV2Model` (Ming-Image `modeling_bailing_moe_v2.py`), text path only (no vision tower), as run by
`BailingMM2NativeForConditionalGeneration.get_condition_embeds_for_image_gen` for text-to-image:

* attention: fused `query_key_value` (split into `query` / `key` / `value` at conversion), per-head `q_norm` /
  `k_norm` RMS norms, GQA, `video_rope` 3D RoPE on the first `partial_rotary_factor * head_dim` channels;
* MLP: dense SwiGLU for the first `first_k_dense_replace` layers, then a sparse MoE block (sigmoid router with
  `expert_bias`, group-limited top-k, `routed_scaling_factor`, one shared expert). Router type "MultiRouter": tokens at
  `image_mask` positions (the generation query tokens, inserted as `<image_patch>` tokens) are routed by
  `mlp.image_gate`, every other token by `mlp.gate` (`mlp.audio_gate` is unused without audio and not ported);
* `hidden_states[i]` semantics of HF: index 0 is the input embeddings, index i < num_layers is the input of layer i
  (output of layer i - 1, before any norm), index num_layers is the final-normed output.

Also: the host-side T2I input builder (`build_t2i_inputs`, `get_t_scale_rope_index`), the reference chat template
(`format_t2i_prompt`, `load_tokenizer`) and the HF -> MaxText parameter mapping (`mllm_param_mapping`).
"""

from collections.abc import Sequence
import os
from typing import Any, Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.sharding import Sharding
from maxtext.m3.models.wan.utils_wan import constrain, param_metadata

# NNX modules take model-specific `__call__` arguments; architecture kwargs mirror the HF configs.
# pylint: disable=arguments-differ,too-many-positional-arguments

BATCH = "activation_batch"
LENGTH = "activation_length"
EMBED = "activation_embed"
HEADS = "activation_heads"
MLP = "activation_mlp"

# Hard-coded in `BailingMoeV2Attention.__init__` (`rope_scaling = {"type": "mrope", "mrope_section": [8, 12, 12]}`).
MROPE_SECTION = (8, 12, 12)

# Reference chat template (`processing_bailingmm2.BailingMM2Processor.apply_chat_template`, no thinking).
SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
SYSTEM_PROMPT_THINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking on"
USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
# `mllm/special_tokens_map.json`: eos (= pad) token of the Ming tokenizer.
EOS_TOKEN = "<|role_end|>"
PAD_TOKEN_ID = 156895


def _llm_cfg(config: dict) -> dict:
  """The BailingMoeV2 config: `config["llm_config"]` (mllm/config.json) or `config` itself."""
  return config.get("llm_config", config)


# ==============================================================================
# Building blocks
# ==============================================================================


class MingRMSNorm(nnx.Module):
  """`BailingMoeV2RMSNorm` / `Qwen2RMSNorm`: fp32 statistics, cast back to the input dtype, then `weight * x`."""

  def __init__(self, dim: int, eps: float, weights_dtype=jnp.float32, sharding: Sharding | None = None):
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((dim,), weights_dtype), **param_metadata(sharding, "norm_scale", ("norm",)))

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
    return self.weight[...] * x32.astype(input_dtype)


def _linear(in_features, out_features, name, axes, *, use_bias, dtype, weights_dtype, precision, rngs, sharding):
  return nnx.Linear(
      in_features,
      out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.initializers.lecun_normal(),
      kernel_metadata=param_metadata(sharding, name, axes),
      bias_metadata=param_metadata(sharding, f"{name}_bias", (axes[-1],)),
      rngs=rngs,
  )


def rotate_half(x: jax.Array) -> jax.Array:
  half = x.shape[-1] // 2
  return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def video_rope_axes(mrope_section: Sequence[int] = MROPE_SECTION) -> np.ndarray:
  """Position axis (0 = t, 1 = h, 2 = w) of every rotary channel, as `apply_3d_rotary_pos_emb(rope_type="video_rope")`.

  The reference regroups `[t, h, w]` sections into `[t, h + w] * 2`, reverses them and walks the channels: inside an
  `h + w` section channels alternate h / w (restarting at h), a `t` section is all t. For [8, 12, 12] and 64 rotary
  channels: 24 x (h, w alternating), 8 x t, 24 x (h, w alternating), 8 x t.
  """
  sections = [mrope_section[0], mrope_section[1] + mrope_section[2]] * 2
  axes = []
  for i, section in enumerate(sections[::-1]):
    if i % 2 == 0:
      axes += [1 if j % 2 == 0 else 2 for j in range(section)]
    else:
      axes += [0] * section
  return np.asarray(axes, dtype=np.int32)


def video_rope_cos_sin(
    position_ids: jax.Array, inv_freq: np.ndarray, mrope_section: Sequence[int], dtype
) -> tuple[jax.Array, jax.Array]:
  """`BailingMoeV2RotaryEmbedding3D` + the video_rope channel selection: cos / sin `[B, L, rope_dim]` in `dtype`.

  Args:
    position_ids: `[3, B, L]` (t, h, w) positions.
    inv_freq: `[rope_dim // 2]` inverse frequencies.
    mrope_section: `[t, h, w]` sections, `2 * sum == rope_dim`.
    dtype: activation dtype (the reference casts cos / sin to the hidden-state dtype).
  """
  axes = video_rope_axes(mrope_section)
  inv_freq_full = np.concatenate([inv_freq, inv_freq]).astype(np.float32)  # emb = cat(freqs, freqs)
  if axes.shape[0] != inv_freq_full.shape[0]:
    raise ValueError(
        f"video_rope needs 2 * sum(mrope_section) == rope_dim, got {axes.shape[0]} vs {inv_freq_full.shape[0]}"
    )
  pos = jnp.moveaxis(position_ids.astype(jnp.float32)[axes], 0, -1)  # [B, L, rope_dim], axis per channel
  angles = pos * jnp.asarray(inv_freq_full)
  return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def default_inv_freq(rope_theta: float, rope_dim: int) -> np.ndarray:
  """transformers' "default" RoPE init: `1 / theta ** (arange(0, dim, 2) / dim)` (attention_scaling 1)."""
  exponent = np.arange(0, rope_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(rope_dim)
  return (1.0 / (np.float32(rope_theta) ** exponent)).astype(np.float32)


def attention_mask_4d(attention_mask: jax.Array, causal: bool) -> jax.Array:
  """`[B, 1, Lq, Lk]` boolean "may attend": key padding (and causal) mask."""
  length = attention_mask.shape[-1]
  mask = attention_mask.astype(jnp.bool_)[:, None, None, :]
  if causal:
    mask = mask & jnp.tril(jnp.ones((length, length), jnp.bool_))[None, None]
  return mask


def dot_product_attention(query, key, value, mask, *, precision) -> jax.Array:
  """Eager GQA attention: `softmax(q k^T / sqrt(d) + mask)` in fp32, `[B, L, H, D]` in and out."""
  batch, length, num_heads, head_dim = query.shape
  num_kv = key.shape[2]
  query = query.reshape(batch, length, num_kv, num_heads // num_kv, head_dim)
  scores = jnp.einsum("bqkgd,bskd->bkgqs", query / np.sqrt(head_dim).astype(query.dtype), key, precision=precision)
  scores = scores.astype(jnp.float32).reshape(batch, num_heads, length, length)
  scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
  weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
  weights = weights.reshape(batch, num_kv, num_heads // num_kv, length, length)
  out = jnp.einsum("bkgqs,bskd->bqkgd", weights, value, precision=precision)
  return out.reshape(batch, length, num_heads, head_dim)


# ==============================================================================
# BailingMoeV2 layers
# ==============================================================================


class MingMLP(nnx.Module):
  """SwiGLU MLP (`BailingMoeV2MLP`): `down_proj(silu(gate_proj(x)) * up_proj(x))`."""

  def __init__(self, hidden_size, intermediate_size, *, dtype, weights_dtype, precision, rngs, sharding=None):
    kw = {"use_bias": False, "dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs}
    self.sharding = sharding
    self.gate_proj = _linear(hidden_size, intermediate_size, "mlp_gate", ("embed", "mlp"), sharding=sharding, **kw)
    self.up_proj = _linear(hidden_size, intermediate_size, "mlp_up", ("embed", "mlp"), sharding=sharding, **kw)
    self.down_proj = _linear(intermediate_size, hidden_size, "mlp_down", ("mlp", "embed"), sharding=sharding, **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    hidden = jax.nn.silu(self.gate_proj(x)) * self.up_proj(x)
    hidden = constrain(self.sharding, hidden, "mlp_hidden", (BATCH, LENGTH, MLP))
    return self.down_proj(hidden)


class MingAttention(nnx.Module):
  """`BailingMoeV2Attention` (eager): q/k RMS norm per head, video_rope on the first `rope_dim` channels, GQA."""

  def __init__(self, cfg: dict, *, dtype, weights_dtype, precision, rngs, sharding=None):
    hidden = cfg["hidden_size"]
    self.num_heads = cfg["num_attention_heads"]
    self.num_kv_heads = cfg["num_key_value_heads"]
    self.head_dim = cfg.get("head_dim") or hidden // self.num_heads
    self.rope_dim = int(self.head_dim * cfg.get("partial_rotary_factor", 1.0))
    self.precision = precision
    self.sharding = sharding
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    qkv_bias = bool(cfg.get("use_qkv_bias", False))
    self.query = _linear(hidden, self.num_heads * self.head_dim, "attn_q", ("embed", "heads"), use_bias=qkv_bias, **kw)
    self.key = _linear(
        hidden, self.num_kv_heads * self.head_dim, "attn_k", ("embed", "kv_heads"), use_bias=qkv_bias, **kw
    )
    self.value = _linear(
        hidden, self.num_kv_heads * self.head_dim, "attn_v", ("embed", "kv_heads"), use_bias=qkv_bias, **kw
    )
    eps = cfg.get("rms_norm_eps", 1e-6)
    self.q_norm = MingRMSNorm(self.head_dim, eps, weights_dtype, sharding)
    self.k_norm = MingRMSNorm(self.head_dim, eps, weights_dtype, sharding)
    self.dense = _linear(
        self.num_heads * self.head_dim, hidden, "attn_out", ("heads", "embed"), use_bias=bool(cfg.get("use_bias")), **kw
    )

  def _rope(self, x, cos, sin):
    x_rot, x_pass = x[..., : self.rope_dim], x[..., self.rope_dim :]
    x_rot = x_rot * cos[:, :, None, :] + rotate_half(x_rot) * sin[:, :, None, :]
    return jnp.concatenate([x_rot, x_pass], axis=-1)

  def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array) -> jax.Array:
    batch, length, _ = x.shape
    query = self.query(x).reshape(batch, length, self.num_heads, self.head_dim)
    query = constrain(self.sharding, query, "attn_heads", (BATCH, LENGTH, HEADS, None))
    key = self.key(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
    value = self.value(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
    query, key = self._rope(self.q_norm(query), cos, sin), self._rope(self.k_norm(key), cos, sin)
    out = dot_product_attention(query, key, value, mask, precision=self.precision)
    out = constrain(self.sharding, out, "attn_heads", (BATCH, LENGTH, HEADS, None))
    return self.dense(out.reshape(batch, length, self.num_heads * self.head_dim))


class MingGate(nnx.Module):
  """`BailingMoeV2Gate`: sigmoid scores, `expert_bias` for selection only, group-limited top-k, scaled weights.

  `weight` is stored `[hidden, num_experts]` (HF `[num_experts, hidden]`, transposed at conversion). Logits are fp32.
  """

  def __init__(self, cfg: dict, *, rngs, sharding=None):
    hidden, self.num_experts = cfg["hidden_size"], cfg["num_experts"]
    self.top_k, self.n_group, self.topk_group = cfg["num_experts_per_tok"], cfg["n_group"], cfg["topk_group"]
    self.routed_scaling_factor = cfg["routed_scaling_factor"]
    self.weight = nnx.Param(
        nnx.initializers.normal(0.02)(rngs.params(), (hidden, self.num_experts), jnp.float32),
        **param_metadata(sharding, "router_weight", ("embed", None)),
    )
    self.expert_bias = nnx.Param(
        jnp.zeros((self.num_experts,), jnp.float32), **param_metadata(sharding, "router_bias", (None,))
    )

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`[T, hidden]` -> (top-k expert indices `[T, k]` int32, top-k weights `[T, k]` fp32)."""
    logits = jnp.matmul(x.astype(jnp.float32), self.weight[...].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    scores = jax.nn.sigmoid(logits)
    scores_for_routing = scores + self.expert_bias[...].astype(jnp.float32)
    grouped = scores_for_routing.reshape(-1, self.n_group, self.num_experts // self.n_group)
    group_scores = jnp.sum(jax.lax.top_k(grouped, 2)[0], axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, self.topk_group)
    group_mask = jnp.any(jnp.arange(self.n_group)[None, :, None] == group_idx[:, None, :], axis=-1)
    score_mask = jnp.repeat(group_mask, self.num_experts // self.n_group, axis=-1)
    _, topk_idx = jax.lax.top_k(jnp.where(score_mask, scores_for_routing, -jnp.inf), self.top_k)
    topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
    if self.top_k > 1:  # The reference normalizes whenever top_k > 1 (it ignores `norm_topk_prob`).
      topk_weight = topk_weight / (jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20)
    return topk_idx.astype(jnp.int32), topk_weight * self.routed_scaling_factor


class MingExperts(nnx.Module):
  """Routed SwiGLU experts, weights stacked `[E, in, out]`; sorted-token `jax.lax.ragged_dot` dispatch."""

  def __init__(self, cfg: dict, *, dtype, weights_dtype, precision, rngs, sharding=None):
    e, h, i = cfg["num_experts"], cfg["hidden_size"], cfg["moe_intermediate_size"]
    self.num_experts, self.top_k = e, cfg["num_experts_per_tok"]
    self.dtype, self.precision = dtype, precision
    init = nnx.initializers.normal(0.02)

    def param(name, shape, axes):
      return nnx.Param(init(rngs.params(), shape, weights_dtype), **param_metadata(sharding, name, axes))

    self.gate_proj = param("experts_gate", (e, h, i), ("expert", "embed", "mlp"))
    self.up_proj = param("experts_up", (e, h, i), ("expert", "embed", "mlp"))
    self.down_proj = param("experts_down", (e, i, h), ("expert", "mlp", "embed"))

  def __call__(self, x: jax.Array, topk_idx: jax.Array, topk_weight: jax.Array) -> jax.Array:
    """`x [T, hidden]`, `topk_idx / topk_weight [T, k]` -> `sum_k w_k * expert_k(x) [T, hidden]`."""
    x = x.astype(self.dtype)
    flat_idx = topk_idx.reshape(-1)
    perm = jnp.argsort(flat_idx, stable=True)
    sorted_x = x[perm // self.top_k]
    group_sizes = jnp.bincount(flat_idx, length=self.num_experts).astype(jnp.int32)

    def ragged(lhs, rhs):
      return jax.lax.ragged_dot(
          lhs, rhs[...].astype(self.dtype), group_sizes, precision=self.precision, preferred_element_type=self.dtype
      )

    hidden = jax.nn.silu(ragged(sorted_x, self.gate_proj)) * ragged(sorted_x, self.up_proj)
    sorted_out = ragged(hidden, self.down_proj)
    out = sorted_out[jnp.argsort(perm)].astype(jnp.float32).reshape(x.shape[0], self.top_k, -1)
    return jnp.sum(out * topk_weight[..., None].astype(jnp.float32), axis=1).astype(self.dtype)


class MingSparseMoeBlock(nnx.Module):
  """`BailingMoeV2SparseMoeBlock` with router_type "MultiRouter" (text `gate` / `image_gate`) + shared experts."""

  def __init__(self, cfg: dict, *, dtype, weights_dtype, precision, rngs, sharding=None):
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.multi_router = cfg.get("router_type", "topN") == "MultiRouter"
    self.experts = MingExperts(cfg, **kw)
    self.gate = MingGate(cfg, rngs=rngs, sharding=sharding)
    if self.multi_router:
      self.image_gate = MingGate(cfg, rngs=rngs, sharding=sharding)
    self.has_shared_experts = bool(cfg.get("num_shared_experts"))
    if self.has_shared_experts:
      shared = cfg["moe_intermediate_size"] * cfg["num_shared_experts"]
      self.shared_experts = MingMLP(cfg["hidden_size"], shared, **kw)

  def __call__(self, x: jax.Array, image_mask: Optional[jax.Array]) -> jax.Array:
    batch, length, hidden = x.shape
    x_flat = x.reshape(-1, hidden)
    topk_idx, topk_weight = self.gate(x_flat)
    if self.multi_router and image_mask is not None:
      image_idx, image_weight = self.image_gate(x_flat)
      is_image = image_mask.reshape(-1, 1).astype(jnp.bool_)
      topk_idx = jnp.where(is_image, image_idx, topk_idx)
      topk_weight = jnp.where(is_image, image_weight, topk_weight)
    out = self.experts(x_flat, topk_idx, topk_weight).reshape(batch, length, hidden)
    if self.has_shared_experts:
      out = out + self.shared_experts(x)
    return out


class MingDecoderLayer(nnx.Module):
  """`BailingMoeV2DecoderLayer`: pre-norm attention and (dense or MoE) MLP residual blocks."""

  def __init__(self, cfg: dict, layer_idx: int, *, dtype, weights_dtype, precision, rngs, sharding=None):
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.sharding = sharding
    self.attention = MingAttention(cfg, **kw)
    if cfg.get("num_experts") and layer_idx >= cfg.get("first_k_dense_replace", 0):
      self.mlp = MingSparseMoeBlock(cfg, **kw)
      self.is_moe = True
    else:
      self.mlp = MingMLP(cfg["hidden_size"], cfg["intermediate_size"], **kw)
      self.is_moe = False
    eps = cfg.get("rms_norm_eps", 1e-6)
    self.input_layernorm = MingRMSNorm(cfg["hidden_size"], eps, weights_dtype, sharding)
    self.post_attention_layernorm = MingRMSNorm(cfg["hidden_size"], eps, weights_dtype, sharding)

  def __call__(self, x, cos, sin, mask, image_mask):
    x = x + self.attention(self.input_layernorm(x), cos, sin, mask)
    x = constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))
    hidden = self.post_attention_layernorm(x)
    x = x + (self.mlp(hidden, image_mask) if self.is_moe else self.mlp(hidden))
    return constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))


class MingTextEncoder(nnx.Module):
  """Flax NNX port of the Ming-Image MLLM language model (`BailingMoeV2Model`, text path) for T2I conditioning.

  `MingTextEncoder(llm_config=<mllm/config.json "llm_config">, rngs=...)` matches the config stored by the checkpoint
  converter. NNX param paths mirror the HF keys `model.model.<path>` (fused `query_key_value` is split into
  `query` / `key` / `value`; experts are stacked `[E, in, out]`; `audio_gate` is not ported).

  Sharding (`sharding=None`: no annotations, for CPU / conversion): parameter logical axes `vocab`, `embed`, `heads`,
  `kv_heads`, `mlp`, `expert`, `norm` (router weights / biases: `embed` + replicated); activation axes
  `activation_batch`, `activation_length`, `activation_embed`, `activation_heads`, `activation_mlp`.
  """

  def __init__(
      self,
      llm_config: dict,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    cfg = _llm_cfg(llm_config)
    self.dtype = jnp.dtype(dtype)
    self.sharding = sharding
    weights_dtype = jnp.dtype(weights_dtype)
    self.vocab_size = cfg["vocab_size"]
    self.hidden_size = cfg["hidden_size"]
    self.num_layers = cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    rope_dim = int(head_dim * cfg.get("partial_rotary_factor", 1.0))
    rope_type = (cfg.get("rope_scaling") or {}).get("rope_type", (cfg.get("rope_scaling") or {}).get("type"))
    if rope_type != "video_rope":
      raise NotImplementedError(f"Only rope_scaling type 'video_rope' is supported, got {rope_type!r}.")
    self.mrope_section = MROPE_SECTION
    self.rope_theta = float(cfg.get("rope_theta", 10000.0))
    self.rope_dim = rope_dim
    self.word_embeddings = nnx.Embed(
        self.vocab_size,
        self.hidden_size,
        dtype=self.dtype,
        param_dtype=weights_dtype,
        embedding_init=nnx.initializers.normal(0.02),
        embedding_metadata=param_metadata(sharding, "word_embeddings", ("vocab", "embed")),
        rngs=rngs,
    )
    kw = {"dtype": self.dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs}
    self.layers = nnx.List([MingDecoderLayer(cfg, i, sharding=sharding, **kw) for i in range(self.num_layers)])
    self.norm = MingRMSNorm(self.hidden_size, cfg.get("rms_norm_eps", 1e-6), weights_dtype, sharding)

  def embed(
      self, input_ids: jax.Array, image_mask: Optional[jax.Array] = None, query_embeds: Optional[jax.Array] = None
  ) -> jax.Array:
    """Word embeddings with the k-th `image_mask` position of each row replaced by `query_embeds[k]`.

    Matches `prompt_wrap_navit` / `masked_scatter` for the T2I case, where every sample gets the same query tokens
    (`query_embeds [Q, hidden]`, or per-sample `[B, Q, hidden]`).
    """
    input_ids = jnp.clip(input_ids, 0, self.vocab_size - 1)
    x = self.word_embeddings(input_ids)
    if query_embeds is not None and image_mask is not None:
      image_mask = image_mask.astype(jnp.bool_)
      num_query = query_embeds.shape[-2]
      slot = jnp.clip(jnp.cumsum(image_mask, axis=-1) - 1, 0, num_query - 1)
      if query_embeds.ndim == 2:
        injected = query_embeds[slot]
      else:
        injected = jnp.take_along_axis(query_embeds, slot[..., None], axis=1)
      x = jnp.where(image_mask[..., None], injected.astype(x.dtype), x)
    return x

  def __call__(
      self,
      input_ids: jax.Array,
      position_ids: jax.Array,
      attention_mask: Optional[jax.Array] = None,
      image_mask: Optional[jax.Array] = None,
      query_embeds: Optional[jax.Array] = None,
      output_layers: Sequence[int] = (5, 12, 20),
      inputs_embeds: Optional[jax.Array] = None,
  ) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Runs the LLM and returns `(tuple(hidden_states[i] for i in output_layers), hidden_states[-1])`.

    Args:
      input_ids: `[B, L]` token ids (from `build_t2i_inputs`).
      position_ids: `[3, B, L]` video_rope (t, h, w) positions (from `build_t2i_inputs`).
      attention_mask: `[B, L]`, nonzero for real tokens; causal + key-padding mask. Defaults to all ones.
      image_mask: `[B, L]` bool, positions of the query tokens: their embeddings come from `query_embeds` and they
        are routed by `mlp.image_gate`.
      query_embeds: `[Q, hidden]` (the conditioner's `query_tokens`) or `[B, Q, hidden]`.
      output_layers: HF `hidden_states` indices to return (0 = embeddings, i = input of layer i, num_layers = output
        after the final norm; negative indices count from the end like python).
      inputs_embeds: optional precomputed `[B, L, hidden]` embeddings (skips `embed`).

    Returns:
      (selected hidden states, last hidden state after the final norm), each `[B, L, hidden]`.
    """
    if attention_mask is None:
      attention_mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)
    x = self.embed(input_ids, image_mask, query_embeds) if inputs_embeds is None else inputs_embeds.astype(self.dtype)
    x = constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))
    inv_freq = default_inv_freq(self.rope_theta, self.rope_dim)
    cos, sin = video_rope_cos_sin(position_ids, inv_freq, self.mrope_section, self.dtype)
    mask = attention_mask_4d(attention_mask, causal=True)
    wanted = {i % (self.num_layers + 1) for i in output_layers}
    hidden_states = {0: x} if 0 in wanted else {}
    for i, layer in enumerate(self.layers):
      x = layer(x, cos, sin, mask, image_mask)
      if i + 1 in wanted and i + 1 < self.num_layers:
        hidden_states[i + 1] = x
    last = self.norm(x)
    hidden_states[self.num_layers] = last
    return tuple(hidden_states[i % (self.num_layers + 1)] for i in output_layers), last


# ==============================================================================
# Host-side T2I inputs (numpy)
# ==============================================================================


def get_t_scale_rope_index(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: Sequence[Sequence[int]],
    *,
    image_start_token: int,
    image_patch_token: int,
    spatial_merge_size: int = 2,
    scale_factor: float = 2.0,
) -> np.ndarray:
  """Numpy port of `get_t_scale_rope_index` (video_rope prefill positions) for text + image inputs.

  `BailingMoeV2Model.forward` calls it with `scale_factor=2.0` and no `second_per_grid_ts`, overriding any
  `position_ids` argument. Images are consumed from `image_grid_thw` in order over the flattened batch. Padded
  positions (attention_mask 0) get position 1, like the reference.

  Returns:
    `[3, B, L]` int32 positions.
  """
  input_ids, attention_mask = np.asarray(input_ids), np.asarray(attention_mask)
  batch, length = input_ids.shape
  position_ids = np.ones((3, batch, length), dtype=np.int64)
  image_index = 0
  for b in range(batch):
    valid = attention_mask[b] == 1
    tokens = input_ids[b][valid]
    starts = np.flatnonzero(tokens == image_start_token)
    starts = starts[starts + 1 < len(tokens)]
    num_images = int(np.sum(tokens[starts + 1] == image_patch_token))
    token_list = tokens.tolist()
    chunks, st = [], 0
    for _ in range(num_images):
      ed = token_list.index(image_patch_token, st)
      t, h, w = image_grid_thw[image_index]
      image_index += 1
      grid_t, grid_h, grid_w = int(t), int(h) // spatial_merge_size, int(w) // spatial_merge_size
      text_len = ed - st
      st_idx = chunks[-1][0].max() + 1 if chunks else 0
      chunks.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
      t_index = np.repeat(np.arange(grid_t), grid_h * grid_w).astype(np.float64) * scale_factor + text_len + st_idx
      h_index = np.tile(np.repeat(np.arange(grid_h), grid_w), grid_t) - (grid_h - 1) // 2 + t_index
      w_index = np.tile(np.arange(grid_w), grid_t * grid_h) - (grid_w - 1) // 2 + t_index
      chunks.append(np.stack([t_index, h_index, w_index]))
      st = ed + grid_t * grid_h * grid_w
    if st < len(token_list):
      st_idx = chunks[-1][0].max() + 1 if chunks else 0
      text_len = len(token_list) - st
      chunks.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
    if chunks:
      position_ids[:, b, valid] = np.concatenate(chunks, axis=1).astype(np.int64)  # torch `.to(long)` truncates
  return position_ids.astype(np.int32)


def build_t2i_inputs(token_ids: Sequence[int], max_length: int, llm_config: dict, img_gen_scales=(16,)) -> dict:
  """Per-sample LLM inputs of the T2I condition path, padded to the static `max_length`.

  Reproduces `append_input_ids_with_multiscale_learnable_tokens` (+ `appand_learnable_tokens` /
  `prompt_wrap_navit` masks and the model's `get_t_scale_rope_index` positions): the prompt is followed, per scale s,
  by `<image_start>`, s*s `<image_patch>` and `<image_end>`, then padding. Trailing padding never changes the valid
  outputs (causal attention, per-token MoE), so the static length is exact.

  Args:
    token_ids: prompt token ids after the chat template (`format_t2i_prompt`), without padding.
    max_length: static sequence length L (>= len(token_ids) + sum(s*s + 2)).
    llm_config: mllm/config.json (or its "llm_config") for the special token ids.
    img_gen_scales: `img_gen_scales` of mlp/config.json.

  Returns:
    numpy dict: `input_ids [L]` int32, `attention_mask [L]` int32, `position_ids [3, L]` int32, `image_mask [L]`
    bool (query-token slots; also the `image_gate` routing mask), `gen_mask [L]` bool (query-token positions whose
    last hidden state feeds `proj_in`), `prompt_mask [L]` bool (directvlm tokens: reference
    `labels < 0 & attention_mask`, i.e. the prompt tokens).
  """
  cfg = _llm_cfg(llm_config)
  patch = cfg["image_patch_token"]
  start, end = patch + 1, patch + 2  # The reference passes image_patch_token + 1 / + 2.
  token_ids = [int(t) for t in token_ids]
  n = len(token_ids)
  gen_ids, gen = [], []
  for s in img_gen_scales:
    gen_ids += [start] + [patch] * (s * s) + [end]
    gen += [False] + [True] * (s * s) + [False]
  valid = n + len(gen_ids)
  if valid > max_length:
    raise ValueError(f"Prompt of {n} tokens + {len(gen_ids)} generation tokens exceeds max_length={max_length}.")
  pad = max_length - valid
  input_ids = np.asarray(token_ids + gen_ids + [PAD_TOKEN_ID] * pad, dtype=np.int32)
  attention_mask = np.asarray([1] * valid + [0] * pad, dtype=np.int32)
  gen_mask = np.asarray([False] * n + gen + [False] * pad)
  prompt_mask = np.asarray([True] * n + [False] * (max_length - n))
  image_mask = input_ids == patch
  grid = [[1, 2, s * s * 2] for s in img_gen_scales]  # `appand_learnable_tokens`: a 1 x (s*s) row of patches.
  position_ids = get_t_scale_rope_index(
      input_ids[None],
      attention_mask[None],
      grid,
      image_start_token=cfg.get("image_start_token", start),
      image_patch_token=patch,
      spatial_merge_size=cfg.get("spatial_merge_size", 2),
  )[:, 0]
  return {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "position_ids": position_ids,
      "image_mask": image_mask,
      "gen_mask": gen_mask,
      "prompt_mask": prompt_mask,
  }


def t2i_prompt_text(prompt: str, eos_token: str = EOS_TOKEN, use_cot_system_prompt: bool = False) -> str:
  """The reference T2I chat text: `apply_chat_template([{"role": "HUMAN", "content": [text]}])`."""
  system = SYSTEM_PROMPT_THINK if use_cot_system_prompt else SYSTEM_PROMPT
  return system + eos_token + USER_PREFIX + prompt + eos_token + ASSISTANT_PREFIX


def load_tokenizer(mllm_dir: str):
  """The Ming tokenizer (`BailingTokenizer`: a `PreTrainedTokenizerFast` with no added BOS / EOS) from `mllm/`.

  Built from `tokenizer.json` + the special-token files with plain `transformers.PreTrainedTokenizerFast` (the repo's
  `tokenizer_config.json` names a remote-code class), which encodes identically.
  """
  import transformers  # pylint: disable=import-outside-toplevel

  return transformers.PreTrainedTokenizerFast(
      tokenizer_file=os.path.join(mllm_dir, "tokenizer.json"),
      bos_token="<|startoftext|>",
      eos_token=EOS_TOKEN,
      pad_token=EOS_TOKEN,
      cls_token="[CLS]",
  )


def format_t2i_prompt(prompt: str, tokenizer: Any) -> list[int]:
  """Token ids of `prompt` after the reference T2I chat template (what the processor feeds the model)."""
  return list(tokenizer(t2i_prompt_text(prompt, tokenizer.eos_token or EOS_TOKEN))["input_ids"])


# ==============================================================================
# HF -> MaxText parameter mapping
# ==============================================================================


def mllm_param_mapping(hf_config: dict) -> tuple[dict, dict]:
  """`({maxtext_key: hf_key | [hf_keys]}, {maxtext_key: hook})` for `MingTextEncoder` from the mllm checkpoint.

  `hf_config` is `{"llm_config": {...}}` (the converter's `model_init_config`) or the llm config itself. List values
  (stacked experts) are stacked along axis 0 by the converter; hooks see each list entry with `target_shape[1:]`.
  """
  cfg = _llm_cfg(hf_config)
  nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
  hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
  q_end, k_end = nh * hd, (nh + nkv) * hd
  mapping, hooks = {}, {}

  def transpose(x, target_shape):
    del target_shape
    return x.T

  def qkv_split(lo, hi):
    def hook(x, target_shape):
      del target_shape
      return x[lo:hi].T if x.ndim == 2 else x[lo:hi]

    return hook

  def add(key, hf, hook=None):
    mapping[key] = hf
    if hook is not None:
      hooks[key] = hook

  add("word_embeddings-embedding", "model.model.word_embeddings.weight")
  add("norm-weight", "model.model.norm.weight")
  qkv_bias = bool(cfg.get("use_qkv_bias", False))
  for i in range(cfg["num_hidden_layers"]):
    mt, hf = f"layers-{i}", f"model.model.layers.{i}"
    for name, (lo, hi) in (("query", (0, q_end)), ("key", (q_end, k_end)), ("value", (k_end, k_end + nkv * hd))):
      add(f"{mt}-attention-{name}-kernel", f"{hf}.attention.query_key_value.weight", qkv_split(lo, hi))
      if qkv_bias:
        add(f"{mt}-attention-{name}-bias", f"{hf}.attention.query_key_value.bias", qkv_split(lo, hi))
    add(f"{mt}-attention-dense-kernel", f"{hf}.attention.dense.weight", transpose)
    if cfg.get("use_bias"):
      add(f"{mt}-attention-dense-bias", f"{hf}.attention.dense.bias")
    for norm in ("q_norm", "k_norm"):
      add(f"{mt}-attention-{norm}-weight", f"{hf}.attention.{norm}.weight")
    for norm in ("input_layernorm", "post_attention_layernorm"):
      add(f"{mt}-{norm}-weight", f"{hf}.{norm}.weight")
    mlps = []
    if cfg.get("num_experts") and i >= cfg.get("first_k_dense_replace", 0):
      gates = ("gate", "image_gate") if cfg.get("router_type", "topN") == "MultiRouter" else ("gate",)
      for gate in gates:
        add(f"{mt}-mlp-{gate}-weight", f"{hf}.mlp.{gate}.weight", transpose)
        add(f"{mt}-mlp-{gate}-expert_bias", f"{hf}.mlp.{gate}.expert_bias")
      for proj in ("gate_proj", "up_proj", "down_proj"):
        experts = [f"{hf}.mlp.experts.{e}.{proj}.weight" for e in range(cfg["num_experts"])]
        add(f"{mt}-mlp-experts-{proj}", experts, transpose)
      if cfg.get("num_shared_experts"):
        mlps.append(("mlp-shared_experts", "mlp.shared_experts"))
    else:
      mlps.append(("mlp", "mlp"))
    for mt_mlp, hf_mlp in mlps:
      for proj in ("gate_proj", "up_proj", "down_proj"):
        add(f"{mt}-{mt_mlp}-{proj}-kernel", f"{hf}.{hf_mlp}.{proj}.weight", transpose)
  return mapping, hooks
