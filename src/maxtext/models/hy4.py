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

"""Tencent Hy4-preview (HYV4ForCausalLM) model implementation.

Reference: https://huggingface.co/tencent/Hy4-preview (`transformers.models.hy_v4.modeling_hy_v4`).

Text decoder layer `i`:
  * Hyper-connections: 4-stream Identity Hyper-Connections (iHC) wrapping both attention (`attn_hc`)
    and MLP (`ffn_hc`), plus `Hy4HyperHead` (`hc_head`) before the final decoder RMSNorm.
  * Attention: Gated MLA (`q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`,
    `qk_rope_head_dim=64`, `v_head_dim=256`, `q_norm`/`kv_norm` epsilon=1e-6, non-interleaved RoPE,
    learnable per-head attention sinks `sinks`, and elementwise sigmoid output gate `gate_proj`).
  * DSA Indexer: `"full"` when `i == 0 or (i - 1) % 4 == 0`, `"shared"` otherwise (reusing the
    preceding `"full"` layer's top-k indices / sparse attention mask).
  * MLP: dense SwiGLU for `i < first_num_dense_layers` (Layer 0), and sigmoid-routed MoE
    (256 routed experts with clamped SwiGLU `mlp_activations_limit=10.0` + 1 unclamped shared expert)
    for `i >= first_num_dense_layers`.
"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any, Optional

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    Array,
    BATCH_ATTN,
    Config,
    DEFAULT_MASK_VALUE,
    D_KV,
    HEAD,
    LENGTH,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_KV_BATCH,
    PREFILL_LENGTH,
)
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.initializers import default_bias_init, default_scalar_init, nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical


HY4_HC_EPS = 1e-6
HY4_HC_MAGNITUDE = 2.0
HY4_MLA_LORA_NORM_EPS = 1e-6


def is_full_indexer_layer(layer_idx: int, config: Config) -> bool:
  """Mirrors `HYV4Config`: `"full" if layer_idx == 0 or (layer_idx - 1) % cycle == 0 else "shared"`."""
  cycle = config.inhomogeneous_layer_cycle_interval
  if layer_idx < config.first_num_dense_layers:
    return True
  return (layer_idx - config.first_num_dense_layers) % cycle == 0


def is_moe_layer(layer_idx: int, config: Config) -> bool:
  """Layer 0 is dense MLP; layers >= first_num_dense_layers are sparse MoE."""
  return config.num_experts > 1 and layer_idx >= config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int, int]:
  """Static `scan_layers=True` layout: `(num_prefix_layers, num_scanned_blocks, remainder_start)`.

  Layers `[0, num_prefix_layers)` are unrolled (`first_num_dense_layers`, i.e. Layer 0).
  The following `num_scanned_blocks` cycles of `inhomogeneous_layer_cycle_interval` (4) layers are
  homogeneous MoE cycles `[full_indexer, shared_indexer, shared_indexer, shared_indexer]` and are
  scanned as `Hy4ScannableBlock`s. Any trailing layers `[remainder_start, num_decoder_layers)`
  (e.g. Layer 77 in the 78-layer full model, or Layers 1..2 in the 3-layer mini model) are unrolled.
  """
  cycle = config.inhomogeneous_layer_cycle_interval
  num_layers = config.num_decoder_layers
  num_prefix = min(num_layers, config.first_num_dense_layers)
  num_blocks = (num_layers - num_prefix) // cycle
  return num_prefix, num_blocks, num_prefix + num_blocks * cycle


# -----------------------------------------
# Identity Hyper-Connections (iHC)
# -----------------------------------------


class Hy4HyperConnection(nnx.Module):
  """HYV4 Identity Hyper-Connection (`HYV4HyperConnection`) wrapping an attention or FFN branch."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.k = config.mhc_expansion_rate
    self.emb_dim = config.emb_dim
    self.rms_norm_eps = config.normalization_layer_epsilon
    self.hc_eps = HY4_HC_EPS
    self.hc_magnitude = HY4_HC_MAGNITUDE

    weight_init = nd_dense_init(1.0, "fan_in", "normal")
    # Stored in MaxText as [k * emb_dim, 2 * k]; HF `fn` is [2 * k, k * emb_dim].
    self.fn = nnx.Param(
        weight_init(
            rngs.params(),
            (self.k * self.emb_dim, 2 * self.k),
            jnp.float32,
            in_axis=0,
            out_axis=1,
        ),
        out_sharding=("activation_embed", None),
    )
    self.base = nnx.Param(
        default_bias_init(rngs.params(), (2 * self.k,), jnp.float32),
        out_sharding=(None,),
    )
    self.scale = nnx.Param(
        default_scalar_init(rngs.params(), (2,), jnp.float32),
        out_sharding=(None,),
    )

  def compute_pre_post(self, hidden_streams: Array) -> tuple[Array, Array]:
    """Computes `(branch_input [B, S, D], post [B, S, K])` from `hidden_streams [B, S, K, D]`."""
    b, s, k, d = hidden_streams.shape
    assert k == self.k and d == self.emb_dim

    flat = jnp.reshape(hidden_streams, (b, s, k * d)).astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + self.rms_norm_eps)

    fn = jnp.asarray(self.fn[...], jnp.float32)
    base = jnp.asarray(self.base[...], jnp.float32)
    scale = jnp.asarray(self.scale[...], jnp.float32)

    mixes = jnp.einsum("bsm,mn->bsn", flat, fn, precision=jax.lax.Precision(self.config.matmul_precision)) * inv_rms
    pre_b, post_b = jnp.split(base, 2, axis=-1)
    scale_flat = scale.reshape(2)
    pre_scale = scale_flat[0]
    post_scale = scale_flat[1]
    pre_logits, post_logits = jnp.split(mixes, 2, axis=-1)

    pre = jax.nn.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps
    post = self.hc_magnitude * jax.nn.sigmoid(post_logits * post_scale + post_b) + self.hc_eps

    branch_in = jnp.sum(pre[:, :, :, None] * hidden_streams.astype(jnp.float32), axis=2).astype(hidden_streams.dtype)
    return branch_in, post

  def apply_residual(self, residual: Array, branch_out: Array, post: Array) -> Array:
    """Updates `residual [B, S, K, D]` with `branch_out [B, S, D]` and `post [B, S, K]`."""
    out_f32 = post.astype(jnp.float32)[:, :, :, None] * branch_out.astype(jnp.float32)[:, :, None, :] + residual.astype(
        jnp.float32
    )
    return out_f32.astype(branch_out.dtype)


class Hy4HyperHead(nnx.Module):
  """Final HC-stream collapse (`HYV4HyperHead`) before `decoder_norm`."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.mhc_expansion_rate = config.mhc_expansion_rate
    self.emb_dim = config.emb_dim
    self.rms_norm_eps = config.normalization_layer_epsilon
    self.hc_eps = HY4_HC_EPS

    weight_init = nd_dense_init(1.0, "fan_in", "normal")
    self.hc_fn = nnx.Param(
        weight_init(
            rngs.params(),
            (self.mhc_expansion_rate * self.emb_dim, self.mhc_expansion_rate),
            jnp.float32,
            in_axis=0,
            out_axis=1,
        ),
        out_sharding=("activation_embed", None),
    )
    self.hc_base = nnx.Param(
        default_bias_init(rngs.params(), (self.mhc_expansion_rate,), jnp.float32),
        out_sharding=(None,),
    )
    self.hc_scale = nnx.Param(
        default_scalar_init(rngs.params(), (1,), jnp.float32),
        out_sharding=(None,),
    )

  def __call__(self, x: Array) -> Array:
    b, s, k, d = x.shape
    assert k == self.mhc_expansion_rate
    assert d == self.emb_dim

    flat_f32 = jnp.reshape(x, (b, s, k * d)).astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True) + self.rms_norm_eps)

    hc_fn = jnp.asarray(self.hc_fn[...], jnp.float32)
    hc_base = jnp.asarray(self.hc_base[...], jnp.float32)
    hc_scale = jnp.asarray(self.hc_scale[...], jnp.float32)

    mixes = (
        jnp.einsum("bsm,mk->bsk", flat_f32, hc_fn, precision=jax.lax.Precision(self.config.matmul_precision)) * inv_rms
    )
    pre = jax.nn.sigmoid(mixes * hc_scale[None, None, :] + hc_base[None, None, :]) + self.hc_eps

    out = jnp.sum(pre[:, :, :, None] * x.astype(jnp.float32), axis=2)
    return out.astype(x.dtype)


# -----------------------------------------
# DSA Indexer & Gated MLA with Learnable Sinks
# -----------------------------------------


class Hy4Indexer(attention_mla.Indexer):
  """HYV4 Sparse Attention Indexer (`HYV4Indexer`).

  Differs from DeepSeek V3.2's `Indexer` in three ways:
    1. RoPE split order: splits `head_dim` (128) into `[nope (64), rope (64)]` and applies
       non-interleaved RoPE to the SECOND half (matching `HYV4Indexer.forward`).
    2. `k_norm`: standard LayerNorm (`eps=1e-5`, with bias) computed in float32 and cast back
       to `inputs_q.dtype` before RoPE.
    3. Scoring: `q @ k^T` and `weights @ scores` are computed in `float32` as in `HYV4Indexer`.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.k_norm = nnx.LayerNorm(
        num_features=self.head_dim,
        epsilon=1e-5,
        use_bias=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=self.rngs,
    )
    self.weights_proj = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.n_heads,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", None),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

  def apply_partial_rope(
      self,
      inputs: Array,
      inputs_positions: Optional[Array | None] = None,
  ) -> Array:
    """Splits `[head_dim - rope_head_dim, rope_head_dim]` and applies RoPE to the second part."""
    nope_dim = self.head_dim - self.rope_head_dim
    x_nope, x_pe = jnp.split(inputs, [nope_dim], axis=-1)
    x_pe = self.rotary_embedding(x_pe, position=inputs_positions)
    return jnp.concatenate([x_nope, x_pe], axis=-1)

  def __call__(
      self,
      inputs_q: Array,
      low_rank_q: Array,
      inputs_kv: Array,
      inputs_positions: Optional[Array | None] = None,
      attention_mask: Optional[Array | None] = None,
      decoder_segment_ids: Optional[Array | None] = None,
      previous_chunk: Any = None,
      kv_cache: Any = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ):
    bsz, seqlen, _ = inputs_q.shape
    inputs_q = jax.lax.stop_gradient(inputs_q)
    low_rank_q = jax.lax.stop_gradient(low_rank_q)
    inputs_kv = jax.lax.stop_gradient(inputs_kv)

    q = self.wq_b(low_rank_q).reshape(bsz, seqlen, self.n_heads, self.head_dim)
    q = self.apply_partial_rope(q, inputs_positions=inputs_positions)

    k = self.wk(inputs_kv)
    k = self.k_norm(k.astype(jnp.float32)).astype(inputs_kv.dtype)
    k = k[:, :, None, :]
    k = self.apply_partial_rope(k, inputs_positions=inputs_positions)
    k = k.squeeze(2)

    cached_s = None
    if model_mode != MODEL_MODE_TRAIN and kv_cache is not None:
      k_cached, cached_s = self.update_indexer_cache(kv_cache, k, decoder_segment_ids, model_mode, previous_chunk)
      k = k_cached if k_cached is not None else k

    if k.shape[1] <= self.indexer_topk:
      return None, None, None

    weights = self.weights_proj(inputs_q).astype(jnp.float32) * (self.n_heads**-0.5) * self.softmax_scale
    logits = jnp.einsum(
        "bthd,bsd->btsh",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    logits = jax.nn.relu(logits)
    indexer_score = jnp.einsum(
        "btsh,bth->bts",
        logits,
        weights,
        precision=jax.lax.Precision.HIGHEST,
    )

    internal_padding_mask = None
    if cached_s is not None:
      internal_padding_mask = jnp.where(cached_s > 0, 0.0, DEFAULT_MASK_VALUE)
      indexer_score = indexer_score + internal_padding_mask[:, None, :]

    if attention_mask is not None:
      indexer_score = indexer_score + attention_mask

    topk_values, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)
    del topk_values
    # Match HYV4Attention: scatter False at topk_indices (unmasked=0.0, masked=DEFAULT_MASK_VALUE)
    selected = (
        jnp.zeros_like(indexer_score, dtype=jnp.bool_)
        .at[
            jnp.arange(bsz)[:, None, None],
            jnp.arange(seqlen)[None, :, None],
            topk_indices,
        ]
        .set(True)
    )
    indexer_mask = jnp.where(
        selected,
        jnp.array(0.0, dtype=self.dtype),
        jnp.array(DEFAULT_MASK_VALUE, dtype=self.dtype),
    )

    if attention_mask is not None:
      indexer_mask = indexer_mask + attention_mask
    if internal_padding_mask is not None:
      indexer_mask = indexer_mask + internal_padding_mask[:, None, :]

    return indexer_mask, topk_indices, indexer_score


class Hy4Attention(attention_mla.MLA):
  """HYV4 Gated Multi-head Latent Attention with learnable sinks and full/shared DSA indexer."""

  def __init__(self, *args, is_full_indexer: bool = True, **kwargs):
    self.is_full_indexer = is_full_indexer
    inputs_kv_shape = kwargs.get("inputs_kv_shape")
    super().__init__(*args, **kwargs)

    # In HYV4Attention.__init__, q_a_layernorm and kv_a_layernorm use default eps=1e-6
    # whereas decoder layer norms use config.rms_norm_eps=1e-5.
    self.q_norm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=HY4_MLA_LORA_NORM_EPS,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.kv_norm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=HY4_MLA_LORA_NORM_EPS,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Elementwise sigmoid gate projection (`gate_proj` in HF model / `linear_gate` in safetensors)
    self.gate_proj = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(self.num_query_heads, self.v_head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_attn", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Learnable per-head attention sinks (`sinks` in HF model / `learnable_sink_param` in safetensors)
    self.sinks = nnx.Param(
        default_bias_init(self.rngs.params(), (self.num_query_heads,), jnp.float32),
        out_sharding=(None,),
    )

    # Replace base Indexer with Hy4Indexer on "full" indexer layers; "shared" layers have no indexer.
    if self.config.use_indexer and self.is_full_indexer:
      self.indexer = Hy4Indexer(
          self.config,
          rngs=self.rngs,
          rotary_embedding=self.rotary_embedding,
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
      )
      self.IndexerKVCache_0 = (
          self.init_indexer_cache(inputs_kv_shape)
          if self.model_mode != MODEL_MODE_TRAIN and inputs_kv_shape is not None
          else None
      )
    else:
      self.indexer = None
      self.IndexerKVCache_0 = None

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding=None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      bidirectional_mask: Optional[Any] = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
      prev_indexer_mask: Optional[Array] = None,
  ) -> tuple[Array, Optional[Array], Optional[Array]]:
    del out_sharding, deterministic, slot, rope_kwargs, attention_metadata
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
      out_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
      out_logical_name = (BATCH_ATTN, LENGTH, HEAD, D_KV)

    if model_mode != MODEL_MODE_TRAIN and decoder_segment_ids is None:
      decoder_segment_ids = jnp.ones(inputs_q.shape[:2], dtype=jnp.int32)

    query, low_rank_q = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    indexer_mask = prev_indexer_mask
    if self.config.use_indexer and self.is_full_indexer and self.indexer is not None:
      attention_mask = self.attention_op.generate_attention_mask(
          query,
          key,
          decoder_segment_ids,
          model_mode,
          previous_chunk,
          bidirectional_mask,
          segment_positions=inputs_positions,
      )
      if attention_mask is not None:
        attention_mask = attention_mask.squeeze(axis=(1, 2))
      indexer_mask, _, _ = self.indexer(
          inputs_q=inputs_q,
          low_rank_q=low_rank_q,
          inputs_kv=inputs_kv,
          inputs_positions=inputs_positions,
          attention_mask=attention_mask,
          decoder_segment_ids=decoder_segment_ids,
          previous_chunk=previous_chunk,
          kv_cache=self.IndexerKVCache_0,
          model_mode=model_mode,
      )

    out = self.attention_op(
        query,
        key,
        value,
        decoder_segment_ids,
        inputs_positions,
        model_mode,
        cached_values,
        sinks=self.sinks[...],
        indexer_mask=indexer_mask,
    )

    if self.use_absorbed_mqa:
      out = self.absorb_output(out)

    # Apply elementwise sigmoid gate (`gate_proj`) before `out_projection` (`o_proj`)
    gate_states = self.gate_proj(inputs_q)  # [B, S, H, v_head_dim]
    out = out.astype(self.dtype) * jax.nn.sigmoid(gate_states)

    out = self._maybe_shard_with_logical(out, self.out_axis_names)
    out = checkpoint_name(out, "attention_out")

    out_sharding = create_sharding(self.mesh, out_logical_name)
    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache, indexer_mask


# -----------------------------------------
# Decoder Layer & Scannable Block
# -----------------------------------------


class Hy4DecoderLayer(nnx.Module):
  """HYV4 decoder layer: iHC(RMSNorm + GatedMLA) + iHC(RMSNorm + DenseMLP/MoE)."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | quantizations.AqtQuantization = None,
      layer_idx: int = 0,
      *,
      is_full_indexer: bool | None = None,
      is_moe: bool | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx
    cfg = config

    self.is_full_indexer = is_full_indexer_layer(layer_idx, cfg) if is_full_indexer is None else is_full_indexer
    self.is_moe = is_moe_layer(layer_idx, cfg) if is_moe is None else is_moe

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    length_name = "prefill_activation_norm_length" if model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    self.activation_axis_names = ("activation_batch", length_name, "activation_embed")
    self.stream_axis_names = ("activation_batch", length_name, None, "activation_embed")
    self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
    self.mlp_intermediate_sharding = create_sharding(
        mesh, ("activation_batch", length_name, "activation_mlp"), rules=get_logical_axis_rules()
    )

    norm = functools.partial(
        RMSNorm,
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.pre_self_attention_layer_norm = norm()
    self.post_self_attention_layer_norm = norm()

    self.attn_hc = Hy4HyperConnection(config=cfg, mesh=mesh, rngs=rngs)
    self.ffn_hc = Hy4HyperConnection(config=cfg, mesh=mesh, rngs=rngs)

    self.self_attention = Hy4Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        original_max_position_embeddings=cfg.original_max_position_embeddings,
        mscale=cfg.mscale,
        rope_factor=cfg.rope_factor,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        model_mode=model_mode,
        is_full_indexer=self.is_full_indexer,
        rngs=rngs,
    )

    if self.is_moe:
      self.moe_block = moe.RoutedAndSharedMoE(
          config=cfg,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = linears.MlpBlock(
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          config=cfg,
          quant=quant,
          model_mode=model_mode,
          mesh=mesh,
          rngs=rngs,
      )

  def _shard(self, x: Array) -> Array:
    axes = self.stream_axis_names if x.ndim == 4 else self.activation_axis_names
    return maybe_shard_with_logical(
        x,
        logical_axes=axes,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        rules=get_logical_axis_rules(),
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      prev_indexer_mask=None,
  ):
    del attention_metadata, decoder_input_tokens
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    # Auto-expand 3D inputs [B, S, D] to 4D [B, S, K, D] if called directly in unit tests
    if inputs.ndim == 3:
      inputs = jnp.repeat(jnp.expand_dims(inputs, axis=2), self.config.mhc_expansion_rate, axis=2)

    residual = self._shard(inputs)
    residual = checkpoint_name(residual, "decoder_layer_input")

    # 1. Attention Hyper-Connection + Pre-Norm + Gated MLA
    attn_in, attn_post = self.attn_hc.compute_pre_post(residual)
    lnx = self._shard(self.pre_self_attention_layer_norm(self._shard(attn_in)))
    attn_out, _, indexer_mask = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        prev_indexer_mask=prev_indexer_mask,
    )
    hidden_states = self._shard(self.attn_hc.apply_residual(residual, self._shard(attn_out), attn_post))

    # 2. FFN Hyper-Connection + Post-Norm + Dense MLP / Sparse MoE
    residual = hidden_states
    ffn_in, ffn_post = self.ffn_hc.compute_pre_post(residual)
    hx = self._shard(self.post_self_attention_layer_norm(self._shard(ffn_in)))

    load_balance_loss = None
    moe_bias_updates = None
    if self.is_moe:
      mlp_out, load_balance_loss, moe_bias_updates = self.moe_block(
          hx, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
      )
    else:
      mlp_out = self.mlp(
          hx,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )
    out = self._shard(self.ffn_hc.apply_residual(residual, self._shard(mlp_out), ffn_post))

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    return out, kv_cache, indexer_mask


class Hy4ScannableBlock(nnx.Module):
  """One homogeneous 4-layer MoE cycle: `layers_0` has a full indexer; `layers_1..layers_{cycle-1}` share it."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | quantizations.AqtQuantization = None,
  ):
    self.config = config
    self.num_layers = config.inhomogeneous_layer_cycle_interval
    for j in range(self.num_layers):
      setattr(
          self,
          f"layers_{j}",
          Hy4DecoderLayer(
              config=config,
              model_mode=model_mode,
              mesh=mesh,
              rngs=rngs,
              quant=quant,
              is_full_indexer=(j == 0),
              is_moe=True,
          ),
      )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    del attention_metadata, decoder_input_tokens
    y = inputs[0] if isinstance(inputs, tuple) else inputs
    indexer_mask = None
    for j in range(self.num_layers):
      layer = getattr(self, f"layers_{j}")
      y, _, indexer_mask = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          prev_indexer_mask=indexer_mask,
      )
    return y, kv_cache
