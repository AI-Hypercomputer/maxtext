# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 dense model in m3 format.

This module provides a self-contained implementation of the Qwen3 model
family (dense variant) following the m3 modern MaxText architecture:
- Composed directly from Flax NNX built-ins: nnx.Linear, nnx.Einsum, nnx.RMSNorm, nnx.Embed.
- Completely free of legacy maxtext.layers dependencies (AttentionOp, DenseGeneral,
  Linen RMSNorm, Linen Embed, quantizations).
- Pure NNX module composition with clean parameter paths matching existing checkpoints.
- Uses m3 core RoPE.
- Designed with clear extension points for future MoE and hybrid variants.
"""

import inspect
from typing import Any, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core import kernels
from maxtext.m3.core.rope import apply_rope
from maxtext.m3.core.sharding import DefaultSharding, Sharding


class Qwen3MLP(nnx.Module):
  """Qwen3 feed-forward network with SwiGLU activation.

  Submodule and parameter names match standard checkpoint paths:
  wi_0 (gate), wi_1 (up), wo (down).
  """

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Optional[Sharding] = None,
  ):
    """Initializes Qwen3MLP feed-forward network.

    Args:
      config: Model configuration specifying dimensions and dtypes.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
      sharding: Logical-to-mesh sharding strategy; defaults to `DefaultSharding`.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()

    kw = {
        "use_bias": False,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.wi_0 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        **kw,
    )
    self.wi_1 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        **kw,
    )
    self.wo = nnx.Linear(
        in_features=config.mlp_dim,
        out_features=config.emb_dim,
        **kw,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies gated SwiGLU feed-forward projection: wo(silu(wi_0(x)) * wi_1(x))."""
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    hidden = (gate * up).astype(self.config.dtype)
    return self.wo(hidden)


class Qwen3Attention(nnx.Module):
  """Qwen3 multi-head / grouped-query attention with QK-norm, RoPE, and self-contained attention.

  Submodule and parameter names match checkpoint paths:
  query, key, value, out, query_norm, key_norm.
  """

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Optional[Sharding] = None,
  ):
    """Initializes Qwen3 Attention module with Q, K, V projections and QK-Norm.

    Args:
      config: Model configuration specifying dimensions, head counts, and RoPE settings.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
      sharding: Logical-to-mesh sharding strategy; defaults to `DefaultSharding`.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    self.attention_kernel = kernels.attention_from_config(config, self.sharding)

    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim

    kw = {
        "bias_shape": None,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    # Q, K, V multi-axis projections
    self.query = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_query_heads, self.head_dim),
        **kw,
    )
    self.key = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        **kw,
    )
    self.value = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (self.num_query_heads, self.head_dim, config.emb_dim),
        **kw,
    )

    # QK-Norm
    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.query_norm = nnx.RMSNorm(
        num_features=self.head_dim,
        **norm_kw,
    )
    self.key_norm = nnx.RMSNorm(
        num_features=self.head_dim,
        **norm_kw,
    )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Executes Qwen3 self-attention forward pass with RoPE and configured attention kernel."""
    is_default_positions = decoder_positions is None
    if decoder_positions is None:
      decoder_positions = jnp.broadcast_to(jnp.arange(inputs.shape[1], dtype=jnp.int32)[None, :], inputs.shape[:2])
    query = self.query(inputs)
    key = self.key(inputs)
    value = self.value(inputs)

    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)

    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_max_timescale)

    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar

    attn_out = self.attention_kernel(
        query,
        key,
        value,
        segment_ids=decoder_segment_ids,
        positions=None if (is_default_positions or decoder_segment_ids is None) else decoder_positions,
    )
    return self.out(attn_out.astype(self.config.dtype))


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (dense).

  Pre-norm architecture with self-attention and MLP residual blocks.
  Leaves clean extension points for MoE or hybrid variants.
  """

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Optional[Sharding] = None,
  ):
    """Initializes pre-norm attention and MLP residual blocks.

    Args:
      config: Model configuration specifying dimensions, layers, and dtypes.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
      sharding: Logical-to-mesh sharding strategy; defaults to `DefaultSharding`.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()

    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        **norm_kw,
    )
    self.self_attention = Qwen3Attention(
        config=config,
        rngs=rngs,
        sharding=self.sharding,
    )
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        **norm_kw,
    )
    self.mlp = Qwen3MLP(
        config=config,
        rngs=rngs,
        sharding=self.sharding,
    )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Applies pre-norm attention and pre-norm MLP blocks with residual connections."""
    # Pre-attention norm & Self-attention
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out = self.self_attention(
        normed_attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    x = (inputs + attn_out).astype(self.config.dtype)

    # Post-attention norm & MLP
    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.config.dtype)

    return output


class Qwen3Decoder(nnx.Module):
  """Qwen3 decoder stack with an `nnx.Sequential` of decoder layers."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      layer_cls: type[nnx.Module] = Qwen3DecoderLayer,
      sharding: Optional[Sharding] = None,
  ):
    """Initializes the Qwen3 decoder stack of N layers and final layer normalization.

    Args:
      config: Model configuration specifying layer count, dimensions, and dtypes.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
      layer_cls: Decoder layer constructor for model-family subclasses.
      sharding: Logical-to-mesh sharding strategy; defaults to `DefaultSharding`.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    params = inspect.signature(layer_cls.__init__).parameters
    layer_kw = (
        {"sharding": self.sharding}
        if ("sharding" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()))
        else {}
    )

    self.layers = nnx.Sequential(
        *(layer_cls(config=config, rngs=rngs, **layer_kw) for _ in range(config.num_decoder_layers))
    )

    self.decoder_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        rngs=rngs,
    )

  def apply_output_head(
      self,
      shared_embedding: Any,
      y: jax.Array,
      deterministic: bool = False,
  ) -> jax.Array:
    """Applies final decoder norm and projects hidden states to logits via tied embedding."""
    y = self.decoder_norm(y)
    attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype
    logits = jnp.dot(y.astype(attend_dtype), shared_embedding.embedding[...].astype(attend_dtype).T)

    if self.config.normalize_embedding_logits:
      logits = logits / jnp.sqrt(y.shape[-1])
    if self.config.final_logits_soft_cap:
      logits = logits / self.config.final_logits_soft_cap
      logits = jnp.tanh(logits) * self.config.final_logits_soft_cap
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  def __call__(
      self,
      shared_embedding: Any,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> tuple[jax.Array, jax.Array]:
    """Iterates through all decoder layers and projects to logits via tied embeddings."""
    y = shared_embedding(decoder_input_tokens)

    y = self.layers(
        y,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        deterministic=deterministic,
    )

    logits = self.apply_output_head(shared_embedding, y, deterministic=deterministic)
    return logits, y


class Qwen3Model(nnx.Module):
  """Top-level Qwen3 autoregressive language model in m3 format."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Optional[Sharding] = None,
      **kwargs,
  ):
    """Initializes the top-level Qwen3 language model.

    Args:
      config: Model configuration specifying dimensions, layers, and vocabulary.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
      sharding: Logical-to-mesh sharding strategy; defaults to `DefaultSharding`.
      **kwargs: Additional keyword arguments.
    """
    self.config = config
    self.mesh = mesh
    self.sharding = sharding or DefaultSharding()

    self.token_embedder = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(
        config=config,
        rngs=rngs,
        sharding=self.sharding,
    )

  def logits_from_hidden_states_for_vocab_tiling(
      self, hidden_states: jax.Array, deterministic: bool, **kwargs
  ) -> jax.Array:
    """Projects hidden states to logits for vocab-tiled computation."""
    return self.decoder.apply_output_head(
        shared_embedding=self.token_embedder,
        y=hidden_states,
        deterministic=deterministic,
    )

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray] = None,
      cache: Any = None,
      enable_dropout: bool = False,
      true_length: Optional[int] = None,
      **kwargs,
  ) -> jax.Array:
    """Executes full Qwen3 forward pass from input token IDs to output logits."""
    deterministic = kwargs.get("deterministic", not enable_dropout)
    logits, _ = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    return logits


def create_qwen3_model(
    config: Config,
    mesh: Optional[Mesh] = None,
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Qwen3Model:
  """Factory function creating a Qwen3Model instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3Model(config, mesh, rngs=rngs, **kwargs)
