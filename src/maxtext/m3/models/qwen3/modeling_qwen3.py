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

Sharding is declared here once, in logical axis names, and resolved by the
`Sharding` instance every block receives at construction (`DefaultSharding`
unless the driver passes another strategy). Each parameter carries
`out_sharding` metadata, so Flax places it under the mesh in scope the moment
it is created; a few activations carry `constrain` calls at block boundaries.
Nothing here names a mesh axis.
"""

from typing import Any, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core.rope import apply_rope
from maxtext.m3.core.sharding import DefaultSharding, Sharding, param_metadata


def norm_kwargs(config: Config, sharding: Sharding, name: str, rngs: nnx.Rngs) -> dict[str, Any]:
  """Keyword arguments shared by every `nnx.RMSNorm` in the model, with the scale's sharding."""
  return {
      "epsilon": config.normalization_layer_epsilon,
      "dtype": config.dtype,
      "param_dtype": config.weight_dtype,
      "scale_metadata": param_metadata(sharding, name, ("norm",)),
      "rngs": rngs,
  }


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
      sharding: Resolves logical axis names to the mesh; `DefaultSharding` if omitted.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    s = self.sharding

    kw = {
        "use_bias": False,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.wi_0 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_metadata=param_metadata(s, "wi_0", ("embed", "mlp")),
        **kw,
    )
    self.wi_1 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_metadata=param_metadata(s, "wi_1", ("embed", "mlp")),
        **kw,
    )
    self.wo = nnx.Linear(
        in_features=config.mlp_dim,
        out_features=config.emb_dim,
        kernel_metadata=param_metadata(s, "wo", ("mlp", "embed")),
        **kw,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies gated SwiGLU feed-forward projection: wo(silu(wi_0(x)) * wi_1(x))."""
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    hidden = (gate * up).astype(self.config.dtype)
    hidden = self.sharding.constrain(hidden, "mlp_hidden", ("batch", "length", "mlp"))
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
      sharding: Resolves logical axis names to the mesh; `DefaultSharding` if omitted.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    s = self.sharding

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
        kernel_metadata=param_metadata(s, "query", ("embed", "heads", "head_dim")),
        **kw,
    )
    self.key = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        kernel_metadata=param_metadata(s, "key", ("embed", "kv_heads", "head_dim")),
        **kw,
    )
    self.value = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        kernel_metadata=param_metadata(s, "value", ("embed", "kv_heads", "head_dim")),
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (self.num_query_heads, self.head_dim, config.emb_dim),
        kernel_metadata=param_metadata(s, "out", ("heads", "head_dim", "embed")),
        **kw,
    )

    # QK-Norm
    self.query_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kwargs(config, s, "query_norm", rngs))
    self.key_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kwargs(config, s, "key_norm", rngs))

  def _compute_attention(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Computes self-contained scaled dot-product attention with GQA and causal masking."""
    b, t, n_q, d = query.shape
    n_kv = key.shape[2]
    g = n_q // n_kv
    query_g = jnp.reshape(query, (b, t, n_kv, g, d))

    def _compute_local_block(q_g, k, v, mask=None):
      """Computes local attention output, max, and sum for scaled dot-product block."""
      weights = jnp.einsum("btkgd,bskd->bkgts", q_g, k)
      if mask is not None:
        weights = jnp.where(mask, weights, -1e10)
      if self.config.float32_logits:
        weights = weights.astype(jnp.float32)
      logits = jnp.reshape(weights, (b, n_q, t, -1))
      local_max = jnp.max(logits, axis=-1, keepdims=True)
      local_exp = jnp.exp(logits - local_max)
      local_sum = jnp.sum(local_exp, axis=-1, keepdims=True)

      local_exp = jnp.reshape(local_exp.astype(v.dtype), (b, n_kv, g, t, -1))
      local_out = jnp.einsum("bkgts,bskd->btkgd", local_exp, v)
      local_out = jnp.reshape(local_out, (b, t, n_q, d))

      local_max = jnp.transpose(local_max, (0, 2, 1, 3))
      local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))
      return local_out, local_max, local_sum

    s = key.shape[1]
    q_pos = jnp.arange(t) if decoder_positions is None else decoder_positions
    k_pos = jnp.arange(s) if decoder_positions is None else decoder_positions
    causal_mask = k_pos[..., None, :] <= q_pos[..., :, None]
    if decoder_segment_ids is not None:
      seg_mask = decoder_segment_ids[..., :, None] == decoder_segment_ids[..., None, :]
      mask = causal_mask & seg_mask
    else:
      mask = causal_mask
    mask = mask[:, None, None, :, :]
    out, _, lsum = _compute_local_block(query_g, key, value, mask)
    return (out / lsum).astype(self.config.dtype)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Executes Qwen3 self-attention forward pass with RoPE."""
    s = self.sharding
    query = s.constrain(self.query(inputs), "query", ("batch", "length", "heads", "head_dim"))
    key = s.constrain(self.key(inputs), "key", ("batch", "length", "kv_heads", "head_dim"))
    value = s.constrain(self.value(inputs), "value", ("batch", "length", "kv_heads", "head_dim"))

    # QK-Norm before RoPE
    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)

    # RoPE
    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_max_timescale)

    # Scale query by 1/sqrt(head_dim)
    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar

    attn_out = self._compute_attention(
        query=query,
        key=key,
        value=value,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    attn_out = s.constrain(attn_out, "attention_out", ("batch", "length", "heads", "head_dim"))
    return self.out(attn_out)


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
      sharding: Resolves logical axis names to the mesh; `DefaultSharding` if omitted.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    s = self.sharding

    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim, **norm_kwargs(config, s, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = Qwen3Attention(config=config, rngs=rngs, sharding=s)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim, **norm_kwargs(config, s, "post_self_attention_layer_norm", rngs)
    )
    self.mlp = Qwen3MLP(config=config, rngs=rngs, sharding=s)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Applies pre-norm attention and pre-norm MLP blocks with residual connections."""
    inputs = self.sharding.constrain(inputs, "decoder_layer_input", ("batch", "length", "embed"))
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
      sharding: Resolves logical axis names to the mesh; `DefaultSharding` if omitted.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()
    s = self.sharding

    self.layers = nnx.Sequential(
        *(layer_cls(config=config, rngs=rngs, sharding=s) for _ in range(config.num_decoder_layers))
    )
    self.decoder_norm = nnx.RMSNorm(num_features=config.emb_dim, **norm_kwargs(config, s, "decoder_norm", rngs))

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
    logits = self.sharding.constrain(logits, "logits", ("batch", "length", "vocab"))

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
      sharding: Resolves logical axis names to the mesh; `DefaultSharding` if omitted.
        Parameters are placed from it at construction, under the mesh in scope.
      **kwargs: Additional keyword arguments.
    """
    self.config = config
    self.sharding = sharding or DefaultSharding()

    self.token_embedder = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata=param_metadata(self.sharding, "token_embedder", ("vocab", "embed")),
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
