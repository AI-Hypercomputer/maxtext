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
from maxtext.m3.core import kernels
from maxtext.m3.core.rope import apply_rope
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.core.sharding import (
    Sharding,
    check_mesh,
    end_eager_placement,
    enter_mesh_scope,
    param_metadata,
    sharding_from_config,
)
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config


def _jnp_dtype(dtype: Any) -> jnp.dtype:
  """Converts a `DType` enum, string, or JAX/NumPy dtype to `jnp.dtype`."""
  return jnp.dtype(getattr(dtype, "value", dtype))


def norm_kwargs(config: Config | Qwen3Config, sharding: Sharding, name: str, rngs: nnx.Rngs) -> dict[str, Any]:
  """Keyword arguments shared by every `nnx.RMSNorm` in the model, with the scale's sharding."""
  cfg = Qwen3Config.from_config(config)
  return {
      "epsilon": cfg.rms_norm_eps,
      "dtype": _jnp_dtype(cfg.compute_dtype),
      "param_dtype": _jnp_dtype(cfg.param_dtype),
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
      config: Config | Qwen3Config,
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
    cfg = Qwen3Config.from_config(config)
    self.config = cfg
    self.dtype = _jnp_dtype(cfg.compute_dtype)
    self.sharding = sharding or sharding_from_config(config)
    s = self.sharding

    kw = {
        "use_bias": False,
        "dtype": self.dtype,
        "param_dtype": _jnp_dtype(cfg.param_dtype),
        "rngs": rngs,
    }
    self.wi_0 = nnx.Linear(
        in_features=cfg.hidden_size,
        out_features=cfg.intermediate_size,
        kernel_metadata=param_metadata(s, "wi_0", ("embed", "mlp")),
        **kw,
    )
    self.wi_1 = nnx.Linear(
        in_features=cfg.hidden_size,
        out_features=cfg.intermediate_size,
        kernel_metadata=param_metadata(s, "wi_1", ("embed", "mlp")),
        **kw,
    )
    self.wo = nnx.Linear(
        in_features=cfg.intermediate_size,
        out_features=cfg.hidden_size,
        kernel_metadata=param_metadata(s, "wo", ("mlp", "embed")),
        **kw,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies gated SwiGLU feed-forward projection: wo(silu(wi_0(x)) * wi_1(x))."""
    s = self.sharding
    x = s.constrain(x, "mlp_in", ("batch", "length", "embed"))
    gate = jax.nn.silu(s.project(self.wi_0, x, "wi_0_out", ("batch", "length", "mlp")))
    up = s.project(self.wi_1, x, "wi_1_out", ("batch", "length", "mlp"))
    hidden = (gate * up).astype(self.dtype)
    hidden = s.constrain(hidden, "mlp_hidden", ("batch", "length", "mlp"))
    return s.project(self.wo, hidden, "mlp_out", ("batch", "norm_length", "embed"))


class Qwen3Attention(nnx.Module):
  """Qwen3 multi-head / grouped-query attention with QK-norm, RoPE, and self-contained attention.

  Submodule and parameter names match checkpoint paths:
  query, key, value, out, query_norm, key_norm.
  """

  def __init__(
      self,
      config: Config | Qwen3Config,
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
    entered_mesh = mesh is not None and jax.sharding.get_abstract_mesh().empty
    with enter_mesh_scope(mesh):
      cfg = Qwen3Config.from_config(config)
      self.config = cfg
      self.dtype = _jnp_dtype(cfg.compute_dtype)
      self.sharding = sharding or sharding_from_config(config)
      s = self.sharding
      self.attention_kernel = kernels.attention_from_config(config, s, mesh=mesh)

      self.num_query_heads = cfg.num_attention_heads
      self.num_kv_heads = cfg.num_key_value_heads
      self.head_dim = cfg.head_dim

      kw = {
          "bias_shape": None,
          "dtype": self.dtype,
          "param_dtype": _jnp_dtype(cfg.param_dtype),
          "rngs": rngs,
      }
      # Q, K, V multi-axis projections
      self.query = nnx.Einsum(
          "...e,ehd->...hd",
          (cfg.hidden_size, self.num_query_heads, self.head_dim),
          kernel_metadata=param_metadata(s, "query", ("embed", "heads", "head_dim")),
          **kw,
      )
      self.key = nnx.Einsum(
          "...e,ehd->...hd",
          (cfg.hidden_size, self.num_kv_heads, self.head_dim),
          kernel_metadata=param_metadata(s, "key", ("embed", "kv_heads", "head_dim")),
          **kw,
      )
      self.value = nnx.Einsum(
          "...e,ehd->...hd",
          (cfg.hidden_size, self.num_kv_heads, self.head_dim),
          kernel_metadata=param_metadata(s, "value", ("embed", "kv_heads", "head_dim")),
          **kw,
      )
      self.out = nnx.Einsum(
          "...hd,hde->...e",
          (self.num_query_heads, self.head_dim, cfg.hidden_size),
          kernel_metadata=param_metadata(s, "out", ("heads", "head_dim", "embed")),
          **kw,
      )

      # QK-Norm
      self.query_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kwargs(cfg, s, "query_norm", rngs))
      self.key_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kwargs(cfg, s, "key_norm", rngs))
    if entered_mesh:
      end_eager_placement(self)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Executes Qwen3 self-attention forward pass with RoPE and configured attention kernel."""
    s = self.sharding
    is_default_positions = decoder_positions is None
    inputs = s.constrain(inputs, "attn_in", ("batch", "length", "embed"))
    query = s.project(self.query, inputs, "query", ("batch", "length", "heads", "head_dim"))
    key = s.project(self.key, inputs, "key", ("batch", "length", "kv_heads", "head_dim"))
    value = s.project(self.value, inputs, "value", ("batch", "length", "kv_heads", "head_dim"))

    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)

    # RoPE
    if decoder_positions is None:
      b, t = query.shape[:2]
      decoder_positions = s.constrain(
          jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t)),
          "decoder_positions",
          ("batch", "length"),
      )
    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_theta)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_theta)

    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar

    # Only pass positions for attention masking when packed (`decoder_segment_ids` present) or
    # decoding (`q_len != kv_len`), keeping unpacked training on the `is_causal=True` fast path.
    use_mask_positions = not is_default_positions and (decoder_segment_ids is not None or query.shape[1] != key.shape[1])
    attn_out = self.attention_kernel(
        query,
        key,
        value,
        segment_ids=decoder_segment_ids,
        positions=decoder_positions if use_mask_positions else None,
    )
    attn_out = s.constrain(
        attn_out.astype(self.dtype),
        "attention_out",
        ("batch", "length", "heads", "head_dim"),
    )
    return s.project(self.out, attn_out, "attn_proj", ("batch", "norm_length", "embed"))


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (dense).

  Pre-norm architecture with self-attention and MLP residual blocks.
  Leaves clean extension points for MoE or hybrid variants.
  """

  def __init__(
      self,
      config: Config | Qwen3Config,
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
    cfg = Qwen3Config.from_config(config)
    self.config = cfg
    self.dtype = _jnp_dtype(cfg.compute_dtype)
    self.sharding = sharding or sharding_from_config(config)
    s = self.sharding

    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        num_features=cfg.hidden_size, **norm_kwargs(cfg, s, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = Qwen3Attention(config=config, mesh=mesh, rngs=rngs, sharding=s)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        num_features=cfg.hidden_size, **norm_kwargs(cfg, s, "post_self_attention_layer_norm", rngs)
    )
    self.mlp = Qwen3MLP(config=cfg, rngs=rngs, sharding=s)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Applies pre-norm attention and pre-norm MLP blocks with residual connections."""
    inputs = self.sharding.constrain(inputs, "decoder_layer_input", ("batch", "norm_length", "embed"))
    # Pre-attention norm & Self-attention
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out = self.self_attention(
        normed_attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    x = (inputs + attn_out).astype(self.dtype)

    # Post-attention norm & MLP
    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.dtype)

    return output


class Qwen3Decoder(nnx.Module):
  """Qwen3 decoder stack with an `nnx.Sequential` of decoder layers."""

  def __init__(
      self,
      config: Config | Qwen3Config,
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
    entered_mesh = mesh is not None and jax.sharding.get_abstract_mesh().empty
    with enter_mesh_scope(mesh):
      cfg = Qwen3Config.from_config(config)
      self.config = cfg
      self.dtype = _jnp_dtype(cfg.compute_dtype)
      self.sharding = sharding or sharding_from_config(config)
      s = self.sharding

      self.layers = nnx.Sequential(
          *(layer_cls(config=config, mesh=mesh, rngs=rngs, sharding=s) for _ in range(cfg.num_hidden_layers))
      )
      self.decoder_norm = nnx.RMSNorm(num_features=cfg.hidden_size, **norm_kwargs(cfg, s, "decoder_norm", rngs))
    if entered_mesh:
      end_eager_placement(self)

  def apply_output_head(
      self,
      shared_embedding: Any,
      y: jax.Array,
      deterministic: bool = False,
  ) -> jax.Array:
    """Applies final decoder norm and projects hidden states to logits via tied embedding."""
    s = self.sharding
    norm_axes = ("batch", "norm_length", "embed") if y.ndim == 3 else ("batch", "embed")
    pre_logits_axes = ("batch", "length", "embed") if y.ndim == 3 else ("batch", "embed")
    y = s.constrain(y, "pre_logits_norm", norm_axes)
    y = self.decoder_norm(y)
    y = s.constrain(y, "pre_logits", pre_logits_axes)
    attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.dtype
    # Vocab tiling (`logits_from_hidden_states_for_vocab_tiling`) passes `(tile, emb)` chunks.
    logits_axes = ("batch", "length", "vocab") if y.ndim == 3 else ("batch", "vocab")
    logits = jnp.dot(
        y.astype(attend_dtype),
        shared_embedding.embedding[...].astype(attend_dtype).T,
        out_sharding=s.out_sharding("logits", logits_axes),
    )
    logits = s.constrain(logits, "logits", logits_axes)

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
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> tuple[jax.Array, jax.Array]:
    """Iterates through all decoder layers and projects to logits via tied embeddings."""
    s = self.sharding
    decoder_input_tokens = s.constrain(decoder_input_tokens, "decoder_input_tokens", ("batch", "length"))
    if decoder_positions is not None:
      decoder_positions = s.constrain(decoder_positions, "decoder_positions", ("batch", "length"))
    if decoder_segment_ids is not None:
      decoder_segment_ids = s.constrain(decoder_segment_ids, "decoder_segment_ids", ("batch", "length"))
    y = s.project(shared_embedding, decoder_input_tokens, "embedded_inputs", ("batch", "norm_length", "embed"))

    y = self.layers(
        y,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        deterministic=deterministic,
    )

    logits = self.apply_output_head(shared_embedding, y, deterministic=deterministic)
    return logits, y


def check_construction_mesh() -> None:
  """Refuses a mesh in scope that the model's `Sharding` cannot use (`core.sharding.check_mesh`).

  Drivers other than the m3 loop hand the model their own mesh; this is where
  the strategy first meets it, before any parameter is placed.
  """
  mesh = jax.sharding.get_abstract_mesh()
  if not mesh.empty:
    check_mesh(mesh, m3_sharding.DEFAULT_SHARDING_AXES)


class Qwen3Model(nnx.Module):
  """Top-level Qwen3 autoregressive language model in m3 format."""

  def __init__(
      self,
      config: Config | Qwen3Config,
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
    check_construction_mesh()
    cfg = Qwen3Config.from_config(config)
    self.config = cfg
    self.mesh = mesh
    self.sharding = sharding or sharding_from_config(config)

    self.token_embedder = nnx.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.hidden_size,
        dtype=_jnp_dtype(cfg.compute_dtype),
        param_dtype=_jnp_dtype(cfg.param_dtype),
        embedding_metadata=param_metadata(self.sharding, "token_embedder", ("vocab", "embed")),
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(
        config=config,
        mesh=mesh,
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
      decoder_positions: Optional[jnp.ndarray] = None,
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
    config: Config | Qwen3Config,
    mesh: Optional[Mesh] = None,
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Qwen3Model:
  """Factory function creating a Qwen3Model instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3Model(config, mesh, rngs=rngs, **kwargs)
