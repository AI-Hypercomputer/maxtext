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
- Self-contained module composition with no dependencies on shared `maxtext.layers` blocks.
- Pure NNX module composition with `nnx.Sequential` layer stacks and `DefaultSharding` annotations.
- Uses m3 core RoPE and returns `CausalLMOutput` when `return_output=True` or `return_logits=False`.
"""

# pylint: disable=arguments-differ

import inspect
from typing import Any, Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.core.rope import apply_rope
from maxtext.m3.core.sharding import DefaultSharding, MESH_AXIS_NAMES, Sharding


def _resolve_param_metadata(spec: P, mesh: Optional[Mesh]) -> dict[str, Any]:
  """Resolves `out_sharding` metadata against the ambient mesh (or `mesh` if provided).

  Under an m3 mesh (or any mesh containing all `MESH_AXIS_NAMES`), returns `{"out_sharding": spec}`
  verbatim. Under partial test meshes (such as single-axis `Mesh(devices, ("data",))`), projects
  axes absent from the active mesh to `None` so eager NNX sharding succeeds cleanly. When no mesh
  is active and `mesh` is `None`, omits `out_sharding`.
  """
  ambient = jax.sharding.get_abstract_mesh()
  if not ambient.empty:
    active_axes = set(ambient.axis_names)
  elif mesh is not None:
    active_axes = set(mesh.axis_names)
  else:
    return {}

  if set(MESH_AXIS_NAMES).issubset(active_axes):
    return {"out_sharding": spec}

  projected = []
  for entry in spec:
    if entry is None:
      projected.append(None)
    elif isinstance(entry, str):
      projected.append(entry if entry in active_axes else None)
    else:
      kept = tuple(ax for ax in entry if ax in active_axes)
      projected.append(kept[0] if len(kept) == 1 else (kept if kept else None))
  return {"out_sharding": P(*projected)}


class Qwen3MLP(nnx.Module):
  """Qwen3 feed-forward network with SwiGLU activation.

  Submodule and parameter names match standard checkpoint paths:
  wi_0 (gate), wi_1 (up), wo (down).
  """

  def __init__(
      self,
      config: Any,
      mesh: Optional[Mesh] = None,
      *,
      sharding: Optional[Sharding] = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    s = sharding or DefaultSharding()

    kw = {
        "use_bias": False,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.wi_0 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_metadata=_resolve_param_metadata(s("wi_0", ("embed", "mlp")), mesh),
        **kw,
    )
    self.wi_1 = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_metadata=_resolve_param_metadata(s("wi_1", ("embed", "mlp")), mesh),
        **kw,
    )
    self.wo = nnx.Linear(
        in_features=config.mlp_dim,
        out_features=config.emb_dim,
        kernel_metadata=_resolve_param_metadata(s("wo", ("mlp", "embed")), mesh),
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
      config: Any,
      mesh: Optional[Mesh] = None,
      *,
      sharding: Optional[Sharding] = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    s = sharding or DefaultSharding()

    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim

    kw = {
        "bias_shape": None,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.query = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_query_heads, self.head_dim),
        kernel_metadata=_resolve_param_metadata(s("query", ("embed", "heads", "head_dim")), mesh),
        **kw,
    )
    self.key = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        kernel_metadata=_resolve_param_metadata(s("key", ("embed", "kv_heads", "head_dim")), mesh),
        **kw,
    )
    self.value = nnx.Einsum(
        "...e,ehd->...hd",
        (config.emb_dim, self.num_kv_heads, self.head_dim),
        kernel_metadata=_resolve_param_metadata(s("value", ("embed", "kv_heads", "head_dim")), mesh),
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (self.num_query_heads, self.head_dim, config.emb_dim),
        kernel_metadata=_resolve_param_metadata(s("out", ("heads", "head_dim", "embed")), mesh),
        **kw,
    )

    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "scale_metadata": _resolve_param_metadata(s("qk_norm", ("norm",)), mesh),
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

  def _compute_attention(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Computes self-contained scaled dot-product attention with GQA and causal segment masking."""
    b, t, n_q, d = query.shape
    n_kv = key.shape[2]
    g = n_q // n_kv
    query_g = jnp.reshape(query, (b, t, n_kv, g, d))

    def _compute_local_block(q_g, k, v, mask=None):
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
    if q_pos.ndim == 3:
      q_pos = q_pos[..., 0]
      k_pos = k_pos[..., 0]
    causal_mask = k_pos[..., None, :] <= q_pos[..., :, None]
    if decoder_segment_ids is not None:
      seg_mask = (decoder_segment_ids[..., :, None] == decoder_segment_ids[..., None, :]) & (
          decoder_segment_ids[..., :, None] != 0
      )
      mask = causal_mask & seg_mask
    else:
      mask = causal_mask
    mask = mask[:, None, None, :, :]
    out, _, lsum = _compute_local_block(query_g, key, value, mask)
    return (out / jnp.maximum(lsum, 1e-12)).astype(self.config.dtype)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    """Executes Qwen3 self-attention forward pass with QK-Norm and RoPE."""
    query = jax.ad_checkpoint.checkpoint_name(self.query(inputs), "query_proj")
    key = jax.ad_checkpoint.checkpoint_name(self.key(inputs), "key_proj")
    value = jax.ad_checkpoint.checkpoint_name(self.value(inputs), "value_proj")

    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)

    if decoder_positions is None:
      rope_pos = jnp.broadcast_to(
          jnp.arange(inputs.shape[1], dtype=jnp.int32)[None, :],
          inputs.shape[:2],
      )
    elif decoder_positions.ndim == 3:
      rope_pos = decoder_positions[..., 0]
    else:
      rope_pos = decoder_positions
    query = apply_rope(query, rope_pos, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, rope_pos, max_timescale=self.config.rope_max_timescale)

    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar

    attn_out = self._compute_attention(
        query=query,
        key=key,
        value=value,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return jax.ad_checkpoint.checkpoint_name(self.out(attn_out), "out_proj")


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Any,
      mesh: Optional[Mesh] = None,
      *,
      sharding: Optional[Sharding] = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    s = sharding or DefaultSharding()

    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "scale_metadata": _resolve_param_metadata(s("layer_norm", ("norm",)), mesh),
        "rngs": rngs,
    }
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        **norm_kw,
    )
    self.self_attention = Qwen3Attention(
        config=config,
        mesh=mesh,
        sharding=s,
        rngs=rngs,
    )
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        **norm_kw,
    )
    self.mlp = Qwen3MLP(
        config=config,
        mesh=mesh,
        sharding=s,
        rngs=rngs,
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
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out = self.self_attention(
        normed_attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    x = (inputs + attn_out).astype(self.config.dtype)

    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.config.dtype)
    return output


class Qwen3Decoder(nnx.Module):
  """Qwen3 decoder stack with an `nnx.Sequential` of decoder layers."""

  def __init__(
      self,
      config: Any,
      mesh: Optional[Mesh] = None,
      *,
      sharding: Optional[Sharding] = None,
      rngs: nnx.Rngs,
      layer_cls: type[nnx.Module] = Qwen3DecoderLayer,
  ):
    self.config = config
    self.num_layers = config.num_decoder_layers
    s = sharding or DefaultSharding()

    accepts_sharding = "sharding" in inspect.signature(layer_cls).parameters
    layer_kw: dict[str, Any] = {"sharding": s} if accepts_sharding else {}
    self.layers = nnx.Sequential(
        *(layer_cls(config=config, mesh=mesh, rngs=rngs, **layer_kw) for _ in range(config.num_decoder_layers))
    )

    self.decoder_norm = nnx.RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        scale_metadata=_resolve_param_metadata(s("decoder_norm", ("norm",)), mesh),
        rngs=rngs,
    )
    if not getattr(config, "logits_via_embedding", True):
      self.logits_dense = nnx.Linear(
          in_features=config.emb_dim,
          out_features=config.vocab_size,
          use_bias=False,
          dtype=config.dtype,
          param_dtype=config.weight_dtype,
          kernel_metadata=_resolve_param_metadata(s("logits_dense", ("embed", "vocab")), mesh),
          rngs=rngs,
      )

  def apply_output_head(
      self,
      shared_embedding: Any,
      y: jax.Array,
      deterministic: bool = False,
  ) -> jax.Array:
    """Applies final decoder norm and projects hidden states to vocabulary logits."""
    y = self.decoder_norm(y)
    attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype
    if hasattr(self, "logits_dense"):
      kernel = self.logits_dense.kernel[...]
      logits = jnp.dot(y.astype(attend_dtype), kernel.astype(attend_dtype))
    else:
      table = (
          shared_embedding.embedding[...]
          if hasattr(shared_embedding, "embedding")
          else getattr(shared_embedding, "value", shared_embedding)
      )
      logits = jnp.dot(y.astype(attend_dtype), jnp.asarray(table, attend_dtype).T)

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
      *,
      apply_output_head: bool = True,
      **kwargs,
  ) -> tuple[jax.Array | None, jax.Array]:
    """Runs decoder layers and optionally projects hidden states to logits."""
    y = shared_embedding(decoder_input_tokens)
    if isinstance(self.layers, nnx.Sequential):
      for layer in self.layers.layers:
        y = layer(
            y,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            deterministic=deterministic,
        )
    else:
      y = self.layers(
          y,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
      )
    if not apply_output_head:
      return None, y
    logits = self.apply_output_head(shared_embedding, y, deterministic=deterministic)
    return logits, y


class Qwen3Model(nnx.Module):
  """Top-level Qwen3 autoregressive language model in m3 format."""

  def __init__(
      self,
      config: Any,
      mesh: Optional[Mesh] = None,
      *,
      sharding: Optional[Sharding] = None,
      rngs: nnx.Rngs,
      **kwargs,
  ):
    self.config = config
    s = sharding or DefaultSharding()

    self.token_embedder = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata=_resolve_param_metadata(s("token_embedder", ("vocab", "embed")), mesh),
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(
        config=config,
        mesh=mesh,
        sharding=s,
        rngs=rngs,
    )

  def logits_from_hidden_states_for_vocab_tiling(
      self, hidden_states: jax.Array, deterministic: bool = True, **kwargs
  ) -> jax.Array:
    """Projects hidden states to logits for chunked/tiled computation."""
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
      *,
      return_output: bool = False,
      return_logits: bool = True,
      **kwargs,
  ) -> jax.Array | CausalLMOutput:
    """Executes Qwen3 forward pass; returns `CausalLMOutput` when `return_output=True` or `not return_logits`."""
    deterministic = kwargs.get("deterministic", not enable_dropout)
    logits, hidden_states = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        apply_output_head=return_logits,
    )
    if return_output or not return_logits:
      return CausalLMOutput(
          logits=logits,
          hidden_states=hidden_states,
          aux_loss=None,
          router_token_counts=None,
      )
    assert logits is not None
    return logits


def create_qwen3_model(
    config: Any,
    mesh: Optional[Mesh] = None,
    *,
    sharding: Optional[Sharding] = None,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Qwen3Model:
  """Factory function creating a `Qwen3Model` instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3Model(config, mesh, sharding=sharding, rngs=rngs, **kwargs)
