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

Composed from NNX built-ins (`nnx.Linear`, `nnx.Einsum`, `nnx.RMSNorm`,
`nnx.Embed`) plus m3's RoPE. Parameter names match the legacy MaxText Qwen3
checkpoint layout, so a checkpoint loads into either stack; the one path
difference is that repeated layers live in a `Layers` container
(`decoder/layers/<i>/...` here, `decoder/layers_<i>/...` in unscanned legacy).

Everything that is not the architecture is absent on purpose. Sharding is a
`Sharding` instance resolving logical axis names; remat, scan, quantization
and offload are applied afterwards by `maxtext.m3.infra`. The only traces
they leave here are `checkpoint_name` tags and `constrain` calls.

Qwen3-MoE and Qwen3-VL build on the attention and MLP defined here.
"""

from typing import Any

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core import kernels
from maxtext.m3.core.attention import dot_product_attention
from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.core.layers import Layers
from maxtext.m3.core.rope import apply_rope
from maxtext.m3.core.sharding import DefaultSharding, Sharding


def norm_kwargs(config: Config, sharding: Sharding, name: str, rngs: nnx.Rngs) -> dict[str, Any]:
  """Keyword arguments shared by every `nnx.RMSNorm` in the model."""
  return {
      "epsilon": config.normalization_layer_epsilon,
      "dtype": config.dtype,
      "param_dtype": config.weight_dtype,
      "scale_metadata": {"out_sharding": sharding(name, ("norm",))},
      "rngs": rngs,
  }


class Qwen3MLP(nnx.Module):
  """SwiGLU feed-forward block: `wo(silu(wi_0(x)) * wi_1(x))`."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    kw = {"use_bias": False, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    self.wi_0 = nnx.Linear(
        config.emb_dim, config.mlp_dim, kernel_metadata={"out_sharding": sharding("wi_0", ("embed", "mlp"))}, **kw
    )
    self.wi_1 = nnx.Linear(
        config.emb_dim, config.mlp_dim, kernel_metadata={"out_sharding": sharding("wi_1", ("embed", "mlp"))}, **kw
    )
    self.wo = nnx.Linear(
        config.mlp_dim, config.emb_dim, kernel_metadata={"out_sharding": sharding("wo", ("mlp", "embed"))}, **kw
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    s = self.sharding
    gate = checkpoint_name(self.wi_0(x), "mlpwi_0")
    up = checkpoint_name(self.wi_1(x), "mlpwi_1")
    hidden = (jax.nn.silu(gate) * up).astype(self.config.dtype)
    hidden = s.constrain(hidden, "mlp_hidden", ("batch", "length", "mlp"))
    return checkpoint_name(self.wo(hidden), "mlpwo")


class Qwen3Attention(nnx.Module):
  """Grouped-query attention with QK-norm and RoPE.

  Submodule names (`query`, `key`, `value`, `out`, `query_norm`, `key_norm`)
  match the checkpoint layout.
  """

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.attention_kernel = kernels.attention_from_config(config, sharding)

    kw = {"bias_shape": None, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    q_shape = (config.emb_dim, self.num_query_heads, self.head_dim)
    kv_shape = (config.emb_dim, self.num_kv_heads, self.head_dim)
    self.query = nnx.Einsum(
        "...e,ehd->...hd",
        q_shape,
        kernel_metadata={"out_sharding": sharding("query", ("embed", "heads", "head_dim"))},
        **kw,
    )
    self.key = nnx.Einsum(
        "...e,ehd->...hd",
        kv_shape,
        kernel_metadata={"out_sharding": sharding("key", ("embed", "kv_heads", "head_dim"))},
        **kw,
    )
    self.value = nnx.Einsum(
        "...e,ehd->...hd",
        kv_shape,
        kernel_metadata={"out_sharding": sharding("value", ("embed", "kv_heads", "head_dim"))},
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (self.num_query_heads, self.head_dim, config.emb_dim),
        kernel_metadata={"out_sharding": sharding("out", ("heads", "head_dim", "embed"))},
        **kw,
    )
    self.query_norm = nnx.RMSNorm(self.head_dim, **norm_kwargs(config, sharding, "query_norm", rngs))
    self.key_norm = nnx.RMSNorm(self.head_dim, **norm_kwargs(config, sharding, "key_norm", rngs))

  def __call__(self, x: jax.Array, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None):
    s = self.sharding
    query = s.constrain(self.query(x), "query", ("batch", "length", "heads", "head_dim"))
    key = s.constrain(self.key(x), "key", ("batch", "length", "kv_heads", "head_dim"))
    value = s.constrain(self.value(x), "value", ("batch", "length", "kv_heads", "head_dim"))

    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)
    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_max_timescale)
    query = query * self.head_dim**-0.5

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = dot_product_attention(
        query,
        key,
        value,
        positions=decoder_positions,
        segment_ids=decoder_segment_ids,
        float32_logits=self.config.float32_logits,
        kernel=self.attention_kernel,
    )
    out = checkpoint_name(s.constrain(out, "attention_out", ("batch", "length", "heads", "head_dim")), "attention_out")
    return checkpoint_name(self.out(out), "out_proj")


class Qwen3DecoderLayer(nnx.Module):
  """Pre-norm attention block followed by a pre-norm MLP block."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = Qwen3Attention(config, sharding=sharding, rngs=rngs)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "post_self_attention_layer_norm", rngs)
    )
    self.mlp = Qwen3MLP(config, sharding=sharding, rngs=rngs)

  def __call__(self, x: jax.Array, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None):
    x = checkpoint_name(
        self.sharding.constrain(x, "decoder_layer_input", ("batch", "length", "embed")), "decoder_layer_input"
    )
    attn_out = self.self_attention(
        self.pre_self_attention_layer_norm(x),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    x = (x + attn_out).astype(self.config.dtype)
    mlp_out = self.mlp(self.post_self_attention_layer_norm(x))
    return (x + mlp_out).astype(self.config.dtype)


class Qwen3Decoder(nnx.Module):
  """Decoder stack: embedded tokens in, logits out.

  Takes the layer class as an argument so Qwen3-VL (different attention) and
  Qwen3-MoE (different MLP) reuse the stack, the final norm and the output
  head without subclassing their way around them.
  """

  def __init__(
      self,
      config: Config,
      *,
      sharding: Sharding,
      rngs: nnx.Rngs,
      layer_cls: type[nnx.Module] = Qwen3DecoderLayer,
  ):
    self.config = config
    self.sharding = sharding
    self.layers = Layers([layer_cls(config, sharding=sharding, rngs=rngs) for _ in range(config.num_decoder_layers)])
    self.decoder_norm = nnx.RMSNorm(config.emb_dim, **norm_kwargs(config, sharding, "decoder_norm", rngs))
    if not config.logits_via_embedding:
      self.logits_dense = nnx.Linear(
          config.emb_dim,
          config.vocab_size,
          use_bias=False,
          dtype=config.dtype,
          param_dtype=config.weight_dtype,
          kernel_metadata={"out_sharding": sharding("logits_dense", ("embed", "vocab"))},
          rngs=rngs,
      )

  def apply_output_head(self, shared_embedding: nnx.Embed, y: jax.Array) -> jax.Array:
    """Final norm, then logits from the tied embedding or the `logits_dense` head."""
    y = self.decoder_norm(y)
    if self.config.logits_via_embedding:
      attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype
      logits = jnp.dot(y.astype(attend_dtype), shared_embedding.embedding[...].astype(attend_dtype).T)
      if self.config.normalize_embedding_logits:
        logits = logits / jnp.sqrt(y.shape[-1])
      if self.config.final_logits_soft_cap:
        logits = jnp.tanh(logits / self.config.final_logits_soft_cap) * self.config.final_logits_soft_cap
    else:
      logits = self.logits_dense(y)
    logits = self.sharding.constrain(logits, "logits", ("batch", "length", "vocab"))
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  def __call__(
      self,
      shared_embedding: nnx.Embed,
      decoder_input_tokens: jax.Array,
      *,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    y = shared_embedding(decoder_input_tokens)
    y = self.layers(y, decoder_positions=decoder_positions, decoder_segment_ids=decoder_segment_ids)
    return self.apply_output_head(shared_embedding, y)


class Qwen3Model(nnx.Module):
  """Qwen3 language model: token embedding plus decoder."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None = None,
      *,
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
      **unused_kwargs,
  ):
    """Builds the model. Must run inside `jax.set_mesh(mesh)`; see `maxtext.m3.core.sharding`.

    Args:
      config: Legacy `pyconfig` surface; only architecture fields are read.
      mesh: Accepted for the legacy `create_model` signature. The mesh that
        matters is the ambient one.
      sharding: Logical-to-mesh axis rules; `DefaultSharding` if omitted.
      rngs: Parameter initialization streams.
      **unused_kwargs: Legacy construction keywords (`model_mode`,
        `quant_mode_str`) that m3 does not act on.
    """
    del mesh, unused_kwargs
    self.config = config
    self.sharding = sharding or DefaultSharding()
    self.token_embedder = nnx.Embed(
        config.vocab_size,
        config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata={"out_sharding": self.sharding("token_embedder", ("vocab", "embed"))},
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(config, sharding=self.sharding, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      **unused_kwargs,
  ) -> CausalLMOutput:
    """Next-token logits for a batch of token ids.

    Args:
      decoder_input_tokens: `(batch, length)` token ids.
      decoder_positions: `(batch, length)` positions, restarting at zero for
        each packed segment.
      decoder_segment_ids: `(batch, length)` segment ids for packed sequences;
        attention stays within a segment. `None` means one segment per row.
      **unused_kwargs: Legacy call keywords (`model_mode`, `enable_dropout`,
        `decoder_target_tokens`, ...) that m3 does not act on.
    """
    del unused_kwargs
    logits = self.decoder(
        self.token_embedder,
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return CausalLMOutput(logits=logits)


def create_qwen3_model(config: Config, mesh: Mesh | None = None, *, rngs: nnx.Rngs | None = None, **kwargs) -> Qwen3Model:
  """Registry entry point; see `maxtext.m3.models.create_model`."""
  return Qwen3Model(config, mesh, rngs=rngs or nnx.Rngs(0), **kwargs)
