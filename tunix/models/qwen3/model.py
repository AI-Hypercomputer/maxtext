# Copyright 2025 Google LLC
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

"""Qwen3 model."""

import dataclasses
from typing import Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping


K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen3 model."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_ndh: Tuple[str | None, ...]
  kv_weight_ndh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_ndh=('tp', fsdp, None),
        kv_weight_ndh=('tp', fsdp, None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  """Configuration for the Qwen3 model."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()

  @classmethod
  def qwen3_0_6_b(cls):  # qwen3-0.6B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1024,
        hidden_dim=3072,
        num_heads=16,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )

  @classmethod
  def qwen3_1_7_b(cls):  # qwen3-1.7B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=2048,
        hidden_dim=6144,
        num_heads=16,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )

  @classmethod
  def qwen3_14_b(cls):  # qwen3-14B
    return cls(
        num_layers=40,
        vocab_size=151936,
        embed_dim=5120,
        hidden_dim=17408,
        num_heads=40,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.einsum(self.einsum_str, x, self.w.value)


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.dot(x, self.input_embedding.value.T)


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    dtype = x.dtype
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
        + self.norm_eps
    )
    return jnp.astype(self.w * x / rms, dtype)


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.shd_config = shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=shd_config.q_weight_ndh,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=shd_config.kv_weight_ndh,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=shd_config.kv_weight_ndh,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=shd_config.o_weight_nhd,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]

    query_proj = self.q_norm(self.q_proj(x))
    key_proj = self.k_norm(self.k_proj(x))
    value_proj = self.v_proj(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape

    # GQA
    query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
    attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
    attn = attn.reshape((b, qh, t, s))

    if attn_mask is not None:
      attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    attn = attn.reshape((b, kh, qh // kh, t, s))
    qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
    qkv = qkv.reshape((b, t, qh, d))

    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @property
  def head_dim(self):
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    return self.kv_proj.shape[1]


class MLP(nnx.Module):
  """MLP module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.shd_config = shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs


class DecoderLayer(nnx.Module):
  """DecoderLayer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
        shd_config=shd_config,
    )

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs


class Qwen3(nnx.Module):
  """Qwen3 model."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=shd_config,
    )
    self.layers = [
        DecoderLayer(config=config, rngs=rngs, shd_config=shd_config)
        for _ in range(config.num_layers)
    ]
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=shd_config,
    )
    self.lm_head = Einsum(
        einsum_str='BTD,DV->BTV',
        shape=(config.embed_dim, config.vocab_size),
        rngs=rngs,
        sharding=shd_config.emb_dv,
    )

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen3 model.

    Args:
      input_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens)

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    logits = self.lm_head(x)

    return logits, new_cache  # pytype: disable=bad-return-type
