"""LLaMA model with explicit sharding-in-types annotations in NNX."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import numpy as np

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.nnx_exp.rope import apply_rope_factors, rope_factors, rope_frequencies
from maxtext.nnx_exp.sharding import Sharding


_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def _sharded_init(init_fn: Callable[..., jax.Array], sharding):
  return partial(init_fn, out_sharding=sharding)


def _sharded_constant_init(value, sharding):
  def init(_key, shape, dtype=jnp.float32):
    match value:
      case 0:
        return jnp.zeros(shape, dtype=dtype, out_sharding=sharding)
      case 1:
        return jnp.ones(shape, dtype=dtype, out_sharding=sharding)
      case _:
        return jnp.zeros(shape, dtype=dtype, out_sharding=sharding) + jnp.asarray(value, dtype=dtype)

  return init


def _stamp_sharding(x, sharding):
  return jnp.broadcast_to(x, x.shape, out_sharding=sharding)


@dataclass
class LlamaConfig:
  vocab_size: int = 128256
  emb_dim: int = 4096
  num_heads: int = 32
  num_kv_heads: int = 8
  num_layers: int = 32
  mlp_dim: int = 14336
  head_dim: int = 128
  rope_max_timescale: float = 500_000.0
  norm_eps: float = 1e-5
  dtype: str = "float32"

  def __post_init__(self):
    if self.num_heads % self.num_kv_heads != 0:
      raise ValueError(f"num_heads must be divisible by num_kv_heads, got {self.num_heads=} {self.num_kv_heads=}")
    if self.emb_dim != self.num_heads * self.head_dim:
      raise ValueError(f"emb_dim must equal num_heads * head_dim, got {self.emb_dim=} {self.num_heads=} {self.head_dim=}")


class Attention(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding: Sharding):
    self.num_heads = config.num_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.kv_repeat = config.num_heads // config.num_kv_heads
    self.sharding = sharding
    self.scale = config.head_dim ** -0.5
    self.rope_freq = nnx.static(
      tuple(np.asarray(rope_frequencies(config.head_dim, config.rope_max_timescale), dtype=np.float32))
    )

    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.qkv_proj = nnx.LinearGeneral(
      config.emb_dim,
      (config.num_heads + 2 * config.num_kv_heads, config.head_dim),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=_sharded_init(
        kernel_init,
        sharding.init_weight_spec("qkv_proj", ("embed",), ("qkv_heads", "head_dim")),
      ),
    )
    self.o_proj = nnx.LinearGeneral(
      (config.num_heads, config.head_dim),
      config.emb_dim,
      axis=(-2, -1),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=_sharded_init(
        kernel_init,
        sharding.init_weight_spec("o_proj", ("heads", "head_dim"), ("embed",)),
      ),
    )

  def __call__(self, x, positions, mask=None):
    s = self.sharding
    qkv = self.qkv_proj(x, out_sharding=s.attention_spec("qkv", head_axis="qkv_heads"))
    q, k, v = jnp.split(qkv, (self.num_heads, self.num_heads + self.num_kv_heads), axis=-2)

    cos, sin = rope_factors(positions, jnp.asarray(self.rope_freq, dtype=jnp.float32), q.ndim, q.dtype)
    q = checkpoint_name(_stamp_sharding(apply_rope_factors(q, cos, sin), s.query_spec("query")), "query_proj")
    k = checkpoint_name(_stamp_sharding(apply_rope_factors(k, cos, sin), s.kv_spec("key")), "key_proj")
    v = checkpoint_name(_stamp_sharding(v, s.kv_spec("value")), "value_proj")

    if self.kv_repeat != 1:
      k = jnp.repeat(k, self.kv_repeat, axis=-2, out_sharding=s.query_spec("key_repeated"))
      v = jnp.repeat(v, self.kv_repeat, axis=-2, out_sharding=s.query_spec("value_repeated"))

    out = jax.nn.dot_product_attention(
      q * self.scale,
      k,
      v,
      mask=mask,
      is_causal=True,
    )
    out = checkpoint_name(_stamp_sharding(out, s.query_spec("attn_out")), "attention_out")
    return self.o_proj(out, out_sharding=s.sequence_spec("post_attn"))


class MLP(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding: Sharding):
    self.sharding = sharding
    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.gate_up = nnx.Linear(
      config.emb_dim,
      2 * config.mlp_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=_sharded_init(
        kernel_init,
        sharding.init_weight_spec("gate_up", ("embed",), ("mlp",)),
      ),
    )
    self.down = nnx.Linear(
      config.mlp_dim,
      config.emb_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=_sharded_init(
        kernel_init,
        sharding.init_weight_spec("down", ("mlp",), ("embed",)),
      ),
    )

  def __call__(self, x):
    mlp_spec = self.sharding.mlp_spec("mlpwi")
    gate_kernel, up_kernel = jnp.split(self.gate_up.kernel[...], 2, axis=-1)
    
    # Simple GLU implementation
    gate = jnp.tensordot(x, gate_kernel, axes=((-1,), (0,)), out_sharding=mlp_spec)
    up = jnp.tensordot(x, up_kernel, axes=((-1,), (0,)), out_sharding=mlp_spec)
    hidden = jax.nn.silu(gate) * up
    
    hidden = checkpoint_name(_stamp_sharding(hidden, mlp_spec), "mlpwi")
    return self.down(hidden, out_sharding=self.sharding.sequence_spec("post_mlp"))


class DecoderLayer(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding: Sharding):
    self.sharding = sharding
    dt = _DTYPES[config.dtype]
    self.attn_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=_sharded_constant_init(1, sharding.weight_spec("attn_norm", ("norm",))),
    )
    self.attn = Attention(config, rngs=rngs, sharding=sharding)
    self.mlp_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=_sharded_constant_init(1, sharding.weight_spec("mlp_norm", ("norm",))),
    )
    self.mlp = MLP(config, rngs=rngs, sharding=sharding)

  def __call__(self, x, positions, mask=None):
    attn_in = _stamp_sharding(self.attn_norm(x), self.sharding.sequence_spec("attn_input"))
    x = _stamp_sharding(x + self.attn(attn_in, positions, mask), self.sharding.sequence_spec("post_attn"))
    mlp_in = _stamp_sharding(self.mlp_norm(x), self.sharding.sequence_spec("mlp_input"))
    x = _stamp_sharding(x + self.mlp(mlp_in), self.sharding.sequence_spec("post_mlp"))
    return x


class Llama(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding: Sharding):
    self.sharding = sharding
    self.config = config
    dt = _DTYPES[config.dtype]
    embedding_init = nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.embed = nnx.Embed(
      config.vocab_size,
      config.emb_dim,
      dtype=dt,
      rngs=rngs,
      embedding_init=_sharded_init(embedding_init, sharding.init_weight_spec("embed", ("vocab",), ("embed",))),
    )
    self.layers = nnx.List([DecoderLayer(config, rngs=rngs, sharding=sharding) for _ in range(config.num_layers)])
    self.norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=_sharded_constant_init(1, sharding.weight_spec("final_norm", ("norm",))),
    )

  def embed_tokens(self, tokens):
    return self.embed.embedding.at[tokens].get(
      mode="promise_in_bounds",
      out_sharding=self.sharding.sequence_spec("embed_tokens"),
    )

  def __call__(self, tokens, positions, mask=None):
    x = self.embed_tokens(tokens)
    for layer in self.layers:
      x = layer(x, positions, mask)
    x = _stamp_sharding(self.norm(x), self.sharding.sequence_spec("final_norm"))
    
    # Compute logits by sharing weights with embedding
    logits = jax.lax.dot_general(
      x,
      self.embed.embedding.T,
      (((x.ndim - 1,), (0,)), ((), ())),
      out_sharding=self.sharding.logits_spec("logits")
    )
    return logits
