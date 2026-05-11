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


_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


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
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.num_heads = config.num_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.kv_repeat = config.num_heads // config.num_kv_heads
    self.scale = config.head_dim ** -0.5
    self.rope_freq = tuple(np.asarray(rope_frequencies(config.head_dim, config.rope_max_timescale), dtype=np.float32))
    self.sharding_hook = sharding_hook or (lambda x, name: x)

    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.qkv_proj = nnx.LinearGeneral(
      config.emb_dim,
      (config.num_heads + 2 * config.num_kv_heads, config.head_dim),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "qkv_proj_kernel_init"),
    )
    self.o_proj = nnx.LinearGeneral(
      (config.num_heads, config.head_dim),
      config.emb_dim,
      axis=(-2, -1),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "o_proj_kernel_init"),
    )

  def __call__(self, x, positions, mask=None, qkv_kernel=None):
    hook = self.sharding_hook
    
    if qkv_kernel is None:
      qkv_kernel = self.qkv_proj.kernel[...]
      qkv_kernel = hook(qkv_kernel, "qkv_proj_kernel")
      
    qkv = hook(jnp.tensordot(x, qkv_kernel, axes=((-1,), (0,))), "qkv")
    q, k, v = jnp.split(qkv, (self.num_heads, self.num_heads + self.num_kv_heads), axis=-2)

    cos, sin = rope_factors(positions, jnp.asarray(self.rope_freq, dtype=jnp.float32), q.ndim, q.dtype)
    q = checkpoint_name(hook(apply_rope_factors(q, cos, sin), "query"), "query_proj")
    k = checkpoint_name(hook(apply_rope_factors(k, cos, sin), "key"), "key_proj")
    v = checkpoint_name(hook(v, "value"), "value_proj")

    if self.kv_repeat != 1:
      k_sharding = hook.get_spec("key_repeated") if hasattr(hook, "get_spec") else None
      v_sharding = hook.get_spec("value_repeated") if hasattr(hook, "get_spec") else None
      k = hook(jnp.repeat(k, self.kv_repeat, axis=-2, out_sharding=k_sharding), "key_repeated")
      v = hook(jnp.repeat(v, self.kv_repeat, axis=-2, out_sharding=v_sharding), "value_repeated")

    out = jax.nn.dot_product_attention(
      q * self.scale,
      k,
      v,
      mask=mask,
      is_causal=True,
    )
    out = checkpoint_name(hook(out, "attn_out"), "attention_out")
    out_sharding = hook.get_spec("post_attn") if hasattr(hook, "get_spec") else None
    return hook(self.o_proj(out, out_sharding=out_sharding), "post_attn")


class MLP(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.gate_up = nnx.Linear(
      config.emb_dim,
      2 * config.mlp_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "gate_up_kernel_init"),
    )
    self.down = nnx.Linear(
      config.mlp_dim,
      config.emb_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "down_kernel_init"),
    )

  def __call__(self, x):
    hook = self.sharding_hook
    gate_kernel, up_kernel = jnp.split(self.gate_up.kernel[...], 2, axis=-1)
    
    gate_kernel = hook(gate_kernel, "gate_up_kernel")
    up_kernel = hook(up_kernel, "gate_up_kernel")
    
    gate = hook(jnp.tensordot(x, gate_kernel, axes=((-1,), (0,))), "gate")
    up = hook(jnp.tensordot(x, up_kernel, axes=((-1,), (0,))), "up")
    hidden = jax.nn.silu(gate) * up
    
    hidden = checkpoint_name(hook(hidden, "mlpwi"), "mlpwi")
    out_sharding = hook.get_spec("post_mlp") if hasattr(hook, "get_spec") else None
    return hook(self.down(hidden, out_sharding=out_sharding), "post_mlp")


class DecoderLayer(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    self.attn_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "attn_norm_scale_init"),
    )
    self.attn = Attention(config, rngs=rngs, sharding_hook=sharding_hook)
    self.mlp_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "mlp_norm_scale_init"),
    )
    self.mlp = MLP(config, rngs=rngs, sharding_hook=sharding_hook)

  def __call__(self, x, positions, mask=None):
    hook = self.sharding_hook
    qkv_kernel = self.attn.sharding_hook(self.attn.qkv_proj.kernel[...], "qkv_proj_kernel")
    
    attn_in = hook(self.attn_norm(x), "attn_input")
    x = hook(x + self.attn(attn_in, positions, mask, qkv_kernel=qkv_kernel), "post_attn")
    mlp_in = hook(self.mlp_norm(x), "mlp_input")
    x = hook(x + self.mlp(mlp_in), "post_mlp")
    return x


class Llama(nnx.Module):
  def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None, scan: bool = False):
    self.config = config
    self.scan = scan
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    embedding_init = nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.embed = nnx.Embed(
      config.vocab_size,
      config.emb_dim,
      dtype=dt,
      rngs=rngs,
      embedding_init=self.sharding_hook(embedding_init, "embed_embedding_init"),
    )
    if scan:
      from maxtext.nnx_exp.infra.scan import create_scanned_layers
      self.layers = nnx.data(create_scanned_layers(
          DecoderLayer,
          config,
          config.num_layers,
          rngs=rngs,
          sharding_hook=sharding_hook
      ))
    else:
      self.layers = nnx.Sequential(*[DecoderLayer(config, rngs=rngs, sharding_hook=sharding_hook) for _ in range(config.num_layers)])
    self.use_remat = False
    self.remat_policy = None
    self.norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "final_norm_scale_init"),
    )

  def embed_tokens(self, tokens):
    hook = self.sharding_hook
    out_sharding = hook.get_spec("embed_tokens") if hasattr(hook, "get_spec") else None
    return hook(
      self.embed.embedding.at[tokens].get(mode="promise_in_bounds", out_sharding=out_sharding),
      "embed_tokens"
    )

  def __call__(self, tokens, positions, mask=None):
    hook = self.sharding_hook
    x = self.embed_tokens(tokens)
    if self.scan:
      from maxtext.nnx_exp.infra.scan import scan_forward
      x = scan_forward(x, self.layers, positions, mask, use_remat=getattr(self, "use_remat", False), remat_policy=getattr(self, "remat_policy", None))
    else:
      for layer in self.layers.layers:
        remat_policy = getattr(self, "remat_policy", None)
        if remat_policy is not None:
          x = nnx.remat(layer, policy=remat_policy)(x, positions, mask)
        else:
          x = layer(x, positions, mask)
    x = hook(self.norm(x), "final_norm")
    
    out_sharding = hook.get_spec("logits") if hasattr(hook, "get_spec") else None
    logits = jax.lax.dot_general(
      x,
      self.embed.embedding.T,
      (((x.ndim - 1,), (0,)), ((), ())),
      out_sharding=out_sharding,
    )
    return hook(logits, "logits")
