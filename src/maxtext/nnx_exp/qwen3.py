"""Qwen3 MoE model architecture logic in NNX."""

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.nnx_exp.rope import apply_rope_factors, rope_factors, rope_frequencies
from maxtext.nnx_exp.kernels import dot_product_attention, gated_linear_unit, KernelBackend, KernelImplementation
from maxtext.nnx_exp.moe.moe_types import MoEExecutor, MoERuntimeConfig, MegabloxConfig
from maxtext.nnx_exp.moe.moe import execute_routed_moe


_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


@dataclass
class Qwen3CausalLMOutput:
  logits: jax.Array
  aux_loss: jax.Array | None = None
  router_logits: tuple[jax.Array, ...] | None = None


@dataclass
class Qwen3MoEConfig:
  vocab_size: int = 152064
  emb_dim: int = 2048
  num_heads: int = 16
  num_kv_heads: int = 16
  num_layers: int = 24
  dense_mlp_dim: int = 5632
  moe_mlp_dim: int = 1408
  head_dim: int = 128
  rope_max_timescale: float = 10000.0
  norm_eps: float = 1e-6
  dtype: str = "float32"
  attention_backend: KernelBackend = "legacy"
  attention_implementation: KernelImplementation = None
  glu_backend: KernelBackend = "legacy"
  glu_implementation: KernelImplementation = None
  
  # MoE config
  num_experts: int = 64
  num_experts_per_tok: int = 8
  norm_topk_prob: bool = True
  decoder_sparse_step: int = 1
  mlp_only_layers: tuple[int, ...] = ()
  router_aux_loss_coef: float = 0.01
  output_router_logits: bool = True
  moe_executor: MoEExecutor = "ragged_dot"
  capacity_factor: float = -1.0

  def __post_init__(self):
    if self.num_heads % self.num_kv_heads != 0:
      raise ValueError(f"num_heads must be divisible by num_kv_heads, got {self.num_heads=} {self.num_kv_heads=}")

  def moe_runtime_config(self) -> MoERuntimeConfig:
    return MoERuntimeConfig(
        executor=self.moe_executor,
        capacity_factor=self.capacity_factor,
        megablox=MegabloxConfig(),
    )


def qwen_router_aux_loss(
    router_logits: tuple[jax.Array, ...] | None,
    *,
    num_experts: int,
    top_k: int,
) -> jax.Array:
  if not router_logits:
    return jnp.array(0.0, dtype=jnp.float32)

  concatenated = jnp.concatenate(
    [jnp.reshape(layer_logits, (-1, num_experts)) for layer_logits in router_logits],
    axis=0,
  )
  routing_weights = jax.nn.softmax(concatenated.astype(jnp.float32), axis=-1)
  _, selected_experts = jax.lax.top_k(routing_weights, top_k)
  expert_mask = jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32)
  tokens_per_expert = jnp.mean(expert_mask, axis=0)
  router_prob_per_expert = jnp.mean(routing_weights, axis=0)
  return jnp.sum(tokens_per_expert * router_prob_per_expert[None, :]) * num_experts


def qwen_uses_sparse_moe(config: Qwen3MoEConfig, layer_idx: int) -> bool:
  return (
    config.num_experts > 0
    and layer_idx not in config.mlp_only_layers
    and (layer_idx + 1) % config.decoder_sparse_step == 0
  )


class Qwen3Attention(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.num_heads = config.num_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.qk_groups = config.num_heads // config.num_kv_heads
    self.scale = config.head_dim ** -0.5
    self.rope_freq = tuple(np.asarray(rope_frequencies(config.head_dim, config.rope_max_timescale), dtype=np.float32))
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.backend = config.attention_backend
    self.implementation = config.attention_implementation

    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.q_proj = nnx.LinearGeneral(
      config.emb_dim,
      (config.num_heads, config.head_dim),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "q_proj_kernel_init"),
    )
    self.k_proj = nnx.LinearGeneral(
      config.emb_dim,
      (config.num_kv_heads, config.head_dim),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "k_proj_kernel_init"),
    )
    self.v_proj = nnx.LinearGeneral(
      config.emb_dim,
      (config.num_kv_heads, config.head_dim),
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "v_proj_kernel_init"),
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
    self.q_norm = nnx.RMSNorm(
      config.head_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "q_norm_scale_init"),
    )
    self.k_norm = nnx.RMSNorm(
      config.head_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "k_norm_scale_init"),
    )

  def __call__(self, x, positions, mask=None):
    hook = self.sharding_hook
    
    q = self.q_proj(x, out_sharding=hook.get_spec("query") if hasattr(hook, "get_spec") else None)
    q = checkpoint_name(hook(self.q_norm(q), "query"), "query_proj")
    k = self.k_proj(x, out_sharding=hook.get_spec("key") if hasattr(hook, "get_spec") else None)
    k = checkpoint_name(hook(self.k_norm(k), "key"), "key_proj")
    v = checkpoint_name(hook(self.v_proj(x, out_sharding=hook.get_spec("value") if hasattr(hook, "get_spec") else None), "value"), "value_proj")

    cos, sin = rope_factors(positions, jnp.asarray(self.rope_freq, dtype=jnp.float32), q.ndim, q.dtype)
    q = checkpoint_name(hook(apply_rope_factors(q, cos, sin), "query"), "query_rope")
    k = checkpoint_name(hook(apply_rope_factors(k, cos, sin), "key"), "key_rope")

    if self.num_kv_heads < self.num_heads:
      k_sharding = hook.get_spec("key_repeated") if hasattr(hook, "get_spec") else None
      v_sharding = hook.get_spec("value_repeated") if hasattr(hook, "get_spec") else None
      k = hook(jnp.repeat(k, self.qk_groups, axis=-2, out_sharding=k_sharding), "key_repeated")
      v = hook(jnp.repeat(v, self.qk_groups, axis=-2, out_sharding=v_sharding), "value_repeated")

    out = dot_product_attention(
      q * self.scale,
      k,
      v,
      mask=mask,
      backend=self.backend,
      implementation=self.implementation,
    )
    out = checkpoint_name(hook(out, "attn_out"), "attention_out")
    out_sharding = hook.get_spec("post_attn") if hasattr(hook, "get_spec") else None
    return hook(self.o_proj(out, out_sharding=out_sharding), "post_attn")


class Qwen3DenseMLP(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.backend = config.glu_backend
    self.implementation = config.glu_implementation
    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.gate_proj = nnx.Linear(
      config.emb_dim,
      config.dense_mlp_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "gate_proj_kernel_init"),
    )
    self.up_proj = nnx.Linear(
      config.emb_dim,
      config.dense_mlp_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "up_proj_kernel_init"),
    )
    self.down_proj = nnx.Linear(
      config.dense_mlp_dim,
      config.emb_dim,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "down_proj_kernel_init"),
    )

  def __call__(self, x):
    hook = self.sharding_hook
    gate_kernel = hook(self.gate_proj.kernel[...], "gate_proj_kernel")
    up_kernel = hook(self.up_proj.kernel[...], "up_proj_kernel")
    
    hidden = gated_linear_unit(
      x,
      (gate_kernel, up_kernel),
      activation=jax.nn.silu,
      backend=self.backend,
      implementation=self.implementation,
      out_sharding=hook.get_spec("mlpwi") if hasattr(hook, "get_spec") else None,
    )
    hidden = checkpoint_name(hook(hidden, "mlpwi"), "mlpwi")
    out_sharding = hook.get_spec("post_mlp") if hasattr(hook, "get_spec") else None
    return hook(self.down_proj(hidden, out_sharding=out_sharding), "post_mlp")


class Qwen3TopKRouter(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    self.proj = nnx.Linear(
      config.emb_dim,
      config.num_experts,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "router_kernel_init"),
    )

  def __call__(self, x):
    hook = self.sharding_hook
    router_logits = self.proj(x, out_sharding=hook.get_spec("router") if hasattr(hook, "get_spec") else None)
    router_logits = hook(router_logits, "router_logits")
    router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1).astype(router_logits.dtype)
    weights, indices = jax.lax.top_k(router_probs, k=self.top_k)
    if self.norm_topk_prob:
      weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)
    return router_logits, weights, indices


class Qwen3RoutedExperts(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    kernel_init = nnx.initializers.lecun_normal()
    
    gate_up_shape = (config.num_experts, config.emb_dim, 2 * config.moe_mlp_dim)
    down_shape = (config.num_experts, config.moe_mlp_dim, config.emb_dim)
    
    key1, key2 = jax.random.split(rngs.params())
    gate_up_val = kernel_init(key1, gate_up_shape, dtype=dt)
    down_val = kernel_init(key2, down_shape, dtype=dt)
    
    self.gate_up = nnx.Param(self.sharding_hook(gate_up_val, "experts_gate_up_init"))
    self.down = nnx.Param(self.sharding_hook(down_val, "experts_down_init"))


class Qwen3SparseMoEBlock(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.runtime_config = config.moe_runtime_config()
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.router = Qwen3TopKRouter(config, rngs=rngs, sharding_hook=sharding_hook)
    self.experts = Qwen3RoutedExperts(config, rngs=rngs, sharding_hook=sharding_hook)

  def __call__(self, x):
    hook = self.sharding_hook
    router_logits, weights, indices = self.router(x)
    gate_up = hook(self.experts.gate_up[...], "experts_gate_up")
    gate, up = jnp.split(gate_up, 2, axis=-1)
    
    down = hook(self.experts.down[...], "experts_down")
    
    routed_out = execute_routed_moe(
      x,
      weights,
      indices,
      gate,
      up,
      down,
      runtime_config=self.runtime_config,
      num_experts=gate_up.shape[0],
      combine_sharding=hook.get_spec("combine_weights") if hasattr(hook, "get_spec") else None,
      hidden_sharding=hook.get_spec("expert_mlpwi") if hasattr(hook, "get_spec") else None,
      outputs_sharding=hook.get_spec("all_expert_out") if hasattr(hook, "get_spec") else None,
      out_sharding=hook.get_spec("routed_out") if hasattr(hook, "get_spec") else None,
    )
    return routed_out, router_logits


class Qwen3DenseDecoderLayer(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    self.attn_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "attn_norm_scale_init"),
    )
    self.attn = Qwen3Attention(config, rngs=rngs, sharding_hook=sharding_hook)
    self.mlp_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "mlp_norm_scale_init"),
    )
    self.mlp = Qwen3DenseMLP(config, rngs=rngs, sharding_hook=sharding_hook)

  def __call__(self, x, positions, mask=None):
    hook = self.sharding_hook
    attn_in = hook(self.attn_norm(x), "attn_input")
    x = hook(x + self.attn(attn_in, positions, mask), "post_attn")
    mlp_in = hook(self.mlp_norm(x), "mlp_input")
    x = hook(x + self.mlp(mlp_in), "post_mlp")
    return x, None


class Qwen3MoEDecoderLayer(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    dt = _DTYPES[config.dtype]
    self.attn_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "attn_norm_scale_init"),
    )
    self.attn = Qwen3Attention(config, rngs=rngs, sharding_hook=sharding_hook)
    self.mlp_norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "mlp_norm_scale_init"),
    )
    self.mlp = Qwen3SparseMoEBlock(config, rngs=rngs, sharding_hook=sharding_hook)

  def __call__(self, x, positions, mask=None):
    hook = self.sharding_hook
    attn_in = hook(self.attn_norm(x), "attn_input")
    x = hook(x + self.attn(attn_in, positions, mask), "post_attn")
    mlp_in = hook(self.mlp_norm(x), "mlp_input")
    mlp_out, router_logits = self.mlp(mlp_in)
    x = hook(x + mlp_out, "post_mlp")
    return x, router_logits


class Qwen3MoE(nnx.Module):
  def __init__(self, config: Qwen3MoEConfig, *, rngs: nnx.Rngs, sharding_hook: Callable = None):
    self.config = config
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
    
    def make_layer(layer_idx):
      if qwen_uses_sparse_moe(config, layer_idx):
        return Qwen3MoEDecoderLayer(config, rngs=rngs, sharding_hook=sharding_hook)
      return Qwen3DenseDecoderLayer(config, rngs=rngs, sharding_hook=sharding_hook)

    self.layers = nnx.Sequential(*[make_layer(i) for i in range(config.num_layers)])
    self.norm = nnx.RMSNorm(
      config.emb_dim,
      epsilon=config.norm_eps,
      dtype=dt,
      rngs=rngs,
      scale_init=self.sharding_hook(1, "final_norm_scale_init"),
    )
    kernel_init = nnx.initializers.lecun_normal()
    self.lm_head = nnx.Linear(
      config.emb_dim,
      config.vocab_size,
      use_bias=False,
      dtype=dt,
      rngs=rngs,
      kernel_init=self.sharding_hook(kernel_init, "lm_head_kernel_init"),
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
    
    router_logits_layers = [] if self.config.output_router_logits else None
    for layer in self.layers.layers:
      x, router_logits = layer(x, positions, mask)
      if router_logits_layers is not None and router_logits is not None:
        router_logits_layers.append(router_logits)
        
    x = hook(self.norm(x), "final_norm")
    
    out_sharding = hook.get_spec("logits") if hasattr(hook, "get_spec") else None
    logits = jax.lax.dot_general(
      x,
      self.lm_head.kernel[...],
      (((x.ndim - 1,), (0,)), ((), ())),
      out_sharding=out_sharding,
    )
    logits = hook(logits, "logits")
    
    aux_loss = None
    if router_logits_layers:
      router_logits_out = tuple(router_logits_layers)
      aux_loss = qwen_router_aux_loss(
        router_logits_out,
        num_experts=self.config.num_experts,
        top_k=self.config.num_experts_per_tok,
      ) * self.config.router_aux_loss_coef
      
    return Qwen3CausalLMOutput(logits=logits, aux_loss=aux_loss, router_logits=router_logits_layers)
