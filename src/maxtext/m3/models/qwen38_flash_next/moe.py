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

"""Qwen3.8 Sparse MoE block (routed experts and gated shared expert)."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp


class Qwen38MLP(nnx.Module):
  """SwiGLU MLP used as shared expert in Qwen3.8-Flash-Next MoE."""

  def __init__(
      self,
      emb_dim: int = 2560,
      intermediate_dim: int = 640,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.gate_proj = nnx.Linear(emb_dim, intermediate_dim, **kw)
    self.up_proj = nnx.Linear(emb_dim, intermediate_dim, **kw)
    self.down_proj = nnx.Linear(intermediate_dim, emb_dim, **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass computing down_proj(silu(gate_proj(x)) * up_proj(x))."""
    return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen38Experts(nnx.Module):
  """Collection of routed expert weights stored as 3D tensors.

  Weight Layout Contract (for CL 9 checkpoint conversion):
    - gate_up_proj: (num_experts, emb_dim, 2 * moe_mlp_dim) -> (E, H, 2I).
      HuggingFace stores gate_up_proj as (E, 2I, H); transpose last two axes.
    - down_proj: (num_experts, moe_mlp_dim, emb_dim) -> (E, I, H).
      HuggingFace stores down_proj as (E, H, I); transpose last two axes.
  """

  def __init__(
      self,
      num_experts: int = 512,
      emb_dim: int = 2560,
      moe_mlp_dim: int = 640,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.emb_dim = emb_dim
    self.moe_mlp_dim = moe_mlp_dim
    self.dtype = dtype

    self.gate_up_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (num_experts, emb_dim, 2 * moe_mlp_dim), weight_dtype)
    )
    self.down_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (num_experts, moe_mlp_dim, emb_dim), weight_dtype)
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      top_k_index: jax.Array,
      top_k_weights: jax.Array,
  ) -> jax.Array:
    """Dispatches tokens via sorted token-expert grouping and jax.lax.ragged_dot."""
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
      hidden_states = hidden_states.reshape(-1, orig_shape[-1])
      top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
      top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

    num_tokens, top_k = top_k_index.shape
    flat_expert_ids = top_k_index.reshape(-1)
    perm = jnp.argsort(flat_expert_ids)
    group_sizes = jnp.bincount(flat_expert_ids, length=self.num_experts).astype(jnp.int32)
    flat_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
    sorted_x = hidden_states[flat_token_ids[perm]]

    w_gate_up = jnp.asarray(self.gate_up_proj.value, dtype=self.dtype)
    gate_up = jax.lax.ragged_dot(sorted_x.astype(self.dtype), w_gate_up, group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    act = jax.nn.silu(gate) * up

    w_down = jnp.asarray(self.down_proj.value, dtype=self.dtype)
    sorted_out = jax.lax.ragged_dot(act, w_down, group_sizes)

    inv_perm = jnp.argsort(perm)
    expert_outs = sorted_out[inv_perm].reshape(num_tokens, top_k, self.emb_dim)
    out = jnp.sum(expert_outs * top_k_weights[:, :, None].astype(expert_outs.dtype), axis=1)
    out = out.astype(hidden_states.dtype)

    if len(orig_shape) == 3:
      out = out.reshape(orig_shape)
    return out

  def reference_forward(
      self,
      hidden_states: jax.Array,
      top_k_index: jax.Array,
      top_k_weights: jax.Array,
  ) -> jax.Array:
    """Per-token weight gather reference forward used in parity tests."""
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
      hidden_states = hidden_states.reshape(-1, orig_shape[-1])
      top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
      top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

    w_gate_up = jnp.asarray(self.gate_up_proj.value, dtype=self.dtype)[top_k_index]
    gate_up = jnp.einsum("nd,nkdf->nkf", hidden_states.astype(self.dtype), w_gate_up)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    act = jax.nn.silu(gate) * up
    w_down = jnp.asarray(self.down_proj.value, dtype=self.dtype)[top_k_index]
    expert_out = jnp.einsum("nkm,nkmd->nkd", act, w_down)
    out = jnp.einsum("nkd,nk->nd", expert_out, top_k_weights.astype(expert_out.dtype))
    out = out.astype(hidden_states.dtype)

    if len(orig_shape) == 3:
      out = out.reshape(orig_shape)
    return out


class Qwen38TopKRouter(nnx.Module):
  """Top-k router computing routing weights, expert indices, and full probabilities."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_experts: int = 512,
      num_experts_per_tok: int = 10,
      norm_topk_prob: bool = True,
      routed_scaling_factor: float = 1.0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.dtype = dtype
    self.linear = nnx.Linear(
        emb_dim,
        num_experts,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes routing weights, selected experts, and full float32 router probabilities."""
    router_logits = self.linear(hidden_states)
    router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    router_top_value, router_indices = jax.lax.top_k(router_probs, self.num_experts_per_tok)
    if self.norm_topk_prob:
      router_top_value = router_top_value / jnp.sum(router_top_value, axis=-1, keepdims=True)
    routing_weights = (router_top_value * self.routed_scaling_factor).astype(router_logits.dtype)
    return routing_weights, router_indices, router_probs


class Qwen38SparseMoeBlock(nnx.Module):
  """Sparse MoE block with routed experts and gated shared expert."""

  def __init__(
      self,
      emb_dim: int = 2560,
      moe_mlp_dim: int = 640,
      num_experts: int = 512,
      num_experts_per_tok: int = 10,
      norm_topk_prob: bool = True,
      routed_scaling_factor: float = 1.0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      shared_expert_intermediate_size: int,
      rngs: nnx.Rngs,
  ):
    del sharding
    if shared_expert_intermediate_size <= 0:
      raise ValueError(f"shared_expert_intermediate_size must be > 0, got {shared_expert_intermediate_size}.")
    self.emb_dim = emb_dim
    self.moe_mlp_dim = moe_mlp_dim
    self.shared_expert_intermediate_size = shared_expert_intermediate_size
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.dtype = dtype

    self.gate = Qwen38TopKRouter(
        emb_dim=emb_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    self.experts = Qwen38Experts(
        num_experts=num_experts,
        emb_dim=emb_dim,
        moe_mlp_dim=moe_mlp_dim,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    self.shared_expert = Qwen38MLP(
        emb_dim=emb_dim,
        intermediate_dim=shared_expert_intermediate_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.shared_expert_gate = nnx.Linear(emb_dim, 1, **kw)

  def __call__(
      self,
      hidden_states: jax.Array,
      return_router_topk: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through routed experts and gated shared expert."""
    b, s, d = hidden_states.shape
    flat_states = hidden_states.reshape(-1, d)

    routing_weights, router_indices, router_probs = self.gate(flat_states)
    routed_out = self.experts(flat_states, router_indices, routing_weights)
    shared_out = self.shared_expert(flat_states)
    shared_gate = jax.nn.sigmoid(self.shared_expert_gate(flat_states).astype(jnp.float32)).astype(shared_out.dtype)

    combined = (routed_out + shared_gate * shared_out).reshape(b, s, d)
    if return_router_topk:
      return (
          combined,
          routing_weights.reshape(b, s, self.num_experts_per_tok),
          router_indices.reshape(b, s, self.num_experts_per_tok),
          router_probs.reshape(b, s, self.num_experts),
      )
    return combined
