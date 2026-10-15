# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clamped-SwiGLU Dense MLP and 288-expert MoE for GLM-5.3-Flash."""

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config


class Glm5RouterBias(nnx.Variable):
  """Router bias state. Subclasses nnx.Variable to exclude from gradients."""


class Glm5MLP(nnx.Module):
  """Clamped SwiGLU dense MLP matching Glm5NextTextMLP."""

  def __init__(
      self,
      config: Glm5Config,
      intermediate_size: int,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    self.hidden_size = h = config.hidden_size
    self.intermediate_size = inter = intermediate_size
    self.swiglu_limit = config.mlp_clamp_limit

    dt, p_dt = getattr(jnp, config.dtype), getattr(jnp, config.weight_dtype)

    def _lin(in_f, out_f, name, axes):
      km = {"sharding": sharding(name, axes)} if sharding else {}
      return nnx.Linear(
          in_f,
          out_f,
          use_bias=False,
          dtype=dt,
          param_dtype=p_dt,
          kernel_metadata=km,
          rngs=rngs,
      )

    self.gate_proj = _lin(h, inter, "gate_proj", ("embed", "mlp"))
    self.up_proj = _lin(h, inter, "up_proj", ("embed", "mlp"))
    self.down_proj = _lin(inter, h, "down_proj", ("mlp", "embed"))

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jnp.minimum(self.gate_proj(x), self.swiglu_limit)
    up = jnp.clip(self.up_proj(x), -self.swiglu_limit, self.swiglu_limit)
    return self.down_proj(jax.nn.silu(gate) * up)


class Glm5TopkRouter(nnx.Module):
  """Top-k sigmoid router with non-Param e_score_correction_bias."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    if (sf := config.scoring_func) != "sigmoid":
      raise ValueError(f"scoring_func must be 'sigmoid': {sf!r}.")

    self.hidden_dim = h = config.hidden_size
    self.num_experts = e = config.n_routed_experts
    self.top_k, self.num_group = config.num_experts_per_tok, config.n_group
    self.topk_group = config.topk_group
    self.routed_scaling_factor = config.routed_scaling_factor
    self.norm_topk_prob = config.norm_topk_prob

    p_dt = getattr(jnp, config.weight_dtype)
    w_spec = sharding("router_weight", ("embed", None)) if sharding else None
    b_spec = sharding("router_bias", (None,)) if sharding else None
    self.weight = nnx.Param(jnp.zeros((h, e), dtype=p_dt), sharding=w_spec)
    self.e_score_correction_bias = Glm5RouterBias(
        jnp.zeros((e,), dtype=jnp.float32),
        sharding=b_spec,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    h_flat = hidden_states.reshape(-1, self.hidden_dim).astype(jnp.float32)
    router_logits = jnp.matmul(h_flat, self.weight[...].astype(jnp.float32))
    scores = jax.nn.sigmoid(router_logits)
    bias_arr = self.e_score_correction_bias[...].astype(jnp.float32)
    scores_for_choice = jax.lax.stop_gradient(scores + bias_arr)

    exp_per_group = self.num_experts // self.num_group
    grouped = scores_for_choice.reshape(-1, self.num_group, exp_per_group)
    group_scores = jnp.sum(jax.lax.top_k(grouped, 2)[0], axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, self.topk_group)
    g_arange = jnp.arange(self.num_group)[None, :]
    group_mask = jnp.any(g_arange[:, :, None] == group_idx[:, None, :], axis=-1)
    score_mask = jnp.repeat(group_mask, exp_per_group, axis=-1)
    masked_scores = jnp.where(
        score_mask,
        scores_for_choice,
        jnp.finfo(jnp.float32).min,
    )

    _, topk_indices = jax.lax.top_k(masked_scores, self.top_k)
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
    if self.norm_topk_prob:
      denom = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
      topk_weights = topk_weights / denom
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices


class Glm5Experts(nnx.Module):
  """Collection of expert weights stored as 3D tensors."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    e, h = config.n_routed_experts, config.hidden_size
    inter = config.moe_intermediate_size
    self.num_experts, self.top_k = e, config.num_experts_per_tok
    self.swiglu_limit = config.mlp_clamp_limit

    p_dt = getattr(jnp, config.weight_dtype)
    gu_axes, d_axes = ("expert", "embed", "mlp"), ("expert", "mlp", "embed")
    gu_spec = sharding("gate_up_proj", gu_axes) if sharding else None
    d_spec = sharding("down_proj", d_axes) if sharding else None

    self.gate_up_proj = nnx.Param(
        jnp.zeros((e, h, 2 * inter), dtype=p_dt),
        sharding=gu_spec,
    )
    self.down_proj = nnx.Param(
        jnp.zeros((e, inter, h), dtype=p_dt),
        sharding=d_spec,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      topk_indices: jax.Array,
      topk_weights: jax.Array,
  ) -> jax.Array:
    flat_idx = topk_indices.reshape(-1)
    flat_w = topk_weights.reshape(-1, 1).astype(hidden_states.dtype)
    rep_h = jnp.repeat(hidden_states, self.top_k, axis=0)
    perm = jnp.argsort(flat_idx)
    sorted_h, inv_perm = rep_h[perm], jnp.argsort(perm)
    group_sizes = jnp.bincount(flat_idx, length=self.num_experts)

    dt = hidden_states.dtype
    gu_w = self.gate_up_proj[...].astype(dt)
    gate_up = jax.lax.ragged_dot(sorted_h, gu_w, group_sizes=group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    act = jax.nn.silu(gate) * up
    d_w = self.down_proj[...].astype(dt)
    sorted_out = jax.lax.ragged_dot(act, d_w, group_sizes=group_sizes)
    n = hidden_states.shape[0]
    weighted = (sorted_out[inv_perm] * flat_w).reshape(n, self.top_k, -1)
    return jnp.sum(weighted, axis=1)


class Glm5MoE(nnx.Module):
  """Mixed expert module with routed experts and shared experts."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    self.experts = Glm5Experts(config, sharding=sharding)
    self.gate = Glm5TopkRouter(config, sharding=sharding)
    self.shared_experts = Glm5MLP(
        config,
        config.moe_intermediate_size * config.n_shared_experts,
        rngs=rngs,
        sharding=sharding,
    )

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
    orig_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, orig_shape[-1])
    router_logits, topk_weights, topk_indices = self.gate(h_flat)
    expert_out = self.experts(h_flat, topk_indices, topk_weights)
    expert_out = expert_out.reshape(orig_shape)
    shared_out = self.shared_experts(hidden_states)
    return expert_out + shared_out, router_logits


def compute_moe_load_balancing_loss(
    gate_logits: tuple[jax.Array, ...] | list[jax.Array],
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes auxiliary load balancing loss matching HF load_balancing_loss_func."""
  if not gate_logits:
    raise ValueError("gate_logits must be non-empty.")

  tokens_sum = prob_sum = jnp.zeros((num_experts,), dtype=jnp.float32)
  total_rows = 0.0

  flat_mask = None
  if attention_mask is not None:
    flat_mask = attention_mask.reshape(-1).astype(jnp.float32)

  for layer_gate in gate_logits:
    routing_weights = jax.nn.softmax(layer_gate.astype(jnp.float32), axis=-1)
    selected_flat = jax.lax.top_k(routing_weights, top_k)[1].reshape(-1)

    if flat_mask is None:
      b_cnt = jnp.bincount(selected_flat, length=num_experts)
      probs, rows = routing_weights, routing_weights.shape[0]
    else:
      mask_rep = jnp.repeat(flat_mask, top_k)
      b_cnt = jnp.bincount(selected_flat, weights=mask_rep, length=num_experts)
      probs, rows = routing_weights * flat_mask[:, None], jnp.sum(flat_mask)

    tokens_sum = tokens_sum + b_cnt.astype(jnp.float32)
    prob_sum, total_rows = prob_sum + jnp.sum(probs, axis=0), total_rows + rows

  t_exp, r_prob = tokens_sum / total_rows, prob_sum / total_rows
  return jnp.sum(t_exp * r_prob) * num_experts
