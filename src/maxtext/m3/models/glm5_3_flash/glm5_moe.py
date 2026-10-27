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

from collections.abc import Sequence
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Sharding


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
          precision=jax.lax.Precision.HIGHEST,
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
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    if (sf := config.scoring_func) != "sigmoid":
      raise ValueError(f"scoring_func must be 'sigmoid': {sf!r}.")

    self.hidden_dim = h = config.hidden_size
    self.num_experts = e = config.n_routed_experts
    self.top_k, self.num_group = config.num_experts_per_tok, config.n_group
    self.topk_group = config.topk_group
    self.routed_scaling_factor = config.routed_scaling_factor
    self.norm_topk_prob = config.norm_topk_prob

    r_p_dt = getattr(jnp, config.router_param_dtype)
    w_spec = sharding("router_weight", ("embed", None)) if sharding else None
    b_spec = sharding("router_bias", (None,)) if sharding else None
    w_init = nnx.initializers.normal(0.02)(rngs.params(), (h, e), r_p_dt)
    self.weight = nnx.Param(w_init, sharding=w_spec)
    self.e_score_correction_bias = Glm5RouterBias(
        jnp.zeros((e,), dtype=r_p_dt),
        sharding=b_spec,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    h_flat = hidden_states.reshape(-1, self.hidden_dim).astype(jnp.float32)
    router_logits = jnp.matmul(
        h_flat,
        self.weight[...].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
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
  """Collection of expert weights stored as 3D/4D tensors.

  Replicated-EP (expert == 1) global-dispatch jax.lax.ragged_dot reference
  implementation (when data or fsdp > 1, GSPMD all-gathers the [B*S*K]
  token-expert indices for global argsort and ragged_dot; expert > 1 is rejected
  by Glm5Experts and Glm5MoE).
  """

  def __init__(
      self,
      config: Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    e, h = config.n_routed_experts, config.hidden_size
    inter = config.moe_intermediate_size
    self.hidden_size = h
    self.intermediate_size = inter
    self.num_experts, self.top_k = e, config.num_experts_per_tok
    self.swiglu_limit = config.mlp_clamp_limit
    self.sharding = sharding
    self.ep_axis = sharding.ep_axis if isinstance(sharding, Glm5Sharding) else "expert"
    if isinstance(sharding, Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5Experts expert parallelism ({self.ep_axis}={sharding.ep_size} >"
          " 1) requires shard_map EP dispatch (follow-up F34)."
      )
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError(
          "Glm5Experts expert parallelism"
          f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires shard_map"
          " EP dispatch (follow-up F34)."
      )

    p_dt = getattr(jnp, config.weight_dtype)
    gu_axes = ("expert", "embed", None, "mlp")
    d_axes = ("expert", "mlp", "embed")
    gu_spec = sharding("gate_up_proj", gu_axes) if sharding else None
    d_spec = sharding("down_proj", d_axes) if sharding else None

    gu_init = nnx.initializers.normal(0.02)(rngs.params(), (e, h, 2, inter), p_dt)
    self.gate_up_proj = nnx.Param(gu_init, sharding=gu_spec)

    d_init = nnx.initializers.normal(0.02)(rngs.params(), (e, inter, h), p_dt)
    self.down_proj = nnx.Param(d_init, sharding=d_spec)

  def __call__(
      self,
      hidden_states: jax.Array,
      topk_indices: jax.Array,
      topk_weights: jax.Array,
  ) -> jax.Array:
    if isinstance(self.sharding, Glm5Sharding) and self.sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map EP"
          f" dispatch (follow-up F34); got ep_size={self.sharding.ep_size}."
      )
    mesh = None
    x_shd = jax.typeof(hidden_states).sharding
    param_shd = jax.typeof(self.gate_up_proj.value).sharding
    if isinstance(x_shd, jax.sharding.NamedSharding):
      mesh = x_shd.mesh
    elif isinstance(param_shd, jax.sharding.NamedSharding):
      mesh = param_shd.mesh
    elif not jax.sharding.get_abstract_mesh().empty:
      mesh = jax.sharding.get_abstract_mesh()

    if mesh is not None and not mesh.empty:
      if self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
        raise NotImplementedError(
            f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map"
            " EP dispatch (follow-up F34); got mesh"
            f" {self.ep_axis}={mesh.shape[self.ep_axis]}."
        )

    flat_idx = topk_indices.reshape(-1)
    flat_w = topk_weights.reshape(-1, 1).astype(jnp.float32)
    perm = jnp.argsort(flat_idx)
    sorted_h = hidden_states[perm // self.top_k]
    inv_perm = jnp.argsort(perm)
    group_sizes = jnp.bincount(flat_idx, length=self.num_experts)

    dt = hidden_states.dtype
    gu = self.gate_up_proj[...].astype(dt)
    gate = jax.lax.ragged_dot(
        sorted_h,
        gu[:, :, 0, :],
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    up = jax.lax.ragged_dot(
        sorted_h,
        gu[:, :, 1, :],
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    act = jax.nn.silu(gate) * up
    d_w = self.down_proj[...].astype(dt)
    sorted_out = jax.lax.ragged_dot(
        act,
        d_w,
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    n = hidden_states.shape[0]
    weighted = (sorted_out[inv_perm].astype(jnp.float32) * flat_w).reshape(n, self.top_k, -1)
    return jnp.sum(weighted, axis=1).astype(dt)


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
    self.ep_axis = sharding.ep_axis if isinstance(sharding, Glm5Sharding) else "expert"
    if isinstance(sharding, Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5MoE expert parallelism ({self.ep_axis}={sharding.ep_size} > 1)"
          " requires all-to-all dispatch (planned for G8); single-host bringup"
          f" requires {self.ep_axis}=1."
      )
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError(
          "Glm5MoE expert parallelism"
          f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires"
          " all-to-all dispatch (planned for G8); single-host bringup requires"
          f" {self.ep_axis}=1."
      )
    self.experts = Glm5Experts(config, rngs=rngs, sharding=sharding)
    self.gate = Glm5TopkRouter(config, rngs=rngs, sharding=sharding)
    self.shared_experts = Glm5MLP(
        config,
        config.moe_intermediate_size * config.n_shared_experts,
        rngs=rngs,
        sharding=sharding,
    )

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    orig_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, orig_shape[-1])
    router_logits, topk_weights, topk_indices = self.gate(h_flat)
    expert_out = self.experts(h_flat, topk_indices, topk_weights)
    expert_out = expert_out.reshape(orig_shape)
    shared_out = self.shared_experts(hidden_states)
    topk_reshaped = topk_indices.reshape(orig_shape[:-1] + (self.config.num_experts_per_tok,))
    return expert_out + shared_out, (router_logits, topk_reshaped)


def init_moe_lb_stats(
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Initializes [E]-sized accumulators (tokens_sum, prob_sum, total_rows) for MoE LB loss."""
  z = jnp.zeros((num_experts,), dtype=jnp.float32)
  return z, z, jnp.zeros((), dtype=jnp.float32)


def accumulate_moe_lb_stats(
    stats: tuple[jax.Array, jax.Array, jax.Array],
    layer_gate: jax.Array,
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Accumulates one MoE layer's (bincount, prob_sum, rows) into [E]-sized carry stats.

  Matches HuggingFace's load_balancing_loss_func (Switch Transformer eq. 4-6):
  routing_weights = softmax(layer_gate, axis=-1) and selected_experts =
  top_k(routing_weights) without e_score_correction_bias.
  """
  tokens_sum, prob_sum, total_rows = stats
  gate_2d = layer_gate.reshape(-1, layer_gate.shape[-1]).astype(jnp.float32)
  routing_weights = jax.nn.softmax(gate_2d, axis=-1)
  _, selected_experts = jax.lax.top_k(routing_weights, top_k)
  selected_flat = selected_experts.reshape(-1)

  if attention_mask is None:
    b_cnt = jnp.bincount(selected_flat, length=num_experts)
    probs = routing_weights
    rows = jnp.asarray(routing_weights.shape[0], dtype=jnp.float32)
  else:
    flat_mask = attention_mask.reshape(-1).astype(jnp.float32)
    mask_rep = jnp.repeat(flat_mask, top_k)
    b_cnt = jnp.bincount(selected_flat, weights=mask_rep, length=num_experts)
    probs = routing_weights * flat_mask[:, None]
    rows = jnp.sum(flat_mask)

  return (
      tokens_sum + b_cnt.astype(jnp.float32),
      prob_sum + jnp.sum(probs, axis=0),
      total_rows + rows,
  )


def finalize_moe_lb_loss(
    stats: tuple[jax.Array, jax.Array, jax.Array],
    num_experts: int,
) -> jax.Array:
  """Finalizes MoE load-balancing loss scalar from accumulated [E]-sized stats."""
  tokens_sum, prob_sum, total_rows = stats
  safe_rows = jnp.maximum(total_rows, 1.0)
  t_exp, r_prob = tokens_sum / safe_rows, prob_sum / safe_rows
  return jnp.sum(t_exp * r_prob) * num_experts


def compute_moe_load_balancing_loss(
    gate_logits: Sequence[jax.Array],
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes auxiliary load balancing loss matching HF load_balancing_loss_func."""
  if not gate_logits:
    raise ValueError("gate_logits must be non-empty.")

  stats = init_moe_lb_stats(num_experts)
  for layer_gate in gate_logits:
    stats = accumulate_moe_lb_stats(
        stats,
        layer_gate,
        num_experts,
        top_k,
        attention_mask=attention_mask,
    )
  return finalize_moe_lb_loss(stats, num_experts)
