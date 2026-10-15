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

"""Loss functions for Qwen3.8-Flash-Next including primary CE, router aux, and MTP."""

from typing import Any
import jax
import jax.numpy as jnp


def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
  """Computes cross-entropy loss between logits and target token IDs in float32."""
  if logits.size == 0 or targets.size == 0:
    raise ValueError("logits and targets must be non-empty.")
  if logits.shape[:-1] != targets.shape:
    raise ValueError(f"logits prefix shape {logits.shape[:-1]} must match targets shape {targets.shape}.")
  if mask is not None and mask.shape != targets.shape:
    raise ValueError(f"mask shape {mask.shape} must match targets shape {targets.shape}.")
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  loss = -jnp.take_along_axis(log_probs, jnp.expand_dims(targets, -1), axis=-1).squeeze(-1)
  if mask is not None:
    m = mask.astype(jnp.float32)
    return jnp.sum(loss * m) / jnp.maximum(jnp.sum(m), 1.0)
  return jnp.mean(loss)


def router_load_balancing_loss(
    aux_list: Any,
    num_experts: int,
    mask: jax.Array | None = None,
    mtp_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes MoE router load balancing auxiliary loss across decoder layers."""
  if aux_list is None:
    raise ValueError("aux_list cannot be None.")
  if isinstance(aux_list, dict):
    if "layer_aux" not in aux_list:
      raise ValueError("aux_list dict must contain 'layer_aux'.")
    layer_items = [(layer, mask) for layer in aux_list["layer_aux"]]
    if aux_list.get("mtp_aux") is not None:
      layer_items.append((aux_list["mtp_aux"], mtp_mask if mtp_mask is not None else mask))
  elif isinstance(aux_list, (list, tuple)):
    layer_items = [(layer, mask) for layer in aux_list]
  else:
    raise ValueError(f"Unsupported aux_list type: {type(aux_list)}.")
  if not layer_items:
    raise ValueError("aux_list contains no layers.")
  for i, (layer_aux, _) in enumerate(layer_items):
    if not isinstance(layer_aux, dict) or "router_probs" not in layer_aux or "router_indices" not in layer_aux:
      raise ValueError(f"Layer {i} aux must be a dict with 'router_probs' and 'router_indices'.")

  tokens_per_expert_sum = jnp.zeros(num_experts, dtype=jnp.float32)
  router_prob_sum = jnp.zeros(num_experts, dtype=jnp.float32)
  total_rows = jnp.array(0.0, dtype=jnp.float32)

  for layer_aux, layer_mask in layer_items:
    r_probs = layer_aux["router_probs"].astype(jnp.float32)
    r_idx = layer_aux["router_indices"]
    oh = jax.nn.one_hot(r_idx, num_experts, dtype=jnp.float32)
    oh_reduce_axes = tuple(range(oh.ndim - 1))
    prob_reduce_axes = tuple(range(r_probs.ndim - 1))
    if layer_mask is not None:
      m = layer_mask.astype(jnp.float32)
      tokens_per_expert_sum = tokens_per_expert_sum + jnp.sum(oh * jnp.expand_dims(m, (-1, -2)), axis=oh_reduce_axes)
      router_prob_sum = router_prob_sum + jnp.sum(r_probs * jnp.expand_dims(m, -1), axis=prob_reduce_axes)
      total_rows = total_rows + jnp.sum(m)
    else:
      tokens_per_expert_sum = tokens_per_expert_sum + jnp.sum(oh, axis=oh_reduce_axes)
      router_prob_sum = router_prob_sum + jnp.sum(r_probs, axis=prob_reduce_axes)
      total_rows = total_rows + float(r_probs.size // num_experts)

  denom = jnp.maximum(total_rows, 1.0)
  tokens_per_expert = tokens_per_expert_sum / denom
  router_prob_per_expert = router_prob_sum / denom
  return (num_experts * jnp.sum(tokens_per_expert * router_prob_per_expert)).astype(jnp.float32)


def compute_total_loss(
    model: Any,
    input_tokens: jax.Array,
    target_tokens: jax.Array,
    *,
    target_mask: jax.Array | None = None,
    positions: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    router_aux_coeff: float | None = None,
    mtp_loss_coeff: float | None = None,
    use_mtp: bool = True,
    **kwargs: Any,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes primary CE loss, router load balancing loss, and optional MTP loss."""
  if router_aux_coeff is None:
    router_aux_coeff = getattr(model.config, "load_balance_loss_weight", None)
  if router_aux_coeff is None:
    raise ValueError("router_aux_coeff must be specified or present in model.config.load_balance_loss_weight.")
  router_aux_coeff = float(router_aux_coeff)

  run_mtp, need_aux = use_mtp and (model.mtp is not None), router_aux_coeff > 0.0
  out = model(input_tokens, positions, segment_ids, return_aux=need_aux, return_mtp=run_mtp, **kwargs)
  logits = out[0] if (run_mtp or need_aux) else out
  aux = out[1] if (run_mtp or need_aux) else None
  mtp_logits = aux["mtp_logits"] if (run_mtp and need_aux) else (out[1] if run_mtp else None)

  primary_loss = cross_entropy_loss(logits, target_tokens, mask=target_mask)

  mtp_aux_mask = None
  mtp_loss = None
  if run_mtp and mtp_logits is not None:
    if mtp_logits.shape[1] <= 1:
      mtp_loss = jnp.array(0.0, dtype=jnp.float32)
      mtp_aux_mask = jnp.zeros_like(target_tokens, dtype=jnp.float32)
    else:
      m = (
          target_mask[:, 1:].astype(jnp.float32)
          if target_mask is not None
          else jnp.ones_like(target_tokens[:, 1:], dtype=jnp.float32)
      )
      if segment_ids is not None:
        m = m * ((segment_ids[:, :-1] == segment_ids[:, 1:]) & (segment_ids[:, :-1] != 0)).astype(jnp.float32)
      mtp_loss = cross_entropy_loss(mtp_logits[:, :-1, :], target_tokens[:, 1:], mask=m)
      mtp_aux_mask = jnp.pad(m, ((0, 0), (0, 1)))

  router_loss = (
      router_load_balancing_loss(
          aux,
          int(model.config.num_experts),
          mask=target_mask,
          mtp_mask=mtp_aux_mask,
      )
      if need_aux
      else jnp.array(0.0, dtype=jnp.float32)
  )
  total_loss = primary_loss + router_aux_coeff * router_loss
  metrics = {"total_loss": total_loss, "primary_loss": primary_loss, "router_loss": router_loss}

  if mtp_loss is not None:
    if mtp_loss_coeff is None:
      mtp_loss_coeff_val = getattr(model.config, "mtp_loss_scaling_factor", None)
    else:
      mtp_loss_coeff_val = mtp_loss_coeff
    if mtp_loss_coeff_val is None:
      raise ValueError(
          "mtp_loss_coeff must be specified or present in model.config.mtp_loss_scaling_factor when MTP is enabled."
      )
    alpha = float(mtp_loss_coeff_val)
    total_loss = total_loss + alpha * mtp_loss
    metrics["total_loss"], metrics["mtp_loss"] = total_loss, mtp_loss

  return total_loss, metrics
