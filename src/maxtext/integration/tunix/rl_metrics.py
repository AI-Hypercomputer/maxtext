# Copyright 2026 Google LLC
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

"""Compact MoE observations for the Tunix GRPO/GSPO loss.

Tunix's log-probability helper discards model auxiliary outputs. Compute the
actor logits once, collect its routing observations, then let the original
Tunix loss consume those logits. This preserves its masks, sampling temperature,
reference KL, importance correction and loss aggregation without copying the
algorithm or running the Transformer a second time.
"""

import functools

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.sft import utils as sft_utils

from maxtext.layers.moe import RoutedMoE


class _PolicyLogits(nnx.Module):
  """NNX-compatible, differentiable input to Tunix's existing log-prob helper."""

  def __init__(self, logits):
    self.logits = nnx.Variable(logits)

  def __call__(self, input_tokens, positions, cache, attention_mask, segment_ids=None, forced_routed_experts=None):  # pylint: disable=arguments-differ
    del input_tokens, positions, cache, attention_mask, segment_ids, forced_routed_experts
    return self.logits[...], None


def routing_statistics(entropy, indices, token_mask, num_experts):
  """Per-layer statistics on global arrays, with padding excluded.

  Leading dimensions enumerate scanned layers; final dimensions are [B, S]
  (entropy) and [B, S, K] (expert ids). Counts are reduced before nonlinear load
  statistics, so GSPMD inserts global reductions for sharded batch/length axes.
  """
  mask = jnp.broadcast_to(token_mask, entropy.shape).astype(jnp.float32)
  entropy = entropy.reshape((-1, token_mask.size))
  mask = mask.reshape(entropy.shape)
  indices = indices.reshape((entropy.shape[0], token_mask.size, -1))

  # Bincount avoids a [tokens, top_k, experts] one-hot allocation.
  def count(ids, weights):
    return jnp.bincount(
        jnp.clip(ids, 0, num_experts - 1).reshape(-1),
        weights=(weights[:, None] * ((ids >= 0) & (ids < num_experts))).reshape(-1),
        length=num_experts,
    )

  counts = jax.vmap(count)(indices, mask)
  tokens = mask.sum(axis=-1)
  mean_entropy = jnp.where(mask > 0, entropy, 0).sum(axis=-1) / jnp.maximum(tokens, 1)
  assignments = counts.sum(axis=-1)
  loads = counts / jnp.maximum(assignments[:, None], 1)
  load_entropy = -jnp.sum(loads * jnp.log(jnp.maximum(loads, 1e-30)), axis=-1)
  mean_load = counts.mean(axis=-1)
  # A one-expert distribution has zero entropy; avoid dividing by log(1).
  entropy_scale = jnp.log(float(max(num_experts, 2)))
  return {
      "router_entropy": mean_entropy,
      "router_entropy_normalized": mean_entropy / entropy_scale,
      "expert_load_entropy_normalized": load_entropy / entropy_scale,
      "expert_load_cv": counts.std(axis=-1) / jnp.maximum(mean_load, 1e-30),
      "max_to_mean_load": counts.max(axis=-1) / jnp.maximum(mean_load, 1e-30),
      "unused_expert_fraction": jnp.where(assignments > 0, (counts == 0).mean(axis=-1), 0),
      "valid_tokens": tokens,
  }


def _observed_loss(model, train_example, algo_config, *, policy_loss, pad_id, eos_id, supported, total):
  """Runs the actor once and augments the unchanged Tunix objective with observations."""
  # Match the model inputs of Tunix common.compute_per_token_logps exactly.
  token_mask = None
  attention_mask = getattr(train_example, "completion_attention_mask", None)
  if isinstance(attention_mask, (jax.Array, np.ndarray)):
    token_mask = jnp.concatenate([train_example.prompt_mask, attention_mask], axis=1)
  segment_ids = getattr(train_example, "segment_ids", None)
  tokens, positions, attention, input_segments = common.process_ids(
      train_example.prompt_ids,
      train_example.completion_ids,
      pad_id,
      eos_id,
      segment_ids,
      getattr(train_example, "segment_positions", None),
      token_mask,
  )
  graph, state = nnx.split(model)
  observed = nnx.merge(graph, state)
  routed_experts = getattr(train_example, "routed_experts", None)
  if routed_experts is not None:
    routed_experts = jnp.asarray(routed_experts, dtype=jnp.int32)
  logits, _ = observed(
      tokens,
      positions=positions,
      cache=None,
      attention_mask=attention,
      segment_ids=segment_ids if segment_ids is not None else input_segments,
      forced_routed_experts=routed_experts,
  )
  result = policy_loss(_PolicyLogits(logits), train_example, algo_config)
  intermediates = dict(nnx.state(observed, nnx.Intermediate).flat_state())
  layer_metrics = {}
  for path, value in intermediates.items():
    if path[-1] != "rl_router_entropy":
      continue
    ids = intermediates[path[:-1] + ("rl_expert_indices",)][...]
    valid_tokens = segment_ids > 0 if segment_ids is not None else input_segments != 0
    statistics = routing_statistics(value[...], ids, valid_tokens, model.base.config.num_experts)
    for key, values in statistics.items():
      layer_metrics.setdefault(key, []).append(values)
  aux = dict(result.aux_metrics)
  aux["moe/observed_router_fraction"] = jnp.float32(supported / total)
  for key, values in layer_metrics.items():
    values = jnp.concatenate(values)
    aux[f"moe/{key}_mean_layer"] = jax.lax.stop_gradient(jnp.mean(values))
    if key == "router_entropy":
      aux["moe/router_entropy_min_layer"] = jax.lax.stop_gradient(jnp.min(values))
  return sft_utils.LossOutput(primary_loss=result.primary_loss, aux_metrics=aux)


def configure_moe_metrics(cluster, enabled=False):
  """Wraps only the actor loss; unsupported routers remain unobserved."""
  if not enabled:
    return
  if cluster.cluster_config.training_config.compute_logps_chunk_size:
    raise ValueError("rl.record_moe_metrics requires compute_logps_chunk_size=0.")
  model = cluster.actor_trainer.model
  total = supported = 0
  for _, module in nnx.iter_graph(model):
    if isinstance(module, RoutedMoE):
      total += 1
      cfg = module.config
      module.record_rl_metrics = not (
          module.is_hash_routing
          or cfg.use_random_routing
          or cfg.te_moe_block
          or cfg.attention in ("vllm_rpa", "vllm_batched_rpa")
      )
      supported += int(module.record_rl_metrics)
  if not total:
    return
  # Preserve the learner-installed loss, including its bound algorithm config
  # and options. Do not select a replacement objective from the registry.
  loss = functools.partial(
      _observed_loss,
      policy_loss=cluster.actor_trainer.loss_fn,
      pad_id=cluster.rollout.pad_id(),
      eos_id=cluster.rollout.eos_id(),
      supported=supported,
      total=total,
  )

  def loss_fn(model, train_example, algo_config):
    return loss(model, train_example, algo_config)

  cluster.actor_trainer.with_loss_fn(loss_fn, has_aux=True)
