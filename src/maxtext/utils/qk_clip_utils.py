# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for QK-Clip (Muon Clip)."""

import jax
import jax.numpy as jnp


def _get_key_name(k):
  """Helper to unwrap JAX path keys."""
  if hasattr(k, "key"):
    return k.key
  if hasattr(k, "idx"):
    return k.idx
  return k


def calculate_max_logit_metric(intermediate_outputs):
  """Extracts and computes the global maximum logit from intermediate outputs.

  Args:
    intermediate_outputs: A pytree containing model intermediates, potentially
      including 'max_logits' sowed by Attention layers.

  Returns:
    The global maximum logit scalar, or None if no logits were found.
  """
  all_max_logits = []

  def extract_logits(path, val):
    # 'sow' stores values in a tuple/list. tree_map descends into it.
    # The path to the leaf array will look like: (..., 'max_logits', 0)
    # So we check if the parent key (path[-2]) is 'max_logits'.
    if len(path) >= 2:
      parent_key = _get_key_name(path[-2])
      if parent_key == "max_logits":
        all_max_logits.append(val)

  jax.tree_util.tree_map_with_path(extract_logits, intermediate_outputs)

  if not all_max_logits:
    return None

  # Compute max per layer first to handle potential shape mismatches
  return jnp.max(jnp.stack([jnp.max(x) for x in all_max_logits]))


def apply_qk_clip(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights based on max_logits.

  Iterates over parameters. If a parameter belongs to an MLA attention layer,
  it finds the corresponding max_logits statistics from intermediate_outputs,
  calculates the clipping factor, and applies it to W_q and W_k components.

  Args:
    state: The current training state containing model parameters.
    intermediate_outputs: A dictionary of intermediate outputs from the model
      forward pass. It is expected to contain 'max_logits' entries sowed by
      Attention layers if QK-Clip is enabled.
    config: The model configuration object, containing QK-Clip hyperparameters
      (e.g. qk_clip_threshold, qk_nope_head_dim) and attention_type.

  Returns:
    A new training state with updated (clipped) parameters.

  Raises:
    ValueError: If the configured attention_type is not 'mla'.
  """
  if getattr(config, "attention_type", None) != "mla":
    raise ValueError(
        f"QK-Clip is only supported for MLA attention (attention_type='mla'). "
        f"Current configuration: {getattr(config, 'attention_type', 'None')}"
    )

  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    """Applies QK-Clip to a single parameter if it's an MLA projection weight.

    Args:
      path: A tuple of JAX Key objects representing the hierarchy path to the parameter in the state PyTree.
      param: The actual JAX array (weight tensor) at the given path.

    Returns:
      The scaled parameter if it is an MLA projection ('wq_b' or 'wkv_b'), otherwise the original parameter.
    """
    # Skip irrelevant weights (embeddings, norms, etc.).
    # We only care about specific MLA projection matrices ('wq_b', 'wkv_b').
    if len(path) < 2:
      return param

    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param

    # Search for max_logits in intermediate_outputs
    curr = intermediate_outputs.get("intermediates", intermediate_outputs)
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param  # Path not found in intermediates, skip

    if not isinstance(curr, dict) or "max_logits" not in curr:
      return param

    # max_logits was sowed as a tuple (array,)
    # shape: [batch, num_heads]
    max_logits_sowed = curr["max_logits"]
    if not max_logits_sowed:
      return param

    max_logits_batch = max_logits_sowed[0]

    # Calculate S_max (per head)
    # We want the global maximum across the batch dimension.
    # Result shape: [num_heads]
    s_max = jnp.max(max_logits_batch, axis=0)

    # Calculate scaling factor gamma
    # gamma = tau / s_max. Clip if s_max > tau.
    scale = jnp.minimum(1.0, tau / (s_max + 1e-6))

    # Apply qk clipping based on weight type
    if layer_name == "wq_b":
      # MLA Up-projection for Query [rank, heads, q_head_dim]
      qk_nope = config.qk_nope_head_dim
      w_qc = param[..., :qk_nope]
      w_qr = param[..., qk_nope:]
      scale_b = scale[None, :, None]  # Broadcast: [1, heads, 1]
      w_qc_new = w_qc * jnp.sqrt(scale_b)
      w_qr_new = w_qr * scale_b
      return jnp.concatenate([w_qc_new, w_qr_new], axis=-1)

    elif layer_name == "wkv_b":
      # MLA Up-projection for Key/Value [rank, heads, kv_head_dim]
      qk_nope = config.qk_nope_head_dim
      w_kc = param[..., :qk_nope]
      w_v = param[..., qk_nope:]
      scale_b = scale[None, :, None]
      w_kc_new = w_kc * jnp.sqrt(scale_b)
      return jnp.concatenate([w_kc_new, w_v], axis=-1)

    return param

  # Apply transformation
  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)
