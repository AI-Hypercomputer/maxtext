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
from maxtext.utils import max_logging


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
    if path and path[-1] == "max_logits":
      # val is tuple from sow: (max_logits_array,)
      all_max_logits.append(val[0])

  jax.tree_util.tree_map_with_path(extract_logits, intermediate_outputs)

  if not all_max_logits:
    return None

  layer_maxes = [jnp.max(m) for m in all_max_logits]
  global_max_logit = jnp.max(jnp.stack(layer_maxes))
  return global_max_logit


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
      (e.g. qk_clip_threshold, qk_nope_head_dim).

  Returns:
    A new training state with updated (clipped) parameters.
  """
  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    # path matches keys in state.params, e.g. ('decoder', 'layers_0', 'self_attention', 'wq_b', 'kernel')

    # 1. Attempt to locate corresponding max_logits in intermediate_outputs
    curr = intermediate_outputs
    try:
      # Traverse down to the layer level
      for node in path[:-2]:
        # Handle JAX tree_util Key objects if present
        if hasattr(node, "key"):
          key = node.key
        elif hasattr(node, "idx"):
          key = node.idx
        else:
          key = node

        if isinstance(curr, dict) and key in curr:
          curr = curr[key]
        else:
          return param  # Not found, skip

      if "max_logits" not in curr:
        return param

      # max_logits was sowed as a tuple (array,)
      # shape: [batch, num_heads]
      max_logits_batch = curr["max_logits"][0]

      # 2. Calculate S_max (per head)
      # We want the global maximum across the batch dimension.
      # In GSPMD (MaxText default), jnp.max on a sharded array automatically
      # handles the cross-device reduction (all-reduce).
      # Result shape: [num_heads]
      s_max = jnp.max(max_logits_batch, axis=0)

      # 3. Calculate scaling factor gamma
      # gamma = tau / s_max. Clip if s_max > tau.
      scale = jnp.minimum(1.0, tau / (s_max + 1e-6))

      # 4. Apply clipping based on weight type
      layer_node = path[-2]
      layer_name = layer_node.key if hasattr(layer_node, "key") else layer_node

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

    except Exception as e:  # pylint: disable=broad-except
      # Fail safe: return param as is if anything goes wrong
      max_logging.log(f"QK-Clip exception for param at path {path}: {e}")
      return param

    return param

  # Apply transformation
  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)
