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


def apply_qk_clip(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights based on max_logits.

  Iterates over parameters. If a parameter belongs to an MLA attention layer,
  it finds the corresponding max_logits statistics from intermediate_outputs,
  calculates the clipping factor, and applies it to W_q and W_k components.
  """
  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    # path matches keys in state.params, e.g. ('decoder', 'layers_0', 'self_attention', 'wq_b', 'kernel')

    # 1. Attempt to locate corresponding max_logits in intermediate_outputs
    # intermediate_outputs structure usually mirrors params but nested under 'intermediates' collection logic
    curr = intermediate_outputs
    try:
      # Traverse down to the layer level (e.g. 'self_attention')
      # We assume path structure: ['decoder', 'layers_X', 'self_attention', 'weight_name', 'kernel']
      # We stop 2 levels before end to get to 'self_attention' dict
      for key in path[:-2]:
        if isinstance(curr, dict) and key in curr:
          curr = curr[key]
        else:
          return param  # Not found in intermediates, skip

      if "max_logits" not in curr:
        return param

      # max_logits was sowed, so it's a tuple (array,)
      # shape: [batch, num_heads]
      max_logits_batch = curr["max_logits"][0]

      # 2. Calculate S_max (per head)
      # Reduce over batch dimension
      local_max = jnp.max(max_logits_batch, axis=0)

      # Sync across data parallelism to get global max per head
      # Assuming 'data' axis exists in the mesh
      s_max = jax.lax.pmax(local_max, axis_name="data")

      # 3. Calculate scaling factor gamma
      # gamma = tau / s_max. Clip if s_max > tau.
      scale = jnp.minimum(1.0, tau / (s_max + 1e-6))

      # 4. Apply clipping based on weight type
      layer_name = path[-2]  # e.g. 'wq_b' or 'wkv_b'

      if layer_name == "wq_b":
        # MLA Up-projection for Query
        # Shape: [rank, heads, q_head_dim]
        # q_head_dim = qk_nope + qk_rope
        qk_nope = config.qk_nope_head_dim

        w_qc = param[..., :qk_nope]
        w_qr = param[..., qk_nope:]

        # Reshape scale for broadcasting: [1, heads, 1]
        scale_b = scale[None, :, None]

        # W_qc <- W_qc * sqrt(gamma)
        # W_qr <- W_qr * gamma
        w_qc_new = w_qc * jnp.sqrt(scale_b)
        w_qr_new = w_qr * scale_b

        return jnp.concatenate([w_qc_new, w_qr_new], axis=-1)

      elif layer_name == "wkv_b":
        # MLA Up-projection for Key/Value
        # Shape: [rank, heads, kv_head_dim]
        # kv_head_dim = qk_nope + v_head_dim
        qk_nope = config.qk_nope_head_dim

        w_kc = param[..., :qk_nope]
        w_v = param[..., qk_nope:]

        # Reshape scale
        scale_b = scale[None, :, None]

        # W_kc <- W_kc * sqrt(gamma)
        # W_v is NOT clipped
        w_kc_new = w_kc * jnp.sqrt(scale_b)

        return jnp.concatenate([w_kc_new, w_v], axis=-1)

    except Exception:
      # If any structure mismatch or error, return param as is
      return param

    return param

  # Apply transformation
  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)
