# Copyright 2023–2025 Google LLC
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

"""Adapter for integrating MaxText Transformer models with Tunix.

This module provides the `TunixMaxTextAdapter` class, which wraps a MaxText
Transformer model to expose a call signature compatible with Tunix Trainers.
It also handles weight mapping for compatibility with Hugging Face models.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from flax import nnx
from jax import Array
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=ungrouped-imports
from maxtext.integration.tunix.utils import VllmWeightMapping
from maxtext.models.models import Transformer


# --- MONKEY-PATCH Weight Synchronization Bug in Tunix ---
try:
  from collections import abc
  import re
  import gc
  from typing import Any
  from flax import traverse_util
  from maxtext.utils import max_logging
  from tunix.generate import utils as tunix_utils

  def patched_transfer_state_directly(
      src_state,
      dst_state,
      reshard_fn,
      scan_axis: int = 1,
      delete_dst_buffers: bool = False,
      reshard_chunk_size: Any = None,
  ):
    max_logging.log("MONKEY-PATCH: transfer_state_directly running!")
    if delete_dst_buffers:
      if hasattr(tunix_utils, '_delete_target_buffers'):
        tunix_utils._delete_target_buffers(dst_state, src_state)
      gc.collect()

    def safe_has_key(obj, key: str) -> bool:
      if isinstance(obj, abc.Mapping):
        return key in obj
      return hasattr(obj, key)

    # Unwrap 'base' from src_state
    if isinstance(src_state, abc.Mapping) and safe_has_key(src_state, 'base'):
      max_logging.log("MONKEY-PATCH: Unwrapping 'base' from source state.")
      src_state = src_state['base']

    # Unwrap 'model' from dst_state
    while isinstance(dst_state, abc.Mapping) and safe_has_key(dst_state, 'model'):
      max_logging.log("MONKEY-PATCH: Unwrapping 'model' from destination state.")
      dst_state = dst_state['model']

    def to_pure_spec(node: Any) -> Any:
      if hasattr(node, 'to_pure_dict'):
        node = node.to_pure_dict()
      if isinstance(node, abc.Mapping):
        return {k: to_pure_spec(v) for k, v in node.items()}
      if isinstance(node, nnx.Variable):
        return to_pure_spec(node[...])
      if hasattr(node, 'value'):
        return node.value
      return node

    def intersect_trees(src, tgt_spec):
      if not isinstance(src, abc.Mapping) or not isinstance(tgt_spec, abc.Mapping):
        return src, tgt_spec

      src_flat = traverse_util.flatten_dict(src)
      tgt_flat = traverse_util.flatten_dict(tgt_spec)
      src_flat = tunix_utils._fuse_moe_weights(src_flat, tgt_flat)

      filtered_src_flat = {}
      filtered_tgt_flat = {}
      unstacked_cache = {}
      layer_pattern = re.compile(r'^layers_(\d+)$')

      for key_tuple, tgt_val in tgt_flat.items():
        path_str = '.'.join(str(k) for k in key_tuple)
        if key_tuple in src_flat:
          src_val = src_flat[key_tuple]
          src_val = tunix_utils._apply_dtype_cast(src_val, tgt_val.dtype, path_str)
          src_val = tunix_utils._align_to_model_shape(src_val, tgt_val, path_str)
          filtered_src_flat[key_tuple] = src_val
          filtered_tgt_flat[key_tuple] = tgt_val
          continue

        # Try scanned layer mapping
        layer_idx = -1
        match_index = -1
        for i, part in enumerate(key_tuple):
          if isinstance(part, str) and part.startswith('layers_'):
            m = layer_pattern.match(part)
            if m:
              layer_idx = int(m.group(1))
              match_index = i
              break

        if match_index != -1:
          candidate_a = list(key_tuple)
          candidate_a[match_index] = 'layers'
          candidate_b = list(key_tuple)
          candidate_b.pop(match_index)

          found_candidate = None
          for cand in [tuple(candidate_a), tuple(candidate_b)]:
            if cand in src_flat:
              found_candidate = cand
              break

          if found_candidate:
            cache_key = (found_candidate, tgt_val.shape, 'aligned')
            if cache_key not in unstacked_cache:
              src_val = src_flat[found_candidate]
              candidate_path = '.'.join(str(k) for k in found_candidate)
              src_val = tunix_utils._apply_dtype_cast(src_val, tgt_val.dtype, candidate_path)
              scanned_per_layer_shape = src_val.shape[:scan_axis] + src_val.shape[scan_axis + 1:]
              if scanned_per_layer_shape == tgt_val.shape:
                unstacked_cache[cache_key] = tunix_utils._unstack_scanned_param(
                    src_val, tgt_val, candidate_path, scan_axis=scan_axis
                )
              else:
                unstacked_cache[cache_key] = tunix_utils._bulk_align_and_unstack(
                    src_val, scan_axis, tgt_val, candidate_path
                )

            sliced_val = unstacked_cache[cache_key][layer_idx]
            sliced_val = tunix_utils._align_to_model_shape(sliced_val, tgt_val, path_str)
            filtered_src_flat[key_tuple] = sliced_val
            filtered_tgt_flat[key_tuple] = tgt_val
            continue

      return (
          traverse_util.unflatten_dict(filtered_src_flat),
          traverse_util.unflatten_dict(filtered_tgt_flat),
      )

    full_source_dict = to_pure_spec(src_state)
    full_target_spec = to_pure_spec(dst_state)

    final_source, final_spec = intersect_trees(full_source_dict, full_target_spec)

    dst_shardings_flat = {
        k: tunix_utils._snapshot_dst_sharding(
            tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
        )
        for k, tgt_val in traverse_util.flatten_dict(final_spec).items()
    }

    resharded_weights = reshard_fn(
        source=final_source,
        target=traverse_util.unflatten_dict(dst_shardings_flat),
    )

    # Assign to target State
    if isinstance(dst_state, nnx.State):
      flat_resharded = traverse_util.flatten_dict(resharded_weights)
      dst_vars = {path: var for path, var in dst_state.flat_state()}
      for path, value in flat_resharded.items():
        if path in dst_vars:
          var = dst_vars[path]
          if isinstance(var, nnx.Variable):
            var.value = value
          else:
            dst_state[path] = value
    elif isinstance(dst_state, dict):
      flat_resharded = traverse_util.flatten_dict(resharded_weights)
      flat_dst = traverse_util.flatten_dict(dst_state)
      for path, value in flat_resharded.items():
        if path in flat_dst:
          var = flat_dst[path]
          if isinstance(var, nnx.Variable):
            var.value = value
          else:
            node = dst_state
            for part in path[:-1]:
              node = node[part]
            node[path[-1]] = value
    else:
      nnx.update(dst_state, resharded_weights)

    gc.collect()
    max_logging.log("MONKEY-PATCH: transfer_state_directly finished successfully!")

  tunix_utils.transfer_state_directly = patched_transfer_state_directly
  max_logging.log("MONKEY-PATCH: Successfully applied tunix weight sync patch inside MaxText!")
except Exception as e:
  from maxtext.utils import max_logging
  max_logging.log(f"MONKEY-PATCH ERROR: Failed to apply tunix weight sync patch: {e}")


class TunixMaxTextAdapter(nnx.Module):
  """Adapter exposing Tunix Trainer call signature over a Transformer model."""

  def __init__(
      self,
      base_model: Transformer,
      use_standalone_mappings: bool = True,
      use_no_op_mappings: bool = False,
      mesh: Optional[Any] = None,
  ):
    super().__init__()
    config = base_model.config
    if config and hasattr(config, "lora") and config.lora.enable_lora:
      from maxtext.utils import lora_utils
      from maxtext.utils import max_logging
      import types
      max_logging.log("Applying LoRA parameters to model via TunixMaxTextAdapter...")
      base_model = lora_utils.apply_lora_to_model(base_model, mesh, config)
      if config.lora.lora_restore_path:
        max_logging.log(f"Restoring LoRA parameters from path in TunixMaxTextAdapter: {config.lora.lora_restore_path}")
        lora_utils.restore_lora_from_path(base_model, config)

        # Mathematically premerge LoRA weights into base parameters so vLLM decoding is accurate
        import jax.numpy as jnp
        lora_rank = config.lora.lora_rank
        lora_alpha = config.lora.lora_alpha
        lora_scale_factor = lora_alpha / lora_rank
        max_logging.log(f"Premerging LoRA weights into base parameters with scale factor: {lora_scale_factor}...")

        def merge_lora_recursively(module):
          if hasattr(module, "kernel") and hasattr(module, "kernel_lora_a") and hasattr(module, "kernel_lora_b"):
            lora_a = module.kernel_lora_a.value
            lora_b = module.kernel_lora_b.value
            base_w = module.kernel.value
            if lora_a is not None and lora_b is not None and base_w is not None:
              if len(base_w.shape) == 3:  # Scanned layers: (input_dim, scan_dim, output_dim)
                delta = jnp.einsum("isr,rsd->isd", lora_a, lora_b)
              elif len(base_w.shape) == 2:  # Non-scanned layers: (input_dim, output_dim)
                delta = jnp.einsum("ir,rd->id", lora_a, lora_b)
              else:
                raise ValueError(f"Unexpected base weight shape: {base_w.shape}")
              module.kernel.value = base_w + delta * lora_scale_factor

          # Recurse into all submodules/attributes of module
          for name, attr in list(module.__dict__.items()):
            if isinstance(attr, nnx.Module):
              merge_lora_recursively(attr)
            elif isinstance(attr, dict):
              for k, v in attr.items():
                if isinstance(v, nnx.Module):
                  merge_lora_recursively(v)
            elif isinstance(attr, (list, tuple)):
              for item in attr:
                if isinstance(item, nnx.Module):
                  merge_lora_recursively(item)

        merge_lora_recursively(base_model)
        max_logging.log("Successfully premerged LoRA weights into base parameters!")

    self.base = base_model
    self._vllm_weight_mapping = VllmWeightMapping(
        self.base.config.model_name,
        HF_MODEL_CONFIGS[self.base.config.model_name].to_dict(),
        use_standalone_mappings,
    )
    self.use_no_op_mappings = use_no_op_mappings

  # ------------------------------------------------------------------ #
  # Tunix call signature
  # ------------------------------------------------------------------ #
  def __call__(
      self,
      input_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Optional[Any],  # Tunix currently passes None from Trainers
      attention_mask: Optional[Array],  # [B, L, L] or None
      decoder_segment_ids: Optional[Array] = None,
      output_hidden_states: bool = False,  # ignored
  ) -> Tuple[Array, None]:
    """Forward compatible with Tunix Trainers default loss.
    Returns logits, None.
    """
    logits = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return logits, None

  def to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_mapping()

  def to_hf_transpose_keys(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}

    return self._vllm_weight_mapping.lora_to_hf_mappings()
