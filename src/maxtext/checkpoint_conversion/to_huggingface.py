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

"""Converts a MaxText checkpoint to a HuggingFace-compatible format.

This script supports three conversion modes:
1. Base: Converts a standard MaxText model to a full Hugging Face model.
   (Requires `load_parameters_path`)
2. Adapter: Converts a standalone MaxText LoRA checkpoint to HF PEFT format.
   (Requires `lora.lora_restore_path`)
3. Merged: Merges MaxText LoRA weights into the base model to produce a full HF model.
   (Requires both `load_parameters_path` and `lora.lora_restore_path`)

It is invoked using MaxText's pyconfig, which means you provide a base config
file and can override parameters on the command line.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `maxtext.utils.globals.HF_IDS`.
  load_parameters_path: (Required for Base/Merged) Path to the MaxText base
                        parameter-only checkpoint.
  lora.lora_restore_path: (Required for Adapter/Merged) Path to the MaxText
                          LoRA checkpoint directory.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.
  weight_dtype: (Optional) It affects the resulting Hugging Face weight dtype.
                Default value is `float32`. We recommend using `bfloat16`
                to save memory and speed up conversion.

Optional Flags:
  --override_model_architecture: If set, overrides the HF model configuration
                                 with values from the MaxText configuration
                                 (e.g., num_heads, hidden_size) instead of failing.

Environment Variables:
  HF_AUTH_TOKEN: (Required) A HuggingFace authentication token. This is needed
                 to download the correct tokenizer configuration and to upload
                 the converted model to the HuggingFace Hub if `base_output_directory`
                 is a Hub repo ID (e.g., "hf://my-user/my-model").

Example Usage:
  To merge a LoRA adapter into a base model and save as a full HF model:

  export HF_AUTH_TOKEN="hf_YOUR_TOKEN"
  python -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name="gemma3-4b" \
    load_parameters_path="/path/to/base/checkpoint/" \
    lora.lora_restore_path="/path/to/lora/checkpoint/" \
    base_output_directory="/path/to/output/" \
    scan_layers=True
"""

import ctypes
import gc
import os
import re
import time
from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np

from transformers import AutoTokenizer, AutoProcessor

from absl import app
from absl import flags

from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    MemoryMonitorTqdm,
    print_peak_memory,
    save_adapter_files,
    get_local_save_path_manager,
    upload_file_to_gcs,
    save_config_file,
    save_safetensor_file,
    save_index_file,
    SAFE_TENSORS_CONFIG_FILE,
    SAFE_TENSORS_INDEX_FILE,
    SAFE_TENSORS_WEIGHTS_FILE,
    DEFAULT_MAX_SHARD_SIZE,
    _process,
)
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.globals import HF_IDS
from maxtext.utils.lora_utils import sync_lora_metadata
from maxtext.utils.model_creation_utils import verify_and_sync_scan_layers


flags.DEFINE_bool(
    "override_model_architecture",
    False,
    "If True, overrides Hugging Face config architecture parameters (heads, layers, dims) "
    "with values from the MaxText config. If False, raises a ValueError on mismatch.",
)

flags.DEFINE_string(
    "hf_model_path",
    None,
    "(Optional) Customized remote HF repo (or local path) to source the tokenizer/processor "
    "from. If not specified, defaults to maxtext.utils.globals.HF_IDS[model_name]. Use this to "
    "bundle a different tokenizer than the default (e.g., the base model's tokenizer instead of "
    "the instruction-tuned one).",
)

flags.DEFINE_integer(
    "parallel_threads",
    4,
    "Number of threads used for parallel saving of weights. Lower this value to reduce peak host RAM usage.",
)
flags.register_validator(
    "parallel_threads",
    lambda value: value > 0,
    message="--parallel_threads must be a positive integer.",
)

FLAGS = flags.FLAGS


def _get_lora_delta(key, lora_state_dict, lora_scaling):
  """Calculates the LoRA delta for a given parameter key."""
  a_key, b_key = key + "_lora_a", key + "_lora_b"
  if a_key not in lora_state_dict and key.startswith("params-"):
    a_key, b_key = key[7:] + "_lora_a", key[7:] + "_lora_b"

  if a_key in lora_state_dict and b_key in lora_state_dict:
    data_a = jnp.asarray(lora_state_dict[a_key], dtype=jnp.float32)
    data_b = jnp.asarray(lora_state_dict[b_key], dtype=jnp.float32)

    is_attention = "attention" in key.lower() or "attn" in key.lower()

    if is_attention and data_a.ndim > 2:
      if data_a.ndim == 4:
        # Scanned attention projection: [num_layers, input_dim, heads, rank] & [num_layers, rank, heads, output_dim]
        return jnp.einsum("lipr,lrpo->lipo", data_a, data_b) * lora_scaling
      # Unscanned attention projection: [input_dim, heads, rank] & [rank, heads, output_dim]
      return jnp.einsum("ipr,rpo->ipo", data_a, data_b) * lora_scaling
    else:
      if data_a.ndim == 3:
        # Scanned standard linear projection: can be [num_layers, input_dim, rank] or [input_dim, num_layers, rank]
        rank = data_a.shape[2]
        if rank == data_b.shape[1] and rank != data_b.shape[0]:
          # Case A: [num_layers, input_dim, rank] & [num_layers, rank, output_dim]
          return jnp.einsum("lir,lro->lio", data_a, data_b) * lora_scaling
        elif rank == data_b.shape[0] and rank != data_b.shape[1]:
          # Case B: [input_dim, num_layers, rank] & [rank, num_layers, output_dim]
          return jnp.einsum("ilr,rlo->ilo", data_a, data_b) * lora_scaling
        else:
          # Disambiguate using key names (Case B is typically 'wo' or 'out-kernel' / 'out_proj')
          if any(term in key for term in ["wo", "out-kernel", "out_proj"]):
            return jnp.einsum("ilr,rlo->ilo", data_a, data_b) * lora_scaling
          else:
            return jnp.einsum("lir,lro->lio", data_a, data_b) * lora_scaling
      # Unscanned standard linear projection
      return jnp.matmul(data_a, data_b) * lora_scaling
  return None


def _get_model_mappings(
    model_name: str, scan_layers: bool, hf_config_dict: dict, maxtext_config: pyconfig.HyperParameters
):
  """Retrieves parameter, shape, and hook function mappings for the model.

  This handles both `atomic` keys and `composite_mt_key` architectures.
  A `composite_mt_key` occurs when multiple MaxText keys must be fused back into a
  single HF parameter (e.g., fusing MT `wi_0` and `wi_1` back into HF `gate_up_proj`).

  Args:
    model_name: The name of the model (e.g., "gemma2-2b").
    scan_layers: Boolean indicating if the model was trained with scanned layers.
    hf_config_dict: The Hugging Face model configuration dictionary.
    maxtext_config: The maxtext model configuration.

  Returns:
    A dictionary containing the parameter mapping, shape mapping, and hook
    function mapping required for the conversion.

  Raises:
    ValueError: If mappings for the specified `model_name` are not found.
  """
  if model_name not in PARAM_MAPPING or model_name not in HF_SHAPE or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  param_mapping = PARAM_MAPPING[model_name](hf_config_dict, maxtext_config, scan_layers)
  hook_fn_mapping = HOOK_FNS[model_name](hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True)

  # Promote composite hook keys into param_mapping.
  # If HOOK_FNS defines a composite tuple key (e.g., (wi_0, wi_1) for MoE gate_up_proj),
  # replace the individual entries in param_mapping with one composite entry so
  # process_maxtext_param receives both tensors together and passes them to the hook.
  for hook_key in list(hook_fn_mapping.keys()):
    if isinstance(hook_key, tuple):
      hf_path = param_mapping.get(hook_key[0])
      if hf_path is not None:
        param_mapping[hook_key] = hf_path
        for k in hook_key:
          param_mapping.pop(k, None)

  return {
      "param_mapping": param_mapping,
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": hook_fn_mapping,
  }


def _validate_or_update_architecture(hf_config, max_config, override: bool):
  """Validates consistency between HF and MaxText configs or overrides HF config if requested.

  Args:
    hf_config: The Hugging Face configuration object.
    max_config: The MaxText configuration object (HyperParameters).
    override: Boolean, if True, update hf_config with max_config values.
              If False, raise error on mismatch.
  """
  # Mapping from Hugging Face config attribute -> MaxText config attribute
  # Note: We use derived MaxText attributes (e.g. emb_dim) which account for scale factors.
  attributes_to_check = [
      ("hidden_size", "emb_dim"),
      ("intermediate_size", "mlp_dim"),
      ("kv_lora_rank", "kv_lora_rank"),
      ("moe_intermediate_size", "moe_mlp_dim"),
      ("n_routed_experts", "num_experts"),
      ("n_shared_experts", "shared_experts"),
      ("num_attention_heads", "num_query_heads"),
      ("num_experts", "num_experts"),
      ("num_experts_per_tok", "num_experts_per_tok"),
      ("num_hidden_layers", "num_decoder_layers"),
      ("num_key_value_heads", "num_kv_heads"),
      ("num_local_experts", "num_experts"),
      ("q_lora_rank", "q_lora_rank"),
      ("qk_nope_head_dim", "qk_nope_head_dim"),
      ("qk_rope_head_dim", "qk_rope_head_dim"),
      ("v_head_dim", "v_head_dim"),
      ("vocab_size", "vocab_size"),
      ("global_head_dim", "global_head_dim"),
      ("num_global_key_value_heads", "global_num_kv_heads"),
  ]

  if max_config.attention_type == "mla":
    attributes_to_check.extend(
        [
            ("qk_nope_head_dim", "qk_nope_head_dim"),
            ("qk_rope_head_dim", "qk_rope_head_dim"),
            ("v_head_dim", "v_head_dim"),
            ("kv_lora_rank", "kv_lora_rank"),
            ("q_lora_rank", "q_lora_rank"),
        ]
    )
  else:
    attributes_to_check.append(("head_dim", "head_dim"))

  mismatches = []
  target_cfg = getattr(hf_config, "text_config", hf_config) or hf_config

  for hf_attr, mt_attr in attributes_to_check:
    # Skip checks if MaxText config doesn't have the attribute (shouldn't happen for valid configs)
    if not hasattr(max_config, mt_attr):
      continue

    # Skip checks if the HF config doesn't have this attribute or raises AmbiguousGlobalPerLayerAttributeError
    try:
      hf_value = getattr(target_cfg, hf_attr)
    except (AttributeError, ValueError, RuntimeError):
      continue

    mt_value = getattr(max_config, mt_attr)

    # Handle None values
    if hf_value is None or mt_value is None:
      continue

    # Special handling for Gemma 2 where local and global layers are bundled
    if max_config.model_name.startswith("gemma2") and hf_attr == "num_hidden_layers":
      if isinstance(mt_value, int):
        mt_value = mt_value * 2

    # Special handling for Qwen3-MoE: hf.intermediate_size is the aggregated dense MLP dim, but mt.mlp_dim is dim per expert
    if "qwen3" in max_config.model_name and getattr(max_config, "num_experts", 0) > 1 and hf_attr == "intermediate_size":
      mt_value = mt_value * getattr(max_config, "num_experts_per_tok", 1)

    # Handle vocab size padding
    if hf_attr == "vocab_size" and isinstance(mt_value, int) and isinstance(hf_value, int):
      # MaxText often pads vocab size to a multiple of 128 or 256 for TPU efficiency
      if mt_value >= hf_value and (mt_value - hf_value) < 256:
        mt_value = hf_value

    # Compare values (with tolerance for floats)
    is_match = False
    if isinstance(hf_value, float) or isinstance(mt_value, float):
      try:
        is_match = abs(float(hf_value) - float(mt_value)) < 1e-6
      except (ValueError, TypeError):
        is_match = hf_value == mt_value
    else:
      is_match = hf_value == mt_value

    if not is_match:
      if override:
        max_logging.log(f"⚠️ Overwriting HF Config '{hf_attr}': {hf_value} -> {mt_value} (from MaxText '{mt_attr}')")
        setattr(target_cfg, hf_attr, mt_value)
      else:
        mismatches.append(f"{hf_attr} (HF={hf_value} vs MaxText={mt_value})")

  if mismatches:
    error_msg = (
        "Architecture mismatches detected between standard Hugging Face config and provided MaxText config:\n  - "
        + "\n  - ".join(mismatches)
        + "\n\nAction Required: Pass the flag `--override_model_architecture` to force the conversion using MaxText values."
    )
    raise ValueError(error_msg)


def _apply_yarn_rope_config(hf_config, max_config) -> None:
  """Apply MaxText YaRN RoPE values without dropping HF model-specific fields."""
  rope_scaling = getattr(hf_config, "rope_scaling", None)
  if isinstance(rope_scaling, dict):
    rope_scaling = dict(rope_scaling)
  else:
    rope_scaling = {}

  rope_type_key = "type" if "type" in rope_scaling and "rope_type" not in rope_scaling else "rope_type"
  rope_scaling.update(
      {
          "beta_fast": float(getattr(max_config, "beta_fast", 32.0)),
          "beta_slow": float(getattr(max_config, "beta_slow", 1.0)),
          "factor": float(getattr(max_config, "rope_factor", 32.0)),
          "original_max_position_embeddings": getattr(max_config, "original_max_position_embeddings", 4096),
          "rope_theta": getattr(max_config, "rope_max_timescale"),
          rope_type_key: "yarn",
          "truncate": getattr(max_config, "rope_truncate", False),
      }
  )

  hf_config.rope_theta = getattr(max_config, "rope_max_timescale")
  hf_config.rope_scaling = rope_scaling
  hf_config.rope_parameters = rope_scaling


def _transform_weights_to_adapter(param_map, state_dict):
  """Extracts standalone PEFT weights from MaxText state dict."""
  processed_params_list = []
  found_hf_modules = set()
  for mt_key, hf_paths in param_map.items():
    a_key, b_key = mt_key + "_lora_a", mt_key + "_lora_b"
    if a_key not in state_dict and mt_key.startswith("params-"):
      a_key, b_key = mt_key[7:] + "_lora_a", mt_key[7:] + "_lora_b"
    if a_key in state_dict and b_key in state_dict:
      data_a, data_b = state_dict[a_key], state_dict[b_key]
      hf_paths = [hf_paths] if not isinstance(hf_paths, list) else hf_paths
      for i, hf_path in enumerate(hf_paths):
        found_hf_modules.add(hf_path.split(".")[-2])
        name = hf_path.replace(".weight", "")

        if data_a.ndim > 2:
          if data_a.shape[0] == len(hf_paths):
            # Case A: layer dimension is axis 0
            layer_a = data_a[i, ...]
            layer_b = data_b[i, ...]
          else:
            # Case B: layer dimension is axis 1
            layer_a = data_a[:, i, ...]
            layer_b = data_b[:, i, ...]
        else:
          layer_a = data_a
          layer_b = data_b

        if layer_a.ndim > 2:
          layer_a = layer_a[:, 0, :]
        if layer_b.ndim > 2:
          layer_b = layer_b[:, 0, :]

        processed_params_list.append(
            (
                f"base_model.model.{name}.lora_A.weight",
                jax.numpy.asarray(layer_a.T),
            )
        )
        processed_params_list.append(
            (
                f"base_model.model.{name}.lora_B.weight",
                jax.numpy.asarray(layer_b.T),
            )
        )
  return dict(processed_params_list), found_hf_modules


def _transform_weights_to_full_model(config, filtered_map_keys, state_dict, param_map, hook_fn_map, shape_map):
  """Transforms MaxText weights to HF full model format, with optional LoRA merging."""
  processed_params_list = []
  lora_scaling = config.lora.lora_alpha / config.lora.lora_rank if config.lora.lora_rank > 0 else 1.0
  for key in MemoryMonitorTqdm(filtered_map_keys, leave=True):
    if isinstance(key, tuple):
      weight = [state_dict.pop(subkey, None) for subkey in key]
    else:
      weight = state_dict.pop(key, None)
    if weight is not None and not isinstance(key, tuple):
      delta = _get_lora_delta(key, state_dict, lora_scaling)
      if delta is not None:
        if delta.shape != weight.shape and delta.size == weight.size:
          delta = delta.reshape(weight.shape)
        weight = (jnp.asarray(weight, dtype=jnp.float32) + delta).astype(weight.dtype)
    if weight is not None:
      processed_params_list.extend(process_maxtext_param(key, weight, param_map, hook_fn_map, shape_map, config))
    del weight
    _trim_memory()
  return dict(processed_params_list)


def _trim_memory():
  gc.collect()
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except (OSError, AttributeError):
    pass


def _transform_and_save_weights(
    config,
    lora_restore_path,
    load_parameters_path,
    param_map,
    maxtext_state_dict,
    filtered_map_keys,
    hook_fn_map,
    shape_map,
    output_directory,
    hf_config_obj,
    tokenizer,
    processor,
):
  """Orchestrates weight transformation and saving based on conversion mode with streaming safetensors."""
  start = time.time()

  if lora_restore_path and not load_parameters_path:
    # Adapter Mode
    transformed_hf_weights, found_hf_modules = _transform_weights_to_adapter(param_map, maxtext_state_dict)
    save_adapter_files(output_directory, transformed_hf_weights, config, found_hf_modules, HF_IDS.get(config.model_name))
    max_logging.log(f"✅ LoRA adapter successfully saved at {output_directory}")
    max_logging.log(f"Elapse for transform and save: {(time.time() - start) / 60:.2f} min")
    return

  # Base or Merged Mode: Streaming transform & save
  max_logging.log(f"\n-> Streaming transform and save to {output_directory}...")

  with get_local_save_path_manager(output_directory) as (current_save_path, is_temp_path):
    remove_local_copy = is_temp_path

    # Save tokenizer/processor and config.json
    if jax.process_index() == 0:
      if processor is not None:
        max_logging.log(f"    Saving image processor files to {current_save_path}...")
        processor.save_pretrained(current_save_path)
      elif tokenizer is not None:
        max_logging.log(f"    Saving tokenizer files to {current_save_path}...")
        tokenizer.save_pretrained(current_save_path)

      files_to_upload = [os.path.join(current_save_path, f) for f in os.listdir(current_save_path)]
      if output_directory.startswith("gs://"):
        for local_file_path in files_to_upload:
          if os.path.exists(local_file_path):
            file_name = os.path.basename(local_file_path)
            upload_file_to_gcs(
                local_file_path,
                os.path.join(output_directory, file_name),
                remove_local_file_after_upload=remove_local_copy,
            )

      save_config_file(hf_config_obj, current_save_path, output_directory, SAFE_TENSORS_CONFIG_FILE, remove_local_copy)

    # Precalculate deterministic sharding plan from shape_map
    dtype_str = getattr(config, "weight_dtype", "bfloat16") or "bfloat16"
    itemsize = np.dtype(dtype_str).itemsize

    def _natural_key(name):
      return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]

    all_target_keys = sorted(shape_map.keys(), key=_natural_key)
    shards_plan = []
    current_shard_keys = set()
    current_shard_size = 0
    total_size = 0

    for k in all_target_keys:
      shape = shape_map[k]
      tensor_bytes = int(np.prod(shape)) * itemsize
      if (current_shard_size + tensor_bytes > DEFAULT_MAX_SHARD_SIZE) and current_shard_keys:
        shards_plan.append(current_shard_keys)
        current_shard_keys = set()
        current_shard_size = 0
      current_shard_keys.add(k)
      current_shard_size += tensor_bytes
      total_size += tensor_bytes

    if current_shard_keys:
      shards_plan.append(current_shard_keys)

    num_shards = len(shards_plan)
    key_to_shard_name = {}
    shard_name_to_required_keys = {}

    for idx, shard_keys in enumerate(shards_plan, 1):
      if num_shards == 1:
        sname = SAFE_TENSORS_WEIGHTS_FILE
      else:
        sname = SAFE_TENSORS_WEIGHTS_FILE.replace(".safetensors", f"-{idx:05d}-of-{num_shards:05d}.safetensors")
      shard_name_to_required_keys[sname] = set(shard_keys)
      for k in shard_keys:
        key_to_shard_name[k] = sname

    # Save index file upfront if sharded
    if num_shards > 1 and jax.process_index() == 0:
      index_dict = {
          "metadata": {"total_size": total_size},
          "weight_map": key_to_shard_name,
      }
      save_index_file(index_dict, current_save_path, output_directory, SAFE_TENSORS_INDEX_FILE, remove_local_copy)

    # Streaming transformation and shard flushing
    buffered_tensors = {}
    completed_shards = set()
    lora_scaling = config.lora.lora_alpha / config.lora.lora_rank if config.lora.lora_rank > 0 else 1.0

    # Separate unstacked parameters (e.g. embeddings, norms) and stacked parameters (layer weights)
    unstacked_keys = []
    stacked_keys = []
    for k in filtered_map_keys:
      target_paths = param_map[k]
      if target_paths is None:
        continue
      if isinstance(target_paths, list) and len(target_paths) > 0 and not isinstance(target_paths, tuple):
        stacked_keys.append(k)
      else:
        unstacked_keys.append(k)

    max_logging.log("\nProcessing unstacked weights...")
    for key in unstacked_keys:
      if isinstance(key, tuple):
        weight = [maxtext_state_dict.pop(subkey, None) for subkey in key]
      else:
        weight = maxtext_state_dict.pop(key, None)
      if weight is not None and not isinstance(key, tuple):
        delta = _get_lora_delta(key, maxtext_state_dict, lora_scaling)
        if delta is not None:
          if delta.shape != weight.shape and delta.size == weight.size:
            delta = delta.reshape(weight.shape)
          weight = (jnp.asarray(weight, dtype=jnp.float32) + delta).astype(weight.dtype)
      if weight is not None:
        transformed_pairs = process_maxtext_param(key, weight, param_map, hook_fn_map, shape_map, config)
        for hf_name, hf_arr in transformed_pairs:
          buffered_tensors[hf_name] = hf_arr

        # Check and flush any planned shard that is fully ready in buffered_tensors
        for sname, req_keys in shard_name_to_required_keys.items():
          if sname not in completed_shards and req_keys.issubset(buffered_tensors.keys()):
            shard_dict = {k: buffered_tensors.pop(k) for k in req_keys}
            max_logging.log(f"   Writing completed shard {sname} ({len(shard_dict)} tensors)...")
            save_safetensor_file(shard_dict, current_save_path, output_directory, sname)
            del shard_dict
            _trim_memory()
            completed_shards.add(sname)

      del weight
      _trim_memory()

    # Determine total number of layers from the first stacked parameter
    if stacked_keys:
      num_layers = len(param_map[stacked_keys[0]])
      axis_to_slice = config.param_scan_axis if config.scan_layers else 0
      max_logging.log(f"\nProcessing {num_layers} layers with zero-memory streaming...")

      # Convert stacked tensors in maxtext_state_dict to numpy arrays upfront so slicing never touches JAX/XLA allocator
      for key in stacked_keys:
        w = maxtext_state_dict.pop(key, None)
        if w is not None:
          if isinstance(w, list):
            maxtext_state_dict[key] = [np.asarray(x) if not isinstance(x, np.ndarray) else x for x in w]
          elif not isinstance(w, np.ndarray):
            maxtext_state_dict[key] = np.asarray(w)
          del w
      _trim_memory()

      for layer_idx in MemoryMonitorTqdm(range(num_layers), desc="Streaming layers", leave=True):
        for key in stacked_keys:
          hf_path = param_map[key][layer_idx]
          weight = maxtext_state_dict.get(key)
          if weight is None:
            continue

          if isinstance(weight, list):
            weight_slice = [
                x.take(layer_idx, axis=axis_to_slice)
                if isinstance(x, np.ndarray)
                else np.take(np.asarray(x), layer_idx, axis=axis_to_slice)
                for x in weight
            ]
          else:
            weight_slice = (
                weight.take(layer_idx, axis=axis_to_slice)
                if isinstance(weight, np.ndarray)
                else np.take(np.asarray(weight), layer_idx, axis=axis_to_slice)
            )

          layer_output = []
          _process(
              hf_path,
              weight_slice,
              layer_output,
              hook_fn_map.get(key),
              shape_map,
              save_dtype=config.weight_dtype,
          )
          for hf_name, hf_arr in layer_output:
            buffered_tensors[hf_name] = hf_arr

        # Check and flush any planned shard that is fully ready in buffered_tensors
        for sname, req_keys in shard_name_to_required_keys.items():
          if sname not in completed_shards and req_keys.issubset(buffered_tensors.keys()):
            shard_dict = {k: buffered_tensors.pop(k) for k in req_keys}
            max_logging.log(f"   Writing completed shard {sname} ({len(shard_dict)} tensors)...")
            save_safetensor_file(shard_dict, current_save_path, output_directory, sname)
            del shard_dict
            _trim_memory()
            completed_shards.add(sname)

        if "weight_slice" in locals():
          del weight_slice
        if "layer_output" in locals():
          del layer_output
        _trim_memory()

      # All layers processed: clear state_dict
      maxtext_state_dict.clear()
      gc.collect()

    # Final flush for any remaining tensors across all shards
    for sname, req_keys in shard_name_to_required_keys.items():
      if sname not in completed_shards:
        remaining_keys = [k for k in req_keys if k in buffered_tensors]
        if remaining_keys:
          shard_dict = {k: buffered_tensors.pop(k) for k in remaining_keys}
          max_logging.log(f"   Writing final shard {sname} ({len(shard_dict)} tensors)...")
          save_safetensor_file(shard_dict, current_save_path, output_directory, sname)
          del shard_dict
          gc.collect()
          completed_shards.add(sname)

    if jax.process_index() == 0:
      max_logging.log(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")

  max_logging.log(f"Elapse for transform and save: {(time.time() - start) / 60:.2f} min")


def main(argv: Sequence[str]) -> None:
  """Main function to convert a MaxText checkpoint to HuggingFace format.

  This function orchestrates the entire conversion process. It loads the
  MaxText checkpoint, transforms the parameter keys and weights according to
  pre-defined mappings, and saves the resulting model, configuration, and
  tokenizer in a format compatible with the Hugging Face ecosystem.

  Args:
    argv: Command-line arguments, which are parsed by `pyconfig`.
  """
  # Initialize maxtext config
  config = pyconfig.initialize_pydantic(argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."

  if not config.load_parameters_path and config.lora.lora_restore_path:
    # Standalone LoRA conversion uses lora_restore_path for metadata
    temp_config = config.model_copy(update={"load_parameters_path": config.lora.lora_restore_path})
    temp_config = verify_and_sync_scan_layers(temp_config)
    config = config.model_copy(update={"scan_layers": temp_config.scan_layers})
  else:
    config = verify_and_sync_scan_layers(config)
  max_utils.print_system_information()
  overall_start = time.time()

  lora_restore_path = config.lora.lora_restore_path
  load_parameters_path = config.load_parameters_path

  if not load_parameters_path and not lora_restore_path:
    raise ValueError("Either load_parameters_path or lora_restore_path must be specified.")

  if lora_restore_path:
    sync_lora_metadata(config)

  # Load Maxtext checkpoint using Orbax (now smart enough to load both if present)
  max_logging.log("\nLoading Orbax checkpoint(s)...")
  start = time.time()
  checkpoint_dict = load_orbax_checkpoint(config)
  max_logging.log(f"Elapse for checkpoint load: {(time.time() - start) / 60:.2f} min")

  # Define output directory
  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  # 1. Get HuggingFace Model Configuration
  model_key = config.model_name
  if model_key not in HF_IDS:
    raise ValueError(f"Unsupported model name: {config.model_name}. Supported models are: {list(HF_IDS.keys())}")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]

  # Validate architecture consistency (raising ValueError on mismatch) or override HF config if specified.
  _validate_or_update_architecture(hf_config_obj, config, override=FLAGS.override_model_architecture)

  # Ensure YaRN rope params are preserved if specified in the maxtext config
  if getattr(config, "rope_type", None) == "yarn":
    _apply_yarn_rope_config(hf_config_obj, config)

  # 2. Load Tokenizer
  if model_key not in HF_IDS:
    raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
  hf_token = config.hf_access_token
  hf_tokenizer_id = FLAGS.hf_model_path or HF_IDS[model_key]
  tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token=hf_token)

  # For multi-modal case:
  processor = AutoProcessor.from_pretrained(hf_tokenizer_id, token=hf_token) if config.use_multimodal else None

  # 3. Get parameter mappings
  mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict(), config)
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]  # HF target shapes
  hook_fn_map = mappings["hook_fn_mapping"]

  # 4. Extract and transform weights for Linen/NNX-SFT/NNX-RL checkpoints
  maxtext_state_dict = detect_and_extract_checkpoint(checkpoint_dict)

  # Validate that checkpoint keys match the parameter mapping
  state_keys = {k.replace("_lora_a", "").replace("_lora_b", "") for k in maxtext_state_dict}
  filtered_map_keys = validate_and_filter_param_map_keys(param_map, state_keys)

  # When not converting a multimodal model, skip vision encoder weights even if
  # they are present in the checkpoint (e.g. converting text-only from a
  # multimodal checkpoint).
  if not config.use_multimodal:
    filtered_map_keys = [
        k
        for k in filtered_map_keys
        if not (isinstance(k, str) and "vision_encoder" in k)
        and not (isinstance(k, tuple) and any("vision_encoder" in sub for sub in k))
    ]

  # Iterate through the parameter map to transform and collect weights.
  max_logging.log("\nProcessing weights...")
  _transform_and_save_weights(
      config,
      lora_restore_path,
      load_parameters_path,
      param_map,
      maxtext_state_dict,
      filtered_map_keys,
      hook_fn_map,
      shape_map,
      output_directory,
      hf_config_obj,
      tokenizer,
      processor,
  )

  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

  app.run(main)
