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

"""Converts a Tunix/NNX MaxText checkpoint to a HuggingFace-compatible model checkpoint.
Modified from to_huggingface.py to handle 'base/' prefix and 'value' suffix in NNX checkpoints.
"""

import jax
import os
from typing import Sequence
import time
from tqdm import tqdm
import numpy as np
import orbax.checkpoint

from transformers import AutoTokenizer, AutoProcessor

from absl import app

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig
from MaxText import max_logging
from MaxText.utils.ckpt_conversion.utils.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from MaxText.utils.ckpt_conversion.utils.hf_shape import HF_SHAPE
from MaxText.utils.ckpt_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from MaxText.utils.ckpt_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    save_model_files,
    HF_IDS,
)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"


def _get_model_mappings(
    model_name: str, scan_layers: bool, hf_config_dict: dict, maxtext_config: pyconfig.HyperParameters
):
  if model_name not in PARAM_MAPPING or model_name not in HF_SHAPE or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  return {
      "param_mapping": PARAM_MAPPING[model_name](hf_config_dict, maxtext_config, scan_layers),
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": HOOK_FNS[model_name](hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True),
  }


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Initialize maxtext config
  config = pyconfig.initialize(argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  max_utils.print_system_information()
  overall_start = time.time()

  # Load Maxtext checkpoint MANUALLY for Tunix/NNX compatibility
  max_logging.log("\nLoading Orbax checkpoint (Tunix mode)...")
  start = time.time()
  
  checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
  try:
      raw_restored = checkpointer.restore(config.load_parameters_path)
  except Exception as e:
      raise RuntimeError(f"Failed to restore checkpoint from {config.load_parameters_path}: {e}")

  # Flatten and reconstruct maxtext_state_dict with corrected keys
  flat_restored, _ = jax.tree_util.tree_flatten_with_path(raw_restored)
  maxtext_state_dict = {}
  
  max_logging.log("Processing checkpoint keys...")
  for path, val in flat_restored:
      key_parts = [str(p.key) for p in path]
      
      # 1. Replace 'base' with 'params' at the root
      if key_parts[0] == 'base':
          key_parts[0] = 'params'
      elif key_parts[0] == 'params':
          pass # already params
      else:
          # If neither, prepend params? Assuming Tunix/NNX usually has 'base'
          # If it's a standard checkpoint, it might start with something else? 
          # For now, let's assume we want to convert whatever is there to fit 'params-...' 
          pass

      # 2. Remove 'value' suffix (NNX state)
      if key_parts[-1] == 'value':
          key_parts = key_parts[:-1]
      
      # 3. Filter out NNX internal keys
      if 'to_nnx__rngs' in key_parts:
          continue
          
      # 4. Construct hyphenated key
      # keys in PARAM_MAPPING are like "params-decoder-..."
      maxtext_param_key = "-".join(key_parts)
      
      # Basic validation: check if it looks like a param we care about
      if not isinstance(val, (jax.Array, np.ndarray)):
           max_logging.log(f"Skipping non-array key: {maxtext_param_key} type {type(val)}")
           continue
           
      maxtext_state_dict[maxtext_param_key] = val

  max_logging.log(f"Restored {len(maxtext_state_dict)} parameters.")
  max_logging.log(f"Elapse for checkpoint load and key process: {(time.time() - start) / 60:.2f} min")

  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  # 1. Get HuggingFace Model Configuration
  model_key = config.model_name
  if model_key not in HF_IDS:
    raise ValueError(f"Unsupported model name: {config.model_name}. Supported models are: {list(HF_IDS.keys())}")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]

  # 2. Load Tokenizer
  if model_key not in HF_IDS:
    raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
  hf_token = config.hf_access_token
  hf_tokenizer_id = HF_IDS[model_key]
  tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token=hf_token)

  # For multi-modal case:
  processor = AutoProcessor.from_pretrained(hf_tokenizer_id, token=hf_token) if config.use_multimodal else None

  # 3. Get parameter mappings
  mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict(), config)
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]  # HF target shapes
  hook_fn_map = mappings["hook_fn_mapping"]

  # 4. Transform Weights - Already have maxtext_state_dict
  
  # The param_map may contain tuples as keys, which represent N-to-1 mappings from maxtext to huggingface
  # Check maxtext_state_dict is a subset of flattened param_map
  # Skip extra keys from param_map
  filtered_map_keys = validate_and_filter_param_map_keys(param_map.keys(), maxtext_state_dict.keys())

  # Iterate through the parameter map to transform and collect weights.
  max_logging.log("\nProccessing weight...")
  start = time.time()
  processed_params_list = []

  for key in tqdm(filtered_map_keys, total=len(filtered_map_keys)):
    if isinstance(key, tuple):
      # if key is tuple of param names, weight is list of param weights
      weight = [maxtext_state_dict[subkey] for subkey in key]
    else:
      # if key is single param name, weight is single param weight
      weight = maxtext_state_dict[key]

    processed_params = process_maxtext_param(key, weight, param_map, hook_fn_map, shape_map, config)
    processed_params_list.extend(processed_params)

  transformed_hf_weights = dict(processed_params_list)
  max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")

  # 5. Save in HuggingFace Format
  if not transformed_hf_weights:
    print("Error: No weights were transformed. Check mappings and parameter paths.")
    return

  max_logging.log("\nSaving HuggingFace model...")
  start = time.time()
  save_model_files(
      weight_arrays=transformed_hf_weights,
      config=hf_config_obj,
      tokenizer=tokenizer,
      processor=processor,
      output_dir=output_directory,
  )
  max_logging.log(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")
  max_logging.log(f"Elapse for save: {(time.time() - start) / 60:.2f} min")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")


if __name__ == "__main__":
  app.run(main)
