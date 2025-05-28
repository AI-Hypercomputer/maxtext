# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import time
import jax
import os
from typing import Sequence, Dict, Any
import jax.numpy as jnp
import numpy as np 
from transformers import AutoTokenizer
from absl import app
import flax

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig
from MaxText import multimodal_utils

from MaxText.ckpt_conversion.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from MaxText.ckpt_conversion.shape_mapping import SHAPE_MAPPING
from MaxText.ckpt_conversion.model_configs import MODEL_CONFIGS
from MaxText.ckpt_conversion.utils import convert_jax_weight_to_numpy, save_model_files, apply_hook_fns

"""Convert MaxText unscanned ckpt into HF format"""

# Mapping from MaxText model key to Hugging Face tokenizer identifiers
TOKENIZER_HF_IDS = {
    "GEMMA2_2B": "google/gemma-2-2b",
    "GEMMA2_9B": "google/gemma-2-9b",
    "GEMMA2_27B": "google/gemma-2-27b",
    "LLAMA31_8B": "meta-llama/Llama-3.1-8B", 
    "LLAMA31_70B": "meta-llama/Llama-3.1-70B",
    "LLAMA31_405B": "meta-llama/Llama-3.1-405B", 
}

def _get_model_mappings(model_name: str, scan_layers: bool, config_dict: dict): # Changed config to config_dict
    """Retrieves parameter, shape, and hook function mappings for the model."""
    if model_name not in PARAM_MAPPING or \
       model_name not in SHAPE_MAPPING or \
       model_name not in HOOK_FNS:
        raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

    return {
        "param_mapping": PARAM_MAPPING[model_name](config_dict, scan_layers),
        "shape_mapping": SHAPE_MAPPING[model_name](config_dict),
        "hook_fn_mapping": HOOK_FNS[model_name](
            config_dict, scan_layers, saving_to_hf=True
        ),
    }

_NUM_STREAMS = 1

def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    config = pyconfig.initialize(argv)
    assert config.load_full_state_path == "", (
        "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
    )
    max_utils.print_system_information()

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    # load params from maxengine
    loaded_params_from_engine = engine.load_params(rng_load_params)

    if not config.base_output_directory:
        output_directory = os.path.expanduser("~/.hf_output/")
    else:
        output_directory = config.base_output_directory

    # 1. Get HuggingFace Model Configuration
    model_key = None
    if config.model_name == "gemma2-2b": 
      model_key = "GEMMA2_2B"
    elif config.model_name == "gemma2-9b": 
      model_key = "GEMMA2_9B"
    elif config.model_name == "llama3.1-8b": 
      model_key = "LLAMA31_8B"
    elif config.model_name == "llama3.1-70b": 
      model_key = "LLAMA31_70B"
    elif config.model_name == "llama3.1-405b":
      model_key = "LLAMA31_405B"
    else:
      raise ValueError(f"Unsupported MaxText model_name for Kithara HF config lookup: {config.model_name}") # type: ignore

    if model_key not in MODEL_CONFIGS: 
      raise ValueError(f"Kithara HF configuration not found for model key: {model_key}")
    hf_config_obj = MODEL_CONFIGS[model_key] 

    # 2. Load Tokenizer
    if model_key not in TOKENIZER_HF_IDS: 
        raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
    hf_tokenizer_id = TOKENIZER_HF_IDS[model_key] 
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token = hf_token)

    # 3. Get parameter mappings
    mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict())
    param_map = mappings["param_mapping"]
    shape_map = mappings["shape_mapping"]  # HF target shapes
    hook_fn_map = mappings["hook_fn_mapping"]



    # 4. Transform Weights
    transformed_hf_weights: Dict[str, Any] = {}

    # Helper function to recursively traverse the MaxText params tree
    def _traverse_and_transform(
        current_maxtext_params_subtree: Dict[str, Any],
        current_maxtext_path_parts: list[str],
        output_hf_weights_dict: Dict[str, Any]
    ):
        for name, value in current_maxtext_params_subtree.items():
            maxtext_param_key = "-".join(current_maxtext_path_parts + [name])

            if isinstance(value, dict) or (isinstance(value, flax.core.FrozenDict)):  # Recurse for sub-trees
              _traverse_and_transform(value, current_maxtext_path_parts + [name], output_hf_weights_dict)
            elif isinstance(value, (jax.Array, np.ndarray)):  # This is a weight array
              if maxtext_param_key not in param_map:
                  print(f"Warning: MaxText param key '{maxtext_param_key}' not found in param_map. Skipping.")
                  continue

              hf_target_paths = param_map[maxtext_param_key]
              if not isinstance(hf_target_paths, list):
                  hf_target_paths = [hf_target_paths]

              if not hf_target_paths or hf_target_paths[0] not in shape_map:
                  print(f"Warning: HF path '{hf_target_paths[0] if hf_target_paths else 'None'}' not found in shape_map for MaxText key '{maxtext_param_key}'. Skipping.")
                  continue
              target_shape_for_hooks = shape_map[hf_target_paths[0]]

              current_hook_fns = hook_fn_map.get(maxtext_param_key)

              if len(hf_target_paths) == 1:
                  hf_path = hf_target_paths[0]
                  processed_weight = value
                  if current_hook_fns:
                      processed_weight = apply_hook_fns(processed_weight, target_shape_for_hooks, current_hook_fns)
                  numpy_weight = convert_jax_weight_to_numpy(processed_weight)
                  output_hf_weights_dict[hf_path] = numpy_weight
              else:  # Stacked MaxText weight
                  if not (value.ndim > 0 and value.shape[config.param_scan_axis] == len(hf_target_paths)): # type: ignore
                      print(f"Warning: Mismatch for stacked layer {maxtext_param_key}. MaxText shape {value.shape}, expected {len(hf_target_paths)} slices on axis {config.param_scan_axis}. Skipping.") # type: ignore
                      continue

                  for i, hf_path in enumerate(hf_target_paths):
                      # Slicing the JAX array
                      weight_slice = jax.lax.index_in_dim(value, i, axis=config.param_scan_axis, keepdims=False) # type: ignore
                      processed_slice = weight_slice
                      if current_hook_fns:
                          processed_slice = apply_hook_fns(processed_slice, target_shape_for_hooks, current_hook_fns)
                      numpy_slice = convert_jax_weight_to_numpy(processed_slice)
                      output_hf_weights_dict[hf_path] = numpy_slice
            #else:
            #    print(f"Info: Skipping non-array/non-dict item: {maxtext_param_key} of type {type(value)}")

    # MaxText `engine.load_params()` returns `state.params` (a FrozenDict).
    # The actual weights are typically under `state.params['params']`.
    actual_weights_dict = loaded_params_from_engine.get('params')
    if actual_weights_dict is None:
        raise ValueError("Loaded parameters from engine do not contain a 'params' key. Structure might be unexpected.")

    _traverse_and_transform(actual_weights_dict, ["params"], transformed_hf_weights)

    # 5. Save in HuggingFace Format
    if not transformed_hf_weights:
        print("Error: No weights were transformed. Check mappings and parameter paths.")
        return

    save_model_files(
        weight_arrays=transformed_hf_weights,
        config=hf_config_obj,
        tokenizer=tokenizer,
        output_dir=output_directory,
    )
    print(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")

if __name__ == "__main__":
  app.run(main)
