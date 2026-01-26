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

"""Converts a MaxText checkpoint to a HuggingFace-compatible model checkpoint.

It is invoked using MaxText's pyconfig, which means you provide a base config
file and can override parameters on the command line.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `MaxText.utils.ckpt_conversion.utils.utils.HF_IDS`.
  load_parameters_path: (Required) Path to the MaxText checkpoint directory
                        containing the parameter-only checkpoint.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.

Environment Variables:
  HF_AUTH_TOKEN: (Required) A HuggingFace authentication token. This is needed
                 to download the correct tokenizer configuration and to upload
                 the converted model to the HuggingFace Hub if `base_output_directory`
                 is a Hub repo ID (e.g., "hf://my-user/my-model").

Example Usage:
  To convert a gemma2-2b MaxText checkpoint and save it to a local directory:

  export HF_AUTH_TOKEN="hf_YOUR_TOKEN"
  python src/MaxText/utils/ckpt_conversion/to_huggingface.py \
    src/MaxText/configs/base.yml \
    model_name="gemma2-2b" \
    load_parameters_path="/path/to/your/maxtext/checkpoint/" \
    base_output_directory="/path/to/your/output/directory" \
    scan_layers=False

  Note: Other parameters in base.yml (like per_device_batch_size, max_target_length, etc.)
  are used to initialize the model structure and should be consistent with the
  checkpoint being converted, but often don't need to be changed from their defaults.
"""

import jax
import os
from typing import Sequence
import time
from tqdm import tqdm

from transformers import AutoTokenizer, AutoProcessor

from absl import app

from MaxText import max_utils
from MaxText import pyconfig
from MaxText import max_logging
from MaxText.utils.ckpt_conversion.strategies.factory import ModelMapperFactory
from MaxText.utils.ckpt_conversion.utils.hf_shape import HF_SHAPE
from MaxText.utils.ckpt_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from MaxText.utils.ckpt_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    save_model_files,
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    HF_IDS,
)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"


def _get_model_mappings(
    model_name: str, scan_layers: bool, hf_config_dict: dict, maxtext_config: pyconfig.HyperParameters
):
  """Retrieves parameter, shape, and hook function mappings for the model.

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
  try:
    strategy = ModelMapperFactory.get_strategy(model_name)
  except ValueError as e:
    # Check old maps if factory fails (or relying on shim)
    raise ValueError(f"Mappings not found for model: {model_name}") from e

  if model_name not in HF_SHAPE:
    raise ValueError(f"Shape mapping not found for {model_name}")

  return {
      "param_mapping": strategy.get_mapping(hf_config_dict, maxtext_config, scan_layers),
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": strategy.get_hooks(hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True),
  }


def main(argv: Sequence[str]) -> None:
  """Main function to convert a MaxText checkpoint to HuggingFace format.

  This function orchestrates the entire conversion process. It loads the
  MaxText checkpoint, transforms the parameter keys and weights according to
  pre-defined mappings, and saves the resulting model, configuration, and
  tokenizer in a format compatible with the Hugging Face ecosystem.

  Args:
    argv: Command-line arguments, which are parsed by `pyconfig`.
  """
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Initialize maxtext config
  config = pyconfig.initialize(argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  max_utils.print_system_information()
  overall_start = time.time()

  # Load Maxtext checkpoint using Orbax to get full parameter dict
  max_logging.log(f"\nLoading Orbax checkpoint from: {config.load_parameters_path}")
  start = time.time()
  checkpoint_dict = load_orbax_checkpoint(config)
  max_logging.log(f"Elapse for checkpoint load: {(time.time() - start) / 60:.2f} min")

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

  # 4. Extract and transform weights for Linen/NNX-SFT/NNX-RL checkpoints
  maxtext_state_dict = detect_and_extract_checkpoint(checkpoint_dict)

  # Validate that checkpoint keys match the parameter mapping
  filtered_map_keys = validate_and_filter_param_map_keys(param_map.keys(), maxtext_state_dict.keys())

  # Iterate through the parameter map to transform and collect weights.
  # This loop handles both simple 1-to-1 mappings and complex N-to-1 mappings
  # (where multiple MaxText weights are combined into a single HF weight).
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
