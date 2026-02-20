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

"""Converts a MaxText checkpoint to a HuggingFace-compatible model checkpoint.

It is invoked using MaxText's pyconfig, which means you provide a base config
file and can override parameters on the command line.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `maxtext.checkpoint_conversion.utils.utils.HF_IDS`.
  load_parameters_path: (Required) Path to the MaxText checkpoint directory
                        containing the parameter-only checkpoint.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.

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
  To convert a gemma2-2b MaxText checkpoint and save it to a local directory:

  export HF_AUTH_TOKEN="hf_YOUR_TOKEN"
  python src/MaxText/checkpoint_conversion/to_huggingface.py \
    src/maxtext/configs/base.yml \
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

from transformers import AutoTokenizer, AutoProcessor

from absl import app
from absl import flags

from MaxText import pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    save_model_files,
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    HF_IDS,
    MemoryMonitorTqdm,
    print_peak_memory,
)
from maxtext.utils import max_logging
from maxtext.utils import max_utils

flags.DEFINE_bool(
    "override_model_architecture",
    False,
    "If True, overrides Hugging Face config architecture parameters (heads, layers, dims) "
    "with values from the MaxText config. If False, raises a ValueError on mismatch.",
)

FLAGS = flags.FLAGS


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
  if model_name not in PARAM_MAPPING or model_name not in HF_SHAPE or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  return {
      "param_mapping": PARAM_MAPPING[model_name](hf_config_dict, maxtext_config, scan_layers),
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": HOOK_FNS[model_name](hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True),
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
      ("num_attention_heads", "num_query_heads"),
      ("num_key_value_heads", "num_kv_heads"),
      ("head_dim", "head_dim"),
      ("hidden_size", "emb_dim"),
      ("intermediate_size", "mlp_dim"),
      ("num_hidden_layers", "num_decoder_layers"),
      ("vocab_size", "vocab_size"),
  ]

  mismatches = []

  for hf_attr, mt_attr in attributes_to_check:
    # Skip checks if the HF config doesn't have this attribute (e.g. layer_norm_eps vs rms_norm_eps)
    if not hasattr(hf_config, hf_attr):
      continue

    # Skip checks if MaxText config doesn't have the attribute (shouldn't happen for valid configs)
    if not hasattr(max_config, mt_attr):
      continue

    hf_value = getattr(hf_config, hf_attr)
    mt_value = getattr(max_config, mt_attr)

    # Handle None values
    if hf_value is None or mt_value is None:
      continue

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
        setattr(hf_config, hf_attr, mt_value)
      else:
        mismatches.append(f"{hf_attr} (HF={hf_value} vs MaxText={mt_value})")

  if mismatches:
    error_msg = (
        "Architecture mismatches detected between standard Hugging Face config and provided MaxText config:\n  - "
        + "\n  - ".join(mismatches)
        + "\n\nAction Required: Pass the flag `--override_model_architecture` to force the conversion using MaxText values."
    )
    raise ValueError(error_msg)


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

  # Validate architecture consistency (raising ValueError on mismatch) or override HF config if specified.
  _validate_or_update_architecture(hf_config_obj, config, override=FLAGS.override_model_architecture)

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

  for key in MemoryMonitorTqdm(filtered_map_keys, total=len(filtered_map_keys), leave=True):
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
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

  app.run(main)
