# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""
Get MaxText unscanned ckpt param Naming and shape
Get Huggingface ckpt Naming

This is used for the ckpt conversion agent
"""

import jax
import os
from typing import List
import json
import argparse

from transformers import AutoModelForCausalLM, AutoConfig

from maxtext.src.maxtext import max_utils
from maxtext.src.maxtext import maxengine
from maxtext.src.maxtext import pyconfig
from maxtext.src.maxtext import max_logging
from maxtext.src.maxtext.globals import MAXTEXT_PKG_DIR


def main(parsed_args: argparse.Namespace, unknown_pyconfig_args: List[str]) -> None:

  # step 1: get shape and naming of mt checkpoint
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  print(parsed_args)

  pyconfig_argv = [" "] + unknown_pyconfig_args
  print(pyconfig_argv)

  config = pyconfig.initialize(pyconfig_argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  # load params from maxengine
  loaded_params_from_engine = engine.load_params(rng_load_params)

  actual_weights_dict = loaded_params_from_engine.get("params")
  if actual_weights_dict is None:
    raise ValueError("Loaded parameters from engine do not contain a 'params' key. Structure might be unexpected.")

  leaves_with_paths = jax.tree_util.tree_leaves_with_path(actual_weights_dict)

  # Prepare MaxText params info
  maxtext_params_info = {}
  for path_tuple, leaf_value in leaves_with_paths:
    # Construct maxtext_param_key from path_tuple
    key_parts = []
    for p_entry in path_tuple:
      if isinstance(p_entry, jax.tree_util.DictKey):
        key_parts.append(p_entry.key)
      elif isinstance(p_entry, jax.tree_util.SequenceKey):
        key_parts.append(str(p_entry.idx))
      # Add other key types if necessary, e.g., jax.tree_util.GetAttrKey
      else:
        max_logging.log(
            f"Warning: Path tuple {path_tuple} contains unhandled key type '{type(p_entry)}'. Skipping this part of path."
        )
        # Decide how to handle this: skip, raise error, or use a placeholder
        key_parts.append(f"__unhandled_key_type_{type(p_entry).__name__}__")

    maxtext_param_key = "params-" + "-".join(key_parts)
    maxtext_params_info[maxtext_param_key] = list(leaf_value.shape)

  hf_token = config.hf_access_token
  # step 2: get shape and naming of hf checkpoint
  hf_config = AutoConfig.from_pretrained(parsed_args.hf_model_config, token=hf_token)
  hf_model = AutoModelForCausalLM.from_config(hf_config)
  hf_model_dict = hf_model.state_dict()

  # Prepare Hugging Face params info
  hf_params_info = {}
  for name, tensor in hf_model_dict.items():
    hf_params_info[name] = list(tensor.shape)

  if not config.base_output_directory:
    output_directory = os.path.join(MAXTEXT_PKG_DIR, "experimental", "agent",
                                    "ckpt_conversion_agent", "context", config.model_name)
  else:
    output_directory = config.base_output_directory
  os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists

  # step 3: save the two param names and shapes in to json file, in output_directory
  maxtext_json_path = os.path.join(output_directory, "maxtext_params.json")
  hf_json_path = os.path.join(output_directory, "hf_params.json")

  if jax.process_index() == 0:  # Ensure only one process writes the files
    with open(maxtext_json_path, "wt", encoding="utf8") as f:
      json.dump(maxtext_params_info, f, indent=2)
    max_logging.log(f"MaxText parameters and shapes saved to {maxtext_json_path}")

    with open(hf_json_path, "wt", encoding="utf8") as f:
      json.dump(hf_params_info, f, indent=2)
    max_logging.log(f"Hugging Face parameters and shapes saved to {hf_json_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Get HF and MaxText checkpoints parameter names and shapes.")
  parser.add_argument(
      "--hf_model_config",
      type=str,
      default="google/gemma-2-2b",
  )

  main(*parser.parse_known_args())
