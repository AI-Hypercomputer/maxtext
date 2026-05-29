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

"""Merges a MaxText LoRA checkpoint into a MaxText base model checkpoint.

Example Usage:
  python src/maxtext/checkpoint_conversion/merge_maxtext_lora.py \
    src/maxtext/configs/base.yml \
    model_name="gemma3-4b" \
    load_parameters_path="/path/to/base/checkpoint/" \
    lora.lora_restore_path="/path/to/lora/checkpoint/" \
    base_output_directory="/path/to/merged/output/" \
    scan_layers=True
"""

import jax
import jax.numpy as jnp
import os
import re
from typing import Sequence
import time

from absl import app
from absl import flags

from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion.utils.utils import (
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    save_weights_to_checkpoint,
    MemoryMonitorTqdm,
    print_peak_memory,
    get_lora_delta,
)
from maxtext.checkpoint_conversion.to_maxtext import get_maxtext_model_info
from maxtext.utils import max_logging
from maxtext.utils import max_utils


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize_pydantic(argv)
  max_utils.print_system_information()
  overall_start = time.time()

  lora_restore_path = config.lora.lora_restore_path
  load_parameters_path = config.load_parameters_path

  if not load_parameters_path or not lora_restore_path:
    raise ValueError("Both load_parameters_path and lora.lora_restore_path must be specified.")

  # 1. Load and Extract Checkpoints
  max_logging.log("\nLoading Orbax checkpoint(s)...")
  start = time.time()
  checkpoint_dict = load_orbax_checkpoint(config)
  max_logging.log(f"Elapse for checkpoint load: {(time.time() - start) / 60:.2f} min")

  maxtext_state_dict = detect_and_extract_checkpoint(checkpoint_dict)

  # 2. Get Model Info (Needed for unflattening later)
  maxtext_abstract_dict, abstract_params_treedef = get_maxtext_model_info(config)

  # 3. Merging Logic
  max_logging.log("\nMerging LoRA weights into base model...")
  lora_scaling = config.lora.lora_alpha / config.lora.lora_rank if config.lora.lora_rank > 0 else 1.0
  final_mt_weights = [None] * len(maxtext_abstract_dict)
  merged_params_count = 0
  merged_layers = set()

  for mt_param_key_abs, (mt_target_idx, mt_target_shape) in MemoryMonitorTqdm(
      maxtext_abstract_dict.items(), desc="Merging weights", unit="param"
  ):
    weight = maxtext_state_dict.get(mt_param_key_abs)
    if weight is not None:
      delta = get_lora_delta(mt_param_key_abs, maxtext_state_dict, lora_scaling)
      if delta is not None:
        if delta.shape != weight.shape and delta.size == weight.size:
          delta = delta.reshape(weight.shape)
        weight = (jnp.asarray(weight, dtype=jnp.float32) + delta).astype(weight.dtype)
        merged_params_count += 1
        # Extract layer index for reporting
        layer_match = re.search(r"layers[_\-/](\d+)", mt_param_key_abs)
        if layer_match:
          merged_layers.add(int(layer_match.group(1)))
        else:
          # If not found in the key itself, check if the weights were scanned.
          # For scanned Llama 3.1, the key is e.g. 'params-decoder-layers-self_attention-query-kernel'
          # mt_target_shape is a tuple of the individual layer's shape.
          if weight.ndim > len(mt_target_shape) or (len(mt_target_shape) > 0 and weight.shape[0] > 1):
             # This is a scanned tensor, it likely covers all layers.
             # We use the config to get the actual number of layers if available.
             n_layers = getattr(config, "num_decoder_layers", 32)
             for i in range(n_layers):
               merged_layers.add(i)

    final_mt_weights[mt_target_idx] = weight

  if merged_params_count > 0:
    max_logging.log(
        f"Successfully merged LoRA weights into {merged_params_count} parameters "
        f"across {len(merged_layers)} layers."
    )
  else:
    max_logging.log("Warning: No LoRA weights were merged. Check if your checkpoint keys match.")

  # 4. Save Final Weights
  max_logging.log("\nSaving merged MaxText checkpoint...")
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  output_directory = config.base_output_directory or f"tmp/{config.run_name}_merged"

  simulated_cpu_devices_count = max(
      16, config.ici_tensor_parallelism * config.ici_data_parallelism * config.ici_expert_parallelism
  )

  save_weights_to_checkpoint(
      output_directory,
      jax_weights,
      simulated_cpu_devices_count,
      config.checkpoint_storage_use_ocdbt,
      config.checkpoint_storage_use_zarr3,
  )
  max_logging.log(f"✅ Merged checkpoint successfully saved at {output_directory}")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
  app.run(main)
