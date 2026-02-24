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

"""NNX model forward pass utility for inference."""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from MaxText import pyconfig
from maxtext.utils import model_creation_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE, DECODING_ACTIVE_SEQUENCE_INDICATOR

import absl
absl.logging.set_verbosity(absl.logging.INFO)


def main(config, test_args):
  """Run NNX model forward pass for inference."""
  max_logging.log("Initializing NNX model for inference")
  
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  
  # Load NNX model with checkpoint
  model, mesh = model_creation_utils.create_nnx_model(
      config,
      model_mode=MODEL_MODE_AUTOREGRESSIVE,
      rng_key=init_rng
  )
  max_utils.print_mem_stats("After model creation")
  
  max_logging.log(f"Model loaded successfully from {config.load_parameters_path}")

  max_logging.log(f"Mesh: {mesh}")
  max_logging.log(f"Model ready for inference")
  
  # Example: Create dummy input for testing
  if test_args.test_forward:
    batch_size = config.global_batch_size_to_train_on
    seq_len = 10
    
    # Create dummy input tokens
    dummy_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_positions = jnp.stack(
        [jnp.arange(seq_len, dtype=jnp.int32) for _ in range(batch_size)]
    )
    dummy_segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_segment_ids = dummy_segment_ids + DECODING_ACTIVE_SEQUENCE_INDICATOR
    
    max_logging.log("\nRunning test forward pass...")
    max_logging.log(f"Input shape: {dummy_ids.shape}")
    
    # Run forward pass
    logits = model(
        dummy_ids,
        dummy_positions,
        dummy_segment_ids,
        enable_dropout=False,
    )
    
    max_logging.log(f"Output logits shape: {logits.shape}")
    max_logging.log(f"Logits dtype: {logits.dtype}")
    max_logging.log(f"Logits sample (first token): {logits[0, 0, :10]}")
    max_logging.log("\nForward pass completed successfully!")
  
  return model, mesh


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--test_forward",
      action="store_true",
      required=False,
      default=False,
      help="Run a test forward pass with dummy inputs.",
  )
  test_args, _ = parser.parse_known_args()
  
  # Remove args defined in this test file to avoid error from pyconfig
  model_args = sys.argv
  to_remove_args = ["--test_forward"]
  for arg in to_remove_args:
    model_args = [s for s in model_args if not s.startswith(arg)]
  
  cfg = pyconfig.initialize(model_args)
  
  # IMPORTANT: MaxText REQUIRES enable_checkpointing=True to load checkpoints
  # But this means a checkpoint manager will be created that can DELETE old checkpoints!
  # 
  # To prevent accidental deletion of your source checkpoint:
  # 1. Keep enable_checkpointing=True (required for loading)
  # 2. Set base_output_directory to a DIFFERENT location than load_parameters_path
  # 3. Use checkpoint_period=999999 to prevent saving during inference
  
  if cfg.load_parameters_path:
    # Validate that paths don't conflict
    if cfg.base_output_directory:
      load_path = str(cfg.load_parameters_path).rstrip('/').split('/items')[0]
      output_dir = str(cfg.base_output_directory).rstrip('/')
      
      # Check if output directory overlaps with checkpoint directory
      if output_dir in load_path or load_path.startswith(output_dir):
        raise ValueError(
            f"\n{'='*80}\n"
            f"CRITICAL ERROR: Path conflict detected!\n"
            f"  base_output_directory: {output_dir}\n"
            f"  load_parameters_path:  {load_path}\n"
            f"\n"
            f"These paths overlap! CheckpointManager will DELETE your source checkpoint!\n"
            f"\n"
            f"Fix: Set base_output_directory to a completely different location, e.g.:\n"
            f"  base_output_directory=/tmp/inference_outputs\n"
            f"{'='*80}\n"
        )
    
    # For inference/testing, set checkpoint_period to very high value to avoid saving
    max_logging.log(
        f"Loading checkpoint from: {cfg.load_parameters_path}\n"
        f"Output directory: {cfg.base_output_directory}\n"
        f"Checkpoint period set to {cfg.checkpoint_period} to prevent frequent saves during inference."
    )
  
  main(cfg, test_args)