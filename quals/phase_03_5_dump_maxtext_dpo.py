# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 3.5 Modular Block 2: MaxText-Tunix DPO Loss Stack Dumper.
Processes raw strings using grain mapper DPODataFormatting, runs eager JAX forward pass,
and computes the Tunix dpo_loss_fn on CPU.
"""

import os
import sys
import numpy as np
import optax

# MaxText / JAX imports
import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils

# Disable JAX remat
def no_op_remat(target, *args, **kwargs): return target
nn.remat = no_op_remat

# Monkeypatch build_positions_from_mask to match PyTorch's absolute sequential positions
from tunix.rl import common as tunix_common
from tunix.sft import utils as tunix_sft_utils

def sequential_positions(input_mask):
  return jnp.arange(input_mask.shape[-1])[None, :]

tunix_common.build_positions_from_mask = sequential_positions
tunix_sft_utils.build_positions_from_mask = sequential_positions

def main():
  print("\n=== Step 3.5.2: Extracting MaxText DPO Loss Reference ===")
  
  # Pre-processed un-scanned checkpoint path
  temp_unscanned_path = "/tmp/unscanned_sft_baseline"
  if not os.path.exists(temp_unscanned_path):
    print("CRITICAL ERROR: Un-scanned parameters not found! Run Phase 3 un-scan checkpoint first.", file=sys.stderr)
    sys.exit(3)

  # 1. Initialize MaxText configurations
  argv = [
      "src/maxtext/configs/base.yml",
      "model_name=qwen2.5-1.5b",
      "tokenizer_path=Qwen/Qwen2.5-1.5B-Instruct",
      f"load_parameters_path={temp_unscanned_path}",
      "scan_layers=False",
      "attention=dot_product",
      "per_device_batch_size=1",
      "max_target_length=1024", # max_prompt_len = 512, max_response_len = 512
      "skip_jax_distributed_system=True",
      "enable_nnx=True",
      "pure_nnx=True",
      "pure_nnx_decoder=False",
      "remat_policy=full",
      "log_config=0"
  ]
  config = pyconfig.initialize_pydantic(argv)
  
  # 2. Create mesh and restore model parameters
  model, mesh = model_creation_utils.from_pretrained(config, wrap_with_tunix_adapter=True)

  # 3. Invoke MaxText grain preprocessing pipelines dynamically
  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline.dpo_utils import DPODataFormatting
  
  # Initialize standard tokenizer adapter
  from tunix.generate.tokenizer_adapter import TokenizerAdapter
  from transformers import AutoTokenizer
  
  native_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
  tokenizer = TokenizerAdapter(native_tok)

  prompt_str = "What is DPO?"
  chosen_str = "DPO stands for Direct Preference Optimization, an algorithm for aligning LLMs."
  rejected_str = "DPO is a marketing strategy used to target customers' preferences."

  # Prepare elements dict matching JAX grain pipeline inputs
  element = {
      "prompt": tokenizer.encode(prompt_str),
      "chosen": tokenizer.encode(chosen_str),
      "rejected": tokenizer.encode(rejected_str)
  }

  # Apply DPODataFormatting exactly as done during dataset mapping
  pad_id = native_tok.pad_token_id if native_tok.pad_token_id is not None else 151643
  data_columns = ("prompt", "chosen", "rejected")
  
  grain_formatter = DPODataFormatting(
      pad_id=pad_id,
      max_target_length=1024,
      data_column_names=data_columns,
      max_prompt_length=512
  )
  processed = grain_formatter.map(element)

  # Build TrainingInput
  from tunix.sft.dpo.dpo_trainer import TrainingInput, dpo_loss_fn
  from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig
  
  # Convert NumPy formatted records to JAX Arrays
  training_input = TrainingInput(
      prompt_ids=jnp.array([processed["prompt_ids"]]),
      prompt_mask=jnp.array([processed["prompt_mask"]]),
      chosen_ids=jnp.array([processed["chosen_ids"]]),
      chosen_mask=jnp.array([processed["chosen_mask"]]),
      rejected_ids=jnp.array([processed["rejected_ids"]]),
      rejected_mask=jnp.array([processed["rejected_mask"]])
  )

  # Instantiate Tunix DPOTrainingConfig and DPOTrainer boundaries
  tunix_config = DPOTrainingConfig(
      eval_every_n_steps=2,
      max_steps=10,
      checkpoint_root_directory="/tmp/dpo_ckpts",
      algorithm="dpo",
      beta=0.1,
      label_smoothing=0.0,
      max_prompt_length=512,
      max_response_length=512
  )
  
  trainer = DPOTrainer(
      model=model,
      ref_model=nnx.clone(model),
      optimizer=optax.sgd(0.0), # No-op since we only run forward/loss checks
      training_config=tunix_config
  )

  # Invoke the input preparation boundary _prepare_inputs (converts to unified TrainExample)
  train_example = trainer._prepare_inputs(training_input)

  # Execute loss evaluations inside the mesh context
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    # Compute DPO Loss using Tunix loss function
    loss_jax, aux_metrics = dpo_loss_fn(
        model=model,
        train_example=train_example,
        algorithm="dpo",
        beta=0.1,
        label_smoothing=0.0
    )

    # Extract chosen and rejected log-probs
    from tunix.sft.dpo.dpo_trainer import compute_logps
    chosen_logps_jax, rejected_logps_jax = compute_logps(
        model=model,
        input_ids=train_example.input_ids,
        positions=train_example.positions,
        attention_mask=train_example.attention_mask,
        logits_to_keep=train_example.logits_to_keep,
        completion_mask=train_example.completion_mask
    )

    # Evaluate reference logps
    ref_chosen_logps_jax, ref_rejected_logps_jax = compute_logps(
        model=trainer.ref_model,
        input_ids=train_example.input_ids,
        positions=train_example.positions,
        attention_mask=train_example.attention_mask,
        logits_to_keep=train_example.logits_to_keep,
        completion_mask=train_example.completion_mask
    )

    # Pull matrices back to CPU hosts
    loss_np = np.array(loss_jax)
    chosen_logps_np = np.array(chosen_logps_jax)
    rejected_logps_np = np.array(rejected_logps_jax)
    
    ref_chosen_logps_np = np.array(ref_chosen_logps_jax)
    ref_rejected_logps_np = np.array(ref_rejected_logps_jax)

  # Pull auxiliary metrics
  rewards_accuracy = np.array(aux_metrics["rewards/accuracy"])
  rewards_margin = np.array(aux_metrics["rewards/margin"])
  rewards_chosen = np.array(aux_metrics["rewards/chosen"])
  rewards_rejected = np.array(aux_metrics["rewards/rejected"])

  print(f"MaxText Chosen logps: {chosen_logps_np[0]:.6f}")
  print(f"MaxText Rejected logps: {rejected_logps_np[0]:.6f}")
  print(f"MaxText DPO Loss: {loss_np:.6f}")
  print(f"MaxText Rewards Accuracy: {rewards_accuracy:.6f}")

  # Save MaxText DPO parameters locally for modular comparative check
  np.save("quals/logs/maxtext_dpo_chosen_ids.npy", np.array(processed["prompt_ids"].tolist() + processed["chosen_ids"].tolist()))
  np.save("quals/logs/maxtext_dpo_rejected_ids.npy", np.array(processed["prompt_ids"].tolist() + processed["rejected_ids"].tolist()))
  np.save("quals/logs/maxtext_dpo_chosen_mask.npy", np.array([0] * 512 + processed["chosen_mask"].tolist()))
  np.save("quals/logs/maxtext_dpo_rejected_mask.npy", np.array([0] * 512 + processed["rejected_mask"].tolist()))
  
  np.save("quals/logs/maxtext_dpo_chosen_logps.npy", chosen_logps_np)
  np.save("quals/logs/maxtext_dpo_rejected_logps.npy", rejected_logps_np)
  np.save("quals/logs/maxtext_dpo_loss.npy", loss_np)
  np.save("quals/logs/maxtext_dpo_margin.npy", rewards_margin)
  np.save("quals/logs/maxtext_dpo_accuracy.npy", rewards_accuracy)

  print("SUCCESS: MaxText DPO stack baseline parameters dumped successfully to quals/logs/")

if __name__ == "__main__":
  main()
