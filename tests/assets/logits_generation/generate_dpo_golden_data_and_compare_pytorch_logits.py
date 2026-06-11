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

"""Script to execute JAX and HuggingFace TRL DPO in parallel, verify DPO correctness parity,
and output golden JAX metrics to a JSON file.

The parity is verified by running JAX and PyTorch/TRL on an identical, miniaturized 2-layer Qwen2 model.
- The JAX model is configured in `tests/post_training/integration/dpo_correctness_base.py`.
- The PyTorch model is configured in `tests/assets/logits_generation/dpo_pytorch_helpers.py`.

The flow of this test is:
* Load the same Qwen2 model (Qwen/Qwen2.5-1.5B-Instruct) in both JAX and PyTorch/TRL.
* Run the MaxText DPO training loop for a few steps (train_steps=2 in this implementation).
  This is needed to make sure that the policy model diverges from the reference model and that we can calculate
  the DPO loss and margin.
* Use custom training hooks to intercept the model parameters after 2 training steps and copy them into the Pytorch model.
* Compare the model parameters between JAX and PyTorch/TRL.
* Assert that the model parameters are identical.
* Calculate the DPO loss and margin for the last batch.
* Assert that the DPO loss and margin between JAX and PyTorch/TRL are identical.
* Save the metrics in the golden data json file.

Note: Both JAX and PyTorch/TRL are executed on CPU to ensure maximum reproducibility and
eliminate GPU/TPU floating point differences.

How to run:
  1. Install required PyTorch and Hugging Face dependencies if they are not already present in your
     virtual environment:
     $ uv pip install torch transformers datasets trl
  2. Run the script:
     $ python3 -m tests.assets.logits_generation.generate_dpo_golden_data_and_compare_pytorch_logits
"""

import json
import os
import tempfile
import jax
from transformers import AutoTokenizer, Qwen2ForCausalLM

from maxtext.trainers.post_train.dpo import hooks as dpo_hooks
from tests.post_training.integration.dpo_correctness_base import (
    DPOCorrectnessTestBase,
    run_jax_training,
    InterceptingTrainingHooks,
)
from tests.assets.logits_generation.dpo_pytorch_helpers import (
    create_pytorch_config,
    get_pytorch_reference,
    sync_jax_to_pytorch,
)


class PyTorchSyncTrainingHooks(InterceptingTrainingHooks):
  """Training hooks subclass that synchronizes JAX model parameters to PyTorch."""

  torch_policy_model = None
  torch_ref_model = None

  def on_train_step_start(self, train_ctx):
    super().on_train_step_start(train_ctx)
    if train_ctx.train_steps == 0:
      # Grab the reference model on step 0 to add extra validation that the reference model is not changing during
      # training.
      if train_ctx.ref_model is not None:
        sync_jax_to_pytorch(train_ctx.ref_model, PyTorchSyncTrainingHooks.torch_ref_model)
      else:
        sync_jax_to_pytorch(train_ctx.model, PyTorchSyncTrainingHooks.torch_ref_model)

    elif train_ctx.train_steps == 2:
      # Copy the trained policy model weights after 2 steps of training. By this time we expect the policy model to have
      # diverged from the reference model.
      sync_jax_to_pytorch(train_ctx.model, PyTorchSyncTrainingHooks.torch_policy_model)


def run_parity_and_generate_golden():
  """Runs DPO scenarios to verify parity and outputs golden JAX metrics."""
  # Setup JAX CPU options to align with base test environment setup
  jax.config.update("jax_platforms", "cpu")
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  # Instantiate a dummy base test class to invoke config builders
  base_test = DPOCorrectnessTestBase()
  DPOCorrectnessTestBase.setUpClass()
  dpo_hooks.DPOTrainingHooks = PyTorchSyncTrainingHooks

  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  max_target_length = 256
  beta = 0.1
  init_weights_seed = 0

  scenarios = {
      "explicit_prompt_len_3_column": (144, "dpo_3_column_dataset.json", ["prompt", "chosen", "rejected"]),
      "default_prompt_len_2_column": (None, "dpo_2_column_dataset.json", ["chosen", "rejected"]),
  }

  # ============================================================================
  # DPO GENERATION
  # ============================================================================
  dpo_results = {}
  print("\n>>> Running DPO Parity & Golden Generation...")
  for name, (max_prompt_len, dataset_filename, data_columns) in scenarios.items():
    print(f"\n--- Scenario: {name} ---")
    InterceptingTrainingHooks.captured_metrics = []

    # Initialize Pytorch structures
    torch_config = create_pytorch_config(max_target_length)
    torch_policy_model = Qwen2ForCausalLM(torch_config)
    torch_ref_model = Qwen2ForCausalLM(torch_config)

    PyTorchSyncTrainingHooks.torch_policy_model = torch_policy_model
    PyTorchSyncTrainingHooks.torch_ref_model = torch_ref_model

    with tempfile.TemporaryDirectory() as temp_dir:
      config = base_test.build_jax_config(
          model_id=model_id,
          max_target_length=max_target_length,
          temp_dir=temp_dir,
          init_weights_seed=init_weights_seed,
          dataset_filename=dataset_filename,
          data_columns=data_columns,
          max_prompt_len=max_prompt_len,
          extra_args=["run_name=dpo_correctness_gen"],
      )
      jax_ref = run_jax_training(config)

    if len(data_columns) == 2:
      prompt_str = f"\n\nHuman: {base_test.COMMON_PROMPT}\n\nAssistant:"
      # Add a space prefix to chosen/rejected to avoid BPE prefix mismatch.
      chosen_str = " " + base_test.COMMON_CHOSEN
      rejected_str = " " + base_test.COMMON_REJECTED
    else:
      prompt_str = base_test.COMMON_PROMPT
      chosen_str = base_test.COMMON_CHOSEN
      rejected_str = base_test.COMMON_REJECTED

    py_ref = get_pytorch_reference(
        policy_model=torch_policy_model,
        ref_model=torch_ref_model,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        prompt_str=prompt_str,
        chosen_str=chosen_str,
        rejected_str=rejected_str,
        beta=beta,
        tokenize_together=(len(data_columns) == 2),
    )

    # Parity verification before writing
    chosen_diff = abs(jax_ref["chosen_logps"] - py_ref["chosen_logps"])
    rejected_diff = abs(jax_ref["rejected_logps"] - py_ref["rejected_logps"])
    loss_diff = abs(jax_ref["loss"] - py_ref["loss"])

    print(
        f"JAX Chosen: {jax_ref['chosen_logps']:.6f} | "
        f"PyTorch Chosen: {py_ref['chosen_logps']:.6f} "
        f"(diff: {chosen_diff:.6f})"
    )
    print(
        f"JAX Rejected: {jax_ref['rejected_logps']:.6f} | "
        f"PyTorch Rejected: {py_ref['rejected_logps']:.6f} "
        f"(diff: {rejected_diff:.6f})"
    )
    print(f"JAX Loss: {jax_ref['loss']:.6f} | PyTorch Loss: {py_ref['loss']:.6f} (diff: {loss_diff:.6f})")

    assert chosen_diff < DPOCorrectnessTestBase.LOG_PROBS_TOLERANCE, f"Chosen logps diff {chosen_diff} exceeds tolerance!"
    assert (
        rejected_diff < DPOCorrectnessTestBase.LOG_PROBS_TOLERANCE
    ), f"Rejected logps diff {rejected_diff} exceeds tolerance!"
    assert loss_diff < DPOCorrectnessTestBase.DPO_LOSS_TOLERANCE, f"Loss diff {loss_diff} exceeds tolerance!"

    dpo_results[name] = jax_ref

  # Write DPO Golden Logits
  dpo_output_path = "tests/assets/golden_logits/golden_dpo_correctness.json"
  with open(dpo_output_path, "w", encoding="utf-8") as f:
    json.dump(dpo_results, f, indent=2)
  print(f"\nWrote DPO golden metrics to: {dpo_output_path}")

  # Cleanup hooks
  PyTorchSyncTrainingHooks.torch_policy_model = None
  PyTorchSyncTrainingHooks.torch_ref_model = None

  # Clean up class environment setup
  DPOCorrectnessTestBase.tearDownClass()


if __name__ == "__main__":
  run_parity_and_generate_golden()
