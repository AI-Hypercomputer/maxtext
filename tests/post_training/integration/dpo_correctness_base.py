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

"""Shared utilities, weight synchronization logic, and base class for DPO
integration correctness tests.
"""

import os
from typing import Any
from absl.testing import parameterized
import jax

# MaxText / Tunix imports
from maxtext.configs import pyconfig
from maxtext.trainers.post_train.dpo import train_dpo
from maxtext.trainers.post_train.dpo import hooks as dpo_hooks


# ==============================================================================
# 1. LOW-LEVEL STATE INTERCEPTION & WEIGHT SYNCHRONIZATION
# ==============================================================================
_original_training_hooks = dpo_hooks.DPOTrainingHooks


class InterceptingTrainingHooks(_original_training_hooks):
  """Custom training hooks class to intercept loss and rewards margin during real trainer step execution."""

  captured_metrics = []

  def on_train_step_end(self, train_ctx, train_step, train_loss, step_time=0.0):
    super().on_train_step_end(train_ctx, train_step, train_loss, step_time)

    prefix = train_ctx.metrics_prefix
    accuracy = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/accuracy", "train")[-1])
    margin = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/margin", "train")[-1])
    chosen_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/chosen", "train")[-1])
    rejected_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/rejected", "train")[-1])

    self.captured_metrics.append(
        {
            "loss": float(train_loss),
            "accuracy": accuracy,
            "margin": margin,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
        }
    )


# ==============================================================================
# 2. HIGH-LEVEL MODEL EXECUTION RUNNERS
# ==============================================================================
def run_jax_training(config):
  """Runs JAX DPO training and returns a flat dict of captured step metrics."""
  print("Executing JAX DPO train_dpo.train()...")
  train_dpo.train(config)  # Perform 3 steps of DPO training.

  # We override the TrainingHooks (see below) to capture the metrics that we need to compare against PyTorch.
  captured = InterceptingTrainingHooks.captured_metrics
  assert len(captured) == 3, f"Expected 3 steps metrics, got {len(captured)}"

  step_1 = captured[0]  # We will print the stats for this step, but won't use them for comparison.
  step_3 = captured[2]  # These are the values we want to compare against PyTorch.

  return {
      "loss_step_1": step_1["loss"],
      "margin_step_1": step_1["margin"],
      "loss": step_3["loss"],
      "margin": step_3["margin"],
      "chosen_logps": step_3["chosen_logps"],
      "rejected_logps": step_3["rejected_logps"],
  }


# ==============================================================================
# 3. SHARED BASE CLASS DEFINITION
# ==============================================================================
class DPOCorrectnessTestBase(parameterized.TestCase):
  """Shared base class establishing environment setup and configuration helpers for DPO parity tests."""

  COMMON_PROMPT = "What is preference optimization?"
  COMMON_CHOSEN = "Aligning LLMs using pairs of chosen and rejected responses is called preference optimization."
  COMMON_REJECTED = "Database operations to choose preferred options are called preference optimization."

  # Constants for sanity checking & parity.
  # These tolerances were validated by running sensitivity studies across 5 different
  # random initialization seeds (0, 42, 123, 2026, 99999).
  # Findings:
  # 1. JAX vs PyTorch CPU Parity (validated on the same host):
  #    - Loss Difference: stable across all seeds, ranging from 0.001 to 0.057.
  #    - Log Probs Difference: max diff ~0.60.
  # 2. JAX Local CPU vs Remote CPU/TPU Divergence (Option B):
  #    - Running JAX on different CPU architectures introduces float32 numerical noise.
  #    - Loss / Margin Difference: max observed diff is 0.113 (Seed 0, 2-column). We set
  #      DPO_LOSS_TOLERANCE to 0.15 to safely cover this hardware-specific divergence.
  #    - Log Probs Difference: max observed diff is 1.020 (Seed 99999, 2-column). We set
  #      LOG_PROBS_TOLERANCE to 1.5 to safely cover this hardware-specific divergence.
  LOG_PROBS_TOLERANCE = 1.5
  DPO_LOSS_TOLERANCE = 0.15

  @classmethod
  def setUpClass(cls):
    """Set up the test class by setting JAX default platform to CPU and monkeypatching the training hooks class."""
    # Set JAX default platform to CPU to eliminate CPU/TPU accelerator math variations
    jax.config.update("jax_platforms", "cpu")
    # Set JAX to CPU/TPU PRNG and SPMD defaults
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = (
          os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
      )

    # Monkeypatch the training hooks class dynamically
    dpo_hooks.DPOTrainingHooks = InterceptingTrainingHooks

  @classmethod
  def tearDownClass(cls):
    # Restore original hooks to prevent polluting other tests
    dpo_hooks.DPOTrainingHooks = _original_training_hooks

  def setUp(self):
    super().setUp()
    InterceptingTrainingHooks.captured_metrics = []

  def tearDown(self):
    super().tearDown()
    InterceptingTrainingHooks.captured_metrics = []

  # ----------------------------------------------------------------------------
  # Private Helpers for test configuration and input generation
  # ----------------------------------------------------------------------------

  def build_jax_config(
      self,
      model_id: str,
      max_target_length: int,
      temp_dir: str,
      init_weights_seed: int,
      dataset_filename: str,
      data_columns: list[str],
      max_prompt_len: int | None = None,
      extra_args: list[str] | None = None,
  ) -> Any:
    """Helper to build MaxText JAX config object for DPO."""
    dataset_path = os.path.abspath(f"tests/assets/local_datasets/dpo/{dataset_filename}")
    argv = [
        "src/maxtext/configs/base.yml",
        "model_name=qwen2.5-1.5b",
        f"tokenizer_path={model_id}",
        "scan_layers=False",
        "attention=dot_product",
        "per_device_batch_size=1",
        f"max_target_length={max_target_length}",
        "skip_jax_distributed_system=True",
        "enable_nnx=True",
        "pure_nnx=True",
        "pure_nnx_decoder=False",
        "remat_policy=full",
        "log_config=0",
        # Tiny architecture specifications.
        # This JAX model configuration must be kept structurally identical to the PyTorch
        # model configuration created in `create_pytorch_config` inside
        # `tests/assets/logits_generation/dpo_pytorch_helpers.py` to allow direct parameter
        # synchronization and logit comparison.
        "base_emb_dim=64",
        "head_dim=32",
        "base_num_query_heads=2",
        "base_num_kv_heads=2",
        "base_mlp_dim=128",
        "base_num_decoder_layers=2",
        "override_model_config=True",
        # Native input pipeline dataset specifications
        "use_dpo=True",
        "packing=False",
        "dataset_type=hf",
        "hf_path=json",
        f"hf_train_files={dataset_path}",
        "tokenize_train_data=True",
        f"train_data_columns={data_columns}",
        f"eval_data_columns={data_columns}",
        "enable_data_shuffling=False",
        "steps=3",
        f"base_output_directory={temp_dir}",
        # Set rope_interleave=False to match Hugging Face Qwen2's concatenated [x_i, x_(i+d/2)] RoPE layout,
        # rather than MaxText's default adjacent [x_(2i), x_(2i+1)] interleaved RoPE layout.
        "rope_interleave=False",
        # Explicitly hard-code init_weights_seed to ensure model initialization is reproducible and self-contained
        f"init_weights_seed={init_weights_seed}",
    ]
    if max_prompt_len is not None:
      argv.append(f"dpo.max_prompt_length={max_prompt_len}")
    if extra_args:
      argv.extend(extra_args)
    return pyconfig.initialize_pydantic(argv)
