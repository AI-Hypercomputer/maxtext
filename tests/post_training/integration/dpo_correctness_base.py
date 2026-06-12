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

# Force JAX/XLA to only create 1 virtual CPU device to prevent batch size scaling
# issues on large multi-core CI runners (which would cause the 20-example test
# dataset to be dropped completely due to drop_remainder=True).
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"

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
class InterceptingTrainingHooks(dpo_hooks.DPOTrainingHooks):
  """Custom training hooks class to intercept loss and rewards margin during real trainer step execution."""

  captured_metrics = []

  def on_train_step_end(self, train_ctx, train_step, train_loss, step_time=0.0):
    super().on_train_step_end(train_ctx, train_step, train_loss, step_time)

    prefix = train_ctx.metrics_prefix
    accuracy = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/accuracy", "train")[-1])
    margin = float(train_ctx.metrics_logger.get_metric_history(prefix, "rewards/margin", "train")[-1])
    chosen_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/chosen", "train")[-1])
    rejected_logps = float(train_ctx.metrics_logger.get_metric_history(prefix, "log_probs/rejected", "train")[-1])

    InterceptingTrainingHooks.captured_metrics.append(
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
def run_jax_training(config, test_only_training_hooks_class=InterceptingTrainingHooks):
  """Runs JAX DPO training and returns a flat dict of captured step metrics."""
  print("Executing JAX DPO train_dpo.train()...")
  train_dpo.train(
      config, test_only_training_hooks_class=test_only_training_hooks_class
  )  # Perform 3 steps of DPO training.

  captured = test_only_training_hooks_class.captured_metrics
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

  MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

  # These prompt and response strings must match the content of the pre-generated local
  # dataset JSON files (e.g., dpo_2_column_dataset.json and dpo_3_column_dataset.json)
  # loaded in the integration tests. If these strings are modified, the dataset files
  # must be regenerated using the parity generator script.
  COMMON_PROMPT = "What is preference optimization?"
  COMMON_CHOSEN = "Aligning LLMs using pairs of chosen and rejected responses is called preference optimization."
  COMMON_REJECTED = "Database operations to choose preferred options are called preference optimization."

  # Constants for sanity checking & parity.
  # These tolerances and platform constraints were validated by running extensive
  # sensitivity and hardware studies:
  #
  # 1. JAX vs PyTorch CPU Parity (Same Host):
  #    - Loss Difference: stable across seeds, ranging from 0.001 to 0.057.
  #    - Log Probs Difference: max observed difference ~0.60.
  #
  # 2. JAX CPU Cross-Platform and Version Drift:
  #    - Running JAX CPU on different host environments (e.g., local workstation vs remote GCE VM)
  #      or upgrading JAX versions (0.9.2 -> 0.10.0) introduces minor float32 compiler drift.
  #      - Max observed cross-platform logprobs noise: ~1.02.
  #      - Max observed cross-platform loss noise: ~0.113.
  #
  # 3. JAX CPU vs. TPU Backend Divergence (Enforcing CPU constraint):
  #    - Running JAX on TPU introduces massive compiler and float32 rounding shifts relative to CPU:
  #      - Log Probs Shift: ~42.68 (TPU v4) and ~14.83 (TPU v5p) on the exact same inputs.
  #      - Loss Shift: ~0.039 (TPU v4) and ~0.219 (TPU v5p).
  #    - Because DPO loss is based on log-ratios, the hardware shift can cancel out on some chips
  #      (like TPU v4), but diverges significantly on others (e.g., TPU v5p loss shift of 0.219
  #      exceeds safe tolerances).
  #    - Therefore, the test strictly enforces CPU execution to ensure cross-platform reproducibility.
  #
  # 4. Sensitivity to Semantic Mutations vs. Tolerances:
  #    - A 1-character dataset mutation ("responses" -> "response") shifts logprobs by 4.73 to 10.84.
  #    - A minor 1-word mutation ("Aligning" -> "Training") shifts logprobs by 18.02 to 42.86.
  #    - Since the CPU version/platform noise (~1.02) is far below the smallest semantic mutation (4.73),
  #      the tolerances are calibrated to maximize robustness to compiler drift while remaining highly
  #      sensitive to real regressions.
  LOG_PROBS_TOLERANCE = 3.0
  DPO_LOSS_TOLERANCE = 0.20

  @classmethod
  def setUpClass(cls):
    """Set up the test class by setting JAX default platform to CPU."""
    # Assert that the JAX CPU device count flag was successfully respected.
    # If JAX was imported/initialized elsewhere before our os.environ["XLA_FLAGS"]
    # statement, this assertion will fail, preventing silent test suite failures in CI.
    assert jax.local_device_count() == 1, (
        f"Expected exactly 1 local JAX device (CPU), but got {jax.local_device_count()}. "
        "This indicates that JAX was initialized before the XLA_FLAGS environment variable "
        "could be set in dpo_correctness_base.py."
    )

    # Set JAX to CPU/TPU PRNG and SPMD defaults
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = (
          os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
      )

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "cpu":
      raise RuntimeError(
          "DPO correctness tests must run on CPU (please run with environment variable JAX_PLATFORMS=cpu), "
          f"but JAX default backend is '{jax.default_backend()}'."
      )
    InterceptingTrainingHooks.captured_metrics = []

  def tearDown(self):
    super().tearDown()
    InterceptingTrainingHooks.captured_metrics = []

  # ----------------------------------------------------------------------------
  # Private Helpers for test configuration and input generation
  # ----------------------------------------------------------------------------

  def build_tiny_qwen2_jax_config(
      self,
      max_target_length: int,
      temp_dir: str,
      init_weights_seed: int,
      dataset_filename: str,
      data_columns: list[str],
      max_prompt_len: int | None = None,
      extra_args: list[str] | None = None,
  ) -> Any:
    """Helper to build a tiny Qwen2 MaxText JAX config object for DPO correctness testing."""
    dataset_path = os.path.abspath(f"tests/assets/local_datasets/dpo/{dataset_filename}")

    # Hermetically resolve the tokenizer from the local pre-packaged assets
    assets_root = os.environ.get("MAXTEXT_ASSETS_ROOT", "src/maxtext/assets")
    tokenizer_path = os.path.join(assets_root, "tokenizers", "qwen3-tokenizer")

    argv = [
        "src/maxtext/configs/base.yml",
        "model_name=qwen2.5-1.5b",
        f"tokenizer_path={tokenizer_path}",
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
