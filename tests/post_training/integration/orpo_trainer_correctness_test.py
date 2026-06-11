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

"""Integration test validating ORPO training step metrics against stored golden outputs.
This runs JAX-only training and does not require PyTorch at test execution time.
The test runs on CPU to ensure maximum reproducibility and eliminate GPU/TPU floating point differences.

How to regenerate the golden data:
  If the model implementation or training logic changes and you need to regenerate
  the golden logits, please follow the instructions in the parity generation script:
  tests/assets/logits_generation/generate_dpo_golden_data_and_compare_pytorch_logits.py
"""

import json
import tempfile
import unittest
import pytest
from absl.testing import parameterized

# Import shared base class and helper functions from shared correctness base
from tests.post_training.integration.dpo_correctness_base import (
    DPOCorrectnessTestBase,
    run_jax_training,
)

# We only use CPU in this test to avoid GPU/TPU differences. However, "integration_test" does not have "cpu_only" runners
# at the moment.
pytestmark = [pytest.mark.post_training, pytest.mark.integration_test]


class ORPOTRLCorrectnessTest(DPOCorrectnessTestBase):

  @parameterized.named_parameters(
      (
          "explicit_prompt_len_3_column",
          "explicit_prompt_len_3_column",
          144,
          "dpo_3_column_dataset.json",
          ["prompt", "chosen", "rejected"],
      ),
      (
          "default_prompt_len_2_column",
          "default_prompt_len_2_column",
          None,
          "dpo_2_column_dataset.json",
          ["chosen", "rejected"],
      ),
  )
  def test_maxtext_orpo_correctness(self, name, max_prompt_len, dataset_filename, data_columns):
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    max_target_length = 256
    init_weights_seed = 0
    lambda_orpo = 0.1

    # Load golden JAX metrics
    golden_path = "tests/assets/golden_logits/golden_orpo_correctness.json"
    with open(golden_path, "r", encoding="utf-8") as f:
      golden_metrics = json.load(f)

    self.assertIn(name, golden_metrics, msg=f"Scenario {name} not found in golden metrics!")
    golden = golden_metrics[name]

    # Configure JAX MaxText Config
    with tempfile.TemporaryDirectory() as temp_dir:
      config = self.build_jax_config(
          model_id=model_id,
          max_target_length=max_target_length,
          temp_dir=temp_dir,
          init_weights_seed=init_weights_seed,
          dataset_filename=dataset_filename,
          data_columns=data_columns,
          max_prompt_len=max_prompt_len,
          extra_args=[
              "dpo.algo=orpo",
              f"dpo.orpo_lambda={lambda_orpo}",
              "run_name=orpo_correctness_test",
          ],
      )

      # Run JAX ORPO Native Training Loop and get flat metrics
      jax_metrics = run_jax_training(config)

    print(f"\n=== Parity Check against Golden Assets for scenario: {name} ===")
    for key in ["loss_step_1", "margin_step_1", "loss", "margin", "chosen_logps", "rejected_logps"]:
      print(f"Metric: {key:15s} | JAX: {jax_metrics[key]:.6f} | Golden: {golden[key]:.6f}")

    # Verify JAX policy model did mutate and diverge after training steps (margin is non-zero)
    self.assertNotEqual(
        jax_metrics["margin"], 0.0, msg="JAX policy model did not mutate and diverge after training steps!"
    )

    # Assert parity between JAX run and golden reference within safe thresholds (Option B).
    # This allows the test to pass on both local CPU (CloudTop) and remote TPU VM/TPU
    # environments by accommodating hardware-specific float32 divergence.
    for key in ["loss_step_1", "margin_step_1", "loss", "margin"]:
      self.assertAlmostEqual(
          jax_metrics[key],
          golden[key],
          delta=self.ORPO_LOSS_TOLERANCE,
          msg=f"Metric {key} diverges from golden: JAX {jax_metrics[key]:.6f} vs Golden {golden[key]:.6f}",
      )
    for key in ["chosen_logps", "rejected_logps"]:
      self.assertAlmostEqual(
          jax_metrics[key],
          golden[key],
          delta=self.ORPO_LOG_PROBS_TOLERANCE,
          msg=f"Metric {key} diverges from golden: JAX {jax_metrics[key]:.6f} vs Golden {golden[key]:.6f}",
      )


if __name__ == "__main__":
  unittest.main()
