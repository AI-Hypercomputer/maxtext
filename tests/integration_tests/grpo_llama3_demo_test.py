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

"""
Integration test for GRPO Llama3 demo.

This test runs the grpo_llama3_demo.py script with minimal configuration
to verify the GRPO training pipeline works correctly on TPU.

ATTENTION: This test should only be run on TPU (v4-8, v5p-8, or v6e-8).

Usage:
  pytest tests/integration_tests/grpo_llama3_demo_test.py
"""

import os
import subprocess
import sys
import tempfile
import pytest

from MaxText.globals import MAXTEXT_REPO_ROOT


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_grpo_llama3_demo():
  """
  Test the GRPO Llama3 demo script with minimal configuration.

  This test verifies that the demo can:
  1. Load the dataset
  2. Initialize the model
  3. Run a few training steps
  4. Complete evaluation
  """

  # Create a temporary directory for outputs
  with tempfile.TemporaryDirectory() as temp_dir:
    # Set environment variables for the test
    env = os.environ.copy()
    env["HOME"] = temp_dir
    env["SKIP_JAX_PRECOMPILE"] = "1"

    # Build the command to run the demo
    demo_script = os.path.join(MAXTEXT_REPO_ROOT, "src/MaxText/examples/grpo_llama3_demo.py")

    # We'll modify the script to run with minimal steps for testing
    # by setting environment variables that the script can read
    env["GRPO_TEST_MODE"] = "1"
    env["GRPO_NUM_BATCHES"] = "2"  # Minimal batches for testing
    env["GRPO_NUM_TEST_BATCHES"] = "2"  # Minimal test batches
    env["GRPO_MAX_STEPS"] = "2"  # Minimal training steps

    # Run the demo script
    result = subprocess.run(
        [sys.executable, demo_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
        check=False,
    )

    # Check if the script executed successfully
    if result.returncode != 0:
      print("STDOUT:", result.stdout)
      print("STDERR:", result.stderr)
      raise AssertionError(f"GRPO Llama3 demo failed with return code {result.returncode}")

    # Verify expected outputs in the logs
    assert "HBM usage before loading model:" in result.stdout, "Model initialization not found in output"
    assert "HBM usage after loading ref model:" in result.stdout, "Reference model loading not found in output"
    assert "HBM usage after loading policy model:" in result.stdout, "Policy model loading not found in output"
    assert "Pre GRPO Training:" in result.stdout, "Pre-training evaluation not found in output"
    assert "Post GRPO Training:" in result.stdout, "Post-training evaluation not found in output"

    print("GRPO Llama3 demo test passed successfully!")


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
