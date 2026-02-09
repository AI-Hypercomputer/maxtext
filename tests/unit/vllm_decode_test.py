# Copyright 2025 Google LLC
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

"""Tests for vllm_decode with dummy weights."""

import io
import unittest
from contextlib import redirect_stdout
from src.maxtext.vllm_decode import decode_with_vllm

import pytest

pytestmark = [pytest.mark.tpu_only, pytest.mark.external_serving]


class VllmDecodeTest(unittest.TestCase):
  """Tests for vLLM decode with dummy weights."""

  def test_decode_with_vllm_dummy_weights(self):
    """Test decode_with_vllm function with dummy weights for qwen3-30b-a3b."""
    # Import here to avoid import errors when dependencies are not available

    # Test parameters matching the command provided
    model_name = "qwen3-30b-a3b"
    hf_model_name = "Qwen/Qwen3-30B-A3B"
    hf_config_path = "src/MaxText/integration/vllm/maxtext_vllm_adapter"
    load_parameters_path = None  # Use dummy weights
    ici_data_parallelism = 1
    ici_tensor_parallelism = 4
    ici_expert_parallelism = 2
    enable_dp_attention = False
    max_prefill_length = 512
    max_target_length = 1024
    gpu_memory_utilization = 0.6
    enable_expert_parallel = True
    prompt = "Suggest some famous landmarks in London."
    decode_sampling_temperature = 0.0
    decode_sampling_nucleus_p = 1.0
    decode_sampling_top_k = 1.0
    debug_sharding = True

    # Capture stdout to verify output
    f = io.StringIO()
    with redirect_stdout(f):
      decode_with_vllm(
          model_name=model_name,
          hf_model_name=hf_model_name,
          hf_config_path=hf_config_path,
          load_parameters_path=load_parameters_path,
          ici_data_parallelism=ici_data_parallelism,
          ici_tensor_parallelism=ici_tensor_parallelism,
          ici_expert_parallelism=ici_expert_parallelism,
          enable_dp_attention=enable_dp_attention,
          max_prefill_length=max_prefill_length,
          max_target_length=max_target_length,
          gpu_memory_utilization=gpu_memory_utilization,
          enable_expert_parallel=enable_expert_parallel,
          prompt=prompt,
          decode_sampling_temperature=decode_sampling_temperature,
          decode_sampling_nucleus_p=decode_sampling_nucleus_p,
          decode_sampling_top_k=decode_sampling_top_k,
          debug_sharding=debug_sharding,
      )

    captured_output = f.getvalue()

    # Verify that the function executed and generated output
    self.assertIn("Initializing LLM", captured_output)
    self.assertIn("Generating output", captured_output)
    self.assertIn("Prompt:", captured_output)
    self.assertIn(prompt, captured_output)

  def test_decode_with_vllm_smaller_config(self):
    """Test decode_with_vllm with a smaller configuration for faster testing."""
    # Smaller test configuration
    model_name = "qwen3-30b-a3b"
    hf_model_name = "Qwen/Qwen3-30B-A3B"
    hf_config_path = "src/MaxText/integration/vllm/maxtext_vllm_adapter"
    load_parameters_path = None  # Use dummy weights
    ici_data_parallelism = 1
    ici_tensor_parallelism = 2  # Smaller TP for faster test
    ici_expert_parallelism = 1  # Smaller EP for faster test
    enable_dp_attention = False
    max_prefill_length = 64  # Smaller for faster test
    max_target_length = 128  # Smaller for faster test
    gpu_memory_utilization = 0.5
    enable_expert_parallel = False  # Disable for simpler test
    prompt = "Hello"
    decode_sampling_temperature = 0.0
    decode_sampling_nucleus_p = 1.0
    decode_sampling_top_k = 1.0
    debug_sharding = False

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
      decode_with_vllm(
          model_name=model_name,
          hf_model_name=hf_model_name,
          hf_config_path=hf_config_path,
          load_parameters_path=load_parameters_path,
          ici_data_parallelism=ici_data_parallelism,
          ici_tensor_parallelism=ici_tensor_parallelism,
          ici_expert_parallelism=ici_expert_parallelism,
          enable_dp_attention=enable_dp_attention,
          max_prefill_length=max_prefill_length,
          max_target_length=max_target_length,
          gpu_memory_utilization=gpu_memory_utilization,
          enable_expert_parallel=enable_expert_parallel,
          prompt=prompt,
          decode_sampling_temperature=decode_sampling_temperature,
          decode_sampling_nucleus_p=decode_sampling_nucleus_p,
          decode_sampling_top_k=decode_sampling_top_k,
          debug_sharding=debug_sharding,
      )

    captured_output = f.getvalue()

    # Verify execution
    self.assertIn("Initializing LLM", captured_output)
    self.assertIn(prompt, captured_output)

  def test_decode_with_vllm_parameters_validation(self):
    """Test that decode_with_vllm validates and uses parameters correctly."""
    # Test with different sampling parameters
    model_name = "qwen3-30b-a3b"
    hf_model_name = "Qwen/Qwen3-30B-A3B"
    hf_config_path = "src/MaxText/integration/vllm/maxtext_vllm_adapter"
    load_parameters_path = None
    ici_data_parallelism = 1
    ici_tensor_parallelism = 1
    ici_expert_parallelism = 1
    enable_dp_attention = False
    max_prefill_length = 32
    max_target_length = 64
    gpu_memory_utilization = 0.5
    enable_expert_parallel = False
    prompt = "Test"
    decode_sampling_temperature = 0.7  # Non-zero temperature
    decode_sampling_nucleus_p = 0.9  # Nucleus sampling
    decode_sampling_top_k = 50.0  # Top-k sampling
    debug_sharding = False

    # This should execute without errors
    f = io.StringIO()
    with redirect_stdout(f):
      decode_with_vllm(
          model_name=model_name,
          hf_model_name=hf_model_name,
          hf_config_path=hf_config_path,
          load_parameters_path=load_parameters_path,
          ici_data_parallelism=ici_data_parallelism,
          ici_tensor_parallelism=ici_tensor_parallelism,
          ici_expert_parallelism=ici_expert_parallelism,
          enable_dp_attention=enable_dp_attention,
          max_prefill_length=max_prefill_length,
          max_target_length=max_target_length,
          gpu_memory_utilization=gpu_memory_utilization,
          enable_expert_parallel=enable_expert_parallel,
          prompt=prompt,
          decode_sampling_temperature=decode_sampling_temperature,
          decode_sampling_nucleus_p=decode_sampling_nucleus_p,
          decode_sampling_top_k=decode_sampling_top_k,
          debug_sharding=debug_sharding,
      )

    captured_output = f.getvalue()
    self.assertIn("Initializing LLM", captured_output)


if __name__ == "__main__":
  unittest.main()
