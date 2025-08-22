"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Test for verifying model training size calculations
For dense models (llama2/3, gpt, etc.), it computes full parameter size
For sparse models (mixtral, deepseek, etc.), it computes active parameter size
"""

import unittest
import pytest
import os

from MaxText.max_utils import calculate_active_training_param_size
from MaxText.globals import PKG_DIR
from MaxText import pyconfig


class ParamSizeCalculationTest(unittest.TestCase):
  """
  Tests for verifying (active) parameter size calculation with 5% relative tolerance
  """

  @pytest.mark.cpu_only
  def test_llama3_8b(self):
    """Tests the parameter count for Llama 3 8B."""
    llama3_8b_config = {
      "model_name": "llama3-8b",
      "skip_jax_distributed_system": True,
    }
    # Llama3-8b has ~8.03B parameters:
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    golden_params = 8.03e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **llama3_8b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_llama2_7b(self):
    """Tests the parameter count for Llama 2 7B."""
    llama2_7b_config = {
      "model_name": "llama2-7b",
      "skip_jax_distributed_system": True,
    }
    # Llama2-7b has ~6.74B ACTIVE parameters:
    # https://adithyask.medium.com/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88
    golden_params = 6.74e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **llama2_7b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_llama4_17b_16e(self):
    """Tests the parameter count for Llama 4 Scout."""
    llama4_17b_16e_config = {
      "model_name": "llama4-17b-16e",
      "skip_jax_distributed_system": True,
    }
    # Llama4 Scout has ~17B ACTIVE parameters:
    # https://ai.meta.com/blog/llama-4-multimodal-intelligence/
    golden_params = 17e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **llama4_17b_16e_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_llama4_17b_128e(self):
    """Tests the parameter count for Llama 4 Maverick."""
    llama4_17b_128e_config = {
      "model_name": "llama4-17b-128e",
      "skip_jax_distributed_system": True,
    }
    # Llama4 Maverick has ~17B parameters:
    # https://ai.meta.com/blog/llama-4-multimodal-intelligence/
    golden_params = 17e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **llama4_17b_128e_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_gpt3_175b(self):
    """Tests the parameter count for GPT 3."""
    gpt3_175b_config = {
      "model_name": "gpt3-175b",
      "skip_jax_distributed_system": True,
    }
    # GPT3-175b has ~175B parameters
    golden_params = 175e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **gpt3_175b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_qwen3_14b(self):
    """Tests the parameter count for Qwen3-14b."""
    qwen3_14b_config = {
      "model_name": "qwen3-14b",
      "skip_jax_distributed_system": True,
    }
    # Qwen3-14b has ~14.8B parameters:
    # https://huggingface.co/Qwen/Qwen3-14B
    golden_params = 14.8e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **qwen3_14b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_gemma2_27b(self):
    """Tests the parameter count for Gemma2 27B."""
    gemma2_27b_config = {
      "model_name": "gemma2-27b",
      "skip_jax_distributed_system": True,
    }
    # Gemma2-27b has ~27B parameters
    golden_params = 27e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **gemma2_27b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_gemma3_27b(self):
    """Tests the parameter count for Gemma3 27B."""
    gemma3_27b_config = {
      "model_name": "gemma3-27b",
      "skip_jax_distributed_system": True,
    }
    # Gemma3-27b has ~27B parameters
    golden_params = 27e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **gemma3_27b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_params,
      delta=0.05*max(calculated_params, golden_params),
    )

  @pytest.mark.cpu_only
  def test_mistral_7b_params(self):
    """Tests the parameter count for the Mistral 7B model."""
    mistral_7b_config = {
      "model_name": "mistral-7b",
      "skip_jax_distributed_system": True,
    }

    # mistral-7b has ~7.3B active parameters:
    # https://mistral.ai/news/announcing-mistral-7b
    golden_active_params = 7.3e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **mistral_7b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_active_params,
      delta=0.05*max(calculated_params, golden_active_params),
    )

  @pytest.mark.cpu_only
  def test_mixtral_8x7b_active_params(self):
    """Tests the ACTIVE parameter count for the Mixtral 8x7B model."""
    mixtral_8x7b_config = {
      "model_name": "mixtral-8x7b",
      "skip_jax_distributed_system": True,
    }

    # mixtral-8x7b has ~12.9B active parameters:
    # https://mistral.ai/news/mixtral-of-experts
    golden_active_params = 12.9e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **mixtral_8x7b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_active_params,
      delta=0.05*max(calculated_params, golden_active_params),
    )

  @pytest.mark.cpu_only
  def test_deepseek_v2_236b_active(self):
    """Tests the ACTIVE parameter count for the DeepSeek-V2 236B model."""
    deepseek_v2_236b_config = {
      "model_name": "deepseek2-236b",
      "skip_jax_distributed_system": True,
    }
    # DeepSeek V2 has ~21B active training parameters:
    # https://arxiv.org/pdf/2405.04434 (Abstract)
    golden_active_params = 21e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **deepseek_v2_236b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_active_params,
      delta=0.05*max(calculated_params, golden_active_params),
    )

  @pytest.mark.cpu_only
  def test_deepseek_v3_671b_active(self):
    """Tests the ACTIVE parameter count for the DeepSeek-V2 236B model."""
    deepseek_v3_671b_config = {
      "model_name": "deepseek3-671b",
      "skip_jax_distributed_system": True,
    }
    # DeepSeek V3 has ~37B active parameters:
    # https://huggingface.co/deepseek-ai/DeepSeek-V3
    golden_active_params = 37e9

    cfg = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      **deepseek_v3_671b_config,
    )

    # Compute param size
    calculated_params = calculate_active_training_param_size(cfg)

    self.assertAlmostEqual(
      calculated_params,
      golden_active_params,
      delta=0.05*max(calculated_params, golden_active_params),
    )
