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
Test suite for validating MaxText YAML configurations against Pydantic models. 

This test suite uses explicit, hardcoded lists of configuration files grouped
by model family (e.g., gemma, llama) to test them directly against the Pydantic
`MaxTextConfig` model. It avoids programmatic file discovery and the complex
`pyconfig.initialize` function to provide fast, targeted feedback on validation
errors like "Extra inputs are not permitted." 
"""

import os
import functools
from copy import deepcopy

import pytest

import yaml

from pydantic import ValidationError
from yaml import YAMLError

from MaxText.configs import types as pydantic_types
from MaxText.globals import MAXTEXT_REPO_ROOT

# Define the root directory where configuration files are located.
CONFIGS_DIR = os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "configs")


@functools.lru_cache(maxsize=None)
def load_and_merge_yamls(yaml_path: str) -> dict:
  """
  Recursively loads a YAML file and merges it with its base configurations.

  A cache is used to avoid re-reading and re-parsing the same base files
  multiple times (e.g., base.yml).

  Args:
      yaml_path: The absolute path to the YAML file to load.

  Returns:
      A single merged dictionary representing the fully resolved configuration.
  """
  with open(yaml_path, "rt", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if data and "base_config" in data:
    base_path_str = data["base_config"]
    # base_config paths are relative to the current YAML file's directory.
    base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), base_path_str))
    if not os.path.exists(base_path):
      # Fallback to the main configs directory
      base_path = os.path.join(CONFIGS_DIR, base_path_str)

    base_data = deepcopy(load_and_merge_yamls(base_path))
    # The child's values overwrite the base's values.
    base_data.update(data)
    return base_data

  return data if data is not None else {}


def run_config_validation(config_file_path: str):
  """
  Core validation logic: loads, merges, and validates a single config file.
  """
  print(f"\nTesting configuration file: {config_file_path}")
  try:
    # Step 1: Load the YAML file and all its parents.
    config_dict = load_and_merge_yamls(config_file_path)

    # Pre-process dictionary to align with Pydantic model before validation.
    if "base_config" in config_dict:
      del config_dict["base_config"]
    if "num_epochs" in config_dict:
      config_dict["num_epoch"] = config_dict.pop("num_epochs")
    for key, value in config_dict.items():
      if isinstance(value, str) and value.lower() == "none":
        config_dict[key] = None

    # Step 2: Attempt to instantiate the Pydantic model.
    # This is where validation happens. If there are extra fields,
    # missing fields, or type mismatches, a ValidationError is raised.
    pydantic_instance = pydantic_types.MaxTextConfig(**config_dict)

    # Step 3: Test the "emit" part by dumping the model back to a dict.
    dumped_config = pydantic_instance.model_dump()
    assert isinstance(dumped_config, dict), "model_dump() did not return a dictionary."

  except ValidationError as e:
    pytest.fail(f"Pydantic validation FAILED for {config_file_path}:\n{e}", pytrace=False)
  except (TypeError, IOError, YAMLError) as e:
    pytest.fail(f"An unexpected error occurred for {config_file_path}:\n{type(e).__name__}: {e}", pytrace=True)


# ==============================================================================
# Begin Test Functions
# ==============================================================================

# --- Test Group 1: Base and Top-Level Configs ---

BASE_CONFIGS = [
    os.path.join(CONFIGS_DIR, "base.yml"),
    os.path.join(CONFIGS_DIR, "dpo.yml"),
    os.path.join(CONFIGS_DIR, "gpu_smoke_test.yml"),
    os.path.join(CONFIGS_DIR, "rl.yml"),
    os.path.join(CONFIGS_DIR, "rl_mt_jt.yml"),
    os.path.join(CONFIGS_DIR, "sft.yml"),
    os.path.join(CONFIGS_DIR, "sft-vision-chartqa.yml"),
    os.path.join(CONFIGS_DIR, "sft-vision-slidevqa.yml"),
    os.path.join(CONFIGS_DIR, "tpu_smoke_test.yml"),
]


@pytest.mark.parametrize("config_file", BASE_CONFIGS)
def test_base_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 2: Gemma Model Family ---

GEMMA_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "gemma-2b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma-7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma2-2b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma2-9b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma2-27b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma3-4b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma3-12b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma3-27b.yml"),
]


@pytest.mark.parametrize("config_file", GEMMA_CONFIGS)
def test_gemma_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 3: Llama Model Family ---

LLAMA_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "gpu", "llama2_7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "llama2_70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "llama3_8b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "llama3_70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "llama3.1_405b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama2-7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama2-13b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama2-70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3-8b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3-70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3-405b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3.1-8b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3.1-70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3.1-405b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama3.3-70b.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama4-17b-16e.yml"),
    os.path.join(CONFIGS_DIR, "models", "llama4-17b-128e.yml"),
]


@pytest.mark.parametrize("config_file", LLAMA_CONFIGS)
def test_llama_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 4: GPT Model Family ---

GPT_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "gpt3-52k.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpt3-6b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpt3-22b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpt3-175b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpt-oss-20b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpt-oss-120b.yml"),
]


@pytest.mark.parametrize("config_file", GPT_CONFIGS)
def test_gpt_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 5: DeepSeek Model Family ---

DEEPSEEK_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "deepseek2-16b.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek2-236b.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-tiny.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-test.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-671b.yml"),
]


@pytest.mark.parametrize("config_file", DEEPSEEK_CONFIGS)
def test_deepseek_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 6: Mistral & Mixtral Model Family ---

MISTRAL_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "mistral-7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "mixtral-8x7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "mixtral-8x22b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "mixtral_8x1b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "mixtral_8x2b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gpu", "mixtral_8x7b.yml"),
]


@pytest.mark.parametrize("config_file", MISTRAL_CONFIGS)
def test_mistral_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 7: Qwen Model Family ---

QWEN_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "qwen3-0.6b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-4b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-4b-thinking-2507.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-8b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-14b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-32b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-235b-a22b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-30b-a3b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-480b-a35b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-next-80b-a3b.yml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-omni-30b-a3b.yml"),
]


@pytest.mark.parametrize("config_file", QWEN_CONFIGS)
def test_qwen_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 8: Kimi Model Family ---

KIMI_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "kimi-k2-1t.yml"),
]


@pytest.mark.parametrize("config_file", KIMI_CONFIGS)
def test_kimi_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 9: Inference-specific Configs ---

INFERENCE_CONFIGS = [
    os.path.join(CONFIGS_DIR, "inference.yml"),
    os.path.join(CONFIGS_DIR, "inference_jetstream.yml"),
    os.path.join(CONFIGS_DIR, "v5e", "llama2_70b_v5e-16.yml"),
    os.path.join(CONFIGS_DIR, "v5e", "llama3_70b_v5e-16.yml"),
    os.path.join(CONFIGS_DIR, "v5e", "llama3_405b_v5e-64.yml"),
    os.path.join(CONFIGS_DIR, "v6e", "inference", "llama4_maverick_v6e-64.yml"),
    os.path.join(
        MAXTEXT_REPO_ROOT,
        "src",
        "MaxText",
        "inference",
        "configs",
        "multi_host",
        "disaggregation",
        "llama3_405b_v6e-16-16.yml",
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "MaxText", "inference", "configs", "multi_host", "interleaved", "llama2_70b_v5e-16.yml"
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "MaxText", "inference", "configs", "multi_host", "interleaved", "llama3_70b_v5e-16.yml"
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "MaxText", "inference", "configs", "multi_host", "interleaved", "llama3_405b_v5e-64.yml"
    ),
]


@pytest.mark.parametrize("config_file", INFERENCE_CONFIGS)
def test_inference_configs(config_file):
  run_config_validation(config_file)
