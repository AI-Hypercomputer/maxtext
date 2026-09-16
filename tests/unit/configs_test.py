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
import ast
import functools
from copy import deepcopy

import pytest

import yaml

from pydantic import ValidationError
from yaml import YAMLError

from maxtext.configs import types as pydantic_types
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

# Define the root directory where configuration files are located.
CONFIGS_DIR = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs")


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


def normalize_config_dict(config_dict: dict) -> dict:
  """Aligns a merged YAML dict with the Pydantic model before validation.

  Drops the base_config pointer, renames num_epochs, and turns the YAML string
  "none" into a real None.
  """
  config_dict = dict(config_dict)
  if "base_config" in config_dict:
    del config_dict["base_config"]
  if "num_epochs" in config_dict:
    config_dict["num_epoch"] = config_dict.pop("num_epochs")
  for key, value in config_dict.items():
    if isinstance(value, str) and value.lower() == "none":
      config_dict[key] = None
  return config_dict


def run_config_validation(config_file_path: str):
  """
  Core validation logic: loads, merges, and validates a single config file.
  """
  print(f"\nTesting configuration file: {config_file_path}")
  try:
    # Step 1: Load the YAML file and all its parents.
    config_dict = load_and_merge_yamls(config_file_path)

    # Pre-process dictionary to align with Pydantic model before validation.
    config_dict = normalize_config_dict(config_dict)

    # Step 2: Attempt to instantiate the Pydantic model.
    # This is where validation happens. If there are extra fields,
    # missing fields, or type mismatches, a ValidationError is raised.
    if "rl.yml" in config_file_path:
      pydantic_instance = pydantic_types.RLConfig(**config_dict)
    else:
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
    os.path.join(CONFIGS_DIR, "post_train", "dpo.yml"),
    os.path.join(CONFIGS_DIR, "gpu/gpu_smoke_test.yml"),
    os.path.join(CONFIGS_DIR, "post_train", "rl.yml"),
    os.path.join(CONFIGS_DIR, "post_train", "rl_mt_jt.yml"),
    os.path.join(CONFIGS_DIR, "post_train", "sft.yml"),
    os.path.join(CONFIGS_DIR, "post_train", "sft-vision-chartqa.yml"),
    os.path.join(CONFIGS_DIR, "post_train", "sft-vision-slidevqa.yml"),
    os.path.join(CONFIGS_DIR, "tpu/tpu_smoke_test.yml"),
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
    os.path.join(CONFIGS_DIR, "models", "gemma4-e2b.yml"),
    os.path.join(CONFIGS_DIR, "models", "gemma4-e4b.yml"),
]


@pytest.mark.parametrize("config_file", GEMMA_CONFIGS)
def test_gemma_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 3: Llama Model Family ---

LLAMA_CONFIGS = [
    os.path.join(CONFIGS_DIR, "gpu", "models", "llama2_7b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "llama2_70b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "llama3_8b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "llama3_70b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "llama3.1_405b.yml"),
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
    os.path.join(CONFIGS_DIR, "models", "deepseek3-test.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-671b.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-671b-2dfsdp.yml"),
    os.path.join(CONFIGS_DIR, "models", "deepseek3-671b-batchsplit.yml"),
]


@pytest.mark.parametrize("config_file", DEEPSEEK_CONFIGS)
def test_deepseek_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 6: Mistral & Mixtral Model Family ---

MISTRAL_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "mistral-7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "mixtral-8x7b.yml"),
    os.path.join(CONFIGS_DIR, "models", "mixtral-8x22b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "mixtral_8x1b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "mixtral_8x2b.yml"),
    os.path.join(CONFIGS_DIR, "gpu", "models", "mixtral_8x7b.yml"),
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
    os.path.join(CONFIGS_DIR, "inference", "inference.yml"),
    os.path.join(CONFIGS_DIR, "inference", "inference_jetstream.yml"),
    os.path.join(CONFIGS_DIR, "tpu", "v5e", "llama2_70b_v5e-16.yml"),
    os.path.join(CONFIGS_DIR, "tpu", "v5e", "llama3_70b_v5e-16.yml"),
    os.path.join(CONFIGS_DIR, "tpu", "v5e", "llama3_405b_v5e-64.yml"),
    os.path.join(CONFIGS_DIR, "tpu", "v6e", "inference", "llama4_maverick_v6e-64.yml"),
    os.path.join(
        MAXTEXT_REPO_ROOT,
        "src",
        "maxtext",
        "configs",
        "inference",
        "multihost",
        "disaggregation",
        "llama3_405b_v6e-16-16.yml",
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "inference", "multihost", "interleaved", "llama2_70b_v5e-16.yml"
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "inference", "multihost", "interleaved", "llama3_70b_v5e-16.yml"
    ),
    os.path.join(
        MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "inference", "multihost", "interleaved", "llama3_405b_v5e-64.yml"
    ),
]


@pytest.mark.parametrize("config_file", INFERENCE_CONFIGS)
def test_inference_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 10: remat_policy "custom" tensor selection ---

# Fields declared as RematLocation but with no matching checkpoint_name(...) call
# anywhere in the model code, so remat_policy: custom cannot actually act on them.
# Listing one in CUSTOM_REMAT_TENSORS would promise control the model code cannot
# honour, so they are excluded on purpose and this set records why. Either the
# layer gains a checkpoint_name or the field should go.
NON_SELECTABLE_REMAT_FIELDS = frozenset({"engram"})

# Selectable, but only on a config that also enables the feature that owns them,
# so the parity test below cannot set them on a stock base.yml. Excluded from that
# test only -- the declared-fields test still covers them.
GATED_REMAT_FIELDS = frozenset({"indexer_cutoff_threshold"})  # needs use_indexer=True, hence MLA

MAXTEXT_SRC_DIR = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext")


def declared_remat_fields() -> set[str]:
  """Every config field whose type is RematLocation."""
  return {
      name
      for name, field in pydantic_types.MaxTextConfig.model_fields.items()
      if field.annotation is pydantic_types.RematLocation
  }


def checkpoint_names_in_tree() -> set[str]:
  """Literal names passed to checkpoint_name(...) anywhere under src/maxtext.

  Parsed with ast rather than a regex for two reasons: calls nest, as in
  checkpoint_name(checkpoint_name(x, "mlpwi_0"), "moe_mlpwi_0"), and a textual
  scan also matches checkpoint_name( written inside a comment or docstring --
  including the one in configs/types.py explaining which names are absent, which
  would make this check vacuously pass for exactly the names it should catch.
  """
  names = set()
  for dirpath, _, filenames in os.walk(MAXTEXT_SRC_DIR):
    for filename in filenames:
      if not filename.endswith(".py"):
        continue
      with open(os.path.join(dirpath, filename), "rt", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filename)
      for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
          continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called != "checkpoint_name":
          continue
        # checkpoint_name(x, "name") or checkpoint_name(x, name="name").
        argument = next((kw.value for kw in node.keywords if kw.arg == "name"), None)
        if argument is None and len(node.args) > 1:
          argument = node.args[1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
          names.add(argument.value)
  return names


def test_custom_remat_tensors_match_declared_fields():
  """The selectable list must track the declared fields, minus the inert ones."""
  selectable = set(pydantic_types.CUSTOM_REMAT_TENSORS)
  assert len(pydantic_types.CUSTOM_REMAT_TENSORS) == len(selectable), "duplicate name in CUSTOM_REMAT_TENSORS"
  assert not (selectable & NON_SELECTABLE_REMAT_FIELDS), (
      "CUSTOM_REMAT_TENSORS lists a field that has no checkpoint_name: "
      f"{sorted(selectable & NON_SELECTABLE_REMAT_FIELDS)}"
  )
  assert selectable == declared_remat_fields() - NON_SELECTABLE_REMAT_FIELDS


def test_custom_remat_tensors_have_checkpoint_names():
  """A name with no checkpoint_name call is silently inert; catch that here."""
  missing = sorted(set(pydantic_types.CUSTOM_REMAT_TENSORS) - checkpoint_names_in_tree())
  assert not missing, f"CUSTOM_REMAT_TENSORS names with no checkpoint_name(...) in the tree: {missing}"


@pytest.mark.parametrize("location", ["device", "offload"])
def test_custom_remat_parity_between_maxtext_and_rl(location):
  """MaxTextConfig and RLConfig must honour the same tensors under remat_policy: custom.

  These lists were written out twice and drifted: RLConfig's was missing
  "kv_proj", so under RL a custom policy with kv_proj: device validated cleanly
  and was then dropped with no warning.
  """
  attribute = "tensors_on_device" if location == "device" else "tensors_to_offload"
  selectable = set(pydantic_types.CUSTOM_REMAT_TENSORS) - GATED_REMAT_FIELDS

  overrides = {"remat_policy": "custom", **{name: location for name in selectable}}

  def config_dict_with_overrides(*path_parts):
    config_dict = normalize_config_dict(deepcopy(load_and_merge_yamls(os.path.join(CONFIGS_DIR, *path_parts))))
    config_dict.update(overrides)
    return config_dict

  maxtext_config = pydantic_types.MaxTextConfig(**config_dict_with_overrides("base.yml"))
  rl_config = pydantic_types.RLConfig(**config_dict_with_overrides("post_train", "rl.yml"))

  maxtext_honoured = set(getattr(maxtext_config, attribute))
  rl_honoured = set(getattr(rl_config, attribute))

  assert maxtext_honoured == rl_honoured, (
      f"only MaxTextConfig honours: {sorted(maxtext_honoured - rl_honoured)}; "
      f"only RLConfig honours: {sorted(rl_honoured - maxtext_honoured)}"
  )
  assert maxtext_honoured == selectable
