# Copyright 2026 Google LLC
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

"""Static checks for the Qwen3.5 35B GSM8K post-training recipe."""

import json
from pathlib import Path

import pytest
import yaml


pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RECIPE_PATH = _REPO_ROOT / "src/maxtext/configs/post_train/rl_gsm8k_qwen35_35b_v5p64.yml"
_TEMPLATE_PATH = _REPO_ROOT / "src/maxtext/examples/chat_templates/qwen35_math_rl.json"


def test_qwen35_recipe_keeps_only_reproducible_runtime_configuration():
  recipe = yaml.safe_load(_RECIPE_PATH.read_text(encoding="utf-8"))

  assert recipe["model_name"] == "qwen3.5-35b-a3b"
  assert recipe["scan_layers"]
  assert recipe["vllm_hf_overrides"] == {"architectures": ["MaxTextForCausalLM"]}
  assert recipe["vllm_additional_config"]["maxtext_config"] == {
      "model_name": "qwen3.5-35b-a3b",
      "model_call_mode": "inference",
      "attention": "vllm_rpa",
      "allow_split_physical_axes": True,
      "log_config": False,
      "weight_dtype": "bfloat16",
      "prefuse_moe_weights": True,
  }
  assert not recipe["enable_prefix_caching"]
  assert recipe["reasoning_start_token_in_prompt"]
  assert recipe["stop_strings"] == ["</answer>"]
  assert not recipe["debug"]

  experiment_only_keys = {
      "base_output_directory",
      "enable_tunix_perf_metrics",
      "load_parameters_path",
      "run_name",
  }
  assert experiment_only_keys.isdisjoint(recipe)


def test_qwen35_template_does_not_nest_model_specific_chat_or_reasoning_tags():
  template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))

  assert "{solution_start_token}" in template["SYSTEM_PROMPT"]
  assert "{solution_end_token}" in template["SYSTEM_PROMPT"]
  assert "{reasoning_start_token}" not in template["SYSTEM_PROMPT"]
  assert "{reasoning_end_token}" not in template["SYSTEM_PROMPT"]
  assert "<|im_start|>" not in template["TEMPLATE"]
  assert "<start_of_turn>" not in template["TEMPLATE"]
