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

"""Global variable constants used throughout the codebase"""

import os.path

# This is the maxtext package root (src/maxtext)
# Since this file is at src/maxtext/utils/globals.py, we need to go up 2 levels
MAXTEXT_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This is the maxtext repo root: with ".git" folder; "README.md"; "pyproject.toml"; &etc.
MAXTEXT_REPO_ROOT = os.environ.get(
    "MAXTEXT_REPO_ROOT",
    r
    if os.path.isdir(
        os.path.join(r := os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".git")
    )
    else MAXTEXT_PKG_DIR,
)

# This is the configs root: with "base.yml"; "models/"; &etc.
MAXTEXT_CONFIGS_DIR = os.environ.get("MAXTEXT_CONFIGS_DIR", os.path.join(MAXTEXT_PKG_DIR, "configs"))

# This is the assets root: with "tokenizers/"; &etc.
MAXTEXT_ASSETS_ROOT = os.environ.get("MAXTEXT_ASSETS_ROOT", os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "assets"))

# This is the test assets root: with "golden_logits"; &etc.
MAXTEXT_TEST_ASSETS_ROOT = os.environ.get("MAXTEXT_TEST_ASSETS_ROOT", os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets"))

EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

# Mapping from MaxText model key to Hugging Face tokenizer identifiers
HF_IDS = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma3-4b": "google/gemma-3-4b-it",  # hf multi-modal should also support the pure-text
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gemma4-26b": "google/gemma-4-26b-a4b-it",
    "gemma4-31b": "google/gemma-4-31b-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.1-405b": "meta-llama/Llama-3.1-405B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B-Base",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "qwen3-480b-a35b": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek2-16b": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek3-671b": "deepseek-ai/DeepSeek-V3",
    "deepseek3.2-671b": "deepseek-ai/DeepSeek-V3.2",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-omni-30b-a3b": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3-next-80b-a3b": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "olmo3-7b-pt": "allenai/Olmo-3-1025-7B",
    "olmo3-32b": "allenai/Olmo-3-32B-Think",
    # "default" is not HF model, but adding to to avoid confusing warning about tokenizer_path
    "default": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers/tokenizer.llama2"),
}

__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "MAXTEXT_ASSETS_ROOT",
    "MAXTEXT_CONFIGS_DIR",
    "MAXTEXT_PKG_DIR",
    "MAXTEXT_REPO_ROOT",
    "MAXTEXT_TEST_ASSETS_ROOT",
    "HF_IDS",
]
