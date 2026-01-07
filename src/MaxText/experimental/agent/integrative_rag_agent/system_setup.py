# Copyright 2023–2026 Google LLC
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
This file sets up the necessary directories and configurations for the Integrative RAG Agent.
It ensures that all required directories are created based on the configuration defined in `config.py`.
"""

import os

from MaxText.experimental.agent.integrative_rag_agent import config


def setup_directories():
  """Creates all necessary directories defined in the config."""
  os.makedirs(config.data_set_folder, exist_ok=True)
  os.makedirs(config.logs_folder, exist_ok=True)
  os.makedirs(config.results_folder, exist_ok=True)
  os.makedirs(config.status_folder, exist_ok=True)
  os.makedirs(config.cache_folder, exist_ok=True)
  os.makedirs(config.similar_block_folder, exist_ok=True)
  # This directory seems to be unused, but we'll create it for now.
  # Consider adding it to config.py if it's a configurable path.
  os.makedirs("Rag_Found", exist_ok=True)
