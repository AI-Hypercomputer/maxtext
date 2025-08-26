# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
main script to execute the multi-agent workflow for model-specific mappings generation
"""
import argparse

from MaxText.experimental.agent.ckpt_conversion_agent.analysis import AnalysisAgent
from MaxText.experimental.agent.ckpt_conversion_agent.dsl import DSLAgent


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A script to process model transformations.")

  parser.add_argument("--target_model", type=str, required=True, help='The name of the target model (e.g., "GEMMA3").')
  parser.add_argument(
      "--dir_path", type=str, required=True, help='The file path to the context directory (e.g., "context/gemma3").'
  )
  parser.add_argument("--api_key", type=str, help="Optional API key for external services.")
  args = parser.parse_args()

  TARGET_MODEL = args.target_model
  dir_path = args.dir_path
  api_key = args.api_key

  analysisAgent = AnalysisAgent(api_key=api_key, dir_path=dir_path, target_model=TARGET_MODEL)
  analysisAgent.analyze_model_structures()

  dslAgent = DSLAgent(api_key=api_key, dir_path=dir_path, target_model=TARGET_MODEL)
  dslAgent.verify_dsl()

  # Human interaction needed,
  # Before proceed, check outputs/proposed_dsl.txt to consider if new ops are needed
