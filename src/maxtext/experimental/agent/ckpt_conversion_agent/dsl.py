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
A DSL agent class, to propose potential new DSL rules/ops based on the previous analysis call. 
"""
import argparse
import os
from maxtext.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template, load_text_file
from maxtext.experimental.agent.ckpt_conversion_agent.base import BaseAgent


class DSLAgent(BaseAgent):
  """DSL Agent"""

  def __init__(self, api_key, dir_path, target_model="gemma3-4b", max_retries=3):
    """
    Initializes the DSLAgent.

    Args:
        target_model (str): The target model for conversion.
        max_retries (int): The maximum number of retries for generation.
    """
    # Initialize the parent BaseAgent with the client
    super().__init__(api_key)

    self.target_model = target_model
    self.max_retries = max_retries
    self.dir_path = dir_path
    self.analysis = load_text_file(f"{self.dir_path}/outputs/analysis.txt")
    self.dsl = load_text_file(f"{self.dir_path}/context/dsl.txt")
    self.prompt_templates = self._load_prompt_templates()

  def _load_prompt_templates(self):
    """Loads all necessary prompt templates."""
    templates = {
        "dsl": load_prompt_template(f"{self.dir_path}/prompts/04_dsl.txt"),
    }
    return templates

  def verify_dsl(self):
    """Verify DSL."""
    prompt = self.prompt_templates["dsl"].format(
        analysis=self.analysis,
        target_model=self.target_model,
        dsl=self.dsl,
    )

    verification_dsl = self.generate_text(prompt)

    # Save the analysis to a file
    output_dir = f"{self.dir_path}/outputs"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "proposed_dsl.txt")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        f.write(verification_dsl)
      print(f"Proposed new DSL successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving analysis file: {e}")

    print(f"\nGrammar Validation is done, Check the results in {file_path} for potential revisions\n")
    print("-----------------------------------------------------\n")
    return verification_dsl


if __name__ == "__main__":
  TARGET_MODEL = "gemma3-4b"
  parser = argparse.ArgumentParser(description="A script to process model transformations.")
  parser.add_argument("--target_model", type=str, required=True, help='The name of the target model (e.g., "GEMMA3").')
  parser.add_argument(
      "--dir_path", type=str, required=True, help='The file path to the context directory (e.g., "context/gemma3").'
  )
  parser.add_argument("--api_key", type=str, help="Optional API key for external services.")
  args = parser.parse_args()
  agent = DSLAgent(api_key=args.api_key, dir_path=args.dir_path, target_model=TARGET_MODEL)
  global_verification_dsl = agent.verify_dsl()
