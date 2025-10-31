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
A plan agent to analysis the HF & Maxtext models architecture and generate a conversion plan in json format.
"""

import argparse
import json
import os
from MaxText.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template, load_json, load_text_file
from MaxText.experimental.agent.ckpt_conversion_agent.base import BaseAgent


class PlanAgent(BaseAgent):
  """
  An agent that demonstrates a multi-step prompt chain to generate a model
  conversion script, with verification that every parameter is mapped.
  """

  def __init__(self, api_key, dir_path, target_model="gemma3", max_retries=3):
    """
    Initializes the PlanAgent.

    Args:
        target_model (str): The target model for conversion.
        max_retries (int): The maximum number of retries for generation.
    """
    super().__init__(api_key)

    self.target_model = target_model
    self.max_retries = max_retries
    self.dir_path = dir_path
    self.maxtext_params = load_json(f"{dir_path}/context/{target_model}/maxtext_params.json")
    self.hf_params = load_json(f"{dir_path}/context/{target_model}/hf_params.json")
    self.dsl = load_text_file(f"{dir_path}/context/dsl.txt")
    self.analysis = load_text_file(f"{dir_path}/outputs/analysis.txt")
    self.prompt_templates = self._load_prompt_templates()

  def _load_prompt_templates(self):
    """Loads all necessary prompt templates."""
    templates = {
      "plan": load_prompt_template(f"{self.dir_path}/prompts/01_plan.txt"),
      "plan_check": load_prompt_template(f"{self.dir_path}/prompts/01_plan_check.txt"),
      "pitfalls": load_prompt_template(f"{self.dir_path}/prompts/04_pitfalls.txt"),
    }
    return templates

  def plan_conversion(self):
    """Json Plan"""
    plan = None
    feedback = ""
    for attempt in range(1, self.max_retries + 1):
      prompt2 = self.prompt_templates["plan"].format(
        analysis=self.analysis,
        dsl=self.dsl,
        feedback=feedback,
      )
      plan = self.generate_text(prompt2)

      check_prompt = self.prompt_templates["plan_check"].format(
        analysis=self.analysis,
        plan=plan,
      )

      feedback = self.generate_text(check_prompt)

      print(f"  Validator Call {attempt}...")
      print(feedback)

      if "yes" in feedback.lower():
        candidate_code = plan
        print("  Passed Validator...")
        break
      else:
        if attempt == self.max_retries:
          raise RuntimeError(f"Max attempts tried for {attempt}")

    # Save the conversion plan json
    output_dir = f"{self.dir_path}/outputs"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "plan.json")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        json.dump(candidate_code, f, ensure_ascii=False, indent=4)
      print(f"Plan successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving analysis file: {e}")
    print("-----------------------------------------------------\n")
    return candidate_code


if __name__ == "__main__":
  # 1. Define the target model
  TARGET_MODEL = "gemma3-4b"
  parser = argparse.ArgumentParser(description="A script to process model transformations.")
  parser.add_argument("--target_model", type=str, required=True, help='The name of the target model (e.g., "GEMMA3").')
  parser.add_argument(
    "--dir_path", type=str, required=True, help='The file path to the context directory (e.g., "context/gemma3").'
  )
  parser.add_argument("--api_key", type=str, help="Optional API key for external services.")
  args = parser.parse_args()
  agent = PlanAgent(api_key=args.api_key, dir_path=args.dir_path, target_model=TARGET_MODEL)
  agent.plan_conversion()
