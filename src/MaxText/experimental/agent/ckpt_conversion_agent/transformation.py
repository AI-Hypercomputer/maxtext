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
A transformation agent to generate the 
layerwise and bidirectional transformation hook functions between HF & Maxtext
"""


import os
from MaxText.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template, load_text_file, load_json
from MaxText.experimental.agent.ckpt_conversion_agent.base import BaseAgent


class TransformationAgent(BaseAgent):
  """
  An agent that generates transformation hook functions for model conversion.
  """

  def __init__(self, api_key, dir_path, target_model="gemma3"):
    """
    Initializes the TransformationAgent.

    Args:
        target_model (str): The target model for conversion.
    """
    super().__init__(api_key)
    self.target_model = target_model
    self.dir_path = dir_path
    self.dsl = load_text_file(f"{self.dir_path}/context/dsl.txt")
    self.analysis = load_json(f"{self.dir_path}/outputs/plan.json")
    self.param_mapping_code = load_text_file(f"{self.dir_path}/outputs/param_mapping.py")
    self.pitfalls = load_text_file(f"{self.dir_path}/prompts/04_pitfalls.txt")
    self.prompt_templates = self._load_prompt_templates()

  def _load_prompt_templates(self):
    """Loads all necessary prompt templates."""
    templates = {
        "hook_fn": load_prompt_template(f"{self.dir_path}/prompts/04_hook_fn_dsl.txt"),
    }
    return templates

  def generate_hook_functions(self):
    """
    Generates layer-wise transformation hook functions.
    """
    if not all([self.analysis, self.dsl, self.param_mapping_code]):
      print("Could not generate hook functions due to missing input files.")
      return None

    prompt = self.prompt_templates["hook_fn"].format(
        plan=self.analysis,
        target_model=self.target_model,
        dsl=self.dsl,
        pitfalls=self.pitfalls,
        param_mapping_code=self.param_mapping_code,
    )

    hook_fn_code = self.generate_text(prompt)

    # Save the generated code to a file
    output_dir = f"{self.dir_path}/outputs"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "hook_fn.py")
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        f.write(hook_fn_code)
      print(f"Hook functions successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving hook functions file: {e}")

    print(f"\nTransformation Functions are saved in:{file_path}\n")
    print("-----------------------------------------------------\n")
    return hook_fn_code


if __name__ == "__main__":
  TARGET_MODEL = "gemma3-4b"
  agent = TransformationAgent(target_model=TARGET_MODEL)
  hook_fn_code = agent.generate_hook_functions()
