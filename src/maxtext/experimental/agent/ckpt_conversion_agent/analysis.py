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
import json
import os
from maxtext.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template, load_json, load_text_file
from maxtext.experimental.agent.ckpt_conversion_agent.base import BaseAgent


class AnalysisAgent(BaseAgent):
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
    self.prompt_templates = self._load_prompt_templates()

  def _load_prompt_templates(self):
    """Loads all necessary prompt templates."""
    templates = {
        "analysis": load_prompt_template(f"{self.dir_path}/prompts/01_analysis.txt"),
        "pitfalls": load_prompt_template(f"{self.dir_path}/prompts/04_pitfalls.txt"),
    }
    return templates

  def analyze_model_structures(self):
    """
    Analyzes the model structures of MaxText and Hugging Face parameters.
    """
    if not self.maxtext_params or not self.hf_params:
      print("Could not perform analysis due to missing parameter files.")
      return

    print("Analysis Agent: Analyzing model structures...")

    # analysis
    prompt1 = self.prompt_templates["analysis"].format(
        target_model=self.target_model,
        maxtext_params_json=json.dumps(self.maxtext_params, indent=2),
        hf_params_json=json.dumps(self.hf_params, indent=2),
        dsl=self.dsl,
        pitfalls=self.prompt_templates["pitfalls"],
    )

    # Generate the analysis
    analysis = self.generate_text(prompt1)

    # Save the analysis to a file
    output_dir = f"{self.dir_path}/outputs"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "analysis.txt")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        f.write(analysis)
      print(f"Analysis successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving analysis file: {e}")
