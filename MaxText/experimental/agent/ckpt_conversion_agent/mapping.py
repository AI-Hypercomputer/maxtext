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
A mapping agent, to generate param_mappings and hf_shape
"""

import json
import os
import re
from MaxText.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template, load_json, load_text_file
from MaxText.experimental.agent.ckpt_conversion_agent.base import BaseAgent


class MappingAgent(BaseAgent):
  """
  An agent that generates and verifies mapping functions for model conversion.
  """

  def __init__(self, api_key, dir_path, target_model="gemma3-4b", max_retries=3):
    """
    Initializes the MappingAgent.

    Args:
        target_model (str): The target model for conversion.
        max_retries (int): The maximum number of retries for generation.
    """
    super().__init__(api_key)

    self.target_model = target_model
    self.max_retries = max_retries
    self.dir_path = dir_path
    self.maxtext_params = load_json(f"{self.dir_path}/context/{target_model}/maxtext_params.json")
    self.hf_params = load_json(f"{self.dir_path}/context/{target_model}/hf_params.json")
    self.analysis = load_text_file(f"{self.dir_path}/outputs/analysis.txt")
    self.prompt_templates = self._load_prompt_templates()

  def _load_prompt_templates(self):
    """Loads all necessary prompt templates."""
    templates = {
        "param_mapping": load_prompt_template(f"{self.dir_path}/prompts/03_param_mapping.txt"),
        "param_mapping_check": load_prompt_template(f"{self.dir_path}/prompts/03_param_mapping_check.txt"),
        "shape_mapping": load_prompt_template(f"{self.dir_path}/prompts/05_shape_mapping.txt"),
        "shape_mapping_check": load_prompt_template(f"{self.dir_path}/prompts/05_shape_mapping_check.txt"),
        "pitfalls": load_prompt_template(f"{self.dir_path}/prompts/04_pitfalls.txt"),
    }
    return templates

  def _generate_and_verify_code(
      self, step_name, gen_prompt_key, check_prompt_key, gen_prompt_args, check_prompt_args_base, outputfile
  ):
    """
    A generic loop to generate code and verify it with a validator prompt.
    """
    print(f"Mapping Agent: Generating {step_name}...")
    candidate_code = None
    feedback = ""
    for attempt in range(1, self.max_retries + 1):
      print(f"  Attempt {attempt}...")

      gen_prompt_args["feedback"] = feedback
      prompt = self.prompt_templates[gen_prompt_key].format(**gen_prompt_args)
      candidate = self.generate_text(prompt)

      check_prompt_args = check_prompt_args_base.copy()
      check_prompt_args["code"] = candidate
      check_prompt = self.prompt_templates[check_prompt_key].format(**check_prompt_args)
      feedback = self.generate_text(check_prompt)

      print(f"  Validator Call {attempt}...")
      print(feedback)

      if "passed" in feedback.lower():
        candidate_code = candidate
        print("  Passed Validator...")
        break
      else:
        if attempt == self.max_retries:
          raise RuntimeError(f"Max attempts tried for {step_name}")

    # Save the code to a file
    output_dir = f"{self.dir_path}/outputs"
    file_path = os.path.join(output_dir, outputfile)
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        f.write(candidate_code)
      print(f"Code successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving analysis file: {e}")

    print(f"\nFinal {step_name} are saved in:{file_path}\n")
    print("-----------------------------------------------------\n")
    return candidate_code

  def generate_param_mapping(self):
    """
    Generates and verifies the parameter mapping function.
    """
    if not self.analysis or not self.maxtext_params or not self.hf_params:
      print("Could not generate param mapping due to missing files.")
      return None

    gen_args = {
        "target_model": self.target_model,
        "analysis": self.analysis,
        "maxtext_params_json": json.dumps(self.maxtext_params, indent=2),
        "hf_params_json": json.dumps(self.hf_params, indent=2),
        "pitfalls": self.prompt_templates["pitfalls"],
        "request_options": {"timeout": 300},
    }

    check_args = {
        "maxtext_params_json": json.dumps(self.maxtext_params, indent=2),
        "hf_params_json": json.dumps(self.hf_params, indent=2),
        "analysis": self.analysis,
    }

    return self._generate_and_verify_code(
        step_name="Parameter Mapping",
        gen_prompt_key="param_mapping",
        check_prompt_key="param_mapping_check",
        gen_prompt_args=gen_args,
        check_prompt_args_base=check_args,
        outputfile="param_mapping.py",
    )

  def generate_shape_mapping(self):
    """
    Generates and verifies the Hugging Face weights shape mapping function.
    """
    if not self.analysis or not self.hf_params:
      print("Could not generate shape mapping due to missing files.")
      return None

    gen_args = {
        "target_model": self.target_model,
        "hf_params_json": json.dumps(self.hf_params, indent=2),
        "analysis": self.analysis,
        "pitfalls": self.prompt_templates["pitfalls"],
    }

    check_args = {
        "hf_params_json": json.dumps(self.hf_params, indent=2),
    }

    return self._generate_and_verify_code(
        step_name="HF Shape",
        gen_prompt_key="shape_mapping",
        check_prompt_key="shape_mapping_check",
        gen_prompt_args=gen_args,
        check_prompt_args_base=check_args,
        outputfile="hf_shape.py",
    )


if __name__ == "__main__":
  TARGET_MODEL = "gemma3-4b"
  agent = MappingAgent(target_model=TARGET_MODEL)
  try:
    param_mapping_code = agent.generate_param_mapping()
    shape_mapping_code = agent.generate_shape_mapping()

  except RuntimeError as e:
    print(f"An error occurred during mapping generation: {e}")
