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
This is a baseline agent, using prompt-chain + validator/executer architecture
"""
import os
import json
import argparse

from maxtext.src.maxtext.experimental.agent.ckpt_conversion_agent.base import BaseAgent
from maxtext.src.maxtext.experimental.agent.ckpt_conversion_agent.utils.utils import load_prompt_template


class prompt_chaining_agent(BaseAgent):
  """
  Demonstrates a multi-step prompt chain to generate a model conversion script,
  with verification that every parameter is actually mapped.
  """

  def __init__(self, api_key, target_model="gemma3", max_retries=3, dir_path="context"):
    # Initialize the parent BaseAgent with the client
    super().__init__(api_key)
    self.target_model = target_model
    self.max_retries = max_retries
    self.dir_path = dir_path

  def run_chain(self, max_retries=3):
    """Run chain"""
    # Load context data
    with open(
        os.path.join(self.dir_path, "context", self.target_model, "maxtext_params.json"), "rt", encoding="utf8"
    ) as f:
      maxtext_params = json.load(f)
    with open(os.path.join(self.dir_path, "context", self.target_model, "hf_params.json"), "rt", encoding="utf8") as f:
      hf_params = json.load(f)

    # Load prompt templates
    prompt_templates = {
        "analysis": load_prompt_template(f"{self.dir_path}/prompts/01_analysis.txt"),
        "param_mapping": load_prompt_template(f"{self.dir_path}/prompts/03_param_mapping.txt"),
        "param_mapping_check": load_prompt_template(f"{self.dir_path}/prompts/03_param_mapping_check.txt"),
        "hook_fn": load_prompt_template(f"{self.dir_path}/prompts/04_hook_fn_prompt_chain.txt"),
        "pitfalls": load_prompt_template(f"{self.dir_path}/prompts/04_pitfalls.txt"),
        "shape_mapping": load_prompt_template(f"{self.dir_path}/prompts/05_shape_mapping.txt"),
        "shape_mapping_check": load_prompt_template(f"{self.dir_path}/prompts/05_shape_mapping_check.txt"),
    }

    # ======== Analyze Model Structures ========
    print("Step 1: Analyzing model structures...")
    prompt1 = prompt_templates["analysis"].format(
        target_model=self.target_model,
        maxtext_params_json=json.dumps(maxtext_params, indent=2),
        hf_params_json=json.dumps(hf_params, indent=2),
        dsl=None,
        pitfalls=load_prompt_template(f"{self.dir_path}/prompts/04_pitfalls.txt"),
    )
    analysis = self.generate_text(prompt1)

    # ======== Generate & Verify Parameter Mapping Function ========
    print("Step 2: Generating and verifying parameter mapping function...")
    param_mapping_code = None
    feedback = ""
    for attempt in range(1, max_retries + 1):
      print(f"  Attempt {attempt}...")
      prompt3 = prompt_templates["param_mapping"].format(
          target_model=self.target_model,
          analysis=analysis,
          pitfalls=None,
          maxtext_params_json=json.dumps(maxtext_params, indent=2),
          hf_params_json=json.dumps(hf_params, indent=2),
          feedback=feedback,
          request_options={"timeout": 300},
      )
      candidate = self.generate_text(prompt3)

      prompt3_1 = prompt_templates["param_mapping_check"].format(
          maxtext_params_json=json.dumps(maxtext_params, indent=2),
          hf_params_json=json.dumps(hf_params, indent=2),
          code=candidate,
          analysis=analysis,
      )
      feedback = self.generate_text(prompt3_1)

      print(f"  Validator Call {attempt}...")
      print(feedback)

      if "passed" in feedback:
        param_mapping_code = candidate
        print("  Passed Validator...")
        break
      else:
        if attempt == max_retries:
          raise RuntimeError("Max attempts tried")

    output_dir = f"{self.dir_path}/outputs"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "param_mapping.py")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        f.write(param_mapping_code)
      print(f"Parameter mapping successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving hook functions file: {e}")

    # ======== Generate HF Shape Function ========
    candidate = None
    feedback = ""
    shape_mapping_code = None
    for attempt in range(1, max_retries + 1):
      print("Step 3: Generating HF weights shape mapping function...")
      prompt2 = prompt_templates["shape_mapping"].format(
          target_model=self.target_model,
          hf_params_json=json.dumps(hf_params, indent=2),
          analysis=analysis,
          feedback=feedback,
          pitfalls=None,
      )
      candidate = self.generate_text(prompt2)

      prompt2_1 = prompt_templates["shape_mapping_check"].format(
          hf_params_json=json.dumps(hf_params, indent=2),
          code=candidate,
      )
      feedback = self.generate_text(prompt2_1)

      print(f"  Validator Call {attempt}...")
      print(feedback)

      if "yes" in feedback.lower():
        shape_mapping_code = candidate
        print("  Passed Validator...")
        break
      else:
        if attempt == max_retries:
          raise RuntimeError("Max attempts tried")

    file_path = os.path.join(output_dir, "hf_shape.py")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        f.write(shape_mapping_code)
      print(f"hf_shape successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving hook functions file: {e}")

    # ======== Generate Hook Functions ========
    print("Step 4: Generating layerwise transformation hook functions...")
    prompt4 = prompt_templates["hook_fn"].format(
        target_model=self.target_model,
        analysis=analysis,
        param_mapping=param_mapping_code,
    )
    hook_fn_code = self.generate_text(prompt4)

    file_path = os.path.join(output_dir, "hook_fn.py")
    try:
      with open(file_path, "wt", encoding="utf-8") as f:
        f.write(hook_fn_code)
      print(f"Hook functions successfully saved to {file_path}")
    except IOError as e:
      print(f"Error saving hook functions file: {e}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A prompt-chain Agent.")

  parser.add_argument("--target_model", type=str, required=True, help='The name of the target model (e.g., "GEMMA3").')
  parser.add_argument("--dir_path", type=str, required=True, help="The file path to the ckpt conversion agent directory.")
  parser.add_argument("--api_key", type=str, required=True, help="Gemini API key.")
  args = parser.parse_args()
  agent = prompt_chaining_agent(api_key=args.api_key, target_model=args.target_model, dir_path=args.dir_path)
  agent.run_chain()
