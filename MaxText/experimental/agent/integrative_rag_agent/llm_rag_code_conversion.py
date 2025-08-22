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

"""
This file provides tools to convert Hugging Face Transformers modules into MaxText-style
JAX modules using an LLM and RAG context.

High-level flow:
  1) Read source modules from the Transformers repository (local or GitHub).
  2) Use prompts and prior examples to guide an LLM to generate JAX code.
  3) Persist generated modules into the MaxText experimental agent directory.

Command-line entry points allow customizing which module to process and how many
MaxText code blocks to include as guidance examples.

Example Invocation:

python llm_rag_code_coversion.py --module-name "Qwen3ForCausalLM"
"""
import json
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from integrative_rag_agent import system_setup
from code_generation_agent.llm_agent import GeminiAgent
from integrative_rag_agent.prompts_integrative_rag import CODE_CONVERSION, MODULE_NAMING_PROMPT, CODE_DESCRIPTION
from orchestration_agent.SplitPythonFile import get_modules_from_file
from orchestration_agent.Utils import parse_python_code
from code_evaluation_agent.utils import get_last_defined_module
from integrative_rag_agent.utils import read_code_blocks
from integrative_rag_agent.config import maxtext_block_description
from integrative_rag_agent.llm_rag_embedding_generation import get_code_description_with_gemini
from integrative_rag_agent.config import files_order_file_format, maxtext_code_block, processed_module_file_format, new_module_file_format

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def arg_parser():
  """Create and return the CLI argument parser for code conversion.

  Returns:
      argparse.Namespace: Parsed arguments containing:
          - number_of_maxtext_blocks (int)
          - module_name (str)
          - destination_base_directory (str)
          - destination_source_url (str)
  """
  parser = argparse.ArgumentParser(description="LLM code conversion utility.")
  parser.add_argument("--number-of-maxtext-blocks", type=int, default=5, help="Number of maxtext blocks to process.")
  parser.add_argument("--module-name", type=str, default="Qwen3ForCausalLM", help="Name of the module to process.")
  parser.add_argument(
      "--destination-base-directory",
      type=str,
      default="maxtext/",
      help="Base directory for output files. Should point to the root of the MaxText repo.",
  )
  parser.add_argument(
      "--destination-source-url",
      type=str,
      default="https://github.com/huggingface/transformers/blob/6ce8f0537537455806ab7bfd39b59ad37803ead9/src/",
      help="Base directory for source files. Should point to the root of the Transformer repo repo.",
  )
  args = parser.parse_args()
  return args


args = arg_parser()
number_of_maxtext_blocks = args.number_of_maxtext_blocks
module_name = args.module_name


destination_base_directory = args.destination_base_directory
destination_source_url = args.destination_source_url

module_list_path = files_order_file_format.format(module_name=module_name)
destination_relative_directory = f"MaxText/experimental/agent/{module_name}/"
destination_directory = destination_base_directory + destination_relative_directory


def get_exisiting_jax_modules():
  """Load known MaxText JAX module descriptions keyed by module path.

  Reads `maxtext_block_description` JSON and returns a dict where keys are
  Python-qualified module names (e.g., "path.to.file#Object" normalized) and
  values are textual analyses used as LLM guidance.

  Returns:
      dict[str, str]: Mapping of module key to analysis/description.
  """
  with open(maxtext_block_description) as f:
    module_list = json.load(f)
  module_list = {m["block_name"].replace("/", ".").replace("py#", ""): m["analysis"] for m in module_list}
  return module_list


def find_existing_agent_files():
  """Find existing Python files in the experimental/agent directory and its subdirectories."""
  agent_dir = destination_directory
  existing_files = []

  if os.path.exists(agent_dir):
    for root, dirs, files in os.walk(agent_dir):
      for file in files:
        if file.endswith(".py"):
          # Get relative path from agent directory
          rel_path = os.path.relpath(os.path.join(root, file), agent_dir)
          # Remove .py extension for display
          module_name = rel_path.replace(".py", "").replace("/", "_")
          existing_files.append({"file_path": rel_path, "module_name": module_name, "full_path": os.path.join(root, file)})

  return existing_files


def generate_file_name_prompt(module_description, existing_files):
  """Generate a prompt for the LLM to find appropriate file names for new modules."""

  existing_files_info = ""
  if existing_files:
    existing_files_info = "Existing Python files in the experimental/agent directory:\n"
    for file_info in existing_files:
      existing_files_info += f"- {file_info['file_path']} (module: {file_info['module_name']})\n"

  # Use the prompt template from prompts_integrative_rag.py
  prompt = (
      MODULE_NAMING_PROMPT.replace("{module_base_path}", destination_directory)
      .replace("{existing_files_info}", existing_files_info)
      .replace("{module_description}", module_description)
  )

  return prompt


def find_appropriate_file_name(module_description):
  """Find an appropriate file name for a new generated module."""
  existing_files = find_existing_agent_files()
  prompt = generate_file_name_prompt(module_description, existing_files)

  # Get LLM response for module naming
  llm_agent = GeminiAgent(system_instruction=MODULE_NAMING_PROMPT)
  response = llm_agent(prompt)
  file_name = response.text.strip()

  # Clean up the response to get just the module name
  if "\n" in file_name:
    file_name = file_name.split("\n")[0].strip()

  # Remove any quotes or extra formatting
  file_name = file_name.strip("\"'`")

  return file_name, existing_files


def convert_given_file(module, jax_modules):
  """Convert a single Transformer component into a JAX module using the LLM.

  Builds a prompt from the target component's code, the full source file,
  example MaxText blocks, and previously known/generated JAX modules. Invokes
  the LLM to produce code, writes it to the destination agent directory, and
  returns a description for the generated module.

  Args:
      module (dict): Component metadata with keys like "filepath",
          "comp_name", and optional "JaxDependencies".
      jax_modules (dict): Existing JAX module descriptions to provide context
          to the LLM.

  Returns:
      dict | None: Mapping of fully-qualified package name to generated
      description, or None if generation did not produce a detectable module.
  """
  maxtext_blocks_code = read_code_blocks(maxtext_code_block, number_of_maxtext_blocks)
  module_code, file_code = get_modules_from_file(destination_source_url + module["filepath"], module=module["comp_name"])
  prompt = (
      CODE_CONVERSION.replace("<CODE_BLOCK>", module_code)
      .replace("<FULL_FILE_CODE>", file_code)
      .replace("<JAX_MODULES_DICT>", json.dumps(jax_modules))
  )
  prompt = prompt.replace("<MAXTEXT_EXAMPLE_CODE>", maxtext_blocks_code)
  maxtext_dependency = json.dumps(module["JaxDependencies"]) if len(module["JaxDependencies"]) > 0 else ""
  prompt = prompt.replace("<MAXTEXT_MATCHED_DEPENDENCIES>", maxtext_dependency)
  llm_agent = GeminiAgent(system_instruction=CODE_CONVERSION)
  resp = llm_agent(prompt)
  module_code = parse_python_code(resp.text)
  file_name, existing_files = find_appropriate_file_name(module_code)
  # Save the generated code to the appropriate file
  dest_path = os.path.join(destination_directory, f"{file_name}.py")
  os.makedirs(os.path.dirname(dest_path), exist_ok=True)
  with open(dest_path, "a") as f:
    f.write(module_code)
  logger.info(f"Generated JAX code saved to: {dest_path}")
  with open(dest_path) as f:
    full_context = f.read()
  last_module = get_last_defined_module(module_code)
  if last_module is None:
    return
  packagename = dest_path.removeprefix(destination_base_directory).replace("/", ".").replace(".py", ".") + last_module
  logger.info(f"New Module Pakage Name {packagename}")
  return {packagename: get_code_description_with_gemini(module_code, full_context, prompt=CODE_DESCRIPTION)}


def convert_all_modules():
  """Convert all components listed for the selected module into JAX code.

  Reads the ordered component list for `module_name`, loads/merges existing
  and newly generated JAX module descriptions, and iteratively converts each
  remaining component by calling `convert_given_file`. Results and progress
  are persisted to JSON files configured via `config`.
  """
  with open(module_list_path) as f:
    modules_List = json.load(f)
  jax_modules = get_exisiting_jax_modules()
  new_module_file = new_module_file_format.format(module_Name=module_name)
  processed_module_file = processed_module_file_format.format(module_Name=module_name)
  os.makedirs(Path(new_module_file).parent, exist_ok=True)
  if os.path.exists(new_module_file):
    with open(new_module_file) as f:
      new_modules = json.load(f)
  else:
    new_modules = {}
  if os.path.exists(processed_module_file):
    with open(processed_module_file) as f:
      processed_module = json.load(f)
  else:
    processed_module = []
  for module in modules_List:
    if module["comp_id"] in processed_module:
      continue
    logger.info(f"Processing {module['comp_id']}....")
    generated_module = convert_given_file(module, {**jax_modules, **new_modules})
    if generated_module is not None:
      new_modules.update(generated_module)
    processed_module.append(module["comp_id"])
    with open(new_module_file, "w") as f:
      json.dump(new_modules, f, indent=4)
    with open(processed_module_file, "w") as f:
      json.dump(processed_module, f, indent=4)

  logger.info(f"Conversion finish Check at {new_module_file} and {processed_module_file}")


if __name__ == "__main__":
  system_setup.setup_directories()
  convert_all_modules()
