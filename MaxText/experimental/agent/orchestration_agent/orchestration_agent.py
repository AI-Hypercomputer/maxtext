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
This file orchestrates the analysis of Python file dependencies within a GitHub repository.
It first determines the hierarchical order of files based on their import relationships,
then further breaks down each file into its constituent components (functions, classes,
variables, and imports), identifying internal dependencies within each file.

The output provides a comprehensive view of the codebase structure,
first by file-level dependencies, and then by component-level dependencies within each file.

Example Invocation:

python orchestrationAgent.py \
  --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
  --entry-file-path "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py" \
  --no-exclude-conditional-imports
"""
import json
import os.path
import argparse, logging

from MaxText.experimental.agent.orchestration_agent.orchestration_agent_utils import check_github_file_exists
from MaxText.experimental.agent.orchestration_agent.get_files_in_hierarchical_order import get_dependency_sorted_files
from MaxText.experimental.agent.orchestration_agent.split_python_file import get_modules_in_order

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ArgParser():
  parser = argparse.ArgumentParser(description="Dependency sorter for Python files on GitHub.")
  parser.add_argument(
      "--base-path",
      type=str,
      default="https://github.com/huggingface/transformers/blob/main/src/",
      help="Root URL of the source directory on GitHub (default: %(default)s)",
  )
  parser.add_argument(
      "--entry-file-path",
      type=str,
      default="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py",
      help="Full GitHub URL for the entry Python file to analyze (default: %(default)s)",
  )
  parser.add_argument(
      "--exclude-conditional-imports",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Exclude imports inside functions/classes (use --no-exclude-conditional-imports to disable)",
  )
  args = parser.parse_args()
  if not args.entry_file_path.startswith(args.base_path):
    raise ValueError("Entry file path must be inside the base path.")
  return args


def saveFilewithComponents(sorted_files, dependencies, basepath, outFile="FilesWithComponents.txt"):
  """
  Saves the sorted files and their internal components to a file.

  Args:
      sorted_files (list): A list of file paths sorted by dependency.
      dependencies (dict): A dictionary where keys are file paths and values are lists of their dependencies.
      basepath (str): The base URL of the project.
      outFile (str): The name of the output file.
  """
  standalone_modules = [mod for mod in sorted_files if mod not in dependencies or len(dependencies[mod]) == 0]
  dependent_sorted_modules = {
      mod: dependencies[mod] for mod in sorted_files if mod in dependencies and len(dependencies[mod]) > 0
  }
  with open(outFile, "w") as f:
    f.write(f"BasePath {args.base_path}\n")
    f.write(f"Entry File {args.entry_file_path}\n")
  # f.write(f"StandAlone Files:\n {json.dumps(standalone_modules, indent=4)}\n")
  # f.write(f"Dependent Files\n {json.dumps(dependent_sorted_modules, indent=4)}\n")
  for m in standalone_modules:
    result = get_modules_in_order(basepath + m)
    standalone_modules = [mod for mod in result["sorted_modules"].keys() if mod not in result["component_dependencies"]]
    dependent_sorted_files = {
        mod: result["component_dependencies"][mod]
        for mod in result["sorted_modules"].keys()
        if mod in result["component_dependencies"]
    }
    with open(outFile, "a") as f:
      f.write(f"\nComponents for {m}\n")
      f.write(f"StandAlone Modules: {json.dumps(standalone_modules)}\n")
      f.write(f"Dependent Modules\n {json.dumps(dependent_sorted_files, indent=4)}\n")
  for m, dep in dependent_sorted_modules.items():
    result = get_modules_in_order(basepath + m)
    standalone_modules = [mod for mod in result["sorted_modules"].keys() if mod not in result["component_dependencies"]]
    dependent_sorted_modules = {
        mod: result["component_dependencies"][mod]
        for mod in result["sorted_modules"].keys()
        if mod in result["component_dependencies"]
    }
    with open(outFile, "a") as f:
      f.write(f"\nComponents for {m}\n")
      f.write(f"There File Dependencies {dep}\n")
      f.write(f"StandAlone Modules: {json.dumps(standalone_modules)}\n")
      f.write(f"Dependent Modules\n {json.dumps(dependent_sorted_modules, indent=4)}\n")
  logger.info(f"Check Results at {outFile}")


if __name__ == "__main__":
  args = ArgParser()
  BASE_PATH = args.base_path
  ENTRY_FILE_PATH = args.entry_file_path
  EXCLUDE_CONDITIONAL_IMPORTS = args.exclude_conditional_imports
  if not check_github_file_exists(ENTRY_FILE_PATH)[0]:
    logger.error(f"Error: Entry file not found at '{ENTRY_FILE_PATH}'")
  else:
    # Use rstrip to handle base paths that may or may not have a trailing slash
    relative_entry = ENTRY_FILE_PATH.replace(BASE_PATH.rstrip("/"), "")
    mode = "Excluding Conditional Imports" if EXCLUDE_CONDITIONAL_IMPORTS else "Including All Imports"
    logger.info(f"Analyzing dependencies for: {relative_entry}")
    logger.info(f"Mode: {mode}")
    logger.info("-" * 40)
    all_files_info = "all_files.json"
    modules_found = False
    if os.path.exists(all_files_info):
      with open(all_files_info) as f:
        data = json.load(f)
        sorted_files, dependencies = data["sorted_files"], data["dependencies"]
        if data["entery_file"] == ENTRY_FILE_PATH:
          modules_found = True
          logger.info("---> Reading Files order from all Files.json You can delete this if have some update in code. ")
    if not modules_found:
      sorted_files, dependencies = get_dependency_sorted_files(
          ENTRY_FILE_PATH, BASE_PATH, EXCLUDE_CONDITIONAL_IMPORTS, returnDependencies=True
      )
      with open(all_files_info, "w") as f:
        json.dump({"entery_file": ENTRY_FILE_PATH, "sorted_files": sorted_files, "dependencies": dependencies}, f)
    saveFilewithComponents(sorted_files, dependencies, args.base_path)

    if sorted_files:
      logger.info("\n--- Dependency Sorted Files ---")
      for file_path in sorted_files:
        logger.info(file_path)
    else:
      logger.info("\nCould not generate sorted file list due to errors.")
