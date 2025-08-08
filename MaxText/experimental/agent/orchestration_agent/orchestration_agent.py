"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
This file orchestrates the analysis of Python file dependencies within a GitHub repository.
It first determines the hierarchical order of files based on their import relationships,
then further breaks down each file into its constituent components (functions, classes,
variables, and imports), identifying internal dependencies within each file.

The output provides a comprehensive view of the codebase structure,
first by file-level dependencies, and then by component-level dependencies within each file.

Example Invocation:

python orchestration_agent.py \
  --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
  --entry-file-path "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py" \
  --no-exclude-conditional-imports
"""
import argparse
import json
import logging
import os.path

from .utils import check_github_file_exists
from .get_files_in_hierarchical_order import get_dependency_sorted_files
from .split_python_file import get_modules_in_order

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def arg_parser():
  """Parses command-line arguments for the dependency analysis script.

  Returns:
      argparse.Namespace: An object containing the parsed command-line arguments.
  """
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


def save_file_with_components(sorted_files, dependencies, base_path, entry_file_path, out_file="FilesWithComponents.txt"):
  """Saves the sorted files and their internal components to a file.

  This function iterates through a dependency-sorted list of files, and for
  each file, it analyzes its internal components (functions, classes, etc.)
  and their dependencies. The results are written to an output file.

  Args:
      sorted_files (list): A list of file paths sorted by dependency.
      dependencies (dict): A dictionary where keys are file paths and values are
        lists of their file-level dependencies.
      base_path (str): The base URL of the project.
      entry_file_path (str): The full GitHub URL for the entry Python file.
      out_file (str): The name of the output file.
  """
  with open(out_file, "w", encoding="utf-8") as f:
    f.write(f"BasePath {base_path}\n")
    f.write(f"Entry File {entry_file_path}\n")

    for file_path in sorted_files:
      result = get_modules_in_order(base_path + file_path)
      component_dependencies = result.get("component_dependencies", {})
      all_components = result.get("sorted_modules", {}).keys()

      standalone_components = [comp for comp in all_components if comp not in component_dependencies]
      dependent_components = {
          comp: component_dependencies[comp] for comp in all_components if comp in component_dependencies
      }

      f.write(f"\nComponents for {file_path}\n")
      file_deps = dependencies.get(file_path)
      if file_deps:
        f.write(f"There File Dependencies {file_deps}\n")

      f.write(f"StandAlone Modules: {json.dumps(standalone_components)}\n")
      f.write(f"Dependent Modules\n {json.dumps(dependent_components, indent=4)}\n")

  logger.info("Check Results at %s", out_file)


def main():
  """Main function to orchestrate the dependency analysis.

  It parses arguments, gets the dependency-sorted list of files,
  analyzes component-level dependencies within each file, and saves the
  results. It also handles caching of file-level dependency analysis to
  avoid re-computing for the same entry file.
  """
  args = arg_parser()
  BASE_PATH = args.base_path
  ENTRY_FILE_PATH = args.entry_file_path
  EXCLUDE_CONDITIONAL_IMPORTS = args.exclude_conditional_imports
  if not check_github_file_exists(ENTRY_FILE_PATH)[0]:
    logger.error("Error: Entry file not found at '%s'", ENTRY_FILE_PATH)
  else:
    # Use rstrip to handle base paths that may or may not have a trailing slash
    relative_entry = ENTRY_FILE_PATH.replace(BASE_PATH.rstrip(os.path.sep), "")
    mode = "Excluding Conditional Imports" if EXCLUDE_CONDITIONAL_IMPORTS else "Including All Imports"
    logger.info("Analyzing dependencies for: %s", relative_entry)
    logger.info("Mode: %s", mode)
    logger.info("-" * 40)
    all_files_info = "all_files.json"
    modules_found = False
    if os.path.exists(all_files_info):
      with open(all_files_info, "rt", encoding="utf8") as f:
        data = json.load(f)
        sorted_files, dependencies = data["sorted_files"], data["dependencies"]
        if data["entery_file"] == ENTRY_FILE_PATH:
          modules_found = True
          logger.info("---> Reading Files order from all Files.json You can delete this if have some update in code. ")
    if not modules_found:
      sorted_files, dependencies = get_dependency_sorted_files(
          ENTRY_FILE_PATH, BASE_PATH, EXCLUDE_CONDITIONAL_IMPORTS, returnDependencies=True
      )
      with open(all_files_info, "wt", encoding="utf8") as f:
        json.dump({"entry_file": ENTRY_FILE_PATH, "sorted_files": sorted_files, "dependencies": dependencies}, f)
    save_file_with_components(sorted_files, dependencies, args.base_path, args.entry_file_path)

    if sorted_files:
      logger.info("\n--- Dependency Sorted Files ---")
      for file_path in sorted_files:
        logger.info(file_path)
    else:
      logger.info("\nCould not generate sorted file list due to errors.")


if __name__ == "__main__":
  main()
