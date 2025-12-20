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
This file provides functionality to analyze Python file dependencies within a GitHub repository
and generate a topologically sorted list of files based on their import relationships.
It handles both absolute and relative imports, and can optionally exclude conditional imports.

Example Invocations:

1. Analyze a specific entry file in a repository, excluding conditional imports (default):
   python get_files_in_hierarchical_order.py \
     --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
--entry-file-path https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

2. Analyze a specific entry file, including all imports (even conditional ones):
   python get_files_in_hierarchical_order.py \
     --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
     --no-exclude-conditional-imports \
--entry-file-path https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import argparse
import ast
import json
import logging
import os.path
from collections import deque

from MaxText.experimental.agent.orchestration_agent.utils import find_cycle, check_github_file_exists, get_github_file_content, resolve_import_path

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_file_dependencies(file_path_url, base_path_url, exclude_conditional_imports=True):
  """
  Finds all direct Python file dependencies for a given file.

  Args:
    file_path_url (str): The full GitHub URL of the Python file to analyze.
    base_path_url (str): The base URL of the GitHub repository's source directory.
    exclude_conditional_imports (bool): If True, imports inside functions,
      classes, or `if TYPE_CHECKING:` blocks are ignored.

  Returns:
    set: A set of full GitHub URLs of the dependent Python files.
  """
  dependencies = set()
  flag, content = get_github_file_content(file_path_url)
  if not flag:
    logger.warning("Warning: Could not read or parse %s. Error: %s", file_path_url, content)
    return dependencies

  try:
    tree = ast.parse(content, filename=file_path_url)
  except (SyntaxError, ValueError) as e:
    logger.warning("Warning: Could not parse %s. Error: %s", file_path_url, e)
    return dependencies

  parent_map = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

  def is_import_conditional(node):
    current = node
    while current in parent_map:
      parent = parent_map[current]
      if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True
      if isinstance(parent, ast.If):
        test = parent.test
        if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute)
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
            and test.attr == "TYPE_CHECKING"
        ):
          return True
      current = parent
    return False

  for node in ast.walk(tree):
    is_import_node = isinstance(node, (ast.Import, ast.ImportFrom))
    if not is_import_node:
      continue

    if exclude_conditional_imports and is_import_conditional(node):
      continue

    if isinstance(node, ast.Import):
      for alias in node.names:
        resolved_path = resolve_import_path(file_path_url, alias.name, 0, base_path_url)
        if resolved_path and resolved_path.startswith(base_path_url):
          dependencies.add(resolved_path)
    elif isinstance(node, ast.ImportFrom):
      module_name = node.module or ""
      for alias in node.names:
        resolved_path = resolve_import_path(file_path_url, module_name, node.level, base_path_url, alias.name)
        if resolved_path and resolved_path.startswith(base_path_url):
          dependencies.add(resolved_path)
  return dependencies


def get_dependency_sorted_files(entry_file_path, base_path, exclude_conditional_imports=True, returnDependencies=False):
  """
  Analyzes a given entry Python file within a GitHub repository and returns a
  topologically sorted list of all dependent Python files.

  Args:
    entry_file_path (str): The full GitHub URL of the entry Python file.
    base_path (str): The base URL of the GitHub repository's source directory.
    exclude_conditional_imports (bool): If True, imports inside functions,
      classes, or `if TYPE_CHECKING:` blocks are ignored for dependency
      analysis.
    returnDependencies (bool): If True, returns a tuple of (sorted_files,
      dependency_graph), otherwise just sorted_files.

  Returns:
    list: A list of file paths (relative to base_path) in topological order.
      Returns an empty list if a circular dependency is detected.
    dict (optional): A dictionary representing the dependency graph if
      returnDependencies is True.
  """
  dependency_graph = {}
  reverse_graph = {}
  processed_files = set()
  # The queue starts with the entry file. The script will iteratively process
  # this queue, adding newly discovered dependencies as it goes. This ensures
  # that it traverses the entire nested dependency graph.
  files_to_process = deque([entry_file_path])
  i = 0
  # This while loop performs a breadth-first search (BFS) to discover all
  # nested dependencies. It stops only when the queue is empty, which means
  # all reachable files have been analyzed.
  while files_to_process:
    current_file = files_to_process.popleft()
    i += 1
    logger.info("Processing %d %s", i, current_file.replace(base_path, ""))
    # Skip files that have already been processed to avoid redundant work and cycles.
    if current_file in processed_files:
      continue
    processed_files.add(current_file)

    dependency_graph.setdefault(current_file, [])
    reverse_graph.setdefault(current_file, [])

    dependencies = find_file_dependencies(current_file, base_path, exclude_conditional_imports)
    dependencies_name = {k.replace(base_path, "") for k in dependencies}
    logger.info("File %s Have %s Dependencies", current_file.replace(base_path, ""), dependencies_name)
    dependency_graph[current_file] = list(dependencies)

    for dep in dependencies:
      reverse_graph.setdefault(dep, []).append(current_file)
      # If a dependency is new to us, add it to the queue to be
      # processed in a future iteration.
      if dep not in processed_files and dep not in files_to_process:
        files_to_process.append(dep)

  logger.info("Parsing Finished")

  all_files_in_graph = set(dependency_graph.keys())
  for deps in dependency_graph.values():
    all_files_in_graph.update(deps)

  in_degree = {file: 0 for file in all_files_in_graph}
  for _, dependencies in reverse_graph.items():
    for dep in dependencies:
      if dep in in_degree:
        in_degree[dep] += 1

  zero_in_degree_queue = deque([file for file, degree in in_degree.items() if degree == 0])
  sorted_list = []

  while zero_in_degree_queue:
    current_file = zero_in_degree_queue.popleft()
    sorted_list.append(current_file)
    for dependent_file in reverse_graph.get(current_file, []):
      in_degree[dependent_file] -= 1
      if in_degree[dependent_file] == 0:
        zero_in_degree_queue.append(dependent_file)

  rel_dependency_graph = {
      k.replace(base_path, ""): [f.replace(base_path, "") for f in v] for k, v in dependency_graph.items()
  }

  if len(sorted_list) == len(all_files_in_graph):
    sorted_relative = [path.replace(base_path, "") for path in sorted_list]
    if returnDependencies:
      return sorted_relative, rel_dependency_graph
    else:
      return sorted_relative
  else:
    cycle = find_cycle(dependency_graph)
    cycle_msg = " -> ".join([c.replace(base_path, "") for c in cycle]) if cycle else "unknown"
    logger.error("\nError: A circular dependency was detected. Cycle: %s", cycle_msg)
    if returnDependencies:
      return [], rel_dependency_graph
    else:
      return []


def parse_args():
  """
  Parses command-line arguments for file or folder processing.

  Returns:
    argparse.Namespace: The parsed command-line arguments.
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


def save_results_in_file(sorted_files, dependencies, args, outFile="FileOrder.txt"):
  """
  Saves the sorted files and their dependencies to a file.

  Args:
      sorted_files (list): A list of file paths sorted by dependency.
      dependencies (dict): A dictionary where keys are file paths and values are lists of their dependencies.
      args (argparse.Namespace): The command-line arguments.
      outFile (str): The name of the output file.
  """
  with open("all_files.json", "wt", encoding="utf-8") as f:
    json.dump({"sorted_files": sorted_files, "dependencies": dependencies}, f)
  standalone_module = [mod for mod in sorted_files if mod not in dependencies or len(dependencies[mod]) == 0]
  dependent_sorted_modules = {
      mod: dependencies[mod] for mod in sorted_files if mod in dependencies and len(dependencies[mod]) > 0
  }
  with open(outFile, "wt", encoding="utf-8") as f:
    f.write(f"BasePath {args.base_path}\n")
    f.write(f"Entry File {args.entry_file_path}\n")
    f.write(f"Standalone Files:\n {json.dumps(standalone_module,indent=4)}\n")
    f.write(f"Dependent Files\n {json.dumps(dependent_sorted_modules,indent=4)}\n")


def main():
  args = parse_args()
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

    sorted_files, dependencies = get_dependency_sorted_files(
        ENTRY_FILE_PATH, BASE_PATH, EXCLUDE_CONDITIONAL_IMPORTS, returnDependencies=True
    )

    save_results_in_file(sorted_files, dependencies, args)

    if sorted_files:
      logger.info("\n--- Dependency Sorted Files ---")
      for file_path in sorted_files:
        logger.info(file_path)
    else:
      logger.info("\nCould not generate sorted file list due to errors.")


if __name__ == "__main__":
  main()
