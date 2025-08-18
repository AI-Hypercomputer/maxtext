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
   python GetFilesInHierarchicalOrder.py \
     --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
     --entry-file-path "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"

2. Analyze a specific entry file, including all imports (even conditional ones):
   python GetFilesInHierarchicalOrder.py \
     --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
     --entry-file-path "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py" \
     --no-exclude-conditional-imports
"""
import ast, json
from collections import deque
import argparse, logging

from MaxText.experimental.agent.orchestration_agent.orchestration_agent_utils import find_cycle, check_github_file_exists, get_github_file_content, url_join

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def have_module(target_name, file_url):
  """
  Checks if a given module (function, class, or variable) exists in a Python file.

  Args:
      target_name (str): The name of the module to search for.
      file_url (str): The URL of the Python file to check.

  Returns:
      bool: True if the module is found, False otherwise.
      tuple: ("ImportFrom", full_module) if the target_name is an alias from an import statement.
  """
  flag, content = get_github_file_content(file_url)
  if not flag:
    logger.warning(f"Warning: Could not read or parse {file_url}. Error: {content}")
    return False  # Fail if content cannot be retrieved

  try:
    tree = ast.parse(content, filename=file_url)
  except (SyntaxError, ValueError):
    logger.warning(f"Warning: Could not parse {file_url}")
    return False

  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == target_name:
      return True
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name) and target.id == target_name:
          return True
    if isinstance(node, ast.ImportFrom):
      for alias in node.names:
        if alias.asname == target_name or alias.name == target_name:
          module = node.module or ""
          level = node.level
          full_module = "." * level + module if level > 0 else module
          return ("ImportFrom", full_module)
  return False


def resolve_complex_import(module_path_base_url, importPackage, base_url, current_dir_url, Try=0, Message=""):
  """
  Resolves a complex import statement, looking for the imported package/module
  within a directory structure. This handles cases where 'importPackage' might
  refer to a file or a directory (package) with an __init__.py.

  Args:
      module_path_base_url (str): The base URL for the module path (e.g., 'https://github.com/.../transformers/models/llama').
      importPackage (str): The specific name being imported (e.g., 'modeling_llama', 'configuration_llama').
      base_url (str): The base URL of the repository.
      current_dir_url (str): The URL of the directory containing the original import statement.
      Try (int): Counter for recursion depth.
      Message (str): Accumulates error messages for recursion depth.

  Returns:
      str: The resolved full GitHub URL of the imported file, or None if not found.

  Example:
      If `module_path_base_url` is "https://github.com/org/repo/blob/main/src/transformers/models/llama",
      `importPackage` is "modeling_llama", `base_url` is "https://github.com/org/repo/blob/main/src/",
      and `current_dir_url` is "https://github.com/org/repo/blob/main/src/transformers/models/llama",
      this function would first check for "https://github.com/org/repo/blob/main/src/transformers/models/llama.py".
      If not found, it would then check for "https://github.com/org/repo/blob/main/src/transformers/models/llama/__init__.py".
      If `__init__.py` exists and contains `from . import modeling_llama`, it would then check for
      "https://github.com/org/repo/blob/main/src/transformers/models/llama/modeling_llama.py".
  """

  if Try == 0:
    Message = f"There is an issue with import {importPackage} in {module_path_base_url} current dir is {current_dir_url}"
  if Try > 4:  # Increased recursion limit slightly for network latency
    logger.error(f"Error: Exceeded recursion depth. {Message}")
    return None
  # Check for a direct .py file containing the definition
  potential_py_url = f"{module_path_base_url}.py"
  if check_github_file_exists(potential_py_url)[0] and have_module(importPackage, potential_py_url) == True:
    return potential_py_url

  # Check for a package (directory with __init__.py)
  potential_pkg_init_url = url_join(module_path_base_url, "__init__.py")
  if check_github_file_exists(potential_pkg_init_url)[0] and potential_pkg_init_url.startswith(base_url):
    has_module = have_module(importPackage, potential_pkg_init_url)
    if has_module:
      if has_module == True:
        return potential_pkg_init_url
      elif has_module[0] == "ImportFrom":
        # The name is re-exported from another module
        re_export_module = has_module[1]
        if re_export_module in (".", ""):
          # from . import X -> look for X.py in the same directory
          potential_file_in_pkg_url = url_join(module_path_base_url, f"{importPackage}.py")
          if check_github_file_exists(potential_file_in_pkg_url)[0]:
            return potential_file_in_pkg_url
        else:
          # from .foo import X -> recurse into foo
          new_module_path_base_url = url_join(module_path_base_url, re_export_module)
          return resolve_complex_import(
              new_module_path_base_url, importPackage, base_url, new_module_path_base_url, Try + 1, Message
          )
    else:
      # The package exists, but the import is not in __init__. It could be a submodule.
      potential_file_in_pkg_url = url_join(module_path_base_url, f"{importPackage}.py")
      if check_github_file_exists(potential_file_in_pkg_url)[0] and potential_file_in_pkg_url.startswith(base_url):
        return potential_file_in_pkg_url
  return None


def resolve_import_path(importer_url, module_name, level, base_url, importPackage=None):
  """
  Resolves an import statement to a full GitHub URL.

  Args:
      importer_url (str): The URL of the file containing the import statement.
      module_name (str): The name of the module being imported (e.g., 'os', 'transformers.models.llama').
      level (int): The level of the import (0 for absolute, 1+ for relative).
      base_url (str): The base URL of the repository (e.g., 'https://github.com/huggingface/transformers/blob/main/src/').
      importPackage (str, optional): The specific package or module being imported from a 'from ... import ...' statement.

  Returns:
      str: The resolved full GitHub URL of the imported file, or None if not found.
  """
  current_dir_url = importer_url[: importer_url.rfind("/")]
  if level > 0:  # Relative import
    path_parts = [current_dir_url] + [".."] * (level - 1)
    if module_name:
      path_parts.extend(module_name.split("."))
    module_path_base_url = url_join(*path_parts)
  else:  # Absolute import
    module_path_base_url = url_join(base_url, *module_name.split("."))
  # Check for a direct .py file
  potential_url_py = f"{module_path_base_url}.py"
  if check_github_file_exists(potential_url_py)[0]:
    return potential_url_py

  # If it's not a direct file, it might be a complex package import
  return resolve_complex_import(module_path_base_url, importPackage, base_url, current_dir_url)


def find_file_dependencies(file_path_url, base_path_url, exclude_conditional_imports=True):
  """
  Finds all direct Python file dependencies for a given file.

  Args:
      file_path_url (str): The full GitHub URL of the Python file to analyze.
      base_path_url (str): The base URL of the GitHub repository's source directory.
      exclude_conditional_imports (bool): If True, imports inside functions,
                                          classes, or `if TYPE_CHECKING:` blocks
                                          are ignored.

  Returns:
      set: A set of full GitHub URLs of the dependent Python files.
  """
  dependencies = set()
  flag, content = get_github_file_content(file_path_url)
  if not flag:
    logger.warning(f"Warning: Could not read or parse {file_path_url}. Error: {content}")
    return dependencies

  try:
    tree = ast.parse(content, filename=file_path_url)
  except (SyntaxError, ValueError) as e:
    logger.warning(f"Warning: Could not parse {file_path_url}. Error: {e}")
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
                                          classes, or `if TYPE_CHECKING:` blocks
                                          are ignored for dependency analysis.
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
    logger.info(f"Processing {i} {current_file.replace(base_path,'')}")
    # Skip files that have already been processed to avoid redundant work and cycles.
    if current_file in processed_files:
      continue
    processed_files.add(current_file)

    dependency_graph.setdefault(current_file, [])
    reverse_graph.setdefault(current_file, [])

    dependencies = find_file_dependencies(current_file, base_path, exclude_conditional_imports)
    dependenciesname = {k.replace(base_path, "") for k in dependencies}
    logger.info(f"File {current_file.replace(base_path,'')} Have {dependenciesname}")
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
    logger.error(f"\nError: A circular dependency was detected. Cycle: {cycle_msg}")
    if returnDependencies:
      return [], rel_dependency_graph
    else:
      return []


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


def save_results_in_file(sorted_files, dependencies, args, outFile="FileOrder.txt"):
  """
  Saves the sorted files and their dependencies to a file.

  Args:
      sorted_files (list): A list of file paths sorted by dependency.
      dependencies (dict): A dictionary where keys are file paths and values are lists of their dependencies.
      args (argparse.Namespace): The command-line arguments.
      outFile (str): The name of the output file.
  """
  with open("all_files.json", "w") as f:
    json.dump({"sorted_files": sorted_files, "dependencies": dependencies}, f)
  standalone_module = [mod for mod in sorted_files if mod not in dependencies or len(dependencies[mod]) == 0]
  dependent_sorted_modules = {
      mod: dependencies[mod] for mod in sorted_files if mod in dependencies and len(dependencies[mod]) > 0
  }
  with open(outFile, "w") as f:
    f.write(f"BasePath {args.base_path}\n")
    f.write(f"Entry File {args.entry_file_path}\n")
    f.write(f"Standalone Files:\n {json.dumps(standalone_module,indent=4)}\n")
    f.write(f"Dependent Files\n {json.dumps(dependent_sorted_modules,indent=4)}\n")


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
