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
This file provides functionality to analyze Python file dependencies within a GitHub repository
and generate a topologically sorted list of files based on their import relationships.
It handles both absolute and relative imports, and can optionally exclude conditional imports.

Example Invocations:

1. Analyze a specific entry file in a repository, excluding conditional imports (default):
```sh
python get_files_in_hierarchical_order.py \
   --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
--entry-file-path https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
```
2. Analyze a specific entry file, including all imports (even conditional ones):
```sh
python get_files_in_hierarchical_order.py \
  --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
--entry-file-path https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
  --no-exclude-conditional-imports
```
"""
from collections import deque
import argparse
import ast
import json
import logging

from .utils import find_cycle, check_github_file_exists, get_github_file_content, url_join

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def have_module(target_name, file_url):
  """Checks if a given name is defined in a Python file.

  This function parses a Python file from a URL to determine if `target_name`
  is defined as a function, class, or top-level variable. It can also detect
  if `target_name` is an alias re-exported from another module via an
  `ast.ImportFrom` statement.

  Args:
    target_name: The name of the function, class, or variable to find.
    file_url: The full GitHub URL of the Python file to inspect.

  Returns:
    A boolean or a tuple indicating the result:
    - True: If `target_name` is explicitly defined in the file.
    - ("ImportFrom", str): If `target_name` is an alias in an `ImportFrom`
      statement. The second element is the full module path.
    - False: If the name is not found, or if the file content cannot be
      retrieved or parsed.
  """
  flag, content = get_github_file_content(file_url)
  if not flag:
    logger.warning("Warning: Could not read or parse %s. Error: %s", file_url, content)
    return False  # Fail if content cannot be retrieved

  try:
    tree = ast.parse(content, filename=file_url)
  except (SyntaxError, ValueError):
    logger.warning("Warning: Could not parse %s", file_url)
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
        if target_name in (alias.asname, alias.name):
          module = node.module or ""
          level = node.level
          full_module = "." * level + module if level > 0 else module
          return "ImportFrom", full_module
  return False


def resolve_complex_import(module_path_base_url, import_package, base_url, current_dir_url, current_recursion_depth=0):
  """Recursively resolves a complex import to a file URL.

  This function handles imports that might refer to a package directory
  (with an `__init__.py`) or a submodule. It attempts to resolve the import
  by checking for `.py` files and package `__init__.py` files, and can
  follow re-exports within `__init__.py` files.

  Args:
    module_path_base_url: The base URL for the module path being resolved.
    import_package: The specific name being imported (e.g., 'modeling_llama').
    base_url: The root URL of the repository's source directory.
    current_dir_url: The URL of the directory containing the import statement.
    current_recursion_depth: Internal counter to prevent infinite recursion.

  Returns:
    The resolved full GitHub URL of the imported file, or None if not found.

  Example:
    If `module_path_base_url` is
    "https://.../src/transformers/models/llama", `import_package` is
    "modeling_llama", this function would check for:
    1. `.../src/transformers/models/llama.py`
    2. `.../src/transformers/models/llama/__init__.py`
    If `__init__.py` re-exports the name, it will recurse to find the source.
  """

  if current_recursion_depth > 4:  # Increased recursion limit slightly for network latency
    logger.error(
        "Error: Exceeded recursion depth while resolving import for '%s' starting from '%s'",
        import_package,
        module_path_base_url,
    )
    return None
  # Check for a direct .py file containing the definition
  potential_py_url = f"{module_path_base_url}.py"
  if check_github_file_exists(potential_py_url)[0]:
    if have_module(import_package, potential_py_url) is True:
      return potential_py_url

  # Check for a package (directory with __init__.py)
  potential_pkg_init_url = url_join(module_path_base_url, "__init__.py")
  if check_github_file_exists(potential_pkg_init_url)[0] and potential_pkg_init_url.startswith(base_url):
    has_module = have_module(import_package, potential_pkg_init_url)
    if has_module:
      return potential_pkg_init_url
    else:
      # The package exists, but the import is not in __init__. It could be a submodule.
      potential_file_in_pkg_url = url_join(module_path_base_url, f"{import_package}.py")
      if check_github_file_exists(potential_file_in_pkg_url)[0] and potential_file_in_pkg_url.startswith(base_url):
        return potential_file_in_pkg_url
  return None


def resolve_import_path(importer_url, module_name, level, base_url, import_package=None):
  """Resolves an import statement to a full GitHub file URL.

  Handles both absolute and relative imports. For a given import statement,
  it constructs the potential path to the imported module and checks for its
  existence as a `.py` file or as a package.

  Args:
    importer_url: The URL of the file containing the import.
    module_name: The name of the module from the import statement (e.g.,
      'transformers.models.llama').
    level: The relative import level (0 for absolute, 1 for `.` , 2 for `..`).
    base_url: The root URL of the repository's source directory.
    import_package: The specific name being imported in a `from ... import ...`
      statement (e.g., `LlamaModel` from `... import LlamaModel`).

  Returns:
    The resolved full GitHub URL of the imported file, or None if not found.
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
  return resolve_complex_import(module_path_base_url, import_package, base_url, current_dir_url)


def find_file_dependencies(file_path_url, base_path_url, exclude_conditional_imports=True):
  """Finds all direct Python file dependencies for a given file.

  Parses a Python file to find all `import` and `from ... import ...`
  statements. It then resolves these imports to their corresponding file URLs
  within the repository.

  Args:
    file_path_url: The full GitHub URL of the Python file to analyze.
    base_path_url: The base URL of the repository's source directory, used to
      filter out external dependencies.
    exclude_conditional_imports: If True, ignores imports inside functions,
      classes, or `if TYPE_CHECKING:` blocks.

  Returns:
    A set of full GitHub URLs corresponding to the dependent Python files.
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
  """Generates a dependency-sorted list of all Python files in a project.

  Starting from an entry file, this function performs a dependency analysis
  across a Python project on GitHub. It builds a graph of file-level imports
  and returns a topologically sorted list of all discovered files.

  Args:
    entry_file_path: The full GitHub URL of the starting Python file.
    base_path: The base URL of the repository's source directory.
    exclude_conditional_imports: If True, imports inside functions, classes,
      or `if TYPE_CHECKING:` blocks are ignored during dependency analysis.
    returnDependencies: If True, returns both the sorted file list and the
      dependency graph.

  Returns:
    If `returnDependencies` is False (default):
      A list of file paths (relative to `base_path`) in topological order.
      Returns an empty list if a circular dependency is detected.
    If `returnDependencies` is True:
      A tuple containing:
        - A list of relative file paths in topological order.
        - A dictionary representing the dependency graph, where keys are
          relative file paths and values are lists of their dependencies.
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
    logger.info("File %s Have %s", current_file.replace(base_path, ""), dependencies_name)
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


def arg_parser():
  """Creates and configures the argument parser for the script.

  Returns:
    An `argparse.ArgumentParser` instance.
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
  """Saves the sorted file list and dependencies to output files.

  This function writes the analysis results into two files:
  1. `all_files.json`: A JSON file containing the sorted file list and the
     full dependency graph.
  2. `FileOrder.txt`: A human-readable text file summarizing standalone and
     dependent modules.

  Args:
    sorted_files: A list of file paths sorted by dependency.
    dependencies: A dictionary where keys are file paths and values are lists
      of their dependencies.
    args: The parsed command-line arguments from `argparse`.
    outFile: The name of the text output file.
  """
  with open("all_files.json", "wt", encoding="utf8") as f:
    json.dump({"sorted_files": sorted_files, "dependencies": dependencies}, f)
  standalone_module = [mod for mod in sorted_files if mod not in dependencies or len(dependencies[mod]) == 0]
  dependent_sorted_modules = {
      mod: dependencies[mod] for mod in sorted_files if mod in dependencies and len(dependencies[mod]) > 0
  }
  with open(outFile, "wt", encoding="utf8") as f:
    f.write(f"BasePath {args.base_path}\n")
    f.write(f"Entry File {args.entry_file_path}\n")
    f.write(f"Standalone Files:\n {json.dumps(standalone_module,indent=4)}\n")
    f.write(f"Dependent Files\n {json.dumps(dependent_sorted_modules,indent=4)}\n")


def main():
  """Main entry point for the dependency analysis script.

  Parses command-line arguments, runs the file dependency analysis starting
  from the entry file, and saves the results to disk. It logs the progress
  and the final sorted list of files.
  """
  args = arg_parser()
  BASE_PATH = args.base_path
  ENTRY_FILE_PATH = args.entry_file_path
  EXCLUDE_CONDITIONAL_IMPORTS = args.exclude_conditional_imports
  if not check_github_file_exists(ENTRY_FILE_PATH)[0]:
    logger.error("Error: Entry file not found at '%s'", ENTRY_FILE_PATH)
  else:
    # Use rstrip to handle base paths that may or may not have a trailing slash
    relative_entry = ENTRY_FILE_PATH.replace(BASE_PATH.rstrip("/"), "")
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
