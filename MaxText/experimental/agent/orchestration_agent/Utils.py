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
import sys
from urllib.parse import urlparse, urljoin
import ast
import os

import requests  # if this is not available, please try ``pip install requests``


def github_blob_to_raw(blob_url):
  """
  Converts a GitHub blob URL to its raw content URL.

  Args:
      blob_url (str): The URL of the GitHub blob.

  Returns:
      str: The URL of the raw content of the GitHub blob.
  """
  parsed = urlparse(blob_url)
  if "github.com" not in parsed.netloc or "/blob/" not in parsed.path:
    raise ValueError(f"Invalid GitHub blob URL: {blob_url}")  # Not a valid blob URL

  parts = parsed.path.split(os.path.sep)
  if len(parts) < 5:
    raise ValueError(f"Invalid GitHub blob URL: {blob_url}")  # Not a valid blob URL

  user = parts[1]
  repo = parts[2]
  commit = parts[4]
  file_path = os.path.sep.join(parts[5:])

  raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{commit}/{file_path}"
  return raw_url


def check_github_file_exists(blob_url):
  """
  Checks if a file exists on GitHub given its blob URL.

  Args:
      blob_url (str): The URL of the GitHub blob.

  Returns:
      bool: True if the file exists, False otherwise.
      str: The raw URL of the file if it exists, or an error message.

  """
  raw_url = github_blob_to_raw(blob_url)
  if not raw_url:
    return False, "Invalid Github blob URL"
  try:
    response = requests.head(raw_url, timeout=10)
    return response.status_code == 200, raw_url
  except requests.RequestException as e:
    return False, str(e)


def get_github_file_content(blob_url):
  """
  Retrieves the content of a file from GitHub given its blob URL.

  Args:
      blob_url (str): The URL of the GitHub blob.

  Returns:
      tuple: A tuple containing:
          - bool: True if the content was retrieved successfully, False otherwise.
          - str: The content of the file if successful, or an error message if not.

  """
  exists, raw_url_or_error = check_github_file_exists(blob_url)
  if not exists:
    return False, f"File does not exist or failed to check: {raw_url_or_error}"
  try:
    response = requests.get(raw_url_or_error, timeout=10)
    if response.status_code == 200:
      return True, response.text
    else:
      return False, f"Failed to fetch file. Status code: {response.status_code}"
  except requests.RequestException as e:
    return False, str(e)


def check_if_file_exists(url):
  """Check whether a GitHub blob URL or local file path exists.

  Args:
    url (str): Either a GitHub blob URL (e.g.,
      'https://github.com/.../blob/.../path.py') or a local filesystem path.

  Returns:
    tuple[bool, str]:
      - True with the resolved raw URL/path if the file exists
      - False with an error message otherwise
  """
  if "http" in url and "github.com" in url:
    return check_github_file_exists(url)
  if os.path.exists(url):
    return True, url
  else:
    return False, "Not a GitHub URL and no local file found"


def get_file_content(url):
  """Retrieve file contents from a GitHub blob URL or a local path.

  Args:
    url (str): Either a GitHub blob URL or a local filesystem path.

  Returns:
    tuple[bool, str]:
      - (True, contents) on success
      - (False, error_message) on failure
  """
  if "http" in url and "github.com" in url:
    return get_github_file_content(url)
  if os.path.exists(url):
    with open(url, "rt", encoding="utf-8") as f:
      return True, f.read()
  else:
    return False, "Not a GitHub URL and no local file found"


def url_join(*args):
  """
  Joins multiple URL parts intelligently, handling relative paths.
  """
  if not args:
    return ""
  full_url = args[0]
  for part in args[1:]:
    if part.startswith("."):
      part = part[1:]
      while len(part) > 0 and "." == part[0]:
        full_url = full_url.removesuffix("/").rsplit("/", 1)[0]
        part = part[1:]
    if not full_url.endswith("/"):
      full_url += "/"
    full_url = urljoin(full_url, part)
  return full_url


def find_cycle(graph):
  """
  Finds a cycle in a directed graph using DFS.

  Args:
      graph (dict): A dictionary representing the graph where keys are nodes
                    and values are lists of their direct dependencies.

  Returns:
      list: A list of nodes forming a cycle if one is found, otherwise None.

  """
  visited = set()
  stack = set()
  path = []

  def dfs(node):
    visited.add(node)
    stack.add(node)
    path.append(node)

    for neighbor in graph.get(node, []):
      if neighbor not in visited:
        if dfs(neighbor):
          return True
      elif neighbor in stack:
        return True
    stack.remove(node)
    path.pop()
    return False

  for node in graph:
    if node not in visited:
      if dfs(node):
        try:
          idx = path.index(path[-1])
          return path[idx:]
        except (ValueError, IndexError):
          return path
  return None


def remove_local_imports(source_code, filepath=None):
  """
  Removes local imports from Python source code.

  This function parses the provided source code and removes import statements
  that are considered 'local'. Local imports are defined as:
  1. Relative imports (e.g., `from . import my_module`).
  2. Imports from the base module if a `filepath` is provided (e.g., if
     filepath is 'my_project/module.py', then `import my_project.utils`
     would be removed).

  Args:
      source_code (str): The Python source code as a string.
      filepath (str, optional): The path to the file containing the source
                                code. Used to determine the base module for
                                identifying local imports. Defaults to None.

  Returns:
      tuple: A tuple containing:
          - str: The modified source code with local imports removed.
          - str: A newline-separated string of the names of the removed imports.
  """
  if filepath is not None:
    # Determine the base module from the filepath
    basemodule = filepath.lstrip(os.path.sep).split(os.path.sep)[0]
  else:
    basemodule = ""
  tree = ast.parse(source_code)
  lines = source_code.splitlines()
  new_lines = []
  removed_imports = []
  # Build a mapping from line number to node for import nodes
  import_nodes = {node.lineno: node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}

  i = 0
  while i < len(lines):
    line = lines[i]
    node = import_nodes.get(i + 1, None)
    if node:
      # Detect multi-line import (parenthesis)
      if "(" in line.strip() and ")" not in line:
        # Detect multi-line import (parenthesis)
        # Find the closing parenthesis, even if comments are present
        start = i
        while i < len(lines) and not lines[i].strip().endswith(")"):
          i += 1
        i += 1  # include the closing parenthesis line
        import_block = "\n".join(lines[start:i])
        # Parse the import block to check if it should be removed
        try:
          import_tree = ast.parse(import_block)
          import_node = import_tree.body[0]
        except Exception:
          import_node = node  # fallback

        remove = False
        import_names = []
        if isinstance(import_node, ast.ImportFrom):
          # Remove relative imports
          if import_node.level > 0:
            remove = True
          # Remove imports from basemodule
          elif import_node.module and import_node.module.split(".")[0] == basemodule:
            remove = True
          if remove:
            import_names = [alias.name for alias in import_node.names]
        elif isinstance(import_node, ast.Import):
          for alias in import_node.names:
            if alias.name.split(".")[0] == basemodule:
              remove = True
              import_names.append(alias.name)
          # Only add names if remove is True
          if not remove:
            import_names = []
        if remove:
          removed_imports.extend(import_names)
        else:
          new_lines.extend(lines[start:i])
        # increment i to avoid infinite loop
        continue  # already incremented i
      else:
        remove = False
        import_names = []
        if isinstance(node, ast.ImportFrom):
          if node.level > 0:
            remove = True
          elif node.module and node.module.split(".")[0] == basemodule:
            remove = True
          if remove:
            import_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
          for alias in node.names:
            if alias.name.split(".")[0] == basemodule:
              remove = True
              import_names.append(alias.name)
          if not remove:
            import_names = []
        if remove:
          removed_imports.extend(import_names)
        else:
          new_lines.append(line)
        i += 1
        continue
    else:
      new_lines.append(line)
      i += 1

  return "\n".join(new_lines), "\n".join(removed_imports)


def parse_python_code(code):
  """
  Extracts Python code from a string that might contain markdown code blocks.

  This function looks for Python code blocks delimited by triple backticks.
  It supports both '```python' and generic '```' delimiters.

  Args:
      code (str): The input string potentially containing Python code blocks.

  Returns:
      str: The extracted Python code. Returns the original string if no
           code blocks are found.
  """
  if "```python" in code:
    code = code.split("```python")[1]
    if "```" in code:
      code = code.split("```")[0]
  elif "```" in code:
    code = code.split("```")[1]
  return code


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
  flag, content = get_file_content(file_url)
  if not flag:
    print(f"Warning: Could not read or parse {file_url}. Error: {content}")
    return False  # Fail if content cannot be retrieved

  try:
    tree = ast.parse(content, filename=file_url)
  except (SyntaxError, ValueError):
    print(f"Warning: Could not parse {file_url}")
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
          return ("ImportFrom", node.module or "." * node.level)
  return False


def resolve_complex_import(module_path_base_url, importPackage, base_url, current_dir_url, num_try=0, Message=""):
  """
  Resolves a complex import statement, looking for the imported package/module
  within a directory structure. This handles cases where 'importPackage' might
  refer to a file or a directory (package) with an __init__.py.

  Args:
      module_path_base_url (str): The base URL for the module path
         (ex. 'https://github.com/.../transformers/models/llama').
      importPackage (str): The specific name being imported (e.g., 'modeling_llama', 'configuration_llama').
      base_url (str): The base URL of the repository.
      current_dir_url (str): The URL of the directory containing the original import statement.
      num_try (int): Counter for recursion depth.
      Message (str): Accumulates error messages for recursion depth.

  Returns:
      str: The resolved full GitHub URL of the imported file, or None if not found.

  Example:
      If `module_path_base_url` is "https://github.com/org/repo/blob/main/src/transformers/models/llama",
      `importPackage` is "modeling_llama", `base_url` is "https://github.com/org/repo/blob/main/src/",
      and `current_dir_url` is "https://github.com/org/repo/blob/main/src/transformers/models/llama",
      this function would first check for "https://github.com/org/repo/blob/main/src/transformers/models/llama.py".
      If not found, it would then check for
      "https://github.com/org/repo/blob/main/src/transformers/models/llama/__init__.py".
      If `__init__.py` exists and contains `from . import modeling_llama`, it would then check for
      "https://github.com/org/repo/blob/main/src/transformers/models/llama/modeling_llama.py".
  """

  message = ""
  if num_try == 0:
    message = f"There is an issue with import {importPackage} in {module_path_base_url} current dir is {current_dir_url}"
  if num_try > 4:  # Increased recursion limit slightly for network latency
    print("Error: Exceeded recursion depth.", message, file=sys.stderr)
    return None
  # Check for a direct .py file containing the definition
  potential_py_url = f"{module_path_base_url}.py"
  if check_if_file_exists(potential_py_url)[0] and have_module(importPackage, potential_py_url):
    return potential_py_url

  # Check for a package (directory with __init__.py)
  potential_pkg_init_url = url_join(module_path_base_url, "__init__.py")
  if check_if_file_exists(potential_pkg_init_url)[0] and (
      base_url in ("", "./") or potential_pkg_init_url.startswith(base_url)
  ):
    has_module = have_module(importPackage, potential_pkg_init_url)
    if has_module:
      return potential_pkg_init_url
    else:
      # The package exists, but the import is not in __init__. It could be a submodule.
      potential_file_in_pkg_url = url_join(module_path_base_url, f"{importPackage}.py")
      if check_if_file_exists(potential_file_in_pkg_url)[0] and (
          base_url in ("", "./") or potential_file_in_pkg_url.startswith(base_url)
      ):
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
  if check_if_file_exists(potential_url_py)[0]:
    return potential_url_py

  # If it's not a direct file, it might be a complex package import
  return resolve_complex_import(module_path_base_url, importPackage, base_url, current_dir_url)


def get_absolute_imports(import_line, file_url, project_root="transformers"):
  """Resolve a relative 'from ... import ...' into absolute import lines.

  This converts relative imports found in `import_line` into absolute imports
  anchored at `project_root`, using `file_url` (GitHub blob URL or local path)
  to determine the base location. Multi-line parenthesized imports are
  flattened before resolution.

  Behavior:
    - If `import_line` does not start with 'from ', it is returned unchanged.
    - If `import_line` is already absolute (no leading dots), it is returned unchanged.
    - On successful resolution, returns one or more absolute import lines
      joined by newlines.
    - If resolution fails, returns None.

  Args:
    import_line (str): The original import statement text.
    file_url (str): The URL or path of the file containing the import.
    project_root (str): The root package name used to compute absolute paths.

  Returns:
    str | None: The converted absolute import line(s), the original line if
    unchanged, or None if resolution failed.
  """
  if not import_line.startswith("from "):
    return import_line
  elif not import_line.removeprefix("from ").startswith("."):
    return import_line
  import_line = import_line.strip()
  if "(" in import_line and ")" in import_line:
    import_line = import_line.replace("\n", "").replace("(", "").replace(")", "").removesuffix(",")
  parts = import_line.split()
  from_part = parts[1]
  module_rest = from_part.lstrip(".")
  level = len(from_part) - len(module_rest)
  base_url = file_url.rsplit(project_root + "/", 1)[0]
  packages_and_aliases = [pkg.split(" as ") if " as " in pkg else (pkg, None) for pkg in " ".join(parts[3:]).split(",")]
  packages_and_aliases = [(pkg, "") if alies is None else (pkg, " as " + alies) for pkg, alies in packages_and_aliases]
  packages = [
      (resolve_import_path(file_url, module_rest, level, base_url, pkg.strip()), pkg, alies)
      for pkg, alies in packages_and_aliases
  ]
  packages = [
      "from "
      + import_path.removeprefix(base_url).removesuffix(".py").replace(os.path.sep, ".")
      + " import "
      + pkg.strip()
      + alies
      for import_path, pkg, alies in packages
      if import_path is not None
  ]
  return "\n".join(packages) if packages else None
