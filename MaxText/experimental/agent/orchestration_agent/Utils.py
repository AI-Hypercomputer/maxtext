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

import ast
from urllib.parse import urlparse, urljoin

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

  parts = parsed.path.split("/")
  if len(parts) < 5:
    raise ValueError(f"Invalid GitHub blob URL: {blob_url}")  # Not a valid blob URL

  user = parts[1]
  repo = parts[2]
  commit = parts[4]
  file_path = "/".join(parts[5:])

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
    basemodule = filepath.lstrip("/").split("/")[0]
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
        except (SyntaxError, TypeError, ValueError, RecursionError):
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
