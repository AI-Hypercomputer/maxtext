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

import requests
from urllib.parse import urlparse, urljoin


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
