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
This file defines the main orchestration agent for analyzing Python file dependencies
within a GitHub repository. It first determines the hierarchical order of files based
on their import relationships, then further breaks down each file into its constituent
components (functions, classes, variables, and imports), identifying internal dependencies
within each file. The output provides a comprehensive view of the codebase structure,
first by file-level dependencies, and then by component-level dependencies within each file. This
script scrapes all python files from the MaxText repository, extracts top-level classes,
functions, and try/if blocks, and saves them to a JSON file.

Example Invocation:
python scrap_all_python_blocks.py
"""

import ast
import base64
import json
import os

import dotenv
import requests

from MaxText.experimental.agent.integrative_rag_agent import system_setup
from MaxText.experimental.agent.integrative_rag_agent.config import (
  repo_name,
  repo_owner,
  maxtext_code_block,
  block_for_rag,
)

dotenv.load_dotenv()


def scrape_python_blocks(source_code, file_path_for_logging):
  """
  Parses Python source code from a string and extracts top-level classes,
  functions, and try/if blocks. It ignores methods inside classes and
  any blocks nested within functions.

  Args:
      source_code (str): The Python source code as a string.
      file_path_for_logging (str): The path of the file being scraped, for logging purposes.

  Returns:
      list: A list of strings, where each string is a source code block.
  """
  blocks = []
  try:
    # Parse the source code into an Abstract Syntax Tree (AST)
    tree = ast.parse(source_code, filename=file_path_for_logging)
  except (SyntaxError, ValueError) as e:
    print(f"Error parsing {file_path_for_logging}: {e}")
    return []

  # Iterate over only the top-level nodes in the module's body
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      # Extracts top-level functions and full classes
      blocks.append(ast.get_source_segment(source_code, node))
    elif isinstance(node, ast.Try):
      # Extracts top-level try...except...finally blocks
      blocks.append(ast.get_source_segment(source_code, node))
    elif isinstance(node, ast.If):
      # Extracts top-level if blocks, as requested
      blocks.append(ast.get_source_segment(source_code, node))

  return blocks


def scrape_github_repository(owner, repo, path, all_scraped_blocks, all_full_codes, token=None):
  """
  Recursively scrapes Python files from a GitHub repository path.

  Args:
      owner (str): The owner of the GitHub repository.
      repo (str): The name of the GitHub repository.
      path (str): The path to a file or directory within the repository.
      all_scraped_blocks (dict): A dict to accumulate the scraped code blocks.
      all_full_codes (dict): A dict to accumulate the full source code of each file.
      token (str, optional): A GitHub Personal Access Token for authentication.
  """
  api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
  headers = {"Accept": "application/vnd.github.v3+json"}

  if token:
    headers["Authorization"] = f"token {token}"

  try:
    response = requests.get(api_url, headers=headers, timeout=360)
    response.raise_for_status()
  except requests.exceptions.RequestException as e:
    print(f"Error fetching {api_url}: {e}")
    return

  content_data = response.json()

  if isinstance(content_data, dict) and content_data.get("type") == "file":
    if content_data["name"].endswith(".py"):
      file_path = content_data["path"]
      print(f"Scraping file: {file_path}")

      file_content_b64 = content_data["content"]
      file_content_bytes = base64.b64decode(file_content_b64)
      source_code = file_content_bytes.decode("utf-8")

      # Store the full source code
      all_full_codes[file_path] = source_code

      # Store the individual blocks
      all_scraped_blocks[file_path] = scrape_python_blocks(source_code, file_path)

  elif isinstance(content_data, list):
    print(f"Scraping directory: {path}")
    for item in content_data:
      scrape_github_repository(owner, repo, item["path"], all_scraped_blocks, all_full_codes, token)


def find_and_scrape_from_github(owner, repo, paths, token=None):
  """
  Finds Python files in GitHub paths and scrapes their blocks and full code.

  Args:
      owner (str): The owner of the GitHub repository.
      repo (str): The name of the GitHub repository.
      paths (list): A list of file or directory paths within the repo.
      token (str, optional): A GitHub Personal Access Token for authentication.

  Returns:
      tuple: A tuple containing two dictionaries:
             - A dictionary of scraped code blocks.
             - A dictionary of the full source code for each file.
  """
  all_scraped_blocks = {}
  all_full_codes = {}
  for path in paths:
    scrape_github_repository(owner, repo, path, all_scraped_blocks, all_full_codes, token)
  return all_scraped_blocks, all_full_codes


def save_scrapped_code_blocks(scraped_blocks):
  with open(maxtext_code_block, "wt", encoding="utf-8") as f:
    json.dump(scraped_blocks, f, indent=4)


def main():
  system_setup.setup_directories()
  GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

  # Run the scraper
  print(f"Starting scraper for repository: {repo_owner}/{repo_name}\n")
  scraped_blocks, full_codes = find_and_scrape_from_github(repo_owner, repo_name, block_for_rag, GITHUB_TOKEN)
  save_scrapped_code_blocks({"scraped_blocks": scraped_blocks, "full_codes": full_codes})

  # 1. Print the scraped blocks for each file
  print("\n" + "=" * 20 + " SCRAPED BLOCKS " + "=" * 20)
  total_blocks = 0
  if scraped_blocks:
    for file_path, blocks in scraped_blocks.items():
      print(f"\n--- Blocks from: {file_path} ---")
      if blocks:
        for i, block in enumerate(blocks):
          print(f"--- Block {i + 1} ---\n{block}\n")
          total_blocks += 1
      else:
        print("No blocks found in this file.")
  else:
    print("No code blocks were scraped.")

  # 2. Print the full source code for each file
  print("\n" + "=" * 20 + " FULL SOURCE CODE " + "=" * 20)
  if full_codes:
    for file_path, source in full_codes.items():
      print(f"\n--- Source from: {file_path} ---\n")
      print(source)
  else:
    print("No full source code was scraped.")

  print(f"\nTotal files scraped: {len(full_codes)}\nTotal Blocks {total_blocks}")


if __name__ == "__main__":
  main()
