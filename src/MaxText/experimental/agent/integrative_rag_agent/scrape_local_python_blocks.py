# Copyright 2023–2025 Google LLC
# ... (License headers omitted for brevity) ...

"""
Local version of the scraping agent.
Scrapes directly from the filesystem relative to this script's location.
"""
import ast
import json
import os
import sys

# Attempt to import config. If running locally without full package install, 
# you might need to adjust python path or just hardcode these for this test.
try:
    from MaxText.experimental.agent.integrative_rag_agent import system_setup
    from MaxText.experimental.agent.integrative_rag_agent.config import maxtext_code_block, block_for_rag
except ImportError:
    # Fallback if running as a standalone script outside package structure
    print("Warning: Could not import MaxText config. Using default paths.")
    maxtext_code_block = "maxtext_code_blocks.json"
    # Assuming standard MaxText structure, these are likely the targets:
    block_for_rag = [
        "src/MaxText/layers",
        "src/MaxText/inference",
        "src/MaxText/common_types.py",
        "src/MaxText/maxtext_utils.py"
    ]

def scrape_python_blocks(source_code, file_path_for_logging):
  """
  Parses Python source code and extracts top-level blocks.
  (Same logic as before, using ast)
  """
  blocks = []
  try:
    tree = ast.parse(source_code, filename=file_path_for_logging)
  except (SyntaxError, ValueError) as e:
    print(f"Error parsing {file_path_for_logging}: {e}")
    return []

  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      blocks.append(ast.get_source_segment(source_code, node))
    elif isinstance(node, ast.Try):
      blocks.append(ast.get_source_segment(source_code, node))
    elif isinstance(node, ast.If):
      blocks.append(ast.get_source_segment(source_code, node))

  return blocks


def scrape_local_path(repo_root, relative_path, all_scraped_blocks, all_full_codes):
  """
  Scrapes a file or directory recursively from the local file system.

  Args:
      repo_root (str): The absolute path to the root of the repository.
      relative_path (str): The path to the file or dir relative to repo_root.
  """
  full_path = os.path.join(repo_root, relative_path)

  if not os.path.exists(full_path):
    print(f"Warning: Path not found: {full_path}")
    return

  # Case 1: It's a single file
  if os.path.isfile(full_path) and full_path.endswith(".py"):
    _process_file(full_path, relative_path, all_scraped_blocks, all_full_codes)

  # Case 2: It's a directory
  elif os.path.isdir(full_path):
    print(f"Scraping directory: {relative_path}")
    for root, _, files in os.walk(full_path):
      for file in files:
        if file.endswith(".py"):
          file_abs_path = os.path.join(root, file)
          # Get path relative to the repo root for clean keys
          file_rel_path = os.path.relpath(file_abs_path, repo_root)
          _process_file(file_abs_path, file_rel_path, all_scraped_blocks, all_full_codes)


def _process_file(abs_path, rel_path, all_scraped_blocks, all_full_codes):
  """Helper to read and parse a single file."""
  print(f"Scraping file: {rel_path}")
  try:
    with open(abs_path, "r", encoding="utf-8") as f:
      source_code = f.read()
    
    all_full_codes[rel_path] = source_code
    all_scraped_blocks[rel_path] = scrape_python_blocks(source_code, rel_path)
  except Exception as e:
    print(f"Failed to read {abs_path}: {e}")


def find_and_scrape_local(repo_root, paths):
  all_scraped_blocks = {}
  all_full_codes = {}
  for path in paths:
    scrape_local_path(repo_root, path, all_scraped_blocks, all_full_codes)
  return all_scraped_blocks, all_full_codes


def save_scrapped_code_blocks(scraped_blocks):
  with open(maxtext_code_block, "wt", encoding="utf-8") as f:
    json.dump(scraped_blocks, f, indent=4)


def main():
  # 1. Determine the Repo Root
  # Assuming this script is at: maxtext/src/MaxText/experimental/agent/integrative_rag_agent/
  # We need to go up 5 levels to get to 'maxtext' root folder.
  # Adjust this number if your folder structure is different.
  script_dir = os.path.dirname(os.path.abspath(__file__))
  # Go up 5 levels: integrative_rag_agent -> agent -> experimental -> MaxText -> src -> maxtext(repo)
  repo_root = os.path.abspath(os.path.join(script_dir, "../../../../.."))
  
  print(f"Detected Repo Root: {repo_root}")

  # 2. Run the scraper
  print(f"Starting local scraper...\n")
  scraped_blocks, full_codes = find_and_scrape_local(repo_root, block_for_rag)
  
  save_scrapped_code_blocks({"scraped_blocks": scraped_blocks, "full_codes": full_codes})

  # 3. Summary
  total_blocks = sum(len(b) for b in scraped_blocks.values())
  print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
  print(f"Output saved to: {maxtext_code_block}")
  print(f"Total files scraped: {len(full_codes)}")
  print(f"Total Blocks scraped: {total_blocks}")


if __name__ == "__main__":
  main()