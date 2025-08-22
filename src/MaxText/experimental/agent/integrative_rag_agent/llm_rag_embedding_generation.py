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
This file contains functions for generating structured descriptions and embedding
vectors for Python code blocks. It leverages a large language model (LLM)
(Gemini) to create human-readable summaries and a separate embedding model to
convert these descriptions into numerical vectors for use in a Retrieval-Augmented
Generation (RAG) system.

It includes:
- `_init_cache` and `_make_cache_key`: For managing a SQLite cache to store
  generated descriptions and embeddings, avoiding redundant LLM calls and
  embedding computations.
- `get_code_embedding`: The primary function to retrieve or generate a code
  block's description and embedding, utilizing the cache.
- `get_code_description_with_gemini`: Interacts with the Gemini LLM to generate
  a structured description of a given code block.
- `save_analysis_blocks`: Persists the generated descriptions to a JSON file.
- `description_generation`: Orchestrates the process of generating descriptions
  for all scraped code blocks.
- `embedding_generation`: Orchestrates the process of generating embeddings
  for the descriptions and saving them to a SQLite database.

Example Invocations:

To generate descriptions and embeddings, skipping existing records:
```bash
python llm_rag_embedding_generation.py
```

To regenerate all descriptions and embeddings, overriding existing records:
```bash
python llm_rag_embedding_generation.py --override-existing-records
```
"""

import os
import sys
import time
import argparse
import pickle, numpy as np
import json
import sqlite3
import hashlib

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integrative_rag_agent import system_setup
from integrative_rag_agent.config import maxtext_code_block, maxtext_block_description, enable_cache
from code_generation_agent.llm_agent import GeminiAgent
from integrative_rag_agent.llm_rag_agent import EmbeddingAgent
from code_evaluation_agent.utils import get_last_defined_module
from orchestration_agent.SplitPythonFile import get_modules_from_file
from integrative_rag_agent.database_operations import save_document, load_all_documents
from integrative_rag_agent.prompts_integrative_rag import Description_Prompt, CODE_DESCRIPTION


# Create cache table if it doesn't exist
def _init_cache(db_path="embedding_cache.db"):
  """Initialize the SQLite cache table if it does not exist.

  Args:
    db_path (str): Path to the SQLite database file.
  """
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute(
      """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            module_code TEXT,
            code_description TEXT,
            embedding BLOB
        )
    """
  )
  conn.commit()
  conn.close()


def _make_cache_key(file_path, project_root, comp_name):
  """Create a deterministic cache key for the given module request.

  The key incorporates the file path, project root, component name and a hash
  of the file path string to avoid accidental collisions.

  Args:
    file_path (str): Full path or URL of the source file.
    project_root (str): Project root used to resolve imports.
    comp_name (str): Component (function/class/variable) name within the file.

  Returns:
    str: A hex SHA-256 hash string to be used as the cache key.
  """
  # Include file content + parameters in the key so it updates when file changes
  # file_bytes = Path(file_path).read_bytes()
  key_input = f"{file_path}|{project_root}|{comp_name}|{hashlib.sha256(file_path.encode('utf-8')).hexdigest()}"
  return hashlib.sha256(key_input.encode()).hexdigest()


def get_code_embedding(file_path, project_root, comp_name, db_path="dataset/embedding_cache.db"):
  """Return module code, its description, and an embedding vector.

  Uses caching when `enable_cache` is True. If a cache hit exists, the
  function returns cached values. Otherwise, it extracts the module code,
  generates a structured description via LLM, computes an embedding, and
  optionally stores all results in the cache.

  Args:
    file_path (str): URL or path to the source file containing the component.
    project_root (str): Base project package path for absolute import resolution.
    comp_name (str): Name of the component (function/class/variable) to extract.
    db_path (str): SQLite database path for caching.

  Returns:
    tuple[str|None, dict|None, list|numpy.ndarray|None]: A tuple of
      (module_code, code_description, embedding). Values may be None when
      extraction fails.
  """
  if enable_cache:
    _init_cache(db_path)
    key = _make_cache_key(file_path, project_root, comp_name)

    # Try to read from cache
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT module_code, code_description, embedding FROM cache WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()

    if row:
      module_code, code_description, embedding = row
      # Convert embedding back from JSON string
      return module_code, json.loads(code_description), pickle.loads(embedding)

  # If not cached, compute and store
  module_code, full_source_code = get_modules_from_file(
      file_path, module=comp_name, project_root=project_root, add_external_dependencies=True
  )
  if module_code is None:
    return None, None, None
  code_description = get_code_description_with_gemini(module_code, full_source_code)
  embedding_agent = EmbeddingAgent()
  embedding = embedding_agent(json.dumps(code_description))

  if enable_cache:
    # Save to cache
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO cache (key, module_code, code_description, embedding) VALUES (?, ?, ?, ?)",
        (key, module_code, json.dumps(code_description), pickle.dumps(np.array(embedding).astype(np.float32))),
    )
    conn.commit()
    conn.close()

  return module_code, code_description, embedding


def get_code_description_with_gemini(code_block, full_code_context, user_prompt=CODE_DESCRIPTION):
  """
  Analyzes a Python code block using the Gemini API to generate a structured description.

  Args:
      code_block (str): The specific Python function or class to analyze.
      full_code_context (str): The full source code of the file for context.
      user_prompt (str): The prompt template for the user message.
  Returns:
      dict: A dictionary containing the structured analysis, or an error message.
  """
  llm_agent = GeminiAgent(system_instruction=Description_Prompt)
  for i in range(5):
    resp = None
    try:
      resp = llm_agent(user_prompt.replace("{code_block}", code_block).replace("{full_code_context}", full_code_context))
      return json.loads(resp.text.removeprefix("```json").removesuffix("```"))
    except Exception as e:
      print("Exception in analyze_code_with_gemini", e)
      print("Response", resp)


def save_analysis_blocks(scraped_blocks):
  """Persist analyzed code block data to `maxtext_block_description` JSON file.

  Args:
    scraped_blocks (list[dict]): List of analysis records to write.
  """
  with open(maxtext_block_description, "w") as f:
    json.dump(scraped_blocks, f, indent=4)


def description_generation(skip_existing_records):
  """Generate structured descriptions for scraped code blocks via LLM.

  Reads block snippets and full file contexts from `maxtext_code_block`. For
  each block, calls `get_code_description_with_gemini` to produce a structured
  description JSON, deduplicating existing entries if `skip_existing_records`
  is True. Writes progress incrementally to `maxtext_block_description`.

  Args:
    skip_existing_records (bool): If True, do not re-analyze blocks already
      present in the output JSON.
  """
  # --- Configuration ---
  with open(maxtext_code_block) as f:
    scraped_data = json.load(f)
  scraped_blocks, full_codes = scraped_data["scraped_blocks"], scraped_data["full_codes"]

  # 2. Analyze each block with Gemini
  all_analyses = []
  if os.path.exists(maxtext_block_description):
    with open(maxtext_block_description) as f:
      all_analyses = json.load(f)
  print("\n" + "=" * 20 + " ANALYZING CODE WITH GEMINI " + "=" * 20)

  for file_path, blocks in scraped_blocks.items():
    if not blocks:
      continue
    full_context = full_codes.get(file_path, "")

    for i, block in enumerate(blocks):
      block_name = get_last_defined_module(block)
      if block_name is None:
        continue
      block_name = file_path + "#" + block_name
      if skip_existing_records and any([analyses["block_name"] == block_name for analyses in all_analyses]):
        print(f"Skipping block {block_name} as exists")
        continue
      print(f"  - Analyzing block {i + 1}/{len(blocks)}...")
      analysis = get_code_description_with_gemini(block, full_context)

      # Store the analysis with its context
      analysis_result = {"block_name": block_name, "file_path": file_path, "code_block": block, "analysis": analysis}
      all_analyses.append(analysis_result)
      save_analysis_blocks(all_analyses)

      # Be respectful of API rate limits
      time.sleep(1)

  print(f"\nAnalysis complete. Generated {len(all_analyses)} analyses.")


def embedding_generation(skip_existing_records):
  """Compute and store embeddings for generated code block descriptions.

  Loads analyses from `maxtext_block_description`, skips any blocks already in
  the RAG database (unless overrides are requested), and inserts new records
  via `save_document`.

  Args:
    skip_existing_records (bool): If True, skip blocks already present in
      the RAG database.
  """
  _, name_list, _, _, _ = load_all_documents()
  embedding_agent = EmbeddingAgent()
  with open(maxtext_block_description) as f:
    all_analyses = json.load(f)
  for doc in all_analyses:
    if skip_existing_records and doc["block_name"] in name_list:
      print(f"Block {doc['block_name']} existing skiping")
      continue
    print(f"Generating embeddings for: {doc['block_name']}")
    desc = json.dumps(doc["analysis"])
    embeddings = embedding_agent(
        desc
    )  # Make embedding of description not code block as we want some similar code in different framework
    if embeddings:
      save_document(doc["block_name"], doc["code_block"], desc, doc["file_path"], np.array(embeddings))
      print(f"\nSuccessfully generated embeddings for {doc['block_name']} with Vector dimension: {len(embeddings)}")
    else:
      print("\nFailed to generate embeddings.")
      print("\n---")


if __name__ == "__main__":
  system_setup.setup_directories()
  parser = argparse.ArgumentParser(description="Generate code block descriptions and embeddings.")
  parser.add_argument(
      "--override-existing-records",
      action="store_true",
      help="Override existing records for both description and embedding generation.",
  )
  args = parser.parse_args()
  skip_existing_records = not args.override_existing_records

  description_generation(skip_existing_records)
  embedding_generation(skip_existing_records)
