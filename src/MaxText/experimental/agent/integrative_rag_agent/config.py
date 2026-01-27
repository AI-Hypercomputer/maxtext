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
Configuration file for the Code Conversion Agent
This file contains all the configurable parameters for the PyTorch to JAX code conversion process
"""

# Similarity threshold for finding similar code blocks between PyTorch and JAX implementations
# Lower values (closer to 0) mean more strict similarity matching
# Higher values (closer to 1) allow more lenient matching
torch_jax_similarity_threshold = 0.20

# Enables or disables caching. If True, results of previous computations are reused; else rerun computations every time.
enable_cache = True

# Stores blocks of code/data that were found to be similar above the given similarity_threshold.
save_similar_blocks = True

# Saves the single most similar block for each input, which is useful for debugging and verifying similarity matches.
save_most_similar_block_for_debugging = True

# Path to the folder containing the dataset or source files that need to be processed.
data_set_folder = "dataset/"

# Directory used to store cached intermediate results, allowing faster re-runs without recomputing from scratch.
cache_folder = "Cache/"


# GitHub or repository hosting owner/organization name where the codebase resides.
repo_owner = "AI-Hypercomputer"

# Repository name under the given owner. Unique when combined with repo_owner, e.g., AI-Hypercomputer/maxtext.
repo_name = "maxtext"

# Directory where the final processed outputs, analysis, or reports are saved.
results_folder = "results/"

# Directory used to store status checkpoints or progress information, useful for resuming long-running processes.
status_folder = "status/"

# Path where log files are written, capturing runtime information, errors, and debugging details.
logs_folder = "logs/"


# Flag to save dependency lists for each module
# Useful for understanding the dependency graph and debugging conversion issues
save_dependency_list = True

# Folder path for storing temporary similar code blocks during analysis
similar_block_folder = "Temp/"


# Database file path for storing RAG (Retrieval-Augmented Generation) embeddings and metadata
# Used for semantic search and code similarity matching
rag_db_file = data_set_folder + "rag_store.db"

# JSON file containing extracted code blocks from the MaxText repository
# Each block represents a functional unit of code for analysis
maxtext_code_block = data_set_folder + "maxtext_blocks.json"

# JSON file containing detailed descriptions of each code block
# Includes metadata about functionality, parameters, and usage
maxtext_block_description = data_set_folder + "maxtext_blocks_description.json"

new_module_file_format = status_folder + "{module_Name}/new_modules.json"

processed_module_file_format = status_folder + "{module_Name}/Processed_modules.json"

# Cache file for storing dependency search results based on similarity threshold
# Filename includes the threshold value to allow multiple threshold configurations
torch_jax_similar_dependency_cache_file = f"{cache_folder}search_dependency_cache_{torch_jax_similarity_threshold}.json"

# Cache file for storing filtered dependency results
# Contains dependencies after applying filtering rules and constraints
dependency_filter_cache_file = f"{cache_folder}dependency_cache.json"

# File format string for saving dependency lists for each module
# {module_name} placeholder gets replaced with actual module names
dependency_list_file_format = f"{logs_folder}dependencies_{{module_name}}.txt"

# File format string for saving the ordered list of files to convert
# Contains the sequence of files in dependency order for conversion
files_order_file_format = f"{results_folder}{{module_name}}_files_order.json"

# File format string for saving progress status during conversion
# Tracks which modules have been processed and their current state
progress_status_file_format = f"{status_folder}{{module_name}}_Status.json"

# List of specific paths within the repository to scrape for code blocks
# These paths contain the JAX implementations that will be used as reference
# for converting PyTorch code to JAX
block_for_rag = [
    "src/MaxText/layers",  # Neural network layers and building blocks
    "src/maxtext/inference",  # Inference and prediction code
    "src/MaxText/common_types.py",  # Common data types and structures
    "src/MaxText/maxtext_utils.py",  # Utility functions and helpers
]
