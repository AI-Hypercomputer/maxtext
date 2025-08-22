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
This script analyzes Python component dependencies within a GitHub repository,
starting from a specific entry module (e.g., a class or function). It generates
a topologically sorted list of all required components (functions, classes, etc.)
and their source code.

This tool combines file-level dependency resolution with intra-file component
analysis to build a complete and ordered dependency graph.

Example Invocation:

python get_components_in_hierarchical_order.py \
  --base-path "https://github.com/huggingface/transformers/blob/main/src/" \
  --entry-file-path "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py" \
  --entry-module "LlamaForCausalLM"
```
"""

import copy

from config import (
    save_most_similar_block_for_debugging,
    similar_block_folder,
    dependency_filter_cache_file,
    save_dependency_list,
    files_order_file_format,
    progress_status_file_format,
    enable_cache,
    torch_jax_similarity_threshold,
    torch_jax_similar_dependency_cache_file,
    dependency_list_file_format,
)

import os, hashlib
import sys
import json
import argparse
from integrative_rag_agent import system_setup
import logging

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import deque, defaultdict
from integrative_rag_agent.prompts_integrative_rag import Dependency_Filter_Prompt
from code_generation_agent.llm_agent import GeminiAgent
from integrative_rag_agent.llm_rag_embedding_generation import get_code_embedding
from integrative_rag_agent.database_operations import make_embedding_index, search_embedding
from orchestration_agent.SplitPythonFile import get_modules_in_order as get_file_components

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ids, names, code_texts, files, embedding_index = make_embedding_index()


def search_similar_dependency(depend, base_path, project_root):
  """
  Searches for similar code dependencies using embedding similarity.

  This function first checks a cache for the given dependency. If not found, it
  generates an embedding for the dependency's code and then searches an
  embedding index for similar code blocks. Results with a similarity score below
  a predefined threshold are considered matches.

  Args:
    depend (str): The dependency to search for, in the format "file_path#component_name".
    base_path (str): The base URL of the source repository.
    project_root (str): The root directory of the project.

  Returns:
    tuple: A tuple containing:
           - A list of tuples, where each tuple contains (distance, name, file, code_text)
             for similar dependencies found. Returns None if no similar dependencies are found.
           - The code of the original module.
  """
  if enable_cache:
    if os.path.exists(torch_jax_similar_dependency_cache_file):
      with open(torch_jax_similar_dependency_cache_file, "r") as f:
        search_cache = json.load(f)
    else:
      search_cache = {}

    dep_hash = hashlib.sha256(depend.encode()).hexdigest()
    cache_key = dep_hash

    if cache_key in search_cache:
      return search_cache[cache_key]["final_result"], search_cache[cache_key]["module_code"]

  # llm call
  dep_file_path, dep_comp_name = depend.split("#")
  module_code, description, embedding = get_code_embedding(base_path + dep_file_path, project_root, dep_comp_name)
  if module_code is None:
    return None, None

  results = search_embedding(embedding, embedding_index, code_texts, top_k=3)
  logger.info("Distances %s", list(map(lambda x: x[1], results)))
  if save_most_similar_block_for_debugging:
    os.makedirs(similar_block_folder + dep_comp_name, exist_ok=True)
    for i, (code, dis, index) in enumerate(results):
      with open(similar_block_folder + dep_comp_name + "/" + str(i) + "_" + str(dis) + ".py", "w") as f:
        f.write("# Distance:  " + str(dis) + "\n")
        f.write("##Base_code\n" + module_code)
        f.write("\n\n\n\n##Similarcode\n" + code)

  results = list(filter(lambda x: x[1] < torch_jax_similarity_threshold, results))
  final_result = None
  if len(results) > 0:
    final_result = [(float(r[1]), names[r[2]], files[r[2]], code_texts[r[2]]) for r in results]

  if enable_cache:
    search_cache[cache_key] = {"final_result": final_result, "module_code": module_code}
    with open(torch_jax_similar_dependency_cache_file, "w") as f:
      json.dump(search_cache, f, indent=4)

  return final_result, module_code


def filter_out_dependency(code, dependency_list):
  """
  Filters a list of dependencies using an LLM agent.

  This function sends the provided code and a list of dependencies to a
  language model to determine which dependencies are actually used. It uses a
  caching mechanism to avoid redundant LLM calls.

  Args:
    code (str): The source code block to analyze.
    dependency_list (list): A list of dependency strings, e.g., ["file_path#component_name"].

  Returns:
    list: A filtered list of dependencies that are deemed necessary by the LLM.
  """
  if enable_cache:
    if os.path.exists(dependency_filter_cache_file):
      with open(dependency_filter_cache_file, "r") as f:
        cache = json.load(f)
    else:
      cache = {}

    code_hash = hashlib.sha256(code.encode()).hexdigest()
    cache_key = f"{code_hash}"

    if cache_key in cache:
      return cache[cache_key]

  prompt = Dependency_Filter_Prompt.replace("<CODE_HERE>", code).replace(
      "<DEPENDENCY_LIST_HERE> ", json.dumps(dependency_list)
  )
  llm_agent = GeminiAgent(system_instruction=Dependency_Filter_Prompt)
  response = llm_agent(prompt)
  if response is not None:
    filter_dependency_list = response.text
    result_str = filter_dependency_list.split("\n")
    result = list(filter(lambda x: "#" in x and x in dependency_list, result_str))
    if len(filter_dependency_list) > 10 and len(result) == 0:
      result = dependency_list
  else:
    result = dependency_list

  if enable_cache:
    cache[cache_key] = result
    with open(dependency_filter_cache_file, "w") as f:
      json.dump(cache, f, indent=4)

  return result


def save_status(
    q,
    processed_components,
    processed_count,
    file_analysis_cache,
    files_to_convert,
    original_dependencies,
    jax_found_depencies,
    jax_dependencies_list,
    outfile,
):
  """
  Saves the current state of the dependency analysis process.

  This is used for checkpointing, allowing the process to be resumed later
  if it's interrupted. It serializes various data structures like queues and sets
  into a JSON format.

  Args:
    q (deque): The queue of components yet to be processed.
    processed_components (set): A set of components that have been fully processed.
    processed_count (int): The total number of components processed so far.
    file_analysis_cache (dict): A cache of file analysis results.
    files_to_convert (list): A list of components identified for conversion.
    original_dependencies (dict): A mapping of components to their original dependencies.
    jax_found_depencies (defaultdict): A dictionary mapping components to their JAX dependencies.
    jax_dependencies_list (list): A list of all JAX dependencies found.
    outfile (str): The path to the output file where the status will be saved.
  """
  all_variables = {
      "q": q,
      "processed_components": processed_components,
      "processed_count": processed_count,
      "file_analysis_cache": file_analysis_cache,
      "files_to_convert": files_to_convert,
      "original_dependencies": original_dependencies,
      "jax_found_depencies": jax_found_depencies,
      "jax_dependencies_list": jax_dependencies_list,
  }
  with open(outfile, "w") as f:
    json.dump(
        {
            "data": {k: list(v) if isinstance(v, (set, deque)) else v for k, v in all_variables.items()},
            "metadata": {k: str(type(v)) for k, v in all_variables.items()},
        },
        f,
        indent=4,
    )


def load_status(file_path):
  """
  Loads a previously saved analysis state from a JSON file.

  This function deserializes the data and restores the original data structures
  (like deques and sets) to allow for the continuation of a paused analysis.

  Args:
      file_path (str): The path to the JSON file containing the saved status.

  Returns:
      list: A list of restored variables representing the analysis state.
  """
  with open(file_path) as f:
    status = json.load(f)
  data = status["data"]
  metadata = status["metadata"]
  variable_list = []
  for variable, vtype in metadata.items():
    if vtype == "<class 'collections.deque'>":
      variable_list.append(deque(data[variable]))
    elif vtype == "<class 'set'>":
      variable_list.append(set(data[variable]))
    elif vtype in ("<class 'dict'>", "<class 'int'>", "<class 'list'>"):
      variable_list.append(data[variable])
    elif vtype == "<class 'collections.defaultdict'>":
      variable_list.append(defaultdict(dict, data[variable]))
    else:
      logger.warning(f"{variable} {vtype} Not covered")
  return variable_list


# --- Main Component Dependency Analysis ---
def sort_and_search_dependency(base_path, file_path, module):
  """
  Analyzes component dependencies starting from an entry module and returns a
  topologically sorted list of all required components and their code.

  This function uses a queue-based approach to traverse the dependency graph.
  It identifies and processes all internal dependencies, filters them using an
  LLM, and searches for similar components in a database to find potential
  substitutions (e.g., JAX-compatible code).

  Args:
    base_path (str): The root URL of the source directory on GitHub.
    file_path (str): The relative path to the entry Python file.
    module (str): The name of the entry function or class to start the analysis from.
  """
  dependency_list_file = dependency_list_file_format.format(module_name=module)
  files_order_file = files_order_file_format.format(module_name=module)
  progress_status_file = progress_status_file_format.format(module_name=module)
  if save_dependency_list:
    open(dependency_list_file, "w").close()
  project_root = file_path.split("/")[0]

  if os.path.exists(progress_status_file):
    (
        q,
        processed_components,
        processed_count,
        file_analysis_cache,
        files_to_convert,
        original_dependencies,
        jax_found_dependencies,
        jax_dependencies_list,
    ) = load_status(progress_status_file)
    logger.info(f"Loaded status for {processed_count} processed")
  else:
    q = deque([(file_path, module)])
    processed_components = set()
    processed_count = 0
    file_analysis_cache = {}
    files_to_convert = []
    original_dependencies = {}
    jax_found_dependencies = defaultdict(dict)
    jax_dependencies_list = []

  while q:
    relative_file_path, comp_name = q.popleft()
    if "." in comp_name:
      comp_name, sub_comp_name = comp_name.split(".")
    else:
      sub_comp_name = None
    file_url = base_path + relative_file_path
    comp_id = f"{relative_file_path}#{comp_name}"

    if comp_id in processed_components:
      logger.info(f"{comp_id} is in processed_components")
      continue
    processed_count += 1
    logger.info(f"Processing ({processed_count}): {comp_id}  {sub_comp_name if sub_comp_name is not None else ''}")

    if comp_id not in file_analysis_cache:
      logger.info(f"---> Analyzing file: {comp_id}")
      try:
        # Uses the function from SplitPythonFile.py
        file_analysis_cache[comp_id] = get_file_components(
            file_url, module=comp_name, project_root=project_root, add_external_dependencies=True
        )
      except FileNotFoundError:
        try:
          file_url, comp_name = file_url.replace(".py", "/" + comp_name + ".py"), sub_comp_name
          file_analysis_cache[comp_id] = get_file_components(
              file_url, module=comp_name, project_root=project_root, add_external_dependencies=True
          )
        except Exception as e:
          logger.error(f"Could not analyze file with subcompnenet {file_url}. Error: {e}")
          continue  # Skip this file if it cannot be analyzed
      except Exception as e:
        logger.error(f"Could not analyze file {file_url}. Error: {e}")
        continue  # Skip this file if it cannot be analyzed
    else:
      logger.info(f"{comp_id}. already in file_analysis_cache")

    processed_comp_ids = list(map(lambda x: x["comp_id"], files_to_convert))
    analysis = copy.deepcopy(file_analysis_cache[comp_id])
    if comp_name in analysis["component_dependencies"]:
      filtered_dependencies = filter_out_dependency(
          analysis["sorted_modules"][comp_name], analysis["component_dependencies"][comp_name]
      )
      logger.info(
          f"Filter dependencies from \n {analysis['component_dependencies'][comp_name]} \nto {filtered_dependencies}"
      )
      analysis["component_dependencies"][comp_name] = filtered_dependencies
      if save_dependency_list:
        with open(dependency_list_file, "a") as f:
          f.write(f"--- Dependency for {comp_name}\n")
          f.write(
              json.dumps(list(filter(lambda x: x not in processed_comp_ids, analysis["component_dependencies"][comp_name])))
              + "\n"
          )
          f.write("\n\n\n\n\n")
      if comp_name not in original_dependencies:
        original_dependencies[comp_name] = analysis["component_dependencies"][comp_name]
    else:
      if save_dependency_list:
        with open(dependency_list_file, "a") as f:
          f.write(f"--- No Depenency for {comp_name}\n\n\n\n\n")
    analysis["component_dependencies"][comp_name] = list(
        filter(
            lambda x: x not in processed_components and x not in jax_dependencies_list,
            analysis["component_dependencies"].get(comp_name, []),
        )
    )
    if comp_name not in analysis["component_dependencies"] or len(analysis["component_dependencies"][comp_name]) == 0:
      processed_components.add(comp_id)
      files_to_convert.append(
          {
              "comp_id": comp_id,
              "filepath": relative_file_path,
              "comp_name": comp_name,
              "Dependencies": original_dependencies.get(comp_name, []),
              "JaxDependencies": jax_found_dependencies.get(comp_name, {}),
          }
      )
      logger.info(f"➔➔➔➔Added {comp_id} to Modules to convert total founded modules {len(processed_components)}")
    else:
      logger.info(f"{comp_name} from {file_url} have dependencies {analysis['component_dependencies'][comp_name]}")
      for dependency in analysis["component_dependencies"][comp_name]:
        logger.info(f"Searching for {dependency} for {comp_name}")
        dependencies, m_code = search_similar_dependency(dependency, base_path, project_root)
        if m_code is None:
          processed_components.add(dependency)
          continue
        elif dependencies is None:
          logger.info(f"Maxtext similar dependency not found for {dependency} so adding to search there dependencies")
          f_path, m_name = dependency.split("#")
          if (f_path, m_name) not in q:
            q.append((f_path, m_name))
        else:
          jax_found_dependencies[comp_name][dependency] = dependencies
          jax_dependencies_list.append(dependency)
          logger.info(f"🌟🌟Dependency Found Check for {dependency}, at {similar_block_folder}/{comp_name}")
      logger.info("Re added %s %s", relative_file_path, comp_name)
      if len(analysis["component_dependencies"][comp_name]) > 0:
        if (relative_file_path, comp_name) not in q:
          q.append((relative_file_path, comp_name))
      else:
        processed_components.add(comp_id)
        files_to_convert.append(
            {
                "comp_id": comp_id,
                "filepath": relative_file_path,
                "comp_name": comp_name,
                "Dependencies": original_dependencies.get(comp_name, []),
                "JaxDependencies": jax_found_dependencies.get(comp_name, {}),
            }
        )
    with open(files_order_file, "w") as f:
      json.dump(files_to_convert, f, indent=4)
    save_status(
        q,
        processed_components,
        processed_count,
        file_analysis_cache,
        files_to_convert,
        original_dependencies,
        jax_found_dependencies,
        jax_dependencies_list,
        progress_status_file,
    )

  logger.info(f"Total Module to convert {len(files_to_convert)}")


def arg_parser():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description="Get a hierarchical list of Python components from a GitHub repo.")
  parser.add_argument(
      "--base-path",
      type=str,
      default="https://github.com/huggingface/transformers/blob/main/src/",
      help="Root URL of the source directory on GitHub.",
  )
  parser.add_argument(
      "--entry-file-path",
      type=str,
      default="transformers/models/llama/modeling_llama.py",
      help="Full GitHub URL for the entry Python file.",
  )
  parser.add_argument(
      "--entry-module",
      type=str,
      default="LlamaForCausalLM",
      help="The name of the entry function or class to start the analysis from.",
  )
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  system_setup.setup_directories()
  args = arg_parser()
  base_path = args.base_path
  entry_file_path = args.entry_file_path
  entry_module = args.entry_module

  logger.info(f"Starting analysis from: {entry_module} in {entry_file_path}")
  logger.info("-" * 60)
  sort_and_search_dependency(base_path, entry_file_path, entry_module)
