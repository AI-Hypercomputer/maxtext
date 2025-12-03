# Copyright 2023–2025 Google LLC
# Licensed under the Apache License, Version 2.0.
"""
Topologically sorts Python component dependencies from a GitHub repo.
"""

from collections import deque, defaultdict
from typing import Any, DefaultDict, Deque, Dict, List, Set, Tuple, cast
import argparse
import copy
import hashlib
import json
import logging
import os
import sys

# --- PATCH: Force Print to verify startup ---
print("--> Script is initializing imports...", flush=True)

from MaxText.experimental.agent.integrative_rag_agent import system_setup
from MaxText.experimental.agent.integrative_rag_agent.prompts_integrative_rag import Dependency_Filter_Prompt, Dependency_Filter_Fast
from MaxText.experimental.agent.code_generation_agent.llm_agent import GeminiAgent
from MaxText.experimental.agent.integrative_rag_agent.llm_rag_embedding_generation import get_code_embedding
from MaxText.experimental.agent.integrative_rag_agent.database_operations import make_embedding_index, search_embedding
from MaxText.experimental.agent.orchestration_agent.split_python_file import get_modules_in_order_fixed as get_file_components
from MaxText.experimental.agent.integrative_rag_agent.config import (
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

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout # Force logging to stdout
)
logger = logging.getLogger(__name__)

# --- PATCH: Lazy Load Global Variables ---
# We initialize these as None and load them only when needed to prevent
# startup hangs or "silent" failures during global scope execution.
ids, names, code_texts, files, embedding_index = None, None, None, None, None


def search_similar_dependency(depend, base_path, project_root):
  """
  Searches for similar code dependencies using embedding similarity.
  """
  # --- PATCH: Load Index Here ---
  global ids, names, code_texts, files, embedding_index
  if embedding_index is None:
      print("--> ⏳ Loading embedding index (this happens once)...", flush=True)
      ids, names, code_texts, files, embedding_index = make_embedding_index()
      print("--> ✅ Embedding index loaded.", flush=True)
  # ------------------------------

  if enable_cache:
    if os.path.exists(torch_jax_similar_dependency_cache_file):
      with open(torch_jax_similar_dependency_cache_file, "rt", encoding="utf-8") as f:
        search_cache = json.load(f)
    else:
      search_cache = {}

    dep_hash = hashlib.sha256(depend.encode()).hexdigest()
    cache_key = dep_hash

    if cache_key in search_cache:
      return search_cache[cache_key]["final_result"], search_cache[cache_key]["module_code"]

  # llm call
  dep_file_path, dep_comp_name = depend.split("#")
  module_code, _, embedding = get_code_embedding(base_path + dep_file_path, project_root, dep_comp_name)
  if module_code is None:
    return None, None

  results = search_embedding(embedding, embedding_index, code_texts, top_k=3)
  logger.info("Distances %s", list(map(lambda x: x[1], results)))
  if save_most_similar_block_for_debugging:
    os.makedirs(similar_block_folder + dep_comp_name, exist_ok=True)
    for i, (code, dis, _) in enumerate(results):
      with open(f"{similar_block_folder}{dep_comp_name}{os.path.sep}{i}_{dis}.py", "wt", encoding="utf-8") as f:
        f.write("# Distance:  " + str(dis) + "\n")
        f.write("##Base_code\n" + module_code)
        f.write("\n\n\n\n##Similarcode\n" + code)

  results = list(filter(lambda x: x[1] < torch_jax_similarity_threshold, results))
  final_result = None
  if len(results) > 0:
    final_result = [(float(r[1]), names[r[2]], files[r[2]], code_texts[r[2]]) for r in results]

  if enable_cache:
    search_cache[cache_key] = {"final_result": final_result, "module_code": module_code}
    with open(torch_jax_similar_dependency_cache_file, "wt", encoding="utf-8") as f:
      json.dump(search_cache, f, indent=4)

  return final_result, module_code


def filter_out_dependency(code, dependency_list, prompt_template: str):
  """
  Filters a list of dependencies using an LLM agent.
  """
  if enable_cache:
    if os.path.exists(dependency_filter_cache_file):
      with open(dependency_filter_cache_file, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    else:
      cache = {}

    code_hash = hashlib.sha256(code.encode()).hexdigest()
    cache_key = f"{code_hash}"

    if cache_key in cache:
      return cache[cache_key]

  prompt = prompt_template.replace("<CODE_HERE>", code).replace(
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
    with open(dependency_filter_cache_file, "wt", encoding="utf-8") as f:
      json.dump(cache, f, indent=4)

  return result


Status = Tuple[
    Deque[Tuple[str, str]],
    Set[str],
    int,
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, List[str]],
    DefaultDict[str, Dict[str, Any]],
    List[str],
]

def save_status(
    q,
    processed_components,
    processed_count,
    file_analysis_cache,
    files_to_convert,
    original_dependencies,
    jax_found_dependencies,
    jax_dependencies_list,
    outfile,
):
  all_variables = {
      "q": q,
      "processed_components": processed_components,
      "processed_count": processed_count,
      "file_analysis_cache": file_analysis_cache,
      "files_to_convert": files_to_convert,
      "original_dependencies": original_dependencies,
      "jax_found_dependencies": jax_found_dependencies,
      "jax_dependencies_list": jax_dependencies_list,
  }
  with open(outfile, "wt", encoding="utf-8") as f:
    json.dump(
        {
            "data": {k: list(v) if isinstance(v, (set, deque)) else v for k, v in all_variables.items()},
            "metadata": {k: str(type(v)) for k, v in all_variables.items()},
        },
        f,
        indent=4,
    )


def load_status(file_path: str) -> Status:
  with open(file_path, "rt", encoding="utf-8") as f:
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
      logger.warning("%s %s Not covered", variable, vtype)
  return cast(Status, tuple(variable_list))


# --- Main Component Dependency Analysis ---
def sort_and_search_dependency(base_path, file_path, module, filter_mode: str = "standard"):
  # --- PATCH: Fix Paths Automatically ---
  if not base_path.endswith("/"):
      base_path += "/"
  # Remove leading slash from file path if present to avoid double-slash issues
  if file_path.startswith("/"):
      file_path = file_path[1:]
  # --------------------------------------

  dependency_list_file = dependency_list_file_format.format(module_name=module)
  files_order_file = files_order_file_format.format(module_name=module)
  progress_status_file = progress_status_file_format.format(module_name=module)
  
  if save_dependency_list:
    with open(dependency_list_file, "a", encoding="utf-8"):
      pass
  
  # Ensure we can find the project root
  project_root = file_path.split(os.path.sep)[0]

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
    print(f"--> Loaded status for {processed_count} processed components", flush=True)
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
      logger.info("%s is in processed_components", comp_id)
      continue
    processed_count += 1
    # Use print for visibility during debugging
    print(f"--> Processing ({processed_count}): {comp_id}", flush=True)
    
    if comp_id not in file_analysis_cache:
      logger.info("---> Analyzing file: %s", comp_id)
      try:
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
          logger.error("Could not analyze file with subcomponent %s. Error: %s", file_url, e)
          continue  
      except Exception as e:
        logger.error("Could not analyze file %s. Error: %s", file_url, e)
        continue  
    else:
      logger.info("%s already in file_analysis_cache", comp_id)

    processed_comp_ids = list(map(lambda x: x["comp_id"], files_to_convert))
    analysis = copy.deepcopy(file_analysis_cache[comp_id])
    if comp_name in analysis["component_dependencies"]:
      original_deps = analysis["component_dependencies"][comp_name]
      filtered_dependencies = original_deps
      if filter_mode == "standard":
        logger.info("Running 'standard' filter for %s...", comp_name)
        filtered_dependencies = filter_out_dependency(
            analysis["sorted_modules"][comp_name],
            original_deps,
            Dependency_Filter_Prompt
        )
      elif filter_mode == "aggressive":
        logger.info("Running 'aggressive' filter for %s...", comp_name)
        filtered_dependencies = filter_out_dependency(
            analysis["sorted_modules"][comp_name],
            original_deps,
            Dependency_Filter_Fast
        )
      else: # filter_mode == "none"
        pass
        
      analysis["component_dependencies"][comp_name] = filtered_dependencies
      if save_dependency_list:
        with open(dependency_list_file, "a", encoding="utf-8") as f:
          f.write(f"--- Dependency for {comp_name}\n")
          f.write(
              json.dumps(
                  list(filter(lambda x: x not in processed_comp_ids, analysis["component_dependencies"][comp_name]))
              )
              + "\n"
          )
          f.write("\n\n\n\n\n")
      if comp_name not in original_dependencies:
        original_dependencies[comp_name] = analysis["component_dependencies"][comp_name]
    else:
      if save_dependency_list:
        with open(dependency_list_file, "a", encoding="utf-8") as f:
          f.write(f"--- No Dependency for {comp_name}\n\n\n\n\n")
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
      logger.info("➔➔➔➔ Added %s to Modules to convert", comp_id)
    else:
      logger.info("%s from %s have dependencies", comp_name, file_url)
      for dependency in analysis["component_dependencies"][comp_name]:
        # Search happens here, will trigger lazy load if needed
        dependencies, m_code = search_similar_dependency(dependency, base_path, project_root)
        if m_code is None:
          processed_components.add(dependency)
          continue
        elif dependencies is None:
          logger.info("Similar dependency not found for %s, adding to queue", dependency)
          f_path, m_name = dependency.split("#")
          if (f_path, m_name) not in q:
            q.append((f_path, m_name))
        else:
          jax_found_dependencies[comp_name][dependency] = dependencies
          jax_dependencies_list.append(dependency)
          logger.info("🌟🌟 Dependency Found Check for %s", dependency)
      logger.info("Re-added %s %s", relative_file_path, comp_name)
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
    with open(files_order_file, "wt", encoding="utf-8") as f:
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

  print(f"--> ✅ Total Modules to convert: {len(files_to_convert)}", flush=True)


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
  parser.add_argument(
      "--filter-mode",
      type=str,
      default="standard",
      choices=["aggressive", "standard", "none"],
      help="Sets the dependency filter mode."
  )
  args = parser.parse_args()
  return args


def main():
  system_setup.setup_directories()
  args = arg_parser()
  base_path = args.base_path
  entry_file_path = args.entry_file_path
  entry_module = args.entry_module

  # --- PATCH: Force Print ---
  print("-" * 60, flush=True)
  print(f"--> Starting analysis from: {entry_module}", flush=True)
  print(f"--> File: {entry_file_path}", flush=True)
  print(f"--> Mode: {args.filter_mode}", flush=True)
  print("-" * 60, flush=True)
  # --------------------------

  logger.info("-" * 60)
  sort_and_search_dependency(base_path, entry_file_path, entry_module, args.filter_mode)


if __name__ == "__main__":
  main()
