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
This file is intended for creating test PyTorch files to evaluate the code
conversion agent. These test files can also be used to further tune the
prompts for the agent if needed in the future. It is not designed for use
in the actual code conversion pipeline but rather to generate standalone
PyTorch modules that can be converted to JAX using the Gemini model.

This file provides functionality to analyze a Python file and extract
standalone PyTorch modules (classes or functions) from it. It identifies
modules that do not have internal dependencies on other components within the
same file and are related to PyTorch (e.g., inherit from `nn.Module` or use
`torch` in annotations/body).

The extracted modules are saved as individual Python files in a specified
output directory.

Example Invocation:

python make_pytorch_file.py \
  --entry_file_path "transformers/models/llama/modeling_llama.py" \
  --base_path "https://github.com/huggingface/transformers/blob/main/src" 

Make sure to set the `output_dir` variable to your desired output directory.
"""
import os.path
import ast, argparse
import os, sys

# Add parent directory to path to allow imports from orchestration_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration_agent.SplitPythonFile import get_modules_in_order
from orchestration_agent.GetFilesInHierarchicalOrder import get_dependency_sorted_files
from orchestration_agent.Utils import remove_local_imports, get_github_file_content

BASE_PATH = "https://github.com/huggingface/transformers/blob/main/src"
outFolder = "dataset/PyTorch"  # Specify your output folder here, e.g., "dataset/PyTorch"
os.makedirs(outFolder, exist_ok=True)


def is_torch_function_or_class(node):
  """
  Checks if an AST node represents a PyTorch-related function or class.

  This is determined by:
  - A class inheriting from 'nn.Module'.
  - A function having 'torch' in its annotations.
  - A function body containing references to 'torch'.

  Args:
      node: An AST node (ast.FunctionDef or ast.ClassDef).

  Returns:
      bool: True if the node is a PyTorch-related function or class,
            False otherwise.
  """
  if isinstance(node, ast.FunctionDef):
    # Look for 'torch' in annotations or function body
    for arg in node.args.args + ([node.returns] if node.returns else []):
      annotation = getattr(arg, "annotation", None)
      if annotation and "torch" in ast.unparse(annotation):
        return True

    for sub_node in ast.walk(node):
      if isinstance(sub_node, ast.Attribute) and "torch" in ast.unparse(sub_node):
        return True
    return False

  elif isinstance(node, ast.ClassDef):
    # Check for inheritance from 'nn.Module'
    for base in node.bases:
      # Using ast.unparse for more reliable base class name retrieval
      base_str = ast.unparse(base) if hasattr(ast, "unparse") else getattr(base, "id", "")
      if "nn.Module" in base_str or base_str.endswith("Module"):
        return True
  return False


def file_uses_torch(tree):
  """
  Checks if a file's AST contains any top-level imports of the 'torch' module.

  Args:
      tree: The AST of the entire file.

  Returns:
      bool: True if 'torch' is imported, False otherwise.
  """
  for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
      module = getattr(node, "module", "")
      if module and module.startswith("torch"):
        return True
      if any(name.name.startswith("torch") for name in getattr(node, "names", [])):
        return True
  return False


def has_external_dependencies(code, removed_names=None, local_components=None):
  """
  Checks if the given code depends on any names from a list of removed
  or local components.

  This is used to ensure a module is truly standalone by verifying it
  doesn't reference other components from the same file that were
  filtered out.

  Args:
      code (str): The source code of the component to check.
      removed_names (list, optional): A list of names (functions, classes)
                                      that were removed by `remove_local_imports`.
                                      Defaults to None.
      local_components (list, optional): A list of names of other components
                                         in the same file. Defaults to None.

  Returns:
      bool: True if a dependency on a removed or local name is found,
            False otherwise.
  """
  if not removed_names and not local_components:
    return False

  # Normalize the removed names to a set for efficient lookup
  if isinstance(removed_names, str):
    removed_names = set(removed_names.splitlines())
  else:
    removed_names = set(removed_names)

  # Add other local components to the set of names to check
  if local_components:
    for comp in local_components:
      if comp:
        removed_names.add(comp)

  if not removed_names:
    return False

  # Now check if any of these names are used in the code's AST
  tree = ast.parse(code)
  for node in ast.walk(tree):
    # Check for direct name usage (e.g., `MyClass`)
    if isinstance(node, ast.Name):
      if node.id in removed_names:
        return True
    # Check for attribute access on a name (e.g., `MyClass.method`)
    elif isinstance(node, ast.Attribute):
      if isinstance(node.value, ast.Name) and node.value.id in removed_names:
        return True
  return False


def extract_python_independent_modules(filepath, base_path):
  """
  Analyzes a single Python file to extract and save standalone PyTorch modules.

  This function performs the following steps:
  1. Gets the sorted components of the file using `get_modules_in_order`.
  2. Filters out non-standalone components (those with dependencies).
  3. For the remaining standalone components, it checks if they are
     PyTorch-related (e.g., a class inheriting from `nn.Module`).
  4. It ensures the component doesn't depend on any local imports or other
     components from the same file.
  5. Saves the valid, standalone PyTorch modules to the output folder.

  Args:
      filepath (str): The relative path to the Python file.
      base_path (str): The base URL or path to the source repository.
  """
  full_path = base_path + filepath
  file_name = os.path.splitext(os.path.basename(filepath))[0]

  # Analyze the file to get components and dependencies
  analysis_result = get_modules_in_order(full_path)

  # Get the original source code to parse global imports
  _, source_code = get_github_file_content(full_path)
  tree = ast.parse(source_code)

  # Separate global and local imports
  global_imports, removed_imports = remove_local_imports(analysis_result["sorted_modules"].get("imports", ""), filepath)
  conditional_imports, _ = remove_local_imports(analysis_result["sorted_modules"].get("conditional_imports", ""), filepath)

  all_global_imports = global_imports + "\n" + conditional_imports

  standalone_modules = []

  for comp_name, code in analysis_result["sorted_modules"].items():
    # Exclude special components like imports
    if comp_name == "imports" or comp_name == "conditional_imports" or comp_name.startswith("Extra"):
      continue
    dependencies = analysis_result["component_dependencies"].get(comp_name, [])
    if dependencies:
      continue  # Not a standalone component
    other_local_components = [
        codename for compname, codename in analysis_result["sorted_modules"].items() if compname != comp_name
    ]
    # Parse the component's code to check its type and dependencies
    comp_tree = ast.parse(code)

    # Check if the component is a PyTorch-related class or function
    for node in comp_tree.body:
      if is_torch_function_or_class(node):
        # Ensure the file itself imports torch and the component has no local dependencies
        if file_uses_torch(tree):
          if not has_external_dependencies(code, removed_imports, other_local_components):
            # This is a valid standalone PyTorch module
            module_name = node.name if hasattr(node, "name") else comp_name
            standalone_modules.append((comp_name, module_name))

            # Save the component to a file
            output_filename = f"{file_name}__{module_name}.py"
            output_path = os.path.join(outFolder, output_filename)

            if os.path.exists(output_path):
              print(f"Overwriting {output_path}")

            with open(output_path, "w", encoding="utf-8") as f:
              # Prepend warning and imports if they exist
              if analysis_result.get("warning"):
                f.write(f"# {analysis_result['warning']}\n\n")
              if all_global_imports.strip():
                f.write(all_global_imports.strip() + "\n\n")
              f.write(code.strip() + "\n")

            print(f"✅ Saved: {output_path}")

  print("\n--- Standalone PyTorch Modules Extracted ---")
  if standalone_modules:
    for comp, name in standalone_modules:
      print(f"- {name} ({comp}) from {filepath}")
  else:
    print("No standalone PyTorch modules found.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract standalone PyTorch modules from a codebase.")
  parser.add_argument(
      "--entry_file_path",
      type=str,
      default="transformers/models/llama4/modeling_llama4.py",
      help="Relative path to the entry Python file.",
  )
  parser.add_argument("--base_path", type=str, default=BASE_PATH, help="Base directory containing the source files.")
  parser.add_argument(
      "--exclude_conditional_imports",
      action="store_true",
      default=True,
      help="Whether to exclude conditional imports when sorting dependencies.",
  )

  args = parser.parse_args()

  # The get_dependency_sorted_files function expects a base path without the file path part
  entry_file_url = os.path.join(args.base_path, args.entry_file_path)

  # Split the base path and entry file for correct dependency sorting
  base_url_for_dependency = args.base_path

  sorted_files = get_dependency_sorted_files(entry_file_url, base_url_for_dependency, args.exclude_conditional_imports)

  # Process each file in the dependency-sorted order
  for file in sorted_files:
    extract_python_independent_modules(file, args.base_path)
