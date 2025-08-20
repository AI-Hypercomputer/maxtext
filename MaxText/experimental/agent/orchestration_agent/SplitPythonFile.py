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
This file provides functionality to analyze a Python file and break it down into
its constituent components (functions, classes, variables, and imports),
identifying internal dependencies between these components. It then provides
a topologically sorted list of these components, grouping circular dependencies
into single modules.

Example Invocation:

python SplitPythonFile.py \
  "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"
"""
import ast
import os.path
import sys
import json
from collections import defaultdict, deque
import argparse
from .Utils import get_github_file_content


class ReferenceVisitor(ast.NodeVisitor):
  """
  Traverses an AST node to find all references to a known set of names.
  This is used to detect dependencies between code blocks.
  """

  def __init__(self, defined_names):
    self.defined_names = defined_names
    self.found_dependencies = set()

  def visit_Name(self, node):
    if node.id in self.defined_names:
      self.found_dependencies.add(node.id)
    self.generic_visit(node)


class dependency_analyzer:
  """
  Analyzes a Python source file to find dependencies between all top-level
  definitions. It groups circular dependencies (Strongly Connected Components)
  into single modules for sequential processing.
  """

  def __init__(self, source_code):
    self.source_code = source_code
    self.tree = ast.parse(self.source_code)
    self.docstring = ast.get_docstring(self.tree)
    self.imports = []
    self.conditional_imports = []
    self.definitions = {}
    self.dependencies = defaultdict(set)
    # Adjacency list for graph algorithms
    self.adj = defaultdict(list)

  def analyze(self):
    """
    Performs a two-pass analysis of the source code to build a
    dependency graph.
    """
    # --- Pass 1: Categorize all top-level nodes ---
    for node in self.tree.body:
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        self.imports.append(node)
      elif isinstance(node, ast.If):
        # Check for conditional imports at top-level
        for subnode in node.body:
          if isinstance(subnode, (ast.Import, ast.ImportFrom)):
            self.conditional_imports.append((node, subnode))
      elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        self.definitions[node.name] = node
      elif isinstance(node, ast.Assign):
        for target in node.targets:
          if isinstance(target, ast.Name):
            self.definitions[target.id] = node

    # --- Pass 2: Find dependencies and build graph ---
    defined_names = set(self.definitions.keys())
    for name, node in self.definitions.items():
      visitor = ReferenceVisitor(defined_names)
      if isinstance(node, ast.Assign):
        visitor.visit(node.value)
        targets_in_assignment = {t.id for t in node.targets if isinstance(t, ast.Name)}
        self.dependencies[name] = visitor.found_dependencies - targets_in_assignment
      else:
        visitor.visit(node)
        self.dependencies[name] = visitor.found_dependencies - {name}

      # Build the adjacency list for the graph
      for dep in self.dependencies[name]:
        if dep in self.definitions:
          self.adj[dep].append(name)

  def _find_sccs(self):
    """
    Finds all Strongly Connected Components (SCCs) in the graph using
    Tarjan's algorithm. This is how we find and group cycles.
    """
    nodes = list(self.definitions.keys())
    disc = {node: -1 for node in nodes}
    low = {node: -1 for node in nodes}
    on_stack = {node: False for node in nodes}
    stack = []
    self.step = 0
    sccs = []

    def tarjan(u):
      disc[u] = low[u] = self.step
      self.step += 1
      stack.append(u)
      on_stack[u] = True

      for v in self.adj.get(u, []):
        if disc[v] == -1:
          tarjan(v)
          low[u] = min(low[u], low[v])
        elif on_stack[v]:
          low[u] = min(low[u], disc[v])

      if low[u] == disc[u]:
        scc = []
        while True:
          node = stack.pop()
          on_stack[node] = False
          scc.append(node)
          if u == node:
            break
        sccs.append(scc)

    for node in nodes:
      if disc[node] == -1:
        tarjan(node)

    return sccs

  def get_import_components(self):
    # Component 0 is always imports
    lines = self.source_code.splitlines()
    import_components = []
    conditional_import_components = []
    last_lineno = None
    extra_index = 1

    # If module-level docstring exists, prepend it
    if self.docstring:
      import_components.append(f'"""{self.docstring}"""')

    for node in sorted(self.imports, key=lambda n: n.lineno):
      curr_lineno = node.lineno

      # If there's a line gap from the previous node, collect the gap as extra
      if last_lineno is not None and curr_lineno > last_lineno + 1:
        gap_lines = lines[last_lineno : curr_lineno - 1]
        extra_text = "\n".join(gap_lines).strip()
        if extra_text:
          key = f"Extra_{extra_index}" if extra_index > 0 else "Extra"
          self.sorted_components[key] = extra_text
          extra_index += 1

      # Add current import node's source
      import_components.append(ast.get_source_segment(self.source_code, node))
      last_lineno = getattr(node, "end_lineno", node.lineno)

    # Handle conditional imports
    for _, import_node in self.conditional_imports:
      # Get the full if block source
      # Only add the import statement(s) inside the conditional, not the full if block
      conditional_import_components.append(ast.get_source_segment(self.source_code, import_node))

    # Final collected imports
    if import_components:
      self.sorted_components["imports"] = "\n".join(import_components)
    if conditional_import_components:
      self.sorted_components["conditional_imports"] = "\n\n".join(conditional_import_components)

  def get_sorted_structure(self):
    """
    Topologically sorts the definitions and returns a structured output
    where cyclic dependencies are grouped into single components.
    The sorting ensures that a component is defined only after all its
    dependencies are defined.
    """
    self.analyze()

    # 1. Find all cycles (Strongly Connected Components)
    sccs = self._find_sccs()

    # 2. Create a "condensation graph" where each SCC is a single node.
    scc_map = {node: i for i, scc in enumerate(sccs) for node in scc}
    scc_adj = defaultdict(set)
    scc_in_degree = defaultdict(int)

    for u, neighbors in self.adj.items():
      scc_u = scc_map[u]
      for v in neighbors:
        scc_v = scc_map[v]
        if scc_u != scc_v:
          if scc_v not in scc_adj[scc_u]:
            scc_adj[scc_u].add(scc_v)
            scc_in_degree[scc_v] += 1

    # 3. Topologically sort the condensation graph. This gives us the
    #    correct execution order for the components.
    queue = deque([i for i, scc in enumerate(sccs) if scc_in_degree[i] == 0])
    sorted_scc_indices = []
    while queue:
      u_scc_idx = queue.popleft()
      sorted_scc_indices.append(u_scc_idx)
      for v_scc_idx in sorted(list(scc_adj[u_scc_idx])):  # sort for determinism
        scc_in_degree[v_scc_idx] -= 1
        if scc_in_degree[v_scc_idx] == 0:
          queue.append(v_scc_idx)

    # 4. Reconstruct the modules based on the sorted components.
    self.sorted_components = {}
    comp_to_name_map = {}
    warning_message = None

    self.get_import_components()

    comp_counter = 1
    for scc_idx in sorted_scc_indices:
      scc = sorted(sccs[scc_idx])  # sort nodes within component for determinism
      is_cycle = len(scc) > 1

      # Combine source code for all nodes in the component
      component_source = []
      added_nodes = set()
      lines = self.source_code.splitlines()

      for name in scc:
        node = self.definitions[name]
        if node not in added_nodes:
          # Adjust start line to include decorators if present
          if hasattr(node, "decorator_list") and node.decorator_list:
            start_lineno = min(decorator.lineno for decorator in node.decorator_list) - 1
          else:
            start_lineno = node.lineno - 1

          end_lineno = getattr(node, "end_lineno", node.lineno)

          # Backtrack to include leading comments or blank lines
          while start_lineno > 0 and (
              lines[start_lineno - 1].strip() == "" or lines[start_lineno - 1].lstrip().startswith("#")
          ):
            start_lineno -= 1

          full_source = "\n".join(lines[start_lineno:end_lineno])
          component_source.append(full_source)
          added_nodes.add(node)
      comp_name = scc[-1]
      modulename = comp_name
      cindex = 2
      while comp_name in self.sorted_components:
        comp_name = modulename + "_" + str(cindex)
        cindex += 1
      if is_cycle:
        comp_name += "_cycle"
        warning_message = (
            "Warning: Circular dependencies were detected and grouped into single components ending in '_cycle'."
        )

      self.sorted_components[comp_name] = "\n\n".join(component_source)

      for name in scc:
        comp_to_name_map[name] = comp_name

      comp_counter += 1

    # 5. Create the dependency list between components.
    dependency_list = set()
    for dependant, deps in self.dependencies.items():
      dependant_comp = comp_to_name_map.get(dependant)
      for dependency in deps:
        dependency_comp = comp_to_name_map.get(dependency)
        if dependant_comp and dependency_comp and dependant_comp != dependency_comp:
          dependency_list.add((dependant_comp, dependency_comp))

    sorted_dependency_list = sorted(list(dependency_list))

    # 6. Create the component-wise dependency dictionary (adjacency list format).
    #    This shows which components each component depends on.
    component_dependencies = defaultdict(list)
    for dependant_comp, dependency_comp in sorted_dependency_list:
      component_dependencies[dependant_comp].append(dependency_comp)

    # Sort dependency lists for deterministic output
    for comp in component_dependencies:
      component_dependencies[comp].sort()

    return {
        "sorted_modules": self.sorted_components,
        "component_dependencies": dict(component_dependencies),
        "warning": warning_message,
    }


def get_modules_in_order(filepath):
  try:
    if filepath.startswith("https"):
      flag, source_code = get_github_file_content(filepath)
      if flag is False:
        print("Issue in reading", filepath, source_code)
    elif os.path.exists(filepath):
      with open(filepath, "r", encoding="utf-8") as f:
        source_code = f.read()
  except FileNotFoundError:
    print(f"Error: File not found at '{filepath}'")
    sys.exit(1)
  except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
  print(f"--- Analyzing '{filepath}' and creating structured output ---\n")
  analyzer = dependency_analyzer(source_code)
  return analyzer.get_sorted_structure()


def get_argparser():
  parser = argparse.ArgumentParser(description="Analyze Python file dependencies and split into components.")
  parser.add_argument(
      "filepath",
      nargs="?",
      default="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py",
      help="Path to the Python file to analyze.",
  )
  return parser


def save_results_in_file(result, filename, outFile="file_component.txt"):
  standalone_modules = [mod for mod in result["sorted_modules"].keys() if mod not in result["component_dependencies"]]
  dependent_sorted_modules = {
      mod: result["component_dependencies"][mod]
      for mod in result["sorted_modules"].keys()
      if mod in result["component_dependencies"]
  }
  with open(outFile, "w") as f:
    f.write(f"Components for {filename}\n")
    f.write(f"Standalone Modules: {json.dumps(standalone_modules)}\n")
    f.write(f"Dependent  Modules\n {json.dumps(dependent_sorted_modules,indent=4)}")


def main():
  parser = get_argparser()
  args = parser.parse_args()
  filepath = args.filepath
  result = get_modules_in_order(filepath)

  save_results_in_file(result, filename=filepath.split("/")[-1])
  print("--- Sorted Modules (Topological Order) ---")
  print(json.dumps(list(result["sorted_modules"].keys()), indent=2))

  print("\n--- Component Dependencies (Adjacency List) ---")
  print(json.dumps(result["component_dependencies"], indent=2))

  if result["warning"]:
    print("\n" + "=" * 80)
    print(result["warning"])
    print("=" * 80)


if __name__ == "__main__":
  main()
