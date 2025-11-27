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

python split_python_file.py \
  "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"
"""

from collections import defaultdict, deque
import argparse
import ast
import hashlib
import json
import logging
import os.path
from typing import TypedDict, cast

from MaxText.experimental.agent.orchestration_agent.utils import (
  get_github_file_content,
  get_absolute_imports,
  check_github_file_exists,
)

logger = logging.getLogger("__name__")

SortedStructure = TypedDict(
  "SortedStructure", {"sorted_modules": dict, "component_dependencies": list, "warning": str}, total=False
)

# Cache file for storing results of Python file splitting operations
# Avoids re-parsing files that have already been analyzed
enable_cache = True
split_python_cache_file = "Cache/split_python_cache.json"
if enable_cache:
  os.makedirs("Cache", exist_ok=True)


class ReferenceVisitor(ast.NodeVisitor):
  """
  Traverses an AST node to find all references to a known set of names.
  This is used to detect dependencies between code blocks.
  """

  def __init__(self, defined_names, git_dependencies=None):
    self.defined_names = defined_names
    self.found_dependencies = set()
    self.git_dependencies = git_dependencies or {}
    self.git_aliases = set(self.git_dependencies.keys())
    # Keep track of AST nodes that have already been processed as part of a larger dependency
    self._handled_nodes = set()

  def get_attribute_chain(self, node):
    """
    Recursively builds the full attribute access chain (e.g., 'a.b.c')
    starting from an ast.Attribute node.
    Returns the base name and the attribute chain as a string.
    """
    chain = []
    curr = node
    while isinstance(curr, ast.Attribute):
      chain.append(curr.attr)
      curr = curr.value
    if isinstance(curr, ast.Name):
      return curr.id, ".".join(reversed(chain))
    return None, None  # The base of the attribute access is not a simple name

  def visit_Attribute(self, node):
    """visit ast.Attribute"""
    # If this node has already been processed as part of a larger attribute chain, skip it.
    if node in self._handled_nodes:
      return

    base_name, attr_chain = self.get_attribute_chain(node)
    if base_name and base_name in self.defined_names:
      # The base of this attribute access (e.g., 'page_manager') is a known definition.

      # Check if the base is an alias for a git dependency.
      if base_name in self.git_aliases:
        # It's an external dependency. We need to format it with the attribute path.
        # Example: base_name='page_manager', attr_chain='PageState'
        # self.git_dependencies['page_manager'] might be 'src/MaxText/inference/page_manager.py#page_manager'
        path, obj = self.git_dependencies[base_name].split("#", 1)

        # As per the user request, we append the attribute access to the object name.
        # e.g., 'page_manager' becomes 'page_manager.PageState'
        new_obj = f"{obj}.{attr_chain}"
        self.found_dependencies.add(f"{path}#{new_obj}")
      else:
        # It's an internal dependency (a class, function, etc. in the same file).
        # The dependency is on the base object itself.
        self.found_dependencies.add(base_name)

      # Mark this node and all its sub-nodes (the entire attribute chain) as handled
      # to prevent visit_Name from creating a less specific, duplicate dependency.
      curr = node
      while isinstance(curr, ast.Attribute):
        self._handled_nodes.add(curr)
        curr = curr.value
      if isinstance(curr, ast.Name):
        self._handled_nodes.add(curr)
    else:
      # The base of the attribute is not a dependency we're tracking,
      # so continue traversing down the tree.
      self.generic_visit(node)

  def visit_Name(self, node):
    """Record standalone name usages that match known definitions.

    Skips names that were already accounted for as part of a larger
    attribute chain in `visit_Attribute`. Adds matching names to
    `self.found_dependencies`.
    """
    # If this node was already part of an attribute chain we handled, skip it.
    if node in self._handled_nodes:
      return

    # If the name is a standalone dependency we are tracking.
    if node.id in self.defined_names:
      # This handles direct usage of internal definitions or imported objects
      # that are not used with further attribute access.
      self.found_dependencies.add(node.id)


class DependencyAnalyzer:
  """
  Analyzes a Python source file to find dependencies between all top-level
  definitions. It groups circular dependencies (Strongly Connected Components)
  into single modules for sequential processing.
  """

  def __init__(self, file_path, project_root="transformers", add_external_dependencies=False):
    self.file_path = file_path
    self.source_code = self.get_source_code()
    self.project_root = project_root
    self.add_external_dependencies = add_external_dependencies
    self.tree = ast.parse(self.source_code)
    self.docstring = ast.get_docstring(self.tree)
    self.imports = []
    self.conditional_imports = []
    self.definitions = {}
    self.dependencies = defaultdict(set)
    self.sorted_components = {}
    self.git_dependencies = {}
    # Adjacency list for graph algorithms
    self.adj = defaultdict(list)
    root_index = len(file_path.split(os.path.sep)) - 1 - file_path.split(os.path.sep)[::-1].index(project_root)
    self.package_path = os.path.sep.join(file_path.split(os.path.sep)[root_index:])
    self.split_python_cache_file = split_python_cache_file

  def get_source_code(self):
    """Return the source code for `self.file_path`.

    If the path is an HTTPS GitHub URL, fetches via `get_github_file_content`
    and validates existence with `check_github_file_exists`. If the path is
    a local file, reads from disk.

    Returns:
        str: The file contents.

    Raises:
        FileNotFoundError: When the remote file does not exist or a local
            path is missing.
        IOError: When a remote file exists but cannot be read.
    """
    source_code = ""
    if self.file_path.startswith("https"):
      flag, source_code = get_github_file_content(self.file_path)
      if flag is False:
        exists, _ = check_github_file_exists(self.file_path)
        if exists:
          print("not able to read seems have some issue in file", self.file_path, source_code)
          raise IOError(f"Could not read remote file content: {source_code}")
        else:
          print("Unable to read file: does not exist", self.file_path)
          raise FileNotFoundError(self.file_path)
    elif os.path.exists(self.file_path):
      with open(self.file_path, "rt", encoding="utf-8") as f:
        source_code = f.read()
    return source_code

  def convert_package_to_path(self, path):
    """Convert an absolute import line to a mapping of names to file anchors.

    Example:
        "from MaxText.inference import page_manager, utils" ->
        {"page_manager": "src/MaxText/inference.py#page_manager",
         "utils": "src/MaxText/inference.py#utils"}

    Args:
        path (str): A normalized absolute import string.

    Returns:
        dict[str, str]: Mapping of imported names to "file.py#name" anchors.
    """
    path_form, path_imports = path.removeprefix("from ").replace(".", os.path.sep).split(" import ")
    import_dict = {}
    for pkg in path_imports.split(","):
      # This part might need to be smarter to distinguish module imports from object imports
      # For now, it assumes an object 'pkg' is in a file named 'path_form.py'
      # or a module 'pkg' corresponds to 'path_form/pkg.py'
      # The logic in get_absolute_imports should ideally resolve this ambiguity.
      # A heuristic could be used here (e.g., checking casing) but we stick to the current logic.
      # The user's example `from MaxText.inference import page_manager` creates a path
      # `src/MaxText/inference.py#page_manager`, which is what the new visitor expects to correct.
      import_dict[pkg.strip()] = path_form + ".py#" + pkg.strip()
    return import_dict

  def analyze(self):
    """
    Performs a two-pass analysis of the source code to build a
    dependency graph.
    """
    # --- Pass 1: Categorize all top-level nodes ---
    for node in self.tree.body:
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        self.imports.append(node)
      elif isinstance(node, (ast.If, ast.Try)):
        # Check for conditional imports at top-level
        for sub_node in node.body:
          if isinstance(sub_node, (ast.Import, ast.ImportFrom)):
            self.conditional_imports.append((node, node))
      elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        self.definitions[node.name] = node
      elif isinstance(node, ast.Assign):
        for target in node.targets:
          if isinstance(target, ast.Name):
            self.definitions[target.id] = node

    self.git_dependencies = {}
    if self.add_external_dependencies:
      for node in self.imports:
        scode = ast.get_source_segment(self.source_code, node)
        absimports = get_absolute_imports(scode, self.file_path, project_root=self.project_root)
        if absimports is not None:
          deque(
            (
              self.git_dependencies.update(self.convert_package_to_path(absimport))
              for absimport in absimports.split("\n")
              if absimport.startswith(f"from {self.project_root}")
            ),
            maxlen=0,
          )

      for node in self.conditional_imports:
        for snode in node[1].body:
          scode = ast.get_source_segment(self.source_code, snode)
          absimports = get_absolute_imports(scode, self.file_path, project_root=self.project_root)
          if absimports is not None:
            deque(
              (
                self.git_dependencies.update(self.convert_package_to_path(absimport))
                for absimport in absimports.split("\n")
                if absimport.startswith(f"from {self.project_root}")
              ),
              maxlen=0,
            )

    # --- Pass 2: Find dependencies and build graph ---
    self.defined_names = set(self.definitions.keys()).union(set(self.git_dependencies.keys()))

    for name, node in self.definitions.items():
      # Pass git_dependencies to the visitor to resolve attribute access on modules.
      visitor = ReferenceVisitor(self.defined_names, self.git_dependencies)
      if isinstance(node, ast.Assign):
        visitor.visit(node.value)
        targets_in_assignment = {t.id for t in node.targets if isinstance(t, ast.Name)}
        self.dependencies[name] = visitor.found_dependencies - targets_in_assignment
      else:
        visitor.visit(node)
        self.dependencies[name] = visitor.found_dependencies - {name}

      # Build the adjacency list for the graph
      for dep in self.dependencies[name]:
        # The dependency 'dep' can be a simple name or a full path string.
        # We only add simple names to the graph for cycle detection.
        # The full path strings are handled later.
        if dep in self.defined_names:
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
    """Collect import-related components and store them into `sorted_components`.

    Builds two special components:
      - "imports": All top-level imports, optionally preceded by the module
        docstring if present.
      - "conditional_imports": Text for import statements nested under
        top-level conditionals (e.g., if/try blocks).
    """
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

  def sort_structure(self):
    """
    Topologically sorts the definitions and returns a structured output
    where cyclic dependencies are grouped into single components.
    The sorting ensures that a component is defined only after all its
    dependencies are defined.
    """
    # actual Code
    self.analyze()

    # 1. Find all cycles (Strongly Connected Components)
    sccs = self._find_sccs()

    # 2. Create a "condensation graph" where each SCC is a single node.
    scc_map = {node: i for i, scc in enumerate(sccs) for node in scc}

    scc_adj = defaultdict(set)
    scc_in_degree = defaultdict(int)

    for u, neighbors in self.adj.items():
      if u not in scc_map:
        continue
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
    # Track aliases for cyclic components so dependencies can be duplicated
    cycle_aliases = defaultdict(list)
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

      # If this SCC is a cycle, also expose the combined component under each
      # original member name so lookups by any individual name will work.
      if is_cycle:
        for original_name in scc:
          if original_name not in self.sorted_components:
            self.sorted_components[original_name] = self.sorted_components[comp_name]
          cycle_aliases[comp_name].append(original_name)

      comp_counter += 1

    # 5. Create the dependency list between components.
    dependency_list = set()
    for dependent, deps in self.dependencies.items():
      dependant_comp = comp_to_name_map.get(dependent)
      if not dependant_comp:
        continue

      for dependency in deps:
        # Case 1: The dependency is a pre-formatted full path string (e.g., from attribute access).
        if isinstance(dependency, str) and "#" in dependency:
          dependency_list.add((dependant_comp, dependency))
          continue

        # Case 2: The dependency is another component in the same file.
        dependency_comp = comp_to_name_map.get(dependency)
        if dependency_comp and dependant_comp != dependency_comp:
          if self.add_external_dependencies:
            dependency_list.add((dependant_comp, self.package_path + "#" + dependency_comp))
          else:
            dependency_list.add((dependant_comp, dependency_comp))

        # Case 3: The dependency is an alias for an external git dependency.
        elif self.add_external_dependencies and dependency in self.git_dependencies:
          dependency_list.add((dependant_comp, self.git_dependencies[dependency]))

    sorted_dependency_list = sorted(list(dependency_list))

    # 6. Create the component-wise dependency dictionary (adjacency list format).
    #    This shows which components each component depends on.
    component_dependencies = defaultdict(list)
    for dependant_comp, dependency_comp in sorted_dependency_list:
      component_dependencies[dependant_comp].append(dependency_comp)

    # Sort dependency lists for deterministic output
    for comp in component_dependencies:
      component_dependencies[comp].sort()
    component_dependencies = dict(component_dependencies)

    # For cyclic components, duplicate dependency lists under each original
    # member name so lookups by those names work as expected.
    for combined_name, original_names in cycle_aliases.items():
      deps = component_dependencies.get(combined_name, [])
      for original_name in original_names:
        if original_name not in component_dependencies:
          component_dependencies[original_name] = list(deps)

    self.sorted_structure = cast(
      SortedStructure,
      {
        "sorted_modules": self.sorted_components,
        "component_dependencies": component_dependencies,
        "warning": warning_message,
      },
    )

  def load_cache(self):
    """Load cached analysis result if caching is enabled.

    Returns:
        tuple[str|None, dict]: A `(cache_key, search_cache)` pair. When
        caching is disabled, returns `(None, {})`.
    """
    search_cache = {}
    if enable_cache:
      if os.path.exists(self.split_python_cache_file):
        with open(self.split_python_cache_file, "rt", encoding="utf-8") as f:
          search_cache = json.load(f)

      dep_hash = hashlib.sha256(self.file_path.encode()).hexdigest()
      cache_key = dep_hash
      return cache_key, search_cache
    return None, search_cache

  def save_in_cache(self, cache_key, search_cache):
    """Persist the current `sorted_structure` under `cache_key` if enabled."""
    if enable_cache:
      search_cache[cache_key] = self.sorted_structure
      with open(self.split_python_cache_file, "wt", encoding="utf-8") as f:
        json.dump(search_cache, f, indent=4)

  def get_sorted_structure(self) -> SortedStructure:
    """Compute (or load) and return the full sorted structure for the file.

    Returns:
        dict: A dictionary with keys:
          - "sorted_modules": Mapping of component name to source code.
          - "component_dependencies": Adjacency lists by component.
          - "warning": Optional warning message about cycles.
    """
    cache_key, search_cache = self.load_cache()
    if cache_key is not None and cache_key in search_cache:
      logger.info("loading from cache")
      self.sorted_structure = search_cache[cache_key]
      self.sorted_components = self.sorted_structure["sorted_modules"]
      return self.sorted_structure
    # sort the structure
    self.sort_structure()
    self.save_in_cache(cache_key, search_cache)
    return self.sorted_structure

  def get_structure_for_module(self, module):
    """Return a structure filtered to a single component `module`."""
    return {
      "sorted_modules": {module: self.sorted_structure["sorted_modules"][module]}
      if module in self.sorted_structure["sorted_modules"]
      else {},
      "component_dependencies": {module: self.sorted_structure["component_dependencies"][module]}
      if module in self.sorted_structure["component_dependencies"]
      else {},
      "warning": self.sorted_structure["warning"],
    }

  def get_module_code(self, module_name):
    """
    Returns the source code for a given module/component name.

    Parameters:
        module_name (str): The name of the module/component to retrieve.

    Returns:
        str: The source code of the requested module/component.

    Raises:
        KeyError: If no component with the provided name exists.
    """
    # Ensure analysis has been done
    if not hasattr(self, "sorted_components"):
      self.get_sorted_structure()

    if module_name not in self.sorted_components:
      available = ", ".join(self.sorted_components.keys())
      raise KeyError(f"Module '{module_name}' not found. Available: {available}")
    return self.sorted_components[module_name]


def get_modules_from_file(file_path, module, project_root="transformers", add_external_dependencies=True):
  """Return `(module_source, full_source)` from a Python file.

  On failure, logs the error and returns `(None, None)` when the analyzer
  could not be created, otherwise `(None, full_source)`.
  """
  analyzer = None
  try:
    logger.info("--- Analyzing '%s' for %s ---\n", file_path, module)
    analyzer = DependencyAnalyzer(file_path, project_root, add_external_dependencies=add_external_dependencies)
    return analyzer.get_module_code(module), analyzer.source_code
  except (FileNotFoundError, IOError, KeyError) as e:
    logger.error("Unable to find module %s from %s due to %s", module, file_path, e)
    if analyzer is None:
      return None, None
    else:
      return None, analyzer.source_code


def get_modules_in_order(file_path, module=None, project_root="transformers", add_external_dependencies=True):
  """Return the dependency-sorted structure for `file_path`.

  If `module` is provided, returns a structure filtered to that component.
  """
  logger.info("\n--- Analyzing '%s' and creating structured output for module %s ---\n", file_path, module)
  analyzer = DependencyAnalyzer(file_path, project_root, add_external_dependencies=add_external_dependencies)
  analyzer.get_sorted_structure()
  if module is None:
    return analyzer.sorted_structure
  return analyzer.get_structure_for_module(module=module)


def parse_args():
  """
  Parses command-line arguments for file or folder processing.

  Returns:
      argparse.Namespace: The parsed command-line arguments.
  """
  parser = argparse.ArgumentParser(description="Analyze Python file dependencies and split into components.")
  parser.add_argument(
    "filepath",
    nargs="?",
    default="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py",
    help="Path to the Python file to analyze.",
  )
  parser.add_argument(
    "--project-root",
    nargs="?",
    default="transformers",
    help="Path to the Python file to analyze.",
  )
  parser.add_argument(
    "--module",
    nargs="?",
    default=None,
    help="Path to the Python file to analyze.",
  )
  return parser


def save_results_in_file(result: SortedStructure, filename, outFile="file_component.txt"):
  """Write a summary of components and dependencies to `outFile`."""
  standalone_modules = [mod for mod in result["sorted_modules"].keys() if mod not in result["component_dependencies"]]
  dependent_sorted_modules = {
    mod: result["component_dependencies"][mod]
    for mod in result["sorted_modules"].keys()
    if mod in result["component_dependencies"]
  }
  with open(outFile, "wt", encoding="utf-8") as f:
    f.write(f"Components for {filename}\n")
    f.write(f"Standalone Modules: {json.dumps(standalone_modules)}\n")
    f.write(f"Dependent  Modules\n {json.dumps(dependent_sorted_modules, indent=4)}")


def main():
  parser = parse_args()
  args = parser.parse_args()
  filepath = args.filepath
  result = get_modules_in_order(filepath)

  save_results_in_file(result, filename=filepath.split(os.path.sep)[-1])
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
