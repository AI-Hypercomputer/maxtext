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

import pdb
import os
import requests
import tempfile

from typing import Tuple, Optional, List, Dict, Any

from MaxText.experimental.agent.orchestration_agent.utils import get_github_file_content, get_absolute_imports, check_github_file_exists

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

COMMON_LIBRARIES = {
  "torch", "torchvision", "numpy", "scipy", "jax", "flax", "tensorflow",
  "math", "os", "sys", "collections", "typing", "argparse", "time", "json",
  "logging", "copy", "hashlib", "pdb", "tempfile", "requests", "abc", "enum",
  "pathlib", "re", "subprocess", "warnings", "pil", "cv2", "matplotlib", 
  "dataclasses", "functools", "itertools", "random", "shutil", "unittest",
  "inspect", "pickle", "ast", "types", "contextlib"
}


class ReferenceVisitor(ast.NodeVisitor):
    """
    Traverses an AST node to find all references to a known set of names,
    including base classes and type hints.
    """

    def __init__(self, defined_names, git_dependencies=None):
        self.defined_names = defined_names
        self.found_dependencies = set()
        self.git_dependencies = git_dependencies or {}
        self.git_aliases = set(self.git_dependencies.keys())
        self._handled_nodes = set()
        self._annotation_names = set()

    def _add_dependency(self, name):
        """Helper to add a dependency if it's one we are tracking."""
        if name in self.defined_names:
            if name in self.git_aliases:
                 self.found_dependencies.add(self.git_dependencies[name])
            else:
                 self.found_dependencies.add(name)

    def _extract_names_from_annotation(self, annotation_node):
        """Recursively find all ast.Name nodes within a type annotation."""
        if isinstance(annotation_node, ast.Name):
            self._add_dependency(annotation_node.id)
            self._annotation_names.add(annotation_node.id) 
        elif isinstance(annotation_node, ast.Subscript):
            self._extract_names_from_annotation(annotation_node.value)
            if isinstance(annotation_node.slice, ast.Tuple):
                for elt in annotation_node.slice.elts:
                    self._extract_names_from_annotation(elt)
            else:
                 self._extract_names_from_annotation(annotation_node.slice)
        elif isinstance(annotation_node, ast.Constant) or annotation_node is None:
             pass 
        elif isinstance(annotation_node, ast.Attribute):
             base_name, _ = self.get_attribute_chain(annotation_node)
             if base_name:
                 self._add_dependency(base_name)
                 self._annotation_names.add(base_name)

    def visit_ClassDef(self, node):
        """Visit Class Definitions to find base classes."""
        for base in node.bases:
            if isinstance(base, ast.Name):
                self._add_dependency(base.id)
            elif isinstance(base, ast.Attribute):
                 base_name, _ = self.get_attribute_chain(base)
                 if base_name:
                      self._add_dependency(base_name)

        # CRITICAL: Continue visiting the *body* of the class
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Visit Function Definitions to find type hints in args and return."""
        for arg in node.args.args:
            if arg.annotation:
                self._extract_names_from_annotation(arg.annotation)
        for arg in node.args.posonlyargs:
             if arg.annotation:
                  self._extract_names_from_annotation(arg.annotation)
        for arg in node.args.kwonlyargs:
             if arg.annotation:
                  self._extract_names_from_annotation(arg.annotation)
        if node.args.vararg and node.args.vararg.annotation:
             self._extract_names_from_annotation(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
             self._extract_names_from_annotation(node.args.kwarg.annotation)

        if node.returns:
            self._extract_names_from_annotation(node.returns)

        # CRITICAL: Continue visiting the *body* of the function
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        """Visit Annotated Assignments (typed variables)."""
        if node.annotation:
            self._extract_names_from_annotation(node.annotation)
        
        # CRITICAL: Continue visiting the value being assigned (if any)
        if node.value:
            self.visit(node.value)

    def get_attribute_chain(self, node):
        chain = []
        curr = node
        while isinstance(curr, ast.Attribute):
            chain.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            return curr.id, ".".join(reversed(chain))
        return None, None

    def visit_Attribute(self, node):
        if node in self._handled_nodes:
            return

        base_name, attr_chain = self.get_attribute_chain(node)
        
        if base_name and base_name in self.defined_names:
            if base_name in self.git_aliases:
                path, obj = self.git_dependencies[base_name].split("#", 1)
                new_obj = f"{obj}.{attr_chain}" if attr_chain else obj 
                self.found_dependencies.add(f"{path}#{new_obj}")
            else:
                 self.found_dependencies.add(base_name)

            curr = node
            while isinstance(curr, ast.Attribute):
                self._handled_nodes.add(curr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                self._handled_nodes.add(curr)
        else:
            self.generic_visit(node)


    def visit_Name(self, node):
        if node in self._handled_nodes:
            return
        if node.id in self._annotation_names:
             return

        if node.id in self.defined_names:
            if node.id in self.git_aliases:
                 self.found_dependencies.add(self.git_dependencies[node.id])
            else:
                 self.found_dependencies.add(node.id)


class DependencyAnalyzer:
  """
  Analyzes a Python source file to find dependencies between all top-level
  definitions. It groups circular dependencies (Strongly Connected Components)
  into single modules for sequential processing.
  """

  def __init__(self, file_path, project_root="transformers", add_external_dependencies=False, original_path: Optional[str] = None):
      self.file_path = file_path 
      self.source_code = self.get_source_code() 
      self.project_root = project_root
      self.add_external_dependencies = add_external_dependencies

      if not self.source_code:
          raise SyntaxError(f"Source code is empty for file: {self.file_path}")

      self.tree = ast.parse(self.source_code)
      self.docstring = ast.get_docstring(self.tree)
      self.imports = []
      self.conditional_imports = []
      self.definitions = {}
      self.dependencies = defaultdict(set)
      self.git_dependencies = {}
      self.adj = defaultdict(list)
      
      path_for_logic = original_path if original_path is not None else self.file_path
      
      logical_path = path_for_logic.replace(os.path.sep, '/') 
      
      try:
          path_parts = logical_path.split('/')
          
          if not self.project_root:
            self.package_path = logical_path
          else:
              root_index = len(path_parts) - 1 - path_parts[::-1].index(self.project_root)
              self.package_path = "/".join(path_parts[root_index:])
      except (IndexError, ValueError):
          print(f"Warning: Could not find project_root '{self.project_root}' in path '{logical_path}'. Using fallback path.")
          self.package_path = logical_path.split('/')[-1]

      self.split_python_cache_file = split_python_cache_file

  def get_source_code(self):
      """
      Return the source code for `self.file_path`.
      Assumes `self.file_path` is a valid, local file.
      """

      if not os.path.exists(self.file_path):
          raise FileNotFoundError(f"DependencyAnalyzer: Local file does not exist: {self.file_path}")
      
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
    import_dict = {}

    # CASE 1: "from X import Y"
    if " import " in path:
      try:
        path_form, path_imports = path.removeprefix("from ").replace(".", os.path.sep).split(" import ")
        for pkg in path_imports.split(","):
          import_dict[pkg.strip()] = path_form + ".py#" + pkg.strip()
      except ValueError:
        # Fallback if splitting fails unexpectedly
        pass
      
    elif path.startswith("import "):
        # e.g., "import util.misc"
        pkg = path.replace("import ", "").strip()
        # Convert dots to slashes for the file path: "util.misc" -> "util/misc.py"
        file_path = pkg.replace(".", os.path.sep) + ".py"
        # Map the top-level name. e.g. "util" -> "util/misc.py#util" 
        # (This is a heuristic; strict mapping requires more context, but this prevents crashes)
        import_dict[pkg] = f"{file_path}#{pkg}"
        
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
          
          if isinstance(node, ast.ImportFrom) and node.level > 0:
              base_dir = os.path.dirname(self.package_path)
              go_up_dirs = "../" * (node.level - 1)
              module_file_part = node.module.replace('.', '/') if node.module else ""
              if module_file_part:
                  combined_path = os.path.join(base_dir, go_up_dirs, module_file_part + ".py")
              else:
                  combined_path = os.path.join(base_dir, go_up_dirs, "__init__.py")
              new_file_path = os.path.normpath(combined_path)
              for alias in node.names:
                  imported_name = alias.name
                  self.git_dependencies[imported_name] = f"{new_file_path}#{imported_name}"
          
          absimports = get_absolute_imports(scode, self.file_path, project_root=self.project_root)
          if absimports is not None:
            for absimport in absimports.split("\n"):
              if not self.project_root or absimport.startswith("from " + self.project_root):
                self.git_dependencies.update(self.convert_package_to_path(absimport))

        for node in self.conditional_imports:
          for snode in node[1].body:
            scode = ast.get_source_segment(self.source_code, snode)
            absimports = get_absolute_imports(scode, self.file_path, project_root=self.project_root)
            if absimports is not None:
              for absimport in absimports.split("\n"):
                if absimport.startswith("from " + self.project_root):
                  self.git_dependencies.update(self.convert_package_to_path(absimport))

      # --- Pass 2: Find dependencies and build graph ---
      self.defined_names = set(self.definitions.keys()).union(set(self.git_dependencies.keys()))

      for name, node in self.definitions.items():
        visitor = ReferenceVisitor(self.defined_names, self.git_dependencies)
        if isinstance(node, ast.Assign):
          visitor.visit(node.value)
          targets_in_assignment = {t.id for t in node.targets if isinstance(t, ast.Name)}
          self.dependencies[name] = visitor.found_dependencies - targets_in_assignment
        else:
          visitor.visit(node)
          self.dependencies[name] = visitor.found_dependencies - {name}
        for dep in self.dependencies[name]:
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
    lines = self.source_code.splitlines()
    import_components = []
    conditional_import_components = []
    last_lineno = None
    extra_index = 1

    if self.docstring:
      import_components.append(f'"""{self.docstring}"""')

    for node in sorted(self.imports, key=lambda n: n.lineno):
      curr_lineno = node.lineno

      if last_lineno is not None and curr_lineno > last_lineno + 1:
        gap_lines = lines[last_lineno : curr_lineno - 1]
        extra_text = "\n".join(gap_lines).strip()
        if extra_text:
          key = f"Extra_{extra_index}" if extra_index > 0 else "Extra"
          self.sorted_components[key] = extra_text
          extra_index += 1

      import_components.append(ast.get_source_segment(self.source_code, node))
      last_lineno = getattr(node, "end_lineno", node.lineno)

    for _, import_node in self.conditional_imports:
      conditional_import_components.append(ast.get_source_segment(self.source_code, import_node))

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
  print('+------ get modules from file ----------+')
  analyzer = None
  try:
    logger.info("--- Analyzing '%s' for %s ---\n", file_path, module)
    analyzer = DependencyAnalyzer(file_path, project_root, add_external_dependencies=add_external_dependencies)
    return analyzer.get_module_code(module), analyzer.source_code
  except Exception as e:
    logger.error("Unable to find module %s from %s due to %s", module, file_path, e)
    if analyzer is None:
      return None, None
    else:
      return None, analyzer.source_code


def get_modules_from_file_ast_fixed(file_url: str, module_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    A robust function to fetch a Python file from a URL, find a specific
    component, and return its source code along with the full file content.
    Uses Python's AST for reliable parsing of classes, functions, and type aliases.
    """
    logger.info("AST_FIX: Analyzing '%s' for module '%s'", file_url, module_name)
    full_file_code = None
    
    if os.path.exists(file_url):
        # Case 1: Local File
        try:
            with open(file_url, "r", encoding="utf-8") as f:
                full_file_code = f.read()
        except Exception as e:
            logger.error("AST_FIX: Failed to read local file %s. Error: %s", file_url, e)
            return None, None

    elif file_url.startswith(("http://", "https://")):
        # Case 2: Remote URL
        try:
            raw_url = file_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_url)
            response.raise_for_status()
            full_file_code = response.text
        except requests.exceptions.RequestException as e:
            logger.error("AST_FIX: Failed to download %s. Error: %s", file_url, e)
            # Try as a package (__init__.py) as a fallback
            try:
                if file_url.endswith(".py"):
                    init_url = file_url.replace(".py", "/__init__.py")
                else:
                    init_url = file_url.rstrip('/') + "/__init__.py"
                
                raw_init_url = init_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                response = requests.get(raw_init_url)
                response.raise_for_status()
                full_file_code = response.text
            except requests.exceptions.RequestException:
                 logger.error("AST_FIX: Could not download %s or its __init__.py", file_url)
                 return None, None
    else:
        logger.error("AST_FIX: File not found locally and invalid URL scheme: %s", file_url)
        return None, None

    if not full_file_code:
        return None, None

    try:
        tree = ast.parse(full_file_code)
        target_node = None

        # Walk the tree to find the specific class, function, or assignment node
        for node in ast.walk(tree):
            node_name = ""
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == module_name:
                target_node = node
                break
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == module_name:
                target_node = node
                break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == module_name:
                target_node = node
                break
        
        if target_node:
            module_code = ast.get_source_segment(full_file_code, target_node)
            logger.info("AST_FIX: Successfully found and extracted module '%s'.", module_name)
            return module_code, full_file_code
        else:
            logger.error("AST_FIX: Unable to find module '%s' in file %s.", module_name, file_url)
            return None, full_file_code

    except SyntaxError as e:
        logger.error("AST_FIX: Syntax error parsing file %s: %s", file_url, e)
        return None, None

def _download_file_content(file_url: str) -> Tuple[Optional[str], str]:
    """
    Downloads content from a URL, handling GitHub raw URL conversion and init fallback.
    
    Returns:
        Tuple[Optional[str], str]: (file_content, actual_url_downloaded)
    """
    try:
        raw_url = file_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text, file_url # Success, return original URL
    except requests.exceptions.RequestException as e:
        logger.info(f"Failed to download {file_url}. Trying __init__.py fallback. Error: {e}")
        # Try as a package (__init__.py) as a fallback
        try:
            if file_url.endswith(".py"):
                init_url = file_url.replace(".py", "/__init__.py")
            else:
                init_url = file_url.rstrip('/') + "/__init__.py"
                
            raw_init_url = init_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_init_url)
            response.raise_for_status()
            logger.info(f"Successfully downloaded __init__.py from: {init_url}")
            return response.text, init_url # SUCCESS, return the NEW __init__-based URL
        except requests.exceptions.RequestException:
             raise FileNotFoundError(f"Could not download {file_url} or its __init__.py")
           
def get_modules_from_file_robust(
    file_path: str, 
    module: str, 
    project_root: str = "transformers", 
    add_external_dependencies: bool = True
) -> Tuple[Optional[str], Optional[str]]:
  
  # ... (logger info) ...
  
  analyzer = None
  full_file_code: Optional[str] = None
  temp_file_path: Optional[str] = None
  
  try:
    # 1. Determine paths
    if file_path.startswith(('http://', 'https://', 'github.com')):
      file_name_part = file_path.split('/')[-1].replace('.py', '')
      if file_name_part in COMMON_LIBRARIES:
          logger.warning(f"ROBUST_FETCH: Skipping likely library file: {file_path}")
          return None, None
      
      full_file_code, actual_url_downloaded = _download_file_content(file_path) 

      with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp:
        tmp.write(full_file_code)
        temp_file_path = tmp.name
      
      analyzer_path = temp_file_path             
      
      original_path_for_analyzer = actual_url_downloaded 
      
    elif os.path.exists(file_path):
      # ... (local file logic is fine) ...
      with open(file_path, "rt", encoding="utf-8") as f:
        full_file_code = f.read()
      analyzer_path = file_path
      original_path_for_analyzer = file_path 
    else:
      raise FileNotFoundError(f"File not found locally and is not a URL: {file_path}")

    # 3. Initialize the analyzer with the CORRECT paths
    analyzer = DependencyAnalyzer(
        analyzer_path, 
        project_root, 
        add_external_dependencies=add_external_dependencies,
        original_path=original_path_for_analyzer # This is now the CORRECT path
    )
    
    # ... (rest of the function) ...
    
    # 4. Get the specific module code.
    module_code = analyzer.get_module_code(module)
    
    return module_code, full_file_code

  except FileNotFoundError as e:
    logger.warning("ROBUST_FETCH: Skipping %s, likely a library file. Message: %s", file_path, e)
    return None, None
  except Exception as e:
    logger.info("ROBUST_FETCH: Module '%s' is not defined in %s (it is likely imported). Error: %s", module, file_path, e)
    return None, full_file_code
  finally:
    if temp_file_path and os.path.exists(temp_file_path):
      os.remove(temp_file_path)


def get_modules_in_order_fixed(file_path: str, module: Optional[str] = None, project_root: str = "transformers", add_external_dependencies: bool = True) -> Optional[List[Dict[str, Any]]]:
  """
  Return the dependency-sorted structure for the file at `file_path`.
  Robustly handles remote URLs by downloading content before analysis.
  
  If `module` is provided, returns a structure filtered to that component.
  """
  logger.info("\n--- ROBUST get_modules_in_order: Analyzing '%s' for module %s ---\n", file_path, module)
  temp_file_path: Optional[str] = None
  try:
    if file_path.startswith(('http://', 'https://', 'github.com')):
      full_file_code, actual_url_downloaded = _download_file_content(file_path) 
      with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp:
        tmp.write(full_file_code)
        temp_file_path = tmp.name
      analyzer_path = temp_file_path
      original_path_for_analyzer = actual_url_downloaded
    elif os.path.exists(file_path):
      # Local file logic (this is fine)
      analyzer_path = file_path
      original_path_for_analyzer = file_path
    else:
      raise FileNotFoundError(f"File not found locally and is not a URL: {file_path}")
    # 2. Proceed with analysis
    analyzer = DependencyAnalyzer(
        analyzer_path, 
        project_root, 
        add_external_dependencies=add_external_dependencies,
        original_path=original_path_for_analyzer # This is now the CORRECT path
    )
    analyzer.get_sorted_structure()
    # 3. Return the sorted structure
    if module is None:
      return analyzer.sorted_structure
    return analyzer.get_structure_for_module(module=module)

  except FileNotFoundError as e:
    logger.error("ROBUST: Analysis failed because file could not be found or downloaded: %s", e)
    return None
  except Exception as e:
    logger.error("ROBUST: An unexpected error occurred during dependency analysis for %s: %s", file_path, e)
    return None
  finally:
    # 4. Clean up the temporary file
    if temp_file_path and os.path.exists(temp_file_path):
      os.remove(temp_file_path)
      
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
    f.write(f"Dependent  Modules\n {json.dumps(dependent_sorted_modules,indent=4)}")


def main():
  parser = parse_args()
  args = parser.parse_args()
  filepath = args.filepath
  result = get_modules_in_order_fixed(filepath, module=args.module, project_root=args.project_root)

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
