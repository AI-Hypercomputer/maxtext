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

"""Utility functions for code evaluation agent."""

import ast
import io
import os
import pytest
import re
import sys
import subprocess
import traceback
from contextlib import redirect_stdout, redirect_stderr


def get_assigned_names(node):
  """Extract target names from an assignment AST node.

  Handles simple names, attributes (e.g., self.value), tuple unpacking, and
  subscripts (e.g., arr[0]). For non-name targets, uses ast.unparse to
  reconstruct a readable string.

  Args:
    node: An ast.Assign node whose `targets` will be inspected.

  Returns:
    A list of strings representing targets in left-to-right order.
  """
  names = []
  for target in node.targets:
    match target:
      case ast.Name(id=name):  # simple variable
        names.append(name)
      case ast.Attribute() | ast.Subscript():  # e.g., self.value or arr[0]
        names.append(ast.unparse(target))
      case ast.Tuple(elts=elts):  # multiple assignment
        names.extend(elt.id for elt in elts if isinstance(elt, ast.Name))
  return names


def get_last_defined_module(code_str):
  """Returns the name of the last defined class or function in a string of Python code.

  This function parses the provided code string into an Abstract Syntax Tree (AST)
  and iterates through the top-level nodes to find the last `ast.FunctionDef` or
  `ast.ClassDef` node.

  Args:
      code_str: A string containing Python code.

  Returns:
      The name of the last defined function or class, or a syntax error message
      if the code is invalid.
  """
  try:
    tree = ast.parse(code_str)
    last_name = None

    for node in tree.body:
      if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        last_name = node.name
      elif isinstance(node, ast.Assign):
        last_name = get_assigned_names(node)[-1]

    return last_name
  except SyntaxError as e:
    return f"Syntax error in code: {e}"


def run_pytest_capture_output(test_file: str, code_folder: None | str = None) -> tuple[str, int, int, int]:
  """
  Runs a specified pytest test file as an ISOLATED SUBPROCESS and captures the output.
  
  Uses `sys.executable` to ensure it runs `pytest` with the
  exact same Python interpreter (and virtual environment) as the main script.

  Args:
      test_file: The name of the pytest file to run (e.g., "test_attention_utils.py").
      code_folder: The directory to run the test from.

  Returns:
      A tuple containing:
          - output (str): The complete stdout and stderr from the test run.
          - exit_code (int): The exit code of the pytest process (0 for success, non-zero otherwise).
          - passed (int): The number of tests that passed.
          - failed (int): The number of tests that failed.
  """
  run_directory = code_folder if code_folder is not None else "."
  
  try:
    # Use sys.executable to run pytest as a module.
    # This guarantees we use the correct virtual environment.
    command = [sys.executable, "-m", "pytest", "-q", test_file]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=run_directory
    )

    output = result.stdout + result.stderr
    exit_code = result.returncode

    # Extract passed and failed counts
    passed, failed = 0, 0
    
    # Try to find the full summary line first
    match = re.search(r"(\d+)\s+passed.*(\d+)\s+failed", output)
    if match:
      passed = int(match.group(1))
      failed = int(match.group(2))
    else:
      # If not, find them separately
      match_passed = re.search(r"(\d+)\s+passed", output)
      if match_passed:
        passed = int(match_passed.group(1))
        
      match_failed = re.search(r"(\d+)\s+failed", output)
      if match_failed:
        failed = int(match_failed.group(1))
    
    # Handle "1 error" as 1 failure if no other "failed" count is found
    if passed == 0 and failed == 0 and "error" in output:
        match_error = re.search(r"(\d+)\s+error", output)
        if match_error:
            failed = int(match_error.group(1)) # Treat errors as failures

    return output, exit_code, passed, failed

  except Exception as e:
    # Catch any unexpected errors (e.g., subprocess failed to even start)
    print(f"CRITICAL ERROR in run_pytest_capture_output: {e}")
    return str(e), -1, 0, 0
