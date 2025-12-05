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


def run_pytest_capture_output(test_file: str, code_folder: None | str = None) -> tuple[str, int, bool, int, int]:
  """Runs a specified pytest test file and captures the output.

  This function temporarily changes the current working directory to the specified
  `code_folder` to ensure tests can find the code they need to import, then
  changes back upon completion. It uses `redirect_stdout` and `redirect_stderr`
  to capture all print statements and error messages from the test run.

  Args:
      test_file: The path to the pytest file to run.
      code_folder: The directory to change into before running the tests.

  Returns:
      A tuple containing:
          - output (str): The complete stdout and stderr from the test run.
          - exit_code (int): The exit code of the pytest process (0 for success, non-zero otherwise).
          - is_dependency_error (bool): True if a common dependency error was found in the output.
          - passed (int): The number of tests that passed.
          - failed (int): The number of tests that failed.
  """
  current_path = os.path.abspath(".")
  try:
    if code_folder is not None:
      os.chdir(code_folder)
    buffer = io.StringIO()
    if os.path.abspath("../") not in sys.path:
      sys.path.append(os.path.abspath("../"))
    with redirect_stdout(buffer), redirect_stderr(buffer):
      exit_code = pytest.main(["-q", test_file])

    output = buffer.getvalue()
    buffer.close()

    # Check for common dependency errors
    dependency_error_keywords = [
        "ModuleNotFoundError",
        "ImportError",
        "No module named",
        "pkg_resources.DistributionNotFound",
        "cannot import name",
    ]

    is_dependency_error = any(err in output for err in dependency_error_keywords)

    # Extract passed and failed counts using regex
    passed, failed = 0, 0
    match = re.search(r"(\d+) passed.*?(\d+) failed", output)
    if match:
      passed = int(match.group(1))
      failed = int(match.group(2))
    else:
      match = re.search(r"(\d+) passed", output)
      if match:
        passed = int(match.group(1))
      match = re.search(r"(\d+) failed", output)
      if match:
        failed = int(match.group(1))

    return output, exit_code, is_dependency_error, passed, failed

  except FileNotFoundError as e:
    print(f"\033[91m[ERROR] File not found:\033[0m {e.filename}", file=sys.stderr)
    return "", 1, False, 0, 0

  except IsADirectoryError as e:
    print(f"\033[91m[ERROR] Expected a file but got a directory:\033[0m {e.filename}", file=sys.stderr)
    return "", 1, False, 0, 0

  except (NotADirectoryError, PermissionError) as e:
    print(f"\033[91m[ERROR] OS error accessing path:\033[0m {e}", file=sys.stderr)
    return "", 1, False, 0, 0

  except (ValueError, TypeError, AttributeError) as e:
    # Catches potential errors during output parsing or other logic.
    error_message = "\033[91m[ERROR] An unexpected internal error occurred while processing test results:\033[0m\n"
    error_message += f"{type(e).__name__}: {str(e)}\n"
    error_message += "\n\033[93mTraceback:\033[0m\n"
    error_message += "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print(error_message, file=sys.stderr)
    return "", 1, False, 0, 0

  finally:
    os.chdir(current_path)
