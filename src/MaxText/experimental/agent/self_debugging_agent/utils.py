"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os.path
import py_compile


def check_code_syntax(file_path: str):
  """Checks the Python syntax of a specified file.

  This function attempts to compile the Python file at the given path.
  If the compilation is successful, it indicates that the syntax is correct.
  If an error occurs during compilation, it means there is a syntax error.

  Args:
    file_path: The path to the Python file to be checked.

  Returns:
    A tuple containing

    * An integer exit code (0 for success, 1 for error).
    * A string message indicating the result (e.g., "Syntax OK"
      or a detailed error message).
  """
  try:
    py_compile.compile(file_path, doraise=True)
    return 0, "Syntax OK"
  except py_compile.PyCompileError as e:
    return 1, f"Syntax Error:\n{e}"


def save_in_file_and_check_code_syntax(code, file_path):
  """Saves a string of code to a file and then checks its syntax.

  This utility function first writes the provided code string to a file at the
  specified path. After the file is created, it calls `check_code_syntax`
  to validate the syntax of the newly written file.

  Args:
    code: A string containing the Python code to be saved and checked.
    file_path: The path where the code should be saved.

  Returns:
    A tuple containing

    * An integer exit code (0 for success, 1 for error).
    * A string message indicating the result.
  """
  with open(file_path, "wt", encoding="utf-8") as f:
    f.write(code)
  return check_code_syntax(file_path)


def parse_json_response(response):
  """Parses a JSON dictionary from a string response.

  This function is designed to extract a JSON object embedded within a
  larger string, typically from a language model's output. It looks for
  a code block delimited by ```json and ```, and then loads the content
  of that block as a JSON dictionary.

  Args:
    Response: The string containing the JSON code block.

  Returns:
    A dictionary containing the parsed JSON data. Returns an empty
    dictionary if no JSON block is found or if parsing fails.
  """
  response_dict = {}
  if "```json" in response:
    code = response.split("```json")[1]
    if "```" in code:
      code = code.split("```")[0]
    response_dict.update(json.loads(code))
  return response_dict


def smartly_copy_code(filename, base_jax_path, base_testcase_path, dest_jax_path, dest_testcase_path):
  """Copies a JAX module and its corresponding test file to new locations.

  This function copies a specified Python file from a base JAX module
  directory to a destination JAX module directory. It also copies the
  associated test file from the base test directory to the destination
  test directory, updating the package import path inside the test file
  to reflect the new location.

  Args:
    filename: Name of the Python file to be copied.
    base_jax_path: Path to the source JAX module directory.
    base_testcase_path: Path to the source test files directory.
    dest_jax_path: Path to the destination JAX module directory.
    dest_testcase_path: Path to the destination test files directory.

  Returns:
    True if both the module file and the test file exist in the
    destination after copying, otherwise False.
  """
  base_jax_package = base_jax_path.removeprefix("../").removeprefix("./").replace(os.path.sep, ".")
  dest_jax_package = dest_jax_path.removeprefix("../").removeprefix("./").replace(os.path.sep, ".")
  if os.path.exists(base_jax_path + filename):
    with open(dest_jax_path + filename, "wt", encoding="utf-8") as fwrite:
      with open(base_jax_path + filename, "rt", encoding="utf-8") as fread:
        fwrite.write(fread.read())
  if os.path.exists(base_testcase_path + filename):
    with open(dest_testcase_path + filename, "wt", encoding="utf-8") as fwrite:
      with open(base_testcase_path + filename, "rt", encoding="utf-8") as fread:
        fwrite.write(fread.read().replace(base_jax_package, dest_jax_package))
  return os.path.exists(dest_jax_path + filename) and os.path.exists(dest_testcase_path + filename)
