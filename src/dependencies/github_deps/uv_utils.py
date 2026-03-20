# Copyright 2026 Google LLC
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

"""Helper utilities for working with uv in installation scripts."""

import os
import shutil
import subprocess
import sys


def get_uv_command():
  """
  Returns the command to run uv, either as a binary in PATH or as a module.
  Attempts to install uv via pip if not found.
  """
  # 1. Try finding 'uv' in PATH
  uv_binary = shutil.which("uv")
  if uv_binary:
    return [uv_binary]

  # 2. Try running it as a module
  try:
    subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
    return [sys.executable, "-m", "uv"]
  except (subprocess.CalledProcessError, FileNotFoundError):
    pass

  # 3. Fall back to installing via pip
  try:
    print("uv not found in PATH or as a module. Attempting to install it via pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, capture_output=True)
    # Check PATH again after installation
    uv_binary = shutil.which("uv")
    if uv_binary:
      return [uv_binary]
    return [sys.executable, "-m", "uv"]
  except subprocess.CalledProcessError as e:
    print(f"Error installing uv via pip: {e}")
    print(f"Stderr: {e.stderr.decode()}")
    sys.exit(1)


def run_install(requirements_files=None, paths=None, editable_paths=None):
  """
  Executes the appropriate uv install command (uv add or uv pip install).

  Args:
    requirements_files: List of paths to requirements.txt files.
    paths: List of paths to local packages or directories (non-editable).
    editable_paths: List of paths to local packages or directories (editable).
  """
  uv_command = get_uv_command()
  is_uv_project = os.path.exists("uv.lock")

  # We run installations in two steps if we have both standard and editable items,
  # because 'uv add --editable' cannot be mixed with non-local requirements.

  # Step 1: Standard installations
  if requirements_files or paths:
    if is_uv_project:
      cmd = uv_command + ["add", "--frozen"]
    else:
      cmd = uv_command + ["pip", "install", "--no-deps"]

    if requirements_files:
      for req in requirements_files:
        cmd.extend(["-r", str(req)])
    if paths:
      cmd.extend(paths)

    _execute_command(cmd)

  # Step 2: Editable installations
  if editable_paths:
    if is_uv_project:
      cmd = uv_command + ["add", "--frozen", "--editable"]
    else:
      cmd = uv_command + ["pip", "install", "--no-deps", "-e"]

    cmd.extend(editable_paths)
    _execute_command(cmd)


def _execute_command(cmd):
  """Helper to execute a command with logging and error handling."""
  try:
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Success!")
  except subprocess.CalledProcessError as e:
    print(f"Command failed with exit status {e.returncode}.")
    print("--- Stderr ---")
    print(e.stderr)
    print("--- Stdout ---")
    print(e.stdout)
    sys.exit(e.returncode)
  except (OSError, FileNotFoundError) as e:
    print(f"An OS-level error occurred: {e}")
    sys.exit(1)
