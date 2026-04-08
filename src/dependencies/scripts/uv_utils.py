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


def _get_uv_command():
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


def install_requirements(requirements_files):
  """Installs packages from requirements files using uv."""
  if not requirements_files:
    return

  uv_command = _get_uv_command()
  is_uv_project = os.path.exists("uv.lock")

  if is_uv_project:
    cmd = uv_command + ["add", "--frozen"]
  else:
    cmd = uv_command + ["pip", "install", "--no-deps"]

  for req in requirements_files:
    cmd.extend(["-r", str(req)])

  _execute_command(cmd)


def install_editable(paths):
  """Installs local packages in editable mode using uv."""
  if not paths:
    return

  uv_command = _get_uv_command()
  is_uv_project = os.path.exists("uv.lock")

  if is_uv_project:
    cmd = uv_command + ["add", "--frozen", "--editable"]
  else:
    cmd = uv_command + ["pip", "install", "--no-deps", "-e"]

  cmd.extend(paths)

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
