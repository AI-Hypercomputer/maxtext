# Copyright 2025 Google LLC
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

"""Installs extra dependencies from a requirements file using uv.

This script is designed to be run to install dependencies specified in
'extra_deps_from_github.txt', which is expected to be in the same directory.
It first ensures 'uv' is installed and then uses it to install the packages
listed in the requirements file.
"""

import subprocess
import sys
from pathlib import Path


def main():
  """
  Installs extra dependencies specified in extra_deps_from_github.txt using uv.

  This script looks for 'extra_deps_from_github.txt' relative to its own location.
  It executes 'uv pip install -r <path_to_extra_deps.txt> --resolution=lowest'.
  """
  script_dir = Path(__file__).resolve().parent

  # Adjust this path if your extra_deps_from_github.txt is in a different location,
  # e.g., script_dir / "data" / "extra_deps_from_github.txt"
  extra_deps_file = script_dir / "extra_deps_from_github.txt"

  if not extra_deps_file.exists():
    print(f"Error: '{extra_deps_file}' not found.")
    print("Please ensure 'extra_deps_from_github.txt' is in the correct location relative to the script.")
    sys.exit(1)
  # Check if 'uv' is available in the environment
  try:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    print(f"Error checking uv version: {e}")
    print(f"Stderr: {e.stderr.decode()}")
    sys.exit(1)

  command = [
      sys.executable,  # Use the current Python executable's pip to ensure the correct environment
      "-m",
      "uv",
      "pip",
      "install",
      "-r",
      str(extra_deps_file),
      "--no-deps",
  ]

  print(f"Installing extra dependencies from '{extra_deps_file}' using uv...")
  print(f"Running command: {' '.join(command)}")

  try:
    # Run the command
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Extra dependencies installed successfully!")
    print("--- Output from uv ---")
    print(process.stdout)
    if process.stderr:
      print("--- Errors/Warnings from uv (if any) ---")
      print(process.stderr)
  except subprocess.CalledProcessError as e:
    print("Failed to install extra dependencies.")
    print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
    print("--- Stderr ---")
    print(e.stderr)
    print("--- Stdout ---")
    print(e.stdout)
    sys.exit(e.returncode)
  except (OSError, FileNotFoundError) as e:
    print(f"An OS-level error occurred while trying to run uv: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
