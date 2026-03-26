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
'post_train_deps.txt', which is expected to be in the same directory.
It first ensures 'uv' is installed and then uses it to install the packages
listed in the requirements file.
"""

import os
import subprocess
import sys


def main():
  """
  Installs extra dependencies specified in post_train_deps.txt using uv.

  This script looks for 'post_train_deps.txt' relative to its own location.
  It executes 'uv pip install -r <path_to_extra_deps.txt> --resolution=lowest'.
  """
  os.environ["VLLM_TARGET_DEVICE"] = "tpu"

  # Adjust this path if your post_train_deps.txt is in a different location,
  # e.g., script_dir / "data" / "post_train_deps.txt"
  extra_deps_file = script_dir / "post_train_deps.txt"

  if not extra_deps_file.exists():
    print(f"Error: '{extra_deps_file}' not found.")
    print("Please ensure 'post_train_deps.txt' is in the correct location relative to the script.")
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
      str(extra_deps_path),
      "--no-deps",
  ]

  local_vllm_install_command = [
      sys.executable,  # Use the current Python executable's pip to ensure the correct environment
      "-m",
      "uv",
      "pip",
      "install",
      f"{repo_root}/maxtext/integration/vllm",  # MaxText on vllm installations
      "--no-deps",
  ]

  try:
    # Run the command to install Github dependencies
    print(f"Installing extra dependencies: {' '.join(command)}")
    _ = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Extra dependencies installed successfully!")

    # Run the command to install the MaxText vLLM directory
    print(f"Installing MaxText vLLM dependency: {' '.join(local_vllm_install_command)}")
    _ = subprocess.run(local_vllm_install_command, check=True, capture_output=True, text=True)
    print("MaxText vLLM dependency installed successfully!")
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
