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

This script is designed to install dependencies specified in 'dependencies/extra_deps/pre_train_*.txt'
and optional TensorFlow dependencies from 'dependencies/extra_deps/tf_requirements.txt'.
It first ensures 'uv' is installed and then uses it to install the packages listed in the requirements file.
"""

import argparse
import os
import subprocess
import sys


def parse_args():
  parser = argparse.ArgumentParser(description="Install pre-training extra dependencies.")
  parser.add_argument(
      "--with-tf",
      action="store_true",
      default=os.getenv("WITH_TF", "").lower() in ("true", "1", "yes"),
      help="Install optional TensorFlow, TFDS, SeqIO, and JetStream dependencies.",
  )
  return parser.parse_args()


def main():
  """Installs extra dependencies specified in 'dependencies/extra_deps/pre_train_*.txt' using uv."""
  args = parse_args()
  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  github_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "pre_train_github_deps.txt")
  if not os.path.exists(github_deps_path):
    raise FileNotFoundError(f"Github dependencies file not found at {github_deps_path}")

  # Check if 'uv' is available in the environment
  try:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    print(f"Error checking uv version: {e}")
    print(f"Stderr: {e.stderr.decode()}")
    sys.exit(1)

  github_deps_command = [
      sys.executable,  # Use the current Python executable's pip to ensure the correct environment
      "-m",
      "uv",
      "pip",
      "install",
      "-r",
      str(github_deps_path),
      "--no-deps",
  ]

  try:
    print(f"Installing Github dependencies: {' '.join(github_deps_command)}")
    subprocess.run(github_deps_command, check=True)
    print("Github dependencies installed successfully!")
  except subprocess.CalledProcessError as e:
    print("Failed to install extra dependencies.")
    print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
    sys.exit(e.returncode)
  except (OSError, FileNotFoundError) as e:
    print(f"An OS-level error occurred while trying to run uv: {e}")
    sys.exit(1)

  if args.with_tf:
    tf_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "tf_requirements.txt")
    if not os.path.exists(tf_deps_path):
      raise FileNotFoundError(f"TensorFlow dependencies file not found at {tf_deps_path}")

    tf_deps_command = [
        sys.executable,
        "-m",
        "uv",
        "pip",
        "install",
        "-r",
        str(tf_deps_path),
    ]
    try:
      print(f"Installing optional TensorFlow dependencies: {' '.join(tf_deps_command)}")
      subprocess.run(tf_deps_command, check=True)
      print("TensorFlow dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
      print("Failed to install TensorFlow dependencies.")
      print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
      sys.exit(e.returncode)
    except (OSError, FileNotFoundError) as e:
      print(f"An OS-level error occurred while trying to run uv: {e}")
      sys.exit(1)

    jetstream_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "jetstream_github_deps.txt")
    if os.path.exists(jetstream_deps_path):
      jetstream_deps_command = [
          sys.executable,
          "-m",
          "uv",
          "pip",
          "install",
          "-r",
          str(jetstream_deps_path),
          "--no-deps",
      ]
      try:
        print(f"Installing JetStream dependencies: {' '.join(jetstream_deps_command)}")
        subprocess.run(jetstream_deps_command, check=True)
        print("JetStream dependencies installed successfully!")
      except subprocess.CalledProcessError as e:
        print("Failed to install JetStream dependencies.")
        print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)
      except (OSError, FileNotFoundError) as e:
        print(f"An OS-level error occurred while trying to run uv: {e}")
        sys.exit(1)


if __name__ == "__main__":
  main()
