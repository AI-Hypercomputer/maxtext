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

This script is designed to install dependencies specified in 'dependencies/extra_deps/post_train_*.txt'.
It first ensures 'uv' is installed and then uses it to install the packages listed in the requirements file.
"""

import os
import shutil
import subprocess
import sys


def ensure_cpp20_compiler():
  """Ensures a compiler supporting C++20 (GCC >= 11.3 or Clang >= 11) is used for building vLLM."""
  if sys.platform != "linux":
    return
  try:
    res = subprocess.run(["gcc", "-dumpversion"], capture_output=True, text=True, check=False)
    major_ver = int(res.stdout.strip().split(".")[0])
    if major_ver >= 11:
      return
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # If gcc-11 / g++-11 exist, point CC and CXX to them (preferred over clang to avoid mwaitxintrin header issue)
  if shutil.which("gcc-11") and shutil.which("g++-11"):
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    return
  if shutil.which("clang") and shutil.which("clang++"):
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    return

  # If running as root (e.g. inside CI docker container), install C++20 compiler via apt
  is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
  if is_root and shutil.which("apt-get"):
    try:
      print("Ensuring C++20 compiler (gcc-11/clang) for vLLM compilation...")
      if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r", encoding="utf-8") as f:
          os_rel = f.read()
        if "bullseye" in os_rel and not os.path.exists("/etc/apt/sources.list.d/backports.list"):
          with open("/etc/apt/sources.list.d/backports.list", "w", encoding="utf-8") as f:
            f.write("deb http://deb.debian.org/debian bullseye-backports main\n")
      # Use check=False for apt-get update as index warnings/errors return exit code 100 on Debian
      subprocess.run(["apt-get", "update", "-y"], check=False, capture_output=True)
      subprocess.run(
          ["apt-get", "install", "-y", "--no-install-recommends", "-t", "bullseye-backports", "gcc-11", "g++-11"],
          check=False,
          capture_output=True,
      )
      apt_cmd = [
          "apt-get",
          "install",
          "-y",
          "--no-install-recommends",
          "gcc-11",
          "g++-11",
          "clang",
          "llvm",
          "build-essential",
          "cmake",
          "ninja-build",
      ]
      subprocess.run(apt_cmd, check=False, capture_output=True)
      if shutil.which("gcc-11") and shutil.which("g++-11"):
        os.environ["CC"] = "gcc-11"
        os.environ["CXX"] = "g++-11"
        print("Using C++20 compiler: CC=gcc-11 CXX=g++-11")
      elif shutil.which("clang") and shutil.which("clang++"):
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
        print("Using C++20 compiler: CC=clang CXX=clang++")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: Failed to install C++20 compiler via apt-get: {e}")


def main():
  """
  Installs extra dependencies specified in 'dependencies/extra_deps/post_train_*.txt' using uv.
  It executes 'uv pip install -r <path_to_extra_deps.txt> --resolution=lowest'.
  """
  os.environ["VLLM_TARGET_DEVICE"] = "tpu"
  os.environ["UV_TORCH_BACKEND"] = "cpu"
  ensure_cpp20_compiler()

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  github_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "post_train_github_deps.txt")
  if not os.path.exists(github_deps_path):
    raise FileNotFoundError(f"Github dependencies file not found at {github_deps_path}")

  # Ensure setuptools is installed via uv without calling pip
  try:
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "setuptools"], check=True, capture_output=True)
  except (subprocess.CalledProcessError, FileNotFoundError):
    try:
      subprocess.run(["uv", "pip", "install", "setuptools"], check=True, capture_output=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: could not install setuptools via uv: {e}")

  github_deps_command = [
      "uv",
      "pip",
      "install",
      "--python",
      sys.executable,
      "-r",
      str(github_deps_path),
      "--no-deps",
      "--no-build-isolation",
  ]

  raiden_keyring_command = [
      sys.executable,
      "-m",
      "pip",
      "install",
      "keyrings.google-artifactregistry-auth",
      "--extra-index-url",
      "https://pypi.org/simple",
  ]

  raiden_deps_command = [
      sys.executable,
      "-m",
      "pip",
      "install",
      "tpu-raiden-jax",
      "--extra-index-url",
      "https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/",
      "--extra-index-url",
      "https://pypi.org/simple",
      "--no-deps",
  ]

  local_vllm_install_command = [
      "uv",
      "pip",
      "install",
      "--python",
      sys.executable,
      f"{repo_root}/maxtext/integration/vllm",  # MaxText on vllm installations
      "--no-deps",
  ]

  try:
    # Run the command to install Github dependencies
    print(f"Installing Github dependencies: {' '.join(github_deps_command)}")
    _ = subprocess.run(github_deps_command, check=True, capture_output=True, text=True, env=os.environ)
    print("Github dependencies installed successfully!")

    # Attempt optional Raiden dependencies installation (non-blocking for outside users)
    try:
      print(f"Installing optional Raiden keyring dependency: {' '.join(raiden_keyring_command)}")
      _ = subprocess.run(raiden_keyring_command, check=True, capture_output=True, text=True, env=os.environ)
      print("Raiden keyring dependency installed successfully!")
      print(f"Installing optional Raiden dependencies: {' '.join(raiden_deps_command)}")
      _ = subprocess.run(raiden_deps_command, check=True, capture_output=True, text=True, env=os.environ)
      print("Raiden dependencies installed successfully!")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: Optional Raiden dependencies installation skipped/failed: {e}")

    # Run the command to install the MaxText vLLM directory
    print(f"Installing MaxText vLLM dependency: {' '.join(local_vllm_install_command)}")
    _ = subprocess.run(local_vllm_install_command, check=True, capture_output=True, text=True, env=os.environ)
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
