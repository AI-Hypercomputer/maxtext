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

# This block makes the script a bit more flexible. It allows `uv_utils` to be imported whether this module is run as a
# standalone script or as part of a larger Python package. It also allows us to not worry whether the full package name
# starts with "src." (this happens when running inside a docker image as part of setup.sh).
try:
  from . import uv_utils
except ImportError:
  import uv_utils


def main():
  """
  Installs extra dependencies specified in 'dependencies/extra_deps/post_train_*.txt' using uv.

  It executes 'uv add' (if uv.lock is present) or 'uv pip install'.
  """
  os.environ["VLLM_TARGET_DEVICE"] = "tpu"

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  github_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "post_train_github_deps.txt")
  if not os.path.exists(github_deps_path):
    raise FileNotFoundError(f"Github dependencies file not found at {github_deps_path}")

  # Install both requirements file and the local vLLM integration
  uv_utils.install_requirements(requirements_files=[github_deps_path])
  uv_utils.install_editable(paths=[os.path.join(repo_root, "maxtext", "integration", "vllm")])


if __name__ == "__main__":
  main()
