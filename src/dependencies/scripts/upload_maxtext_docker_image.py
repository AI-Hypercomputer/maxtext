# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper to run docker_upload_runner.sh from pip install."""

import os
import sys


def main():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  # Use relative path for Docker
  os.environ["PACKAGE_DIR"] = os.path.relpath(repo_root, os.getcwd())

  script_path = os.path.join(current_dir, "docker_upload_runner.sh")
  if not os.path.exists(script_path):
    raise FileNotFoundError(f"Script not found at {script_path}")

  cmd = ["bash", script_path] + sys.argv[1:]
  os.execvp("bash", cmd)
