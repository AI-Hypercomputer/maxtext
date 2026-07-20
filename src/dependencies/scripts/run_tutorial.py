# Copyright 2023-2026 Google LLC
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

"""Wrapper to run a tutorial script from pip install."""

import os
import sys


def main():
  tutorial_path = sys.argv[1]
  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  tutorial_path = os.path.join(repo_root, tutorial_path)
  if not os.path.exists(tutorial_path):
    raise FileNotFoundError(f"Tutorial not found at {tutorial_path}")

  cmd = ["bash", tutorial_path]
  os.execvp("bash", cmd)
