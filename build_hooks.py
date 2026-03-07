# Copyright 2023–2025 Google LLC
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

"""Custom build hooks for PyPI."""

import os
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

TPU_REQUIREMENTS_PATH = "dependencies/requirements/generated_requirements/tpu-requirements.txt"


def get_tpu_dependencies():
  """Reads the TPU requirements file and returns a list of dependencies."""
  if not os.path.exists(TPU_REQUIREMENTS_PATH):
    print(f"Warning: TPU requirements file not found at {TPU_REQUIREMENTS_PATH}. Skipping dependency injection.")
    return []

  with open(TPU_REQUIREMENTS_PATH, "r") as f:  # pylint: disable=unspecified-encoding
    # Filter out comments and empty lines
    deps = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
  return deps


class CustomBuildHook(BuildHookInterface):
  """A custom hook to inject TPU dependencies into the core wheel dependencies."""

  def initialize(self, version, build_data):  # pylint: disable=unused-argument
    """Initializes the build hook."""
    tpu_deps = get_tpu_dependencies()
    extras = ["cuda12", "docs"]
    extras_str = " and ".join([f"extra != '{extra}'" for extra in extras])
    build_data["dependencies"] = [
        # If the dependency already has a marker (indicated by a semicolon), we need to combine it with the extras condition.
        f"{dep.split(';')[0]}; ({dep.split(';')[-1]}) and ({extras_str})"
        if dep.count(";") == 1
        else f"{dep} ; {extras_str}"
        for dep in tpu_deps
    ]
    print(f"Successfully injected {len(tpu_deps)} TPU dependencies into the wheel's core requirements.")
