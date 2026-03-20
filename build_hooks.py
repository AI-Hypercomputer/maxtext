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

TPU_REQUIREMENTS_PATH = "src/dependencies/requirements/generated_requirements/tpu-requirements.txt"


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
  """A custom hook to handle platform-specific package configuration for MaxText."""

  def initialize(self, version, build_data):  # pylint: disable=unused-argument
    """Adjusts the build_data dictionary to customize the wheel's package structure."""

    # Avoid case-sensitivity issues with `MaxText` and `maxtext` directories on case-insensitive platforms.
    build_data["force_include"] = build_data.get("force_include", {})

    # Detect case-insensitivity by checking if this file can be accessed via a different case.
    # On case-insensitive filesystems flipping the case of the filename still points to the same file.
    is_case_insensitive = os.path.exists(__file__.upper()) and os.path.exists(__file__.lower())

    if is_case_insensitive:
      print("Skipping legacy MaxText shims to avoid case-sensitivity conflicts.")
      # Always include the __init__.py in the lowercase 'maxtext'.
      # This ensures that 'import maxtext' (and thus 'import MaxText') has the proper version and metadata.
      build_data["force_include"]["src/MaxText/__init__.py"] = "maxtext/__init__.py"
    else:
      # On other platforms, include 'src/MaxText' as its own top-level package for legacy support.
      # We do NOT add __init__.py to 'maxtext' here to maintain exact parity with previous builds.
      build_data["force_include"]["src/MaxText"] = "MaxText"
