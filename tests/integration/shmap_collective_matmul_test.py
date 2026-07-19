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

"""Integration test for maxtext/examples/shmap_collective_matmul.py"""

import os.path
import sys

import pytest

from maxtext.utils.globals import MAXTEXT_REPO_ROOT

sys.path.append(os.path.join(MAXTEXT_REPO_ROOT, "pedagogical_examples"))

# Uncomment the import when b/415022795 is fixed
# from maxtext.examples.shmap_collective_matmul import main


@pytest.mark.skip(reason="Enable when b/415022795 is fixed")
@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_shmap_collective_matmul_example():
  """Validate Pedagogical Example, Shmap_collective_matmul."""
  # Uncomment main() assertion when b/415022795 is fixed
  # assert main() is True
