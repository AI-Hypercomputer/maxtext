# SPDX-License-Identifier: Apache-2.0

"""Integration test for pedagogical_examples/shmap_collective_matmul.py"""

import os.path
import sys

import pytest

from MaxText.globals import PKG_DIR

sys.path.append(os.path.join(os.path.dirname(PKG_DIR), "pedagogical_examples"))

# Uncomment the import when b/415022795 is fixed
# from pedagogical_examples.shmap_collective_matmul import main


@pytest.mark.skip(reason="Enable when b/415022795 is fixed")
@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_shmap_collective_matmul_example():
  """Validate Pedagogical Example, Shmap_collective_matmul."""
  # Uncomment main() assertion when b/415022795 is fixed
  # assert main() is True
