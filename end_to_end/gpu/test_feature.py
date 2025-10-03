# pylint: skip-file
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
"""Test module for XLA GPU feature verification.

This module provides test utilities for verifying GPU-specific features in HLO files,
including collective matmul operations and FP8 GEMM operations.
"""

from absl import app
from typing import Sequence

import numpy as np
import re


class XlaGpuFeatureTestCase:
  """Test case class for verifying GPU-specific features in HLO files."""

  def check_collective_matmul(
      self,
      hlo_content: str,
      expected_unrolled_ag: int,
      expected_unrolled_rs: int,
  ) -> None:
    """Verify correctness of collective matmul in the given HLO content.

    Args:
      hlo_content: The HLO file content as a string.
      expected_unrolled_ag: Expected number of unrolled all-gather operations.
      expected_unrolled_rs: Expected number of unrolled reduce-scatter operations.

    Raises:
      AssertionError: If the HLO content is not a string or if counts don't match
        expected values.
    """
    if not isinstance(hlo_content, str):
      raise AssertionError("hlo_content must be a string")

    expected_unrolled_ag = int(expected_unrolled_ag)
    expected_unrolled_rs = int(expected_unrolled_rs)

    pattern_ag = r"%unrolled_windowed_dot_general_body_ag"
    pattern_rs = r"%unrolled_windowed_dot_general_body_rs"

    actual_unrolled_ag = len(re.findall(pattern_ag, hlo_content, re.MULTILINE))
    actual_unrolled_rs = len(re.findall(pattern_rs, hlo_content, re.MULTILINE))

    np.testing.assert_equal(
        (actual_unrolled_ag, actual_unrolled_rs),
        (expected_unrolled_ag, expected_unrolled_rs),
        err_msg=(
            f"Collective matmul operation counts mismatch. "
            f"Expected all-gather: {expected_unrolled_ag}, "
            f"Actual all-gather: {actual_unrolled_ag}, "
            f"Expected reduce-scatter: {expected_unrolled_rs}, "
            f"Actual reduce-scatter: {actual_unrolled_rs}"
        ),
    )

  def check_fp8_gemm(
      self,
      hlo_content: str,
      expected_fp8_gemm: int,
  ) -> None:
    """Check the number of FP8 GEMM operations in the given HLO content.

    Args:
      hlo_content: The HLO file content as a string.
      expected_fp8_gemm: The expected number of FP8 GEMM operations.

    Raises:
      AssertionError: If the HLO content is not a string, or if the actual count
        of FP8 GEMM operations doesn't match the expected count.
    """
    if not isinstance(hlo_content, str):
      raise AssertionError("hlo_content must be a string")

    expected_fp8_gemm = int(expected_fp8_gemm)

    pattern_fp8_gemm = r"__cublas\$lt\$matmul\$f8"

    actual_fp8_gemm = len(re.findall(pattern_fp8_gemm, hlo_content, re.MULTILINE))

    np.testing.assert_equal(
        actual_fp8_gemm,
        expected_fp8_gemm,
        err_msg=(f"FP8 GEMM operation count mismatch. " f"Expected: {expected_fp8_gemm}, " f"Actual: {actual_fp8_gemm}"),
    )


def test_collective_matmul(
    test_case: XlaGpuFeatureTestCase,
    hlo_file: str,
    expected_unrolled_ag: int,
    expected_unrolled_rs: int,
) -> None:
  """Test collective matmul correctness in HLO file.

  Args:
    test_case: The JAX test case object.
    hlo_file: Path to the HLO file.
    expected_unrolled_ag: Expected number of unrolled all-gather operations.
    expected_unrolled_rs: Expected number of unrolled reduce-scatter operations.
  """
  with open(hlo_file, "r") as hlo_file:
    hlo_content = hlo_file.read()
    test_case.check_collective_matmul(hlo_content, expected_unrolled_ag, expected_unrolled_rs)
    print("Collective matmul test passed.")


def test_fp8_gemm(
    test_case: XlaGpuFeatureTestCase,
    hlo_file: str,
    expected_fp8_gemm: int,
) -> None:
  """Test FP8 GEMM correctness in HLO file.

  Args:
    test_case: The JAX test case object.
    hlo_file: Path to the HLO file.
    expected_fp8_gemm: Expected number of FP8 GEMM operations.
  """
  with open(hlo_file, "r") as hlo_file:
    hlo_content = hlo_file.read()
    test_case.check_fp8_gemm(hlo_content, expected_fp8_gemm)
    print("FP8 GEMM test passed.")


def main(argv: Sequence[str]) -> None:
  """Main entry point for the test module.

  Args:
    argv: Command line arguments. The first argument should be the test scenario
      ('collective_matmul' or 'fp8_gemm'), followed by the test parameters.

  Raises:
    ValueError: If the test scenario is not recognized.
  """
  test_case = XlaGpuFeatureTestCase()
  _, test_scenario, *test_vars = argv

  if test_scenario == "collective_matmul":
    test_collective_matmul(test_case, *test_vars)
  elif test_scenario == "fp8_gemm":
    test_fp8_gemm(test_case, *test_vars)
  else:
    raise ValueError(f"Unrecognized test scenario: {test_scenario}")


if __name__ == "__main__":
  app.run(main)
