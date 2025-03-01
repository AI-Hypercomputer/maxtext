# pylint: skip-file
from absl import app
from typing import Sequence

from jax.test_util import XlaGpuFeatureTestCase

def test_collective_matmul(test_case, hlo_file, expected_unrolled_ag, expected_unrolled_rs):
  """
  Test collective matmul correctness in HLO file.

  Args:
  test_case: The JAX test case object.
  hlo_file: Path to the HLO file.
  expected_unrolled_ag: Expected number of unrolled all-gather operations.
  expected_unrolled_rs: Expected number of unrolled reduce-scatter operations.
  """
  with open(hlo_file, 'r') as hlo_file:
    hlo_content = hlo_file.read()
    test_case.check_collective_matmul(hlo_content, expected_unrolled_ag, expected_unrolled_rs)
    print('collective matmul test passed.')

def test_fp8_gemm(test_case, hlo_file, expected_fp8_gemm):
  """
  Test FP8 GEMM correctness in HLO file.

  Args:
  test_case: The JAX test case object.
  hlo_file: Path to the HLO file.
  expected_fp8_gemm: Expected number of FP8 GEMM operations.
  """
  with open(hlo_file, 'r') as hlo_file:
    hlo_content = hlo_file.read()
    test_case.check_fp8_gemm(hlo_content, expected_fp8_gemm)
    print('fp8_gemm test passed.')

def main(argv: Sequence[str]) -> None:
  test_case = XlaGpuFeatureTestCase()
  _, test_scenario, *test_vars = argv

  if test_scenario == 'collective_matmul':
    test_collective_matmul(test_case, *test_vars)
  elif test_scenario == 'fp8_gemm':
    test_fp8_gemm(test_case, *test_vars)
  else:
     raise ValueError(f"Unrecognized test_scenario {test_scenario}")

if __name__ == "__main__":
  app.run(main)
