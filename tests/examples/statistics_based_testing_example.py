#!/usr/bin/env python3
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

"""
Example script demonstrating how to use statistics-based testing.

This shows how to:
1. Generate statistics from an output array
2. Compare a new array against expected statistics
3. Use this in your own tests
"""

import numpy as np
import sys
import os

# Add parent directory to path to import the utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utils.generate_test_statistics import (
    generate_statistics_dict,
    print_statistics_for_hardcoding,
    generate_and_print_stats,
)


def compare_with_statistics(array, expected_stats, rtol=1e-2, atol=1e-2, name="output"):
  """Compare array against precomputed statistics.
  
  Args:
    array: Array to compare
    expected_stats: Dictionary with expected statistics
    rtol: Relative tolerance
    atol: Absolute tolerance
    name: Name for error messages
    
  Raises:
    AssertionError: If statistics don't match
  """
  actual_stats = generate_statistics_dict(array, name)
  
  # Compare shape
  if actual_stats["shape"] != expected_stats["shape"]:
    raise AssertionError(
        f"{name} shape mismatch: expected {expected_stats['shape']}, got {actual_stats['shape']}"
    )
  
  # Compare scalar statistics
  for key in ["mean", "std", "max", "min", "median"]:
    expected = expected_stats[key]
    actual = actual_stats[key]
    if not np.allclose([actual], [expected], rtol=rtol, atol=atol):
      raise AssertionError(
          f"{name} {key} mismatch: expected {expected:.6f}, got {actual:.6f} "
          f"(diff: {abs(actual - expected):.6f}, rtol: {rtol}, atol: {atol})"
      )
  
  # Compare first and last values
  for key in ["first_5", "last_5"]:
    expected = np.array(expected_stats[key])
    actual = np.array(actual_stats[key])
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
      raise AssertionError(
          f"{name} {key} mismatch:\nExpected: {expected}\nActual: {actual}"
      )
  
  print(f"✓ {name} statistics match within tolerance (rtol={rtol}, atol={atol})")


def example_1_generate_statistics():
  """Example 1: Generate statistics from a random array."""
  print("=" * 80)
  print("EXAMPLE 1: Generate Statistics")
  print("=" * 80)
  
  # Create a random array (simulating model output)
  np.random.seed(42)
  output = np.random.randn(16, 1024).astype(np.float32)
  
  print(f"\nGenerated array with shape: {output.shape}")
  print(f"Mean: {output.mean():.6f}, Std: {output.std():.6f}")
  
  # Generate and print statistics
  print("\nStatistics in dictionary format:")
  stats = generate_statistics_dict(output, "example_output")
  for key, value in stats.items():
    print(f"  {key}: {value}")
  
  # Print in format for hardcoding
  print("\nStatistics in hardcodable format:")
  print_statistics_for_hardcoding(output, "example_output")


def example_2_compare_against_statistics():
  """Example 2: Compare a new array against expected statistics."""
  print("\n" + "=" * 80)
  print("EXAMPLE 2: Compare Against Statistics")
  print("=" * 80)
  
  # Expected statistics (as if generated from a previous run)
  EXPECTED_STATS = {
    "shape": (16, 1024),
    "mean": -0.00019758939743041992,
    "std": 0.023577280715107918,
    "max": 0.11169242858886719,
    "min": -0.10485196113586426,
    "median": -0.000415802001953125,
    "first_5": [-0.017169952392578125, -0.020233154296875, 0.01904296875, 0.020172119140625, 0.0008268356323242188],
    "last_5": [0.008697509765625, -0.002193450927734375, 0.030975341796875, 0.0057220458984375, -0.013702392578125],
  }
  
  # Generate a new array with same random seed (should match)
  np.random.seed(100)  # Different seed for this example
  # Simulate an array that's close but not exact
  new_output = np.random.randn(16, 1024).astype(np.float32) * 0.023 - 0.0002
  
  print(f"\nGenerated new array with shape: {new_output.shape}")
  print(f"Mean: {new_output.mean():.6f}, Std: {new_output.std():.6f}")
  
  print("\nComparing against expected statistics...")
  try:
    compare_with_statistics(new_output, EXPECTED_STATS, rtol=1e-2, atol=1e-2, name="new_output")
    print("✓ Comparison passed!")
  except AssertionError as e:
    print(f"✗ Comparison failed: {e}")


def example_3_test_pattern():
  """Example 3: Show test pattern with flag."""
  print("\n" + "=" * 80)
  print("EXAMPLE 3: Test Pattern with USE_TORCH_FORWARD Flag")
  print("=" * 80)
  
  USE_TORCH_FORWARD = os.environ.get("USE_TORCH_FORWARD", "true").lower() in ("true", "1", "yes")
  
  print(f"\nUSE_TORCH_FORWARD = {USE_TORCH_FORWARD}")
  
  # Expected statistics
  EXPECTED_STATS = {
    "shape": (8, 512),
    "mean": 0.001,
    "std": 0.05,
    "max": 0.2,
    "min": -0.2,
    "median": 0.0,
    "first_5": [0.01, -0.02, 0.03, -0.04, 0.05],
    "last_5": [-0.01, 0.02, -0.03, 0.04, -0.05],
  }
  
  if USE_TORCH_FORWARD:
    print("\n📊 Running FULL mode (with PyTorch simulation)")
    print("   - Would create PyTorch model")
    print("   - Would run PyTorch forward pass")
    print("   - Would compare JAX vs PyTorch outputs element-wise")
    
    # Simulate JAX output
    np.random.seed(42)
    jax_output = np.random.randn(8, 512).astype(np.float32) * 0.05 + 0.001
    
    # Simulate PyTorch output (should be same)
    torch_output = jax_output.copy()
    
    print(f"\n   JAX output shape: {jax_output.shape}")
    print(f"   PyTorch output shape: {torch_output.shape}")
    print(f"   Max difference: {np.max(np.abs(jax_output - torch_output)):.6e}")
    print("\n   ✓ Full comparison passed")
    
    # Optionally print statistics
    if os.environ.get("PRINT_STATS", "false").lower() in ("true", "1", "yes"):
      print("\n   Printing statistics for hardcoding:")
      print_statistics_for_hardcoding(jax_output, "test_output")
  else:
    print("\n📈 Running LIGHTWEIGHT mode (statistics-based)")
    print("   - Skipping PyTorch model creation")
    print("   - Running JAX forward pass only")
    print("   - Comparing against precomputed statistics")
    
    # Simulate JAX output
    np.random.seed(42)
    jax_output = np.random.randn(8, 512).astype(np.float32) * 0.05 + 0.001
    
    print(f"\n   JAX output shape: {jax_output.shape}")
    print(f"   JAX output mean: {jax_output.mean():.6f}")
    print(f"   JAX output std: {jax_output.std():.6f}")
    
    try:
      compare_with_statistics(jax_output, EXPECTED_STATS, rtol=0.5, atol=0.5, name="test_output")
      print("\n   ✓ Statistics comparison passed")
    except AssertionError as e:
      print(f"\n   ✗ Statistics comparison failed (expected for this demo)")
      print(f"   Error: {e}")


def main():
  """Run all examples."""
  print("\n")
  print("╔" + "=" * 78 + "╗")
  print("║" + " " * 20 + "Statistics-Based Testing Examples" + " " * 24 + "║")
  print("╚" + "=" * 78 + "╝")
  
  example_1_generate_statistics()
  example_2_compare_against_statistics()
  example_3_test_pattern()
  
  print("\n" + "=" * 80)
  print("Summary:")
  print("=" * 80)
  print("""
The statistics-based testing approach allows you to:
1. Run tests without PyTorch dependencies (USE_TORCH_FORWARD=false)
2. Generate hardcodable statistics from outputs (PRINT_STATS=true)
3. Compare new outputs against expected statistics

Use cases:
- CI environments without PyTorch
- Quick sanity checks without full model comparison
- Regression testing with statistical signatures

See STATISTICS_BASED_TESTING.md for detailed documentation.
""")


if __name__ == "__main__":
  main()
