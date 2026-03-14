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
Utility script to generate statistics for test outputs.

This script helps generate hardcoded statistics that can be used in tests
to avoid PyTorch dependencies in CI.

Usage:
  python generate_test_statistics.py <output_file.npy>
  
  Or use in Python:
    from generate_test_statistics import generate_and_print_stats
    generate_and_print_stats(your_array, "my_output")
"""

import numpy as np
import sys


def generate_statistics_dict(array, name="output"):
  """Generate statistics dictionary from an array for comparison.
  
  This function extracts key statistics that can be hardcoded for CI testing
  without requiring PyTorch forward passes.
  
  Args:
    array: NumPy array, JAX array, or PyTorch tensor
    name: Name identifier for the statistics
    
  Returns:
    Dictionary with statistics that can be printed and hardcoded
  """
  # Convert to numpy for consistent handling
  if hasattr(array, 'detach'):  # PyTorch tensor
    arr_np = array.detach().cpu().numpy()
  elif hasattr(array, '__array__'):  # JAX or NumPy
    arr_np = np.array(array)
  else:
    arr_np = array
  
  # Flatten to 1D for easier statistics
  flat = arr_np.flatten()
  
  stats = {
      "name": name,
      "shape": tuple(arr_np.shape),
      "mean": float(np.mean(flat)),
      "std": float(np.std(flat)),
      "max": float(np.max(flat)),
      "min": float(np.min(flat)),
      "median": float(np.median(flat)),
      "first_5": flat[:5].tolist() if len(flat) >= 5 else flat.tolist(),
      "last_5": flat[-5:].tolist() if len(flat) >= 5 else flat.tolist(),
  }
  
  return stats


def print_statistics_for_hardcoding(array, name="output", indent=2):
  """Print statistics in a format that can be copy-pasted into code.
  
  Args:
    array: NumPy array, JAX array, or PyTorch tensor
    name: Name identifier for the statistics
    indent: Indentation level for formatting
  """
  stats = generate_statistics_dict(array, name)
  ind = " " * indent
  
  print(f"\n{ind}# Statistics for {name}:")
  print(f"{ind}EXPECTED_STATS = {{")
  print(f'{ind}  "shape": {stats["shape"]},')
  print(f'{ind}  "mean": {stats["mean"]},')
  print(f'{ind}  "std": {stats["std"]},')
  print(f'{ind}  "max": {stats["max"]},')
  print(f'{ind}  "min": {stats["min"]},')
  print(f'{ind}  "median": {stats["median"]},')
  print(f'{ind}  "first_5": {stats["first_5"]},')
  print(f'{ind}  "last_5": {stats["last_5"]},')
  print(f"{ind}}}")


def generate_and_print_stats(array, name="output"):
  """Convenience function to generate and print statistics.
  
  Args:
    array: Array to generate statistics from
    name: Name for the statistics
  """
  print_statistics_for_hardcoding(array, name)


def main():
  """Main function for CLI usage."""
  if len(sys.argv) < 2:
    print("Usage: python generate_test_statistics.py <output_file.npy>")
    print("\nOr use as a library:")
    print("  from generate_test_statistics import generate_and_print_stats")
    print("  generate_and_print_stats(your_array, 'my_output')")
    sys.exit(1)
  
  filepath = sys.argv[1]
  name = sys.argv[2] if len(sys.argv) > 2 else "output"
  
  print(f"Loading array from {filepath}...")
  array = np.load(filepath)
  
  print(f"Array shape: {array.shape}")
  print(f"Array dtype: {array.dtype}")
  
  print_statistics_for_hardcoding(array, name)
  
  print("\n# To use these statistics in a test:")
  print("# 1. Copy the EXPECTED_STATS dictionary above")
  print("# 2. Add it to your test method")
  print("# 3. Use compare_with_statistics(jax_output, EXPECTED_STATS, name='...')")


if __name__ == "__main__":
  main()
