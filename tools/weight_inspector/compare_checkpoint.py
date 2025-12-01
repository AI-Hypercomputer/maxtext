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


r"""This is to inspect/analyze two checkpoint weights with the same structure to find differences.


Usage:

python tools/weight_inspector/compare_checkpoint.py --lhs /model-left/runner_direct_1/checkpoints/0/items --rhs /model-right/runner_direct_1/checkpoints/0/items

"""

import argparse
import jax
import orbax.checkpoint as ocp
from typing import Any, Dict, Set
import pprint
import numpy as np

def load_params_from_path(checkpoint_dir: str) -> Dict[str, Any] | None:

  if not checkpoint_dir:
    raise ValueError("checkpoint_dir must be provided.")
  print(f"Loading quantized params checkpoint from: {checkpoint_dir}")
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  try:
    restored_object = orbax_checkpointer.restore(checkpoint_dir)
    if "params" in restored_object:
      print(f"Successfully restored checkpoint from {checkpoint_dir}")
      return restored_object["params"]
    else:
      print(f"Error: 'params' key not found in the restored checkpoint at {checkpoint_dir}")
      return None
  except Exception as e:
    print(f"An error occurred during checkpoint restoration from {checkpoint_dir}: {e}")
    return None

def get_tree_paths(tree: Any) -> Set[str]:
    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(p) for p, _ in flat_with_path}

def compare_quantized_checkpoints(left_path: str, right_path: str, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
    print(f"\n--- Comparing Checkpoints ---")
    print(f"  Left checkpoint path: {left_path}")
    print(f"  Right checkpoint path: {right_path}")

    params_left = load_params_from_path(left_path)
    params_right = load_params_from_path(right_path)

    if params_left is None or params_right is None:
        print("❌ Loading failed for one or both checkpoints. Cannot compare.")
        return False

    flat_left, struct1 = jax.tree_util.tree_flatten_with_path(params_left)
    flat_right, struct2 = jax.tree_util.tree_flatten_with_path(params_right)

    if struct1 != struct2:
        print("❌ Tree structures differ.")
        paths1 = get_tree_paths(params_left)
        paths2 = get_tree_paths(params_right)
        in_left_only = sorted(list(paths1 - paths2))
        if in_left_only:
            print("\n  Paths unique to left checkpoint:")
            for p in in_left_only: print(f"    {p}")
        in_right_only = sorted(list(paths2 - paths1))
        if in_right_only:
            print("\n  Paths unique to right checkpoint:")
            for p in in_right_only: print(f"    {p}")
        return False

    print("✅ Tree structures are identical.")

    map_left = {jax.tree_util.keystr(p): v for p, v in flat_left}
    map_right = {jax.tree_util.keystr(p): v for p, v in flat_right}

    all_equal = True
    print("\n--- Comparing Leaf Values ---")
    for key in sorted(map_left.keys()):
        left_values = map_left[key]
        right_values = map_right[key]

        if type(left_values) is not type(right_values):
            print(f"❌ Type mismatch at {key}: {type(left_values)} vs {type(right_values)}")
            all_equal = False
            continue

        if isinstance(left_values, jax.Array):
            if left_values.shape != right_values.shape:
                print(f"❌ Shape mismatch at {key}: {left_values.shape} vs {right_values.shape}")
                all_equal = False; continue
            if left_values.dtype != right_values.dtype:
                print(f"❌ Dtype mismatch at {key}: {left_values.dtype} vs {right_values.dtype}")
                all_equal = False; continue

            try:
                left_cpu = jax.device_get(left_values)
                right_cpu = jax.device_get(right_values)
            except Exception as e:
                print(f"❌ Error during jax.device_get at {key}: {e}")
                all_equal = False; continue

            if not np.allclose(left_cpu, right_cpu, rtol=rtol, atol=atol):
                print(f"❌ Numerical difference in JAX Array at {key}.")
                diff = np.abs(left_cpu - right_cpu)
                print(f"      Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
                all_equal = False
        elif isinstance(left_values, dict):
            if left_values != right_values:
                print(f"❌ Dict difference at {key}:")
                pprint.pprint(f"    Left: {left_values}", width=120)
                pprint.pprint(f"    Right: {right_values}", width=120)
                all_equal = False
        elif left_values != right_values:
            try:
                if np.isscalar(left_values) and np.isscalar(right_values) and np.allclose(np.array(left_values), np.array(right_values), rtol=rtol, atol=atol):
                    continue
            except (TypeError, ValueError):
                pass
            print(f"❌ Value difference at {key}: {left_values} vs {right_values}")
            all_equal = False

    if all_equal:
        print("\n✅ All compared leaf values are identical or numerically close.")
    else:
        print("\n❌ Differences found in leaf values. See details above.")
    return all_equal

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--lhs", type=str, required=True)
  parser.add_argument("--rhs", type=str, required=True)

  args = parser.parse_args()
  are_checkpoints_same = compare_quantized_checkpoints(args.lhs, args.rhs)
  print(f"\nComparison result: {are_checkpoints_same}")



