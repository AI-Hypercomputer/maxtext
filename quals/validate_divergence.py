# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to mathematically validate parameter divergence between SFT and DPO checkpoints.
"""

import sys
import argparse
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from etils import epath


def compute_pytree_diff(tree1, tree2):
  """Computes the L2 differences of leaf arrays between tree1 and tree2."""
  # Flatten both trees with their path elements
  dict1 = jax.tree_util.tree_leaves_with_path(tree1)
  dict2 = jax.tree_util.tree_leaves_with_path(tree2)

  # Convert paths to readable string keys
  map1 = {str(p): val for p, val in dict1}
  map2 = {str(p): val for p, val in dict2}

  common_keys = set(map1.keys()).intersection(set(map2.keys()))
  if not common_keys:
    print("Error: No common keys found between SFT and DPO parameter trees.", file=sys.stderr)
    return None

  total_diff = 0.0
  matched_count = 0

  for k in sorted(common_keys):
    val1 = map1[k]
    val2 = map2[k]

    # Ensure both are arrays/leafs
    if hasattr(val1, "shape") and hasattr(val2, "shape"):
      if val1.shape != val2.shape:
        print(f"Shape mismatch for parameter '{k}': {val1.shape} vs {val2.shape}", file=sys.stderr)
        continue

      # Compute L2 distance
      diff = jnp.linalg.norm(val1 - val2)
      total_diff += float(diff)
      matched_count += 1
      if diff > 1e-6:
        print(f"Parameter '{k}' diverged by L2: {diff:.6f}")

  print(f"Matched and compared {matched_count} parameters.")
  return total_diff


def main():
  parser = argparse.ArgumentParser(description="Validate parameter divergence between two checkpoints.")
  parser.add_argument("--sft_path", type=str, required=True, help="Path to SFT baseline checkpoint directory.")
  parser.add_argument("--dpo_path", type=str, required=True, help="Path to DPO checkpoint directory.")
  parser.add_argument("--threshold", type=float, default=1e-5, help="Minimum L2 divergence threshold to pass.")
  args = parser.parse_args()

  print(f"Restoring SFT baseline from: {args.sft_path}")
  print(f"Restoring DPO checkpoint from: {args.dpo_path}")

  # Restore using Orbax PyTree checkpoint handler
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))

  try:
    sft_state = ckptr.restore(epath.Path(args.sft_path))
    if isinstance(sft_state, dict) and "model" in sft_state:
      sft_state = sft_state["model"]
    elif isinstance(sft_state, dict) and "params" in sft_state:
      sft_state = sft_state["params"]
  except Exception as e:
    print(f"Error loading SFT checkpoint: {e}", file=sys.stderr)
    sys.exit(1)

  try:
    dpo_state = ckptr.restore(epath.Path(args.dpo_path))
    if isinstance(dpo_state, dict) and "model" in dpo_state:
      dpo_state = dpo_state["model"]
    elif isinstance(dpo_state, dict) and "params" in dpo_state:
      dpo_state = dpo_state["params"]
  except Exception as e:
    print(f"Error loading DPO checkpoint: {e}", file=sys.stderr)
    sys.exit(1)

  diff = compute_pytree_diff(sft_state, dpo_state)
  if diff is None:
    sys.exit(1)

  print(f"Total L2 Parameter Divergence: {diff:.8f}")
  if diff < args.threshold:
    print(
        f"CRITICAL ERROR: Parameter divergence {diff:.8f} is below the threshold of {args.threshold:.8f}!",
        file=sys.stderr,
    )
    print("The checkpoints are mathematically identical or too similar!", file=sys.stderr)
    sys.exit(2)
  else:
    print("SUCCESS: Checkpoints have mathematically diverged.")
    sys.exit(0)


if __name__ == "__main__":
  main()
