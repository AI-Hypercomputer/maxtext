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

"""Tests for Hugging Face checkpoint conversion."""

import argparse
import glob
import pathlib
import os

import torch

from safetensors import safe_open

import numpy as np

import jax


def load_hf(hf_checkpoint_folder):
  safetensor_files = glob.glob(os.path.join(hf_checkpoint_folder, "*.safetensors"))

  hf_tensor = {}
  for st_f in safetensor_files:
    with safe_open(st_f, framework="pt", device="cpu") as f:
      for key in f.keys():
        hf_tensor[key] = f.get_tensor(key).to(torch.float16)
  return hf_tensor


def load_meta(meta_checkpoint_folder):
  meta_tensor = {}
  ckpt_paths = sorted(pathlib.Path(meta_checkpoint_folder).glob("[!.]*.pth"))
  for ckpt_path in ckpt_paths:
    meta_tensor = torch.load(ckpt_path, map_location="cpu")
  return meta_tensor


def compare_pytrees(tree1, tree2, atol=0.001):
  """
  Compares two JAX pytrees to check if all leaf values are within the given absolute tolerance.

  Args:
      tree1: First pytree.
      tree2: Second pytree.
      atol: Absolute tolerance for comparison (default: 0.001).

  Returns:
      A boolean indicating if all leaf values are within the specified range.
  """
  # Ensure both trees have the same structure
  if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(tree2):
    print(
        "Pytrees have different structures! Tree1:"
        f"{jax.tree_util.tree_structure(tree1)} \n\n\n"
        f"Tree2: {jax.tree_util.tree_structure(tree2)}"
    )
    return

  # Compare leaves with names
  def get_named_leaves(pytree, parent_key=""):
    named_leaves = {}
    for key, value in pytree.items():
      new_key = f"{parent_key}.{key}" if parent_key else key
      if isinstance(value, dict):
        named_leaves.update(get_named_leaves(value, new_key))
      else:
        named_leaves[new_key] = value
    return named_leaves

  named_leaves1 = get_named_leaves(tree1)
  named_leaves2 = get_named_leaves(tree2)

  print(f"There are {len(named_leaves1.keys())} leaves to check.")
  for key in named_leaves1:  # pylint: disable=C0206
    if key not in named_leaves2:
      print(f"Missing key in second tree: {key}")
      return
    try:
      if not np.allclose(named_leaves1[key], named_leaves2[key], atol=atol):
        print(f"Mismatch at leaf '{key}' with shape {named_leaves1[key].shape}:\n")
        for i in range(10):
          print(f"{named_leaves1[key][..., i, :]}\n")
        print("The second tensor:\n")
        for i in range(10):
          print(f"{named_leaves2[key][..., i, :]}\n")
        return
    except:  # pylint: disable=W0702
      print(f"The issue is with {key}")

  print(f"All {len(named_leaves1.keys())} leaves match within tolerance.")


def main():
  parser = argparse.ArgumentParser(description="Compares the original checkpoint and converted back checkpoint.")
  parser.add_argument(
      "--original_ckpt",
      type=str,
      default="",
      help="The original huggingface checkpoint",
  )
  parser.add_argument(
      "--converted_ckpt",
      type=str,
      default="",
      help="The original huggingface checkpoint",
  )
  args = parser.parse_args()

  hf_tensor = load_hf(args.original_ckpt)
  meta_tensor = load_hf(args.converted_ckpt)

  compare_pytrees(hf_tensor, meta_tensor)


if __name__ == "__main__":
  main()
