import glob
import os
import torch
from safetensors import safe_open
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import torch
from llama_or_mistral_ckpt import permute_to_match_maxtext_rope
from llama_mistral_mixtral_orbax_to_hf import unpermute_from_match_maxtext_rope
import sys
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import pyconfig
import pytest

import unittest

def load_hf(hf_checkpoint_folder):
  safetensor_files = glob.glob(os.path.join(hf_checkpoint_folder, "*.safetensors"))

  hf_tensor = {}
  for st_f in safetensor_files:
    with safe_open(st_f, framework="pt", device="cpu") as f:
      for key in f.keys():
        hf_tensor[key] = f.get_tensor(key).to(torch.float16)
        # print(f"Weight name {key}, Shape: {hf_tensor.shape}, dtype: {hf_tensor[key].dtype}")
  return hf_tensor

def load_meta(meta_checkpoint_folder):
  meta_tensor = {}
  ckpt_paths = sorted(pathlib.Path(meta_checkpoint_folder).glob("[!.]*.pth"))
  for i, ckpt_path in enumerate(ckpt_paths):
    meta_tensor = torch.load(ckpt_path, map_location="cpu")
    # chkpt_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
  # chkpt_vars = [chkpt_vars[i] for i in sorted(list(chkpt_vars.keys()))]
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
        print(f"Pytrees have different structures! Tree1: {jax.tree_util.tree_structure(tree1)} \n\n\nTree2: {jax.tree_util.tree_structure(tree2)}")
        return False

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

    for key in named_leaves1:
        # import pdb; pdb.set_trace()
        if key not in named_leaves2:
            print(f"Missing key in second tree: {key}")
            return False
        try:
          if not np.allclose(named_leaves1[key], named_leaves2[key], atol=atol):
              # print(f"Mismatch at leaf '{key}':\n{named_leaves1[key]}\n{named_leaves2[key]}")
              # return False
              # print(f"Mismatch at leaf '{key}'")
              mismatch_values1 = named_leaves1[key].flatten()[:10]
              mismatch_values2 = named_leaves2[key].flatten()[:10]
              # print(f"Mismatch at leaf '{key}':\nFirst 10 elements:\n{mismatch_values1}\n{mismatch_values2}")
              print(f"Mismatch at leaf '{key}' with shape {named_leaves1[key].shape}:\n")
              for i in range(10):
                print(f"{named_leaves1[key][..., i, :]}\n")
              print(f"The second tensor:\n")
              for i in range(10):
                print(f"{named_leaves2[key][..., i, :]}\n")
              return
        except:
          print(f"The issue is with {key}")
        # print(f"Checking {key} done")

    print("All leaves match within tolerance.")
    return True

def test_huggingface_to_maxtext_back_to_huggingface_flow():
  base_num_query_heads = base_num_kv_heads = 16
  head_dim = 32
  # import pdb; pdb.set_trace()
  wq = np.arange(base_num_query_heads * head_dim * base_num_query_heads * head_dim, dtype=np.float16).reshape(base_num_query_heads * head_dim , base_num_query_heads * head_dim )
  wq1 = wq.transpose()
  wq2 = np.reshape(wq1, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

  wq3 = permute_to_match_maxtext_rope(wq2)
  stack_shape = (1,)
  x = np.zeros(stack_shape + wq3.shape, dtype=np.float16)
  x[0, ...] = wq3
  x = np.transpose(x, axes=(1, 0, 2, 3))

  x = x[:, 0, :, :]
  wq4 = unpermute_from_match_maxtext_rope(x, "llama3.1")
  wq5 = wq4.reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim)
  wq6 = wq5.transpose()

  if not np.array_equal(wq, wq6):
    print("Test failed: wq does not match wq6")

  if not np.array_equal(wq1, wq5):
    print("Test failed: wq1 does not match wq5")

  if not np.array_equal(wq2, wq4):
    print("Test failed: wq2 does not match wq4")

def test_permutate_unpermutate():
  x = np.arange(24*36).reshape(24, 36)
  permutated = permute_to_match_maxtext_rope(x)
  x_ = unpermute_from_match_maxtext_rope(permutated, "llama3.1")
  assert np.array_equal(x, x_), "Test failed: x does not match x_"

if __name__ == "__main__":
  hf_checkpoint_folder = "/mnt/disks/persist/checkpoints/huggingface/Llama3.1-8B"
  hf_tensor = load_hf(hf_checkpoint_folder)

  meta_checkpoint_folder = "/tmp/hf_llama3_1_no_perm"
  meta_tensor = load_hf(meta_checkpoint_folder)


  compare_pytrees(hf_tensor, meta_tensor)

