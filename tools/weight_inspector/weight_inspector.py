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

r"""This is to inspect/analyze two weights with the same structure to find differences.
This assumes weights are dumped in a pickle file

Usage:

python3 -m MaxText.weight_inspector --lhs left_hand.pkl --rhs right_hand.pkl

"""

import argparse
import pickle
import numpy as np
import torch
from maxtext.utils import max_logging


def inspect_weights(left_path, right_path):
  """Load the pickle files and compare contents."""
  with open(left_path, "rb") as file:
    left_weights = pickle.load(file)

  with open(right_path, "rb") as file:
    right_weights = pickle.load(file)
  assert sorted(left_weights.keys()) == sorted(
      right_weights.keys()
  ), f"Weights structure does not match! {list(set(left_weights.keys()).symmetric_difference(right_weights.keys()))}"

  mismatched_keys = []
  # Iterate through keys common to both dictionaries
  for key in left_weights.keys() & right_weights.keys():  # Intersection of keys
    if ".0." in key:  # check only layer 0 of the model
      assert (
          left_weights[key].shape == right_weights[key].shape
      ), f"Mismatched shapes left {left_weights[key].shape}, right right_weights[key].shape"
      if not np.allclose(
          left_weights[key].type(torch.float16).numpy(), right_weights[key].type(torch.float16).numpy(), atol=1e-8
      ):
        mismatched_keys.append(key)

  if mismatched_keys:
    max_logging.log("Contents of mismatched keys")
    for key in mismatched_keys:
      max_logging.log(f"Key: {key}")
      max_logging.log(f"{left_weights[key]=}")
      max_logging.log(f"{right_weights[key]=}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--lhs", type=str, required=True)
  parser.add_argument("--rhs", type=str, required=True)

  args = parser.parse_args()

  inspect_weights(args.lhs, args.rhs)

  args = parser.parse_args()
