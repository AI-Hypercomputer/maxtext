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
This script automatically searches for the optimal rematerialization policy
and batch size for a given MaxText model configuration. It aims to find the
Pareto frontier of batch size vs. rematerialization policy, allowing users to
make informed trade-offs between training throughput and memory usage.

The script works by iteratively testing different rematerialization policies,
from keeping all tensors in device memory to rematerializing all of them. For
each policy, it performs a binary search to find the largest possible batch
size that does not cause an out-of-memory (OOM) error.

The key functions in this script are:
- `is_oom`: Checks if a given configuration results in an OOM error.
- `largest_batch_size`: Finds the largest batch size for a given policy.
- `search`: The main algorithm that iterates through policies and batch sizes.

By automating this search, the script helps to efficiently find the most
performant and memory-efficient training configurations.
"""
import os
import sys
import contextlib
from typing import Sequence
from absl import app
import time
import jax
from enum import IntEnum

from MaxText import pyconfig
from MaxText import train_compile


class Action(IntEnum):
  REMAT = 0
  OFFLOAD = 1
  DEVICE = 2


class RematPolicy:
  """RematPolicy representing different remat policy combinations"""

  def __init__(self, tensor_names: list[str], tensors: dict | None = None, initial_level: Action = Action.REMAT):
    self.tensors = {name: initial_level for name in tensor_names} if tensors is None else tensors
    self.tensor_order = tensor_names

  @property
  def to_dict(self) -> dict[str, str]:
    """Convert internal action to strings for MaxText"""
    mapping = {0: "remat", 1: "offload", 2: "device"}
    return {k: mapping[v.value] if isinstance(v, Action) else mapping[v] for k, v in self.tensors.items()}

  def next_policy(self):
    """
    Moves from Remat -> Offload -> Device.
    Iterates through tensors in priority order to increase memory but reduce time usage
    TODO: it is not necessary offload is strictly better than remat. We simplify the order
    here. We need to update this logic for better time-memory modeling purpose.
    """
    new_policy = RematPolicy(tensor_names=self.tensor_order, tensors=self.tensors.copy())

    # Search for the first tensor that can be moved to a more compute-efficient state
    # reverse tensor order to on-device high AI later
    for action_to_check in [Action.OFFLOAD, Action.DEVICE]:
      for name in reversed(self.tensor_order):
        if new_policy.tensors[name] < action_to_check:
          new_policy.tensors[name] = action_to_check
          return new_policy
    return None

  def previous_policy(self):
    """
    Moves from Device -> Offload -> Remat.
    Iterates through tensors in priority order to reduce memory but increase time usage
    """
    new_policy = RematPolicy(tensor_names=self.tensor_order, tensors=self.tensors.copy())

    # Search for the first tensor that can be moved to a more memory-efficient state
    for action_to_check in [Action.OFFLOAD, Action.REMAT]:
      for name in self.tensor_order:
        if new_policy.tensors[name] > action_to_check:
          new_policy.tensors[name] = action_to_check
          return new_policy
    return None

  def __repr__(self):
    return str(self.to_dict)


def generate_priority_list(config, provided_tensor_names):
  """
  Generates a sorted list of tensors based on their scores.

  Args:
    config: The model configuration.
    provided_tensor_names: tensor names already provided that gonna skipped

  Returns:
    A sorted list of tensor names.
  """
  keys = {
      (True, 1): ["context", "qkv_proj", "mlpwi", "mlpwo", "out_proj"],
      (True, 2): ["context", "qkv_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"],
      (False, 1): ["context", "query_proj", "key_proj", "value_proj", "mlpwi", "mlpwo", "out_proj"],
      (False, 2): ["context", "query_proj", "key_proj", "value_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"],
  }
  sort_tensor_names = sorted(keys[config.fused_mlp, len(config.mlp_activations)], key=lambda x: tensor_score(x, config))
  return [key for key in sort_tensor_names if key not in provided_tensor_names]


def tensor_score(tensor_name: str, config) -> tuple:
  """
  Calculates a score for a given tensor.

  The score is used to prioritize which tensors to offload/remat first. Tensors
  with a higher score are rematerialized later. The scoring is based on tensor
  arithmetic intensity and memory size, with larger tensors getting lower scores
  (higher priority for remat).

  Args:
    tensor_name: The name of the tensor.
    config: The model configuration.

  Returns:
    A tuple representing the score.
  """
  tensor_score_map = {
      "context": (
          -config.max_target_length,
          -config.num_query_heads * config.head_dim,
      ),
      "mlpwi_0": (-config.emb_dim, -config.mlp_dim),
      "mlpwi_1": (-config.emb_dim, -config.mlp_dim),
      "mlpwo": (-config.mlp_dim, -config.emb_dim),
      "query_proj": (
          -config.emb_dim,
          -config.num_query_heads * config.head_dim,
      ),
      "key_proj": (-config.emb_dim, -config.num_kv_heads * config.head_dim),
      "value_proj": (
          -config.emb_dim,
          -config.num_kv_heads * config.head_dim,
      ),
      "out_proj": (
          -config.num_query_heads * config.head_dim,
          -config.emb_dim,
      ),
      "qkv_proj": (
          -config.emb_dim,
          -(config.num_query_heads + 2 * config.num_kv_heads) * config.head_dim,
      ),
      "mlpwi": (-config.emb_dim, -config.mlp_dim),
  }
  return tensor_score_map[tensor_name]


def find_pdb_scalar(config):
  """Calculates the scaling factor to normalize the Per-Device Batch (PDB) size.

  In distributed training, the batch size is divided across various mesh axes.
  When using non-batch-based sharding (like Tensor Parallelism), the raw
  per-device batch size can become a fractional value.

  This function identifies those non-batch axes (e.g., 'tensor') and calculates
  a multiplier. This scalar represents the value by which a fractional per-device
  batch size must be multiplied to result in an integer value, ensuring
  compatibility with memory and compute estimation logic.

  Args:
    config: The configuration object containing 'mesh_axes' and the
      corresponding 'ici_{axis}_parallelism' values.

  Returns:
    float: The aggregate parallelism degree of all non-data/non-FSDP axes,
      serving as the integer-normalization constant for the PDB.
  """
  pdb_scalar = 1.0
  for mesh_axis in config.mesh_axes:
    if mesh_axis not in ("data", "fsdp", "fsdp_transpose", "expert", "stage"):
      pdb_scalar *= getattr(config, f"ici_{mesh_axis}_parallelism")
  return pdb_scalar


def largest_batch_size(base_argv, policy, min_pdb=None, max_pdb=32.0, pdb_scalar=1.0) -> int:
  """
  Finds the largest possible per_device_batch_size (pdb) that does not cause an OOM error.

  This function uses a binary search algorithm within the provided min and max
  range to efficiently find the optimal batch size.

  Args:
    policy: The rematerialization policy.
    min_pdb: The minimum per_device_batch_size to test.
    max_pdb: The maximum per_device_batch_size to test.

  Returns:
    The largest per_device_batch_size within the range that does not result in an OOM error.
  """
  if pdb_scalar == 0.0:
    raise ValueError("pdb_scalar cannot be value zero.")

  if min_pdb <= 0.0 or max_pdb < min_pdb:
    raise ValueError(f"Abnormal {min_pdb=} or {max_pdb=} values.")

  min_pdb = 1.0 / pdb_scalar if min_pdb is None else min_pdb
  print(f"Starting binary search for the largest batch size between {min_pdb} and {max_pdb}.")

  if min_pdb == max_pdb:
    oom_res = is_oom(base_argv, policy, min_pdb)
    return min_pdb if oom_res else min_pdb - 1 / pdb_scalar

  if is_oom(base_argv, policy, min_pdb):
    print(f"OOM at minimum batch size {min_pdb}.")
    return min_pdb - 1 / pdb_scalar
  if not is_oom(base_argv, policy, max_pdb):
    print(f"No OOM at maximum batch size {max_pdb}.")
    return max_pdb

  low, high, result = int(min_pdb * pdb_scalar), int(max_pdb * pdb_scalar), int(min_pdb * pdb_scalar)
  while low <= high:
    mid = (low + high) // 2
    if mid < min_pdb:
      low = mid + 1
      continue

    if not is_oom(base_argv, policy, mid / pdb_scalar):
      result = mid
      low = mid + 1
    else:
      high = mid - 1
  return result / pdb_scalar


def is_oom(base_argv, policy: RematPolicy, pdb: int) -> bool:
  """
  Checks if the given policy and batch size cause an OOM error.

  Args:
    policy: The rematerialization policy.
    pdb: The per_device_batch_size.

  Returns:
    True if an OOM error occurs, False otherwise.
  """
  compile_argv = build_argv(base_argv, policy, pdb)
  print(f"Checking whether batch_size={pdb} and policy={policy} is OOM")

  # Save the original file descriptors for stdout (1) and stderr (2)
  orig_stdout_fd = os.dup(sys.stdout.fileno())
  orig_stderr_fd = os.dup(sys.stderr.fileno())

  try:
    # Open the null device
    with open(os.devnull, "w") as devnull:  # pylint: disable=unspecified-encoding
      devnull_fd = devnull.fileno()

      # Redirect stdout and stderr FDs to the null device
      os.dup2(devnull_fd, sys.stdout.fileno())
      os.dup2(devnull_fd, sys.stderr.fileno())

      # All output now goes to devnull
      result = train_compile.is_oom(compile_argv)

  finally:
    # This happens even if the 'try' block fails
    os.dup2(orig_stdout_fd, sys.stdout.fileno())
    os.dup2(orig_stderr_fd, sys.stderr.fileno())

    os.close(orig_stdout_fd)
    os.close(orig_stderr_fd)

  print(f"Is OOM: {result}")
  return result


def search_policy_only(
    tensor_names,
    base_argv,
    pdb,
    init_policy: dict = None,
) -> dict:
  """
  Finds the "lightest" remat policy that fits in memory for a *fixed* batch size.

  It starts with an initial policy (e.g., no remat) and iteratively adds
  more tensors to rematerialize (`next_policy`) until it no longer
  causes an Out-Of-Memory (OOM) error.

  Args:
    tensor_names: Prioritized list of all tensor names available for remat.
    base_argv: The base command-line arguments.
    pdb: The fixed per-device batch size to test against.
    init_policy: The policy to start searching from. If None, defaults to
                  'full_device_policy' (no remat).

  Returns:
    The first rematerialization policy that did *not* OOM.

  Raises:
    ValueError: If even a full remat policy causes an OOM for the given batch size.
  """
  # Sanity check: If full remat OOMs, this batch size is impossible.
  full_remat_policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.REMAT)
  if is_oom(base_argv, full_remat_policy, pdb):
    raise ValueError(f"Given batch size {pdb} leads to OOM even with full remat.")

  # Start with the lightest policy (e.g., no remat fully on device)
  policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.DEVICE) if init_policy is None else init_policy
  pre_policy = None  # To track the last policy that *did not* OOM

  # Iteratively reduce memory usage until it fits
  while is_oom(base_argv, policy, pdb):
    pre_policy = policy
    policy = policy.previous_policy()

  # Return the first policy that *fit* (did not OOM).
  return pre_policy


def search(
    tensor_names,
    base_argv,
    min_pdb: float | None = None,
    max_pdb: float = 64.0,
    init_policy: RematPolicy = None,
    pdb_scalar: float = 1.0,
) -> list[tuple[int, dict]]:
  """
  Performs the core search algorithm to find the Pareto frontier points.

  Args:
    tensor_names: Priority list of tensors.
    base_argv: base arguments for training.
    min_pdb: minimal pdb value from full device policy.
    max_pdb: maximal pdb value from full remat policy.
    pdb_scalar: batch scalar enabling fractional batch size search.

  Returns:
    A list of tuples, where each tuple contains a batch size and its
    corresponding rematerialization policy dict.
  """
  output_lst = []
  policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.REMAT) if init_policy is None else init_policy
  min_pdb = 1 / pdb_scalar if min_pdb is None else min_pdb
  pdb = max_pdb
  while policy is not None:
    pdb = largest_batch_size(base_argv, policy, min_pdb=min_pdb, max_pdb=pdb, pdb_scalar=pdb_scalar)
    if pdb > 0:
      output_lst.append((pdb, policy.to_dict))

    policy = policy.next_policy()
    if pdb < min_pdb:
      print(f"The value of {pdb=} hits {min_pdb=}. Stop iterating policies.")
      break
  return output_lst


def generate_remat_config(policy: RematPolicy) -> tuple:
  """Generate remat-related configs"""
  return ("remat_policy=custom",) + tuple(f"{key}={value}" for key, value in policy.to_dict.items())


def generate_pdb_config(pdb: int):
  """Generate batch size configs"""
  return (f"per_device_batch_size={pdb}",)


def build_argv(base_argv, remat_policy: RematPolicy, pdb: int) -> tuple[str, ...]:
  """Builds the argument vector for train_compile."""
  remat_args = generate_remat_config(remat_policy)
  pdb_args = generate_pdb_config(pdb)

  # Check if decoder_layer_input is already specified in base_argv
  has_layer_input_config = any(
      "decoder_layer_input=device" in arg or "decoder_layer_input=offload" in arg for arg in base_argv
  )

  # If not specified, append the default offload setting to remat_args
  if not has_layer_input_config:
    remat_args += ("decoder_layer_input=device",)
  return base_argv + pdb_args + remat_args


def get_parameter_value(config_tuple, prefix):
  """
  Searches a tuple for an item starting with a specific prefix
  and returns whether it was found and its value.

  Args:
    config_tuple: A tuple of strings to search.
    prefix: The prefix string to look for (e.g., 'key=').

  Returns:
    A tuple of (bool, str or None).
    - (True, value) if the prefix is found.
    - (False, None) if the prefix is not found.
  """
  for item in config_tuple:
    if item.startswith(prefix):
      # Found it. Get the length of the prefix
      # and slice the string to get everything after it.
      value = item[len(prefix) :]
      return (True, value)

  # If the loop finishes without finding the prefix
  return (False, None)


def find_batch_size(base_argv):
  """
  Parses the base arguments to find the 'per_device_batch_size'.

  Args:
      base_argv: The tuple of command-line arguments.

  Returns:
      A tuple of (bool, int or None):
      - (True, batch_size) if 'per_device_batch_size=...' was found.
      - (False, None) if it was not found.
  """
  pdb_provided, pdb_str = get_parameter_value(base_argv, prefix="per_device_batch_size=")

  return pdb_provided, float(pdb_str) if pdb_provided else None


def find_remat_policy_tensor_names(base_argv):
  """
  Finds tensors explicitly provided as flags in the command line.

  This allows a user to force certain tensors (e.g., 'context', 'query_proj')
  to be considered for rematerialization.

  Args:
      base_argv: The tuple of command-line arguments.

  Returns:
      A list of tensor names that were passed as flags.
  """
  full_tensor_list = [
      "context",
      "query_proj",
      "key_proj",
      "value_proj",
      "mlpwi_0",
      "mlpwi_1",
      "mlpwo",
      "out_proj",
      "qkv_proj",
      "mlpwi",
  ]
  provided_tensor_names = []
  for tensor_name in full_tensor_list:
    # get_parameter_value returns (bool, value). We only care if it exists.
    if get_parameter_value(base_argv, prefix=tensor_name)[0]:
      provided_tensor_names.append(tensor_name)
  return provided_tensor_names


def main(argv_list: Sequence[str]) -> None:
  """
  Main entry point for the remat policy estimation script.
  """
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting batch size and remat policy search...", flush=True)

  # Convert list to tuple for immutability and hashing
  base_argv = tuple(argv_list)

  # Check if user provided a specific batch size or specific tensors
  pdb_provided, pdb = find_batch_size(base_argv)
  provided_tensor_names = find_remat_policy_tensor_names(base_argv)

  # Load the base MaxText configuration from the provided args
  # (Assuming pyconfig and train_compile are imported)
  with open(os.devnull, "w") as devnull:  # pylint: disable=unspecified-encoding
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
      config = pyconfig.initialize(base_argv)
  train_compile.validate_config(config)
  pdb_scalar = find_pdb_scalar(config)

  # Get the prioritized list of tensors to try rematerializing
  tensor_names = generate_priority_list(config, provided_tensor_names)
  # Define the two extremes: all remat vs. no remat
  full_remat_policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.REMAT)
  full_device_policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.DEVICE)

  start_time = time.time()
  suggested_list = []

  if pdb_provided:
    # MODE 1: Batch size is fixed, just find the best policy.
    print(f"Batch size provided ({pdb}). Searching for best policy...")
    best_policy = search_policy_only(tensor_names, base_argv, pdb=pdb, init_policy=full_device_policy)
    suggested_list = [(pdb, best_policy)]
  else:
    # MODE 2: No batch size. Search for both batch size and policy.
    print("No batch size provided. Searching for max batch size and policies...")
    # First, find the absolute max batch size that fits *even with full remat*
    max_pdb = largest_batch_size(base_argv, full_remat_policy, min_pdb=1.0 / pdb_scalar, pdb_scalar=pdb_scalar)
    # suggested_list = [(max_pdb, full_remat_policy.to_dict())]

    # Now, search for combinations, starting from full-remat up to min_pdb
    suggested_list.extend(
        search(
            tensor_names,
            base_argv,
            min_pdb=1.0 / pdb_scalar,
            max_pdb=max_pdb,
            init_policy=full_remat_policy,
            pdb_scalar=pdb_scalar,
        )
    )

  end_time = time.time()
  print(f"\nSearch completed in {end_time - start_time:.2f} seconds.")

  output_filename = "remat_commands_from_estimator.txt"

  # Only open the file and print the status if the config allows writing
  if config.write_estimator_result:
    print(f"Writing {len(suggested_list)} suggested command(s) to {output_filename}...")

    with open(output_filename, "w", encoding="utf-8") as f:
      for pdb_result, policy_result in suggested_list:
        # Build the full, runnable command string
        final_argv = build_argv(base_argv[1:], policy_result, pdb_result)
        command = "python -m MaxText.train " + " ".join(final_argv)

        f.write(command + "\n")
        print(f"  - Found valid combo: pdb={pdb_result}, policy={policy_result}")

    print("Done.")
  else:
    for pdb_result, policy_result in suggested_list:
      print(f"  - Found valid combo: pdb={pdb_result}, policy={policy_result}")

  print("Done.")


if __name__ == "__main__":
  app.run(main)
