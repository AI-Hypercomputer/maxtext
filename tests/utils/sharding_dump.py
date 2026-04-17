# Copyright 2023–2026 Google LLC
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

"""Dump sharding of models implementing in linen with various topology to serve as baselines for comparison against

sharding strategies during migration to NNX.
"""

import json
import os
from pathlib import Path
from typing import Any, List, Sequence, Union
from absl import app
import jax
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten_with_path
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.utils.sharding import _ACTIVATION_SHARDINGS_DUMP

from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train.train_compile import get_shaped_inputs, get_topology_mesh, validate_config

Transformer = models.Transformer

TEST_CASES = [
    # (model_name, topology, num_slice, custom_mesh_and_rule, sharding_strategy_overrides)
    ("deepseek2-16b", "tpu7x-16", 1, "", ()),
    ("deepseek2-16b", "tpu7x-16", 1, "pure-fsdp", ()),
    ("deepseek2-16b", "v6e-16", 1, "", ("ici_fsdp_parallelism=-1", "ici_expert_parallelism=4")),
    (
        "deepseek2-16b",
        "v6e-16",
        1,
        "pipeline-large-moe",
        ("ici_fsdp_parallelism=-1", "ici_expert_parallelism=4", "use_ring_of_experts=true"),
    ),
    (
        "deepseek2-16b",
        "tpu7x-8",
        1,
        "ep-as-cp",
        ("ici_fsdp_parallelism=-1", "ici_expert_parallelism=2"),
    ),
    ("qwen3-0.6b", "tpu7x-16", 1, "", ()),
    ("gpt-oss-20b", "tpu7x-16", 1, "", ()),
    ("gpt-oss-20b", "tpu7x-16", 1, "", ("ici_fsdp_parallelism=-1", "ici_expert_parallelism=2")),
]


def _json_spec(spec: PartitionSpec) -> List[Union[List[str], str, None]]:
  """Convert PartitionSpec into JSON format."""

  def convert(entry):
    if isinstance(entry, tuple):
      return list(convert(e) for e in entry)
    elif entry is None:
      return None
    else:
      return str(entry)

  return list(convert(e) for e in spec)


def named_shardings_to_json(train_state, shape_tree) -> dict[str, dict]:
  """Extract NamedSharding instances from a trainstate and save to JSON file."""

  named_shardings = {}
  flat_items = tree_flatten_with_path(train_state)[0]
  flat_shapes, _ = tree_flatten_with_path(shape_tree)

  for (path_s, leaf_s), (_, leaf_sh) in zip(flat_items, flat_shapes):
    if isinstance(leaf_s, NamedSharding):
      name = "/".join(str(p) for p in path_s)
      mesh = leaf_s.mesh
      spec = leaf_s.spec
      # Extract shape from the shape_tree leaf (likely a ShapeDtypeStruct)
      shape = list(leaf_sh.shape) if hasattr(leaf_sh, "shape") else None

      named_shardings[name] = {
          "mesh": {
              "axis_names": list(mesh.axis_names),
              "shape": dict(mesh.shape),
          },
          "partition_spec": _json_spec(spec),
          "shape": shape,
      }

  print(f"Got {len(named_shardings)} NamedSharding entries.")
  return named_shardings


def partition_specs_to_json(logical_tree, shape_tree) -> dict[str, Any]:
  """Extract PartitionSpecs (Logical) from the logical tree.

  Leaf nodes are expected to be PartitionSpec (or None).
  """
  logical_dict = {}
  flat_items = tree_flatten_with_path(logical_tree)[0]
  flat_shapes, _ = tree_flatten_with_path(shape_tree)

  for (path_l, leaf_l), (_, leaf_sh) in zip(flat_items, flat_shapes):
    # leaf should be PartitionSpec or None
    if isinstance(leaf_l, PartitionSpec) or leaf_l is None:
      name = "/".join(str(p) for p in path_l)
      # Extract shape
      shape = list(leaf_sh.shape) if hasattr(leaf_sh, "shape") else None

      logical_dict[name] = {
          "partition_spec": _json_spec(leaf_l),
          "shape": shape,
      }
  print(f"Got {len(logical_dict)} Logical entries.")
  return logical_dict


def input_sharding_to_json() -> dict[str, Any]:
  input_sharding = {}
  input_sharding["Activation Sharding Dump"] = _ACTIVATION_SHARDINGS_DUMP
  print(f"Got {len(_ACTIVATION_SHARDINGS_DUMP)} Input entries.")
  return input_sharding


def save_json(output_path: str | Path, sharding_dict: dict) -> None:
  """Save dict to a JSON file."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sharding_dict, f, indent=2)


def load_json(json_path: str | Path) -> dict:
  """Loads json file into a plain Python dict."""
  json_path = Path(json_path)
  with open(json_path, "r", encoding="utf-8") as f:
    return json.load(f)


def main(argv: Sequence[str]) -> None:
  """Load a config that describes a model with topology and slices to be dumped."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting sharding_tests.py...", flush=True)

  config = pyconfig.initialize(argv)
  validate_config(config)
  print(f"Sharding debug: {config.debug_sharding}")

  rule_name = f"rule_{config.custom_mesh_and_rule}" if config.custom_mesh_and_rule else "rule_default"
  # Find overrides from argv to append to rule_name
  overrides = []
  for arg in argv:
    if "=" in arg:
      k, _ = arg.split("=", 1)
      if k not in [
          "compile_topology",
          "compile_topology_num_slices",
          "model_name",
          "custom_mesh_and_rule",
          "weight_dtype",
          "log_config",
          "debug_sharding",
      ]:
        overrides.append(arg)
  if overrides:
    rule_name += "_" + "_".join(overrides)

  base_path = Path(
      f"{MAXTEXT_REPO_ROOT}/tests/utils/sharding_info/{config.model_name}/"
      f"{config.compile_topology}/slice_{config.compile_topology_num_slices}/{rule_name}/"
  )
  json_path_named = base_path / "named_shardings.json"
  json_path_logical = base_path / "logical_shardings.json"
  json_path_input = base_path / "input_shardings.json"

  try:
    topology_mesh = get_topology_mesh(config)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    optimizers.get_optimizer(config, learning_rate_schedule)
    shaped_train_args, _, state_mesh_shardings, logical_annotations, _ = get_shaped_inputs(topology_mesh, config)
  except Exception as e:  # pylint: disable=broad-except
    print(f"Error generating inputs: {e}")
    return

  if not state_mesh_shardings:
    print("No shardings generated.")
    return

  # 1. Generate New Output
  # Physical: Tree of NamedSharding
  named_shardings = named_shardings_to_json(state_mesh_shardings, shaped_train_args[0])
  # Logical: Tree of PartitionSpec (direct from get_shaped_inputs)
  logical_shardings = partition_specs_to_json(logical_annotations, shaped_train_args[0])

  # Input
  input_shardings = input_sharding_to_json()

  print(f"Got {len(named_shardings)} Physical entries and {len(logical_shardings)} Logical entries.")

  # 2. Save New Output (Overwrite)
  print(f"\nSaving updated shardings to {base_path}...")
  save_json(json_path_named, named_shardings)
  save_json(json_path_logical, logical_shardings)
  save_json(json_path_input, input_shardings)

  print(f"Finished: {config.model_name} {config.compile_topology}")


if __name__ == "__main__":
  app.run(main)
