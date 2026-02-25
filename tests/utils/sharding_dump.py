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

import itertools
import json
import os
from pathlib import Path
from typing import Any, List, Sequence, Union
from absl import app
import jax
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten_with_path
from MaxText import maxtext_utils
from MaxText import optimizers
from MaxText.globals import MAXTEXT_REPO_ROOT
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.trainers.pre_train.train_compile import get_shaped_inputs, get_topology_mesh, validate_config


Transformer = models.Transformer

MODEL_NAMES = [
    # "default",
    # "llama2-7b",
    # "llama2-13b",
    # "llama2-70b",
    # "llama3-8b",
    # "llama3-70b",
    # "llama3.1-8b",
    # "llama3.1-70b",
    # "llama3.1-405b",
    # "llama3.3-70b",
    # "mistral-7b",
    # "mixtral-8x7b",
    # "mixtral-8x22b",
    "deepseek2-16b",
    # "deepseek2-236b",
    # "deepseek3-671b",
    # "deepseek3-671b-2dfsdp",
    # "deepseek3-test",
    # "deepseek3-tiny",
    # "deepseek3.2-671b",
    # "gemma-7b",
    # "gemma-2b",
    # "gemma2-2b",
    # "gemma2-9b",
    # "gemma2-27b",
    # "gemma3-4b",
    # "gemma3-12b",
    # "gemma3-27b",
    "qwen3-0.6b",
    # "qwen3-4b",
    # "qwen3-4b-thinking-2507",
    # "qwen3-8b",
    # "qwen3-14b",
    # "qwen3-32b",
    # "qwen3-235b-a22b",
    # "qwen3-30b-a3b",
    # "qwen3-480b-a35b",
    # "qwen3-next-80b-a3b",
    # "qwen3-omni-30b-a3b",
    # "gpt3-175b",
    # "gpt3-22b",
    # "gpt3-6b",
    # "gpt3-52k",
    "gpt-oss-20b",
    # "gpt-oss-120b",
    # "llama4-17b-16e",
    # "llama4-17b-128e",
]

TOPOLOGIES = [
    # "tpu7x-2",
    # "tpu7x-8",
    "tpu7x-16",
    # "tpu7x-32",
    # "tpu7x-64",
    # "tpu7x-128",
    # "tpu7x-256",
    # "tpu7x-384",
    # "tpu7x-512",
    # "tpu7x-640",
    # "tpu7x-768",
    # "tpu7x-896",
    # "tpu7x-1024",
    # "tpu7x-1152",
    # "tpu7x-1280",
    # "tpu7x-1408",
    # "tpu7x-1536",
    # "tpu7x-1664",
    # "tpu7x-1792",
    # "tpu7x-1920",
    # "tpu7x-2048",
    # "tpu7x-2176",
    # "tpu7x-2304",
    # "tpu7x-2432",
    # "tpu7x-2560",
    # "tpu7x-2688",
    # "tpu7x-2816",
    # "tpu7x-2944",
    # "tpu7x-3072",
    # "tpu7x-3200",
    # "tpu7x-3328",
    # "tpu7x-3456",
    # "tpu7x-3584",
    # "tpu7x-3712",
    # "tpu7x-3840",
    # "tpu7x-3968",
    # "tpu7x-4096",
    # "tpu7x-4224",
    # "tpu7x-4352",
    # "tpu7x-4480",
    # "tpu7x-4608",
    # "tpu7x-4736",
    # "tpu7x-4864",
    # "tpu7x-4992",
    # "tpu7x-5120",
    # "tpu7x-5248",
    # "tpu7x-5376",
    # "tpu7x-5504",
    # "tpu7x-5632",
    # "tpu7x-5760",
    # "tpu7x-5888",
    # "tpu7x-6016",
    # "tpu7x-6144",
    # "tpu7x-6272",
    # "tpu7x-6400",
    # "tpu7x-6528",
    # "tpu7x-6656",
    # "tpu7x-6784",
    # "tpu7x-6912",
    # "tpu7x-7040",
    # "tpu7x-7168",
    # "tpu7x-7296",
    # "tpu7x-7424",
    # "tpu7x-7552",
    # "tpu7x-7680",
    # "tpu7x-7808",
    # "tpu7x-7936",
    # "tpu7x-8064",
    # "tpu7x-8192",
    # "tpu7x-8320",
    # "tpu7x-8448",
    # "tpu7x-8704",
    # "tpu7x-8832",
    # "tpu7x-8960",
    # "tpu7x-9216",
    # "tpu7x-9472",
    # "tpu7x-9600",
    # "tpu7x-9728",
    # "tpu7x-9856",
    # "tpu7x-9984",
    # "tpu7x-10240",
    # "tpu7x-10368",
    # "tpu7x-10496",
    # "tpu7x-10752",
    # "tpu7x-10880",
    # "tpu7x-11008",
    # "tpu7x-11136",
    # "tpu7x-11264",
    # "tpu7x-11520",
    # "tpu7x-11648",
    # "tpu7x-11776",
    # "tpu7x-11904",
    # "tpu7x-12032",
    # "tpu7x-12160",
    # "tpu7x-12288",
    # "tpu7x-13824",
    # "tpu7x-16384",
    # "tpu7x-17920",
    # "tpu7x-18432",
    # "v6e-1",
    # "v6e-4",
    # "v6e-8",
    "v6e-16",
    # "v6e-32",
    # "v6e-64",
    # "v6e-128",
    # "v6e-256",
    # "v5e-1",
    # "v5e-4",
    # "v5e-8",
    # "v5e-16",
    # "v5e-32",
    # "v5e-64",
    # "v5e-128",
    # "v5e-256",
    # "v4-8",
    # "v4-16",
    # "v4-32",
    # "v4-64",
    # "v4-128",
    # "v4-256",
    # "v4-384",
    # "v4-512",
    # "v4-1024",
    # "v4-1536",
    # "v4-2048",
    # "v4-4096",
    # "v5p-8",
    "v5p-16",
    # "v5p-32",
    # "v5p-64",
    # "v5p-128",
    # "v5p-256",
    # "v5p-384",
    # "v5p-512",
    # "v5p-640",
    # "v5p-768",
    # "v5p-896",
    # "v5p-1024",
    # "v5p-1152",
    # "v5p-1280",
    # "v5p-1408",
    # "v5p-1536",
    # "v5p-1664",
    # "v5p-1792",
    # "v5p-1920",
    # "v5p-2048",
    # "v5p-2176",
    # "v5p-2304",
    # "v5p-2432",
    # "v5p-2560",
    # "v5p-2688",
    # "v5p-2816",
    # "v5p-2944",
    # "v5p-3072",
    # "v5p-3200",
    # "v5p-3328",
    # "v5p-3456",
    # "v5p-3584",
    # "v5p-3712",
    # "v5p-3840",
    # "v5p-3968",
    # "v5p-4096",
    # "v5p-4224",
    # "v5p-4352",
    # "v5p-4480",
    # "v5p-4608",
    # "v5p-4736",
    # "v5p-4864",
    # "v5p-4992",
    # "v5p-5120",
    # "v5p-5248",
    # "v5p-5376",
    # "v5p-5504",
    # "v5p-5632",
    # "v5p-5760",
    # "v5p-5888",
    # "v5p-6016",
    # "v5p-6144",
    # "v5p-6272",
    # "v5p-6400",
    # "v5p-6528",
    # "v5p-6656",
    # "v5p-6784",
    # "v5p-6912",
    # "v5p-7040",
    # "v5p-7168",
    # "v5p-7296",
    # "v5p-7424",
    # "v5p-7552",
    # "v5p-7680",
    # "v5p-7808",
    # "v5p-7936",
    # "v5p-8064",
    # "v5p-8192",
    # "v5p-8320",
    # "v5p-8448",
    # "v5p-8704",
    # "v5p-8832",
    # "v5p-8960",
    # "v5p-9216",
    # "v5p-9472",
    # "v5p-9600",
    # "v5p-9728",
    # "v5p-9856",
    # "v5p-9984",
    # "v5p-10240",
    # "v5p-10368",
    # "v5p-10496",
    # "v5p-10752",
    # "v5p-10880",
    # "v5p-11008",
    # "v5p-11136",
    # "v5p-11264",
    # "v5p-11520",
    # "v5p-11648",
    # "v5p-11776",
    # "v5p-11904",
    # "v5p-12032",
    # "v5p-12160",
    # "v5p-12288",
    # "v5p-13824",
    # "v5p-17920",
    # "a3"
]

SLICES = [1, 4]

TEST_CASES = list(itertools.product(MODEL_NAMES, TOPOLOGIES, SLICES))


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


def save_json(output_path: str | Path, sharding_dict: dict) -> None:
  """Save dict to a JSON file."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sharding_dict, f, indent=2)


def load_json(json_path: str | Path) -> dict:
  """Loads the named_shardings.json file into a plain Python dict."""
  json_path = Path(json_path)
  with open(json_path, "r", encoding="utf-8") as f:
    return json.load(f)


def main(argv: Sequence[str]) -> None:
  """Load a config that describes a model with topology and slices to be dumped."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "")
      + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting sharding_tests.py...", flush=True)

  config = pyconfig.initialize(argv)
  validate_config(config)

  base_path = Path(
      f"{MAXTEXT_REPO_ROOT}/tests/utils/sharding_info/{config.model_name}/"
      f"{config.compile_topology}/slice_{config.compile_topology_num_slices}/"
  )
  json_path_named = base_path / "named_shardings.json"
  json_path_logical = base_path / "logical_shardings.json"

  try:
    topology_mesh = get_topology_mesh(config)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    optimizers.get_optimizer(config, learning_rate_schedule)
    shaped_train_args, _, state_mesh_shardings, logical_annotations, _ = (
        get_shaped_inputs(topology_mesh, config)
    )
  except Exception as e:  # pylint: disable=broad-except
    print(f"Error generating inputs: {e}")
    return

  if not state_mesh_shardings:
    print("No shardings generated.")
    return

  # 1. Generate New Output
  # Physical: Tree of NamedSharding
  named_shardings = named_shardings_to_json(
      state_mesh_shardings, shaped_train_args[0]
  )
  # Logical: Tree of PartitionSpec (direct from get_shaped_inputs)
  logical_shardings = partition_specs_to_json(
      logical_annotations, shaped_train_args[0]
  )

  print(
      f"Got {len(named_shardings)} Physical entries and"
      f" {len(logical_shardings)} Logical entries."
  )

  # 2. Save New Output (Overwrite)
  print(f"\nSaving updated shardings to {base_path}...")
  save_json(json_path_named, named_shardings)
  save_json(json_path_logical, logical_shardings)

  print(f"Finished: {config.model_name} {config.compile_topology}")


if __name__ == "__main__":
  app.run(main)
