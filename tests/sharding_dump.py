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

"""
Dump sharding of models implementing in linen with various topology to serve as baselines for comparison against
sharding strategies during migration to NNX.
"""

import os
import json
import itertools
from pathlib import Path
from typing import List, Sequence, Union
import jax
from absl import app
from jax.tree_util import tree_flatten_with_path
from jax.sharding import NamedSharding, PartitionSpec
from MaxText import pyconfig
from MaxText.train_compile import get_shaped_inputs, get_topology_mesh, validate_config
from MaxText.layers import models


Transformer = models.Transformer

MODEL_NAMES = [
    # "default",
    # "llama2-7b",
    # "llama2-13b",
    # "llama2-70b",
    # "llama3-8b",
    # "llama3-70b",
    # "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    # "llama3.3-70b",
    # "mistral-7b",
    # "mixtral-8x7b",
    # "mixtral-8x22b",
    # "deepseek2-16b",
    # "deepseek2-236b",
    # "deepseek3-671b",
    # "deepseek3-test",
    # "gemma-7b",
    # "gemma-2b",
    # "gemma2-2b",
    # "gemma2-9b",
    # "gemma2-27b",
    # "gemma3-4b",
    # "gemma3-12b",
    # "gemma3-27b",
    # "qwen3-0.6b",
    # "qwen3-4b",
    # "qwen3-8b",
    # "gpt3-175b",
    # "gpt3-22b",
    # "gpt3-6b",
    # "gpt3-52k",
    # "llama4-17b-16e",
    # "llama4-17b-128e",
]

TOPOLOGIES = [
    # "v6e-1",
    # "v6e-4",
    # "v6e-8",
    "v6e-16",
    # "v6e-32",
    # "v6e-64",
    # "v6e-128",
    "v6e-256",
    # "v5e-1",
    # "v5e-4",
    # "v5e-8",
    "v5e-16",
    # "v5e-32",
    # "v5e-64",
    # "v5e-128",
    "v5e-256",
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
    "v5p-256",
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

SLICES = [1, 4, 8192]

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


def named_shardings_to_json(train_state) -> dict[str, dict]:
  """Extract NamedSharding instances from a trainstate and save to JSON file."""

  named_shardings = {}
  flat_items = tree_flatten_with_path(train_state)[0]
  for path, leaf in flat_items:
    if isinstance(leaf, NamedSharding):
      name = "/".join(str(p) for p in path)
      mesh = leaf.mesh
      spec = leaf.spec

      named_shardings[name] = {
          "mesh": {
              "axis_names": list(mesh.axis_names),
              "shape": dict(mesh.shape),
          },
          "partition_spec": _json_spec(spec),
      }

  print(f"Got {len(named_shardings)} NamedSharding entries.")
  return named_shardings


def save_named_sharding_dict(output_path: str | Path, sharding_dict: dict) -> None:
  """Save the sharding dict directly to a JSON file."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sharding_dict, f, indent=2)


def load_named_sharding_json(json_path: str | Path) -> dict:
  """Loads the named_shardings.json file into a plain Python dict."""
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

  json_path = (
      f"sharding_info/{config.model_name}/"
      f"{config.compile_topology}/"
      f"slice_{config.compile_topology_num_slices}/"
      f"named_shardings.json"
  )

  try:
    topology_mesh = get_topology_mesh(config)
    _, _, state_mesh_shardings, _ = get_shaped_inputs(topology_mesh, config)
  except:  # pylint: disable=bare-except
    state_mesh_shardings = {}

  if state_mesh_shardings == {}:
    return

  sharding_dict = named_shardings_to_json(state_mesh_shardings)
  save_named_sharding_dict(json_path, sharding_dict)
  load_named_sharding_json(json_path)
  print(config.model_name, config.compile_topology)


if __name__ == "__main__":
  app.run(main)
