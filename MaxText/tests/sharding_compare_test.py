"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Compare expected sharding of models with actual sharding of models."""


import hashlib
from MaxText.train_compile import get_shaped_inputs, get_topology_mesh, validate_config
from MaxText.tests.sharding_dump import valid_test_cases, sharding_info_folder, named_shardings_to_json, load_named_sharding_json
from MaxText import pyconfig
import pytest
import json


def compute_checksum(d: dict) -> str:
  """Compute a checksum (SHA256) of a dictionary."""
  json_str = json.dumps(d, sort_keys=True)
  return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def compare_named_sharding_jsons(json1: dict, model1_name: str, json2: dict, model2_name: str) -> bool:
  """Compare two json files and print the differences if any."""
  keys1 = set(json1.keys())
  keys2 = set(json2.keys())

  only_in_1 = keys1 - keys2
  only_in_2 = keys2 - keys1
  shared_keys = keys1 & keys2

  if only_in_1:
    print(f"Keys only in {model1_name}:")
    for k in sorted(only_in_1):
      print(f"  {k}")

  if only_in_2:
    print(f"Keys only in {model2_name}:")
    for k in sorted(only_in_2):
      print(f"  {k}")

  for key in sorted(shared_keys):
    entry1 = json1[key]
    entry2 = json2[key]

    mesh1 = entry1.get("mesh", {})
    mesh2 = entry2.get("mesh", {})
    spec1 = entry1.get("partition_spec", [])
    spec2 = entry2.get("partition_spec", [])

    if mesh1 != mesh2:
      print(f"\nMesh mismatch at '{key}':")
      print(f"  mesh1: {mesh1}")
      print(f"  mesh2: {mesh2}")

    if spec1 != spec2:
      print(f"\nPartitionSpec mismatch at '{key}':")
      print(f"  spec1: {spec1}")
      print(f"  spec2: {spec2}")

  return not only_in_1 and not only_in_2 and all(json1[k] == json2[k] for k in shared_keys)


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_name, topology, num_slice", valid_test_cases())
def test_sharding_dump_for_model(model_name: str, topology: str, num_slice: str) -> None:
  """Test if the sharding of new model implementation is as expected."""
  params = [
      "MaxText/tests/sharding_compare_test",
      "MaxText/configs/base.yml",
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
  ]

  json_path = sharding_info_folder / model_name / topology / f"slice_{num_slice}" / "named_shardings.json"

  expected_json_exists = json_path.exists()
  assert expected_json_exists, f"Expected sharding JSON does not exist: {json_path}"

  config = pyconfig.initialize(params)
  validate_config(config)

  try:
    topology_mesh = get_topology_mesh(config)
    _, _, state_mesh_shardings, _ = get_shaped_inputs(topology_mesh, config)
  except Exception:  # pylint: disable=broad-except
    state_mesh_shardings = {}

  assert state_mesh_shardings != {}, "No sharding information was produced."

  actual_json = named_shardings_to_json(state_mesh_shardings)
  expected_json = load_named_sharding_json(json_path)

  actual_checksum = compute_checksum(actual_json)
  expected_checksum = compute_checksum(expected_json)

  result = actual_checksum == expected_checksum

  if not result:
    compare_named_sharding_jsons(expected_json, f"expected_{model_name}", actual_json, f"actual_{model_name}")

  assert result, "Sharding JSONs do not match."
