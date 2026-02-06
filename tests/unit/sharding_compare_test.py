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

"""Compare expected sharding of models with actual sharding of models."""

import hashlib
import json
import os
import pytest
import jax
import jax.numpy as jnp
# import optax

from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.train_compile import get_shaped_inputs, get_topology_mesh, validate_config
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText import optimizers
from maxtext.models import models
from maxtext.layers import quantizations
from tests.utils.sharding_dump import load_json, TEST_CASES, named_shardings_to_json, partition_specs_to_json

Transformer = models.transformer_as_linen


def compute_checksum(d: dict) -> str:
  """Compute a checksum (SHA256) of a dictionary."""
  # Serialize the dictionary into a JSON string (ensuring consistent ordering of keys)
  json_str = json.dumps(d, sort_keys=True)

  # Compute the SHA256 checksum of the serialized string
  checksum = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

  return checksum


def compare_sharding_jsons(json1: dict, model1_name: str, json2: dict, model2_name: str) -> bool:
  """Compare two json files and print the differences if any."""
  keys1 = set(json1.keys())
  keys2 = set(json2.keys())

  only_in_1 = keys1 - keys2
  only_in_2 = keys2 - keys1
  shared_keys = keys1 & keys2

  has_diff = False

  if only_in_1:
    print(f"Keys only in {model1_name}:")
    for k in sorted(only_in_1):
      print(f"  {k}")
    has_diff = True

  if only_in_2:
    print(f"Keys only in {model2_name}:")
    for k in sorted(only_in_2):
      print(f"  {k}")
    has_diff = True

  for key in sorted(shared_keys):
    entry1 = json1[key]
    entry2 = json2[key]

    if isinstance(entry1, dict) and isinstance(entry2, dict):
      mesh1 = entry1.get("mesh", {})
      mesh2 = entry2.get("mesh", {})

      spec1 = entry1.get("partition_spec", [])
      spec2 = entry2.get("partition_spec", [])

      shape1 = entry1.get("shape")
      shape2 = entry2.get("shape")

      if mesh1 != mesh2:
        print(f"\nMesh mismatch at '{key}':")
        print(f"  {model1_name}: {mesh1}")
        print(f"  {model2_name}: {mesh2}")
        has_diff = True

      if spec1 != spec2:
        print(f"\nPartitionSpec mismatch at '{key}':")
        print(f"  {model1_name}: {spec1}")
        print(f"  {model2_name}: {spec2}")
        has_diff = True

      if shape1 != shape2:
        print(f"\nShape mismatch at '{key}':")
        print(f"  {model1_name}: {shape1}")
        print(f"  {model2_name}: {shape2}")
        has_diff = True

    else:
      print(f"\nFormat mismatch at '{key}':")
      print(f"  {model1_name} type: {type(entry1)}")
      print(f"  {model2_name} type: {type(entry2)}")
      has_diff = True

  return has_diff


@pytest.mark.parametrize("model_name, topology, num_slice", TEST_CASES)
def test_sharding_dump_for_model(model_name: str, topology: str, num_slice: str) -> None:
  """
  Test sharding configurations from train_compile.get_shaped_inputs.
  This test verifies that the sharding configurations for various models and topologies remain consistent with golden files.
  """
  params = [
      "/deps/MaxText/tests/unit/sharding_compare_test",
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
  ]

  root_dir = "tests/utils/sharding_info"
  base_path = os.path.join(root_dir, model_name, topology, f"slice_{num_slice}")

  named_json_path = os.path.join(base_path, "named_shardings.json")
  logical_json_path = os.path.join(base_path, "logical_shardings.json")

  if not os.path.exists(named_json_path):
    pytest.skip(f"Missing named_shardings.json for {model_name} {topology} slice {num_slice}")
    return
  if not os.path.exists(logical_json_path):
    pytest.skip(f"Missing logical_shardings.json for {model_name} {topology} slice {num_slice}")
    return

  config = pyconfig.initialize(params)
  validate_config(config)

  topology_mesh = get_topology_mesh(config)
  shaped_train_args, _, state_mesh_shardings, logical_shardings, _ = get_shaped_inputs(topology_mesh, config)

  error_messages = []

  # 1. Compare Named Shardings
  actual_named = named_shardings_to_json(state_mesh_shardings, shaped_train_args[0])
  expected_named = load_json(named_json_path)
  # calculate checksum
  actual_named_sum = compute_checksum(actual_named)
  expected_named_sum = compute_checksum(expected_named)
  named_match = actual_named_sum == expected_named_sum

  if not named_match:
    print(f"\n[FAIL] Physical Sharding Mismatch: {model_name} {topology} slice {num_slice}", flush=True)
    compare_sharding_jsons(expected_named, "Expected (Physical)", actual_named, "Actual (Physical)")
    error_messages.append(f" Physical sharding mismatch for {model_name} on {topology} slice {num_slice}")

  # 2. Compare Logical Shardings
  actual_logical = partition_specs_to_json(logical_shardings, shaped_train_args[0])
  expected_logical = load_json(logical_json_path)
  # calculate checksum
  actual_logical_sum = compute_checksum(actual_logical)
  expected_logical_sum = compute_checksum(expected_logical)
  logical_match = actual_logical_sum == expected_logical_sum

  if not logical_match:
    print(f"\n[FAIL] Logical Sharding Mismatch: {model_name} {topology} slice {num_slice}", flush=True)
    compare_sharding_jsons(expected_logical, "Expected (Logical)", actual_logical, "Actual (Logical)")
    error_messages.append(f"Logical sharding mismatch for {model_name} on {topology} slice {num_slice}")

  assert not error_messages, "\n".join(error_messages)


@pytest.fixture(
    scope="module",
    params=[pytest.param(case, id=f"{case[0]}-{case[1]}-{case[2]}") for case in TEST_CASES],
)
def abstract_state_and_shardings(request):
  """Pytest fixture to set up model, config, and generate abstract state once per test case."""
  model_name, topology, num_slice = request.param
  print(f"Testing model: {model_name}, topology: {topology}, num_slices: {num_slice}", flush=True)
  params = [
      "/deps/MaxText/tests/unit/sharding_compare_test",
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
      "weight_dtype=float32",
  ]
  config = pyconfig.initialize(params)
  validate_config(config)

  topology_mesh = get_topology_mesh(config)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=topology_mesh, quant=quant)

  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # tx = optax.adam(learning_rate=learning_rate_schedule)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  rng = jax.random.PRNGKey(0)

  # Get abstract state and physical shardings from maxtext_utils
  abstract_state, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
      model, tx, config, rng, topology_mesh, is_training=True
  )

  # Get logical shardings from maxtext_utils
  logical_shardings = maxtext_utils.get_logical_annotations(model, tx, config, rng, topology_mesh, is_training=True)

  return model_name, topology, num_slice, abstract_state, state_mesh_shardings, logical_shardings


class TestGetAbstractState:
  """Test class for get_abstract_state function and sharding comparison."""

  def test_get_abstract_state_sharding(self, abstract_state_and_shardings):  # pylint: disable=redefined-outer-name
    """Tests that get_abstract_state returns a state with the correct abstract structure and compares sharding."""

    model_name, topology, num_slice, abstract_state, state_mesh_shardings, logical_shardings = (
        abstract_state_and_shardings
    )

    assert hasattr(abstract_state, "params")
    assert hasattr(abstract_state, "opt_state")
    param_leaf = jax.tree_util.tree_leaves(abstract_state.params)[0]
    assert isinstance(param_leaf, jax.ShapeDtypeStruct)
    assert param_leaf.dtype == jnp.float32

    root_dir = "tests/utils/sharding_info"  # Or your target directory
    base_path = os.path.join(root_dir, model_name, topology, f"slice_{num_slice}")
    os.makedirs(base_path, exist_ok=True)  # Ensure directory exists for saving actual

    error_messages = []

    # 1. Compare Physical/Named Shardings
    named_json_path = os.path.join(base_path, "named_shardings.json")
    if not os.path.exists(named_json_path):
      pytest.skip(f"Missing named_shardings.json for {model_name} {topology} slice {num_slice}")
      return

    # Use state_mesh_shardings from the fixture
    actual_named = named_shardings_to_json(state_mesh_shardings, abstract_state)
    expected_named = load_json(named_json_path)

    if compare_sharding_jsons(expected_named, "Expected (Physical)", actual_named, "Actual (Physical)"):
      error_messages.append(f"Physical sharding mismatch for {model_name} on {topology} slice {num_slice}")

    # 2. Compare Logical Shardings
    logical_json_path = os.path.join(base_path, "logical_shardings.json")
    if not os.path.exists(logical_json_path):
      pytest.skip(f"Missing logical_shardings.json for {model_name} {topology} slice {num_slice}")
      return

    # Use logical_shardings from the fixture
    actual_logical = partition_specs_to_json(logical_shardings, abstract_state)
    expected_logical = load_json(logical_json_path)

    if compare_sharding_jsons(expected_logical, "Expected (Logical)", actual_logical, "Actual (Logical)"):
      error_messages.append(f"Logical sharding mismatch for {model_name} on {topology} slice {num_slice}")

    assert not error_messages, "\n".join(error_messages)
