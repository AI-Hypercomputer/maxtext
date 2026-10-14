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

"""Compare expected sharding of models with actual sharding of models.

Each case in `tests.utils.sharding_dump.TEST_CASES` is checked against the golden
files under `tests/utils/sharding_info`, which `tests.utils.run_sharding_dump`
generates. Goldens dumped from the Linen TrainState before the NNX-only migration
cannot match the NNX state and are skipped until they are regenerated.
"""

import os

import pytest

from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train_compile import get_shaped_inputs, get_topology_mesh, validate_config
from maxtext.utils.sharding import clear_input_shardings_dump
from tests.utils.sharding_dump import (
    TEST_CASES,
    input_sharding_to_json,
    load_json,
    named_shardings_to_json,
    partition_specs_to_json,
)
from tests.utils.test_helpers import get_test_config_path

_SHARDING_INFO_DIR = "tests/utils/sharding_info"


def _diff_entries(expected: dict, actual: dict, limit: int = 20) -> list[str]:
  """Returns a readable line per key that is missing, unexpected, or different."""
  lines = [f"missing: {k}" for k in sorted(expected.keys() - actual.keys())]
  lines += [f"unexpected: {k}" for k in sorted(actual.keys() - expected.keys())]
  for k in sorted(expected.keys() & actual.keys()):
    if expected[k] != actual[k]:
      lines.append(f"changed: {k}\n    expected {expected[k]}\n    actual   {actual[k]}")
  if len(lines) > limit:
    lines = lines[:limit] + [f"... and {len(lines) - limit} more"]
  return lines


# Requires JAX TPU support to generate the simulated TPU topology.
@pytest.mark.tpu_backend
@pytest.mark.parametrize("model_name, topology, num_slice, custom_mesh_and_rule, overrides", TEST_CASES)
def test_sharding_dump_for_model(
    model_name: str, topology: str, num_slice: int, custom_mesh_and_rule: str, overrides: tuple
) -> None:
  """The state and activation shardings from `get_shaped_inputs` match the golden files."""
  rule_name = f"rule_{custom_mesh_and_rule}" if custom_mesh_and_rule else "rule_default"
  if overrides:
    rule_name += "_" + "_".join(overrides)
  base_path = os.path.join(_SHARDING_INFO_DIR, model_name, topology, f"slice_{num_slice}", rule_name)
  file_names = ("named_shardings.json", "logical_shardings.json", "input_shardings.json")
  if not all(os.path.exists(os.path.join(base_path, name)) for name in file_names):
    pytest.skip(f"Missing golden files under {base_path}")
  expected_named, expected_logical, expected_input = (load_json(os.path.join(base_path, name)) for name in file_names)
  if any(key.startswith(".params/") for key in expected_named):
    pytest.skip(f"Golden under {base_path} predates the NNX migration; regenerate it with tests.utils.run_sharding_dump")

  # Same arguments as tests.utils.run_sharding_dump.run_single_dump.
  argv = [
      "sharding_compare_test",
      get_test_config_path(),
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
      "weight_dtype=float32",
      "log_config=false",
      "debug_sharding=true",
  ]
  if custom_mesh_and_rule:
    argv.append(f"custom_mesh_and_rule={custom_mesh_and_rule}")
  argv.extend(overrides)
  config = pyconfig.initialize(argv)
  validate_config(config)

  clear_input_shardings_dump()
  shaped_train_args, _, state_mesh_shardings, logical_annotations, _ = get_shaped_inputs(
      get_topology_mesh(config), config
  )
  abstract_state = shaped_train_args[0]

  errors = []
  for kind, expected, actual in (
      ("physical", expected_named, named_shardings_to_json(state_mesh_shardings, abstract_state)),
      ("logical", expected_logical, partition_specs_to_json(logical_annotations, abstract_state)),
      ("input", expected_input, input_sharding_to_json()),
  ):
    if expected != actual:
      errors.append(f"{kind} sharding mismatch:\n  " + "\n  ".join(_diff_entries(expected, actual)))
  assert not errors, f"{model_name} {topology} slice {num_slice} {rule_name}\n" + "\n".join(errors)
