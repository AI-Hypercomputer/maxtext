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

"""Run script to dump sharding of various combination of model and topology.

This script is a utility to generate and save the sharding configurations
(both physical and logical) for various model and hardware topology combinations.
These saved configurations act as "golden" files for regression testing.

There are two primary ways to use the script:

1. Generate Sharding for All Predefined Test Cases
----------------------------------------------------
Run the script without any command-line arguments to iterate through all test
cases defined in `tests.utils.sharding_dump.TEST_CASES`. It will skip any
combination for which the output files already exist.

Command:
  python3 -m tests.utils.run_sharding_dump

2. Generate Sharding for a Single, Specific Case
-------------------------------------------------
Provide the `model_name`, `topology`, and `num_slice` as command-line arguments
to generate sharding information for a single configuration. You must provide
all three arguments.

Command:
  python3 -m tests.utils.run_sharding_dump --model_name <model> --topology <topology> --num_slice <slices>

Example:
  python3 -m tests.utils.run_sharding_dump --model_name gemma-7b --topology v5p-256 --num_slice 1

"""


from typing import Sequence

from maxtext.utils.globals import MAXTEXT_PKG_DIR, MAXTEXT_REPO_ROOT
from tests.utils.sharding_dump import TEST_CASES
import os
import subprocess
from absl import app, flags
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string("model_name", None, "Specific model name to dump.")
flags.DEFINE_string("topology", None, "Specific topology to dump.")
flags.DEFINE_string("num_slice", None, "Specific number of slices to dump.")


def run_single_dump(model_name: str, topology: str, num_slice: str) -> None:
  """Generate sharding json file for one specific model, topology and slice."""
  subprocess.run(
      [
          "python3",
          "-m",
          "tests.utils.sharding_dump",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"compile_topology={topology}",
          f"compile_topology_num_slices={num_slice}",
          f"model_name={model_name}",
          "weight_dtype=float32",
          "log_config=false",
          "debug_sharding=true",
      ],
      check=True,
  )


def main(argv: Sequence[str]) -> None:
  """Generate json files for every combination of model, topology and slices."""
  if FLAGS.model_name and FLAGS.topology and FLAGS.num_slice:
    cases_to_run = [(FLAGS.model_name, FLAGS.topology, FLAGS.num_slice)]
    print(
        "Running specific case from command line: "
        f"Model={FLAGS.model_name}, Topology={FLAGS.topology}, NumSlice={FLAGS.num_slice}"
    )
  elif FLAGS.model_name or FLAGS.topology or FLAGS.num_slice:
    print("Error: To specify a single test case, --model_name, --topology, and --num_slice must all be provided.")
    return
  else:
    cases_to_run = TEST_CASES
    print(f"Running all {len(TEST_CASES)} predefined test cases.")

  total = len(cases_to_run)
  for i, (model_name, topology, num_slice) in enumerate(cases_to_run):
    print(f"\n[{i+1}/{total}] Processing: {model_name} | {topology} | Slice {num_slice}")

    base_path = Path(f"{MAXTEXT_REPO_ROOT}/tests/utils/sharding_info/{model_name}/" f"{topology}/slice_{num_slice}/")
    json_path_named = base_path / "named_shardings.json"
    json_path_logical = base_path / "logical_shardings.json"

    if json_path_named.exists() and json_path_logical.exists():
      print("  -> Sharding files already exist. Regenerating to overwrite.")

    try:
      run_single_dump(model_name, topology, str(num_slice))
    except subprocess.CalledProcessError:
      print(f"!!! FAILED: {model_name} {topology} {num_slice}")


if __name__ == "__main__":
  app.run(main)
