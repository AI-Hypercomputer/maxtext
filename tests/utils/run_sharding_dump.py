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

""" Run script to dump sharding of various combination of model and topology. """


from typing import Sequence

from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_REPO_ROOT
from tests.utils.sharding_dump import TEST_CASES
import os
import subprocess
from absl import app
from pathlib import Path


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
      ],
      check=True,
  )


def main(argv: Sequence[str]) -> None:
  """Generate json files for every combination of model, topology and slices."""
  total = len(TEST_CASES)
  for i, (model_name, topology, num_slice) in enumerate(TEST_CASES):
    print(f"\n[{i+1}/{total}] Processing: {model_name} | {topology} | Slice {num_slice}")

    base_path = Path(f"{MAXTEXT_REPO_ROOT}/tests/utils/sharding_info_new/{model_name}/" f"{topology}/slice_{num_slice}/")
    json_path_named = base_path / "named_shardings.json"
    json_path_logical = base_path / "logical_shardings.json"

    if json_path_named.exists() and json_path_logical.exists():
      print("  -> Sharding files already exist. Skipping.")
      continue

    try:
      run_single_dump(model_name, topology, str(num_slice))
    except subprocess.CalledProcessError:
      print(f"!!! FAILED: {model_name} {topology} {num_slice}")


if __name__ == "__main__":
  app.run(main)
