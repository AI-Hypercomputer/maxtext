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

""" Run script to dump sharding of various combination of model and topology. """


import json
from typing import Sequence
from MaxText.tests.sharding_dump import TEST_CASES, sharding_info_folder
import subprocess
from absl import app


def run_single_dump(model_name: str, topology: str, num_slice: str) -> None:
  """Generate sharding json file for one specific model, topology and slice."""
  subprocess.run(
      [
          "python",
          "-m",
          "MaxText.tests.sharding_dump",
          "MaxText/configs/base.yml",
          f"compile_topology={topology}",
          f"compile_topology_num_slices={num_slice}",
          f"model_name={model_name}",
      ],
      check=True,
  )



def main(argv: Sequence[str]) -> None:
  """Generate sharding json files for every combination of model, topology and slices."""
  for model_name, topology, num_slice in TEST_CASES:
    run_single_dump(model_name, topology, num_slice)


if __name__ == "__main__":
  app.run(main)
