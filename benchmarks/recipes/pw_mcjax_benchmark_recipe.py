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

"""Used to perf benchmarks between Pathways and McJax."""
import argparse
import dataclasses
import sys

from google.cloud import storage

from benchmarks.recipes import args_helper as helper
from benchmarks.recipes import parser_utils
from benchmarks.recipes.pw_utils import check_and_create_bucket
from benchmarks.recipes.runner_utils import generate_and_run_workloads
from benchmarks.recipes.user_configs import USER_CONFIG
from benchmarks.recipes.user_configs import UserConfig


def main(user_config) -> int:
  """
  Initializes necessary resources and executes the benchmark workload.

  This function performs essential setup, including:
  1. Setting up the Google Cloud Storage (GCS) client.
  2. Ensuring the benchmark output bucket exists.
  3. Launching the Pathways/McJax training or testing run.

  Args:
    user_config: An object containing all configuration settings (project,
                 region, slices, steps, etc.) for the current benchmark run.

  Returns:
    An integer exit code from the workload execution (0 for success, non-zero for failure).
  """
  storage_client = storage.Client(project=user_config.project)
  bucket_name = user_config.base_output_directory.removeprefix("gs://").split("/")[0]
  gcs_bucket = check_and_create_bucket(
      storage_client,
      bucket_name,
      user_config.region,
  )
  if not gcs_bucket:
    print("Exiting due to failure in GCS bucket creation or access.")
    return 1
  return_code = generate_and_run_workloads(
      user_config,
      user_config.num_slices_list,
      user_config.benchmark_steps,
      user_config.priority,
  )

  return return_code


def main_cli_execution() -> None:
  """
  Entry point for the command-line interface (CLI) execution.

  This function manages the overall lifecycle of the benchmark script by:
  1. Parsing all command-line arguments using argparse.
  2. Determining the configuration mode (custom arguments vs. default settings).
  3. Invoking the core benchmark logic defined in `main()`.

  """
  parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  parser_utils.add_arguments(parser)
  parser.set_defaults(**{f.name: getattr(USER_CONFIG, f.name) for f in dataclasses.fields(USER_CONFIG) if f.init})
  args = parser.parse_args()
  user_config = UserConfig(**vars(args))

  should_continue = helper.handle_cmd_args(
      user_config.cluster_config,
      is_delete=user_config.delete,
      user=user_config.user,
  )
  if not should_continue:
    sys.exit(0)

  return_code = main(user_config)
  sys.exit(return_code)


if __name__ == "__main__":
  main_cli_execution()
