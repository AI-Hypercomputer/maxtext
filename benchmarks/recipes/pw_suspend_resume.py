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

"""
This recipe tests the suspend and resume functionality in Pathways.

It launches a MaxText workload using Cluster Toolkit (gcluster) and then uses
the DisruptionManager to send a SIGTERM signal at a specific training step. This
simulates a planned preemption event, allowing validation of the framework's
ability to gracefully suspend, checkpoint, and later resume training.
"""
import argparse
import dataclasses

from benchmarks.disruption_management.disruption_handler import DisruptionMethod
from benchmarks.recipes import args_helper as helper
from benchmarks.recipes import parser_utils
from benchmarks.recipes import user_configs
from benchmarks.recipes.runner_utils import generate_and_run_workloads

DISRUPTION_METHOD = DisruptionMethod.SIGTERM
DISRUPTIONS = {
    # "time_seconds":[120,600],
    "step": [3]
}


def main():
  """Main function to run the suspend/resume disruption test."""
  parser = argparse.ArgumentParser()
  parser_utils.add_arguments(parser)
  parser.set_defaults(
      **{
          f.name: getattr(user_configs.USER_CONFIG, f.name)
          for f in dataclasses.fields(user_configs.USER_CONFIG)
          if f.init
      }
  )
  args, _ = parser.parse_known_args()
  user_config = dataclasses.replace(user_configs.UserConfig(**vars(args)), max_restarts=3, headless=False)
  should_continue = helper.handle_cmd_args(
      user_config.cluster_config,
      is_delete=user_config.delete,
      user=user_config.user,
  )

  if not should_continue:
    return 0

  return_code = generate_and_run_workloads(
      user_config,
      user_config.num_slices_list,
      user_config.benchmark_steps,
      user_config.priority,
      DISRUPTION_METHOD,
      DISRUPTIONS,
  )

  print("Suspend/Resume disruptions completed. Please check logs for results.")

  return return_code


if __name__ == "__main__":
  main()
