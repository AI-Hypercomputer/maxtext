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

It launches a MaxText workload and then uses the DisruptionManager to send a
SIGTERM signal at a specific training step. This simulates a planned preemption
event, allowing validation of the framework's ability to gracefully suspend,
checkpoint, and later resume training.
"""
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from . import args_helper as helper
from . import user_configs

from benchmarks.disruption_management.disruption_handler import DisruptionMethod
from .runner_utils import generate_and_run_workloads

user_configs.USER_CONFIG.max_restarts = 3
DISRUPTION_METHOD = DisruptionMethod.SIGTERM
DISRUPTIONS = {
    # "time_seconds":[120,600],
    "step": [3]
}


def main():
  """Main function to run the suspend/resume disruption test."""
  user_configs.USER_CONFIG.headless = False
  should_continue = helper.handle_cmd_args(
      user_configs.USER_CONFIG.cluster_config, helper.DELETE, xpk_path=user_configs.USER_CONFIG.xpk_path
  )

  if not should_continue:
    return 0

  return_code = generate_and_run_workloads(
      user_configs.USER_CONFIG,
      user_configs.USER_CONFIG.num_slices_list,
      user_configs.USER_CONFIG.benchmark_steps,
      user_configs.USER_CONFIG.priority,
      DISRUPTION_METHOD,
      DISRUPTIONS,
  )

  print("Suspend/Resume disruptions completed. Please check logs for results.")

  return return_code


if __name__ == "__main__":
  main()
