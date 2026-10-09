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

"""A recipe for running an elastic training benchmark with disruptions.

This script configures and launches a MaxText workload on a GKE cluster using
Cluster Toolkit (gcluster), and then introduces disruptions (e.g., killing pods)
to test the resilience and recovery capabilities of the training job. It can be
configured to run with both Pathways and McJAX to compare their elastic training
behavior.
"""

import dataclasses
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmarks.disruption_management.disruption_handler import DisruptionMethod
from benchmarks.recipes import args_helper as helper
from benchmarks.recipes import user_configs
from benchmarks.recipes.runner_utils import generate_and_run_workloads

COMPARE_WITH_MCJAX = True

DISRUPTION_METHOD = DisruptionMethod.SIGILL
DISRUPTIONS = {
    "time_seconds": [120, 600],
    # "step":[3]
}


def main() -> int:
  """Main function to run the elastic training disruption test."""
  user_config = dataclasses.replace(user_configs.USER_CONFIG, max_restarts=10, headless=False)
  is_delete = user_config.delete or ("--delete" in sys.argv)
  should_continue = helper.handle_cmd_args(user_config.cluster_config, is_delete, user_config.user)

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

  print("Elastic Training disruptions completed. Please check logs for results.")

  return return_code


if __name__ == "__main__":
  main()
