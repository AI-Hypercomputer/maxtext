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

"""A recipe for running a long-running MaxText benchmark using Pathways/McJAX.

This script is designed for stability and long-duration runs. It configures
and launches a workload on a GKE cluster using XPK, with a high number of
restarts enabled. It defines the cluster, Docker images, and model
configurations for a Pathways/McJAX-based run.
"""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from . import args_helper as helper
from . import user_configs
from .runner_utils import generate_and_run_workloads

user_configs.USER_CONFIG.max_restarts = 10_000
user_configs.USER_CONFIG.benchmark_steps = 10_000_000


def main():
  """Main program entry point"""
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
  )

  return return_code


if __name__ == "__main__":
  main()
