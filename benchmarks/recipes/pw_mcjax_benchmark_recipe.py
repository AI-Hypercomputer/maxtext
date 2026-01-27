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

"""Used to perf benchmarks between Pathways and McJax."""

from benchmarks.recipes import args_helper as helper
from benchmarks.recipes import user_configs
from benchmarks.recipes.runner_utils import generate_and_run_workloads


def main() -> int:
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
  return_code = generate_and_run_workloads(
      user_configs.USER_CONFIG,
      user_configs.USER_CONFIG.num_slices_list,
      user_configs.USER_CONFIG.benchmark_steps,
      user_configs.USER_CONFIG.priority,
  )

  return return_code


if __name__ == "__main__":
  main()
