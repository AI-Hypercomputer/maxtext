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
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from . import args_helper as helper
from .user_configs import UserConfig
from .user_configs import USER_CONFIG
from .runner_utils import generate_and_run_workloads
from . import parser_utils
import argparse
from google.cloud import storage
from .pw_utils import check_and_create_bucket

def main(user_config) -> int:
  """Main program entry point"""
  storage_client = storage.Client(project=user_config.project)
  check_and_create_bucket(storage_client, user_config.base_output_directory[5:].split('/')[0], user_config.region)
  return_code = generate_and_run_workloads(user_config, user_config.num_slices_list, user_config.benchmark_steps, user_config.priority)

  return return_code


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  parser_utils.add_arguments(parser)
  args = parser.parse_args()

  if len(sys.argv) > 2:
    print("Multiple command line arguments detected. Custom configuration will be used.")
    user_config = UserConfig(**vars(args))
    should_continue = helper.handle_cmd_args(
        user_config.cluster_config,
        is_delete=user_config.delete,
        user=user_config.user,
        xpk_path=user_config.xpk_path
    )
    if not should_continue:
      sys.exit(0)

    print(f"configuration used：{user_config}")
    return_code = main(user_config)
    sys.exit(return_code)

  else:
    print("No command line or only a single --delete argument was detected. The default configuration will be used.")
    user_config = USER_CONFIG
    if "--delete" in sys.argv:
      user_config.delete = True
      should_continue = helper.handle_cmd_args(
          user_config.cluster_config,
          is_delete=user_config.delete,
          user=user_config.user,
          xpk_path=user_config.xpk_path
      )
      if not should_continue:
        sys.exit(0)    
    print(f"configuration used：{user_config}")
    return_code= main(user_config)
    sys.exit(return_code)
