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

import benchmarks.recipes.args_helper as helper
import maxtext_trillium_model_configs as model_configs
import maxtext_xpk_runner as mxr
from recipes.user_configs import (
    cluster_config,
    xpk_path,
    pathways_config,
    base_output_directory,
)


def main() -> int:
  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=xpk_path
  )

  if not should_continue:
    return 0

  # Verify Pause Resume
  num_slices = 4
  virtual_slices = num_slices - 2
  model = model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds
  model.tuning_params["colocated_python_data_input"] = True

  # Add one additional slice to the virtaul slices.
  if cluster_config.device_type == "v6e-256":
    pathways_config.proxy_flags = (
        "--virtual_slices=" + "tpuv6e:16x16," * (virtual_slices)
    )[:-1]

  # Define the workload configuration directly
  wl_config = mxr.WorkloadConfig(
      model=model,
      num_slices=num_slices,
      device_type=cluster_config.device_type,
      base_output_directory=base_output_directory,
      max_restarts=5,
      libtpu_type=None,
      libtpu_nightly_version="",
      base_docker_image="",
      pathways_config=pathways_config,
      xpk_path=xpk_path,
      num_steps=50,
      priority="high",
      elastic_slices=virtual_slices,
      max_slice_restarts=1000,
  )

  # Generate and run the single workload command
  command, name = mxr.generate_xpk_workload_cmd(
      cluster_config=cluster_config, wl_config=wl_config
  )

  print(f"Name of the workload is: {name} \n")
  print(f"XPK command to be used is: {command} \n")

  return_code = mxr.run_command_with_updates(command, name)
  if return_code != 0:
    print(f"Unable to run xpk workload: {name}")


if __name__ == "__main__":
  main()
