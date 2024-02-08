# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to construct configs for maxtext DAG."""

from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner, gcs_bucket
from dags.multipod.configs import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion
import datetime

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value


def get_mxla_collective_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    bytes_to_transfer: int,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    num_slices: int = 1,
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
      dataset_project=project_name,
      composer_project=project_name,
  )

  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")

  base_output_directory = f"{gcs_bucket.XLML_OUTPUT_DIR}/multipod/mxla/nightly/automated/{current_date}/{num_slices}slice-V{tpu_version.value}_{tpu_cores}-mxla-collective-{bytes_to_transfer}transferBytes-{current_datetime}"

  test_platform = common.Platform.GCE
  set_up_cmds = common.setup_mxla_collective()
  run_model_cmds = (
      (
          "cd /tmp/mxla_collective &&"
          f" sudo bash run_mxla_collective_benchmark_gcp.sh BYTES_TO_TRANSFER={bytes_to_transfer} NUM_STEPS=100 SLICES={num_slices} ACCELERATOR_TYPE=v{tpu_version.value}_{tpu_cores} GCS_PATH={base_output_directory}"
          " exit $(cat /tmp/benchmark_status.txt)"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.TONY_C,
      num_slices=num_slices,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
