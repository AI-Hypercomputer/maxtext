# Copyright 2023 Google LLC
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


def get_maxtext_nightly_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    test_mode: common.SetupMode,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    automated_test: bool = True,
    num_slices: int = 1,
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")

  trigger = "automated" if automated_test else "manual"
  base_output_directory = f"{gcs_bucket.XLML_OUTPUT_DIR}/maxtext/{test_mode.value}/{trigger}/{current_date}"

  run_name = f"{num_slices}slice-V{tpu_version.value}_{tpu_cores}-maxtext-{test_mode.value}-{current_datetime}"

  test_platform = common.Platform.GCE
  set_up_cmds = common.setup_maxtext(test_mode, test_platform)
  run_model_cmds = (
      (
          "cd /tmp/maxtext &&"
          " JAX_PLATFORM_NAME=TPU XLA_FLAGS='--xla_dump_to=/tmp/xla_dump/'"
          " ENABLE_PJRT_COMPATIBILITY=true"
          f" python3 MaxText/train.py MaxText/configs/base.yml run_name={run_name}"
          f" base_output_directory={base_output_directory}"
          " dataset_path=gs://max-datasets-rogue dataset_type=synthetic"
          " per_device_batch_size=12 reuse_example_batch=1 global_parameter_scale=1 metrics_file='metrics.txt'"
          " steps=50 enable_checkpointing=false enable_profiler=true upload_all_profiler_results=true skip_first_n_steps_for_profiler=10 profiler_steps=10 gcs_metrics=true"
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


def get_maxtext_end_to_end_test_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    test_script: str,
    test_mode: common.SetupMode,
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
  )

  test_platform = common.Platform.GCE
  set_up_cmds = common.setup_maxtext(test_mode, test_platform)
  run_model_cmds = (f"cd /tmp/maxtext && bash end_to_end/{test_script}.sh",)

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
      task_owner=test_owner.JON_B,
      num_slices=num_slices,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
