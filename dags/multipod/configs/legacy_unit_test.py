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

"""Utilities to run legacy tests from the old infrastructure."""

import datetime
import os
from xlml.apis import gcp_config, metric_config, task, test_config
from base64 import b64encode
from collections.abc import Iterable
from dags import test_owner
from dags.multipod.configs import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion, ClusterName


def get_legacy_unit_test_config(
    script_to_copy: str,
    test_cmd: Iterable,
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    test_owner: str,
    docker_image: str,
    num_slices: int = 1,
    cluster_name: str = ClusterName.V4_8_MULTISLICE_CLUSTER.value,
    project_name: str = Project.TPU_PROD_ENV_MULTIPOD.value,
) -> task.TpuXpkTask:
  """
  Run a legacy unit test script.
  `script_to_copy` is a script in the `dags/multipod/legacy_tests` folder to be
  copied into the workload container, and `test_cmd` will run with the script
  in the working directory.
  """
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  unit_test_folder = os.environ.get(
      'XLMLTEST_MULTIPOD_LEGACY_TEST_DIR',
      '/home/airflow/gcs/dags/dags/multipod/legacy_tests',
  )
  with open(os.path.join(unit_test_folder, script_to_copy), 'rb') as f:
    encoded_script = b64encode(f.read()).decode()

  run_model_cmds = (
      f'echo {encoded_script} | base64 -d > {script_to_copy}',
      'export TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 JAX_USE_PJRT_C_API_ON_TPU=1 TF_CPP_MIN_LOG_LEVEL=0',
      *test_cmd,
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=test_name,
      run_model_cmds=run_model_cmds,
      set_up_cmds=None,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner,
      num_slices=num_slices,
      cluster_name=cluster_name,
      docker_image=docker_image,
  )

  return task.TpuXpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
