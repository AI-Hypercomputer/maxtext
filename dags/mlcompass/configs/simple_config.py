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

"""Utilities to construct configs for simple DAG."""

from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.vm_resource import TpuVersion, Zone, Project, RuntimeVersion


def get_simple_config() -> task.TpuQueuedResourceTask:
  set_up_cmds = (
      "set +x",
      "echo {{params.commit_sha}}",
      "pip install -U pip",
      "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
  )

  run_model_cmds = (
      "set +x",
      "echo {{params.commit_sha}}",
      "ls -ltrh /dev/accel*",
      "python3 -c 'import jax; print(jax.device_count()); print(jax.numpy.add(1,1))'",
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=TpuVersion.V4,
          cores=8,
          runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=False,
          network="default",
          subnetwork="default",
      ),
      test_name="simple-jax-code",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=60,
      task_owner=test_owner.ORTI_B,
      num_slices=1,
  )

  project_name = Project.CLOUD_ML_AUTO_SOLUTIONS.value
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=Zone.US_CENTRAL2_B.value,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
      dataset_project=project_name,
      composer_project=project_name,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
