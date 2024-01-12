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

"""Utilities to construct configs for example_dag."""

from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.vm_resource import TpuVersion


def get_flax_resnet_xpk_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    test_name: str,
    project_name: str,
    cluster_name: str,
    docker_image: str,
    time_out_in_min: int,
    num_slices: int = 1,
) -> task.TpuXpkTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  run_model_cmds = (
      "python3 /tmp/flax/examples/imagenet/main.py"
      " --config=/tmp/flax/examples/imagenet/configs/tpu.py"
      " --workdir=/tmp/imagenet --config.num_epochs=1"
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=test_name,
      cluster_name=cluster_name,
      docker_image=docker_image,
      run_model_cmds=run_model_cmds,
      set_up_cmds=None,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.RAN_R,
      num_slices=num_slices,
  )

  return task.TpuXpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
