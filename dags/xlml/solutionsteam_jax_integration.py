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

import datetime
from airflow import models
from apis import gcp_config, metric_config, task, test_config
from configs import composer_env, vm_resource


# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None
US_CENTRAL1_A = gcp_config.GCPConfig(
    vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
    vm_resource.Zone.US_CENTRAL1_A.value,
    metric_config.DatasetOption.XLML_DATASET,
)
US_CENTRAL1_C = gcp_config.GCPConfig(
    vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
    vm_resource.Zone.US_CENTRAL1_C.value,
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id="jax_integration",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "jax", "integration", "xlml"],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  compilation_cache = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_jax(
          "jax-compilation-cache-test-func-v2-8-1vm"
      ),
      US_CENTRAL1_C,
  ).run()

  pod_latest = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_jax(
          "jax-pod-latest-tpu-ubuntu2204-base-func-v2-32-1vm"
      ),
      US_CENTRAL1_A,
  ).run()

  pod_head = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_jax(
          "jax-pod-head-tpu-ubuntu2204-base-func-v2-32-1vm"
      ),
      US_CENTRAL1_A,
  ).run()
