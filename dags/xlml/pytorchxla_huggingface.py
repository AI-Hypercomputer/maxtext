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


# Run once a day at 10 pm UTC (2 pm PST)
SCHEDULED_TIME = "0 22 * * *" if composer_env.is_prod_env() else None
US_CENTRAL1_C = gcp_config.GCPConfig(
    vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
    vm_resource.Zone.US_CENTRAL1_C.value,
    metric_config.DatasetOption.XLML_DATASET,
)
US_CENTRAL2_B = gcp_config.GCPConfig(
    vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
    vm_resource.Zone.US_CENTRAL2_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id="pytorchxla-huggingface",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  accelerate_v2_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-accelerate-smoke-v2-8-1vm"
      ),
      US_CENTRAL1_C,
  ).run()
  accelerate_v4_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-accelerate-smoke-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
  diffusers_v4_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-hf-diffusers-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()

  accelerate_v4_8 >> accelerate_v2_8
  accelerate_v4_8 >> diffusers_v4_8

  task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-hf-fsmt-pjrt-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
