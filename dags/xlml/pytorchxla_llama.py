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
from configs import vm_resource


US_CENTRAL2_B = gcp_config.GCPConfig(
    vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
    vm_resource.Zone.US_CENTRAL2_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id="pytorchxla-llama",
    schedule=None,
    tags=["pytorchxla", "latest", "supported"],
    start_date=datetime.datetime(2023, 7, 12),
):
  llama_inference_v4_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-i-infer-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
  llama_train_v4_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-t-train-spmd-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
