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
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env
from dags.vm_resource import Project, Zone


# Run once a day at 2 pm UTC (6 am PST)
SCHEDULED_TIME = "0 14 * * *" if composer_env.is_prod_env() else None
US_CENTRAL1_C = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_CENTRAL1_C.value,
    metric_config.DatasetOption.XLML_DATASET,
)
US_CENTRAL2_B = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_CENTRAL2_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)

US_CENTRAL1 = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    # HACK: use region in place of zone, since clusters are regional
    zone="us-central1",
    dataset_name=...,
)


with models.DAG(
    dag_id="pytorchxla-torchvision",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  mnist_v2_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch("pt-nightly-mnist-pjrt-func-v2-8-1vm"),
      US_CENTRAL1_C,
  ).run()
  resnet_v2_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v2-8-1vm"
      ),
      US_CENTRAL1_C,
  ).run()
  resnet_v4_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()

  mnist_v2_8 >> resnet_v2_8
  mnist_v2_8 >> resnet_v4_8

  resnet_v100_2x2 = task.GpuGkeTask(
      test_config.JSonnetGpuTest.from_pytorch("pt-nightly-resnet50-mp-fake-v100-x2x2"),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()
