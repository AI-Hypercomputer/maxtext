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
from airflow.decorators import task_group
from airflow import models
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env
from dags.vm_resource import Project, Zone, V5_NETWORKS, V5E_SUBNETWORKS


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
US_EAST1_D = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_EAST1_D.value,
    metric_config.DatasetOption.XLML_DATASET,
)


US_CENTRAL1 = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    # HACK: use region in place of zone, since clusters are regional
    zone="us-central1",
    dataset_name=...,
)

US_EAST1_C = gcp_config.GCPConfig(
    project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
    zone=Zone.US_EAST1_C.value,
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
)


@task_group(prefix_group_id=False)
def torchvision():
  mnist_v2_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch("pt-nightly-mnist-pjrt-func-v2-8-1vm"),
      US_CENTRAL1_C,
  ).run()
  resnet_v2_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v2-8-1vm",
          reserved=True,
      ),
      US_CENTRAL1_C,
  ).run()
  resnet_v3_8_tests = [
      task.TpuQueuedResourceTask(
          test_config.JSonnetTpuVmTest.from_pytorch(test, reserved=True),
          US_EAST1_D,
      ).run()
      for test in (
          "pt-nightly-resnet50-pjrt-fake-v3-8-1vm",
          "pt-nightly-resnet50-pjrt-ddp-fake-v3-8-1vm",
      )
  ]
  resnet_v4_8_tests = [
      task.TpuQueuedResourceTask(
          test_config.JSonnetTpuVmTest.from_pytorch(test),
          US_CENTRAL2_B,
      ).run()
      for test in (
          "pt-nightly-resnet50-pjrt-fake-v4-8-1vm",
          "pt-nightly-resnet50-pjrt-ddp-fake-v4-8-1vm",
          "pt-nightly-resnet50-spmd-batch-fake-v4-8-1vm",
          "pt-nightly-resnet50-spmd-spatial-fake-v4-8-1vm",
      )
  ]
  resnet_v4_32 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v4-32-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
  resnet_v5lp_4 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v5litepod-4-1vm",
          network=V5_NETWORKS,
          subnetwork=V5E_SUBNETWORKS,
          reserved=True,
      ),
      US_EAST1_C,
  ).run()

  mnist_v2_8 >> (resnet_v2_8, *resnet_v4_8_tests, resnet_v4_32, resnet_v5lp_4)
  resnet_v2_8 >> resnet_v3_8_tests

  resnet_v100_2x2 = task.GpuGkeTask(
      test_config.JSonnetGpuTest.from_pytorch("pt-nightly-resnet50-mp-fake-v100-x2x2"),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x2_spmd = task.GpuGkeTask(
      test_config.JSonnetGpuTest.from_pytorch(
          "pt-nightly-resnet50-spmd-batch-fake-v100-x2x2"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x2 >> resnet_v100_2x2_spmd

  resnet_v100_2x1_plugin = task.GpuGkeTask(
      test_config.JSonnetGpuTest.from_pytorch(
          "pt-nightly-resnet50-mp-plugin-fake-v100-x2x1"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x2_plugin = task.GpuGkeTask(
      test_config.JSonnetGpuTest.from_pytorch(
          "pt-nightly-resnet50-mp-plugin-fake-v100-x2x2"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x1_plugin >> resnet_v100_2x2_plugin


@task_group(prefix_group_id=False)
def huggingface():
  accelerate_v2_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-accelerate-smoke-v2-8-1vm", reserved=True
      ),
      US_CENTRAL1_C,
  ).run()
  accelerate_v4_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch("pt-nightly-accelerate-smoke-v4-8-1vm"),
      US_CENTRAL2_B,
  ).run()
  diffusers_v4_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-hf-diffusers-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()

  accelerate_v4_8 >> accelerate_v2_8
  accelerate_v4_8 >> diffusers_v4_8

  task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-hf-bert-pjrt-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()


@task_group(prefix_group_id=False)
def llama():
  llama_inference_v4_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-pjrt-infer-func-v4-8-1vm-1vm"
      ),
      US_CENTRAL2_B,
  ).run()
  llama_train_v4_8 = task.TpuQueuedResourceTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-pjrt-train-spmd-func-v4-8-1vm-1vm"
      ),
      US_CENTRAL2_B,
  ).run()


with models.DAG(
    dag_id="pytorchxla-nightly",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  torchvision()
  huggingface()
  llama()
