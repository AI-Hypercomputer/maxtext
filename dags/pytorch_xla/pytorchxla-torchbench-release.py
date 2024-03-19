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

"""A DAG to run all TorchBench tests with nightly version."""

from airflow import models
import datetime
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
import dags.vm_resource as resource

SCHEDULED_TIME = None


with models.DAG(
    dag_id="pytorchxla-torchbench-release",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "release", "torchbench"],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  model = "all" if composer_env.is_prod_env() else "BERT_pytorch"
  torchbench_extra_flags = [f"--filter={model}"]
  test_version = config.VERSION.R2_2
  # Running on V4-8:
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V4,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_AUTO_SOLUTIONS,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      test_version=test_version,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5P
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V5P,
      tpu_cores=8,
      project=resource.Project.TPU_PROD_ENV_AUTOMATED,
      tpu_zone=resource.Zone.US_EAST5_A,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV5,
      network=resource.V5_NETWORKS,
      subnetwork=resource.V5P_SUBNETWORKS,
      test_version=test_version,
      time_out_in_min=700,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5E
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V5E,
      tpu_cores=4,
      project=resource.Project.TPU_PROD_ENV_AUTOMATED,
      tpu_zone=resource.Zone.US_EAST1_C,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE,
      network=resource.V5_NETWORKS,
      subnetwork=resource.V5E_SUBNETWORKS,
      test_version=test_version,
      time_out_in_min=1600,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V100 GPU
  config.get_torchbench_gpu_config(
      machine_type=resource.MachineVersion.N1_STANDARD_8,
      image_project=resource.ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=resource.ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=resource.GpuVersion.V100,
      count=1,
      gpu_zone=resource.Zone.US_CENTRAL1_C,
      test_version=test_version,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on A100 GPU
  config.get_torchbench_gpu_config(
      machine_type=resource.MachineVersion.A2_HIGHGPU_1G,
      image_project=resource.ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=resource.ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=resource.GpuVersion.A100,
      count=1,
      gpu_zone=resource.Zone.US_CENTRAL1_F,
      test_version=test_version,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on H100 GPU
  # Note: H100 must use ssd.
  config.get_torchbench_gpu_config(
      machine_type=resource.MachineVersion.A3_HIGHGPU_8G,
      image_project=resource.ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=resource.ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=resource.GpuVersion.H100,
      count=8,
      gpu_zone=resource.Zone.US_CENTRAL1_A,
      nvidia_driver_version="535.86.10",
      test_version=test_version,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on L4 GPU
  config.get_torchbench_gpu_config(
      machine_type=resource.MachineVersion.G2_STAND_4,
      image_project=resource.ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=resource.ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=resource.GpuVersion.L4,
      count=1,
      gpu_zone=resource.Zone.US_CENTRAL1_C,
      test_version=test_version,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()
