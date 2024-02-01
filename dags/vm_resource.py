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

"""The file for common projects, zone, and runtime versions."""

import enum


V5_NETWORKS_PREFIX = "projects/tpu-prod-env-automated"
V5_NETWORKS = f"{V5_NETWORKS_PREFIX}/global/networks/mas-test"
V5E_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east1/subnetworks/mas-test"
V5P_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east5/subnetworks/mas-test"


class Project(enum.Enum):
  CLOUD_ML_AUTO_SOLUTIONS = "cloud-ml-auto-solutions"
  TPU_PROD_ENV_MULTIPOD = "tpu-prod-env-multipod"
  TPU_PROD_ENV_AUTOMATED = "tpu-prod-env-automated"


class ImageProject(enum.Enum):
  DEEP_LEARNING_PLATFORM_RELEASE = "deeplearning-platform-release"


class ImageFamily(enum.Enum):
  COMMON_CU121_DEBIAN_11 = "common-cu121-debian-11"


class Zone(enum.Enum):
  # reserved/on-demand v2-32 in cloud-ml-auto-solutions
  US_CENTRAL1_A = "us-central1-a"
  # on-demand v3-8 in cloud-ml-auto-solutions
  US_CENTRAL1_B = "us-central1-b"
  # reserved v4-8 & v4-32 in cloud-ml-auto-solutions
  US_CENTRAL2_B = "us-central2-b"
  # reserved/on-demand v2-8 in cloud-ml-auto-solutions
  US_CENTRAL1_C = "us-central1-c"
  # committed resource for A100
  US_CENTRAL1_F = "us-central1-f"
  # reserved v5e in tpu-prod-env-automated
  US_EAST1_C = "us-east1-c"
  # reserved v3-8 & reserved/on-demand v3-32 in cloud-ml-auto-solutions
  US_EAST1_D = "us-east1-d"
  # reserved v5p in tpu-prod-env-automated
  US_EAST5_A = "us-east5-a"


class MachineVersion(enum.Enum):
  N1_STANDARD_8 = "n1-standard-8"
  N1_STANDARD_32 = "n1-standard-32"
  A2_HIGHGPU_1G = "a2-highgpu-1g"
  A2_HIGHGPU_4G = "a2-highgpu-4g"
  A3_HIGHGPU_8G = "a3-highgpu-8g"
  G2_STAND_4 = "g2-standard-4"


class TpuVersion(enum.Enum):
  V2 = "2"
  V3 = "3"
  V4 = "4"
  V5E = "5litepod"
  V5P = "5p"


class GpuVersion(enum.Enum):
  L4 = "nvidia-l4"
  A100 = "nvidia-tesla-a100"
  H100 = "nvidia-h100-80gb"
  V100 = "nvidia-tesla-v100"


class RuntimeVersion(enum.Enum):
  TPU_VM_TF_NIGHTLY = "tpu-vm-tf-nightly"
  TPU_VM_TF_NIGHTLY_POD = "tpu-vm-tf-nightly-pod"
  TPU_VM_TF_2150_SE = "tpu-vm-tf-2.15.0-se"
  TPU_VM_TF_2150_POD_SE = "tpu-vm-tf-2.15.0-pod-se"
  TPU_VM_TF_2150_PJRT = "tpu-vm-tf-2.15.0-pjrt"
  TPU_VM_TF_2150_POD_PJRT = "tpu-vm-tf-2.15.0-pod-pjrt"
  TPU_UBUNTU2204_BASE = "tpu-ubuntu2204-base"
  TPU_VM_V4_BASE = "tpu-vm-v4-base"
  V2_ALPHA_TPUV5_LITE = "v2-alpha-tpuv5-lite"
  V2_ALPHA_TPUV5 = "v2-alpha-tpuv5"


class ClusterName(enum.Enum):
  V4_8_CLUSTER = "mas-v4-8"
  V4_32_CLUSTER = "mas-v4-32"
  V5E_4_CLUSTER = "mas-v5e-4"
  V5E_16_CLUSTER = "mas-v5e-16"
  V4_128_MULTISLICE_CLUSTER = "v4-bodaborg"
  V5E_16_MULTISLICE_CLUSTER = "v5e-16-bodaborg"
  V5E_256_MULTISLICE_CLUSTER = "v5e-256-bodaborg"


class DockerImage(enum.Enum):
  XPK_JAX_TEST = "gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest"
  XPK_MAXTEXT_TEST = "gcr.io/tpu-prod-env-multipod/xpk_maxtext_test:latest"
