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


PROJECT_CLOUD_ML_AUTO_SOLUTIONS = "cloud-ml-auto-solutions"


class Zone(enum.Enum):
  US_CENTRAL1_A = "us-central1-a"  # reservation for v2-32 in cloud-ml-auto-solutions
  US_CENTRAL2_B = "us-central2-b"  # reservation for v4-8 & v4-32 in cloud-ml-auto-solutions
  US_CENTRAL1_C = "us-central1-c"  # reservation for v2-8 in cloud-ml-auto-solutions
  US_EAST1_D = "us-east1-d"  # reservation for v3-8 & v3-32 in cloud-ml-auto-solutions


class RuntimeVersion(enum.Enum):
  TPU_VM_TF_NIGHTLY = "tpu-vm-tf-nightly"
  TPU_VM_TF_NIGHTLY_POD = "tpu-vm-tf-nightly-pod"
  TPU_UBUNTU2204_BASE = "tpu-ubuntu2204-base"
  TPU_VM_V4_BASE = "tpu-vm-v4-base"


class ClusterName(enum.Enum):
  V4_8_CLUSTER = "mas-v4-8"
  V4_32_CLUSTER = "mas-v4-32"
  V5E_4_CLUSTER = "mas-v5e-4"
  V5E_16_CLUSTER = "mas-v5e-16"


class DockerImage(enum.Enum):
  XPK_JAX_TEST = "gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest"
