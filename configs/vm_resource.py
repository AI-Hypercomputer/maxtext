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
  VM_NIGHTLY = "1vm-nightly"
  VM_NIGHTLY_POD = "1vm-nightly-pod"
  TPU_UBUNTU2204_BASE = "tpu-ubuntu2204-base"
  TPU_VM_V4_BASE = "tpu-vm-v4-base"
