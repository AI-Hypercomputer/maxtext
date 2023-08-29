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


# TODO(ranran): Update reservation once bugs are closed: b/297217984 & b/297218059.
class Zone(enum.Enum):
  US_CENTRAL1_A = "us-central1-a"  # reservation for v2-32
  US_CENTRAL2_B = "us-central2-b"  # reservation for v4-8 & v4-32
  US_EAST1_D = "us-east1-d"  # reservation for v3-64


class RuntimeVersion(enum.Enum):
  VM_NIGHTLY = "1vm-nightly"
  TPU_UBUNTU2204_BASE = "tpu-ubuntu2204-base"
