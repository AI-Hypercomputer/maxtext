# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the XpkClusterConfig dataclass.

This file is separated to prevent circular dependencies between modules that
both need to reference cluster configuration details (e.g., maxtext_xpk_runner
and disruption_manager).
"""

import dataclasses


# This is needed to prevent circular imports.
@dataclasses.dataclass
class XpkClusterConfig:
  """Holds details for an XPK cluster to run workloads on.

  Attributes:
    cluster_name: The name of the GKE cluster.
    project: The Google Cloud project where the cluster is located.
    zone: The zone where the cluster is located.
    device_type: The type of TPU device in the cluster (e.g., 'v5litepod-256').
  """

  cluster_name: str
  project: str
  zone: str
  device_type: str
