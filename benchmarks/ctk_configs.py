# Copyright 2023–2026 Google LLC
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

"""Defines the ClusterConfig dataclass for Cluster Toolkit (gcluster) benchmarks.

This file is separated to prevent circular dependencies between modules that
both need to reference cluster configuration details (e.g., maxtext_ctk_runner
and disruption_manager).
"""

import dataclasses


# This is needed to prevent circular imports.
@dataclasses.dataclass
class ClusterConfig:
  """Holds details for a GKE cluster managed by Cluster Toolkit (gcluster).

  Attributes:
    cluster_name: The name of the GKE cluster.
    project: The Google Cloud project where the cluster is located.
    zone: The GCE zone where the cluster is located (e.g., 'us-central2-b').
    device_type: The type of TPU device in the cluster (e.g., 'v6e-256',
      'v5litepod-256', 'v5p-128').
    location: The GCP location (region or zone) passed to `gcluster
      --location` (e.g., 'us-central2' or 'us-central2-b'). Defaults to `zone`
      if not explicitly specified. For regional GKE clusters, `location` must be
      set to the cluster's region.
    compute_type: Optional explicit `gcluster --compute-type` override.
    topology: Optional explicit `gcluster --topology` override.
  """

  cluster_name: str
  project: str
  zone: str = ""
  device_type: str = ""
  location: str | None = None
  compute_type: str | None = None
  topology: str | None = None

  def __post_init__(self):
    if not self.location and self.zone:
      self.location = self.zone
    elif not self.zone and self.location and len(self.location.split("-")) == 3:
      self.zone = self.location


# Backward-compatible alias for existing callers and external DAGs.
XpkClusterConfig = ClusterConfig
