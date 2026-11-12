# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verified GKE TPU platform profiles for v7x, v6e, and v5e benchmark runs."""

from dataclasses import dataclass, field
import math
from typing import Any

CONTAINER_IMAGE = "gcr.io/cloud-tpu-shared-capacity/yaoyuchen-runner:ds41-2026-10-22-16-03-48"
DEFAULT_GCS_DATA_ROOT = "gs://mesa-maxtext/mesa-bench-v2"
DEFAULT_GCS_CODE_DIR = f"{DEFAULT_GCS_DATA_ROOT}/code"


@dataclass(frozen=True)
class PlatformProfile:
  """Cluster, queue, scheduling, and TPU hardware parameters for one benchmark platform."""

  name: str
  gke_cluster_name: str
  gcp_project_id: str
  gcp_region: str
  kube_context: str
  kueue_queue: str | None
  kueue_topology_aware_scheduling: bool
  priority_class_name: str | None
  node_selectors: dict[str, str]
  tolerations: list[dict[str, Any]]
  required_node_affinity: list[dict[str, Any]]
  service_account_name: str | None
  image_pull_secrets: tuple[str, ...]
  container_image: str
  tpu_chips_per_host_by_topology: dict[str, int]
  tpu_devices_per_chip: int
  hosts_per_slice_by_topology: dict[str, int]
  host_machine_type_by_topology: dict[str, str]
  host_ram_gib_by_topology: dict[str, int]
  hf_token_secret_name: str = "mesa-hf-token"
  hf_token_secret_optional: bool = True
  gcs_code_dir: str = DEFAULT_GCS_CODE_DIR
  gcs_data_root: str = DEFAULT_GCS_DATA_ROOT
  regional_gcs_data_root: str = ""
  pod_labels: dict[str, str] = field(default_factory=lambda: {"app": "mesa-bench-v2"})
  workload_time_limit_minutes: int | None = None
  capacity_type: str = "spot"
  zone_hint: str = ""
  cpu_node_pool: str = "cpu-np"
  cpu_machine_type: str = ""
  step1b_cpu_node_pool: str = ""
  step1b_cpu_machine_type: str = ""
  step2_ram_margin_ratio: float = 0.93
  min_avail_host_ram_gb: float = 8.5
  sparse_min_avail_host_ram_gb: float = 24.0
  default_phase_timeout_s: int = 14400
  default_barrier_timeout_s: int = 1800
  max_download_workers: int = 16
  max_concurrent_jobsets: int = 2

  def _require_topology(self, topology: str) -> None:
    if topology not in self.hosts_per_slice_by_topology:
      supported = ", ".join(sorted(self.hosts_per_slice_by_topology))
      raise ValueError(f"Unknown topology {topology!r} for platform {self.name!r} (expected one of: {supported})")

  def hosts_per_slice(self, topology: str) -> int:
    """Returns the number of VM hosts (Indexed Job pods) in one slice of `topology`."""
    self._require_topology(topology)
    return self.hosts_per_slice_by_topology[topology]

  def chips_per_host(self, topology: str) -> int:
    """Returns `google.com/tpu` chips requested by each host pod in `topology`."""
    self._require_topology(topology)
    return self.tpu_chips_per_host_by_topology[topology]

  def devices_per_slice(self, topology: str) -> int:
    """Returns total JAX devices in one slice across all hosts of `topology`."""
    self._require_topology(topology)
    try:
      chips = math.prod(int(part) for part in topology.split("x"))
    except ValueError as ex:
      raise ValueError(f"Invalid topology shape {topology!r} for platform {self.name!r}") from ex
    return chips * self.tpu_devices_per_chip


_V7X_TOPOLOGIES = ("2x2x1", "2x2x2", "2x2x4", "2x4x4")

PROFILES: dict[str, PlatformProfile] = {
    "v7x": PlatformProfile(
        name="v7x",
        gke_cluster_name="bodaborg-tpu7x-nap",
        gcp_project_id="cloud-tpu-shared-capacity",
        gcp_region="us-central1",
        kube_context="gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap",
        kueue_queue="default",
        kueue_topology_aware_scheduling=True,
        priority_class_name=None,
        node_selectors={
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            "cloud.google.com/gke-spot": "true",
            "tpu7x.gke.io/capacity": "spot",
        },
        tolerations=[
            {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
            {"key": "cloud.google.com/gke-spot", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
        ],
        required_node_affinity=[{"key": "cloud.google.com/reservation-name", "operator": "DoesNotExist"}],
        service_account_name=None,
        image_pull_secrets=("gitimg",),
        container_image=CONTAINER_IMAGE,
        tpu_chips_per_host_by_topology={t: 4 for t in _V7X_TOPOLOGIES},
        tpu_devices_per_chip=2,
        hosts_per_slice_by_topology={"2x2x1": 1, "2x2x2": 2, "2x2x4": 4, "2x4x4": 8},
        host_machine_type_by_topology={t: "tpu7x-standard-4t" for t in _V7X_TOPOLOGIES},
        host_ram_gib_by_topology={t: 960 for t in _V7X_TOPOLOGIES},
        regional_gcs_data_root="gs://mesa-maxtext-us-central1/mesa-bench-v2",
        workload_time_limit_minutes=180,
        capacity_type="spot",
        zone_hint="us-central1-c",
        cpu_node_pool="cpu-np",
        cpu_machine_type="n2d-standard-64",
        step1b_cpu_node_pool="job-manager-np",
        step1b_cpu_machine_type="n2d-standard-224",
        step2_ram_margin_ratio=0.93,
        min_avail_host_ram_gb=8.5,
        sparse_min_avail_host_ram_gb=24.0,
        default_phase_timeout_s=3600,
        default_barrier_timeout_s=1800,
        max_download_workers=16,
        max_concurrent_jobsets=2,
    ),
    "v6e": PlatformProfile(
        name="v6e",
        gke_cluster_name="bodaborg-v6e-nap",
        gcp_project_id="cloud-tpu-shared-capacity",
        gcp_region="southamerica-west1",
        kube_context="gke_cloud-tpu-shared-capacity_southamerica-west1_bodaborg-v6e-nap",
        kueue_queue="default",
        kueue_topology_aware_scheduling=False,
        priority_class_name="medium",
        node_selectors={"cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice"},
        tolerations=[{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
        required_node_affinity=[],
        service_account_name=None,
        image_pull_secrets=(),
        container_image=CONTAINER_IMAGE,
        tpu_chips_per_host_by_topology={"2x4": 8, "2x2": 4, "4x4": 4, "8x8": 4, "16x16": 4},
        tpu_devices_per_chip=1,
        hosts_per_slice_by_topology={"2x4": 1, "2x2": 1, "4x4": 4, "8x8": 16, "16x16": 64},
        host_machine_type_by_topology={
            "2x4": "ct6e-standard-8t",
            "2x2": "ct6e-standard-4t",
            "4x4": "ct6e-standard-4t",
            "8x8": "ct6e-standard-4t",
            "16x16": "ct6e-standard-4t",
        },
        host_ram_gib_by_topology={"2x4": 1440, "2x2": 720, "4x4": 720, "8x8": 720, "16x16": 720},
        regional_gcs_data_root="gs://mesa-maxtext-southamerica-west1/mesa-bench-v2",
        workload_time_limit_minutes=180,
        capacity_type="on_demand",
        zone_hint="southamerica-west1-a",
        cpu_node_pool="cpu-np",
        cpu_machine_type="n4-standard-64",
        step2_ram_margin_ratio=0.93,
        min_avail_host_ram_gb=8.5,
        sparse_min_avail_host_ram_gb=24.0,
        default_phase_timeout_s=3600,
        default_barrier_timeout_s=1800,
        max_download_workers=16,
        max_concurrent_jobsets=2,
    ),
    "v5e": PlatformProfile(
        name="v5e",
        gke_cluster_name="v5e-256-bodaborg-europe-west4",
        gcp_project_id="tpu-prod-env-multipod",
        gcp_region="europe-west4",
        kube_context="gke_tpu-prod-env-multipod_europe-west4_v5e-256-bodaborg-europe-west4",
        kueue_queue=None,
        kueue_topology_aware_scheduling=False,
        priority_class_name=None,
        node_selectors={"cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice"},
        tolerations=[{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
        required_node_affinity=[],
        service_account_name="mesa-bench",
        image_pull_secrets=(),
        container_image=CONTAINER_IMAGE,
        tpu_chips_per_host_by_topology={"16x16": 4},
        tpu_devices_per_chip=1,
        hosts_per_slice_by_topology={"16x16": 64},
        host_machine_type_by_topology={"16x16": "ct5lp-hightpu-4t"},
        host_ram_gib_by_topology={"16x16": 192},
        regional_gcs_data_root="gs://mesa-maxtext-europe-west4/mesa-bench-v2",
        workload_time_limit_minutes=None,
        capacity_type="reserved",
        zone_hint="europe-west4-b",
        cpu_node_pool="cpu-np",
        cpu_machine_type="n2-standard-64",
        step2_ram_margin_ratio=0.90,
        min_avail_host_ram_gb=6.5,
        sparse_min_avail_host_ram_gb=18.0,
        default_phase_timeout_s=3600,
        default_barrier_timeout_s=1800,
        max_download_workers=16,
        max_concurrent_jobsets=1,
    ),
}


def get_profile(name: str) -> PlatformProfile:
  """Returns the verified PlatformProfile for `name` or raises ValueError."""
  if name not in PROFILES:
    supported = ", ".join(sorted(PROFILES))
    raise ValueError(f"Unknown platform {name!r} (expected one of: {supported})")
  return PROFILES[name]
