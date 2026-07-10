"""Programmatic launcher for Non-SPMD DiLoCo on Shared Pathways Service.

Runs the test in the same process as `isc_pathways.connect` to avoid child-process
socket lock contention on the IFRT proxy port forwarder.
"""

import jax
from pathwaysutils.experimental.shared_pathways_service import isc_pathways
import sys
import os

sys.path.insert(0, os.path.abspath("tests"))
import diloco_colocated_cpu_test

from pathwaysutils.experimental.shared_pathways_service import gke_utils
import subprocess


def custom_check_pod_ready(pod_name: str) -> str:
  target = f"pod/{pod_name}" if not pod_name.startswith("pod/") else pod_name
  print(f"Waiting up to 300s for {target} to be ready...")
  wait_command = ["kubectl", "wait", "--for=condition=Ready", "--timeout=300s", "--", target]
  subprocess.run(wait_command, check=True)
  return pod_name


gke_utils.check_pod_ready = custom_check_pod_ready

print("=" * 70)
print("Connecting to Shared Pathways Service (V5p Cluster across 2 slices)...")
print("=" * 70)

with isc_pathways.connect(
    cluster="auto-v5p-8-bodaborg",
    project="cloud-tpu-multipod-dev",
    region="europe-west4",
    gcs_bucket="gs://rostam-sps-bucket",
    pathways_service="sps8-0512jx10-pathways-head-0-0.sps8-0512jx10:29001",
    expected_tpu_instances={"tpuv5:2x2x1": 2},
    proxy_server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260512_RC00-jax_0.10.0",
    collect_service_metrics=True,
):
  print("✓ Connected! JAX Devices:", jax.devices())
  print("=" * 70)
  diloco_colocated_cpu_test.main()
