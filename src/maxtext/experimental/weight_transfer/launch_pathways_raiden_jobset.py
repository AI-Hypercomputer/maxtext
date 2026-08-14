#!/usr/bin/env python3
"""Launcher for Standard 2-Slice Pathways TPU Raiden JobSets on GKE.

Deploys a standard 2-slice (8 TPU v5p devices) JobSet on Google Cloud GKE,
waits for all pods to be scheduled onto GKE nodes, and generates direct Cloud
Logging query links with precise pod/container filters.
"""

from datetime import datetime
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Sequence
import urllib.parse

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "jobset_name",
    "",
    "Name for the JobSet. Defaults to raiden-2slice-YYYYMMDD-HHMMSS.",
)
flags.DEFINE_string(
    "cluster",
    "auto-v5p-8-bodaborg",
    "GKE cluster name.",
)
flags.DEFINE_string(
    "region",
    "europe-west4",
    "GCP region of the cluster.",
)
flags.DEFINE_string(
    "project",
    "cloud-tpu-multipod-dev",
    "GCP project ID.",
)
flags.DEFINE_string(
    "namespace",
    "default",
    "Kubernetes namespace.",
)
flags.DEFINE_integer(
    "timeout_seconds",
    300,
    "Maximum time in seconds to wait for pods to be scheduled.",
)
flags.DEFINE_string(
    "yaml_template",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pathways_2slice_raiden_jobset.yaml",
    ),
    "Path to standard Pathways JobSet YAML template.",
)


def generate_cloud_logging_url(
    project: str,
    cluster: str,
    namespace: str,
    jobset_name: str,
    container_name: str = "",
) -> str:
  """Builds a direct Cloud Logging URL with targeted resource and label filters."""
  query_lines = [
      'resource.type="k8s_container"',
      f'resource.labels.cluster_name="{cluster}"',
      f'resource.labels.namespace_name="{namespace}"',
      f'labels."k8s-pod/jobset-name"="{jobset_name}"',
  ]
  if container_name:
    query_lines.append(f'resource.labels.container_name="{container_name}"')

  query_str = "\n".join(query_lines)
  encoded_query = urllib.parse.quote(query_str, safe="")
  return f"https://console.cloud.google.com/logs/query;query={encoded_query};" f"duration=PT1H?project={project}"


def wait_for_jobset_pods_scheduled(
    jobset_name: str,
    namespace: str,
    expected_pods: int = 3,
    timeout_s: int = 300,
) -> List[Dict[str, Any]]:
  """Waits for JobSet pods to be created and scheduled onto GKE nodes."""
  print(f"\n3. Waiting up to {timeout_s}s for all {expected_pods} JobSet pods to" " be scheduled on GKE nodes...")
  start_time = time.time()
  last_report = ""

  while time.time() - start_time < timeout_s:
    cmd = [
        "kubectl",
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        f"jobset.sigs.k8s.io/jobset-name={jobset_name}",
        "-o",
        "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if res.returncode == 0 and res.stdout.strip():
      try:
        data = json.loads(res.stdout)
        items = data.get("items", [])
        scheduled_count = 0
        running_count = 0
        pod_details = []

        for item in items:
          p_name = item.get("metadata", {}).get("name", "")
          p_phase = item.get("status", {}).get("phase", "Pending")
          node_name = item.get("spec", {}).get("nodeName", "")

          if node_name:
            scheduled_count += 1
          if p_phase == "Running":
            running_count += 1

          pod_details.append(
              {
                  "name": p_name,
                  "phase": p_phase,
                  "node": node_name or "Pending Assignment",
              }
          )

        status_line = (
            f"   [Progress] {scheduled_count}/{expected_pods} pods scheduled,"
            f" {running_count}/{expected_pods} Running (Elapsed:"
            f" {int(time.time() - start_time)}s)"
        )
        if status_line != last_report:
          print(status_line, flush=True)
          last_report = status_line

        if scheduled_count >= expected_pods:
          print(f"\n✓ All {expected_pods} pods successfully scheduled on GKE" " nodes!")
          print("-" * 85)
          print(f"{'Pod Name':<45} | {'Status':<12} | {'Assigned Node'}")
          print("-" * 85)
          for p in pod_details:
            print(f"{p['name']:<45} | {p['phase']:<12} | {p['node']}")
          print("-" * 85 + "\n")
          return pod_details

      except (json.JSONDecodeError, KeyError):
        pass

    # Check if JobSet reached a terminal Failed state
    cmd_js = ["kubectl", "get", "jobset", jobset_name, "-n", namespace, "-o", "json"]
    res_js = subprocess.run(cmd_js, capture_output=True, text=True, check=False)
    if res_js.returncode == 0 and res_js.stdout.strip():
      try:
        js_data = json.loads(res_js.stdout)
        conditions = js_data.get("status", {}).get("conditions", [])
        for c in conditions:
          if c.get("type") == "Failed" and c.get("status") == "True":
            msg = c.get("message", "JobSet failed")
            print(f"\n❌ [ERROR] JobSet encountered a failure: {msg}")
            return []
      except (json.JSONDecodeError, KeyError):
        pass

    time.sleep(3.0)

  print(f"\n⚠ Warning: Timed out after {timeout_s}s waiting for all" f" {expected_pods} pods to be scheduled.")
  return []


def launch_jobset() -> None:
  """Submits the JobSet to GKE, waits for scheduling, and prints Cloud Logging URLs."""
  now_str = datetime.utcnow().strftime("%H%M%S")
  jobset_name = FLAGS.jobset_name or f"r2s-{now_str}"

  if not os.path.isfile(FLAGS.yaml_template):
    raise FileNotFoundError(f"Template YAML not found at: {FLAGS.yaml_template}")

  with open(FLAGS.yaml_template, "r", encoding="utf-8") as f:
    template_content = f.read()

  rendered_yaml = template_content.replace("JOB_NAME_PLACEHOLDER", jobset_name).replace(
      "JOBSET_NAME_PLACEHOLDER", jobset_name
  )

  tmp_yaml_path = f"/tmp/{jobset_name}.yaml"
  with open(tmp_yaml_path, "w", encoding="utf-8") as f:
    f.write(rendered_yaml)

  print("=" * 90)
  print("[Pathways JobSet Launcher] Submitting 2-Slice TPU Raiden Workload...")
  print(f"  JobSet Name : {jobset_name}")
  print(f"  Cluster     : {FLAGS.cluster} ({FLAGS.region})")
  print(f"  Project     : {FLAGS.project}")
  print(f"  Namespace   : {FLAGS.namespace}")
  print("  Topology    : 2 Slices of TPU v5p (8 chips total: 4 Trainer, 4 Sampler)")
  print("=" * 90)

  # 1. Fetch credentials
  print("\n1. Authenticating to GKE cluster...")
  cmd_auth = [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      FLAGS.cluster,
      f"--region={FLAGS.region}",
      f"--project={FLAGS.project}",
  ]
  subprocess.run(cmd_auth, check=True)

  # 2. Apply JobSet
  print("\n2. Deploying JobSet manifest...")
  cmd_apply = ["kubectl", "apply", "-f", tmp_yaml_path, "-n", FLAGS.namespace]
  subprocess.run(cmd_apply, check=True)

  # 3. Wait for Pods to be Scheduled
  wait_for_jobset_pods_scheduled(
      jobset_name=jobset_name,
      namespace=FLAGS.namespace,
      expected_pods=3,
      timeout_s=FLAGS.timeout_seconds,
  )

  # 4. Generate Cloud Logging Links
  main_logs_url = generate_cloud_logging_url(
      project=FLAGS.project,
      cluster=FLAGS.cluster,
      namespace=FLAGS.namespace,
      jobset_name=jobset_name,
      container_name="jax-tpu",
  )
  proxy_logs_url = generate_cloud_logging_url(
      project=FLAGS.project,
      cluster=FLAGS.cluster,
      namespace=FLAGS.namespace,
      jobset_name=jobset_name,
      container_name="pathways-proxy",
  )
  worker_logs_url = generate_cloud_logging_url(
      project=FLAGS.project,
      cluster=FLAGS.cluster,
      namespace=FLAGS.namespace,
      jobset_name=jobset_name,
      container_name="pathways-worker",
  )
  all_logs_url = generate_cloud_logging_url(
      project=FLAGS.project,
      cluster=FLAGS.cluster,
      namespace=FLAGS.namespace,
      jobset_name=jobset_name,
  )

  print("=" * 90)
  print("DIRECT CLOUD LOGGING LINKS (FILTERED FOR THIS JOBSET):")
  print("=" * 90)
  print(f"  • Main Benchmark Runner Logs : {main_logs_url}")
  print(f"  • Pathways Proxy Server Logs : {proxy_logs_url}")
  print(f"  • TPU Worker Slices Logs     : {worker_logs_url}")
  print(f"  • All JobSet Containers Logs : {all_logs_url}")

  print("\n" + "-" * 90)
  print("COMMAND LINE MONITORING COMMANDS:")
  print("-" * 90)
  print("  # Watch Pod Status:")
  print(f"  kubectl get pods -n {FLAGS.namespace} -l" f" 'jobset.sigs.k8s.io/jobset-name={jobset_name}' -w")
  print("\n  # Stream Main Benchmark Output in Terminal:")
  print(f"  kubectl logs -n {FLAGS.namespace}" f" pod/{jobset_name}-pathways-head-0-0 -c jax-tpu -f")
  print("\n  # Stream TPU Worker Slice Logs in Terminal:")
  print(f"  kubectl logs -n {FLAGS.namespace} pod/{jobset_name}-worker-0-0 -c" " pathways-worker -f")
  print("\n  # Delete JobSet when complete:")
  print(f"  kubectl delete jobset {jobset_name} -n {FLAGS.namespace}")
  print("=" * 90 + "\n")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  launch_jobset()


if __name__ == "__main__":
  app.run(main)
