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

"""Builds stage-scoped GCP Log Explorer URLs and verifies anchor log lines via gcloud."""

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.parse


def build_logs_explorer_url(
    project: str,
    cluster: str,
    pod: str,
    start_iso: str,
    end_iso: str,
    container: str = "workload-container",
) -> str:
  """Constructs a time-bounded GCP Logs Explorer query URL for a specific worker pod."""
  query = (
      'resource.type="k8s_container"\n'
      f'resource.labels.cluster_name="{cluster}"\n'
      f'resource.labels.pod_name="{pod}"\n'
      f'resource.labels.container_name="{container}"'
  )
  enc_query = urllib.parse.quote(query, safe="")
  time_range = f"{start_iso}/{end_iso}"
  enc_time = urllib.parse.quote(time_range, safe="/:")
  return f"https://console.cloud.google.com/logs/query;query={enc_query};timeRange={enc_time}?project={project}"


def verify_stage_logs(
    project: str,
    cluster: str,
    pod: str,
    start_iso: str,
    end_iso: str,
    anchor: str,
    container: str = "workload-container",
) -> tuple[bool, int, list[str]]:
  """Queries Cloud Logging for the stage window and checks whether anchor text is present."""
  log_filter = (
      'resource.type="k8s_container" '
      f'resource.labels.cluster_name="{cluster}" '
      f'resource.labels.pod_name="{pod}" '
      f'resource.labels.container_name="{container}" '
      f'timestamp>="{start_iso}" timestamp<="{end_iso}"'
  )
  fd, local_log = tempfile.mkstemp(prefix="stage_log_", suffix=".txt")
  os.close(fd)
  try:
    cmd = [
        "gcloud",
        "logging",
        "read",
        log_filter,
        f"--project={project}",
        "--order=asc",
        "--format=value(timestamp,textPayload)",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
      return False, 0, [res.stderr.strip()]
    with open(local_log, "w", encoding="utf-8") as f:
      f.write(res.stdout)
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    matched = [line for line in lines if anchor in line] if anchor else lines
    return len(matched) > 0, len(lines), matched[:5]
  finally:
    if os.path.exists(local_log):
      os.remove(local_log)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument("--cluster", required=True, help="GKE cluster name")
  parser.add_argument("--pod", required=True, help="Worker-0 pod name")
  parser.add_argument("--container", default="workload-container", help="Container name")
  parser.add_argument(
      "--stage",
      action="append",
      required=True,
      help="Stage spec 'name|start_iso|end_iso|anchor_substring' (may be passed multiple times)",
  )
  parser.add_argument("--no-verify", action="store_true", help="Only print URLs without calling gcloud logging read")
  args = parser.parse_args()

  all_ok = True
  for spec in args.stage:
    parts = spec.split("|")
    if len(parts) != 4:
      print(f"Invalid --stage '{spec}'. Expected 'name|start_iso|end_iso|anchor'.", file=sys.stderr)
      return 2
    name, start_iso, end_iso, anchor = parts
    url = build_logs_explorer_url(args.project, args.cluster, args.pod, start_iso, end_iso, args.container)
    print(f"[{name}] {start_iso} .. {end_iso}\n  URL: {url}")
    if not args.no_verify:
      ok, total_lines, sample = verify_stage_logs(
          args.project, args.cluster, args.pod, start_iso, end_iso, anchor, args.container
      )
      status = "PASS" if ok else "FAIL"
      print(f"  Verify ({status}): {total_lines} lines in window, {len(sample)} anchor sample(s)")
      for s in sample:
        print(f"    {s}")
      if not ok:
        all_ok = False

  return 0 if all_ok else 1


if __name__ == "__main__":
  sys.exit(main())
