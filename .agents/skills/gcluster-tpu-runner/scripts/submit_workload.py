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

"""Wrapper for gcluster TPU job submission with .dockerignore and /deps overlay checks."""

import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight_check  # pylint: disable=wrong-import-position


def build_gcluster_command(
    cluster: str,
    zone: str,
    project: str,
    device_type: str,
    num_slices: int,
    job_name: str,
    base_image: str,
    wrapped_command: str,
) -> list[str]:
  """Builds the gcluster job submit command line."""
  return [
      "gcluster",
      "job",
      "submit",
      f"--cluster={cluster}",
      f"--zone={zone}",
      f"--project={project}",
      f"--tpu-type={device_type}",
      f"--num-slices={num_slices}",
      f"--job-name={job_name}",
      f"--base-image={base_image}",
      "--",
      "bash",
      "-c",
      wrapped_command,
  ]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cluster", required=True, help="GKE TPU cluster name")
  parser.add_argument("--zone", required=True, help="GCP zone")
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument("--device-type", required=True, help="TPU device type (e.g. v5p-128, v6e-256)")
  parser.add_argument("--num-slices", type=int, default=1, help="Number of slices")
  parser.add_argument("--job-name", required=True, help="Unique workload name")
  parser.add_argument("--base-image", required=True, help="Base Docker image URI")
  parser.add_argument("--maxtext-root", default=".", help="Path to MaxText repository root")
  parser.add_argument("--command", required=True, help="Workload command to run inside the TPU container")
  parser.add_argument("--no-deps-overlay", action="store_true", help="Do not prepend /src -> /deps copy")
  parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
  args = parser.parse_args()

  root = os.path.abspath(args.maxtext_root)
  di_ok, missing = preflight_check.validate_dockerignore(root, fix=True)
  if not di_ok:
    print(f"Error: .dockerignore missing required patterns: {missing}", file=sys.stderr)
    return 1

  py_ok, py_errs = preflight_check.check_python_syntax(root)
  if not py_ok:
    for err in py_errs:
      print(err, file=sys.stderr)
    return 1

  wrapped = preflight_check.wrap_deps_overlay_command(args.command, use_deps_overlay=not args.no_deps_overlay)
  cmd = build_gcluster_command(
      cluster=args.cluster,
      zone=args.zone,
      project=args.project,
      device_type=args.device_type,
      num_slices=args.num_slices,
      job_name=args.job_name,
      base_image=args.base_image,
      wrapped_command=wrapped,
  )
  print("Command:", " ".join(shlex.quote(c) for c in cmd))
  if args.dry_run:
    return 0
  return subprocess.run(cmd, cwd=root, check=False).returncode


if __name__ == "__main__":
  sys.exit(main())
