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

"""Generates Kubernetes JobSet YAML manifests for single-slice regression & multi-slice DeepSeek TPU parity."""

import argparse
import copy
import os
import pathlib
import re
import shlex
import sys
from typing import Any
import yaml

_BENCHMARK_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import platforms  # pylint: disable=wrong-import-position

_FALLBACK_OVERLAY_URI = "gs://mesa-maxtext/mesa-bench-v2/code/site_packages_overlay.tar.gz"
_DEFAULT_TOPOLOGY_BY_PLATFORM = {
    "v7x": "2x2x1",
    "v5p": "2x2x1",
    "v6e": "2x4",
    "v5e": "16x16",
}


def _dns_slug(value: str, max_len: int = 28) -> str:
  cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  cleaned = re.sub(r"-+", "-", cleaned)[:max_len].rstrip("-")
  if not cleaned:
    raise ValueError(f"Cannot derive valid DNS-1035 prefix from {value!r}")
  return cleaned


def _bootstrap_script(gcs_code_dir: str, python_cmd: str) -> str:
  code_dir = gcs_code_dir.rstrip("/")
  return f"""set -euo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export EPATH_USE_TF=false
mkdir -p /tmp/site_packages_raw /tmp/maxtext /ramdisk/tmp
export TMPDIR=/ramdisk/tmp
python3 - <<'PYEOF'
import json, os, shutil, site, tarfile
from etils import epath

overlay_candidates = [
    "{code_dir}/site_packages_overlay.tar.gz",
    "{_FALLBACK_OVERLAY_URI}",
]
overlay_uri = next((u for u in overlay_candidates if epath.Path(u).exists()), None)
if overlay_uri is not None:
  open("/tmp/ov.tar.gz", "wb").write(epath.Path(overlay_uri).read_bytes())
  with tarfile.open("/tmp/ov.tar.gz", "r:gz") as tf:
    tf.extractall("/tmp/site_packages_raw", filter="tar")
  target_sp = site.getsitepackages()[0]
  for item in os.listdir("/tmp/site_packages_raw"):
    src = os.path.join("/tmp/site_packages_raw", item)
    dst = os.path.join(target_sp, item)
    if not os.path.exists(dst):
      shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy2(src, dst)

tarball_uri = "{code_dir}/maxtext_lance_prs.tar.gz"
open("/tmp/maxtext_lance_prs.tar.gz", "wb").write(epath.Path(tarball_uri).read_bytes())
with tarfile.open("/tmp/maxtext_lance_prs.tar.gz", "r:gz") as tf:
  tf.extractall("/tmp/maxtext", filter="tar")
ver = json.load(open("/tmp/maxtext/CODE_VERSION.json", "r", encoding="utf-8"))
print(f"[Bootstrap] Extracted {{tarball_uri}} (git_commit={{ver.get('git_commit')}})", flush=True)
PYEOF

export PYTHONPATH=/tmp/maxtext/src:/tmp/maxtext/checkpoint_load_benchmark
{python_cmd}
"""


def _build_replicated_job(
    *,
    profile: platforms.PlatformProfile,
    jobset_name: str,
    slice_idx: int,
    num_slices: int,
    topology: str,
    bootstrap: str,
    multislice: bool,
    duration_minutes: int,
) -> dict[str, Any]:
  """Builds one `replicatedJobs` slice entry from `profile`."""
  hosts = profile.hosts_per_slice(topology)
  chips_per_host = profile.chips_per_host(topology)
  coord_host = f"{jobset_name}-slice0-0-0.{jobset_name}"

  base_labels = {**profile.pod_labels, "mesa-run-id": jobset_name}
  pod_labels = dict(base_labels)
  if profile.workload_time_limit_minutes is not None:
    pod_labels["declared-duration-minutes"] = str(duration_minutes)

  pod_annotations: dict[str, str] = {}
  if profile.kueue_topology_aware_scheduling:
    pod_annotations.update(
        {
            "kueue.x-k8s.io/podset-slice-required-topology": "cloud.google.com/gke-nodepool",
            "kueue.x-k8s.io/podset-slice-size": str(hosts),
            "kueue.x-k8s.io/podset-unconstrained-topology": "true",
        }
    )

  if multislice:
    env_vars = [
        {"name": "NUM_SLICES", "value": str(num_slices)},
        {"name": "SLICE_INDEX", "value": str(slice_idx)},
        {"name": "COORD_HOST", "value": coord_host},
    ]
  else:
    env_vars = [
        {"name": "NUM_HOSTS", "value": str(hosts)},
        {"name": "COORD_HOST", "value": coord_host},
    ]

  pod_spec: dict[str, Any] = {
      "restartPolicy": "Never",
      "nodeSelector": {**profile.node_selectors, "cloud.google.com/gke-tpu-topology": topology},
      "tolerations": copy.deepcopy(profile.tolerations),
      "volumes": [{"name": "ramdisk", "emptyDir": {"medium": "Memory"}}],
      "containers": [
          {
              "name": "worker",
              "image": profile.container_image,
              "imagePullPolicy": "IfNotPresent",
              "command": ["bash", "-c", bootstrap],
              "env": env_vars,
              "volumeMounts": [{"name": "ramdisk", "mountPath": "/ramdisk"}],
              "resources": {
                  "requests": {"google.com/tpu": str(chips_per_host)},
                  "limits": {"google.com/tpu": str(chips_per_host)},
              },
          }
      ],
  }
  if profile.priority_class_name is not None:
    pod_spec["priorityClassName"] = profile.priority_class_name
  if profile.service_account_name is not None:
    pod_spec["serviceAccountName"] = profile.service_account_name
  if profile.image_pull_secrets:
    pod_spec["imagePullSecrets"] = [{"name": s} for s in profile.image_pull_secrets]
  if profile.required_node_affinity:
    pod_spec["affinity"] = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": copy.deepcopy(profile.required_node_affinity)}]
            }
        }
    }

  return {
      "name": f"slice{slice_idx}",
      "replicas": 1,
      "template": {
          "metadata": {
              "labels": dict(base_labels),
              "annotations": {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"},
          },
          "spec": {
              "parallelism": hosts,
              "completions": hosts,
              "completionMode": "Indexed",
              "backoffLimit": 0,
              "template": {
                  "metadata": {"labels": pod_labels, "annotations": pod_annotations},
                  "spec": pod_spec,
              },
          },
      },
  }


def build_regression_jobset(
    *,
    profile: platforms.PlatformProfile,
    job_prefix: str,
    topology: str,
    gcs_code_dir: str,
    gcs_data_root: str,
    gcs_out_dir: str,
) -> dict[str, Any]:
  """Builds the single-slice 4-model regression JobSet manifest."""
  name = f"{job_prefix}-reg"
  out_json = f"{gcs_out_dir.rstrip('/')}/{name}_result.json"
  cmd = (
      "python3 -u /tmp/maxtext/checkpoint_load_benchmark/scripts/repro/tpu_verify_single_slice.py "
      f"--suite regression --repo-root /tmp/maxtext --gcs-root {shlex.quote(gcs_data_root)} "
      f"--out-gcs {shlex.quote(out_json)}"
  )
  bootstrap = _bootstrap_script(gcs_code_dir, cmd)
  jobset_labels = {**profile.pod_labels, "mesa-run-id": name}
  if profile.kueue_queue:
    jobset_labels["kueue.x-k8s.io/queue-name"] = profile.kueue_queue
  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {"name": name, "namespace": "default", "labels": jobset_labels},
      "spec": {
          "failurePolicy": {"maxRestarts": 0},
          "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
          "replicatedJobs": [
              _build_replicated_job(
                  profile=profile,
                  jobset_name=name,
                  slice_idx=0,
                  num_slices=1,
                  topology=topology,
                  bootstrap=bootstrap,
                  multislice=False,
                  duration_minutes=45,
              )
          ],
      },
  }


def build_multislice_deepseek_jobset(
    *,
    profile: platforms.PlatformProfile,
    job_prefix: str,
    topology: str,
    num_slices: int,
    suite: str,
    gcs_code_dir: str,
    gcs_data_root: str,
    gcs_out_dir: str,
) -> dict[str, Any]:
  """Builds the multi-slice (`num_slices` x `topology`) full 284B/671B DeepSeek parity JobSet manifest."""
  name = f"{job_prefix}-dsv"
  out_json = f"{gcs_out_dir.rstrip('/')}/{name}_result.json"
  cmd = (
      "python3 -u /tmp/maxtext/checkpoint_load_benchmark/scripts/repro/tpu_verify_multislice.py "
      f"--suite {shlex.quote(suite)} --num-slices {num_slices} --repo-root /tmp/maxtext "
      f"--gcs-root {shlex.quote(gcs_data_root)} --out-gcs {shlex.quote(out_json)} --resume"
  )
  bootstrap = _bootstrap_script(gcs_code_dir, cmd)
  jobset_labels = {**profile.pod_labels, "mesa-run-id": name}
  if profile.kueue_queue:
    jobset_labels["kueue.x-k8s.io/queue-name"] = profile.kueue_queue
  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {"name": name, "namespace": "default", "labels": jobset_labels},
      "spec": {
          "failurePolicy": {"maxRestarts": 4},
          "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
          "replicatedJobs": [
              _build_replicated_job(
                  profile=profile,
                  jobset_name=name,
                  slice_idx=slice_idx,
                  num_slices=num_slices,
                  topology=topology,
                  bootstrap=bootstrap,
                  multislice=True,
                  duration_minutes=60,
              )
              for slice_idx in range(num_slices)
          ],
      },
  }


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for generating parameterized TPU verification JobSet manifests."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--platform", default="v7x", choices=["v7x", "v5p", "v6e", "v5e"])
  parser.add_argument(
      "--job-prefix",
      default=f"{os.environ.get('USER', 'user')}-repro",
      help="DNS-safe prefix for generated JobSets (<prefix>-reg and <prefix>-dsv).",
  )
  parser.add_argument(
      "--topology",
      default=None,
      help="Slice topology (defaults to 2x2x1 on v7x/v5p, 2x4 on v6e, 16x16 on v5e).",
  )
  parser.add_argument(
      "--num-slices",
      type=int,
      default=4,
      help="Number of replicated slices for the multi-slice DeepSeek parity JobSet (default: 4).",
  )
  parser.add_argument(
      "--multislice-suite",
      default="both",
      choices=["dsv4", "dsv3", "both"],
      help="DeepSeek suite to run in the multi-slice JobSet.",
  )
  parser.add_argument(
      "--gcs-code-dir",
      required=True,
      help="GCS directory holding maxtext_lance_prs.tar.gz built by scripts/build_code_tarball.py.",
  )
  parser.add_argument(
      "--gcs-data-root",
      default=None,
      help="Regional GCS root containing hf_cache/ and hf_cache_bf16/ (defaults to platform regional root).",
  )
  parser.add_argument(
      "--gcs-out-dir",
      required=True,
      help="GCS directory where summary JSONs and step-digest caches will be written.",
  )
  parser.add_argument(
      "--out-dir",
      required=True,
      help="Local output directory for generated JobSet YAML files.",
  )
  args = parser.parse_args(argv)

  profile = platforms.get_profile(args.platform)
  prefix = _dns_slug(args.job_prefix)
  topology = args.topology or _DEFAULT_TOPOLOGY_BY_PLATFORM[args.platform]
  data_root = args.gcs_data_root or profile.regional_gcs_data_root or profile.gcs_data_root

  reg_doc = build_regression_jobset(
      profile=profile,
      job_prefix=prefix,
      topology=topology,
      gcs_code_dir=args.gcs_code_dir,
      gcs_data_root=data_root,
      gcs_out_dir=args.gcs_out_dir,
  )
  dsv_doc = build_multislice_deepseek_jobset(
      profile=profile,
      job_prefix=prefix,
      topology=topology,
      num_slices=args.num_slices,
      suite=args.multislice_suite,
      gcs_code_dir=args.gcs_code_dir,
      gcs_data_root=data_root,
      gcs_out_dir=args.gcs_out_dir,
  )

  os.makedirs(args.out_dir, exist_ok=True)
  reg_path = os.path.join(args.out_dir, f"{prefix}-reg.yaml")
  dsv_path = os.path.join(args.out_dir, f"{prefix}-dsv.yaml")
  with open(reg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(reg_doc, f, sort_keys=False)
  with open(dsv_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(dsv_doc, f, sort_keys=False)

  print(f"Wrote single-slice regression JobSet (1x {topology}) -> {reg_path}")
  print(f"Wrote multi-slice DeepSeek JobSet ({args.num_slices}x {topology}, suite={args.multislice_suite}) -> {dsv_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
