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

"""Generates a single-slice JobSet YAML manifest for a v7x, v6e, or v5e smoke test run."""

import argparse
import copy
import os
import re
import shlex
import sys
from typing import Any
import yaml

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import platforms  # pylint: disable=wrong-import-position

_MAX_JOBSET_NAME_LEN = 40
_HARNESS_PATH = "/tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py"


def _dns_slug(value: str) -> str:
  cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower().replace(".", "p")).strip("-")
  return re.sub(r"-+", "-", cleaned)


def _label_value(value: str) -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-_")[:63]
  return cleaned or "run"


def jobset_name(platform: str, model: str, run_id: str) -> str:
  """Returns a DNS-1035 JobSet name <= 40 chars so child pods stay within 63 chars."""
  plat_slug = _dns_slug(platform)
  model_slug = _dns_slug(model)
  run_slug = _dns_slug(run_id)
  if not plat_slug or not model_slug or not run_slug:
    raise ValueError(f"platform={platform!r}, model={model!r}, and run_id={run_id!r} must be non-empty")
  prefix = f"mesa-smoke-{plat_slug}-{model_slug}-"
  max_run_len = _MAX_JOBSET_NAME_LEN - len(prefix)
  if max_run_len < 1:
    full = f"mesa-smoke-{plat_slug}-{model_slug}-{run_slug}"
    return full[:_MAX_JOBSET_NAME_LEN].rstrip("-")
  return f"{prefix}{run_slug[:max_run_len].rstrip('-')}"


def _bootstrap_command(code_dir: str) -> str:
  return (
      'if [ "${SLICE_INDEX:-0}" = 0 ] && [ "$JOB_COMPLETION_INDEX" = 0 ]; then export ROLE=runner; '
      "else export ROLE=daemon; fi; "
      "mkdir -p /ramdisk/hf_cache /ramdisk/tmp; "
      'python3 -c "from etils import epath; '
      f"open('/tmp/bootstrap.py','wb').write(epath.Path('{code_dir.rstrip('/')}/bootstrap_tpu7x.py').read_bytes())\" "
      "&& python3 -u /tmp/bootstrap.py"
  )


def build_runner_script(
    *,
    profile: platforms.PlatformProfile,
    topology: str,
    model: str,
    run_id: str,
    scan_modes: list[str],
    pipelines: str,
    name: str,
) -> str:
  """Returns the bash command executed on pod slice0-0-0 (ROLE=runner)."""
  hosts = profile.hosts_per_slice(topology)
  devices = profile.devices_per_slice(topology)
  gcs_base = f"{profile.gcs_data_root.rstrip('/')}/smoke"
  pipeline_flags = {
      "both": "",
      "safetensors_dynamic": "  --skip-to-maxtext \\\n",
      "to_maxtext": "  --skip-safetensors \\\n",
  }[pipelines]
  work_dir = f"/ramdisk/bench_{_dns_slug(run_id)}"
  csv_local = f"/tmp/{name}.csv"
  csv_gcs = f"{gcs_base}/run_{run_id}/{name}.csv"
  quoted_scans = " ".join(shlex.quote(m) for m in scan_modes)
  banner = (
      f"=== [{name}] Smoke test on {profile.name} ({topology}: {hosts} hosts, {devices} devices) "
      "commit=${BENCH_GIT_COMMIT:-unknown} ==="
  )
  return f"""set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 KEEP_HF_CACHE=1 HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp
echo "{banner}"
python3 -u {_HARNESS_PATH} \\
  --models {shlex.quote(model)} \\
  --scan-modes {quoted_scans} \\
  --num-slices-list 1 \\
  --num-hosts-per-slice {hosts} \\
  --devices-per-slice {devices} \\
  --hardware tpu \\
  --gcs-base {shlex.quote(gcs_base)} \\
  --run-id {shlex.quote(run_id)} \\
  --local-work-dir {shlex.quote(work_dir)} \\
  --csv-report {shlex.quote(csv_local)} \\
  --reuse-model-hf-cache \\
{pipeline_flags}  --skip-completed \\
  --allow-step-failure
python3 - <<'EOF'
import os
from etils import epath
if os.path.exists({csv_local!r}):
  epath.Path({csv_gcs!r}).write_bytes(open({csv_local!r}, "rb").read())
EOF
echo "=== [{name}] SMOKE TEST COMPLETE ==="
"""


def build_smoke_jobset(
    platform: str,
    topology: str,
    model: str,
    run_id: str,
    scan_modes: list[str] | None = None,
    pipelines: str = "both",
) -> dict[str, Any]:
  """Builds the Kubernetes JobSet manifest dict for one smoke test run."""
  profile = platforms.get_profile(platform)
  hosts = profile.hosts_per_slice(topology)
  chips_per_host = profile.chips_per_host(topology)
  modes = list(scan_modes or ["scanned", "unscanned"])
  if pipelines not in ("both", "to_maxtext", "safetensors_dynamic"):
    raise ValueError(f"Unsupported --pipelines {pipelines!r}")

  name = jobset_name(platform, model, run_id)
  run_label = _label_value(run_id)
  base_labels = {**profile.pod_labels, "mesa-run-id": run_label}
  jobset_labels = dict(base_labels)
  pod_labels = dict(base_labels)
  if profile.kueue_queue:
    jobset_labels["kueue.x-k8s.io/queue-name"] = profile.kueue_queue
  if profile.workload_time_limit_minutes is not None:
    pod_labels["declared-duration-minutes"] = "60"

  pod_annotations: dict[str, str] = {}
  if profile.kueue_topology_aware_scheduling:
    pod_annotations.update(
        {
            "kueue.x-k8s.io/podset-slice-required-topology": "cloud.google.com/gke-nodepool",
            "kueue.x-k8s.io/podset-slice-size": str(hosts),
            "kueue.x-k8s.io/podset-unconstrained-topology": "true",
        }
    )

  daemons = [f"{name}-slice0-0-{h}.{name}" for h in range(1, hosts)]
  runner_script = build_runner_script(
      profile=profile,
      topology=topology,
      model=model,
      run_id=run_id,
      scan_modes=modes,
      pipelines=pipelines,
      name=name,
  )

  env_list: list[dict[str, Any]] = [
      {"name": "SLICE_INDEX", "value": "0"},
      {"name": "DAEMON_PORT", "value": "55555"},
      {"name": "BENCH_CODE_DIR", "value": profile.gcs_code_dir},
      {"name": "BENCH_PLATFORM", "value": profile.name},
      {"name": "BENCH_CLUSTER", "value": profile.gke_cluster_name},
      {"name": "BENCH_TOPOLOGY", "value": topology},
      {"name": "BENCH_RUN_ID", "value": run_id},
      {"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"},
      {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
      {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
      {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
      {
          "name": "HF_TOKEN",
          "valueFrom": {
              "secretKeyRef": {
                  "name": profile.hf_token_secret_name,
                  "key": "HF_TOKEN",
                  "optional": profile.hf_token_secret_optional,
              }
          },
      },
      {"name": "WAIT_FOR_DAEMON_HOST", "value": ",".join(daemons)},
      {"name": "RUNNER_SCRIPT", "value": runner_script},
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
              "command": ["bash", "-c", _bootstrap_command(profile.gcs_code_dir)],
              "env": env_list,
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
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {
          "name": name,
          "namespace": "default",
          "labels": jobset_labels,
      },
      "spec": {
          "failurePolicy": {"maxRestarts": 0},
          "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
          "replicatedJobs": [
              {
                  "name": "slice0",
                  "replicas": 1,
                  "template": {
                      "metadata": {
                          "labels": dict(base_labels),
                          "annotations": {
                              "alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool",
                          },
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
          ],
      },
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--platform", required=True, choices=["v7x", "v6e", "v5e"])
  p.add_argument("--topology", required=True)
  p.add_argument("--model", default="qwen3-0.6b")
  p.add_argument("--run-id", required=True)
  p.add_argument("--scan-modes", nargs="+", default=["scanned", "unscanned"], choices=["scanned", "unscanned"])
  p.add_argument("--pipelines", default="both", choices=["both", "to_maxtext", "safetensors_dynamic"])
  p.add_argument("--out", default=None, help="Output YAML path (prints to stdout if omitted)")
  args = p.parse_args(argv)

  doc = build_smoke_jobset(
      platform=args.platform,
      topology=args.topology,
      model=args.model,
      run_id=args.run_id,
      scan_modes=args.scan_modes,
      pipelines=args.pipelines,
  )
  rendered = yaml.safe_dump(doc, sort_keys=False)
  if args.out:
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(rendered)
  else:
    sys.stdout.write(rendered)
  return 0


if __name__ == "__main__":
  sys.exit(main())
