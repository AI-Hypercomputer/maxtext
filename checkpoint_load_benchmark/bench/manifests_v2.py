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

"""Profile-driven Kubernetes Job, JobSet, PVC, and GCS task-queue generators for v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import re
import shlex
from typing import Any

from bench import platforms
from bench import schema_v2

_MAX_JOBSET_NAME_LEN = 40
_MAX_JOB_NAME_LEN = 63
_HARNESS_REL = "checkpoint_load_benchmark/benchmark_ckpt_load_time.py"
_STEP1_REL = "checkpoint_load_benchmark/bench/step1_download_job.py"
_QUEUE_REL = "checkpoint_load_benchmark/scripts/tpu7x_task_queue.py"

_TIER_ORDER: dict[str, int] = {"S": 0, "A": 1, "C": 2, "B": 3}


def dns_slug(value: str) -> str:
  """Converts `value` into a lowercase alphanumeric hyphen-separated DNS slug."""
  cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower().replace(".", "p")).strip("-")
  return re.sub(r"-+", "-", cleaned) or "item"


def k8s_name(parts: Sequence[str], *, max_len: int = _MAX_JOB_NAME_LEN) -> str:
  """Builds a bounded RFC-1035 DNS label starting with `mesa-`."""
  joined = "-".join(dns_slug(p) for p in parts if p)
  if not joined.startswith("mesa-"):
    joined = f"mesa-{joined}"
  return joined[:max_len].rstrip("-")


def _label_value(value: str) -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-_")[:63]
  return cleaned or "run"


def _bootstrap_shell_cmd(code_dir: str, *, tpu_jobset: bool = False) -> str:
  role_prefix = (
      'if [ "${SLICE_INDEX:-0}" = 0 ] && [ "${JOB_COMPLETION_INDEX:-0}" = 0 ]; then export ROLE=runner; '
      "else export ROLE=daemon; fi; "
      if tpu_jobset
      else "export ROLE=runner; "
  )
  return (
      f"{role_prefix}"
      "mkdir -p /ramdisk/hf_cache /ramdisk/tmp; "
      'python3 -c "from etils import epath; '
      f"open('/tmp/bootstrap.py','wb').write(epath.Path('{code_dir.rstrip('/')}/bootstrap_tpu7x.py').read_bytes())\" "
      "&& python3 -u /tmp/bootstrap.py"
  )


def _hf_token_env(profile: platforms.PlatformProfile, *, required: bool = False) -> dict[str, Any]:
  return {
      "name": "HF_TOKEN",
      "valueFrom": {
          "secretKeyRef": {
              "name": profile.hf_token_secret_name,
              "key": "HF_TOKEN",
              "optional": False if required else profile.hf_token_secret_optional,
          }
      },
  }


def _downward_api_envs() -> list[dict[str, Any]]:
  return [
      {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
      {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
      {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
  ]


def _step1_cpu_tolerations(node_pool: str) -> list[dict[str, Any]]:
  """Returns tolerations compatible with GKE `cpu-np` / `job-manager-np` taints."""
  tols: list[dict[str, Any]] = [
      {"key": "cloud.google.com/gke-spot", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
  ]
  if node_pool == "job-manager-np":
    tols.append({"key": "dedicated", "operator": "Equal", "value": "job-manager", "effect": "NoSchedule"})
  elif node_pool:
    tols.append(
        {"key": "cloud.google.com/gke-nodepool", "operator": "Equal", "value": node_pool, "effect": "NoSchedule"}
    )
  return tols


def _build_cpu_job(
    *,
    name: str,
    profile: platforms.PlatformProfile,
    node_pool: str,
    runner_script: str,
    labels: Mapping[str, str],
    memory_request: str = "64Gi",
    cpu_request: str = "16",
    extra_volumes: Sequence[dict[str, Any]] = (),
    extra_mounts: Sequence[dict[str, Any]] = (),
    extra_env: Sequence[dict[str, Any]] = (),
    require_hf_token: bool = False,
) -> dict[str, Any]:
  """Constructs a `batch/v1` CPU Kubernetes Job manifest honoring `profile` cluster constraints."""
  pod_spec: dict[str, Any] = {
      "restartPolicy": "Never",
      "nodeSelector": {"cloud.google.com/gke-nodepool": node_pool},
      "tolerations": _step1_cpu_tolerations(node_pool),
      "volumes": [{"name": "ramdisk", "emptyDir": {"medium": "Memory"}}, *extra_volumes],
      "containers": [
          {
              "name": "worker",
              "image": profile.container_image,
              "imagePullPolicy": "IfNotPresent",
              "command": ["bash", "-c", _bootstrap_shell_cmd(profile.gcs_code_dir, tpu_jobset=False)],
              "env": [
                  {"name": "BENCH_CODE_DIR", "value": profile.gcs_code_dir},
                  {"name": "BENCH_PLATFORM", "value": profile.name},
                  {"name": "BENCH_CLUSTER", "value": profile.gke_cluster_name},
                  {"name": "BENCH_PROJECT", "value": profile.gcp_project_id},
                  {"name": "BENCH_REGION", "value": profile.gcp_region},
                  {"name": "BENCH_NODE_POOL", "value": node_pool},
                  {"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"},
                  *_downward_api_envs(),
                  _hf_token_env(profile, required=require_hf_token),
                  *extra_env,
                  {"name": "RUNNER_SCRIPT", "value": runner_script},
              ],
              "volumeMounts": [{"name": "ramdisk", "mountPath": "/ramdisk"}, *extra_mounts],
              "resources": {
                  "requests": {"cpu": cpu_request, "memory": memory_request},
                  "limits": {"memory": memory_request},
              },
          }
      ],
  }
  if profile.service_account_name is not None:
    pod_spec["serviceAccountName"] = profile.service_account_name
  if profile.image_pull_secrets:
    pod_spec["imagePullSecrets"] = [{"name": s} for s in profile.image_pull_secrets]

  return {
      "apiVersion": "batch/v1",
      "kind": "Job",
      "metadata": {"name": name, "namespace": "default", "labels": dict(labels)},
      "spec": {
          "backoffLimit": 1,
          "template": {"metadata": {"labels": dict(labels)}, "spec": pod_spec},
      },
  }


def regional_hf_cache_dir(profile: platforms.PlatformProfile, model_name: str, *, dequantized: bool = False) -> str:
  """Returns the canonical regional GCS path holding `.safetensors` weights for `model_name`."""
  root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")
  sub = "hf_cache_bf16" if dequantized else "hf_cache"
  return f"{root}/{sub}/{model_name}"


def build_step1_cpu_job(
    *,
    platform: str,
    model_name: str,
    run_id: str,
    clean_target: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
  """Builds a regional CPU Kubernetes Job for cold Step 1 (`HF -> GCS`) download."""
  profile = platforms.get_profile(platform)
  cfg = schema_v2.load_models_v2()
  model_info = next((m for m in cfg["models"] if m["model_name"] == model_name), None)
  if model_info is None:
    raise ValueError(f"Unknown model {model_name!r} in models_v2.json")

  gcs_cache_dir = regional_hf_cache_dir(profile, model_name, dequantized=False)
  root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")
  result_gcs_uri = f"{root}/step1/{model_name}_step1_result.json"
  n_workers = workers or profile.max_download_workers
  clean_flag = " --clean-target" if clean_target else ""
  script = (
      "set -euo pipefail\n"
      f"python3 -u /tmp/maxtext/{_STEP1_REL} download \\\n"
      f"  --platform {shlex.quote(platform)} \\\n"
      f"  --region {shlex.quote(profile.gcp_region)} \\\n"
      f"  --model {shlex.quote(model_name)} \\\n"
      f"  --hf-repo-id {shlex.quote(model_info['hf_repo_id'])} \\\n"
      f"  --gcs-cache-dir {shlex.quote(gcs_cache_dir)} \\\n"
      f"  --run-id {shlex.quote(run_id)} \\\n"
      f"  --workers {n_workers}{clean_flag} \\\n"
      "  --local-result-json /tmp/step1_result.json \\\n"
      f"  --result-gcs-uri {shlex.quote(result_gcs_uri)}\n"
  )
  name = k8s_name(["mesa", "step1", platform, model_name, run_id])
  labels = {**profile.pod_labels, "mesa-run-id": _label_value(run_id), "mesa-step": "step1"}
  return _build_cpu_job(
      name=name,
      profile=profile,
      node_pool=profile.cpu_node_pool,
      runner_script=script,
      labels=labels,
      memory_request="64Gi",
      cpu_request="16",
      extra_env=[
          {"name": "BENCH_HOST_MACHINE_TYPE", "value": profile.cpu_machine_type},
          {"name": "BENCH_RUN_ID", "value": run_id},
      ],
      require_hf_token=True,
  )


def recommended_step1b_pvc_gib(model_info: Mapping[str, Any]) -> int:
  """Returns scratch PVC size in GiB covering HF quantized download + converted BF16 output."""
  raw_gb = float(model_info.get("hf_download_gb") or 200.0)
  bf16_gb = float(model_info.get("loaded_bf16_gb") or 600.0)
  return max(1500, int((raw_gb + bf16_gb) * 1.25))


def build_deepseek_step1b_manifests(
    *,
    model_name: str,
    run_id: str,
    pvc_size_gib: int | None = None,
    clean_target: bool = False,
    cache_size: int = 4,
    storage_class: str = "standard-rwo",
    node_pool: str | None = None,
    memory_request: str = "400Gi",
    cpu_request: str = "64",
) -> list[dict[str, Any]]:
  """Builds `[PersistentVolumeClaim, Job]` for DeepSeek Step 1b (`FP8/FP4 -> BF16`) on `v7x` `job-manager-np`."""
  profile = platforms.get_profile("v7x")
  cfg = schema_v2.load_models_v2()
  model_info = next((m for m in cfg["models"] if m["model_name"] == model_name), None)
  if model_info is None:
    raise ValueError(f"Unknown model {model_name!r} in models_v2.json")

  size_gib = pvc_size_gib or recommended_step1b_pvc_gib(model_info)
  pvc_name = k8s_name(["mesa", "step1b", model_name, "pvc"])
  job_name = k8s_name(["mesa", "step1b", model_name, run_id])
  labels = {**profile.pod_labels, "mesa-run-id": _label_value(run_id), "mesa-step": "step1b"}

  pvc = {
      "apiVersion": "v1",
      "kind": "PersistentVolumeClaim",
      "metadata": {"name": pvc_name, "namespace": "default", "labels": labels},
      "spec": {
          "accessModes": ["ReadWriteOnce"],
          "storageClassName": storage_class,
          "resources": {"requests": {"storage": f"{size_gib}Gi"}},
      },
  }

  src_gcs = regional_hf_cache_dir(profile, model_name, dequantized=False)
  dst_gcs = regional_hf_cache_dir(profile, model_name, dequantized=True)
  root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")
  result_gcs_uri = f"{root}/step1b/{model_name}_step1b_result.json"
  clean_flag = " --clean-target" if clean_target else ""
  script = (
      "set -euo pipefail\n"
      f"mkdir -p /scratch/{shlex.quote(model_name)}\n"
      f"python3 -u /tmp/maxtext/{_STEP1_REL} deepseek-convert \\\n"
      "  --platform v7x \\\n"
      f"  --region {shlex.quote(profile.gcp_region)} \\\n"
      f"  --model {shlex.quote(model_name)} \\\n"
      f"  --hf-repo-id {shlex.quote(model_info['hf_repo_id'])} \\\n"
      f"  --source-gcs-dir {shlex.quote(src_gcs)} \\\n"
      f"  --dest-gcs-dir {shlex.quote(dst_gcs)} \\\n"
      f"  --scratch-dir /scratch/{shlex.quote(model_name)} \\\n"
      f"  --run-id {shlex.quote(run_id)} \\\n"
      f"  --workers {profile.max_download_workers} \\\n"
      f"  --cache-size {cache_size}{clean_flag} \\\n"
      "  --local-result-json /tmp/step1b_result.json \\\n"
      f"  --result-gcs-uri {shlex.quote(result_gcs_uri)}\n"
  )
  target_pool = node_pool or profile.step1b_cpu_node_pool or profile.cpu_node_pool
  target_machine = (
      profile.cpu_machine_type
      if target_pool == profile.cpu_node_pool
      else (profile.step1b_cpu_machine_type or profile.cpu_machine_type)
  )
  job = _build_cpu_job(
      name=job_name,
      profile=profile,
      node_pool=target_pool,
      runner_script=script,
      labels=labels,
      memory_request=memory_request,
      cpu_request=cpu_request,
      extra_volumes=[{"name": "scratch", "persistentVolumeClaim": {"claimName": pvc_name}}],
      extra_mounts=[{"name": "scratch", "mountPath": "/scratch"}],
      extra_env=[
          {"name": "BENCH_HOST_MACHINE_TYPE", "value": target_machine},
          {"name": "BENCH_RUN_ID", "value": run_id},
      ],
  )
  return [pvc, job]


def build_copy_region_cpu_job(
    *,
    dest_platform: str,
    model_name: str,
    source_gcs_dir: str,
    run_id: str,
    clean_target: bool = False,
    bf16_source: str = "converted_by_this_benchmark",
) -> dict[str, Any]:
  """Builds a destination-region CPU Job copying converted BF16 weights into the regional bucket."""
  profile = platforms.get_profile(dest_platform)
  cfg = schema_v2.load_models_v2()
  model_info = next((m for m in cfg["models"] if m["model_name"] == model_name), None)
  if model_info is None:
    raise ValueError(f"Unknown model {model_name!r} in models_v2.json")

  dest_gcs = regional_hf_cache_dir(profile, model_name, dequantized=True)
  root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")
  result_gcs_uri = f"{root}/step1b/{model_name}_copy_result.json"
  clean_flag = " --clean-target" if clean_target else ""
  script = (
      "set -euo pipefail\n"
      f"python3 -u /tmp/maxtext/{_STEP1_REL} copy-region \\\n"
      f"  --platform {shlex.quote(dest_platform)} \\\n"
      f"  --dest-region {shlex.quote(profile.gcp_region)} \\\n"
      f"  --model {shlex.quote(model_name)} \\\n"
      f"  --hf-repo-id {shlex.quote(model_info['hf_repo_id'])} \\\n"
      f"  --source-gcs-dir {shlex.quote(source_gcs_dir)} \\\n"
      f"  --dest-gcs-dir {shlex.quote(dest_gcs)} \\\n"
      f"  --run-id {shlex.quote(run_id)} \\\n"
      f"  --workers {profile.max_download_workers}{clean_flag} \\\n"
      f"  --bf16-source {shlex.quote(bf16_source)} \\\n"
      "  --local-result-json /tmp/copy_result.json \\\n"
      f"  --result-gcs-uri {shlex.quote(result_gcs_uri)}\n"
  )
  name = k8s_name(["mesa", "copy", dest_platform, model_name, run_id])
  labels = {**profile.pod_labels, "mesa-run-id": _label_value(run_id), "mesa-step": "copy-region"}
  return _build_cpu_job(
      name=name,
      profile=profile,
      node_pool=profile.cpu_node_pool,
      runner_script=script,
      labels=labels,
      memory_request="64Gi",
      cpu_request="16",
      extra_env=[
          {"name": "BENCH_HOST_MACHINE_TYPE", "value": profile.cpu_machine_type},
          {"name": "BENCH_RUN_ID", "value": run_id},
      ],
  )


def build_tpu_jobset(
    *,
    platform: str,
    topology: str,
    num_slices: int,
    name: str,
    runner_script: str,
    run_id: str,
    experiment_name: str = "mesa-bench-v2",
    repetition_index: int = 1,
    max_restarts: int = 3,
    workload_duration_minutes: int | None = None,
) -> dict[str, Any]:
  """Builds a multi-slice / multi-host TPU `JobSet` manifest strictly from `PlatformProfile`."""
  if not name.startswith("mesa-") or len(name) > _MAX_JOBSET_NAME_LEN:
    raise ValueError(f"JobSet name {name!r} must start with 'mesa-' and be <= {_MAX_JOBSET_NAME_LEN} chars")
  profile = platforms.get_profile(platform)
  hosts = profile.hosts_per_slice(topology)
  chips_per_host = profile.chips_per_host(topology)
  run_label = _label_value(run_id)
  base_labels = {**profile.pod_labels, "mesa-run-id": run_label}
  jobset_labels = dict(base_labels)
  pod_labels = dict(base_labels)
  if profile.kueue_queue:
    jobset_labels["kueue.x-k8s.io/queue-name"] = profile.kueue_queue
  dur = workload_duration_minutes if workload_duration_minutes is not None else profile.workload_time_limit_minutes
  if dur is not None:
    pod_labels["declared-duration-minutes"] = str(dur)

  pod_annotations: dict[str, str] = {}
  if profile.kueue_topology_aware_scheduling:
    pod_annotations.update(
        {
            "kueue.x-k8s.io/podset-slice-required-topology": "cloud.google.com/gke-nodepool",
            "kueue.x-k8s.io/podset-slice-size": str(hosts),
            "kueue.x-k8s.io/podset-unconstrained-topology": "true",
        }
    )

  daemons = [
      f"{name}-slice{s}-0-{h}.{name}" for s in range(num_slices) for h in range(hosts) if not (s == 0 and h == 0)
  ]
  daemons_env = ",".join(f"{d}:55555" for d in daemons)
  # Wait host list carries hostnames without port so bootstrap_tpu7x.py exports REMOTE_SLICE1_ADDR="h:55555,...".
  wait_hosts = ",".join(daemons)
  del daemons_env

  replicated_jobs = []
  for s in range(num_slices):
    is_runner_slice = s == 0
    env_list: list[dict[str, Any]] = [
        {"name": "SLICE_INDEX", "value": str(s)},
        {"name": "DAEMON_PORT", "value": "55555"},
        {"name": "BENCH_CODE_DIR", "value": profile.gcs_code_dir},
        {"name": "BENCH_PLATFORM", "value": profile.name},
        {"name": "BENCH_CLUSTER", "value": profile.gke_cluster_name},
        {"name": "BENCH_PROJECT", "value": profile.gcp_project_id},
        {"name": "BENCH_REGION", "value": profile.gcp_region},
        {"name": "BENCH_ZONE", "value": profile.zone_hint},
        {"name": "BENCH_CAPACITY_TYPE", "value": profile.capacity_type},
        {"name": "BENCH_KUEUE_QUEUE", "value": profile.kueue_queue or ""},
        {"name": "BENCH_HOST_MACHINE_TYPE", "value": profile.host_machine_type_by_topology[topology]},
        {
            "name": "BENCH_TPU_ACCELERATOR",
            "value": profile.node_selectors.get("cloud.google.com/gke-tpu-accelerator", ""),
        },
        {"name": "BENCH_IMAGE", "value": profile.container_image},
        {"name": "BENCH_SLICE_TOPOLOGY", "value": topology},
        {"name": "BENCH_RUN_ID", "value": run_id},
        {"name": "BENCH_EXPERIMENT_NAME", "value": experiment_name},
        {"name": "BENCH_REPETITION_INDEX", "value": str(repetition_index)},
        {"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"},
        *_downward_api_envs(),
        _hf_token_env(profile, required=False),
        {"name": "WAIT_FOR_DAEMON_HOST", "value": wait_hosts if is_runner_slice else ""},
        {"name": "RUNNER_SCRIPT", "value": runner_script if is_runner_slice else ""},
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
                "command": ["bash", "-c", _bootstrap_shell_cmd(profile.gcs_code_dir, tpu_jobset=True)],
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
      pod_spec["imagePullSecrets"] = [{"name": sec} for sec in profile.image_pull_secrets]
    if profile.required_node_affinity:
      pod_spec["affinity"] = {
          "nodeAffinity": {
              "requiredDuringSchedulingIgnoredDuringExecution": {
                  "nodeSelectorTerms": [{"matchExpressions": copy.deepcopy(profile.required_node_affinity)}]
              }
          }
      }
    replicated_jobs.append(
        {
            "name": f"slice{s}",
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
                        "metadata": {"labels": dict(pod_labels), "annotations": pod_annotations},
                        "spec": pod_spec,
                    },
                },
            },
        }
    )

  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {"name": name, "namespace": "default", "labels": jobset_labels},
      "spec": {
          "failurePolicy": {"maxRestarts": max_restarts},
          "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
          "replicatedJobs": replicated_jobs,
      },
  }


def build_queue_worker_jobset(
    *,
    platform: str,
    topology: str,
    num_slices: int,
    queue_uri: str,
    worker_id: str,
    run_id: str,
    results_gcs_uri: str,
    experiment_name: str = "mesa-bench-v2",
    min_slices: int = 1,
) -> dict[str, Any]:
  """Builds a queue-draining TPU JobSet supporting multi-host (`hosts_per_slice`) and multi-slice tasks."""
  profile = platforms.get_profile(platform)
  hosts = profile.hosts_per_slice(topology)
  name = k8s_name(["mesa", "q", platform, topology, f"{num_slices}s", worker_id], max_len=_MAX_JOBSET_NAME_LEN)
  script = (
      "set -euo pipefail\n"
      "export TORCH_DEVICE_BACKEND_AUTOLOAD=0 KEEP_HF_CACHE=1 HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp\n"
      f"python3 -u /tmp/maxtext/{_QUEUE_REL} \\\n"
      f"  --queue {shlex.quote(queue_uri)} \\\n"
      f"  --worker {shlex.quote(worker_id)} \\\n"
      f"  --platform {shlex.quote(platform)} \\\n"
      f"  --topology {shlex.quote(topology)} \\\n"
      f"  --hosts-per-slice {hosts} \\\n"
      f"  --max-slices {num_slices} \\\n"
      f"  --min-slices {min_slices} \\\n"
      f"  --results {shlex.quote(results_gcs_uri)} \\\n"
      "  --prefer-max-slices --no-wait\n"
  )
  return build_tpu_jobset(
      platform=platform,
      topology=topology,
      num_slices=num_slices,
      name=name,
      runner_script=script,
      run_id=run_id,
      experiment_name=experiment_name,
  )


def build_planned_queue_tasks(
    *,
    platform: str,
    experiment_name: str,
    run_prefix: str,
    tiers: Sequence[str] = ("S", "A", "C", "B"),
    tree: str = "/tmp/maxtext",
    verify_checksums_tiers: Sequence[str] = ("A",),
    clean_sparse_env: bool = True,
) -> list[dict[str, Any]]:
  """Generates ordered GCS task-queue items for `platform` from `models_v2.json`."""
  profile = platforms.get_profile(platform)
  cfg = schema_v2.load_models_v2()
  reps_by_tier: Mapping[str, int] = cfg["repetitions_by_tier"]
  want_tiers = set(tiers)
  want_cs_tiers = set(verify_checksums_tiers)
  root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")

  models = [m for m in cfg["models"] if m["tier"] in want_tiers]
  models.sort(key=lambda m: (_TIER_ORDER.get(m["tier"], 99), float(m.get("loaded_bf16_gb") or 0.0)))

  tasks: list[dict[str, Any]] = []
  for model in models:
    m_name = model["model_name"]
    tier = model["tier"]
    needs_1b = bool(model.get("needs_step1b"))
    staged_gcs_dir = regional_hf_cache_dir(profile, m_name, dequantized=needs_1b)
    reps = int(reps_by_tier.get(tier, 1))
    placements = model.get("topologies_by_platform", {}).get(platform, [])

    # Convert Orbax once on a single host before fan-out if multi-slice placements exist, or run all placements.
    for placement in placements:
      slices = int(placement["slices"])
      topology = str(placement["topology"])
      hosts = profile.hosts_per_slice(topology)
      devices = profile.devices_per_slice(topology)
      for rep in range(1, reps + 1):
        topo_slug = dns_slug(topology)
        run_id = f"{run_prefix}-{platform}-{m_name}-{slices}x{topo_slug}-r{rep}"
        tid = f"{platform}_{m_name}_{slices}x{topo_slug}_r{rep}"
        args = [
            "--models",
            m_name,
            "--scan-modes",
            "scanned",
            "unscanned",
            "--num-slices-list",
            str(slices),
            "--num-hosts-per-slice",
            str(hosts),
            "--devices-per-slice",
            str(devices),
            "--hardware",
            "tpu",
            "--platform",
            platform,
            "--slice-topology",
            topology,
            "--gcs-base",
            f"{root}/runs/{tier}",
            "--run-id",
            run_id,
            "--local-work-dir",
            f"/ramdisk/bench_{tid}",
            "--reuse-model-hf-cache",
            "--skip-completed",
            "--allow-step-failure",
        ]
        if tier in want_cs_tiers:
          args.append("--verify-checksums")
        env: dict[str, str] = {
            "BENCH_PLATFORM": platform,
            "BENCH_SLICE_TOPOLOGY": topology,
            "BENCH_EXPERIMENT_NAME": experiment_name,
            "BENCH_REPETITION_INDEX": str(rep),
            "BENCH_RUN_ID": run_id,
            "BENCH_SAFETENSORS_GCS_DIR": staged_gcs_dir,
            "KEEP_HF_CACHE": "1",
            "SKIP_FAILED_RESULTS": "1",
        }
        if needs_1b and clean_sparse_env:
          env["STAGE_SPARSE_DEEPSEEK_OOM"] = "1"
        tasks.append(
            {
                "id": tid,
                "platform": platform,
                "tier": tier,
                "model": m_name,
                "slices": slices,
                "hosts_per_slice": hosts,
                "topology": topology,
                "repetition": rep,
                "tree": tree,
                "args": args,
                "env": env,
                "csv": f"{tid}.csv",
            }
        )
  return tasks
