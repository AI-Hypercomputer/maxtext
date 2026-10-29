"""Generates Kubernetes YAML manifests for 2-slice `tpu7x` benchmark pod pairs on `bodaborg-tpu7x-nap`.

Each benchmark pair `<id>` (`a`..`f`) consists of:
  - Headless Kubernetes `Service` (`mesakh-w1-<id>`, port `55555`) for Slice 0 -> Slice 1 discovery
  - Slice 1 Worker Pod (`mesakh-tpu7x-<id>1`, `ROLE=daemon`): 1 `tpu7x-standard-4t` node (`2x2x1`, 4 chips, 8 TPU cores, 813.8 GB HBM, 1,014.5 GB RAM)
  - Slice 0 Runner Pod (`mesakh-tpu7x-<id>0`, `ROLE=runner`): 1 `tpu7x-standard-4t` node (`2x2x1`, 4 chips, 8 TPU cores, 813.8 GB HBM, 1,014.5 GB RAM)

Pod Pair Assignments:
  - Priority 1 (`tpu7x_p1_pods.yaml` — `>200B` Models: Experiment 1A Before/After Lance's PRs + Experiment 1B `to_maxtext` vs. `safetensors_dynamic`):
      * Pair `a`: `qwen3-235b-a22b` (235B MoE, 470.38 GB BF16)
      * Pair `b`: `qwen3.5-397b-a17b` (397B Hybrid GatedDeltaNet MoE, 795.48 GB BF16)
      * Pair `c`: `qwen3-480b-a35b` (480B MoE, 961.03 GB BF16)
      * Pair `f`: `deepseek4-284b` (284B MoE, 581.89 GB BF16) & `deepseek3-671b` (671B MoE, 1,342.35 GB BF16)
  - Priority 2 (`tpu7x_p2_pods.yaml` — 16-Model Benchmark Suite across 1 & 2 slices):
      * Pair `d`: `qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `llama3.1-8b`, `gemma2-2b`, `gemma3-4b`, `deepseek2-16b`, `gpt-oss-20b`, `qwen3-30b-a3b`
      * Pair `e`: `qwen3-32b`, `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b`
  - Priority 1 completion (`tpu7x_p1_completion_pods.yaml` + `tpu7x_ctrl_f1_pod.yaml`): workers draining the GCS task queue
    `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1` (tasks: `configs/p1_completion_tasks.json`, worker:
    `scripts/tpu7x_task_queue.py`):
      * Pair `g`: 2-slice worker (`--max-slices 2`).
      * `mesakh-ctrl-f1`: CPU controller (`scripts/tpu7x_daemon_controller.py`) that runs single-slice tasks on the
        surviving daemon pod `mesakh-tpu7x-f1`, one task per daemon command.
  - Multi-host slices (`tpu7x_multihost_<run>.yaml`, one per `MH_RUNS` entry): a JobSet `<owner>-mh-<run>` on
    `2x2x2` (2 hosts per slice) or `2x2x4` (4 hosts per slice) nodes. `<owner>` is `$BENCH_OWNER` (default `$USER`);
    code, results and logs go to `gs://mesa-maxtext/<owner>_tpu7x_bench/`.
"""

import copy
import os
import yaml

IMAGE = "gcr.io/cloud-tpu-shared-capacity/yaoyuchen-runner:ds41-2026-10-22-16-03-48"


def make_bootstrap_cmd(code_dir: str) -> str:
  return (
      'python3 -c "from etils import epath; '
      f"open('/tmp/bootstrap.py','wb').write(epath.Path('{code_dir}/bootstrap_tpu7x.py').read_bytes())\" "
      "&& python3 -u /tmp/bootstrap.py"
  )


BOOTSTRAP_CMD = make_bootstrap_cmd("gs://mesa-maxtext/mesakh_tpu7x_bench/code")


def make_p1_runner_script(pair_id: str, models_str: str, extra_exports: str = "") -> str:
  return f"""set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export KEEP_HF_CACHE=1
export SKIP_FAILED_RESULTS=1
{extra_exports}
echo "=== [Pair {pair_id}] Step 1: Priority 1A/1B safetensors_dynamic WITH Lance's 3 PRs (#5706+#5828+#5705) ==="
python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \\
  --models {models_str} \\
  --scan-modes scanned unscanned \\
  --num-slices-list 1 2 \\
  --devices-per-slice 8 \\
  --hardware tpu \\
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \\
  --run-id tpu7x_200b \\
  --local-work-dir /tmp/p1_lance_prs \\
  --csv-report /tmp/p1_st_lance_{pair_id}.csv \\
  --reuse-model-hf-cache \\
  --skip-to-maxtext \\
  --skip-completed \\
  --allow-step-failure
python3 -c "from etils import epath; epath.Path('gs://mesa-maxtext/mesakh_tpu7x_bench/results/p1_st_lance_{pair_id}.csv').write_bytes(open('/tmp/p1_st_lance_{pair_id}.csv','rb').read())"

echo "=== [Pair {pair_id}] Step 2: Priority 1A safetensors_dynamic BEFORE Lance's 3 PRs (origin/main) ==="
export SHARED_HF_CACHE_BASE_DIR="gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/safetensors_dynamic"
export KEEP_HF_CACHE=1
export PHASE_TIMEOUT_S=900
python3 -u /tmp/maxtext_origin_main/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \\
  --models {models_str} \\
  --scan-modes scanned unscanned \\
  --num-slices-list 1 2 \\
  --devices-per-slice 8 \\
  --hardware tpu \\
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main \\
  --run-id tpu7x_200b \\
  --local-work-dir /tmp/p1_origin_main \\
  --csv-report /tmp/p1_origin_main_{pair_id}.csv \\
  --reuse-model-hf-cache \\
  --skip-to-maxtext \\
  --skip-completed \\
  --allow-step-failure
python3 -c "from etils import epath; epath.Path('gs://mesa-maxtext/mesakh_tpu7x_bench/results/p1_origin_main_{pair_id}.csv').write_bytes(open('/tmp/p1_origin_main_{pair_id}.csv','rb').read())"
unset SHARED_HF_CACHE_BASE_DIR
unset PHASE_TIMEOUT_S

echo "=== [Pair {pair_id}] Step 3: Priority 1B Full to_maxtext vs safetensors_dynamic comparison ==="
export KEEP_HF_CACHE=1
export REUSE_SHARED_ST_GCS_CACHE=1
python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \\
  --models {models_str} \\
  --scan-modes scanned unscanned \\
  --num-slices-list 1 2 \\
  --devices-per-slice 8 \\
  --hardware tpu \\
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \\
  --run-id tpu7x_200b \\
  --local-work-dir /tmp/p1_lance_prs \\
  --csv-report /tmp/p1_full_{pair_id}.csv \\
  --reuse-model-hf-cache \\
  --skip-completed \\
  --allow-step-failure
python3 -c "from etils import epath; epath.Path('gs://mesa-maxtext/mesakh_tpu7x_bench/results/p1_full_{pair_id}.csv').write_bytes(open('/tmp/p1_full_{pair_id}.csv','rb').read())"
echo "=== [Pair {pair_id}] ALL DONE ==="
"""


def make_p2_runner_script(pair_id: str, models_str: str) -> str:
  return f"""set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
echo "=== [Pair {pair_id}] Priority 2: 16-Model Benchmark on bodaborg-tpu7x-nap ==="
python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \\
  --models {models_str} \\
  --scan-modes scanned unscanned \\
  --num-slices-list 1 2 \\
  --devices-per-slice 8 \\
  --hardware tpu \\
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs \\
  --run-id tpu7x_16m \\
  --local-work-dir /tmp/p2_lance_prs \\
  --csv-report /tmp/p2_full_{pair_id}.csv \\
  --reuse-model-hf-cache \\
  --skip-completed \\
  --allow-step-failure
python3 -c "from etils import epath; epath.Path('gs://mesa-maxtext/mesakh_tpu7x_bench/results/p2_full_{pair_id}.csv').write_bytes(open('/tmp/p2_full_{pair_id}.csv','rb').read())"
echo "=== [Pair {pair_id}] ALL DONE ==="
"""


P1_PAIRS = [
    ("a", make_p1_runner_script("a", "qwen3-235b-a22b")),
    ("b", make_p1_runner_script("b", "qwen3.5-397b-a17b")),
    ("c", make_p1_runner_script("c", "qwen3-480b-a35b")),
    ("f", make_p1_runner_script("f", "deepseek4-284b deepseek3-671b", "export STAGE_SPARSE_DEEPSEEK_OOM=1")),
]

P2_PAIRS = [
    (
        "d",
        make_p2_runner_script(
            "d",
            "qwen3-0.6b qwen3-4b qwen3-8b qwen3-14b llama3.1-8b gemma2-2b gemma3-4b deepseek2-16b gpt-oss-20b qwen3-30b-a3b",
        ),
    ),
    (
        "e",
        make_p2_runner_script(
            "e",
            "qwen3-32b mixtral-8x7b llama3.1-70b gpt-oss-120b",
        ),
    ),
]

# Priority 1 completion: a GCS task queue (configs/p1_completion_tasks.json uploaded to QUEUE_P1/tasks.json) that
# any number of workers drain together, so spot preemptions and Kueue reclamation only delay the run.
CODE_DIR = "gs://mesa-maxtext/mesakh_tpu7x_bench/code"
QUEUE_P1 = "gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1"


def make_queue_runner_script(pair_id: str, queue: str, max_slices: int = 2, min_slices: int = 1) -> str:
  return f"""set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 KEEP_HF_CACHE=1 HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp
mkdir -p /ramdisk/hf_cache /ramdisk/tmp
python3 -c "from etils import epath; open('/tmp/tpu7x_task_queue.py','wb').write(epath.Path('{CODE_DIR}/tpu7x_task_queue.py').read_bytes())"
echo "=== [Worker {pair_id}] Task-queue worker for {queue} (slices {min_slices}..{max_slices}) ==="
python3 -u /tmp/tpu7x_task_queue.py --queue {queue} --worker {pair_id} --max-slices {max_slices} --min-slices {min_slices} --prefer-max-slices
echo "=== [Worker {pair_id}] ALL DONE ==="
"""


P1C_PAIRS = [("g", make_queue_runner_script("g", QUEUE_P1))]

# Scale-out while spot 2x2x1 capacity is idle: two more 2-slice workers, three single-slice workers for the
# to_maxtext CPU conversions (no idle slice-1 node during multi-hour conversions), and a 4-slice group that only
# takes 4-slice tasks (deepseek3-671b, 1.34 TB BF16: needs 32 TPU cores / 3.26 TB HBM).
P1C_SCALEOUT_PAIRS = [(pid, make_queue_runner_script(pid, QUEUE_P1)) for pid in ("i", "j")]
P1C_SCALEOUT_SINGLES = [(wid, make_queue_runner_script(wid, QUEUE_P1, max_slices=1)) for wid in ("s1", "s2", "s3")]
P1C_SCALEOUT_GROUPS = [("m", 4, make_queue_runner_script("m", QUEUE_P1, max_slices=4, min_slices=4))]

# Second scale-out wave (11 spot 2x2x1 nodes were still idle at 13:22 UTC): five more single-slice workers so the
# long unscanned to_maxtext conversions (480B, 397B), which gate their 2-slice Orbax loads, start right away, and
# one more 2-slice pair. Workers exit when no task they can run is left, which frees their nodes.
P1C_SCALEOUT2_PAIRS = [("k", make_queue_runner_script("k", QUEUE_P1))]
P1C_SCALEOUT2_SINGLES = [
    (wid, make_queue_runner_script(wid, QUEUE_P1, max_slices=1)) for wid in ("s4", "s5", "s6", "s7", "s8")
]


def group_to_docs(gid: str, n_slices: int, runner_script: str) -> list[dict]:
  """An N-slice group: runner `mesakh-tpu7x-<gid>0` (slice 0) and daemons `mesakh-tpu7x-<gid><k>` (slices 1..N-1).

  Each daemon gets a headless Service `mesakh-w<k>-<gid>`; the runner waits for all of them (bootstrap_tpu7x.py
  exports them as REMOTE_SLICE1_ADDR, in slice order).
  """
  svc1, daemon1, runner = pairs_to_docs([(gid, runner_script)])
  docs = [svc1, daemon1]
  for k in range(2, n_slices):
    svc, daemon = copy.deepcopy(svc1), copy.deepcopy(daemon1)
    svc["metadata"]["name"] = svc["spec"]["selector"]["role"] = f"mesakh-w{k}-{gid}"
    daemon["metadata"]["name"] = f"mesakh-tpu7x-{gid}{k}"
    daemon["metadata"]["labels"]["role"] = f"mesakh-w{k}-{gid}"
    docs += [svc, daemon]
  for env in runner["spec"]["containers"][0]["env"]:
    if env["name"] == "WAIT_FOR_DAEMON_HOST":
      env["value"] = ",".join(f"mesakh-w{k}-{gid}" for k in range(1, n_slices))
  return docs + [runner]


def single_worker_docs(wid: str, runner_script: str) -> list[dict]:
  """A single-slice worker pod `mesakh-tpu7x-<wid>` (runner without a slice-1 daemon)."""
  runner = pairs_to_docs([(wid, runner_script)])[2]
  runner["metadata"]["name"] = f"mesakh-tpu7x-{wid}"
  container = runner["spec"]["containers"][0]
  container["env"] = [e for e in container["env"] if e["name"] != "WAIT_FOR_DAEMON_HOST"]
  return [runner]


def make_daemon_worker_command(worker: str, queue: str, max_slices: int = 1, remote: str | None = None) -> str:
  """Shell command a controller sends to a daemon pod: sync the harness, then run one task from the queue.

  With `remote` (a second daemon's host:port), the daemon pod acts as slice 0 of a 2-slice pair.
  """
  remote_flag = f" --remote-slice-addrs {remote}" if remote else ""
  return f"""set -e
export PYTHONPATH=/tmp/maxtext/src:/tmp/maxtext HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp KEEP_HF_CACHE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p /ramdisk/hf_cache /ramdisk/tmp
python3 - <<'EOF'
import os
from etils import epath
code = "{CODE_DIR}"
harness = epath.Path(code + "/benchmark_ckpt_load_time.py").read_bytes()
for tree in ("/tmp/maxtext", "/tmp/maxtext_origin_main", "/tmp/maxtext_hbmfix"):
    if os.path.isdir(tree + "/checkpoint_load_benchmark"):
        open(tree + "/checkpoint_load_benchmark/benchmark_ckpt_load_time.py", "wb").write(harness)
open("/tmp/tpu7x_task_queue.py", "wb").write(epath.Path(code + "/tpu7x_task_queue.py").read_bytes())
EOF
set +e
python3 -u /tmp/tpu7x_task_queue.py --queue {queue} --worker {worker} --max-slices {max_slices}{remote_flag} --prefer-max-slices --max-tasks 1 --no-wait
exit $?
"""


def make_daemon_controller_pod(
    pod_name: str, daemon_addr: str, worker: str, queue: str, max_slices: int = 1, remote: str | None = None
) -> dict:
  """A CPU pod running scripts/tpu7x_daemon_controller.py against a daemon pod (e.g. one whose runner is gone)."""
  with open(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "tpu7x_daemon_controller.py"), encoding="utf-8"
  ) as f:
    controller_src = f.read()
  return {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {
          "name": pod_name,
          "namespace": "default",
          "labels": {"kueue.x-k8s.io/queue-name": "default", "app": "mesakh-tpu7x-bench", "role": "controller"},
      },
      "spec": {
          "restartPolicy": "Never",
          "containers": [
              {
                  "name": "controller",
                  "image": "python:3.11-slim",
                  "command": [
                      "python3",
                      "-u",
                      "-c",
                      controller_src,
                      "--daemon",
                      daemon_addr,
                      "--command",
                      make_daemon_worker_command(worker, queue, max_slices, remote),
                      "--shutdown-at-end",
                  ],
                  "resources": {"requests": {"cpu": "250m", "memory": "256Mi"}},
              }
          ],
      },
  }


def pairs_to_docs(pairs_list):
  docs = []
  for pid, runner_script in pairs_list:
    svc_name = f"mesakh-w1-{pid}"
    w1_name = f"mesakh-tpu7x-{pid}1"
    w0_name = f"mesakh-tpu7x-{pid}0"

    docs.append(
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": svc_name, "namespace": "default"},
            "spec": {
                "clusterIP": "None",
                "selector": {"role": svc_name},
                "ports": [{"port": 55555, "targetPort": 55555}],
            },
        }
    )

    for pod_name, role_env, is_w1 in [(w1_name, "daemon", True), (w0_name, "runner", False)]:
      env_list = [
          {"name": "ROLE", "value": role_env},
          {"name": "DAEMON_PORT", "value": "55555"},
          {"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"},
          {
              "name": "HF_TOKEN",
              "valueFrom": {"secretKeyRef": {"name": "mesakh-hf-token", "key": "HF_TOKEN"}},
          },
      ]
      if not is_w1:
        env_list.append({"name": "WAIT_FOR_DAEMON_HOST", "value": svc_name})
        env_list.append({"name": "RUNNER_SCRIPT", "value": runner_script})

      pod_doc = {
          "apiVersion": "v1",
          "kind": "Pod",
          "metadata": {
              "name": pod_name,
              "namespace": "default",
              "annotations": {
                  "kueue.x-k8s.io/podset-unconstrained-topology": "true",
              },
              "labels": {
                  "kueue.x-k8s.io/queue-name": "default",
                  "app": "mesakh-tpu7x-bench",
                  "pair": pid,
                  **({"role": svc_name} if is_w1 else {}),
              },
          },
          "spec": {
              "restartPolicy": "Never",
              "imagePullSecrets": [{"name": "gitimg"}],
              "affinity": {
                  "nodeAffinity": {
                      "requiredDuringSchedulingIgnoredDuringExecution": {
                          "nodeSelectorTerms": [
                              {
                                  "matchExpressions": [
                                      {
                                          "key": "cloud.google.com/reservation-name",
                                          "operator": "DoesNotExist",
                                      }
                                  ]
                              }
                          ]
                      }
                  }
              },
              "nodeSelector": {
                  "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                  "cloud.google.com/gke-tpu-topology": "2x2x1",
                  "cloud.google.com/gke-spot": "true",
                  "tpu7x.gke.io/capacity": "spot",
              },
              "tolerations": [
                  {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
                  {"key": "cloud.google.com/gke-spot", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
              ],
              "volumes": [
                  {"name": "ramdisk", "emptyDir": {"medium": "Memory"}},
              ],
              "containers": [
                  {
                      "name": "worker",
                      "image": IMAGE,
                      "imagePullPolicy": "IfNotPresent",
                      "command": ["bash", "-c", BOOTSTRAP_CMD],
                      "env": env_list,
                      "volumeMounts": [{"name": "ramdisk", "mountPath": "/ramdisk"}],
                      "resources": {
                          "requests": {"google.com/tpu": "4"},
                          "limits": {"google.com/tpu": "4"},
                      },
                  }
              ],
          },
      }
      docs.append(pod_doc)
  return docs


# Multi-host slices. A tpu7x-standard-4t host has 4 chips (8 TPU cores), so a 2x2x2 slice is 2 hosts / 16 cores /
# 1,627.6 GB HBM, and a 2x2x4 slice is 4 hosts / 32 cores / 3,255.2 GB HBM. A v5p host (ct5p-hightpu-4t,
# bodaborg-v5p-nap) has 4 chips (4 devices, 95 GB HBM each), so 2x2x4 is 4 hosts / 1,520 GB HBM and 4x4x4 is 16 hosts /
# 6,080 GB HBM. These runs use their owner's GCS prefix, pod names and app label, so they never touch the
# mesakh_tpu7x_bench code, queue or pods.
MH_OWNER = os.environ.get("BENCH_OWNER", os.environ.get("USER", "bench"))
MH_BASE = f"gs://mesa-maxtext/{MH_OWNER}_tpu7x_bench"
MH_CODE_DIR = f"{MH_BASE}/code"
MH_APP = f"{MH_OWNER}-tpu7x-bench"
HOSTS_PER_SLICE = {"2x2x2": 2, "2x2x4": 4, "4x4x4": 16}
# Per platform: TPU devices per host and the harness run id, which keeps the result paths of the platforms apart.
MH_PLATFORMS = {
    "tpu7x": {"devices_per_host": 8, "harness_run_id": "tpu7x_multihost"},
    "v5p": {"devices_per_host": 4, "harness_run_id": "v5p_multihost"},
}
# HF caches that the earlier runs filled (every shard present). The multi-host runs only read them.
P1_HF_CACHE = "gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/safetensors_dynamic"
P2_HF_CACHE = "gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/safetensors_dynamic"


def make_multihost_runner_script(
    run_id: str,
    models_str: str,
    topology: str,
    num_slices: int,
    hf_cache: str,
    *,
    platform: str = "tpu7x",
    scan_modes: str = "scanned unscanned",
    extra_exports: str = "",
) -> str:
  """safetensors_dynamic WITH Lance's 3 PRs on `num_slices` multi-host slices of `topology`."""
  hosts = HOSTS_PER_SLICE[topology]
  plat = MH_PLATFORMS[platform]
  layout = topology if platform == "tpu7x" else f"{platform} {topology}"
  return f"""set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 KEEP_HF_CACHE=1 HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp
export SHARED_HF_CACHE_BASE_DIR="{hf_cache}"
{extra_exports}
echo "=== [{run_id}] safetensors_dynamic WITH Lance's 3 PRs on {num_slices} x {layout} ({hosts} hosts per slice) ==="
python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \\
  --models {models_str} \\
  --scan-modes {scan_modes} \\
  --num-slices-list {num_slices} \\
  --num-hosts-per-slice {hosts} \\
  --devices-per-slice {plat["devices_per_host"] * hosts} \\
  --hardware tpu \\
  --gcs-base {MH_BASE}/lance_prs \\
  --run-id {plat["harness_run_id"]} \\
  --local-work-dir /tmp/mh_lance_prs \\
  --csv-report /tmp/{run_id}.csv \\
  --reuse-model-hf-cache \\
  --skip-to-maxtext \\
  --skip-completed \\
  --allow-step-failure
python3 - <<'EOF'
from etils import epath
epath.Path("{MH_BASE}/results/{run_id}.csv").write_bytes(open("/tmp/{run_id}.csv", "rb").read())
EOF
echo "=== [{run_id}] ALL DONE ==="
"""


def multihost_jobset_doc(name: str, topology: str, num_slices: int, runner_script: str, platform: str = "tpu7x") -> dict:
  """A JobSet with one Indexed Job per slice (`slice<k>`, one pod per host).

  GKE sets TPU_WORKER_ID and TPU_WORKER_HOSTNAMES only on Indexed Job pods, and a multi-host slice needs them, so
  these runs cannot use bare pod pairs. Pod `<name>-slice0-0-0` (slice 0, host 0) is the runner and every other pod
  runs the remote slice daemon. WAIT_FOR_DAEMON_HOST lists the daemons slice by slice and host by host, the order in
  which the harness assigns JAX process ids 1..N-1.

  tpu7x runs on spot capacity with Kueue topology-aware scheduling. bodaborg-v5p-nap runs on reservation capacity:
  Kueue adds the reservation selector, and the cluster has no Kueue topology, so the pods carry no TAS annotations.
  """
  hosts = HOSTS_PER_SLICE[topology]
  if platform == "tpu7x":
    placement = {
        "imagePullSecrets": [{"name": "gitimg"}],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [{"key": "cloud.google.com/reservation-name", "operator": "DoesNotExist"}]}
                    ]
                }
            }
        },
        "nodeSelector": {
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            "cloud.google.com/gke-tpu-topology": topology,
            "cloud.google.com/gke-spot": "true",
            "tpu7x.gke.io/capacity": "spot",
        },
        "tolerations": [
            {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
            {"key": "cloud.google.com/gke-spot", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
        ],
    }
    pod_annotations = {
        "kueue.x-k8s.io/podset-slice-required-topology": "cloud.google.com/gke-nodepool",
        "kueue.x-k8s.io/podset-slice-size": str(hosts),
        "kueue.x-k8s.io/podset-unconstrained-topology": "true",
    }
  else:
    placement = {
        "nodeSelector": {
            "cloud.google.com/gke-tpu-accelerator": "tpu-v5p-slice",
            "cloud.google.com/gke-tpu-topology": topology,
        },
        "tolerations": [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
    }
    pod_annotations = {}
  daemons = [f"{name}-slice{s}-0-{h}.{name}" for s in range(num_slices) for h in range(hosts) if (s, h) != (0, 0)]
  command = (
      'if [ "$SLICE_INDEX" = 0 ] && [ "$JOB_COMPLETION_INDEX" = 0 ]; then export ROLE=runner; '
      "else export ROLE=daemon; fi; mkdir -p /ramdisk/hf_cache /ramdisk/tmp; " + make_bootstrap_cmd(MH_CODE_DIR)
  )
  replicated_jobs = []
  for s in range(num_slices):
    env_list = [
        {"name": "SLICE_INDEX", "value": str(s)},
        {"name": "DAEMON_PORT", "value": "55555"},
        {"name": "BENCH_CODE_DIR", "value": MH_CODE_DIR},
        {"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"},
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "mesakh-hf-token", "key": "HF_TOKEN", "optional": True}},
        },
    ]
    if s == 0:
      env_list.append({"name": "WAIT_FOR_DAEMON_HOST", "value": ",".join(daemons)})
      env_list.append({"name": "RUNNER_SCRIPT", "value": runner_script})
    # Each slice gets its own copies: YAML would otherwise write the shared objects as anchors and aliases.
    pod_spec = {
        "restartPolicy": "Never",
        **copy.deepcopy(placement),
        "volumes": [{"name": "ramdisk", "emptyDir": {"medium": "Memory"}}],
        "containers": [
            {
                "name": "worker",
                "image": IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c", command],
                "env": env_list,
                "volumeMounts": [{"name": "ramdisk", "mountPath": "/ramdisk"}],
                "resources": {"requests": {"google.com/tpu": "4"}, "limits": {"google.com/tpu": "4"}},
            }
        ],
    }
    replicated_jobs.append(
        {
            "name": f"slice{s}",
            "replicas": 1,
            "template": {
                "metadata": {
                    "annotations": {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"}
                },
                "spec": {
                    "parallelism": hosts,
                    "completions": hosts,
                    "completionMode": "Indexed",
                    "backoffLimit": 0,
                    "template": {
                        "metadata": {"labels": {"app": MH_APP}, "annotations": dict(pod_annotations)},
                        "spec": pod_spec,
                    },
                },
            },
        }
    )
  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {
          "name": name,
          "namespace": "default",
          "labels": {"kueue.x-k8s.io/queue-name": "default", "app": MH_APP},
      },
      "spec": {
          # A spot preemption restarts every pod; --skip-completed resumes from the finished phases.
          "failurePolicy": {"maxRestarts": 2},
          "network": {"enableDNSHostnames": True, "publishNotReadyAddresses": True},
          "replicatedJobs": replicated_jobs,
      },
  }


# (run id, topology, num_slices, models, scan modes, HF cache, extra exports). `smoke` checks the multi-host plumbing
# on a small model first. 2x2x2 has the HBM of the earlier 2 x 2x2x1 runs over ICI instead of DCN, and 2x2x4 that of
# 4 x 2x2x1. When scanned fails before the load finishes, the harness skips unscanned for the same config, so a model
# whose scanned run is known to fail gets a separate unscanned run (`397b-u`).
# The `-2s` runs use 2 x 2x2x2 (4 hosts, ICI within a slice and DCN between them), the HBM of 2x2x4 and 4 x 2x2x1.
# `397b-2s` and `397b-4h` run unscanned first, so a scanned failure cannot skip it. `1600b-4s` (3,198 GB) uses
# 4 x 2x2x2 (8 hosts, 6,510 GB HBM).
BOTH = "scanned unscanned"
SPARSE = "export STAGE_SPARSE_DEEPSEEK_OOM=1"
MH_RUNS = [
    ("smoke", "2x2x2", 1, "qwen3-0.6b", BOTH, P2_HF_CACHE, ""),
    ("235b", "2x2x2", 1, "qwen3-235b-a22b", BOTH, P1_HF_CACHE, ""),
    ("284b", "2x2x2", 1, "deepseek4-284b", BOTH, P1_HF_CACHE, SPARSE),
    ("397b", "2x2x2", 1, "qwen3.5-397b-a17b", BOTH, P1_HF_CACHE, ""),
    ("397b-u", "2x2x2", 1, "qwen3.5-397b-a17b", "unscanned", P1_HF_CACHE, ""),
    ("480b", "2x2x2", 1, "qwen3-480b-a35b", BOTH, P1_HF_CACHE, ""),
    ("671b", "2x2x4", 1, "deepseek3-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("397b-4h", "2x2x4", 1, "qwen3.5-397b-a17b", "unscanned scanned", P1_HF_CACHE, ""),
    ("smoke-2s", "2x2x2", 2, "qwen3-0.6b", BOTH, P2_HF_CACHE, ""),
    ("235b-2s", "2x2x2", 2, "qwen3-235b-a22b", BOTH, P1_HF_CACHE, ""),
    ("284b-2s", "2x2x2", 2, "deepseek4-284b", BOTH, P1_HF_CACHE, SPARSE),
    ("397b-2s", "2x2x2", 2, "qwen3.5-397b-a17b", "unscanned scanned", P1_HF_CACHE, ""),
    ("480b-2s", "2x2x2", 2, "qwen3-480b-a35b", BOTH, P1_HF_CACHE, ""),
    ("671b-2s", "2x2x2", 2, "deepseek3-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("ds32-2s", "2x2x2", 2, "deepseek3.2-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("1600b-4s", "2x2x2", 4, "deepseek4-1600b", BOTH, P1_HF_CACHE, SPARSE),
]

# bodaborg-v5p-nap runs for the DeepSeek checklist of the v5p sheet (single slice and multi slice, multi host), in the
# same tuple format. Each layout keeps the model under ~50% of HBM: 284B on 2x2x4 (1,520 GB HBM), 671B and 1600B on
# 4x4x4 (6,080 GB) for one slice, and 2 x 2x2x4 (3,040 GB) or 2 x 4x4x4 for two.
V5P_MH_RUNS = [
    ("v5p-smoke-2s", "2x2x2", 2, "qwen3-0.6b", BOTH, P2_HF_CACHE, ""),
    ("v5p-284b", "2x2x4", 1, "deepseek4-284b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-284b-2s", "2x2x4", 2, "deepseek4-284b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-671b", "4x4x4", 1, "deepseek3-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-671b-2s", "2x2x4", 2, "deepseek3-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-ds32", "4x4x4", 1, "deepseek3.2-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-ds32-2s", "2x2x4", 2, "deepseek3.2-671b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-1600b", "4x4x4", 1, "deepseek4-1600b", BOTH, P1_HF_CACHE, SPARSE),
    ("v5p-1600b-2s", "4x4x4", 2, "deepseek4-1600b", BOTH, P1_HF_CACHE, SPARSE),
]


def build_manifests():
  out_dir = os.environ.get("MANIFEST_OUT_DIR", "/tmp/tpu7x_manifests")
  os.makedirs(out_dir, exist_ok=True)
  for fname, plist in [
      ("tpu7x_p1_pods.yaml", P1_PAIRS),
      ("tpu7x_p2_pods.yaml", P2_PAIRS),
      ("tpu7x_benchmark_pods.yaml", P1_PAIRS + P2_PAIRS),
      ("tpu7x_p1_completion_pods.yaml", P1C_PAIRS),
  ]:
    docs = pairs_to_docs(plist)
    p = os.path.join(out_dir, fname)
    with open(p, "w", encoding="utf-8") as f:
      yaml.safe_dump_all(docs, f, sort_keys=False)
    print(f"Wrote {len(docs)} resources to {p}")

  # Controller that keeps the surviving slice-1 daemon of pair f (mesakh-tpu7x-f1) draining the P1 queue.
  p = os.path.join(out_dir, "tpu7x_ctrl_f1_pod.yaml")
  with open(p, "w", encoding="utf-8") as f:
    yaml.safe_dump(make_daemon_controller_pod("mesakh-ctrl-f1", "mesakh-w1-f:55555", "f1", QUEUE_P1), f, sort_keys=False)
  print(f"Wrote 1 resource to {p}")

  # Recovery after the runner pod of pair g was evicted: its daemon mesakh-tpu7x-g1 becomes slice 0 (driven by the
  # CPU controller mesakh-ctrl-g1) and a new daemon pod mesakh-tpu7x-h1 becomes slice 1. Until h1 is admitted, the
  # queue worker on g1 only takes single-slice tasks.
  h_docs = [d for d in pairs_to_docs([("h", "")]) if d["metadata"]["name"] != "mesakh-tpu7x-h0"]
  ctrl = make_daemon_controller_pod(
      "mesakh-ctrl-g1", "mesakh-w1-g:55555", "g1", QUEUE_P1, max_slices=2, remote="mesakh-w1-h:55555"
  )
  p = os.path.join(out_dir, "tpu7x_p1_completion_recovery.yaml")
  with open(p, "w", encoding="utf-8") as f:
    yaml.safe_dump_all(h_docs + [ctrl], f, sort_keys=False)
  print(f"Wrote {len(h_docs) + 1} resources to {p}")

  scaleout = pairs_to_docs(P1C_SCALEOUT_PAIRS)
  for wid, script in P1C_SCALEOUT_SINGLES:
    scaleout += single_worker_docs(wid, script)
  for gid, n_slices, script in P1C_SCALEOUT_GROUPS:
    scaleout += group_to_docs(gid, n_slices, script)
  p = os.path.join(out_dir, "tpu7x_p1_completion_scaleout.yaml")
  with open(p, "w", encoding="utf-8") as f:
    yaml.safe_dump_all(scaleout, f, sort_keys=False)
  print(f"Wrote {len(scaleout)} resources to {p}")

  scaleout2 = pairs_to_docs(P1C_SCALEOUT2_PAIRS)
  for wid, script in P1C_SCALEOUT2_SINGLES:
    scaleout2 += single_worker_docs(wid, script)
  p = os.path.join(out_dir, "tpu7x_p1_completion_scaleout2.yaml")
  with open(p, "w", encoding="utf-8") as f:
    yaml.safe_dump_all(scaleout2, f, sort_keys=False)
  print(f"Wrote {len(scaleout2)} resources to {p}")

  for platform, runs in (("tpu7x", MH_RUNS), ("v5p", V5P_MH_RUNS)):
    for run_id, topology, num_slices, models_str, scan_modes, hf_cache, extra_exports in runs:
      script = make_multihost_runner_script(
          run_id,
          models_str,
          topology,
          num_slices,
          hf_cache,
          platform=platform,
          scan_modes=scan_modes,
          extra_exports=extra_exports,
      )
      doc = multihost_jobset_doc(f"{MH_OWNER}-mh-{run_id}", topology, num_slices, script, platform=platform)
      p = os.path.join(out_dir, f"{platform}_multihost_{run_id}.yaml")
      with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
      print(f"Wrote 1 resource to {p}")


if __name__ == "__main__":
  build_manifests()
