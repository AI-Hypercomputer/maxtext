#!/usr/bin/env python3
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

"""Autonomous background orchestrator for Checkpoint-Load Benchmark v2 across v7x, v6e, and v5e."""

import argparse
from collections.abc import Mapping, Sequence
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import manifests_v2  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

DEFAULT_SPREADSHEET_ID = "1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I"
MAX_CPU_JOBS_BY_PLATFORM = {"v5e": 3, "v6e": 5, "v7x": 5}
TIER_ORDER = {"S": 0, "A": 1, "C": 2, "B": 3}


def utc_now_str() -> str:
  """Returns current UTC timestamp formatted as `YYYY-MM-DD HH:MM:SS UTC`."""
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _run_cmd(
    cmd: Sequence[str], *, input_text: str | None = None, timeout: int = 180
) -> subprocess.CompletedProcess[str]:
  """Runs a subprocess command without raising on non-zero exit."""
  return subprocess.run(
      list(cmd),
      input=input_text,
      text=True,
      capture_output=True,
      timeout=timeout,
      check=False,
  )


def apply_k8s_manifests(context: str, docs: Sequence[Mapping[str, Any]]) -> bool:
  """Applies one or more Kubernetes JSON manifests to `context`."""
  for doc in docs:
    payload = json.dumps(doc, indent=2) + "\n"
    res = _run_cmd(["kubectl", f"--context={context}", "apply", "-f", "-"], input_text=payload, timeout=30)
    if res.returncode != 0:
      print(f"[WARN] kubectl apply failed on {context}: {res.stderr}", flush=True)
      return False
  return True


def sync_gcs_to_mirror(mirror_dir: str) -> int:
  """Syncs regional GCS `runs` (excluding `to_maxtext/` weights), `val`, `step1`, `step1b`, and `queue/v2/done`."""
  for plat in ("v7x", "v6e", "v5e"):
    prof = platforms.get_profile(plat)
    region = prof.gcp_region
    gcs_root = (prof.regional_gcs_data_root or prof.gcs_data_root).rstrip("/")
    for sub in ("runs", "val", "step1", "step1b", "queue/v2/done"):
      dst = os.path.join(mirror_dir, region, sub)
      os.makedirs(dst, exist_ok=True)
      cmd = ["gcloud", "storage", "rsync", "-r", "--exclude=(^|.*/)to_maxtext/.*", f"{gcs_root}/{sub}", dst]
      _run_cmd(cmd, timeout=120)

  total_jsons = 0
  for _, _, files in os.walk(mirror_dir):
    for fname in files:
      if fname.endswith(".json"):
        total_jsons += 1
  return total_jsons


def load_completed_step1_and_1b(mirror_dir: str, platform: str) -> tuple[set[str], set[str]]:
  """Returns `(done_step1_models, done_step1b_models)` for `platform` from mirrored JSON files."""
  prof = platforms.get_profile(platform)
  region_dir = os.path.join(mirror_dir, prof.gcp_region)
  done_s1: set[str] = set()
  done_s1b: set[str] = set()

  s1_dir = os.path.join(region_dir, "step1")
  if os.path.isdir(s1_dir):
    for fname in os.listdir(s1_dir):
      if not fname.endswith("_step1_result.json"):
        continue
      fpath = os.path.join(s1_dir, fname)
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          rec = json.load(f)
        if rec.get("run_status") == "OK" and rec.get("model_name"):
          done_s1.add(str(rec["model_name"]))
      except (OSError, ValueError):
        continue

  s1b_dir = os.path.join(region_dir, "step1b")
  if os.path.isdir(s1b_dir):
    for fname in os.listdir(s1b_dir):
      if not fname.endswith(".json"):
        continue
      fpath = os.path.join(s1b_dir, fname)
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          rec = json.load(f)
        if rec.get("run_status") == "OK" and rec.get("model_name"):
          done_s1b.add(str(rec["model_name"]))
      except (OSError, ValueError):
        continue

  return done_s1, done_s1b


def load_done_queue_task_ids(mirror_dir: str, platform: str) -> set[str]:
  """Returns task IDs present under `mirror_dir/<region>/queue/v2/done/*.json`."""
  prof = platforms.get_profile(platform)
  done_dir = os.path.join(mirror_dir, prof.gcp_region, "queue/v2/done")
  done_ids: set[str] = set()
  if os.path.isdir(done_dir):
    for fname in os.listdir(done_dir):
      if fname.endswith(".json"):
        done_ids.add(fname[:-5])
  return done_ids


def manage_cpu_jobs(
    platform: str,
    done_s1: set[str],
    done_s1b: set[str],
    all_done_s1b: Mapping[str, set[str]],
    events: list[str],
) -> list[str]:
  """Cleans up finished standalone CPU jobs and launches next needed Step 1 / 1b / copy jobs."""
  prof = platforms.get_profile(platform)
  ctx = prof.kube_context
  cfg = schema_v2.load_models_v2()
  team_buckets: Mapping[str, str] = cfg.get("deepseek_bf16_team_buckets", {})

  res = _run_cmd(["kubectl", f"--context={ctx}", "get", "jobs", "-l", "app=mesa-bench-v2", "-o", "json"])
  active_job_names: list[str] = []
  active_models_s1: set[str] = set()
  active_models_s1b: set[str] = set()

  if res.returncode == 0 and res.stdout.strip():
    try:
      items = json.loads(res.stdout).get("items", [])
    except ValueError:
      items = []
    for item in items:
      meta = item.get("metadata", {})
      if meta.get("ownerReferences"):
        continue
      jname = str(meta.get("name") or "")
      status = item.get("status", {})
      succeeded = int(status.get("succeeded") or 0)
      failed = int(status.get("failed") or 0)
      matched_model = ""
      for m in cfg["models"]:
        m_slug = manifests_v2.dns_slug(m["model_name"])
        if f"-{m_slug}-" in jname:
          matched_model = m["model_name"]
          break
      if succeeded >= 1:
        if matched_model:
          if jname.startswith("mesa-step1-"):
            done_s1.add(matched_model)
          elif jname.startswith(("mesa-copy-", "mesa-step1b-")):
            done_s1b.add(matched_model)
        _run_cmd(["kubectl", f"--context={ctx}", "delete", "job", jname, "--ignore-not-found=true"])
        events.append(f"[{utc_now_str()}] `{platform}` CPU job `{jname}` Completed and cleaned up.")
        continue
      if failed >= 2:
        _run_cmd(["kubectl", f"--context={ctx}", "delete", "job", jname, "--ignore-not-found=true"])
        events.append(f"[{utc_now_str()}] `{platform}` CPU job `{jname}` Failed and cleaned up.")
        continue
      active_job_names.append(jname)
      if matched_model:
        if jname.startswith("mesa-step1-"):
          active_models_s1.add(matched_model)
        elif jname.startswith(("mesa-copy-", "mesa-step1b-")):
          active_models_s1b.add(matched_model)

  max_cpu = MAX_CPU_JOBS_BY_PLATFORM.get(platform, 3)
  models_for_plat = [m for m in cfg["models"] if platform in m.get("topologies_by_platform", {})]
  models_for_plat.sort(key=lambda m: (TIER_ORDER.get(m["tier"], 99), float(m.get("loaded_bf16_gb") or 0.0)))

  for m in models_for_plat:
    if len(active_job_names) >= max_cpu:
      break
    m_name = m["model_name"]
    if m_name not in done_s1 and m_name not in active_models_s1:
      job = manifests_v2.build_step1_cpu_job(
          platform=platform,
          model_name=m_name,
          run_id=f"v2-s1-{platform}",
          clean_target=True,
      )
      if apply_k8s_manifests(ctx, [job]):
        jname = job["metadata"]["name"]
        active_job_names.append(jname)
        active_models_s1.add(m_name)
        events.append(f"[{utc_now_str()}] Launched `{platform}` Step 1 CPU job `{jname}` for `{m_name}`.")

    if len(active_job_names) >= max_cpu:
      break
    if m.get("needs_step1b") and m_name not in done_s1b and m_name not in active_models_s1b:
      if m_name in team_buckets:
        job = manifests_v2.build_copy_region_cpu_job(
            dest_platform=platform,
            model_name=m_name,
            source_gcs_dir=team_buckets[m_name],
            run_id=f"v2-cp-{platform}",
            clean_target=True,
            bf16_source="copied_from_maxtext_team_bucket",
        )
        if apply_k8s_manifests(ctx, [job]):
          jname = job["metadata"]["name"]
          active_job_names.append(jname)
          active_models_s1b.add(m_name)
          events.append(f"[{utc_now_str()}] Launched `{platform}` BF16 regional copy job `{jname}` for `{m_name}`.")
      elif platform == "v7x" and m_name in done_s1:
        docs = manifests_v2.build_deepseek_step1b_manifests(
            model_name=m_name,
            run_id="v2-s1b-v7x",
            clean_target=True,
        )
        if apply_k8s_manifests(ctx, docs):
          jname = docs[1]["metadata"]["name"]
          active_job_names.append(jname)
          active_models_s1b.add(m_name)
          events.append(f"[{utc_now_str()}] Launched `v7x` Step 1b conversion job `{jname}` for `{m_name}`.")
      elif platform in ("v6e", "v5e") and m_name in all_done_s1b.get("v7x", set()):
        v7x_prof = platforms.get_profile("v7x")
        src_bf16 = manifests_v2.regional_hf_cache_dir(v7x_prof, m_name, dequantized=True)
        job = manifests_v2.build_copy_region_cpu_job(
            dest_platform=platform,
            model_name=m_name,
            source_gcs_dir=src_bf16,
            run_id=f"v2-cp-{platform}",
            clean_target=True,
            bf16_source="converted_by_this_benchmark",
        )
        if apply_k8s_manifests(ctx, [job]):
          jname = job["metadata"]["name"]
          active_job_names.append(jname)
          active_models_s1b.add(m_name)
          events.append(f"[{utc_now_str()}] Launched `{platform}` BF16 copy job `{jname}` for `{m_name}`.")

  return active_job_names


def update_queue_and_tpu_workers(
    platform: str,
    done_s1: set[str],
    done_s1b: set[str],
    done_task_ids: set[str],
    active_cpu_jobs: Sequence[str],
    last_uploaded_task_ids: dict[str, set[str]],
    worker_seq: dict[str, int],
    events: list[str],
) -> tuple[list[str], int, int]:
  """Updates regional `tasks.json` with ready models and ensures a TPU worker JobSet is active."""
  prof = platforms.get_profile(platform)
  ctx = prof.kube_context
  cfg = schema_v2.load_models_v2()
  models_by_name = {m["model_name"]: m for m in cfg["models"]}

  busy_cpu_models: set[str] = set()
  for jname in active_cpu_jobs:
    for m_name in models_by_name:
      m_slug = manifests_v2.dns_slug(m_name)
      if f"-{m_slug}-" in jname:
        busy_cpu_models.add(m_name)

  staged_ready: set[str] = set()
  for m_name, m_info in models_by_name.items():
    if m_name in busy_cpu_models:
      continue
    if m_name in done_s1 and (not m_info.get("needs_step1b") or m_name in done_s1b):
      staged_ready.add(m_name)

  all_tasks = manifests_v2.build_planned_queue_tasks(
      platform=platform,
      experiment_name="mesa-bench-v2",
      run_prefix="v2",
      tiers=("S", "A", "C", "B"),
  )
  ready_tasks = [t for t in all_tasks if t["model"] in staged_ready]
  ready_ids = {str(t["id"]) for t in ready_tasks}

  root = (prof.regional_gcs_data_root or prof.gcs_data_root).rstrip("/")
  queue_uri = f"{root}/queue/v2"
  results_uri = f"{root}/runs"

  if ready_ids and ready_ids != last_uploaded_task_ids.get(platform, set()):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
      json.dump({"tasks": ready_tasks}, tf, indent=2)
      tmp_path = tf.name
    up = _run_cmd(["gcloud", "storage", "cp", tmp_path, f"{queue_uri}/tasks.json"], timeout=60)
    try:
      os.remove(tmp_path)
    except OSError:
      pass
    if up.returncode == 0:
      last_uploaded_task_ids[platform] = set(ready_ids)
      events.append(
          f"[{utc_now_str()}] Updated `{platform}` queue `{queue_uri}/tasks.json` "
          f"with {len(ready_tasks)}/{len(all_tasks)} staged tasks."
      )

  pending_ready_tasks = [t for t in ready_tasks if str(t["id"]) not in done_task_ids]

  res = _run_cmd(["kubectl", f"--context={ctx}", "get", "jobsets", "-l", "app=mesa-bench-v2", "-o", "json"])
  active_jobsets: list[str] = []
  has_queue_worker = False
  if res.returncode == 0 and res.stdout.strip():
    try:
      items = json.loads(res.stdout).get("items", [])
    except ValueError:
      items = []
    for item in items:
      js_name = str(item.get("metadata", {}).get("name") or "")
      term_state = str(item.get("status", {}).get("terminalState") or "")
      if term_state in ("Completed", "Failed"):
        _run_cmd(["kubectl", f"--context={ctx}", "delete", "jobset", js_name, "--ignore-not-found=true"])
        events.append(f"[{utc_now_str()}] `{platform}` TPU JobSet `{js_name}` `{term_state}` and cleaned up.")
        continue
      active_jobsets.append(js_name)
      if js_name.startswith(f"mesa-q-{platform}-"):
        has_queue_worker = True

  if pending_ready_tasks and not has_queue_worker and not active_jobsets:
    nxt = pending_ready_tasks[0]
    topo = str(nxt["topology"])
    slices = int(nxt["slices"])
    min_slices = 1
    if platform == "v6e" and topo == "2x4":
      same_topo_slices = {int(t["slices"]) for t in pending_ready_tasks if str(t["topology"]) == topo}
      slices = max(same_topo_slices)
      min_slices = min(same_topo_slices)
    else:
      min_slices = slices

    seq = worker_seq.get(platform, 1) + 1
    worker_seq[platform] = seq
    wid = f"w{seq}"
    js = manifests_v2.build_queue_worker_jobset(
        platform=platform,
        topology=topo,
        num_slices=slices,
        queue_uri=queue_uri,
        worker_id=wid,
        run_id=f"v2-{platform}-{wid}",
        results_gcs_uri=results_uri,
        min_slices=min_slices,
    )
    if apply_k8s_manifests(ctx, [js]):
      js_name = js["metadata"]["name"]
      active_jobsets.append(js_name)
      events.append(
          f"[{utc_now_str()}] Launched `{platform}` TPU queue worker `{js_name}` "
          f"(`{slices}x{topo}`, min_slices={min_slices}) for {len(pending_ready_tasks)} runnable tasks."
      )

  return active_jobsets, len(done_task_ids), len(all_tasks)


def sync_sheet_v2(mirror_dir: str, v1_dir: str, sync_dir: str, spreadsheet_id: str) -> bool:
  """Runs `build_sheet_v2.py --apply` with both v2 mirror and legacy v1 roots."""
  py_bin = sys.executable
  script = os.path.join(_BENCHMARK_DIR, "scripts", "build_sheet_v2.py")
  multihost_csv = os.path.join(_BENCHMARK_DIR, "reports", "tpu7x_multihost_results.csv")
  impl_json = os.path.join(sync_dir, "impl.json")
  cmd = [
      py_bin,
      script,
      "--v2-root",
      mirror_dir,
      "--v1-p2-root",
      os.path.join(v1_dir, "p2_lance_prs/run_tpu7x_16m/logs"),
      "--v1-p2-root",
      os.path.join(v1_dir, "p2_lance_prs/run_tpu7x_16m_rerun/logs"),
      "--v1-p2-root",
      os.path.join(v1_dir, "p2_lance_prs/run_tpu7x_16m_tmfix/logs"),
      "--v1-p1-root",
      os.path.join(v1_dir, "p1_lance_prs/run_tpu7x_200b/logs"),
      "--v1-p1-root",
      os.path.join(v1_dir, "p1_lance_prs/run_tpu7x_200b_pstream/logs"),
      "--v1-p1-root",
      os.path.join(v1_dir, "p1_lance_prs/run_tpu7x_200b_tmfix/logs"),
      "--v1-variant-root",
      f"with fix={os.path.join(v1_dir, 'p2_lance_prs/run_tpu7x_16m_fix/logs')}",
      "--v1-variant-root",
      f"with fix={os.path.join(v1_dir, 'p1_lance_prs/run_tpu7x_200b_fix/logs')}",
      "--v1-before-root",
      os.path.join(v1_dir, "p1_origin_main/run_tpu7x_200b/logs"),
      "--v1-before-root",
      os.path.join(v1_dir, "p1_origin_main/run_tpu7x_200b_t1800/logs"),
      "--v1-multihost-csv",
      multihost_csv,
      "--v1-cold-s1",
      "qwen3.5-397b-a17b=814.53",
      "--impl-json",
      impl_json,
      "--out-dir",
      sync_dir,
      "--spreadsheet-id",
      spreadsheet_id,
      "--apply",
  ]
  res = _run_cmd(cmd, timeout=300)
  return res.returncode == 0


def write_progress_markdown(
    progress_md: str,
    status_by_plat: Mapping[str, Mapping[str, Any]],
    events: Sequence[str],
    spreadsheet_id: str,
) -> None:
  """Writes live status and chat/event log to `progress_md` for team visibility."""
  repo_dir = os.path.dirname(_BENCHMARK_DIR)
  lines = [
      "# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log",
      "",
      f"- **Last Updated (UTC):** `{utc_now_str()}`",
      "- **Git Branch:** `docs/checkpoint-load-benchmark`",
      f"- **Repository Path on Cloudtop:** `{repo_dir}`",
      f"- **This Progress File:** `{progress_md}`",
      f"- **Handoff Runbook:** `{_BENCHMARK_DIR}/handoff.md`",
      f"- **Design & Task Plan:** `{_BENCHMARK_DIR}/plans/benchmark_v2_plan.md`",
      f"- **Live Google Sheet v2:** https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
      "- **Background Orchestrator Script:** `checkpoint_load_benchmark/scripts/run_v2_orchestrator.py`",
      "",
      "## 1. Live Cluster & Experiment Progress Summary",
      "",
      "| Platform | Region | Step 1 Done | TPU Queue Tasks Done | Active TPU JobSets | Active CPU Jobs |",
      "|---|---|---:|---:|---|---|",
  ]
  for plat in ("v7x", "v6e", "v5e"):
    st = status_by_plat.get(plat, {})
    s1_done = st.get("s1_done", 0)
    s1_total = st.get("s1_total", 0)
    q_done = st.get("q_done", 0)
    q_total = st.get("q_total", 0)
    js_list = ", ".join(f"`{x}`" for x in st.get("active_jobsets", [])) or "_none_"
    cpu_list = ", ".join(f"`{x}`" for x in st.get("active_cpu_jobs", [])) or "_none_"
    lines.append(
        f"| `{plat}` | `{st.get('region', '')}` | **{s1_done} / {s1_total}** | "
        f"**{q_done} / {q_total}** | {js_list} | {cpu_list} |"
    )

  lines.extend(
      [
          "",
          "## 2. Chat Session Summary & Handoff Guide for Other Engineers",
          "",
          "1. **Summary of What Has Been Completed in This Session:**",
          (
              "   - **Code & Tests (`docs/checkpoint-load-benchmark`):** Completed"
              " all Phase 0 through Phase 4 deliverables (`bench/schema_v2.py`,"
              " `bench/harness_v2.py`, `bench/platforms.py`, `bench/stages.py`,"
              " `bench/manifests_v2.py`, `scripts/generate_v2_manifests.py`,"
              " `scripts/run_v2_cluster.sh`, `scripts/build_sheet_v2.py`,"
              " `scripts/run_v2_orchestrator.py`). Fixed 64-host stage timing"
              " aggregation on `v5e` and TPU HBM metric keys on `v5e`/`v6e`/`v7x`."
              " All 94 unit tests pass (`pytest`) and all modular files score"
              " `10.00/10` on `pylint`."
          ),
          (
              "   - **Cross-Platform Validation (`T3.5`) & Sheet Sync (`T4.1-T4.3`):**"
              " Verified end-to-end on `v5e` (`16x16` = 64 hosts / 256 chips in"
              " `europe-west4`), `v6e` (`2x4` and `4x4` in `southamerica-west1`),"
              " and `v7x` (`2x2x1` in `us-central1`). Populated Google Sheet v2"
              f" (`{spreadsheet_id}`) with all 555 Summary rows (352 planned v2"
              " rows + 203 legacy v1 rows)."
          ),
          (
              "   - **Background Execution (`Phase 5`):** To avoid blocking the"
              " interactive Jetski chat session while multi-hour HuggingFace"
              " downloads and TPU benchmarks run, `scripts/run_v2_orchestrator.py`"
              " runs in the background (`nohup` daemon logging to"
              " `/tmp/mesa_v2_orchestrator.log`). Every 60 seconds it polls all"
              " 3 GKE clusters and regional GCS buckets, cleans up finished K8s"
              " Jobs/JobSets, launches the next Step 1 / Step 1b / TPU Queue"
              " Worker waves in `S -> A -> C -> B` tier order, syncs Google"
              " Sheet v2, and updates this Markdown file."
          ),
          "2. **How Another Engineer Can Inspect or Take Over on Cloudtop:**",
          "   ```bash",
          "   # 1. Go to the repo and verify branch is docs/checkpoint-load-benchmark",
          "   cd /usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3",
          "   git status",
          "   git pull origin docs/checkpoint-load-benchmark",
          "",
          "   # 2. Read this live progress file and tail the background orchestrator log",
          "   cat checkpoint_load_benchmark/PROGRESS_LOG.md",
          "   ps -ef | grep run_v2_orchestrator.py",
          "   tail -f /tmp/mesa_v2_orchestrator.log",
          "",
          "   # 3. Check live K8s jobs/jobsets/pods across all 3 clusters (v7x, v6e, v5e)",
          "   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v7x --mode status",
          "   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v6e --mode status",
          "   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v5e --mode status",
          "   ```",
          "",
          "## 3. Chronological Event Log (Newest Last)",
          "",
      ]
  )
  for ev in events[-150:]:
    lines.append(f"- {ev}")
  lines.append("")

  with open(progress_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the background orchestration loop across `v7x`, `v6e`, and `v5e`."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mirror-dir", required=True)
  parser.add_argument("--v1-dir", required=True)
  parser.add_argument("--sync-dir", required=True)
  parser.add_argument(
      "--progress-md",
      default=os.path.join(_BENCHMARK_DIR, "PROGRESS_LOG.md"),
  )
  parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
  parser.add_argument("--interval-seconds", type=int, default=60)
  parser.add_argument("--once", action="store_true", help="Run a single orchestration pass and exit")
  parser.add_argument("--skip-gcs-sync", action="store_true", help="Skip initial GCS rsync (use existing mirror)")
  args = parser.parse_args(argv)

  cfg = schema_v2.load_models_v2()
  expected_s1_by_plat = {
      plat: sum(1 for m in cfg["models"] if plat in m.get("topologies_by_platform", {})) for plat in ("v7x", "v6e", "v5e")
  }

  events: list[str] = [f"[{utc_now_str()}] Background orchestrator started on branch `docs/checkpoint-load-benchmark`."]
  last_uploaded_task_ids: dict[str, set[str]] = {}
  worker_seq: dict[str, int] = {"v7x": 1, "v6e": 1, "v5e": 1}
  last_json_count = -1

  while True:
    if args.skip_gcs_sync:
      json_count = last_json_count
    else:
      json_count = sync_gcs_to_mirror(args.mirror_dir)
    done_s1_by_plat: dict[str, set[str]] = {}
    done_s1b_by_plat: dict[str, set[str]] = {}
    done_q_by_plat: dict[str, set[str]] = {}
    for plat in ("v7x", "v6e", "v5e"):
      s1, s1b = load_completed_step1_and_1b(args.mirror_dir, plat)
      done_s1_by_plat[plat] = s1
      done_s1b_by_plat[plat] = s1b
      done_q_by_plat[plat] = load_done_queue_task_ids(args.mirror_dir, plat)

    status_by_plat: dict[str, dict[str, Any]] = {}
    all_complete = True
    for plat in ("v5e", "v6e", "v7x"):
      prof = platforms.get_profile(plat)
      active_cpu = manage_cpu_jobs(
          plat,
          done_s1_by_plat[plat],
          done_s1b_by_plat[plat],
          done_s1b_by_plat,
          events,
      )
      active_js, q_done, q_total = update_queue_and_tpu_workers(
          plat,
          done_s1_by_plat[plat],
          done_s1b_by_plat[plat],
          done_q_by_plat[plat],
          active_cpu,
          last_uploaded_task_ids,
          worker_seq,
          events,
      )
      s1_done = len(done_s1_by_plat[plat])
      s1_total = expected_s1_by_plat[plat]
      if s1_done < s1_total or q_done < q_total or active_cpu or active_js:
        all_complete = False
      status_by_plat[plat] = {
          "region": prof.gcp_region,
          "s1_done": s1_done,
          "s1_total": s1_total,
          "q_done": q_done,
          "q_total": q_total,
          "active_jobsets": active_js,
          "active_cpu_jobs": active_cpu,
      }

    if json_count != last_json_count:
      if sync_sheet_v2(args.mirror_dir, args.v1_dir, args.sync_dir, args.spreadsheet_id):
        events.append(
            f"[{utc_now_str()}] Synced Google Sheet v2 (`{args.spreadsheet_id}`) with {json_count} mirrored JSONs."
        )
        last_json_count = json_count

    write_progress_markdown(args.progress_md, status_by_plat, events, args.spreadsheet_id)

    if args.once or all_complete:
      if all_complete:
        events.append(f"[{utc_now_str()}] All Step 1 and TPU queue tasks completed across v7x, v6e, and v5e.")
        write_progress_markdown(args.progress_md, status_by_plat, events, args.spreadsheet_id)
      break

    time.sleep(max(10, args.interval_seconds))

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
