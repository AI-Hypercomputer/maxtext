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

"""Runs the units of a benchmark v3 task file, one task per worker JobSet, and logs what happens.

  driver.py --tasks tasks.json --state-dir DIR [--lanes v7x=2 v6e=2 v5p=3 v5e=1] [--delete-finished-jobsets]

Rules it enforces:
  * One task per pod: every task gets a new JobSet, whose queue worker takes exactly one task and exits.
  * The tasks of a unit (baseline and candidate of one configuration) run back to back in one lane.
  * At most one unit with 64 or more hosts runs per region, because of the GCS egress quota.
  * The run directory of a task must be empty before its first attempt.
  * A unit starts only when the weights of its model are in the region: the Step 1 result (for DeepSeek the Step 1b
    or regional copy result) is OK, and every .safetensors file has the size that its header implies.
  * A task whose pods died is not retried silently: the results of the attempt are copied to
    <run dir>/attempts/a<N>/, the reason is logged, and the next attempt re-runs the phases without an OK result.
  * Nothing is submitted to a cluster on which v2 benchmark pods are running, nor to v5e without free slices.

It never deletes anything on GCS. It deletes a finished JobSet of its own only with --delete-finished-jobsets, after
saving the pod list and the runner log. State is in DIR/state.json, so a restarted driver continues where it was;
events are appended to DIR/events.log, one line each: `<UTC time> <INFO|ATTN> <text>`.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl import common  # pylint: disable=wrong-import-position
from wl import wl_bench  # pylint: disable=wrong-import-position
from bench import manifests_v2  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

DEFAULT_LANES = {"v7x": 2, "v6e": 2, "v5p": 3, "v5e": 1}
BIG_UNIT_HOSTS = 64
V5E_NODES = 128
QUEUE_TAIL = "  --prefer-max-slices --no-wait\n"
# The queue worker exits 2 after its one task and 3 when another worker holds the task: neither is a failed pod.
ONE_TASK_TAIL = (
    '  --prefer-max-slices --no-wait --max-tasks 1 || rc=$?\ncase "${rc:-0}" in 0|2|3) exit 0;; *) exit "$rc";; esac\n'
)


def task_key(task: dict) -> str:
  return f"{task['build']}/{task['id']}"


def phase_result_uris(task: dict) -> dict[str, str]:
  """Returns the result JSON of every measured phase of `task`, by `<scan mode>/<phase>`."""
  args = task["args"]
  scans = args[args.index("--scan-modes") + 1 : args.index("--num-slices-list")]
  out = {}
  for scan in scans:
    multi = scan + (f"_slices{task['slices']}" if task["slices"] > 1 else "")
    multi += f"_hosts{task['hosts_per_slice']}" if task["hosts_per_slice"] > 1 else ""
    phases = (
        {"safetensors_dynamic": multi}
        if task["pipe"] == "sd"
        else {
            "to_maxtext_step1": scan,
            "to_maxtext_step2": scan,
            "to_maxtext_step3": multi,
        }
    )
    for phase, sub in phases.items():
      out[f"{scan}/{phase}"] = f"{task['run_dir']}/logs/{task['model']}/{sub}/{phase}_result.json"
  return out


def summarize_results(task: dict) -> tuple[bool, str]:
  """Returns whether every phase of `task` is OK with valid cache evidence, and a one-line summary."""
  parts, good = [], True
  for name, uri in phase_result_uris(task).items():
    rec = common.gcs_cat_json(uri)
    if not isinstance(rec, dict):
      parts.append(f"{name}=MISSING")
      good = False
      continue
    status = rec.get("run_status") or rec.get("status")
    total = rec.get("safetensors_dynamic_step2_total_seconds", rec.get(f"{rec.get('phase_name')}_total_seconds"))
    valid = rec.get("cache_valid")
    text = f"{name}={status}"
    if isinstance(total, (int, float)):
      text += f" {total:.0f}s"
    if status == "OK" and valid is not True:
      text += f" CACHE-INVALID({rec.get('cache_invalid_reason')})"
    if status != "OK":
      text += f" [{rec.get('failure_category')}: {str(rec.get('error_summary'))[:120]}]"
    good &= status == "OK" and valid is True
    parts.append(text)
  return good, " | ".join(parts)


class Driver:
  """The state machine of one driver run: lanes per platform, each running one unit at a time."""

  def __init__(self, args):
    self.args = args
    with open(args.tasks, "r", encoding="utf-8") as f:
      plan = json.load(f)
    self.phase = plan["phase"]
    self.units = [u for u in plan["units"] if not args.platforms or u["platform"] in args.platforms]
    os.makedirs(os.path.join(args.state_dir, "pods"), exist_ok=True)
    os.makedirs(os.path.join(args.state_dir, "logs"), exist_ok=True)
    self.state_path = os.path.join(args.state_dir, "state.json")
    self.events_path = os.path.join(args.state_dir, "events.log")
    self.state = {"tasks": {}, "notes": {}}
    if os.path.exists(self.state_path):
      with open(self.state_path, "r", encoding="utf-8") as f:
        self.state = json.load(f)
    self.weights_asked: dict[str, float] = {}
    self.needs_step1b = {m["model_name"]: bool(m.get("needs_step1b")) for m in schema_v2.load_models_v2()["models"]}
    self.lanes = dict(DEFAULT_LANES)
    for item in args.lanes or []:
      name, _, count = item.partition("=")
      self.lanes[name] = int(count)

  # ---- bookkeeping ----
  def event(self, level: str, text: str) -> None:
    """Prints an event and appends it to the event log."""
    line = f"{common.utc_now()} {level} {text}"
    print(line, flush=True)
    with open(self.events_path, "a", encoding="utf-8") as f:
      f.write(line + "\n")

  def note_once(self, key: str, level: str, text: str) -> None:
    """Logs `text` when it differs from the last text logged under `key`."""
    if self.state["notes"].get(key) != text:
      self.state["notes"][key] = text
      if text:
        self.event(level, text)

  def save(self) -> None:
    """Writes the state file atomically."""
    tmp = self.state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(self.state, f, indent=1)
    os.replace(tmp, self.state_path)

  def task_state(self, task: dict) -> dict:
    """Returns the saved state of `task`: its status and its attempts."""
    return self.state["tasks"].setdefault(task_key(task), {"status": "pending", "attempts": []})

  def unit_status(self, unit: dict) -> str:
    """Returns `pending`, `running` or `finished` for `unit`, from the statuses of its tasks."""
    statuses = [self.task_state(t)["status"] for t in unit["tasks"]]
    if all(s in ("done", "failed", "blocked") for s in statuses):
      return "finished"
    return "pending" if all(s == "pending" for s in statuses) else "running"

  # ---- cluster checks ----
  def cluster_free(self, unit: dict) -> bool:
    """Returns whether `unit` may be submitted now: no v2 benchmark pods on its cluster, and room on v5e."""
    platform = unit["platform"]
    res = common.kubectl(
        platform,
        "get",
        "pods",
        "-A",
        "-l",
        f"app={common.V2_APP}",
        "--field-selector=status.phase=Running",
        "-o",
        "name",
        check=False,
    )
    v2_pods = len(res.stdout.split()) if res.returncode == 0 else -1
    if v2_pods:
      self.note_once(
          f"v2:{platform}", "ATTN", f"HOLD {platform}: {v2_pods} running {common.V2_APP} pods (-1 = query failed)"
      )
      return False
    self.note_once(f"v2:{platform}", "INFO", "")
    if platform != "v5e":
      return True
    res = common.kubectl(
        platform, "get", "pods", "-A", "--field-selector=status.phase=Running", "-o", "json", check=False
    )
    if res.returncode:
      return False
    busy = sum(
        1
        for pod in json.loads(res.stdout)["items"]
        if any("google.com/tpu" in c.get("resources", {}).get("limits", {}) for c in pod["spec"]["containers"])
    )
    need = unit["slices"] * unit["hosts_per_slice"]
    if V5E_NODES - busy < need:
      self.note_once("v5e:busy", "ATTN", f"HOLD v5e: {busy} TPU pods running, {need} hosts needed of {V5E_NODES}")
      return False
    self.note_once("v5e:busy", "INFO", "")
    return True

  def weights_ready(self, unit: dict) -> bool:
    """Returns whether the weights of the unit's model are complete in its region; asks GCS at most every 5 min."""
    if self.args.skip_weights_check:
      return True
    platform, model = unit["platform"], unit["model"]
    root = common.regional_root(platform)
    key = f"{root}/{model}"
    known = self.state.setdefault("weights", {})
    if key in known:
      return known[key] == "ok"
    if time.time() - self.weights_asked.get(key, 0.0) < 300:
      return False
    self.weights_asked[key] = time.time()
    if self.needs_step1b[model]:
      at_source = platform == wl_bench.STEP1B_SOURCE_PLATFORM
      result = f"{root}/step1b/{model}_{'step1b' if at_source else 'copy'}_result.json"
      weights = f"{root}/hf_cache_bf16/{model}"
    else:
      result, weights = f"{root}/step1/{model}_step1_result.json", f"{root}/hf_cache/{model}"
    rec = common.gcs_cat_json(result)
    if not isinstance(rec, dict) or rec.get("run_status") != "OK":
      return False
    count, size, bad = wl_bench.scan_weights([weights])
    if bad or not count:
      known[key] = f"bad: {bad[:3] or 'no .safetensors files'}"
      self.event("ATTN", f"WEIGHTS_BAD {weights}: {known[key]}")
      return False
    known[key] = "ok"
    self.event("INFO", f"WEIGHTS_OK {weights}: {count} files, {size / 1e9:.1f} GB, headers match sizes")
    return True

  # ---- launching ----
  def launch(self, unit: dict, task: dict) -> None:
    """Starts the next attempt of `task`: its own queue with this one task, and a worker JobSet that takes it."""
    del unit
    ts = self.task_state(task)
    attempt = len(ts["attempts"]) + 1
    if attempt == 1 and common.gcs_ls(task["run_dir"]):
      # Not ours to overwrite, unless an earlier driver run (e.g. the smoke) finished this very task.
      if any("/done/" in u for u in common.gcs_ls(task["queue_root"], recursive=True)):
        good, summary = summarize_results(task)
        ts["status"] = "done"
        self.event("INFO" if good else "ATTN", f"ALREADY_DONE {task_key(task)} by an earlier driver run: {summary}")
      else:
        ts["status"] = "blocked"
        self.event("ATTN", f"BLOCKED {task_key(task)}: run directory is not empty: {task['run_dir']}")
      return
    queue = f"{task['queue_root']}/a{attempt}"
    queued = copy.deepcopy(task)
    if attempt > 1:
      # A retry runs again every phase that has no OK result; the earlier results are in attempts/.
      queued["env"]["SKIP_FAILED_RESULTS"] = "0"
    common.gcs_write_verified(f"{queue}/tasks.json", json.dumps({"tasks": [queued]}, indent=1))
    # The name depends on the task and the attempt only, so a restarted driver finds the JobSet it created.
    worker = f"wl3{task['build'][0]}{hashlib.sha1(task_key(task).encode()).hexdigest()[:8]}a{attempt}"
    doc = manifests_v2.build_queue_worker_jobset(
        platform=task["platform"],
        topology=task["topology"],
        num_slices=task["slices"],
        queue_uri=queue,
        worker_id=worker,
        run_id=f"{worker}-{task['platform']}",
        results_gcs_uri=f"{common.regional_root(task['platform'])}/runs-{task['build']}/csv",
        experiment_name=common.EXPERIMENT,
        min_slices=task["slices"],
        code_dir=common.code_dir(task["build"]),
    )
    doc["spec"]["failurePolicy"]["maxRestarts"] = 0
    patched = 0
    for job in doc["spec"]["replicatedJobs"]:
      for env in job["template"]["spec"]["template"]["spec"]["containers"][0]["env"]:
        if env["name"] == "RUNNER_SCRIPT" and env.get("value"):
          if not env["value"].endswith(QUEUE_TAIL):
            raise ValueError("the queue worker script changed; update ONE_TASK_TAIL")
          env["value"] = env["value"][: -len(QUEUE_TAIL)] + ONE_TASK_TAIL
          patched += 1
    if patched != 1:
      raise ValueError(f"expected one runner script in the worker JobSet, found {patched}")
    name = doc["metadata"]["name"]
    verb = "LAUNCH"
    if not common.apply_manifest(task["platform"], doc):
      if name not in self.jobsets(task["platform"]):
        ts["status"] = "blocked"
        self.event("ATTN", f"BLOCKED {task_key(task)}: {name} exists and is not labelled app={common.APP}")
        return
      verb = "ADOPT"  # Created by a driver that stopped before it saved its state.
    ts["status"] = "running"
    ts["attempts"].append({"n": attempt, "jobset": name, "queue": queue, "started": common.utc_now()})
    self.event("INFO", f"{verb} {task_key(task)} attempt {attempt}: {name}")

  # ---- polling ----
  def jobsets(self, platform: str) -> dict[str, dict]:
    """Returns our JobSets on `platform` by name."""
    res = common.kubectl(platform, "get", "jobsets", "-l", f"app={common.APP}", "-o", "json", check=False)
    if res.returncode:
      return {}
    return {j["metadata"]["name"]: j for j in json.loads(res.stdout)["items"]}

  def pods(self, platform: str, jobset: str) -> list[dict]:
    """Returns name, node, phase, exit reason, start time and image of every pod of `jobset`."""
    res = common.kubectl(
        platform, "get", "pods", "-l", f"jobset.sigs.k8s.io/jobset-name={jobset}", "-o", "json", check=False
    )
    if res.returncode:
      return []
    out = []
    for pod in json.loads(res.stdout)["items"]:
      status = pod.get("status", {})
      container = (status.get("containerStatuses") or [{}])[0]
      terminated = container.get("state", {}).get("terminated") or {}
      out.append(
          {
              "name": pod["metadata"]["name"],
              "node": pod["spec"].get("nodeName", ""),
              "phase": status.get("phase", ""),
              "reason": status.get("reason", "") or terminated.get("reason", ""),
              "message": status.get("message", ""),
              "exit_code": terminated.get("exitCode"),
              "start_time": status.get("startTime", ""),
              "image_id": container.get("imageID", ""),
          }
      )
    return out

  def snapshot_pods(self, platform: str, attempt: dict) -> list[dict]:
    """Saves the pod list of the attempt's JobSet to the state directory and returns it."""
    pods = self.pods(platform, attempt["jobset"])
    if pods:
      with open(os.path.join(self.args.state_dir, "pods", f"{attempt['jobset']}.json"), "w", encoding="utf-8") as f:
        json.dump(pods, f, indent=1)
    return pods

  def finish_jobset(self, platform: str, attempt: dict) -> None:
    """Saves the runner log of a finished JobSet and, if allowed, deletes the JobSet."""
    name = attempt["jobset"]
    runner = f"jobset.sigs.k8s.io/jobset-name={name},jobset.sigs.k8s.io/replicatedjob-name=slice0"
    runner += ",batch.kubernetes.io/job-completion-index=0"
    log = common.kubectl(platform, "logs", "-l", runner, "--tail=-1", check=False)
    if log.returncode == 0 and log.stdout:
      with open(os.path.join(self.args.state_dir, "logs", f"{name}.log"), "w", encoding="utf-8") as f:
        f.write(log.stdout)
    if self.args.delete_finished_jobsets:
      common.kubectl(platform, "delete", "jobset", name, "-l", f"app={common.APP}", "--wait=false", check=False)

  def poll(self, unit: dict, task: dict, jobsets: dict[str, dict]) -> None:
    """Checks the running attempt of `task`; when its JobSet has ended, records the task or retries it."""
    ts = self.task_state(task)
    attempt = ts["attempts"][-1]
    platform, key = task["platform"], task_key(task)
    jobset = jobsets.get(attempt["jobset"])
    if jobset is None:
      self.note_once(f"gone:{key}", "ATTN", f"MISSING JobSet {attempt['jobset']} of {key}")
      return
    terminal = jobset.get("status", {}).get("terminalState")
    if not terminal:
      now = time.time()
      if jobset["spec"].get("suspend"):
        # Kueue has not admitted the JobSet yet, or has evicted it and will resume it with new pods.
        since = attempt.setdefault("suspended_since", now)
        if now - since > 4 * 3600:
          self.note_once(f"susp:{key}", "ATTN", f"SUSPENDED {attempt['jobset']} ({key}) for over 4 h")
        elif "running_since" in attempt:
          self.note_once(f"susp:{key}", "ATTN", f"SUSPENDED {attempt['jobset']} ({key}) after it had started")
        return
      attempt.pop("suspended_since", None)
      if "running_since" not in attempt:
        pods = self.snapshot_pods(platform, attempt)
        want = task["slices"] * task["hosts_per_slice"]
        if len(pods) == want and all(p["phase"] == "Running" for p in pods):
          attempt["running_since"] = common.utc_now()
          self.event("INFO", f"RUNNING {attempt['jobset']} ({key}): {want} pods")
        elif now - attempt.setdefault("created_epoch", now) > 1800:
          pending = sum(1 for p in pods if p["phase"] == "Pending")
          self.note_once(
              f"pend:{key}", "ATTN", f"WAITING {attempt['jobset']} ({key}): {pending}/{want} pods Pending for over 30 min"
          )
      return

    pods = self.snapshot_pods(platform, attempt)
    attempt["ended"] = common.utc_now()
    attempt["terminal_state"] = terminal
    done = any(u.endswith(f"/{task['id']}.json") for u in common.gcs_ls(f"{attempt['queue']}/done"))
    self.finish_jobset(platform, attempt)
    if done:
      good, summary = summarize_results(task)
      ts["status"] = "done"
      attempt["outcome"] = "done"
      self.event("INFO" if good else "ATTN", f"TASK_DONE {key} attempt {attempt['n']}: {summary}")
      return
    bad = sorted(
        {
            f"{p['reason'] or p['phase']}(exit {p['exit_code']}) {p['message']}".strip()
            for p in pods
            if p["phase"] != "Succeeded"
        }
    )
    reason = f"JobSet {terminal}; pods: {', '.join(bad) or 'none left'}"
    attempt["outcome"], attempt["reason"] = "interrupted", reason
    archive = f"{task['run_dir']}/attempts/a{attempt['n']}"
    if common.gcs_ls(f"{task['run_dir']}/logs"):
      common.run(["gcloud", "storage", "cp", "-r", f"{task['run_dir']}/logs", archive], check=False, timeout=1800)
    if attempt["n"] >= self.args.max_attempts:
      ts["status"] = "failed"
      self.event("ATTN", f"TASK_FAILED {key}: attempt {attempt['n']} interrupted ({reason}); no attempts left")
      return
    self.event("ATTN", f"TASK_INTERRUPTED {key}: attempt {attempt['n']} ({reason}); results kept in {archive}; retrying")
    self.launch(unit, task)

  # ---- main loop ----
  def step(self) -> bool:
    """Advances every lane once. Returns whether any unit is unfinished."""
    for platform in common.PLATFORMS:
      units = [(i, u) for i, u in enumerate(self.units) if u["platform"] == platform]
      running = [(i, u) for i, u in units if self.unit_status(u) == "running"]
      jobsets = self.jobsets(platform) if running else {}
      for _, unit in running:
        for task in unit["tasks"]:
          ts = self.task_state(task)
          if ts["status"] == "running":
            self.poll(unit, task, jobsets)
            break
          if ts["status"] == "pending":
            self.launch(unit, task)
            break
      running = [(i, u) for i, u in units if self.unit_status(u) == "running"]
      free = self.lanes.get(platform, 1) - len(running)
      for _, unit in units:
        if free <= 0:
          break
        if self.unit_status(unit) != "pending":
          continue
        hosts = unit["slices"] * unit["hosts_per_slice"]
        region = platforms.get_profile(platform).gcp_region
        if hosts >= BIG_UNIT_HOSTS and any(
            u["slices"] * u["hosts_per_slice"] >= BIG_UNIT_HOSTS
            and platforms.get_profile(u["platform"]).gcp_region == region
            and self.unit_status(u) == "running"
            for u in self.units
        ):
          continue
        if not self.weights_ready(unit):
          continue
        if not self.cluster_free(unit):
          break
        self.launch(unit, unit["tasks"][0])
        free -= 1
    self.save()
    return any(self.unit_status(u) != "finished" for u in self.units)

  def summary(self) -> str:
    """Returns the number of tasks per status and platform, as one line."""
    counts: dict[str, dict[str, int]] = {}
    for unit in self.units:
      for task in unit["tasks"]:
        per = counts.setdefault(unit["platform"], {})
        status = self.task_state(task)["status"]
        per[status] = per.get(status, 0) + 1
    return " | ".join(f"{p}: " + " ".join(f"{k}={v}" for k, v in sorted(c.items())) for p, c in counts.items())

  def run(self) -> None:
    """Runs until every unit is finished."""
    self.event("INFO", f"DRIVER_START phase={self.phase} units={len(self.units)} lanes={self.lanes}")
    last_beat = time.time()
    while self.step():
      if time.time() - last_beat >= self.args.heartbeat_s:
        self.event("INFO", f"HEARTBEAT {self.summary()}")
        last_beat = time.time()
      time.sleep(self.args.poll_s)
    self.event("INFO", f"DRIVER_DONE {self.summary()}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--tasks", required=True, help="Task file written by plan.py tasks")
  parser.add_argument("--state-dir", required=True)
  parser.add_argument("--platforms", nargs="+", choices=common.PLATFORMS)
  parser.add_argument("--lanes", nargs="*", help="Units running at once per platform, e.g. v5p=3")
  parser.add_argument("--max-attempts", type=int, default=3)
  parser.add_argument("--poll-s", type=int, default=60)
  parser.add_argument("--heartbeat-s", type=int, default=1800)
  parser.add_argument("--delete-finished-jobsets", action="store_true")
  parser.add_argument("--skip-weights-check", action="store_true", help="Do not wait for Step 1 results (tests only)")
  common.use_v3_profiles()
  Driver(parser.parse_args()).run()


if __name__ == "__main__":
  main()
