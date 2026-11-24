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

"""Tests the benchmark v3 plan and driver against a fake cluster and a fake GCS."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from wl import common  # pylint: disable=wrong-import-position
from wl import driver  # pylint: disable=wrong-import-position
from wl import plan  # pylint: disable=wrong-import-position


class FakeWorld:
  """Stands in for kubectl and gcloud: JobSets created by the driver, and the GCS objects it reads and writes."""

  def __init__(self):
    self.jobsets: dict[str, dict] = {}
    self.objects: dict[str, str] = {}
    self.copies: list[tuple[str, str]] = []
    self.v2_pods = 0

  def apply_manifest(self, platform, doc, dry_run=False):
    """Records the JobSet instead of creating it."""
    del dry_run
    common.check_isolated(doc)
    name = doc["metadata"]["name"]
    if name in self.jobsets:
      return False
    self.jobsets[name] = {"platform": platform, "doc": doc, "terminal": None}
    return True

  def kubectl(self, platform, *args, input_text=None, check=True):
    """Answers the `kubectl get` calls of the driver from the recorded JobSets."""
    del input_text, check
    if args[:2] == ("get", "jobsets"):
      items = [
          {"metadata": {"name": n}, "spec": {}, "status": {"terminalState": j["terminal"]} if j["terminal"] else {}}
          for n, j in self.jobsets.items()
          if j["platform"] == platform
      ]
      return types.SimpleNamespace(returncode=0, stdout=json.dumps({"items": items}))
    if args[:2] == ("get", "pods") and f"app={common.V2_APP}" in args:
      return types.SimpleNamespace(returncode=0, stdout="pod/x\n" * self.v2_pods)
    if args[:2] == ("get", "pods"):
      return types.SimpleNamespace(returncode=0, stdout=json.dumps({"items": []}))
    return types.SimpleNamespace(returncode=1, stdout="")

  def gcs_ls(self, uri):
    """Lists the fake objects under `uri`."""
    prefix = uri.rstrip("/") + "/"
    return sorted(k for k in self.objects if k.startswith(prefix))

  def gcs_write_verified(self, uri, text):
    """Stores a fake object."""
    self.objects[uri] = text

  def gcs_cat_json(self, uri):
    """Returns a fake object as JSON, or None."""
    return json.loads(self.objects[uri]) if uri in self.objects else None

  def run(self, cmd, **kwargs):
    """Records recursive copies; every other command succeeds silently."""
    del kwargs
    if cmd[:4] == ["gcloud", "storage", "cp", "-r"]:
      self.copies.append((cmd[4], cmd[5]))
    return types.SimpleNamespace(returncode=0, stdout="")

  def env_of(self, name):
    """Returns the environment of the runner container of JobSet `name`."""
    containers = self.jobsets[name]["doc"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    return {e["name"]: e.get("value") for e in containers["containers"][0]["env"]}

  def finish(self, name, task, ok=True, results=True):
    """Ends JobSet `name`; with `ok`, as the queue worker does after its task: a done marker and phase results."""
    self.jobsets[name]["terminal"] = "Completed" if ok else "Failed"
    if not ok:
      return
    queue = self.env_of(name)["RUNNER_SCRIPT"].split("--queue ")[1].split()[0]
    self.objects[f"{queue}/done/{task['id']}.json"] = "{}"
    if results:
      for uri in driver.phase_result_uris(task).values():
        rec = {"run_status": "OK", "cache_valid": True, "safetensors_dynamic_step2_total_seconds": 12.0}
        self.objects[uri] = json.dumps(rec)


class WlDriverTest(unittest.TestCase):
  """Verifies the plan and the driver's launch, ordering, retry and hold rules."""

  def setUp(self):
    """Replaces the kubectl and gcloud helpers of `common` with the fake world."""
    common.use_v3_profiles()
    self.addCleanup(common.restore_v2_profiles)
    self.world = FakeWorld()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    for name in ("apply_manifest", "kubectl", "gcs_ls", "gcs_write_verified", "gcs_cat_json", "run"):
      patcher = mock.patch.object(common, name, getattr(self.world, name))
      patcher.start()
      self.addCleanup(patcher.stop)

  def _driver(self, units, **overrides):
    """Returns a driver for `units` with its state in the test's temporary directory."""
    tasks_path = os.path.join(self.tmp, "tasks.json")
    with open(tasks_path, "w", encoding="utf-8") as f:
      json.dump({"phase": "sd", "units": units}, f)
    args = argparse.Namespace(
        tasks=tasks_path,
        state_dir=os.path.join(self.tmp, "state"),
        platforms=None,
        lanes=None,
        max_attempts=2,
        poll_s=0,
        heartbeat_s=10**9,
        delete_finished_jobsets=False,
    )
    for key, value in overrides.items():
      setattr(args, key, value)
    return driver.Driver(args)

  def _events(self):
    with open(os.path.join(self.tmp, "state", "events.log"), "r", encoding="utf-8") as f:
      return f.read()

  def test_planned_rows_match_the_matrix_and_tasks_stay_inside_wl_bench_v3(self):
    rows = plan.planned_rows()
    candidate_sd = [r for r in rows if r["build"] == "candidate" and r["pipeline"] == "safetensors_dynamic"]
    by_platform = {p: sum(1 for r in candidate_sd if r["platform"] == p) for p in common.PLATFORMS}
    self.assertEqual(by_platform, {"v7x": 82, "v6e": 70, "v5p": 82, "v5e": 24})
    self.assertEqual({r["model"] for r in rows if r["build"] == "baseline"}, set(plan.BASELINE_SUBSET))
    self.assertFalse([r for r in rows if r["build"] == "baseline" and r["pipeline"] == "to_maxtext"])
    for unit in plan.build_units("smoke"):
      for task in unit["tasks"]:
        text = json.dumps(task)
        self.assertNotIn(common.V2_APP, text)
        self.assertIn(f"/{common.APP}/runs-{task['build']}/", task["run_dir"])
        self.assertIn("--verify-checksums", task["args"])
        self.assertEqual(task["env"]["BENCH_FULL_CHECKSUMS"], "1")
        self.assertIn("--skip-to-maxtext" if task["pipe"] == "sd" else "--skip-safetensors", task["args"])

  def test_a_pair_runs_back_to_back_in_one_task_per_jobset(self):
    unit = plan.build_units("sd", ["v5p"], ["qwen3-14b"])[0]
    first, second = unit["tasks"]
    self.assertEqual({first["build"], second["build"]}, {"baseline", "candidate"})
    drv = self._driver([unit])
    self.assertTrue(drv.step())
    self.assertEqual(len(self.world.jobsets), 1)
    name = next(iter(self.world.jobsets))
    doc = self.world.jobsets[name]["doc"]
    self.assertEqual(doc["spec"]["failurePolicy"]["maxRestarts"], 0)
    env = self.world.env_of(name)
    self.assertIn("--max-tasks 1 || rc=$?", env["RUNNER_SCRIPT"])
    self.assertEqual(env["BENCH_CODE_DIR"], common.code_dir(first["build"]))
    queued = json.loads(self.world.objects[f"{first['queue_root']}/a1/tasks.json"])["tasks"]
    self.assertEqual([t["id"] for t in queued], [first["id"]])
    self.assertTrue(drv.step())  # Still running: the second task must wait.
    self.assertEqual(len(self.world.jobsets), 1)
    self.world.finish(name, first)
    self.assertTrue(drv.step())  # Records the first task ...
    self.assertTrue(drv.step())  # ... and launches the second.
    self.assertEqual(len(self.world.jobsets), 2)
    second_name = [n for n in self.world.jobsets if n != name][0]
    self.assertEqual(self.world.env_of(second_name)["BENCH_CODE_DIR"], common.code_dir(second["build"]))
    self.world.finish(second_name, second)
    self.assertFalse(drv.step())
    self.assertEqual(self._events().count("TASK_DONE"), 2)
    self.assertNotIn("ATTN", self._events())

  def test_an_interrupted_task_keeps_its_results_and_is_retried_once_more(self):
    unit = plan.build_units("tm", ["v5p"], ["qwen3-0.6b"])[0]
    task = unit["tasks"][0]
    drv = self._driver([unit])
    drv.step()
    first = next(iter(self.world.jobsets))
    self.world.objects[f"{task['run_dir']}/logs/x/to_maxtext_step1_result.json"] = "{}"
    self.world.finish(first, task, ok=False)
    self.assertTrue(drv.step())
    self.assertIn((f"{task['run_dir']}/logs", f"{task['run_dir']}/attempts/a1"), self.world.copies)
    retry = json.loads(self.world.objects[f"{task['queue_root']}/a2/tasks.json"])["tasks"][0]
    self.assertEqual(retry["env"]["SKIP_FAILED_RESULTS"], "0")
    self.assertEqual(len(self.world.jobsets), 2)
    self.world.finish([n for n in self.world.jobsets if n != first][0], task, ok=False)
    self.assertFalse(drv.step())
    self.assertIn("TASK_INTERRUPTED", self._events())
    self.assertIn("TASK_FAILED", self._events())

  def test_a_used_run_directory_and_running_v2_pods_block_a_launch(self):
    unit = plan.build_units("tm", ["v6e"], ["qwen3-0.6b"])[0]
    task = unit["tasks"][0]
    self.world.v2_pods = 3
    drv = self._driver([unit])
    self.assertTrue(drv.step())
    self.assertFalse(self.world.jobsets)
    self.assertIn("HOLD v6e", self._events())
    self.world.v2_pods = 0
    self.world.objects[f"{task['run_dir']}/logs/old.json"] = "{}"
    self.assertFalse(drv.step())
    self.assertFalse(self.world.jobsets)
    self.assertIn("run directory is not empty", self._events())

  def test_only_one_64_host_unit_runs_per_region_and_state_survives_a_restart(self):
    units = plan.build_units("tm", ["v5e"], ["qwen3-0.6b", "llama3.1-8b"])[:2]
    drv = self._driver(units, lanes=["v5e=2"])
    drv.step()
    self.assertEqual(len(self.world.jobsets), 1)
    restarted = self._driver(units, lanes=["v5e=2"])
    restarted.step()
    self.assertEqual(len(self.world.jobsets), 1)
    self.world.finish(next(iter(self.world.jobsets)), units[0]["tasks"][0])
    restarted.step()
    restarted.step()
    self.assertEqual(len(self.world.jobsets), 2)


if __name__ == "__main__":
  unittest.main()
