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

"""Tests bench/manifests_v2.py and scripts/generate_v2_manifests.py."""

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import yaml

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import manifests_v2  # pylint: disable=wrong-import-position
from scripts import generate_v2_manifests  # pylint: disable=wrong-import-position

DNS_LABEL_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


class ManifestsV2Test(unittest.TestCase):
  """Verifies generated CPU Jobs, DeepSeek PVCs, TPU JobSets, and queue tasks match cluster profiles."""

  def _env_map(self, pod_spec: dict) -> dict[str, dict]:
    return {e["name"]: e for e in pod_spec["containers"][0]["env"]}

  def test_step1_cpu_jobs_across_all_platforms(self):
    for plat, region, sa, pull_sec in (
        ("v7x", "us-central1", None, True),
        ("v6e", "southamerica-west1", None, False),
        ("v5e", "europe-west4", "mesa-bench", False),
    ):
      job = manifests_v2.build_step1_cpu_job(platform=plat, model_name="llama3.1-8b", run_id="r1")
      name = job["metadata"]["name"]
      self.assertTrue(name.startswith("mesa-step1-"))
      self.assertRegex(name, DNS_LABEL_RE)
      self.assertLessEqual(len(name), 63)

      pod_spec = job["spec"]["template"]["spec"]
      self.assertEqual(pod_spec["nodeSelector"], {"cloud.google.com/gke-nodepool": "cpu-np"})
      self.assertEqual(pod_spec.get("serviceAccountName"), sa)
      self.assertEqual("imagePullSecrets" in pod_spec, pull_sec)
      tol_keys = {(t["key"], t.get("value")) for t in pod_spec["tolerations"]}
      self.assertIn(("cloud.google.com/gke-nodepool", "cpu-np"), tol_keys)

      env = self._env_map(pod_spec)
      script = env["RUNNER_SCRIPT"]["value"]
      self.assertIn(f"gs://mesa-maxtext-{region}/mesa-bench-v2/hf_cache/llama3.1-8b", script)
      self.assertIn("step1_download_job.py download", script)

  def test_deepseek_step1b_pvc_and_job_manager_tolerations(self):
    pvc, job = manifests_v2.build_deepseek_step1b_manifests(model_name="deepseek4-1600b", run_id="ds1b")
    self.assertEqual(pvc["kind"], "PersistentVolumeClaim")
    self.assertGreaterEqual(int(pvc["spec"]["resources"]["requests"]["storage"].removesuffix("Gi")), 5000)

    pod_spec = job["spec"]["template"]["spec"]
    self.assertEqual(pod_spec["nodeSelector"], {"cloud.google.com/gke-nodepool": "job-manager-np"})
    tol_keys = {(t["key"], t.get("value")) for t in pod_spec["tolerations"]}
    self.assertIn(("dedicated", "job-manager"), tol_keys)
    script = self._env_map(pod_spec)["RUNNER_SCRIPT"]["value"]
    self.assertIn("deepseek-convert", script)
    self.assertIn("hf_cache_bf16/deepseek4-1600b", script)

  def test_multislice_multihost_queue_worker_daemon_addressing(self):
    doc = manifests_v2.build_queue_worker_jobset(
        platform="v5e",
        topology="16x16",
        num_slices=2,
        queue_uri="gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/q1",
        worker_id="v5e_w1",
        run_id="qrun1",
        results_gcs_uri="gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue_results",
    )
    js_name = doc["metadata"]["name"]
    self.assertTrue(js_name.startswith("mesa-q-"))
    self.assertNotIn("kueue.x-k8s.io/queue-name", doc["metadata"]["labels"])
    self.assertEqual(len(doc["spec"]["replicatedJobs"]), 2)

    s0_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]
    self.assertEqual(s0_spec["parallelism"], 64)
    env0 = self._env_map(s0_spec["template"]["spec"])
    daemons = [d for d in env0["WAIT_FOR_DAEMON_HOST"]["value"].split(",") if d]
    # 2 slices * 64 hosts - 1 runner = 127 daemons, starting with slice0 host 1 and ending with slice1 host 63.
    self.assertEqual(len(daemons), 127)
    self.assertEqual(daemons[0], f"{js_name}-slice0-0-1.{js_name}")
    self.assertEqual(daemons[63], f"{js_name}-slice1-0-0.{js_name}")
    self.assertEqual(daemons[-1], f"{js_name}-slice1-0-63.{js_name}")
    self.assertIn("--hosts-per-slice 64", env0["RUNNER_SCRIPT"]["value"])

  def test_planned_queue_tasks_and_cli_generation(self):
    tasks_v6e = manifests_v2.build_planned_queue_tasks(platform="v6e", experiment_name="exp2", run_prefix="v2")
    # Tier S (2) + Tier A (9 placements * 3 reps = 27) + Tier C (6) + Tier B (18) = 53 tasks.
    self.assertEqual(len(tasks_v6e), 53)
    tier_a_tasks = [t for t in tasks_v6e if t["tier"] == "A"]
    self.assertEqual(len(tier_a_tasks), 27)
    self.assertTrue(all("--verify-checksums" in t["args"] for t in tier_a_tasks))
    self.assertTrue(
        all(
            "STAGE_SPARSE_DEEPSEEK_OOM" in t["env"]
            for t in tasks_v6e
            if t["model"].startswith("deepseek3") or t["model"].startswith("deepseek4")
        )
    )

    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    q_json = os.path.join(tmp_dir, "tasks.json")
    cp_yaml = os.path.join(tmp_dir, "copy.yaml")
    self.assertEqual(
        generate_v2_manifests.main(["queue-tasks", "--platform", "v5e", "--run-prefix", "v2", "--out", q_json]),
        0,
    )
    with open(q_json, "r", encoding="utf-8") as f:
      loaded = json.load(f)
    self.assertEqual(loaded["platform"], "v5e")
    self.assertEqual(len(loaded["tasks"]), 22)

    self.assertEqual(
        generate_v2_manifests.main(
            [
                "copy-region",
                "--dest-platform",
                "v6e",
                "--model",
                "deepseek3-671b",
                "--source-gcs-dir",
                "gs://mesa-maxtext-us-central1/mesa-bench-v2/hf_cache_bf16/deepseek3-671b",
                "--run-id",
                "cp1",
                "--out",
                cp_yaml,
            ]
        ),
        0,
    )
    with open(cp_yaml, "r", encoding="utf-8") as f:
      docs = list(yaml.safe_load_all(f))
    self.assertEqual(docs[0]["kind"], "Job")


if __name__ == "__main__":
  unittest.main()
