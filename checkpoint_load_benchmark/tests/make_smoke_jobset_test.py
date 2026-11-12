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

"""Tests scripts/make_smoke_jobset.py for v7x, v6e, and v5e smoke JobSets."""

import importlib.util
import os
import re
import shutil
import tempfile
import unittest
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GEN_PATH = os.path.join(_REPO_ROOT, "checkpoint_load_benchmark", "scripts", "make_smoke_jobset.py")

_spec = importlib.util.spec_from_file_location("make_smoke_jobset", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

DNS_LABEL_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


class MakeSmokeJobsetTest(unittest.TestCase):
  """Verifies generated JobSet manifests match preflight cluster rules and mesa- naming."""

  def _env_map(self, doc: dict) -> dict[str, dict]:
    pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    return {entry["name"]: entry for entry in pod_spec["containers"][0]["env"]}

  def test_v7x_v6e_and_v5e_manifests(self):
    cases = [
        ("v7x", "2x2x1", 1, "4", 8, "default", True, "60", None, None),
        ("v6e", "2x4", 1, "8", 8, "default", False, "60", "medium", None),
        ("v5e", "16x16", 64, "4", 256, None, False, None, None, "mesa-bench"),
    ]
    for platform, topo, hosts, tpu_req, devices, queue, tas, duration, prio, sa in cases:
      with self.subTest(platform=platform, topology=topo):
        doc = gen.build_smoke_jobset(
            platform=platform,
            topology=topo,
            model="qwen3-0.6b",
            run_id="smoke_test_20261112_long_id",
        )
        name = doc["metadata"]["name"]
        self.assertTrue(name.startswith("mesa-smoke-"))
        self.assertLessEqual(len(name), 40)
        self.assertRegex(name, DNS_LABEL_RE)

        meta_labels = doc["metadata"]["labels"]
        self.assertEqual(meta_labels["app"], "mesa-bench-v2")
        self.assertIn("mesa-run-id", meta_labels)
        if queue is None:
          self.assertNotIn("kueue.x-k8s.io/queue-name", meta_labels)
        else:
          self.assertEqual(meta_labels.get("kueue.x-k8s.io/queue-name"), queue)

        self.assertEqual(doc["spec"]["failurePolicy"], {"maxRestarts": 0})
        self.assertEqual(doc["spec"]["network"], {"enableDNSHostnames": True, "publishNotReadyAddresses": True})

        rjob = doc["spec"]["replicatedJobs"][0]
        job_tmpl = rjob["template"]
        self.assertEqual(
            job_tmpl["metadata"]["annotations"]["alpha.jobset.sigs.k8s.io/exclusive-topology"],
            "cloud.google.com/gke-nodepool",
        )
        job_spec = job_tmpl["spec"]
        self.assertEqual(job_spec["parallelism"], hosts)
        self.assertEqual(job_spec["completions"], hosts)
        self.assertEqual(job_spec["completionMode"], "Indexed")
        self.assertEqual(job_spec["backoffLimit"], 0)

        pod_meta = job_spec["template"]["metadata"]
        pod_labels = pod_meta["labels"]
        self.assertEqual(pod_labels["app"], "mesa-bench-v2")
        self.assertEqual(pod_labels.get("declared-duration-minutes"), duration)

        pod_ann = pod_meta.get("annotations", {})
        if tas:
          self.assertEqual(pod_ann.get("kueue.x-k8s.io/podset-slice-required-topology"), "cloud.google.com/gke-nodepool")
          self.assertEqual(pod_ann.get("kueue.x-k8s.io/podset-slice-size"), str(hosts))
          self.assertEqual(pod_ann.get("kueue.x-k8s.io/podset-unconstrained-topology"), "true")
        else:
          self.assertEqual(pod_ann, {})

        pod_spec = job_spec["template"]["spec"]
        self.assertEqual(pod_spec.get("priorityClassName"), prio)
        self.assertEqual(pod_spec.get("serviceAccountName"), sa)
        self.assertEqual(pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-topology"], topo)
        if platform == "v7x":
          self.assertIn("imagePullSecrets", pod_spec)
          self.assertIn("affinity", pod_spec)
        else:
          self.assertNotIn("imagePullSecrets", pod_spec)
          self.assertNotIn("affinity", pod_spec)

        res = pod_spec["containers"][0]["resources"]
        self.assertEqual(res["requests"]["google.com/tpu"], tpu_req)
        self.assertEqual(res["limits"]["google.com/tpu"], tpu_req)

        env = self._env_map(doc)
        self.assertEqual(env["BENCH_CODE_DIR"]["value"], "gs://mesa-maxtext/mesa-bench-v2/code")
        hf_ref = env["HF_TOKEN"]["valueFrom"]["secretKeyRef"]
        self.assertEqual(hf_ref["name"], "mesa-hf-token")
        self.assertTrue(hf_ref["optional"])
        for downward in ("POD_NAME", "POD_NAMESPACE", "NODE_NAME"):
          self.assertIn("fieldRef", env[downward]["valueFrom"])

        daemons_raw = env["WAIT_FOR_DAEMON_HOST"]["value"]
        daemons = [d for d in daemons_raw.split(",") if d]
        self.assertEqual(len(daemons), hosts - 1)
        if hosts > 1:
          self.assertEqual(daemons[0], f"{name}-slice0-0-1.{name}")
          self.assertEqual(daemons[-1], f"{name}-slice0-0-{hosts - 1}.{name}")

        runner_script = env["RUNNER_SCRIPT"]["value"]
        self.assertIn("--hardware tpu", runner_script)
        self.assertIn(f"--devices-per-slice {devices}", runner_script)
        self.assertIn(f"--num-hosts-per-slice {hosts}", runner_script)
        self.assertIn("--gcs-base gs://mesa-maxtext/mesa-bench-v2/smoke", runner_script)

  def test_pipeline_flags_and_cli_out(self):
    st_doc = gen.build_smoke_jobset("v7x", "2x2x1", "qwen3-0.6b", "s1", pipelines="safetensors_dynamic")
    self.assertIn("--skip-to-maxtext", self._env_map(st_doc)["RUNNER_SCRIPT"]["value"])
    self.assertNotIn("--skip-safetensors", self._env_map(st_doc)["RUNNER_SCRIPT"]["value"])

    tm_doc = gen.build_smoke_jobset("v7x", "2x2x1", "qwen3-0.6b", "s1", pipelines="to_maxtext")
    self.assertIn("--skip-safetensors", self._env_map(tm_doc)["RUNNER_SCRIPT"]["value"])
    self.assertNotIn("--skip-to-maxtext", self._env_map(tm_doc)["RUNNER_SCRIPT"]["value"])

    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    out_file = os.path.join(tmp_dir, "smoke.yaml")
    rc = gen.main(
        ["--platform", "v5e", "--topology", "16x16", "--model", "qwen3-0.6b", "--run-id", "s1", "--out", out_file]
    )
    self.assertEqual(rc, 0)
    with open(out_file, "r", encoding="utf-8") as f:
      loaded = yaml.safe_load(f)
    self.assertEqual(loaded["kind"], "JobSet")
    self.assertEqual(loaded["spec"]["replicatedJobs"][0]["template"]["spec"]["parallelism"], 64)


if __name__ == "__main__":
  unittest.main()
