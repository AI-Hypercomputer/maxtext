# Copyright 2026 Google LLC
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

"""CPU-only tests of render_e2e.py on a hand-written DRY_RUN fixture (no cluster, no Tunix, no JAX)."""

import io
import os
import re
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import render_e2e as R  # pylint: disable=g-import-not-at-top,wrong-import-position
import structdiff_render as SD  # pylint: disable=g-import-not-at-top,wrong-import-position

FIXTURE = os.path.join(HERE, "testdata", "e2e_dryrun_fixture.yaml")
DRIVER = os.path.join(PKG, "e2e_driver.py")

pytestmark = pytest.mark.cpu_only


def _client_env(js):
  """The driver's inline env (E2E_TRAINER_ARGV, E2E_ORCH_ARGV, MAXTEXT_EXTRA_FLAGS, ...) of a rendered JobSet."""
  _, main_c = R._proc_main(js)  # pylint: disable=protected-access
  line = next(l for l in main_c["command"][-1].splitlines() if "exec python /tmp/e2e_driver.py" in l)
  env = {}
  for t in shlex.split(line):
    k, eq, v = t.partition("=")
    if eq:
      env[k] = v
  return env


class RenderE2ETest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

  def tearDown(self):
    self.tmp.cleanup()

  def render(self, *extra, name="tb-test"):
    out = os.path.join(self.tmp.name, f"{name}.yaml")
    argv = ["--e2e-render", FIXTURE, "--driver", DRIVER, "--out", out, "--name", name, "--namespace", "bench-ns",
            "--output-prefix", "/bench-out", *extra]
    with redirect_stdout(io.StringIO()):
      self.assertEqual(R.main(argv), 0)
    with open(out) as f:
      return yaml.safe_load(f), out

  def test_selects_trainer_and_orchestrator(self):
    docs = R.load_documents(FIXTURE)
    self.assertEqual([d["metadata"]["name"] for d in docs], ["example-orch", "example-train", "example-roll-0"])
    train, orch = R.select_jobsets(docs)
    self.assertEqual(train["metadata"]["name"], "example-train")
    self.assertEqual(orch["metadata"]["name"], "example-orch")
    with self.assertRaises(ValueError):
      R.select_jobsets(docs + [train])  # two trainers

  def test_base_render(self):
    js, _ = self.render()
    md = js["metadata"]
    self.assertEqual((md["name"], md["namespace"]), ("tb-test", "bench-ns"))
    self.assertEqual(md["labels"]["scheduling.x-k8s.io/gang-id"], "tb-test")
    self.assertEqual(md["labels"]["kueue.x-k8s.io/queue-name"], "example-queue")  # scheduling fields kept
    self.assertEqual(js["spec"]["failurePolicy"]["maxRestarts"], 0)
    proc, main_c = R._proc_main(js)  # pylint: disable=protected-access
    self.assertNotIn("ports", main_c)
    self.assertEqual(main_c["image"], "example.invalid/tunix:tag")
    script = main_c["command"][-1]
    self.assertIn("cat > /tmp/e2e_driver.py <<'E2E_DRIVER_EOF'", script)
    self.assertNotIn("FT_REGISTERED_MARKER", script)  # Tunix fail-fast wrapper dropped
    with open(DRIVER) as f:
      self.assertIn(f.read(), script)

    env = _client_env(js)
    targv = shlex.split(env["E2E_TRAINER_ARGV"])
    self.assertIn("--maxtext_ckpt_path=", targv)
    self.assertIn("--maxtext_output_directory=/bench-out/tb-test", targv)
    self.assertIn("--worker_id=tb-test", targv)
    self.assertFalse([t for t in targv if t.startswith(("--port=", "--discovery_addrs=", "--process_main="))])
    self.assertEqual([t for t in targv if t.startswith("--mesh_fsdp=")], ["--mesh_fsdp=32"])
    oargv = shlex.split(env["E2E_ORCH_ARGV"])
    self.assertIn("--trainer_fsdp=32", oargv)
    self.assertIn("--seq_logprob_error_threshold=2.0", oargv)
    self.assertEqual((env["STEPS"], env["MICRO_STEPS"]), ("3", "4"))
    self.assertEqual(env["MAXTEXT_EXTRA_FLAGS"], "context=offload num_moe_token_chunks=4 packing=True")
    # Without libtpu knobs, the proxy and worker flags are untouched.
    proxy = R.find(proc["initContainers"], "pathways-proxy")
    self.assertEqual(proxy["args"][-1], "--sidecar_name=external")
    self.assertIn("--xla_tpu_flag_b=4", proxy["args"])

  def test_libtpu_knobs_reach_all_three_places(self):
    js, _ = self.render("--libtpu-extra=--xla_tpu_flag_b=8 --xla_new=1", "--libtpu-drop=--xla_tpu_flag_a")
    env = _client_env(js)
    self.assertEqual(env["LIBTPU_INIT_ARGS"].split(), ["--RUNTIME_only_flag=true", "--xla_tpu_flag_b=8", "--xla_new=1"])
    proc, _ = R._proc_main(js)  # pylint: disable=protected-access
    proxy = R.find(proc["initContainers"], "pathways-proxy")["args"]
    self.assertEqual([a for a in proxy if a.startswith("--xla_")], ["--xla_tpu_flag_b=8", "--xla_new=1"])
    self.assertEqual(proxy[-1], "--sidecar_name=external")
    self.assertFalse([a for a in proxy if a.startswith("--RUNTIME")])  # runtime-only flags stay off the proxy
    pw = R.find(js["spec"]["replicatedJobs"], "pw-node")["template"]["spec"]["template"]["spec"]
    wenv = R.find(R.find(pw["containers"], "pathways-worker")["env"], "LIBTPU_INIT_ARGS")["value"].split()
    self.assertEqual(wenv, ["--RUNTIME_only_flag=true", "--xla_tpu_flag_b=8", "--xla_new=1", "--worker_only_flag=1"])

  def test_maxtext_flag_knobs(self):
    js, _ = self.render("--replace-maxtext-flags", "context=remat", "--extra-maxtext-flags", "log_moe_drops=true")
    self.assertEqual(_client_env(js)["MAXTEXT_EXTRA_FLAGS"],
                     "context=remat num_moe_token_chunks=4 packing=True log_moe_drops=true")
    with self.assertRaises(ValueError):
      self.render("--replace-maxtext-flags", "not_a_key=1")

  def test_trainer_and_orch_flag_knobs(self):
    js, _ = self.render("--set-trainer-flags", "train_micro_batch_size=128", "--set-orch-flags",
                        "train_micro_batch_size=128", "--compute-logps-chunk-size", "1024",
                        "--profiler-skip", "16", "--profiler-steps", "3")
    env = _client_env(js)
    targv = shlex.split(env["E2E_TRAINER_ARGV"])
    for flag in ("--train_micro_batch_size=128", "--compute_logps_chunk_size=1024", "--profiler_steps=3",
                 "--skip_first_n_profiler_steps=16"):
      self.assertEqual(sum(t.split("=", 1)[0] == flag.split("=", 1)[0] for t in targv), 1, flag)
      self.assertIn(flag, targv)
    self.assertIn("--train_micro_batch_size=128", shlex.split(env["E2E_ORCH_ARGV"]))
    with self.assertRaises(ValueError):
      self.render("--set-trainer-flags", "no_such_flag=1")
    with self.assertRaises(ValueError):
      self.render("--set-orch-flags", "no_such_flag=1")

  def test_trainer_slice(self):
    base, base_path = self.render(name="tb-a")
    js, path = self.render("--trainer-slice", "4x8x8", name="tb-b")
    proc, _ = R._proc_main(js)  # pylint: disable=protected-access
    rm = R.find(proc["initContainers"], "pathways-rm")["args"]
    self.assertIn("--instance_type=tpu7x:4x8x8", rm)
    pw = R.find(js["spec"]["replicatedJobs"], "pw-node")["template"]["spec"]
    self.assertEqual((pw["completions"], pw["parallelism"]), (64, 64))
    self.assertEqual(pw["template"]["metadata"]["annotations"]["cloud.google.com/gke-tpu-slice-topology"], "4x8x8")
    env = _client_env(js)
    self.assertIn("--mesh_fsdp=64", shlex.split(env["E2E_TRAINER_ARGV"]))
    self.assertIn("--trainer_fsdp=64", shlex.split(env["E2E_ORCH_ARGV"]))
    # Only the documented fields (plus the names) differ from the 4x4x8 render.
    buf = io.StringIO()
    with redirect_stdout(buf):
      SD.main_args(base_path, path)
    diffs = [l for l in buf.getvalue().splitlines() if l.startswith("[")]
    keys = sorted(re.match(r"\[(\w+)\] (\S+?):", l).group(1, 2) for l in diffs)
    self.assertEqual(keys, sorted([
        ("leaf", ".metadata.name"), ("leaf", ".metadata.labels.scheduling.x-k8s.io/gang-id"),
        ("leaf", ".spec.replicatedJobs[proc].template.spec.template.spec.initContainers[pathways-rm].args[3]"),
        ("leaf", ".spec.replicatedJobs[pw-node].template.spec.completions"),
        ("leaf", ".spec.replicatedJobs[pw-node].template.spec.parallelism"),
        ("leaf", ".spec.replicatedJobs[pw-node].template.spec.template.metadata.annotations."
                 "cloud.google.com/gke-tpu-slice-topology"),
        ("trainer_argv", "--maxtext_output_directory"), ("trainer_argv", "--worker_id"),
        ("trainer_argv", "--mesh_fsdp"), ("orch_argv", "--trainer_fsdp"),
    ]))
    del base

  def test_driver_env_and_image(self):
    js, _ = self.render("--driver-env", "PAYLOAD=realistic ROUTING_TARGETS=1.5,2.0", "--image", "example.invalid/x:y",
                        "--steps", "2", "--micro-steps", "5")
    env = _client_env(js)
    self.assertEqual((env["PAYLOAD"], env["ROUTING_TARGETS"], env["STEPS"], env["MICRO_STEPS"]),
                     ("realistic", "1.5,2.0", "2", "5"))
    self.assertEqual(R._proc_main(js)[1]["image"], "example.invalid/x:y")  # pylint: disable=protected-access
    with self.assertRaises(ValueError):
      self.render("--driver-env", "lower=1")

  def test_print_aot_env(self):
    buf = io.StringIO()
    with redirect_stdout(buf):
      R.main(["--print-aot-env", "--e2e-render", FIXTURE, "--trainer-slice", "4x8x8", "--libtpu-extra=--xla_new=1"])
    out = buf.getvalue()
    self.assertIn("export MAXTEXT_EXTRA_FLAGS='context=offload num_moe_token_chunks=4 packing=True'", out)
    self.assertIn("export LIBTPU_INIT_ARGS='--xla_tpu_flag_a=true --xla_tpu_flag_b=4 --xla_new=1'", out)
    self.assertIn("export AOT_MESH_FSDP=64", out)
    self.assertIn("export AOT_LOGPS_CHUNK_SIZE=512", out)

  def test_name_validation(self):
    with self.assertRaises(SystemExit):
      self.render(name="Bad_Name")


if __name__ == "__main__":
  unittest.main()
