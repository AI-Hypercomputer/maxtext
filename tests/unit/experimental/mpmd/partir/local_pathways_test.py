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

"""Tests for local_pathways.sh's arguments and XLA flags file, with docker stood in for (no container starts)."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from maxtext.experimental.mpmd import partir

SCRIPT = os.path.join(os.path.dirname(partir.__file__), "local_pathways.sh")
GCS = "--gcs_scratch_location=gs://bucket/pathways"
INSTANCE = "--instance_type=tpu7x:2x2x1"
UP = ("up", GCS, INSTANCE)
IMAGES = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways"
FLAGS_FILE = "# a comment\n--xla_a=1\n\n  --xla_b=true  \n"
# up's first docker call asks for existing containers: under DOCKER=echo the answer is echo's own output, so up stops.
EXISTING = "containers exist already"
# A docker that records its calls, has no containers, and whose logs say the slice and the proxy are ready.
FAKE_DOCKER = """#!/usr/bin/env bash
printf '%q ' "$@" >> {calls}; echo >> {calls}
case $1 in
  logs) echo '1/1 Pathways Slices Now Ready'; echo 'IFRT proxy server started with status OK' ;;
  inspect) echo true ;;
esac
"""


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class LocalPathwaysTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)

  def write(self, name, text, mode=0o644):
    path = os.path.join(self.tmp, name)
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(text)
    os.chmod(path, mode)
    return path

  def run_script(self, *args, docker="echo", bin_dir=None):
    env = dict(os.environ, DOCKER=docker)
    if bin_dir:
      env["PATH"] = bin_dir + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", SCRIPT, *args], env=env, capture_output=True, text=True, timeout=60, check=False)
    return result.returncode, result.stdout + result.stderr

  def test_refusals_come_before_any_docker_call(self):
    """Under DOCKER=echo any docker call would stop up as if containers existed: each refusal comes first."""
    good = self.write("good.flags", FLAGS_FILE)
    cases = [
        ((), 2, "usage:"),
        (("start",), 2, "usage:"),
        ((*UP, "--bogus"), 2, "unknown argument: --bogus"),
        (("up", INSTANCE), 2, "--gcs_scratch_location=gs://... is required"),
        (("up", "--gcs_scratch_location=/tmp/pathways", INSTANCE), 2, "--gcs_scratch_location=gs://... is required"),
        (("up", GCS), 2, "--instance_type is required"),
        ((*UP, f"--xla_flags_file={self.tmp}/missing.flags"), 1, "cannot read --xla_flags_file="),
    ]
    for i, line in enumerate(("xla_b=2", "--xla_a=1 --xla_b=2", "--xla_a", "--=1")):
      bad = self.write(f"bad{i}.flags", f"--xla_ok=1\n{line}\n")
      cases.append(((*UP, f"--xla_flags_file={bad}"), 1, f"expected one --key=value a line, got '{line}'"))
    for args, returncode, message in cases:
      with self.subTest(args=args):
        got, out = self.run_script(*args)
        self.assertEqual(got, returncode, out)
        self.assertIn(message, out)
        self.assertNotIn(EXISTING, out)
    # Accepted arguments reach docker, which echo answers as if containers existed.
    for args in (UP, (*UP, "--tag=my-tag", f"--xla_flags_file={good}")):
      with self.subTest(args=args):
        returncode, out = self.run_script(*args)
        self.assertEqual(returncode, 1, out)
        self.assertIn(EXISTING, out)

  def test_the_flags_reach_the_resource_manager_worker_and_proxy(self):
    """The flags file's flags end each server's arguments, after the scratch location and the instance type."""
    calls = os.path.join(self.tmp, "calls")
    docker = self.write("docker", FAKE_DOCKER.format(calls=shlex.quote(calls)), 0o755)
    os.mkdir(os.path.join(self.tmp, "bin"))
    self.write("bin/sudo", "#!/bin/sh\nexit 1\n", 0o755)  # sudo lsof /dev/vfio/*: nothing holds the TPU
    good = self.write("good.flags", FLAGS_FILE)
    for extra, flags in (((), []), ((f"--xla_flags_file={good}",), ["--xla_a=1", "--xla_b=true"])):
      with self.subTest(flags=flags):
        if os.path.exists(calls):
          os.remove(calls)
        returncode, out = self.run_script(
            *UP, "--tag=my-tag", *extra, docker=docker, bin_dir=os.path.join(self.tmp, "bin")
        )
        self.assertEqual(returncode, 0, out)
        self.assertIn(f"Pathways up (tpu7x:2x2x1, my-tag, {len(flags)} XLA flags)", out)
        with open(calls, encoding="utf-8") as handle:
          runs = [argv for argv in map(shlex.split, handle) if argv[0] == "run"]
        self.assertEqual(
            [argv[argv.index("--name") + 1] for argv in runs],
            [f"maxtext-pathways-{n}" for n in ("rm", "worker", "proxy")],
        )
        for argv, image in zip(runs, ("server", "server", "proxy_server")):
          server_args = argv[argv.index(f"{IMAGES}/{image}:my-tag") + 1 :]
          self.assertIn(GCS, server_args)
          self.assertEqual(server_args[len(server_args) - len(flags) :], flags)
          self.assertEqual(len([a for a in server_args if a.startswith("--xla_")]), len(flags))
        self.assertIn(INSTANCE, runs[0])


if __name__ == "__main__":
  unittest.main()
