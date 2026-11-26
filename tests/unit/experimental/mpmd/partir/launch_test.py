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

"""Tests for the Pathways launcher's command lines (no JAX, no cluster)."""

import contextlib
import io
import os
import shlex
import tempfile
import unittest

from absl import app
from absl import flags

from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import launch

REQUIRED = [
    "--cluster=my-cluster",
    "--project=my-project",
    "--zone=us-east5-b",
    "--docker_image=example.com/maxtext:latest",
    "--pathways_gcs_location=gs://bucket/pathways",
    "--device_type=v5p-16",
    "--workload=mpmd-pp4",
]
MODEL = ["--pp=4", "--fsdp=2", "--num_microbatches=8", "--per_device_batch_size=8", "--model_name=deepseek3-tiny"]


def parse(*argv):
  flag_values = flags.FlagValues()
  launch.define_flags(flag_values)
  flag_values(["launch", *argv])
  return flag_values


def printed(*argv, extra=()):
  """The commands main prints for `argv` (flags) and `extra` (the arguments after --)."""
  out = io.StringIO()
  with contextlib.redirect_stdout(out):
    launch.main(["launch", *extra], flag_values=parse(*argv))
  return [shlex.split(line) for line in out.getvalue().splitlines()]


def command_of(argv):
  """The entry point's command inside an xpk or gcluster argv."""
  (command,) = [a.partition("=")[2] for a in argv if a.startswith("--command=")]
  return shlex.split(command)


class ModuleArgsTest(unittest.TestCase):

  def test_only_the_trainer_gets_its_flags(self):
    hparams = hp.Hparams(maxtext_overrides=("dtype=float32",))
    train = launch.module_args(hparams, parse(*REQUIRED))
    self.assertIn("--noenable_profile", train)
    self.assertIn("--run_name=mpmd-pp4", train)
    self.assertIn("--maxtext_overrides=dtype=float32", train)
    parity = launch.module_args(hparams, parse(*REQUIRED, "--module=parity"), ["--loss_tolerance=1e-3"])
    self.assertFalse([a for a in parity if a.startswith(("--enable_profile", "--profile_", "--run_name"))])
    self.assertEqual(parity[-1], "--loss_tolerance=1e-3")
    self.assertIn("--compile_topology=local", launch.module_args(hparams, parse(*REQUIRED, "--module=train_compile")))
    # What the launcher sends parses back into the same hyperparameters.
    flag_values = flags.FlagValues()
    hp.define_flags(flag_values)
    flag_values(["parity", *[a for a in parity if not a.startswith("--loss_tolerance")]])
    self.assertEqual(hp.from_flags(flag_values), hparams)

  def test_extra_arguments_must_be_entry_point_flags(self):
    flag_values = parse(*REQUIRED, "--module=parity")
    with self.assertRaisesRegex(app.UsageError, "must be --flags"):
      launch.module_args(hp.Hparams(), flag_values, ["stray"])
    # Hyperparameter and launcher flags after -- would bypass main()'s validation.
    for item in (
        "--num_microbatches=3",
        "--pp=2",
        "--noremat",
        "--pipeline_schedule=Circular",
        "--maxtext_overrides=dtype=float32",
        "--enable_profile",
        "--cluster=x",
    ):
      with self.assertRaisesRegex(app.UsageError, "before --", msg=item):
        launch.module_args(hp.Hparams(), flag_values, [item])
    self.assertEqual(
        launch.module_args(hp.Hparams(), flag_values, ["--noise_floor_factor=3"])[-1], "--noise_floor_factor=3"
    )


class ChecksTest(unittest.TestCase):

  def test_slices_and_stages_divide(self):
    hparams = hp.Hparams(num_stages=4, num_microbatches=8, per_device_batch_size=8)
    launch.check_slices(hparams, "v5p-16", 2)  # 2 stages per slice
    launch.check_slices(hp.Hparams(num_stages=2, num_microbatches=8, per_device_batch_size=8), "v5p-16", 4)
    with self.assertRaisesRegex(app.UsageError, "divide one another"):
      launch.check_slices(hparams, "v5p-16", 3)  # 24 devices split in 4, but not by slice
    with self.assertRaisesRegex(app.UsageError, "does not split into 4 stages"):
      launch.check_slices(hparams, "v5p-unknown", 1)

  def test_microbatches_split_over_the_stage(self):
    # v5p-16 at PP=4: 2 devices a stage; 8 rows in 8 microbatches is 1 row a microbatch, over 2 devices.
    with self.assertRaisesRegex(app.UsageError, "must be a multiple of the 2 devices"):
      launch.check_slices(hp.Hparams(num_stages=4, num_microbatches=8, per_device_batch_size=4), "v5p-16", 1)

  def test_workload_names(self):
    launch.check_workload_name("mpmd-pp4")
    launch.check_workload_name("a" * launch.MAX_PATHWAYS_NAME)
    launch.check_workload_name("a")
    # A leading digit passed, but xpk and JobSet's webhook want a DNS-1035 label, which starts with a letter.
    for bad in ("a" * (launch.MAX_PATHWAYS_NAME + 1), "Upper", "under_score", "-dash", "dash-", "1mpmd", "4"):
      with self.assertRaisesRegex(app.UsageError, "at most 22 characters", msg=bad):
        launch.check_workload_name(bad)


class CommandTest(unittest.TestCase):

  def test_xpk_create_pathways(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "x.flags")
      with open(path, "w", encoding="utf-8") as f:
        f.write("# comment\n--xla_a=true\n--xla_b=1\n")
      (argv,) = printed(
          *REQUIRED,
          *MODEL,
          "--launcher=xpk",
          f"--xla_flags_file={path}",
          "--extra_xla_flags=xla_b=2",
          "--launcher_args=--priority=low",
      )
    self.assertEqual(argv[:3], ["xpk", "workload", "create-pathways"])
    for want in (
        "--cluster=my-cluster",
        "--project=my-project",
        "--zone=us-east5-b",
        "--workload=mpmd-pp4",
        "--tpu-type=v5p-16",
        "--num-slices=1",
        "--docker-image=example.com/maxtext:latest",
        f"--server-image={launch.PATHWAYS_IMAGES}/server:{launch.DEFAULT_PATHWAYS_TAG}",
        f"--proxy-server-image={launch.PATHWAYS_IMAGES}/proxy_server:{launch.DEFAULT_PATHWAYS_TAG}",
        "--pathways-gcs-location=gs://bucket/pathways",
        "--priority=low",
    ):
      self.assertIn(want, argv)
    # The XLA flags reach the proxy server, which compiles, as well as the resource manager and the workers.
    for component in ("proxy-server", "server", "worker"):
      self.assertIn(f"--custom-pathways-{component}-args=--xla_a=true --xla_b=2", argv)
    command = command_of(argv)
    self.assertEqual(command[:4], ["python3", "-u", "-m", "maxtext.experimental.mpmd.partir.train"])
    self.assertIn("--num_stages=4", command)

  def test_gcluster_is_the_default(self):
    """Cluster Toolkit, which MaxText uses in place of the deprecated XPK; the slice is derived from --device_type."""
    commands = printed(*REQUIRED, *MODEL, "--extra_xla_flags=xla_a=1")
    self.assertEqual(
        commands[:3],
        [
            ["gcluster", "job", "config", "set", "project", "my-project"],
            ["gcluster", "job", "config", "set", "cluster", "my-cluster"],
            ["gcluster", "job", "config", "set", "location", "us-east5-b"],
        ],
    )
    submit = commands[3]
    self.assertEqual(submit[:4], ["gcluster", "job", "submit", "--pathways"])
    for want in (
        "--name=mpmd-pp4",
        "--image=example.com/maxtext:latest",
        "--compute-type=ct5p-hightpu-4t",
        "--topology=2x2x2",
        "--num-slices=1",
        "--pathways-gcs-location=gs://bucket/pathways",
        f"--pathways-server-image={launch.PATHWAYS_IMAGES}/server:{launch.DEFAULT_PATHWAYS_TAG}",
        f"--pathways-proxy-server-image={launch.PATHWAYS_IMAGES}/proxy_server:{launch.DEFAULT_PATHWAYS_TAG}",
        "--pathways-proxy-args=--xla_a=1",
    ):
      self.assertIn(want, submit)
    self.assertEqual(command_of(submit)[3], "maxtext.experimental.mpmd.partir.train")

  def test_gcluster_warns_that_it_reruns_a_finished_job(self):
    """gcluster v1.108.0 and v1.109.0 restart a Pathways JobSet whose entry point ended (cluster-toolkit PR #6739)."""
    with self.assertLogs("absl", level="WARNING") as logs:
      printed(*REQUIRED, *MODEL)
    (warning,) = [line for line in logs.output if "PR #6739" in line]
    self.assertIn("gcluster v1.108.0 and v1.109.0", warning)
    self.assertIn("--launcher=xpk", warning)
    with self.assertNoLogs("absl", level="WARNING"):
      printed(*REQUIRED, *MODEL, "--launcher=xpk")

  def test_gcluster_slices(self):
    self.assertEqual(launch.gcluster_slice("v5p-16"), ("ct5p-hightpu-4t", "2x2x2"))
    self.assertEqual(launch.gcluster_slice("v6e-16"), ("ct6e-standard-4t", "4x4"))
    self.assertEqual(launch.gcluster_slice("tpu7x-128"), ("tpu7x-standard-4t", "4x4x4"))
    self.assertEqual(launch.gcluster_slice("tpu7x-4x4x4"), ("tpu7x-standard-4t", "4x4x4"))
    self.assertEqual(launch.gcluster_slice("v5litepod-8"), ("ct5lp-hightpu-4t", "2x4"))
    self.assertEqual(launch.gcluster_slice("v6e-2x2"), ("ct6e-standard-4t", "2x2"))
    for device_type, error in (
        ("v5p-12345", "MaxText knows no v5p-12345 slice"),
        ("gpu-8", r"'gpu' is not one of the known TPU generations \(tpu7x, v6e, v5p, v5e, v5litepod, v4\)"),
        # The compute types are 4-chip hosts: a 1-chip slice needs another (e.g. ct6e-standard-1t).
        ("v6e-1", "a 1x1 slice has 1-chip hosts, and ct6e-standard-4t is a host of 4 chips"),
        ("tpu7x-2", "a 1x1x1 slice has 1-chip hosts"),
        ("v5litepod-1", "a 1x1 slice has 1-chip hosts"),
        ("tpu7x-1x1x1", "a 1x1x1 slice has 1-chip hosts"),
        ("tpu7x-4x4", r"a tpu7x topology has 3 dimensions \(AxBxC\), not '4x4'"),
        ("v6e-4x4x4", r"a v6e topology has 2 dimensions \(AxB\), not '4x4x4'"),
        # The right number of dimensions, one of them not a number of chips.
        ("v5p-2x2x0", r"^a topology's dimensions are positive integers, not '0' \(in '2x2x0'\)$"),
        ("tpu7x-4x4x", r"^a topology's dimensions are positive integers, not '' \(in '4x4x'\)$"),
        ("v6e-2xa", r"^a topology's dimensions are positive integers, not 'a' \(in '2xa'\)$"),
    ):
      with self.subTest(device_type=device_type), self.assertRaisesRegex(ValueError, error):
        launch.gcluster_slice(device_type)

  def test_gcluster_topologies(self):
    """The slice's chip box, and the chips on each of its hosts: all of them for a slice smaller than a host."""
    self.assertEqual(launch.gcluster_topology("v5p-16"), ("2x2x2", 4))
    self.assertEqual(launch.gcluster_topology("tpu7x-4x4x4"), ("4x4x4", 4))
    self.assertEqual(launch.gcluster_topology("tpu7x-2"), ("1x1x1", 1))
    self.assertEqual(launch.gcluster_topology("v6e-2x2"), ("2x2", 4))
    self.assertEqual(launch.gcluster_topology("v6e-1x1"), ("1x1", 1))
    self.assertEqual(launch.gcluster_topology("v5litepod-8"), ("2x4", 4))
    for device_type, error in (
        ("v5p-12345", "MaxText knows no v5p-12345 slice"),
        ("gpu-2x2", "'gpu' is not one of the known TPU generations"),
        ("tpu7x-4x4", r"a tpu7x topology has 3 dimensions \(AxBxC\), not '4x4'"),
        ("v5p-2x0x2", r"a topology's dimensions are positive integers, not '0' \(in '2x0x2'\)"),
    ):
      with self.subTest(device_type=device_type), self.assertRaisesRegex(ValueError, error):
        launch.gcluster_topology(device_type)

  def test_gcluster_derives_only_what_no_flag_gives(self):
    """An explicit compute type or topology is never blamed: an error names only the flags still needed."""
    # tpu7x-2 is one chip, so its 1-chip hosts fit no 4-chip compute type; 2 devices hold PP=2.
    small = ["--pp=2", "--fsdp=1", "--num_microbatches=4", "--per_device_batch_size=4", "--model_name=deepseek3-tiny"]

    def slice_flags(*argv):
      submit = printed(*REQUIRED, *small, *argv)[-1]
      return [a for a in submit if a.startswith(("--compute-type=", "--topology="))]

    self.assertEqual(
        slice_flags("--device_type=tpu7x-2", "--gcluster_compute_type=tpu7x-standard-1t"),
        ["--compute-type=tpu7x-standard-1t", "--topology=1x1x1"],
    )
    self.assertEqual(
        slice_flags("--device_type=v5p-16", "--gcluster_topology=4x2x1"),
        ["--compute-type=ct5p-hightpu-4t", "--topology=4x2x1"],
    )
    hosts = r"--device_type=tpu7x-2: a 1x1x1 slice has 1-chip hosts, and tpu7x-standard-4t is a host of 4 chips"
    unknown = r"--device_type=v5p-16384: MaxText knows no v5p-16384 slice \(maxtext.utils.accelerator_to_spec_map\)"
    for argv, error in (
        # The slice is known: only the compute type is in question, with or without an explicit topology.
        (["--device_type=tpu7x-2"], rf"^{hosts}; pass --gcluster_compute_type\.$"),
        (["--device_type=tpu7x-2", "--gcluster_topology=1x1x1"], rf"^{hosts}; pass --gcluster_compute_type\.$"),
        # No value is derived from a slice that is not known: the error names the flags not given.
        (["--device_type=v5p-16384"], rf"^{unknown}; pass --gcluster_compute_type and --gcluster_topology\.$"),
        (
            ["--device_type=v5p-16384", "--gcluster_compute_type=ct5p-hightpu-4t"],
            rf"^{unknown}; pass --gcluster_topology\.$",
        ),
        (["--device_type=v5p-16384", "--gcluster_topology=16x16x32"], rf"^{unknown}; pass --gcluster_compute_type\.$"),
    ):
      with self.subTest(argv=argv), self.assertRaisesRegex(app.UsageError, error):
        printed(*REQUIRED, *small, *argv)

  def test_gcluster_explicit_slice(self):
    commands = printed(
        *REQUIRED, *MODEL, "--launcher=gcluster", "--gcluster_compute_type=ct5p-hightpu-4t", "--gcluster_topology=2x2x2"
    )
    self.assertEqual(
        commands[:3],
        [
            ["gcluster", "job", "config", "set", "project", "my-project"],
            ["gcluster", "job", "config", "set", "cluster", "my-cluster"],
            ["gcluster", "job", "config", "set", "location", "us-east5-b"],
        ],
    )
    submit = commands[3]
    self.assertEqual(submit[:4], ["gcluster", "job", "submit", "--pathways"])
    for want in ("--name=mpmd-pp4", "--compute-type=ct5p-hightpu-4t", "--topology=2x2x2"):
      self.assertIn(want, submit)
    self.assertFalse([a for a in submit if a.startswith("--pathways-proxy-args")])  # no XLA flags given
    self.assertEqual(command_of(submit)[3], "maxtext.experimental.mpmd.partir.train")

  def test_refuses_what_would_fail_on_the_cluster(self):
    with self.assertRaisesRegex(app.UsageError, "Missing required flags: --cluster, --project"):
      printed(*REQUIRED[2:], *MODEL)
    with self.assertRaisesRegex(app.UsageError, "pass --gcluster_compute_type and --gcluster_topology"):
      printed(*REQUIRED, *MODEL, "--device_type=v5p-16384")
    with self.assertRaisesRegex(app.UsageError, "--device_type=tpu7x-4x4: a tpu7x topology has 3 dimensions"):
      printed(*REQUIRED, *MODEL, "--device_type=tpu7x-4x4")
    with self.assertRaisesRegex(app.UsageError, "gs://"):
      printed(*REQUIRED, *MODEL, "--pathways_gcs_location=/tmp/x")
    with self.assertRaisesRegex(app.UsageError, "gs:// --profile_dir"):
      printed(*REQUIRED, *MODEL, "--enable_profile")
    # The hyperparameters' refusals are usage errors, not tracebacks.
    with self.assertRaisesRegex(app.UsageError, "decoder_block=gpt3"):
      printed(*REQUIRED, "--pp=4", "--model_name=gpt3-52k")
    with self.assertRaisesRegex(app.UsageError, "MaxText model config .*no-such-model.yml' is missing"):
      printed(*REQUIRED, *MODEL, "--model_name=no-such-model")
    # Once accepted, it failed only when the trainer traced its first step on the cluster.
    with self.assertRaisesRegex(app.UsageError, "remat_policy=qkv_proj_offloaded is not one"):
      printed(*REQUIRED, *MODEL, "--remat", "--remat_policy=qkv_proj_offloaded")


if __name__ == "__main__":
  unittest.main()
