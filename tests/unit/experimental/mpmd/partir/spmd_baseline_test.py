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

"""Tests for the equivalent MaxText SPMD baseline's settings and command, and that MaxText traces them (no cluster)."""

import contextlib
import dataclasses
import io
import os
import re
import shlex
import unittest

from absl import app
from absl import flags

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import runner
from maxtext.experimental.mpmd.partir import spmd_baseline
from maxtext.experimental.mpmd.partir import train_step

FLAGS = [
    "--device_type=v5p-32",
    "--pp=4",
    "--fsdp=4",
    "--num_microbatches=8",
    "--per_device_batch_size=8",
    "--model_name=llama3.1-8b",
    "--num_layers=32",
    "--max_target_length=2048",
    "--steps=5",
    "--warmup_steps=2",
    "--run_name=sb-run",
    "--output_dir=gs://bucket/out",
]
# The README's spmd_baseline example: its GKE llama3.1-8b run on two slices, two layers a stage.
README_EXAMPLE = (
    "--device_type=tpu7x-4x4x4",
    "--num_slices=2",
    "--pp=4",
    "--num_microbatches=4",
    "--per_device_batch_size=4",
    "--model_name=llama3.1-8b",
    "--max_target_length=1024",
    "--learning_rate=1e-5",
)
# One DeepSeek layer a pipeline stage, on a v5p-16.
ONE_DEEPSEEK_LAYER_A_STAGE = (
    "--device_type=v5p-16",
    "--pp=4",
    "--fsdp=2",
    "--num_microbatches=8",
    "--per_device_batch_size=8",
    "--model_name=deepseek3-tiny",
    "--num_layers=8",
    "--max_target_length=256",
)


def run(*extra):
  flag_values = flags.FlagValues()
  spmd_baseline.define_flags(flag_values)
  flag_values(["spmd_baseline", *FLAGS, *extra])
  out = io.StringIO()
  with contextlib.redirect_stdout(out):
    spmd_baseline.main(["spmd_baseline"], flag_values=flag_values)
  return out.getvalue()


def settings_of(out):
  return dict(
      line.strip().split("=", 1)
      for line in out.split("Run MaxText's trainer with:")[0].splitlines()
      if line.startswith("  ") and "=" in line
  )


class SpmdFlagsTest(unittest.TestCase):

  def test_same_tokens_per_step(self):
    """MPMD's batch is per stage, MaxText's per device of the whole slice: the baseline divides by the stages."""
    hparams = hp.Hparams(model_name="simple", num_layers=8, num_stages=4, num_microbatches=4, per_device_batch_size=8)
    settings = spmd_baseline.spmd_flags(hparams, total_devices=16)
    self.assertEqual(settings["per_device_batch_size"], "2.0")  # 8 x 4 devices per stage, over 16 devices
    self.assertEqual(settings["ici_pipeline_parallelism"], "4")
    self.assertEqual(settings["num_pipeline_microbatches"], "4")
    with self.assertRaisesRegex(ValueError, "multiple of the stage count"):
      spmd_baseline.spmd_flags(dataclasses.replace(hparams, num_microbatches=6), total_devices=16)

  def test_remat_policy(self):
    """MaxText refuses remat_policy=none with pipeline parallelism; the pipeline iterations checkpoint only with remat."""
    hparams = hp.Hparams(model_name="simple", num_layers=8, num_stages=4, num_microbatches=4, per_device_batch_size=8)
    pipelined = spmd_baseline.spmd_flags(hparams, total_devices=16)
    self.assertEqual((pipelined["remat_policy"], pipelined["set_remat_policy_on_pipeline_iterations"]), ("full", "False"))
    self.assertEqual(spmd_baseline.spmd_flags(hparams, total_devices=16, pure_fsdp=True)["remat_policy"], "none")
    remat = spmd_baseline.spmd_flags(dataclasses.replace(hparams, remat=True, remat_policy="minimal"), total_devices=16)
    self.assertEqual((remat["remat_policy"], remat["set_remat_policy_on_pipeline_iterations"]), ("minimal", "True"))


class MultiSliceTest(unittest.TestCase):
  """Over several slices MaxText's ici_* degrees describe one slice and its dcn_* degrees the slices.

  The baseline once put every stage on ICI: on two slices its ICI degrees multiplied to the devices of both, and
  MaxText's mesh check (create_device_mesh) stopped the printed command at start-up.
  """

  def mesh(self, hparams, device_type, num_slices, pure_fsdp=False):
    """{mesh axis: (ICI degree, DCN degree)} of the baseline on `num_slices` slices of `device_type`, axes above 1.

    pyconfig resolves the settings for that topology (compile_topology, as MaxText's AOT compilation takes it), and
    the degrees are filled in and checked as MaxText's create_device_mesh does.
    """
    devices = hp.devices_in_device_type(device_type)
    settings = spmd_baseline.spmd_flags(
        hparams, total_devices=devices * num_slices, num_slices=num_slices, pure_fsdp=pure_fsdp
    )
    cfg = pyconfig.initialize(
        [
            "spmd_baseline",
            os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
            *(f"{k}={v}" for k, v in settings.items()),
            f"compile_topology={device_type}",
            f"compile_topology_num_slices={num_slices}",
        ]
    )
    self.assertEqual(cfg.num_slices, num_slices)
    ici = max_utils.fill_unspecified_mesh_axes(list(cfg.ici_parallelism), devices, "ICI")
    dcn = max_utils.fill_unspecified_mesh_axes(list(cfg.dcn_parallelism), num_slices, "DCN")
    return {axis: (i, d) for axis, i, d in zip(cfg.mesh_axes, ici, dcn) if i * d > 1}

  def test_whole_stages_on_each_slice(self):
    """deepseek3-tiny on two v5p-16 slices: two stages a slice, the stage boundary between them over DCN."""
    tiny = hp.Hparams(
        model_name="deepseek3-tiny", num_layers=8, num_stages=4, fsdp=2, num_microbatches=8, per_device_batch_size=8
    )
    self.assertEqual(self.mesh(tiny, "v5p-16", 1), {"stage": (4, 1), "fsdp": (2, 1)})
    # 4 devices a stage: FSDP 2 x tensor 2, as the MPMD run's stage meshes.
    self.assertEqual(self.mesh(tiny, "v5p-16", 2), {"stage": (2, 2), "fsdp": (2, 1), "tensor": (2, 1)})
    expert = dataclasses.replace(tiny, fsdp=-1, expert=2)
    self.assertEqual(self.mesh(expert, "v5p-16", 2), {"stage": (2, 2), "fsdp": (2, 1), "expert": (2, 1)})
    # MaxText fills in the FSDP degree of each slice's stages.
    self.assertEqual(self.mesh(dataclasses.replace(tiny, fsdp=-1), "v5p-16", 2), {"stage": (2, 2), "fsdp": (4, 1)})
    # The README's example: 2 x tpu7x-4x4x4 (MaxText's compile_topology calls it tpu7x-128), two stages a slice.
    flag_values = flags.FlagValues()
    spmd_baseline.define_flags(flag_values)
    flag_values(["spmd_baseline", *README_EXAMPLE])
    self.assertEqual(self.mesh(hp.from_flags(flag_values), "tpu7x-128", 2), {"stage": (2, 2), "fsdp": (64, 1)})
    llama = hp.Hparams(model_name="llama3.1-8b", num_layers=8, num_stages=4, fsdp=4, num_microbatches=4)
    self.assertEqual(self.mesh(llama, "v5p-32", 2), {"stage": (2, 2), "fsdp": (4, 1), "tensor": (2, 1)})

  def test_whole_slices_in_each_stage(self):
    """Two stages on four slices: each stage holds two, with its outermost axis, fsdp, over DCN as in the MPMD run."""
    hparams = hp.Hparams(model_name="llama3.1-8b", num_layers=8, num_stages=2, num_microbatches=4)
    self.assertEqual(self.mesh(hparams, "v5p-16", 4), {"stage": (1, 2), "fsdp": (8, 2)})
    # 16 devices a stage at FSDP 8: tensor 2 within each slice, as in the MPMD run's stage mesh (fsdp outermost).
    self.assertEqual(
        self.mesh(dataclasses.replace(hparams, fsdp=8), "v5p-16", 4), {"stage": (1, 2), "fsdp": (4, 2), "tensor": (2, 1)}
    )
    with self.assertRaisesRegex(ValueError, "FSDP degree \\(1\\) must be a multiple of 2"):
      spmd_baseline.spmd_flags(dataclasses.replace(hparams, fsdp=1), total_devices=32, num_slices=4)

  def test_pure_fsdp_data_parallel_across_slices(self):
    hparams = hp.Hparams(model_name="llama3.1-8b", num_layers=8, num_stages=4, fsdp=4, num_microbatches=4)
    self.assertEqual(self.mesh(hparams, "v5p-32", 1, pure_fsdp=True), {"fsdp": (16, 1)})
    # FSDP 4 x 4 stages over the 32 devices, 2-way tensor parallel as the MPMD run's stages.
    self.assertEqual(self.mesh(hparams, "v5p-32", 2, pure_fsdp=True), {"data": (1, 2), "fsdp": (8, 1), "tensor": (2, 1)})
    self.assertEqual(
        self.mesh(dataclasses.replace(hparams, fsdp=-1), "v5p-32", 2, pure_fsdp=True), {"data": (1, 2), "fsdp": (16, 1)}
    )

  def test_refuses_stages_that_would_straddle_slices(self):
    hparams = hp.Hparams(model_name="simple", num_layers=8, num_stages=4, num_microbatches=4)
    with self.assertRaisesRegex(ValueError, "3 slices and 4 stages must divide one another"):
      spmd_baseline.spmd_flags(hparams, total_devices=24, num_slices=3)
    with self.assertRaisesRegex(ValueError, "must split evenly over num_slices"):
      spmd_baseline.spmd_flags(hparams, total_devices=16, num_slices=3)

  def test_command_on_two_slices(self):
    out = run("--num_slices=2", "--cluster=my-cluster", "--project=my-project", "--zone=us-east5-b", "--docker_image=img")
    self.assertIn("(32 devices in 2 slices)", out)
    settings = settings_of(out)
    self.assertEqual(
        {k: settings[k] for k in ("ici_pipeline_parallelism", "dcn_pipeline_parallelism", "per_device_batch_size")},
        {"ici_pipeline_parallelism": "2", "dcn_pipeline_parallelism": "2", "per_device_batch_size": "2.0"},
    )
    self.assertIn("--num-slices=2", out)


class DeepSeekBaselineTest(unittest.TestCase):

  HPARAMS = hp.Hparams(
      model_name="deepseek3-tiny",
      num_layers=8,
      num_stages=4,
      num_microbatches=4,
      per_device_batch_size=4,
      emb_dim=64,
      mlp_dim=128,
      vocab_size=256,
      max_target_length=32,
  )

  def test_pipeline_holds_only_moe_layers(self):
    """deepseek3-tiny's 3 dense layers run before MaxText's pipeline, which holds 4 of its 5 MoE layers."""
    settings = spmd_baseline.spmd_flags(self.HPARAMS, total_devices=16)
    self.assertEqual((settings["pipeline_parallel_layers"], settings["num_layers_per_pipeline_stage"]), ("4", "1"))
    self.assertIn(
        "4 of the 8 layers, all MoE; DeepSeek's 3 dense layers and the 1 MoE layer past a multiple of the 4 stages run"
        " before it",
        spmd_baseline.baseline_notes(self.HPARAMS)[0],
    )
    all_moe = dataclasses.replace(self.HPARAMS, first_num_dense_layers=0)
    self.assertEqual(spmd_baseline.spmd_flags(all_moe, total_devices=16)["pipeline_parallel_layers"], "8")
    # Every layer is in MaxText's pipeline, as in the MPMD run: nothing to note.
    self.assertEqual(spmd_baseline.baseline_notes(all_moe), [])
    uneven = dataclasses.replace(all_moe, num_layers=9, stage_layer_counts="2,2,2,3")
    self.assertIn(
        "8 of the 9 layers, all MoE; the 1 MoE layer past a multiple of the 4 stages runs before it",
        spmd_baseline.baseline_notes(uneven)[1],
    )
    with self.assertRaisesRegex(ValueError, "fewer than the 4 stages"):
      spmd_baseline.spmd_flags(
          dataclasses.replace(self.HPARAMS, num_layers=4, first_num_dense_layers=1), total_devices=16
      )

  def test_scan_layers(self):
    """Without --remat the layers outside MaxText's pipeline run unscanned, unless a stage holds one DeepSeek layer.

    Unscanned they are not rematerialised, as in the MPMD run; but with one layer per stage scan_layers=False makes
    MaxText's pipeline shift the layer's output tuple and fail, as the README's example did (see
    MaxTextTrainStepTest).
    """
    one_per_stage = spmd_baseline.spmd_flags(self.HPARAMS, total_devices=16)
    self.assertNotIn("scan_layers", one_per_stage)  # MaxText's default: scanned
    self.assertIn("MaxText still rematerialises those layers", spmd_baseline.baseline_notes(self.HPARAMS)[1])
    self.assertIn("--remat on both sides", spmd_baseline.baseline_notes(self.HPARAMS)[1])
    two_per_stage = dataclasses.replace(self.HPARAMS, num_stages=2, num_microbatches=2)
    settings = spmd_baseline.spmd_flags(two_per_stage, total_devices=16)
    self.assertEqual((settings["num_layers_per_pipeline_stage"], settings["scan_layers"]), ("2", "False"))
    notes = spmd_baseline.baseline_notes(two_per_stage)
    self.assertIn("those layers run unscanned (scan_layers=False)", notes[1])
    self.assertIn("--remat on both sides", notes[1])
    for remat in (dataclasses.replace(two_per_stage, remat=True), dataclasses.replace(self.HPARAMS, remat=True)):
      self.assertNotIn("scan_layers", spmd_baseline.spmd_flags(remat, total_devices=16))
      self.assertEqual(len(spmd_baseline.baseline_notes(remat)), 1)
    all_moe = dataclasses.replace(two_per_stage, first_num_dense_layers=0)  # nothing outside the pipeline
    self.assertNotIn("scan_layers", spmd_baseline.spmd_flags(all_moe, total_devices=16))
    self.assertEqual(spmd_baseline.spmd_flags(self.HPARAMS, total_devices=16, pure_fsdp=True)["scan_layers"], "True")

  def test_uneven_stage_layer_counts(self):
    """MaxText's pipeline splits the layers evenly: uneven counts are noted, and indivisible ones refused."""
    simple = hp.Hparams(model_name="simple", num_layers=6, num_stages=2, num_microbatches=2, stage_layer_counts="2,4")
    self.assertIn("--stage_layer_counts=2,4 has no MaxText equivalent", spmd_baseline.baseline_notes(simple)[0])
    self.assertEqual(spmd_baseline.spmd_flags(simple, total_devices=4)["num_layers_per_pipeline_stage"], "3")
    with self.assertRaisesRegex(ValueError, "layers to divide into the 2 stages"):
      spmd_baseline.spmd_flags(dataclasses.replace(simple, num_layers=5, stage_layer_counts="2,3"), total_devices=4)

  def test_same_model_as_the_mpmd_run(self):
    """MaxText builds the baseline with exactly the MPMD model's parameters (it once built 3 dense + 8 MoE layers)."""
    if len(jax.devices("cpu")) < 4:
      self.skipTest("needs 4 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    settings = spmd_baseline.spmd_flags(self.HPARAMS, total_devices=4)
    cfg = pyconfig.initialize(
        ["spmd_baseline", os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), *(f"{k}={v}" for k, v in settings.items())]
    )
    baseline = nnx.eval_shape(lambda: model_creation_utils.from_config(cfg, rngs=nnx.Rngs(0)))
    _, baseline_params, _ = nnx.split(baseline, nnx.Param, ...)
    mpmd = runner.build(self.HPARAMS, devices=jax.devices("cpu")[:4], mode=train_step.ExecutionMode.SPMD, abstract=True)

    def count(tree):
      return sum(int(leaf.size) for leaf in jax.tree.leaves(tree))

    self.assertEqual(count(baseline_params), count(mpmd.state.params))


class MaxTextTrainStepTest(unittest.TestCase):
  """MaxText's own training step traces with the settings the baseline prints, on the CPU devices.

  It once did not: with one DeepSeek layer per pipeline stage, ``scan_layers=False`` made MaxText's pipeline shift
  the layer's output tuple (``TypeError: zeros_like requires ndarray or scalar arguments, got <class 'tuple'>``).
  """

  def hparams_of(self, argv, **changes):
    flag_values = flags.FlagValues()
    spmd_baseline.define_flags(flag_values)
    flag_values(["spmd_baseline", *argv])
    return dataclasses.replace(hp.from_flags(flag_values), **changes)

  def trace_train_step(self, hparams):
    """Traces MaxText's `train_step` with the baseline of `hparams` on every CPU device, its FSDP filling each stage.

    As MaxText's ahead-of-time compilation (`train_compile`) lowers it, from abstract state and data; returns the
    settings.
    """
    devices = len(jax.devices())
    if devices % hparams.num_stages:
      self.skipTest(f"needs a multiple of {hparams.num_stages} CPU devices, has {devices}")
    hparams = dataclasses.replace(hparams, fsdp=devices // hparams.num_stages)
    settings = spmd_baseline.spmd_flags(hparams, total_devices=devices)
    cfg = pyconfig.initialize(
        ["spmd_baseline", os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), *(f"{k}={v}" for k, v in settings.items())]
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    shaped_args, shaped_kwargs, state_mesh_shardings, _, model = train_compile.get_shaped_inputs(mesh, cfg)
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(cfg, state_mesh_shardings)
    step, in_shardings, out_shardings, static_argnums, donate_argnums = maxtext_utils.get_functional_train_with_signature(
        train.train_step,
        sharding.get_input_data_sharding(cfg, mesh),
        sharding.build_zero1_input_state_mesh_shardings(cfg, state_mesh_shardings, params_shardings),
        model,
        cfg,
        params_shardings,
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      jax.jit(
          step,
          in_shardings=in_shardings,
          out_shardings=out_shardings,
          static_argnums=static_argnums,
          donate_argnums=donate_argnums,
      ).trace(*shaped_args, **shaped_kwargs)
    return settings

  def test_readme_example(self):
    """The README's example on one slice of the CPU devices, its FSDP scaled to them: 2 layers a stage."""
    with open(os.path.join(os.path.dirname(spmd_baseline.__file__), "README.md"), encoding="utf-8") as readme:
      self.assertIn("spmd_baseline " + " ".join(README_EXAMPLE), re.sub(r"\s*\\\n\s*", " ", readme.read()))
    settings = self.trace_train_step(self.hparams_of(README_EXAMPLE))
    self.assertEqual(settings["num_layers_per_pipeline_stage"], "2")

  def test_one_deepseek_layer_a_stage(self):
    """deepseek3-tiny at PP=4 x FSDP=2 on a v5p-16, its FSDP scaled to the CPU devices: 1 layer a stage."""
    settings = self.trace_train_step(self.hparams_of(ONE_DEEPSEEK_LAYER_A_STAGE))
    self.assertEqual(settings["num_layers_per_pipeline_stage"], "1")

  def test_unscanned_layers_outside_the_pipeline(self):
    """Two layers a stage: the dense and remaining MoE layers run unscanned before MaxText's pipeline."""
    settings = self.trace_train_step(self.hparams_of(ONE_DEEPSEEK_LAYER_A_STAGE, num_stages=2))
    self.assertEqual((settings["num_layers_per_pipeline_stage"], settings["scan_layers"]), ("2", "False"))


class LaunchCommandTest(unittest.TestCase):

  def test_trainer_command(self):
    out = run()
    command = out.split("Run MaxText's trainer with:", 1)[1].strip().splitlines()[0]
    argv = shlex.split(command)
    self.assertEqual(argv[:4], ["python3", "-u", "-m", "maxtext.trainers.pre_train.train"])
    self.assertIn("run_name=sb-run", argv)
    self.assertIn("base_output_directory=gs://bucket/out", argv)
    self.assertNotIn("On the cluster", out)

  def test_profiler_after_the_timed_steps(self):
    settings = settings_of(run())
    self.assertEqual(settings["steps"], "12")  # 2 warm-up + 5 timed + 5 profiled
    self.assertEqual(settings["skip_first_n_steps_for_profiler"], "7")
    self.assertEqual(settings["profiler_steps"], "5")
    # one TPU, one SparseCore and one SparseCore tile, as the MPMD side traces
    self.assertEqual(
        {
            k: settings[k]
            for k in (
                "enable_tpu_profiling_options",
                "tpu_num_chips_to_profile_per_task",
                "tpu_num_sparse_cores_to_trace",
                "tpu_num_sparse_core_tiles_to_trace",
            )
        },
        {
            "enable_tpu_profiling_options": "True",
            "tpu_num_chips_to_profile_per_task": "1",
            "tpu_num_sparse_cores_to_trace": "1",
            "tpu_num_sparse_core_tiles_to_trace": "1",
        },
    )

  def test_gcluster_command(self):
    out = run("--cluster=my-cluster", "--project=my-project", "--zone=us-east5-b", "--docker_image=example.com/mt")
    lines = out.split("On the cluster:", 1)[1].strip().splitlines()
    self.assertEqual(
        lines[:3],
        [
            "gcluster job config set project my-project",
            "gcluster job config set cluster my-cluster",
            "gcluster job config set location us-east5-b",
        ],
    )
    argv = shlex.split("\n".join(lines[3:]).replace("\\\n", " "))
    self.assertEqual(argv[:3], ["gcluster", "job", "submit"])
    options = dict(arg[2:].split("=", 1) for arg in argv[3:])
    self.assertEqual(
        {k: options[k] for k in ("name", "image", "compute-type", "topology", "num-slices")},
        {
            "name": "sb-run",
            "image": "example.com/mt",
            "compute-type": "ct5p-hightpu-4t",
            "topology": "2x2x4",
            "num-slices": "1",
        },
    )
    self.assertTrue(options["command"].startswith("python3 -u -m maxtext.trainers.pre_train.train "))

  def test_default_run_names_differ(self):
    """A fixed default made every repeated run (and its workload) collide; the default carries a timestamp."""
    flag_values = flags.FlagValues()
    spmd_baseline.define_flags(flag_values)
    flag_values(["spmd_baseline", *[f for f in FLAGS if not f.startswith("--run_name")]])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      spmd_baseline.main(["spmd_baseline"], flag_values=flag_values)
    self.assertRegex(out.getvalue(), r"run_name=spmd-baseline-\d{10}\b")

  def test_xpk_command(self):
    out = run(
        "--cluster=my-cluster",
        "--project=my-project",
        "--zone=us-east5-b",
        "--docker_image=example.com/mt",
        "--launcher=xpk",
    )
    argv = shlex.split(out.split("On the cluster:", 1)[1].replace("\\\n", " "))
    self.assertEqual(argv[:3], ["xpk", "workload", "create"])
    pairs = dict(zip(argv[3::2], argv[4::2]))
    self.assertEqual(
        {k: pairs[k] for k in ("--cluster", "--project", "--zone", "--tpu-type", "--workload", "--docker-image")},
        {
            "--cluster": "my-cluster",
            "--project": "my-project",
            "--zone": "us-east5-b",
            "--tpu-type": "v5p-32",
            "--workload": "sb-run",
            "--docker-image": "example.com/mt",
        },
    )
    self.assertTrue(pairs["--command"].startswith("python3 -u -m maxtext.trainers.pre_train.train "))
    with self.assertRaisesRegex(app.UsageError, "--cluster needs --project, --zone, --docker_image"):
      run("--cluster=my-cluster")

  def test_refusals_are_usage_errors_before_any_output(self):
    out = io.StringIO()
    for extra, error in (
        (("--num_slices=3", "--total_devices=48"), "3 slices and 4 stages must divide one another"),
        (("--model_name=no-such-model",), "MaxText model config .*no-such-model.yml' is missing"),
    ):
      flag_values = flags.FlagValues()
      spmd_baseline.define_flags(flag_values)
      flag_values(["spmd_baseline", *FLAGS, *extra])
      with self.subTest(extra=extra), self.assertRaisesRegex(app.UsageError, error), contextlib.redirect_stdout(out):
        spmd_baseline.main(["spmd_baseline"], flag_values=flag_values)
    self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
  unittest.main()
