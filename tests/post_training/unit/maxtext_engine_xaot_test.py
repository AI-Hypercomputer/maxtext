# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ahead-of-time compilation of the engine compiles the same thing training runs.

`training_engine/maxtext_engine_compile.py` reports a configuration's cost and peak memory from
shapes alone, with no weights and no hardware. That report is worth only the claim behind it --
that the kernels it compiled are the kernels training will run -- and `AbstractMaxTextEngine`
reaches them differently: it traces its optimizer moments rather than allocating them, and
derives their shardings by propagation rather than off real arrays.

So these tests compare the optimized HLO of all three kernels against a live engine that has
actually stepped. `test_the_comparison_can_fail` guards the vacuous case.

The parity tests run in a subprocess. Their shardings need more than one device, and the CPU
backend reads `--xla_force_host_platform_device_count` only at backend initialization -- which is
already past by the time pytest imports this file, because sibling modules under
`tests/post_training` touch JAX while being collected. Setting the flag at import time here is
therefore a no-op that leaves `jax.device_count() == 1` and skips the parity classes green.
`test_engine_aot_parity_on_a_four_device_cpu_mesh` re-execs the module with the flag appended
instead, and refuses to pass unless the child reports tests actually run.
"""

# Every `_engine._private` below reads a member of the class under test from its own test.
# pylint: disable=protected-access

import os
import re
import subprocess
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import maxtext_engine_compile
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_XAOT_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")


@pytest.mark.cpu_only
def test_engine_aot_parity_on_a_four_device_cpu_mesh():
  """Runs the two parity classes below in a child process with four CPU devices.

  See the module docstring for why re-exec is the only option. The flag is *appended* rather
  than defaulted so it wins over whatever a sibling module set -- XLA takes the last occurrence
  of a repeated flag.
  """
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  # The child imports `tests.utils.test_helpers`, which pytest puts on the path for us and a
  # bare interpreter does not.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # An exit status of 0 is also what a run that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


def _config(**overrides) -> pyconfig.HyperParameters:
  """A tiny real decoder, big enough that every sharding decision is visible in the HLO.

  `adamw` rather than `sgd`: its moments are the part of the train state the abstract path has
  to invent, and so the part most likely to be wrong.
  """
  argv = [
      "maxtext_engine_xaot_test.py",
      get_test_config_path("base.yml"),
      "model_name=default",
      "run_name=engine_xaot_test",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "enable_dropout=False",
      "init_weights_seed=0",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "remat_policy=none",
      "scan_layers=False",
      "attention=dot_product",
      "shard_mode=explicit",
      f"ici_data_parallelism={_REQUIRED_DEVICES}",
      "ici_fsdp_parallelism=1",
      "ici_tensor_parallelism=1",
      "per_device_batch_size=1",
      "vocab_size=128",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "head_dim=16",
      "max_target_length=32",
      "opt_type=adamw",
      "learning_rate=1e-2",
      "gradient_clipping_threshold=0.0",
      "warmup_steps_fraction=0.0",
      "learning_rate_final_fraction=1.0",
      "gradient_accumulation_steps=1",
  ]
  argv.extend(f"{key}={value}" for key, value in overrides.items())
  return pyconfig.initialize(argv)


def _config_and_mesh(**overrides) -> tuple[pyconfig.HyperParameters, jax.sharding.Mesh]:
  cfg = _config(**overrides)
  return cfg, maxtext_utils.get_mesh_from_config(cfg)


def _batch(cfg: pyconfig.HyperParameters, seed: int) -> dict[str, np.ndarray]:
  """One micro-batch shaped for `maxtext.trainers.pre_train.train.loss_fn`.

  NumPy rather than `jnp`, as a real driver's batch is: a committed array built here would
  arrive replicated and be rejected by `jax.jit`'s exact `in_shardings` match.
  """
  batch, seq = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length
  rng = np.random.default_rng(seed)
  tokens = rng.integers(1, cfg.vocab_size, size=(batch, seq)).astype(np.int32)
  positions = np.tile(np.arange(seq, dtype=np.int32), (batch, 1))
  segmentation = np.ones((batch, seq), dtype=np.int32)
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1),
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets_segmentation": segmentation,
  }


def _abstract(batch: dict[str, np.ndarray]) -> dict[str, jax.ShapeDtypeStruct]:
  """The same batch with the data taken out -- all an AOT compile is given."""
  return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), batch)


def _script_argv(*overrides: str) -> tuple[str, ...]:
  """`main`'s argv for a model small enough to compile inside a test."""
  return (
      "",
      get_test_config_path("base.yml"),
      "compile_topology=v6e-4",
      "compile_topology_num_slices=1",
      "per_device_batch_size=1",
      "max_target_length=128",
      "base_emb_dim=128",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "enable_checkpointing=false",
  ) + overrides


class _LiveKernels:
  """Recompiles the kernels a running engine actually dispatched.

  An executable keeps no handle on the arguments it was compiled from, so reading a kernel's HLO
  means compiling it a second time -- through the engine's own jitted wrappers, from the avals of
  the arguments it was dispatched with. Avals rather than the arrays, because `_compiled_update`
  donates its state. That recompilation is only sound if it reproduces the original, which
  `test_recompiling_reproduces_the_kernels_training_ran` checks and the rest of the file rests on.
  """

  _ATTRIBUTES = {
      "fwd_bwd": "_compiled_fwd_bwd",
      "fwd_bwd_accum": "_compiled_fwd_bwd_accum",
      "update": "_compiled_update",
  }

  def __init__(self, engine: maxtext_engine.MaxTextTrainingEngine):
    self._engine = engine
    self._dispatched = {}
    self.calls: dict[str, list] = {}
    for name, attribute in self._ATTRIBUTES.items():
      self._dispatched[name] = getattr(engine, attribute)
      setattr(engine, attribute, self._spy(name))

  def _spy(self, name: str):
    """Returns a stand-in for one kernel's dispatch handle that records its arguments and forwards."""

    def call(*args):
      self.calls.setdefault(name, []).append(
          # The sharding as well as the shape: `jax.jit` keys its cache on both.
          jax.tree.map(
              lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
              args,
          )
      )
      return self._dispatched[name](*args)

    return call

  def compiled(self) -> dict[str, jax.stages.Compiled]:
    """The executable behind each kernel's first dispatch.

    `jax.jit` exposes no way to compile without lowering first, so this goes through
    `Lowered.compile()` -- the same two steps the engine's `compile_kernels` takes.
    """
    missing = sorted(set(self._ATTRIBUTES) - set(self.calls))
    if missing:
      raise AssertionError(f"these kernels were never dispatched, so they have no HLO to read: {missing}")
    # The mesh and the logical axis rules have to be live while tracing, exactly as they were at
    # dispatch: without the rules every `maybe_shard_with_logical` is a no-op.
    with self._engine._sharding_ctx():
      return {name: self._engine._jitted_kernels[name].lower(*calls[0]).compile() for name, calls in self.calls.items()}


# Optimized HLO carries a source-location index -- `<id> "<name>"` header lines, plus the
# `metadata={...}` and `stack_frame_id=N` referring into it -- which names where the Python was,
# not what the program does, and differs between two lowerings of the same kernel. Same
# normalization as `tests/integration/aot_identical_test.py` and `hlo_diff_test.py`.
_SOURCE_ID_LINE = re.compile(r'^\s*\d+\s+(?:"[^"]*"|\{[^}]*\})\s*$')
_METADATA = re.compile(r"metadata=\{[^}]*\}")
_STACK_FRAME = re.compile(r"stack_frame_id=\d+")


def _normalize(hlo: str) -> str:
  """Strips source-location bookkeeping from optimized HLO, leaving the program."""
  lines = []
  for line in hlo.splitlines():
    if _SOURCE_ID_LINE.match(line):
      continue
    line = _METADATA.sub("metadata={}", line)
    lines.append(_STACK_FRAME.sub("stack_frame_id=0", line))
  return "\n".join(lines)


class EngineAotParityTest(parameterized.TestCase):
  """A live engine and an abstract one compile to the same bytes."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def _live(self, cfg, mesh, micro_batches: int = 2):
    """Runs one full optimizer step and returns the engine plus its dispatched kernels.

    Two micro-batches: the accumulating kernel is traced lazily, so a single-micro-batch run
    would leave `fwd_bwd_accum` with nothing on the live side to compare against.
    """
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    engine.compile(_batch(cfg, 0))
    kernels = _LiveKernels(engine)
    for micro in range(micro_batches):
      engine.fwd_bwd(_batch(cfg, micro))
    engine.update()
    return engine, kernels

  @parameterized.named_parameters(
      # The gradient all-reduce is deferred to the update, so the reduced/unreduced tags have to
      # survive onto the abstract gradients too.
      ("data_parallel", {}),
      # Zero-1 shards the moments by moving real arrays on the live path and by restating avals
      # on the abstract one.
      ("zero1", {"shard_optimizer_over_data": "True"}),
      ("fsdp", {"ici_data_parallelism": 1, "ici_fsdp_parallelism": _REQUIRED_DEVICES}),
      # The hard case for the abstract path: JAX propagates a layout through `zeros_like` only on
      # Explicit axes, so the moments' shardings have to be observed on a stand-in mesh.
      ("auto_shard_mode", {"shard_mode": "auto", "ici_data_parallelism": 1, "ici_fsdp_parallelism": _REQUIRED_DEVICES}),
      # Gradients narrower than the parameters, which the AOT path must read off the
      # forward/backward kernel's outputs rather than assume.
      ("bfloat16_grads", {"grad_dtype": "bfloat16"}),
  )
  def test_aot_compiles_the_same_hlo_the_trainer_runs(self, overrides):
    cfg, mesh = _config_and_mesh(**overrides)

    _, kernels = self._live(cfg, mesh)
    live_compiled = kernels.compiled()
    aot_compiled = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh).compile_kernels(_abstract(_batch(cfg, 0)))

    self.assertEqual(sorted(live_compiled), sorted(aot_compiled))
    for name in live_compiled:
      # Optimized HLO, which is what the cost and memory analyses are measured on and what the
      # hardware actually runs.
      self.assertEqual(
          _normalize(live_compiled[name].as_text()),
          _normalize(aot_compiled[name].as_text()),
          f"{name}: the AOT executable differs from the one training ran",
      )

  def test_the_comparison_can_fail(self):
    """The vacuity guard for every assertion above.

    Width rather than sequence length, which would be the obvious knob and the wrong one:
    `_update_kernel` sees gradients and moments, never a sequence, so a shorter sequence leaves
    its HLO byte-identical and the guard would silently test nothing on that kernel.
    """
    cfg, mesh = _config_and_mesh()
    wider = _config(base_mlp_dim=256)

    compiled = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh).compile_kernels(_abstract(_batch(cfg, 0)))
    compiled_wider = maxtext_engine_compile.AbstractMaxTextEngine(wider, mesh).compile_kernels(
        _abstract(_batch(wider, 0))
    )

    for name, narrow in compiled.items():
      self.assertNotEqual(
          _normalize(narrow.as_text()),
          _normalize(compiled_wider[name].as_text()),
          f"{name}: doubling the MLP width changed nothing that survives normalization",
      )

  def test_recompiling_reproduces_the_kernels_training_ran(self):
    """`engine.compile_kernels()` is not a second code path -- on a live engine it re-derives its own.

    This is what licenses `_LiveKernels`: without it every comparison here would be between two
    compilations and none against training.
    """
    cfg, mesh = _config_and_mesh()

    engine, kernels = self._live(cfg, mesh)
    dispatched = kernels.compiled()
    recompiled = engine.compile_kernels(_batch(cfg, 0))

    for name in dispatched:
      self.assertEqual(_normalize(dispatched[name].as_text()), _normalize(recompiled[name].as_text()), name)

  def test_compile_covers_the_eval_kernel_too(self):
    """`compile()` is a promise that nothing after it stalls on XLA, and eval is part of a step.

    The eval kernel is compiled off the same dummy batch as the training ones, so a first
    `eval_step` of that shape has to dispatch through the executable rather than replace it.
    """
    cfg, mesh = _config_and_mesh()

    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    engine.compile(_batch(cfg, 0))
    compiled = engine._compiled_eval

    self.assertIsInstance(compiled, jax.stages.Compiled)
    with engine.eval_context():
      engine.eval_step(_batch(cfg, 1))
    self.assertIs(engine._compiled_eval, compiled, "the first eval_step recompiled instead of dispatching")

  def test_the_abstract_train_state_matches_the_live_one_leaf_for_leaf(self):
    """Where a divergence would come from, stated directly rather than through the HLO.

    Same tree, paths, shapes, dtypes and shardings -- including the adamw moments and the
    `count` scalars, whose eager shardings come from two different places on the two paths.
    """
    cfg, mesh = _config_and_mesh()

    live = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    live.compile(_batch(cfg, 0))
    abstract = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh)

    live_leaves = jax.tree_util.tree_flatten_with_path(nnx.split(live.state)[1])[0]
    abstract_leaves = jax.tree_util.tree_flatten_with_path(nnx.split(abstract.state)[1])[0]

    self.assertNotEmpty(live_leaves)
    self.assertEqual(len(live_leaves), len(abstract_leaves))
    for (live_path, live_leaf), (abstract_path, abstract_leaf) in zip(live_leaves, abstract_leaves):
      where = jax.tree_util.keystr(live_path)
      self.assertEqual(where, jax.tree_util.keystr(abstract_path))
      self.assertIsInstance(abstract_leaf, jax.ShapeDtypeStruct, f"{where} was materialized")
      self.assertEqual(live_leaf.shape, abstract_leaf.shape, where)
      self.assertEqual(live_leaf.dtype, abstract_leaf.dtype, where)
      self.assertEqual(live._mesh_sharding(live_leaf), abstract._mesh_sharding(abstract_leaf), where)

  def test_every_step_after_the_first_runs_the_same_kernels(self):
    """Otherwise an AOT report describes a program that runs once and is then replaced.

    `nnx.Optimizer` builds optax's `count` and its own `step` with `jnp.zeros` under no mesh, so
    they reach the first update uncommitted and come back from it committed -- a second argument
    signature, and a second compile of the largest kernel in the engine. `_place_state_on_mesh`
    settles them up front; this pins it.
    """
    cfg, mesh = _config_and_mesh()

    engine, kernels = self._live(cfg, mesh)
    for micro in range(2):
      engine.fwd_bwd(_batch(cfg, micro))
    engine.update()

    for name, calls in kernels.calls.items():
      self.assertLen(calls, 2, f"{name} should have been dispatched once per step")
      first, second = (jax.tree_util.tree_flatten_with_path(call)[0] for call in calls)
      differing = [jax.tree_util.keystr(path) for (path, a), (_, b) in zip(first, second) if a != b]
      self.assertEmpty(differing, f"{name} is dispatched with a different signature on the second step")

  def test_zero1_and_the_deferred_all_reduce_engage_on_both_paths(self):
    """Guards the two parameterizations that would otherwise pass by both doing nothing."""
    cfg, mesh = _config_and_mesh(shard_optimizer_over_data=True)

    # `compile()`, not a full step: both flags are set while the kernels are traced.
    live = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    live.compile(_batch(cfg, 0))
    abstract = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh)
    abstract.compile_kernels(_abstract(_batch(cfg, 0)))

    for engine, label in ((live, "live"), (abstract, "abstract")):
      self.assertIsNotNone(engine._zero1_params_shardings, f"Zero-1 never engaged on the {label} engine")
      self.assertIsNotNone(engine._unreduced_grad_shardings, f"the deferral never engaged on the {label} engine")

  def test_the_hlo_dump_filters_match_the_names_xla_gives_the_kernels(self):
    """Those filters are a claim about names XLA derives from the jitted callables.

    Nothing checks the claim at runtime: a filter that matches nothing leaves an empty dump, not
    an error, so a renamed kernel would go unnoticed until someone went looking for HLO.
    """
    cfg, mesh = _config_and_mesh()
    compiled = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh).compile_kernels(_abstract(_batch(cfg, 0)))

    self.assertLen(compiled, len(maxtext_engine_compile.KERNEL_NAMES))
    for kernel in compiled.values():
      module = re.search(r"^HloModule (\S+?),", kernel.as_text(), re.MULTILINE)
      self.assertIsNotNone(module, "the executable has no module name to match against")
      self.assertRegex(module.group(1), maxtext_engine_compile.HLO_DUMP_DEFAULTS["dump_hlo_local_module_name"])
      self.assertIn(maxtext_engine_compile.HLO_DUMP_DEFAULTS["dump_hlo_module_name"], module.group(1))


class AbstractMaxTextEngineTest(absltest.TestCase):
  """What an engine built without weights will and will not do."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def setUp(self):
    super().setUp()
    self.cfg, self.mesh = _config_and_mesh()

  def test_refuses_to_run_anything(self):
    """Silence here would be worse than an error: `update()` would report a step count."""
    engine = maxtext_engine_compile.AbstractMaxTextEngine(self.cfg, self.mesh)

    for operation, call in (
        ("fwd_bwd", lambda: engine.fwd_bwd(_batch(self.cfg, 0))),
        ("update", engine.update),
        ("save_checkpoint", lambda: engine.save_checkpoint({"step": 0})),
        ("restore_checkpoint", engine.restore_checkpoint),
    ):
      with self.subTest(operation=operation):
        with self.assertRaisesRegex(RuntimeError, "has only shapes"):
          call()

  def test_requires_a_mesh(self):
    with self.assertRaisesRegex(ValueError, "requires a mesh"):
      maxtext_engine_compile.AbstractMaxTextEngine(self.cfg, None)

  def test_compiling_needs_something_to_compile_against(self):
    engine = maxtext_engine_compile.AbstractMaxTextEngine(self.cfg, self.mesh)

    with self.assertRaisesRegex(ValueError, "needs a dummy payload"):
      engine.compile_kernels(None)


@pytest.mark.tpu_backend
class Qwen3TopologyTest(absltest.TestCase):
  """The worked example: qwen3-0.6b compiled for four v6e chips this host does not have.

  Needs libtpu, which knows a v6e's shape, but no TPU: the mesh is a topology description and
  nothing is executed on it. That is the case the script exists for, and the one the parity
  tests above miss, since they run on a mesh of real (if simulated) devices.
  """

  # Compiling qwen3-0.6b for a v6e-4 is the slowest thing in this file, and the two tests that
  # read the report only read it, so they share one.
  _compiled: dict | None = None

  def setUp(self):
    super().setUp()
    # `maxtext_engine_compile.main` sets this process-wide; put it back so the tests above cannot
    # be reordered into a different RNG implementation.
    previous = jax.config.jax_default_prng_impl
    self.addCleanup(jax.config.update, "jax_default_prng_impl", previous)

  def _qwen3_config(self) -> pyconfig.HyperParameters:
    """qwen3-0.6b at a sequence length short enough to compile inside a test."""
    return pyconfig.initialize(
        [
            "",
            get_test_config_path("base.yml"),
            "model_name=qwen3-0.6b",
            "run_name=engine_aot_qwen3_test",
            "compile_topology=v6e-4",
            "compile_topology_num_slices=1",
            "ici_fsdp_parallelism=4",
            "per_device_batch_size=1",
            "max_target_length=512",
            "attention=flash",
            "enable_checkpointing=false",
        ]
    )

  def _compiled_kernels(self) -> dict:
    compiled = Qwen3TopologyTest._compiled
    if compiled is None:
      cfg = self._qwen3_config()
      compiled = maxtext_engine_compile.compile_engine_kernels(cfg, pre_train_compile.get_topology_mesh(cfg))
      Qwen3TopologyTest._compiled = compiled
    return compiled

  def test_compiles_all_three_kernels_with_no_weights_and_no_hardware(self):
    compiled = self._compiled_kernels()

    self.assertEqual(sorted(compiled), sorted(maxtext_engine_compile.KERNEL_NAMES))
    for name, executable in compiled.items():
      memory = executable.memory_analysis()
      with self.subTest(kernel=name):
        # A report of zero would mean the kernel was lowered with its arguments closed over as
        # constants rather than passed.
        self.assertGreater(memory.argument_size_in_bytes, 0)
        self.assertGreater(executable.cost_analysis()["flops"], 0)

  def test_reports_the_forward_backward_peak_rather_than_the_updates(self):
    """A sanity check on the numbers, not just on their existence.

    Activations dominate a training step and the optimizer's arithmetic is elementwise, so a
    report where the update needs the most scratch describes something other than this model --
    most likely a batch that never reached the kernel.
    """
    compiled = self._compiled_kernels()

    fwd_bwd = compiled["fwd_bwd"].memory_analysis()
    update = compiled["update"].memory_analysis()
    self.assertGreater(fwd_bwd.temp_size_in_bytes, update.temp_size_in_bytes)
    # Two adamw moments per parameter on top of the parameters themselves, so the update's
    # arguments outweigh a single micro-batch's.
    self.assertGreater(update.argument_size_in_bytes, fwd_bwd.argument_size_in_bytes)

  def test_the_whole_script_writes_one_executable_per_kernel(self):
    output = os.path.join(self.create_tempdir().full_path, "engine.pickle")

    maxtext_engine_compile.main(_script_argv("run_name=engine_aot_save_test", f"compiled_trainstep_file={output}"))

    for name in maxtext_engine_compile.KERNEL_NAMES:
      written = maxtext_engine_compile.kernel_save_path(output, name)
      self.assertTrue(os.path.exists(written), f"{name} was not written to {written}")
      self.assertGreater(os.path.getsize(written), 0, written)

  def test_a_dump_that_matched_nothing_says_which_filter_was_wrong(self):
    """`upload_dump` deletes what it uploaded, and cannot be handed a directory XLA skipped.

    `jit_train_step` is `train.py`'s name for its fused step and the config default, and it is
    exactly what none of the engine's three kernels are called.
    """
    local_dir = os.path.join(self.create_tempdir().full_path, "xla_dump")

    with self.assertRaisesRegex(FileNotFoundError, "matched none of the engine's kernels"):
      maxtext_engine_compile.main(
          _script_argv(
              "run_name=engine_aot_dump_test",
              "dump_hlo=true",
              f"dump_hlo_local_dir={local_dir}",
              "dump_hlo_local_module_name=jit_train_step",
          )
      )

  def test_diloco_is_refused_rather_than_silently_reported_on(self):
    """The engine has no outer step, so these numbers would describe a different run."""
    with self.assertRaisesRegex(NotImplementedError, "enable_diloco"):
      maxtext_engine_compile.main(_script_argv("run_name=engine_aot_diloco_test", "enable_diloco=true"))


class CompileHelpersTest(absltest.TestCase):
  """The pieces of the script that decide what gets compiled, and under which name."""

  def test_the_shaped_batch_is_one_micro_batch_not_the_global_one(self):
    cfg = _config(gradient_accumulation_steps=4)
    shaped = maxtext_engine_compile.get_shaped_micro_batch(cfg)

    self.assertNotEmpty(shaped)
    for key, aval in shaped.items():
      self.assertEqual(aval.shape[0], int(cfg.micro_batch_size_to_train_on), key)

  def test_the_shaped_batch_matches_the_shapes_a_driver_feeds(self):
    """`train.loss_fn` slices its batch, so a mismatch here is a recompile, not an error."""
    cfg = _config()
    shaped = maxtext_engine_compile.get_shaped_micro_batch(cfg)
    driven = _batch(cfg, 0)

    self.assertContainsSubset(driven.keys(), shaped.keys())
    for key, array in driven.items():
      self.assertEqual(shaped[key].shape, array.shape, key)
      self.assertEqual(shaped[key].dtype, array.dtype, key)

  def test_an_abstract_batch_is_traced_rather_than_closed_over(self):
    """`_is_jax_dynamic` decides this, and a batch it classifies static is not passed at all."""
    batch = _batch(_config(), 0)
    dynamic, static = maxtext_engine._split_static_and_dynamic(_abstract(batch))

    self.assertEmpty(static)
    self.assertEqual(sorted(dynamic), sorted(batch))

  def test_each_kernel_is_saved_to_its_own_file(self):
    paths = [
        maxtext_engine_compile.kernel_save_path("/tmp/engine.pickle", name)
        for name in maxtext_engine_compile.KERNEL_NAMES
    ]

    self.assertEqual(len(set(paths)), len(maxtext_engine_compile.KERNEL_NAMES))
    self.assertEqual(paths[0], "/tmp/engine_fwd_bwd.pickle")

  def test_a_requested_dump_is_pointed_at_the_kernels(self):
    argv = maxtext_engine_compile.with_engine_hlo_dump_defaults(["", "base.yml", "dump_hlo=True"])

    self.assertEqual(argv[:3], ["", "base.yml", "dump_hlo=True"])
    self.assertContainsSubset([f"{key}={value}" for key, value in maxtext_engine_compile.HLO_DUMP_DEFAULTS.items()], argv)

  def test_a_run_that_asked_for_no_dump_is_left_alone(self):
    """The regex reaches `XLA_FLAGS` either way, so widening it here would dump on every run."""
    argv = ["", "base.yml", "compile_topology=v6e-4"]

    self.assertEqual(maxtext_engine_compile.with_engine_hlo_dump_defaults(argv), argv)

  def test_an_explicit_filter_is_not_overridden(self):
    argv = maxtext_engine_compile.with_engine_hlo_dump_defaults(
        ["", "base.yml", "dump_hlo=true", "dump_hlo_module_name=first"]
    )

    self.assertIn("dump_hlo_module_name=first", argv)
    self.assertNotIn("dump_hlo_module_name=kernel", argv)
    self.assertIn("dump_hlo_local_module_name=jit_.*kernel", argv)


_SUITE = (EngineAotParityTest, AbstractMaxTextEngineTest)


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _loader = unittest.defaultTestLoader
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.TestSuite(_loader.loadTestsFromTestCase(cls) for cls in _SUITE)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
