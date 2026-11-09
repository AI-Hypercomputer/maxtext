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

import contextlib
import dataclasses

import io
import os
import re
import subprocess
import sys
import types
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P
from maxtext.configs import maxtext_engine_types
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import grpo_engine_compile
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import maxtext_engine_compile
from maxtext.training_engine import memory_report
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import packing

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_XAOT_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

_MIB = 2**20
_GIB = 2**30
_FLOAT32_BYTES = 4

# Two ways, so that a sharded leaf's per-device bytes differ from its global ones. Abstract, so no
# second device is needed: `shard_shape` and `memory_kind` are all the report reads off a sharding.
_MESH = AbstractMesh((2,), ("data",), axis_types=(AxisType.Explicit,))


@pytest.mark.cpu_only
def test_engine_aot_parity_on_a_four_device_cpu_mesh():
  """Runs the multi-device classes below in a child process with four CPU devices.

  See the module docstring for why re-exec is the only option. The flag is
  *appended* rather
  than defaulted so it wins over whatever a sibling module set -- XLA takes the
  last occurrence
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


def _tiny_overrides(data_parallelism: int = _REQUIRED_DEVICES) -> list[str]:
  """A tiny real decoder, big enough that every sharding decision is visible in the HLO.

  `adamw` rather than `sgd`: its moments are the part of the train state the abstract path has
  to invent, and so the part most likely to be wrong.
  """
  return [
      "model_name=default",
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
      f"ici_data_parallelism={data_parallelism}",
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
      "profiler_steps=0",
  ]


def _config(*extra: str, data_parallelism: int = _REQUIRED_DEVICES, **overrides: Any) -> pyconfig.HyperParameters:
  """The tiny decoder, with each override replacing the default of the same key rather than repeating it."""
  extra_list = list(extra) + [f"{key}={value}" for key, value in overrides.items()]
  replaced = {override.split("=", 1)[0] for override in extra_list}
  defaults = [override for override in _tiny_overrides(data_parallelism) if override.split("=", 1)[0] not in replaced]
  return pyconfig.initialize(
      [
          "maxtext_engine_xaot_test.py",
          get_test_config_path("base.yml"),
          "run_name=engine_xaot_test",
      ]
      + defaults
      + extra_list,
      config_class=maxtext_engine_types.CompileConfig,
  )


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
    signature, and a second compile of the largest kernel in the engine. `_settle_state`
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
            "profiler_steps=0",
        ],
        config_class=maxtext_engine_types.CompileConfig,
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
    defaults = maxtext_engine_compile.HLO_DUMP_DEFAULTS
    self.assertIn("dump_hlo_module_name=first", argv)
    self.assertNotIn(f"dump_hlo_module_name={defaults['dump_hlo_module_name']}", argv)
    self.assertIn(f"dump_hlo_local_module_name={defaults['dump_hlo_local_module_name']}", argv)


def _aval(num_bytes: int, spec: P = P(), memory_kind: str | None = None) -> jax.ShapeDtypeStruct:
  """A float32 aval of `num_bytes` global bytes on `_MESH`.

  Nothing is allocated, so size is free.
  """
  sharding = NamedSharding(_MESH, spec)
  if memory_kind is not None:
    sharding = sharding.with_memory_kind(memory_kind)
  return jax.ShapeDtypeStruct((num_bytes // _FLOAT32_BYTES,), jnp.float32, sharding=sharding)


def _state(optimizer: dict[str, Any]) -> nnx.State:
  """A train state shaped as `nnx.split(TrainStateNNX(...))` shapes it: a model and an optimizer."""
  return nnx.State({"model": {"kernel": _aval(64 * _MIB)}, "optimizer": optimizer})


def _grpo_config_obj(temperature: float = 1.0, **kwargs: Any) -> algorithm_config.GRPOConfig:
  """Builds a `GRPOConfig` and sets its non-init `temperature` field."""
  cfg = algorithm_config.GRPOConfig(**kwargs)
  cfg.temperature = temperature
  return cfg


class _Compiled:
  """Stands in for `jax.stages.Compiled`, of which the report reads only `memory_analysis()`."""

  def __init__(self, **sizes: int):
    stats = {
        f"{prefix}{field}_size_in_bytes": 0
        for prefix in ("", "host_")
        for field in ("argument", "output", "alias", "temp", "generated_code")
    }
    unknown = set(sizes) - set(stats)
    if unknown:
      raise TypeError(f"not CompiledMemoryStats fields: {sorted(unknown)}")
    stats.update(sizes)
    self._stats = types.SimpleNamespace(**stats)

  def memory_analysis(self) -> types.SimpleNamespace:
    return self._stats


def _kernels(**per_kernel: dict[str, int]) -> dict[str, _Compiled]:
  """One stand-in per kernel, in `KERNEL_NAMES` order; a kernel not given reports all zeros."""
  return {name: _Compiled(**per_kernel.get(name, {})) for name in maxtext_engine_compile.KERNEL_NAMES}


def _by_kernel(rows) -> dict[str, Any]:
  return {row.kernel: row for row in rows}


def _header(lines: list[str]) -> str:
  """The table's column-header line; matched on its first columns, since legend prose can start with "kernel"."""
  return next(line for line in lines if re.match(r"kernel\s+arg\s+out\b", line))


class MemoryReportTest(absltest.TestCase):
  """The arithmetic, on stand-in executables whose numbers are chosen to tell the cases apart."""

  def test_state_not_passed_added_to_the_kernels_that_run_beside_it(self):
    """What a kernel runs beside without being passed it is in HBM while it runs, so it is added to its total.

    `fwd_bwd` and `fwd_bwd_accum` are not passed the optimizer; `update` is
    passed all of it.
    """
    optimizer = {"mu": _aval(8 * _MIB), "nu": _aval(8 * _MIB)}
    compiled = _kernels(
        fwd_bwd={"argument_size_in_bytes": 100 * _MIB},
        fwd_bwd_accum={
            "argument_size_in_bytes": 105 * _MIB,
            "output_size_in_bytes": 5 * _MIB,
            "alias_size_in_bytes": 5 * _MIB,
        },
        update={"argument_size_in_bytes": 100 * _MIB},
    )

    rows = _by_kernel(memory_report.memory_report(compiled, _state(optimizer)))

    self.assertEqual(rows["fwd_bwd"].not_passed, ("optimizer",))
    self.assertEqual(rows["fwd_bwd"].device_not_passed, 16 * _MIB)
    self.assertEqual(rows["fwd_bwd"].device_total, 116 * _MIB)
    self.assertEqual(rows["fwd_bwd_accum"].not_passed, ("optimizer",))
    self.assertEqual(rows["fwd_bwd_accum"].device_not_passed, 16 * _MIB)
    self.assertEqual(rows["fwd_bwd_accum"].device_total, 105 * _MIB + 16 * _MIB)
    # The update's own arguments already include all of it; adding it again double-counts it.
    self.assertEqual(rows["update"].not_passed, ())
    self.assertEqual(rows["update"].device_not_passed, 0)
    self.assertEqual(rows["update"].device_total, 100 * _MIB)

  def test_donated_buffer_counted_once(self):
    """`update` donates its 40 MiB state: that buffer is in the 100 MiB of arguments and the 60 of outputs."""
    compiled = _kernels(
        update={
            "argument_size_in_bytes": 100 * _MIB,
            "output_size_in_bytes": 60 * _MIB,
            "alias_size_in_bytes": 40 * _MIB,
            "temp_size_in_bytes": 10 * _MIB,
        }
    )

    row = _by_kernel(memory_report.memory_report(compiled, _state({})))["update"]

    self.assertEqual(row.resident, 130 * _MIB)

  def test_host_bytes_excluded_from_device_peak(self):
    """`update` with host-memory buffers, with byte counts taken from a TPU compile.

    The offloaded state is donated on the host, so it is in the host arguments,
    the host outputs
    and the host alias alike. None of it counts toward the device peak.
    """
    offloaded = 1_192_568_320
    compiled = _kernels(
        update={
            "argument_size_in_bytes": 1_192_578_048,
            "output_size_in_bytes": 596_295_392,
            "alias_size_in_bytes": 596_294_144,
            "temp_size_in_bytes": 1_529_469_760,
            "host_argument_size_in_bytes": offloaded,
            "host_output_size_in_bytes": offloaded,
            "host_alias_size_in_bytes": offloaded,
        }
    )

    row = _by_kernel(memory_report.memory_report(compiled, _state({})))["update"]

    self.assertEqual(
        (row.host_argument, row.host_output, row.host_alias),
        (offloaded, offloaded, offloaded),
    )
    self.assertEqual(row.resident, 1_192_578_048 + 596_295_392 - 596_294_144 + 1_529_469_760)
    self.assertEqual(row.device_total, row.resident)

  def test_host_temp_excluded_from_device_peak(self):
    """Host temp, e.g. activations offloaded by `decoder_layer_input=offload`, has its own column.

    It is host RAM, so it is not part of `resident`.
    """
    offloaded = 58_720_256
    forward = {
        "argument_size_in_bytes": 100 * _MIB,
        "temp_size_in_bytes": 10 * _MIB,
        "host_temp_size_in_bytes": offloaded,
    }
    compiled = _kernels(fwd_bwd=forward)

    rows = memory_report.memory_report(compiled, _state({}))
    report = memory_report.format_memory_report(rows).splitlines()

    self.assertIn("host temp", _header(report))
    row = _by_kernel(rows)["fwd_bwd"]
    self.assertEqual(row.host_temp, offloaded)
    self.assertEqual(row.resident, 110 * _MIB)
    self.assertEqual(row.device_total, 110 * _MIB)
    cells = next(line for line in report if line.startswith("fwd_bwd ")).split()
    # kernel, 7 device columns, host arg, host out, host alias, then host temp.
    self.assertEqual(cells[11], f"{offloaded / _GIB:.2f}")
    self.assertEqual(_by_kernel(rows)["update"].host_temp, 0)

  def test_xla_counts_donated_buffer_twice(self):
    """On a real executable, a donated buffer is in `argument` and `output`, and `resident` counts it once."""
    buffer = jnp.zeros((256 * 1024,), jnp.float32)
    update = jax.jit(lambda state, grads: state - grads, donate_argnums=(0,))
    compiled = {"update": update.lower(buffer, buffer).compile()}

    row = memory_report.memory_report(compiled, _state({}))[0]

    self.assertEqual(row.alias, buffer.nbytes)
    self.assertEqual(
        row.argument + row.output,
        3 * buffer.nbytes,
        "the raw sum counts the donated state twice",
    )
    self.assertEqual(row.resident - row.temp, 2 * buffer.nbytes)

  def test_host_optimizer_not_counted_as_device(self):
    """An optimizer in host memory moves from `+state` to `host state` for each kernel it is not passed to."""
    for memory_kind in ("pinned_host", "unpinned_host"):
      with self.subTest(memory_kind=memory_kind):
        optimizer = {
            "mu": _aval(8 * _MIB, memory_kind=memory_kind),
            "nu": _aval(8 * _MIB, memory_kind=memory_kind),
        }

        rows = _by_kernel(memory_report.memory_report(_kernels(), _state(optimizer)))

        self.assertEqual(rows["fwd_bwd"].device_not_passed, 0)
        self.assertEqual(rows["fwd_bwd"].host_not_passed, 16 * _MIB)
        self.assertEqual(rows["fwd_bwd_accum"].device_not_passed, 0)
        self.assertEqual(rows["fwd_bwd_accum"].host_not_passed, 16 * _MIB)

  def test_device_and_unset_memory_kind_count_as_device(self):
    """A concrete mesh reports `device`; an abstract one, or a leaf not yet placed, reports None."""
    optimizer = {
        "mu": _aval(8 * _MIB, memory_kind="device"),
        "nu": _aval(8 * _MIB),
        "count": _aval(8 * _MIB, memory_kind="pinned_host"),
    }

    row = _by_kernel(memory_report.memory_report(_kernels(), _state(optimizer)))["fwd_bwd"]

    self.assertEqual(row.device_not_passed, 16 * _MIB)
    self.assertEqual(row.host_not_passed, 8 * _MIB)

  def test_per_device_bytes_follow_sharding(self):
    """What fits is per device: a leaf split two ways costs each device half of it."""
    self.assertEqual(
        memory_report.per_device_bytes(_aval(8 * _MIB, P("data"))),
        (4 * _MIB, 0),
    )
    self.assertEqual(memory_report.per_device_bytes(_aval(8 * _MIB)), (8 * _MIB, 0))
    unsharded = jax.ShapeDtypeStruct((2 * _MIB,), jnp.float32)
    self.assertEqual(memory_report.per_device_bytes(unsharded), (8 * _MIB, 0))

    optimizer = {"mu": _aval(8 * _MIB, P("data")), "nu": _aval(8 * _MIB)}
    row = _by_kernel(memory_report.memory_report(_kernels(), _state(optimizer)))["fwd_bwd"]
    self.assertEqual(row.device_not_passed, 12 * _MIB)

  def test_prng_key_leaf_bytes(self):
    """The model's RNG state carries key dtypes, which `np.dtype` refuses; threefry keys are 8 bytes."""
    keys = jax.ShapeDtypeStruct((3,), jax.random.key(0).dtype)

    self.assertEqual(memory_report.per_device_bytes(keys), (24, 0))

  def test_peak_includes_state_not_passed(self):
    """The peak is the largest `device_total`, even when another kernel has the largest `resident`."""
    optimizer = {"mu": _aval(8 * _MIB), "nu": _aval(8 * _MIB)}
    compiled = _kernels(
        fwd_bwd={
            "argument_size_in_bytes": 60 * _MIB,
            "temp_size_in_bytes": 40 * _MIB,
        },
        update={
            "argument_size_in_bytes": 100 * _MIB,
            "temp_size_in_bytes": 10 * _MIB,
        },
    )

    rows = memory_report.memory_report(compiled, _state(optimizer))
    by_kernel = _by_kernel(rows)
    peak = memory_report.device_peak(rows)

    self.assertGreater(by_kernel["update"].resident, by_kernel["fwd_bwd"].resident)
    self.assertEqual(peak.kernel, "fwd_bwd")
    self.assertEqual(peak.device_total, 116 * _MIB)

  def test_verdict_names_peak_and_parts(self):
    # Sized in whole hundredths of a GiB so the expected strings are exact.
    compiled = _kernels(
        fwd_bwd={
            "argument_size_in_bytes": 40 * _GIB,
            "temp_size_in_bytes": 20 * _GIB,
        },
        fwd_bwd_accum={
            "argument_size_in_bytes": int(42.5 * _GIB),
            "output_size_in_bytes": int(1.25 * _GIB),
            "alias_size_in_bytes": int(1.25 * _GIB),
            "temp_size_in_bytes": int(18.75 * _GIB),
        },
        update={"argument_size_in_bytes": 50 * _GIB},
    )
    on_device = {"mu": _aval(int(4.375 * _GIB)), "nu": _aval(int(4.375 * _GIB))}
    on_host = {key: _aval(int(4.375 * _GIB), memory_kind="pinned_host") for key in ("mu", "nu")}
    cases = (
        (
            on_device,
            ("TRAIN-KERNEL DEVICE PEAK: 70.00 GiB (fwd_bwd_accum 61.25 +" " optimizer state 8.75 not passed to it)"),
        ),
        (
            on_host,
            (
                "TRAIN-KERNEL DEVICE PEAK: 61.25 GiB (fwd_bwd_accum 61.25 +"
                " optimizer state 0.00 not passed to it; 8.75 GiB more of it"
                " is on host)"
            ),
        ),
    )
    for optimizer, verdict in cases:
      with self.subTest(verdict=verdict):
        report = memory_report.format_memory_report(memory_report.memory_report(compiled, _state(optimizer)))
        lines = report.splitlines()

        self.assertEqual(lines[-1], verdict)
        for name in maxtext_engine_compile.KERNEL_NAMES:
          self.assertLen([line for line in lines if line.startswith(f"{name} ")], 1, name)

  def test_verdict_when_update_is_peak(self):
    compiled = _kernels(update={"argument_size_in_bytes": 50 * _GIB})

    report = memory_report.format_memory_report(memory_report.memory_report(compiled, _state({})))

    self.assertEqual(
        report.splitlines()[-1],
        "TRAIN-KERNEL DEVICE PEAK: 50.00 GiB (update, which is passed all the" " train state it runs with)",
    )

  def test_table_cells_align_with_headers(self):
    """The table body, cell by cell: every value distinct, so a swapped column or a wrong field shows.

    Sized in quarter GiBs so every cell prints exactly.
    """
    compiled = _kernels(
        fwd_bwd={
            "argument_size_in_bytes": 3 * _GIB,
            "output_size_in_bytes": 2 * _GIB,
            "alias_size_in_bytes": 1 * _GIB,
            "temp_size_in_bytes": 4 * _GIB,
            "host_argument_size_in_bytes": _GIB // 2,
            "host_output_size_in_bytes": 3 * _GIB // 4,
            "host_alias_size_in_bytes": _GIB // 4,
            "host_temp_size_in_bytes": 7 * _GIB // 4,
        }
    )
    optimizer = {
        "mu": _aval(9 * _GIB // 4),
        "nu": _aval(11 * _GIB // 4, memory_kind="pinned_host"),
    }
    expected = {
        "arg": "3.00",
        "out": "2.00",
        "alias": "1.00",
        "temp": "4.00",
        "resident": "8.00",
        "+state": "2.25",
        "total": "10.25",
        "host arg": "0.50",
        "host out": "0.75",
        "host alias": "0.25",
        "host temp": "1.75",
        "host state": "2.75",
    }

    lines = memory_report.format_memory_report(memory_report.memory_report(compiled, _state(optimizer))).splitlines()
    header = _header(lines)
    row = next(line for line in lines if line.startswith("fwd_bwd "))

    self.assertEqual(row.split()[: len(expected) + 1], ["fwd_bwd", *expected.values()])
    self.assertTrue(row.endswith("   optimizer"), row)
    # Right-aligned: each cell ends where its header ends, in this order left to right.
    position = len("kernel")
    for name, cell in expected.items():
      with self.subTest(column=name):
        end = header.index(name, position) + len(name)
        self.assertEqual(row[end - len(cell) : end], cell)
        self.assertEqual(row[end - len(cell) - 1], " ")
        position = end
    self.assertGreater(header.index("state not passed", position), position)

  def test_legend_lists_exclusions(self):
    """The legend lists what the train-kernel peak excludes, so it is not read as the whole step's peak."""
    report = memory_report.format_memory_report(memory_report.memory_report(_kernels(), _state({})))
    lines = report.splitlines()
    legend = " ".join(lines[: lines.index(_header(lines))])

    self.assertNotIn("ENGINE DEVICE PEAK", report)
    for excluded in (
        "weight-sync staging",
        "release_weight_sync",
        "fwd_only",
        "eval",
        "generated code",
        "the caller",
    ):
      with self.subTest(excluded=excluded):
        self.assertIn(excluded, legend)

  def test_rejects_unaccounted_state(self):
    """Each of these would otherwise report a peak lower than the real one."""
    with self.assertRaisesRegex(ValueError, "no entry for kernel 'fwd_only'"):
      memory_report.memory_report({"fwd_only": _Compiled()}, _state({}))
    with self.assertRaisesRegex(ValueError, r"no \['optimizer'\] subtree"):
      memory_report.memory_report(_kernels(), nnx.State({"model": {"kernel": _aval(_MIB)}}))
    # A third subtree, e.g. an EMA of the weights: resident, and in no kernel's arguments or `+state`.
    third = nnx.State(
        {
            "model": {"kernel": _aval(_MIB)},
            "optimizer": {},
            "ema": {"kernel": _aval(_MIB)},
        }
    )
    with self.assertRaisesRegex(
        ValueError,
        r"subtrees \['ema'\] that STATE_NOT_PASSED does not classify",
    ):
      memory_report.memory_report(_kernels(), third)

    no_analysis = _Compiled()
    no_analysis.memory_analysis = lambda: None
    with self.assertRaisesRegex(ValueError, "has no memory analysis"):
      memory_report.memory_report({"update": no_analysis}, _state({}))

  def test_every_kernel_has_state_entry(self):
    """A kernel added to the engine must be classified here, or the report refuses to run."""
    self.assertEqual(
        sorted(memory_report.STATE_NOT_PASSED),
        sorted(maxtext_engine_compile.KERNEL_NAMES),
    )


def _subtree_bytes(state, key: str, memory: int = 0) -> int:
  """Bytes per device of one top-level subtree, by the report's own per-leaf rule.

  `memory` indexes `per_device_bytes`'s `(device, host)`: device bytes by
  default, 1 for host.
  """
  return sum(memory_report.per_device_bytes(leaf)[memory] for leaf in jax.tree.leaves(state[key]))


def _run_main(test: absltest.TestCase, *extra: str) -> str:
  """Runs `main` itself on this host's CPU and returns what it printed.

  Only the topology lookup, which needs libtpu, is replaced. `main` sets the
  PRNG implementation
  process-wide, so it is put back. Each of `extra` replaces the default of the
  same key.
  """
  previous = jax.config.jax_default_prng_impl
  test.addCleanup(jax.config.update, "jax_default_prng_impl", previous)
  replaced = {override.split("=", 1)[0] for override in extra}
  defaults = [
      override
      for override in _tiny_overrides(1)
      if not override.startswith("ici_") and override.split("=", 1)[0] not in replaced
  ]
  argv = (
      [
          "",
          get_test_config_path("base.yml"),
          "run_name=engine_compile_main_test",
      ]
      + ["compile_topology=v6e-1", "compile_topology_num_slices=1"]
      + defaults
      + list(extra)
  )
  stdout = io.StringIO()
  with (
      mock.patch.dict(os.environ),
      mock.patch.object(
          pre_train_compile,
          "get_topology_mesh",
          maxtext_utils.get_mesh_from_config,
      ),
      contextlib.redirect_stdout(stdout),
  ):
    maxtext_engine_compile.main(argv)
  return stdout.getvalue()


def _run_main_and_capture(test: absltest.TestCase, *extra: str) -> tuple[str, Any, Any, dict[str, Any]]:
  """`_run_main`, also returning the engine `main` compiled, the micro-batch it compiled for, and the executables."""
  calls = []
  compile_kernels = maxtext_engine_compile.AbstractMaxTextEngine.compile_kernels

  def capture(engine, micro_batch, *args, **kwargs):
    compiled = compile_kernels(engine, micro_batch, *args, **kwargs)
    calls.append((engine, micro_batch, compiled))
    return compiled

  with mock.patch.object(maxtext_engine_compile.AbstractMaxTextEngine, "compile_kernels", capture):
    output = _run_main(test, *extra)
  test.assertLen(calls, 1)
  return (output, *calls[0])


class CompiledMemoryReportTest(absltest.TestCase):
  """The report on the engine's real kernels, checked against what XLA says it allocated."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cfg = _config(data_parallelism=1)
    cls._engine, micro_batch = maxtext_engine_compile._engine_and_micro_batch(
        cfg, maxtext_utils.get_mesh_from_config(cfg)
    )
    cls._compiled = cls._engine.compile_kernels(micro_batch)

  # CPU only: the report counts logical shard bytes, while a TPU executable's buffers also carry
  # tiled-layout padding, so the byte counts are exactly equal only on XLA:CPU.
  @pytest.mark.cpu_only
  def test_report_matches_xla_allocation(self):
    """The report's bytes match XLA's, and each kernel is charged with what it runs beside.

    `update` donates the whole train state and `fwd_bwd_accum` the running
    gradient
    sum, so their aliases are XLA's own counts of those. `fwd_bwd` and
    `fwd_bwd_accum`
    are not passed the optimizer.
    """
    state = self._engine.train_state_avals()
    model, optimizer = _subtree_bytes(state, "model"), _subtree_bytes(state, "optimizer")
    params, _ = nnx.split_state(state["model"], nnx.Param, ...)
    # One float32 gradient per parameter, plus the float32 denominator.
    gradient_sum = sum(memory_report.per_device_bytes(leaf)[0] for leaf in jax.tree.leaves(params)) + 4
    rows = _by_kernel(memory_report.memory_report(self._compiled, state))

    self.assertGreater(optimizer, 0)
    self.assertEqual(rows["update"].alias, model + optimizer)
    self.assertEqual(rows["update"].device_not_passed, 0)
    self.assertEqual(rows["fwd_bwd_accum"].alias, gradient_sum)
    self.assertEqual(rows["fwd_bwd_accum"].device_not_passed, optimizer)
    self.assertGreaterEqual(rows["fwd_bwd"].argument, model)
    self.assertLess(
        rows["fwd_bwd"].argument,
        model + optimizer,
        "the optimizer is among fwd_bwd's arguments",
    )
    self.assertEqual(rows["fwd_bwd"].device_not_passed, optimizer)
    self.assertGreaterEqual(rows["fwd_bwd_accum"].output, gradient_sum)

  def test_train_state_avals_returns_copy(self):
    """The engine's cached state is re-placed in place on a recompile, so a caller must not share it."""
    cfg = _config(data_parallelism=1)
    engine, micro_batch = maxtext_engine_compile._engine_and_micro_batch(cfg, maxtext_utils.get_mesh_from_config(cfg))
    engine.compile_kernels(micro_batch)

    handed_out = engine.train_state_avals()
    self.assertIsNot(handed_out, engine._read_state_pure())
    handed_out["optimizer"] = nnx.State({})

    self.assertGreater(
        _subtree_bytes(engine._read_state_pure(), "optimizer"),
        0,
        "the caller's edit reached the engine",
    )
    self.assertEqual(
        _subtree_bytes(engine.train_state_avals(), "optimizer"),
        _subtree_bytes(engine._read_state_pure(), "optimizer"),
    )

  def test_train_state_avals_requires_compile(self):
    """Before compiling, the optimizer is not yet on the layout it runs on."""
    cfg = _config(data_parallelism=1)
    engine = maxtext_engine_compile.AbstractMaxTextEngine(cfg, maxtext_utils.get_mesh_from_config(cfg))

    with self.assertRaisesRegex(RuntimeError, "only meaningful after compile_kernels"):
      engine.train_state_avals()

  def test_main_prints_report_after_analyses(self):
    lines = _run_main(self).splitlines()
    raw = [index for index, line in enumerate(lines) if line.startswith("Memory analysis: ")]
    verdict = [index for index, line in enumerate(lines) if line.startswith("TRAIN-KERNEL DEVICE PEAK: ")]
    self.assertLen(
        raw,
        len(maxtext_engine_compile.KERNEL_NAMES),
        "the raw analyses must still be printed",
    )
    self.assertLen(verdict, 1)
    self.assertGreater(verdict[0], raw[-1])
    self.assertRegex(
        lines[verdict[0]],
        r"^TRAIN-KERNEL DEVICE PEAK: \d+\.\d\d GiB" r" \((fwd_bwd|fwd_bwd_accum|update)\b",
    )
    for name in maxtext_engine_compile.KERNEL_NAMES:
      self.assertLen(
          [line for line in lines[raw[-1] : verdict[0]] if line.startswith(f"{name} ")],
          1,
          name,
      )

  def test_main_saves_executables_before_report(self):
    """When `memory_report` raises (here: no memory analysis), the executables are already saved."""
    output = os.path.join(self.create_tempdir().full_path, "engine.pickle")

    with mock.patch.object(jax.stages.Compiled, "memory_analysis", return_value=None):
      with self.assertRaisesRegex(ValueError, "has no memory analysis on this backend"):
        _run_main(self, f"compiled_trainstep_file={output}")

    for name in maxtext_engine_compile.KERNEL_NAMES:
      written = maxtext_engine_compile.kernel_save_path(output, name)
      self.assertTrue(os.path.exists(written), f"{name} was not written to {written}")
      self.assertGreater(os.path.getsize(written), 0, written)


# The Tunix adapter needs a model with a Hugging Face config. Two sequences per micro-batch and two
# micro-batches per update, so the micro-batch is not the global batch; 8 of the 32 tokens are prompt,
# so the prompt and the completion differ in length. `shard_mode=auto` is the default, which the config
# Tunix builds for its trainer keeps. The temperature is not `GRPOConfig`'s default, so a compile that
# ignored `decode_sampling_temperature` would be seen.
_GRPO_OVERRIDES = (
    "loss=grpo",
    "model_name=qwen3-0.6b",
    "override_model_config=True",
    "shard_mode=auto",
    "per_device_batch_size=2",
    "gradient_accumulation_steps=2",
    "max_prefill_predict_length=8",
    "decode_sampling_temperature=0.7",
)


# A Qwen3.5 small enough to compile on CPU, whose four layers route each token to two of four experts: router
# replay needs a decoder whose MoE takes forced routing. `sparse_matmul=False` keeps the MoE off megablox, which
# XLA:CPU can only interpret, and the model's `mrope_section` needs `head_dim * partial_rotary_factor / 2 == 32`.
_QWEN35_OVERRIDES = (
    "model_name=qwen3.5-35b-a3b",
    "sparse_matmul=False",
    "base_num_decoder_layers=4",
    "base_emb_dim=64",
    "base_num_query_heads=2",
    "base_num_kv_heads=2",
    "head_dim=256",
    "base_mlp_dim=64",
    "base_moe_mlp_dim=64",
    "num_experts=4",
    "num_experts_per_tok=2",
    "gdn_key_head_dim=16",
    "gdn_value_head_dim=16",
    "gdn_num_key_heads=2",
    "gdn_num_value_heads=4",
    "gdn_chunk_size=16",
)


def _grpo_overrides(*extra: str) -> list[str]:
  """`_GRPO_OVERRIDES`, each of `extra` replacing the override of the same key."""
  replaced = {override.split("=", 1)[0] for override in extra}
  return [override for override in _GRPO_OVERRIDES if override.split("=", 1)[0] not in replaced] + list(extra)


def _grpo_config(*extra: str) -> pyconfig.HyperParameters:
  """The tiny decoder with `_grpo_overrides(*extra)`."""
  return _config(*_grpo_overrides(*extra), data_parallelism=1)


def _tree_bytes(tree) -> int:
  """Device bytes per device of every leaf in `tree`, by the report's own per-leaf rule."""
  return sum(memory_report.per_device_bytes(leaf)[0] for leaf in jax.tree.leaves(tree))


def _layout(payload) -> dict[str, Any]:
  """Each field of an `RLTrainerPayload` as `(shape, dtype)`, or as itself where it is None or `num_segments`."""
  layout = {}
  for field in dataclasses.fields(payload):
    value = getattr(payload, field.name)
    if field.name != "metadata":
      layout[field.name] = (tuple(value.shape), jnp.dtype(value.dtype)) if hasattr(value, "shape") else value
  return layout


def _assembled_micro_batch(cfg: pyconfig.HyperParameters, algo) -> datatypes.RLTrainerPayload:
  """One micro-batch as Tunix's orchestrator builds it for `algo`, from rollouts shorter than the padded lengths.

  The rollouts report their log-probabilities and a status. They go through
  `algo.create_trainer_payloads`
  and `PaddedBatchAssembler`, and get the reference model's log-probabilities
  when `algo` needs them.
  """
  micro_batch_size = cfg.micro_batch_size_to_train_on
  prompt_length = cfg.max_prefill_predict_length
  completion_length = cfg.max_target_length - prompt_length
  rng = np.random.default_rng(0)
  items = []
  for index in range(micro_batch_size):
    completion = rng.integers(1, cfg.vocab_size, size=completion_length - 1 - index, dtype=np.int32)
    traj = {
        "prompt_tokens": rng.integers(1, cfg.vocab_size, size=prompt_length - 1, dtype=np.int32),
        "conversation_tokens": completion,
        "conversation_masks": np.ones(completion.size, np.float32),
        "old_logprobs": np.full(completion.size, -1.0, np.float32),
        "status": "SUCCEEDED",
    }
    items.append(datatypes.TrajectoryItem(prompt_id="prompt", group_index=index, traj=traj))
  payloads = algo.create_trainer_payloads(items, rewards=[float(index % 2) for index in range(micro_batch_size)])
  assembler = batch_assembly.PaddedBatchAssembler(
      batch_size=micro_batch_size,
      max_prompt_length=prompt_length,
      max_response_length=completion_length,
      pad_id=grpo_engine_compile.GRPO_PAD_ID,
      num_generations=algo.num_generations,
      mini_batch_size=1,
  )
  batch = assembler.pack(payloads)[0]
  if algo.requires_reference_kl:
    batch = batch_assembly.with_ref_per_token_logps(batch, np.zeros(np.shape(batch.completion_ids), np.float32))
  return batch


def _orchestrator_micro_batch(
    cfg: pyconfig.HyperParameters,
    algo,
    assembler,
    num_rollouts: int,
    routed: bool = False,
) -> datatypes.RLTrainerPayload:
  """The micro-batch Tunix's orchestrator hands the trainer from `num_rollouts` rollouts of the longest length allowed.

  As `StandardRLProgram` handles them: `algo.create_trainer_payloads` a group at
  a time, tagged with their
  trajectory ids, fed to `assembler`, and given the reference model's
  log-probabilities when `algo` needs them.
  The rollouts are as long as `get_rl_micro_batch`'s, since lengths decide the
  layout, but their tokens,
  log-probabilities, rewards and, with `routed`, experts are random.
  """
  prompt_length = cfg.max_prefill_predict_length
  completion_length = cfg.max_target_length - prompt_length
  rng = np.random.default_rng(0)
  group_size = algo.num_generations
  payloads = []
  for prompt_index in range(-(-num_rollouts // group_size)):
    items = []
    for index in range(group_size):
      traj = {
          "prompt_tokens": rng.integers(1, cfg.vocab_size, size=prompt_length, dtype=np.int32),
          "conversation_tokens": rng.integers(1, cfg.vocab_size, size=completion_length, dtype=np.int32),
          "conversation_masks": np.ones(completion_length, np.float32),
          "old_logprobs": -rng.random(completion_length, dtype=np.float32),
          "status": "SUCCEEDED",
      }
      if routed:
        routing_shape = (
            prompt_length + completion_length,
            cfg.num_decoder_layers,
            cfg.num_experts_per_tok,
        )
        traj["routed_experts"] = rng.integers(0, cfg.num_experts, size=routing_shape, dtype=np.int16)
      items.append(datatypes.TrajectoryItem(prompt_id=f"prompt_{prompt_index}", group_index=index, traj=traj))
    for item, payload in zip(
        items,
        algo.create_trainer_payloads(items, rewards=rng.random(group_size).tolist()),
    ):
      payloads.append(
          dataclasses.replace(
              payload,
              metadata={**payload.metadata, "traj_id": item.traj_id},
          )
      )
  batches = [*assembler.feed(payloads[:num_rollouts]), *assembler.flush()]
  assert len(batches) == 1, f"the rollouts filled {len(batches)} micro-batches"
  batch = batches[0].payload
  if algo.requires_reference_kl:
    batch = batch_assembly.with_ref_per_token_logps(batch, np.zeros(np.shape(batch.completion_ids), np.float32))
  return batch


def _assert_same_layout(test: absltest.TestCase, actual, expected) -> None:
  """Asserts that two `RLTrainerPayload`s are laid out alike.

  The same `_layout`, and the same values wherever a value places a token: the
  masks, the segment ids and
  positions, which token ids are padding and which routing slots are unset.
  """
  test.assertEqual(_layout(actual), _layout(expected))
  for name in (
      "prompt_mask",
      "completion_mask",
      "segment_ids",
      "segment_positions",
  ):
    if getattr(expected, name) is not None:
      np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)
  for name in ("prompt_ids", "completion_ids"):
    padding = [np.asarray(getattr(payload, name)) == grpo_engine_compile.GRPO_PAD_ID for payload in (actual, expected)]
    np.testing.assert_array_equal(*padding, err_msg=f"padding in {name}")
  if expected.routed_experts is not None:
    unset = [
        packing.decode_routed_experts(np.asarray(payload.routed_experts), batched=True) == datatypes.UNSET_ROUTED_EXPERT
        for payload in (actual, expected)
    ]
    np.testing.assert_array_equal(*unset, err_msg="unset routing slots")


def _fwd_bwd_memory(compiled: dict[str, Any]) -> tuple[int, int]:
  """`fwd_bwd`'s argument and temporary bytes."""
  stats = compiled["fwd_bwd"].memory_analysis()
  return stats.argument_size_in_bytes, stats.temp_size_in_bytes


class GrpoCompileTest(parameterized.TestCase):
  """`loss`: the loss the engine's kernels are compiled with."""

  def test_main_compiles_grpo_loss(self):
    """`grpo` compiles every kernel through Tunix's GRPO loss, set up as Tunix's RL trainer sets up the engine."""
    output, engine, micro_batch, compiled = _run_main_and_capture(self, *_GRPO_OVERRIDES)
    cfg = engine._config

    self.assertIsInstance(engine.model, TunixMaxTextAdapter)
    self.assertEqual(engine.model._pad_id, grpo_engine_compile.GRPO_PAD_ID)
    self.assertIs(engine._loss_fn, algo_core.grpo_loss_fn)
    self.assertTrue(engine._has_aux)
    inputs = engine._gen_model_input_fn(micro_batch)
    self.assertEqual(sorted(inputs), ["algo_config", "eos_id", "pad_id", "train_example"])
    self.assertIs(inputs["train_example"], micro_batch)
    self.assertEqual(
        (inputs["pad_id"], inputs["eos_id"]),
        (
            grpo_engine_compile.GRPO_PAD_ID,
            grpo_engine_compile.GRPO_EOS_ID,
        ),
    )
    # `GRPOConfig`'s defaults, but for the temperature.
    self.assertEqual(
        inputs["algo_config"],
        _grpo_config_obj(temperature=cfg.decode_sampling_temperature),
    )

    # Two sequences of 8 prompt and 24 completion tokens, laid out as Tunix lays out a micro-batch for this
    # algorithm.
    self.assertEqual(micro_batch.prompt_ids.shape, (2, 8))
    self.assertEqual(micro_batch.completion_ids.shape, (2, 24))
    algo = algorithm_adapter.GRPOAdapter(inputs["algo_config"])
    self.assertEqual(_layout(micro_batch), _layout(_assembled_micro_batch(cfg, algo)))

    # `fwd_bwd` takes the payload and returns GRPO's metrics, not those of MaxText's loss.
    batch = compiled["fwd_bwd"].args_info[0][2]
    self.assertEqual(list(batch), ["train_example"])
    self.assertEqual(_layout(batch["train_example"]), _layout(micro_batch))
    aux = compiled["fwd_bwd"].out_info[1]
    self.assertContainsSubset({"pg_clipfrac", "advantage/abs_mean", "kl"}, set(aux))
    self.assertNotIn("xent_sum", aux)

    # The payload is among `fwd_bwd`'s arguments, beside the model. `jax.jit` drops the arguments a program
    # never reads, which for this loss is the prompt mask.
    state = engine.train_state_avals()
    rows = _by_kernel(memory_report.memory_report(compiled, state))
    unread = _tree_bytes(micro_batch.prompt_mask)
    self.assertEqual(
        rows["fwd_bwd"].argument,
        _subtree_bytes(state, "model") + _tree_bytes(micro_batch) - unread,
    )

    lines = output.splitlines()
    split = "Compiling Tunix's GRPO loss, for micro-batches of 2 sequences of 8" " prompt and 24 completion tokens."
    self.assertIn(split, lines)
    raw = [index for index, line in enumerate(lines) if line.startswith("Memory analysis: ")]
    verdict = [index for index, line in enumerate(lines) if line.startswith("TRAIN-KERNEL DEVICE PEAK: ")]
    self.assertLen(raw, len(maxtext_engine_compile.KERNEL_NAMES))
    self.assertLen(verdict, 1)
    self.assertLess(lines.index(split), raw[0])
    self.assertLess(
        lines.index(split),
        lines.index("Jitting and compiling the engine's kernels..."),
    )
    self.assertGreater(verdict[0], raw[-1])

  def test_main_compiles_trainer_grpo_options(self):
    """The trainer's `GRPOConfig` options and log-probability chunk size reach the compiled program."""
    _, engine, micro_batch, compiled = _run_main_and_capture(
        self,
        *_GRPO_OVERRIDES,
        "grpo_config={beta: 0.0, use_rollout_logps: false}",
        "logps_chunk_size=4",
    )
    inputs = engine._gen_model_input_fn(micro_batch)
    self.assertEqual(
        inputs["algo_config"],
        _grpo_config_obj(
            temperature=engine._config.decode_sampling_temperature,
            beta=0.0,
            use_rollout_logps=False,
        ),
    )
    self.assertIsNone(micro_batch.ref_per_token_logps)
    self.assertIsNone(micro_batch.old_per_token_logps)
    self.assertEqual(inputs["compute_logps_chunk_size"], 4)

    # Tunix scans over the chunks, forward and backward, so the chunked program has more loops than the other.
    _, _, _, unchunked = _run_main_and_capture(self, *_GRPO_OVERRIDES)
    self.assertGreater(
        compiled["fwd_bwd"].as_text().count(" while("),
        unchunked["fwd_bwd"].as_text().count(" while("),
    )

  def test_main_compiles_maxtext_loss_by_default(self):
    """`maxtext`, the default, compiles MaxText's own loss on a pre-training batch, with no Tunix adapter."""
    self.assertEqual(_config(data_parallelism=1).loss, "maxtext")
    for extra in ((), ("loss=maxtext",)):
      with self.subTest(extra=extra):
        output, engine, micro_batch, compiled = _run_main_and_capture(self, *extra)

        self.assertNotIsInstance(engine.model, TunixMaxTextAdapter)
        self.assertIsNone(engine._loss_fn)
        self.assertIsNone(engine._gen_model_input_fn)
        self.assertEqual(
            micro_batch,
            maxtext_engine_compile.get_shaped_micro_batch(engine._config),
        )
        self.assertIn("xent_sum", compiled["fwd_bwd"].out_info[1])
        self.assertNotIn("GRPO", output)

  def test_rejects_unknown_loss(self):
    with self.assertRaisesRegex(ValueError, r"loss\s+Input should be 'maxtext' or 'grpo'"):
      _config("loss=ppo", data_parallelism=1)

  @parameterized.named_parameters(
      # A non-zero `beta`, so the reference model's log-probabilities are present too.
      ("grpo_config_defaults", {}),
      ("no_kl_term", {"beta": 0.0}),
      ("old_logps_recomputed", {"use_rollout_logps": False}),
  )
  def test_rl_micro_batch_matches_tunix(self, grpo_config):
    """The payload has the fields Tunix gives such a micro-batch, with the same shapes and dtypes, and no others."""
    cfg = _grpo_config()
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0, **grpo_config))

    micro_batch = grpo_engine_compile.get_shaped_rl_micro_batch(cfg, algo, maxtext_utils.get_mesh_from_config(cfg))

    self.assertEqual(_layout(micro_batch), _layout(_assembled_micro_batch(cfg, algo)))

  def test_rl_micro_batch_needs_prompt_and_completion(self):
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))
    for prompt_length in (0, 32):
      with self.subTest(prompt_length=prompt_length):
        cfg = _grpo_config(f"max_prefill_predict_length={prompt_length}")
        with self.assertRaisesRegex(
            ValueError,
            r"needs 0 < max_prefill_predict_length < max_target_length",
        ):
          grpo_engine_compile.get_shaped_rl_micro_batch(cfg, algo, maxtext_utils.get_mesh_from_config(cfg))

  def test_micro_batch_is_padded_and_unrouted_by_default(self):
    """Packing and router replay are off unless asked for, which leaves the micro-batch above."""
    cfg = _grpo_config()
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))

    micro_batch = grpo_engine_compile.get_shaped_rl_micro_batch(cfg, algo, maxtext_utils.get_mesh_from_config(cfg))

    self.assertEqual(
        (
            cfg.max_seq_token_per_tpu,
            cfg.max_segments_per_packed_row,
            cfg.router_replay,
        ),
        (0, 0, False),
    )
    self.assertEqual(
        (
            micro_batch.segment_ids,
            micro_batch.num_segments,
            micro_batch.routed_experts,
        ),
        (None, None, None),
    )


# A 2 x 2 fsdp x expert mesh, over which a packed micro-batch has four rows. Abstract: only its axis sizes are read.
_FSDP_BY_EXPERT = AbstractMesh((2, 2), ("fsdp", "expert"))


class GrpoPackingCompileTest(parameterized.TestCase):
  """`max_seq_token_per_tpu`: the micro-batch Tunix's `SequencePackedBatchAssembler` builds."""

  @parameterized.named_parameters(
      # Two 32-token rollouts fill a 64-token row, under the cap of three. `GRPOConfig`'s defaults, so the
      # old policy's and the reference model's log-probabilities are packed too.
      ("tokens_bind", 64, 3, 2, {}),
      # Three reach the cap with 32 tokens of a 128-token row to spare, which stay padding.
      ("segments_bind", 128, 3, 3, {"beta": 0.0}),
      # One to a row, and uncapped, so Tunix allows as many sequences as a row has tokens.
      ("one_per_row_uncapped", 32, 0, 1, {"use_rollout_logps": False}),
      # Three fill a 96-token row exactly: the prompt counts towards a rollout's length, not just the completion.
      ("three_fill_a_row", 96, 0, 3, {}),
      # Groups of 16, more than the eight rollouts the four rows hold: the group count rounds up.
      ("groups_larger_than_the_micro_batch", 64, 3, 2, {"num_generations": 16}),
  )
  def test_packed_micro_batch_matches_tunix(self, tokens, segments, per_row, grpo_config):
    """The packed micro-batch is laid out as Tunix's assembler lays out the same rollouts.

    The assembler here is built with the numbers expected -- four rows of
    `tokens` -- rather than by
    `create_batch_assembler`, so a micro-batch sized or packed any other way
    fails. Four rows, for the 2 x 2
    mesh; the two sequences `per_device_batch_size` gives do not decide it.
    """
    cfg = _grpo_config(
        f"max_seq_token_per_tpu={tokens}",
        f"max_segments_per_packed_row={segments}",
    )
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0, **grpo_config))
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=4,
        num_generations=algo.num_generations,
        mini_batch_size=1,
        max_packed_len=tokens,
        pad_id=grpo_engine_compile.GRPO_PAD_ID,
        max_segments_per_packed_row=segments or None,
    )

    micro_batch = grpo_engine_compile.get_rl_micro_batch(cfg, algo, _FSDP_BY_EXPERT)

    _assert_same_layout(
        self,
        micro_batch,
        _orchestrator_micro_batch(cfg, algo, assembler, num_rollouts=4 * per_row),
    )
    # The layout, spelled out: in every row, `per_row` sequences of 8 prompt and 24 scored completion tokens,
    # numbered from 1, each with its own positions, then padding, which is segment 0.
    padding = [0] * (tokens - 32 * per_row)
    row = {
        "segment_ids": (np.repeat(np.arange(1, per_row + 1), 32).tolist() + padding),
        "segment_positions": list(range(32)) * per_row + padding,
        "completion_mask": ([0.0] * 8 + [1.0] * 24) * per_row + padding,
    }
    for name, expected in row.items():
      self.assertEqual(getattr(micro_batch, name).tolist(), [expected] * 4, name)
    self.assertEqual(micro_batch.prompt_ids.shape, (4, 0))
    self.assertEqual(micro_batch.num_segments, (segments or tokens) + 1)
    # And the shapes the kernels are compiled for.
    self.assertEqual(
        _layout(grpo_engine_compile.get_shaped_rl_micro_batch(cfg, algo, _FSDP_BY_EXPERT)),
        _layout(micro_batch),
    )

  @parameterized.named_parameters(
      ("fsdp_by_expert", (2, 2), ("fsdp", "expert"), 4),
      ("data_by_fsdp_transpose", (3, 2), ("data", "fsdp_transpose"), 6),
      # Tunix does not count axes that do not split the batch.
      (
          "context_and_tensor_not_counted",
          (3, 2, 2),
          ("fsdp", "context", "tensor"),
          3,
      ),
  )
  def test_packed_rows_follow_the_mesh(self, sizes, names, rows):
    """A packed micro-batch has a row per device of the axes the batch is split over, not a set number of sequences."""
    cfg = _grpo_config("max_seq_token_per_tpu=32")
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))

    micro_batch = grpo_engine_compile.get_shaped_rl_micro_batch(cfg, algo, AbstractMesh(sizes, names))

    self.assertEqual(micro_batch.completion_ids.shape, (rows, 32))
    self.assertNotEqual(rows, cfg.micro_batch_size_to_train_on)

  def test_main_compiles_packed_micro_batch(self):
    """The compiled program reads the packed micro-batch and aggregates the loss per packed sequence.

    One row, as the mesh here is one device, holding two 32-token sequences; six
    segments, so `num_segments`
    is 7, a count no other dimension of this model has.
    """
    packing_overrides = (
        "max_seq_token_per_tpu=64",
        "max_segments_per_packed_row=6",
    )
    output, engine, micro_batch, compiled = _run_main_and_capture(self, *_grpo_overrides(*packing_overrides))

    self.assertEqual(
        (micro_batch.completion_ids.shape, micro_batch.num_segments),
        ((1, 64), 7),
    )
    batch = compiled["fwd_bwd"].args_info[0][2]["train_example"]
    self.assertEqual(_layout(batch), _layout(micro_batch))
    self.assertEqual(batch.num_segments, 7)
    # Every field is read; the prompt part is empty.
    state = engine.train_state_avals()
    rows = _by_kernel(memory_report.memory_report(compiled, state))
    self.assertEqual(
        rows["fwd_bwd"].argument,
        _subtree_bytes(state, "model") + _tree_bytes(micro_batch),
    )
    # Tunix's loss sums per segment, into `[rows, num_segments]`, only for a packed micro-batch.
    _, _, _, padded = _run_main_and_capture(self, *_GRPO_OVERRIDES)
    per_segment = re.compile(r"f32\[1,7\]")
    self.assertRegex(compiled["fwd_bwd"].as_text(), per_segment)
    self.assertNotRegex(padded["fwd_bwd"].as_text(), per_segment)
    self.assertIn(
        "Compiling Tunix's GRPO loss, for packed micro-batches of 1 x 64"
        " tokens, each row holding up to 6 sequences of up to 8 prompt and 24"
        " completion tokens.",
        output.splitlines(),
    )

  def test_micro_batch_follows_the_compiled_mesh(self):
    """The packed micro-batch `main` compiles for has a row per device of the mesh it compiles for, not of this host."""
    cfg = _grpo_config("max_seq_token_per_tpu=32")

    with mock.patch.object(maxtext_engine_compile, "AbstractMaxTextEngine"):
      _, micro_batch = maxtext_engine_compile._engine_and_micro_batch(cfg, _FSDP_BY_EXPERT)

    self.assertEqual(micro_batch.completion_ids.shape, (4, 32))

  def test_rejects_a_row_shorter_than_a_rollout(self):
    """Tunix's own check: 31 tokens cannot hold a 32-token rollout."""
    cfg = _grpo_config("max_seq_token_per_tpu=31")
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))

    with self.assertRaisesRegex(
        ValueError,
        r"max_seq_token_per_tpu=31 is smaller than the longest possible" r" sequence",
    ):
      grpo_engine_compile.get_rl_micro_batch(cfg, algo, _FSDP_BY_EXPERT)

  def test_rejects_options_that_would_be_ignored(self):
    for option, value in (
        ("grpo_config", "{beta: 0.0}"),
        ("logps_chunk_size", 4),
        ("max_seq_token_per_tpu", 64),
        ("router_replay", True),
    ):
      with self.subTest(option=option):
        with self.assertRaisesRegex(ValueError, rf"so {option} would be ignored"):
          _config(f"{option}={value}", data_parallelism=1)
    with self.assertRaisesRegex(ValueError, r"so it needs max_seq_token_per_tpu"):
      _grpo_config("max_segments_per_packed_row=4")
    with self.assertRaisesRegex(
        ValueError,
        r"max_seq_token_per_tpu\s+Input should be greater than" r" or equal to 0",
    ):
      _grpo_config("max_seq_token_per_tpu=-1")


class GrpoRouterReplayCompileTest(parameterized.TestCase):
  """`router_replay`: the rollouts' MoE routing, wherever Tunix's batch assembler carries it."""

  def test_padded_micro_batch_carries_routed_experts(self):
    """The routing is laid out as `PaddedBatchAssembler` lays it out, in the compact dtype the trainer receives."""
    cfg = _grpo_config(*_QWEN35_OVERRIDES, "router_replay=True")
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=8,
        max_response_length=24,
        pad_id=grpo_engine_compile.GRPO_PAD_ID,
        num_generations=algo.num_generations,
        mini_batch_size=1,
    )

    micro_batch = grpo_engine_compile.get_rl_micro_batch(cfg, algo, maxtext_utils.get_mesh_from_config(cfg))

    _assert_same_layout(
        self,
        micro_batch,
        _orchestrator_micro_batch(cfg, algo, assembler, num_rollouts=2, routed=True),
    )
    # A row per sequence, a slot per prompt and completion token, and top 2 of 4 experts in each of 4 layers.
    self.assertEqual(
        (micro_batch.routed_experts.shape, micro_batch.routed_experts.dtype),
        ((2, 32, 4, 2), np.uint8),
    )

  def test_packed_micro_batch_routing_matches_tunix(self):
    """The packed micro-batch carries `routed_experts` exactly where `SequencePackedBatchAssembler` does."""
    cfg = _grpo_config(
        *_QWEN35_OVERRIDES,
        "router_replay=True",
        "max_seq_token_per_tpu=64",
    )
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=4,
        num_generations=algo.num_generations,
        mini_batch_size=1,
        max_packed_len=64,
        pad_id=grpo_engine_compile.GRPO_PAD_ID,
    )
    expected = _orchestrator_micro_batch(cfg, algo, assembler, num_rollouts=8, routed=True)

    micro_batch = grpo_engine_compile.get_rl_micro_batch(cfg, algo, _FSDP_BY_EXPERT)

    _assert_same_layout(self, micro_batch, expected)

  def test_main_compiles_router_replay(self):
    """The compiled program reads the routing: it takes exactly the routing's bytes more than without it."""
    _, _, _, unrouted = _run_main_and_capture(self, *_grpo_overrides(*_QWEN35_OVERRIDES))
    output, _, micro_batch, compiled = _run_main_and_capture(
        self,
        *_grpo_overrides(*_QWEN35_OVERRIDES, "router_replay=True"),
    )

    routed_experts = compiled["fwd_bwd"].args_info[0][2]["train_example"].routed_experts
    self.assertEqual((routed_experts.shape, routed_experts.dtype), ((2, 32, 4, 2), jnp.uint8))
    # `jax.jit` drops the arguments a program never reads.
    argument, _ = _fwd_bwd_memory(compiled)
    self.assertEqual(
        argument - _fwd_bwd_memory(unrouted)[0],
        _tree_bytes(micro_batch.routed_experts),
    )
    self.assertIn(
        "The trainer replays the rollouts' routing: each token's 2 experts in" " each of 4 layers.",
        output.splitlines(),
    )

  def test_main_compiles_packed_micro_batch_without_replay(self):
    """When the packed micro-batch has no routing, router replay leaves the program as it is, and the report says why."""
    packed = _grpo_overrides(*_QWEN35_OVERRIDES, "max_seq_token_per_tpu=64")
    _, _, _, unrouted = _run_main_and_capture(self, *packed)
    output, _, micro_batch, compiled = _run_main_and_capture(self, *packed, "router_replay=True")

    if micro_batch.routed_experts is not None:
      self.skipTest("Tunix's packed assembler carries routed_experts;" " test_main_compiles_router_replay covers that.")
    self.assertEqual(_fwd_bwd_memory(compiled), _fwd_bwd_memory(unrouted))
    self.assertIn(
        "router_replay is set, but Tunix's batch assembler drops"
        " routed_experts from this micro-batch, so the compiled trainer routes"
        " every token itself.",
        output.splitlines(),
    )

  def test_rejects_a_model_without_routed_experts(self):
    cfg = _grpo_config("router_replay=True")
    algo = algorithm_adapter.GRPOAdapter(_grpo_config_obj(temperature=1.0))

    with self.assertRaisesRegex(ValueError, r"num_experts=1 leaves this model without routed experts"):
      grpo_engine_compile.get_rl_micro_batch(cfg, algo, _FSDP_BY_EXPERT)


class Zero1MemoryReportTest(absltest.TestCase):
  """Zero-1 shards the optimizer at compile time, so the report must read the state after it."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def test_report_counts_sharded_optimizer(self):
    cfg = _config("shard_optimizer_over_data=True")
    engine = maxtext_engine_compile.AbstractMaxTextEngine(cfg, maxtext_utils.get_mesh_from_config(cfg))
    # Counted before compiling: compiling re-places this tree in place, so counting it afterwards
    # would read the sharded layout.
    unsharded = _subtree_bytes(engine._read_state_pure(), "optimizer")
    compiled = engine.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
    state = engine.train_state_avals()
    rows = _by_kernel(memory_report.memory_report(compiled, state))

    self.assertIsNotNone(engine._zero1_params_shardings, "Zero-1 never engaged")
    model = _subtree_bytes(state, "model")
    optimizer = rows["fwd_bwd"].device_not_passed
    self.assertEqual(rows["update"].alias, model + optimizer)
    self.assertEqual(rows["fwd_bwd_accum"].device_not_passed, optimizer)
    # The pre-compile, unsharded optimizer is ~4x larger and does not match XLA's count.
    self.assertNotEqual(rows["update"].alias, model + unsharded)
    self.assertGreater(unsharded, 3 * optimizer)


_SUITE = (EngineAotParityTest, AbstractMaxTextEngineTest, Zero1MemoryReportTest)


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
