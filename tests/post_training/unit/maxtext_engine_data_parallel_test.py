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

"""The engine's two data-axis optimizations: the deferred all-reduce, and Zero-1.

Under gradient accumulation the engine used to pay one cross-replica all-reduce of the
whole parameter tree per *micro*-batch. Tagging the differentiated parameters `reduced`
over the data axis makes their cotangents `unreduced`, so the accumulation stays
replica-local and the all-reduce happens once, in `update()`. `shard_optimizer_over_data`
(Zero-1) then shards the parameter-shaped optimizer state over that same axis and runs the
update on those slices, gathering the new parameters back at the end.

They are tested together because they meet in one place -- the reshard at the top of
`_update_kernel`, which both discharges the `unreduced` tag and moves the gradients onto
the moments' layout -- and because they have to hold on one config rather than two.

Neither is a functional change: get either wrong and the model still trains, just without
the saving, which makes them exactly the kind that rots into a no-op without anything
failing. So the assertions are on the compiled HLO, where the collectives actually are, and
on where the arrays sit; and each one is mirrored by the same probe run with the feature
off, so a probe that has stopped finding anything fails rather than passes.

Everything runs in a subprocess. A data-parallel mesh needs more than one device, and the
CPU backend reads `--xla_force_host_platform_device_count` only at backend initialization --
which is already past by the time pytest imports this file, because sibling modules under
`tests/post_training` touch JAX while being collected, and one of them *assigns*
`XLA_FLAGS=--xla_force_host_platform_device_count=1`. Setting the flag at import time here
is therefore a no-op that leaves `jax.device_count() == 1` and skips the whole file green.
`test_engine_data_parallelism_on_a_four_device_cpu_mesh` re-execs the module with the flag
appended instead, and refuses to pass unless the child reports tests actually run.
"""

import contextlib
import dataclasses
import functools
import os
import re
import subprocess
import sys
import tempfile
import unittest

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_DATA = maxtext_engine._DATA_AXIS  # pylint: disable=protected-access
_SENTINEL = "MAXTEXT_ENGINE_DATA_PARALLEL_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_engine_data_parallelism_on_a_four_device_cpu_mesh():
  """The only test pytest collects here; the classes below run inside the child process.

  See the module docstring for why re-exec is the only option: the device count is fixed
  when the CPU backend initializes, and something else has already initialized it. The flag
  is *appended* rather than defaulted so it wins over whatever a sibling module set --
  XLA takes the last occurrence of a repeated flag.
  """
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  # The child imports `tests.utils.test_helpers`, which pytest puts on the path for us and
  # a bare interpreter does not.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # Without this the whole suite could go quietly empty again: an exit status of 0 is also
  # what a run that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


# ---------------------------------------------------------------------------------------
# The rig.
# ---------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _config(**overrides) -> pyconfig.HyperParameters:
  """A tiny real model on an explicit, purely data-parallel mesh.

  Small enough to compile in seconds, but a *real* MaxText decoder rather than a stub: the
  reduced/unreduced tags have to survive every layer that touches a parameter, and it was a
  norm layer indexing `spec[...]` directly that broke first.

  Cached because `pyconfig.initialize` costs ~0.2s and the tests below want a handful of
  configs many times over. `HyperParameters` refuses `__setattr__`, so sharing one is safe.
  """
  argv = [
      "maxtext_engine_data_parallel_test.py",
      get_test_config_path("base.yml"),
      "model_name=default",
      "run_name=engine_data_parallel_test",
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
      # The deferral tag only goes on when "data" is the sole batch axis of size > 1.
      "shard_mode=explicit",
      f"ici_data_parallelism={_REQUIRED_DEVICES}",
      "ici_fsdp_parallelism=1",
      "ici_tensor_parallelism=1",
      "per_device_batch_size=1",
      # Tiny model. These must be the `base_*` names: emb_dim and mlp_dim are derived.
      "vocab_size=128",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "head_dim=16",
      "max_target_length=32",
      # adamw rather than sgd, which carries no parameter-shaped state at all: Zero-1 would
      # then have nothing to shard and every assertion about it would pass vacuously. A
      # constant schedule with no clipping, so `update()` is the optimizer and nothing else.
      "opt_type=adamw",
      "learning_rate=1e-2",
      "gradient_clipping_threshold=0.0",
      "warmup_steps_fraction=0.0",
      "learning_rate_final_fraction=1.0",
      "gradient_accumulation_steps=1",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


@functools.lru_cache(maxsize=None)
def _mesh(**overrides):
  """The mesh `_config(**overrides)` resolves to, cached alongside it."""
  return maxtext_utils.get_mesh_from_config(_config(**overrides))


def _rig(**overrides):
  """`(config, mesh)` for one set of overrides. Almost every test wants both."""
  return _config(**overrides), _mesh(**overrides)


def _batch(cfg: pyconfig.HyperParameters, seed: int) -> dict[str, np.ndarray]:
  """Minimal batch shaped for `maxtext.trainers.pre_train.train.loss_fn`.

  NumPy, not `jnp`: the compiled kernel takes its batch on `P("data", None)`, and a
  committed device array built here would arrive on `P()` and be rejected by `jax.jit`'s
  exact `in_shardings` match. Host arrays are placed by the kernel itself.
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


def _no_deferral():
  """Patches the deferral gate shut, leaving everything else about the engine identical."""
  return mock.patch.object(maxtext_engine, "_deferred_all_reduce_shardings", return_value=(None, None))


def _axes(spec) -> list[str]:
  """The mesh axes a `PartitionSpec` names.

  A `PartitionSpec` is a pytree *leaf*, so flattening one gives back the spec itself; its
  entries have to be opened first, and they nest -- a dimension sharded over two axes is a
  tuple. The same trap the guard in `add_data_to_sharding` was written with. Read through
  `.partitions` because iterating a spec that carries a reduced/unreduced tag raises.
  """
  return jax.tree.leaves(spec.partitions)


# Matches the result shape of an all-reduce in optimized HLO, covering both the fused
# `%x = f32[64]{0} all-reduce(...)` form and the tupled `ROOT %y = (f32[], f32[64]{0})
# all-reduce-start(...)` one. Same shape of pattern for all-gather.
_ALL_REDUCE = re.compile(r"=\s*(.+?)\s+all-reduce(?:-start|-done)?\(")
_ALL_GATHER = re.compile(r"=\s*(.+?)\s+all-gather(?:-start|-done)?\(")
# A shape with any dimension at all, i.e. not the `f32[]` of a scalar loss term.
_NON_SCALAR = re.compile(r"\[\s*\d")
# The dimensions inside one `f32[128,64]{1,0}`; a tupled result yields one match per element.
_SHAPE_DIMS = re.compile(r"\[([\d,]*)\]")


def _array_all_reduces(hlo: str) -> list[str]:
  """The result shapes of every all-reduce in `hlo` that moves more than a scalar.

  Scalars are ignored on purpose: the loss and its denominator are reduced across replicas
  every micro-batch and always will be. What the deferral is about is the parameter-sized
  traffic, which is four orders of magnitude larger even on the toy model here.
  """
  shapes = [m.group(1) for line in hlo.splitlines() if (m := _ALL_REDUCE.search(line))]
  return [s for s in shapes if _NON_SCALAR.search(s)]


def _gathered_elements(hlo: str) -> int:
  """Total result size of every all-gather in `hlo`, as a stand-in for gathered volume.

  Counting instructions would be brittle -- XLA fuses and splits them freely -- and the
  absolute number here means little, since an async gather's start and done both count. Only
  the difference against the same kernel compiled without Zero-1 is ever asserted on, and
  both sides of that are counted the same way.
  """
  total = 0
  for line in hlo.splitlines():
    if match := _ALL_GATHER.search(line):
      for dims in _SHAPE_DIMS.findall(match.group(1)):
        if dims:
          total += int(np.prod([int(dim) for dim in dims.split(",")]))
  return total


class _KernelHlo:
  """Captures the arguments the engine passes one jitted kernel, to re-lower it later.

  `jax.jit` keeps no handle on the executable it cached, so the only way to read a kernel's
  optimized HLO is to lower it again. Recording `ShapeDtypeStruct`s rather than the arrays
  themselves keeps that independent of donation -- `_compiled_update` donates its state, so
  by the time a test asks for the HLO those buffers are gone.

  Lowering is lazy and memoized: it is a full XLA compile, and only a third of the tests
  below want HLO at all. It needs the mesh set, because the fwd/bwd kernels apply the
  `reduced` tag while tracing, and by then the run that recorded the avals is long over.
  """

  def __init__(self, engine, attr: str, mesh):
    self._jitted = getattr(engine, attr)
    self._mesh = mesh
    self._avals = None
    self._text = None
    setattr(engine, attr, self._spy)

  def _spy(self, *args):
    if self._avals is None:
      # `sharding=None` for the host-side batch arrays, which have none; `jax.jit` places
      # those from its own `in_shardings` either way.
      self._avals = jax.tree.map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
          args,
      )
    return self._jitted(*args)

  def text(self) -> str:
    if self._avals is None:
      raise AssertionError("kernel was never called, so there is no HLO to read")
    if self._text is None:
      with jax.set_mesh(self._mesh):
        self._text = self._jitted.lower(*self._avals).compile().as_text()
    return self._text


@dataclasses.dataclass(frozen=True)
class _Recipe:
  """What distinguishes one shared run from another. Hashable, so it can key the run cache."""

  zero1: bool = False
  defer: bool = True
  opt_type: str = "adamw"
  micro_batches: int = 3
  steps: int = 2

  def overrides(self) -> dict[str, Any]:
    """The config overrides this recipe implies. The deferral is patched, not configured."""
    return {"shard_optimizer_over_data": self.zero1, "opt_type": self.opt_type}


_KERNELS = {"first": "_compiled_fwd_bwd", "accum": "_compiled_fwd_bwd_accum", "update": "_compiled_update"}


class _Run:
  """One finished engine run: `steps` optimizer steps of `micro_batches` each.

  Building an engine and compiling its three kernels is essentially the entire runtime of
  this file, and each recipe below is wanted by three or four tests -- hence `_run`'s cache.
  Treat an instance as read-only: the engine is left wherever its last `update()` put it,
  and stepping it again would move it under every other test holding the same recipe.
  """

  def __init__(self, recipe: _Recipe):
    self.recipe = recipe
    self.config = _config(**recipe.overrides())
    self.mesh = _mesh(**recipe.overrides())
    with contextlib.ExitStack() as stack:
      if not recipe.defer:
        stack.enter_context(_no_deferral())
      stack.enter_context(jax.set_mesh(self.mesh))
      self.engine = maxtext_engine.MaxTextTrainingEngine(self.config, mesh=self.mesh)
      self.engine.compile(_batch(self.config, 0))
      self._probes = {name: _KernelHlo(self.engine, attr, self.mesh) for name, attr in _KERNELS.items()}
      norms = []
      for step in range(recipe.steps):
        for micro in range(recipe.micro_batches):
          self.engine.fwd_bwd(_batch(self.config, step * recipe.micro_batches + micro))
        self.engine.update()
        # One norm per update, recorded as a length-1 array rather than a scalar.
        recorded = self.engine.get_metrics(clear_cache=True).scalar_metrics["gradient_norm"]
        norms.append(float(np.asarray(recorded).reshape(-1)[0]))
      self.grad_norms = tuple(norms)

  def hlo(self, kernel: str) -> str:
    return self._probes[kernel].text()

  @property
  def defer_engaged(self) -> bool:
    return self.engine._plain_grad_shardings is not None  # pylint: disable=protected-access

  @property
  def zero1_engaged(self) -> bool:
    return self.engine._zero1_params_shardings is not None  # pylint: disable=protected-access

  @property
  def moments(self) -> dict[str, jax.Array]:
    return _moments(self.engine)

  @property
  def params(self) -> dict[str, jax.Array]:
    return _params(self.engine)


def _moments(engine) -> dict[str, jax.Array]:
  """`{path: array}` for every parameter-shaped optimizer moment in the engine's state."""
  _, state_pure = nnx.split(engine.state)
  return {
      jax.tree_util.keystr(path): leaf
      for path, leaf in jax.tree_util.tree_leaves_with_path(state_pure)
      if "['mu']" in jax.tree_util.keystr(path) or "['nu']" in jax.tree_util.keystr(path)
  }


def _params(engine) -> dict[str, jax.Array]:
  """`{path: array}` for the model's parameters."""
  return {
      jax.tree_util.keystr(path): leaf
      for path, leaf in jax.tree.flatten_with_path(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))[0]
  }


@functools.lru_cache(maxsize=None)
def _run(recipe: _Recipe) -> _Run:
  return _Run(recipe)


# The shared runs. Every combination of the two features, on one model and one mesh, so that
# "with the feature off" always means the same thing and the comparisons are exact.
_BASE = _Recipe(zero1=False, defer=True)
_NO_DEFER = _Recipe(zero1=False, defer=False)
_ZERO1 = _Recipe(zero1=True, defer=True)
_ZERO1_ONLY = _Recipe(zero1=True, defer=False)
# And the same pair again under sgd, for the one comparison adamw is the wrong lens for; see
# `EquivalenceTest.test_deferring_does_not_change_the_weights`.
_SGD = _Recipe(opt_type="sgd", defer=True)
_SGD_NO_DEFER = _Recipe(opt_type="sgd", defer=False)


def _assert_close(test, got: dict[str, jax.Array], want: dict[str, jax.Array], what: str):
  """Compares two parameter dicts leaf by leaf, at the tolerance an accelerator needs."""
  test.assertEqual(sorted(want), sorted(got))
  test.assertNotEmpty(want, f"no parameters to compare, so {what} is not checked")
  for path, expected in want.items():
    np.testing.assert_allclose(
        np.asarray(got[path]), np.asarray(expected), rtol=1e-6, atol=1e-6, err_msg=f"{what}: parameter {path}"
    )


# ---------------------------------------------------------------------------------------
# The gates: pure functions of (config, mesh), decided before an engine exists.
# ---------------------------------------------------------------------------------------

# Zero-1's answer for a row the config layer refuses to build at all, so that the two gates
# can still be read off one table.
_CONFIG_REJECTS = "rejected by the config validator"

# One situation per row, checked against both gates -- which is the point of the table. The
# two mostly agree, and where they diverge they say something: a second sharded axis makes
# the *tag* illegal (JAX requires the unreduced set to be exactly the contracted axes) but
# leaves Zero-1 sound, since it only ever reshards -- yet `fsdp` specifically is stopped one
# layer earlier, by a config validator, because sharding the optimizer over `data` is what
# FSDP already does over `fsdp`.
_SITUATIONS = (
    {
        "testcase_name": "purely_data_parallel",
        "overrides": {},
        "auto_axis_mesh": False,
        "no_mesh": False,
        "deferral": True,
        "zero1_declined_for": None,
    },
    {
        "testcase_name": "auto_shard_mode",
        "overrides": {"shard_mode": "auto"},
        "auto_axis_mesh": False,
        "no_mesh": False,
        "deferral": False,
        "zero1_declined_for": "explicit",
    },
    # A caller can hand the engine a mesh built by bare `jax.sharding.Mesh(...)` whatever
    # `shard_mode` says. Reduced/unreduced specs and real reshards both need Explicit axes,
    # so the mesh wins over the config.
    {
        "testcase_name": "auto_axis_mesh_in_explicit_mode",
        "overrides": {},
        "auto_axis_mesh": True,
        "no_mesh": False,
        "deferral": False,
        "zero1_declined_for": "Explicit",
    },
    {
        "testcase_name": "no_data_replicas",
        "overrides": {"ici_data_parallelism": 1, "ici_tensor_parallelism": _REQUIRED_DEVICES},
        "auto_axis_mesh": False,
        "no_mesh": False,
        "deferral": False,
        "zero1_declined_for": _DATA,
    },
    # "unreduced axes should be equal to the contracting specs. Got unreduced
    # axes=frozenset({'data'}) and contracting spec=(('data', 'fsdp'), None)". Widening the
    # tag to `fsdp` is not available either: the parameters are sharded over it.
    {
        "testcase_name": "fsdp_also_shards_the_batch",
        "overrides": {"ici_data_parallelism": 2, "ici_fsdp_parallelism": 2},
        "auto_axis_mesh": False,
        "no_mesh": False,
        "deferral": False,
        "zero1_declined_for": _CONFIG_REJECTS,
    },
    # The batch is on `data` alone here and the backward pass is still rejected: `tensor`
    # breaks the tag through the *feature* dimension of the same activation, which a
    # batch-axis check cannot see -- "... and contracting spec=('data', None, 'tensor')".
    # Measured on 4x v6e with qwen3-0.6b at dp2 x tp2, where it crashed the first micro-batch.
    {
        "testcase_name": "tensor_shards_the_features",
        "overrides": {"ici_data_parallelism": 2, "ici_tensor_parallelism": 2},
        "auto_axis_mesh": False,
        "no_mesh": False,
        "deferral": False,
        "zero1_declined_for": None,
    },
    {
        "testcase_name": "no_mesh_at_all",
        "overrides": {},
        "auto_axis_mesh": False,
        "no_mesh": True,
        "deferral": False,
        "zero1_declined_for": "mesh",
    },
)


class GateTest(parameterized.TestCase):
  """Both gates must decline widely: a tag or a reshard in the wrong place is a hard error."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def _situation(self, overrides, auto_axis_mesh, no_mesh, **extra):
    """Returns `(config, mesh)` for one row of the table above."""
    cfg, mesh = _rig(**overrides, **extra)
    if auto_axis_mesh:
      mesh = jax.sharding.Mesh(mesh.devices, mesh.axis_names)
    return cfg, None if no_mesh else mesh

  @parameterized.named_parameters(*_SITUATIONS)
  def test_the_deferral_gate(self, overrides, auto_axis_mesh, no_mesh, deferral, zero1_declined_for):
    del zero1_declined_for  # read by the companion test below.
    cfg, mesh = self._situation(overrides, auto_axis_mesh, no_mesh)
    # `_deferred_all_reduce_shardings` needs somewhere to put the tag; the mesh it is placed
    # on is irrelevant to the decision, which is why the no-mesh row can still pass one in.
    shardings = {"w": jax.sharding.NamedSharding(_mesh(), jax.sharding.PartitionSpec())}

    reduced, unreduced = maxtext_engine._deferred_all_reduce_shardings(cfg, mesh, shardings)  # pylint: disable=protected-access

    if deferral:
      self.assertEqual(reduced["w"].spec.reduced, {_DATA})
      self.assertEqual(unreduced["w"].spec.unreduced, {_DATA})
    else:
      self.assertEqual((None, None), (reduced, unreduced))

  @parameterized.named_parameters(*_SITUATIONS)
  def test_the_zero1_gate(self, overrides, auto_axis_mesh, no_mesh, deferral, zero1_declined_for):
    del deferral  # read by the companion test above.
    if zero1_declined_for == _CONFIG_REJECTS:
      # Never reaches the gate: `MaxTextConfig` refuses the combination while validating.
      with self.assertRaisesRegex(ValueError, "cannot be combined with FSDP"):
        _config(**overrides, shard_optimizer_over_data=True)
      return

    # The flag on, so that what is being read is the situation and not the flag.
    cfg, mesh = self._situation(overrides, auto_axis_mesh, no_mesh, shard_optimizer_over_data=True)

    declined = maxtext_engine._zero1_active(cfg, mesh)  # pylint: disable=protected-access

    if zero1_declined_for is None:
      self.assertIsNone(declined)
    else:
      self.assertIn(zero1_declined_for, declined)

  def test_the_zero1_gate_declines_when_the_flag_is_off(self):
    """The flag being off is a reason like any other, so one call answers "should this run"."""
    self.assertIsNotNone(maxtext_engine._zero1_active(_config(), _mesh()))  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("over_data_alone", jax.sharding.PartitionSpec(_DATA)),
      # A dimension sharded over two axes at once is a *nested* entry. A per-dimension check
      # reads `('data', 'fsdp')` as one unrecognised name, tags a parameter that is already
      # sharded over `data`, and dies on "partitions cannot overlap with reduced axes".
      ("over_data_and_fsdp_together", jax.sharding.PartitionSpec((_DATA, "fsdp"))),
  )
  def test_a_parameter_already_sharded_over_data_is_left_alone(self, spec):
    """`reduced` and a shard over the same axis are contradictory, and JAX says so."""
    mesh = _mesh()
    shardings = {"w": jax.sharding.NamedSharding(mesh, spec)}

    reduced, unreduced = maxtext_engine._deferred_all_reduce_shardings(_config(), mesh, shardings)  # pylint: disable=protected-access

    self.assertEmpty(reduced["w"].spec.reduced)
    self.assertEmpty(unreduced["w"].spec.unreduced)


class LeafPlacementTest(parameterized.TestCase):
  """`_zero1_sharding` places one leaf. Everything Zero-1 moves goes through it."""

  __test__ = False

  def _place(self, shape, spec=None):
    mesh = _mesh(shard_optimizer_over_data=True)
    replicated = jax.sharding.PartitionSpec(*(None,) * len(shape))
    base = jax.sharding.NamedSharding(mesh, replicated if spec is None else spec)
    aval = jax.ShapeDtypeStruct(shape, jax.numpy.float32)
    return maxtext_engine._zero1_sharding(mesh, aval, base)  # pylint: disable=protected-access

  def test_adds_the_data_axis_to_the_first_dimension_that_divides(self):
    self.assertEqual(self._place((128, 64)).spec, jax.sharding.PartitionSpec(_DATA, None))

  def test_skips_a_dimension_the_data_axis_does_not_divide(self):
    self.assertEqual(self._place((3, 64)).spec, jax.sharding.PartitionSpec(None, _DATA))

  def test_leaves_a_scalar_alone(self):
    """`adamw`'s step `count`, and every rng counter beside it. Nothing to slice."""
    self.assertIsNone(self._place(()))

  def test_leaves_a_shape_no_dimension_of_which_divides_alone(self):
    self.assertIsNone(self._place((3, 5)))

  def test_leaves_a_leaf_already_sharded_over_data_alone(self):
    self.assertIsNone(self._place((128, 64), jax.sharding.PartitionSpec(_DATA, None)))


# ---------------------------------------------------------------------------------------
# End to end on a real decoder.
# ---------------------------------------------------------------------------------------


class EngagementTest(parameterized.TestCase):
  """Which features actually turned on in each shared run.

  This is the vacuity guard for everything below it: the whole file compares a run with a
  feature against a run without it, and both halves of every one of those comparisons are
  pinned here. If the deferral silently stopped engaging, these fail and the HLO tests --
  which would then be comparing two identical baselines -- do not quietly agree.
  """

  __test__ = False

  @parameterized.named_parameters(
      ("deferral_only", _BASE, True, False),
      ("neither", _NO_DEFER, False, False),
      ("both", _ZERO1, True, True),
      # The two are independent: Zero-1 has to work on its own, which is what a caller on a
      # mesh the deferral declines on gets.
      ("zero1_without_the_deferral", _ZERO1_ONLY, False, True),
  )
  def test_the_engine_engaged_what_it_was_asked_for(self, recipe, defer, zero1):
    run = _run(recipe)

    self.assertEqual(run.defer_engaged, defer, "the deferred all-reduce is not in the state it should be")
    self.assertEqual(run.zero1_engaged, zero1, "Zero-1 is not in the state it should be")


class CollectivePlacementTest(absltest.TestCase):
  """Where the collectives land. The performance claim of both features, in one place."""

  __test__ = False

  def test_micro_batch_kernels_move_only_scalars_across_replicas(self):
    """The whole point: no parameter-sized all-reduce per micro-batch, only per step."""
    run = _run(_BASE)
    first = _array_all_reduces(run.hlo("first"))
    accum = _array_all_reduces(run.hlo("accum"))
    update = _array_all_reduces(run.hlo("update"))

    self.assertEmpty(first, f"first micro-batch still all-reduces arrays: {first}")
    self.assertEmpty(accum, f"accumulating micro-batches still all-reduce arrays: {accum}")
    # Vacuity guard from the other side: the traffic did not vanish, it moved.
    self.assertNotEmpty(update, "no array all-reduce in update() either -- the gradients are never reduced")

  def test_without_the_deferral_every_micro_batch_pays(self):
    """Proves the probe above can fail. Same model, same probe, tag withheld."""
    run = _run(_NO_DEFER)
    first = _array_all_reduces(run.hlo("first"))
    accum = _array_all_reduces(run.hlo("accum"))
    update = _array_all_reduces(run.hlo("update"))

    self.assertNotEmpty(first, "baseline should all-reduce the gradients in the first micro-batch")
    self.assertNotEmpty(accum, "baseline should all-reduce the gradients in every micro-batch")
    self.assertEmpty(update, f"baseline should have nothing left to reduce in update(): {update}")

  def test_zero1_costs_one_all_gather_in_update_and_nothing_per_micro_batch(self):
    """Where the traffic Zero-1 adds is, and where it must not be.

    Each replica updates its own slice, so the new parameters have to be gathered before the
    next forward pass -- once per optimizer step, in `update()`. If that gather ever appears
    in a micro-batch kernel instead, Zero-1 has become a per-micro-batch cost.
    """
    zero1, baseline = _run(_ZERO1), _run(_BASE)

    added = {k: _gathered_elements(zero1.hlo(k)) - _gathered_elements(baseline.hlo(k)) for k in _KERNELS}
    self.assertGreater(added["update"], 0, "update() gathers nothing, so the parameters were never sharded")
    self.assertEqual(added["first"], 0, "Zero-1 added an all-gather to the first micro-batch")
    self.assertEqual(added["accum"], 0, "Zero-1 added an all-gather to the accumulating micro-batches")

  def test_zero1_composes_with_the_deferred_all_reduce(self):
    """The pair is the point: one reduction per step, on 1/N of the optimizer.

    Zero-1 reshards the gradients onto the moments' layout inside `update()`, which is the
    same reshard that discharges the deferral's `unreduced` tag. So turning it on must not
    put parameter-sized traffic back into the micro-batches.
    """
    run = _run(_ZERO1)

    self.assertEmpty(_array_all_reduces(run.hlo("first")))
    self.assertEmpty(_array_all_reduces(run.hlo("accum")))
    self.assertNotEmpty(_array_all_reduces(run.hlo("update")), "the gradients are never reduced at all")


class StatePlacementTest(absltest.TestCase):
  """Where the arrays sit once the run is over. Zero-1's saving, measured directly."""

  __test__ = False

  def test_the_optimizer_moments_are_sharded_over_the_data_axis(self):
    """The saving itself: each replica holds and updates 1/N of every moment."""
    moments = _run(_ZERO1).moments

    self.assertNotEmpty(moments, "adamw kept no parameter-shaped state, so there is nothing to shard")
    for path, leaf in moments.items():
      self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} is not sharded over {_DATA!r}")
      shard = leaf.addressable_shards[0].data.shape
      self.assertEqual(
          np.prod(shard) * _REQUIRED_DEVICES,
          np.prod(leaf.shape),
          f"{path} claims to be sharded but each device still holds {shard} of {leaf.shape}",
      )

  def test_without_the_flag_the_moments_stay_replicated(self):
    """Proves the probe above can fail. Same model, same optimizer, flag off."""
    moments = _run(_BASE).moments

    self.assertNotEmpty(moments, "no moments to check, so this guard proves nothing")
    for path, leaf in moments.items():
      self.assertNotIn(_DATA, _axes(leaf.sharding.spec), f"{path} is sharded with the flag off")

  def test_the_parameters_themselves_stay_replicated(self):
    """Zero-1, not Zero-2/3: only the optimizer state is stored sharded.

    The forward pass wants whole parameters and every kernel signature is unchanged, so the
    slicing lives entirely inside `update()`.
    """
    for path, leaf in _run(_ZERO1).params.items():
      self.assertNotIn(_DATA, _axes(leaf.sharding.spec), f"parameter {path} came back sharded")

  def test_zero1_shards_the_moments_without_the_deferral_too(self):
    """The two are independent, so Zero-1 must not be relying on the tag being there."""
    moments = _run(_ZERO1_ONLY).moments

    self.assertNotEmpty(moments, "no moments to check, so this proves nothing")
    for path, leaf in moments.items():
      self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} is not sharded over {_DATA!r}")


class EquivalenceTest(absltest.TestCase):
  """Neither feature is allowed to change a number, only where the arithmetic happens."""

  __test__ = False

  def test_deferring_does_not_change_the_weights(self):
    """Same sum, different association order.

    Deferring does not drop or duplicate a term -- it moves the cross-replica addition from
    before the micro-batch sum to after it. In exact arithmetic the two agree identically,
    and on this CPU mesh they do; in float32 on a real accelerator the reassociation shows up
    around 1e-6 relative, which is what the tolerance here allows for.

    Under sgd, unlike everything else in this file, because sgd applies the gradient as it
    is: what the comparison then sees is the gradient sum, which is the only thing deferring
    touches. adamw divides by `sqrt(nu) + eps`, and for a parameter whose gradient is near
    zero that turns a last-bit difference in the sum into a ~2e-5 relative difference in the
    weight -- real, reproducible, and not evidence of anything this test is about.
    """
    _assert_close(self, _run(_SGD).params, _run(_SGD_NO_DEFER).params, "deferring moved the weights")

  def test_zero1_does_not_change_the_weights(self):
    """Same optimizer arithmetic, run elementwise on disjoint slices instead of on all of it.

    Every operation `adamw` applies is elementwise in the parameter, so splitting the tensor
    across replicas changes nothing about the result -- on this CPU mesh, not even the last
    bit. The tolerance is for accelerators, where the gather is not exact.
    """
    _assert_close(self, _run(_ZERO1).params, _run(_BASE).params, "Zero-1 moved the weights")

  def test_the_gradient_norm_still_sees_the_whole_gradient(self):
    """The one part of `update()` that is not elementwise.

    `l2norm_pytree` sums squares over every element, and under Zero-1 those elements are
    spread across replicas. Under explicit sharding the cross-replica sum that restores the
    total is inserted by JAX rather than written here, so it is exactly the kind of thing a
    later refactor can drop: a replica-local sum would come out low by a factor of
    sqrt(N) -- 2x on this mesh -- and everything downstream of it, gradient clipping and
    spike detection, would silently stop biting. Comparing the norms is a sharper probe than
    comparing weights clipped with it, which only reacts once the threshold is crossed.
    """
    zero1, baseline = _run(_ZERO1).grad_norms, _run(_BASE).grad_norms

    self.assertNotEmpty(baseline, "no gradient norms were recorded, so this checks nothing")
    np.testing.assert_allclose(zero1, baseline, rtol=1e-6, err_msg="the norm under Zero-1 is not the global one")


class LifecycleTest(absltest.TestCase):
  """The two events that outlive a compilation: a checkpoint restore, and a recompile."""

  __test__ = False

  def test_a_mid_step_checkpoint_round_trip_resumes_where_it_left_off(self):
    """Saving mid-step must write the gradient total, not one replica's partial.

    Orbax cannot serialize an unreduced array at all -- `device_indices_map` is undefined for
    one -- so `save_checkpoint` reduces first, and `restore_checkpoint` puts the accumulator
    back on the layout `_compiled_fwd_bwd_accum` was built against. Restoring does not
    recompile, so the moments have to land back sharded too: `_compiled_update` was built
    against sharded ones and would die on an `in_shardings` mismatch, and if it somehow did
    not, the optimizer would quietly be replicated for the rest of the run.

    Both halves are checked on one interrupted run, and the proof that the value written was
    the right one is that finishing the step from the restored state lands on exactly the
    weights of the uninterrupted `_ZERO1` run.
    """
    # `enterContext` is the `with` pylint is asking for, scoped to the test rather than a block.
    output_dir = self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with
    cfg, mesh = _rig(
        shard_optimizer_over_data=True,
        enable_checkpointing=True,
        base_output_directory=output_dir,
        async_checkpointing=False,
        checkpoint_period=1,
    )
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      # One whole step first, so the moments are non-zero and actually carry information.
      for micro in range(3):
        engine.fwd_bwd(_batch(cfg, micro))
      engine.update()
      # Then interrupt the second step two micro-batches in, with no update() yet.
      engine.fwd_bwd(_batch(cfg, 3))
      engine.fwd_bwd(_batch(cfg, 4))
      self.assertTrue(
          jax.tree.leaves(engine._accumulated_grads)[0].sharding.spec.unreduced,  # pylint: disable=protected-access
          "the accumulator is not unreduced, so this is not exercising the reduce-on-save",
      )
      engine.save_checkpoint(metadata={"marker": 1}, force=True)
      engine._checkpoint_manager.wait_until_finished()  # pylint: disable=protected-access

      engine.restore_checkpoint()
      restored = _moments(engine)
      self.assertNotEmpty(restored, "no moments came back, so nothing below is checked")
      for path, leaf in restored.items():
        self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} came back replicated from the checkpoint")

      # Finish the interrupted step from the restored state.
      engine.fwd_bwd(_batch(cfg, 5))
      engine.update()
      resumed = _params(engine)

    _assert_close(self, resumed, _run(_ZERO1).params, "the resumed run diverged from the uninterrupted one")

  def test_a_recompile_leaves_a_live_accumulator_and_the_sharded_moments_alone(self):
    """A second batch shape re-enters `_compile_for_batch`, mid-step.

    Two things have to survive it. The moments are already on the Zero-1 layout, so
    re-placing them has to be a no-op -- adding the data axis a second time produces
    `P(('data', 'data'), ...)`, which `NamedSharding` rejects outright and which would take
    the whole engine down. And the accumulator is live and `unreduced`, so it has to be
    conformed onto whatever the new compilation expects rather than fed to it as-is.
    """
    cfg, mesh = _rig(shard_optimizer_over_data=True)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      engine.fwd_bwd(_batch(cfg, 0))
      # A shorter sequence: a different dynamic batch shape, so `fwd_bwd` recompiles -- with
      # the accumulator from the micro-batch above still pending.
      short = {name: value[:, : cfg.max_target_length // 2] for name, value in _batch(cfg, 1).items()}
      engine.fwd_bwd(short)
      engine.update()
      moments = _moments(engine)

    self.assertNotEmpty(moments, "no moments to check, so this proves nothing")
    for path, leaf in moments.items():
      self.assertEqual(_axes(leaf.sharding.spec).count(_DATA), 1, f"{path} was sharded over {_DATA!r} twice")


class DeclinedGracefullyTest(absltest.TestCase):
  """A gate that closes has to leave a working engine behind, and say so."""

  __test__ = False

  def test_a_tensor_parallel_mesh_still_trains(self):
    """This does not reproduce the crash the deferral gate exists to prevent.

    The toy model here shards plenty over `tensor` and traces clean on CPU anyway; what
    raised `ShardingTypeError` was qwen3-0.6b at dp2 x tp2 on 4x v6e, whose attention
    kernels contract over a tensor-sharded dimension the `dot_product` path does not. So the
    assertion that pins the fix is the gate one in `GateTest`, and this pins the other half:
    with the tag off, a tensor-parallel mesh completes a step and moves the weights.
    """
    overrides = {"ici_data_parallelism": 2, "ici_tensor_parallelism": 2}
    cfg, mesh = _rig(**overrides)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      before = [np.asarray(leaf) for leaf in _params(engine).values()]
      engine.fwd_bwd(_batch(cfg, 0))
      engine.fwd_bwd(_batch(cfg, 1))
      engine.update()
      after = list(_params(engine).values())

      self.assertIsNone(
          engine._plain_grad_shardings,  # pylint: disable=protected-access
          "the deferral engaged on a tensor-parallel mesh, which JAX rejects",
      )
    self.assertTrue(
        any(not np.array_equal(np.asarray(got), want) for got, want in zip(after, before)),
        "the step ran but no parameter moved, so nothing was trained",
    )

  def test_a_zero1_request_the_engine_cannot_honour_is_reported_once(self):
    """The failure mode this replaces was silence: the flag set, and nothing done about it."""
    overrides = {"shard_optimizer_over_data": True, "shard_mode": "auto"}
    cfg, mesh = _rig(**overrides)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      with self.assertLogs(level="WARNING") as logs:
        engine.compile(_batch(cfg, 0))
        engine.fwd_bwd(_batch(cfg, 0))
        engine.update()

    declined = [line for line in logs.output if "Zero-1" in line]
    self.assertLen(declined, 1, f"expected exactly one Zero-1 warning, got {declined}")
    self.assertIn("shard_mode=explicit", declined[0])
    self.assertIsNone(engine._zero1_params_shardings)  # pylint: disable=protected-access


_SUITE = (
    GateTest,
    LeafPlacementTest,
    EngagementTest,
    CollectivePlacementTest,
    StatePlacementTest,
    EquivalenceTest,
    LifecycleTest,
    DeclinedGracefullyTest,
)


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
