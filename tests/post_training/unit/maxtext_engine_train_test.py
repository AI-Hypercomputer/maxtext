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

"""End-to-end CPU tests of `MaxTextTrainingEngine`, driven the way its callers drive it.

Each suite trains a tiny real MaxText model end to end, through the callers' own code
wherever it can be imported:

- `TrainScriptTest` runs the standalone script `maxtext.experimental.maxtext_engine.train`
  through its own `main()`, with gradient accumulation and no logical-axis rules bound
  outside the engine.
- `TunixPathTest` builds the engine with Tunix's `build_maxtext_config`, `create_maxtext_mesh`
  and `create_maxtext_engine`, hosts it in Tunix's `TrainerWorker`, attaches Tunix's GRPO loss
  the way `DistributedRLEngine` does, trains on payloads from Tunix's `GRPOAdapter` and
  `PaddedBatchAssembler`, and scores log-probs through `TrainerWorker.per_token_logps`, which
  enters the engine's `model_scope`.
- `TrainPyParityTest` trains one tiny dense model on the same data twice, through
  `pre_train/train.py`'s jitted `train_step` and through the engine, and compares the loss at
  every step and the final parameters.
- `MoeLoadBalanceTest` logs how the two paths weight the MoE load-balancing loss under
  gradient accumulation.

The suites need four CPU devices. The CPU backend reads `--xla_force_host_platform_device_count`
only when it initializes, which has already happened by the time pytest imports this module, so
pytest collects one test per suite that re-runs this file in a child process with the flag set.
"""

import collections
import contextlib
import dataclasses
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.core import spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_TRAIN_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")
# Measurements the suites print for a human to read; the parent echoes them.
_MEASURE = "ENGINE_TRAIN_MEASURE"

_SUITES = ("TrainScriptTest", "TunixPathTest", "TrainPyParityTest", "MoeLoadBalanceTest")


@pytest.mark.parametrize("suite", _SUITES)
def test_suite_in_subprocess(suite):
  """Runs one suite in a child process with four CPU devices.

  The device-count flag is appended because XLA takes the last occurrence of a repeated flag.
  A child that skipped every test also exits 0, so the child reports how many tests it ran.
  """
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__, suite], env=env, capture_output=True, text=True, check=False)

  for line in result.stdout.splitlines():
    if line.startswith(_MEASURE):
      print(line)
  report = f"stdout:\n{result.stdout[-20000:]}\nstderr:\n{result.stderr[-20000:]}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


# ---------------------------------------------------------------------------------------
# Shared helpers. MaxText and Tunix modules are imported inside the functions that use
# them, so collecting this file in the parent pytest process imports neither.
# ---------------------------------------------------------------------------------------


def _measure(**fields: Any) -> None:
  """Prints one measurement line the parent echoes, `key=value` separated by spaces."""
  print(_MEASURE, " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


class UnboundRulesTracker:
  """Records every read of flax's ambient logical-axis rules that finds none bound.

  MaxText layers resolve logical axis names through the rules bound by
  `flax.core.spmd.logical_axis_rules`. With none bound, `maybe_shard_with_logical` silently
  becomes a no-op and XLA picks the partitioning, and a layer that validates the rules (such as
  Tokamax ring attention at model init) raises. The first is invisible on CPU, so this
  instruments flax's rule store and records the MaxText call site of each unbound read.

  Two reads are not lookups and are skipped: `logical_axis_rules` saving the previous rules on
  entry, and flax's `apply_rules`, which composes the ambient rules with explicit
  `sharding_rules` from its caller and so is correct with none bound.
  """

  _SKIPPED_READERS = ("logical_axis_rules", "apply_rules")

  def __init__(self):
    self.sites: collections.Counter = collections.Counter()
    self._quiet = threading.local()

  def ambient_rules(self) -> tuple:
    """The rules bound right now, read without counting the read itself as an unbound lookup."""
    self._quiet.on = True
    try:
      return flax_spmd.get_logical_axis_rules()
    finally:
      self._quiet.on = False

  def _record(self) -> None:
    """Counts one unbound read against the MaxText frame that made it."""
    if getattr(self._quiet, "on", False):
      return
    frame = sys._getframe(2)  # pylint: disable=protected-access
    reader = frame.f_code.co_name
    if reader == "get_logical_axis_rules":
      caller = frame.f_back
      if caller is not None and caller.f_code.co_name in self._SKIPPED_READERS:
        return
    elif reader in self._SKIPPED_READERS:
      return
    site = None
    while frame is not None:
      path = frame.f_code.co_filename.replace(os.sep, "/")
      # `site-packages` is excluded by name: a virtualenv under a `maxtext/` directory puts
      # flax itself on a path containing "/maxtext/".
      if (
          "/maxtext/" in path
          and "site-packages" not in path
          and "/tests/" not in path
          and not path.endswith("maxtext/utils/sharding.py")
      ):
        site = f"{path.split('/maxtext/', 1)[1]}:{frame.f_lineno}:{frame.f_code.co_name}"
        break
      frame = frame.f_back
    self.sites[site or f"<non-maxtext:{reader}>"] += 1

  @contextlib.contextmanager
  def watching(self):
    """Swaps the rule store for one that reports unbound reads, for the `with` body."""
    original = flax_spmd._axis_rules  # pylint: disable=protected-access
    tracker = self
    initial = original.rules

    class _ObservedRules(threading.local):
      """flax's thread-local rule store, reporting reads that find it empty."""

      def __init__(self):
        super().__init__()
        self._rules = initial

      @property
      def rules(self):
        if not self._rules:
          tracker._record()  # pylint: disable=protected-access
        return self._rules

      @rules.setter
      def rules(self, value):
        self._rules = value

    flax_spmd._axis_rules = _ObservedRules()  # pylint: disable=protected-access
    try:
      yield self
    finally:
      flax_spmd._axis_rules = original  # pylint: disable=protected-access

  def total(self) -> int:
    """The number of unbound reads recorded."""
    return sum(self.sites.values())

  def top(self, n: int = 8) -> str:
    """The `n` call sites with the most unbound reads, as `site x count` joined by `;`."""
    return ";".join(f"{site}x{count}" for site, count in self.sites.most_common(n)) or "none"


def _dense_overrides(**overrides) -> dict[str, Any]:
  """A tiny real dense decoder: two layers, 64 wide, 32 tokens. Seconds to compile on CPU."""
  base = {
      "model_name": "default",
      "vocab_size": 128,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "max_target_length": 32,
      "per_device_batch_size": 1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "enable_dropout": False,
      "enable_checkpointing": False,
      "enable_tensorboard": False,
      "record_internal_nn_metrics": False,
      "skip_jax_distributed_system": True,
      "profiler_steps": 0,
      "log_config": False,
      "init_weights_seed": 0,
  }
  base.update(overrides)
  return base


def _argv(run_name: str, output_dir: str, overrides: dict[str, Any]) -> list[str]:
  """A MaxText command line over the test `base.yml` with `overrides` as `key=value` arguments."""
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  return [
      sys.argv[0],
      get_test_config_path("base.yml"),
      f"run_name={run_name}",
      f"base_output_directory={output_dir}",
  ] + [f"{k}={v}" for k, v in overrides.items()]


def _lm_batch(rows: int, seq: int, vocab: int, seed: int) -> dict[str, np.ndarray]:
  """A `train.py` loss batch with ragged sequence lengths, as host arrays.

  With every position real, a micro-batch's `total_weights` is a power of two and dividing by
  it is exact in any float format, so a gradient cast to bf16 before the division could not be
  told apart from one cast after it.
  """
  rng = np.random.default_rng(seed)
  lengths = seq - (np.arange(rows) * 3 + seed) % 7
  real = (np.arange(seq)[None, :] < lengths[:, None]).astype(np.int32)
  tokens = rng.integers(1, vocab, size=(rows, seq)).astype(np.int32) * real
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1) * np.roll(real, -1, axis=-1),
      "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (rows, 1)),
      "inputs_segmentation": real,
      "targets_segmentation": real * np.roll(real, -1, axis=-1),
  }


def _memory_kinds(tree) -> set[str]:
  """The memory kinds (`device`, `pinned_host`, ...) the arrays of a tree live in."""
  return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree) if hasattr(leaf, "sharding")}


def _flat_params(model) -> dict[str, np.ndarray]:
  """`{path: float64 array}` over a model's `nnx.Param` leaves."""
  flat = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param))):
    flat[jax.tree_util.keystr(path)] = np.asarray(leaf, dtype=np.float64)
  return flat


def _update_rel(p0: dict, pa: dict, pb: dict) -> float:
  """`||(pa - p0) - (pb - p0)|| / ||pb - p0||` over the whole tree: how far apart two updates are."""
  num = sum(float(np.sum((pa[k] - pb[k]) ** 2)) for k in pb)
  den = sum(float(np.sum((pb[k] - p0[k]) ** 2)) for k in pb)
  if not den:
    raise AssertionError("The reference update left every parameter unchanged, so there is nothing to compare.")
  return math.sqrt(num / den)


def _max_diff(pa: dict, pb: dict) -> tuple[float, float]:
  """Largest absolute and largest relative elementwise difference between two param trees."""
  max_abs = max(float(np.max(np.abs(pa[k] - pb[k]))) for k in pb)
  max_rel = max(float(np.max(np.abs(pa[k] - pb[k]) / np.maximum(np.abs(pb[k]), 1e-12))) for k in pb)
  return max_abs, max_rel


def _copy_arrays(tree):
  """`tree` with every `jax.Array` leaf copied into a fresh buffer."""
  return jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, tree)


def _state_shardings(state, mesh):
  """Each leaf's own sharding if it is a `NamedSharding` on `mesh`, else replicated over `mesh`."""
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def home(leaf):
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == mesh:
      return leaf_sharding
    return replicated

  return jax.tree.map(home, state)


def _jit_train_py_step(cfg, mesh, graphdef, state, state_mesh_shardings):
  """`pre_train/train.py`'s `train_step`, jitted the way train.py jits it; returns it and the data sharding."""
  # pylint: disable=import-outside-toplevel
  from maxtext.trainers.pre_train import train as pre_train
  from maxtext.utils import sharding
  from maxtext.utils import train_utils

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(cfg, state_mesh_shardings)
  data_sharding = sharding.get_input_data_sharding(cfg, mesh)
  train_step = train_utils.jit_train_step(
      cfg, graphdef, state, state_mesh_shardings, data_sharding, pre_train.train_step, params_shardings, mesh
  )
  return train_step, data_sharding


@dataclasses.dataclass
class _FakeChipDevice:
  """The two attributes `count_chips` reads off a device; CPU and GPU devices have no `coords`."""

  coords: Any = None
  slice_index: int = 0


class _FakeMemoryStatsDevice:
  """A device whose `memory_stats()` returns what it was built with; a CPU device returns `None`."""

  def __init__(self, stats: dict[str, int] | None):
    self._stats = stats

  def memory_stats(self):
    return self._stats

  def __str__(self):
    return "TPU_0"


# ---------------------------------------------------------------------------------------
# 1. The standalone script, through its own main().
# ---------------------------------------------------------------------------------------


_SCRIPT_STEPS = 3
_SCRIPT_MICRO_STEPS = 2
# How many devices `count_chips` is told share a chip in the script runs; see `_run_main`.
_SCRIPT_DEVICES_PER_CHIP = 2


@dataclasses.dataclass
class _ScriptRun:
  """What one `main()` run did, as the spies in `TrainScriptTest._run_main` recorded it."""

  # Every `max_logging.log` message, in order.
  lines: list[str] = dataclasses.field(default_factory=list)
  # The same messages, interleaved with "<fwd_bwd>", "<update>" and "<wait_for_all>" markers.
  events: list[str] = dataclasses.field(default_factory=list)
  losses: list[float] = dataclasses.field(default_factory=list)
  # Each global batch as the data loader returned it, and each micro-batch as `fwd_bwd` got it.
  loaded: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
  fed: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
  # The devices each `count_chips` call was given.
  chip_counts: list[list[Any]] = dataclasses.field(default_factory=list)
  # Engine entry points checked for rules its caller bound, and the ones that found some.
  engine_entries: collections.Counter = dataclasses.field(default_factory=collections.Counter)
  caller_rules: list[str] = dataclasses.field(default_factory=list)
  engines: list[Any] = dataclasses.field(default_factory=list)
  tracker: UnboundRulesTracker = dataclasses.field(default_factory=UnboundRulesTracker)


def _measure_script_run(case: str, run: _ScriptRun) -> None:
  """Prints one script run's losses and rule-check counts."""
  _measure(
      suite="script",
      case=case,
      losses=",".join(f"{l:.6f}" for l in run.losses),
      unbound_reads=run.tracker.total(),
      caller_rules=len(run.caller_rules),
      engine_entries_checked=sum(run.engine_entries.values()),
  )


class TrainScriptTest(absltest.TestCase):
  """`maxtext.experimental.maxtext_engine.train.main()`, end to end, with no outer rules.

  Neither this test nor `main()` binds logical-axis rules or a mesh. Spies around the run
  record what the script did -- log lines, the batches loaded and fed, the engine calls and the
  rules bound when the engine is entered -- and `_assert_script_run` checks them. The
  `UnboundRulesTracker` catches rules that are missing; the entry spy catches rules bound
  outside the engine.
  """

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def _run_main(self, run_name: str, **overrides) -> _ScriptRun:
    """Runs `main()` for `_SCRIPT_STEPS` steps of `_SCRIPT_MICRO_STEPS` micro-batches under spies.

    Every spy passes through to the function it wraps and returns its result. Two of them pass
    stand-in devices, because CPU devices cannot show what is checked: `log_peak_memory` reads a
    device that keeps a peak counter (CPU keeps none), and `count_chips` gets as many devices as
    the run has, two to a chip (CPU devices are one to a chip, so a rate divided by the device
    count could not be told apart from one divided by the chip count).
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.experimental.maxtext_engine import train as engine_train_script
    from maxtext.training_engine import inflight_throttler
    from maxtext.training_engine import maxtext_engine
    from maxtext.training_engine import metrics as metrics_module

    self.assertEqual(flax_spmd.get_logical_axis_rules(), (), "the test itself must not bind rules")
    run = _ScriptRun()
    engine_cls = maxtext_engine.MaxTextTrainingEngine
    real_log = engine_train_script.max_logging.log
    real_process = metrics_module.MetricsLogger.process_metrics
    real_log_peak_memory = engine_train_script.log_peak_memory
    real_count_chips = engine_train_script.count_chips
    real_load = engine_train_script.DataLoader.load_next_batch
    real_init = engine_cls.__init__
    real_sharding_ctx = engine_cls._sharding_ctx  # pylint: disable=protected-access
    real_fwd_bwd = engine_cls.fwd_bwd
    real_update = engine_cls.update
    real_wait_for_all = inflight_throttler.InflightThrottler.wait_for_all

    def log(message, *args, **kwargs):
      run.lines.append(str(message))
      run.events.append(str(message))
      return real_log(message, *args, **kwargs)

    def process(self_, buffer):
      processed = real_process(self_, buffer)
      if "loss" in processed:
        run.losses.append(float(processed["loss"]))
      return processed

    def log_peak_memory():
      with mock.patch.object(jax, "local_devices", return_value=[_FakeMemoryStatsDevice({"peak_bytes_in_use": 2**30})]):
        return real_log_peak_memory()

    def count_chips(devices):
      run.chip_counts.append(list(devices))
      return real_count_chips(
          [_FakeChipDevice(coords=(i // _SCRIPT_DEVICES_PER_CHIP, 0, 0)) for i in range(len(devices))]
      )

    def load_next_batch(self_, *args, **kwargs):
      batch = real_load(self_, *args, **kwargs)
      run.loaded.append({name: np.asarray(value) for name, value in batch.items()})
      return batch

    def check_rules(entry: str) -> None:
      run.engine_entries[entry] += 1
      bound = run.tracker.ambient_rules()
      if bound:
        run.caller_rules.append(f"{entry} entered with {len(bound)} rules its caller bound")

    def init(self_, *args, **kwargs):
      run.engines.append(self_)
      check_rules("__init__")
      return real_init(self_, *args, **kwargs)

    ctx_depth = 0

    @contextlib.contextmanager
    def sharding_ctx(self_):
      nonlocal ctx_depth
      # Only the outermost entry: a nested one sees the engine's own rules, as it should.
      if not ctx_depth:
        check_rules("_sharding_ctx")
      ctx_depth += 1
      try:
        with real_sharding_ctx(self_):
          yield
      finally:
        ctx_depth -= 1

    def fwd_bwd(self_, payload, **kwargs):
      run.fed.append({name: np.asarray(value) for name, value in payload.items()})
      run.events.append("<fwd_bwd>")
      return real_fwd_bwd(self_, payload, **kwargs)

    def update(self_, **kwargs):
      run.events.append("<update>")
      return real_update(self_, **kwargs)

    def wait_for_all(self_):
      run.events.append("<wait_for_all>")
      return real_wait_for_all(self_)

    argv = _argv(
        run_name,
        tempfile.mkdtemp(prefix=f"{run_name}_"),
        _dense_overrides(
            dataset_type="synthetic",
            steps=_SCRIPT_STEPS,
            gradient_accumulation_steps=_SCRIPT_MICRO_STEPS,
            learning_rate=1e-3,
            warmup_steps_fraction=0.0,
            **overrides,
        ),
    )
    with contextlib.ExitStack() as stack:
      for patch in (
          mock.patch.object(engine_train_script.max_logging, "log", side_effect=log),
          mock.patch.object(metrics_module.MetricsLogger, "process_metrics", autospec=True, side_effect=process),
          mock.patch.object(engine_train_script, "log_peak_memory", side_effect=log_peak_memory),
          mock.patch.object(engine_train_script, "count_chips", side_effect=count_chips),
          mock.patch.object(engine_train_script.DataLoader, "load_next_batch", new=load_next_batch),
          mock.patch.object(engine_cls, "__init__", new=init),
          mock.patch.object(engine_cls, "_sharding_ctx", new=sharding_ctx),
          mock.patch.object(engine_cls, "fwd_bwd", new=fwd_bwd),
          mock.patch.object(engine_cls, "update", new=update),
          mock.patch.object(inflight_throttler.InflightThrottler, "wait_for_all", new=wait_for_all),
          run.tracker.watching(),
      ):
        stack.enter_context(patch)
      engine_train_script.main(argv)
    return run

  def _assert_peak_memory_logged(self, lines):
    """One peak line right after the first step's memstats, and one after the last step.

    The first means a run that fails in a later step has still reported its peak.
    """
    peaks = [i for i, line in enumerate(lines) if line.startswith("Peak live arrays on")]
    self.assertLen(peaks, 2, "\n".join(lines))
    first_memstats = next(i for i, line in enumerate(lines) if "Memstats: After first optimizer step" in line)
    second_step = next(i for i, line in enumerate(lines) if line.startswith("completed step: 1,"))
    last_step = next(i for i, line in enumerate(lines) if line.startswith(f"completed step: {_SCRIPT_STEPS - 1},"))
    self.assertTrue(first_memstats < peaks[0] < second_step, f"first peak at {peaks[0]}\n" + "\n".join(lines))
    self.assertGreater(peaks[1], last_step, "\n".join(lines))

  def _assert_micro_batch_split(self, run: _ScriptRun):
    """Step s trains on the s-th global batch loaded, micro-batch k being rows k::G of it."""
    num_micro = _SCRIPT_MICRO_STEPS
    self.assertEqual(len(run.loaded), _SCRIPT_STEPS, "one global batch loaded per optimizer step")
    self.assertEqual(len(run.fed), _SCRIPT_STEPS * num_micro, "G micro-batches fed per optimizer step")
    for step, global_batch in enumerate(run.loaded):
      # Precondition: a step's micro-batches differ, so feeding the wrong one would show.
      self.assertFalse(np.array_equal(global_batch["inputs"][0::num_micro], global_batch["inputs"][1::num_micro]))
      for k in range(num_micro):
        fed = run.fed[step * num_micro + k]
        self.assertEqual(set(fed), set(global_batch))
        for name, value in global_batch.items():
          np.testing.assert_array_equal(fed[name], value[k::num_micro], err_msg=f"step {step}, micro-batch {k}, {name}")

  def _assert_throughput(self, run: _ScriptRun):
    """tokens/step is what one step was fed, and both rates divide it by the right count."""
    starting = [line for line in run.lines if line.startswith("Starting training engine loop")]
    self.assertLen(starting, 1, "\n".join(run.lines))
    tokens_per_step, devices, chips = map(
        int, re.search(r"(\d+) tokens/step across (\d+) devices on (\d+) chips", starting[0]).groups()
    )
    fed_tokens = sum(micro_batch["inputs"].size for micro_batch in run.fed[:_SCRIPT_MICRO_STEPS])
    self.assertEqual(tokens_per_step, fed_tokens, starting[0])
    self.assertEqual(devices, jax.device_count(), starting[0])
    self.assertEqual(run.chip_counts, [jax.devices()], "chips are counted once, over every device of the run")
    self.assertEqual(chips, jax.device_count() // _SCRIPT_DEVICES_PER_CHIP, starting[0])
    steps = [line for line in run.lines if line.startswith("completed step:")]
    self.assertLen(steps, _SCRIPT_STEPS, "\n".join(run.lines))
    for i, line in enumerate(steps):
      self.assertTrue(line.startswith(f"completed step: {i},"), line)
      self.assertNotIn("tokens/s/chip", line, "the per-device rate must not be labelled per chip")
      step_ms, per_device, per_chip = (
          float(re.search(rf"{label}: ([\d.]+)", line).group(1))
          for label in ("train_step_time_ms", "Tokens/s/device", "Tokens/s/chip")
      )
      # The bounds are exactly the printed roundings: 2 decimals of ms, 3 of each rate.
      low = (per_device - 5e-4) * devices * (step_ms - 5e-3) / 1e3
      high = (per_device + 5e-4) * devices * (step_ms + 5e-3) / 1e3
      self.assertTrue(low <= tokens_per_step <= high, f"{tokens_per_step} tokens/step outside [{low}, {high}]: {line}")
      self.assertAlmostEqual(
          per_chip, per_device * _SCRIPT_DEVICES_PER_CHIP, delta=5e-4 * (1 + _SCRIPT_DEVICES_PER_CHIP), msg=line
      )

  def _assert_step_timed_after_drain(self, run: _ScriptRun):
    """`wait_for_all` runs between a step's `update` and its step line, which reads the clock."""
    for i in range(_SCRIPT_STEPS):
      line_at = next(j for j, event in enumerate(run.events) if event.startswith(f"completed step: {i},"))
      update_at = max(j for j in range(line_at) if run.events[j] == "<update>")
      self.assertIn("<wait_for_all>", run.events[update_at:line_at], f"step {i} was timed before its update drained")

  def _assert_script_run(self, run: _ScriptRun):
    """What the script owes every run, read off the spies of `_run_main`."""
    self.assertLen(run.losses, _SCRIPT_STEPS, f"one logged loss per optimizer step, got {run.losses}")
    self.assertTrue(all(math.isfinite(l) for l in run.losses), run.losses)
    # No rules missing: nothing read the ambient rules while none were bound (`TunixPathTest`
    # checks that the tracker detects such reads). No rules extra: nothing outside the engine
    # bound any around it, which would hide a call path that forgets to bind them.
    self.assertEqual(run.tracker.total(), 0, f"traced without rules: {run.tracker.top()}")
    self.assertGreater(run.engine_entries["__init__"], 0, "the entry spy never saw the engine built")
    self.assertGreater(run.engine_entries["_sharding_ctx"], 0, "the entry spy never saw the engine bind its rules")
    self.assertEqual(run.caller_rules, [])
    self._assert_micro_batch_split(run)
    self._assert_throughput(run)
    self._assert_step_timed_after_drain(run)
    # Memory is reported once after compile and once after the first step, never per step.
    for label in ("Memstats: After engine compile", "Memstats: After first optimizer step"):
      self.assertLen([line for line in run.lines if label in line], 1, label)
    self._assert_peak_memory_logged(run.lines)

  def test_train_with_grad_accumulation(self):
    run = self._run_main("engine_train_script")
    _measure_script_run("G2", run)
    self._assert_script_run(run)
    # Without optimizer offload the optimizer state stays on the device.
    self.assertEqual(self._optimizer_memory_kinds(run), {"device"})

  def _optimizer_memory_kinds(self, run: _ScriptRun) -> set[str]:
    """The memory kinds of the run's optimizer arrays once it is over; params must be on the device."""
    self.assertLen(run.engines, 1)
    state = run.engines[0].state
    self.assertEqual(_memory_kinds(nnx.state(state.model, nnx.Param)), {"device"})
    return _memory_kinds(nnx.state(state.optimizer))

  def test_log_peak_memory(self):
    """A device with a peak counter logs one line, labelled as not the HBM peak; one without logs none."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.experimental.maxtext_engine import train as engine_train_script

    gib = 2**30
    tpu = _FakeMemoryStatsDevice({"peak_bytes_in_use": int(22.04 * gib) + 1, "bytes_limit": int(94.74 * gib)})
    logged = []
    with mock.patch.object(engine_train_script.max_logging, "log", side_effect=logged.append):
      with mock.patch.object(jax, "local_devices", return_value=[tpu]):
        engine_train_script.log_peak_memory()
      self.assertLen(logged, 1)
      self.assertIn("Peak live arrays on TPU_0: 22.04 GiB", logged[0])
      self.assertIn("not the HBM peak", logged[0])
      self.assertNotIn("Peak memory", logged[0], "must not be labelled as the HBM peak")
      for silent in (None, {}, {"bytes_limit": gib}):
        with mock.patch.object(jax, "local_devices", return_value=[_FakeMemoryStatsDevice(silent)]):
          engine_train_script.log_peak_memory()
      self.assertLen(logged, 1, f"a backend without a peak counter must log nothing, got {logged[1:]}")

  def test_train_with_optimizer_offload(self):
    run = self._run_main("engine_train_script_offload", optimizer_memory_host_offload=True)
    _measure_script_run("G2_offload", run)
    self._assert_script_run(run)
    # The run trains the same whether or not the flag takes effect, so check where the state
    # ends up: every optimizer array, the step count included, must be in pinned host memory.
    self.assertEqual(self._optimizer_memory_kinds(run), {"pinned_host"})

  def test_split_global_batch(self):
    """`_split_global_batch` gives train.py's micro-batches, on the data sharding, with no collectives.

    `gradient_accumulation.py` makes micro-batch k rows `k::G` of the global batch (reshape to
    `[B // G, G, ...]`, swap the first two axes). The batch is sharded over both of its axes
    the way the data loader shards it, and every row is distinct, so an order error cannot
    hide. Both checks are also run on a contiguous split, which must fail them: its rows are in
    the wrong order, and inside one jit it compiles to collectives.
    """
    # pylint: disable-next=import-outside-toplevel
    from maxtext.experimental.maxtext_engine import train as engine_train_script

    num_micro, rows, seq = 4, 16, 8
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2, 2), ("fsdp", "context"))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "context"))
    host = {
        name: np.arange(rows * seq, dtype=np.int32).reshape(rows, seq) + 1000 * i
        for i, name in enumerate(("inputs", "targets", "targets_segmentation"))
    }
    batch = jax.device_put(host, data_sharding)

    def order_errors(micro_batches) -> list[str]:
      if len(micro_batches) != num_micro:
        return [f"{len(micro_batches)} micro-batches, expected {num_micro}"]
      return [
          f"micro-batch {k} {name}"
          for k, micro_batch in enumerate(micro_batches)
          for name, value in host.items()
          if not np.array_equal(np.asarray(micro_batch[name]), value[k::num_micro])
      ]

    def collectives(split_fn) -> list[str]:
      hlo = split_fn.lower(batch).compile().as_text()
      return re.findall(r"\b(all-gather|all-to-all|collective-permute|all-reduce|reduce-scatter)(?:-start)?\(", hlo)

    micro_batches = engine_train_script._split_global_batch(batch, num_micro, data_sharding)  # pylint: disable=protected-access
    self.assertEqual(order_errors(micro_batches), [])
    for micro_batch in micro_batches:
      self.assertEqual(set(micro_batch), set(host))
      for name, value in micro_batch.items():
        self.assertEqual(value.shape, (rows // num_micro, seq), name)
        self.assertEqual(value.sharding, data_sharding, name)
    splitter = engine_train_script._micro_batch_splitter(num_micro, data_sharding)  # pylint: disable=protected-access
    self.assertEqual(collectives(splitter), [], "each device holds its own rows of every micro-batch")

    # Both checks detect a contiguous split: eager slices fail the order check, and the same
    # split inside one jit compiles to collectives.
    micro_rows = rows // num_micro
    contiguous_eager = [
        {
            name: jax.device_put(value[k * micro_rows : (k + 1) * micro_rows], data_sharding)
            for name, value in batch.items()
        }
        for k in range(num_micro)
    ]
    self.assertLen(
        order_errors(contiguous_eager), num_micro * len(host), "contiguous rows must fail for every micro-batch"
    )
    contiguous_jit = jax.jit(
        lambda b: [{n: v[k * micro_rows : (k + 1) * micro_rows] for n, v in b.items()} for k in range(num_micro)],
        out_shardings=data_sharding,
    )
    self.assertNotEqual(collectives(contiguous_jit), [], "a contiguous split must move rows between devices")
    _measure(
        suite="script",
        case="split",
        collectives_strided=len(collectives(splitter)),
        collectives_contiguous_jit=len(collectives(contiguous_jit)),
    )

  def test_count_chips(self):
    """`count_chips` counts distinct `(slice_index, coords)`, and warns once when devices carry no coords."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.experimental.maxtext_engine import train as engine_train_script

    count_chips = engine_train_script.count_chips
    logged = []
    with mock.patch.object(engine_train_script.max_logging, "log", side_effect=logged.append):
      v7x = [_FakeChipDevice(coords=(x, y, 0)) for x in range(2) for y in range(2) for _ in range(2)]
      self.assertEqual(count_chips(v7x), 4, "two TensorCores per v7x chip are one chip")
      multislice = [_FakeChipDevice(coords=(0, 0, 0), slice_index=s) for s in range(3)]
      self.assertEqual(count_chips(multislice), 3, "the same coords on different slices are different chips")
      self.assertEqual(logged, [], "devices with coords are counted without a warning")

      # The fallback is right on CPU and GPU but blind to shared chips, so it says so, once.
      self.assertEqual(count_chips([_FakeChipDevice(), _FakeChipDevice(), _FakeChipDevice()]), 3)
      self.assertLen(logged, 1, "one warning per fallback, not one per device")
      self.assertIn("WARNING", logged[0])
      self.assertIn("counted as one chip", logged[0])
      self.assertEqual(count_chips(jax.devices()), jax.device_count())
      self.assertLen(logged, 2, "CPU devices carry no coords")


# ---------------------------------------------------------------------------------------
# 2. Tunix's own construction, worker, GRPO loss, batch assembly and log-prob scoring.
# ---------------------------------------------------------------------------------------

# Token ids. The random draws below start above both, so neither appears by accident.
_PAD_ID = 0
_EOS_ID = 1
_PROMPT_LEN = 16
_RESPONSE_LEN = 16
_NUM_GENERATIONS = 4
_MINI_BATCH = 2  # prompt groups per optimizer update
_MICRO_BATCH = 4  # sequences per fwd_bwd: 8 trajectories = 2 micro-batches per update

# Every dimension of qwen3-0.6b overridden down to a toy. A real HF_MODEL_CONFIGS key is
# required: `TunixMaxTextAdapter` looks the model up there. Passed the way a Tunix run passes
# extra MaxText flags, through `MAXTEXT_EXTRA_FLAGS`, which `build_maxtext_config` applies last.
_TUNIX_EXTRA_FLAGS = " ".join(
    [
        "override_model_config=true",
        "base_emb_dim=64",
        "base_num_query_heads=2",
        "base_num_kv_heads=2",
        "head_dim=32",
        "base_mlp_dim=64",
        "base_num_decoder_layers=2",
        "vocab_size=128",
        "enable_dropout=false",
        "log_config=false",
    ]
)


class TunixPathTest(absltest.TestCase):
  """The engine as Tunix's maxtext trainer path builds and drives it.

  Mirrors, with Tunix's own functions wherever they import:
  `run_trainer_node.py::_create_maxtext_trainer_factory` (config, mesh, engine factory),
  `run_trainer_node.py` (`TrainerWorker(..., execution_context=mesh)`),
  `distributed_rl_engine.py::configure` (`with_loss_fn(algo.loss_fn(), has_aux=True)` and
  `with_gen_model_input_fn(algo.build_gen_model_input_fn(pad_id, eos_id))`), and the
  orchestrator's `GRPOAdapter.create_trainer_payloads` -> `PaddedBatchAssembler.feed`.
  """

  __test__ = False

  def _worker(self):
    """A `TrainerWorker` around the engine, built exactly as `run_trainer_node.py` builds it."""
    # pylint: disable=import-outside-toplevel
    from tunix.experimental.worker import trainer_worker
    from tunix.utils import maxtext_utils as tunix_maxtext_utils

    with mock.patch.dict(os.environ, {"MAXTEXT_EXTRA_FLAGS": _TUNIX_EXTRA_FLAGS}):
      config = tunix_maxtext_utils.build_maxtext_config(
          model_name="qwen3-0.6b",
          worker_id="engine_tunix",
          train_micro_batch_size=_MICRO_BATCH,
          mesh_fsdp=_REQUIRED_DEVICES,
          num_devices=jax.device_count(),
          max_prompt_length=_PROMPT_LEN,
          max_response_length=_RESPONSE_LEN,
          # Large enough to move bf16 weights, which the live-weights check below relies on: at
          # Tunix's 1e-5 default an Adam step on bf16 weights mostly rounds back to the same value.
          learning_rate=1e-2,
          base_output_directory=tempfile.mkdtemp(prefix="engine_tunix_"),
          gradient_accumulation_steps=_MINI_BATCH * _NUM_GENERATIONS // _MICRO_BATCH,
      )
    mesh = tunix_maxtext_utils.create_maxtext_mesh(config)

    def factory():
      return tunix_maxtext_utils.create_maxtext_engine(
          config, mesh=mesh, tokenizer_pad_id=_PAD_ID, wrap_with_tunix_adapter=True
      )

    return trainer_worker.TrainerWorker(trainer_factory=factory, execution_context=mesh), config

  def _grpo(self):
    """Tunix's `GRPOAdapter` for this suite's group, mini-batch and micro-batch sizes."""
    # pylint: disable=import-outside-toplevel
    from tunix.experimental.orchestrator import algorithm_adapter
    from tunix.rl import algorithm_config

    return algorithm_adapter.GRPOAdapter(
        algorithm_config.GRPOConfig(num_generations=_NUM_GENERATIONS, temperature=1.0, beta=0.0),
        mini_batch_size=_MINI_BATCH,
        train_micro_batch_size=_MICRO_BATCH,
        max_response_length=_RESPONSE_LEN,
    )

  def _assembled_batches(self, algo, vocab: int):
    """Two prompt groups of rollouts, through the adapter and the padded assembler."""
    # pylint: disable=import-outside-toplevel
    from tunix.experimental.common import datatypes
    from tunix.experimental.orchestrator import batch_assembly

    rng = np.random.default_rng(0)
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=_MICRO_BATCH,
        max_prompt_length=_PROMPT_LEN,
        max_response_length=_RESPONSE_LEN,
        pad_id=_PAD_ID,
        num_generations=_NUM_GENERATIONS,
        mini_batch_size=_MINI_BATCH,
    )
    batches = []
    for group in range(_MINI_BATCH):
      prompt = rng.integers(2, vocab, size=10 + group).astype(np.int32)
      items = []
      for g in range(_NUM_GENERATIONS):
        length = 9 + 2 * g
        completion = rng.integers(2, vocab, size=length).astype(np.int32)
        # Agentic-style mask: assistant tokens scored, an environment turn in the middle not.
        mask = np.ones(length, np.float32)
        mask[3:5] = 0.0
        items.append(
            datatypes.TrajectoryItem(
                prompt_id=f"p{group}",
                group_index=g,
                traj={"prompt_tokens": prompt, "conversation_tokens": completion, "conversation_masks": mask},
            )
        )
      # Mixed rewards, so the GRPO advantages are non-zero and the gradient is live.
      payloads = algo.create_trainer_payloads(items, rewards=[1.0, 0.0, 1.0, 0.0])
      batches.extend(assembler.feed(payloads))
    return batches

  def _logps(self, worker, payload):
    """Scores `payload`'s tokens through `TrainerWorker.per_token_logps`, as the orchestrator does."""
    # pylint: disable-next=import-outside-toplevel
    from tunix.experimental.common import datatypes

    response = worker.per_token_logps(
        datatypes.LogprobsRequest(
            prompt_tokens=np.asarray(payload.prompt_ids),
            completion_tokens=np.asarray(payload.completion_ids),
            temperature=1.0,
            model_role="actor",
            pad_id=_PAD_ID,
            eos_id=_EOS_ID,
        )
    )
    return np.asarray(response.per_token_logps)

  def _configured_worker(self):
    """A worker configured exactly as `DistributedRLEngine.configure` leaves it, plus its batches."""
    self.assertEqual(flax_spmd.get_logical_axis_rules(), (), "the test itself must not bind rules")
    worker, config = self._worker()
    engine = worker._trainer  # pylint: disable=protected-access
    self.assertEqual(type(engine.model).__name__, "TunixMaxTextAdapter")
    self.assertEqual(config.gradient_accumulation_steps, 2)
    algo = self._grpo()
    worker.initialize()
    worker.with_loss_fn(algo.loss_fn(), has_aux=True)
    worker.with_gen_model_input_fn(algo.build_gen_model_input_fn(pad_id=_PAD_ID, eos_id=_EOS_ID))
    # Tunix's worker lifecycle compiles with no dummy data; the engine defers to the first batch.
    worker.compile()

    batches = self._assembled_batches(algo, config.vocab_size)
    self.assertLen(batches, 2)
    self.assertEqual([b.is_final_batch for b in batches], [False, True])
    payload = batches[0].payload
    self.assertEqual(np.asarray(payload.prompt_ids).shape, (_MICRO_BATCH, _PROMPT_LEN))
    self.assertEqual(np.asarray(payload.completion_ids).shape, (_MICRO_BATCH, _RESPONSE_LEN))
    return worker, engine, batches

  def _check_tracker(self, engine, payload) -> tuple[int, int]:
    """Checks the unbound-read tracker on one eager forward, with and without the engine's rules.

    With nothing bound the forward must register unbound reads; inside the engine's own
    `_sharding_ctx` it must register none. Returns both counts.
    """
    tokens = jnp.concatenate([jnp.asarray(payload.prompt_ids), jnp.asarray(payload.completion_ids)], axis=-1)
    positions = jnp.broadcast_to(jnp.arange(tokens.shape[-1], dtype=jnp.int32), tokens.shape)
    unbound, bound = UnboundRulesTracker(), UnboundRulesTracker()
    with unbound.watching():
      engine.model(tokens, positions, None, None)
    with bound.watching(), engine._sharding_ctx():  # pylint: disable=protected-access
      engine.model(tokens, positions, None, None)
    self.assertGreater(unbound.total(), 0, "a forward with no rules bound must register unbound reads")
    self.assertEqual(bound.total(), 0, f"a forward inside _sharding_ctx found no rules: {bound.top()}")
    return unbound.total(), bound.total()

  def test_grpo_train_and_score_logps(self):
    # pylint: disable-next=import-outside-toplevel
    from tunix.experimental.common import datatypes

    worker, engine, batches = self._configured_worker()
    payload = batches[0].payload
    unbound_reads_without_rules, unbound_reads_with_rules = self._check_tracker(engine, payload)

    train_tracker, score_tracker = UnboundRulesTracker(), UnboundRulesTracker()
    with score_tracker.watching():
      before = self._logps(worker, payload)
    with train_tracker.watching():
      for batch in batches:
        worker.fwd_bwd(datatypes.TrainRequest(payload=batch.payload))
      train_step = worker.update()
    metrics = engine.get_metrics(clear_cache=False)
    with score_tracker.watching():
      after = self._logps(worker, payload)

    # Nothing but the engine binds rules here, so a read that finds none is a call path that
    # does not bind them.
    self.assertEqual(train_tracker.total(), 0, f"fwd_bwd/update traced without rules: {train_tracker.top()}")
    self.assertEqual(score_tracker.total(), 0, f"per_token_logps traced without rules: {score_tracker.top()}")
    self.assertEqual(train_step, 1)
    loss = metrics.weighted_metrics["loss"]
    loss_value = float(np.sum(loss.unreduced_sum) / np.sum(loss.denominator))
    grad_norm = float(np.asarray(metrics.scalar_metrics["gradient_norm"])[-1])
    self.assertTrue(math.isfinite(loss_value), loss_value)
    self.assertTrue(math.isfinite(grad_norm) and grad_norm > 0.0, f"a live GRPO gradient, got {grad_norm}")

    completion_mask = np.asarray(payload.completion_mask) > 0
    for name, logps in (("before", before), ("after", after)):
      self.assertEqual(logps.shape, (_MICRO_BATCH, _RESPONSE_LEN), name)
      self.assertTrue(np.all(np.isfinite(logps)), name)
      self.assertTrue(np.all(logps[completion_mask] <= 1e-6), f"{name}: log-probs must be <= 0")
    # Scored with the trainer's *live* weights: one update later the same tokens score differently.
    moved = float(np.max(np.abs(after - before)[completion_mask]))
    self.assertGreater(moved, 0.0, "per_token_logps did not see the update")

    _measure(
        suite="tunix",
        loss=f"{loss_value:.6f}",
        grad_norm=f"{grad_norm:.6f}",
        logps_mean_before=f"{float(np.mean(before[completion_mask])):.6f}",
        logps_max_move=f"{moved:.3e}",
        unbound_reads_train=train_tracker.total(),
        unbound_reads_scoring=score_tracker.total(),
        unbound_reads_without_rules=unbound_reads_without_rules,
        unbound_reads_with_rules=unbound_reads_with_rules,
    )

  def test_run_eval_binds_rules(self):
    """`TrainerWorker.run_eval` traces the model under the engine's rules.

    `TrainerWorker.compile()` passes no dummy data, so the eval kernel is first traced inside
    `run_eval` (`eval_context` + `eval_step`); no rule lookup there may find the rules unbound.
    """
    worker, _, batches = self._configured_worker()
    tracker = UnboundRulesTracker()
    with tracker.watching():
      worker.run_eval([batches[0].payload])
    _measure(suite="tunix_eval", unbound_reads=tracker.total(), unbound_sites=tracker.top())
    self.assertEqual(tracker.total(), 0, f"eval_step traced without rules: {tracker.top()}")


# ---------------------------------------------------------------------------------------
# 3. Parity with pre_train/train.py's train_step.
# ---------------------------------------------------------------------------------------


class TrainPyParityTest(absltest.TestCase):
  """The engine against `pre_train/train.py`'s jitted `train_step`: same model, data and initial state.

  Both paths use SGD with a constant learning rate and no clipping, so the parameters after N
  steps are `p0 - lr * sum(grads)` and a gradient difference reaches them linearly. (Adam
  normalises each element by its own magnitude, which hides a relative gradient error at step
  one, so it gets a single case.) train.py's state is a copy of the engine's, taken before
  either trains. Micro-batch k of a G-step update is rows `k::G` of the global batch, the split
  `gradient_accumulation.py` makes, and sequence lengths are ragged (see `_lm_batch`).

  Every case bounds the largest per-step relative loss difference by 1e-5, and
  `update_rel = ||dp_engine - dp_train_py|| / ||dp_train_py||` over the whole tree after
  `_STEPS` steps (`dp = p_final - p0`) by 1e-4 with float32 gradients and 3e-4 with bfloat16
  ones. The forward pass is float32 and the parameters differ by a fraction of one update, so
  for bfloat16 gradients it is the update gate that checks the order of the cast.

  train.py sums micro-batch gradients in the parameters' dtype (`gradient_accumulation.py`
  accumulates into `zeros_like` of them), divides by the token count, and then casts float32
  leaves to `grad_dtype`. With float32 weights that is a float32 sum cast once, so the
  bfloat16-gradient cases set `grad_accumulation_dtype="float32"`; the engine's default sums in
  `grad_dtype` to save memory and is not expected to meet this gate there. With bfloat16
  weights train.py sums in bfloat16, which is the engine's default order, so that case keeps
  the default.
  """

  __test__ = False

  _STEPS = 3
  _LR = 0.1

  def _cfg(self, name: str, **overrides):
    """The tiny dense model with the SGD setup described in the class docstring."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.configs import pyconfig

    fields = {
        "opt_type": "sgd",
        "learning_rate": self._LR,
        "warmup_steps_fraction": 0.0,
        "learning_rate_final_fraction": 1.0,
        "gradient_clipping_threshold": 0.0,
        "steps": self._STEPS,
        "remat_policy": "none",
    }
    fields.update(overrides)
    return pyconfig.initialize(_argv(name, tempfile.mkdtemp(prefix=f"{name}_"), _dense_overrides(**fields)))

  def _run(self, name: str, **overrides) -> dict[str, float]:
    """Trains the same model on the same data through train.py and the engine; returns the gaps."""
    # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import maxtext_engine
    from maxtext.utils import maxtext_utils

    cfg = self._cfg(name, **overrides)
    num_micro = cfg.gradient_accumulation_steps
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)

    # train.py's state: a copy of the engine's, taken before either side trains.
    graphdef, state = nnx.split(engine.state)
    state_mesh_shardings = _state_shardings(state, mesh)
    state = jax.device_put(_copy_arrays(state), state_mesh_shardings)
    p_train_step, data_sharding = _jit_train_py_step(cfg, mesh, graphdef, state, state_mesh_shardings)
    p0 = _flat_params(engine.state.model)

    micro_rows = int(cfg.micro_batch_size_to_train_on)
    engine_losses, train_losses, denominators = [], [], []
    for step in range(self._STEPS):
      global_batch = _lm_batch(micro_rows * num_micro, cfg.max_target_length, cfg.vocab_size, seed=step)
      micro_batches = [{k: v[m::num_micro] for k, v in global_batch.items()} for m in range(num_micro)]
      denominators.extend(int(np.sum(mb["targets_segmentation"] != 0)) for mb in micro_batches)

      # pre_train/train.py, in the context its training loop calls it under (train.py `train_loop`).
      with jax.set_mesh(mesh), flax_spmd.logical_axis_rules(cfg.logical_axis_rules):
        state, metrics = p_train_step(state, jax.device_put(global_batch, data_sharding))
      train_losses.append(float(metrics["scalar"]["learning/loss"]))

      # The engine, as its callers drive it: no context at all.
      if step == 0:
        engine.compile(micro_batches[0])
      for micro_batch in micro_batches:
        engine.fwd_bwd(micro_batch)
      engine.update()
      loss = engine.get_metrics().weighted_metrics["loss"]
      engine_losses.append(float(np.sum(loss.unreduced_sum) / np.sum(loss.denominator)))

    # Precondition: with every denominator a power of two the bf16 cases could not see the cast
    # order (see `_lm_batch`).
    self.assertTrue(any(d & (d - 1) for d in denominators), f"every denominator is a power of two: {denominators}")
    p_engine = _flat_params(engine.state.model)
    p_train = _flat_params(nnx.merge(graphdef, state).model)
    self.assertEqual(set(p_engine), set(p_train))
    loss_rel = max(abs(e - t) / abs(t) for e, t in zip(engine_losses, train_losses))
    max_abs, max_rel = _max_diff(p_engine, p_train)
    result = {
        "loss_rel": loss_rel,
        "update_rel": _update_rel(p0, p_engine, p_train),
        "param_max_abs": max_abs,
        "param_max_rel": max_rel,
    }
    _measure(
        suite="parity",
        case=name,
        engine_losses=",".join(f"{l:.7f}" for l in engine_losses),
        train_losses=",".join(f"{l:.7f}" for l in train_losses),
        **{k: f"{v:.3e}" for k, v in result.items()},
    )
    engine.close()
    return result

  def _check(self, result, update_gate):
    """The loss at every step within 1e-5 relative, and the update within `update_gate`."""
    self.assertLessEqual(result["loss_rel"], 1e-5, result)
    self.assertLessEqual(result["update_rel"], update_gate, result)

  def test_float32_grads_one_micro_batch(self):
    self._check(self._run("parity_fp32_g1", grad_dtype="float32", gradient_accumulation_steps=1), 1e-4)

  def test_float32_grads_two_micro_batches(self):
    self._check(self._run("parity_fp32_g2", grad_dtype="float32", gradient_accumulation_steps=2), 1e-4)

  def test_float32_grads_two_micro_batches_adamw(self):
    self._check(
        self._run("parity_fp32_g2_adamw", grad_dtype="float32", gradient_accumulation_steps=2, opt_type="adamw"), 1e-4
    )

  def test_bfloat16_grads_one_micro_batch(self):
    self._check(
        self._run(
            "parity_bf16_g1", grad_dtype="bfloat16", grad_accumulation_dtype="float32", gradient_accumulation_steps=1
        ),
        3e-4,
    )

  def test_bfloat16_grads_two_micro_batches(self):
    self._check(
        self._run(
            "parity_bf16_g2", grad_dtype="bfloat16", grad_accumulation_dtype="float32", gradient_accumulation_steps=2
        ),
        3e-4,
    )

  def test_bfloat16_weights_default_accumulation(self):
    self._check(
        self._run("parity_bf16w_g2", weight_dtype="bfloat16", grad_dtype="bfloat16", gradient_accumulation_steps=2),
        3e-4,
    )


# ---------------------------------------------------------------------------------------
# 4. MoE load-balancing loss under gradient accumulation.
# ---------------------------------------------------------------------------------------


class MoeLoadBalanceTest(absltest.TestCase):
  """How train.py's accumulation path and the engine weight the MoE load-balancing loss.

  train.py's `loss_fn` under G > 1 returns `xent_sum + moe_lb_loss` unnormalised, and
  `gradient_accumulation.py` divides the summed gradient by the total token count, so the lb
  gradient is scaled by `1 / total_weights`. The engine rebuilds `xent_sum + lb *
  total_weights` and divides by the same total, i.e. weights each micro-batch's lb by its token
  share. So the engine's lb contribution to the router gradient is expected to be about
  `total_weights / G` times train.py's, and close to train.py's own G=1 path on the whole global
  batch (not equal: the lb loss is nonlinear in the batch it is taken over).

  Each path takes one SGD step at lb weight 0 and `_LB_WEIGHT`, with `g = (p0 - p1) / lr`. The
  test asserts that the gradients are finite and that the two G=2 paths agree with the lb loss
  off; the lb weighting itself is logged, not asserted.
  """

  __test__ = False

  _LR = 0.1
  _LB_WEIGHT = 0.01

  def _cfg(self, name, **overrides):
    """A tiny qwen3 MoE (4 experts, top-2) on the dense-matmul path, trained with plain SGD."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.configs import pyconfig

    fields = _dense_overrides(
        model_name="qwen3-30b-a3b",
        override_model_config=True,
        base_emb_dim=64,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=32,
        base_mlp_dim=64,
        base_moe_mlp_dim=32,
        base_num_decoder_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        sparse_matmul=False,
        megablox=False,
        opt_type="sgd",
        learning_rate=self._LR,
        warmup_steps_fraction=0.0,
        learning_rate_final_fraction=1.0,
        gradient_clipping_threshold=0.0,
        grad_dtype="float32",
        steps=1,
        remat_policy="none",
    )
    fields.update(overrides)
    return pyconfig.initialize(_argv(name, tempfile.mkdtemp(prefix=f"{name}_"), fields))

  def _router_grads(self, lb_weight: float) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Router gradients of the engine (G=2), train.py G=2 and train.py G=1, and the batch's token count.

    The gradients are `{path name: {router param: gradient}}`.
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import maxtext_engine
    from maxtext.utils import maxtext_utils

    cfg = self._cfg("moe_lb_engine", gradient_accumulation_steps=2, load_balance_loss_weight=lb_weight)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    graphdef, state0 = nnx.split(engine.state)
    state_mesh_shardings = _state_shardings(state0, mesh)
    state0 = _copy_arrays(state0)
    p0 = _flat_params(engine.state.model)
    rows = int(cfg.micro_batch_size_to_train_on)
    global_batch = _lm_batch(2 * rows, cfg.max_target_length, cfg.vocab_size, seed=0)

    grads = {}
    # train.py, with the accumulation path (G=2) and without it (G=1 over the same 2x rows).
    for label, train_cfg in (
        ("train_py_g2", cfg),
        (
            "train_py_g1",
            self._cfg(
                "moe_lb_train_g1",
                gradient_accumulation_steps=1,
                per_device_batch_size=2,
                load_balance_loss_weight=lb_weight,
            ),
        ),
    ):
      state = jax.device_put(_copy_arrays(state0), state_mesh_shardings)
      step, data_sharding = _jit_train_py_step(train_cfg, mesh, graphdef, state, state_mesh_shardings)
      with jax.set_mesh(mesh), flax_spmd.logical_axis_rules(train_cfg.logical_axis_rules):
        state, _ = step(state, jax.device_put(global_batch, data_sharding))
      p1 = _flat_params(nnx.merge(graphdef, state).model)
      grads[label] = {k: (v - p1[k]) / self._LR for k, v in p0.items()}

    micro_batches = [{k: v[m::2] for k, v in global_batch.items()} for m in range(2)]
    engine.compile(micro_batches[0])
    for micro_batch in micro_batches:
      engine.fwd_bwd(micro_batch)
    engine.update()
    p1 = _flat_params(engine.state.model)
    grads["engine_g2"] = {k: (v - p1[k]) / self._LR for k, v in p0.items()}
    engine.close()
    total_weights = int(np.sum(global_batch["targets_segmentation"] != 0))
    router = [k for k in p0 if "gate" in k and "moe" in k.lower()] or [k for k in p0 if "gate" in k]
    return {label: {k: g[k] for k in router} for label, g in grads.items()}, total_weights

  @staticmethod
  def _norm(tree) -> float:
    return math.sqrt(sum(float(np.sum(v**2)) for v in tree.values()))

  @staticmethod
  def _diff(a, b) -> dict:
    return {k: a[k] - b[k] for k in a}

  def test_load_balance_loss_weighting(self):
    off, total_weights = self._router_grads(0.0)
    on, _ = self._router_grads(self._LB_WEIGHT)
    self.assertTrue(off["engine_g2"], "found no router parameters to measure")
    for grads in (off, on):
      for label, tree in grads.items():
        self.assertTrue(all(np.all(np.isfinite(v)) for v in tree.values()), label)

    # With the lb loss off, the router gradients of the two G=2 paths agree.
    control = self._norm(self._diff(off["engine_g2"], off["train_py_g2"])) / self._norm(off["train_py_g2"])
    self.assertLessEqual(control, 1e-4, f"engine and train.py disagree even with the lb loss off: {control}")

    contribution = {label: self._diff(on[label], off[label]) for label in on}
    norms = {label: self._norm(tree) for label, tree in contribution.items()}
    total_rel = self._norm(self._diff(on["engine_g2"], on["train_py_g2"])) / self._norm(on["train_py_g2"])
    _measure(
        suite="moe_lb",
        router_params=len(off["engine_g2"]),
        total_weights=total_weights,
        predicted_ratio_engine_over_train_py_g2=total_weights / 2,
        lb_grad_norm_engine_g2=f"{norms['engine_g2']:.4e}",
        lb_grad_norm_train_py_g2=f"{norms['train_py_g2']:.4e}",
        lb_grad_norm_train_py_g1=f"{norms['train_py_g1']:.4e}",
        ratio_engine_over_train_py_g2=f"{norms['engine_g2'] / max(norms['train_py_g2'], 1e-30):.3f}",
        ratio_engine_over_train_py_g1=f"{norms['engine_g2'] / max(norms['train_py_g1'], 1e-30):.4f}",
        router_grad_rel_diff_engine_vs_train_py_g2=f"{total_rel:.3e}",
        control_rel_diff_lb_off=f"{control:.3e}",
    )


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _names = sys.argv[1:] or list(_SUITES)
  _loader = unittest.defaultTestLoader
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.TestSuite(_loader.loadTestsFromTestCase(globals()[name]) for name in _names)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
