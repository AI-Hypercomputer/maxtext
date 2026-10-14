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

"""CPU-only tests of bench_driver.py's STAGE_BEFORE_UPDATE switch (NumPy only; no Tunix, no JAX, no Pathways).

`main()` runs end to end against stand-ins for what it imports: Pathways, JAX, the Tunix example modules (argv
parsing, the algorithm, the warm-up payload), the batch assembler, the tokenizer, Tunix's `TrainerWorker` (which, like
the real one, hands `request.payload` to the engine's `fwd_bwd` and calls the engine's `update`) and the MaxText
training engine, which records every call with a digest of the micro-batch it trains. The micro-batches are the
driver's own `realistic_payload` of a small warm-up-style payload, so the regular and the final one differ.

  * Off (unset or `off`): the engine sees exactly the calls of the original loop, per step MICRO_STEPS `fwd_bwd` (the
    regular micro-batch, then the final one, which has fully padded rows, with apply_optimizer=True), one `update`,
    the drain and the metrics read. Neither hook is called, nothing STAGE_BEFORE_UPDATE is logged, and an engine
    without the hooks works.
  * On: the same `fwd_bwd` sequence (request ids, micro-batch digests, apply_optimizer flags) and the same STEP_LOSS /
    STEP_METRICS lines as off. Step k+1's first micro-batch arrives prepared, made by one `prepare_fwd_bwd_input`
    between step k's last `fwd_bwd` and its `update`, with one `wait_for_device` in between; one `update` per step;
    one STAGE_BEFORE_UPDATE line per staged boundary, carrying the wait the engine returned.
  * Refusal: an engine without either hook stops the driver right after the worker initializes, before the compile;
    a worker that runs the update inside the last `fwd_bwd` (Tunix's fused update) stops it at the first boundary,
    before anything is staged; and a value other than on/off stops it before Pathways is initialized.

The checks are functions that return a list of failures, and `CheckerTest` shows each one failing on a wrong run.
`stand_in_modules` mirrors `main()`'s imports, so a new import in `main()` needs a stand-in there.
"""

import dataclasses
import hashlib
import logging
import os
import re
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_driver  # pylint: disable=g-import-not-at-top,wrong-import-position

pytestmark = pytest.mark.cpu_only

ROWS, BUDGET, LAYERS, TOP_K, EXPERTS = 4, 256, 4, 2, 16
# Every environment variable the driver reads, cleared for each run so that the caller's environment cannot leak in.
DRIVER_ENV = (
    "TRAINER_ARGV",
    "ORCH_ARGV",
    "STEPS",
    "MICRO_STEPS",
    "STAGE_BEFORE_UPDATE",
    "LOGP_GATES",
    "PAYLOAD",
    "ROUTING",
    "ROUTING_SCALE",
    "ROUTING_TARGETS",
    "PAD_FRACTION",
    "FINAL_PAD_FRACTION",
    "PAYLOAD_SEED",
    "EP_SIZE",
    "ADVANTAGES",
    "MODEL_DIR",
)


@dataclasses.dataclass(frozen=True)
class StandInPayload:
  """The fields of Tunix's packed RLTrainerPayload that `realistic_payload` reads and writes."""

  completion_ids: np.ndarray
  segment_ids: np.ndarray
  segment_positions: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  old_per_token_logps: np.ndarray | None
  rollout_per_token_logps: np.ndarray | None
  routed_experts: np.ndarray | None


def warmup_payload():
  """A warm-up-style micro-batch: two short segments at the start of each row, the rest padding, routing UNSET."""
  seg = np.zeros((ROWS, BUDGET), np.int32)
  cm = np.zeros((ROWS, BUDGET), np.int32)
  adv = np.zeros((ROWS, BUDGET), np.float32)
  for s in range(2):
    seg[:, s * 40 : (s + 1) * 40] = s + 1
    cm[:, s * 40 + 16 : (s + 1) * 40] = 1
    adv[:, s * 40 + 16 : (s + 1) * 40] = 1.0 if s else -1.0
  return StandInPayload(
      completion_ids=np.where(seg > 0, 7, 0).astype(np.int32),
      segment_ids=seg,
      segment_positions=np.tile(np.arange(BUDGET, dtype=np.int32), (ROWS, 1)),
      completion_mask=cm,
      advantages=adv,
      old_per_token_logps=None,
      rollout_per_token_logps=np.zeros((ROWS, BUDGET), np.float32),
      routed_experts=np.full((ROWS, BUDGET, LAYERS, TOP_K), bench_driver.UNSET_ROUTED_EXPERT, np.int16),
  )


def digest(payload):
  """Content digest of a micro-batch: every field's name, shape, dtype and bytes."""
  h = hashlib.md5()
  for f in dataclasses.fields(payload):
    v = getattr(payload, f.name)
    h.update(f.name.encode())
    if v is not None:
      a = np.ascontiguousarray(np.asarray(v))
      h.update(f"{a.shape}{a.dtype.str}".encode())
      h.update(a.tobytes())
  return h.hexdigest()[:12]


class Prepared:
  """What the stand-in engine's `prepare_fwd_bwd_input` returns (MaxText's `PreparedFwdBwdInput`)."""

  def __init__(self, payload_digest):
    self.digest = payload_digest
    self.used = False


@dataclasses.dataclass
class WeightedMetric:
  unreduced_sum: np.ndarray
  denominator: np.ndarray
  eps: float | None = None
  min_denom: float | None = None


def fully_padded_rows(payload):
  """Rows with no real token: the final micro-batch has FINAL_PAD_FRACTION of them, a regular one has none."""
  return int((np.asarray(payload.segment_ids) == 0).all(axis=1).sum())


class EngineWithoutHooks:
  """Stand-in for MaxText's training engine without the staging methods (as on #5701's base branch).

  Records each call in `calls`: ("fwd_bwd", step, micro index, micro-batch digest, prepared, apply_optimizer),
  ("update", step, micro-batches accumulated), ("drain", step), ("get_metrics", step), and each micro-batch's fully
  padded row count in `pad_rows` (by digest). Its metrics are a function of the micro-batches the step trained, so a
  different micro-batch changes the driver's STEP_LOSS line.
  """

  def __init__(self, calls):
    self.calls = calls
    self._config = types.SimpleNamespace(num_experts=EXPERTS, inhomogeneous_layer_cycle_interval=4)
    self._throttler = types.SimpleNamespace(wait_for_all=lambda: calls.append(("drain", self.step)))
    self.step = 0
    self.micro = []
    self.metrics = None
    self.pad_rows = {}

  @property
  def train_step(self):
    """As MaxText's engine: the number of updates applied."""
    return self.step

  def fwd_bwd(self, payload, **kwargs):
    """Trains one micro-batch: a payload, or what `prepare_fwd_bwd_input` made of one."""
    if isinstance(payload, Prepared):
      if payload.used:
        raise AssertionError("a prepared input was trained twice")
      payload.used = True
      d, prepared = payload.digest, True
    else:
      d, prepared = digest(payload), False
      self.pad_rows[d] = fully_padded_rows(payload)
    self.calls.append(("fwd_bwd", self.step, len(self.micro), d, prepared, kwargs.get("apply_optimizer")))
    self.micro.append(d)

  def update(self, **kwargs):
    """Applies the step: its metrics are a function of the micro-batches it trained."""
    del kwargs
    self.calls.append(("update", self.step, len(self.micro)))
    if not self.micro:  # As MaxText's engine: nothing accumulated, nothing applied.
      return self.step
    sums = np.array([int(d[:6], 16) % 1000 for d in self.micro], np.float64)
    dens = np.array([1 + int(d[6:8], 16) % 3 for d in self.micro], np.float64)
    norm = float(int(hashlib.md5("".join(self.micro).encode()).hexdigest()[:6], 16))
    self.metrics = types.SimpleNamespace(
        weighted_metrics={"loss": WeightedMetric(sums, dens)}, scalar_metrics={"gradient_norm": np.float64(norm)}
    )
    self.micro = []
    self.step += 1
    return self.step

  def get_metrics(self):
    self.calls.append(("get_metrics", self.step))
    return self.metrics


class Engine(EngineWithoutHooks):
  """The engine with the two methods of AI-Hypercomputer/maxtext#5718."""

  def prepare_fwd_bwd_input(self, payload):
    d = digest(payload)
    self.pad_rows[d] = fully_padded_rows(payload)
    self.calls.append(("prepare", self.step, len(self.micro), d))
    return Prepared(d)

  def wait_for_device(self):
    self.calls.append(("wait_for_device", self.step, len(self.micro)))
    return 0.25


class EngineWithPrepareOnly(EngineWithoutHooks):
  prepare_fwd_bwd_input = Engine.prepare_fwd_bwd_input


class TrainerWorker:
  """Stand-in for Tunix's TrainerWorker without a fused update: `fwd_bwd` hands `request.payload` to the engine as
  is, `update` and `get_metrics` call the engine's."""

  def __init__(self, run, trainer_factory, **kwargs):
    del kwargs  # worker_id, logps_chunk_size, logps_micro_batch_size, execution_context
    self._trainer_factory = trainer_factory
    self._trainer = None
    self.request_ids = []
    run.worker = self

  def initialize(self):
    self._trainer = self._trainer_factory()
    self._trainer.calls.append(("initialize",))

  def with_loss_fn(self, loss_fn, has_aux=False):
    del loss_fn, has_aux

  def with_gen_model_input_fn(self, gen_model_input_fn):
    del gen_model_input_fn

  def compile(self, dummy_data=None):
    del dummy_data
    self._trainer.calls.append(("compile",))

  def fwd_bwd(self, request, apply_optimizer=False, **kwargs):
    self.request_ids.append(request.request_id)
    self._trainer.fwd_bwd(request.payload, apply_optimizer=apply_optimizer, **kwargs)

  def update(self, **kwargs):
    return self._trainer.update(**kwargs)

  def get_metrics(self):
    return self._trainer.get_metrics()

  def stop(self):
    self._trainer.calls.append(("stop",))


class FusedUpdateTrainerWorker(TrainerWorker):
  """A TrainerWorker with Tunix's fused update (google/tunix#2719): `fwd_bwd(apply_optimizer=True)` also runs
  `update()`."""

  def fwd_bwd(self, request, apply_optimizer=False, **kwargs):
    super().fwd_bwd(request, apply_optimizer=apply_optimizer, **kwargs)
    if apply_optimizer:
      self.update()


@dataclasses.dataclass
class TrainRequest:
  request_id: str
  payload: object
  metadata: dict


def _package(name, **attrs):
  m = types.ModuleType(name)
  m.__path__ = []  # A package, so that `from <name> import <submodule>` resolves through sys.modules.
  m.__dict__.update(attrs)
  return m


def stand_in_modules(run, engine_cls, worker_cls):
  """sys.modules entries for everything `main()` imports, recording into `run`."""
  o_args = types.SimpleNamespace(
      tokenizer_path="stand-in",
      model_id="stand-in",
      max_prompt_length=64,
      max_response_length=BUDGET - 64,
      max_seq_token_per_tpu=BUDGET,
      max_segments_per_packed_row=2,
      trainer_fsdp=4,
      trainer_dp=1,
      trainer_expert=1,
      num_generations=2,
      mini_batch_size=2,
      trainer_warmup_routed_experts=True,
      trainer_warmup_routed_layers=LAYERS,
      trainer_warmup_routed_top_k=TOP_K,
  )
  algo = types.SimpleNamespace(
      train_micro_batch_size=ROWS, loss_fn=lambda: None, build_gen_model_input_fn=lambda pad_id, eos_id: None
  )
  t_args = types.SimpleNamespace(worker_id="stand-in", compute_logps_chunk_size=0, compute_logps_micro_batch_size=None)
  tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token="<eos>", vocab_size=1000)
  leaves = {
      "tunix.experimental.examples.common.run_trainer_node": {
          "_parse_args": lambda argv: t_args,
          "_create_maxtext_trainer_factory": lambda args: (lambda: engine_cls(run.calls), None),
      },
      "tunix.experimental.examples.deepswe_dist.run_deepswe_dist": {
          "_parse_args": lambda argv: o_args,
          "_build_algo": lambda args: algo,
      },
      "tunix.experimental.examples.deepswe_dist.trainer_warmup": {
          "routed_experts_shape_from_config": lambda path: (LAYERS, TOP_K),
          "build_warmup_payload": lambda **kwargs: warmup_payload(),
      },
      "tunix.experimental.orchestrator.batch_assembly": {
          "BatchConfig": types.SimpleNamespace,
          "create_batch_assembler": lambda **kwargs: types.SimpleNamespace(pad_id=0, eos_id=0),
          "describe_payload": lambda payload: "stand-in micro-batch",
      },
      "tunix.experimental.common.datatypes": {"TrainRequest": TrainRequest},
      "tunix.experimental.worker.trainer_worker": {"TrainerWorker": lambda **kwargs: worker_cls(run, **kwargs)},
  }
  mods = {
      "pathwaysutils": _package("pathwaysutils", initialize=lambda: run.calls.append(("pathways_initialize",))),
      "jax": _package("jax", device_count=lambda: 4, device_get=lambda x: x, block_until_ready=lambda x: x),
      "transformers": _package(
          "transformers", AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
      ),
  }
  for name, attrs in leaves.items():
    mods[name] = _package(name, **attrs)
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
      parent = ".".join(parts[:i])
      mods.setdefault(parent, _package(parent))
      setattr(mods[parent], parts[i], mods[".".join(parts[: i + 1])])
  return mods


class _ListHandler(logging.Handler):

  def __init__(self, lines):
    super().__init__(level=logging.INFO)
    self.lines = lines

  def emit(self, record):
    self.lines.append(record.getMessage())


def run_driver(env, engine_cls=Engine, driver=bench_driver, worker_cls=TrainerWorker):
  """Runs `driver.main()` against the stand-ins with STEPS=3, MICRO_STEPS=4 and `env` on top (a value of None
  unsets the variable). Returns the run: calls (in order), worker.request_ids, log (messages), error (or None)."""
  run = types.SimpleNamespace(calls=[], worker=None, log=[], error=None)
  environ = {k: v for k, v in os.environ.items() if k not in DRIVER_ENV}
  environ.update(TRAINER_ARGV="--worker_id=stand-in", ORCH_ARGV="--model_id=stand-in", STEPS="3", MICRO_STEPS="4")
  for k, v in env.items():
    if v is None:
      environ.pop(k, None)
    else:
      environ[k] = v
  root = logging.getLogger()
  handler, level = _ListHandler(run.log), root.level
  root.addHandler(handler)
  root.setLevel(logging.INFO)
  try:
    with (
        mock.patch.dict(os.environ, environ, clear=True),
        mock.patch.dict(sys.modules, stand_in_modules(run, engine_cls, worker_cls)),
    ):
      try:
        driver.main()
      except Exception as e:  # pylint: disable=broad-except
        run.error = e
  finally:
    root.removeHandler(handler)
    root.setLevel(level)
  return run


def _fwd_bwd(run):
  return [c for c in run.calls if c[0] == "fwd_bwd"]


def _metric_lines(run):
  return [line for line in run.log if line.startswith(("STEP_LOSS ", "STEP_METRICS "))]


def original_loop_failures(run, steps, micro_steps):
  """How `run` departs from the original loop's call sequence (an empty list when it does not)."""
  if run.error is not None:
    return [f"the driver raised {run.error!r}"]
  fb = _fwd_bwd(run)
  if len(fb) != steps * micro_steps:
    return [f"{len(fb)} fwd_bwd calls, want {steps * micro_steps}"]
  regular, final = fb[0][3], fb[micro_steps - 1][3]
  failures = []
  if micro_steps > 1 and regular == final:
    failures.append("the regular and the final micro-batch are identical, so the check cannot tell them apart")
  pad_rows = run.worker._trainer.pad_rows  # pylint: disable=protected-access
  if pad_rows.get(final, 0) == 0 or (micro_steps > 1 and pad_rows.get(regular) != 0):
    failures.append(
        f"the last micro-batch must be the final payload (fully padded rows: {pad_rows.get(final)}) and the others "
        f"the regular one (fully padded rows: {pad_rows.get(regular)})"
    )
  want = [("pathways_initialize",), ("initialize",), ("compile",)]
  for s in range(steps):
    for m in range(micro_steps):
      last = m == micro_steps - 1
      want.append(("fwd_bwd", s, m, final if last else regular, False, last))
    want += [("update", s, micro_steps), ("drain", s + 1), ("get_metrics", s + 1)]
  want.append(("stop",))
  if run.calls != want:
    i = next((i for i, (a, b) in enumerate(zip(run.calls, want)) if a != b), min(len(run.calls), len(want)))
    got_i = run.calls[i] if i < len(run.calls) else None
    want_i = want[i] if i < len(want) else None
    failures.append(f"engine calls differ from the original loop at call {i}: got {got_i}, want {want_i}")
  want_ids = [f"bench_{s}_{m}" for s in range(steps) for m in range(micro_steps)]
  if run.worker.request_ids != want_ids:
    failures.append(f"request ids {run.worker.request_ids}, want {want_ids}")
  if len(_metric_lines(run)) != 2 * steps:
    failures.append(f"{len(_metric_lines(run))} STEP_LOSS/STEP_METRICS lines, want {2 * steps}")
  stage_lines = [line for line in run.log if "STAGE_BEFORE_UPDATE" in line]
  if stage_lines:
    failures.append(f"STAGE_BEFORE_UPDATE lines with the switch off: {stage_lines}")
  return failures


def _index(calls, *head):
  """Index of the first call that starts with `head`, or -1."""
  return next((i for i, c in enumerate(calls) if c[: len(head)] == head), -1)


def _without_staging(run):
  """The run's calls without prepare_fwd_bwd_input / wait_for_device, and without fwd_bwd's `prepared` field."""
  return [c[:4] + c[5:] if c[0] == "fwd_bwd" else c for c in run.calls if c[0] not in ("prepare", "wait_for_device")]


def staged_failures(on, off, steps, micro_steps):
  """How the switch-on run `on` departs from the switch-off run `off` of the same configuration (empty = correct)."""
  if on.error is not None or off.error is not None:
    return [f"a run raised: on {on.error!r}, off {off.error!r}"]
  failures = []
  on_fb, off_fb = _fwd_bwd(on), _fwd_bwd(off)
  # Per call: step, micro index, micro-batch digest, apply_optimizer.
  if [c[1:4] + c[5:] for c in on_fb] != [c[1:4] + c[5:] for c in off_fb]:
    failures.append("the fwd_bwd sequence (step, micro index, micro-batch, apply_optimizer) differs from off")
  if on.worker.request_ids != off.worker.request_ids:
    failures.append("the request ids differ from off")
  want_prepared = [(s, 0) for s in range(1, steps)]
  got_prepared = [(c[1], c[2]) for c in on_fb if c[4]]
  if got_prepared != want_prepared:
    failures.append(f"prepared inputs trained at (step, micro) {got_prepared}, want {want_prepared}")
  updates = [(c[1], c[2]) for c in on.calls if c[0] == "update"]
  if updates != [(s, micro_steps) for s in range(steps)]:
    failures.append(f"updates (step, micro-batches) {updates}, want one per step after its {micro_steps}")
  prepares = [c for c in on.calls if c[0] == "prepare"]
  if len(prepares) != steps - 1:
    failures.append(f"{len(prepares)} prepare_fwd_bwd_input calls, want {steps - 1}")
  waits = [c[1] for c in on.calls if c[0] == "wait_for_device"]
  if waits != list(range(steps - 1)):
    failures.append(f"wait_for_device at steps {waits}, want one per staged boundary {list(range(steps - 1))}")
  first_of = {c[1]: c[3] for c in off_fb if c[2] == 0}
  for s in range(steps - 1):
    order = (
        _index(on.calls, "fwd_bwd", s, micro_steps - 1),
        _index(on.calls, "prepare", s),
        _index(on.calls, "wait_for_device", s),
        _index(on.calls, "update", s),
        _index(on.calls, "fwd_bwd", s + 1, 0),
    )
    if not -1 < order[0] < order[1] < order[2] < order[3] < order[4]:
      failures.append(f"step {s}: want last fwd_bwd < prepare < wait_for_device < update < next fwd_bwd, got {order}")
    if order[1] >= 0 and on.calls[order[1]][3] != first_of.get(s + 1):
      failures.append(
          f"step {s} prepared {on.calls[order[1]][3]}, off trains {first_of.get(s + 1)} first at step {s + 1}"
      )
  if _without_staging(on) != _without_staging(off):
    failures.append("the calls other than prepare_fwd_bwd_input / wait_for_device differ from off")
  if len(_metric_lines(off)) != 2 * steps or _metric_lines(on) != _metric_lines(off):
    failures.append("the STEP_LOSS / STEP_METRICS lines differ from off (or are missing)")
  if sum(line.startswith("STAGE_BEFORE_UPDATE on:") for line in on.log) != 1:
    failures.append("want exactly one 'STAGE_BEFORE_UPDATE on:' line")
  stage = [i for i, line in enumerate(on.log) if line.startswith("STAGE_BEFORE_UPDATE step=")]
  completed = [i for i, line in enumerate(on.log) if line.startswith("completed step: ")]
  if len(stage) != steps - 1 or len(completed) != steps:
    failures.append(f"{len(stage)} STAGE_BEFORE_UPDATE step lines and {len(completed)} completed lines")
  else:
    for s, i in enumerate(stage):
      want = rf"STAGE_BEFORE_UPDATE step={s} staged=bench_{s + 1}_0 stage_s=\d+\.\d{{3}} device_wait_s=0\.250"
      in_place = completed[s] < i < completed[s + 1]
      if not re.fullmatch(want, on.log[i]) or not in_place:
        failures.append(f"STAGE_BEFORE_UPDATE line {s} is wrong or out of place: {on.log[i]!r}")
  return failures


class OffTest(unittest.TestCase):

  def test_unset_runs_the_original_loop(self):
    self.assertEqual(original_loop_failures(run_driver({"STAGE_BEFORE_UPDATE": None}), 3, 4), [])

  def test_off_runs_the_original_loop(self):
    self.assertEqual(original_loop_failures(run_driver({"STAGE_BEFORE_UPDATE": "off"}), 3, 4), [])

  def test_off_needs_no_hooks(self):
    run = run_driver({"STAGE_BEFORE_UPDATE": "off"}, engine_cls=EngineWithoutHooks)
    self.assertEqual(original_loop_failures(run, 3, 4), [])


class OnTest(unittest.TestCase):

  def _check(self, steps, micro_steps):
    env = {"STEPS": str(steps), "MICRO_STEPS": str(micro_steps)}
    off = run_driver({**env, "STAGE_BEFORE_UPDATE": "off"})
    on = run_driver({**env, "STAGE_BEFORE_UPDATE": "on"})
    self.assertEqual(original_loop_failures(off, steps, micro_steps), [])
    self.assertEqual(staged_failures(on, off, steps, micro_steps), [])
    return on, off

  def test_same_micro_batches_order_and_one_update_per_step(self):
    on, _ = self._check(3, 4)
    self.assertEqual([c[1] for c in on.calls if c[0] == "prepare"], [0, 1])

  def test_one_micro_batch_per_step_stages_the_final_one(self):
    on, off = self._check(3, 1)
    final = _fwd_bwd(off)[0][3]
    self.assertEqual([c[3] for c in on.calls if c[0] == "prepare"], [final, final])

  def test_last_step_stages_nothing(self):
    on, off = self._check(1, 3)
    self.assertEqual([c for c in on.calls if c[0] in ("prepare", "wait_for_device")], [])
    self.assertEqual(on.calls, off.calls)


class RefusalTest(unittest.TestCase):

  def test_engine_without_hooks_is_refused_before_the_compile(self):
    run = run_driver({"STAGE_BEFORE_UPDATE": "on"}, engine_cls=EngineWithoutHooks)
    self.assertIsInstance(run.error, RuntimeError)
    for word in ("prepare_fwd_bwd_input", "wait_for_device", "AI-Hypercomputer/maxtext#5718", "EngineWithoutHooks"):
      self.assertIn(word, str(run.error))
    self.assertEqual(run.calls, [("pathways_initialize",), ("initialize",)])

  def test_engine_with_one_hook_is_refused(self):
    run = run_driver({"STAGE_BEFORE_UPDATE": "on"}, engine_cls=EngineWithPrepareOnly)
    self.assertIsInstance(run.error, RuntimeError)
    self.assertIn("lacks ['wait_for_device']", str(run.error))
    self.assertEqual(run.calls, [("pathways_initialize",), ("initialize",)])

  def test_fused_update_worker_is_refused_at_the_first_boundary(self):
    run = run_driver({"STAGE_BEFORE_UPDATE": "on"}, worker_cls=FusedUpdateTrainerWorker)
    self.assertIsInstance(run.error, RuntimeError)
    self.assertIn("google/tunix#2719", str(run.error))
    self.assertEqual([c[0] for c in run.calls if c[0] in ("prepare", "wait_for_device")], [])
    self.assertEqual(run.calls[-1], ("update", 0, 4))  # The worker's own update, inside step 0's last fwd_bwd.

  def test_fused_update_worker_runs_with_the_switch_off(self):
    run = run_driver({"STAGE_BEFORE_UPDATE": "off"}, worker_cls=FusedUpdateTrainerWorker)
    self.assertIsNone(run.error)
    # The worker's update inside the last fwd_bwd, then the driver's own update(), which has nothing to apply.
    updates = [c[1:] for c in run.calls if c[0] == "update"]
    self.assertEqual(updates, [(0, 4), (1, 0), (1, 4), (2, 0), (2, 4), (3, 0)])

  def test_other_values_are_refused_before_pathways(self):
    for value in ("yes", "ON", "1", ""):
      run = run_driver({"STAGE_BEFORE_UPDATE": value})
      self.assertIsInstance(run.error, ValueError, value)
      self.assertIn("STAGE_BEFORE_UPDATE must be on or off", str(run.error))
      self.assertEqual(run.calls, [], value)


class CheckerTest(unittest.TestCase):
  """Each check fails on a run that is wrong in the way it guards against."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.off = run_driver({"STAGE_BEFORE_UPDATE": "off"})
    cls.on = run_driver({"STAGE_BEFORE_UPDATE": "on"})

  def test_original_loop_check_rejects_a_staged_run(self):
    failures = original_loop_failures(self.on, 3, 4)
    self.assertTrue(any("engine calls differ" in f for f in failures), failures)
    self.assertTrue(any("STAGE_BEFORE_UPDATE lines" in f for f in failures), failures)

  def test_original_loop_check_rejects_the_wrong_micro_batch(self):
    regular = _fwd_bwd(self.off)[0][3]
    run = types.SimpleNamespace(**vars(self.off))
    run.calls = [c[:3] + (regular,) + c[4:] if c[:3] == ("fwd_bwd", 1, 3) else c for c in self.off.calls]
    failures = original_loop_failures(run, 3, 4)
    self.assertTrue(any("engine calls differ from the original loop at call" in f for f in failures), failures)

  def test_original_loop_check_rejects_identical_micro_batches(self):
    regular = _fwd_bwd(self.off)[0][3]
    run = types.SimpleNamespace(**vars(self.off))
    run.calls = [c[:3] + (regular,) + c[4:] if c[0] == "fwd_bwd" else c for c in self.off.calls]
    failures = original_loop_failures(run, 3, 4)
    self.assertTrue(any("cannot tell them apart" in f for f in failures), failures)

  def test_staged_check_rejects_a_run_that_never_stages(self):
    failures = staged_failures(self.off, self.off, 3, 4)
    self.assertTrue(any("prepared inputs trained" in f for f in failures), failures)
    self.assertTrue(any("prepare_fwd_bwd_input calls" in f for f in failures), failures)

  def test_staged_check_rejects_a_different_micro_batch(self):
    run = types.SimpleNamespace(**vars(self.on))
    run.calls = [("fwd_bwd", 1, 0, "0" * 12, True, False) if c[:3] == ("fwd_bwd", 1, 0) else c for c in self.on.calls]
    failures = staged_failures(run, self.off, 3, 4)
    self.assertTrue(any("fwd_bwd sequence" in f for f in failures), failures)

  def test_original_loop_check_rejects_swapped_payloads(self):
    regular, final = _fwd_bwd(self.off)[0][3], _fwd_bwd(self.off)[3][3]
    swap = {regular: final, final: regular}
    run = types.SimpleNamespace(**vars(self.off))
    run.calls = [c[:3] + (swap[c[3]],) + c[4:] if c[0] == "fwd_bwd" else c for c in self.off.calls]
    failures = original_loop_failures(run, 3, 4)
    self.assertTrue(any("must be the final payload" in f for f in failures), failures)

  def test_staged_check_rejects_an_extra_wait(self):
    run = types.SimpleNamespace(**vars(self.on))
    i_update = _index(self.on.calls, "update", 2)
    run.calls = self.on.calls[:i_update] + [("wait_for_device", 2, 4)] + self.on.calls[i_update:]
    failures = staged_failures(run, self.off, 3, 4)
    self.assertTrue(any("wait_for_device at steps" in f for f in failures), failures)

  def test_staged_check_rejects_a_wrong_logged_wait(self):
    run = types.SimpleNamespace(**vars(self.on))
    run.log = [line.replace("device_wait_s=0.250", "device_wait_s=0.000") for line in self.on.log]
    failures = staged_failures(run, self.off, 3, 4)
    self.assertTrue(any("is wrong or out of place" in f for f in failures), failures)

  def test_staged_check_rejects_staging_after_the_update(self):
    run = types.SimpleNamespace(**vars(self.on))
    calls = list(self.on.calls)
    i_prep = next(i for i, c in enumerate(calls) if c[0] == "prepare" and c[1] == 0)
    i_update = next(i for i, c in enumerate(calls) if c[0] == "update" and c[1] == 0)
    calls.insert(i_update, calls.pop(i_prep))  # update now precedes prepare
    calls.insert(i_update, calls.pop(i_prep))  # and wait_for_device
    run.calls = calls
    failures = staged_failures(run, self.off, 3, 4)
    self.assertTrue(any("step 0: want last fwd_bwd < prepare" in f for f in failures), failures)


if __name__ == "__main__":
  unittest.main()
