# Copyright 2023-2026 Google LLC
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

"""Unit tests for threaded (non-SPMD) streaming DiLoCo."""

import collections
import concurrent.futures
import contextlib
import datetime
import functools
import os
import queue
import random
import shutil
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from maxtext.configs import pyconfig
from maxtext.trainers.diloco import threaded_diloco
from maxtext.trainers.diloco.threaded_transport import ThreadedTransport, TransportAborted
from maxtext.trainers.diloco.utils import fragment_transfer
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils import exceptions
from maxtext.utils import train_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_test_config_path

_THREADED_ARGS = {
    "enable_diloco": True,
    "enable_streaming_diloco": True,
    "enable_threaded_diloco": True,
    "ici_diloco_parallelism": 2,
    "num_diloco_fragments": 3,
    "diloco_sync_period": 3,
    "enable_checkpointing": False,
    "base_num_decoder_layers": 2,
}


def _config(**overrides):
  return pyconfig.initialize(
      [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
      skip_jax_distributed_system=True,
      **{**_THREADED_ARGS, **overrides},
  )


class ThreadedTransportTest(unittest.TestCase):

  def test_delivers_by_key_and_buffers_early_messages(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=5)
    mailbox = transport.to_learner[0]
    mailbox.put((2, 1), "second")
    mailbox.put((1, 0), "first")
    self.assertEqual(mailbox.get((1, 0)), "first")
    self.assertEqual(mailbox.get((2, 1)), "second")

  def test_abort_wakes_up_waiters(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=60)
    errors = []

    def wait():
      try:
        transport.to_syncer[0].get((1, 0))
      except TransportAborted as e:
        errors.append(e)

    waiter = threading.Thread(target=wait)
    waiter.start()
    transport.abort()
    waiter.join(timeout=10)
    self.assertFalse(waiter.is_alive())
    self.assertEqual(len(errors), 1)

  def test_abort_keeps_the_first_cause(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=60)
    root, later = ValueError("root"), RuntimeError("later")
    transport.abort(cause=root)
    transport.abort(cause=later)
    self.assertIs(transport.cause, root)
    with self.assertRaisesRegex(TransportAborted, "ValueError: root") as ctx:
      transport.to_learner[0].get_next()
    self.assertIs(ctx.exception.__cause__, root)

  def test_times_out(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=0.1)
    start = time.monotonic()
    with self.assertRaises(TimeoutError):
      transport.to_learner[0].get((1, 0))
    self.assertLess(time.monotonic() - start, 0.6)  # Not rounded up to the 1 s abort poll.

  def test_timeout_bounds_the_whole_wait(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=0.5)
    mailbox = transport.to_learner[0]
    stop = threading.Event()

    def send_other_keys():  # Unrelated messages must not extend the wait for (1, 0).
      for i in range(40):
        if stop.wait(0.1):
          return
        mailbox.put((100 + i, 0), None)

    sender = threading.Thread(target=send_other_keys)
    sender.start()
    self.addCleanup(sender.join)
    self.addCleanup(stop.set)
    start = time.monotonic()
    with self.assertRaises(TimeoutError):
      mailbox.get((1, 0))
    self.assertLess(time.monotonic() - start, 2.0)


def _skip_unless_two_devices(test):
  # Two learners split every per-learner field, including num_target_devices, so these need >= 2 devices.
  if jax.device_count() < 2:
    test.skipTest("Needs 2 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=2).")


class ScheduleAndConfigTest(parameterized.TestCase):

  def test_sync_schedule_matches_streaming_diloco(self):
    config = _config(steps=10, num_diloco_fragments=3, diloco_sync_period=6)  # Two steps between syncs.
    self.assertEqual(threaded_diloco.sync_steps(config), [2, 4, 6, 8, 10])
    self.assertEqual([threaded_diloco.fragment_for_sync_step(config, s) for s in (2, 4, 6, 8)], [1, 2, 0, 1])

  @parameterized.parameters(True, False)
  def test_outer_step_targets(self, replicate):
    config = _config(threaded_diloco_replicate_outer_step=replicate)
    self.assertEqual(threaded_diloco.outer_step_targets(config), [0, 1] if replicate else [0])

  def test_learner_config(self):
    _skip_unless_two_devices(self)
    config = _config(
        per_device_batch_size=2,
        profiler="xplane",
        gcs_metrics=True,
        metrics_file="/tmp/metrics.txt",
        managed_mldiagnostics=True,
        managed_mldiagnostics_dir="/tmp/mldiag",
    )
    learner = threaded_diloco.make_learner_config(config, learner_idx=1)
    self.assertEqual(learner.global_batch_size_to_train_on, config.global_batch_size_to_train_on // 2)
    self.assertEqual(learner.global_batch_size_to_load, config.global_batch_size_to_load // 2)
    self.assertNotIn("diloco", learner.mesh_axes)
    self.assertEqual(len(learner.ici_parallelism), len(learner.mesh_axes))
    self.assertEqual(len(learner.dcn_parallelism), len(learner.mesh_axes))
    for rules in (learner.logical_axis_rules, learner.logical_axis_rules_for_eval):
      for _, physical in rules:
        self.assertNotIn("diloco", (physical,) if isinstance(physical, str) else (physical or ()))
    self.assertFalse(learner.enable_diloco)
    self.assertEqual((learner.num_data_replicas_per_process, learner.data_replica_index), (2, 1))
    self.assertEqual(learner.run_name, f"{config.run_name}_learner_1")
    self.assertEqual((learner.profiler, learner.gcs_metrics, learner.metrics_file), ("", False, ""))
    self.assertFalse(learner.managed_mldiagnostics)
    learner0 = threaded_diloco.make_learner_config(config, learner_idx=0)
    self.assertEqual(
        (learner0.profiler, learner0.gcs_metrics, learner0.metrics_file), ("xplane", True, "/tmp/metrics.txt")
    )
    self.assertTrue(learner0.managed_mldiagnostics)

  def test_learner_config_rejects_uneven_split(self):
    _skip_unless_two_devices(self)
    config = _config()
    uneven = config.replace(global_batch_size_to_train_on=config.global_batch_size_to_train_on + 1)
    with self.assertRaisesRegex(ValueError, "global_batch_size_to_train_on"):
      threaded_diloco.make_learner_config(uneven, learner_idx=0)

  @parameterized.named_parameters(
      ("needs_streaming", {"enable_streaming_diloco": False}, "enable_streaming_diloco"),
      ("needs_two_replicas", {"ici_diloco_parallelism": 1}, "at least 2"),
      # The id must not contain "checkpoint": the stack's CPU suite runs with -k "not checkpoint".
      ("saving_state", {"enable_checkpointing": True}, "checkpointing"),
      ("dataset", {"dataset_type": "c4_mlperf"}, "dataset_type"),
      (
          "colocated_input",
          {"colocated_python_data_input": True, "enable_single_controller": True},
          "colocated_python_data_input",
      ),
      ("overlap_of_a_whole_period", {"num_communication_overlapping_steps": 3}, "num_communication_overlapping_steps"),
      ("mllog", {"enable_mllog": True}, "enable_mllog"),
      (
          "no_diloco_mesh_axis",
          {"mesh_axes": ["data", "stage", "fsdp", "fsdp_transpose", "context", "tensor", "expert", "autoregressive"]},
          "'diloco' axis",
      ),
  )
  def test_invalid_configs(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      _config(**overrides)

  def test_overlap_shorter_than_a_period_is_accepted(self):
    config = _config(num_communication_overlapping_steps=2)  # Period 3.
    self.assertEqual(config.num_communication_overlapping_steps, 2)

  def test_tokamax_gmm_is_allowed(self):
    # Learners run the plain train step (no drjax vmap), so the SPMD DiLoCo restriction does not apply.
    self.assertTrue(_config(use_tokamax_gmm=True).use_tokamax_gmm)


class OuterSyncerTest(parameterized.TestCase):
  """Drives the syncer with hand-made fragments and checks it against the reference outer step."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    devices = np.array(jax.devices()[:4])  # Two learners of 2 devices each, as on CI's 4 CPU devices.
    self.meshes = [Mesh(devices[:2], ("fsdp",)), Mesh(devices[2:], ("fsdp",))]

  def _fragment(self, learner, value):
    return {"w": jax.device_put(jnp.full((8,), value, jnp.float32), NamedSharding(self.meshes[learner], P("fsdp")))}

  @parameterized.parameters(True, False)
  def test_outer_steps(self, replicate):
    config = _config(
        steps=2, num_diloco_fragments=2, diloco_sync_period=2, threaded_diloco_replicate_outer_step=replicate
    )
    transport = ThreadedTransport(num_learners=2, timeout_seconds=30)
    syncer = threaded_diloco.OuterSyncer(config, self.meshes, transport)
    self.assertEqual(syncer.expected_syncs, 2)

    initial = 1.0
    init_step = threaded_diloco._INIT_STEP  # pylint: disable=protected-access
    for f in range(2):  # Learner 0 alone seeds the outer state of every target.
      transport.to_syncer[0].put((init_step, f), self._fragment(0, initial))
    learner_values = {1: (0.5, 0.7), 2: (0.2, 0.6)}  # sync_step -> per-learner fragment value
    for step, values in learner_values.items():
      for learner, value in enumerate(values):
        transport.to_syncer[learner].put(
            (step, threaded_diloco.fragment_for_sync_step(config, step)), self._fragment(learner, value)
        )
    syncer.run()
    for (_, target), trace_fragment in syncer._trace.items():  # pylint: disable=protected-access
      self.assertEqual(trace_fragment["w"].sharding, NamedSharding(self.meshes[target], P("fsdp")))

    outer, trace = {1: {"w": jnp.full((8,), initial)}, 0: {"w": jnp.full((8,), initial)}}, {}
    for step, values in learner_values.items():
      f = threaded_diloco.fragment_for_sync_step(config, step)
      outer[f], trace[f] = fragment_transfer.nesterov_outer_step(
          outer[f],
          trace.get(f, {"w": jnp.zeros(8)}),
          [{"w": jnp.full((8,), v)} for v in values],
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
      )
      for learner in range(2):
        result = transport.to_learner[learner].get((step, f))
        np.testing.assert_allclose(result["w"], outer[f]["w"], rtol=1e-6)
        expected_mesh = self.meshes[learner] if replicate else self.meshes[0]
        self.assertEqual(result["w"].sharding, NamedSharding(expected_mesh, P("fsdp")))

  def test_interleaved_arrivals_and_repeated_syncs(self):
    """Each fragment syncs four times (trace carry-over); learners send concurrently with random delays."""
    config = _config(steps=8, num_diloco_fragments=2, diloco_sync_period=2)  # One step between syncs.
    transport = ThreadedTransport(num_learners=2, timeout_seconds=30)
    syncer = threaded_diloco.OuterSyncer(config, self.meshes, transport)
    steps = threaded_diloco.sync_steps(config)
    self.assertEqual(steps, list(range(1, 9)))
    init_step = threaded_diloco._INIT_STEP  # pylint: disable=protected-access

    def value(step, learner):
      return 0.9 - 0.05 * step + 0.1 * learner

    def send(learner, seed):
      rng = random.Random(seed)
      if learner == 0:
        for f in range(2):
          transport.to_syncer[0].put((init_step, f), self._fragment(0, 1.0))
      for step in steps:
        time.sleep(rng.uniform(0, 0.02))
        fragment = threaded_diloco.fragment_for_sync_step(config, step)
        transport.to_syncer[learner].put((step, fragment), self._fragment(learner, value(step, learner)))

    senders = [threading.Thread(target=send, args=(i, i)) for i in range(2)]
    for sender in senders:
      sender.start()
    syncer.run()
    for sender in senders:
      sender.join()

    outer, trace = {f: {"w": jnp.ones(8)} for f in range(2)}, {f: {"w": jnp.zeros(8)} for f in range(2)}
    for step in steps:
      f = threaded_diloco.fragment_for_sync_step(config, step)
      outer[f], trace[f] = fragment_transfer.nesterov_outer_step(
          outer[f],
          trace[f],
          [{"w": jnp.full((8,), value(step, i))} for i in range(2)],
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
      )
      for learner in range(2):
        np.testing.assert_allclose(transport.to_learner[learner].get((step, f))["w"], outer[f]["w"], rtol=1e-6)
    # The ingest threads exit once they have their learner's last fragment instead of waiting out the timeout.
    deadline = time.monotonic() + 5
    while any(t.name.startswith("diloco_ingest_") for t in threading.enumerate()) and time.monotonic() < deadline:
      time.sleep(0.05)
    self.assertFalse([t.name for t in threading.enumerate() if t.name.startswith("diloco_ingest_")])

  def test_outer_steps_of_a_fragment_run_in_sync_order(self):
    config = _config(steps=4, num_diloco_fragments=2, diloco_sync_period=2)  # Fragments 1, 0, 1, 0 at steps 1-4.
    transport = ThreadedTransport(num_learners=2, timeout_seconds=30)
    syncer = threaded_diloco.OuterSyncer(config, self.meshes, transport)
    for f in range(2):
      transport.to_syncer[0].put((threaded_diloco._INIT_STEP, f), self._fragment(0, 1.0))  # pylint: disable=protected-access
    values = {1: 0.8, 2: 0.7, 3: 0.4, 4: 0.3}
    for step in (3, 4, 1, 2):  # Every fragment's second sync arrives before its first.
      for learner in range(2):
        f = threaded_diloco.fragment_for_sync_step(config, step)
        transport.to_syncer[learner].put((step, f), self._fragment(learner, values[step]))
    syncer.run()
    outer, trace = {f: {"w": jnp.ones(8)} for f in range(2)}, {f: {"w": jnp.zeros(8)} for f in range(2)}
    for step in (1, 2, 3, 4):
      f = threaded_diloco.fragment_for_sync_step(config, step)
      outer[f], trace[f] = fragment_transfer.nesterov_outer_step(
          outer[f],
          trace[f],
          [{"w": jnp.full((8,), values[step])}] * 2,
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
      )
      np.testing.assert_allclose(transport.to_learner[0].get((step, f))["w"], outer[f]["w"], rtol=1e-6)

  def test_unexpected_sync_message_fails(self):
    config = _config(steps=2, num_diloco_fragments=2, diloco_sync_period=2)
    transport = ThreadedTransport(num_learners=2, timeout_seconds=30)
    syncer = threaded_diloco.OuterSyncer(config, self.meshes, transport)
    transport.to_syncer[0].put((2, 1), self._fragment(0, 0.0))  # Step 2 syncs fragment 0, not 1.
    with self.assertRaisesRegex(RuntimeError, "not a sync"):
      syncer.run()

  def test_failure_aborts_transport(self):
    config = _config(steps=1, num_diloco_fragments=2, diloco_sync_period=2)
    transport = ThreadedTransport(num_learners=2, timeout_seconds=30)
    syncer = threaded_diloco.OuterSyncer(config, self.meshes, transport)
    for learner in range(2):  # Sync fragment without initial outer state: the outer step must fail loudly.
      transport.to_syncer[learner].put((1, 1), self._fragment(learner, 0.0))
    with self.assertRaises(KeyError):
      syncer.run()
    self.assertTrue(transport.aborted)
    self.assertIsInstance(transport.cause, KeyError)


def _bare_train_loop(**attrs):
  loop = threaded_diloco._TrainLoop.__new__(threaded_diloco._TrainLoop)  # pylint: disable=protected-access
  loop._last_completion = {}  # pylint: disable=protected-access
  loop.log_futures = collections.deque()
  for name, value in attrs.items():
    setattr(loop, name, value)
  return loop


class TrainLoopTest(parameterized.TestCase):
  # pylint: disable=protected-access

  def test_step_time_is_averaged_over_steps_since_the_last_logged_one(self):
    loop = _bare_train_loop(learner=types.SimpleNamespace(mesh=Mesh(np.array(jax.devices()[:1]), ("x",))))
    loop.logger = mock.Mock()
    t0 = datetime.datetime(2026, 1, 1)
    with mock.patch.object(threaded_diloco.datetime, "datetime") as fake_datetime:
      fake_datetime.now.side_effect = [t0, t0 + datetime.timedelta(seconds=5), t0 + datetime.timedelta(seconds=11)]
      loop._reset_step_timer(True)
      loop._write_metrics({}, 0, True)
      loop._write_metrics({}, 3, True)  # Steps 1 and 2 were not logged.
    step_times = [c.kwargs["step_time_delta"] for c in loop.logger.buffer_and_write_metrics.call_args_list]
    self.assertEqual(step_times, [datetime.timedelta(seconds=5), datetime.timedelta(seconds=2)])

  def _loop_for_train(self, steps=5, eval_interval=-1, eval_start_step=0, transport=None):
    """A train loop without syncs whose logging calls and evals are recorded in order in `events`."""
    transport = transport or ThreadedTransport(num_learners=1, timeout_seconds=5)
    learner = types.SimpleNamespace(
        transport=transport, learner_idx=0, recorder=None, tau=0, on_mesh=contextlib.nullcontext, global_config=None
    )
    config = types.SimpleNamespace(steps=steps, eval_interval=eval_interval, eval_start_step=eval_start_step)
    loop = _bare_train_loop(
        learner=learner,
        config=config,
        steps_between_syncs=steps + 1,  # No sync within the run.
        data_loader=mock.Mock(),
        rampup_manager=None,
        state="state",
    )
    loop.p_train_step = mock.Mock(side_effect=lambda state, batch: (state, {"step": "metrics"}))
    events = []
    loop._write_metrics = mock.Mock()
    loop._reset_step_timer = mock.Mock()

    def log(fn, *args):
      events.append(("write", args[1]) if fn is loop._write_metrics else ("other", None))

    loop._log = log
    loop._check_logging = lambda wait=False: None
    loop._eval = lambda: events.append(("eval", None))
    return loop, events

  def test_metrics_are_written_every_step(self):
    loop, events = self._loop_for_train(steps=5)
    loop._train(mock.Mock(), prefetch_future=None)
    self.assertEqual([step for name, step in events if name == "write"], [0, 1, 2, 3, 4])

  @parameterized.named_parameters(
      ("from_step_0", 2, 0, [0, 2, 4]),
      ("delayed_start", 2, 3, [3]),
  )
  def test_eval_cadence_matches_train_loop(self, eval_interval, eval_start_step, expected_steps):
    loop, events = self._loop_for_train(steps=5, eval_interval=eval_interval, eval_start_step=eval_start_step)
    loop._train(mock.Mock(), prefetch_future=None)
    eval_after = []
    for i, (name, _) in enumerate(events):
      if name == "eval":
        eval_after.append(next(step for n, step in reversed(events[:i]) if n == "write"))
    self.assertEqual(eval_after, expected_steps)

  def test_stops_at_the_next_step_after_an_abort(self):
    transport = ThreadedTransport(num_learners=1, timeout_seconds=5)
    transport.abort(cause=RuntimeError("elsewhere"))
    loop, _ = self._loop_for_train(steps=5, transport=transport)
    with self.assertRaisesRegex(TransportAborted, "elsewhere"):
      loop._train(mock.Mock(), prefetch_future=None)
    loop.p_train_step.assert_not_called()

  def test_next_prefetched_takes_a_fragment_queued_after_the_timeout(self):
    item = ((3, 1), {"w": None})
    prefetched = mock.Mock()
    prefetched.get.side_effect = queue.Empty  # The timed get misses the item, which the prefetch thread
    prefetched.get_nowait.return_value = item  # queued just before it finished.
    future = concurrent.futures.Future()
    future.set_result(None)
    loop = _bare_train_loop(prefetched=prefetched, learner=types.SimpleNamespace(transport=None))
    self.assertEqual(loop._next_prefetched(future), item)

  def test_next_prefetched_surfaces_a_prefetch_failure(self):
    future = concurrent.futures.Future()
    future.set_exception(ValueError("prefetch failed"))
    prefetched = mock.Mock()
    prefetched.get.side_effect = queue.Empty
    prefetched.get_nowait.side_effect = queue.Empty
    loop = _bare_train_loop(prefetched=prefetched, learner=types.SimpleNamespace(transport=None))
    with self.assertRaisesRegex(ValueError, "prefetch failed"):
      loop._next_prefetched(future)

  def test_pending_logging_is_bounded(self):
    done = concurrent.futures.Future()
    done.set_result(None)
    loop = _bare_train_loop(log_pool=mock.Mock(submit=mock.Mock(return_value=done)))
    for _ in range(3 * threaded_diloco._MAX_PENDING_LOG_TASKS):
      loop._log(print)
    self.assertLessEqual(len(loop.log_futures), threaded_diloco._MAX_PENDING_LOG_TASKS)


def _train_argv(output_dir, *extra):
  return (
      None,
      get_test_config_path(),
      f"base_output_directory={output_dir}",
      "run_name=threaded_diloco_unit",
      "enable_diloco=true",
      "enable_streaming_diloco=true",
      "enable_threaded_diloco=true",
      "ici_diloco_parallelism=2",
      "dataset_type=synthetic",
      "per_device_batch_size=1",
      "max_target_length=64",
      "base_emb_dim=32",
      "base_num_decoder_layers=2",
      "base_mlp_dim=64",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "head_dim=16",
      "vocab_size=256",
      "enable_checkpointing=false",
      "enable_goodput_recording=false",
      "monitor_goodput=false",
      "skip_jax_distributed_system=true",
      *extra,
  )


# Transport timeout of the failure tests. Without the abort paths under test, a failing run waits this long for a
# message and then ends with TimeoutError/TransportAborted, which fails the test instead of hanging it.
_FAILURE_TIMEOUT_SECONDS = 60
_PROMPT_SECONDS = 30  # A healthy abort ends the whole run, setup and compilation included, well within this.


class EndToEndTest(parameterized.TestCase):
  """Runs train.py with threaded DiLoCo on the CPU devices and checks how failures end the run."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:  # Two learners of >= 2 devices each (CI runs on 4 CPU devices).
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    self.output_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)
    self.wall_seconds = None

  def _train(self, *extra):
    """Runs train.py with the test argv plus `extra`, recording the wall time in `self.wall_seconds`."""
    argv = _train_argv(
        self.output_dir,
        "num_diloco_fragments=3",
        "diloco_sync_period=3",
        "num_communication_overlapping_steps=1",
        "steps=6",
        f"threaded_diloco_transport_timeout_seconds={_FAILURE_TIMEOUT_SECONDS}",
        *extra,
    )
    start = time.monotonic()
    try:
      train_main(argv)
    finally:
      self.wall_seconds = time.monotonic() - start

  def _patch_train_for_learner(self, learner_idx, error):
    original = threaded_diloco._TrainLoop._train  # pylint: disable=protected-access

    def train(loop, prof, prefetch_future):
      if loop.learner.learner_idx == learner_idx:
        raise error
      return original(loop, prof, prefetch_future)

    return mock.patch.object(threaded_diloco._TrainLoop, "_train", train)  # pylint: disable=protected-access

  def test_learner_failing_during_setup_stops_the_run(self):
    original = threaded_diloco.FragmentedTreeManipulator.create
    error = RuntimeError("injected setup failure")

    def create(params, config, *args, **kwargs):
      if config.data_replica_index == 1:
        raise error
      return original(params, config, *args, **kwargs)

    with mock.patch.object(threaded_diloco.FragmentedTreeManipulator, "create", create):
      with self.assertRaises(RuntimeError) as ctx:
        self._train()
    self.assertIs(ctx.exception, error)
    self.assertLess(self.wall_seconds, _PROMPT_SECONDS)

  def test_invalid_profiler_options_stop_the_run(self):
    with self.assertRaisesRegex(ValueError, "Profiling requested"):
      self._train("profiler=xplane", "skip_first_n_steps_for_profiler=6")
    self.assertLess(self.wall_seconds, _PROMPT_SECONDS)

  def test_learner_failure_in_the_loop_is_the_raised_error(self):
    error = RuntimeError("learner 1 failed")
    with self._patch_train_for_learner(1, error):
      with self.assertRaises(RuntimeError) as ctx:
        self._train()
    self.assertIs(ctx.exception, error)  # Not learner 0's TransportAborted.
    self.assertLess(self.wall_seconds, _PROMPT_SECONDS)

  def test_syncer_failure_is_the_raised_error(self):
    error = RuntimeError("outer step failed")

    def outer_step(*args, **kwargs):
      raise error

    with mock.patch.object(threaded_diloco.fragment_transfer, "nesterov_outer_step", outer_step):
      with self.assertRaises(RuntimeError) as ctx:
        self._train()
    self.assertIs(ctx.exception, error)  # Not a learner's TransportAborted.
    self.assertLess(self.wall_seconds, _PROMPT_SECONDS)

  def test_stop_training_ends_the_run_gracefully(self):
    with self._patch_train_for_learner(1, exceptions.StopTraining("target reached")):
      self._train()  # Does not raise, as in train_loop.
    self.assertLess(self.wall_seconds, _PROMPT_SECONDS)


_DECAY = 0.9


def _shift(learner_idx):
  return 0.01 * (learner_idx + 1)


def _decay_and_shift(params, shift):
  """The oracle's replacement parameter update."""
  return jax.tree.map(lambda x: x * _DECAY + shift, params)


def _shardings(tree):
  return jax.tree.map(lambda x: x.sharding, tree)


def _host_leaves(tree):
  """`[(keystr, numpy array)]` of a parameter tree, in flattening order."""
  leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
  return [(jax.tree_util.keystr(path), np.asarray(jax.device_get(leaf))) for path, leaf in leaves]


class OracleTest(parameterized.TestCase):
  """Checks what threaded DiLoCo computes against a host-side reference of streaming DiLoCo.

  The real train step runs, but its parameter update is replaced by `params * _DECAY + _shift(learner)`, which is
  distinct per learner and easy to reproduce on the host. Everything else (when fragments are extracted, the outer
  state's seed, the outer step, routing, when and how synced fragments are applied) is the threaded trainer's own.
  The reference follows the SPMD streaming schedule (diloco.py `build_streaming_diloco_train_step`), with each
  fragment's parameter elements taken from `FragmentedTreeManipulator.get_flat_fragment`.
  """

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:  # Two learners of >= 2 devices each (CI runs on 4 CPU devices).
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    self.output_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)

  def _run(self, extra, learner_1_offset=0.0):
    """Trains with the replaced parameter update; returns per-learner initial/final params and fragment layouts.

    `learner_1_offset` is added to learner 1's parameters before anything reads them, so that the learners start
    from different points.
    """
    initial, final, layouts, configs = {}, {}, {}, {}
    real_jit = train_utils.jit_train_and_eval_step

    def jit_train_and_eval_step(config, graphdef, mesh, state, *args, **kwargs):
      p_train, p_eval = real_jit(config, graphdef, mesh, state, *args, **kwargs)
      idx = config.data_replica_index
      configs[idx] = config
      shift = _shift(idx)
      update = {}
      if idx == 1 and learner_1_offset:
        params = nnx.state(state.model, nnx.Param)
        offset = jax.jit(lambda p: jax.tree.map(lambda x: x + learner_1_offset, p), out_shardings=_shardings(params))
        nnx.update(state.model, offset(params))  # The learner's own `state`: seen by the outer seed and training.

      def train(state, batch):
        params = nnx.state(state.model, nnx.Param)
        if not update:  # First step: the params after any re-layout, before any update.
          initial[idx] = _host_leaves(params)
          layouts[idx] = (threaded_diloco.FragmentedTreeManipulator.create(params, config), jax.tree.structure(params))
          update["fn"] = jax.jit(functools.partial(_decay_and_shift, shift=shift), out_shardings=_shardings(params))
        new_params = jax.block_until_ready(update["fn"](params))
        state, metrics = p_train(state, batch)
        nnx.update(state.model, new_params)
        return state, metrics

      return train, p_eval

    original_train = threaded_diloco._TrainLoop._train  # pylint: disable=protected-access

    def train_loop(loop, prof, prefetch_future):
      original_train(loop, prof, prefetch_future)
      final[loop.learner.learner_idx] = _host_leaves(loop._params())  # pylint: disable=protected-access

    with mock.patch.object(threaded_diloco.train_utils, "jit_train_and_eval_step", jit_train_and_eval_step):
      with mock.patch.object(threaded_diloco._TrainLoop, "_train", train_loop):  # pylint: disable=protected-access
        train_main(_train_argv(self.output_dir, "threaded_diloco_transport_timeout_seconds=60", *extra))
    return initial, final, layouts, configs

  @parameterized.named_parameters(
      ("overlap_1_replicated_outer_step", 1, 0.0, True, False, (), 0.0),
      ("overlap_0_alpha_outer_step_on_learner_0", 0, 0.5, False, False, (), 0.0),
      ("overlap_2_alpha_bucketized", 2, 0.25, True, True, (), 0.0),
      (
          "zero1",
          1,
          0.0,
          True,
          False,
          ("shard_optimizer_over_data=true", "ici_fsdp_parallelism=1", "ici_data_parallelism=-1"),
          0.0,
      ),
      # The outer state starts from learner 0's parameters, as SPMD DiLoCo broadcasts one copy.
      ("learners_start_apart", 1, 0.0, True, False, (), 0.05),
  )
  def test_matches_host_reference(self, tau, alpha, replicate, bucketize, extra, learner_1_offset):
    steps, num_fragments, steps_between_syncs = 8, 3, 2  # Period 6: fragment 1 syncs at steps 2 and 8.
    period = num_fragments * steps_between_syncs
    initial, final, layouts, configs = self._run(
        (
            f"steps={steps}",
            f"num_diloco_fragments={num_fragments}",
            f"diloco_sync_period={period}",
            f"num_communication_overlapping_steps={tau}",
            f"communication_overlapping_alpha={alpha}",
            f"threaded_diloco_replicate_outer_step={str(replicate).lower()}",
            f"diloco_bucketize_non_scanned={str(bucketize).lower()}",
            *extra,
        ),
        learner_1_offset=learner_1_offset,
    )
    self.assertEqual(sorted(final), [0, 1])
    keys = [k for k, _ in initial[0]]
    for (key, value0), (_, value1) in zip(initial[0], initial[1]):
      np.testing.assert_allclose(value1, value0 + learner_1_offset, rtol=1e-6, atol=1e-6, err_msg=f"start of {key}")

    # Element ids of each fragment, as indices into the flattened parameter vector.
    manipulator, treedef = layouts[0]
    sizes = [v.size for _, v in initial[0]]
    offsets = np.cumsum([0] + sizes[:-1])
    ids = [np.arange(o, o + v.size, dtype=np.int32).reshape(v.shape) for o, (_, v) in zip(offsets, initial[0])]
    id_tree = jax.tree.unflatten(treedef, ids)
    fragment_ids = {}
    for f in range(num_fragments):
      flat = manipulator.get_flat_fragment(id_tree, f)
      fragment_ids[f] = np.concatenate([np.asarray(v).reshape(-1) for v in flat.values()])
    covered = np.sort(np.concatenate(list(fragment_ids.values())))
    np.testing.assert_array_equal(covered, np.arange(sum(sizes)), err_msg="The fragments must partition the params.")

    lr, momentum = np.float32(configs[0].diloco_outer_lr), np.float32(configs[0].diloco_outer_momentum)
    theta = {i: np.concatenate([v.reshape(-1).astype(np.float32) for _, v in initial[i]]) for i in range(2)}
    outer = {f: theta[0][i].copy() for f, i in fragment_ids.items()}
    trace = {f: np.zeros_like(o) for f, o in outer.items()}
    synced, applies = {}, 0
    for completed in range(1, steps + 1):
      for i in range(2):
        theta[i] = theta[i] * np.float32(_DECAY) + np.float32(_shift(i))
      if completed % steps_between_syncs == 0:
        f = (completed % period) // steps_between_syncs
        mean = (theta[0][fragment_ids[f]] + theta[1][fragment_ids[f]]) / np.float32(2)
        pseudo_grad = outer[f] - mean
        trace[f] = momentum * trace[f] + pseudo_grad
        outer[f] = outer[f] - lr * (pseudo_grad + momentum * trace[f])
        synced[completed] = (f, outer[f].copy())
      if completed - tau > 0 and (completed - tau) % steps_between_syncs == 0:
        f, value = synced[completed - tau]
        for i in range(2):
          current = theta[i][fragment_ids[f]]
          theta[i][fragment_ids[f]] = np.float32(alpha) * current + np.float32(1 - alpha) * value
        applies += 1
    self.assertGreaterEqual(applies, 3)

    for i in range(2):
      self.assertEqual([k for k, _ in final[i]], keys)
      got = np.concatenate([v.reshape(-1).astype(np.float32) for _, v in final[i]])
      np.testing.assert_allclose(got, theta[i], rtol=1e-5, atol=1e-5, err_msg=f"learner {i}")


if __name__ == "__main__":
  unittest.main()
