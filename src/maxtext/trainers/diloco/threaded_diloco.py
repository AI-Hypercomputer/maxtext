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

"""Threaded (non-SPMD) streaming DiLoCo.

One Python thread per learner drives that learner's slice of the `diloco` mesh axis with the regular MaxText train
step, and a syncer runs the outer optimizer on the learners' colocated CPU devices. Every `steps_between_syncs`
steps each learner extracts one parameter fragment and hands it to the syncer; tau
(`num_communication_overlapping_steps`) steps later it applies the synced fragment. Transfers and the outer step run
off the accelerators' critical path, so the train step only pays for the extract and apply kernels.

The outer state starts from learner 0's initial parameters, as in SPMD DiLoCo. With
`threaded_diloco_replicate_outer_step`, every learner's CPU devices run the same deterministic outer step, so each
learner receives its result without a cross-host copy.

A failure in any thread aborts the shared transport, which stops every other thread within about a second; the
failure that caused the abort is the one re-raised. A run does not checkpoint or export the outer parameters.

All learners and the syncer live in one process, i.e. this needs a single controller such as Pathways.
"""

import collections
import concurrent.futures
import contextlib
import datetime
import queue
import threading
import traceback
from typing import Any, Callable

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp

from maxtext.common import metric_logger
from maxtext.common import profiler
from maxtext.common.goodput import GoodputEvent, RECORD_JOB_END_TIME, maybe_record_goodput, record_goodput
from maxtext.configs.types import ProfilerType
from maxtext.trainers.diloco.threaded_transport import ThreadedTransport, TransportAborted
from maxtext.trainers.diloco.utils import fragment_transfer
from maxtext.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator, get_streaming_schedule
from maxtext.utils import diloco_sharding
from maxtext.utils import exceptions
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils

# Mailbox key step of the initial outer-state fragments; sync fragments are keyed by (sync_step, fragment_idx).
_INIT_STEP = -1
# Config fields that scale with the number of devices and must be divided among the learners.
_PER_LEARNER_FIELDS = (
    "global_batch_size_to_train_on",
    "global_batch_size_to_load",
    "micro_batch_size_to_train_on",
    "global_batch_size_to_eval_on",
    "global_batch_size_to_load_eval",
    "micro_batch_size_to_eval_on",
    "num_target_devices",
)
# Syncer threads running outer steps. Different fragments can be in flight at once (a fragment's outer step can
# still be running when the next sync's fragments arrive); outer steps of one fragment never overlap.
_OUTER_STEP_WORKERS = 4
# Logging tasks a learner may queue before it waits for the logging thread to catch up.
_MAX_PENDING_LOG_TASKS = 64


def make_learner_config(config, learner_idx: int):
  """Returns the config a learner uses to train on its own slice, without the `diloco` mesh axis.

  `config.replace` does not re-run validation, so every field derived from the stripped `diloco` axis is updated
  here. The DiLoCo fragmentation fields (`num_diloco_fragments`, `diloco_bucketize_non_scanned`) are kept although
  `enable_streaming_diloco` is False: the learner builds its fragmenter from them.

  Raises:
    ValueError: A per-learner field is not divisible by the number of learners.
  """
  num_learners = config.num_diloco_replicas
  updates = {}
  for field in _PER_LEARNER_FIELDS:
    value = getattr(config, field)
    if not value:
      continue
    if value % num_learners:
      raise ValueError(f"Threaded DiLoCo splits {field}={value} among {num_learners} learners; it must be divisible.")
    updates[field] = value // num_learners
  keep = [axis != "diloco" for axis in config.mesh_axes]
  updates.update(
      mesh_axes=[axis for axis, k in zip(config.mesh_axes, keep) if k],
      ici_parallelism=[p for p, k in zip(config.ici_parallelism, keep) if k],
      dcn_parallelism=[p for p, k in zip(config.dcn_parallelism, keep) if k],
      logical_axis_rules=diloco_sharding.remove_mesh_axis_from_rules(config.logical_axis_rules),
      logical_axis_rules_for_eval=diloco_sharding.remove_mesh_axis_from_rules(config.logical_axis_rules_for_eval),
      enable_diloco=False,
      enable_streaming_diloco=False,
      num_data_replicas_per_process=num_learners,
      data_replica_index=learner_idx,
      run_name=f"{config.run_name}_learner_{learner_idx}",
  )
  if learner_idx != 0:
    # Outputs with fixed file names or process-wide state are written by learner 0 only; every learner still writes
    # its own TensorBoard run (named after `run_name`). ManagedMLDiagnostics is a process-wide singleton.
    updates.update(
        profiler=ProfilerType.NONE, gcs_metrics=False, metrics_file="", enable_wandb=False, managed_mldiagnostics=False
    )
  return config.replace(**updates)


def outer_step_targets(config) -> list[int]:
  """Learners whose colocated CPU devices run the outer step."""
  return list(range(config.num_diloco_replicas)) if config.threaded_diloco_replicate_outer_step else [0]


def sync_steps(config) -> list[int]:
  """Completed-step counts after which a fragment is synced (the SPMD streaming DiLoCo schedule)."""
  steps_between_syncs, _ = get_streaming_schedule(config)
  return [s for s in range(1, config.steps + 1) if s % steps_between_syncs == 0]


def fragment_for_sync_step(config, sync_step: int) -> int:
  steps_between_syncs, period = get_streaming_schedule(config)
  return (sync_step % period) // steps_between_syncs


def _zeros_like(fragment: fragment_transfer.TransferFragment) -> fragment_transfer.TransferFragment:
  # Not jitted: a jitted `zeros_like` has no data dependency and lands on a single default device.
  return {k: jnp.zeros(v.shape, v.dtype, device=v.sharding) for k, v in fragment.items()}


class OuterSyncer:
  """Averages learner fragments and runs the Nesterov outer step on colocated CPU devices.

  Messages from learner `i` arrive in order on `transport.to_syncer[i]`: learner 0 first sends its initial fragments,
  which seed the outer state of every target, then every learner sends one fragment per sync step. Once every
  learner's fragment for a sync step has arrived, a worker updates that fragment's outer state on every target's CPU
  mesh and sends each learner its result. A fragment's outer steps run in sync-step order, whatever order the
  learners' messages arrive in.
  """

  def __init__(self, config, cpu_meshes: list[jax.sharding.Mesh], transport: ThreadedTransport):
    self.config = config
    self.cpu_meshes = cpu_meshes
    self.transport = transport
    self.num_learners = len(cpu_meshes)
    self.targets = outer_step_targets(config)
    self.expected_syncs = len(sync_steps(config))
    self._outer = {}  # (fragment_idx, target) -> TransferFragment on cpu_meshes[target].
    self._trace = {}
    self._pending = collections.defaultdict(dict)  # (sync_step, fragment_idx) -> {learner_idx: fragment}.
    # Outer steps of one fragment must run in sync-step order; completed syncs wait in `_ready` for their turn.
    self._fragment_sync_steps = collections.defaultdict(list)
    for s in sync_steps(config):
      self._fragment_sync_steps[fragment_for_sync_step(config, s)].append(s)
    self._next_sync = collections.defaultdict(int)  # fragment_idx -> index into _fragment_sync_steps[fragment_idx].
    self._ready = collections.defaultdict(dict)  # fragment_idx -> {sync_step: learner fragments}.
    self._lock = threading.Lock()
    self._fragment_locks = {f: threading.Lock() for f in range(config.num_diloco_fragments)}
    self._completed = 0
    self._done = threading.Event()
    self._error: BaseException | None = None
    self._workers = concurrent.futures.ThreadPoolExecutor(_OUTER_STEP_WORKERS, thread_name_prefix="diloco_syncer")
    if not self.expected_syncs:
      self._done.set()

  def run(self) -> None:
    """Blocks until every sync step has been processed; re-raises the first failure."""
    with self._workers:
      for i in range(self.num_learners):
        threading.Thread(target=self._ingest, args=(i,), name=f"diloco_ingest_{i}", daemon=True).start()
      while not self._done.wait(timeout=1.0):
        if self.transport.aborted:
          try:
            self.transport.raise_if_aborted("while the syncer was waiting")
          except TransportAborted as e:
            self._fail(e)
    if self._error is not None:
      raise self._error

  def _fail(self, error: BaseException) -> None:
    with self._lock:
      if self._error is None:
        self._error = error
    self.transport.abort(cause=error)
    self._done.set()

  def _ingest(self, learner_idx: int) -> None:
    """Stores the initial fragments and dispatches an outer step once a sync step is complete.

    Returns after learner `learner_idx`'s last sync fragment, so that a finished learner's silence cannot time out.
    """
    received = 0
    try:
      while received < self.expected_syncs and not self._done.is_set():
        (step, fragment_idx), fragment = self.transport.to_syncer[learner_idx].get_next()
        if step == _INIT_STEP:
          for target in self.targets:  # One copy seeds every target, as SPMD DiLoCo broadcasts one outer state.
            mesh = self.cpu_meshes[target]
            outer = fragment if target == learner_idx else fragment_transfer.move_fragment(fragment, mesh)
            self._outer[(fragment_idx, target)] = outer
            self._trace[(fragment_idx, target)] = _zeros_like(outer)
          continue
        if step not in self._fragment_sync_steps.get(fragment_idx, ()):
          raise RuntimeError(f"Learner {learner_idx} sent fragment {fragment_idx} at step {step}, which is not a sync.")
        received += 1
        with self._lock:
          arrived = self._pending[(step, fragment_idx)]
          arrived[learner_idx] = fragment
          if len(arrived) < self.num_learners:
            continue
          del self._pending[(step, fragment_idx)]
          self._ready[fragment_idx][step] = [arrived[i] for i in range(self.num_learners)]
        self._workers.submit(self._drain, fragment_idx)
    except BaseException as e:  # pylint: disable=broad-exception-caught
      if not self._done.is_set():
        self._fail(e)

  def _drain(self, fragment_idx: int) -> None:
    """Runs fragment `fragment_idx`'s ready outer steps, strictly in sync-step order."""
    try:
      with self._fragment_locks[fragment_idx]:
        while not self._done.is_set():
          with self._lock:
            order = self._fragment_sync_steps[fragment_idx]
            if self._next_sync[fragment_idx] == len(order):
              return
            step = order[self._next_sync[fragment_idx]]
            learner_fragments = self._ready[fragment_idx].pop(step, None)
            if learner_fragments is None:  # An earlier sync of this fragment is still incomplete.
              return
            self._next_sync[fragment_idx] += 1
          self._outer_step(step, fragment_idx, learner_fragments)
    except BaseException as e:  # pylint: disable=broad-exception-caught
      self._fail(e)

  def _outer_step(self, step: int, fragment_idx: int, learner_fragments: list) -> None:
    """Updates fragment `fragment_idx`'s outer state on every target and sends the result to the learners."""
    for target in self.targets:
      mesh = self.cpu_meshes[target]
      aligned = [
          f if all(v.sharding.mesh == mesh for v in f.values()) else fragment_transfer.move_fragment(f, mesh)
          for f in learner_fragments
      ]
      key = (fragment_idx, target)
      self._outer[key], self._trace[key] = fragment_transfer.nesterov_outer_step(
          self._outer[key],
          self._trace[key],
          aligned,
          learning_rate=self.config.diloco_outer_lr,
          momentum=self.config.diloco_outer_momentum,
      )
      receivers = [target] if self.config.threaded_diloco_replicate_outer_step else range(self.num_learners)
      for learner in receivers:
        self.transport.to_learner[learner].put((step, fragment_idx), self._outer[key])
    with self._lock:
      self._completed += 1
      if self._completed == self.expected_syncs:
        self._done.set()


class Learner:
  """Trains one DiLoCo replica on its slice and exchanges fragments with the syncer."""

  def __init__(
      self,
      learner_idx: int,
      config,
      mesh: jax.sharding.Mesh,
      cpu_mesh: jax.sharding.Mesh,
      transport: ThreadedTransport,
      recorder: Any,
      train_step: Callable[..., Any],
      eval_step: Callable[..., Any],
      init_lock: threading.Lock,
  ):
    self.learner_idx = learner_idx
    self.global_config = config
    self.config = make_learner_config(config, learner_idx)
    self.mesh = mesh
    self.cpu_mesh = cpu_mesh
    self.transport = transport
    # Goodput is recorded once per job, by learner 0.
    self.recorder = recorder if learner_idx == 0 else None
    self.train_step = train_step
    self.eval_step = eval_step
    self.init_lock = init_lock
    self.tau = config.num_communication_overlapping_steps
    self.mailbox = transport.to_syncer[learner_idx]

  @contextlib.contextmanager
  def on_mesh(self, rules=None):
    """Mesh and logical-axis-rule context for dispatching work on this learner's slice."""
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules or self.config.logical_axis_rules):
      yield

  def run(self) -> None:
    """Initializes the learner, then trains for `config.steps` steps; any failure aborts the transport."""
    try:
      self._run()
    except BaseException as e:
      self.transport.abort(cause=e)
      raise

  def _run(self) -> None:
    """Body of `run`: sets up this learner's state, transfers and logger, then runs its train loop."""
    config = self.config
    prof = profiler.Profiler(config)  # Validates the profiler options before the expensive setup.
    with self.on_mesh():
      with self.init_lock:  # setup_train_loop is not safe to run concurrently.
        setup = train_utils.setup_train_loop(config, self.recorder, mesh=self.mesh)
      _, _, state_mesh_shardings, _, _, lr_schedule, _, data_loader, rampup_manager, eval_iterator, state = setup
      params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
          config, state_mesh_shardings
      )
      graphdef, state = nnx.split(state)
      p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
          config,
          graphdef,
          self.mesh,
          state,
          state_mesh_shardings,
          self.train_step,
          self.eval_step,
          eval_iterator,
          params_shardings,
      )
      if config.shard_optimizer_over_data:
        # Zero-1: move the state to the layout the train step was compiled for, as train_loop does. This must come
        # before FragmentTransfer is built, because it pins its apply outputs to the params' layout at build time.
        state = jax.device_put(state, state_mesh_shardings)
      params = nnx.state(state.model, nnx.Param)
      manipulator = FragmentedTreeManipulator.create(params, config)
      transfer = fragment_transfer.FragmentTransfer(manipulator, params, alpha=config.communication_overlapping_alpha)
      if self.learner_idx == 0:
        for f in range(manipulator.num_fragments):  # Seed the outer state with the initial parameters.
          self.mailbox.put((_INIT_STEP, f), fragment_transfer.move_fragment(transfer.extract(params, f), self.cpu_mesh))
      logger = metric_logger.MetricLogger(
          config=config, learning_rate_schedule=lr_schedule, log_prefix=f"[learner {self.learner_idx}] "
      )
      logger.write_setup_info_to_tensorboard(params)
      del params

    loop = _TrainLoop(
        self, state, p_train_step, p_eval_step, transfer, data_loader, rampup_manager, eval_iterator, logger
    )
    loop.run(prof)

  def apply_steps(self) -> list[int]:
    """Sync steps whose result is applied within this run."""
    return [s for s in sync_steps(self.global_config) if s + self.tau <= self.config.steps]


class _TrainLoop:
  """One learner's training loop, with a prefetch thread for synced fragments and a metrics-logging thread."""

  def __init__(
      self,
      learner: Learner,
      state,
      p_train_step,
      p_eval_step,
      transfer: fragment_transfer.FragmentTransfer,
      data_loader,
      rampup_manager,
      eval_iterator,
      logger: metric_logger.MetricLogger,
  ):
    self.learner = learner
    self.config = learner.config
    self.state = state
    self.p_train_step = p_train_step
    self.p_eval_step = p_eval_step
    self.transfer = transfer
    self.data_loader = data_loader
    self.rampup_manager = rampup_manager
    self.eval_iterator = eval_iterator
    self.logger = logger
    self.steps_between_syncs, _ = get_streaming_schedule(learner.global_config)
    # Holds up to tau synced fragments, already on the learner's devices, until their apply step.
    self.prefetched = queue.Queue(maxsize=max(1, learner.tau))
    self.log_futures = collections.deque()
    # is_training -> (completion time, step) of the last logged step; only touched on the logging thread.
    self._last_completion: dict[bool, tuple[datetime.datetime, int]] = {}
    # Each pool starts its thread on the first submit.
    name = f"diloco_learner_{learner.learner_idx}"
    self.log_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=f"{name}_log")
    self.prefetch_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=f"{name}_prefetch")

  def run(self, prof: profiler.Profiler) -> None:
    """Trains for `config.steps` steps; any failure aborts the transport so that the other threads stop too."""
    with self.log_pool, self.prefetch_pool:
      prefetch_future = self.prefetch_pool.submit(self._prefetch)
      try:
        self._train(prof, prefetch_future)
      except BaseException as e:
        self.learner.transport.abort(cause=e)
        raise
      finally:
        self._log(self.logger.flush_metrics_and_cleanup)
      self._check_logging(wait=True)

  def _prefetch(self) -> None:
    """Moves synced fragments onto the learner's devices ahead of their apply step."""
    learner = self.learner
    try:
      for sync_step in learner.apply_steps():
        key = (sync_step, fragment_for_sync_step(learner.global_config, sync_step))
        fragment = learner.transport.to_learner[learner.learner_idx].get(key)
        item = (key, fragment_transfer.move_fragment(fragment, learner.mesh))
        while True:
          try:
            self.prefetched.put(item, timeout=1.0)
            break
          except queue.Full:
            learner.transport.raise_if_aborted("while prefetching")
    except BaseException as e:
      learner.transport.abort(cause=e)
      raise

  def _next_prefetched(
      self, prefetch_future: concurrent.futures.Future
  ) -> tuple[tuple[int, int], fragment_transfer.TransferFragment]:
    """Waits for the next prefetched fragment, surfacing a prefetch failure or a transport abort."""
    while True:
      try:
        return self.prefetched.get(timeout=1.0)
      except queue.Empty:
        if prefetch_future.done():
          try:  # The prefetch thread may have queued its last fragment after the timeout and then finished.
            return self.prefetched.get_nowait()
          except queue.Empty:
            pass
          raise prefetch_future.exception() or RuntimeError("Prefetch ended early.")  # pylint: disable=raise-missing-from
        self.learner.transport.raise_if_aborted("while waiting for a synced fragment")

  def _params(self):
    return nnx.state(self.state.model, nnx.Param)

  def _train(self, prof: profiler.Profiler, prefetch_future: concurrent.futures.Future) -> None:
    """Train steps interleaved with fragment extraction at sync steps and application `tau` steps later.

    Steps are counted by this loop rather than read from the optimizer's step count (which SPMD DiLoCo uses); the
    two agree because every train step advances the optimizer by one.
    """
    config, learner = self.config, self.learner
    self._log(self._reset_step_timer, True)
    for step in range(config.steps):
      learner.transport.raise_if_aborted(f"before step {step}")
      prof.maybe_activate_profiler(step, self.state)
      with jax.profiler.StepTraceAnnotation(f"train_learner_{learner.learner_idx}", step_num=step):
        batch = self.data_loader.load_next_batch(rampup_manager=self.rampup_manager)
        with maybe_record_goodput(learner.recorder, GoodputEvent.STEP, step), learner.on_mesh():
          self.state, metrics = self.p_train_step(self.state, batch)
      completed = step + 1

      with learner.on_mesh():
        if completed % self.steps_between_syncs == 0:
          f = fragment_for_sync_step(learner.global_config, completed)
          fragment = self.transfer.extract(self._params(), f)
          learner.mailbox.put((completed, f), fragment_transfer.move_fragment(fragment, learner.cpu_mesh))
        if completed - learner.tau > 0 and (completed - learner.tau) % self.steps_between_syncs == 0:
          (sync_step, f), fragment = self._next_prefetched(prefetch_future)
          if sync_step != completed - learner.tau:
            raise RuntimeError(f"Expected the fragment synced at step {completed - learner.tau}, got {sync_step}.")
          nnx.update(self.state.model, self.transfer.apply(self._params(), f, fragment))

      # Every step, as in train_loop: the logger applies `log_period` itself and checks every loss for NaN/Inf.
      self._log(self._write_metrics, metrics, step, True)
      self._check_logging()
      # The same eval cadence as train_loop.
      if config.eval_interval > 0 and step >= config.eval_start_step:
        if (step - config.eval_start_step) % config.eval_interval == 0:
          self._eval()
      prof.maybe_deactivate_profiler(step, self.state)

  def _log(self, fn, *args) -> None:
    """Runs `fn` on the logging thread; its failures surface in the training loop."""
    self.log_futures.append(self.log_pool.submit(fn, *args))
    while len(self.log_futures) > _MAX_PENDING_LOG_TASKS:
      self.log_futures.popleft().result()

  def _check_logging(self, wait: bool = False) -> None:
    while self.log_futures and (wait or self.log_futures[0].done()):
      self.log_futures.popleft().result()

  def _reset_step_timer(self, is_training: bool) -> None:
    self._last_completion[is_training] = (datetime.datetime.now(), -1)

  def _write_metrics(self, metrics, step: int, is_training: bool) -> None:
    """Writes one step's metrics, timing the step by when its results are ready.

    Steps are dispatched asynchronously and metrics are synced on this thread rather than in the train loop, so the
    time between dispatches says nothing about step time; the time between completions does.
    """
    jax.block_until_ready(metrics)
    now = datetime.datetime.now()
    last_time, last_step = self._last_completion[is_training]
    self._last_completion[is_training] = (now, step)
    step_time = (now - last_time) / (step - last_step)  # Averaged over the steps since the last logged one.
    # The logger evaluates the LR schedule eagerly; keep that op on this learner's own devices.
    with jax.default_device(self.learner.mesh.devices.flat[0]), jax.set_mesh(self.learner.mesh):
      self.logger.buffer_and_write_metrics(metrics, step, step_time_delta=step_time, is_training=is_training)

  def _eval(self) -> None:
    """Evaluates this learner's current (inner) parameters on its share of the eval data."""
    config = self.config
    if hasattr(self.eval_iterator, "reset"):
      self.eval_iterator.reset()
    self._log(self.logger.reset_eval_metrics)
    self._log(self._reset_step_timer, False)
    data_sharding = sharding.get_input_data_sharding(config, self.learner.mesh, rules=config.logical_axis_rules_for_eval)
    for eval_step, batch in enumerate(self.eval_iterator):
      if 0 < config.eval_steps <= eval_step:
        break
      with self.learner.on_mesh(config.logical_axis_rules_for_eval):
        metrics = self.p_eval_step(self.state, jax.device_put(batch, data_sharding))
      self._log(self._write_metrics, metrics, eval_step, False)


def _root_cause(transport: ThreadedTransport, errors: list[tuple[str, BaseException]]) -> BaseException:
  """The failure that started the abort; every `TransportAborted` is a consequence of it."""
  if transport.cause is not None:
    return transport.cause
  return next((e for _, e in errors if not isinstance(e, TransportAborted)), errors[0][1])


def run_threaded_diloco(config, recorder: Any, train_step: Callable[..., Any], eval_step: Callable[..., Any]) -> None:
  """Runs threaded streaming DiLoCo: one learner thread per `diloco` slice plus the outer syncer.

  Raises:
    BaseException: The failure that stopped the run (the root cause, not another thread's `TransportAborted`).
      `exceptions.StopTraining` is not raised: as in train_loop, it ends the run gracefully.
  """
  if jax.process_count() != 1:
    raise ValueError(f"Threaded DiLoCo needs a single controller (e.g. Pathways), got {jax.process_count()} processes.")
  tpu_meshes = diloco_sharding.split_mesh_along_axis(maxtext_utils.get_mesh_from_config(config), "diloco")
  cpu_meshes = [colocated_python.colocated_cpu_devices(mesh) for mesh in tpu_meshes]
  transport = ThreadedTransport(len(tpu_meshes), config.threaded_diloco_transport_timeout_seconds)
  syncer = OuterSyncer(config, cpu_meshes, transport)
  init_lock = threading.Lock()
  # Built before any thread starts, so that an invalid learner config fails without starting the others.
  learners = [
      Learner(i, config, mesh, cpu_meshes[i], transport, recorder, train_step, eval_step, init_lock)
      for i, mesh in enumerate(tpu_meshes)
  ]
  max_logging.log(f"Threaded DiLoCo: {len(tpu_meshes)} learners, outer step on learners {syncer.targets}.")

  errors: list[tuple[str, BaseException]] = []
  train_utils.maybe_apply_dcn_throttling(config)
  try:
    with concurrent.futures.ThreadPoolExecutor(len(tpu_meshes), thread_name_prefix="diloco_learner") as pool:
      try:
        futures = [pool.submit(learner.run) for learner in learners]
        try:
          syncer.run()
        except BaseException as e:  # pylint: disable=broad-exception-caught
          transport.abort(cause=e)  # A no-op after a syncer failure; needed if e.g. the wait was interrupted.
          errors.append(("syncer", e))
        concurrent.futures.wait(futures)
        for i, f in enumerate(futures):
          if (error := f.exception()) is not None:
            errors.append((f"learner {i}", error))
      except BaseException as e:
        transport.abort(cause=e)  # E.g. a KeyboardInterrupt in this thread: stop the learners before joining them.
        raise
  finally:
    train_utils.maybe_cleanup_dcn_throttling(config)

  if not errors:
    record_goodput(recorder, RECORD_JOB_END_TIME)
    max_logging.log("Threaded DiLoCo: finished.")
    return
  root = _root_cause(transport, errors)
  per_thread = [(name, type(error).__name__) for name, error in errors]
  max_logging.log(f"Threaded DiLoCo: stopped by {type(root).__name__}; per thread: {per_thread}.")
  for name, error in errors:
    if error is not root and not isinstance(error, TransportAborted):
      details = "".join(traceback.format_exception(error)).rstrip()
      max_logging.log(f"Threaded DiLoCo: {name} also failed:\n{details}")
  if isinstance(root, exceptions.StopTraining):
    max_logging.log(f"Training stopped: {root}")
    record_goodput(recorder, RECORD_JOB_END_TIME)
    return
  raise root
