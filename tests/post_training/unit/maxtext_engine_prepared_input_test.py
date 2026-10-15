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

"""Tests for MaxTextTrainingEngine.prepare_fwd_bwd_input and wait_for_device.

A micro-batch prepared ahead (mapped and placed on the mesh while earlier work
still runs) must train exactly like the payload it came from: same compiled
kernel, same inputs, bit-identical weights and metrics. Waiting for the device
afterwards (what Tunix's `TrainerWorker.stage_fwd_bwd` does, to report whether
the transfer hid behind the device) must change nothing either.

`maxtext_engine_prepared_input_dropless_test.py` covers the dropless replay and
the final micro-batch's dropless program, and
`maxtext_engine_prepared_input_sharded_test.py` a four-device mesh with Tunix's
real model input.
"""
# pylint: disable=protected-access

import dataclasses
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import inflight_throttler
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax
import pytest

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]


class _Model(nnx.Module):

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))


@struct.dataclass(frozen=True, kw_only=True)
class _Payload(abstract_engine.TrainerPayload):
  token_ids: Any
  token_mask: Any
  metadata: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


def _payload(seed: int) -> _Payload:
  rng = np.random.default_rng(seed)
  return _Payload(
      token_ids=rng.normal(size=(2, 4)).astype(np.float32),
      token_mask=(rng.random((2, 4)) > 0.25).astype(np.float32),
      metadata={"seed": seed},
  )


def _loss_fn(model, tokens, mask, scale):
  # Depends on the data and on both weights, so any change to either shows.
  pred = model.weights[...][0] * tokens + model.weights[...][1] * tokens**2
  return (
      abstract_engine.WeightedMetric(
          unreduced_sum=jnp.sum(pred * mask) * scale,
          denominator=jnp.sum(mask),
      ),
      {},
  )


def _model_input(payload: _Payload) -> dict[str, Any]:
  # Tunix's shape: traced arrays plus a closed-over Python constant.
  return {"tokens": payload.token_ids, "mask": payload.token_mask, "scale": 2.0}


# Where a step's next first micro-batch is prepared, relative to the step's update.
_BEFORE_UPDATE = "before_update"  # a separate update call follows the last fwd_bwd
_AFTER_UPDATE = "after_update"  # the update runs inside the last fwd_bwd call (google/tunix#2719)


class PrepareFwdBwdInputTest(absltest.TestCase):

  def setUp(self):
    """A one-device mesh and a toy two-weight model, built without a checkpoint."""
    super().setUp()
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        model_name="llama3.1-8b",
        run_name="test_run",
        base_output_directory=self.create_tempdir().full_path,
        init_weights_seed=42,
        micro_batch_size_to_train_on=2,
        gradient_accumulation_steps=1,
        enable_dropout=False,
        record_internal_nn_metrics=False,
        enable_tensorboard=False,
        tensorboard_dir=self.create_tempdir().full_path,
        skip_jax_distributed_system=True,
        enable_checkpointing=False,
        profiler_steps=0,
    )
    self.mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)
    for patcher in (
        mock.patch.object(
            maxtext_engine.train_utils,
            "create_training_optimizer",
            return_value=(lambda step: jnp.array(0.001), optax.adam(0.05)),
        ),
        mock.patch.object(
            maxtext_engine.model_creation_utils,
            "from_pretrained",
            side_effect=lambda *a, **k: (_Model(), self.mesh),
        ),
    ):
      self.addCleanup(patcher.stop)
      patcher.start()

  def _engine(self, compiled: bool) -> maxtext_engine.MaxTextTrainingEngine:
    engine = maxtext_engine.MaxTextTrainingEngine(self.config)
    engine.with_loss_fn(_loss_fn)
    engine.with_gen_model_input_fn(_model_input)
    if compiled:
      engine.compile(_payload(0))
    return engine

  def _train(self, engine, payloads, prepare_at: set[int], wait: bool = False, order: str = _BEFORE_UPDATE):
    """Two steps of two micro-batches; micro-batches in `prepare_at` go in prepared.

    A step's next first micro-batch is prepared right after the step's last `fwd_bwd`, while that micro-batch
    and the update are still on the device: before the update with `order=_BEFORE_UPDATE`, after it with
    `_AFTER_UPDATE` (where Tunix's `TrainerWorker.stage_fwd_bwd` lands once the update runs inside the last
    train call). With `wait`, `wait_for_device` follows each prepare, as it does there too.
    """
    prepared = {}

    def prepare(i):
      if i not in prepare_at or i >= len(payloads):
        return
      prepared[i] = engine.prepare_fwd_bwd_input(payloads[i])
      if wait:
        queued = engine._throttler._inflight_queue.qsize()
        self.assertGreaterEqual(engine.wait_for_device(), 0.0)
        # The in-flight queue is left for the next dispatch, or a drain, to empty as before.
        self.assertEqual(engine._throttler._inflight_queue.qsize(), queued)

    prepare(0)
    metrics = []
    for step in range(2):
      for j in range(2):
        i = 2 * step + j
        engine.fwd_bwd(prepared.pop(i) if i in prepared else payloads[i])
      if order == _BEFORE_UPDATE:
        prepare(2 * step + 2)
      engine.update()
      if order == _AFTER_UPDATE:
        prepare(2 * step + 2)
      buffer = engine.get_metrics(clear_cache=True)
      metrics.append({k: np.asarray(v).tolist() for k, v in sorted(buffer.scalar_metrics.items())})
      metrics[-1]["loss"] = np.asarray(buffer.weighted_metrics["loss"].compute()).tolist()
    self.assertEqual(prepared, {})
    return np.asarray(engine.model.weights[...]), metrics

  def test_prepared_micro_batches_train_bit_identically(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    for compiled in (True, False):
      for order in (_BEFORE_UPDATE, _AFTER_UPDATE):
        with self.subTest(compiled=compiled, order=order):
          ref = self._engine(compiled)
          new = self._engine(compiled)
          with (
              mock.patch.object(new, "_compile_for_batch", wraps=new._compile_for_batch) as new_compiles,
              mock.patch.object(ref, "_compile_for_batch", wraps=ref._compile_for_batch) as ref_compiles,
          ):
            ref_w, ref_m = self._train(ref, payloads, prepare_at=set())
            new_w, new_m = self._train(new, payloads, prepare_at={0, 2}, order=order)
          np.testing.assert_array_equal(ref_w, new_w)
          self.assertEqual(ref_m, new_m)
          # No kernel was rebuilt for a prepared input.
          self.assertEqual(ref_compiles.call_count, new_compiles.call_count)
          self.assertEqual(new.train_step, 2)

  def test_waiting_for_the_device_changes_nothing(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    for compiled in (True, False):
      for order in (_BEFORE_UPDATE, _AFTER_UPDATE):
        with self.subTest(compiled=compiled, order=order):
          ref = self._engine(compiled)
          new = self._engine(compiled)
          with (
              mock.patch.object(new, "_compile_for_batch", wraps=new._compile_for_batch) as new_compiles,
              mock.patch.object(ref, "_compile_for_batch", wraps=ref._compile_for_batch) as ref_compiles,
              mock.patch.object(new._metrics_logger, "write_metrics", wraps=new._metrics_logger.write_metrics) as new_w,
              mock.patch.object(ref._metrics_logger, "write_metrics", wraps=ref._metrics_logger.write_metrics) as ref_w,
          ):
            ref_weights, ref_m = self._train(ref, payloads, prepare_at=set())
            new_weights, new_m = self._train(new, payloads, prepare_at={0, 2}, wait=True, order=order)
          np.testing.assert_array_equal(ref_weights, new_weights)
          self.assertEqual(ref_m, new_m)
          self.assertEqual(ref_compiles.call_count, new_compiles.call_count)
          # The same metrics writes, in the same order (the wait writes none).
          self.assertEqual(
              [c.args[0].id for c in ref_w.call_args_list],
              [c.args[0].id for c in new_w.call_args_list],
          )

  def test_wait_for_device_waits_on_every_queued_computation_and_keeps_the_queue(self):
    engine = self._engine(True)
    engine.fwd_bwd(_payload(1))
    engine.fwd_bwd(_payload(2))
    engine.update()
    queue = engine._throttler._inflight_queue
    queued = list(queue.queue)
    # The update queues its gradient norm with the step's metrics, after the micro-batches' losses.
    self.assertIsNotNone(queued[-1][1])
    waited_on = []
    block = jax.block_until_ready

    def spy(x):
      waited_on.append(x)
      return block(x)

    with mock.patch.object(inflight_throttler.jax, "block_until_ready", side_effect=spy):
      waited = engine.wait_for_device()
    self.assertIsInstance(waited, float)
    self.assertGreaterEqual(waited, 0.0)
    # Nothing dequeued: the next dispatch or drain finds the same queue.
    self.assertEqual(list(queue.queue), queued)
    # Every queued computation was waited on, in order: each micro-batch's loss and the update's gradient norm,
    # so a caller's wait covers the update.
    self.assertEqual([id(x) for x in waited_on], [id(leaves) for leaves, _ in queued])
    token = getattr(engine, "last_update_token", None)
    if token is not None:
      # With the readiness tokens of AI-Hypercomputer/maxtext#5571 the update's token is that same norm, so a
      # caller that waits on it after `wait_for_device` (Tunix's `TrainerWorker.stage_fwd_bwd`) does not wait again.
      self.assertIs(queued[-1][0][0], token)

  def test_the_comparison_sees_a_one_token_difference(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    changed = list(payloads)
    tokens = np.array(payloads[2].token_ids)
    tokens[0, 0] += 1e-3
    changed[2] = dataclasses.replace(payloads[2], token_ids=tokens)
    ref_w, _ = self._train(self._engine(True), payloads, prepare_at=set())
    new_w, _ = self._train(self._engine(True), changed, prepare_at={2})
    self.assertFalse(np.array_equal(ref_w, new_w))

  def test_staged_through_tunix_trainer_worker_trains_bit_identically(self):
    """The path Tunix's pipelined step boundary takes, through a real `TrainerWorker`.

    Each step's last micro-batch runs the update in the same call (`fwd_bwd(apply_optimizer=True)`,
    google/tunix#2719). With staging, that call returns at the update's dispatch (`wait_for_update=False`) and the
    next step's first micro-batch is staged right after it: `stage_fwd_bwd` calls `prepare_fwd_bwd_input` and
    `wait_for_device`, and `fwd_bwd_staged` trains what it returned.
    """
    from tunix.experimental.common import datatypes  # pylint: disable=import-outside-toplevel
    from tunix.experimental.worker import trainer_worker  # pylint: disable=import-outside-toplevel

    if not hasattr(trainer_worker.TrainerWorker, "stage_fwd_bwd"):
      self.skipTest("this Tunix has no TrainerWorker.stage_fwd_bwd (its pipelined step boundary)")

    payloads = [_payload(s) for s in (1, 2, 3, 4, 5, 6)]

    def run(stage: bool):
      engine = self._engine(True)
      worker = trainer_worker.TrainerWorker(trainer_factory=lambda: engine)
      worker.initialize()
      with (
          mock.patch.object(engine, "prepare_fwd_bwd_input", wraps=engine.prepare_fwd_bwd_input) as prepare,
          mock.patch.object(engine, "wait_for_device", wraps=engine.wait_for_device) as wait,
      ):
        staged_id = None
        for step in range(3):
          for j in range(2):
            i = 2 * step + j
            kwargs = {"apply_optimizer": True, "wait_for_update": not stage} if j == 1 else {}
            if staged_id is not None:
              resp = worker.fwd_bwd_staged(staged_id, **kwargs)
              staged_id = None
            else:
              resp = worker.fwd_bwd(datatypes.TrainRequest(request_id=f"r{i}", payload=payloads[i]), **kwargs)
            if j == 1:
              self.assertEqual(resp.metadata["train_step"], step + 1)
              if stage and i + 1 < len(payloads):
                staged = worker.stage_fwd_bwd(datatypes.TrainRequest(request_id=f"r{i + 1}", payload=payloads[i + 1]))
                # The engine's wait_for_device answered how long the device still ran.
                self.assertGreaterEqual(staged.metadata["device_wait_s"], 0.0)
                staged_id = f"r{i + 1}"
      return np.asarray(engine.model.weights[...]), prepare.call_count, wait.call_count

    ref_w, ref_prepared, ref_waits = run(stage=False)
    new_w, new_prepared, new_waits = run(stage=True)
    np.testing.assert_array_equal(ref_w, new_w)
    self.assertEqual((ref_prepared, new_prepared), (0, 2))
    self.assertEqual((ref_waits, new_waits), (0, 2))

  def test_preparing_places_the_traced_inputs_and_leaves_state_alone(self):
    engine = self._engine(True)
    engine.fwd_bwd(_payload(1))
    before = (
        engine.micro_step_count,
        engine.train_step,
        np.asarray(engine.model.weights[...]),
    )
    prepared = engine.prepare_fwd_bwd_input(_payload(2))
    self.assertIsInstance(prepared, maxtext_engine.PreparedFwdBwdInput)
    self.assertEqual(prepared.batch["scale"], 2.0)
    dynamic, _ = maxtext_engine._split_static_and_dynamic(prepared.batch)
    shardings = engine._batch_data_shardings(dynamic)
    for key in ("tokens", "mask"):
      self.assertIsInstance(prepared.batch[key], jax.Array)
      self.assertEqual(prepared.batch[key].sharding, shardings[key])
    np.testing.assert_array_equal(np.asarray(prepared.batch["tokens"]), _payload(2).token_ids)
    after = (
        engine.micro_step_count,
        engine.train_step,
        np.asarray(engine.model.weights[...]),
    )
    self.assertEqual(before[:2], after[:2])
    np.testing.assert_array_equal(before[2], after[2])


class _Metrics:

  def __init__(self):
    self.written = []

  def write_metrics(self, metrics, **kwargs):
    del kwargs
    self.written.append(metrics)


class InflightThrottlerBlockUntilReadyTest(absltest.TestCase):

  def _throttler(self):
    logger = _Metrics()
    config = types.SimpleNamespace(max_inflight_computations=2)
    return inflight_throttler.InflightThrottler(config, logger), logger

  def test_block_until_ready_neither_dequeues_nor_writes(self):
    throttler, logger = self._throttler()
    throttler.add_computation(jnp.ones(3) * 2, "m1")
    throttler.add_computation(jnp.ones(3) * 3, None)
    throttler.block_until_ready()
    self.assertEqual(throttler._inflight_queue.qsize(), 2)
    self.assertEqual(logger.written, [])
    for leaves, _ in list(throttler._inflight_queue.queue):
      self.assertTrue(all(leaf.is_ready() for leaf in leaves))

  def test_a_drain_after_it_writes_what_it_would_have(self):
    written = []
    for block_first in (False, True):
      throttler, logger = self._throttler()
      throttler.add_computation(jnp.ones(3), "m1")
      throttler.add_computation(jnp.ones(3), "m2")
      if block_first:
        throttler.block_until_ready()
      throttler.wait_for_all()
      written.append(logger.written)
      self.assertTrue(throttler._inflight_queue.empty())
    self.assertEqual(written[0], ["m1", "m2"])
    self.assertEqual(written[0], written[1])

  def test_block_until_ready_on_an_empty_queue_returns(self):
    throttler, logger = self._throttler()
    throttler.block_until_ready()
    self.assertEqual(logger.written, [])


if __name__ == "__main__":
  absltest.main()
