# Copyright 2023–2026 Google LLC
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
"""

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
  pred = model.weights.value[0] * tokens + model.weights.value[1] * tokens**2
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


class PrepareFwdBwdInputTest(absltest.TestCase):

  def setUp(self):
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

  def _train(self, engine, payloads, prepare_at: set[int], wait: bool = False):
    """Two steps of two micro-batches; micro-batches in `prepare_at` go in prepared.

    A prepared micro-batch is prepared before the previous update is applied, as
    `TrainerWorker.stage_fwd_bwd` does while that update is still on the device.
    With `wait`, `wait_for_device` follows each prepare, as it does there too.
    """
    prepared = {i: None for i in prepare_at}
    if 0 in prepared:
      prepared[0] = engine.prepare_fwd_bwd_input(payloads[0])
      if wait:
        engine.wait_for_device()
    metrics = []
    for step in range(2):
      for j in range(2):
        i = 2 * step + j
        engine.fwd_bwd(prepared[i] if i in prepared else payloads[i])
        if j == 1 and i + 1 in prepared:
          prepared[i + 1] = engine.prepare_fwd_bwd_input(payloads[i + 1])
          if wait:
            queued = engine._throttler._inflight_queue.qsize()
            self.assertGreaterEqual(engine.wait_for_device(), 0.0)
            # The in-flight queue is left for the update to drain as before.
            self.assertEqual(engine._throttler._inflight_queue.qsize(), queued)
      engine.update()
      buffer = engine.get_metrics(clear_cache=True)
      metrics.append({k: np.asarray(v).tolist() for k, v in sorted(buffer.scalar_metrics.items())})
      metrics[-1]["loss"] = np.asarray(buffer.weighted_metrics["loss"].compute()).tolist()
    return np.asarray(engine.model.weights.value), metrics

  def test_prepared_micro_batches_train_bit_identically(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    for compiled in (True, False):
      ref = self._engine(compiled)
      new = self._engine(compiled)
      with (
          mock.patch.object(new, "_compile_for_batch", wraps=new._compile_for_batch) as new_compiles,
          mock.patch.object(ref, "_compile_for_batch", wraps=ref._compile_for_batch) as ref_compiles,
      ):
        ref_w, ref_m = self._train(ref, payloads, prepare_at=set())
        new_w, new_m = self._train(new, payloads, prepare_at={0, 2})
      np.testing.assert_array_equal(ref_w, new_w)
      self.assertEqual(ref_m, new_m)
      # No kernel was rebuilt for a prepared input.
      self.assertEqual(ref_compiles.call_count, new_compiles.call_count)
      self.assertEqual(new.train_step, 2)

  def test_waiting_for_the_device_changes_nothing(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    for compiled in (True, False):
      ref = self._engine(compiled)
      new = self._engine(compiled)
      with (
          mock.patch.object(new, "_compile_for_batch", wraps=new._compile_for_batch) as new_compiles,
          mock.patch.object(ref, "_compile_for_batch", wraps=ref._compile_for_batch) as ref_compiles,
          mock.patch.object(new._metrics_logger, "write_metrics", wraps=new._metrics_logger.write_metrics) as new_w,
          mock.patch.object(ref._metrics_logger, "write_metrics", wraps=ref._metrics_logger.write_metrics) as ref_w,
      ):
        ref_weights, ref_m = self._train(ref, payloads, prepare_at=set())
        new_weights, new_m = self._train(new, payloads, prepare_at={0, 2}, wait=True)
      np.testing.assert_array_equal(ref_weights, new_weights)
      self.assertEqual(ref_m, new_m)
      self.assertEqual(ref_compiles.call_count, new_compiles.call_count)
      # The same metrics writes, in the same order (the wait writes none).
      self.assertEqual(
          [c.args[0].id for c in ref_w.call_args_list],
          [c.args[0].id for c in new_w.call_args_list],
      )

  def test_wait_for_device_waits_for_the_update_and_keeps_the_queue(self):
    engine = self._engine(True)
    engine.fwd_bwd(_payload(1))
    engine.fwd_bwd(_payload(2))
    engine.update()
    queue = engine._throttler._inflight_queue
    queued = list(queue.queue)
    self.assertNotEmpty(queued)
    waited = engine.wait_for_device()
    self.assertIsInstance(waited, float)
    self.assertGreaterEqual(waited, 0.0)
    self.assertEqual(list(queue.queue), queued)
    # Everything dispatched before the wait, the update included, is done.
    for leaves, _ in queued:
      for leaf in leaves:
        if isinstance(leaf, jax.Array):
          self.assertTrue(leaf.is_ready())

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
    # The path Tunix's pipelined step boundary takes: TrainerWorker.stage_fwd_bwd
    # calls prepare_fwd_bwd_input, fwd_bwd_staged trains what it returned.
    from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
    from tunix.experimental.worker import trainer_worker  # pylint: disable=g-import-not-at-top

    payloads = [_payload(s) for s in (1, 2, 3, 4)]

    def run(stage: bool):
      engine = self._engine(True)
      worker = trainer_worker.TrainerWorker(trainer_factory=lambda: engine)
      worker.initialize()
      prepare = mock.patch.object(engine, "prepare_fwd_bwd_input", wraps=engine.prepare_fwd_bwd_input)
      with prepare as spy:
        for i, payload in enumerate(payloads):
          request = datatypes.TrainRequest(request_id=f"r{i}", payload=payload)
          if stage and i % 2 == 0:
            resp = worker.stage_fwd_bwd(request)
            # The engine's wait_for_device answered how long the device ran on.
            self.assertGreaterEqual(resp.metadata["device_wait_s"], 0.0)
            worker.fwd_bwd_staged(f"r{i}")
          else:
            worker.fwd_bwd(request)
          if i % 2 == 1:
            worker.update()
      return np.asarray(engine.model.weights.value), spy.call_count

    ref_w, ref_prepared = run(stage=False)
    new_w, new_prepared = run(stage=True)
    np.testing.assert_array_equal(ref_w, new_w)
    self.assertEqual((ref_prepared, new_prepared), (0, 2))

  def test_preparing_places_the_traced_inputs_and_leaves_state_alone(self):
    engine = self._engine(True)
    engine.fwd_bwd(_payload(1))
    before = (
        engine.micro_step_count,
        engine.train_step,
        np.asarray(engine.model.weights.value),
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
        np.asarray(engine.model.weights.value),
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
