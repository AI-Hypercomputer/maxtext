# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `MaxTextTrainingEngine.stage_batch` and the `TUNIX_STAGE_BATCH` input-transfer modes.

The engine is the tiny CPU MoE of `maxtext_engine_dropless_replay_test.py` with the dropless replay switched on, so
a consumed micro-batch's inputs are held in the pending overflow check as on the production path. Checked:

  (a) a staged batch is consumed by the next `fwd_bwd` (its device tree is what the kernel ran on), and the losses
      and accumulated gradients are bit-identical to the host-leaf (`off`) path;
  (b) a staged batch whose structure differs from the `fwd_bwd` payload's is dropped and the payload is
      transferred explicitly (and recompiled for), with a warning;
  (c) input transfers never overlap: every transfer after the first is preceded by a wait on the previous one, in
      `stage` (staged, consumed and fallback transfers alike) and in `serialize` mode;
  (d) `serialize` stages nothing, transfers explicitly and serially, and is bit-identical to `off`;
  plus `fwd_bwd(None, stage_key=...)`, the out-of-order key path, and the env parsing.
"""

# pylint: disable=protected-access

import concurrent.futures
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.models import models
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.post_training.unit.maxtext_engine_dropless_replay_test import _cfg, _payload

pytestmark = [pytest.mark.post_training]


def _np_payload(first_token=10, rows=1):
  """`_payload` with host (numpy) leaves, as Tunix's worker hands the engine a deserialized micro-batch; `rows`
  copies of the row (a different row count is a different batch structure, so a different kernel signature)."""
  p = _payload(first_token)

  def rep(x):
    return np.repeat(np.asarray(x), rows, axis=0)

  return abstract_engine.RLTrainerPayload(
      prompt_ids=rep(p.prompt_ids),
      prompt_mask=rep(p.prompt_mask),
      completion_ids=rep(p.completion_ids),
      completion_mask=rep(p.completion_mask),
      advantages=rep(p.advantages),
  )


class StageBatchEngineTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    # The metrics logger writes under base_output_directory; the source tree may be read-only (e.g. in-image runs).
    self._tmp = tempfile.mkdtemp(prefix="stage_batch_test_")
    self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

  def _engine(self, mode, max_device_batches=None):
    """A compiled engine built under TUNIX_STAGE_BATCH=`mode` (the engine reads the env once, at __init__)."""
    cfg = _cfg(base_output_directory=self._tmp)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    env = {maxtext_engine.STAGE_BATCH_ENV: mode} if mode is not None else {}
    if max_device_batches is not None:
      env[maxtext_engine.STAGE_MAX_DEVICE_BATCHES_ENV] = str(max_device_batches)
    with mock.patch.dict(os.environ, env, clear=False):
      if mode is None:
        os.environ.pop(maxtext_engine.STAGE_BATCH_ENV, None)
      if max_device_batches is None:
        os.environ.pop(maxtext_engine.STAGE_MAX_DEVICE_BATCHES_ENV, None)
      with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
        engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    engine._dropless_replay = True  # The production path: inputs are held in the pending overflow check.
    base_loss = maxtext_engine.make_router_replay_loss_fn(cfg)

    def loss_fn(model, **batch):
      loss, aux = base_loss(model, **batch)
      aux = {k: v for k, v in aux.items() if k != "has_moe_overflow"}
      aux["has_moe_overflow"] = jnp.asarray(False)
      return loss, aux

    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(loss_fn, has_aux=True)
    engine.compile(None)
    return engine

  @staticmethod
  def _results(engine):
    engine._resolve_overflow_check()  # as `update` does first
    losses = [float(l.compute()) if hasattr(l, "compute") else float(l) for l in engine._cached_losses]
    grads = jax.tree.map(np.asarray, engine._accumulated_grads)
    return losses, grads, float(engine._accumulated_denominator)

  def _assert_same_results(self, a, b):
    self.assertEqual(a[0], b[0])
    self.assertEqual(a[2], b[2])
    la, lb = jax.tree.leaves(a[1]), jax.tree.leaves(b[1])
    self.assertEqual(len(la), len(lb))
    for x, y in zip(la, lb):
      np.testing.assert_array_equal(x, y)

  @staticmethod
  def _spy_transfers(engine):
    """Records ("wait", id(previous tree)) / ("put", id(new tree)) events of the engine's transfer methods."""
    events = []
    keep_alive = []
    real_wait, real_issue = engine._wait_for_input_transfer, engine._issue_input_transfer

    def wait():
      events.append(("wait", id(engine._last_input_transfer)))
      real_wait()

    def issue(dynamic_batch):
      out = real_issue(dynamic_batch)
      keep_alive.append(out)
      events.append(("put", id(out)))
      return out

    engine._wait_for_input_transfer = wait
    engine._issue_input_transfer = issue
    return events

  @staticmethod
  def _assert_serialized(events):
    """Every put after the first is preceded by a wait on exactly the previous put's tree, with no put between."""
    puts = [e for e in events if e[0] == "put"]
    for i in range(1, len(puts)):
      prev_put = events.index(puts[i - 1])
      this_put = events.index(puts[i])
      between = events[prev_put + 1 : this_put]
      assert ("wait", puts[i - 1][1]) in between, (i, events)
      assert not [e for e in between if e[0] == "put"], events

  def test_mode_parsing(self):
    for value, want in (
        ("", "off"),
        ("0", "off"),
        ("off", "off"),
        ("1", "stage"),
        ("stage", "stage"),
        ("true", "stage"),
        ("serialize", "serialize"),
        ("SERIALIZE", "serialize"),
    ):
      with mock.patch.dict(os.environ, {maxtext_engine.STAGE_BATCH_ENV: value}):
        self.assertEqual(maxtext_engine.stage_batch_mode(), want, value)
    with mock.patch.dict(os.environ, {maxtext_engine.STAGE_BATCH_ENV: "maybe"}):
      with self.assertRaises(ValueError):
        maxtext_engine.stage_batch_mode()
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(maxtext_engine.STAGE_MAX_DEVICE_BATCHES_ENV, None)
      self.assertEqual(maxtext_engine.stage_max_device_batches(), 1)
    with mock.patch.dict(os.environ, {maxtext_engine.STAGE_MAX_DEVICE_BATCHES_ENV: "0"}):
      self.assertEqual(maxtext_engine.stage_max_device_batches(), 0)
    with mock.patch.dict(os.environ, {maxtext_engine.STAGE_MAX_DEVICE_BATCHES_ENV: "2"}):
      self.assertEqual(maxtext_engine.stage_max_device_batches(), 2)

  def test_default_off_and_stage_is_noop(self):
    engine = self._engine(None)
    self.assertEqual(engine.stage_batch_mode, "off")
    self.assertFalse(engine.stage_batch(_np_payload(10)))
    self.assertEqual(engine.staged_batch_count, 0)

  def test_staged_batch_consumed_and_bit_identical(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    off.fwd_bwd(_np_payload(20))
    want = self._results(off)

    engine = self._engine("stage")
    self.assertEqual(engine.stage_batch_mode, "stage")
    self.assertTrue(engine.stage_batch(_np_payload(10), key="mb0"))
    self.assertEqual(engine.staged_batch_count, 1)
    staged_tree = engine._staged_batches[0].dynamic
    for leaf in jax.tree.leaves(staged_tree):
      self.assertIsInstance(leaf, jax.Array)
    engine.fwd_bwd(_np_payload(10), stage_key="mb0")
    self.assertEqual(engine.staged_batch_count, 0)
    # The kernel ran on the staged device tree: it is what the pending overflow check holds.
    self.assertIs(engine._pending_overflow_check.inputs, staged_tree)
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb1"))
    engine.fwd_bwd(_np_payload(20), stage_key="mb1")
    self._assert_same_results(self._results(engine), want)

  def test_fwd_bwd_without_payload_runs_staged_batch(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    want = self._results(off)

    engine = self._engine("stage")
    engine.stage_batch(_np_payload(10), key="r1")
    engine.fwd_bwd(None, stage_key="r1")
    self._assert_same_results(self._results(engine), want)
    with self.assertRaises(ValueError):
      engine.fwd_bwd(None, stage_key="r2")

  def test_out_of_order_key_takes_named_entry(self):
    engine = self._engine("stage", max_device_batches=0)
    engine.stage_batch(_np_payload(10), key="a")
    engine.stage_batch(_np_payload(20), key="b")
    b_tree = engine._staged_batches[1].dynamic
    with self.assertLogs(level="WARNING") as logs:
      engine.fwd_bwd(_np_payload(20), stage_key="b")
    self.assertTrue(any("out of FIFO order" in line for line in logs.output), logs.output)
    self.assertIs(engine._pending_overflow_check.inputs, b_tree)
    self.assertEqual(engine.staged_batch_count, 1)
    self.assertEqual(engine._staged_batches[0].key, "a")

  def test_signature_mismatch_falls_back(self):
    engine = self._engine("stage")
    events = self._spy_transfers(engine)
    engine.fwd_bwd(_np_payload(10))  # compiles for the one-row structure
    self.assertTrue(engine.stage_batch(_np_payload(20), key="stale"))
    # The fwd_bwd payload has two rows: a different signature, so the staged entry is dropped and the payload is
    # transferred explicitly (and the kernel recompiled for it).
    with self.assertLogs(level="WARNING") as logs:
      engine.fwd_bwd(_np_payload(20, rows=2), stage_key="stale")
    self.assertTrue(any("does not match the fwd_bwd payload's structure" in line for line in logs.output), logs.output)
    self.assertEqual(engine.staged_batch_count, 0)
    # Three transfers: the first fwd_bwd's explicit one, the stale staged one, the fallback one.
    self.assertEqual(len([e for e in events if e[0] == "put"]), 3)
    self._assert_serialized(events)
    losses, _, _ = self._results(engine)
    self.assertEqual(len(losses), 2)

  def test_transfers_never_overlap_in_stage_mode(self):
    engine = self._engine("stage", max_device_batches=0)
    events = self._spy_transfers(engine)
    waited_on = []
    real_block = jax.block_until_ready

    def block(tree):
      waited_on.append(tree)
      return real_block(tree)

    with mock.patch.object(maxtext_engine.jax, "block_until_ready", side_effect=block):
      engine.stage_batch(_np_payload(10), key="mb0")  # put 1 (nothing to wait for)
      engine.fwd_bwd(_np_payload(10), stage_key="mb0")  # consumes put 1, no transfer
      engine.stage_batch(_np_payload(20), key="mb1")  # waits put 1, put 2
      engine.stage_batch(_np_payload(30), key="mb2")  # waits put 2, put 3
      engine.fwd_bwd(_np_payload(20), stage_key="mb1")  # consumes put 2
      engine.fwd_bwd(_np_payload(30), stage_key="mb2")  # consumes put 3
      engine.fwd_bwd(_np_payload(40))  # not staged: waits put 3, put 4
    puts = [e for e in events if e[0] == "put"]
    self.assertEqual(len(puts), 4)
    self._assert_serialized(events)
    # The waits went through jax.block_until_ready on the previous transfer's tree (not on the compute). Other
    # callers (the in-flight throttler waits on losses) are filtered out by identity.
    put_ids = {p[1] for p in puts}
    self.assertEqual([id(t) for t in waited_on if id(t) in put_ids], [p[1] for p in puts[:-1]])
    self.assertEqual(engine.staged_batch_count, 0)
    self.assertEqual(len(self._results(engine)[0]), 4)

  def test_serialize_mode(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    off.fwd_bwd(_np_payload(20))
    want = self._results(off)

    engine = self._engine("serialize")
    self.assertEqual(engine.stage_batch_mode, "serialize")
    events = self._spy_transfers(engine)
    self.assertFalse(engine.stage_batch(_np_payload(10)))
    self.assertEqual(engine.staged_batch_count, 0)
    engine.fwd_bwd(_np_payload(10))
    engine.fwd_bwd(_np_payload(20))
    self.assertEqual([e[0] for e in events], ["wait", "put", "wait", "put"])
    self._assert_serialized(events)
    for leaf in jax.tree.leaves(engine._pending_overflow_check.inputs):
      self.assertIsInstance(leaf, jax.Array)
    self._assert_same_results(self._results(engine), want)
    with self.assertRaises(ValueError):
      engine.fwd_bwd(None, stage_key="x")

  def test_host_buffered_queue_promotes_to_device_on_consume(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    off.fwd_bwd(_np_payload(20))
    off.fwd_bwd(_np_payload(30))
    want = self._results(off)

    # Default STAGE_MAX_DEVICE_BATCHES is 1 when unset (B6).
    engine = self._engine("stage")
    self.assertEqual(engine._stage_max_device_batches, 1)
    events = self._spy_transfers(engine)
    self.assertTrue(engine.stage_batch(_np_payload(10), key="mb0"))
    self.assertTrue(engine._staged_batches[0].on_device)
    for leaf in jax.tree.leaves(engine._staged_batches[0].dynamic):
      self.assertIsInstance(leaf, jax.Array)
    # Second and third batches staged before mb0 runs stay in host RAM as np.ndarray leaves (B7).
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb1"))
    self.assertFalse(engine._staged_batches[1].on_device)
    for leaf in jax.tree.leaves(engine._staged_batches[1].dynamic):
      self.assertIsInstance(leaf, np.ndarray)
      self.assertNotIsInstance(leaf, jax.Array)
    self.assertTrue(engine.stage_batch(_np_payload(30), key="mb2"))
    self.assertFalse(engine._staged_batches[2].on_device)
    for leaf in jax.tree.leaves(engine._staged_batches[2].dynamic):
      self.assertIsInstance(leaf, np.ndarray)
      self.assertNotIsInstance(leaf, jax.Array)
    self.assertEqual(len([e for e in events if e[0] == "put"]), 1)

    # Consuming mb0 promotes mb1 to device HBM automatically and logs STAGE_PROMOTE (N1).
    with self.assertLogs(level="INFO") as logs:
      engine.fwd_bwd(None, stage_key="mb0")
    self.assertTrue(any("STAGE_PROMOTE key=mb1" in line for line in logs.output), logs.output)
    self.assertTrue(engine._staged_batches[0].on_device)
    self.assertFalse(engine._staged_batches[1].on_device)
    self.assertEqual(len([e for e in events if e[0] == "put"]), 2)

    engine.fwd_bwd(None, stage_key="mb1")
    self.assertTrue(engine._staged_batches[0].on_device)
    self.assertEqual(len([e for e in events if e[0] == "put"]), 3)

    engine.fwd_bwd(None, stage_key="mb2")
    self.assertEqual(engine.staged_batch_count, 0)
    self._assert_serialized(events)
    self._assert_same_results(self._results(engine), want)

  def test_keyless_fwd_bwd_leaves_keyed_staged_fifo_intact(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    off.fwd_bwd(_np_payload(20))
    want = self._results(off)

    engine = self._engine("stage")
    # Stage mb1 ahead of time, then run an unstaged keyless fwd_bwd(payload) (e.g. sampler agreement / fallback).
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb1"))
    self.assertEqual(engine.staged_batch_count, 1)
    engine.fwd_bwd(_np_payload(10))
    # Keyed entry mb1 must remain untouched in the FIFO.
    self.assertEqual(engine.staged_batch_count, 1)
    self.assertEqual(engine._staged_batches[0].key, "mb1")
    engine.fwd_bwd(None, stage_key="mb1")
    self.assertEqual(engine.staged_batch_count, 0)
    self._assert_same_results(self._results(engine), want)

  def test_stage_batch_idempotent_by_key(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(20))
    off.fwd_bwd(_np_payload(40))
    want = self._results(off)

    engine = self._engine("stage")
    # First stage mb0 and mb1, then re-stage both with the same keys (simulating RPC retry / replacement).
    self.assertTrue(engine.stage_batch(_np_payload(10), key="mb0"))
    self.assertTrue(engine.stage_batch(_np_payload(30), key="mb1"))
    self.assertEqual(engine.staged_batch_count, 2)
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb0"))
    self.assertTrue(engine.stage_batch(_np_payload(40), key="mb1"))
    self.assertEqual(engine.staged_batch_count, 2)
    self.assertEqual([s.key for s in engine._staged_batches], ["mb0", "mb1"])
    self.assertTrue(engine._staged_batches[0].on_device)
    self.assertFalse(engine._staged_batches[1].on_device)

    engine.fwd_bwd(None, stage_key="mb0")
    self.assertEqual(engine.staged_batch_count, 1)
    self.assertTrue(engine._staged_batches[0].on_device)
    engine.fwd_bwd(None, stage_key="mb1")
    self.assertEqual(engine.staged_batch_count, 0)
    self._assert_same_results(self._results(engine), want)

  def test_in_flight_counter_recovered_on_exception_and_reset_staged(self):
    engine = self._engine("stage")
    self.assertTrue(engine.stage_batch(_np_payload(10), key="mb0"))
    with mock.patch.object(engine._throttler, "wait_for_next", side_effect=RuntimeError("simulated failure")):
      with self.assertRaises(RuntimeError):
        engine.fwd_bwd(None, stage_key="mb0")
    # _in_flight_staged_on_device must be decremented in finally so future staging still promotes to device.
    self.assertEqual(engine._in_flight_staged_on_device, 0)
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb1"))
    self.assertTrue(engine._staged_batches[0].on_device)
    self.assertEqual(engine.reset_staged(), 1)
    self.assertEqual(engine.staged_batch_count, 0)
    self.assertEqual(engine._in_flight_staged_on_device, 0)

  def test_concurrent_stage_batch_and_fwd_bwd_thread_safety(self):
    off = self._engine("off")
    tokens = [10, 20, 30, 40]
    for tok in tokens:
      off.fwd_bwd(_np_payload(tok))
    want = self._results(off)

    engine = self._engine("stage")
    # 1. Verify _take_staged_batch does not block while _transfer_lock is held by another transfer (B2).
    self.assertTrue(engine.stage_batch(_np_payload(10), key="probe"))
    with engine._transfer_lock:
      popped = engine._take_staged_batch(None, "probe")
      self.assertIsNotNone(popped)
      self.assertEqual(popped.key, "probe")
    engine.reset_staged()

    # 2. Concurrent worker thread staging batches while main thread consumes them via fwd_bwd (B8).
    staged_events = {tok: threading.Event() for tok in tokens}

    def stage_worker():
      for tok in tokens:
        ok = engine.stage_batch(_np_payload(tok), key=f"mb_{tok}")
        self.assertTrue(ok)
        staged_events[tok].set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
      fut = pool.submit(stage_worker)
      for tok in tokens:
        self.assertTrue(staged_events[tok].wait(timeout=30.0))
        engine.fwd_bwd(None, stage_key=f"mb_{tok}")
      fut.result(timeout=30.0)

    self.assertEqual(engine.staged_batch_count, 0)
    self.assertEqual(engine._in_flight_staged_on_device, 0)
    self._assert_same_results(self._results(engine), want)

  def test_prepare_fwd_bwd_input_coexists_with_stage_batch(self):
    off = self._engine("off")
    off.fwd_bwd(_np_payload(10))
    off.fwd_bwd(_np_payload(20))
    want = self._results(off)

    engine = self._engine("stage")
    prepared = engine.prepare_fwd_bwd_input(_np_payload(10))
    self.assertTrue(engine.stage_batch(_np_payload(20), key="mb1"))
    self.assertEqual(engine.staged_batch_count, 1)
    with mock.patch.object(engine, "_issue_input_transfer", wraps=engine._issue_input_transfer) as issue_mock:
      engine.fwd_bwd(prepared)
      # PreparedFwdBwdInput is already on device and must not re-transfer or pop the staged FIFO.
      issue_mock.assert_not_called()
    self.assertEqual(engine.staged_batch_count, 1)
    engine.fwd_bwd(None, stage_key="mb1")
    self.assertEqual(engine.staged_batch_count, 0)
    self._assert_same_results(self._results(engine), want)


if __name__ == "__main__":
  unittest.main(argv=sys.argv[:1])
