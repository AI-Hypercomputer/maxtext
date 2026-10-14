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
"""The order Tunix uses when it stages a step's first micro-batch before the previous step's update.

Tunix's staged step boundary (`stage_before_update`) calls, at a staged boundary: `fwd_bwd(last, apply_optimizer=True)`,
then `prepare_fwd_bwd_input(next step's first micro-batch)`, then `wait_for_device()`, then `update()`, and later
`fwd_bwd(prepared)`. Under `retry_when_tokens_dropped` without `retry_dropless_final_micro_batch` (the team's MLPerf
trainer configuration sets the first; the second is opt-in), the step's last micro-batch can overflow: its flag is still
pending when the next micro-batch is prepared, and its dropless rerun runs inside `update()`, after the prepare. The
prepared micro-batch must come out of that untouched, and the whole run must train bit-identically to the base order,
`fwd_bwd(last, apply_optimizer=True)`, `update()`, `fwd_bwd(next payload)`.

The rig is `maxtext_engine_prepared_input_dropless_test.py`'s (a one-layer Qwen3.5 MoE on CPU whose loss reports the
overflow flag RoutedMoE would), with the final micro-batch's dropless program off.
"""
# pylint: disable=protected-access

import dataclasses
import os
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.models import models
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.post_training.unit.maxtext_engine_dropless_replay_test import _cfg, _payload, _MARKED
from tests.post_training.unit.maxtext_engine_final_dropless_test import _program

pytestmark = [pytest.mark.post_training]

# Per-program loss multipliers, so the losses say which kernel ran each micro-batch.
_SCALES = {"regular": 1.0, "replay": 2.0, "final": 3.0}

# A second overflowing content, so a rerun can be told from the micro-batch prepared while it is pending.
_MARKED_B = 60

# Three steps of three micro-batches. Every step's last micro-batch overflows. At both staged boundaries (after steps
# 0 and 1) its rerun runs inside `update()` with the next step's first micro-batch already prepared; after step 2 it
# runs inside the last `update()`. Step 1's first (prepared) micro-batch overflows too.
_STEPS = ((10, 20, _MARKED), (_MARKED_B, 20, _MARKED), (40, 20, _MARKED))


class StageBeforeUpdateOrderTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _engine(self, final_dropless: bool = False):
    """The one-layer MoE engine with the dropless replay on; its loss raises the overflow flag on marked batches."""
    cfg = _cfg(override_logical_axis_rules=True, ragged_buffer_factor=2.5)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    # The ring-of-experts prerequisites of retry_when_tokens_dropped cannot run on CPU, so the switches are set
    # directly, as the dropless replay and final-dropless tests do.
    engine._dropless_replay = True
    engine._final_dropless = final_dropless
    base_loss = maxtext_engine.make_router_replay_loss_fn(cfg)

    def loss_fn(model, **batch):
      program = _program(model)
      loss, aux = base_loss(model, **batch)
      aux = {k: v for k, v in aux.items() if k != "has_moe_overflow"}
      marked = jnp.logical_or(batch["inputs"][0, 0] == _MARKED, batch["inputs"][0, 0] == _MARKED_B)
      aux["has_moe_overflow"] = jnp.logical_and(marked, program == "regular")
      aux["xent_sum"] = aux["xent_sum"] * _SCALES[program]
      return loss, aux

    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(loss_fn, has_aux=True)
    engine.compile(None)
    return engine

  def _train(self, engine, staged: bool, prepare=None):
    """Runs `_STEPS`; with `staged`, each next step's first micro-batch is prepared before the update.

    Returns the final parameters, per-step metrics, and per staged boundary whether the last micro-batch's flag was
    still pending when the next one was prepared.
    """
    prepare = prepare or engine.prepare_fwd_bwd_input
    prepared = None
    pending_at_prepare = []
    metrics = []
    for step, tokens in enumerate(_STEPS):
      for j, first_token in enumerate(tokens):
        final = j == len(tokens) - 1
        if j == 0 and prepared is not None:
          self.assertIsInstance(prepared, maxtext_engine.PreparedFwdBwdInput)
          engine.fwd_bwd(prepared, apply_optimizer=False)
          prepared = None
        else:
          engine.fwd_bwd(_payload(first_token), apply_optimizer=final)
      if staged and step + 1 < len(_STEPS):
        pending_at_prepare.append(engine._pending_overflow_check is not None)
        prepared = prepare(_payload(_STEPS[step + 1][0]))
        self.assertGreaterEqual(engine.wait_for_device(), 0.0)
        # Waiting resolves nothing: the flag is read, and the rerun run, by `update()`.
        self.assertIsNotNone(engine._pending_overflow_check)
      with mock.patch.object(engine, "record_metrics", wraps=engine.record_metrics) as record:
        engine.update()
      recorded = {call.args[0]: call.args[1] for call in record.call_args_list}
      buffer = engine.get_metrics(clear_cache=True)
      metrics.append(
          {
              "replays": int(recorded["moe_dropless_replays"]),
              "loss": np.asarray(buffer.weighted_metrics["loss"].compute()).tolist(),
              "gradient_norm": np.asarray(buffer.scalar_metrics["gradient_norm"]).tolist(),
          }
      )
    self.assertIsNone(prepared)
    params = [np.asarray(leaf) for leaf in jax.tree.leaves(nnx.state(engine.model, nnx.Param))]
    return params, metrics, pending_at_prepare

  def _assert_same_training(self, ref, new):
    self.assertEqual(ref[1], new[1])
    self.assertEqual(len(ref[0]), len(new[0]))
    for r, n in zip(ref[0], new[0]):
      np.testing.assert_array_equal(r, n)

  def test_staged_before_a_replayed_update_trains_bit_identically(self):
    reference = self._train(self._engine(), staged=False)
    # Not vacuous: every step's last micro-batch replays (in update), and step 1's first (the prepared one) replays.
    self.assertEqual([m["replays"] for m in reference[1]], [1, 2, 1])
    engine = self._engine()
    with mock.patch.object(engine, "_compile_for_batch", wraps=engine._compile_for_batch) as compiles:
      staged = self._train(engine, staged=True)
    self._assert_same_training(reference, staged)
    # The last micro-batch's flag was still pending at both prepares, i.e. its rerun ran after the prepare.
    self.assertEqual(staged[2], [True, True])
    self.assertEqual(compiles.call_count, 1)  # no kernel was rebuilt for the prepared input

  def test_with_the_final_dropless_program_too(self):
    # retry_dropless_final_micro_batch (#5673): the last micro-batch runs the final program and cannot overflow, but
    # its check is still created and read by update(), so the order is the same; only step 1's prepared first
    # micro-batch replays.
    reference = self._train(self._engine(final_dropless=True), staged=False)
    self.assertEqual([m["replays"] for m in reference[1]], [0, 1, 0])
    staged = self._train(self._engine(final_dropless=True), staged=True)
    self._assert_same_training(reference, staged)
    self.assertEqual(staged[2], [True, True])

  def test_through_tunix_trainer_worker_the_deferred_update_trains_bit_identically(self):
    """The same order, driven by a real Tunix `TrainerWorker`: `fwd_bwd(last, apply_optimizer=True,
    defer_update=True)` runs the last micro-batch and holds the update, `stage_fwd_bwd` prepares the next step's first
    micro-batch without waiting, `apply_deferred_update` waits for the device and runs the update (and with it the
    last micro-batch's rerun), and `fwd_bwd_staged` trains the prepared micro-batch. Against the fused calls
    `fwd_bwd(last, apply_optimizer=True)` and `fwd_bwd(next payload)`."""
    from tunix.experimental.common import datatypes  # pylint: disable=import-outside-toplevel
    from tunix.experimental.worker import trainer_worker  # pylint: disable=import-outside-toplevel

    if not hasattr(trainer_worker.TrainerWorker, "apply_deferred_update"):
      self.skipTest("this Tunix has no TrainerWorker.apply_deferred_update (staging before the update)")

    def run(staged: bool):
      engine = self._engine()
      worker = trainer_worker.TrainerWorker(trainer_factory=lambda: engine)
      worker.initialize()
      metrics, device_waits = [], []
      staged_id = None
      with mock.patch.object(engine, "fwd_bwd", wraps=engine.fwd_bwd) as fwd_bwd:
        for step, tokens in enumerate(_STEPS):
          for j, first_token in enumerate(tokens):
            final = j == len(tokens) - 1
            kwargs = {"apply_optimizer": True} if final else {}
            if final and staged and step + 1 < len(_STEPS):
              kwargs["defer_update"] = True
            if j == 0 and staged_id is not None:
              resp = worker.fwd_bwd_staged(staged_id, **kwargs)
              staged_id = None
            else:
              request = datatypes.TrainRequest(request_id=f"s{step}m{j}", payload=_payload(first_token))
              resp = worker.fwd_bwd(request, **kwargs)
          if resp.metadata.get("update_deferred"):
            nxt = datatypes.TrainRequest(request_id=f"s{step + 1}m0", payload=_payload(_STEPS[step + 1][0]))
            stage = worker.stage_fwd_bwd(nxt)
            self.assertTrue(stage.metadata["before_update"])
            self.assertIsNone(stage.metadata["device_wait_s"])
            update = worker.apply_deferred_update()
            device_waits.append(update.metadata["device_wait_s"])
            self.assertEqual(update.metadata["train_step"], step + 1)
            staged_id = nxt.request_id
          else:
            self.assertEqual(resp.metadata["train_step"], step + 1)
          buffer = engine.get_metrics(clear_cache=True)
          metrics.append(
              {
                  "loss": np.asarray(buffer.weighted_metrics["loss"].compute()).tolist(),
                  "gradient_norm": np.asarray(buffer.scalar_metrics["gradient_norm"]).tolist(),
              }
          )
        # `defer_update` is the worker's, never the engine's.
        self.assertTrue(all("defer_update" not in c.kwargs for c in fwd_bwd.call_args_list))
      self.assertIsNone(staged_id)
      self.assertIsNone(worker._deferred_update)
      params = [np.asarray(leaf) for leaf in jax.tree.leaves(nnx.state(engine.model, nnx.Param))]
      return params, metrics, device_waits

    reference = run(staged=False)
    staged = run(staged=True)
    self.assertEqual(reference[1], staged[1])
    for r, n in zip(reference[0], staged[0]):
      np.testing.assert_array_equal(r, n)
    self.assertEqual(len(staged[2]), len(_STEPS) - 1)
    self.assertTrue(all(w >= 0.0 for w in staged[2]))

  def test_the_comparison_sees_a_rerun_of_the_wrong_micro_batch(self):
    # Control: a rerun inside update() that ran on the prepared micro-batch instead of its own is caught.
    reference = self._train(self._engine(), staged=False)
    engine = self._engine()
    holder = {}
    real_prepare = engine.prepare_fwd_bwd_input
    real_resolve = engine._resolve_overflow_check

    def prepare(payload):
      holder["prepared"] = real_prepare(payload)
      return holder["prepared"]

    def resolve():
      check = engine._pending_overflow_check
      prepared = holder.pop("prepared", None)
      if check is not None and prepared is not None and check.micro_step > 0:
        dynamic, _ = maxtext_engine._split_static_and_dynamic(prepared.batch)
        engine._pending_overflow_check = dataclasses.replace(check, inputs=dynamic)
      return real_resolve()

    with mock.patch.object(engine, "_resolve_overflow_check", side_effect=resolve):
      wrong = self._train(engine, staged=True, prepare=prepare)
    self.assertNotEqual(reference[1], wrong[1])

  def test_the_comparison_sees_a_different_prepared_micro_batch(self):
    # Control: preparing the wrong micro-batch (the next step's second instead of its first) is caught.
    reference = self._train(self._engine(), staged=False)
    engine = self._engine()
    real_prepare = engine.prepare_fwd_bwd_input
    wrong = self._train(engine, staged=True, prepare=lambda payload: real_prepare(_payload(20)))
    self.assertNotEqual(reference[1], wrong[1])


if __name__ == "__main__":
  unittest.main()
