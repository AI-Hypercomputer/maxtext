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
"""prepare_fwd_bwd_input under the dropless replay and the final micro-batch's dropless program.

The MLPerf recipes train with `retry_when_tokens_dropped` (a micro-batch whose MoE ragged buffer overflowed is
accumulated as zero and rerun dropless, from the inputs it ran on) and `retry_dropless_final_micro_batch` (a step's
last micro-batch runs the dropless program directly). A micro-batch prepared ahead must take both paths exactly like
the payload it came from: the overflowing micro-batch is replayed from its prepared inputs, the final one runs the
final program, and the weights, losses and replay counts are bit-identical.

The rig is `maxtext_engine_final_dropless_test.py`'s: a one-layer Qwen3.5 MoE on CPU whose loss reports the overflow
flag RoutedMoE would, from which program its graph is, and scales the loss per program so the kernels can be told
apart.
"""
# pylint: disable=protected-access

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

# Two steps of three micro-batches. Step 1's first micro-batch overflows, so the one prepared ahead is replayed.
_STEPS = ((10, _MARKED, 30), (_MARKED, 20, 30))

# Where the next step's first micro-batch is prepared, relative to the step's update.
_BEFORE_UPDATE = "before_update"
_AFTER_UPDATE = "after_update"  # where Tunix's stage lands when the update runs inside the last train call


class PreparedInputDroplessTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _engine(self):
    """An engine on the one-layer MoE rig, with the dropless replay and the final dropless program on."""
    cfg = _cfg(override_logical_axis_rules=True, ragged_buffer_factor=2.5)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    # The ring-of-experts prerequisites of retry_when_tokens_dropped cannot run on CPU, so the switches are set
    # directly, as the dropless replay and final-dropless tests do.
    engine._dropless_replay = True
    engine._final_dropless = True
    base_loss = maxtext_engine.make_router_replay_loss_fn(cfg)

    def loss_fn(model, **batch):
      program = _program(model)
      loss, aux = base_loss(model, **batch)
      aux = {k: v for k, v in aux.items() if k != "has_moe_overflow"}
      aux["has_moe_overflow"] = jnp.logical_and(batch["inputs"][0, 0] == _MARKED, program == "regular")
      aux["xent_sum"] = aux["xent_sum"] * _SCALES[program]
      return loss, aux

    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(loss_fn, has_aux=True)
    engine.compile(None)  # As Tunix's TrainerWorker does at bring-up: the kernels compile on the first fwd_bwd.
    return engine

  def _train(self, engine, prepare_ahead: bool, order: str = _AFTER_UPDATE, prepare_first: bool = False):
    """Runs `_STEPS`; returns the final parameters, per-step metrics and the prepared micro-batches' replays.

    With `prepare_ahead`, each step's first micro-batch after the first step is prepared right after the previous
    step's last `fwd_bwd` (`apply_optimizer=True`, so it ran the final program), before or after the update, and
    `wait_for_device` follows, as Tunix's `TrainerWorker.stage_fwd_bwd` does.
    """
    prepared = None
    if prepare_first:
      prepared = engine.prepare_fwd_bwd_input(_payload(_STEPS[0][0]))
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
      has_next = step + 1 < len(_STEPS)
      if prepare_ahead and has_next and order == _BEFORE_UPDATE:
        prepared = engine.prepare_fwd_bwd_input(_payload(_STEPS[step + 1][0]))
        self.assertGreaterEqual(engine.wait_for_device(), 0.0)
      with mock.patch.object(engine, "record_metrics", wraps=engine.record_metrics) as record:
        engine.update()
      recorded = {call.args[0]: call.args[1] for call in record.call_args_list}
      if prepare_ahead and has_next and order == _AFTER_UPDATE:
        prepared = engine.prepare_fwd_bwd_input(_payload(_STEPS[step + 1][0]))
        self.assertGreaterEqual(engine.wait_for_device(), 0.0)
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
    return params, metrics

  def _assert_same_training(self, ref, new):
    ref_params, ref_metrics = ref
    new_params, new_metrics = new
    self.assertEqual(ref_metrics, new_metrics)
    self.assertEqual(len(ref_params), len(new_params))
    for r, n in zip(ref_params, new_params):
      np.testing.assert_array_equal(r, n)

  def test_a_prepared_micro_batch_is_replayed_and_trains_bit_identically(self):
    reference = self._train(self._engine(), prepare_ahead=False)
    # Not vacuous: one replay per step (the marked micro-batches), and step 1's is the prepared one.
    self.assertEqual([m["replays"] for m in reference[1]], [1, 1])
    for order in (_AFTER_UPDATE, _BEFORE_UPDATE):
      with self.subTest(order=order):
        engine = self._engine()
        with mock.patch.object(engine, "_compile_for_batch", wraps=engine._compile_for_batch) as compiles:
          prepared = self._train(engine, prepare_ahead=True, order=order)
        self._assert_same_training(reference, prepared)
        self.assertEqual(compiles.call_count, 1)  # no kernel was rebuilt for the prepared input

  def test_the_first_micro_batch_of_the_run_can_be_prepared(self):
    # Tunix never stages a run's first micro-batch, but the background compiles that start from it must still
    # produce the kernels a payload would.
    reference = self._train(self._engine(), prepare_ahead=False)
    prepared = self._train(self._engine(), prepare_ahead=True, prepare_first=True)
    self._assert_same_training(reference, prepared)

  def test_the_comparison_sees_a_different_prepared_micro_batch(self):
    # Control: preparing the wrong micro-batch (step 1's second instead of its first) is caught.
    reference = self._train(self._engine(), prepare_ahead=False)
    engine = self._engine()
    real_prepare = engine.prepare_fwd_bwd_input
    with mock.patch.object(
        engine, "prepare_fwd_bwd_input", side_effect=lambda payload: real_prepare(_payload(_STEPS[1][1]))
    ):
      wrong = self._train(engine, prepare_ahead=True)
    self.assertNotEqual(reference[1], wrong[1])


if __name__ == "__main__":
  unittest.main()
