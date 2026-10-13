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
"""Tests for the final micro-batch's dropless program (retry_dropless_final_micro_batch) and the drop audit
(log_moe_drops) in MaxTextTrainingEngine, on top of the per-micro-batch dropless replay (retry_when_tokens_dropped).

As in `maxtext_engine_dropless_replay_test.py`, the ragged-buffer overflow needs the ring-of-experts kernels, so the
loss sows the flag RoutedMoE would, from which program its graph is: the regular one (overflows when asked to), the
replay one (`apply_dropless_overrides`: force_dropless + scanned chunks) or the final one
(`apply_final_dropless_overrides`: force_dropless only). It returns the drop audit in its aux, as Tunix's GRPO loss
does. The layer-level count and the final program's numerics are checked in `tests/unit/moe_drop_audit_test.py`.
"""

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
import pytest

from tests.post_training.unit.maxtext_engine_dropless_replay_test import _cfg, _moes, _payload, _MARKED

pytestmark = [pytest.mark.post_training]

_DROPPED_PER_LAYER = 7  # What the fake loss reports per layer for an overflowing regular attempt.


def _program(model):
  moes = _moes(model)
  if all(m.force_dropless for m in moes):
    return "replay" if all(m.dropless_scan_chunks for m in moes) else "final"
  return "regular"


class FinalDroplessEngineTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _engine(self, overflow, scales=None, final_flag=True, audit=False, compile_dummy=False, **config_overrides):
    """`overflow`: True (every regular attempt), False, or "marked" (regular attempts of `_payload(_MARKED)`).
    `scales`: per-program multiplier of the loss, to tell the kernels apart (None = all 1)."""
    # The first-phase flags need retry_when_tokens_dropped=True, which the config cannot validate on CPU: set them
    # after validation, as the #5570 tests do.
    first_phase = {
        k: config_overrides.pop(k)
        for k in ("retry_dropless_first_steps", "first_phase_ragged_buffer_factor")
        if k in config_overrides
    }
    cfg = _cfg(override_logical_axis_rules=True, ragged_buffer_factor=2.5, **config_overrides)
    cfg.get_keys().update(first_phase)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    # The ring-of-experts prerequisites of retry_when_tokens_dropped (which the two new flags need) cannot run on
    # CPU, so the switches are set directly, as the #5570 tests do.
    engine._dropless_replay = True  # pylint: disable=protected-access
    engine._final_dropless = final_flag  # pylint: disable=protected-access
    engine._log_moe_drops = audit  # pylint: disable=protected-access
    base_loss = maxtext_engine.make_router_replay_loss_fn(cfg)
    num_layers = cfg.num_decoder_layers

    def loss_fn(model, **batch):
      program = _program(model)
      loss, aux = base_loss(model, **batch)
      aux = {k: v for k, v in aux.items() if k != "has_moe_overflow"}
      if overflow == "marked":
        ov = jnp.logical_and(batch["inputs"][0, 0] == _MARKED, program == "regular")
      else:
        ov = jnp.asarray(bool(overflow) and program == "regular")
      aux["has_moe_overflow"] = ov
      if audit:
        aux[maxtext_engine.MOE_AUDIT_AUX_KEY] = {
            "dropped": jnp.full((num_layers,), _DROPPED_PER_LAYER, jnp.int32) * ov.astype(jnp.int32),
            "max_ratio": jnp.where(ov, 4.0, 1.5) * jnp.ones((num_layers,), jnp.float32),
        }
      if scales:
        aux["xent_sum"] = aux["xent_sum"] * scales[program]
      return loss, aux

    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(loss_fn, has_aux=True)
    engine.compile(None)  # As Tunix's TrainerWorker does at bring-up.
    if compile_dummy:
      engine.compile(_payload())  # As TRAINER_WARMUP_COMPILE does, before run_start.
    return engine

  def _step(self, engine, first_tokens, final_flags):
    """One step; returns the per-micro-batch losses, the summed gradients/denominator and the recorded metrics."""
    for first_token, apply_optimizer in zip(first_tokens, final_flags):
      kwargs = {} if apply_optimizer is None else {"apply_optimizer": apply_optimizer}
      engine.fwd_bwd(_payload(first_token), **kwargs)
    engine._resolve_overflow_check()  # pylint: disable=protected-access  # as `update` does first
    losses = [float(l.compute()) for l in engine._cached_losses]  # pylint: disable=protected-access
    grads = jax.tree.map(jnp.array, engine._accumulated_grads)  # pylint: disable=protected-access
    denom = float(engine._accumulated_denominator)  # pylint: disable=protected-access
    with mock.patch.object(engine, "record_metrics", wraps=engine.record_metrics) as record:
      engine.update()
    recorded = {call.args[0]: call.args[1] for call in record.call_args_list}
    return losses, grads, denom, recorded

  _SCALES = {"regular": 1.0, "replay": 2.0, "final": 3.0}

  def _clean_losses(self, first_tokens):
    engine = self._engine(overflow=False, final_flag=False)
    losses, _, _, _ = self._step(engine, first_tokens, [False] * (len(first_tokens) - 1) + [True])
    return losses

  def test_non_final_micro_batches_run_the_rbf_program_and_the_final_runs_dropless(self):
    tokens = [10, 20, 30]
    clean = self._clean_losses(tokens)
    engine = self._engine(overflow=False, scales=self._SCALES)
    losses, _, _, recorded = self._step(engine, tokens, [False, False, True])
    self.assertEqual(recorded["moe_dropless_replays"], 0)
    for got, want, program in zip(losses, clean, ("regular", "regular", "final")):
      self.assertAlmostEqual(got, self._SCALES[program] * want, places=3)
    # Compiled on the background thread from the first micro-batch, ahead of the final one.
    self.assertIsInstance(engine._compiled_fwd_bwd_final, jax.stages.Compiled)  # pylint: disable=protected-access
    # The live model is untouched.
    self.assertFalse(any(m.force_dropless for m in _moes(engine.model)))

  def test_overflow_on_a_non_final_micro_batch_is_replayed_and_the_final_is_not(self):
    tokens = [10, 20, 30]
    clean = self._clean_losses(tokens)
    engine = self._engine(overflow=True, scales=self._SCALES)
    losses, _, _, recorded = self._step(engine, tokens, [False, False, True])
    # Both regular micro-batches overflowed and were replayed; the final one ran dropless and cannot overflow.
    self.assertEqual(recorded["moe_dropless_replays"], 2)
    for got, want, program in zip(losses, clean, ("replay", "replay", "final")):
      self.assertAlmostEqual(got, self._SCALES[program] * want, places=3)

  def test_accumulated_gradients_equal_the_all_dropless_step(self):
    tokens = [10, _MARKED, 30]
    # Every micro-batch directly through the dropless replay kernel (#5570's retry_dropless_first_steps).
    reference = self._engine(overflow="marked", final_flag=False, retry_dropless_first_steps=1)
    ref_losses, ref_grads, ref_denom, _ = self._step(reference, tokens, [False, False, True])
    engine = self._engine(overflow="marked")
    losses, grads, denom, recorded = self._step(engine, tokens, [False, False, True])
    self.assertEqual(recorded["moe_dropless_replays"], 1)  # the marked micro-batch
    self.assertEqual(denom, ref_denom)
    for got, want in zip(losses, ref_losses):
      self.assertAlmostEqual(got, want, places=4)
    leaves, ref_leaves = jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(ref_grads)
    self.assertEqual(len(leaves), len(ref_leaves))
    for g, r in zip(leaves, ref_leaves):
      self.assertTrue(jnp.allclose(g, r, rtol=2e-2, atol=1e-5))

  def test_without_apply_optimizer_every_micro_batch_runs_the_rbf_program(self):
    tokens = [10, 20, 30]
    clean = self._clean_losses(tokens)
    engine = self._engine(overflow=False, scales=self._SCALES)
    losses, _, _, _ = self._step(engine, tokens, [None, None, None])
    for got, want in zip(losses, clean):
      self.assertAlmostEqual(got, want, places=3)

  def test_warm_up_compile_installs_the_final_kernel(self):
    engine = self._engine(overflow=False, compile_dummy=True)
    final_kernel = engine._compiled_fwd_bwd_final  # pylint: disable=protected-access
    self.assertIsInstance(final_kernel, jax.stages.Compiled)
    self.assertIsInstance(engine._compiled_fwd_bwd_dropless, jax.stages.Compiled)  # pylint: disable=protected-access
    with mock.patch.object(engine, "_start_background_compile") as background:
      self._step(engine, [10, 20], [False, True])
    background.assert_not_called()
    self.assertIs(engine._compiled_fwd_bwd_final, final_kernel)  # pylint: disable=protected-access

  def test_compile_kernels_adds_the_final_kernel_on_an_edited_copy(self):
    engine = self._engine(overflow=False)
    compiled = engine.compile_kernels(_payload())
    self.assertEqual(
        set(compiled),
        {*maxtext_engine.KERNEL_NAMES, maxtext_engine.DROPLESS_FWD_BWD, maxtext_engine.FINAL_DROPLESS_FWD_BWD},
    )
    graph = nnx.merge(engine._final_graphdef, *nnx.split(engine.model, nnx.Param, ...)[1:])  # pylint: disable=protected-access
    for moe in _moes(graph):
      self.assertTrue(moe.force_dropless)
      self.assertFalse(moe.dropless_scan_chunks)  # the regular chunks, not the replay's scan
    self.assertFalse(any(m.force_dropless for m in _moes(engine.model)))

  def test_drop_audit_reports_zero_used_after_a_replay(self):
    engine = self._engine(overflow="marked", audit=True)
    num_layers = engine._config.num_decoder_layers  # pylint: disable=protected-access
    with self.assertLogs("absl", level="INFO") as logs:
      _, _, _, recorded = self._step(engine, [10, _MARKED, 30], [False, False, True])
    lines = [r.getMessage() for r in logs.records if r.getMessage().startswith("MOE_DROP_AUDIT")]
    print("\n".join(lines))
    audit = [l for l in lines if l.startswith("MOE_DROP_AUDIT step=")]
    self.assertEqual(len(audit), 3)
    self.assertIn("micro=0 program=rbf2.5 replayed=False dropped_assignments_used=0 ", audit[0])
    self.assertIn("micro=1 program=rbf2.5 replayed=True dropped_assignments_used=0 ", audit[1])
    self.assertIn("micro=2 program=dropless replayed=False dropped_assignments_used=0 ", audit[2])
    discarded = [l for l in lines if l.startswith("MOE_DROP_AUDIT_DISCARDED")]
    self.assertEqual(len(discarded), 1)
    self.assertIn(f"micro=1 program=rbf2.5 dropped_assignments_discarded={_DROPPED_PER_LAYER * num_layers} ", discarded[0])
    self.assertIn("max_ratio=4.0000", discarded[0])
    step_lines = [l for l in lines if l.startswith("MOE_DROP_AUDIT_STEP")]
    self.assertEqual(len(step_lines), 1)
    self.assertIn("dropped_assignments_used=0 dropless_replays=1 ", step_lines[0])
    self.assertIn(f"dropped_assignments_discarded={_DROPPED_PER_LAYER * num_layers}", step_lines[0])
    self.assertEqual(recorded["moe_dropped_assignments"], 0)
    self.assertEqual(recorded["moe_dropped_assignments_discarded"], _DROPPED_PER_LAYER * num_layers)
    self.assertNotIn(maxtext_engine.MOE_AUDIT_AUX_KEY, recorded)

  def test_drop_audit_without_replay_counts_what_was_dropped(self):
    """Negative control: with the replay switched off, the accumulated program's drops are reported, not zero."""
    engine = self._engine(overflow="marked", audit=True, final_flag=False)
    engine._dropless_replay = False  # pylint: disable=protected-access
    engine._compiled = False  # pylint: disable=protected-access  # recompile without the replay outputs
    num_layers = engine._config.num_decoder_layers  # pylint: disable=protected-access
    with self.assertLogs("absl", level="INFO") as logs:
      _, _, _, recorded = self._step(engine, [10, _MARKED], [False, True])
    audit = [r.getMessage() for r in logs.records if r.getMessage().startswith("MOE_DROP_AUDIT step=")]
    self.assertIn(f"micro=1 program=rbf2.5 replayed=False dropped_assignments_used={_DROPPED_PER_LAYER * num_layers} ", audit[1])
    self.assertEqual(recorded["moe_dropped_assignments"], _DROPPED_PER_LAYER * num_layers)


if __name__ == "__main__":
  unittest.main()
