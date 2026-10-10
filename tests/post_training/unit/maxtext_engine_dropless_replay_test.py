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
"""Tests for MaxTextTrainingEngine's per-micro-batch dropless replay (retry_when_tokens_dropped).

The ragged-buffer overflow itself needs the ring-of-experts kernels, which do not run on CPU, so the
loss here sows the `moe_has_overflow` flag RoutedMoE would, from whether the graph it runs on is the
dropless one (`force_dropless`, set by `apply_dropless_overrides`). Everything else -- the dropless
graphdef, the second kernel, the rerun and the accumulation -- is the engine's own.
"""

import os
import sys
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import pytest

pytestmark = [pytest.mark.post_training]


def _cfg():
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path(), "attention=dot_product"],
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      override_model_config=True,
      model_name="qwen3.5-35b-a3b",
      num_experts=4,
      num_experts_per_tok=2,
      base_emb_dim=256,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=256,
      partial_rotary_factor=0.25,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      vocab_size=1000,
      max_target_length=8,
      max_prefill_predict_length=8,
      per_device_batch_size=1.0,
      weight_dtype="bfloat16",
      inhomogeneous_layer_cycle_interval=1,
      base_num_decoder_layers=1,
      num_decoder_layers=1,
      scan_layers=False,
      run_name="dropless_replay_engine_test",
  )


def _moes(model):
  return [m for _, m in nnx.iter_graph(model) if type(m).__name__ == "RoutedMoE"]


def _payload(first_token=10):
  token_ids = jnp.arange(8, dtype=jnp.int32)[None, :] + first_token
  return abstract_engine.RLTrainerPayload(
      prompt_ids=jnp.zeros((1, 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((1, 0), dtype=jnp.int32),
      completion_ids=token_ids,
      completion_mask=jnp.ones_like(token_ids),
      advantages=jnp.zeros((1,), dtype=jnp.float32),
  )


_MARKED = 50  # First token of the micro-batches that overflow under `overflow_on_normal_path="marked"`.


class DroplessReplayEngineTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _engine(self, overflow_on_normal_path, flag="sown", has_aux=True):
    """`flag` is where the loss leaves the overflow flag: sown on the model, in its aux as MaxText's loss does, or
    nowhere, as a loss that runs the forward on a split/merged copy of the model (Tunix's GRPO loss) does.
    `overflow_on_normal_path="marked"` overflows only the micro-batches built with `_payload(_MARKED)`."""
    cfg = _cfg()
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    # retry_when_tokens_dropped's ring-of-experts prerequisites cannot run on CPU: switch the replay on directly.
    engine._dropless_replay = True  # pylint: disable=protected-access
    base_loss = maxtext_engine.make_router_replay_loss_fn(cfg)

    def loss_fn(model, **batch):
      dropless = all(m.force_dropless for m in _moes(model))
      loss, aux = base_loss(nnx.merge(*nnx.split(model)) if flag == "none" else model, **batch)
      # MaxText's loss returns a constant False flag here (retry_when_tokens_dropped is off in the config).
      aux = {k: v for k, v in aux.items() if k != "has_moe_overflow"}
      # What RoutedMoE sows: the overflow flag on the capped path, never on the dropless one.
      if overflow_on_normal_path == "marked":
        overflow = jnp.logical_and(batch["inputs"][0, 0] == _MARKED, not dropless)
      else:
        overflow = jnp.asarray(overflow_on_normal_path and not dropless)
      if flag == "sown":
        _moes(model)[0].sow(nnx.Intermediate, "moe_has_overflow", overflow)
      elif flag == "aux":
        aux["has_moe_overflow"] = overflow
      # Tells the two kernels' gradients apart: the dropless one's loss is doubled.
      if dropless:
        aux["xent_sum"] = aux["xent_sum"] * 2.0
      return loss, aux

    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(loss_fn, has_aux=has_aux)
    # As Tunix's TrainerWorker does: no dummy data, so the kernels compile on the first fwd_bwd.
    engine.compile(None)
    return engine

  def _run_step(self, overflow, **engine_kwargs):
    """Two micro-batches and an update; returns the summed denominator, the summed gradients, the last loss and
    the `moe_dropless_replays` the update recorded."""
    engine = self._engine(overflow_on_normal_path=overflow, **engine_kwargs)
    engine.fwd_bwd(_payload())
    # Compiling on a background thread from the first micro-batch, not at the first overflow.
    self.assertTrue(
        engine._dropless_compile is not None  # pylint: disable=protected-access
        or isinstance(engine._compiled_fwd_bwd_dropless, jax.stages.Compiled)  # pylint: disable=protected-access
    )
    engine.fwd_bwd(_payload())
    # Flags are read one micro-batch late: the first was settled once the second was dispatched, which is pending.
    self.assertEqual(engine._dropless_replays, int(overflow))  # pylint: disable=protected-access
    self.assertIsNotNone(engine._pending_overflow_check)  # pylint: disable=protected-access
    engine._resolve_overflow_check()  # pylint: disable=protected-access  # as `update` does first
    loss = float(engine._cached_losses[-1].compute())  # pylint: disable=protected-access
    denom = float(engine._accumulated_denominator)  # pylint: disable=protected-access
    grads = jax.tree.map(jnp.array, engine._accumulated_grads)  # pylint: disable=protected-access
    with mock.patch.object(engine, "record_metrics", wraps=engine.record_metrics) as record:
      engine.update()
    recorded = {call.args[0]: call.args[1] for call in record.call_args_list}
    # Joined by the update at the latest.
    self.assertIsInstance(engine._compiled_fwd_bwd_dropless, jax.stages.Compiled)  # pylint: disable=protected-access
    return denom, grads, loss, recorded["moe_dropless_replays"]

  def test_overflow_reruns_each_micro_batch_dropless_and_accumulates_once(self):
    denom_clean, grads_clean, loss_clean, clean_replays = self._run_step(overflow=False)
    self.assertEqual(clean_replays, 0)
    denom_overflow, grads_overflow, loss_overflow, replays = self._run_step(overflow=True)
    self.assertEqual(replays, 2)
    # Each micro-batch's denominator is summed once, not once per attempt.
    self.assertEqual(denom_overflow, denom_clean)
    # The dropless kernel's (doubled) loss and gradients were kept, the overflowed ones dropped.
    self.assertAlmostEqual(loss_overflow, 2.0 * loss_clean, places=3)
    for g_o, g_c in zip(jax.tree_util.tree_leaves(grads_overflow), jax.tree_util.tree_leaves(grads_clean)):
      self.assertTrue(jnp.allclose(g_o, 2.0 * g_c, rtol=2e-2, atol=1e-5))

  def test_flag_in_the_aux_is_read_under_has_aux_false(self):
    _, _, _, replays = self._run_step(overflow=True, flag="aux", has_aux=False)
    self.assertEqual(replays, 2)

  def test_loss_that_hides_the_flag_fails_loudly(self):
    engine = self._engine(overflow_on_normal_path=True, flag="none")
    with self.assertRaisesRegex(ValueError, "moe_has_overflow"):
      engine.fwd_bwd(_payload())

  def _replays_per_micro_batch(self, first_tokens):
    """Runs one step over micro-batches starting with `first_tokens`; returns the replays after each one's
    fwd_bwd, and those the update recorded."""
    engine = self._engine(overflow_on_normal_path="marked")
    after_each = []
    for first_token in first_tokens:
      engine.fwd_bwd(_payload(first_token))
      after_each.append(engine._dropless_replays)  # pylint: disable=protected-access
    with mock.patch.object(engine, "record_metrics", wraps=engine.record_metrics) as record:
      engine.update()
    recorded = {call.args[0]: call.args[1] for call in record.call_args_list}
    return after_each, recorded["moe_dropless_replays"]

  def test_middle_micro_batch_overflow_is_rerun_after_the_next_dispatch(self):
    after_each, replays = self._replays_per_micro_batch([10, _MARKED, 10])
    self.assertEqual(after_each, [0, 0, 1])
    self.assertEqual(replays, 1)

  def test_last_micro_batch_overflow_is_rerun_by_update(self):
    after_each, replays = self._replays_per_micro_batch([10, 10, _MARKED])
    self.assertEqual(after_each, [0, 0, 0])
    self.assertEqual(replays, 1)

  def test_scoring_and_eval_run_the_dropless_graph(self):
    engine = self._engine(overflow_on_normal_path=False)
    with engine.model_scope() as (model, _, _):
      self.assertTrue(all(m.force_dropless for m in _moes(model)))
    with mock.patch.object(engine, "_eval_kernel", wraps=engine._eval_kernel) as eval_kernel:  # pylint: disable=protected-access
      engine.eval_step(_payload())
    graphdef = eval_kernel.call_args.kwargs["graphdef"]
    graph = nnx.merge(graphdef, *nnx.split(engine.model, nnx.Param, ...)[1:])
    self.assertTrue(all(m.force_dropless for m in _moes(graph)))
    # The live model is untouched.
    self.assertFalse(any(m.force_dropless for m in _moes(engine.model)))

  def test_compile_kernels_adds_the_dropless_kernel_on_an_edited_copy(self):
    engine = self._engine(overflow_on_normal_path=False)
    compiled = engine.compile_kernels(_payload())
    self.assertEqual(set(compiled), {*maxtext_engine.KERNEL_NAMES, maxtext_engine.DROPLESS_FWD_BWD})
    graph = nnx.merge(engine._dropless_graphdef, *nnx.split(engine.model, nnx.Param, ...)[1:])  # pylint: disable=protected-access
    for moe in _moes(graph):
      self.assertTrue(moe.force_dropless and moe.dropless_scan_chunks)
    # The live model is untouched.
    self.assertFalse(any(m.force_dropless for m in _moes(engine.model)))


if __name__ == "__main__":
  unittest.main()
