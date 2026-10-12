# Copyright 2025-2026 Google LLC
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

"""Unit tests for the NNX paths of loss_fn / train_step / eval_step in pre_train.train.

These tests exercise the NNX branches without standing up a real Transformer or
data pipeline. We use a tiny NNX module that mimics the call signature the
production loss_fn uses (decoder_input_tokens, decoder_positions, ...).
"""

from dataclasses import dataclass
import datetime
import functools
import types as pytypes
import unittest
from unittest import mock

from flax import nnx
from flax.nnx import variablelib
import jax
import jax.numpy as jnp
from maxtext.layers import nnx_scan
import numpy as np
from maxtext.common import train_state_nnx
from maxtext.common import metric_logger
from maxtext.common.metric_logger import record_activation_metrics
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train as pre_train
from maxtext.layers.multi_token_prediction import mtp_losses
from maxtext.utils import gradient_accumulation
import optax


@dataclass
class _Cfg:
  """Subset of HyperParameters used by loss_fn / train_step / eval_step."""

  model_name: str = ""
  micro_batch_size_to_train_on: int = 2
  micro_batch_size_to_eval_on: int = 2
  vocab_size: int = 8
  z_loss_multiplier: float = 0.0
  enable_dropout: bool = False
  use_multimodal: bool = False
  use_indexer: bool = False
  indexer_sparse_training: bool = False
  indexer_loss_scaling_factor: float = 0.0
  num_vocab_tiling: int = 1
  num_experts: int = 1
  moe_dropless_fallback: str | None = None
  routed_bias: bool = False
  routed_bias_update_rate: float = 0.0
  mtp_num_layers: int = 0
  mtp_eval_target_module: int = 0
  use_qk_clip: bool = False
  use_tunix_gradient_accumulation: bool = False
  gradient_accumulation_steps: int = 1
  shard_optimizer_over_data: bool = False
  optimizer_memory_host_offload: bool = False
  parameter_memory_host_offload: bool = False
  gradient_clipping_threshold: float = 0.0
  grad_dtype: jnp.dtype = jnp.float32
  record_internal_nn_metrics: bool = False
  skip_step_on_spikes: bool = False
  shard_mode: int = 0  # ShardMode.AUTO
  debug_sharding: bool = False
  weight_sparsity_n: int = 0
  weight_sparsity_m: int = 0


class _TinyDecoder(nnx.Module):
  """Mimics NNXDecoder.__call__ enough for loss_fn to run end-to-end.

  Returns logits of shape [batch, seq_len, vocab_size]. Ignores all multimodal
  / dropout / target arguments — they exist only to match the keyword signature.
  """

  def __init__(self, vocab_size: int, hidden: int, rngs: nnx.Rngs):
    self.embed = nnx.Embed(vocab_size, hidden, rngs=rngs)
    self.proj = nnx.Linear(hidden, vocab_size, rngs=rngs)
    # loss_fn shards activations against model.mesh, so the stub needs one.
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    h = self.embed(decoder_input_tokens)
    return self.proj(h)


_OVERWRITE_WITH_GRADIENT = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)


class _CustomGradientModel(nnx.Module):
  """Small model with custom state differentiated outside the optimizer."""

  def __init__(self):
    self.weight = nnx.Param(jnp.array(1.0))
    self.custom_state = _OVERWRITE_WITH_GRADIENT(jnp.array(2.0))


class GateLogit(nnx.Module):
  """Router gate stub holding bias parameter."""

  def __init__(self, bias_shape):
    self.bias = nnx.Param(jnp.zeros(bias_shape))


class _MoEBiasStub(nnx.Module):
  """Sows bias updates for testing."""

  def __init__(self, bias_shape, sow_shape, update_val: float = 1.0):
    self.gate = GateLogit(bias_shape)
    self.MoeBlock_0 = self
    self.DeepSeekMoeBlock_0 = self
    self.transformer_layer = self
    self.moe_layers = self
    self.sow_shape = sow_shape
    self.update_val = update_val

  def __call__(self):
    self.sow(
        nnx.Intermediate,
        "moe_bias_updates",
        jnp.full(self.sow_shape, self.update_val),
    )


class _TinyDecoderMoEBias(_TinyDecoder):
  """`_TinyDecoder` with decoder MoE layers that sow `moe_bias_updates`."""

  def __init__(self, vocab_size: int, hidden: int, rngs: nnx.Rngs):
    super().__init__(vocab_size, hidden, rngs=rngs)
    # Using MoEBiasVar, expected bias shape is (num_layers, num_experts)
    self.decoder = _MoEBiasStub(bias_shape=(2, 3), sow_shape=(2, 3), update_val=1.0)

  def __call__(self, decoder_input_tokens, decoder_positions, **kwargs):
    out = super().__call__(decoder_input_tokens, decoder_positions, **kwargs)
    self.decoder()
    return out


class _TinyDecoderMoEBiasWithMTP(_TinyDecoderMoEBias):
  """`_TinyDecoderMoEBias` that also includes MTP layers."""

  def __init__(
      self,
      vocab_size: int,
      hidden: int,
      rngs: nnx.Rngs,
      num_mtp_layers: int = 2,
  ):
    super().__init__(vocab_size, hidden, rngs=rngs)
    self.num_mtp_layers = num_mtp_layers
    # Use distinct update values (e.g. 2.0 for layer 1, 3.0 for layer 2)
    self.mtp_block = nnx.Dict(
        {
            f"mtp_layer_{i + 1}": _MoEBiasStub(bias_shape=(3,), sow_shape=(3,), update_val=float(i + 2))
            for i in range(num_mtp_layers)
        }
    )

  def __call__(self, decoder_input_tokens, decoder_positions, **kwargs):
    out = super().__call__(decoder_input_tokens, decoder_positions, **kwargs)
    for i in range(self.num_mtp_layers):
      self.mtp_block[f"mtp_layer_{i + 1}"]()
    return out


class _TinyDecoderMoEOverflow(_TinyDecoder):
  """`_TinyDecoder` with a layer that sows moe_has_overflow, like RoutedMoE.sparse_matmul."""

  def __init__(self, vocab_size: int, hidden: int, rngs: nnx.Rngs, has_overflow: bool):
    super().__init__(vocab_size, hidden, rngs=rngs)
    self.has_overflow = has_overflow

  def __call__(self, decoder_input_tokens, decoder_positions, **kwargs):
    out = super().__call__(decoder_input_tokens, decoder_positions, **kwargs)
    self.sow(nnx.Intermediate, "moe_has_overflow", jnp.bool_(self.has_overflow))
    return out


from maxtext.layers.attention_mla import indexer_losses


class _MockIndexerLayer(nnx.Module):

  def __init__(self, rngs):
    self.mock_val = nnx.Param(jnp.zeros(()))

  def __call__(self, carry):
    self.sow(indexer_losses, "indexer_loss", self.mock_val.get_value())
    return carry


class _TinyDecoderIndexerLoss(_TinyDecoder):
  """_TinyDecoder that also sows indexer_loss via a scanned layer."""

  def __init__(self, vocab_size: int, hidden: int, rngs: nnx.Rngs):
    super().__init__(vocab_size, hidden, rngs)

    self.layers = nnx_scan.create_scanned_layers(
        _MockIndexerLayer,
        length=2,
        param_scan_axis=0,
        metadata_axis_name="layer",
        rngs=rngs,
    )

    # Overwrite the empty parameters generated with our mock test metrics!
    _, params, other = nnx.split(self.layers, nnx.Param, ...)
    params.mock_val.value = jnp.array([0.25, 0.75])
    nnx.update(self.layers, params, other)

  def __call__(self, decoder_input_tokens, decoder_positions, **kwargs):
    out = super().__call__(decoder_input_tokens, decoder_positions, **kwargs)

    def apply_fn(module, carry):
      return module(carry)

    nnx_scan.apply_scanned_layers(self.layers, carry=None, length=2, param_scan_axis=0, apply_fn=apply_fn)
    return out


def _make_data(batch=2, seq=4, vocab=8):
  return {
      "inputs": jnp.zeros((batch, seq), dtype=jnp.int32),
      "inputs_position": jnp.broadcast_to(jnp.arange(seq), (batch, seq)),
      "inputs_segmentation": jnp.ones((batch, seq), dtype=jnp.int32),
      "targets": jnp.zeros((batch, seq), dtype=jnp.int32),
      "targets_segmentation": jnp.ones((batch, seq), dtype=jnp.int32),
  }


def _build_state():
  cfg = _Cfg()
  model = _TinyDecoder(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
  ts = train_state_nnx.TrainStateNNX(model, optimizer)
  return cfg, ts


class TestLossFnNNX(unittest.TestCase):
  """Cover the NNX branch of loss_fn (lines 178-213)."""

  def test_returns_loss_and_full_aux_dict(self):
    cfg, ts = _build_state()
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    self.assertTrue(jnp.isfinite(loss))
    # Aux schema relied on by train_step / eval_step / GA.
    for key in (
        "intermediate_outputs",
        "xent_sum",
        "z_loss",
        "total_weights",
        "moe_lb_loss",
        "indexer_loss",
        "moe_bias_updates",
        "mtp_loss",
    ):
      self.assertIn(key, aux)
    # NNX intermediates are captured into a pure-dict snapshot.
    self.assertIsInstance(aux["intermediate_outputs"], dict)

  def test_logits_preserved_during_eval_with_mtp(self):
    """Verifies logits is stored in intermediate_outputs only during eval with MTP target."""
    cfg, ts = _build_state()
    cfg.mtp_eval_target_module = 1
    data = _make_data(batch=cfg.micro_batch_size_to_eval_on, vocab=cfg.vocab_size)

    # 1. During eval: logits must be preserved for acceptance rate calculation
    _, aux_eval = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=False)
    self.assertIn("logits", aux_eval["intermediate_outputs"])

    # 2. During training: logits must NOT be stored to avoid memory bloat
    _, aux_train = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    self.assertNotIn("logits", aux_train["intermediate_outputs"])

  def test_eval_mode_truncates_to_eval_micro_batch(self):
    cfg, ts = _build_state()
    cfg.micro_batch_size_to_eval_on = 1
    data = _make_data(batch=2, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=False)
    self.assertTrue(jnp.isfinite(loss))
    # eval truncated batch to 1 → total_weights = seq_len * 1
    self.assertEqual(int(aux["total_weights"]), data["targets_segmentation"].shape[1])

  def test_multimodal_model_accepts_text_only_batch(self):
    cfg, ts = _build_state()
    cfg.use_multimodal = True
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)

    loss, _ = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)

    self.assertTrue(jnp.isfinite(loss))

  def test_indexer_dense_warmup_skips_xent(self):
    cfg, ts = _build_state()
    cfg.use_indexer = True
    cfg.indexer_sparse_training = False
    cfg.indexer_loss_scaling_factor = 0.1
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    # When dense warm-up is active the loss_fn skips the main loss entirely.
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertEqual(float(loss), 0.0)

  def test_indexer_warmup_precedes_vocab_tiling(self):
    # The indexer dense warm-up branch must be checked before the num_vocab_tiling>1
    # branch. With the order reversed, a warm-up step with tiling on ran the
    # vocab-tiling loss instead of skipping xent. With both on, xent must still be 0.
    cfg, ts = _build_state()
    cfg.use_indexer = True
    cfg.indexer_sparse_training = False
    cfg.indexer_loss_scaling_factor = 0.1
    cfg.num_vocab_tiling = 2
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertEqual(float(loss), 0.0)

  def test_indexer_without_indexer_loss_keeps_xent(self):
    # With indexer_loss_scaling_factor == 0 there is no indexer loss, so the LM loss must be kept;
    # skipping it would make the total objective zero.
    cfg, ts = _build_state()
    cfg.use_indexer = True
    cfg.indexer_sparse_training = False
    cfg.indexer_loss_scaling_factor = 0.0
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    self.assertGreater(float(aux["xent_sum"]), 0.0)
    self.assertGreater(float(loss), 0.0)

  def test_indexer_losses_harvested_and_injected_into_loss(self):
    cfg = _Cfg()
    cfg.use_indexer = True
    cfg.indexer_sparse_training = True
    cfg.indexer_loss_scaling_factor = 0.1
    model = _TinyDecoderIndexerLoss(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0))
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)

    loss_without_indexer, _ = pre_train.loss_fn(
        _TinyDecoder(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0)), cfg, data, None, None, is_train=True
    )

    loss, aux = pre_train.loss_fn(model, cfg, data, None, None, is_train=True)
    expected_indexer_loss = 0.5  # mean of 0.25 and 0.75

    self.assertTrue(jnp.isfinite(loss))
    self.assertAlmostEqual(float(aux["indexer_loss"]), expected_indexer_loss, places=5)
    self.assertAlmostEqual(float(loss), float(loss_without_indexer) + expected_indexer_loss, places=5)


class TestTrainStepNNX(unittest.TestCase):
  """Cover the NNX branch of train_step (the diff_wrapper / nnx.update path)."""

  def test_train_step_returns_state_and_metrics(self):
    cfg, ts = _build_state()
    state_graphdef, state_pure = nnx.split(ts)

    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    new_state, metrics = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    # NNX path returns nnx.State (via nnx.state(new_state)) and a metrics dict.
    self.assertIsInstance(new_state, nnx.State)
    self.assertIn("scalar", metrics)
    self.assertIn("learning/loss", metrics["scalar"])
    self.assertIn("learning/grad_norm", metrics["scalar"])
    self.assertIn("learning/param_norm", metrics["scalar"])
    self.assertTrue(jnp.isfinite(metrics["scalar"]["learning/loss"]))

  def test_train_step_increments_optimizer_step(self):
    cfg, ts = _build_state()
    state_graphdef, state_pure = nnx.split(ts)
    pre_step = int(state_pure.optimizer.step.get_value())
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    new_state, _ = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    self.assertEqual(int(new_state.optimizer.step.get_value()), pre_step + 1)

  def test_train_step_with_gradient_clipping(self):
    """The clipping branch (gradient_clipping_threshold > 0) must run without raising."""
    cfg, ts = _build_state()
    cfg.gradient_clipping_threshold = 1.0
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    new_state, metrics = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    self.assertIsInstance(new_state, nnx.State)
    self.assertTrue(jnp.isfinite(metrics["scalar"]["learning/loss"]))

  def test_custom_state_keeps_gradient_update(self):
    """Checks that old custom state does not overwrite its gradient update."""
    cfg = _Cfg()
    model = _CustomGradientModel()
    ts = train_state_nnx.TrainStateNNX(
        model,
        nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param),
    )
    state_graphdef, state_pure = nnx.split(ts)

    def fake_loss_fn(local_model, *_args, **_kwargs):
      loss = local_model.weight.get_value() + 3.0 * local_model.custom_state.get_value()
      return loss, {
          "intermediate_outputs": {},
          "xent_sum": loss,
          "z_loss": jnp.array(0.0),
          "total_weights": jnp.array(1.0),
          "moe_lb_loss": jnp.array(0.0),
          "indexer_loss": jnp.array(0.0),
          "moe_bias_updates": None,
          "mtp_moe_bias_updates": None,
          "mtp_loss": jnp.array(0.0),
          "batch_stats": None,
      }

    original_loss_fn = pre_train.loss_fn
    try:
      pre_train.loss_fn = fake_loss_fn
      new_state, _ = pre_train.train_step(
          state_graphdef,
          cfg,
          state_mesh_shardings=None,
          params_shardings=None,
          state=state_pure,
          data={},
      )
    finally:
      pre_train.loss_fn = original_loss_fn

    # The custom gradient is 3.0. It must not be replaced by the old state (2.0).
    np.testing.assert_allclose(np.asarray(new_state.model.custom_state.get_value()), 3.0)


class TestMoeOverflowLoggingNNX(unittest.TestCase):
  """Covers train_step/eval_step surfacing has_moe_overflow for training_loop_iteration's retry branches."""

  def _build_state(self, has_overflow, step_fallback):
    cfg = _Cfg(moe_dropless_fallback="step" if step_fallback else None)
    model = _TinyDecoderMoEOverflow(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), has_overflow=has_overflow)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    return cfg, train_state_nnx.TrainStateNNX(model, optimizer)

  def _metrics(self, has_overflow, step_fallback):
    cfg, ts = self._build_state(has_overflow, step_fallback)
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    _, metrics = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    return metrics

  def _eval_metrics(self, has_overflow, step_fallback):
    """Runs eval_step and returns its metrics dict."""
    cfg, ts = self._build_state(has_overflow, step_fallback)
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_eval_on, vocab=cfg.vocab_size)
    return pre_train.eval_step(state_graphdef, cfg, state_pure, data)

  def test_surfaces_overflow_when_flag_on(self):
    metrics = self._metrics(has_overflow=True, step_fallback=True)
    self.assertTrue(bool(metrics["has_moe_overflow"]))

  def test_no_overflow_when_flag_on(self):
    metrics = self._metrics(has_overflow=False, step_fallback=True)
    self.assertFalse(bool(metrics["has_moe_overflow"]))

  def test_key_absent_when_flag_off(self):
    """Key must be absent (not just False) when off: an always-present key would

    change every model's compiled train_step output, even non-MoE ones.
    """
    metrics = self._metrics(has_overflow=True, step_fallback=False)
    self.assertNotIn("has_moe_overflow", metrics)

  def test_eval_step_also_surfaces_overflow(self):
    """eval_step must surface has_moe_overflow the same way train_step does -- it's what
    training_loop_iteration's eval-retry branch checks before replaying with the dropless step.
    """
    metrics = self._eval_metrics(has_overflow=True, step_fallback=True)
    self.assertTrue(bool(metrics["has_moe_overflow"]))

  def test_eval_step_key_absent_when_flag_off(self):
    metrics = self._eval_metrics(has_overflow=True, step_fallback=False)
    self.assertNotIn("has_moe_overflow", metrics)

  def _before_after(self, has_overflow, step_fallback, jit=False):
    """Runs train_step, eagerly or under jit, and returns the input/output state leaves."""
    cfg, ts = self._build_state(has_overflow, step_fallback)
    state_graphdef, state_pure = nnx.split(ts)
    before = jax.tree_util.tree_leaves(state_pure)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)

    def step(state, data):
      return pre_train.train_step(
          state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state, data=data
      )

    new_state, _ = (jax.jit(step) if jit else step)(state_pure, data)
    return before, jax.tree_util.tree_leaves(new_state)

  def test_state_is_rolled_back_on_overflow(self):
    for jit in (False, True):
      with self.subTest(jit=jit):
        before, after = self._before_after(has_overflow=True, step_fallback=True, jit=jit)
        self.assertEqual(len(before), len(after))
        for i, (b, a) in enumerate(zip(before, after)):
          self.assertTrue(bool(jnp.array_equal(b, a)), f"state leaf {i} changed despite a token-drop rollback")

  def test_state_advances_without_overflow(self):
    for jit in (False, True):
      with self.subTest(jit=jit):
        before, after = self._before_after(has_overflow=False, step_fallback=True, jit=jit)
        self.assertEqual(len(before), len(after))
        self.assertTrue(
            any(not bool(jnp.array_equal(b, a)) for b, a in zip(before, after)),
            "train_step left the state untouched with has_overflow=False -- the rollback test proves nothing.",
        )


class TestEvalStepNNX(unittest.TestCase):
  """Cover the NNX branch of eval_step (lines 568-570)."""

  def test_eval_step_returns_metrics(self):
    cfg, ts = _build_state()
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_eval_on, vocab=cfg.vocab_size)
    metrics = pre_train.eval_step(state_graphdef, cfg, state_pure, data)
    self.assertIn("scalar", metrics)
    for key in (
        "evaluation/loss",
        "evaluation/total_loss",
        "evaluation/total_weights",
        "evaluation/moe_lb_loss",
    ):
      self.assertIn(key, metrics["scalar"])
    self.assertTrue(jnp.isfinite(metrics["scalar"]["evaluation/loss"]))


class TestSkipStepOnSpikesNNX(unittest.TestCase):
  """The NNX optimizer must actually skip a loss/grad spike — i.e. apply_gradients forwards
  loss/grad_norm to the GradientTransformationExtraArgs, and a skipped step freezes params."""

  def _is_skipped(self, optimizer):
    return bool(nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]["is_skipped"])

  def test_spike_is_skipped_and_params_frozen(self):
    model = _TinyDecoder(8, hidden=4, rngs=nnx.Rngs(0))
    tx = optimizers.skip_step_on_spikes(optax.sgd(0.1), interval=4, scaling_factor=6.0)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)
    grads = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))

    # Prime a stable baseline (mean≈1, std≈0); these are applied, not skipped.
    for _ in range(3):
      state.apply_gradients(grads, loss=jnp.float32(1.0), grad_norm=jnp.float32(1.0))
    self.assertFalse(self._is_skipped(optimizer))

    before = [np.asarray(x) for x in jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))]
    # A large spike must be skipped (params unchanged). If apply_gradients did NOT forward
    # loss/grad_norm, the optimizer would never skip and this would fail.
    state.apply_gradients(grads, loss=jnp.float32(1e3), grad_norm=jnp.float32(1e3))
    self.assertTrue(self._is_skipped(optimizer))
    after = [np.asarray(x) for x in jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))]
    for b, a in zip(before, after):
      np.testing.assert_allclose(a, b)


class TestRoutedBiasReadNNX(unittest.TestCase):
  """loss_fn must find the DeepSeek `moe_bias_updates` intermediate on the NNX (model-rooted) shape."""

  def test_routed_bias_update_found_by_suffix(self):
    cfg = _Cfg()
    cfg.routed_bias = True
    cfg.routed_bias_update_rate = 0.001
    model = _TinyDecoderMoEBias(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0))
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    _, aux = pre_train.loss_fn(model, cfg, data, None, None, is_train=True)
    self.assertIsNotNone(aux["moe_bias_updates"])
    np.testing.assert_allclose(np.asarray(aux["moe_bias_updates"][0]), np.ones((2, 3)))

  def test_loss_fn_extracts_mtp_moe_bias_updates(self):
    """Verifies loss_fn returns mtp_moe_bias_updates with correct shapes and values."""
    cfg = _Cfg()
    cfg.routed_bias = True
    cfg.routed_bias_update_rate = 0.001
    cfg.mtp_num_layers = 2
    model = _TinyDecoderMoEBiasWithMTP(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), num_mtp_layers=2)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)

    _, aux = pre_train.loss_fn(model, cfg, data, None, None, is_train=True)
    self.assertIsNotNone(aux["mtp_moe_bias_updates"])
    self.assertEqual(len(aux["mtp_moe_bias_updates"]), 2)
    # Layer 1 has update 2.0 of shape (3,) and Layer 2 has update 3.0 of shape (3,)
    np.testing.assert_allclose(np.asarray(aux["mtp_moe_bias_updates"][0]), np.full((3,), 2.0))
    np.testing.assert_allclose(np.asarray(aux["mtp_moe_bias_updates"][1]), np.full((3,), 3.0))

  def test_train_step_updates_decoder_and_mtp_routed_biases(self):
    cfg = _Cfg()
    cfg.routed_bias = True
    cfg.routed_bias_update_rate = 0.001
    cfg.mtp_num_layers = 2
    model = _TinyDecoderMoEBiasWithMTP(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), num_mtp_layers=2)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    ts = train_state_nnx.TrainStateNNX(model, optimizer)
    state_graphdef, state_pure = nnx.split(ts)

    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    new_state, _ = pre_train.train_step(
        state_graphdef,
        cfg,
        state_mesh_shardings=None,
        params_shardings=None,
        state=state_pure,
        data=data,
    )
    # Scanned decoder bias is (num_layers=2, num_experts=3) with update_val=1.0
    dec_gate = new_state.model.decoder.gate
    np.testing.assert_allclose(
        np.asarray(dec_gate.bias.value),
        np.full((2, 3), 1.0),
    )
    # Distinct updates for each MTP layer (2.0 for layer 1, 3.0 for layer 2)
    mtp1_gate = new_state.model.mtp_block.mtp_layer_1.gate
    np.testing.assert_allclose(
        np.asarray(mtp1_gate.bias.value),
        np.full((3,), 2.0),
    )
    mtp2_gate = new_state.model.mtp_block.mtp_layer_2.gate
    np.testing.assert_allclose(
        np.asarray(mtp2_gate.bias.value),
        np.full((3,), 3.0),
    )

  def test_routed_bias_disabled_returns_none(self):
    cfg = _Cfg()  # routed_bias=False
    model = _TinyDecoderMoEBias(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0))
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    _, aux = pre_train.loss_fn(model, cfg, data, None, None, is_train=True)
    self.assertIsNone(aux["moe_bias_updates"])


class _AuxLossMTPBlock(nnx.Module):
  """Sows MTP loss components (token-summed loss and token count) the way MultiTokenPredictionBlock does."""

  def __init__(self, hidden: int, rngs: nnx.Rngs):
    self.v = nnx.Param(jax.random.normal(rngs.params(), (hidden,)))

  def __call__(self, h):
    per_token = jnp.tanh(h @ self.v[...]) ** 2
    self.losses = mtp_losses(jnp.stack([jnp.sum(per_token)]))
    self.weights = mtp_losses(jnp.stack([jnp.array(per_token.size, jnp.float32)]))


class _TinyDecoderAuxLosses(_TinyDecoder):
  """`_TinyDecoder` that also produces MoE load-balance, indexer and MTP losses.

  Each auxiliary loss is a per-token mean, so for equal-sized microbatches the
  mean over microbatches equals the global-batch value and the GA gradient can
  be compared exactly against the non-GA gradient.
  """

  def __init__(self, vocab_size: int, hidden: int, rngs: nnx.Rngs):
    super().__init__(vocab_size, hidden, rngs=rngs)
    self.lb_w = nnx.Param(jax.random.normal(rngs.params(), (hidden,)))
    self.indexer_w = nnx.Param(jax.random.normal(rngs.params(), (hidden,)))
    self.mtp_block = _AuxLossMTPBlock(hidden, rngs)

  def __call__(self, decoder_input_tokens, decoder_positions, **kwargs):
    out = super().__call__(decoder_input_tokens, decoder_positions, **kwargs)
    h = self.embed(decoder_input_tokens)
    self.sow(nnx.Intermediate, "moe_lb_loss", jnp.atleast_1d(jnp.mean(jax.nn.sigmoid(h @ self.lb_w[...]))))
    self.sow(indexer_losses, "indexer_loss", jnp.mean(jnp.sin(h @ self.indexer_w[...]) ** 2))
    self.mtp_block(h)
    return out


def _params_shardings(model):
  """Replicated NamedSharding tree on the model's mesh, shaped like nnx.split(model, nnx.Param, ...)[1]."""
  _, params, _ = nnx.split(model, nnx.Param, ...)
  ns = jax.sharding.NamedSharding(model.mesh, jax.sharding.PartitionSpec())
  return jax.tree.map(lambda _: ns, params)


class TestGradientAccumulationAuxLosses(unittest.TestCase):
  """Under GA the MTP, indexer and load-balance gradients match the non-GA gradient of the global batch.

  The aux-loss weighting and the logged aux values under GA come from #5401 (loss_fn's _add_aux_loss and the mean
  over microbatches in gradient_accumulation_loss_and_grad); these tests check them together with the cotangent
  scale (loss / S, gradient / (W / S)), which must leave every gradient, aux terms included, unchanged.
  """

  def _cfg(self, ga_steps, batch, seq=4):
    """Config with MTP, indexer and load-balance losses on; GA microbatches of batch // ga_steps."""
    cfg = _Cfg(
        gradient_accumulation_steps=ga_steps,
        micro_batch_size_to_train_on=batch // ga_steps,
        num_experts=4,
        mtp_num_layers=1,
        use_indexer=True,
        indexer_sparse_training=True,
        indexer_loss_scaling_factor=1.0,
    )
    cfg.mtp_loss_scaling_factor = 0.5  # not a _Cfg field; read by calculate_mtp_loss
    # Read by gradient_accumulation's cotangent scale (the static global-batch token capacity).
    cfg.global_batch_size_to_train_on = batch
    cfg.max_target_length = seq
    return cfg

  def _data(self, batch=4, seq=4):
    data = _make_data(batch=batch, seq=seq)
    data["inputs"] = jax.random.randint(jax.random.PRNGKey(1), (batch, seq), 0, 8)
    data["targets"] = jax.random.randint(jax.random.PRNGKey(2), (batch, seq), 0, 8)
    return data

  def _grads_as_dict(self, grads):
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_leaves_with_path(grads)}

  def test_ga1_loss_unchanged(self):
    data = self._data()
    model = _TinyDecoderAuxLosses(8, hidden=4, rngs=nnx.Rngs(0))
    loss, aux = pre_train.loss_fn(model, self._cfg(1, 4), data, None, None, is_train=True)
    expected = aux["xent_sum"] / (aux["total_weights"] + 1e-8) + aux["mtp_loss"] + aux["indexer_loss"]
    expected = expected + aux["moe_lb_loss"]
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-6)

  def test_ga2_aux_gradients_match_global_batch(self):
    data = self._data()
    ga_model = _TinyDecoderAuxLosses(8, hidden=4, rngs=nnx.Rngs(0))
    ga_loss, aux, ga_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        pre_train.loss_fn, self._cfg(2, 4), ga_model, None, _params_shardings(ga_model), dict(data), None
    )
    self.assertGreater(float(aux["mtp_loss"]), 0.0)
    self.assertGreater(float(aux["indexer_loss"]), 0.0)
    self.assertGreater(float(aux["moe_lb_loss"]), 0.0)

    ref_model = _TinyDecoderAuxLosses(8, hidden=4, rngs=nnx.Rngs(0))
    grad_fn = nnx.value_and_grad(pre_train.loss_fn, argnums=0, has_aux=True)
    (ref_loss, ref_aux), ref_grads = grad_fn(ref_model, self._cfg(1, 4), dict(data), None, None, is_train=True)

    got, want = self._grads_as_dict(ga_grads), self._grads_as_dict(ref_grads)
    self.assertEqual(set(got), set(want))
    for key in want:
      np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-7, err_msg=key)
    # The aux-only parameters get a gradient of ordinary size (not ~1/tokens).
    for key in want:
      if "lb_w" in key or "indexer_w" in key or "mtp_block" in key:
        self.assertGreater(np.abs(want[key]).max(), 1e-3, key)
    # The reported loss and the logged aux losses equal the GA=1 values of the same global batch (the aux losses are
    # per-token means and the two microbatches carry equal token counts).
    np.testing.assert_allclose(float(ga_loss), float(ref_loss), rtol=1e-5)
    for key in ("moe_lb_loss", "indexer_loss", "mtp_loss"):
      np.testing.assert_allclose(float(aux[key]), float(ref_aux[key]), rtol=1e-5, err_msg=key)


class TestRecordActivationMetricsParity(unittest.TestCase):
  """record_activation_metrics must yield identical metrics for Linen- and NNX-shaped intermediates.

  Linen sows into the "intermediates" collection; NNX's `nnx.pop(...).to_pure_dict()` is
  model-rooted with no "intermediates" prefix. The fix routes the NNX shape through a
  suffix collector — this test pins that both shapes produce the same per-layer numbers.
  """

  def _metrics(self, intermediates, scan_layers, num_layers):
    cfg = pytypes.SimpleNamespace(scan_layers=scan_layers, num_decoder_layers=num_layers)
    out = {"scalar": {}}
    record_activation_metrics(out, intermediates, cfg)
    return out["scalar"]

  def test_scanned_layout_linen_matches_nnx(self):
    num_layers = 3
    mean, std, fz = jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.1, 1.2]), jnp.array([0.5, 0.4, 0.3])
    triples = {"activation_mean": (mean,), "activation_stdev": (std,), "activation_fraction_zero": (fz,)}
    # Linen scanned: intermediates/decoder/decoder/<key>[0][layer]
    linen = {"intermediates": {"decoder": {"decoder": triples}}}
    # NNX scanned: model-rooted, one stacked array per key (no "intermediates" prefix)
    nnx_shaped = {"decoder": {"layers": triples}}

    m_linen = self._metrics(linen, scan_layers=True, num_layers=num_layers)
    m_nnx = self._metrics(nnx_shaped, scan_layers=True, num_layers=num_layers)
    self.assertEqual(set(m_linen), set(m_nnx))
    for key, expected in m_linen.items():
      np.testing.assert_allclose(np.asarray(m_nnx[key]), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(m_nnx["activ_mean/layer_001"]), 0.2)

  def test_unscanned_layout_linen_matches_nnx(self):
    num_layers = 3
    means, stds, fzs = [0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [0.5, 0.4, 0.3]

    def per_layer(d, n):
      return {
          "activation_mean": (jnp.array(d[0][n]),),
          "activation_stdev": (jnp.array(d[1][n]),),
          "activation_fraction_zero": (jnp.array(d[2][n]),),
      }

    data = (means, stds, fzs)
    # Linen unscanned: intermediates/decoder/layers_<n>/<key>[0]
    linen = {"intermediates": {"decoder": {f"layers_{n}": per_layer(data, n) for n in range(num_layers)}}}
    # NNX unscanned: model-rooted per-layer entries (one leaf per layer, matched by suffix)
    nnx_shaped = {"decoder": {f"layers_{n}": per_layer(data, n) for n in range(num_layers)}}

    m_linen = self._metrics(linen, scan_layers=False, num_layers=num_layers)
    m_nnx = self._metrics(nnx_shaped, scan_layers=False, num_layers=num_layers)
    self.assertEqual(set(m_linen), set(m_nnx))
    for key, expected in m_linen.items():
      np.testing.assert_allclose(np.asarray(m_nnx[key]), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(m_nnx["activ_stdev/layer_002"]), 1.2)


class TestTrainingLoopIterationEvalRetry(unittest.TestCase):
  """Covers training_loop_iteration's host-side eval-retry branch: p_eval_step_dropless is
  invoked (and its metrics used) only when moe_dropless_fallback="step", a dropless eval
  step is available, and the primary eval step reports has_moe_overflow.
  """

  _PRIMARY_LOSS = 999.0
  _DROPLESS_LOSS = -1.0

  def _run(
      self,
      step_fallback,
      has_overflow,
      with_dropless,
      retry_dropless_first_steps=0,
      train_calls=None,
      with_first_phase=False,
      first_phase_overflow=False,
      required_rbf=None,
  ):
    """Runs training_loop_iteration with fake step fns and returns (eval loss used, dropless call count).

    When train_calls is a list, a fake p_train_step_dropless is provided and each train step call appends
    "normal", "first_phase" or "dropless" to it. with_first_phase adds a fake p_train_step_first_phase whose
    metrics report has_moe_overflow=first_phase_overflow. required_rbf (a list) turns on
    log_required_ragged_buffer_factor and makes every fake train step return it as metrics["moe_required_rbf"].
    """

    def _train_metrics(overflow=False):
      metrics = {"scalar": {}, "scalars": {}, "has_moe_overflow": jnp.bool_(overflow)}
      if required_rbf is not None:
        metrics["moe_required_rbf"] = jnp.array(required_rbf, dtype=jnp.float32)
      return metrics

    def p_train_step(state, batch, *rng_args):
      del batch, rng_args
      if train_calls is not None:
        train_calls.append("normal")
      return state, _train_metrics()

    def p_train_step_dropless(state, batch, *rng_args):
      del batch, rng_args
      train_calls.append("dropless")
      return state, _train_metrics()

    def p_train_step_first_phase(state, batch, *rng_args):
      del batch, rng_args
      train_calls.append("first_phase")
      return state, _train_metrics(overflow=first_phase_overflow)

    def p_eval_step(state, batch, *rng_args):
      del state, batch, rng_args
      metrics = {"scalar": {"evaluation/total_loss": jnp.array(self._PRIMARY_LOSS)}}
      if step_fallback:
        metrics["has_moe_overflow"] = jnp.bool_(has_overflow)
      return metrics

    dropless_calls = []

    def p_eval_step_dropless(state, batch, *rng_args):
      del state, batch, rng_args
      dropless_calls.append(1)
      return {"scalar": {"evaluation/total_loss": jnp.array(self._DROPLESS_LOSS)}}

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1), ("data",))
    cfg = pytypes.SimpleNamespace(
        elastic_enabled=False,
        enable_diloco=False,
        moe_dropless_fallback="step" if step_fallback else None,
        retry_dropless_first_steps=retry_dropless_first_steps,
        first_phase_ragged_buffer_factor=5.0 if with_first_phase else 0.0,
        log_required_ragged_buffer_factor=required_rbf is not None,
        logical_axis_rules_for_eval=(),
    )
    metric_logger_instance = mock.MagicMock()
    data_loader = mock.MagicMock()
    prof = mock.MagicMock()

    jax_device_state = {
        "state": "fake_state",
        "init_rng": None,
        "mesh": mesh,
        "p_train_step": p_train_step,
        "p_train_step_dropless": p_train_step_dropless if train_calls is not None else None,
        "p_train_step_first_phase": p_train_step_first_phase if with_first_phase else None,
        "p_eval_step": p_eval_step,
        "p_eval_step_dropless": p_eval_step_dropless if with_dropless else None,
    }
    python_vars = {
        "step": 0,
        "last_step_completion": datetime.datetime.now(),
        "data_loader": data_loader,
        "rampup_manager": None,
        "recorder": None,
        "checkpoint_manager": None,
        "data_iterator": None,
        "eval_data_iterator": [jnp.zeros((1,))],
        "metric_logger_instance": metric_logger_instance,
        "prof": prof,
    }
    immutable_data = {
        "config": cfg,
        "logical_axis_rules_for_train": (),
        "logical_axis_rules_for_eval": (),
        "eval_interval": 1,
        "eval_steps": 0,
        "start_step": -1,  # != step, so the print_mem_stats branch is skipped
        "eval_start_step": 0,
        "dump_hlo": False,
        "dump_step": -1,
        "dump_hlo_local_dir": None,
        "dump_hlo_gcs_dir": None,
        "dump_hlo_module_name": None,
        "dump_hlo_delete_local_after": False,
        "dump_hlo_upload_all": False,
    }
    with mock.patch.object(pre_train.sharding, "get_input_data_sharding", return_value=None):
      pre_train.training_loop_iteration(jax_device_state, python_vars, immutable_data)

    eval_call = next(
        c for c in metric_logger_instance.buffer_and_write_metrics.call_args_list if not c.kwargs["is_training"]
    )
    used_loss = float(eval_call.args[0]["scalar"]["evaluation/total_loss"])
    return used_loss, len(dropless_calls)

  def test_replays_with_dropless_metrics_on_overflow(self):
    used_loss, dropless_call_count = self._run(step_fallback=True, has_overflow=True, with_dropless=True)
    self.assertEqual(used_loss, self._DROPLESS_LOSS)
    self.assertEqual(dropless_call_count, 1)

  def test_keeps_primary_metrics_without_overflow(self):
    used_loss, dropless_call_count = self._run(step_fallback=True, has_overflow=False, with_dropless=True)
    self.assertEqual(used_loss, self._PRIMARY_LOSS)
    self.assertEqual(dropless_call_count, 0)

  def test_keeps_primary_metrics_when_dropless_unavailable(self):
    used_loss, dropless_call_count = self._run(step_fallback=True, has_overflow=True, with_dropless=False)
    self.assertEqual(used_loss, self._PRIMARY_LOSS)
    self.assertEqual(dropless_call_count, 0)

  def test_keeps_primary_metrics_when_flag_off(self):
    used_loss, dropless_call_count = self._run(step_fallback=False, has_overflow=True, with_dropless=True)
    self.assertEqual(used_loss, self._PRIMARY_LOSS)
    self.assertEqual(dropless_call_count, 0)

  def test_first_steps_run_dropless_train_program(self):
    # step 0, start_step -1: step - start_step = 1 < retry_dropless_first_steps = 2.
    train_calls = []
    self._run(
        step_fallback=True,
        has_overflow=False,
        with_dropless=True,
        retry_dropless_first_steps=2,
        train_calls=train_calls,
    )
    self.assertEqual(train_calls, ["dropless"])

  def test_steps_after_first_phase_attempt_normal_train_program(self):
    # step - start_step = 1 is not < retry_dropless_first_steps = 1, so the normal program runs and the
    # overflow-free step is kept without a replay.
    train_calls = []
    self._run(
        step_fallback=True,
        has_overflow=False,
        with_dropless=True,
        retry_dropless_first_steps=1,
        train_calls=train_calls,
    )
    self.assertEqual(train_calls, ["normal"])

  def test_first_phase_program_inside_window(self):
    # first_phase_ragged_buffer_factor > 0: steps inside the window attempt the first-phase program; no overflow,
    # so its state is kept and neither the normal nor the dropless program runs.
    train_calls = []
    self._run(
        step_fallback=True,
        has_overflow=False,
        with_dropless=True,
        retry_dropless_first_steps=2,
        train_calls=train_calls,
        with_first_phase=True,
    )
    self.assertEqual(train_calls, ["first_phase"])

  def test_normal_program_after_first_phase_window(self):
    # step - start_step = 1 is not < retry_dropless_first_steps = 1: the normal program runs, not the first-phase one.
    train_calls = []
    self._run(
        step_fallback=True,
        has_overflow=False,
        with_dropless=True,
        retry_dropless_first_steps=1,
        train_calls=train_calls,
        with_first_phase=True,
    )
    self.assertEqual(train_calls, ["normal"])

  def test_first_phase_overflow_replays_with_dropless_program(self):
    # A first-phase step that still drops tokens is discarded and replayed with the dropless program.
    train_calls = []
    self._run(
        step_fallback=True,
        has_overflow=False,
        with_dropless=True,
        retry_dropless_first_steps=2,
        train_calls=train_calls,
        with_first_phase=True,
        first_phase_overflow=True,
    )
    self.assertEqual(train_calls, ["first_phase", "dropless"])

  def test_logs_required_rbf_line(self):
    train_calls = []
    with mock.patch.object(pre_train.max_logging, "log") as log:
      self._run(
          step_fallback=True,
          has_overflow=False,
          with_dropless=True,
          retry_dropless_first_steps=2,
          train_calls=train_calls,
          with_first_phase=True,
          required_rbf=[1.5, 3.25],
      )
    lines = [c.args[0] for c in log.call_args_list if c.args and str(c.args[0]).startswith("REQUIRED_RBF")]
    self.assertEqual(lines, ["REQUIRED_RBF step=0 max=3.2500 per_layer=[1.5000,3.2500] program=first_phase"])


def _state_checksum(state):
  """Host copies of every array leaf of an NNX state (PRNG keys as key data), in leaf order."""
  out = []
  for leaf in jax.tree_util.tree_leaves(state):
    if isinstance(leaf, jax.Array):
      if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
      out.append(np.array(leaf))
  return out


class TestWarmupProgramsInInit(unittest.TestCase):
  """warmup_programs_in_init: warmup_programs, make_synthetic_batch and start_run_clock."""

  def _mesh(self):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1), ("data",))

  def _shaped(self, mesh=None, batch=2, seq=4):
    # The tiny decoder fixture builds its own explicit mesh, so the train/eval tests use unsharded batches.
    sharding = None if mesh is None else jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return {k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sharding) for k, v in _make_data(batch, seq).items()}

  def _routed_bias_programs(self, cfg, donate):
    """Jitted train/eval programs over a tiny routed-bias model; returns (state, p_train, p_eval, mesh)."""
    model = _TinyDecoderMoEBias(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    graphdef, state = nnx.split(train_state_nnx.TrainStateNNX(model, optimizer))
    train_fn = functools.partial(pre_train.train_step, graphdef, cfg, None, None)
    eval_fn = functools.partial(pre_train.eval_step, graphdef, cfg)
    p_train = jax.jit(train_fn, donate_argnums=(0,) if donate else ())
    p_eval = jax.jit(eval_fn)
    return state, p_train, p_eval, model.mesh  # the loop's mesh context must be the model's mesh

  def _check_state_unchanged(self, donate):
    """Warms train (twice, as normal and dropless) and eval programs and checks every state leaf is bit-identical."""
    cfg = _Cfg()
    cfg.routed_bias = True
    cfg.routed_bias_update_rate = 0.001
    state, p_train, p_eval, mesh = self._routed_bias_programs(cfg, donate)
    batch = pre_train.make_synthetic_batch(cfg, self._shaped())
    before = _state_checksum(state)

    # The train program does change params, optimizer state and the routed bias when its output is kept.
    with jax.set_mesh(mesh):
      new_state, _ = p_train(jax.tree_util.tree_map(lambda x: x.copy(), state), batch)
    self.assertEqual(float(np.asarray(new_state.model.decoder.gate.bias.value).sum()), 6.0)
    changed = [not np.array_equal(a, b) for a, b in zip(before, _state_checksum(new_state))]
    self.assertGreaterEqual(sum(changed), 3)

    programs = [("train", "train", p_train), ("train_dropless", "train", p_train), ("eval", "eval", p_eval)]
    pre_train.warmup_programs(programs, state, mesh, (), (), batch, batch, copy_train_state=donate)

    self.assertFalse(any(leaf.is_deleted() for leaf in jax.tree_util.tree_leaves(state)))
    after = _state_checksum(state)
    self.assertEqual(len(before), len(after))
    for a, b in zip(before, after):
      np.testing.assert_array_equal(a, b)
    self.assertEqual(float(np.asarray(state.model.decoder.gate.bias.value).sum()), 0.0)

  def test_state_unchanged_after_warmup_non_donating(self):
    self._check_state_unchanged(donate=False)

  def test_state_unchanged_after_warmup_donating_copy(self):
    self._check_state_unchanged(donate=True)

  def test_runs_each_precompiled_program_exactly_once(self):
    calls = []

    def fake(name):
      def p_step(state, batch, *rng_args):
        del rng_args
        calls.append((name, batch))
        return state, {"loss": jnp.zeros(())}

      return p_step

    programs = [
        ("train", "train", fake("train")),
        ("train_first_phase", "train", None),  # not precompiled
        ("train_dropless", "train", fake("train_dropless")),
        ("eval", "eval", fake("eval")),
        ("eval_dropless", "eval", fake("eval_dropless")),
    ]
    state = {"w": jnp.ones((2,))}
    timings = pre_train.warmup_programs(programs, state, self._mesh(), (), (), "train_batch", "eval_batch")
    self.assertEqual(
        calls,
        [
            ("train", "train_batch"),
            ("train_dropless", "train_batch"),
            ("eval", "eval_batch"),
            ("eval_dropless", "eval_batch"),
        ],
    )
    self.assertEqual(sorted(timings), ["eval", "eval_dropless", "train", "train_dropless"])
    self.assertFalse(state["w"].is_deleted())

  def test_warmup_fills_the_jit_dispatch_cache(self):
    cfg = _Cfg()
    state, p_train, _, mesh = self._routed_bias_programs(cfg, donate=False)
    shaped = self._shaped(mesh)  # NamedSharding like the loader's input_data_shardings
    pre_train.warmup_programs(
        [("train", "train", p_train)], state, mesh, (), (), pre_train.make_synthetic_batch(cfg, shaped), None
    )
    self.assertEqual(p_train._cache_size(), 1)  # pylint: disable=protected-access
    # The loop's first step: a real batch with the same shapes and sharding hits the warmed entry.
    real = jax.device_put({k: np.zeros(v.shape, v.dtype) for k, v in shaped.items()}, shaped["inputs"].sharding)
    with jax.set_mesh(mesh), pre_train.logical_axis_rules(()):
      p_train(state, real)
    self.assertEqual(p_train._cache_size(), 1)  # pylint: disable=protected-access

  def test_synthetic_batch_matches_shapes_and_uses_prng_tokens(self):
    cfg = _Cfg()
    cfg.max_target_length = 4
    shaped = self._shaped(self._mesh(), batch=3, seq=4)
    batch = pre_train.make_synthetic_batch(cfg, shaped, seed=0)
    self.assertEqual(sorted(batch), sorted(shaped))
    for k, v in shaped.items():
      self.assertEqual((batch[k].shape, batch[k].dtype), (v.shape, v.dtype))
      self.assertEqual(batch[k].sharding, v.sharding)
    tokens = np.asarray(batch["inputs"])
    self.assertTrue(((tokens >= 0) & (tokens < cfg.vocab_size)).all())
    np.testing.assert_array_equal(np.asarray(batch["inputs_position"]), np.broadcast_to(np.arange(4), (3, 4)))
    np.testing.assert_array_equal(np.asarray(batch["inputs_segmentation"]), np.ones((3, 4)))

  def _clock_order(self, with_warmup):
    """Runs start_run_clock with mllog and the barrier mocked; returns (call order, python_vars)."""
    order = []
    names = ("init_print", "init_stop", "run_start", "block_start")
    patches = [
        mock.patch.object(pre_train.mllog_utils, n, side_effect=lambda *a, _n=n, **k: order.append(_n)) for n in names
    ]
    patches.append(
        mock.patch.object(
            pre_train.multihost_utils, "sync_global_devices", side_effect=lambda *_: order.append("barrier")
        )
    )
    python_vars = {"last_step_completion": None}
    for p in patches:
      p.start()
    try:
      pre_train.start_run_clock(
          _Cfg(),
          0,
          warmup_fn=(lambda: order.append("warmup")) if with_warmup else None,
          python_vars=python_vars,
      )
    finally:
      for p in patches:
        p.stop()
    return order, python_vars

  def test_run_start_after_warmup_and_barrier(self):
    order, python_vars = self._clock_order(with_warmup=True)
    self.assertEqual(order, ["init_print", "warmup", "barrier", "init_stop", "run_start", "block_start"])
    self.assertIsNotNone(python_vars["last_step_completion"])  # step 0's step time excludes the warmup

  def test_knob_off_keeps_original_mllog_sequence_without_barrier(self):
    order, python_vars = self._clock_order(with_warmup=False)
    self.assertEqual(order, ["init_print", "init_stop", "run_start", "block_start"])
    self.assertIsNone(python_vars["last_step_completion"])


class TestStepDiagnosticsNNX(unittest.TestCase):
  """log_step_diagnostics: the diagnostic scalars are produced, finite and correct, and appear on the step line."""

  _DIAG_KEYS = (
      "learning/raw_grad_norm",
      "learning/grad_norm",
      "learning/param_norm",
      "diag/max_abs_grad",
      "diag/moe_bias_checksum",
      "diag/moe_bias_update_nonzero",
      "diag/moe_overflow",
  )

  def _run(self, log_step_diagnostics, model_cls="bias"):
    """One train_step on the routed-bias + MTP stub ("bias") or the overflow stub; returns (cfg, metrics)."""
    cfg = _Cfg(gradient_clipping_threshold=1e-3, moe_dropless_fallback="step")
    cfg.log_step_diagnostics = log_step_diagnostics
    if model_cls == "bias":
      cfg.routed_bias, cfg.routed_bias_update_rate, cfg.mtp_num_layers = True, 0.001, 2
      model = _TinyDecoderMoEBiasWithMTP(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), num_mtp_layers=2)
    else:
      model = _TinyDecoderMoEOverflow(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), has_overflow=True)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    state_graphdef, state_pure = nnx.split(train_state_nnx.TrainStateNNX(model, optimizer))
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    _, metrics = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    return cfg, metrics

  def test_diagnostics_finite_and_correct(self):
    _, metrics = self._run(True)
    scal = {k: float(v) for k, v in metrics["scalar"].items()}
    for k in self._DIAG_KEYS:
      self.assertIn(k, scal)
      self.assertTrue(np.isfinite(scal[k]), k)
    # Biases start at 0 and get +1 (decoder 2x3), +2 and +3 (two MTP layers of 3): checksum 6 + 6 + 9, 12 nonzero.
    self.assertAlmostEqual(scal["diag/moe_bias_checksum"], 21.0, places=5)
    self.assertEqual(scal["diag/moe_bias_update_nonzero"], 12.0)
    self.assertEqual(scal["diag/moe_overflow"], 0.0)
    self.assertGreater(scal["diag/max_abs_grad"], 0.0)
    # Clipping at 1e-3 makes the post-clip norm the threshold, below the pre-clip norm.
    self.assertLess(scal["learning/grad_norm"], scal["learning/raw_grad_norm"])
    self.assertLessEqual(scal["diag/max_abs_grad"], scal["learning/raw_grad_norm"] + 1e-6)

  def test_overflow_value(self):
    _, metrics = self._run(True, model_cls="overflow")
    self.assertEqual(float(metrics["scalar"]["diag/moe_overflow"]), 1.0)
    self.assertEqual(float(metrics["scalar"]["diag/moe_bias_update_nonzero"]), 0.0)

  def test_off_adds_nothing(self):
    _, metrics = self._run(False)
    self.assertFalse([k for k in metrics["scalar"] if k.startswith("diag/")])

  def test_step_log_line(self):
    cfg, metrics = self._run(True)
    logger = metric_logger.MetricLogger.__new__(metric_logger.MetricLogger)  # skip __init__
    logger.config = pytypes.SimpleNamespace(
        rampup_end_step=0,
        hide_profiler_step_metric=False,
        elastic_enabled=False,
        num_experts=1,
        mtp_num_layers=cfg.mtp_num_layers,
        use_indexer=False,
        log_step_diagnostics=True,
    )
    scalars = {k: float(v) for k, v in metrics["scalar"].items()}
    perf = ("perf/step_time_seconds", "perf/per_device_tflops_per_sec", "perf/per_device_tokens_per_sec")
    scalars.update({k: 1.0 for k in perf})
    with mock.patch.object(metric_logger.max_logging, "log") as log:
      logger.log_metrics({"scalar": scalars}, step=3, metric_type="train")
    line = log.call_args[0][0]
    self.assertIn("completed step: 3", line)
    self.assertIn("diag: raw_grad_norm=", line)
    diag = line.split("diag: ")[1].split(",")[0].split()
    self.assertEqual([d.split("=")[0] for d in diag], [n for n, _ in metric_logger.STEP_DIAGNOSTICS_KEYS])
    for d in diag:
      self.assertTrue(np.isfinite(float(d.split("=")[1])), d)
    self.assertIn("moe_bias_checksum=2.100000000e+01", line)
    # Flag off: no diag part.
    logger.config.log_step_diagnostics = False
    with mock.patch.object(metric_logger.max_logging, "log") as log:
      logger.log_metrics({"scalar": scalars}, step=3, metric_type="train")
    self.assertNotIn("diag:", log.call_args[0][0])


if __name__ == "__main__":
  unittest.main()
