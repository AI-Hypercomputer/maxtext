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
import types as pytypes
import unittest

from flax import nnx
from flax.nnx import variablelib
import jax
import jax.numpy as jnp
from maxtext.layers import nnx_scan
import numpy as np
from maxtext.common import train_state_nnx
from maxtext.common.metric_logger import record_activation_metrics
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train as pre_train
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
  retry_when_tokens_dropped: bool = False
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
    cfg.num_vocab_tiling = 2
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertEqual(float(loss), 0.0)

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
  """Covers train_step surfacing has_moe_overflow for training_loop_iteration's log line."""

  def _build_state(self, has_overflow, retry_when_tokens_dropped):
    cfg = _Cfg(retry_when_tokens_dropped=retry_when_tokens_dropped)
    model = _TinyDecoderMoEOverflow(cfg.vocab_size, hidden=4, rngs=nnx.Rngs(0), has_overflow=has_overflow)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    return cfg, train_state_nnx.TrainStateNNX(model, optimizer)

  def _metrics(self, has_overflow, retry_when_tokens_dropped):
    cfg, ts = self._build_state(has_overflow, retry_when_tokens_dropped)
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    _, metrics = pre_train.train_step(
        state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    return metrics

  def test_surfaces_overflow_when_flag_on(self):
    metrics = self._metrics(has_overflow=True, retry_when_tokens_dropped=True)
    self.assertTrue(bool(metrics["has_moe_overflow"]))

  def test_no_overflow_when_flag_on(self):
    metrics = self._metrics(has_overflow=False, retry_when_tokens_dropped=True)
    self.assertFalse(bool(metrics["has_moe_overflow"]))

  def test_key_absent_when_flag_off(self):
    """Key must be absent (not just False) when off: an always-present key would

    change every model's compiled train_step output, even non-MoE ones.
    """
    metrics = self._metrics(has_overflow=True, retry_when_tokens_dropped=False)
    self.assertNotIn("has_moe_overflow", metrics)


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


if __name__ == "__main__":
  unittest.main()
