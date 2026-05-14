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

import unittest
from dataclasses import dataclass

import jax.numpy as jnp
import optax
from flax import nnx

from maxtext.layers import train_state_nnx
from maxtext.trainers.pre_train import train as pre_train


@dataclass
class _Cfg:
  """Subset of HyperParameters used by loss_fn / train_step / eval_step."""

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
  routed_bias: bool = False
  routed_bias_update_rate: float = 0.0
  mtp_num_layers: int = 0
  mtp_eval_target_module: int = 0
  use_dpo: bool = False
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
    # NNX intermediates are captured into a pure-dict snapshot, then logits attached.
    self.assertIsInstance(aux["intermediate_outputs"], dict)
    self.assertIn("logits", aux["intermediate_outputs"])

  def test_eval_mode_truncates_to_eval_micro_batch(self):
    cfg, ts = _build_state()
    cfg.micro_batch_size_to_eval_on = 1
    data = _make_data(batch=2, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=False)
    self.assertTrue(jnp.isfinite(loss))
    # eval truncated batch to 1 → total_weights = seq_len * 1
    self.assertEqual(int(aux["total_weights"]), data["targets_segmentation"].shape[1])

  def test_indexer_dense_warmup_skips_xent(self):
    cfg, ts = _build_state()
    cfg.use_indexer = True
    cfg.indexer_sparse_training = False
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    loss, aux = pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)
    # When dense warm-up is active the loss_fn skips the main loss entirely.
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertEqual(float(loss), 0.0)

  def test_vocab_tiling_raises_not_implemented(self):
    cfg, ts = _build_state()
    cfg.num_vocab_tiling = 4
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    with self.assertRaises(NotImplementedError):
      pre_train.loss_fn(ts.model, cfg, data, None, None, is_train=True)


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

  def test_train_step_dpo_raises_for_nnx(self):
    cfg, ts = _build_state()
    cfg.use_dpo = True
    state_graphdef, state_pure = nnx.split(ts)
    data = _make_data(batch=cfg.micro_batch_size_to_train_on, vocab=cfg.vocab_size)
    with self.assertRaises(NotImplementedError):
      pre_train.train_step(
          state_graphdef, cfg, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
      )

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
    # NNX path must NOT include DPO eval metric.
    self.assertNotIn("evaluation/dpo_reward_accuracy", metrics["scalar"])
    self.assertTrue(jnp.isfinite(metrics["scalar"]["evaluation/loss"]))


if __name__ == "__main__":
  unittest.main()
