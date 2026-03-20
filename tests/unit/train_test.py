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

"""Unit tests for pure-logic branches in train.py.

Covers functions that do not require a full model training stack:
  - get_first_step: Linen vs NNX step-counter dispatch
  - train_step: NNX + DPO error path
  - loss_fn: NNX + vocab-tiling error path (model call mocked)
  - eval_step: DPO reward-accuracy metric injection (loss mocked)
"""

import contextlib
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from maxtext.trainers.pre_train.train import eval_step, get_first_step, loss_fn, run, main, train_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(batch_size: int = 2, seq_len: int = 4) -> dict:
  """Returns a minimal data batch compatible with loss_fn's expected keys."""
  return {
      "inputs": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
      "inputs_position": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
      "inputs_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
      "targets": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
      "targets_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
  }


@dataclass
class _BaseMockConfig:
  """Minimal config mock that satisfies all loss_fn / eval_step field accesses."""

  micro_batch_size_to_train_on: int = 2
  micro_batch_size_to_eval_on: int = 2
  mtp_num_layers: int = 0
  mtp_eval_target_module: int = 0
  use_multimodal: bool = False
  enable_dropout: bool = False
  num_vocab_tiling: int = 1
  z_loss_multiplier: float = 0.0
  vocab_size: int = 10
  gradient_accumulation_steps: int = 1
  use_tunix_gradient_accumulation: bool = False
  num_experts: int = 1
  routed_bias: bool = False
  routed_bias_update_rate: float = 0.0
  use_dpo: bool = False
  shard_mode: str = "auto"
  shard_optimizer_over_data: bool = False
  gradient_clipping_threshold: float = 1.0
  optimizer_memory_host_offload: bool = False
  parameter_memory_host_offload: bool = False
  record_internal_nn_metrics: bool = False
  use_qk_clip: bool = False
  grad_dtype: str = "bfloat16"
  debug_sharding: bool = False
  use_sparse_indexer: bool = False


# ---------------------------------------------------------------------------
# get_first_step
# ---------------------------------------------------------------------------


class TestGetFirstStep(unittest.TestCase):
  """Tests for get_first_step() — lines 77-80."""

  def test_linen_model_reads_state_step(self):
    """Linen path: returns int(state.step)."""
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    state.step = jnp.array(42)
    result = get_first_step(model, state)
    self.assertEqual(result, 42)
    self.assertIsInstance(result, int)

  def test_nnx_model_reads_optimizer_step(self):
    """NNX path: returns int(state.optimizer.step.get_value())."""
    model = object()  # Not an nn.Module → triggers else branch
    state = MagicMock()
    state.optimizer.step.get_value.return_value = jnp.array(7)
    result = get_first_step(model, state)
    self.assertEqual(result, 7)
    self.assertIsInstance(result, int)

  def test_linen_step_zero_returns_int(self):
    """step=0 returns Python int 0, not jax.Array."""
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    state.step = jnp.array(0)
    result = get_first_step(model, state)
    self.assertIsInstance(result, int)
    self.assertEqual(result, 0)

  def test_nnx_step_zero_returns_int(self):
    """NNX step=0 returns Python int 0."""
    model = object()
    state = MagicMock()
    state.optimizer.step.get_value.return_value = jnp.array(0)
    result = get_first_step(model, state)
    self.assertIsInstance(result, int)
    self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# train_step — NNX + DPO error
# ---------------------------------------------------------------------------


class TestTrainStepNNXDPOError(unittest.TestCase):
  """train_step raises NotImplementedError for NNX model + use_dpo=True — lines 299-300."""

  @dataclass
  class _DPOConfig:
    use_dpo: bool = True
    gradient_accumulation_steps: int = 1

  def test_nnx_plus_dpo_raises_not_implemented(self):
    """Non-linen model with use_dpo=True must raise NotImplementedError immediately."""
    config = self._DPOConfig()
    model = object()  # Not nn.Module → NNX branch
    with self.assertRaises(NotImplementedError):
      train_step(model, config, None, None, None, None)

  def test_error_message_mentions_dpo(self):
    """Error message must reference DPO."""
    config = self._DPOConfig()
    model = object()
    with self.assertRaises(NotImplementedError) as ctx:
      train_step(model, config, None, None, None, None)
    self.assertIn("DPO", str(ctx.exception))


# ---------------------------------------------------------------------------
# loss_fn — NNX + num_vocab_tiling > 1 error
# ---------------------------------------------------------------------------


class TestLossFnNNXVocabTilingError(unittest.TestCase):
  """loss_fn raises NotImplementedError for NNX + num_vocab_tiling > 1 — lines 184-185."""

  @dataclass
  class _VocabTilingConfig(_BaseMockConfig):
    num_vocab_tiling: int = 2

  def test_nnx_vocab_tiling_raises_not_implemented(self):
    """NNX path with num_vocab_tiling > 1 raises NotImplementedError."""
    config = self._VocabTilingConfig()
    data = _make_batch()

    # Non-nn.Module so loss_fn takes the NNX else branch.
    model = MagicMock()
    # Return logits of shape (batch, seq, vocab_size).
    model.return_value = jnp.zeros((2, 4, config.vocab_size))

    # Patch nnx.state so nnx.state(model, nnx.Intermediate) succeeds without a real module.
    with patch("maxtext.trainers.pre_train.train.nnx.state") as mock_nnx_state:
      mock_nnx_state.return_value.to_pure_dict.return_value = {}
      with self.assertRaises(NotImplementedError):
        loss_fn(model, config, data, None, None, is_train=True)

  def test_nnx_vocab_tiling_error_message(self):
    """Error message must mention vocab tiling."""
    config = self._VocabTilingConfig()
    data = _make_batch()
    model = MagicMock()
    model.return_value = jnp.zeros((2, 4, config.vocab_size))
    with patch("maxtext.trainers.pre_train.train.nnx.state") as mock_nnx_state:
      mock_nnx_state.return_value.to_pure_dict.return_value = {}
      with self.assertRaises(NotImplementedError) as ctx:
        loss_fn(model, config, data, None, None, is_train=True)
    self.assertIn("Vocab tiling", str(ctx.exception))


# ---------------------------------------------------------------------------
# eval_step — DPO reward-accuracy metric (Linen path)
# ---------------------------------------------------------------------------


class TestEvalStepDPOMetric(unittest.TestCase):
  """eval_step injects dpo_reward_accuracy into metrics when use_dpo=True — lines 512-513."""

  @dataclass
  class _DPOConfig(_BaseMockConfig):
    use_dpo: bool = True

  def _fake_aux(self, reward_accuracy: float = 0.75) -> dict:
    return {
        "total_loss": jnp.array(1.0),
        "z_loss": jnp.array(0.0),
        "total_weights": jnp.array(8.0),
        "moe_lb_loss": jnp.array(0.0),
        "indexer_loss": jnp.array(0.0),
        "mtp_loss": jnp.array(0.0),
        "intermediate_outputs": {},
        "reward_accuracy": jnp.array(reward_accuracy),
    }

  def test_dpo_reward_accuracy_key_present(self):
    """Linen + DPO eval_step must include 'evaluation/dpo_reward_accuracy' in metrics."""
    config = self._DPOConfig()
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    data = _make_batch()
    dropout_rng = jax.random.PRNGKey(0)

    with (
        patch("maxtext.trainers.pre_train.train._split_dpo_state") as mock_split,
        patch("maxtext.trainers.pre_train.train.dpo_loss_fn") as mock_dpo_fn,
    ):
      mock_split.return_value = (MagicMock(), MagicMock())
      mock_dpo_fn.return_value = (jnp.array(1.0), self._fake_aux(0.75))

      metrics = eval_step(model, config, state, data, dropout_rng)

    self.assertIn("evaluation/dpo_reward_accuracy", metrics["scalar"])

  def test_dpo_reward_accuracy_value_propagated(self):
    """The reward_accuracy value from aux is forwarded into metrics unchanged."""
    config = self._DPOConfig()
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    data = _make_batch()
    dropout_rng = jax.random.PRNGKey(0)

    with (
        patch("maxtext.trainers.pre_train.train._split_dpo_state") as mock_split,
        patch("maxtext.trainers.pre_train.train.dpo_loss_fn") as mock_dpo_fn,
    ):
      mock_split.return_value = (MagicMock(), MagicMock())
      mock_dpo_fn.return_value = (jnp.array(1.0), self._fake_aux(0.9))

      metrics = eval_step(model, config, state, data, dropout_rng)

    actual = float(metrics["scalar"]["evaluation/dpo_reward_accuracy"])
    self.assertAlmostEqual(actual, 0.9, places=5)

  def test_no_dpo_key_when_dpo_disabled(self):
    """Without use_dpo, 'evaluation/dpo_reward_accuracy' must NOT appear in metrics."""
    config = _BaseMockConfig()  # use_dpo=False
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    data = _make_batch()
    dropout_rng = jax.random.PRNGKey(0)

    with patch("maxtext.trainers.pre_train.train.loss_fn") as mock_loss_fn:
      mock_loss_fn.return_value = (jnp.array(1.0), self._fake_aux())

      metrics = eval_step(model, config, state, data, dropout_rng)

    self.assertNotIn("evaluation/dpo_reward_accuracy", metrics["scalar"])


# ---------------------------------------------------------------------------
# loss_fn — NNX path continuation (lines 190–272)
# ---------------------------------------------------------------------------

_NNX_STATE_PATH = "maxtext.trainers.pre_train.train.nnx.state"


def _nnx_loss(config, data, intermediate_outputs=None, is_train=True):
  """Helper: run loss_fn with a mock NNX model."""
  if intermediate_outputs is None:
    intermediate_outputs = {}
  model = MagicMock()
  model.return_value = jnp.zeros((config.micro_batch_size_to_train_on, 4, config.vocab_size))
  with patch(_NNX_STATE_PATH) as mock_st:
    mock_st.return_value.to_pure_dict.return_value = intermediate_outputs
    return loss_fn(model, config, data, None, None, is_train=is_train)


class TestLossFnNNXContinuation(unittest.TestCase):
  """NNX loss_fn path past the vocab-tiling guard (lines 190–272)."""

  def test_basic_nnx_loss_returns_aux_keys(self):
    config = _BaseMockConfig()
    data = _make_batch()
    _, aux = _nnx_loss(config, data)
    for key in ("total_loss", "z_loss", "total_weights", "moe_lb_loss", "mtp_loss", "intermediate_outputs"):
      self.assertIn(key, aux)

  def test_nnx_loss_value_is_finite(self):
    config = _BaseMockConfig()
    data = _make_batch()
    loss, _ = _nnx_loss(config, data)
    self.assertTrue(jnp.isfinite(loss))

  def test_is_train_false_slices_eval_batch(self):
    """is_train=False uses micro_batch_size_to_eval_on — covers lines 111–112."""

    @dataclass
    class EvalConfig(_BaseMockConfig):
      micro_batch_size_to_eval_on: int = 1

    config = EvalConfig()
    data = _make_batch(batch_size=2)
    loss, _ = _nnx_loss(config, data, is_train=False)
    self.assertTrue(jnp.isfinite(loss))

  def test_mtp_loss_added_when_mtp_layers_set(self):
    """mtp_num_layers > 0 and is_train → mtp_losses appended, loss includes mtp (line 117, 226–228)."""

    @dataclass
    class MTPConfig(_BaseMockConfig):
      mtp_num_layers: int = 2

    config = MTPConfig()
    data = _make_batch()
    with patch("maxtext.trainers.pre_train.train.calculate_mtp_loss", return_value=jnp.array(0.1)):
      _, aux = _nnx_loss(config, data, is_train=True)
    self.assertAlmostEqual(float(aux["mtp_loss"]), 0.1, places=5)

  def test_mtp_acceptance_collections_eval(self):
    """mtp_eval_target_module > 0 and not is_train → line 122 executed."""

    @dataclass
    class MTPEvalConfig(_BaseMockConfig):
      mtp_eval_target_module: int = 1

    config = MTPEvalConfig()
    data = _make_batch()
    loss, _ = _nnx_loss(config, data, is_train=False)
    self.assertTrue(jnp.isfinite(loss))

  def test_gradient_accumulation_skips_normalization(self):
    """gradient_accumulation_steps > 1 → loss = total_loss, not divided by weights (line 213)."""

    @dataclass
    class GAConfig(_BaseMockConfig):
      gradient_accumulation_steps: int = 4
      use_tunix_gradient_accumulation: bool = False

    config = GAConfig()
    data = _make_batch()
    loss, aux = _nnx_loss(config, data)
    np.testing.assert_allclose(float(loss), float(aux["total_loss"]), rtol=1e-5)

  def test_tunix_gradient_accumulation_normalizes_loss(self):
    """use_tunix_gradient_accumulation=True with ga_steps>1 still normalizes (else line 219)."""

    @dataclass
    class TunixConfig(_BaseMockConfig):
      gradient_accumulation_steps: int = 4
      use_tunix_gradient_accumulation: bool = True

    config = TunixConfig()
    data = _make_batch()
    loss, _ = _nnx_loss(config, data)
    self.assertTrue(jnp.isfinite(loss))

  def test_num_experts_gt1_no_moe_loss_found(self):
    """num_experts > 1, no matching key → found_loss=False, debug log path (lines 247–248)."""

    @dataclass
    class MoEConfig(_BaseMockConfig):
      num_experts: int = 2

    config = MoEConfig()
    data = _make_batch()
    with patch("maxtext.trainers.pre_train.train.maxtext_utils.get_nested_value", return_value=0.0):
      _, aux = _nnx_loss(config, data)
    self.assertEqual(float(aux["moe_lb_loss"]), 0.0)

  def test_num_experts_gt1_moe_loss_found(self):
    """num_experts > 1, matching key found → found_loss=True, loss increases (lines 243–245)."""

    @dataclass
    class MoEConfig(_BaseMockConfig):
      num_experts: int = 2

    config = MoEConfig()
    data = _make_batch()
    with patch("maxtext.trainers.pre_train.train.maxtext_utils.get_nested_value", return_value=jnp.array(0.5)):
      _, aux = _nnx_loss(config, data)
    self.assertGreater(float(aux["moe_lb_loss"]), 0.0)

  def test_routed_bias_extracts_moe_bias_updates(self):
    """routed_bias=True and update_rate > 0 → moe_bias_updates set (lines 255–257)."""

    @dataclass
    class RoutedBiasConfig(_BaseMockConfig):
      routed_bias: bool = True
      routed_bias_update_rate: float = 0.1

    config = RoutedBiasConfig()
    data = _make_batch()
    bias_val = jnp.zeros((4,))
    with patch("maxtext.trainers.pre_train.train.maxtext_utils.get_nested_value", return_value=bias_val):
      _, aux = _nnx_loss(config, data)
    self.assertIsNotNone(aux["moe_bias_updates"])


# ---------------------------------------------------------------------------
# loss_fn — Linen path (lines 126–171)
# ---------------------------------------------------------------------------


class TestLossFnLinenPath(unittest.TestCase):
  """loss_fn with a mocked Linen nn.Module (lines 126–171)."""

  def _linen_loss(self, config, data, is_train=True):
    logits = jnp.zeros((config.micro_batch_size_to_train_on, 4, config.vocab_size))
    model = MagicMock(spec=nn.Module)
    model.apply.return_value = (logits, {})
    model.mesh = MagicMock()
    dropout_rng = jax.random.PRNGKey(0)
    with patch("maxtext.trainers.pre_train.train.sharding.maybe_shard_with_logical", side_effect=lambda x, *a, **kw: x):
      return loss_fn(model, config, data, dropout_rng, {"params": {}}, is_train=is_train)

  def test_linen_loss_is_finite(self):
    loss, _ = self._linen_loss(_BaseMockConfig(), _make_batch())
    self.assertTrue(jnp.isfinite(loss))

  def test_linen_loss_aux_keys(self):
    _, aux = self._linen_loss(_BaseMockConfig(), _make_batch())
    for key in ("total_loss", "z_loss", "total_weights"):
      self.assertIn(key, aux)

  def test_linen_eval_mode(self):
    @dataclass
    class EvalConfig(_BaseMockConfig):
      micro_batch_size_to_eval_on: int = 1

    loss, _ = self._linen_loss(EvalConfig(), _make_batch(batch_size=2), is_train=False)
    self.assertTrue(jnp.isfinite(loss))

  def test_linen_model_apply_called(self):
    config = _BaseMockConfig()
    data = _make_batch()
    logits = jnp.zeros((2, 4, config.vocab_size))
    model = MagicMock(spec=nn.Module)
    model.apply.return_value = (logits, {})
    model.mesh = MagicMock()
    with patch("maxtext.trainers.pre_train.train.sharding.maybe_shard_with_logical", side_effect=lambda x, *a, **kw: x):
      loss_fn(model, config, data, jax.random.PRNGKey(0), {"params": {}})
    model.apply.assert_called_once()

  def test_linen_num_experts_gt1_no_loss(self):
    @dataclass
    class MoEConfig(_BaseMockConfig):
      num_experts: int = 2

    with patch("maxtext.trainers.pre_train.train.maxtext_utils.get_nested_value", return_value=0.0):
      _, aux = self._linen_loss(MoEConfig(), _make_batch())
    self.assertEqual(float(aux["moe_lb_loss"]), 0.0)

  def test_linen_mtp_loss(self):
    @dataclass
    class MTPConfig(_BaseMockConfig):
      mtp_num_layers: int = 2

    with patch("maxtext.trainers.pre_train.train.calculate_mtp_loss", return_value=jnp.array(0.05)):
      _, aux = self._linen_loss(MTPConfig(), _make_batch(), is_train=True)
    self.assertAlmostEqual(float(aux["mtp_loss"]), 0.05, places=5)

  def test_linen_vocab_tiling_path(self):
    """num_vocab_tiling > 1 → vocab_tiling_linen_loss called (lines 144–146)."""

    @dataclass
    class VTConfig(_BaseMockConfig):
      num_vocab_tiling: int = 2

    config = VTConfig()
    data = _make_batch()
    logits = jnp.zeros((2, 4, config.vocab_size))
    hidden_states = jnp.zeros((2, 4, 8))
    model = MagicMock(spec=nn.Module)
    model.apply.return_value = (logits, {})
    model.mesh = MagicMock()
    dropout_rng = jax.random.PRNGKey(0)
    with (
        patch("maxtext.trainers.pre_train.train.maxtext_utils.get_nested_value", return_value=(hidden_states,)),
        patch(
            "maxtext.trainers.pre_train.train.vocab_tiling_linen_loss",
            return_value=(jnp.array(1.0), jnp.array(0.0)),
        ) as mock_vt,
    ):
      loss, _ = loss_fn(model, config, data, dropout_rng, {"params": {}})
    mock_vt.assert_called_once()
    self.assertTrue(jnp.isfinite(loss))


# ---------------------------------------------------------------------------
# eval_step — NNX path + MTP acceptance rate (lines 493–498)
# ---------------------------------------------------------------------------


class TestEvalStepNNXPath(unittest.TestCase):
  """eval_step with NNX model (lines 493–494)."""

  def _fake_aux(self):
    return {
        "total_loss": jnp.array(1.0),
        "z_loss": jnp.array(0.0),
        "total_weights": jnp.array(8.0),
        "moe_lb_loss": jnp.array(0.0),
        "indexer_loss": jnp.array(0.0),
        "mtp_loss": jnp.array(0.0),
        "intermediate_outputs": {},
    }

  def test_nnx_eval_step_returns_metrics(self):
    config = _BaseMockConfig()
    model = object()  # non-nn.Module → NNX branch
    state = MagicMock()
    mock_merged = MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.nnx.merge", return_value=mock_merged),
        patch("maxtext.trainers.pre_train.train.loss_fn", return_value=(jnp.array(1.0), self._fake_aux())),
    ):
      metrics = eval_step(model, config, state, _make_batch())
    self.assertIn("scalar", metrics)
    self.assertIn("evaluation/loss", metrics["scalar"])

  def test_nnx_eval_step_calls_merge(self):
    config = _BaseMockConfig()
    model = object()
    state = MagicMock()
    mock_merged = MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.nnx.merge", return_value=mock_merged) as mock_merge,
        patch("maxtext.trainers.pre_train.train.loss_fn", return_value=(jnp.array(1.0), self._fake_aux())),
    ):
      eval_step(model, config, state, _make_batch())
    mock_merge.assert_called_once_with(model, state)

  def test_nnx_eval_step_calls_loss_fn_with_eval_mode(self):
    config = _BaseMockConfig()
    model = object()
    state = MagicMock()
    mock_merged = MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.nnx.merge", return_value=mock_merged),
        patch("maxtext.trainers.pre_train.train.loss_fn", return_value=(jnp.array(1.0), self._fake_aux())) as mock_lf,
    ):
      eval_step(model, config, state, _make_batch())
    # Must be called with is_train=False
    _, kwargs = mock_lf.call_args
    self.assertFalse(kwargs.get("is_train", True))

  def test_mtp_acceptance_rate_computed_when_enabled(self):
    """mtp_eval_target_module > 0 → calculate_mtp_acceptance_rate called (line 498)."""

    @dataclass
    class MTPConfig(_BaseMockConfig):
      mtp_eval_target_module: int = 1

    config = MTPConfig()
    model = MagicMock(spec=nn.Module)
    state = MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.loss_fn", return_value=(jnp.array(1.0), self._fake_aux())),
        patch("maxtext.trainers.pre_train.train.calculate_mtp_acceptance_rate", return_value=0.75) as mock_mtp,
    ):
      metrics = eval_step(model, config, state, _make_batch())
    mock_mtp.assert_called_once()
    self.assertAlmostEqual(float(metrics["scalar"]["evaluation/mtp_acceptance_rate_percent"]), 0.75, places=5)


# ---------------------------------------------------------------------------
# run() — contextlib dispatch (lines 719–732)
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
  """Tests for run() — train_loop is invoked under correct context managers."""

  @patch("maxtext.trainers.pre_train.train.train_loop")
  @patch("maxtext.trainers.pre_train.train.max_utils.maybe_get_transformer_engine_context")
  def test_run_calls_train_loop(self, mock_ctx, mock_loop):
    mock_ctx.return_value = contextlib.nullcontext()
    run(MagicMock(), MagicMock(), MagicMock())
    mock_loop.assert_called_once()

  @patch("maxtext.trainers.pre_train.train.train_loop")
  @patch("maxtext.trainers.pre_train.train.max_utils.maybe_get_transformer_engine_context")
  @patch("maxtext.trainers.pre_train.train.is_decoupled", return_value=True)
  def test_run_logs_when_decoupled(self, _mock_decoupled, mock_ctx, mock_loop):
    mock_ctx.return_value = contextlib.nullcontext()
    with patch("maxtext.trainers.pre_train.train.max_logging") as mock_log:
      run(MagicMock(), MagicMock(), MagicMock())
    mock_log.log.assert_called()
    mock_loop.assert_called_once()

  @patch("maxtext.trainers.pre_train.train.train_loop")
  @patch("maxtext.trainers.pre_train.train.max_utils.maybe_get_transformer_engine_context")
  def test_run_passes_config_and_recorder_to_train_loop(self, mock_ctx, mock_loop):
    mock_ctx.return_value = contextlib.nullcontext()
    config = MagicMock()
    recorder = MagicMock()
    run(config, recorder, MagicMock())
    mock_loop.assert_called_once_with(config, recorder)


# ---------------------------------------------------------------------------
# main() (lines 736–739)
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):
  """Tests for main() — wires initialize → record_goodput → run."""

  def _run_main(self):
    config, recorder, diag = MagicMock(), MagicMock(), MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.initialize", return_value=(config, recorder, diag)) as mock_init,
        patch("maxtext.trainers.pre_train.train.record_goodput"),
        patch("maxtext.trainers.pre_train.train.maybe_monitor_goodput", return_value=contextlib.nullcontext()),
        patch("maxtext.trainers.pre_train.train.run") as mock_run,
    ):
      main(["dummy_config"])
    return mock_init, mock_run, config, recorder, diag

  def test_main_calls_initialize(self):
    mock_init, _, _, _, _ = self._run_main()
    mock_init.assert_called_once_with(["dummy_config"])

  def test_main_calls_run_with_correct_args(self):
    _, mock_run, config, recorder, diag = self._run_main()
    mock_run.assert_called_once_with(config, recorder, diag)

  def test_main_records_goodput(self):
    config, recorder, diag = MagicMock(), MagicMock(), MagicMock()
    with (
        patch("maxtext.trainers.pre_train.train.initialize", return_value=(config, recorder, diag)),
        patch("maxtext.trainers.pre_train.train.record_goodput") as mock_record,
        patch("maxtext.trainers.pre_train.train.maybe_monitor_goodput", return_value=contextlib.nullcontext()),
        patch("maxtext.trainers.pre_train.train.run"),
    ):
      main(["dummy_config"])
    mock_record.assert_called_once()


if __name__ == "__main__":
  unittest.main()
