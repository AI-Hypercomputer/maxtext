# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests MaxText preparation of fresh diffusion OPD batches."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.trainers.post_train.distillation import diffusion_opd
from maxtext.trainers.post_train.distillation import distillation_utils


def _student_config(**overrides):
  """Builds the student configuration subset used by OPD validation."""
  values = {
      "distill_data_source": "student_rollout",
      "training_objective": "block_diffusion",
      "attention_type": "block_diffusion",
      "use_sft": True,
      "sft_train_on_completion_only": True,
      "vocab_size": 16,
      "tokenizer_path": "test-tokenizer",
      "tokenizer_type": "huggingface",
      "distill_beta": 0.0,
      "distill_beta_end": None,
      "distill_alpha": 0.75,
      "distill_alpha_end": None,
      "distill_alpha_schedule": "constant",
      "distill_temperature": 2.0,
      "distill_temperature_end": None,
      "distill_temperature_schedule": "constant",
      "eval_interval": -1,
      "steps": 10,
      "gradient_accumulation_steps": 1,
      "use_tunix_gradient_accumulation": False,
      "max_target_length": 16,
      "global_batch_size_to_load": 8,
      "global_batch_size_to_train_on": 8,
      "dataset_type": "hf",
      "generate_padding_batch_train": False,
      "elastic_enabled": False,
      "expansion_factor_real_data": -1.0,
      "enable_rampup_batch_size": False,
      "learn_to_init_mode": False,
      "student_params_to_update": None,
      "distill_rollout_max_denoise_steps": -1,
      "block_diffusion_block_size": 4,
      "block_diffusion_mask_id": 15,
      "block_diffusion_min_noise": 0.001,
      "block_diffusion_logit_alignment": "same_position",
      "block_diffusion_canvas_policy": "all_masked",
      "distill_rollout_confidence_threshold": 0.9,
      "distill_rollout_temperature": 1.0,
      "distill_rollout_algorithm": "low_confidence",
      "distill_rollout_stop_token_ids": [],
      "enable_dropout": True,
      "dropout_rate": 0.0,
      "opt_type": "adamw",
      "learning_rate": 1e-4,
      "warmup_steps_fraction": 0.1,
      "learning_rate_final_fraction": 0.1,
      "gradient_clipping_threshold": 1.0,
      "hardware": "cpu",
      "shard_mode": "auto",
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _teacher_config(**overrides):
  """Builds a causal teacher configuration subset."""
  values = {
      "training_objective": "causal_lm",
      "attention_type": "global",
      "vocab_size": 16,
      "tokenizer_path": "test-tokenizer",
      "tokenizer_type": "huggingface",
      "hardware": "cpu",
      "shard_mode": "auto",
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _training_input(**overrides):
  """Builds one explicit-mask target-aligned distillation batch."""
  values = {
      "input_tokens": jnp.asarray([[3, 4, 31, 31]], dtype=jnp.int32),
      "input_mask": jnp.ones((1, 4), dtype=jnp.bool_),
      "positions": jnp.arange(4, dtype=jnp.int32)[None, :],
      "decoder_segment_ids": jnp.ones((1, 4), dtype=jnp.int32),
      "targets": jnp.asarray([[3, 4, 5, 6]], dtype=jnp.int32),
      "targets_position": jnp.arange(4, dtype=jnp.int32)[None, :],
      "targets_segmentation": jnp.ones((1, 4), dtype=jnp.int32),
      "completion_mask": jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
      "corruption_mask": jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
      "targets_loss_mask": jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
  }
  values.update(overrides)
  return distillation_utils.MaxTextTrainingInput(**values)


class MaxTextDiffusionOPDTest(absltest.TestCase):

  def test_config_validation_is_default_off(self):
    student = _student_config(distill_data_source="dataset")

    self.assertFalse(diffusion_opd.validate_diffusion_opd_configs(student, _teacher_config(), is_offline=False))

  def test_config_validation_accepts_supported_contract(self):
    self.assertTrue(diffusion_opd.validate_diffusion_opd_configs(_student_config(), _teacher_config(), is_offline=False))

  def test_config_validation_rejects_unsafe_modes(self):
    cases = [
        (_student_config(), _teacher_config(), True, "online teacher"),
        (_student_config(training_objective="causal_lm"), _teacher_config(), False, "student training_objective"),
        (_student_config(use_sft=False), _teacher_config(), False, "completion-only SFT"),
        (_student_config(), _teacher_config(training_objective="block_diffusion"), False, "causal teacher"),
        (_student_config(), _teacher_config(attention_type="full"), False, "causal teacher"),
        (_student_config(), _teacher_config(mtp_num_layers=1), False, "MTP teacher"),
        (_student_config(), _teacher_config(tokenizer_path="other-tokenizer"), False, "identical.*tokenizer"),
        (
            _student_config(tokenizer_type="sentencepiece"),
            _teacher_config(tokenizer_type="sentencepiece"),
            False,
            "tokenizer_type='huggingface'",
        ),
        (_student_config(distill_beta=0.1), _teacher_config(), False, "feature distillation"),
        (_student_config(distill_alpha=float("nan")), _teacher_config(), False, "finite"),
        (_student_config(distill_rollout_temperature=float("inf")), _teacher_config(), False, "finite"),
        (_student_config(eval_interval=10), _teacher_config(), False, "evaluation"),
        (
            _student_config(generate_padding_batch_train=True),
            _teacher_config(),
            False,
            "generate_padding_batch_train",
        ),
        (_student_config(elastic_enabled=True), _teacher_config(), False, "elastic data loading"),
        (_student_config(expansion_factor_real_data=2.0), _teacher_config(), False, "expansion_factor_real_data"),
        (_student_config(enable_rampup_batch_size=True), _teacher_config(), False, "ramp-up batch sizing"),
        (
            _student_config(gradient_accumulation_steps=2),
            _teacher_config(),
            False,
            "use_tunix_gradient_accumulation=True",
        ),
        (_student_config(distill_rollout_max_denoise_steps=3), _teacher_config(), False, "at least"),
        (
            _student_config(distill_rollout_stop_token_ids=[7, 7]),
            _teacher_config(),
            False,
            "unique IDs",
        ),
    ]
    for student, teacher, is_offline, message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        diffusion_opd.validate_diffusion_opd_configs(student, teacher, is_offline=is_offline)

  def test_tpu_auto_sharding_is_preserved_for_models_without_explicit_support(self):
    self.assertTrue(
        diffusion_opd.validate_diffusion_opd_configs(
            _student_config(hardware="tpu", shard_mode="auto"),
            _teacher_config(hardware="tpu", shard_mode="auto"),
            is_offline=False,
        )
    )

  def test_causal_teacher_logits_are_shifted_to_physical_targets(self):
    raw_logits = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    model = mock.Mock(return_value=raw_logits)
    logits_fn = diffusion_opd.create_target_aligned_logits_fn(_teacher_config(), enable_dropout=False)
    model_inputs = {
        "input_tokens": jnp.ones((1, 4), dtype=jnp.int32),
        "positions": jnp.arange(4, dtype=jnp.int32)[None, :],
        "decoder_segment_ids": jnp.ones((1, 4), dtype=jnp.int32),
        "targets": jnp.ones((1, 4), dtype=jnp.int32),
        "targets_segmentation": jnp.ones((1, 4), dtype=jnp.int32),
    }

    aligned = logits_fn(model, model_inputs)

    np.testing.assert_array_equal(aligned, raw_logits[:, [0, 0, 1, 2], :])
    self.assertFalse(model.call_args.kwargs["enable_dropout"])

  def test_compiled_rollout_functionalizes_mutating_nnx_model(self):
    class MutatingLogitModel(nnx.Module):

      def __init__(self):
        self.counter = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))

      def __call__(self, decoder_input_tokens, **kwargs):
        del kwargs
        self.counter[...] += 1
        targets = jnp.full_like(decoder_input_tokens, 7)
        return jax.nn.one_hot(targets, 16, dtype=jnp.float32) * 12.0

    model = MutatingLogitModel()
    rollout_fn = diffusion_opd.create_rollout_fn(_student_config())

    generated = rollout_fn(
        model,
        jnp.asarray([[3, 4, 1, 1]], dtype=jnp.int32),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray([[0, 0, 1, 1]], dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(generated, [[3, 4, 7, 7]])
    self.assertEqual(int(model.counter[...]), 0)

  def test_preparation_scores_clean_rollout_and_corrupts_only_student(self):
    generated = jnp.asarray([[3, 4, 7, 8]], dtype=jnp.int32)
    rollout_fn = mock.Mock(return_value=generated)
    teacher_model = mock.sentinel.teacher_model

    def teacher_score_fn(model, model_inputs):
      self.assertIs(model, teacher_model)
      np.testing.assert_array_equal(model_inputs["input_tokens"], generated)
      return jax.nn.one_hot(generated, 16, dtype=jnp.float32)

    batch = diffusion_opd.prepare_diffusion_opd_batch(
        _training_input(),
        mock.sentinel.student_model,
        teacher_model,
        rollout_fn=rollout_fn,
        teacher_score_fn=teacher_score_fn,
        mask_id=15,
    )

    np.testing.assert_array_equal(batch.student_batch.target_ids, generated)
    np.testing.assert_array_equal(batch.student_batch.model_inputs["input_tokens"], [[3, 4, 15, 15]])
    np.testing.assert_array_equal(batch.student_batch.loss_weights, [[0.0, 0.0, 1.0, 1.0]])
    self.assertEqual(batch.teacher_logits.shape, (1, 4, 16))
    rollout_fn.assert_called_once()

  def test_preparation_excludes_tokens_after_generated_eos(self):
    generated = jnp.asarray([[3, 4, 7, 8]], dtype=jnp.int32)
    teacher_score_fn = mock.Mock(return_value=jax.nn.one_hot(generated, 16))

    batch = diffusion_opd.prepare_diffusion_opd_batch(
        _training_input(),
        mock.sentinel.student_model,
        mock.sentinel.teacher_model,
        rollout_fn=mock.Mock(return_value=generated),
        teacher_score_fn=teacher_score_fn,
        mask_id=15,
        stop_token_ids=(7,),
    )

    np.testing.assert_array_equal(batch.student_batch.loss_weights, [[0.0, 0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(batch.student_batch.model_inputs["decoder_segment_ids"], [[1, 1, 1, 0]])
    np.testing.assert_array_equal(teacher_score_fn.call_args.args[1]["targets_segmentation"], [[1, 1, 1, 0]])

  def test_shifted_block_anchor_can_be_weighted_without_corruption(self):
    generated = jnp.asarray([[3, 7, 8, 9]], dtype=jnp.int32)
    batch = diffusion_opd.prepare_diffusion_opd_batch(
        _training_input(
            completion_mask=jnp.asarray([[0, 1, 1, 1]], dtype=jnp.int32),
            corruption_mask=jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32),
            targets_loss_mask=jnp.asarray([[0, 1, 1, 1]], dtype=jnp.int32),
        ),
        mock.sentinel.student,
        mock.sentinel.teacher,
        rollout_fn=mock.Mock(return_value=generated),
        teacher_score_fn=mock.Mock(return_value=jax.nn.one_hot(generated, 16)),
        mask_id=15,
        logit_alignment="shifted",
        block_size=2,
    )

    np.testing.assert_array_equal(batch.student_batch.model_inputs["input_tokens"], [[3, 15, 8, 15]])
    np.testing.assert_array_equal(batch.student_batch.loss_weights, [[0.0, 1.0, 1.0, 1.0]])

  def test_prepared_batch_runs_weighted_tunix_loss_and_student_gradient(self):
    class TrainableLogitModel(nnx.Module):

      def __init__(self):
        self.bias = nnx.Param(jnp.zeros((16,), dtype=jnp.float32))

      def __call__(self, decoder_input_tokens, **kwargs):
        del kwargs
        return jnp.broadcast_to(self.bias[...], (*decoder_input_tokens.shape, 16))

    generated = jnp.asarray([[3, 4, 7, 8]], dtype=jnp.int32)
    batch = diffusion_opd.prepare_diffusion_opd_batch(
        _training_input(),
        mock.sentinel.student,
        mock.sentinel.teacher,
        rollout_fn=mock.Mock(return_value=generated),
        teacher_score_fn=mock.Mock(return_value=jax.nn.one_hot(generated, 16) * 4.0),
        mask_id=15,
    )
    model = TrainableLogitModel()
    logits_fn = diffusion_opd.create_target_aligned_logits_fn(_student_config(), enable_dropout=False)

    def loss_fn(trainable_model):
      output = diffusion_opd.tunix_diffusion_opd.diffusion_opd_loss_fn(
          trainable_model,
          batch,
          logits_fn,
          temperature=2.0,
          soft_loss_weight=0.75,
          hard_loss_weight=0.25,
      )
      return output.primary_loss.unreduced_sum, output

    (loss_sum, output), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    self.assertTrue(bool(jnp.isfinite(loss_sum)))
    self.assertEqual(float(output.primary_loss.denominator), 2.0)
    self.assertTrue(any(bool(jnp.any(leaf[...] != 0)) for leaf in jax.tree.leaves(gradients)))

  def test_same_position_rejects_weighted_clean_targets(self):
    with self.assertRaisesRegex(ValueError, "same-position targets must be corrupted"):
      diffusion_opd.prepare_diffusion_opd_batch(
          _training_input(
              corruption_mask=jnp.asarray([[0, 0, 1, 0]], dtype=jnp.int32),
              targets_loss_mask=jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
          ),
          mock.sentinel.student,
          mock.sentinel.teacher,
          rollout_fn=mock.Mock(),
          teacher_score_fn=mock.Mock(),
          mask_id=15,
          logit_alignment="same_position",
          block_size=4,
      )

  def test_device_safe_weights_zero_invalid_suffix_rows(self):
    weights = diffusion_opd._safe_loss_weights(  # pylint: disable=protected-access
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.ones((1, 4), dtype=jnp.bool_),
        jnp.asarray([[0, 1, 0, 1]], dtype=jnp.bool_),
        jnp.asarray([[0, 1, 0, 1]], dtype=jnp.bool_),
        jnp.asarray([[0.0, 1.0, 0.0, 1.0]], dtype=jnp.float32),
        alignment="same_position",
        block_size=4,
    )

    np.testing.assert_array_equal(weights, jnp.zeros((1, 4), dtype=jnp.float32))

  def test_device_safe_weights_zero_out_of_range_positions(self):
    for positions in (jnp.asarray([[-1, 1]]), jnp.asarray([[0, 2]])):
      with self.subTest(positions=positions):
        weights = diffusion_opd._safe_loss_weights(  # pylint: disable=protected-access
            positions,
            jnp.ones((1, 2), dtype=jnp.bool_),
            jnp.asarray([[0, 1]], dtype=jnp.bool_),
            jnp.asarray([[0, 1]], dtype=jnp.bool_),
            jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
            alignment="same_position",
            block_size=2,
        )

        np.testing.assert_array_equal(weights, jnp.zeros((1, 2), dtype=jnp.float32))

  def test_device_safe_weights_zero_mask_subset_violations(self):
    weights = diffusion_opd._safe_loss_weights(  # pylint: disable=protected-access
        jnp.asarray([[0, 1, 2]]),
        jnp.asarray([[1, 1, 0]], dtype=jnp.bool_),
        jnp.asarray([[0, 1, 1]], dtype=jnp.bool_),
        jnp.asarray([[0, 1, 1]], dtype=jnp.bool_),
        jnp.asarray([[0.0, 1.0, 1.0]], dtype=jnp.float32),
        alignment="same_position",
        block_size=2,
    )

    np.testing.assert_array_equal(weights, jnp.zeros((1, 3), dtype=jnp.float32))

  def test_preparation_rejects_missing_or_cross_role_masks(self):
    rollout_fn = mock.Mock()
    teacher_score_fn = mock.Mock()
    with self.assertRaisesRegex(ValueError, "corruption_mask"):
      diffusion_opd.prepare_diffusion_opd_batch(
          _training_input(corruption_mask=None),
          mock.sentinel.student,
          mock.sentinel.teacher,
          rollout_fn=rollout_fn,
          teacher_score_fn=teacher_score_fn,
          mask_id=15,
      )
    with self.assertRaisesRegex(ValueError, "subset of completion_mask"):
      diffusion_opd.prepare_diffusion_opd_batch(
          _training_input(corruption_mask=jnp.asarray([[0, 1, 1, 0]], dtype=jnp.int32)),
          mock.sentinel.student,
          mock.sentinel.teacher,
          rollout_fn=rollout_fn,
          teacher_score_fn=teacher_score_fn,
          mask_id=15,
      )
    rollout_fn.assert_not_called()
    teacher_score_fn.assert_not_called()

  def test_hf_resume_replays_and_bounds_deterministic_stream(self):
    bounded = diffusion_opd.replay_and_bound_iterator(
        iter(range(20)),
        iter_steps=4,
        train_steps=2,
        max_steps=5,
        accumulation_steps=2,
    )

    self.assertEqual(list(bounded), [4, 5, 6, 7, 8, 9])

  def test_hf_resume_can_replay_before_global_materialization(self):
    local_iterator = iter(range(20))

    class GlobalIterator:

      def __next__(self):
        return next(local_iterator) * 10

      def __iter__(self):
        return self

    bounded = diffusion_opd.replay_and_bound_iterator(
        GlobalIterator(),
        iter_steps=4,
        train_steps=2,
        max_steps=5,
        accumulation_steps=2,
        replay_iterator=local_iterator,
    )

    self.assertEqual(list(bounded), [40, 50, 60, 70, 80, 90])

  def test_hf_resume_rejects_early_exhaustion_during_replay(self):
    with self.assertRaisesRegex(ValueError, "replay exhausted after 2 of 4"):
      diffusion_opd.replay_and_bound_iterator(
          iter(range(2)),
          iter_steps=4,
          train_steps=2,
          max_steps=5,
          accumulation_steps=2,
      )

  def test_hf_resume_rejects_early_exhaustion_during_remaining_training(self):
    bounded = diffusion_opd.replay_and_bound_iterator(
        iter(range(4)),
        iter_steps=4,
        train_steps=2,
        max_steps=5,
        accumulation_steps=2,
        replay_iterator=iter(range(4)),
    )

    with self.assertRaisesRegex(ValueError, "4 of 6 required remaining microbatches"):
      list(bounded)

  def test_hf_resume_rejects_checkpoint_beyond_configured_steps(self):
    with self.assertRaisesRegex(ValueError, "exceeds configured max_steps"):
      diffusion_opd.replay_and_bound_iterator(
          iter(()),
          iter_steps=12,
          train_steps=6,
          max_steps=5,
          accumulation_steps=2,
      )

  def test_hf_resume_at_final_step_does_not_replay(self):
    bounded = diffusion_opd.replay_and_bound_iterator(
        iter(()),
        iter_steps=10,
        train_steps=5,
        max_steps=5,
        accumulation_steps=2,
    )

    self.assertEqual(list(bounded), [])

  def test_checkpoint_contract_captures_batch_and_preprocessing_identity(self):
    base = diffusion_opd._checkpoint_contract(_student_config(), _teacher_config())  # pylint: disable=protected-access
    changed = diffusion_opd._checkpoint_contract(  # pylint: disable=protected-access
        _student_config(gradient_accumulation_steps=2, use_tunix_gradient_accumulation=True), _teacher_config()
    )

    self.assertNotEqual(base, changed)
    self.assertEqual(base["max_target_length"], 16)

  def test_checkpoint_restore_preserves_contract_without_claiming_hf_cursor_state(self):
    trainer = diffusion_opd.MaxTextDiffusionOPDTrainer.__new__(diffusion_opd.MaxTextDiffusionOPDTrainer)
    trainer.checkpoint_manager = mock.Mock()
    trainer.model = mock.sentinel.model
    trainer.optimizer = mock.sentinel.optimizer
    trainer.config = mock.Mock(checkpointing_options=mock.sentinel.options)
    trainer.config.get_with_default.return_value = 1
    trainer._lora_enabled = False  # pylint: disable=protected-access
    trainer.checkpoint_contract = {"version": 1, "teacher_model": "teacher"}
    manager = mock.Mock()
    manager.maybe_restore.return_value = (4, {"diffusion_opd_contract": trainer.checkpoint_contract})
    raw_iterator = SimpleNamespace(local_iterator=mock.sentinel.local_iterator)

    with mock.patch.object(diffusion_opd.distillation_utils, "MaxTextCheckpointManager", return_value=manager) as factory:
      result = trainer.setup_checkpoint_manager_and_restore(
          raw_iterator, SimpleNamespace(dataset_type="hf", checkpoint_dir="/tmp")
      )

    self.assertIs(result, raw_iterator)
    self.assertIsNone(factory.call_args.kwargs["raw_iterator"])
    self.assertEqual(trainer._train_steps, 4)  # pylint: disable=protected-access
    self.assertEqual(trainer.custom_checkpoint_metadata()["diffusion_opd_contract"], trainer.checkpoint_contract)

  def test_checkpoint_restore_rejects_semantic_contract_mismatch(self):
    trainer = diffusion_opd.MaxTextDiffusionOPDTrainer.__new__(diffusion_opd.MaxTextDiffusionOPDTrainer)
    trainer.checkpoint_manager = mock.Mock()
    trainer.model = mock.sentinel.model
    trainer.optimizer = mock.sentinel.optimizer
    trainer.config = mock.Mock(checkpointing_options=mock.sentinel.options)
    trainer.config.get_with_default.return_value = 1
    trainer._lora_enabled = False  # pylint: disable=protected-access
    trainer.checkpoint_contract = {"version": 1, "teacher_model": "expected"}
    manager = mock.Mock()
    manager.maybe_restore.return_value = (4, {"diffusion_opd_contract": {"version": 1, "teacher_model": "other"}})

    with (
        mock.patch.object(diffusion_opd.distillation_utils, "MaxTextCheckpointManager", return_value=manager),
        self.assertRaisesRegex(ValueError, "checkpoint contract"),
    ):
      trainer.setup_checkpoint_manager_and_restore(
          SimpleNamespace(local_iterator=mock.sentinel.local_iterator),
          SimpleNamespace(dataset_type="hf", checkpoint_dir="/tmp"),
      )
