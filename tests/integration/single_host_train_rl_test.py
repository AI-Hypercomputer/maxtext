# Copyright 2023–2025 Google LLC
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

"""
Integration test for train RL GRPO and GSPO-token workflows with actual dataset/checkpoint.
Tests the workflow in src/maxtext/trainers/post_train/rl/train_rl.py
"""

import os
import logging
import unittest
import pytest
import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx
from transformers import AutoTokenizer

from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.utils import max_logging


train_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.train_rl",
    reason="Tunix is not installed on the GPU image",
)


evaluate_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.evaluate_rl",
    reason="math_verify is not installed on the GPU image",
)


@pytest.mark.external_training
class RLTrainerIntegrationTests(unittest.TestCase):
  """Integration tests for the RL GRPO workflow."""

  CONFIGS = {
      "maxtext_on_vllm": [
          "run_name=test_qwen_rl_maxtext",
          "vllm_hf_overrides={architectures: ['MaxTextForCausalLM']}",
          "vllm_additional_config={'maxtext_config': {'model_name': 'qwen3-0.6b', 'log_config': 'false'}}",
      ],
      "vllm_native": [
          "run_name=test_qwen_rl_native",
      ],
  }

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["TPU_BACKEND_TYPE"] = "jax"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    self.base_argv = [
        "train_rl.py",
        os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "rl.yml"),
        "model_name=qwen3-0.6b",
        "tokenizer_path=Qwen/Qwen3-0.6B",
        f"base_output_directory=/tmp/maxtext_rl_test/{jax.random.PRNGKey(0)[0]}",
        "batch_size=4",
        "num_batches=1",  # Minimal batches for integration test speed
        "num_test_batches=1",
        f"chips_per_vm={jax.device_count()}",
        "scan_layers=True",
        "hbm_utilization_vllm=0.4",
        "rollout_data_parallelism=1",
        "rollout_tensor_parallelism=-1",
        "rl.num_generations=16",
        "load_parameters_path=gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items",
    ]

  def _run_rl_workflow_end_to_end(self, extra_argv):
    """Helper method to run the full RL training workflow and verify parameter updates."""
    argv = self.base_argv + extra_argv

    trainer_config, sampler_config, trainer_devices, sampler_devices = train_rl.setup_configs_and_devices(argv)

    if not trainer_config.debug.rl:
      noise_filter = max_logging.NoisyLogFilter()
      logging.getLogger().addFilter(noise_filter)

    max_train_steps = train_rl.get_max_train_steps(trainer_config)
    model_tokenizer = AutoTokenizer.from_pretrained(trainer_config.tokenizer_path)
    train_dataset, test_dataset = train_rl.prepare_datasets(trainer_config, model_tokenizer)

    reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = train_rl.create_models_and_meshes(
        trainer_config, sampler_config, trainer_devices, sampler_devices
    )

    rl_cluster, rl_trainer, _ = train_rl.create_rl_components(
        trainer_config,
        sampler_config,
        sampler_devices,
        actor_model,
        actor_mesh,
        reference_model,
        reference_mesh,
        rollout_mesh,
        model_tokenizer,
        max_train_steps,
    )

    # Pre-train evaluation
    pre_metrics, _ = evaluate_rl.evaluate(
        trainer_config,
        test_dataset,
        rl_cluster=rl_cluster,
        num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst,
        make_lst=trainer_config.eval_make_lst,
    )
    actor_initial_state = nnx.to_pure_dict(nnx.state(actor_model.base, nnx.Param))
    ref_initial_state = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

    # Training
    rl_trainer.train(train_dataset)

    # Block until training is complete and all async operations are done
    actor_final_state = nnx.to_pure_dict(nnx.state(actor_model.base, nnx.Param))
    ref_final_state = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

    # Post-train evaluation
    post_metrics, _ = evaluate_rl.evaluate(
        trainer_config,
        test_dataset,
        rl_cluster=rl_cluster,
        num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst,
        make_lst=trainer_config.eval_make_lst,
    )

    # Verify results
    for metrics in [pre_metrics, post_metrics]:
      corr, total, acc, partial_acc, format_acc = metrics
      self.assertGreaterEqual(total, 1, "There should be at least one eval item")
      self.assertGreaterEqual(corr, 0, "Correct items should be non-negative")
      self.assertTrue(0.0 <= partial_acc <= 100.0, "Partial accuracy should be a percentage between 0 and 100")
      self.assertTrue(0.0 <= acc <= 100.0, "Accuracy should be a percentage between 0 and 100")
      self.assertTrue(0.0 <= format_acc <= 100.0, "Format accuracy should be a percentage between 0 and 100")

    # Reference model parameters should NOT change during training
    self.assertTrue(
        jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, ref_initial_state, ref_final_state)),
        "Reference model parameters should remain unchanged.",
    )

    # Actor model parameters SHOULD change from their initial state
    is_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, actor_initial_state, actor_final_state)
    )
    self.assertTrue(is_changed, "Actor model parameters should have changed after training.")

    # Actor model parameters should be DIFFERENT from reference model after training
    is_different = not jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, ref_final_state, actor_final_state))
    self.assertTrue(is_different, "Actor model parameters should be different from reference model after training.")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_GRPO_workflow_maxtext_on_vllm(self):
    """Tests the RL training workflow with the maxtext_on_vllm config."""
    self._run_rl_workflow_end_to_end(self.CONFIGS["maxtext_on_vllm"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_GRPO_workflow_vllm_native(self):
    """Tests the RL training workflow with the vllm_native config."""
    self._run_rl_workflow_end_to_end(self.CONFIGS["vllm_native"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_GSPO_workflow_maxtext_on_vllm(self):
    """Tests the RL training workflow with the maxtext_on_vllm config."""
    self._run_rl_workflow_end_to_end(self.CONFIGS["maxtext_on_vllm"] + ["rl.loss_algo=gspo-token"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_GSPO_workflow_vllm_native(self):
    """Tests the RL training workflow with the vllm_native config."""
    self._run_rl_workflow_end_to_end(self.CONFIGS["vllm_native"] + ["rl.loss_algo=gspo-token"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_rl_train_maxtext_on_vllm(self):
    """Tests the rl_train API with the maxtext on vllm config."""
    trainer_config, sampler_config, trainer_devices, sampler_devices = train_rl.setup_configs_and_devices(
        self.CONFIGS["maxtext_on_vllm"]
    )
    train_rl.rl_train(trainer_config, sampler_config, trainer_devices, sampler_devices)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @pytest.mark.skip
  def test_rl_train_native(self):
    """Tests the rl_train API with the vllm native config."""
    trainer_config, sampler_config, trainer_devices, sampler_devices = train_rl.setup_configs_and_devices(
        self.CONFIGS["vllm_native"]
    )
    train_rl.rl_train(trainer_config, sampler_config, trainer_devices, sampler_devices)


if __name__ == "__main__":
  absltest.main()
