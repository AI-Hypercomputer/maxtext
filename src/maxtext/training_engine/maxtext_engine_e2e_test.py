# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end RL Orchestrator training loop driver and integration test."""

from collections.abc import Iterator
import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
import optax


class DummyNNXModel(nnx.Module):

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))


@dataclasses.dataclass(kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  token_ids: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  token_mask: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))


class TrainingLoopRunner:
  """Drives an end-to-end RL training loop across MaxTextTrainingEngine APIs."""

  def __init__(
      self,
      trainer_instance: maxtext_engine.MaxTextTrainingEngine,
      microbatches_per_minibatch: int = 2,
      checkpoint_interval: int = 2,
      eval_interval: int = 2,
  ):
    self.trainer = trainer_instance
    self.microbatches_per_minibatch = microbatches_per_minibatch
    self.checkpoint_interval = checkpoint_interval
    self.eval_interval = eval_interval

  def run(
      self,
      train_dataloader: Iterator[abstract_engine.TrainerPayload],
      eval_dataloader: Iterator[abstract_engine.TrainerPayload],
      num_minibatches: int,
      dummy_compile_payload: abstract_engine.TrainerPayload | None = None,
  ) -> list[abstract_engine.MetricsBuffer]:
    """Executes the full RL training loop and returns step metric buffers."""
    history: list[abstract_engine.MetricsBuffer] = []

    _ = self.trainer.restore_checkpoint()
    if dummy_compile_payload is not None:
      self.trainer.compile(dummy_compile_payload)

    for step in range(1, num_minibatches + 1):
      for _ in range(self.microbatches_per_minibatch):
        micro_payload = next(train_dataloader)
        self.trainer.fwd_bwd(micro_payload)

      self.trainer.update()

      if step % self.checkpoint_interval == 0:
        self.trainer.save_checkpoint(metadata={"step": step, "source": "TrainingLoopRunner"})

      if step % self.eval_interval == 0:
        eval_payload = next(eval_dataloader)
        self.trainer.eval_step(eval_payload)

      step_metrics = self.trainer.get_metrics(clear_cache=True)
      history.append(step_metrics)

      _ = self.trainer.prepare_weight_sync()

    self.trainer.close()
    return history


class MaxTextTrainingEngineE2ETest(absltest.TestCase):

  def setup_config(self, enable_checkpointing: bool = False):
    """Sets up mock configuration for testing."""
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.init_weights_seed = 42
    mock_config.model_name = "llama3.1-8b"
    mock_config.tensorboard_dir = "/tmp/tb_dir"
    mock_config.run_name = "test_run"
    mock_config.enable_tensorboard = False
    if enable_checkpointing:
      mock_config.checkpoint_directory = "/tmp/test_out/e2e_checkpoints"
      mock_config.checkpoint_period = 2
      mock_config.max_num_checkpoints_to_keep = 5
      mock_config.async_checkpointing = True
    return mock_config

  @mock.patch.object(maxtext_engine.train_utils, "create_training_optimizer")
  @mock.patch.object(maxtext_engine.checkpointing, "CheckpointManager")
  @mock.patch.object(
      maxtext_engine.gradient_accumulation,
      "gradient_accumulation_loss_and_grad",
  )
  @mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained")
  def test_e2e_training_loop_exercises_all_trainer_apis(
      self, mock_from_pretrained, mock_ga, unused_mock_ckpt_mgr, mock_create_opt
  ):
    mock_config = self.setup_config(enable_checkpointing=True)
    dummy_model = DummyNNXModel()
    mock_from_pretrained.return_value = dummy_model
    dummy_opt = nnx.Optimizer(dummy_model, optax.sgd(0.01), wrt=nnx.Param)
    mock_create_opt.return_value = (lambda step: jnp.array(0.001), dummy_opt)
    mock_ga.return_value = (
        jnp.array(0.25),
        {},
        {"weights": jnp.array([0.1, 0.1])},
    )

    trainer_instance = maxtext_engine.MaxTextTrainingEngine(mock_config)
    trainer_instance._checkpoint_manager.restore_checkpoint.return_value = (  # pylint: disable=protected-access
        None,
        {},
    )

    trainer_instance.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(1.0), denominator=jnp.array(4.0)),
            {},
        )
    )

    runner = TrainingLoopRunner(
        trainer_instance=trainer_instance,
        microbatches_per_minibatch=2,
        checkpoint_interval=2,
        eval_interval=2,
    )

    def payload_generator() -> Iterator[abstract_engine.TrainerPayload]:
      while True:
        yield DummyPayload()

    history = runner.run(
        train_dataloader=payload_generator(),
        eval_dataloader=payload_generator(),
        num_minibatches=4,
        dummy_compile_payload=DummyPayload(),
    )

    self.assertLen(history, 4)
    for metrics_buf_list in history:
      self.assertIsInstance(metrics_buf_list, list)
      self.assertNotEmpty(metrics_buf_list)
      self.assertIsInstance(metrics_buf_list[0], abstract_engine.MetricsBuffer)
    self.assertEqual(trainer_instance.train_step, 4)


if __name__ == "__main__":
  absltest.main()
