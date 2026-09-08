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

"""End-to-end MaxText training engine test."""

from collections.abc import Iterator
import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import optax
import pytest

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]


class DummyNNXModel(nnx.Module):
  """Dummy NNX model for testing."""

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))


@struct.dataclass(frozen=True, kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  """Dummy payload for testing."""

  token_ids: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  token_mask: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))


class TrainingLoopRunner:
  """Drives an end-to-end training loop across MaxTextTrainingEngine APIs."""

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
    """Executes the full training loop and returns step metric buffers."""
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


@pytest.mark.integration_test
class MaxTextTrainingEngineE2ETest(absltest.TestCase):
  """End-to-end MaxText training engine test."""

  def setup_config(self, enable_checkpointing: bool = False, **kwargs):
    """Sets up a MaxText config via pyconfig.initialize."""
    overrides = {
        "model_name": "llama3.1-8b",
        "run_name": "test_run",
        "base_output_directory": self.create_tempdir().full_path,
        "init_weights_seed": 42,
        "micro_batch_size_to_train_on": 2,
        "gradient_accumulation_steps": 1,
        "enable_dropout": False,
        "record_internal_nn_metrics": False,
        "enable_tensorboard": False,
        "tensorboard_dir": self.create_tempdir().full_path,
        "skip_jax_distributed_system": True,
        "enable_checkpointing": enable_checkpointing,
        # Disable scan_layers to prevent prepare_weight_sync from trying to unscan layers on DummyNNXModel
        "scan_layers": False,
    }
    if enable_checkpointing:
      overrides.update(
          {
              "checkpoint_dir": self.create_tempdir().full_path,
              "checkpoint_period": 2,
              "max_num_checkpoints_to_keep": 5,
              "async_checkpointing": True,
          }
      )
    overrides.update(kwargs)
    return pyconfig.initialize([None, get_test_config_path()], **overrides)

  @mock.patch.object(maxtext_engine.train_utils, "create_training_optimizer")
  @mock.patch.object(maxtext_engine.checkpointing, "CheckpointManager")
  @mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained")
  def test_e2e_training_loop_exercises_all_trainer_apis(self, mock_from_pretrained, mock_ckpt_mgr, mock_create_opt):
    mock_config = self.setup_config(enable_checkpointing=True)
    dummy_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(mock_config), mock_config.mesh_axes)
    dummy_model = DummyNNXModel()
    # This test constructs the engine without a mesh, and `from_pretrained` returns
    # `(model, model.mesh)` in that case -- it enters `with mesh:` before returning, so
    # the mesh it hands back is never None. (It returns a bare model only when the
    # caller supplies a mesh, which this test does not.)
    mock_from_pretrained.return_value = (dummy_model, dummy_mesh)
    # `create_training_optimizer` returns `(schedule, tx)` where `tx` is a raw optax
    # GradientTransformation; the engine wraps it in an nnx.Optimizer itself. Returning
    # an already-wrapped nnx.Optimizer here would make the engine wrap it twice.
    mock_create_opt.return_value = (lambda step: jnp.array(0.001), optax.sgd(0.01))

    mock_ckpt_mgr_inst = mock.MagicMock()
    mock_ckpt_mgr_inst.restore_checkpoint.return_value = (None, None, None)
    mock_ckpt_mgr.return_value = mock_ckpt_mgr_inst

    trainer_instance = maxtext_engine.MaxTextTrainingEngine(mock_config)

    trainer_instance.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.25), denominator=jnp.array(1.0)),
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
    for metrics_buf in history:
      self.assertIsInstance(metrics_buf, abstract_engine.MetricsBuffer)
    self.assertEqual(trainer_instance.train_step, 4)


if __name__ == "__main__":
  absltest.main()
