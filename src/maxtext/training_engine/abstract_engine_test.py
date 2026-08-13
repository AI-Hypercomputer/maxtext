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

"""Tests for trainer abstractions."""

import dataclasses
from typing import Any
from absl.testing import absltest
import jax.numpy as jnp
from maxtext.training_engine import abstract_engine


@dataclasses.dataclass(kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  """Dummy payload for testing."""

  data: str = "batch_1"


class DummyTrainingEngine(abstract_engine.AbstractTrainingEngine):
  """Minimal concrete implementation of AbstractTrainingEngine for unit testing."""

  def __init__(self, training_config: abstract_engine.TrainingConfig) -> None:
    self.config = training_config
    self.loss_fn = None
    self.compiled = False
    self.fwd_bwd_called = 0
    self.update_called = 0
    self.checkpoint_saved_with_metadata: Any = None
    self.restored_metadata: Any = {"global_step": 42}

  def with_loss_fn(self, customized_fn: Any) -> None:
    self.loss_fn = customized_fn

  def with_gen_model_input_fn(self, gen_model_input_fn: Any) -> "DummyTrainingEngine":
    self._gen_model_input_fn = gen_model_input_fn
    return self

  def compile(self, dummy_data: abstract_engine.TrainerPayload) -> None:
    self.compiled = True

  def fwd_bwd(self, payload: abstract_engine.TrainerPayload) -> None:
    self.fwd_bwd_called += 1

  def update(self) -> None:
    self.update_called += 1

  def eval_step(self, payload: abstract_engine.TrainerPayload, **kwargs: Any) -> None:
    pass

  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    self.checkpoint_saved_with_metadata = metadata

  def restore_checkpoint(self, **kwargs: Any) -> Any:
    return self.restored_metadata

  def get_metrics(self, clear_cache: bool = True) -> abstract_engine.MetricsBuffer:
    return abstract_engine.MetricsBuffer(id=1)

  def prepare_weight_sync(self, **kwargs: Any) -> Any:
    return {"endpoint": "grpc://dummy-trainer:55555"}


class AbstractTrainingEngineTest(absltest.TestCase):

  def test_abstract_training_engine_cannot_be_instantiated_directly(self):
    config = abstract_engine.TrainingConfig()
    with self.assertRaises(TypeError):
      abstract_engine.AbstractTrainingEngine(config)  # pytype: disable=not-instantiable  # pylint: disable=abstract-class-instantiated

  def test_dummy_training_engine_implements_abstract_interface(self):
    config = abstract_engine.TrainingConfig(max_steps=100)
    t = DummyTrainingEngine(config)
    self.assertEqual(t.config.max_steps, 100)
    payload = DummyPayload(
        data="dummy",
        token_ids=jnp.ones((2, 2)),
        token_mask=jnp.ones((2, 2)),
    )
    t.compile(payload)
    self.assertTrue(t.compiled)
    t.fwd_bwd(payload)
    self.assertEqual(t.fwd_bwd_called, 1)
    t.update()
    self.assertEqual(t.update_called, 1)

  def test_weighted_metric_compute(self):
    m = abstract_engine.WeightedMetric(
        unreduced_sum=jnp.array(10.0),
        denominator=jnp.array(2.0),
    )
    self.assertAlmostEqual(float(m.compute()), 5.0)


if __name__ == "__main__":
  absltest.main()
