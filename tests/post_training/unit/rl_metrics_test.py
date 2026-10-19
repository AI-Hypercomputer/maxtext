# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks of RL scalar delivery, routing statistics and loss invariance."""

from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tunix.rl import algo_core, common
from tunix.rl.grpo.grpo_learner import GrpoConfig
from tunix.sft import metrics_logger

from maxtext.integration.tunix import rl_metrics
from maxtext.trainers.post_train.rl import metrics

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]


def _config(path, **kwargs):
  values = {
      "enable_tensorboard": True,
      "managed_mldiagnostics": False,
      "tensorboard_dir": str(path),
      "log_period": 20,
      "num_eval_passes": 2,
      "eval_mode": "pass_at_1",
  }
  return SimpleNamespace(**(values | kwargs))


def test_eval_events_survive_actor_logger_shutdown(tmp_path):
  accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
  cfg = _config(tmp_path)
  cluster = SimpleNamespace(global_steps=0, actor_trainer=SimpleNamespace(train_steps=0))
  with metrics.RLMetricLogger(cfg) as logger:
    metrics.log_evaluation(
        logger, cluster, (1, 2, 50, 50, 100, 0.6), {"num_samples": 4, "num_reward_samples": 3}, 0.25, cfg
    )
    # Tunix owns and closes a different logger at the end of train().
    actor_logger = metrics_logger.MetricsLogger(
        metrics_logger.MetricsLoggerOptions(log_dir=str(tmp_path), flush_every_n_steps=1)
    )
    actor_logger.close()
    cluster.global_steps = 4
    cluster.actor_trainer.train_steps = 8
    metrics.log_evaluation(
        logger, cluster, (2, 2, 100, 100, 100, 1.2), {"num_samples": 4, "num_reward_samples": 4}, 0.5, cfg
    )
  events = accumulator.EventAccumulator(str(tmp_path))
  events.Reload()
  accuracy = events.Scalars("rl/eval/accuracy_percent")
  assert [(e.step, e.value) for e in accuracy] == [(0, 50), (4, 100)]
  assert [(e.step, e.value) for e in events.Scalars("rl/steps/actor_updates")] == [(0, 0), (4, 8)]
  assert events.Scalars("rl/eval/num_samples")[0].value == 4
  assert events.Scalars("rl/eval/num_reward_samples")[0].value == 3


@pytest.mark.parametrize("enabled,process", [(False, 0), (True, 1)])
def test_disabled_or_non_primary_process_creates_no_backend(tmp_path, enabled, process):
  with (
      mock.patch.object(metrics.jax, "process_index", return_value=process),
      mock.patch.object(metrics, "_TensorboardBackend") as backend,
  ):
    with metrics.RLMetricLogger(_config(tmp_path, enable_tensorboard=enabled)) as logger:
      logger.write({"rl/eval/success": 1}, 0)
    backend.assert_not_called()


def test_failed_or_unavailable_eval_reward_is_not_logged_as_zero():
  logger = mock.Mock()
  cluster = SimpleNamespace(global_steps=3, actor_trainer=SimpleNamespace(train_steps=6))
  metrics.log_evaluation(
      logger, cluster, (1, 2, 50, 50, 100, 0), {"num_samples": 4, "num_reward_samples": 0}, 0.1, _config("unused")
  )
  assert "rl/eval/mean_reward" not in logger.write.call_args.args[0]
  metrics.log_evaluation_failure(logger, cluster, 0.2)
  assert logger.write.call_args.args == ({"rl/eval/success": 0, "rl/eval/duration_seconds": 0.2}, 3)


def test_router_confidence_and_expert_balance_are_distinct():
  mask = jnp.ones((1, 4))
  balanced = jnp.array([[[0], [1], [2], [3]]])
  uniform = rl_metrics.routing_statistics(jnp.full((1, 4), jnp.log(4)), balanced, mask, 4)
  confident = rl_metrics.routing_statistics(jnp.zeros((1, 4)), balanced, mask, 4)
  collapsed = rl_metrics.routing_statistics(jnp.zeros((1, 4)), jnp.zeros((1, 4, 1), jnp.int32), mask, 4)
  np.testing.assert_allclose(uniform["router_entropy_normalized"], [1])
  np.testing.assert_allclose(confident["router_entropy_normalized"], [0])
  for result in (uniform, confident):
    np.testing.assert_allclose(result["expert_load_entropy_normalized"], [1])
    np.testing.assert_allclose(result["expert_load_cv"], [0])
    np.testing.assert_allclose(result["max_to_mean_load"], [1])
  np.testing.assert_allclose(collapsed["expert_load_entropy_normalized"], [0])
  np.testing.assert_allclose(collapsed["max_to_mean_load"], [4])
  np.testing.assert_allclose(collapsed["unused_expert_fraction"], [0.75])


def test_routing_padding_invalid_ids_and_scanned_layers_under_jit():
  mask = jnp.array([[1, 1, 0]])
  entropy = jnp.array([[[1.0, 3.0, jnp.nan]], [[2.0, 4.0, jnp.nan]]])
  ids = jnp.array([[[[0, 1], [2, 3], [0, 0]]], [[[0, -1], [1, 9], [0, 0]]]])
  result = jax.jit(rl_metrics.routing_statistics, static_argnums=3)(entropy, ids, mask, 4)
  np.testing.assert_allclose(result["router_entropy"], [2, 3])
  np.testing.assert_allclose(result["valid_tokens"], [2, 2])
  np.testing.assert_allclose(result["max_to_mean_load"], [1, 2])
  np.testing.assert_allclose(result["unused_expert_fraction"], [0, 0.5])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple CPU devices")
def test_routing_load_is_reduced_across_sharded_tokens():
  mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("data",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
  # Each shard sees only half of the experts, but the global load is balanced.
  entropy = jax.device_put(jnp.zeros((4, 1)), sharding)
  ids = jax.device_put(jnp.arange(4).reshape((4, 1, 1)), sharding)
  mask = jax.device_put(jnp.ones((4, 1)), sharding)
  result = jax.jit(rl_metrics.routing_statistics, static_argnums=3)(entropy, ids, mask, 4)
  np.testing.assert_allclose(result["expert_load_entropy_normalized"], [1])
  np.testing.assert_allclose(result["max_to_mean_load"], [1])


class _ToyPolicy(nnx.Module):
  """A small differentiable policy with router observations for loss parity tests."""

  def __init__(self):
    self.base = nnx.Dict(config=SimpleNamespace(num_experts=4))
    self.embed = nnx.Embed(16, 4, rngs=nnx.Rngs(1))
    self.head = nnx.Linear(4, 16, rngs=nnx.Rngs(2))

  def __call__(self, tokens, positions, cache, attention_mask, segment_ids=None, forced_routed_experts=None):  # pylint: disable=arguments-differ
    del positions, cache, attention_mask, segment_ids, forced_routed_experts
    logits = self.head(self.embed(tokens))
    probs = jax.nn.softmax(logits[..., :4], axis=-1)
    self.rl_router_entropy = nnx.Intermediate(jax.lax.stop_gradient(-jnp.sum(probs * jnp.log(probs), axis=-1)))
    self.rl_expert_indices = nnx.Intermediate(jax.lax.top_k(logits[..., :4], 2)[1])
    return logits, None


@pytest.mark.parametrize("algo", ["grpo", "gspo-token"])
@pytest.mark.parametrize("packed", [False, True])
def test_observation_preserves_loss_aux_and_parameter_gradients(algo, packed):
  cfg = GrpoConfig(num_generations=2, beta=0.08, epsilon=0.2, loss_algo=algo, temperature=0.9)
  example = common.TrainExample(
      prompt_ids=jnp.array([[0, 1, 3], [1, 4, 5]]),
      prompt_mask=jnp.array([[0, 1, 1], [1, 1, 1]]),
      completion_ids=jnp.array([[5, 2, 0], [6, 7, 2]]),
      completion_mask=jnp.array([[1, 1, 0], [1, 1, 1]]),
      advantages=jnp.array([1.0, -1.0]),
      ref_per_token_logps=jnp.full((2, 3), -3.0),
      old_per_token_logps=jnp.full((2, 3), -2.8),
  )
  baseline = _ToyPolicy()
  if packed:
    example = example.replace(
        prompt_ids=jnp.zeros((2, 0), jnp.int32),
        prompt_mask=jnp.zeros((2, 0), jnp.int32),
        completion_ids=jnp.array([[1, 3, 5, 1, 6, 2], [1, 4, 6, 2, 0, 0]]),
        completion_mask=jnp.array([[0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0]]),
        segment_ids=jnp.array([[1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 0, 0]]),
        segment_positions=jnp.array([[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 0, 0]]),
        num_segments=3,
        advantages=jnp.array([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]),
        ref_per_token_logps=jnp.full((2, 6), -3.0),
        old_per_token_logps=jnp.full((2, 6), -2.8),
    )
  observed = nnx.clone(baseline)

  def original(model):
    result = algo_core.grpo_loss_fn(model, example, cfg, pad_id=0, eos_id=2)
    return result.primary_loss.compute(), result

  def with_metrics(model):
    result = rl_metrics._observed_loss(  # pylint: disable=protected-access
        model, example, cfg, policy_loss=algo_core.grpo_loss_fn, pad_id=0, eos_id=2, supported=1, total=1
    )
    return result.primary_loss.compute(), result

  (loss, aux), gradients = nnx.value_and_grad(original, has_aux=True)(baseline)
  (new_loss, new_aux), new_gradients = nnx.value_and_grad(with_metrics, has_aux=True)(observed)
  np.testing.assert_allclose(new_loss, loss, rtol=1e-6)
  for actual, expected in zip(jax.tree.leaves(new_gradients), jax.tree.leaves(gradients), strict=True):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
  for key, value in aux.aux_metrics.items():
    actual = new_aux.aux_metrics[key]
    np.testing.assert_allclose(
        actual.compute() if hasattr(actual, "compute") else actual,
        value.compute() if hasattr(value, "compute") else value,
        rtol=1e-6,
    )
  assert "moe/router_entropy_mean_layer" in new_aux.aux_metrics
