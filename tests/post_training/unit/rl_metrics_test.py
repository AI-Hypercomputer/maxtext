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

import functools
import struct
from types import SimpleNamespace
from unittest import mock

from flax import nnx
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tunix.rl import algo_core, common
from tunix.rl.grpo.grpo_learner import GrpoConfig
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils
from tensorboardX.proto import event_pb2

from maxtext.integration.tunix import rl_metrics
from maxtext.configs import pyconfig, types
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.moe import RoutedMoE
from maxtext.trainers.post_train.rl import metrics
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

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
    actor_logger.log("actor", "loss", 0.125, "train", 1)
    actor_logger.close()
    cluster.global_steps = 4
    cluster.actor_trainer.train_steps = 8
    metrics.log_evaluation(
        logger, cluster, (2, 2, 100, 100, 100, 1.2), {"num_samples": 4, "num_reward_samples": 4}, 0.5, cfg
    )
  # Read TFRecord events using the writer's protobuf dependency, so this test
  # does not silently skip when the TensorBoard server package is absent.
  scalars = {}
  for path in tmp_path.glob("events.out.tfevents.*"):
    with path.open("rb") as stream:
      while length := stream.read(8):
        size = struct.unpack("<Q", length)[0]
        stream.read(4)  # Length CRC.
        event = event_pb2.Event.FromString(stream.read(size))
        stream.read(4)  # Payload CRC.
        for value in event.summary.value:
          scalars.setdefault(value.tag, []).append((event.step, value.simple_value))
  assert scalars["rl/eval/accuracy_percent"] == [(0, 50), (4, 100)]
  assert scalars["rl/steps/actor_updates"] == [(0, 0), (4, 8)]
  assert scalars["rl/eval/num_samples"][0][1] == 4
  assert scalars["rl/eval/num_reward_samples"][0][1] == 3
  assert scalars["actor/train/loss"] == [(1, 0.125)]


@pytest.mark.parametrize("enabled,process", [(False, 0), (True, 1)])
def test_disabled_or_non_primary_process_creates_no_backend(tmp_path, enabled, process):
  with (
      mock.patch.object(metrics.jax, "process_index", return_value=process),
      mock.patch.object(metrics, "_TensorboardBackend") as backend,
  ):
    with metrics.RLMetricLogger(_config(tmp_path, enable_tensorboard=enabled)) as logger:
      logger.write({"rl/eval/success": 1}, 0)
    backend.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_moe_metrics_flag_reaches_rl_config_from_cli(enabled):
  cfg = pyconfig.initialize_pydantic(
      [None, get_test_config_path("post_train/rl.yml"), f"rl.record_moe_metrics={str(enabled).lower()}"],
      config_class=types.RLConfig,
      model_name="llama2-7b",
      override_model_config=True,
      run_name="rl_metric_config",
      enable_checkpointing=False,
      attention="dot_product",
      base_emb_dim=128,
      base_num_decoder_layers=1,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      base_mlp_dim=256,
      max_target_length=8,
      max_prefill_predict_length=4,
      weight_dtype="bfloat16",
      tokenizer_path="meta-llama/Llama-2-7b",
  )
  assert cfg.rl.record_moe_metrics is enabled


def test_failed_or_unavailable_eval_reward_is_not_logged_as_zero():
  logger = mock.Mock()
  cluster = SimpleNamespace(global_steps=3, actor_trainer=SimpleNamespace(train_steps=6))
  metrics.log_evaluation(
      logger, cluster, (1, 2, 50, 50, 100, 0), {"num_samples": 4, "num_reward_samples": 0}, 0.1, _config("unused")
  )
  assert "rl/eval/mean_reward" not in logger.write.call_args.args[0]
  metrics.log_evaluation_failure(logger, cluster, 0.2)
  assert logger.write.call_args.args == ({"rl/eval/success": 0, "rl/eval/duration_seconds": 0.2}, 3)


def test_empty_evaluation_has_counts_but_no_accuracy():
  logger = mock.Mock()
  cluster = SimpleNamespace(global_steps=3, actor_trainer=SimpleNamespace(train_steps=6))
  metrics.log_evaluation(
      logger, cluster, (0, 0, 0, 0, 0, 0), {"num_samples": 0, "num_reward_samples": 0}, 0.1, _config("unused")
  )
  values, step = logger.write.call_args.args
  assert step == 3
  assert values["rl/eval/success"] == 0
  assert values["rl/eval/num_prompts"] == 0
  assert not any("accuracy" in tag or "reward" in tag and "num_" not in tag for tag in values)


def test_evaluation_exception_records_failure_and_propagates():
  logger = mock.Mock()
  cluster = SimpleNamespace(global_steps=7, maxtext_metrics_logger=logger)
  with pytest.raises(RuntimeError, match="generation failed"):
    metrics.evaluate_with_logging(
        mock.Mock(side_effect=RuntimeError("generation failed")), _config("unused"), [], cluster
    )
  values, step = logger.write.call_args.args
  assert step == 7
  assert values["rl/eval/success"] == 0
  assert set(values) == {"rl/eval/success", "rl/eval/duration_seconds"}


def test_mldiagnostics_backend_receives_evaluation_without_tensorboard():
  from tunix.rl.metrics import mldiag_backend  # pylint: disable=import-outside-toplevel

  with mock.patch.object(mldiag_backend, "MLDiagScalarBackend") as factory:
    with metrics.RLMetricLogger(_config("unused", enable_tensorboard=False, managed_mldiagnostics=True)) as logger:
      logger.write({"rl/eval/accuracy_percent": 50}, 7)
    factory.return_value.log_scalar.assert_called_once_with("rl/eval/accuracy_percent", 50.0, step=7)
    factory.return_value.close.assert_called_once()


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
  np.testing.assert_allclose(collapsed["expert_load_cv"], [np.sqrt(3)], rtol=1e-6)
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


def test_single_expert_and_empty_microbatch_are_finite():
  for mask in (jnp.ones((1, 2)), jnp.zeros((1, 2))):
    result = rl_metrics.routing_statistics(jnp.zeros((1, 2)), jnp.zeros((1, 2, 1), jnp.int32), mask, 1)
    assert all(np.isfinite(value).all() for value in result.values())
    np.testing.assert_array_equal(result["router_entropy_normalized"], [0])
    np.testing.assert_array_equal(result["expert_load_entropy_normalized"], [0])


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
    del positions, cache, attention_mask, segment_ids
    if forced_routed_experts is not None:
      assert forced_routed_experts.dtype == jnp.int32
    logits = self.head(self.embed(tokens))
    probs = jax.nn.softmax(logits[..., :4], axis=-1)
    self.rl_router_entropy = nnx.Intermediate(jax.lax.stop_gradient(-jnp.sum(probs * jnp.log(probs), axis=-1)))
    self.rl_expert_indices = nnx.Intermediate(jax.lax.top_k(logits[..., :4], 2)[1])
    return logits, None


@pytest.mark.parametrize("algo", ["grpo", "gspo-token"])
@pytest.mark.parametrize("layout", ["unpacked", "packed", "explicit_mask"])
@pytest.mark.parametrize("sampler_correction", [False, True])
def test_observation_preserves_loss_aux_and_parameter_gradients(algo, layout, sampler_correction):
  cfg = GrpoConfig(
      num_generations=2,
      beta=0.08,
      epsilon=0.2,
      loss_algo=algo,
      temperature=0.9,
      use_rollout_logps=sampler_correction,
      sampler_is="token" if sampler_correction else None,
  )
  example = common.TrainExample(
      prompt_ids=jnp.array([[0, 1, 3], [1, 4, 5]]),
      prompt_mask=jnp.array([[0, 1, 1], [1, 1, 1]]),
      completion_ids=jnp.array([[5, 2, 0], [6, 7, 2]]),
      completion_mask=jnp.array([[1, 1, 0], [1, 1, 1]]),
      advantages=jnp.array([1.0, -1.0]),
      ref_per_token_logps=jnp.full((2, 3), -3.0),
      old_per_token_logps=jnp.full((2, 3), -2.8),
      routed_experts=jnp.zeros((2, 6, 1, 2), dtype=jnp.int16),
  )
  baseline = _ToyPolicy()
  if layout == "explicit_mask":
    example = SimpleNamespace(**vars(example), completion_attention_mask=jnp.array([[1, 1, 1], [1, 1, 0]]))
  if layout == "packed":
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
        model,
        example,
        cfg,
        policy_loss=functools.partial(algo_core.grpo_loss_fn, pad_id=0, eos_id=2),
        pad_id=0,
        eos_id=2,
        supported=1,
        total=1,
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
  assert not nnx.state(observed, nnx.Intermediate), "Observations must not persist in the trainer/checkpoint state"


@pytest.fixture(name="routed_layer", params=["mixtral-8x7b", "deepseek3-tiny", "llama4-17b-16e"])
def _routed_layer(request):
  """Real router and experts, with small CPU shapes and no downloaded weights."""
  cfg = pyconfig.initialize(
      [None, get_test_config_path()],
      model_name=request.param,
      override_model_config=True,
      run_name="rl_router_observation",
      enable_checkpointing=False,
      enable_dropout=False,
      base_emb_dim=16,
      base_num_decoder_layers=1,
      first_num_dense_layers=0,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      base_mlp_dim=32,
      base_moe_mlp_dim=32,
      head_dim=8,
      vocab_size=32,
      max_target_length=8,
      max_prefill_predict_length=4,
      per_device_batch_size=1,
      num_experts=4,
      num_experts_per_tok=1 if request.param.startswith("llama4") else 2,
      n_routing_groups=1,
      topk_routing_group=1,
      sparse_matmul=False,
      dtype="float32",
      weight_dtype="float32",
      float32_gate_logits=True,
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with mesh, partitioning.axis_rules(cfg.logical_axis_rules):
    layer = RoutedMoE(
        config=cfg,
        num_experts=4,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=32,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(3),
    )
    if cfg.routed_bias:
      layer.gate.bias[...] = jnp.array([-0.6, -0.2, 0.2, 0.6])
    yield layer


@pytest.mark.parametrize("replay", [False, True])
def test_actual_router_observation_matches_decisions_and_preserves_gradients(routed_layer, replay):
  baseline = routed_layer
  observed = nnx.clone(baseline)
  observed.record_rl_metrics = True
  inputs = jax.random.normal(jax.random.key(4), (jax.device_count(), 8, 16))
  forced = jnp.tile(jnp.array([[[3, 0], [2, -1]]]), (jax.device_count(), 4, 1)) if replay else None
  if replay:
    forced = forced[..., : baseline.num_experts_per_tok]
  gate_logits, pre_bias = baseline.gate(inputs)
  scores = np.asarray(pre_bias if pre_bias is not None else gate_logits, dtype=np.float64)
  if baseline.config.model_name.startswith("llama4"):
    probs = 1 / (1 + np.exp(-scores))
  elif baseline.gate.score_func:
    probs = scores
  else:
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
  probs /= probs.sum(axis=-1, keepdims=True)
  expected_entropy = -(probs * np.log(probs)).sum(axis=-1)
  # These models select top-k scores (DeepSeek has a single group in this fixture).
  expected_ids = forced if replay else np.argsort(-np.asarray(gate_logits), axis=-1)[..., : baseline.num_experts_per_tok]

  def loss(model):
    output, _, _ = model(inputs, forced_routed_experts=forced)
    return jnp.mean(output**2), output

  (value, output), gradients = nnx.value_and_grad(loss, has_aux=True)(baseline)
  (actual_value, actual_output), actual_gradients = nnx.value_and_grad(loss, has_aux=True)(observed)
  np.testing.assert_allclose(observed.rl_router_entropy[...], expected_entropy, rtol=1e-6)
  np.testing.assert_array_equal(observed.rl_expert_indices[...], expected_ids)
  np.testing.assert_array_equal(actual_output, output)
  np.testing.assert_array_equal(actual_value, value)
  for actual, expected in zip(jax.tree.leaves(actual_gradients), jax.tree.leaves(gradients), strict=True):
    np.testing.assert_array_equal(actual, expected)


def test_configuration_wraps_the_installed_loss_and_only_the_actor(routed_layer):
  actor = _ToyPolicy()
  actor.router = routed_layer
  reference = nnx.clone(actor)

  def installed_loss(model, example, config):
    result = algo_core.grpo_loss_fn(model, example, config, pad_id=0, eos_id=2)
    return sft_utils.LossOutput(
        primary_loss=result.primary_loss,
        aux_metrics={**result.aux_metrics, "learner_specific_metric": jnp.float32(7)},
    )

  trainer = SimpleNamespace(model=actor, loss_fn=installed_loss)
  trainer.with_loss_fn = lambda fn, has_aux: setattr(trainer, "loss_fn", fn)
  cluster = SimpleNamespace(
      actor_trainer=trainer,
      reference=reference,
      rollout=SimpleNamespace(pad_id=lambda: 0, eos_id=lambda: 2),
      cluster_config=SimpleNamespace(training_config=SimpleNamespace(compute_logps_chunk_size=0)),
  )
  example = common.TrainExample(
      prompt_ids=jnp.array([[1]]),
      prompt_mask=jnp.ones((1, 1)),
      completion_ids=jnp.array([[3, 2]]),
      completion_mask=jnp.ones((1, 2)),
      advantages=jnp.ones(1),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )
  config = GrpoConfig(num_generations=2, beta=0, temperature=0.9)
  original = installed_loss(actor, example, config)
  rl_metrics.configure_moe_metrics(cluster, enabled=True)
  observed = trainer.loss_fn(actor, example, config)
  assert actor.router.record_rl_metrics
  assert not reference.router.record_rl_metrics
  np.testing.assert_array_equal(observed.primary_loss.compute(), original.primary_loss.compute())
  assert observed.aux_metrics["learner_specific_metric"] == 7
  assert "moe/router_entropy_mean_layer" in observed.aux_metrics
