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

"""Tests exact MaxText denoising-trace replay for diffusion RL."""

import functools
from types import SimpleNamespace

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.diffusion import denoise
from maxtext.diffusion import scoring
from maxtext.integration.tunix import diffusion_rl
from tunix.rl import diffusion as tunix_diffusion


def _config(**overrides):
  """Builds the minimal diffusion configuration used by these tests."""
  values = {
      "block_diffusion_block_size": 4,
      "block_diffusion_mask_id": 7,
      "block_diffusion_logit_alignment": "same_position",
      "block_diffusion_canvas_policy": "all_masked",
      "vocab_size": 8,
      "data_shuffle_seed": 3,
      "rl": SimpleNamespace(
          diffusion_stop_token_ids=[],
          diffusion_max_denoise_steps=-1,
          diffusion_confidence_threshold=0.99,
      ),
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class _CanvasDependentModel(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(2.0, dtype=jnp.float32))

  def __call__(self, decoder_input_tokens, **kwargs):
    del kwargs
    committed = jnp.sum(decoder_input_tokens != 7, axis=1, keepdims=True)
    target_ids = (committed + jnp.arange(decoder_input_tokens.shape[1])[None, :]) % 7
    return jax.nn.one_hot(target_ids, 8, dtype=jnp.float32) * self.scale[...]


class _NonFiniteModel(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

  def __call__(self, decoder_input_tokens, **kwargs):
    del kwargs
    return jnp.full((*decoder_input_tokens.shape, 8), jnp.nan) * self.scale[...]


class _MeshConstrainedModel(nnx.Module):

  def __init__(self, mesh):
    self.mesh = mesh
    self.scale = nnx.Param(jnp.asarray(2.0, dtype=jnp.float32))
    self._shard = functools.partial(
        jax.lax.with_sharding_constraint,
        shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
    )

  def __call__(self, decoder_input_tokens, **kwargs):
    del kwargs
    tokens = self._shard(decoder_input_tokens)
    target_ids = jnp.arange(tokens.shape[1])[None, :] % 7
    return jax.nn.one_hot(target_ids, 8, dtype=jnp.float32) * self.scale[...]


class _MeshConstrainedWrapper(nnx.Module):

  def __init__(self, mesh):
    self.base = _MeshConstrainedModel(mesh)


class _Tokenizer:
  """Minimal tokenizer contract used by the in-process rollout tests."""

  def encode(self, text):
    del text
    return [4]

  def decode(self, token_ids):
    return " ".join(str(token_id) for token_id in token_ids)

  def pad_id(self):
    return 0

  def eos_id(self):
    return 6


class _PadValuedPromptTokenizer(_Tokenizer):

  def encode(self, text):
    del text
    return [0, 4]


class DiffusionRLTest(absltest.TestCase):

  def test_rollout_rebinds_nested_model_constraints_to_sampler_mesh(self):
    if len(jax.devices()) < 2:
      self.skipTest("requires two CPU devices")
    trainer_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sampler_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[1:2]), ("data",))
    model = _MeshConstrainedWrapper(trainer_mesh)
    graphdef, state = nnx.split(model)
    sampler_state = jax.tree.map(
        lambda value: jax.device_put(
            value,
            jax.sharding.NamedSharding(sampler_mesh, jax.sharding.PartitionSpec()),
        ),
        state,
    )
    model = nnx.merge(graphdef, sampler_state)
    rollout_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
    )

    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=model,
        tokenizer=_Tokenizer(),
        mesh=sampler_mesh,
        rollout_config=rollout_config,
        maxtext_config=_config(),
    )
    output = rollout.generate(["prompt"], rollout_config)

    self.assertEqual(rollout._model.base.mesh, sampler_mesh)  # pylint: disable=protected-access
    self.assertEqual(
        rollout._model.base._shard.keywords["shardings"].mesh,  # pylint: disable=protected-access
        sampler_mesh,
    )
    self.assertEqual(output.diffusion_batch.target_ids.shape, (1, 3))

  def test_trace_replay_matches_action_time_logps_and_has_live_gradient(self):
    model = _CanvasDependentModel()
    initial = jnp.asarray([[4, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion_mask = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)
    temperature = 0.7

    trace = denoise.low_confidence_rollout(
        lambda canvas: model(decoder_input_tokens=canvas),
        initial,
        positions,
        jnp.ones_like(completion_mask),
        completion_mask,
        block_size=4,
        mask_id=7,
        logit_alignment="same_position",
        canvas_policy="all_masked",
        confidence_threshold=0.99,
        temperature=temperature,
        rng=jax.random.PRNGKey(7),
    )
    batch = diffusion_rl.prepare_diffusion_policy_batch(
        prompt_tokens=initial[:, :1],
        prompt_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        completion_tokens=trace.tokens[:, 1:],
        completion_mask=trace.action_steps[:, 1:] >= 0,
        action_steps=trace.action_steps[:, 1:],
    )
    logits_fn = diffusion_rl.make_diffusion_trace_logits_fn(_config())
    replayed = tunix_diffusion.compute_diffusion_per_token_logps(
        model,
        batch,
        logits_fn,
        temperature=temperature,
        stop_gradient=False,
    )

    np.testing.assert_allclose(replayed, trace.action_logps[:, 1:], rtol=1e-6, atol=1e-6)
    gradients = nnx.grad(
        lambda scored_model: jnp.sum(
            tunix_diffusion.compute_diffusion_per_token_logps(
                scored_model,
                batch,
                logits_fn,
                temperature=temperature,
                stop_gradient=False,
            )
        )
    )(model)
    self.assertGreater(float(jnp.abs(gradients.scale[...])), 0.0)

  def test_shifted_seed_trace_replay_matches_action_time_logps(self):
    model = _CanvasDependentModel()
    initial = jnp.asarray([[2, 3, 4, 5, 7, 7, 7, 7]], dtype=jnp.int32)
    positions = jnp.arange(8, dtype=jnp.int32)[None, :]
    completion_mask = jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.bool_)
    temperature = 0.7

    trace = denoise.low_confidence_rollout(
        lambda canvas: scoring.align_logits_to_targets(
            model(decoder_input_tokens=canvas),
            "shifted",
            positions,
            jnp.ones_like(completion_mask),
        ),
        initial,
        positions,
        jnp.ones_like(completion_mask),
        completion_mask,
        block_size=4,
        mask_id=7,
        logit_alignment="shifted",
        canvas_policy="seed_and_mask",
        confidence_threshold=0.99,
        temperature=temperature,
        rng=jax.random.PRNGKey(9),
    )
    batch = diffusion_rl.prepare_diffusion_policy_batch(
        prompt_tokens=initial[:, :4],
        prompt_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        completion_tokens=trace.tokens[:, 4:],
        completion_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        action_steps=trace.action_steps[:, 4:],
    )
    config = _config(
        block_diffusion_logit_alignment="shifted",
        block_diffusion_canvas_policy="seed_and_mask",
    )

    replayed = tunix_diffusion.compute_diffusion_per_token_logps(
        model,
        batch,
        diffusion_rl.make_diffusion_trace_logits_fn(config),
        temperature=temperature,
        stop_gradient=True,
    )

    np.testing.assert_allclose(replayed, trace.action_logps[:, 4:], rtol=1e-6, atol=1e-6)

  def test_prepared_batch_rejects_action_mask_mismatch(self):
    with self.assertRaisesRegex(ValueError, "exactly one nonnegative action step"):
      diffusion_rl.prepare_diffusion_policy_batch(
          prompt_tokens=jnp.asarray([[4]]),
          prompt_mask=jnp.asarray([[1]], dtype=jnp.bool_),
          completion_tokens=jnp.asarray([[5, 6]]),
          completion_mask=jnp.asarray([[1, 1]], dtype=jnp.bool_),
          action_steps=jnp.asarray([[0, -1]]),
      )

  def test_stop_tokens_mask_actions_after_first_match(self):
    keep = diffusion_rl.policy_mask_at_stop(
        jnp.asarray([[4, 5, 6, 9]]),
        jnp.asarray([[0, 1, 2, 3]]),
        (5, 8),
    )

    np.testing.assert_array_equal(keep, [[1, 1, 0, 0]])

  def test_stop_mask_preserves_full_trace_for_exact_replay(self):
    model = _CanvasDependentModel()
    initial = jnp.asarray([[4, 7, 7, 7]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    completion_mask = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.bool_)
    temperature = 0.7
    trace = denoise.low_confidence_rollout(
        lambda canvas: model(decoder_input_tokens=canvas),
        initial,
        positions,
        jnp.ones_like(completion_mask),
        completion_mask,
        block_size=4,
        mask_id=7,
        logit_alignment="same_position",
        canvas_policy="all_masked",
        confidence_threshold=0.99,
        temperature=temperature,
        rng=jax.random.PRNGKey(7),
    )
    completion_tokens = trace.tokens[:, 1:]
    completion_steps = trace.action_steps[:, 1:]
    stop_id = int(completion_tokens[0, 0])
    policy_mask = diffusion_rl.policy_mask_at_stop(completion_tokens, completion_steps, (stop_id,))
    batch = diffusion_rl.prepare_diffusion_policy_batch(
        prompt_tokens=initial[:, :1],
        prompt_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        completion_tokens=completion_tokens,
        completion_mask=jnp.ones_like(completion_tokens, dtype=jnp.bool_),
        action_steps=completion_steps,
        loss_mask=policy_mask,
        inactive_target_id=0,
    )

    replayed = tunix_diffusion.compute_diffusion_per_token_logps(
        model,
        batch,
        diffusion_rl.make_diffusion_trace_logits_fn(_config()),
        temperature=temperature,
        stop_gradient=True,
    )

    np.testing.assert_array_equal(batch.model_inputs["action_steps"], completion_steps)
    np.testing.assert_array_equal(batch.target_ids, jnp.where(policy_mask, completion_tokens, 0))
    np.testing.assert_allclose(
        replayed,
        jnp.where(policy_mask, trace.action_logps[:, 1:], 0.0),
        rtol=1e-6,
        atol=1e-6,
    )

  def test_explicit_stop_ids_do_not_require_tokenizer_eos(self):
    config = _config(
        rl=SimpleNamespace(
            diffusion_stop_token_ids=[5, 6],
            diffusion_max_denoise_steps=-1,
            diffusion_confidence_threshold=0.99,
        )
    )

    self.assertEqual(diffusion_rl.resolve_stop_token_ids(config, None), (5, 6))

  def test_prompt_mask_comes_from_padding_length_not_token_value(self):
    rollout_config = SimpleNamespace(temperature=0.7)
    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=_CanvasDependentModel(),
        tokenizer=_PadValuedPromptTokenizer(),
        mesh=None,
        rollout_config=rollout_config,
        maxtext_config=_config(),
    )

    tokens, mask = rollout._encode_left_padded(["prompt"], 3)  # pylint: disable=protected-access

    np.testing.assert_array_equal(tokens, [[0, 0, 4]])
    np.testing.assert_array_equal(mask, [[0, 1, 1]])

  def test_rollout_exports_padded_trace_and_matching_sampler_logps(self):
    config = _config()
    model = _CanvasDependentModel()
    rollout_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
    )
    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=model,
        tokenizer=_Tokenizer(),
        mesh=None,
        rollout_config=rollout_config,
        maxtext_config=config,
    )

    output = rollout.generate(["prompt", "prompt"], rollout_config)
    batch = output.diffusion_batch
    self.assertIsNotNone(batch)
    self.assertIsInstance(batch.target_ids, np.ndarray)
    self.assertEqual(batch.target_ids.shape, (2, 3))
    np.testing.assert_array_equal(batch.loss_weights > 0, batch.model_inputs["action_steps"] >= 0)
    np.testing.assert_array_equal(batch.target_ids, batch.model_inputs["completion_tokens"])
    replayed = tunix_diffusion.compute_diffusion_per_token_logps(
        model,
        batch,
        diffusion_rl.make_diffusion_trace_logits_fn(config),
        temperature=rollout_config.temperature,
        stop_gradient=True,
    )
    np.testing.assert_allclose(replayed, np.stack(output.logprobs), rtol=1e-6, atol=1e-6)

  def test_rollout_scores_full_trace_while_reward_text_stops_at_eos(self):
    config = _config(
        rl=SimpleNamespace(
            diffusion_stop_token_ids=list(range(7)),
            diffusion_max_denoise_steps=-1,
            diffusion_confidence_threshold=0.99,
        )
    )
    training_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
    )
    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=_CanvasDependentModel(),
        tokenizer=_Tokenizer(),
        mesh=None,
        rollout_config=training_config,
        maxtext_config=config,
    )

    output = rollout.generate(["prompt"], training_config)

    self.assertLen(output.tokens[0], 1)
    self.assertLen(output.logprobs[0], 3)
    np.testing.assert_array_equal(output.diffusion_batch.loss_weights, np.ones((1, 3)))

  def test_greedy_eval_uses_its_own_sampling_contract(self):
    training_config = SimpleNamespace(temperature=0.7)
    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=_CanvasDependentModel(),
        tokenizer=_Tokenizer(),
        mesh=None,
        rollout_config=training_config,
        maxtext_config=_config(),
    )
    eval_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.01,
        return_logprobs=False,
        top_k=1,
        top_p=1.0,
    )

    first = rollout.generate(["prompt"], eval_config)
    second = rollout.generate(["prompt"], eval_config)

    np.testing.assert_array_equal(first.diffusion_batch.target_ids, second.diffusion_batch.target_ids)

  def test_rollout_fails_loudly_on_non_finite_policy_scores(self):
    rollout_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
    )
    rollout = diffusion_rl.MaxTextDiffusionRollout(
        rollout_actor=_NonFiniteModel(),
        tokenizer=_Tokenizer(),
        mesh=None,
        rollout_config=rollout_config,
        maxtext_config=_config(),
    )

    with self.assertRaisesRegex(RuntimeError, "unresolved mask|non-finite"):
      rollout.generate(["prompt"], rollout_config)

  def test_eval_rng_does_not_advance_training_stream(self):
    training_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
        seed=13,
    )
    eval_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.01,
        return_logprobs=False,
        top_k=1,
        top_p=1.0,
    )

    def make_rollout():
      return diffusion_rl.MaxTextDiffusionRollout(
          rollout_actor=_CanvasDependentModel(),
          tokenizer=_Tokenizer(),
          mesh=None,
          rollout_config=training_config,
          maxtext_config=_config(),
      )

    with_eval = make_rollout()
    baseline = make_rollout()
    with_eval.generate(["prompt"], eval_config)

    actual = with_eval.generate(["prompt"], training_config)
    expected = baseline.generate(["prompt"], training_config)

    np.testing.assert_array_equal(actual.diffusion_batch.target_ids, expected.diffusion_batch.target_ids)

  def test_generation_context_restores_training_rng_by_global_step(self):
    training_config = SimpleNamespace(
        max_prompt_length=2,
        max_tokens_to_generate=3,
        temperature=0.7,
        return_logprobs=True,
        top_k=-1,
        top_p=1.0,
        seed=13,
    )

    def make_rollout():
      return diffusion_rl.MaxTextDiffusionRollout(
          rollout_actor=_CanvasDependentModel(),
          tokenizer=_Tokenizer(),
          mesh=None,
          rollout_config=training_config,
          maxtext_config=_config(),
      )

    uninterrupted = make_rollout()
    resumed = make_rollout()
    uninterrupted.set_generation_context(global_step=7, mode="train")
    resumed.set_generation_context(global_step=7, mode="train")

    expected = uninterrupted.generate(["prompt"], training_config)
    actual = resumed.generate(["prompt"], training_config)

    np.testing.assert_array_equal(actual.diffusion_batch.target_ids, expected.diffusion_batch.target_ids)


if __name__ == "__main__":
  absltest.main()
