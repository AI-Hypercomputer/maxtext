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

"""Unit tests for `grpo_loss_fn_nnx`, `compute_log_probs_nnx`, plus a small
Linen-path regression block (the repo's existing Linen GRPO integration test
is TPU-only)."""

import types
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.experimental.rl import grpo_trainer
from maxtext.experimental.rl import grpo_utils


class _MockTransformer(nnx.Module):
  """Tiny NNX module that responds to the kwargs `compute_log_probs_nnx` uses."""

  def __init__(self, vocab_size: int, embed_dim: int, rngs: nnx.Rngs):
    self.embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
    self.proj = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions=None,
      decoder_segment_ids=None,
      enable_dropout=False,
      **kwargs,
  ):
    del decoder_positions, decoder_segment_ids, enable_dropout, kwargs
    return self.proj(self.embed(decoder_input_tokens))


def _make_grpo_config(**overrides):
  """Minimal config namespace covering every field `grpo_loss_fn_nnx` reads."""
  base = {
      "train_data_columns": "prompt",
      "num_generations": 2,
      "grpo_epsilon": 0.2,
      "grpo_beta": 0.1,
      "num_experts": 1,
      "decode_sampling_temperature": 1.0,
      "enable_dropout": False,
      "use_dpo": False,
  }
  base.update(overrides)
  return types.SimpleNamespace(**base)


def _make_grpo_batch(B=2, G=2, S=6):
  """Minimal GRPO batch: `B` prompts, `G` generations each (total `B*G`), seq length `S`."""
  total = B * G
  prompts = jnp.tile(jnp.arange(S, dtype=jnp.int32), (total, 1))
  return {
      "prompt_completions": prompts,
      "prompt_completions_position": prompts,
      "prompt_completions_segmentation": jnp.ones((total, S), dtype=jnp.int32),
      "ar_completions_segmentation": jnp.array([[0, 0, 1, 1, 1, 0]] * total, dtype=jnp.int32),
      "completions_logprobs": None,  # off-policy
  }


class TestGrpoLossFnNnx(unittest.TestCase):
  """Behavior of `grpo_loss_fn_nnx` on a synthetic policy / reference pair."""

  def setUp(self):
    self.policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    self.reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))  # identical seed
    self.config = _make_grpo_config()
    self.data = _make_grpo_batch()

  def test_aux_structure_matches_linen(self):
    """`grpo_loss_fn_nnx` returns the same `LossAux` dataclass shape as `grpo_loss_fn`."""
    loss, aux = grpo_trainer.grpo_loss_fn_nnx(
        self.policy, self.config, self.data, None, None, self.reference, is_train=True
    )
    self.assertIsInstance(aux, grpo_trainer.LossAux)
    for field in (
        "total_loss",
        "avg_reward",
        "avg_reward_std",
        "avg_advantage",
        "completion_length",
        "moe_lb_loss",
        "total_weights",
    ):
      self.assertTrue(hasattr(aux, field), f"aux missing field {field}")
    self.assertTrue(jnp.isfinite(loss))

  def test_unused_dropout_rng_and_params_args_are_ignored(self):
    """`dropout_rng` and `params` are positional placeholders only — values shouldn't matter."""
    a = grpo_trainer.grpo_loss_fn_nnx(self.policy, self.config, self.data, None, None, self.reference, is_train=True)
    b = grpo_trainer.grpo_loss_fn_nnx(
        self.policy, self.config, self.data, jax.random.key(99), {"junk": 1}, self.reference, is_train=True
    )
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), rtol=1e-6)

  def test_identical_policy_and_reference_zero_kl(self):
    """Identical policy and reference → per-token KL is zero, so `aux.avg_kl ≈ 0`."""
    cfg = _make_grpo_config(grpo_beta=0.5)
    _, aux = grpo_trainer.grpo_loss_fn_nnx(self.policy, cfg, self.data, None, None, self.reference, is_train=True)
    self.assertIsNotNone(aux.avg_kl)
    np.testing.assert_allclose(np.asarray(aux.avg_kl), 0.0, atol=1e-5)

  def test_grpo_beta_zero_avg_kl_is_none(self):
    cfg = _make_grpo_config(grpo_beta=0.0)
    _, aux = grpo_trainer.grpo_loss_fn_nnx(self.policy, cfg, self.data, None, None, self.reference, is_train=True)
    self.assertIsNone(aux.avg_kl)

  def test_value_and_grad_flows_only_to_policy(self):
    """`nnx.value_and_grad` over the policy yields finite grads; reference is left alone."""

    def loss_only(policy_model):
      loss, _ = grpo_trainer.grpo_loss_fn_nnx(
          policy_model, self.config, self.data, None, None, self.reference, is_train=True
      )
      return loss

    # nnx.value_and_grad returns (value, grad_state) where grad_state holds nnx.Param leaves.
    _, grads = nnx.value_and_grad(loss_only, argnums=0)(self.policy)
    leaves = jax.tree_util.tree_leaves(grads)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      self.assertTrue(np.all(np.isfinite(np.asarray(leaf))), "policy grad has non-finite entries")


class TestComputeLogProbsNnx(unittest.TestCase):
  """Shape contract of `compute_log_probs_nnx`."""

  def test_returns_correct_shape(self):
    config = _make_grpo_config()
    data = _make_grpo_batch()
    model = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    log_probs, _ = grpo_utils.compute_log_probs_nnx(
        model,
        data["prompt_completions"],
        data["prompt_completions_position"],
        data["prompt_completions_segmentation"],
        data["ar_completions_segmentation"],
        config,
        is_train=False,
    )
    # Inputs are [B, S] → log_probs are [B, S-1].
    self.assertEqual(log_probs.shape, (data["prompt_completions"].shape[0], data["prompt_completions"].shape[1] - 1))


# ---------------------------------------------------------------------------
# Linen-path regression smoke tests
# ---------------------------------------------------------------------------


class _MockLinenTransformer(nn.Module):
  """Tiny Linen module that responds to the same `model.apply(...)` shape Linen `compute_log_probs` uses."""

  vocab_size: int
  embed_dim: int

  @nn.compact
  def __call__(self, inputs, positions, decoder_segment_ids=None, enable_dropout=False):
    del positions, decoder_segment_ids, enable_dropout
    embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="embed")(inputs)
    return nn.Dense(features=self.vocab_size, name="proj")(embed)


class TestLinenGrpoRegression(unittest.TestCase):
  """Smoke test that the Linen `grpo_loss_fn` and `compute_log_probs` still run
  end-to-end with `pure_nnx=False`-style inputs."""

  def setUp(self):
    self.config = _make_grpo_config()
    self.config.pure_nnx = False  # explicit Linen mode
    self.config.gradient_accumulation_steps = 1
    self.data = _make_grpo_batch()
    self.model = _MockLinenTransformer(vocab_size=8, embed_dim=4)
    rng = jax.random.key(0)
    inputs = self.data["prompt_completions"]
    self.params = self.model.init(rng, inputs, inputs, decoder_segment_ids=jnp.ones_like(inputs), enable_dropout=False)
    self.reference_params = jax.tree_util.tree_map(jnp.copy, self.params)

  def test_linen_grpo_loss_fn_still_runs(self):
    """Linen `grpo_loss_fn` returns a finite loss + a `LossAux`."""
    loss, aux = grpo_trainer.grpo_loss_fn(
        self.model,
        self.config,
        self.data,
        jax.random.key(1),
        self.params,
        self.reference_params["params"],  # Linen reference_params is the inner subtree
        is_train=True,
    )
    self.assertTrue(jnp.isfinite(loss))
    self.assertTrue(hasattr(aux, "total_loss"))
    self.assertTrue(hasattr(aux, "moe_lb_loss"))
    self.assertTrue(hasattr(aux, "total_weights"))

  def test_linen_compute_log_probs_still_runs(self):
    """Linen `compute_log_probs` produces shape `[B, S-1]`."""
    log_probs, _ = grpo_utils.compute_log_probs(
        self.model,
        self.params,
        self.data["prompt_completions"],
        self.data["prompt_completions_position"],
        self.data["prompt_completions_segmentation"],
        self.data["ar_completions_segmentation"],
        self.config,
        is_train=False,
        rngs={"dropout": jax.random.key(2), "params": jax.random.key(3)},
    )
    S = self.data["prompt_completions"].shape[1]
    self.assertEqual(log_probs.shape, (self.data["prompt_completions"].shape[0], S - 1))


if __name__ == "__main__":
  unittest.main()
