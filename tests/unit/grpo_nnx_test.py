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

"""Unit tests for `grpo_loss_fn_nnx` and `compute_log_probs_nnx`."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from maxtext.common import train_state_nnx
from maxtext.experimental.rl import grpo_trainer
from maxtext.experimental.rl import grpo_utils
from maxtext.utils import maxtext_utils_nnx


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
    # Use the same seed so the reference starts identical to the policy.
    self.reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    self.config = _make_grpo_config()
    self.data = _make_grpo_batch()

  def test_aux_structure_matches_linen(self):
    """`grpo_loss_fn_nnx` returns a `LossAux` dataclass with the expected fields."""
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
    """`dropout_rng` and `params` are positional placeholders, so their values do not affect the loss."""
    a = grpo_trainer.grpo_loss_fn_nnx(self.policy, self.config, self.data, None, None, self.reference, is_train=True)
    b = grpo_trainer.grpo_loss_fn_nnx(
        self.policy, self.config, self.data, jax.random.key(99), {"junk": 1}, self.reference, is_train=True
    )
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), rtol=1e-6)

  def test_identical_policy_and_reference_zero_kl(self):
    """When the policy and reference are identical, the per-token KL is zero."""
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
    # Inputs are [B, S] and log_probs are [B, S - 1].
    self.assertEqual(log_probs.shape, (data["prompt_completions"].shape[0], data["prompt_completions"].shape[1] - 1))


class TestGrpoTrainStepNnxGradAccum(unittest.TestCase):
  """Gradient accumulation on the NNX GRPO step must match a single full-batch step."""

  def _step(self, ga_steps):
    """Run one `_train_step_nnx` from a fixed init; return (loss, updated policy params)."""
    # pylint: disable=protected-access  # the test drives the internal _train_step_nnx
    policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    # Different seed from the policy so KL(policy||reference) != 0 and the step
    # produces a real (non-zero) gradient — otherwise the equivalence check is vacuous.
    reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(policy, optax.sgd(0.1), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(policy, optimizer)
    state.reference_model = reference
    graphdef, flat_state = nnx.split(state)
    config = _make_grpo_config(
        gradient_accumulation_steps=ga_steps,
        gradient_clipping_threshold=0.0,
        optimizer_memory_host_offload=False,
    )
    # B=2, G=2 -> 4 rows; GA=2 splits into 2 microbatches that each hold one
    # complete generation-group, so per-group advantages are unchanged and the
    # accumulated gradient must equal the full-batch gradient.
    data = _make_grpo_batch(B=2, G=2, S=6)
    new_flat, metrics = grpo_trainer._train_step_nnx(graphdef, config, None, flat_state, data)
    updated = nnx.merge(graphdef, new_flat)
    params = jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(updated.model, nnx.Param)))
    return float(metrics["scalar"]["learning/loss"]), params

  def test_gradient_accumulation_matches_single_shot(self):
    loss_full, params_full = self._step(ga_steps=1)
    loss_ga, params_ga = self._step(ga_steps=2)

    # Guard against a trivial pass: the step must actually move the params.
    init = jax.tree_util.tree_leaves(
        nnx.to_pure_dict(nnx.state(_MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0)), nnx.Param))
    )
    moved = any(not np.allclose(np.asarray(p), np.asarray(i)) for p, i in zip(params_full, init))
    self.assertTrue(moved, "params did not change — test would be trivially true")

    # GA=2 must reproduce the full-batch step's loss and resulting parameters.
    np.testing.assert_allclose(loss_ga, loss_full, rtol=1e-5, atol=1e-5)
    self.assertEqual(len(params_full), len(params_ga))
    # GA reorders the per-microbatch gradient summation, so on lower-precision hardware
    # (TPU bf16 matmuls) the updated params differ from the full-batch step by
    # accumulation rounding (~1e-6 absolute); fp32/CPU matches to ~1e-10. A real GA bug
    # (wrong normalization) would be a gross mismatch, so this tolerance still catches it.
    for pf, pg in zip(params_full, params_ga):
      np.testing.assert_allclose(np.asarray(pg), np.asarray(pf), rtol=1e-2, atol=1e-5)


class TestPathwaysReshardNnxScanLayersFalse(unittest.TestCase):
  """scan_layers=False must no longer raise; the unscanned policy params reshard to the engine.

  The actual `reshard_pytree` runs through pathwaysutils (Pathways infra, not available off-cluster),
  so it's mocked to a pass-through — the test pins *our* change: the guard is gone and the policy
  params are pushed to the inference engine.
  """

  def test_unscanned_policy_pushes_params_to_engine(self):
    captured = {}

    class _Engine:

      def update_params(self, params):
        captured["params"] = params

    original_reshard = grpo_utils.reshard_pytree
    grpo_utils.reshard_pytree = lambda source, target, **kw: source  # pass-through (skip Pathways)
    try:
      policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
      shardings_model = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
      cfg = _make_grpo_config(scan_layers=False)
      grpo_utils.pathways_reshard_nnx(cfg, _Engine(), policy, shardings_model, shardings_model)
    finally:
      grpo_utils.reshard_pytree = original_reshard

    self.assertIn("params", captured)  # no NotImplementedError; engine received params
    pushed = jax.tree_util.tree_leaves(captured["params"])
    expected = jax.tree_util.tree_leaves(nnx.state(_MockTransformer(8, 4, nnx.Rngs(0)), nnx.Param))
    self.assertEqual(len(pushed), len(expected))


class TestGrpoHostOffloadNnx(unittest.TestCase):
  """optimizer_memory_host_offload must run and not change the math (only memory placement).

  The memory-kind move needs TPU host-offload, so `move_memory_to_device` is mocked to identity;
  the test verifies the surrounding plumbing (extract opt_state, device_put, nnx.update, then
  apply_gradients) runs and yields the same params as the no-offload step.
  """

  def _step(self, host_offload):
    """Run one GRPO `_train_step_nnx` with/without host-offload; return (loss, policy params)."""
    # pylint: disable=protected-access  # the test drives the internal _train_step_nnx
    policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(policy, optax.sgd(0.1), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(policy, optimizer)
    state.reference_model = reference
    graphdef, flat_state = nnx.split(state)
    cfg = _make_grpo_config(
        gradient_accumulation_steps=1, gradient_clipping_threshold=0.0, optimizer_memory_host_offload=host_offload
    )
    sms = None
    original = maxtext_utils_nnx.move_memory_to_device
    if host_offload:
      mesh = jax.make_mesh((1,), ("x",))
      replicated = jax.tree.map(
          lambda _: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), nnx.state(state.optimizer)
      )
      sms = types.SimpleNamespace(optimizer=replicated)
      maxtext_utils_nnx.move_memory_to_device = lambda path, x: x  # identity (skip TPU memory kinds)
    try:
      new_flat, metrics = grpo_trainer._train_step_nnx(graphdef, cfg, sms, flat_state, _make_grpo_batch(B=2, G=2, S=6))
    finally:
      maxtext_utils_nnx.move_memory_to_device = original
    params = jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(nnx.merge(graphdef, new_flat).model, nnx.Param)))
    return float(metrics["scalar"]["learning/loss"]), params

  def test_host_offload_matches_no_offload(self):
    loss_off, params_off = self._step(host_offload=False)
    loss_on, params_on = self._step(host_offload=True)
    np.testing.assert_allclose(loss_on, loss_off, rtol=1e-6, atol=1e-6)
    for a, b in zip(params_on, params_off):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
