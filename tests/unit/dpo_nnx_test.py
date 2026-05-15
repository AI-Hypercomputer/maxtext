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

"""NNX DPO unit tests.

Covers the NNX-native DPO surface:
  * `TrainStateNNX(model, optimizer, reference_model=...)` — reference model
    sits alongside policy and is not touched by `apply_gradients`.
  * `dpo_loss_fn_nnx(policy, config, data, None, None, reference, is_train)` —
    aux structure, identical-model invariant (loss = log(2), reward_accuracy = 0.5).
"""

import math
import types
import unittest

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from maxtext.layers import train_state_nnx
from maxtext.trainers.post_train.dpo import dpo_utils


class _MockTransformer(nnx.Module):
  """Tiny NNX transformer-shaped module for DPO tests.

  Accepts the same keyword args that `dpo_loss_fn_nnx` passes:
  `decoder_input_tokens`, `decoder_positions`, `decoder_segment_ids`,
  `enable_dropout`. Other args are tolerated via **kwargs.
  """

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


def _make_dpo_config(**overrides):
  """Build the minimal config surface that `dpo_loss_fn_nnx` reads."""
  base = {
      "dpo_label_smoothing": 0.0,
      "dpo_beta": 0.1,
      "enable_dropout": False,
      "num_experts": 1,
      "micro_batch_size_to_train_on": 2,
  }
  base.update(overrides)
  return types.SimpleNamespace(**base)


def _make_dpo_batch(batch_size=2, seq_len=5):
  """Build a tiny DPO-shaped batch.

  `chosen` and `rejected` share the first 2 tokens (common prefix is masked
  out in the loss), differ at positions 2 and 3, and are padded at position 4.
  """
  chosen = jnp.array([[1, 2, 3, 4, 0]] * batch_size, dtype=jnp.int32)
  rejected = jnp.array([[1, 2, 5, 6, 0]] * batch_size, dtype=jnp.int32)
  positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))
  segmentation = jnp.array([[1, 1, 1, 1, 0]] * batch_size, dtype=jnp.int32)
  return {
      "chosen": chosen,
      "rejected": rejected,
      "chosen_position": positions,
      "rejected_position": positions,
      "chosen_segmentation": segmentation,
      "rejected_segmentation": segmentation,
  }


class TestTrainStateNNXWithReferenceModel(unittest.TestCase):
  """`TrainStateNNX(reference_model=...)` semantics."""

  def setUp(self):
    self.policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    self.reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(1))
    self.tx = optax.adam(1e-3)

  def test_init_with_reference(self):
    optimizer = nnx.Optimizer(self.policy, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.policy, optimizer, reference_model=self.reference)
    self.assertIs(state.model, self.policy)
    self.assertIs(state.reference_model, self.reference)
    self.assertEqual(state.optimizer.step.value, 0)

  def test_init_without_reference_omits_attribute(self):
    optimizer = nnx.Optimizer(self.policy, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.policy, optimizer)
    self.assertFalse(hasattr(state, "reference_model"))

  def test_apply_gradients_does_not_touch_reference(self):
    """Gradient update on policy must leave reference model bit-identical."""
    optimizer = nnx.Optimizer(self.policy, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.policy, optimizer, reference_model=self.reference)

    ref_kernel_before = jnp.asarray(state.reference_model.proj.kernel.value).copy()

    def policy_loss(m):
      return jnp.mean(m(jnp.array([[1, 2]])) ** 2)

    grads = nnx.grad(policy_loss)(state.model)
    state.apply_gradients(grads)

    ref_kernel_after = jnp.asarray(state.reference_model.proj.kernel.value)
    self.assertTrue(jnp.array_equal(ref_kernel_before, ref_kernel_after))


class TestDPOLossFnNNX(unittest.TestCase):
  """`dpo_loss_fn_nnx` numerical and structural sanity checks."""

  def setUp(self):
    self.policy = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    # Reference initialized with the same seed to make policy and reference
    # bit-identical at construction time.
    self.reference = _MockTransformer(vocab_size=8, embed_dim=4, rngs=nnx.Rngs(0))
    self.config = _make_dpo_config()
    self.data = _make_dpo_batch()

  def test_aux_has_expected_keys(self):
    _, aux = dpo_utils.dpo_loss_fn_nnx(
        self.policy, self.config, dict(self.data), None, None, self.reference, is_train=True
    )
    expected_keys = {
        "intermediate_outputs",
        "xent_sum",
        "dpo_loss",
        "total_weights",
        "moe_lb_loss",
        "reward_accuracy",
        "indexer_loss",
        "mtp_loss",
    }
    self.assertEqual(set(aux.keys()), expected_keys)
    self.assertEqual(aux["xent_sum"], 0.0)
    self.assertEqual(aux["moe_lb_loss"], 0.0)  # num_experts=1
    self.assertEqual(aux["total_weights"], self.data["chosen"].shape[0])

  def test_identical_policy_and_reference_yields_log2_loss(self):
    """When policy == reference, all logratios are 0; with label_smoothing=0
    the per-example loss is `-log(sigmoid(0)) = log(2)`. `reward_accuracy`
    uses strict `chosen > rejected`, so equal logratios score 0.0 (no example
    is strictly preferred).
    """
    loss, aux = dpo_utils.dpo_loss_fn_nnx(
        self.policy, self.config, dict(self.data), None, None, self.reference, is_train=True
    )
    self.assertAlmostEqual(float(loss), math.log(2.0), places=4)
    self.assertAlmostEqual(float(aux["dpo_loss"]), math.log(2.0), places=4)
    self.assertAlmostEqual(float(aux["reward_accuracy"]), 0.0, places=4)

  def test_dropout_rng_and_params_args_are_unused(self):
    """The 4th and 5th positional args are signature-compat slots for the
    Linen dispatcher; passing arbitrary values must not affect the result.
    """
    loss_a, _ = dpo_utils.dpo_loss_fn_nnx(
        self.policy, self.config, dict(self.data), None, None, self.reference, is_train=True
    )
    loss_b, _ = dpo_utils.dpo_loss_fn_nnx(
        self.policy,
        self.config,
        dict(self.data),
        jax.random.PRNGKey(123),  # dropout_rng — unused
        {"params": "garbage"},  # params — unused
        self.reference,
        is_train=True,
    )
    self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)

  def test_value_and_grad_argnums0_only_diffs_policy(self):
    """`nnx.value_and_grad(..., argnums=0)` over the policy should produce
    finite grads on policy params and not require reference grads.
    """

    def _loss(policy_module):
      loss, _ = dpo_utils.dpo_loss_fn_nnx(
          policy_module, self.config, dict(self.data), None, None, self.reference, is_train=True
      )
      return loss

    grad_fn = nnx.value_and_grad(_loss, argnums=0)
    loss, grads = grad_fn(self.policy)
    self.assertTrue(jnp.isfinite(loss))
    # Grads is an nnx.State of the policy's nnx.Param leaves; check at least one
    # leaf is finite and non-trivially shaped.
    leaves = jax.tree_util.tree_leaves(grads)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      self.assertTrue(jnp.all(jnp.isfinite(leaf)))


if __name__ == "__main__":
  unittest.main()
