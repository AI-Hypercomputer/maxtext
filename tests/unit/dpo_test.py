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

"""Linen DPO unit tests.

Covers dpo_loss_fn:
  * identical-model invariant (loss = log(2)).
  * gradient-accumulation contract: the loss is the unnormalized sum under manual
    GA, so the accumulator's divide-by-total_weights recovers the mean gradient.
"""

import math
import types
import unittest

import jax
import jax.numpy as jnp
import flax.linen as nn

from maxtext.trainers.post_train.dpo import dpo_utils


class _TinyLinen(nn.Module):
  """Tiny Linen model matching the call signature dpo_loss_fn uses.

  dpo_loss_fn calls `model.apply(params, inputs, inputs_position,
  decoder_segment_ids=..., enable_dropout=..., mutable="intermediates")` and
  expects logits of shape [batch, seq_len, vocab_size]. No intermediates are
  sown; the mutable collection comes back empty, which is fine for num_experts=1.
  """

  vocab_size: int
  hidden: int

  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions=None,
      decoder_segment_ids=None,
      enable_dropout=False,
      **kwargs,
  ):
    del decoder_positions, decoder_segment_ids, enable_dropout, kwargs
    h = nn.Embed(self.vocab_size, self.hidden)(decoder_input_tokens)
    return nn.Dense(self.vocab_size)(h)


def _make_dpo_config(**overrides):
  """Minimal config surface read by dpo_loss_fn."""
  base = {
      "dpo_label_smoothing": 0.0,
      "dpo_beta": 0.1,
      "enable_dropout": False,
      "num_experts": 1,
      "micro_batch_size_to_train_on": 4,
      "gradient_accumulation_steps": 1,
      "use_tunix_gradient_accumulation": False,
  }
  base.update(overrides)
  return types.SimpleNamespace(**base)


def _make_dpo_batch(batch_size=4, seq_len=5):
  """DPO-shaped batch: chosen/rejected share a prefix, differ mid-sequence, pad at the end.

  dpo_loss_fn mutates its data dict in place, so build a fresh one per call.
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


class TestDPOLossFn(unittest.TestCase):
  """Linen dpo_loss_fn numerical and gradient-scaling checks."""

  def setUp(self):
    self.model = _TinyLinen(vocab_size=8, hidden=4)
    variables = self.model.init(jax.random.PRNGKey(0), jnp.ones((2, 5), dtype=jnp.int32))
    self.params = variables
    # Reference uses the same params, so policy and reference are bit-identical:
    # all logratios are 0 and every per-example loss is -log(sigmoid(0)) = log(2).
    self.reference_params = variables["params"]
    self.rng = jax.random.PRNGKey(0)

  def test_identical_policy_and_reference_yields_log2_loss(self):
    loss, aux = dpo_utils.dpo_loss_fn(
        self.model, _make_dpo_config(), _make_dpo_batch(), self.rng, self.params, self.reference_params, is_train=True
    )
    self.assertAlmostEqual(float(loss), math.log(2.0), places=4)
    self.assertAlmostEqual(float(aux["dpo_loss"]), math.log(2.0), places=4)
    self.assertEqual(float(aux["xent_sum"]), 0.0)
    self.assertEqual(int(aux["total_weights"]), 4)

  def test_gradient_accumulation_returns_unnormalized_sum(self):
    """Under manual gradient accumulation the loss must be the unnormalized sum
    (= mean * batch). gradient_accumulation_loss_and_grad sums per-microbatch
    grads then divides once by total_weights, so returning the pre-normalized
    mean here would under-scale the gradient by the microbatch size.
    """
    loss_mean, aux = dpo_utils.dpo_loss_fn(
        self.model,
        _make_dpo_config(gradient_accumulation_steps=1),
        _make_dpo_batch(),
        self.rng,
        self.params,
        self.reference_params,
        is_train=True,
    )
    loss_sum, _ = dpo_utils.dpo_loss_fn(
        self.model,
        _make_dpo_config(gradient_accumulation_steps=2),
        _make_dpo_batch(),
        self.rng,
        self.params,
        self.reference_params,
        is_train=True,
    )
    batch = aux["total_weights"]
    self.assertAlmostEqual(float(loss_sum), float(loss_mean) * batch, places=4)
    # The reported metric (aux["dpo_loss"]) stays the mean regardless of GA.
    self.assertAlmostEqual(float(aux["dpo_loss"]), float(loss_mean), places=6)


if __name__ == "__main__":
  unittest.main()
