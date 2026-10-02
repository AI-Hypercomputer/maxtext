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

"""Tests of per-invocation TE EP capacity sizing without TE or GPU execution."""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from maxtext.utils import max_utils


class TeMoeBootstrapTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(
        te_moe_block=True,
        gradient_accumulation_steps=3,
        micro_batch_size_to_train_on=96,
        micro_batch_size_to_eval_on=96,
        eval_interval=-1,
        num_experts=128,
        num_experts_per_tok=8,
        ragged_buffer_factor=2.0,
        moe_expert_input_dim=0,
        emb_dim=7168,
        dtype=jnp.bfloat16,
    )
    self.mesh = mock.MagicMock(shape={"fsdp": 4, "expert": 4})
    ep = ModuleType("transformer_engine.jax.ep")
    moe = ModuleType("transformer_engine.jax.moe")
    self.bootstrap = ep.ep_bootstrap = mock.Mock()
    self.record = moe.record_ep_bootstrap_signature_for_moe = mock.Mock()
    self.capacity = moe.get_moe_recv_capacity_per_rank = mock.Mock(return_value=393216)
    self.enterContext(mock.patch.dict(sys.modules, {ep.__name__: ep, moe.__name__: moe}))
    self.enterContext(mock.patch.object(max_utils, "_te_moe_bootstrap_signature", None))
    self.enterContext(mock.patch.object(jax, "local_device_count", return_value=1))
    self.enterContext(mock.patch.object(jax, "process_count", return_value=16))
    self.enterContext(mock.patch.object(jax, "process_index", return_value=0))
    self.enterContext(mock.patch.object(jax, "set_mesh", return_value=nullcontext()))

  def bootstrap_batch(self, loaded_batch=288):
    batch = {"inputs": jax.ShapeDtypeStruct((loaded_batch, 4096), jnp.int32)}
    max_utils.maybe_bootstrap_te_moe(self.config, self.mesh, batch)

  def assert_token_bound(self, expected):
    self.assertEqual(self.bootstrap.call_args.kwargs["max_tokens_per_rank"], expected)
    self.assertEqual(self.record.call_args.kwargs["max_tokens_per_rank"], expected)
    self.assertEqual(self.capacity.call_count, 2)
    for call in self.capacity.call_args_list:
      self.assertEqual(call.kwargs["max_tokens_per_rank"], expected)

  def test_capacity_is_independent_of_ga(self):
    for ga in (1, 2, 3, 16, 32):
      with self.subTest(ga=ga), mock.patch.object(max_utils, "_te_moe_bootstrap_signature", None):
        self.capacity.reset_mock()
        self.config.gradient_accumulation_steps = ga
        self.bootstrap_batch(96 * ga)
        self.assert_token_bound(24576)
        self.assertEqual(self.bootstrap.call_args.kwargs["recv_capacity_per_rank"], 393216)
        self.assertEqual(self.record.call_args.kwargs["recv_capacity_per_rank"], 393216)
        self.assertEqual(max_utils.get_te_moe_recv_capacity_per_rank(), 393216)
        self.assertEqual(self.capacity.call_args.kwargs["recv_capacity_factor"], 2.0)

  def test_larger_eval_batch_is_covered(self):
    self.config.eval_interval = 10
    self.config.micro_batch_size_to_eval_on = 192
    self.bootstrap_batch()
    self.assert_token_bound(49152)

  def test_smaller_eval_batch_does_not_shrink_capacity(self):
    self.config.eval_interval = 10
    self.config.micro_batch_size_to_eval_on = 48
    self.bootstrap_batch()
    self.assert_token_bound(24576)

  def test_disabled_eval_does_not_inflate_capacity(self):
    self.config.micro_batch_size_to_eval_on = 192
    self.bootstrap_batch()
    self.assert_token_bound(24576)

  def test_expanded_loaded_microbatch_is_covered(self):
    self.bootstrap_batch(576)
    self.assert_token_bound(49152)

  def test_initialization_batch_is_covered_during_rampup(self):
    self.bootstrap_batch(144)
    self.assert_token_bound(24576)

  def test_indivisible_loaded_batch_raises(self):
    with self.assertRaisesRegex(ValueError, r"loaded batch size \(289\) must be divisible by GA steps \(3\)"):
      self.bootstrap_batch(289)
    self.bootstrap.assert_not_called()

  def test_unshardable_microbatch_raises(self):
    self.config.micro_batch_size_to_train_on = 97
    with self.assertRaisesRegex(ValueError, r"per-call batch size \(97\).*divisible by FSDP \* EP \(16\)"):
      self.bootstrap_batch(291)
    self.bootstrap.assert_not_called()

  def test_nonpositive_microbatch_raises(self):
    self.config.micro_batch_size_to_train_on = 0
    with self.assertRaisesRegex(ValueError, r"per-call batch size \(0\) must be positive.*FSDP \* EP \(16\)"):
      self.bootstrap_batch(0)
    self.bootstrap.assert_not_called()

  def test_repeated_bootstrap_is_cached(self):
    self.bootstrap_batch()
    self.bootstrap_batch()
    self.bootstrap.assert_called_once()

  def test_non_te_model_skips_bootstrap(self):
    self.config.te_moe_block = False
    self.bootstrap_batch()
    self.bootstrap.assert_not_called()


if __name__ == "__main__":
  unittest.main()
