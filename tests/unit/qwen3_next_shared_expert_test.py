# Copyright 2026 Google LLC
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
"""The Qwen3-Next sparse MoE block builds a shared expert only when asked."""

import unittest

from flax import nnx
import jax
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models.qwen3 import Qwen3NextSparseMoeBlock
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _count_params(module) -> int:
  return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))


class Qwen3NextSharedExpertTest(unittest.TestCase):
  """`shared_experts` defaults to 0, so building the expert unconditionally put a
  second full-size MLP in every layer of a dense configuration."""

  def _block(self, shared_experts: int) -> Qwen3NextSparseMoeBlock:
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="qwen3_next_shared_expert_test",
        enable_checkpointing=False,
        override_model_config=True,
        model_name="qwen3-next-80b-a3b",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=8,
        per_device_batch_size=1,
        base_emb_dim=8,
        base_moe_mlp_dim=16,
        base_num_decoder_layers=1,
        num_experts=2,
        num_experts_per_tok=1,
        shared_experts=shared_experts,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    return Qwen3NextSparseMoeBlock(config=config, mesh=mesh, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))

  def test_no_shared_expert_when_zero(self):
    block = self._block(shared_experts=0)
    self.assertFalse(block.use_shared_expert)
    self.assertIsNone(block.shared_expert)
    self.assertIsNone(block.shared_expert_gate)

  def test_shared_expert_when_one(self):
    block = self._block(shared_experts=1)
    self.assertTrue(block.use_shared_expert)
    self.assertIsNotNone(block.shared_expert)
    self.assertIsNotNone(block.shared_expert_gate)

  def test_the_only_difference_is_the_shared_expert(self):
    """The parameter count with the flag off is short by the shared expert and
    its gate, and by nothing else."""
    with_expert = self._block(shared_experts=1)
    without = self._block(shared_experts=0)
    expert_params = _count_params(with_expert.shared_expert) + _count_params(with_expert.shared_expert_gate)
    self.assertGreater(expert_params, 0)
    self.assertEqual(_count_params(with_expert) - _count_params(without), expert_params)


if __name__ == "__main__":
  unittest.main()
