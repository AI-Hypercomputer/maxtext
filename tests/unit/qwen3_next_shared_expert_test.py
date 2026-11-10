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
import jax.numpy as jnp
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

  def _block(self, shared_experts: int, **extra_kwargs) -> Qwen3NextSparseMoeBlock:
    """Build the block with the given `shared_experts` count."""
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
        capacity_factor=-1.0,
        max_target_length=8,
        per_device_batch_size=1,
        base_emb_dim=8,
        base_moe_mlp_dim=16,
        base_num_decoder_layers=1,
        num_experts=2,
        num_experts_per_tok=1,
        shared_experts=shared_experts,
        **extra_kwargs,
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

  def test_shared_expert_gate_and_combine_honor_float32_flags(self):
    """Verifies shared_expert_gate and combine honor float32_gate_logits while keeping default bf16 graphs unchanged."""
    # 1. Default bf16 config (float32_gate_logits=False, float32_weight_sum=True) stays completely in bf16.
    block_default = self._block(shared_experts=1)
    self.assertEqual(block_default.shared_expert_gate.dtype, jnp.bfloat16)
    self.assertEqual(block_default.shared_expert_gate.matmul_precision, "default")
    x_bf16 = jnp.ones((1, 4, 8), dtype=jnp.bfloat16)
    default_jaxpr = jax.make_jaxpr(lambda x: block_default(x, deterministic=True)[0])(x_bf16)
    final_adds_default = [
        eqn for eqn in default_jaxpr.jaxpr.eqns if eqn.primitive.name == "add" and eqn.outvars[0].aval.shape == (1, 4, 8)
    ]
    self.assertEqual(final_adds_default[-1].outvars[0].aval.dtype, jnp.bfloat16)

    # 2. float32_gate_logits=True + float32_weight_sum=True runs gate & combine in fp32 without changing weight_dtype.
    block_fp32 = self._block(
        shared_experts=1,
        float32_gate_logits=True,
        float32_weight_sum=True,
        matmul_precision="highest",
    )
    self.assertEqual(block_fp32.shared_expert_gate.dtype, jnp.float32)
    self.assertEqual(block_fp32.shared_expert_gate.matmul_precision, "highest")
    self.assertEqual(block_fp32.shared_expert_gate.kernel.get_value().dtype, jnp.bfloat16)

    block_fp32.shared_expert_gate.kernel.set_value(jnp.ones((8, 1), dtype=jnp.bfloat16))
    x_fp32 = jnp.ones((1, 4, 8), dtype=jnp.float32)
    x_perturbed = x_fp32.at[..., 0].add(jnp.float32(2.0**-10))
    self.assertTrue(bool(jnp.array_equal(x_fp32.astype(jnp.bfloat16), x_perturbed.astype(jnp.bfloat16))))
    self.assertFalse(
        bool(jnp.array_equal(block_fp32.shared_expert_gate(x_fp32), block_fp32.shared_expert_gate(x_perturbed)))
    )

    out_fp32_sum, _ = block_fp32(x_fp32, deterministic=True)
    self.assertEqual(out_fp32_sum.dtype, jnp.bfloat16)
    fp32_jaxpr = jax.make_jaxpr(lambda x: block_fp32(x, deterministic=True)[0])(x_fp32)
    final_adds_fp32 = [
        eqn for eqn in fp32_jaxpr.jaxpr.eqns if eqn.primitive.name == "add" and eqn.outvars[0].aval.shape == (1, 4, 8)
    ]
    self.assertEqual(final_adds_fp32[-1].outvars[0].aval.dtype, jnp.float32)

    # 3. float32_gate_logits=True + float32_weight_sum=False keeps sigmoid in fp32 and casts its scalar output to bf16.
    block_bf16_sum = self._block(shared_experts=1, float32_gate_logits=True, float32_weight_sum=False)
    out_bf16_sum, _ = block_bf16_sum(x_fp32, deterministic=True)
    self.assertEqual(out_bf16_sum.dtype, jnp.bfloat16)
    bf16_sum_jaxpr = jax.make_jaxpr(lambda x: block_bf16_sum(x, deterministic=True)[0])(x_fp32)
    gate_sigmoid_eqns = [
        eqn
        for eqn in bf16_sum_jaxpr.jaxpr.eqns
        if eqn.outvars
        and getattr(eqn.outvars[0].aval, "shape", None) == (1, 4, 1)
        and eqn.primitive.name != "dot_general"
    ]
    # The sigmoid output is float32, followed by convert_element_type -> bfloat16.
    self.assertEqual(gate_sigmoid_eqns[0].invars[0].aval.dtype, jnp.float32)
    self.assertEqual(gate_sigmoid_eqns[0].outvars[0].aval.dtype, jnp.float32)
    self.assertEqual(gate_sigmoid_eqns[-1].primitive.name, "convert_element_type")
    self.assertEqual(gate_sigmoid_eqns[-1].outvars[0].aval.dtype, jnp.bfloat16)


if __name__ == "__main__":
  unittest.main()
