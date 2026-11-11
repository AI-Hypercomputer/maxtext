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

"""Unit tests for the DeepSeek routed gate bias dtype."""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
import pytest
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only


class DeepSeekV4RoutedBiasTest(unittest.TestCase):
  """Tests for float32 routed gate bias precision."""

  def test_gate_logit_routed_bias_dtype(self):
    rngs = nnx.Rngs(0)
    gate_fp32 = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=8,
        model_name="deepseek4-284b",
        mesh=None,
        rngs=rngs,
        use_bias=True,
        dtype=jnp.float32,
        weight_dtype=jnp.bfloat16,
        bias_dtype=jnp.float32,
    )
    self.assertIsInstance(gate_fp32.bias, moe.MoEBiasVar)
    self.assertEqual(gate_fp32.bias.get_value().dtype, jnp.float32)

    gate_bf16 = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=8,
        model_name="deepseek4-284b",
        mesh=None,
        rngs=rngs,
        use_bias=True,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
    )
    self.assertIsInstance(gate_bf16.bias, moe.MoEBiasVar)
    self.assertEqual(gate_bf16.bias.get_value().dtype, jnp.bfloat16)

  def test_routed_moe_wires_float32_gate_logits_to_bias_dtype(self):
    base_args = [
        None,
        get_test_config_path(),
        "model_name=deepseek4-284b",
        "override_model_config=True",
        "base_emb_dim=16",
        "base_mlp_dim=16",
        "base_moe_mlp_dim=16",
        "num_experts=4",
        "num_experts_per_tok=2",
        "routed_bias=True",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
    ]
    cfg_fp32 = pyconfig.initialize(base_args + ["float32_gate_logits=True"])
    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        maxtext_utils.create_device_mesh(cfg_fp32, devices),
        cfg_fp32.mesh_axes,
    )
    routed_fp32 = moe.RoutedMoE(
        config=cfg_fp32,
        num_experts=cfg_fp32.num_experts,
        num_experts_per_tok=cfg_fp32.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg_fp32.moe_mlp_dim,
        dtype=cfg_fp32.dtype,
        weight_dtype=cfg_fp32.weight_dtype,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(routed_fp32.gate.bias.get_value().dtype, jnp.float32)
    self.assertEqual(routed_fp32.gate.matmul_precision, "default")

    cfg_bf16 = pyconfig.initialize(base_args + ["float32_gate_logits=False"])
    routed_bf16 = moe.RoutedMoE(
        config=cfg_bf16,
        num_experts=cfg_bf16.num_experts,
        num_experts_per_tok=cfg_bf16.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg_bf16.moe_mlp_dim,
        dtype=cfg_bf16.dtype,
        weight_dtype=cfg_bf16.weight_dtype,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(routed_bf16.gate.bias.get_value().dtype, jnp.bfloat16)

  def test_routed_moe_preserves_fp32_routing_inputs_and_respects_matmul_precision(self):
    """Verifies RoutedMoE routes in fp32 before casting inputs to cfg.dtype while honoring config.matmul_precision."""
    x_base = jnp.ones((1, 2, 16), dtype=jnp.float32)
    base_args = [
        None,
        get_test_config_path(),
        "model_name=qwen3-30b-a3b",
        "override_model_config=True",
        "base_emb_dim=16",
        "base_mlp_dim=16",
        "base_moe_mlp_dim=16",
        "num_experts=4",
        "num_experts_per_tok=2",
        "megablox=False",
        "sparse_matmul=False",
        "capacity_factor=-1.0",
        "load_balance_loss_weight=0.01",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "float32_gate_logits=True",
        "matmul_precision=highest",
    ]
    cfg = pyconfig.initialize(base_args)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg, jax.devices()), cfg.mesh_axes)
    routed = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(routed.gate.matmul_precision, "highest")
    # Map features 0..3 to experts 0..3 so a sub-bf16-ULP delta on feature 2 breaks the top-2 tie in favor of expert 2.
    routed.gate.kernel.set_value(jnp.eye(16, 4, dtype=jnp.bfloat16))
    # 1.0 and 1.0 + 2**-10 collapse to identical bfloat16 values (bf16 ULP at 1.0 is 2**-7),
    # so expert selection and MoE output differ iff RoutedMoE.__call__ routes in fp32 before casting `inputs` to bf16.
    x_asym = x_base.at[..., 2].add(jnp.float32(2.0**-10))
    self.assertTrue(bool(jnp.array_equal(x_base.astype(jnp.bfloat16), x_asym.astype(jnp.bfloat16))))
    logits_base, _ = routed.gate(x_base)
    logits_asym, _ = routed.gate(x_asym)
    self.assertFalse(bool(jnp.array_equal(logits_base, logits_asym)))

    out_base, _, _ = routed(x_base)
    out_asym, _, _ = routed(x_asym)
    self.assertFalse(bool(jnp.array_equal(out_base, out_asym)))

    # Trace RoutedMoE.__call__ on float32 inputs to confirm the gate dot_general gets float32 inputs at HIGHEST precision.
    moe_jaxpr = jax.make_jaxpr(lambda x: routed(x)[0])(x_base)
    first_dot = next(eqn for eqn in moe_jaxpr.jaxpr.eqns if eqn.primitive.name == "dot_general")
    self.assertEqual(first_dot.invars[0].aval.dtype, jnp.float32)
    self.assertIn("HIGHEST", str(first_dot.params.get("precision")))


if __name__ == "__main__":
  unittest.main()
