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
        "src/maxtext/configs/base.yml",
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


if __name__ == "__main__":
  unittest.main()
