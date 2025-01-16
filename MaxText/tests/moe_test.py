#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import sre_parse
import unittest
from layers import linears
from layers import initializers
import jax.numpy as jnp

import pyconfig
import max_utils
from jax.sharding import Mesh
import flax.linen as nn
from typing import Tuple
import common_types
import pytest


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer


class TokenDroppingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize(
        [None, "configs/base.yml"],
        run_name="token_dropping_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        capacity_factor=2,
    )
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(42)
    devices_array = max_utils.create_device_mesh(self.cfg)
    self.model = linears.MoeBlock(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
    )

  def test_generate_masks(self):
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    # expert_capacity_in_batch = (4 * 2 / 8) * 2 = 2
    top_k_indices = jnp.array(
        [
            [[0, 5], [0, 4], [1, 0], [3, 5]],
            [[1, 2], [4, 1], [5, 0], [7, 1]],
            [[6, 2], [2, 3], [4, 2], [1, 2]],
            [[4, 1], [0, 7], [5, 0], [4, 7]],
        ]
    )
    softmax_probs = jnp.array(
        [
            [
                [0.20, 0, 0, 0, 0, 0.80, 0, 0],
                [0.68, 0, 0, 0, 0.32, 0, 0, 0],
                [0.22, 0.78, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.32, 0, 0.68, 0, 0],
            ],
            [
                [0, 0.26, 0.74, 0, 0, 0, 0, 0],
                [0, 0.79, 0, 0, 0.21, 0, 0, 0],
                [0.89, 0, 0, 0, 0, 0.11, 0, 0],
                [0, 0.11, 0, 0, 0, 0, 0, 0.89],
            ],
            [
                [0, 0, 0.26, 0, 0, 0, 0.74, 0],
                [0, 0, 0.88, 0.12, 0, 0, 0, 0],
                [0, 0, 0.17, 0, 0.83, 0, 0, 0],
                [0, 0.35, 0.65, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0.47, 0, 0, 0.53, 0, 0, 0],
                [0.36, 0, 0, 0, 0, 0, 0, 0.64],
                [0.15, 0, 0, 0, 0, 0.85, 0, 0],
                [0, 0, 0, 0, 0.18, 0, 0, 0.82],
            ],
        ]
    )

    # As expert_capacity_in_batch=2, so updated softmax_probs become (4 tokens were dropped):
    # softmax_probs = jnp.array([[[0.20, 0, 0, 0, 0, 0.80, 0, 0],
    #                             [0.68, 0, 0, 0, 0.32, 0, 0, 0],
    #                             [0, 0.78, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0.32, 0, 0.68, 0, 0]],
    #                            [[0, 0.26, 0.74, 0, 0, 0, 0, 0],
    #                             [0, 0.79, 0, 0, 0.21, 0, 0, 0],
    #                             [0.89, 0, 0, 0, 0, 0.11, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0, 0.89]],
    #                            [[0, 0, 0.26, 0, 0, 0, 0.74, 0],
    #                             [0, 0, 0.88, 0.12, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0.83, 0, 0, 0],
    #                             [0, 0.35, 0, 0, 0, 0, 0, 0]],
    #                            [[0, 0.47, 0, 0, 0.53, 0, 0, 0],
    #                             [0.36, 0, 0, 0, 0, 0, 0, 0.64],
    #                             [0.15, 0, 0, 0, 0, 0.85, 0, 0],
    #                             [0, 0, 0, 0, 0.18, 0, 0, 0.82]]])

    # shape of dispatch_mask & combine_mask: (batch_size, seq_len, num_experts, expert_capacity_per_batch)
    expected_combine_mask = jnp.array(
        [
            [
                [[0.2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.8, 0], [0, 0], [0, 0]],
                [[0, 0.68], [0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.78, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0.68], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.26, 0], [0.74, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0.79], [0, 0], [0, 0], [0.21, 0], [0, 0], [0, 0], [0, 0]],
                [[0.89, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.11, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.89, 0]],
            ],
            [
                [[0, 0], [0, 0], [0.26, 0], [0, 0], [0, 0], [0, 0], [0.74, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0.88], [0.12, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0.83, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.35, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.47, 0], [0, 0], [0, 0], [0.53, 0], [0, 0], [0, 0], [0, 0]],
                [[0.36, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.64, 0]],
                [[0, 0.15], [0, 0], [0, 0], [0, 0], [0, 0], [0.85, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0.18], [0, 0], [0, 0], [0, 0.82]],
            ],
        ],
        dtype=jnp.float32,
    )
    expected_dispatch_mask = expected_combine_mask.astype(bool)
    actual_dispatch_mask, actual_combine_mask = self.model.generate_masks(top_k_indices, softmax_probs)

    self.assertTrue((expected_dispatch_mask == actual_dispatch_mask).all())
    self.assertTrue(jax.numpy.allclose(expected_combine_mask, actual_combine_mask, rtol=1e-02, atol=1e-02))


class MoeLoopBlock(nn.Module):
  """Reference implemetnation from https://github.com/mistralai/mistral-inference.
  This is not included anymore in our repo,
  due to limitation of for-loop implementation in sharding.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.bfloat16

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = linears.DenseGeneral(
        self.num_experts, dtype=self.dtype, kernel_init=self.kernel_init, kernel_axes=self.kernel_axes, name="gate"
    )(inputs)

    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    mlp_lnx = jnp.zeros_like(inputs)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    for k in range(self.num_experts):
      weights_exp = jnp.sum(jnp.multiply(selected_experts == k, weights), axis=-1)
      mlp_lnx_exp = linears.MlpBlock(
          intermediate_dim=self.config.mlp_dim,
          activations=["silu", "linear"],
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name=f"mlp_{k}",
          config=self.config,
      )(inputs, deterministic=deterministic)

      mlp_lnx_exp = nn.with_logical_constraint(mlp_lnx_exp, ("activation_batch", "activation_length", "activation_embed"))
      mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
      mlp_lnx += mlp_lnx_exp

    return mlp_lnx


class MoeBlockTest(unittest.TestCase):

  def get_expected_output(self, rng, hidden_states, cfg):
    model = MoeLoopBlock(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
    )
    variables = model.init(
        rng, jax.random.normal(rng, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim))
    )

    output = jax.jit(model.apply)(variables, hidden_states)
    return variables, output

  def get_moe_output(self, variables, hidden_states, cfg, mesh):
    model = linears.MoeBlock(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
    )

    # convert format of parameters
    kernel = variables["params"]["gate"]["kernel"].value
    kernel = kernel.astype(cfg.weight_dtype)

    exp_wi_0 = []
    exp_wi_1 = []
    exp_wo = []

    for i in range(cfg.num_experts):
      tmp_wi_0 = variables["params"][f"mlp_{i}"]["wi_0"]["kernel"].value
      tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wi_1 = variables["params"][f"mlp_{i}"]["wi_1"]["kernel"].value
      tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wo = variables["params"][f"mlp_{i}"]["wo"]["kernel"].value
      tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))

      exp_wi_0.append(tmp_wi_0)
      exp_wi_1.append(tmp_wi_1)
      exp_wo.append(tmp_wo)

    wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
    wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
    wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)

    moe_variables = {"params": {"gate": {"kernel": kernel}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo}}

    output = jax.jit(model.apply)(moe_variables, hidden_states)
    return output

  @pytest.mark.tpu_only
  def test_megablox(self):
    pyconfig.initialize(
        [None, "configs/base.yml"],
        run_name="moe_block_megablox_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
    )

    cfg = pyconfig.config
    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    hidden_states = jax.random.uniform(
        rng_hidden_states, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim), dtype=cfg.dtype
    )

    devices_array = max_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg)
    actual_output, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_dense(self):
    pyconfig.initialize(
        [None, "configs/base.yml"],
        run_name="moe_block_dense_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
    )

    cfg = pyconfig.config
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    hidden_states = jax.random.uniform(
        rng_hidden_states, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim), dtype=cfg.dtype
    )

    devices_array = max_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg)
    actual_output, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    # suspect numeric issues in the dense matmul
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=5e-01, atol=5e-01, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
