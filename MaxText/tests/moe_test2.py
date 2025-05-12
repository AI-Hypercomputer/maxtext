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
""" Mixture of Experts (MoE) tests. """

import os.path
import unittest
from typing import Tuple

import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import flax.linen as nn
from flax.linen import partitioning as nn_partitioning

from MaxText.layers import linears
from MaxText.layers import initializers
from MaxText.layers import moe
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.globals import PKG_DIR
from MaxText import common_types


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer



class MoeLoopBlock(nn.Module):
  """Reference implementation from https://github.com/mistralai/mistral-inference.
  This is not included anymore in our repo, due to a limitation of for-loop implementation in sharding.
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
    gate_logits = moe.GateLogit(
        self.num_experts,
        self.config.model_name,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        name="gate",
    )(inputs)[0]

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


class RoutedMoeTest(unittest.TestCase):
  """Routed Mixture of Experts test."""

  def get_expected_output(self, rng, hidden_states, cfg):
    """Retrieve expected output from Routed Mixture of Experts."""
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
    """retrieve expected output from MoE"""
    model = moe.RoutedMoE(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
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
  def test_megablox_expert_parallelism(self):
    cfg = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="moe_block_megablox_ep_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_expert_parallelism=4,
        max_target_length=7,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    hidden_states = jax.random.uniform(
        rng_hidden_states, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim), dtype=cfg.dtype
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg)
    #   print(f"expected_output: {expected_output}")
      actual_output, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      print(f"actual_output: {actual_output}")
    #   self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
