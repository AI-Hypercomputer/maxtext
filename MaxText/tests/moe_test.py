"""
 Copyright 2024 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Tests for Mistral."""

import jax
import unittest
from layers import linears
from layers import initializers
import jax.numpy as jnp

import pyconfig
import max_utils
from jax.sharding import Mesh
from typing import Tuple
import common_types
import flax.linen as nn
from tests import time


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer

class MoeLoopBlock(nn.Module):
  """Mixture of Experts (MoE) block.
  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    dtype: Type for the dense layer.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  weight_dtype: DType = jnp.bfloat16
  dtype: DType = jnp.bfloat16

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = linears.DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name='gate')(inputs)

    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    # print("weights from loop", weights)
    # print("selected_experts from loop", selected_experts)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    mlp_lnx = jnp.zeros_like(inputs)
    mlp_lnx = nn.with_logical_constraint(
            mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
        )

    for k in range(self.num_experts):
        weights_exp = jnp.sum(jnp.multiply(selected_experts==k, weights), axis=-1)
        mlp_lnx_exp = linears.MlpBlock(
          intermediate_dim=self.config.mlp_dim,
          activations=['silu', 'linear'],
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name=f'mlp_{k}',
          config=self.config,
          )(inputs, deterministic=deterministic)

        mlp_lnx_exp = nn.with_logical_constraint(
            mlp_lnx_exp, ('activation_batch', 'activation_length', 'activation_embed')
        )
        mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
        mlp_lnx += mlp_lnx_exp
        # print(f"mlp_lnx.shape: {mlp_lnx.shape}")

    return mlp_lnx


def get_expected_output(rng, hidden_states, cfg):
      model = MoeLoopBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
          kernel_axes=('embed', 'mlp'),
          dtype=cfg.dtype,
      )
      variables = model.init(rng, jax.random.normal(rng, (int(cfg.per_device_batch_size), 
                                                          cfg.max_target_length, 
                                                          cfg.base_emb_dim)))

      # print("get_expected_output variables", variables)
      # breakpoint()
      time.simple_timeit(jax.jit(model.apply), variables, hidden_states, tries=10, task="loop")

      output = jax.jit(model.apply)(variables, hidden_states)
      return variables, output


def get_moe_output(variables, hidden_states, cfg, mesh):
      model = linears.MoeBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
          kernel_axes=(None, 'test'),
          dtype=cfg.dtype,
      )
      # print("jax.tree_util.tree_structure(variables)")
      # print(jax.tree_util.tree_structure(variables))

      kernel = variables['params']['gate']['kernel'].value
      kernel = kernel.astype(cfg.weight_dtype)

      exp_wi_0 = []
      exp_wi_1 = []
      exp_wo = []

      for i in range(cfg.num_experts):

        tmp_wi_0 = variables['params'][f'mlp_{i}']['wi_0']['kernel'].value
        tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
        tmp_wi_1 = variables['params'][f'mlp_{i}']['wi_1']['kernel'].value
        tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))

        tmp_wo = variables['params'][f'mlp_{i}']['wo']['kernel'].value
        tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))

        exp_wi_0.append(tmp_wi_0)
        exp_wi_1.append(tmp_wi_1)
        exp_wo.append(tmp_wo)

      # wi_0 = jnp.array(exp_wi_0, dtype=cfg.weight_dtype)
      # wi_1 = jnp.array(exp_wi_1, dtype=cfg.weight_dtype)
      # wo = jnp.array(exp_wo, dtype=cfg.weight_dtype)
      # print("wi_0: {wi_0}")

      wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
      wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
      wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)

      kernel = nn.with_logical_constraint(
            kernel, ('embed', 'mlp')
        )
      wi_0 = nn.with_logical_constraint(
            wi_0, (None, 'test', None)
        )
      wi_1 = nn.with_logical_constraint(
            wi_1, (None, 'test', None)
        )
      wo = nn.with_logical_constraint(
            wo, (None, 'test', None)
        )

      moe_variables = {'params': {'gate': {'kernel': kernel}, 
                                  'wi_0': wi_0, 
                                  'wi_1': wi_1,
                                  'wo': wo}}

      # print("get_moe_output expected_variables", variables)
      # breakpoint()
      # from jax.sharding import PartitionSpec
      # fsdp_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec('fsdp'))
      # moe_variables = jax.device_put(moe_variables, device=fsdp_sharding)
      # hidden_states = jax.device_put(hidden_states, device=fsdp_sharding)
      
      hidden_states = nn.with_logical_constraint(
            hidden_states, ('activation_batch', 'activation_length', 'activation_embed')
        )
      
      
      time.simple_timeit(jax.jit(model.apply), moe_variables, hidden_states, tries=10, task="matmul")
      output = jax.jit(model.apply)(moe_variables, hidden_states)
      # output = model.apply(moe_variables, hidden_states)
      return output


class MoeTest(unittest.TestCase):

  def setUp(self):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    pyconfig.initialize(
      [None, 'configs/base.yml'],
      run_name='test',
      enable_checkpointing=False,
      model_name='mixtral-test',
      dtype='bfloat16',
      weight_dtype='bfloat16',
      moe_matmul=True,
      megablox=True,
      ici_fsdp_parallelism=4,
      per_device_batch_size=4,
      dataset_type='synthetic',
      attention='flash',
      max_target_length=4096,
    )

    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(42)

    self.hidden_states = jax.random.uniform(self.rng, (int(self.cfg.per_device_batch_size),
                                            self.cfg.max_target_length,
                                            self.cfg.base_emb_dim), dtype=self.cfg.dtype)
    # print(f"{self.hidden_states.shape}=")

    # devices_array = max_utils.create_device_mesh(self.cfg, devices=[jax.devices()[0]])
    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

  def test_moe_block(self):
    variables, expected_output = get_expected_output(self.rng, self.hidden_states, self.cfg)
    actual_output = get_moe_output(variables, self.hidden_states, self.cfg, self.mesh)
    # print("expected_output", expected_output)
    # print("actual_output", actual_output)
    # breakpoint()
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))


if __name__ == '__main__':
  unittest.main()