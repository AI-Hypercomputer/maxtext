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
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = linears.DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name='gate')(inputs)
    
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    print("weights from loop", weights)
    print("selected_experts from loop", selected_experts)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1)
    mlp_lnx = jnp.zeros_like(inputs)
    weights = weights.astype(self.dtype)
    mlp_lnx = nn.with_logical_constraint(
            mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
        )

    for k in range(self.num_experts):
        weights_exp = jnp.sum(jnp.multiply(selected_experts==k, weights), axis=-1)
        mlp_lnx_exp = linears.MlpBlock(
          intermediate_dim=self.config.mlp_dim,
          activations=self.config.mlp_activations,
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
      variables = model.init(rng, jax.random.normal(rng, (cfg.base_num_query_heads, 
                                                          cfg.head_dim, 
                                                          cfg.base_emb_dim)))
      print("get_expected_output variables", variables)
      return variables, model.apply(variables, hidden_states)


def get_moe_output(variables, hidden_states, cfg, mesh):
      model = linears.MoeBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
          kernel_axes=('embed', 'mlp'),
          dtype=cfg.dtype,
      )

      kernel = variables['params']['gate']['kernel'].value
      exp0_wi_0 = variables['params']['mlp_0']['wi_0']['kernel'].value
      exp1_wi_0 = variables['params']['mlp_1']['wi_0']['kernel'].value
      exp2_wi_0 = variables['params']['mlp_2']['wi_0']['kernel'].value

      exp0_wi_1 = variables['params']['mlp_0']['wi_1']['kernel'].value
      exp1_wi_1 = variables['params']['mlp_1']['wi_1']['kernel'].value
      exp2_wi_1 = variables['params']['mlp_2']['wi_1']['kernel'].value

      exp0_wo = variables['params']['mlp_0']['wo']['kernel'].value
      exp1_wo = variables['params']['mlp_1']['wo']['kernel'].value
      exp2_wo = variables['params']['mlp_2']['wo']['kernel'].value

      # construct
      exp0_wi_0 = jnp.reshape(exp0_wi_0, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      exp1_wi_0 = jnp.reshape(exp1_wi_0, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      exp2_wi_0 = jnp.reshape(exp2_wi_0, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      wi_0 = jnp.concat((exp0_wi_0, exp1_wi_0, exp2_wi_0), axis=1).reshape(cfg.base_emb_dim,-1)

      exp0_wi_1 = jnp.reshape(exp0_wi_1, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      exp1_wi_1 = jnp.reshape(exp1_wi_1, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      exp2_wi_1 = jnp.reshape(exp2_wi_1, (cfg.base_emb_dim, 1, cfg.base_mlp_dim))
      wi_1 = jnp.concat((exp0_wi_1, exp1_wi_1, exp2_wi_1), axis=1).reshape(cfg.base_emb_dim,-1)

      exp0_wo = jnp.reshape(exp0_wo, (cfg.base_mlp_dim, 1, cfg.base_emb_dim))
      exp1_wo = jnp.reshape(exp1_wo, (cfg.base_mlp_dim, 1, cfg.base_emb_dim))
      exp2_wo = jnp.reshape(exp2_wo, (cfg.base_mlp_dim, 1, cfg.base_emb_dim))
      wo = jnp.concat((exp0_wo, exp1_wo, exp2_wo), axis=1).reshape(cfg.base_mlp_dim,-1)

      moe_variables = {'params': {'gate': {'kernel': kernel}, 
                                           'wi_0': wi_0, 
                                           'wi_1': wi_1,
                                           'wo': wo}}
      # print("actual_variables", variables)
      # rng = jax.random.PRNGKey(42)
      # variables = model.init(rng, jax.random.normal(rng, (cfg.base_num_query_heads, 
      #                                                     cfg.head_dim, 
      #                                                     cfg.base_emb_dim)))

      
      # print("get_moe_output expected_variables", variables)
      return model.apply(moe_variables, hidden_states)


class MixtralTest(unittest.TestCase):

  def setUp(self):
    pyconfig.initialize(
      [None, 'configs/base.yml'],
      run_name='test',
      enable_checkpointing=False,
      model_name='mixtral-test',
      dtype='float32',
    )

    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(42)

    num = jnp.arange(self.cfg.base_num_query_heads * self.cfg.head_dim * self.cfg.base_emb_dim)
    self.hidden_states = jnp.reshape(num, (self.cfg.base_num_query_heads, 
                                           self.cfg.head_dim, 
                                           self.cfg.base_emb_dim))
    print("hidden_states", self.hidden_states.shape)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    # self.moe_variables = {'params': {'gate': {'kernel': kenel}, 
    #                                   'wi_0': moe_wi_0_stack, 
    #                                   'wi_1': moe_wi_1_stack,
    #                                   'wo': moe_wo_stack}}

    # self.moe_loop_variables = {'params': {'gate': {'kernel': kenel}, 
    #                                       'mlp_0': {'wi_0': {'kernel': moe_loop_wi_0_exp0},
    #                                                 'wi_1': {'kernel': moe_loop_wi_1_exp0},
    #                                                 'wo': {'kernel': moe_loop_wo_exp0}},
    #                                       'mlp_1': {'wi_0': {'kernel': moe_loop_wi_0_exp1},
    #                                                 'wi_1': {'kernel': moe_loop_wi_1_exp1},
    #                                                 'wo': {'kernel': moe_loop_wo_exp1}},
    #                                       'mlp_2': {'wi_0': {'kernel': moe_loop_wi_0_exp2},
    #                                                 'wi_1': {'kernel': moe_loop_wi_1_exp2},
    #                                                 'wo': {'kernel': moe_loop_wo_exp2}}}}

  def test_moe_block(self):
    variables, expected_output = get_expected_output(self.rng, self.hidden_states, self.cfg)
    actual_output = get_moe_output(variables, self.hidden_states, self.cfg, self.mesh)
    print("expected_output", expected_output)
    print("actual_output", actual_output)
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-03, atol=1e-03, equal_nan=False))


if __name__ == '__main__':
  unittest.main()
