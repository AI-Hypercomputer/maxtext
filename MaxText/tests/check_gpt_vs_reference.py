"""
Copyright 2025 Google LLC

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

from typing import Callable, NamedTuple, Optional, Tuple
import os.path
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import jax
import unittest
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText.globals import PKG_DIR
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.layers import attentions, embeddings, moe
import numpy as np
from MaxText.layers.initializers import NdInitializer, nd_dense_init, variable_to_logically_partitioned


"""  
Tests for Attention & MLP in GPT OSS.

GPT OSS PyTorch implementation at:
https://github.com/huggingface/transformers/blob/31ab7168ff7e07f61c90134e5238c4d97606aa70/src/transformers/models/gpt_oss/modular_gpt_oss.py
"""


# Reference implementation
class GptOssExperts(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.intermediate_size = config.intermediate_size
    self.num_experts = config.num_local_experts
    self.hidden_size = config.hidden_size
    self.expert_dim = self.intermediate_size
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
    self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
    self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
    self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
    self.alpha = 1.702
    self.limit = config.limit

  def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

    gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    next_states = torch.bmm(((up + 1) * glu), self.down_proj)
    next_states = next_states + self.down_proj_bias[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
    next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)
    return next_states.reshape(batch_size, seq_len, self.hidden_size)


# Reference implementation
class GptOssTopKRouter(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    self.bias = nn.Parameter(torch.empty(self.num_experts))

  def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices


# Reference implementation
class GptOssMLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.router = GptOssTopKRouter(config)
    self.experts = GptOssExperts(config)

  def forward(self, hidden_states):
    router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class Config:
  hidden_size = 32
  intermediate_size = 16
  num_local_experts = 8
  num_experts_per_tok = 2
  limit = 7.0


class GptOssTest(unittest.TestCase):
  """Test for the MaxText GPT OSS implementation."""

  # TODO(ranran): test dense_matmul first
  def test_mlp_block(self):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    config = Config()
    predefined_weights = {
        "router.weight": torch.randn(config.num_local_experts, config.hidden_size),
        "router.bias": torch.arange(config.num_local_experts),
        "experts.gate_up_proj": torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        "experts.gate_up_proj_bias": torch.rand(config.num_local_experts, 2 * config.intermediate_size),
        "experts.down_proj": torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size),
        "experts.down_proj_bias": torch.rand(config.num_local_experts, config.hidden_size),
    }

    batch_size = 4
    seq_len = 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # reference model
    model = GptOssMLP(config)
    model.load_state_dict(predefined_weights)
    model.eval()
    with torch.no_grad():
      expected_output, _ = model(hidden_states)

    # MaxText model
    cfg = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_mlp_test",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        weight_dtype="float32",
        megablox=False,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        base_emb_dim=config.hidden_size,
        base_mlp_dim=config.intermediate_size,
        mlp_activations=["sigmoid", "linear"],
        mlp_activations_limit=config.limit,
        routed_bias=True,
        mlp_bias=True,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        decoder_block="gpt_oss",
        attention="dot_product",
    )
    jax_hidden_states = to_jax(hidden_states)
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    jax_model = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.base_mlp_dim,
        dtype=cfg.dtype,
    )

    moe_variables = {
        "moe_variables": {
            "gate": {
                "kernel": to_jax(predefined_weights["router.weight"].transpose(0, 1)),
                "bias": to_jax(predefined_weights["router.bias"]),
            },
            "wi_0": to_jax(predefined_weights["experts.gate_up_proj"][..., ::2]),
            "wi_0_bias": to_jax(predefined_weights["experts.gate_up_proj_bias"][..., ::2]),
            "wi_1": to_jax(predefined_weights["experts.gate_up_proj"][..., 1::2]),
            "wi_1_bias": to_jax(predefined_weights["experts.gate_up_proj_bias"][..., 1::2]),
            "wo": to_jax(predefined_weights["experts.down_proj"]),
            "wo_bias": to_jax(predefined_weights["experts.down_proj_bias"]),
        },
    }
    actual_output, _ = jax.jit(jax_model.apply)(moe_variables, jax_hidden_states)
    mse = jnp.mean((to_jax(expected_output) - actual_output) ** 2)
    self.assertLess(mse, 1e-1, f"expected_output mismatch with actual_output, MSE {mse} exceeds threshold 1e-1")

  def test_full_attention_block(self):
    pass

  def test_sliding_attention_block(self):
    pass


if __name__ == "__main__":
  unittest.main()
