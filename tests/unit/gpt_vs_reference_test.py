# Copyright 2023–2025 Google LLC
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

"""
Tests for GPT OSS: Attention, MLP, RoPE

GPT OSS PyTorch implementation at:
https://github.com/huggingface/transformers/blob/31ab7168ff7e07f61c90134e5238c4d97606aa70/src/transformers/models/gpt_oss/modular_gpt_oss.py
"""

from types import SimpleNamespace
from typing import Optional
import math
import unittest

import numpy as np
import pytest

import torch
from torch import nn
import torch.nn.functional as F

from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
import jax
import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.layers import attentions, moe, embeddings
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


# Reference implementation
class GptOssExperts(nn.Module):
  """PyTorch reference implementation for GPT-OSS Experts layer."""

  def __init__(self, config):
    """Initializes the GptOssExperts module.

    Args:
      config: A configuration object with model hyperparameters.
    """
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
    """Forward pass for the GptOssExperts module.

    Args:
      hidden_states: The input tensor.
      router_indices: Indices of the selected experts (not used in this simplified forward pass).
      routing_weights: Weights for combining expert outputs.

    Returns:
      The output tensor after processing by the experts.
    """
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
  """PyTorch reference implementation for GPT-OSS Top-K Router."""

  def __init__(self, config):
    """Initializes the GptOssTopKRouter module.

    Args:
      config: A configuration object with model hyperparameters.
    """
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    self.bias = nn.Parameter(torch.empty(self.num_experts))

  def forward(self, hidden_states):
    """Forward pass for the GptOssTopKRouter module.

    Args:
      hidden_states: The input tensor.

    Returns:
      A tuple containing the router scores and the indices of the top-k experts.
    """
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices


# Reference implementation
class GptOssMLP(nn.Module):
  """PyTorch reference implementation for the complete GPT-OSS MLP block."""

  def __init__(self, config):
    """Initializes the GptOssMLP module.

    Args:
      config: A configuration object with model hyperparameters.
    """
    super().__init__()
    self.router = GptOssTopKRouter(config)
    self.experts = GptOssExperts(config)

  def forward(self, hidden_states):
    """Forward pass for the GptOssMLP module.

    Args:
      hidden_states: The input tensor.

    Returns:
      A tuple containing the output of the MoE block and the router scores.
    """
    router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


# Standard implementation of repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  Repeats the key and value heads to match the number of query heads in GQA.
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Reference implementation
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
  """PyTorch reference implementation for eager attention.

  This function computes attention, including support for attention sinks,
  and is used as a reference to validate the MaxText implementation.

  Args:
    module: A module-like object containing configuration (e.g., sinks).
    query: The query tensor.
    key: The key tensor.
    value: The value tensor.
    attention_mask: An optional mask to apply to the attention weights.
    scaling: The scaling factor for the attention scores.
    dropout: The dropout rate.

  Returns:
    A tuple containing the attention output and the attention weights.
  """
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  if hasattr(module, "sinks") and module.sinks is not None:
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
  else:
    probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
    scores = probs

  attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()
  return attn_output, attn_weights


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  """Converts a PyTorch tensor to a JAX array.

  Args:
    pt_tensor: The PyTorch tensor to convert.

  Returns:
    The equivalent JAX array.
  """
  return jnp.asarray(pt_tensor.detach().numpy())


class Config:
  """A configuration class for holding hyperparameters for the tests."""

  hidden_size = 16
  intermediate_size = 16
  num_local_experts = 8
  num_experts_per_tok = 2
  limit = 7.0
  # attention
  num_attention_heads = 8
  num_key_value_heads = 4
  head_dim = 8
  attention_dropout = 0.0
  # rope
  rope_type = "yarn"
  rope_max_timescale = 150_000
  max_position_embeddings = 131072
  original_max_position_embeddings = 4096
  rope_factor = 32
  beta_fast = 32
  beta_slow = 1
  rope_interleave = False
  rope_truncate = False
  rope_attention_scaling = True


class GptOssMLPTest(unittest.TestCase):
  """Tests for the MaxText GPT-OSS MLP implementation against a PyTorch reference."""

  def test_mlp_block(self):
    """Validates the MaxText MoE MLP block against the PyTorch reference."""
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    config = Config()
    # Set print options to show full tensors and arrays
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)
    predefined_weights = {
        "router.weight": torch.randn(config.num_local_experts, config.hidden_size),
        "router.bias": torch.randn(config.num_local_experts),
        "experts.gate_up_proj": torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        "experts.gate_up_proj_bias": torch.randn(config.num_local_experts, 2 * config.intermediate_size),
        "experts.down_proj": torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size),
        "experts.down_proj_bias": torch.randn(config.num_local_experts, config.hidden_size),
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
        [None, get_test_config_path()],
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
        base_moe_mlp_dim=config.intermediate_size,
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
    # Add normalization to let logits at same scale
    normalized_expected = to_jax(expected_output) / jnp.linalg.norm(to_jax(expected_output))
    normalized_actual = actual_output / jnp.linalg.norm(actual_output)
    mse = jnp.mean((normalized_expected - normalized_actual) ** 2)
    self.assertLess(mse, 1e-3, f"mlp mismatch, MSE: {mse}")
    np.testing.assert_allclose(normalized_expected, normalized_actual, rtol=1e-3, atol=1e-2)


class GptOssAttentionTest(unittest.TestCase):
  """Tests for the MaxText GPT-OSS attention implementation."""

  def setUp(self):
    """Sets up the test environment, preparing tensors and configurations."""
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    self.config = Config()
    self.batch_size = 4
    self.seq_len = 128

    self.mock_module_with_sinks = SimpleNamespace(
        num_key_value_groups=self.config.num_attention_heads // self.config.num_key_value_heads,
        sinks=torch.randn(self.config.num_attention_heads),
        training=False,
    )

    self.query = torch.randn(self.batch_size, self.config.num_attention_heads, self.seq_len, self.config.head_dim)
    self.key = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, self.config.head_dim)
    self.value = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, self.config.head_dim)
    self.attention_mask = None
    self.scaling = 1.0 / (self.config.head_dim**0.5)

    # JAX tensors
    self.jax_query = to_jax(self.query)
    self.jax_key = to_jax(self.key)
    self.jax_value = to_jax(self.value)
    self.jax_sinks = to_jax(self.mock_module_with_sinks.sinks)
    self.jax_query_t = jnp.transpose(self.jax_query, (0, 2, 1, 3))
    self.jax_key_t = jnp.transpose(self.jax_key, (0, 2, 1, 3))
    self.jax_value_t = jnp.transpose(self.jax_value, (0, 2, 1, 3))

  def test_dot_product_attention_with_sinks(self):
    """Validates dot-product attention with sinks against the reference."""
    expected_attn_output, _ = eager_attention_forward(
        module=self.mock_module_with_sinks,
        query=self.query,
        key=self.key,
        value=self.value,
        attention_mask=self.attention_mask,
        scaling=self.scaling,
        dropout=0.0,
    )

    cfg_dot = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="gpt_oss_attention_test_dot",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        base_num_query_heads=self.config.num_attention_heads,
        base_num_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        attention="dot_product",
        attention_bias=False,
        attention_sink=True,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_dot)
    mesh = Mesh(devices_array, cfg_dot.mesh_axes)

    attention_op_dot = attentions.AttentionOp(
        config=cfg_dot,
        mesh=mesh,
        attention_kernel="dot_product",
        max_target_length=self.seq_len,
        num_query_heads=self.config.num_attention_heads,
        num_kv_heads=self.config.num_key_value_heads,
        dtype=jnp.float32,
        attention_type=attentions.AttentionType.FULL,
    )

    @jax.jit
    def run_dot_product_attention(q, k, v):
      unnormalized_output, _, sum_val = attention_op_dot.apply_attention_dot(
          query=q,
          key=k,
          value=v,
          decoder_segment_ids=jnp.ones((self.batch_size, self.seq_len)),
          model_mode="train",
          sinks=self.jax_sinks,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )
      return unnormalized_output / sum_val

    scaled_jax_query_t = self.jax_query_t * self.scaling
    actual_attn_output_dot = run_dot_product_attention(scaled_jax_query_t, self.jax_key_t, self.jax_value_t)

    mse_dot = jnp.mean((to_jax(expected_attn_output) - actual_attn_output_dot) ** 2)
    self.assertLess(mse_dot, 1e-3, f"dot-product attention mismatch, MSE: {mse_dot}")
    np.testing.assert_allclose(to_jax(expected_attn_output), actual_attn_output_dot, rtol=1e-3, atol=1e-2)

  def test_flash_attention_with_sinks(self):
    """Validates flash attention with sinks against the reference."""
    expected_attn_output, _ = eager_attention_forward(
        module=self.mock_module_with_sinks,
        query=self.query,
        key=self.key,
        value=self.value,
        attention_mask=self.attention_mask,
        scaling=self.scaling,
        dropout=0.0,
    )

    cfg_flash = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="gpt_oss_attention_test_flash",
        enable_checkpointing=False,
        model_name="default",
        dtype="float32",
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        base_num_query_heads=self.config.num_attention_heads,
        base_num_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        attention="flash",
        attention_bias=False,
        attention_sink=True,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_flash)
    mesh = Mesh(devices_array, cfg_flash.mesh_axes)

    attention_op_flash = attentions.AttentionOp(
        config=cfg_flash,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=self.seq_len,
        num_query_heads=self.config.num_attention_heads,
        num_kv_heads=self.config.num_key_value_heads,
        dtype=jnp.float32,
        attention_type=attentions.AttentionType.FULL,
    )

    @jax.jit
    def run_flash_attention(q, k, v, sinks_logits):
      output = attention_op_flash.tpu_flash_attention(
          query=q,
          key=k,
          value=v,
          decoder_segment_ids=None,
          sinks=sinks_logits,
      )
      return output

    scaled_jax_query_flash_t = self.jax_query_t * self.scaling
    actual_attn_output_flash = run_flash_attention(
        scaled_jax_query_flash_t, self.jax_key_t, self.jax_value_t, self.jax_sinks
    )
    mse_flash = jnp.mean((to_jax(expected_attn_output) - actual_attn_output_flash) ** 2)
    self.assertLess(mse_flash, 1e-3, f"flash attention mismatch, MSE: {mse_flash}")
    np.testing.assert_allclose(to_jax(expected_attn_output), actual_attn_output_flash, rtol=1e-3, atol=1e-2)

  def _run_cudnn_flash_te_attention_with_sinks(
      self,
      attention_type,
      packing,
      check_sinks_grad=False,
  ):
    """Compare MaxText `cudnn_flash_attention` to the PyTorch reference for one (type, packing) pair.

    Sinks reach Transformer Engine through its internal `softmax_offset` parameter, which MaxText
    overwrites right after `lazy_init`. The original FULL / packing=False case also checks that
    the grafted array stays part of the computation, i.e. that gradients still flow back into
    `sinks`.
    """
    te_attention = pytest.importorskip("transformer_engine.jax.attention")

    is_local = attention_type == attentions.AttentionType.LOCAL_SLIDING
    # Window must be clearly smaller than seq_len or local and global degenerate into the same
    # computation. Asserted against the causal reference rather than assumed.
    sliding_window_size = 32
    # A realistic head_dim for the TE fused kernels, set locally because the shared Config's
    # head_dim of 8 is what the other tests in this file expect. The head counts are left alone so
    # that the sinks prepared in setUp can be reused as-is.
    head_dim = 64
    tensors = self._pytorch_sinks_attention_reference(
        is_local=is_local, sliding_window_size=sliding_window_size, head_dim=head_dim
    )
    expected_attn_output = tensors["expected_attn_output"]
    query_bf16 = tensors["query_bf16"]
    key_bf16 = tensors["key_bf16"]
    value_bf16 = tensors["value_bf16"]

    cfg_kwargs = {
        "run_name": f"gpt_oss_attention_test_cudnn_flash_te_{attention_type.name}_packing_{packing}",
        "enable_checkpointing": False,
        "model_name": "default",
        # The fused kernels need bf16/fp16; on fp32 TE falls back to its unfused path.
        "dtype": "bfloat16",
        "per_device_batch_size": 1,
        "max_target_length": self.seq_len,
        "max_prefill_predict_length": self.seq_len,
        "base_num_query_heads": self.config.num_attention_heads,
        "base_num_kv_heads": self.config.num_key_value_heads,
        "head_dim": head_dim,
        "attention": "cudnn_flash_te",
        "attention_bias": False,
        "attention_sink": True,
        "packing": packing,
    }
    if packing:
      # Required by configs/types.py when hardware=gpu, packing, and cudnn_flash_te. dataset_type
      # is left at base.yml's tfds so this combination actually takes the THD packed branch.
      cfg_kwargs["max_segments_per_seq"] = 32
    if is_local:
      cfg_kwargs["attention_type"] = "local_sliding"
      cfg_kwargs["sliding_window_size"] = sliding_window_size

    cfg_te = pyconfig.initialize([None, get_test_config_path()], **cfg_kwargs)
    devices_array = maxtext_utils.create_device_mesh(cfg_te)
    mesh = Mesh(devices_array, cfg_te.mesh_axes)

    def run_cudnn_flash_te_attention(q, k, v, sinks_logits):
      # Building the layer forks the rng streams, so AttentionOp has to be created inside the
      # traced function to keep flax from complaining about a mutation across trace levels.
      attention_op_kwargs = {
          "config": cfg_te,
          "mesh": mesh,
          "attention_kernel": "cudnn_flash_te",
          "max_target_length": self.seq_len,
          "num_query_heads": self.config.num_attention_heads,
          "num_kv_heads": self.config.num_key_value_heads,
          "dtype": jnp.bfloat16,
          "attention_type": attention_type,
          "rngs": nnx.Rngs(0),
      }
      if is_local:
        attention_op_kwargs["sliding_window_size"] = sliding_window_size
      attention_op_te = attentions.AttentionOp(**attention_op_kwargs)
      # Non-zero segment ids: TE treats 0 as padding, and zeros would validate attending to nothing.
      decoder_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
      # TE 2.17+ requires segment_pos for the THD SequenceDescriptor; a single packed segment is
      # positions 0..S-1. Passing None hits ValueError before the kernel runs.
      segment_positions = None
      if packing:
        segment_positions = jnp.broadcast_to(
            jnp.arange(self.seq_len, dtype=jnp.int32)[None, :],
            (self.batch_size, self.seq_len),
        )
      return attention_op_te.cudnn_flash_attention(
          q,
          k,
          v,
          decoder_segment_ids,
          segment_positions,
          "train",
          sinks=sinks_logits,
      )

    scaled_query_te_t = jnp.transpose(to_jax(query_bf16.to(torch.float32)), (0, 2, 1, 3)).astype(jnp.bfloat16)
    key_te_t = jnp.transpose(to_jax(key_bf16.to(torch.float32)), (0, 2, 1, 3)).astype(jnp.bfloat16)
    value_te_t = jnp.transpose(to_jax(value_bf16.to(torch.float32)), (0, 2, 1, 3)).astype(jnp.bfloat16)

    qkv_layout = te_attention.QKVLayout.THD_THD_THD if packing else te_attention.QKVLayout.BSHD_BSHD_BSHD
    # Must match AttentionOp.cudnn_flash_attention: TE inclusive [i-left, i] vs MaxText [i-(w-1), i].
    window_size = (sliding_window_size - 1, 0) if is_local else None
    fused_kernel_available = te_attention.is_fused_attn_kernel_available(
        True,  # is_training
        jnp.bfloat16,
        jnp.bfloat16,
        qkv_layout,
        te_attention.AttnBiasType.NO_BIAS,
        te_attention.AttnMaskType.PADDING_CAUSAL_MASK,
        te_attention.AttnSoftmaxType.LEARNABLE_SOFTMAX,
        0.0,  # dropout_probability
        self.config.num_attention_heads,
        self.config.num_key_value_heads,
        self.seq_len,
        self.seq_len,
        head_dim,
        head_dim,
        window_size,
    )
    if not fused_kernel_available:
      self.skipTest(
          "no TE fused attention kernel for "
          f"{attention_type.name} packing={packing} qkv_layout={qkv_layout.name} "
          f"window_size={window_size}; would silently validate the unfused fallback"
      )

    with mesh, nn_partitioning.axis_rules(cfg_te.logical_axis_rules):
      actual_attn_output_te = jax.jit(run_cudnn_flash_te_attention)(
          scaled_query_te_t, key_te_t, value_te_t, self.jax_sinks
      )
      if check_sinks_grad:

        def sinks_output_sum_of_squares(sinks_logits):
          output = run_cudnn_flash_te_attention(scaled_query_te_t, key_te_t, value_te_t, sinks_logits)
          return jnp.sum(output.astype(jnp.float32) ** 2)

        sinks_grad = jax.jit(jax.grad(sinks_output_sum_of_squares))(self.jax_sinks)

    actual_attn_output_te = actual_attn_output_te.astype(jnp.float32)
    mse_te = float(jnp.mean((to_jax(expected_attn_output) - actual_attn_output_te) ** 2))
    combo = f"{attention_type.name} packing={packing}"
    print(f"cudnn_flash_te vs pytorch MSE ({combo}): {mse_te:.6e}")
    self.assertLess(mse_te, 1e-3, f"cudnn_flash_te attention mismatch ({combo}), MSE: {mse_te}")
    np.testing.assert_allclose(to_jax(expected_attn_output), actual_attn_output_te, rtol=1e-3, atol=1e-2)

    if check_sinks_grad:
      # The whole point of grafting sinks into `softmax_offset` is that they stay differentiable.
      self.assertEqual(sinks_grad.shape, (self.config.num_attention_heads,))
      self.assertTrue(bool(jnp.all(jnp.isfinite(sinks_grad))), f"non-finite gradient w.r.t. sinks: {sinks_grad}")
      self.assertGreater(
          float(jnp.max(jnp.abs(sinks_grad))),
          0.0,
          "gradient w.r.t. sinks is identically zero, softmax_offset injection is not taking effect",
      )

  @pytest.mark.gpu_only
  def test_cudnn_flash_te_attention_with_sinks(self):
    """FULL, packing=False: the original TE fused-attn baseline, including sink gradients."""
    self._run_cudnn_flash_te_attention_with_sinks(attentions.AttentionType.FULL, packing=False, check_sinks_grad=True)

  @pytest.mark.gpu_only
  def test_cudnn_flash_te_attention_with_sinks_packed(self):
    """FULL, packing=True: THD SequenceDescriptor branch."""
    self._run_cudnn_flash_te_attention_with_sinks(attentions.AttentionType.FULL, packing=True)

  @pytest.mark.gpu_only
  def test_cudnn_flash_te_attention_with_sinks_local_sliding(self):
    """LOCAL_SLIDING, packing=False: padding seqlens plus window_size=[w-1, 0]."""
    self._run_cudnn_flash_te_attention_with_sinks(attentions.AttentionType.LOCAL_SLIDING, packing=False)

  @pytest.mark.gpu_only
  def test_cudnn_flash_te_attention_with_sinks_local_sliding_packed(self):
    """LOCAL_SLIDING, packing=True: THD SequenceDescriptor plus window_size=[w-1, 0]."""
    self._run_cudnn_flash_te_attention_with_sinks(attentions.AttentionType.LOCAL_SLIDING, packing=True)

  def _pytorch_sinks_attention_reference(self, *, is_local: bool, sliding_window_size: int, head_dim: int):
    """QKV, causal/window mask, and PyTorch output used by the TE fused-attn checks."""
    scaling = 1.0 / (head_dim**0.5)
    query = torch.randn(self.batch_size, self.config.num_attention_heads, self.seq_len, head_dim)
    key = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, head_dim)
    value = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, head_dim)

    neg_inf = torch.finfo(torch.float32).min
    causal_2d = torch.triu(torch.full((self.seq_len, self.seq_len), neg_inf), diagonal=1)
    attention_mask_2d = causal_2d
    if is_local:
      row_ids = torch.arange(self.seq_len).view(self.seq_len, 1)
      col_ids = torch.arange(self.seq_len).view(1, self.seq_len)
      sliding_ok = (col_ids > (row_ids - sliding_window_size)) & (col_ids <= row_ids)
      sliding_2d = torch.where(
          sliding_ok,
          torch.zeros(self.seq_len, self.seq_len),
          torch.full((self.seq_len, self.seq_len), neg_inf),
      )
      attention_mask_2d = torch.minimum(causal_2d, sliding_2d)

    causal_mask = causal_2d.reshape(1, 1, self.seq_len, self.seq_len).expand(self.batch_size, 1, -1, -1)
    attention_mask = attention_mask_2d.reshape(1, 1, self.seq_len, self.seq_len).expand(self.batch_size, 1, -1, -1)

    query_bf16 = (query * scaling).to(torch.bfloat16)
    key_bf16 = key.to(torch.bfloat16)
    value_bf16 = value.to(torch.bfloat16)
    query_ref = query_bf16.to(torch.float32)
    key_ref = key_bf16.to(torch.float32)
    value_ref = value_bf16.to(torch.float32)

    expected_attn_output, _ = eager_attention_forward(
        module=self.mock_module_with_sinks,
        query=query_ref,
        key=key_ref,
        value=value_ref,
        attention_mask=attention_mask,
        scaling=1.0,
        dropout=0.0,
    )

    if is_local:
      expected_causal, _ = eager_attention_forward(
          module=self.mock_module_with_sinks,
          query=query_ref,
          key=key_ref,
          value=value_ref,
          attention_mask=causal_mask,
          scaling=1.0,
          dropout=0.0,
      )
      window_vs_causal_mse = float(torch.mean((expected_causal - expected_attn_output) ** 2))
      self.assertGreater(
          window_vs_causal_mse,
          1e-4,
          "windowed PyTorch reference is indistinguishable from the causal reference at "
          f"seq_len={self.seq_len} sliding_window_size={sliding_window_size} "
          f"(MSE={window_vs_causal_mse}); the local case would not actually cover the window",
      )

    return {
        "expected_attn_output": expected_attn_output,
        "query_bf16": query_bf16,
        "key_bf16": key_bf16,
        "value_bf16": value_bf16,
    }


class GptOssRotaryEmbedding(nn.Module):
  """
  https://github.com/huggingface/transformers/blob/b9282355bea846b54ed850a066901496b19da654/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L169
  """

  inv_freq: torch.Tensor  # fix linting for `register_buffer`

  def __init__(self, config, device=None):
    super().__init__()
    # BC: "rope_type" was originally "type"
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
      self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
    else:
      self.rope_type = "default"
    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings

    self.config = config
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
  def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the rotary positional embeddings for the given positions.

    This method calculates the cosine and sine values for RoPE based on the
    input positions. These values are then applied to the query and key
    tensors in the attention mechanism.

    Args:
      x: The input tensor, used to determine the device and dtype for the
        output. Shape is not used otherwise.
      position_ids: A 1D tensor of token positions for which to compute the
        embeddings.

    Returns:
      A tuple containing the cosine and sine components of the rotary embeddings.
    """
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = freqs
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling

    return cos.to(x.dtype), sin.to(x.dtype)


def _compute_yarn_parameters(
    config, device: "torch.device", seq_len: Optional[int] = None
) -> tuple["torch.Tensor", float]:
  """
  https://github.com/huggingface/transformers/blob/b9282355bea846b54ed850a066901496b19da654/src/transformers/modeling_rope_utils.py#L197C1-L281C38
  """

  base = config.rope_theta
  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)
  factor = config.rope_scaling["factor"]
  attention_factor = config.rope_scaling.get("attention_factor")
  mscale = config.rope_scaling.get("mscale")
  mscale_all_dim = config.rope_scaling.get("mscale_all_dim")
  original_max_position_embeddings = (
      config.rope_scaling.get("original_max_position_embeddings") or config.max_position_embeddings
  )

  def get_mscale(scale, mscale=1):
    if scale <= 1:
      return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if mscale and mscale_all_dim:
      attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
    else:
      attention_factor = get_mscale(factor)

  # Optional config options
  # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
  beta_fast = config.rope_scaling.get("beta_fast") or 32
  beta_slow = config.rope_scaling.get("beta_slow") or 1

  # Compute the inverse frequencies
  def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Inverse dimension formula to find the dimension based on the number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

  def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
    """Find dimension range bounds based on rotations"""
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(minimum, maximum, dimensions):
    if minimum == maximum:
      maximum += 0.001  # Prevent singularity

    linear_func = (torch.arange(dimensions, dtype=torch.float32) - minimum) / (maximum - minimum)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

  # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
  # to expand the possible context length. In other words, interpolation = apply scaling factor.
  pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
  inv_freq_extrapolation = 1.0 / pos_freqs
  inv_freq_interpolation = 1.0 / (factor * pos_freqs)

  truncate = config.rope_scaling.get("truncate", True)
  low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

  # Get n-dimensional rotational scaling corrected for extrapolation
  inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
  inv_freq = (
      inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
      + inv_freq_extrapolation * inv_freq_extrapolation_factor
  )
  return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {"yarn": _compute_yarn_parameters}


# https://github.com/huggingface/transformers/blob/b9282355bea846b54ed850a066901496b19da654/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L217C1-L233C28
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
  """Applies rotary embedding to a tensor.

  Args:
    x: The input tensor.
    cos: The cosine part of the rotary embedding.
    sin: The sine part of the rotary embedding.

  Returns:
    The tensor with rotary embeddings applied.
  """
  first_half, second_half = torch.chunk(x, 2, dim=-1)
  first_ = first_half * cos - second_half * sin
  second_ = second_half * cos + first_half * sin
  return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
  """Applies rotary positional embedding to query and key tensors.

  Args:
    q: The query tensor.
    k: The key tensor.
    cos: The cosine part of the rotary embedding.
    sin: The sine part of the rotary embedding.
    position_ids: Optional position IDs.
    unsqueeze_dim: The dimension to unsqueeze cos and sin.

  Returns:
    A tuple containing the query and key tensors with rotary embeddings applied.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = _apply_rotary_emb(q, cos, sin)
  k_embed = _apply_rotary_emb(k, cos, sin)
  return q_embed, k_embed


class GptOssYarnTest(unittest.TestCase):
  """Tests for the MaxText GPT-OSS Yarn RoPE implementation."""

  def setUp(self):
    """Sets up the test environment for Yarn RoPE validation."""
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    # jax config
    self.config = Config()
    # data, test long context
    self.batch_size = 1
    self.seq_len = 4096
    self.dtype = "float32"
    # torch tensors
    self.query = torch.randn(self.batch_size, self.config.num_attention_heads, self.seq_len, self.config.head_dim)
    self.key = torch.randn(self.batch_size, self.config.num_key_value_heads, self.seq_len, self.config.head_dim)
    self.positions = torch.arange(self.seq_len).unsqueeze(0)
    # torch config
    pt_config = {
        "head_dim": self.config.head_dim,
        "max_position_embeddings": self.config.max_position_embeddings,
        "rope_scaling": {
            "beta_fast": self.config.beta_fast,
            "beta_slow": self.config.beta_slow,
            "factor": self.config.rope_factor,
            "original_max_position_embeddings": self.config.original_max_position_embeddings,
            "rope_type": self.config.rope_type,
            "truncate": self.config.rope_truncate,
        },
        "rope_theta": self.config.rope_max_timescale,
        # placeholder, to get past `getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)`
        "hidden_size": float("inf"),
        "num_attention_heads": float("inf"),
    }
    self.pt_config = SimpleNamespace(**pt_config)
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)

  def test_yarn(self):
    """Validates the JAX Yarn RoPE implementation against the HF reference."""
    # HF Yarn RoPE
    torch_embedding = GptOssRotaryEmbedding(self.pt_config)
    cos, sin = torch_embedding(self.query, self.positions)
    q_rope_pt, k_rope_pt = apply_rotary_pos_emb(self.query, self.key, cos, sin)
    # JAX Yarn RoPE
    model_jax = embeddings.YarnRotaryEmbedding(
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mesh=self.mesh,
        beta_fast=self.config.beta_fast,
        beta_slow=self.config.beta_slow,
        rope_theta=self.config.rope_max_timescale,
        rope_factor=self.config.rope_factor,
        embedding_dims=self.config.head_dim,
        fprop_dtype=self.dtype,
        interleave=self.config.rope_interleave,
        truncate=self.config.rope_truncate,
        attention_scaling=self.config.rope_attention_scaling,
    )
    jax_positions = to_jax(self.positions)
    q_rope_jax = model_jax(to_jax(self.query).transpose(0, 2, 1, 3), jax_positions)
    k_rope_jax = model_jax(to_jax(self.key).transpose(0, 2, 1, 3), jax_positions)
    # Compare outputs from the HF and JAX implementations
    np.testing.assert_allclose(to_jax(q_rope_pt).transpose(0, 2, 1, 3), q_rope_jax, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(to_jax(k_rope_pt).transpose(0, 2, 1, 3), k_rope_jax, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
