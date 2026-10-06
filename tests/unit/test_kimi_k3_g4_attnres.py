# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kimi-K3 Group 4 Attention Residuals (AttnRes) Highway Subsystem."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from maxtext.layers.attn_res import (
    AttnResState,
    KimiAttnResLayer,
    _apply_attn_res,
    _apply_output_attn_res,
)
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params

_ORIG_MATMUL_PRECISION = None


def setUpModule():
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")
  jax.config.update("jax_platforms", "cpu")


def tearDownModule():
  if _ORIG_MATMUL_PRECISION is not None:
    jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


# ==============================================================================
# Reference PyTorch Implementation (from kimi-k3-hf-reference/modeling_kimi_linear.py)
# ==============================================================================
class RefKimiRMSNorm(nn.Module):
  """Reference PyTorch KimiRMSNorm (modeling_kimi_linear.py:L226-L236)."""

  def __init__(self, hidden_size, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * x.to(dtype)


def ref_apply_attn_res(prefix_sum, block_residual, proj, norm):
  """Reference PyTorch _apply_attn_res (modeling_kimi_linear.py:L1075-L1089)."""
  v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
  v_float = v.float()
  variance = v_float.pow(2).mean(-1, keepdim=True)
  k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
  score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
  scores = (k * score_weight).sum(-1)
  probs = scores.softmax(-1).unsqueeze(1)
  hidden_states = torch.matmul(probs, v_float).squeeze(1)
  return hidden_states.to(v.dtype)


def ref_apply_output_attn_res(hidden_states, block_residual, proj, norm):
  """Reference PyTorch _apply_output_attn_res (modeling_kimi_linear.py:L1226-L1233)."""
  batch_size, seq_len, hidden_size = hidden_states.shape
  return ref_apply_attn_res(
      hidden_states.view(-1, hidden_size),
      block_residual,
      proj,
      norm,
  ).view(batch_size, seq_len, hidden_size)


# ==============================================================================
# Unit Test Suite
# ==============================================================================
class KimiK3Group4AttnResTest(unittest.TestCase):
  """Comprehensive test suite for Kimi-K3 Group 4 Attention Residuals."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 7168
    self.eps = 1e-5
    self.max_blocks = 8

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    self.hf_config = {
        "hidden_size": self.hidden_size,
        "num_hidden_layers": 93,
        "rms_norm_eps": self.eps,
        "attn_res_block_size": 12,
        "first_k_dense_replace": 1,
    }

  def test_attn_res_state_lifecycle(self):
    """Verifies creation, block appending, prefix sum accumulation, and slicing."""
    state = AttnResState.create(
        batch_size=self.batch_size,
        seq_len=self.seq_len,
        hidden_size=self.hidden_size,
        max_blocks=self.max_blocks,
    )
    self.assertEqual(state.block_residuals.shape, (self.batch_size, self.seq_len, 8, self.hidden_size))
    self.assertIsNone(state.prefix_sum)
    self.assertEqual(state.num_blocks, 0)

    # Initial embedding delta
    x0 = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    state = state.update_prefix_sum(x0)
    self.assertIsNotNone(state.prefix_sum)
    np.testing.assert_allclose(state.prefix_sum, x0)

    # Checkpoint block 0 (layer 0 checkpoint)
    state = state.add_block(state.prefix_sum)
    self.assertEqual(state.num_blocks, 1)
    self.assertIsNone(state.prefix_sum)
    np.testing.assert_allclose(state.get_active_blocks()[:, :, 0, :], x0)

    # Accumulate layer 0 attention and mlp deltas
    attn_out = 2.0 * jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    mlp_out = 3.0 * jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    state = state.update_prefix_sum(attn_out)
    state = state.update_prefix_sum(mlp_out)
    np.testing.assert_allclose(state.prefix_sum, attn_out + mlp_out)

    # Checkpoint block 1 (layer 12 checkpoint)
    state = state.add_block(state.prefix_sum)
    self.assertEqual(state.num_blocks, 2)
    self.assertIsNone(state.prefix_sum)
    self.assertEqual(state.get_active_blocks().shape, (self.batch_size, self.seq_len, 2, self.hidden_size))

  def test_apply_attn_res_empty_block_residual(self):
    """Verifies that empty block residuals return prefix_sum unchanged."""
    prefix_sum = np.random.randn(self.batch_size, self.seq_len, self.hidden_size).astype(np.float32)
    norm_w = np.ones((self.hidden_size,), dtype=np.float32)
    proj_w = np.ones((self.hidden_size, 1), dtype=np.float32)

    # None block residuals
    out_none = _apply_attn_res(jnp.array(prefix_sum), None, jnp.array(proj_w), jnp.array(norm_w), self.eps)
    np.testing.assert_allclose(out_none, prefix_sum)

    # Empty tensor [B, S, 0, D]
    empty_blocks = jnp.zeros((self.batch_size, self.seq_len, 0, self.hidden_size), dtype=jnp.float32)
    out_empty = _apply_attn_res(jnp.array(prefix_sum), empty_blocks, jnp.array(proj_w), jnp.array(norm_w), self.eps)
    np.testing.assert_allclose(out_empty, prefix_sum)

  def test_apply_attn_res_numerical_parity(self):
    """Verifies numerical parity with PyTorch reference _apply_attn_res across varying block counts."""
    for num_blocks in [1, 2, 4, 8]:
      with self.subTest(num_blocks=num_blocks):
        # 1. Generate synthetic PyTorch weights and inputs
        pt_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
        pt_proj = nn.Linear(self.hidden_size, 1, bias=False)
        with torch.no_grad():
          pt_norm.weight.copy_(torch.randn_like(pt_norm.weight) * 0.1 + 1.0)
          pt_proj.weight.copy_(torch.randn_like(pt_proj.weight) * 0.02)

        pt_prefix_sum = torch.randn(self.batch_size * self.seq_len, self.hidden_size, dtype=torch.float32)
        pt_block_residuals = torch.randn(
            self.batch_size * self.seq_len, num_blocks, self.hidden_size, dtype=torch.float32
        )

        # 2. PyTorch reference forward pass
        pt_out = ref_apply_attn_res(pt_prefix_sum, pt_block_residuals, pt_proj, pt_norm)
        y_torch = pt_out.view(self.batch_size, self.seq_len, self.hidden_size).detach().cpu().numpy()

        # 3. MaxText forward pass
        norm_w = pt_norm.weight.detach().cpu().numpy()  # pylint: disable=not-callable
        proj_w = pt_proj.weight.detach().cpu().numpy().T  # pylint: disable=not-callable

        jax_prefix_sum = jnp.array(pt_prefix_sum.view(self.batch_size, self.seq_len, self.hidden_size).numpy())
        jax_block_residuals = jnp.array(
            pt_block_residuals.view(self.batch_size, self.seq_len, num_blocks, self.hidden_size).numpy()
        )

        y_jax = _apply_attn_res(
            prefix_sum=jax_prefix_sum,
            block_residual=jax_block_residuals,
            proj_weight=jnp.array(proj_w),
            norm_weight=jnp.array(norm_w),
            epsilon=self.eps,
        )

        np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

  def test_static_padding_and_masking_parity(self):
    """Verifies that static shape padding with masking produces identical outputs to exact slicing."""
    num_blocks = 3
    prefix_sum = np.random.randn(self.batch_size, self.seq_len, self.hidden_size).astype(np.float32)
    full_block_buffer = np.zeros((self.batch_size, self.seq_len, self.max_blocks, self.hidden_size), dtype=np.float32)
    active_blocks = np.random.randn(self.batch_size, self.seq_len, num_blocks, self.hidden_size).astype(np.float32)
    full_block_buffer[:, :, :num_blocks, :] = active_blocks

    norm_w = np.random.randn(self.hidden_size).astype(np.float32)
    proj_w = np.random.randn(self.hidden_size, 1).astype(np.float32)

    # 1. Unpadded dynamic slice
    y_unpadded = _apply_attn_res(
        prefix_sum=jnp.array(prefix_sum),
        block_residual=jnp.array(active_blocks),
        proj_weight=jnp.array(proj_w),
        norm_weight=jnp.array(norm_w),
        epsilon=self.eps,
    )

    # 2. Statically padded with num_blocks
    y_padded = _apply_attn_res(
        prefix_sum=jnp.array(prefix_sum),
        block_residual=jnp.array(full_block_buffer),
        proj_weight=jnp.array(proj_w),
        norm_weight=jnp.array(norm_w),
        epsilon=self.eps,
        num_blocks=num_blocks,
    )

    np.testing.assert_allclose(y_padded, y_unpadded, rtol=1e-5, atol=1e-5)

  def test_apply_output_attn_res_parity(self):
    """Verifies output AttnRes pooling (all 8 checkpointed blocks at backbone output)."""
    pt_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
    pt_proj = nn.Linear(self.hidden_size, 1, bias=False)
    with torch.no_grad():
      pt_norm.weight.copy_(torch.randn_like(pt_norm.weight) * 0.1 + 1.0)
      pt_proj.weight.copy_(torch.randn_like(pt_proj.weight) * 0.02)

    pt_prefix_sum = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    pt_block_residuals = torch.randn(self.batch_size * self.seq_len, 8, self.hidden_size, dtype=torch.float32)

    # PyTorch reference output pooling:
    pt_out = ref_apply_output_attn_res(
        pt_prefix_sum,
        pt_block_residuals,
        pt_proj,
        pt_norm,
    )
    y_torch = pt_out.detach().cpu().numpy()

    # MaxText _apply_output_attn_res
    norm_w = pt_norm.weight.detach().cpu().numpy()  # pylint: disable=not-callable
    proj_w = pt_proj.weight.detach().cpu().numpy().T  # pylint: disable=not-callable

    jax_prefix_sum = jnp.array(pt_prefix_sum.numpy())
    jax_block_residuals = jnp.array(pt_block_residuals.view(self.batch_size, self.seq_len, 8, self.hidden_size).numpy())

    y_jax = _apply_output_attn_res(
        hidden_states=jax_prefix_sum,
        block_residual=jax_block_residuals,
        proj_weight=jnp.array(proj_w),
        norm_weight=jnp.array(norm_w),
        epsilon=self.eps,
    )

    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

  def test_kimi_attn_res_layer_nnx_and_state(self):
    """Verifies KimiAttnResLayer NNX module pooling and AttnResState integration."""
    pt_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
    pt_proj = nn.Linear(self.hidden_size, 1, bias=False)
    with torch.no_grad():
      pt_norm.weight.copy_(torch.randn_like(pt_norm.weight) * 0.1 + 1.0)
      pt_proj.weight.copy_(torch.randn_like(pt_proj.weight) * 0.02)

    pt_prefix_sum = torch.randn(self.batch_size * self.seq_len, self.hidden_size, dtype=torch.float32)
    pt_block_residuals = torch.randn(self.batch_size * self.seq_len, 2, self.hidden_size, dtype=torch.float32)

    pt_out = ref_apply_attn_res(pt_prefix_sum, pt_block_residuals, pt_proj, pt_norm)
    y_torch = pt_out.view(self.batch_size, self.seq_len, self.hidden_size).detach().cpu().numpy()

    # Instantiate NNX module and load weights
    layer = KimiAttnResLayer(hidden_size=self.hidden_size, epsilon=self.eps)
    layer.norm_scale[...] = jnp.array(pt_norm.weight.detach().cpu().numpy())  # pylint: disable=not-callable
    layer.proj_kernel[...] = jnp.array(pt_proj.weight.detach().cpu().numpy().T)  # pylint: disable=not-callable

    jax_prefix_sum = jnp.array(pt_prefix_sum.view(self.batch_size, self.seq_len, self.hidden_size).numpy())
    jax_block_residuals = jnp.array(pt_block_residuals.view(self.batch_size, self.seq_len, 2, self.hidden_size).numpy())

    # Forward via pool
    y_jax = layer.pool(jax_block_residuals, jax_prefix_sum)
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

    # Forward via __call__
    y_call = layer(jax_prefix_sum, jax_block_residuals)
    np.testing.assert_allclose(y_call, y_torch, rtol=1e-4, atol=1e-4)

    # Forward via apply_to_state
    state = AttnResState.create(self.batch_size, self.seq_len, self.hidden_size, max_blocks=8)
    state = state.add_block(jax_block_residuals[:, :, 0, :])
    state = state.add_block(jax_block_residuals[:, :, 1, :])
    state = state.update_prefix_sum(jax_prefix_sum)
    y_state = layer.apply_to_state(state)
    np.testing.assert_allclose(y_state, y_torch, rtol=1e-4, atol=1e-4)

  def test_jit_compilation(self):
    """Verifies that _apply_attn_res is JIT-compilable without tracing issues."""
    norm_w = jnp.ones((self.hidden_size,), dtype=jnp.float32)
    proj_w = jnp.ones((self.hidden_size, 1), dtype=jnp.float32)
    prefix_sum = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    block_residuals = jnp.ones((self.batch_size, self.seq_len, 4, self.hidden_size), dtype=jnp.float32)

    jitted_fn = jax.jit(_apply_attn_res)
    out = jitted_fn(prefix_sum, block_residuals, proj_w, norm_w, self.eps)
    self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

  def test_group4_attention_residuals_parity(self):
    """Verifies AttnRes dynamic pooling and block residual carry against PyTorch reference via conversion pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup PyTorch reference modules (KimiRMSNorm and Linear proj)
    pt_self_attn_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
    pt_self_attn_proj = nn.Linear(self.hidden_size, 1, bias=False)
    pt_mlp_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
    pt_mlp_proj = nn.Linear(self.hidden_size, 1, bias=False)
    pt_output_norm = RefKimiRMSNorm(self.hidden_size, eps=self.eps)
    pt_output_proj = nn.Linear(self.hidden_size, 1, bias=False)

    with torch.no_grad():
      pt_self_attn_norm.weight.copy_(torch.randn_like(pt_self_attn_norm.weight) * 0.1 + 1.0)
      pt_self_attn_proj.weight.copy_(torch.randn_like(pt_self_attn_proj.weight) * 0.02)
      pt_mlp_norm.weight.copy_(torch.randn_like(pt_mlp_norm.weight) * 0.1 + 1.0)
      pt_mlp_proj.weight.copy_(torch.randn_like(pt_mlp_proj.weight) * 0.02)
      pt_output_norm.weight.copy_(torch.randn_like(pt_output_norm.weight) * 0.1 + 1.0)
      pt_output_proj.weight.copy_(torch.randn_like(pt_output_proj.weight) * 0.02)

    # Build PyTorch state dict for layer 0 and output
    pt_state_dict = {
        "layers.0.self_attention_res_norm.weight": pt_self_attn_norm.weight,
        "layers.0.self_attention_res_proj.weight": pt_self_attn_proj.weight,
        "layers.0.mlp_res_norm.weight": pt_mlp_norm.weight,
        "layers.0.mlp_res_proj.weight": pt_mlp_proj.weight,
        "output_attn_res_norm.weight": pt_output_norm.weight,
        "output_attn_res_proj.weight": pt_output_proj.weight,
    }

    # 2. Convert PyTorch weights to MaxText format via param_mapping
    converted_params = convert_pytorch_module_to_maxtext_params(pt_state_dict, self.hf_config, None)

    # Verify converted parameter shapes
    self.assertEqual(
        converted_params["params-decoder-layers_0-self_attention_res_norm-scale"].shape,
        (self.hidden_size,),
    )
    self.assertEqual(
        converted_params["params-decoder-layers_0-self_attention_res_proj-kernel"].shape,
        (self.hidden_size, 1),
    )
    self.assertEqual(
        converted_params["params-decoder-output_attn_res_proj-kernel"].shape,
        (self.hidden_size, 1),
    )

    # 3. Test self-attention AttnRes pooling (with 2 checkpointed blocks)
    pt_prefix_sum = torch.randn(self.batch_size * self.seq_len, self.hidden_size, dtype=torch.float32)
    pt_block_residuals = torch.randn(self.batch_size * self.seq_len, 2, self.hidden_size, dtype=torch.float32)

    pt_out_attn = ref_apply_attn_res(pt_prefix_sum, pt_block_residuals, pt_self_attn_proj, pt_self_attn_norm)
    y_torch_attn = pt_out_attn.view(self.batch_size, self.seq_len, self.hidden_size).detach().cpu().numpy()

    # MaxText KimiAttnResLayer forward pass
    jax_prefix_sum = jnp.array(pt_prefix_sum.view(self.batch_size, self.seq_len, self.hidden_size).numpy())
    jax_block_residuals = jnp.array(pt_block_residuals.view(self.batch_size, self.seq_len, 2, self.hidden_size).numpy())

    layer_self_attn = KimiAttnResLayer(hidden_size=self.hidden_size, epsilon=self.eps)
    layer_self_attn.norm_scale[...] = jnp.array(converted_params["params-decoder-layers_0-self_attention_res_norm-scale"])
    layer_self_attn.proj_kernel[...] = jnp.array(
        converted_params["params-decoder-layers_0-self_attention_res_proj-kernel"]
    )

    y_jax_attn = layer_self_attn.pool(jax_block_residuals, jax_prefix_sum)
    np.testing.assert_allclose(y_jax_attn, y_torch_attn, rtol=1e-4, atol=1e-4)

    # 4. Test MLP AttnRes pooling
    pt_out_mlp = ref_apply_attn_res(pt_prefix_sum, pt_block_residuals, pt_mlp_proj, pt_mlp_norm)
    y_torch_mlp = pt_out_mlp.view(self.batch_size, self.seq_len, self.hidden_size).detach().cpu().numpy()

    layer_mlp = KimiAttnResLayer(hidden_size=self.hidden_size, epsilon=self.eps)
    layer_mlp.norm_scale[...] = jnp.array(converted_params["params-decoder-layers_0-mlp_res_norm-scale"])
    layer_mlp.proj_kernel[...] = jnp.array(converted_params["params-decoder-layers_0-mlp_res_proj-kernel"])

    y_jax_mlp = layer_mlp.pool(jax_block_residuals, jax_prefix_sum)
    np.testing.assert_allclose(y_jax_mlp, y_torch_mlp, rtol=1e-4, atol=1e-4)

    # 5. Test Output AttnRes pooling (all 8 checkpointed blocks)
    pt_final_residuals = torch.randn(self.batch_size * self.seq_len, 8, self.hidden_size, dtype=torch.float32)
    pt_out_final = ref_apply_attn_res(pt_prefix_sum, pt_final_residuals, pt_output_proj, pt_output_norm)
    y_torch_final = pt_out_final.view(self.batch_size, self.seq_len, self.hidden_size).detach().cpu().numpy()

    jax_final_residuals = jnp.array(pt_final_residuals.view(self.batch_size, self.seq_len, 8, self.hidden_size).numpy())
    layer_output = KimiAttnResLayer(hidden_size=self.hidden_size, epsilon=self.eps)
    layer_output.norm_scale[...] = jnp.array(converted_params["params-decoder-output_attn_res_norm-scale"])
    layer_output.proj_kernel[...] = jnp.array(converted_params["params-decoder-output_attn_res_proj-kernel"])

    y_jax_final = layer_output.pool(jax_final_residuals, jax_prefix_sum)
    np.testing.assert_allclose(y_jax_final, y_torch_final, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
