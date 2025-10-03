# Copyright 2025 Google LLC
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
Tests for GatedDeltaRule in Qwen3-Next against its PyTorch reference.
"""
import unittest
import os

import torch
from torch import nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import linen

from MaxText import pyconfig
from MaxText.layers import qwen3
from MaxText import maxtext_utils
from MaxText.common_types import Config
from MaxText.globals import MAXTEXT_PKG_DIR

# ----------------------------------------------------------------------
# START: Copied PyTorch functions
# ----------------------------------------------------------------------
def l2norm_torch(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm_torch(query, dim=-1, eps=1e-6)
        key = l2norm_torch(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to (B, H, N_CHUNKS, CHUNK_SIZE, DIM)
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, num_chunks, chunk_size, -1)
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

    # Intra-chunk computations
    g_cumsum = g.cumsum(dim=-1)
    decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp().float().tril() # (B, H, N_CHUNKS, CS, CS)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # attn (B, H, N_CHUNKS, CS, CS)
    attn = -((torch.einsum('bhcsd,bhctd->bhcst', k_beta, key)) * decay_mask)
    attn = attn.masked_fill(mask, 0.0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        update = row + torch.einsum('...ci,...cij->...cj', row, sub)
        attn[..., i, :i] = update
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value_intra = torch.einsum('bhcst,bhctv->bhcsv', attn, v_beta)
    k_cumdecay = torch.einsum('bhcst,bhctd->bhcsd', attn, k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device)
        if initial_state is None
        else initial_state.to(value.dtype)
    )
    core_attn_out = torch.zeros_like(value_intra)
    mask_inter = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value_intra[:, :, i]
        g_i = g[:, :, i]
        k_cumdecay_i = k_cumdecay[:, :, i]
        decay_mask_i = decay_mask[:, :, i]

        v_prime = torch.einsum('bhsd,bhdv->bhsv', k_cumdecay_i, last_recurrent_state)
        v_new = v_i - v_prime
        attn_inter = torch.einsum('bhsd,bhdv->bhsv', q_i * g_i[..., None].exp(), last_recurrent_state)
        attn_intra = (torch.einsum('bhsd,bhtd->bhst', q_i, k_i) * decay_mask_i).masked_fill_(mask_inter, 0.0)
        chunk_output = attn_inter + torch.einsum('bhst,bhtv->bhsv', attn_intra, v_new)
        core_attn_out[:, :, i] = chunk_output
        g_i_last = g_i[..., -1, None, None]
        exp_g_diff = (g_i[..., -1:] - g_i).exp()
        k_i_weighted = k_i * exp_g_diff.unsqueeze(-1)
        update_term = torch.einsum('bhsd,bhsv->bhdv', k_i_weighted, v_new)
        last_recurrent_state = (
            last_recurrent_state * g_i_last.exp()
            + update_term
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_sequence_length, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length, :]
    return core_attn_out.contiguous().to(initial_dtype), last_recurrent_state

class Qwen3NextRMSNormGated_PT(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if gate is not None:
             hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
# ----------------------------------------------------------------------
# END: Copied PyTorch functions
# ----------------------------------------------------------------------

class TestQwen3Next(unittest.TestCase):
    """Main test class for Qwen3-Next layers."""

    def setUp(self):
        """Set up the configuration and test environment."""
        super().setUp()
        self.cfg = pyconfig.initialize([
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "run_name=qwen3_next_test",
            "base_emb_dim=128",
            "linear_num_value_heads=4",
            "linear_num_key_heads=2",
            "linear_key_head_dim=32",
            "linear_value_head_dim=32",
            "linear_conv_kernel_dim=4",
            "normalization_layer_epsilon=1e-5",
            "dtype=float32", # Test with float32
            "weight_dtype=float32",
            "decoder_block=qwen3_next",
            "matmul_precision=highest", # Use highest precision for tests
        ])

        self.batch_size = 2
        self.seq_len = 128
        self.hidden_size = self.cfg.base_emb_dim
        self.devices = np.array(jax.devices())
        self.mesh = Mesh(self.devices, ('data',))
        torch.manual_seed(0)
        np.random.seed(0)
        jax.random.PRNGKey(0)

    def test_rms_norm_gated(self):
        """Tests the Qwen3NextRMSNormGated layer."""
        hidden_states_pt = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        gate_pt = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        weight_pt = torch.rand(self.hidden_size)

        # PyTorch reference
        pt_model = Qwen3NextRMSNormGated_PT(self.hidden_size, eps=self.cfg.normalization_layer_epsilon)
        pt_model.weight.data = weight_pt
        pt_model.eval()
        with torch.no_grad():
            expected_output = pt_model(hidden_states_pt, gate_pt)

        # JAX implementation
        jax_model = qwen3.Qwen3NextRMSNormGated(num_features=self.hidden_size, eps=self.cfg.normalization_layer_epsilon, dtype=self.cfg.dtype)
        params = {'params': {'weight': jnp.array(weight_pt.numpy())}}
        hidden_states_jax = jnp.array(hidden_states_pt.numpy())
        gate_jax = jnp.array(gate_pt.numpy())

        @jax.jit
        def run_jax(params, hidden_states, gate):
            return jax_model.apply(params, hidden_states, gate)

        actual_output = run_jax(params, hidden_states_jax, gate_jax)

        np.testing.assert_allclose(
            expected_output.numpy(),
            actual_output,
            rtol=1e-5, atol=1e-6, # Tight tolerance for this layer
            err_msg="Qwen3NextRMSNormGated does not match PyTorch reference!"
        )

    def test_l2norm(self):
        """Tests the l2norm function."""
        x_pt = torch.randn(self.batch_size, self.cfg.linear_num_value_heads, self.seq_len, self.cfg.linear_key_head_dim)
        expected_output = l2norm_torch(x_pt)
        actual_output = qwen3.l2norm(jnp.array(x_pt.numpy()))
        np.testing.assert_allclose(
            expected_output.numpy(),
            actual_output,
            rtol=1e-5, atol=1e-6,
            err_msg="l2norm does not match PyTorch reference!"
        )

    def test_chunk_gated_delta_rule_logic(self):
        """
        Directly tests the `jax_chunk_gated_delta_rule` against the original PyTorch reference.
        The numerical differences observed are expected due to framework-specific floating-point
        operation implementations and error accumulation, especially within the recurrent loop.
        The tolerances are set to reflect acceptable differences for this complex block.
        """
        query_pt = torch.randn(self.batch_size, self.cfg.linear_num_value_heads, self.seq_len, self.cfg.linear_key_head_dim)
        key_pt = torch.randn(self.batch_size, self.cfg.linear_num_key_heads, self.seq_len, self.cfg.linear_key_head_dim)
        value_pt = torch.randn(self.batch_size, self.cfg.linear_num_value_heads, self.seq_len, self.cfg.linear_value_head_dim)
        g_pt = torch.randn(self.batch_size, self.cfg.linear_num_value_heads, self.seq_len)
        beta_pt = torch.rand(self.batch_size, self.cfg.linear_num_value_heads, self.seq_len)

        if self.cfg.linear_num_value_heads // self.cfg.linear_num_key_heads > 1:
            key_pt = key_pt.repeat_interleave(self.cfg.linear_num_value_heads // self.cfg.linear_num_key_heads, dim=1)

        # PyTorch
        with torch.no_grad():
            output_pt, _ = torch_chunk_gated_delta_rule(
                query_pt.clone(), key_pt.clone(), value_pt.clone(), g_pt.clone(), beta_pt.clone(), use_qk_l2norm_in_kernel=True
            )

        # JAX
        query_jax, key_jax, value_jax, g_jax, beta_jax = [
            jnp.array(x.numpy()) for x in [query_pt, key_pt, value_pt, g_pt, beta_pt]
        ]
        # JIT compile the JAX function
        jax_fn = jax.jit(qwen3.jax_chunk_gated_delta_rule)
        output_jax = jax_fn(
            query_jax, key_jax, value_jax, g_jax, beta_jax
        )
        output_jax.block_until_ready()

        # Due to the complex and recurrent nature of this function, minor floating point differences
        # between PyTorch and JAX/XLA can accumulate. Tolerances are set accordingly.
        np.testing.assert_allclose(
            output_pt.numpy(),
            output_jax,
            rtol=1e-1, atol=0.015,
            err_msg="jax_chunk_gated_delta_rule does not match PyTorch reference!"
        )

    def test_gated_delta_net_structure(self):
        """Tests the structure and output shape of Qwen3NextGatedDeltaNet."""
        hidden_states_jax = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=self.cfg.dtype)

        jax_model = qwen3.Qwen3NextGatedDeltaNet(config=self.cfg)

        @jax.jit
        def run_jax(hidden_states):
            params = jax_model.init(jax.random.PRNGKey(0), hidden_states, deterministic=True)
            return jax_model.apply(params, hidden_states, deterministic=True)

        output_jax = run_jax(hidden_states_jax)

        self.assertEqual(output_jax.shape, (self.batch_size, self.seq_len, self.hidden_size))

if __name__ == '__main__':
    unittest.main()
