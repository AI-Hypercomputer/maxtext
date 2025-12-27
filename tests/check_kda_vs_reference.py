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
Tests for KDA against its PyTorch reference.
"""
import unittest
import os

import torch
import jax
import jax.numpy as jnp
import numpy as np

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import kimi_delta_attention

import torch
from einops import rearrange

def torch_native_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S

def torch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    q, k, v, g, beta = map(lambda x: rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float), [q, k, v, g, beta])
    q = q * scale
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, H, BT, ...]
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        A = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            A[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        A = A.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


class TestQwen3Next(unittest.TestCase):
  """Main test class for Qwen3-Next layers."""

  def setUp(self):
    """Set up a complete configuration and test environment for all Qwen3-Next tests."""
    super().setUp()
    # This setup now includes all necessary parameters for both linear attention and MoE tests.
    self.cfg = pyconfig.initialize(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "kda_num_value_heads=4",
            "kda_num_key_heads=4",
            "kda_key_head_dim=32",
            "kda_value_head_dim=32",
            "kda_conv_kernel_dim=4",
            "kda_chunk_size=64",
            "normalization_layer_epsilon=1e-6",
        ]
    )
    torch.manual_seed(42)
    np.random.seed(42)
    self.batch_size = 4
    self.seq_len = 128
    print("setUp complete!")

  def test_kda_precision(self):
    """
    Directly tests the `jax_chunk_kda` against the original PyTorch reference.
    """
    print("Running test_kda_precision...")
    # Use renamed config parameters
    num_heads = self.cfg.kda_num_value_heads
    k_head_dim = self.cfg.kda_key_head_dim
    v_head_dim = self.cfg.kda_value_head_dim
    chunk_size = self.cfg.kda_chunk_size

    key = jax.random.PRNGKey(42)
    key_q, key_k, key_v, key_g, key_beta = jax.random.split(key, 5)

    scale_factor = 0.1

    # Shapes are (B, S, H, D)
    q_jax = (
        jax.random.normal(
            key_q,
            (self.batch_size, self.seq_len, num_heads, k_head_dim),
            dtype=jnp.float32,
        ) * scale_factor
    )
    k_jax = (
        jax.random.normal(
            key_k,
            (self.batch_size, self.seq_len, num_heads, k_head_dim),
            dtype=jnp.float32,
        ) * scale_factor
    )
    v_jax = (
        jax.random.normal(
            key_v,
            (self.batch_size, self.seq_len, num_heads, v_head_dim),
            dtype=jnp.float32,
        ) * scale_factor
    )
    initial_state_jax = (
        jax.random.normal(
            key_v,
            (self.batch_size, num_heads, k_head_dim, v_head_dim),
            dtype=jnp.float32,
        )
    )
    g_jax = jax.random.normal(key_g, (self.batch_size, self.seq_len, num_heads, k_head_dim), dtype=jnp.float32) * scale_factor
    beta_jax = jax.random.uniform(key_beta, (self.batch_size, self.seq_len, num_heads), dtype=jnp.float32)

    q_torch = torch.from_numpy(np.asarray(q_jax).copy())
    k_torch = torch.from_numpy(np.asarray(k_jax).copy())
    v_torch = torch.from_numpy(np.asarray(v_jax).copy())
    g_torch = torch.from_numpy(np.asarray(g_jax).copy())
    beta_torch = torch.from_numpy(np.asarray(beta_jax).copy())
    initial_state_torch = torch.from_numpy(np.asarray(initial_state_jax).copy())

    target_atol = 1e-6
    target_rtol = 1e-6

    torch_chunk_output, _ = torch_chunk_kda(
        q_torch.clone(),
        k_torch.clone(),
        v_torch.clone(),
        g_torch.clone(),
        beta_torch.clone(),
        chunk_size=chunk_size,
        initial_state=initial_state_torch,
        output_final_state=False,
    )
    torch_native_output, _ = torch_native_kda(
        q_torch.clone(),
        k_torch.clone(),
        v_torch.clone(),
        g_torch.clone(),
        beta_torch.clone(),
        initial_state=initial_state_torch,
        output_final_state=False,
    )
    jax_output, _  = kimi_delta_attention.chunk_parallel_delta_attention(
        q_jax,
        k_jax,
        v_jax,
        g_jax,
        beta_jax,
        chunk_size=chunk_size,
        initial_state=initial_state_jax,
    )
    np.testing.assert_allclose(
        torch_chunk_output.detach().numpy(),
        torch_native_output.detach().numpy(),
        atol=target_atol,
        rtol=target_rtol,
        err_msg=f"PyTorch Chunk and Native outputs are NOT close within atol={target_atol}, rtol={target_rtol}!",
    )
    np.testing.assert_allclose(
        torch_native_output.detach().numpy(),
        np.asarray(jax_output),
        atol=target_atol,
        rtol=target_rtol,
        err_msg=f"JAX and PyTorch outputs are NOT close within atol={target_atol}, rtol={target_rtol}!",
    )
    print("test_kda_precision passed!")

if __name__ == "__main__":
  unittest.main()
