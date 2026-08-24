# Copyright 2026 Google LLC
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

"""Unit tests for Kimi Decoupled Attention (KDA) in MaxText."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from maxtext.configs import pyconfig
from maxtext.layers.kda import KimiDecoupledAttention, ShortConv1D, kda_recurrent_kernel


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
  """Self-contained PyTorch reference for KDA recurrent attention."""
  if scale is None:
    scale = q.shape[-1] ** -0.5
  q = q * scale
  B, T, H, K = q.shape
  V = v.shape[-1]
  S = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device) if initial_state is None else initial_state
  outputs = []
  for i in range(T):
    q_i = q[:, i]
    k_i = k[:, i]
    v_i = v[:, i]
    g_i = g[:, i]
    b_i = beta[:, i]

    S = S * torch.exp(g_i).unsqueeze(-1)
    k_S = torch.sum(k_i.unsqueeze(-1) * S, dim=-2)
    v_diff = v_i - k_S
    bk = b_i.unsqueeze(-1) * k_i
    S = S + bk.unsqueeze(-1) * v_diff.unsqueeze(-2)
    o_i = torch.sum(q_i.unsqueeze(-1) * S, dim=-2)
    outputs.append(o_i)

  o = torch.stack(outputs, dim=1)
  if output_final_state:
    return o, S
  return o


def test_short_conv1d_shape_and_causality():
  """Test that ShortConv1D preserves shape and is strictly causal."""
  rngs = nnx.Rngs(0)
  conv = ShortConv1D(features=16, kernel_size=4, rngs=rngs)

  # Shape check
  x = jnp.ones((2, 10, 16))
  out, state = conv(x)
  assert out.shape == (2, 10, 16)
  assert state.shape == (2, 3, 16)

  # Causality check: changing x at t=5 should not affect out at t=0..4
  x1 = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 16))
  x2 = x1.at[:, 5:, :].add(10.0)

  out1, _ = conv(x1)
  out2, _ = conv(x2)

  np.testing.assert_allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)


def test_short_conv1d_autoregressive_caching():
  """Test that step-by-step decoding with conv_state matches sequence-level convolution."""
  rngs = nnx.Rngs(0)
  conv = ShortConv1D(features=16, kernel_size=4, rngs=rngs)
  x_seq = jax.random.normal(jax.random.PRNGKey(42), (2, 8, 16))

  # 1. Full sequence forward pass
  out_seq, final_conv_state = conv(x_seq)

  # 2. Step-by-step autoregressive forward pass
  step_outputs = []
  conv_state = None
  for t in range(8):
    x_t = x_seq[:, t : t + 1, :]
    out_t, conv_state = conv(x_t, conv_state=conv_state)
    step_outputs.append(out_t)
  out_steps = jnp.concatenate(step_outputs, axis=1)

  np.testing.assert_allclose(out_seq, out_steps, atol=1e-6)
  np.testing.assert_allclose(final_conv_state, conv_state, atol=1e-6)


@pytest.mark.parametrize("T", [1, 16, 64, 128])

def test_kda_recurrent_kernel_parity_with_fla(T):
  """Test kda_recurrent_kernel against fla naive_recurrent_kda."""
  np.random.seed(42)
  B, H, K, HV, V = 2, 4, 32, 4, 32
  A_log_np = np.random.uniform(1, 4, (H,)).astype(np.float32)
  dt_bias_np = np.random.randn(H * K).astype(np.float32).reshape(H, K)

  q_np = np.random.randn(B, T, H, K).astype(np.float32)
  k_np = np.random.randn(B, T, H, K).astype(np.float32)
  # L2-normalize q and k as defined in KDA
  q_np = q_np / np.maximum(np.linalg.norm(q_np, axis=-1, keepdims=True), 1e-6)
  k_np = k_np / np.maximum(np.linalg.norm(k_np, axis=-1, keepdims=True), 1e-6)

  v_np = np.random.randn(B, T, HV, V).astype(np.float32)
  g_raw_np = np.random.randn(B, T, HV, K).astype(np.float32)
  beta_np = 1.0 / (1.0 + np.exp(-np.random.randn(B, T, HV).astype(np.float32)))

  # Compute g_np using Kimi K3 decay formula: g = gmin * Sigmoid(exp(A_log) * (g_raw + dt_bias))
  g_np = -5.0 / (1.0 + np.exp(-np.exp(A_log_np)[None, None, :, None] * (g_raw_np + dt_bias_np[None, None, :, :])))

  # PyTorch
  o_pt, S_pt = naive_recurrent_kda(
      torch.from_numpy(q_np),
      torch.from_numpy(k_np),
      torch.from_numpy(v_np),
      torch.from_numpy(g_np),
      torch.from_numpy(beta_np),
      output_final_state=True,
  )

  # JAX
  o_jax, S_jax = kda_recurrent_kernel(
      jnp.array(q_np),
      jnp.array(k_np),
      jnp.array(v_np),
      jnp.array(g_np),
      jnp.array(beta_np),
  )

  max_diff_o = np.max(np.abs(o_pt.numpy() - np.array(o_jax)))
  max_diff_S = np.max(np.abs(S_pt.numpy() - np.array(S_jax)))

  assert max_diff_o < 1e-4, f"o Max diff too large for T={T}: {max_diff_o}"
  assert max_diff_S < 1e-4, f"S Max diff too large for T={T}: {max_diff_S}"


def test_kimi_decoupled_attention_module():
  """Test KimiDecoupledAttention NNX module initialization and forward pass."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
  ])

  rngs = nnx.Rngs(0)
  kda = KimiDecoupledAttention(cfg, layer_idx=0, rngs=rngs)

  # Forward pass check
  x = jnp.ones((2, 8, cfg.emb_dim))
  out, final_state = kda(x)

  assert out.shape == (2, 8, cfg.emb_dim)

  assert final_state.shape == (2, cfg.num_query_heads, cfg.head_dim, cfg.head_dim)
  assert not jnp.isnan(out).any()
  assert not jnp.isnan(final_state).any()
