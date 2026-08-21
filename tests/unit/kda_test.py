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

import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx


from maxtext.configs import pyconfig
from maxtext.layers.kda import KimiDecoupledAttention, ShortConv1D, kda_recurrent_kernel

# Load naive.py directly without triggering fla.ops.__init__ (which requires triton)
spec = importlib.util.spec_from_file_location(
    "kda_naive",
    "/Users/jfacevedo/.gemini/jetski/brain/0487c2aa-4e99-434c-b4e2-9147cc01875b/scratch/venv/lib/python3.12/site-packages/fla/ops/kda/naive.py",
)
kda_naive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kda_naive)
naive_recurrent_kda = kda_naive.naive_recurrent_kda


def test_short_conv1d_shape_and_causality():
  """Test that ShortConv1D preserves shape and is strictly causal."""
  rngs = nnx.Rngs(0)
  conv = ShortConv1D(features=16, kernel_size=4, rngs=rngs)

  # Shape check
  x = jnp.ones((2, 10, 16))
  out = conv(x)
  assert out.shape == (2, 10, 16)

  # Causality check: changing x at t=5 should not affect out at t=0..4
  x1 = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 16))
  x2 = x1.at[:, 5:, :].add(10.0)

  out1 = conv(x1)
  out2 = conv(x2)

  np.testing.assert_allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)


@pytest.mark.parametrize("T", [1, 16, 64, 128])
def test_kda_recurrent_kernel_parity_with_fla(T):
  """Test kda_recurrent_kernel against fla naive_recurrent_kda."""
  np.random.seed(42)
  B, H, K, HV, V = 2, 4, 32, 4, 32
  A_log_np = np.random.uniform(1, 4, (H,)).astype(np.float32)
  dt_bias_np = np.random.randn(H * K).astype(np.float32).reshape(H, K)

  q_np = np.random.randn(B, T, H, K).astype(np.float32)
  k_np = np.random.randn(B, T, H, K).astype(np.float32)
  v_np = np.random.randn(B, T, HV, V).astype(np.float32)
  g_raw_np = np.random.randn(B, T, HV, K).astype(np.float32)
  beta_np = np.random.randn(B, T, HV).astype(np.float32)

  # Compute g_np using Kimi K3 decay formula: g = -exp(A_log) * softplus(g_raw + dt_bias)
  softplus_g = np.log1p(np.exp(g_raw_np + dt_bias_np[None, None, :, :]))
  g_np = -np.exp(A_log_np)[None, None, :, None] * softplus_g

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
