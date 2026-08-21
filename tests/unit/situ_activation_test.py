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

"""Unit tests for SituAndMul activation in MaxText."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from maxtext.layers.linears import _convert_to_activation_function, situ_and_mul


class PyTorchSituAndMul(torch.nn.Module):
  """PyTorch reference implementation of SituAndMul from MoonshotAI Kimi-K3."""

  def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
    super().__init__()
    self.beta = beta
    self.linear_beta = linear_beta

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
    if self.linear_beta is not None:
      up = self.linear_beta * torch.tanh(up / self.linear_beta)
    return (situ_a * up).to(x.dtype)


@pytest.mark.parametrize("beta,linear_beta", [(4.0, 25.0), (1.0, None), (2.0, 10.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_situ_and_mul_parity(beta, linear_beta, dtype):
  """Test JAX situ_and_mul against PyTorch reference for various parameters and dtypes."""
  np.random.seed(42)
  x_np = np.random.randn(2, 4, 128).astype(np.float32)

  # PyTorch
  pt_act = PyTorchSituAndMul(beta=beta, linear_beta=linear_beta)
  pt_dtype = torch.bfloat16 if dtype == jnp.bfloat16 else torch.float32
  pt_out = pt_act(torch.from_numpy(x_np).to(pt_dtype)).to(torch.float32).numpy()

  # JAX
  jax_x = jnp.array(x_np, dtype=dtype)
  jax_out = np.array(situ_and_mul(jax_x, beta=beta, linear_beta=linear_beta).astype(jnp.float32))

  # Compare
  max_diff = np.max(np.abs(pt_out - jax_out))
  threshold = 1e-3 if dtype == jnp.bfloat16 else 1e-6
  assert max_diff < threshold, f"Parity check failed for beta={beta}, linear_beta={linear_beta}, dtype={dtype}: max_diff={max_diff}"


def test_convert_to_activation_function_situ():
  """Test that _convert_to_activation_function resolves 'situ' to situ_and_mul."""
  act_fn = _convert_to_activation_function("situ")
  assert act_fn is situ_and_mul

  # Verify it can be called
  x = jnp.ones((2, 4))
  out = act_fn(x)
  assert out.shape == (2, 2)
