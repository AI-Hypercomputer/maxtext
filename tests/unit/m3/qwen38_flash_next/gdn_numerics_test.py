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

"""Numerical verification for GDN chunked scan at long sequence lengths."""

from typing import Any
import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import gdn_recurrence  # pylint: disable=no-name-in-module
import numpy as np
import pytest

torch: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  torch = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


@pytest.mark.cpu_only
class Qwen38GDNNumericsTest(absltest.TestCase):
  """Validates numerical stability and PyTorch parity at long sequences."""

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_chunked_scan_numerics_long_seq(self):
    """Verifies JAX vs PyTorch chunked scan parity at seq_len=8192."""
    if torch is None or modeling_qwen4_exp is None:
      self.fail("torch and modeling_qwen4_exp must not be None")
    rng = np.random.default_rng(2026)
    b, seq_len, h, d_k, d_v = 1, 8192, 2, 32, 32
    q, k, v = (rng.standard_normal((b, seq_len, h, d), dtype=np.float32) * 0.1 for d in (d_k, d_k, d_v))
    beta = rng.uniform(0.1, 0.9, size=(b, seq_len, h)).astype(np.float32)
    g = -rng.uniform(0.01, 0.2, size=(b, seq_len, h)).astype(np.float32)

    t_inputs = tuple(torch.from_numpy(x) for x in (q, k, v, g, beta))
    pt_out, pt_state = modeling_qwen4_exp.torch_chunk_gated_delta_rule(
        *t_inputs, chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    j_inputs = tuple(jnp.array(x) for x in (q, k, v, g, beta))
    jax_out, jax_state = gdn_recurrence.jax_chunk_gated_delta_rule(*j_inputs, chunk_size=64, output_final_state=True)
    self.assertIsNotNone(jax_state)
    self.assertIsNotNone(pt_state)
    if jax_state is None or pt_state is None:
      self.fail("jax_state and pt_state must not be None")
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_state), pt_state.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_chunk_gated_delta_rule_backward_grad_finite_large_decay(self):
    """Checks backward grad through chunked scan is finite under large decay."""
    rng = np.random.default_rng(2026)
    b, s, h, d_k, d_v = 1, 128, 2, 32, 32
    q, k, v = (jnp.array(rng.standard_normal((b, s, h, d), dtype=np.float32) * 0.1) for d in (d_k, d_k, d_v))
    beta = jnp.array(rng.uniform(0.1, 0.9, size=(b, s, h)).astype(np.float32))
    g = jnp.full((b, s, h), -5.0, dtype=jnp.float32)

    def loss_fn(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array, g_in: jax.Array, beta_in: jax.Array) -> jax.Array:
      out, _ = gdn_recurrence.jax_chunk_gated_delta_rule(q_in, k_in, v_in, g_in, beta_in, chunk_size=64)
      return jnp.sum(out)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for grad in grads:
      self.assertTrue(np.all(np.isfinite(np.array(grad))))


if __name__ == "__main__":
  absltest.main()
