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

"""Numerics tests for gdn_chunk_impl="fast" (jax_chunk_gated_delta_rule_fast)."""

import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path


def _inputs(batch, seq, heads, k_dim, v_dim, dtype, seed=0):
  rng = np.random.default_rng(seed)
  q = rng.standard_normal((batch, seq, heads, k_dim)) / np.sqrt(k_dim)
  k = rng.standard_normal((batch, seq, heads, k_dim)) / np.sqrt(k_dim)
  v = rng.standard_normal((batch, seq, heads, v_dim)) / np.sqrt(v_dim)
  g = -np.abs(rng.standard_normal((batch, seq, heads)))
  beta = rng.uniform(0.1, 0.9, size=(batch, seq, heads))
  h0 = 0.1 * rng.standard_normal((batch, heads, k_dim, v_dim))
  return (
      jnp.asarray(q, dtype),
      jnp.asarray(k, dtype),
      jnp.asarray(v, dtype),
      jnp.asarray(g, jnp.float32),
      jnp.asarray(beta, dtype),
      jnp.asarray(h0, jnp.float32),
  )


class InvertUnitLowerTriangularTest(parameterized.TestCase):

  @parameterized.parameters(
      (64, 1),
      (64, 8),
      (64, 16),
      (64, 64),
      # Odd and non-power-of-two sizes split unevenly (N // 2 != N - N // 2).
      (37, 1),
      (37, 8),
      (50, 16),
      (3, 1),
  )
  def test_matches_solve_triangular(self, n, block_size):
    rng = np.random.default_rng(1)
    strict_lower = np.tril(rng.standard_normal((3, 4, n, n)) * 0.1, k=-1)
    l_mat = jnp.asarray(np.eye(n) + strict_lower, jnp.float32)
    expected = jax.scipy.linalg.solve_triangular(
        l_mat, jnp.broadcast_to(jnp.eye(n, dtype=jnp.float32), l_mat.shape), lower=True, unit_diagonal=True
    )
    actual = qwen3.invert_unit_lower_triangular(l_mat, block_size=block_size)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class FastChunkGatedDeltaRuleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("divisible", 256, 64),
      ("chunk128", 256, 128),
      ("padded", 300, 64),
  )
  def test_fp32_matches_default(self, seq, chunk):
    """With fp32 compute both implementations are the same math, so they agree tightly."""
    q, k, v, g, beta, h0 = _inputs(2, seq, 4, 32, 32, jnp.float32)
    kwargs = {"chunk_size": chunk, "initial_state": h0, "compute_dtype": jnp.float32}
    ref_out, ref_state = qwen3.jax_chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    out, state = qwen3.jax_chunk_gated_delta_rule_fast(q, k, v, g, beta, **kwargs)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state, ref_state, rtol=1e-4, atol=1e-4)

  def test_fp32_gradients_match_default(self):
    q, k, v, g, beta, h0 = _inputs(1, 128, 2, 32, 32, jnp.float32)

    def loss(fn, *args):
      out, state = fn(*args, chunk_size=64, initial_state=h0, compute_dtype=jnp.float32)
      return jnp.sum(out**2) + jnp.sum(state**2)

    ref = jax.grad(lambda *a: loss(qwen3.jax_chunk_gated_delta_rule, *a), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    fast = jax.grad(lambda *a: loss(qwen3.jax_chunk_gated_delta_rule_fast, *a), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for name, r, f in zip(("q", "k", "v", "g", "beta"), ref, fast):
      np.testing.assert_allclose(f, r, rtol=1e-3, atol=1e-4, err_msg=name)

  def test_bf16_close_to_fp32_reference(self):
    """bf16 matmul inputs (fp32 accumulation) stay within bf16 rounding of the fp32 result."""
    q, k, v, g, beta, h0 = _inputs(2, 256, 4, 64, 64, jnp.bfloat16)
    ref_out, ref_state = qwen3.jax_chunk_gated_delta_rule(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        g,
        beta.astype(jnp.float32),
        chunk_size=64,
        initial_state=h0,
        compute_dtype=jnp.float32,
    )
    out, state = qwen3.jax_chunk_gated_delta_rule_fast(
        q, k, v, g, beta, chunk_size=64, initial_state=h0, compute_dtype=jnp.bfloat16
    )
    scale = float(jnp.max(jnp.abs(ref_out)))
    np.testing.assert_allclose(out.astype(jnp.float32), ref_out, atol=3e-2 * scale)
    state_scale = float(jnp.max(jnp.abs(ref_state)))
    np.testing.assert_allclose(state.astype(jnp.float32), ref_state, atol=3e-2 * state_scale)

  def test_fp32_qk_norm_without_initial_state_matches_default(self):
    q, k, v, g, beta, _ = _inputs(2, 192, 2, 32, 32, jnp.float32)
    kwargs = {"chunk_size": 64, "initial_state": None, "use_qk_norm_in_gdn": True, "compute_dtype": jnp.float32}
    ref_out, ref_state = qwen3.jax_chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    out, state = qwen3.jax_chunk_gated_delta_rule_fast(q, k, v, g, beta, **kwargs)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    # Without an initial state neither implementation returns a final state.
    self.assertIsNone(ref_state)
    self.assertIsNone(state)


def _gdn_layer_config(**overrides):
  """A minimal fp32 Qwen3-Next config for a single-device CPU run."""
  argv = [
      None,
      get_test_config_path(),
      "run_name=gdn_fast_layer_test",
      "dtype=float32",
      "weight_dtype=float32",
      "decoder_block=qwen3_next",
      "attention=dot_product",
      "base_emb_dim=64",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "head_dim=32",
      "gdn_num_value_heads=2",
      "gdn_num_key_heads=2",
      "gdn_key_head_dim=32",
      "gdn_value_head_dim=32",
      "gdn_conv_kernel_dim=4",
      "gdn_chunk_size=64",
  ]
  argv += [f"{k}={v}" for k, v in overrides.items()]
  return pyconfig.initialize(argv)


class GatedDeltaNetLayerImplTest(unittest.TestCase):
  """`gdn_chunk_impl` is honoured by `Qwen3NextGatedDeltaNet` on the mesh (shard_map) training path."""

  def test_fast_matches_default(self):
    hidden = jnp.asarray(np.random.default_rng(3).standard_normal((2, 128, 64)) * 0.5, jnp.float32)
    outputs = {}
    for impl in ("default", "fast"):
      cfg = _gdn_layer_config(gdn_chunk_impl=impl)
      mesh = Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
      # Same seed -> identical parameters for both implementations.
      layer = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh, rngs=nnx.Rngs(0), inputs_shape=hidden.shape)
      out = layer(hidden)
      outputs[impl] = out[0] if isinstance(out, tuple) else out
    np.testing.assert_allclose(outputs["fast"], outputs["default"], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
