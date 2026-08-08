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

"""Tests that the Qwen3-Next Gated DeltaNet kernels honor gdn_matmul_precision.

The delta-rule matmuls historically pinned `jax.lax.Precision.HIGHEST`. These
tests lock in the opt-in knob: the default reproduces that pinned behavior
exactly, and the configured value actually reaches the matmuls rather than being
accepted and ignored.
"""

import unittest
import jax
import jax.numpy as jnp
from maxtext.models import qwen3


def _inputs(batch=1, seq_len=128, num_heads=2, head_dim=64, seed=0):
  """Returns (query, key, value, g, beta, initial_state) for the delta rule."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  query = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim), jnp.float32)
  key = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim), jnp.float32)
  value = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim), jnp.float32)
  g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq_len, num_heads), jnp.float32))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq_len, num_heads), jnp.float32))
  initial_state = jax.random.normal(keys[5], (batch, num_heads, head_dim, head_dim), jnp.float32) * 0.1
  return query, key, value, g, beta, initial_state


class Qwen3NextGdnPrecisionTest(unittest.TestCase):
  """The delta-rule matmuls must take their precision from the caller."""

  def test_chunk_rule_default_arg_preserves_previous_behavior(self):
    """Omitting matmul_precision must reproduce the previously hardcoded HIGHEST."""
    query, key, value, g, beta, initial_state = _inputs()
    kwargs = {"chunk_size": 64, "use_qk_norm_in_gdn": True, "initial_state": initial_state}

    omitted, omitted_state = qwen3.jax_chunk_gated_delta_rule(query, key, value, g, beta, **kwargs)
    pinned, pinned_state = qwen3.jax_chunk_gated_delta_rule(
        query, key, value, g, beta, matmul_precision="highest", **kwargs
    )

    self.assertTrue(jnp.array_equal(omitted, pinned))
    self.assertTrue(jnp.array_equal(omitted_state, pinned_state))

  def test_ar_rule_default_arg_preserves_previous_behavior(self):
    """The decode path must be unchanged for callers that omit the argument."""
    query, key, value, g, beta, initial_state = _inputs(seq_len=1)
    kwargs = {"initial_state": initial_state, "use_qk_norm_in_gdn": True}

    omitted, omitted_state = qwen3.jax_ar_gated_delta_rule(query, key, value, g, beta, **kwargs)
    pinned, pinned_state = qwen3.jax_ar_gated_delta_rule(query, key, value, g, beta, matmul_precision="highest", **kwargs)

    self.assertTrue(jnp.array_equal(omitted, pinned))
    self.assertTrue(jnp.array_equal(omitted_state, pinned_state))

  def test_naive_rule_default_arg_preserves_baseline(self):
    """The reference kernel must stay at HIGHEST by default so tests keep their baseline."""
    query, key, value, g, beta, initial_state = _inputs()
    kwargs = {"chunk_size": 64, "use_qk_norm_in_gdn": True, "initial_state": initial_state}

    omitted, omitted_state = qwen3.naive_jax_chunk_gated_delta_rule(query, key, value, g, beta, **kwargs)
    pinned, pinned_state = qwen3.naive_jax_chunk_gated_delta_rule(
        query, key, value, g, beta, precision=jax.lax.Precision.HIGHEST, **kwargs
    )

    self.assertTrue(jnp.array_equal(omitted, pinned))
    self.assertTrue(jnp.array_equal(omitted_state, pinned_state))

  def test_chunk_rule_precision_reaches_the_matmuls(self):
    """The requested precision must show up in the lowered HLO, not be dropped.

    This is the property under test: the value has to reach jnp.matmul rather
    than being accepted and ignored. It is asserted on the HLO because on CPU
    precision does not change the emitted numerics.
    """
    query, key, value, g, beta, initial_state = _inputs()

    def lower(precision):
      def fn(q, k, v, gate, b):
        return qwen3.jax_chunk_gated_delta_rule(
            q,
            k,
            v,
            gate,
            b,
            chunk_size=64,
            use_qk_norm_in_gdn=True,
            initial_state=initial_state,
            matmul_precision=precision,
        )

      return jax.jit(fn).lower(query, key, value, g, beta).as_text()

    self.assertIn("HIGHEST", lower("highest"))
    self.assertNotIn("HIGHEST", lower("default"))

  def test_ar_rule_precision_reaches_the_matmuls(self):
    """Same property for the autoregressive decode path."""
    query, key, value, g, beta, initial_state = _inputs(seq_len=1)

    def lower(precision):
      def fn(q, k, v, gate, b):
        return qwen3.jax_ar_gated_delta_rule(
            q,
            k,
            v,
            gate,
            b,
            initial_state=initial_state,
            use_qk_norm_in_gdn=True,
            matmul_precision=precision,
        )

      return jax.jit(fn).lower(query, key, value, g, beta).as_text()

    self.assertIn("HIGHEST", lower("highest"))
    self.assertNotIn("HIGHEST", lower("default"))

  def test_configured_precisions_are_accepted(self):
    """Every value gdn_matmul_precision may take in base.yml must run."""
    query, key, value, g, beta, initial_state = _inputs()

    for precision in ("default", "high", "highest"):
      with self.subTest(precision=precision):
        out, _ = qwen3.jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            use_qk_norm_in_gdn=True,
            initial_state=initial_state,
            matmul_precision=precision,
        )
        self.assertTrue(bool(jnp.isfinite(out).all()))


if __name__ == "__main__":
  unittest.main()
