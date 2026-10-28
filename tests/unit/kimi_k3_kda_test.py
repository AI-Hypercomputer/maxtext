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

"""Hermetic (JAX-only) tests for the Kimi Delta Attention (KDA) layer.

Torch-oracle parity lives in `kimi_k3_kda_vs_reference_test.py`.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.layers.attention_kda import (
    KimiDeltaAttention,
    jax_ar_kimi_delta_rule,
    jax_chunk_kimi_delta_rule,
    kda_gate,
)
from maxtext.layers.normalizations import l2norm


def _random_core_inputs(seed, b, t, h, d, g_range=(-1.0, -0.1)):
  """Random L2-normalized q/k, v, log-decay g, beta and initial state for the KDA core."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  q = l2norm(jax.random.normal(keys[0], (b, t, h, d))) / np.sqrt(d)
  k = l2norm(jax.random.normal(keys[1], (b, t, h, d)))
  v = jax.random.normal(keys[2], (b, t, h, d))
  g = jax.random.uniform(keys[3], (b, t, h, d), minval=g_range[0], maxval=g_range[1])
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, t, h)))
  h0 = jax.random.normal(keys[5], (b, h, d, d))
  return q, k, v, g, beta, h0


def _recurrent_reference(q, k, v, g, beta, h0):
  """Token-by-token `jax_ar_kimi_delta_rule`, the definition the chunked core must reproduce."""
  state, outs = h0, []
  for t in range(q.shape[1]):
    o_t, state = jax_ar_kimi_delta_rule(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], initial_state=state)
    outs.append(o_t)
  return jnp.stack(outs, axis=1), state


class KimiK3KDATest(unittest.TestCase):
  """JAX-only tests of the KDA gate, chunked / recurrent cores and the layer's state handoff."""

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision("highest"))

  def test_gate_lower_bound_is_bounded_sigmoid_not_clamp(self):
    """With a lower bound the gate is `lb * sigmoid(...)`, not `max(softplus gate, lb)`.

    At `g_raw + dt_bias = 0` the bounded-sigmoid gate is exactly `lb / 2` for every A_log,
    whereas a clamped softplus gate would be `-exp(A_log) * log(2)`.
    """
    h, d = 3, 32
    a_log = jnp.log(jnp.array([1.0, 4.0, 16.0], dtype=jnp.float32))
    g = kda_gate(jnp.zeros((1, 1, h, d)), a_log, jnp.zeros((h * d,)), -5.0)
    np.testing.assert_allclose(np.asarray(g), -2.5, atol=1e-7)
    g_wide = kda_gate(jnp.linspace(-50.0, 50.0, h * d).reshape(1, 1, h, d), a_log, jnp.zeros((h * d,)), -5.0)
    self.assertTrue(bool(jnp.all((g_wide >= -5.0) & (g_wide <= 0.0))))

  def test_gate_softplus_without_lower_bound(self):
    """Without a lower bound the gate is `-exp(A_log) * softplus(g_raw + dt_bias)` in fp32."""
    rng = np.random.default_rng(0)
    h, d = 2, 4
    g_raw = rng.normal(size=(1, 3, h, d)).astype(np.float32)
    a_log = rng.normal(size=(h,)).astype(np.float32)
    dt_bias = rng.normal(size=(h * d,)).astype(np.float32)
    expected = -np.exp(a_log)[:, None] * np.log1p(np.exp(g_raw + dt_bias.reshape(h, d)))
    got = kda_gate(jnp.asarray(g_raw, jnp.bfloat16), jnp.asarray(a_log), jnp.asarray(dt_bias), None)
    self.assertEqual(got.dtype, jnp.float32)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-2, atol=2e-2)  # bf16 g_raw input

  def test_chunked_matches_recurrent(self):
    """Chunked WY core == token-by-token recurrence (output and final state), with an initial state."""
    q, k, v, g, beta, h0 = _random_core_inputs(0, b=2, t=16, h=2, d=8)
    o_ref, s_ref = _recurrent_reference(q, k, v, g, beta, h0)
    o, s = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=h0)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(s), np.asarray(s_ref), rtol=1e-5, atol=1e-5)

  def test_chunked_padding_seq_len_not_multiple_of_chunk(self):
    """seq_len=13 with chunk 8 (3 padded positions): output and final state are unaffected by padding."""
    q, k, v, g, beta, h0 = _random_core_inputs(1, b=2, t=13, h=2, d=8)
    o_ref, s_ref = _recurrent_reference(q, k, v, g, beta, h0)
    o, s = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=h0)
    self.assertEqual(o.shape, v.shape)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(s), np.asarray(s_ref), rtol=1e-5, atol=1e-5)

  def test_chunked_gradients_finite_under_strong_decay(self):
    """At the gate's lower bound the masked upper-triangle exp(g_diff) overflows; grads must stay finite.

    A 64-token chunk at g=-5 accumulates a log-decay of -320, so g_cumsum[j] - g_cumsum[i] for
    j > i reaches +320 and exp() overflows fp32. Masking only the *output* of exp() still yields
    0 * inf = NaN in the backward pass.
    """
    q, k, v, _, beta, _ = _random_core_inputs(2, b=1, t=64, h=2, d=8)
    g = jnp.full(q.shape, -5.0)

    def loss(q, k, v, g, beta):
      out, _ = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=64)
      return jnp.sum(out**2)

    out, _ = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=64)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for name, grad in zip(("q", "k", "v", "g", "beta"), grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(grad))), f"non-finite gradient w.r.t. {name}")

  def test_bf16_compute_dtype_keeps_decay_in_fp32(self):
    """bf16 compute only rounds q/k/v; the log-decay cumsum must stay fp32.

    32 tokens at the gate's lower bound push the chunk's cumsum to -160, where bf16 spacing is 1.0;
    the following slow-decay tokens (g=-0.05) then get `g_cumsum[i] - g_cumsum[j]` rounded to 0 or
    +-1. With a bf16 cumsum this case has ~29% relative error; with an fp32 cumsum it is ~0.3%.
    """
    q, k, v, _, beta, h0 = _random_core_inputs(3, b=1, t=64, h=2, d=16)
    g = jnp.broadcast_to(jnp.where(jnp.arange(64) < 32, -5.0, -0.05)[None, :, None, None], q.shape)
    o32, s32 = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=h0)
    o16, s16 = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=h0, compute_dtype=jnp.bfloat16)
    self.assertEqual(o16.dtype, q.dtype)
    self.assertEqual(s16.dtype, jnp.float32)
    for got, ref in ((o16, o32), (s16, s32)):
      rel = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
      self.assertLess(rel, 1e-2)

    o_ar, s_ar = jax_ar_kimi_delta_rule(q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], h0, jnp.bfloat16)
    self.assertEqual(o_ar.shape, (1, 1, 2, 16))
    self.assertEqual(s_ar.dtype, jnp.float32)

  def _layer(self, compute_dtype=jnp.float32, chunk_size=4):
    return KimiDeltaAttention(
        hidden_size=32,
        num_heads=2,
        head_dim=8,
        conv_size=4,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        rngs=nnx.Rngs(params=0),
    )

  def test_layer_prefill_then_decode_matches_single_pass(self):
    """Chunked prefill (7 tokens, padded chunk) returning conv + recurrent state, then 3 AR steps,
    reproduces one 10-token chunked pass, outputs and final states."""
    layer = self._layer(chunk_size=4)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 32))
    y_full, conv_full, rec_full = layer(x, model_mode=MODEL_MODE_TRAIN, output_final_state=True)

    y_pre, conv_state, rec_state = layer(x[:, :7], model_mode=MODEL_MODE_PREFILL, output_final_state=True)
    ys = [y_pre]
    for t in range(7, 10):
      y_t, conv_state, rec_state = layer(
          x[:, t : t + 1],
          conv_state=conv_state,
          recurrent_state=rec_state,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          output_final_state=True,
      )
      ys.append(y_t)
    np.testing.assert_allclose(np.asarray(jnp.concatenate(ys, axis=1)), np.asarray(y_full), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(rec_state), np.asarray(rec_full), rtol=1e-5, atol=1e-5)
    for got, ref in zip(conv_state, conv_full):
      np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-6)

  def test_layer_compute_dtype_knob(self):
    """`compute_dtype` (config: `kda_compute_dtype`) changes only the q/k/v precision of the core."""
    self.assertEqual(self._layer().compute_dtype, jnp.float32)
    cfg = SimpleNamespace(
        emb_dim=32,
        num_query_heads=2,
        head_dim=8,
        short_conv_kernel_size=4,
        kda_chunk_size=4,
        normalization_layer_epsilon=1e-5,
        kda_gate_lower_bound=-5.0,
        kda_compute_dtype="bfloat16",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        shard_mode=ShardMode.AUTO,
        matmul_precision="default",
    )
    m_cfg = KimiDeltaAttention(config=cfg, rngs=nnx.Rngs(params=0))
    self.assertEqual(m_cfg.compute_dtype, jnp.bfloat16)
    self.assertEqual((m_cfg.chunk_size, m_cfg.gate_lower_bound), (4, -5.0))

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32))
    y32, _, _ = self._layer()(x, model_mode=MODEL_MODE_TRAIN)
    y16, _, _ = self._layer(compute_dtype=jnp.bfloat16)(x, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(y16.dtype, y32.dtype)  # core output is cast back to the activation dtype
    rel = float(jnp.linalg.norm(y16 - y32) / jnp.linalg.norm(y32))
    self.assertLess(rel, 2e-2)
    self.assertGreater(rel, 0.0)  # the knob really changes the core precision


if __name__ == "__main__":
  unittest.main()
