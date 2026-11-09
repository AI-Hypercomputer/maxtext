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

"""Hermetic (JAX/NumPy-only) tests for the Kimi Delta Attention (KDA) layer.

These run in PR CI (no torch, no HF checkout): a float64 NumPy oracle of the fla KDA math for the
core and the full layer, gradient checks, the kimi-k3 PARAM_MAPPING/HOOK_FNS round trip, bf16,
padding and a 4-device CPU mesh. Torch-oracle parity lives in `kimi_k3_kda_vs_reference_test.py`.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.layers.attention_kda import (
    KimiDeltaAttention,
    jax_ar_kimi_delta_rule,
    jax_chunk_kimi_delta_rule,
    kda_gate,
)
from maxtext.layers.normalizations import l2norm
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params, load_params_into_nnx

_PREFIX = "params-decoder-layers_0-self_attention"


# ==============================================================================
# float64 NumPy oracle of the fla KDA math (independent of the JAX implementation)
# ==============================================================================
def _np_recurrent_kda(q, k, v, g, beta, *, h0=None):
  """Token-by-token KDA recurrence in float64; state `[B, H, K, V]`. q is already scaled."""
  q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
  b, t, h, d = q.shape
  state = np.zeros((b, h, d, v.shape[-1])) if h0 is None else np.asarray(h0, np.float64).copy()
  outs = []
  for i in range(t):
    state = state * np.exp(g[:, i])[..., None]
    delta = beta[:, i, :, None] * (v[:, i] - np.einsum("bhk,bhkv->bhv", k[:, i], state))
    state = state + np.einsum("bhk,bhv->bhkv", k[:, i], delta)
    outs.append(np.einsum("bhk,bhkv->bhv", q[:, i], state))
  return np.stack(outs, axis=1), state


def _np_layer(params, x, *, num_heads, head_dim, lower_bound=-5.0, eps=1e-5):
  """float64 forward of HF `KimiDeltaAttention` (fla semantics) from MaxText-layout params."""
  x = np.asarray(x, np.float64)
  b, t, _ = x.shape
  p = {n: np.asarray(v, np.float64) for n, v in params.items()}
  heads = (b, t, num_heads, head_dim)

  def conv(y, kernel):  # kernel [K, 1, C]; causal depthwise conv + silu
    ks = kernel.shape[0]
    y_pad = np.pad(y, ((0, 0), (ks - 1, 0), (0, 0)))
    out = sum(y_pad[:, j : j + t] * kernel[j, 0] for j in range(ks))
    return out / (1.0 + np.exp(-out))

  def l2n(y):
    return y / np.sqrt((y * y).sum(-1, keepdims=True) + 1e-6)

  def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

  q = l2n(conv(x @ p["q_proj"], p["q_conv1d"]).reshape(heads)) / np.sqrt(head_dim)
  k = l2n(conv(x @ p["k_proj"], p["k_conv1d"]).reshape(heads))
  v = conv(x @ p["v_proj"], p["v_conv1d"]).reshape(heads)
  g_in = (x @ p["f_a_proj"] @ p["f_b_proj"]).reshape(heads) + p["dt_bias"].reshape(num_heads, head_dim)
  rate = np.exp(p["A_log"]).reshape(num_heads, 1)
  if lower_bound is not None:
    g = lower_bound * sigmoid(rate * g_in)
  else:
    g = -rate * np.log1p(np.exp(g_in))
  beta = sigmoid(x @ p["b_proj"])
  o, _ = _np_recurrent_kda(q, k, v, g, beta)
  o = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * p["o_norm"] * sigmoid((x @ p["g_proj"]).reshape(heads))
  return o.reshape(b, t, -1) @ p["out"]


def _layer_params_np(layer):
  """MaxText-layout float32 NumPy copies of a `KimiDeltaAttention`'s parameters, by short name."""
  names = ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj", "b_proj")
  params = {n: np.asarray(getattr(layer, n).kernel.get_value(), np.float32) for n in names + ("g_proj", "out")}
  params["A_log"] = np.asarray(layer.A_log.get_value(), np.float32)
  params["dt_bias"] = np.asarray(layer.dt_bias.get_value(), np.float32)
  params["o_norm"] = np.asarray(layer.o_norm.scale.get_value(), np.float32)
  return params


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

  def test_chunked_matches_float64_numpy_oracle(self):
    """Chunked core (several chunks, padded tail, initial state) == float64 NumPy recurrence."""
    q, k, v, g, beta, h0 = _random_core_inputs(4, b=2, t=21, h=2, d=8, g_range=(-5.0, -0.05))
    o_ref, s_ref = _np_recurrent_kda(q, k, v, g, beta, h0=h0)
    o, s = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=h0)
    np.testing.assert_allclose(np.asarray(o), o_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(s), s_ref, rtol=1e-5, atol=1e-5)

  def test_chunked_gradients_match_numerical(self):
    """`check_grads` (reverse mode vs finite differences) on the chunked core, through several chunks
    and the state. The core is float32 by design (decay / WY solve / state), hence the tolerances."""
    q, k, v, g, beta, h0 = _random_core_inputs(5, b=1, t=10, h=2, d=4, g_range=(-2.0, -0.1))

    def f(q, k, v, g, beta, h0):  # pylint: disable=too-many-positional-arguments
      o, s = jax_chunk_kimi_delta_rule(q, k, v, g, beta, chunk_size=4, initial_state=h0)
      return jnp.sum(o * jnp.cos(o)) + jnp.sum(s**2)

    jtu.check_grads(f, (q, k, v, g, beta, h0), order=1, modes=("rev",), eps=1e-3, atol=1e-2, rtol=1e-2)

  def test_layer_matches_float64_numpy_oracle(self):
    """Full layer (fp32) == float64 NumPy forward of HF `KimiDeltaAttention`, with and without lower bound."""
    for lower_bound in (-5.0, None):
      with self.subTest(lower_bound=lower_bound):
        layer = KimiDeltaAttention(
            hidden_size=32,
            num_heads=2,
            head_dim=8,
            chunk_size=4,
            gate_lower_bound=lower_bound,
            rngs=nnx.Rngs(params=0),
        )
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 11, 32))
        y, _, _ = layer(x, model_mode=MODEL_MODE_TRAIN)
        y_ref = _np_layer(_layer_params_np(layer), x, num_heads=2, head_dim=8, lower_bound=lower_bound)
        np.testing.assert_allclose(np.asarray(y), y_ref, rtol=1e-4, atol=1e-5)

  def test_param_mapping_round_trip_with_padded_a_log(self):
    """HF-layout weights (release `[head_dim]`-padded `A_log`) -> kimi-k3 PARAM_MAPPING/HOOK_FNS -> layer -> HF."""
    num_heads, head_dim, hidden = 4, 8, 32
    hf_config = {
        "hidden_size": hidden,
        "num_hidden_layers": 4,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "full_attn_layers": [3],
        "rms_norm_eps": 1e-5,
        "first_k_dense_replace": 4,  # all-dense toy: only the KDA keys are exercised
        "linear_attn_config": {"num_heads": num_heads, "head_dim": head_dim, "use_full_rank_gate": True},
    }
    src = KimiDeltaAttention(hidden_size=hidden, num_heads=num_heads, head_dim=head_dim, rngs=nnx.Rngs(params=1))
    mt = _layer_params_np(src)
    # HF / torch layouts: Linear weight [out, in]; conv weight [C, 1, K]; A_log zero-padded to head_dim.
    hf_sd = {f"{n}.weight": mt[n].T for n in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj", "g_proj")}
    hf_sd["o_proj.weight"] = mt["out"].T
    hf_sd.update({f"{n}.weight": mt[n].transpose(2, 1, 0) for n in ("q_conv1d", "k_conv1d", "v_conv1d")})
    hf_sd["A_log"] = np.pad(mt["A_log"], (0, head_dim - num_heads))
    hf_sd["dt_bias"] = mt["dt_bias"]
    hf_sd["o_norm.weight"] = mt["o_norm"]

    dst = KimiDeltaAttention(hidden_size=hidden, num_heads=num_heads, head_dim=head_dim, rngs=nnx.Rngs(params=2))
    target_shapes = {f"{_PREFIX}-A_log": (num_heads,)}
    for shapes in (target_shapes, None):  # with the MaxText target shape (as `to_maxtext`) and without
      with self.subTest(target_shapes=shapes is not None):
        converted = convert_pytorch_module_to_maxtext_params(
            hf_sd, hf_config, prefix_filter=_PREFIX, target_shapes=shapes
        )
        loaded = load_params_into_nnx(dst, converted, _PREFIX, strict=True)
        self.assertEqual(len(loaded), 14)
        for name, ref in _layer_params_np(dst).items():
          np.testing.assert_array_equal(ref, mt[name], err_msg=name)

    # MaxText -> HF restores the release layout (zero-padded A_log, torch layouts).
    param_map = PARAM_MAPPING["kimi-k3"](hf_config, None, scan_layers=False)
    to_hf = HOOK_FNS["kimi-k3"](hf_config, None, scan_layers=False, saving_to_hf=True)
    for mt_key, hf_key in param_map.items():
      if not mt_key.startswith(_PREFIX + "-"):
        continue
      short = hf_key.split(".self_attn.", 1)[1]
      value = converted[mt_key]
      if mt_key in to_hf:
        value = to_hf[mt_key](value, hf_sd[short].shape)
      np.testing.assert_array_equal(np.asarray(value), hf_sd[short], err_msg=mt_key)

  def test_bf16_layer_keeps_fp32_params_and_tracks_oracle(self):
    """bf16 activations / weights: fp32 KDA params (as released), bf16 output close to the float64 oracle."""
    layer = KimiDeltaAttention(
        hidden_size=32,
        num_heads=2,
        head_dim=8,
        chunk_size=4,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(params=0),
    )
    for p in (layer.A_log, layer.dt_bias, layer.q_conv1d.kernel, layer.k_conv1d.kernel, layer.v_conv1d.kernel):
      self.assertEqual(p.get_value().dtype, jnp.float32)
    self.assertEqual(layer.o_norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(layer.q_proj.kernel.get_value().dtype, jnp.bfloat16)

    x = jax.random.normal(jax.random.PRNGKey(4), (2, 12, 32)).astype(jnp.bfloat16)
    y, conv_state, rec_state = layer(x, model_mode=MODEL_MODE_TRAIN, output_final_state=True)
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertEqual(conv_state[0].dtype, jnp.bfloat16)
    self.assertEqual(rec_state.dtype, jnp.float32)
    y_ref = _np_layer(_layer_params_np(layer), np.asarray(x, np.float32), num_heads=2, head_dim=8)
    rel = np.linalg.norm(np.asarray(y, np.float64) - y_ref) / np.linalg.norm(y_ref)
    self.assertLess(rel, 2e-2)

  def test_right_padding_does_not_touch_outputs_or_state(self):
    """`decoder_segment_ids == 0` (non-zero pad embeddings) leaves valid outputs and the returned states equal
    to running on the unpadded prefix, for prefill and for an inactive decode step."""
    layer = self._layer(chunk_size=4)
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 9, 32))
    lengths = (9, 5)
    seg = jnp.asarray([[1] * n + [0] * (9 - n) for n in lengths], jnp.int32)
    y, conv_state, rec_state = layer(x, model_mode=MODEL_MODE_PREFILL, output_final_state=True, decoder_segment_ids=seg)
    assert conv_state is not None
    assert rec_state is not None
    rec_state_np = np.asarray(rec_state)  # pylint 3.3.8 does not carry the Optional narrowing into the loop.
    for row, n in enumerate(lengths):
      y_ref, conv_ref, rec_ref = layer(x[row : row + 1, :n], model_mode=MODEL_MODE_PREFILL, output_final_state=True)
      assert conv_ref is not None
      assert rec_ref is not None
      np.testing.assert_allclose(np.asarray(y[row : row + 1, :n]), np.asarray(y_ref), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(rec_state_np[row : row + 1], np.asarray(rec_ref), rtol=1e-5, atol=1e-5)
      for got, ref in zip(conv_state, conv_ref):
        np.testing.assert_allclose(np.asarray(got[row : row + 1]), np.asarray(ref), rtol=1e-6, atol=1e-6)

    # A decode step with segment id 0 (inactive slot) is an identity on both states.
    x_t = jax.random.normal(jax.random.PRNGKey(6), (2, 1, 32))
    _, conv_next, rec_next = layer(
        x_t,
        conv_state=conv_state,
        recurrent_state=rec_state,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        output_final_state=True,
        decoder_segment_ids=jnp.zeros((2, 1), jnp.int32),
    )
    assert conv_next is not None
    assert rec_next is not None
    np.testing.assert_array_equal(np.asarray(rec_next), np.asarray(rec_state))
    for got, ref in zip(conv_next, conv_state):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_four_device_mesh_matches_single_device(self):
    """Layer forward + parameter gradients on a 2x2 (data, tensor) CPU mesh match the unsharded run."""
    if jax.device_count() < 4:
      self.skipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    layer = self._layer(chunk_size=4)
    graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
    x = jax.random.normal(jax.random.PRNGKey(7), (4, 10, 32))

    def loss(params, x):
      y, _, _ = nnx.merge(graphdef, params, rest)(x, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(y**2), y

    # Compare jit (sharded) against jit (unsharded): eager vs fused differ by more than the reduction order.
    (_, y_ref), g_ref = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, x)

    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("data", "tensor"))

    def param_sharding(p):
      # Shard the projection output (heads) axis over `tensor` where it divides; replicate the rest.
      spec = [None] * p.ndim
      if p.ndim >= 1 and p.shape[-1] % 2 == 0 and p.shape[-1] >= 16:
        spec[-1] = "tensor"
      return NamedSharding(mesh, P(*spec))

    params_sharded = jax.tree.map(lambda p: jax.device_put(p, param_sharding(p)), params)
    x_sharded = jax.device_put(x, NamedSharding(mesh, P("data")))
    (_, y), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params_sharded, x_sharded)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-5, atol=1e-5)
    for got, ref in zip(jax.tree.leaves(grads), jax.tree.leaves(g_ref)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-4, atol=2e-4)  # sharded reduction order


if __name__ == "__main__":
  unittest.main()
