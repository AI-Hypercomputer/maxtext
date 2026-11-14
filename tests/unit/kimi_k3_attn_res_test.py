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

"""Tests for Kimi-K3 Attention Residuals (AttnRes) pooling.

Parity is checked against a float64 NumPy copy of Moonshot's `_apply_attn_res`
(modeling_kimi_linear.py), so neither the HF reference repo nor torch is needed.
Only the bit-exact bf16 / autograd comparisons against an inlined PyTorch copy are
skipped when torch is unavailable.
"""

import unittest

from absl.testing import parameterized
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.test_util import check_grads
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS
from maxtext.layers.attn_res import (
    _apply_attn_res,
    attn_res_num_buffer_slots,
    commit_attn_res_block,
    init_attn_res_buffer,
)
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params

try:
  import torch  # pylint: disable=import-error
except ImportError:  # torch is not installed in the cpu-unit CI job.
  torch = None

_B, _S, _D = 2, 4, 64
_EPS = 1e-5
_HF_CONFIG = {
    "hidden_size": _D,
    "num_hidden_layers": 4,
    "full_attn_layers": [3],  # 0-indexed; the mapping requires an explicit layer schedule
    "rms_norm_eps": _EPS,
    "attn_res_block_size": 12,
    "first_k_dense_replace": 4,  # all-dense toy: only the AttnRes keys are exercised
}


def np_apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps=_EPS):
  """float64 NumPy copy of the HF `_apply_attn_res` (prefix [..., D], blocks [..., K, D], proj [1, D])."""
  v = np.concatenate([block_residual, prefix_sum[..., None, :]], axis=-2).astype(np.float64)
  k = v / np.sqrt(np.mean(v**2, axis=-1, keepdims=True) + eps)
  score_weight = norm_weight.astype(np.float64) * proj_weight.astype(np.float64)[0]
  scores = np.sum(k * score_weight, axis=-1)
  scores = scores - scores.max(axis=-1, keepdims=True)
  probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
  return np.einsum("...k,...kd->...d", probs, v)


def torch_apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps=_EPS):
  """Inlined copy of the HF `_apply_attn_res` (prefix_sum [N, D], block_residual [N, K, D])."""
  v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
  v_float = v.float()
  k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + eps)
  score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
  probs = (k * score_weight).sum(-1).softmax(-1).unsqueeze(1)
  return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def _make_inputs(num_blocks, seed=0):
  """Returns float32 NumPy arrays (prefix [B, S, D], blocks [B, S, K, D], HF proj [1, D], norm [D])."""
  rng = np.random.default_rng(seed)
  prefix = rng.standard_normal((_B, _S, _D)).astype(np.float32)
  blocks = rng.standard_normal((_B, _S, num_blocks, _D)).astype(np.float32)
  proj = (rng.standard_normal((1, _D)) * 0.5).astype(np.float32)
  norm = (rng.standard_normal((_D,)) * 0.1 + 1.0).astype(np.float32)
  return prefix, blocks, proj, norm


def _to_jax(prefix, blocks, proj, norm):
  # MaxText DenseGeneral kernel layout is [D, 1], the transpose of the HF [1, D] weight.
  return jnp.asarray(prefix), jnp.asarray(blocks), jnp.asarray(proj.T), jnp.asarray(norm)


class KimiK3AttnResTest(parameterized.TestCase):
  """Tests for `_apply_attn_res`."""

  def test_no_blocks_is_identity(self):
    prefix = jnp.asarray(np.random.default_rng(0).standard_normal((_B, _S, _D)), dtype=jnp.float32)
    proj, norm = jnp.ones((_D, 1)), jnp.ones((_D,))
    np.testing.assert_array_equal(_apply_attn_res(prefix, None, proj, norm, _EPS), prefix)
    empty = jnp.zeros((_B, _S, 0, _D))
    np.testing.assert_array_equal(_apply_attn_res(prefix, empty, proj, norm, _EPS), prefix)

  def test_static_num_blocks_validates_args(self):
    """The static-padded path needs a buffer of exactly num_blocks slots and a committed count."""
    prefix = jnp.zeros((_B, _S, _D))
    with self.assertRaisesRegex(ValueError, "block buffer"):
      _apply_attn_res(prefix, jnp.zeros((_B, _S, 2, _D)), jnp.ones((_D, 1)), jnp.ones((_D,)), _EPS, num_blocks=1)
    with self.assertRaisesRegex(ValueError, "num_committed"):
      _apply_attn_res(prefix, jnp.zeros((_B, _S, 2, _D)), jnp.ones((_D, 1)), jnp.ones((_D,)), _EPS, num_blocks=2)

  @parameterized.parameters(1, 2, 8)
  def test_static_padded_parity_vs_hf_formula(self, num_blocks):
    """Masked pooling over a padded buffer equals the HF formula over the committed blocks."""
    prefix, blocks, proj, norm = _make_inputs(num_blocks)
    expected = np_apply_attn_res(prefix, blocks, proj, norm)
    p, b, w, n = _to_jax(prefix, blocks, proj, norm)
    padded = jnp.concatenate([b, jnp.full(b.shape[:-2] + (3, _D), 7.0, b.dtype)], axis=-2)
    actual = _apply_attn_res(
        p, padded, w, n, epsilon=_EPS, num_blocks=num_blocks + 3, num_committed=jnp.int32(num_blocks)
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)

  def test_static_padded_equals_dynamic_for_every_count(self):
    """Under jit with a traced count, masked pooling == the dynamic path over the first n blocks (n = 0..M)."""
    num_slots = 5
    p, b, w, n = _to_jax(*_make_inputs(num_slots, seed=4))
    masked = jax.jit(lambda c: _apply_attn_res(p, b, w, n, epsilon=_EPS, num_blocks=num_slots, num_committed=c))
    for count in range(num_slots + 1):
      want = _apply_attn_res(p, b[..., :count, :], w, n, epsilon=_EPS)
      np.testing.assert_allclose(masked(jnp.int32(count)), want, rtol=1e-6, atol=1e-6)

  def test_static_padded_dtype_and_shape_checks(self):
    """The static path keeps PR6's semantics: promoted output dtype and strict weight shapes."""
    p, b, w, n = _to_jax(*_make_inputs(3))
    out = _apply_attn_res(p.astype(jnp.bfloat16), b, w, n, _EPS, num_blocks=3, num_committed=jnp.int32(0))
    self.assertEqual(out.dtype, jnp.float32)
    np.testing.assert_allclose(out, p.astype(jnp.bfloat16).astype(jnp.float32))
    out = _apply_attn_res(
        p.astype(jnp.bfloat16), b.astype(jnp.bfloat16), w, n, _EPS, num_blocks=3, num_committed=jnp.int32(2)
    )
    self.assertEqual(out.dtype, jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "must have shape"):
      _apply_attn_res(p, b, w.T, n, _EPS, num_blocks=3, num_committed=jnp.int32(1))
    with self.assertRaisesRegex(ValueError, "only valid with num_blocks"):
      _apply_attn_res(p, b, w, n, _EPS, num_committed=jnp.int32(1))

  def test_buffer_commits_match_dynamic_block_list(self):
    """commit_attn_res_block on every layer (count advances only on commits) == the growing concat list."""
    num_layers, block_size = 7, 3
    rng = np.random.default_rng(5)
    xs = jnp.asarray(rng.standard_normal((num_layers, _B, _S, _D)).astype(np.float32))
    _, _, proj, norm = _to_jax(*_make_inputs(1, seed=6))
    slots = attn_res_num_buffer_slots(num_layers, block_size)

    def step(buf, inputs):
      i, x = inputs
      pooled = _apply_attn_res(x, buf.blocks, proj, norm, _EPS, num_blocks=slots, num_committed=buf.count)
      return commit_attn_res_block(buf, x, i % block_size == 0), pooled

    buf, pooled = jax.lax.scan(step, init_attn_res_buffer(xs[0], slots), (jnp.arange(num_layers), xs))
    blocks = jnp.zeros((_B, _S, 0, _D))
    for i in range(num_layers):
      np.testing.assert_allclose(pooled[i], _apply_attn_res(xs[i], blocks, proj, norm, _EPS), rtol=1e-6, atol=1e-6)
      if i % block_size == 0:
        blocks = jnp.concatenate([blocks, xs[i][..., None, :]], axis=-2)
    self.assertEqual(int(buf.count), blocks.shape[-2])
    np.testing.assert_array_equal(buf.blocks[..., : blocks.shape[-2], :], blocks)

  @parameterized.named_parameters(
      ("proj_1x1", (1, 1), (_D,)),
      ("proj_1d", (_D,), (_D,)),
      ("proj_hf_layout", (1, _D), (_D,)),
      ("norm_2d", (_D, 1), (1, _D)),
      ("norm_short", (_D, 1), (_D // 2,)),
  )
  def test_mis_shaped_weights_rejected(self, proj_shape, norm_shape):
    prefix = jnp.zeros((_B, _S, _D))
    with self.assertRaisesRegex(ValueError, "must have shape"):
      _apply_attn_res(prefix, jnp.zeros((_B, _S, 2, _D)), jnp.ones(proj_shape), jnp.ones(norm_shape), _EPS)

  @parameterized.parameters(1, 2, 8)
  def test_parity_vs_hf_formula(self, num_blocks):
    prefix, blocks, proj, norm = _make_inputs(num_blocks)
    expected = np_apply_attn_res(prefix, blocks, proj, norm)
    actual = _apply_attn_res(*_to_jax(prefix, blocks, proj, norm), epsilon=_EPS)
    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_bf16_preserves_dtype(self):
    """bf16 activations and weights (as in config.json): fp32 internals, one bf16 rounding at the end."""
    prefix, blocks, proj, norm = (x.astype(ml_dtypes.bfloat16) for x in _make_inputs(3))
    out = _apply_attn_res(*_to_jax(prefix, blocks, proj, norm), epsilon=_EPS)
    self.assertEqual(out.dtype, jnp.bfloat16)
    expected = np_apply_attn_res(prefix, blocks, proj, norm)
    # Only the final fp32 -> bf16 cast may differ: at most half a bf16 ulp (2**-8 relative).
    np.testing.assert_allclose(np.asarray(out, np.float64), expected, rtol=2**-8 * 1.01, atol=1e-6)

  def test_mixed_dtype_returns_promoted_dtype(self):
    """Like HF (`.to(v.dtype)`), the output dtype is the dtype of the concatenated V."""
    prefix, blocks, proj, norm = _to_jax(*_make_inputs(2))
    out = _apply_attn_res(prefix.astype(jnp.bfloat16), blocks, proj, norm, _EPS)
    self.assertEqual(out.dtype, jnp.float32)

  @unittest.skipIf(torch is None, "torch is not installed")
  def test_bf16_bit_exact_vs_torch(self):
    prefix, blocks, proj, norm = (torch.from_numpy(x).bfloat16() for x in _make_inputs(3))
    expected = torch_apply_attn_res(prefix.reshape(-1, _D), blocks.reshape(-1, 3, _D), proj, norm)
    p, b, w, n = (jnp.asarray(t.float().numpy(), dtype=jnp.bfloat16) for t in (prefix, blocks, proj.T, norm))
    out = _apply_attn_res(p, b, w, n, _EPS)
    np.testing.assert_array_equal(np.asarray(out, np.float32), expected.float().numpy().reshape(_B, _S, _D))

  def test_jit_and_grads(self):
    p, b, w, n = _to_jax(*_make_inputs(3, seed=1))
    eager = _apply_attn_res(p, b, w, n, _EPS)
    jitted = jax.jit(_apply_attn_res, static_argnames=("epsilon",))(p, b, w, n, epsilon=_EPS)
    np.testing.assert_allclose(jitted, eager, rtol=1e-6, atol=1e-6)
    # Finite-difference check of the VJP w.r.t. every input (fp32, so loose tolerances).
    check_grads(lambda *a: _apply_attn_res(*a, epsilon=_EPS), (p, b, w, n), order=1, modes=("rev",), atol=5e-2, rtol=5e-2)

  @unittest.skipIf(torch is None, "torch is not installed")
  def test_grad_matches_torch(self):
    prefix, blocks, proj, norm = _make_inputs(3, seed=1)

    def loss(p, b, w, n):
      return jnp.sum(jnp.sin(_apply_attn_res(p, b, w, n, _EPS)))

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*_to_jax(prefix, blocks, proj, norm))

    tp, tb, tw, tn = (torch.from_numpy(x).requires_grad_(True) for x in (prefix, blocks, proj, norm))
    torch.sin(torch_apply_attn_res(tp.reshape(-1, _D), tb.reshape(-1, 3, _D), tw, tn)).sum().backward()
    expected = (tp.grad.numpy(), tb.grad.numpy(), tw.grad.numpy().T, tn.grad.numpy())
    for got, want in zip(grads, expected):
      np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

  def test_batch_sharded_matches_unsharded(self):
    devices = jax.devices()[:_B]
    if len(devices) < _B:
      self.skipTest(f"needs {_B} devices")
    p, b, w, n = _to_jax(*_make_inputs(2, seed=2))
    mesh = Mesh(np.array(devices), ("data",))
    p_s = jax.device_put(p, NamedSharding(mesh, P("data")))
    b_s = jax.device_put(b, NamedSharding(mesh, P("data")))
    out = jax.jit(_apply_attn_res, static_argnames=("epsilon",))(p_s, b_s, w, n, epsilon=_EPS)
    np.testing.assert_allclose(out, _apply_attn_res(p, b, w, n, _EPS), rtol=1e-6, atol=1e-6)

  def test_multi_layer_block_schedule(self):
    """Chains the pooling through the HF decoder schedule (`_forward_attn_residual` + output pooling).

    Toy tanh-linear maps stand in for attention / MLP. Exercises the empty initial block list,
    pooling *before* the commit at `layer_idx % block_size == 0`, the `prefix_sum = None` reset,
    the growing block count, and the final output pooling.
    """
    num_layers, block_size = 7, 3
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((_B, _S, _D)).astype(np.float32)

    def site():
      return (rng.standard_normal((1, _D)) * 0.5).astype(np.float32), (rng.standard_normal(_D) * 0.1 + 1).astype(
          np.float32
      )

    def dense():
      return (rng.standard_normal((_D, _D)) / 8).astype(np.float32)

    layers = [{"attn": site(), "mlp": site(), "wa": dense(), "wm": dense()} for _ in range(num_layers)]
    out_site = site()

    def run(x, pool, xp):  # xp: jnp or np
      blocks = xp.zeros((_B, _S, 0, _D))
      for i, lyr in enumerate(layers):
        prefix = x
        h = pool(prefix, blocks, *lyr["attn"]) if blocks.shape[-2] > 0 else x
        if i % block_size == 0:
          blocks, prefix = xp.concatenate([blocks, prefix[..., None, :]], axis=-2), None
        attn_out = xp.tanh(xp.matmul(h, lyr["wa"]))
        prefix = attn_out if prefix is None else prefix + attn_out
        h = pool(prefix, blocks, *lyr["mlp"])
        x = prefix + xp.tanh(xp.matmul(h, lyr["wm"]))
      return pool(x, blocks, *out_site), blocks.shape[-2]

    def jax_pool(prefix, blocks, proj, norm):
      return _apply_attn_res(prefix, blocks, jnp.asarray(proj.T), jnp.asarray(norm), epsilon=_EPS)

    actual, k_jax = run(jnp.asarray(x0), jax_pool, jnp)
    expected, k_np = run(x0.astype(np.float64), np_apply_attn_res, np)
    self.assertEqual(k_jax, len(range(0, num_layers, block_size)))
    self.assertEqual(k_jax, k_np)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  def test_param_mapping_round_trip(self):
    """HF [1, D] proj weight <-> [D, 1] kernel and [D] norm weight <-> [D] scale, in both directions."""
    _, _, proj, norm = _make_inputs(1)
    state_dict = {
        "layers.0.self_attention_res_norm.weight": norm,
        "layers.0.self_attention_res_proj.weight": proj,
        "layers.0.mlp_res_norm.weight": norm * 2,
        "layers.0.mlp_res_proj.weight": proj * 2,
        "output_attn_res_norm.weight": norm * 3,
        "output_attn_res_proj.weight": proj * 3,
    }
    params = convert_pytorch_module_to_maxtext_params(state_dict, _HF_CONFIG, None)
    to_hf_hooks = HOOK_FNS["kimi-k3"](_HF_CONFIG, None, scan_layers=False, saving_to_hf=True)
    for mt_prefix, scale in (
        ("params-decoder-layers_0-self_attention_res", 1),
        ("params-decoder-layers_0-mlp_res", 2),
        ("params-decoder-output_attn_res", 3),
    ):
      norm_key, proj_key = f"{mt_prefix}_norm-scale", f"{mt_prefix}_proj-kernel"
      np.testing.assert_allclose(params[norm_key], norm * scale)
      np.testing.assert_allclose(params[proj_key], (proj * scale).T)
      self.assertEqual(params[proj_key].shape, (_D, 1))
      # MaxText -> HF: the [D, 1] kernel goes back to the HF [1, D] weight; the norm scale is unchanged.
      for key, want in ((norm_key, norm * scale), (proj_key, proj * scale)):
        back = to_hf_hooks[key](params[key], want.shape) if key in to_hf_hooks else params[key]
        self.assertEqual(back.shape, want.shape)
        np.testing.assert_allclose(back, want)


if __name__ == "__main__":
  unittest.main()
