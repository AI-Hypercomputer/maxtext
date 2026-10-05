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

Parity is checked against an inlined PyTorch copy of Moonshot's
`_apply_attn_res` (modeling_kimi_linear.py), so the HF reference repo is not
needed; the module is skipped when torch is unavailable.
"""

import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from maxtext.layers.attn_res import _apply_attn_res
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params

_B, _S, _D = 2, 4, 64
_EPS = 1e-5


def ref_apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps=_EPS):
  """Inlined copy of the HF `_apply_attn_res` (prefix_sum [N, D], block_residual [N, K, D])."""
  v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
  v_float = v.float()
  k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + eps)
  score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
  probs = (k * score_weight).sum(-1).softmax(-1).unsqueeze(1)
  return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def _make_inputs(num_blocks, seed=0):
  """Returns torch tensors (prefix [B*S, D], blocks [B*S, K, D], proj [1, D], norm [D])."""
  g = torch.Generator().manual_seed(seed)
  prefix = torch.randn(_B * _S, _D, generator=g)
  blocks = torch.randn(_B * _S, num_blocks, _D, generator=g)
  proj = torch.randn(1, _D, generator=g) * 0.5
  norm = torch.randn(_D, generator=g) * 0.1 + 1.0
  return prefix, blocks, proj, norm


def _to_jax(prefix, blocks, proj, norm):
  num_blocks = blocks.shape[1]
  return (
      jnp.asarray(prefix.numpy().reshape(_B, _S, _D)),
      jnp.asarray(blocks.numpy().reshape(_B, _S, num_blocks, _D)),
      jnp.asarray(proj.numpy().T),  # MaxText DenseGeneral kernel layout [D, 1]
      jnp.asarray(norm.numpy()),
  )


class KimiK3AttnResTest(parameterized.TestCase):
  """Tests for `_apply_attn_res`."""

  def test_no_blocks_is_identity(self):
    prefix = jnp.asarray(np.random.default_rng(0).standard_normal((_B, _S, _D)), dtype=jnp.float32)
    proj, norm = jnp.ones((_D, 1)), jnp.ones((_D,))
    np.testing.assert_array_equal(_apply_attn_res(prefix, None, proj, norm, _EPS), prefix)
    empty = jnp.zeros((_B, _S, 0, _D))
    np.testing.assert_array_equal(_apply_attn_res(prefix, empty, proj, norm, _EPS), prefix)

  def test_static_num_blocks_rejected(self):
    prefix = jnp.zeros((_B, _S, _D))
    with self.assertRaises(NotImplementedError):
      _apply_attn_res(prefix, jnp.zeros((_B, _S, 2, _D)), jnp.ones((_D, 1)), jnp.ones((_D,)), _EPS, num_blocks=1)

  @parameterized.parameters(1, 2, 8)
  def test_parity_vs_hf_formula(self, num_blocks):
    prefix, blocks, proj, norm = _make_inputs(num_blocks)
    expected = ref_apply_attn_res(prefix, blocks, proj, norm).numpy().reshape(_B, _S, _D)
    actual = _apply_attn_res(*_to_jax(prefix, blocks, proj, norm), epsilon=_EPS)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_bf16_preserves_dtype(self):
    prefix, blocks, proj, norm = _make_inputs(3)
    p, b, w, n = _to_jax(prefix, blocks, proj, norm)
    out = _apply_attn_res(p.astype(jnp.bfloat16), b.astype(jnp.bfloat16), w, n, _EPS)
    self.assertEqual(out.dtype, jnp.bfloat16)
    expected = ref_apply_attn_res(prefix.bfloat16(), blocks.bfloat16(), proj, norm).float().numpy()
    np.testing.assert_allclose(np.asarray(out, np.float32), expected.reshape(_B, _S, _D), rtol=2e-2, atol=2e-2)

  def test_jit_and_grad_match_reference(self):
    prefix, blocks, proj, norm = _make_inputs(3, seed=1)
    p, b, w, n = _to_jax(prefix, blocks, proj, norm)

    eager = _apply_attn_res(p, b, w, n, _EPS)
    jitted = jax.jit(_apply_attn_res, static_argnames=("epsilon",))(p, b, w, n, epsilon=_EPS)
    np.testing.assert_allclose(jitted, eager, rtol=1e-6, atol=1e-6)

    def loss(p, b, w, n):
      return jnp.sum(jnp.sin(_apply_attn_res(p, b, w, n, _EPS)))

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(p, b, w, n)

    tp, tb, tw, tn = (t.clone().requires_grad_(True) for t in (prefix, blocks, proj, norm))
    torch.sin(ref_apply_attn_res(tp, tb, tw, tn)).sum().backward()
    expected = (
        tp.grad.numpy().reshape(_B, _S, _D),
        tb.grad.numpy().reshape(_B, _S, 3, _D),
        tw.grad.numpy().T,
        tn.grad.numpy(),
    )
    for got, want in zip(grads, expected):
      np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

  def test_param_mapping_round_trip(self):
    """The HF [1, D] proj weight converts to a [D, 1] kernel and norm weight to a [D] scale."""
    _, _, proj, norm = _make_inputs(1)
    hf_config = {
        "hidden_size": _D,
        "num_hidden_layers": 4,
        "full_attn_layers": [3],  # 0-indexed; the mapping requires an explicit layer schedule
        "rms_norm_eps": _EPS,
        "attn_res_block_size": 12,
        "first_k_dense_replace": 1,
    }
    state_dict = {
        "layers.0.self_attention_res_norm.weight": norm,
        "layers.0.self_attention_res_proj.weight": proj,
        "layers.0.mlp_res_norm.weight": norm * 2,
        "layers.0.mlp_res_proj.weight": proj * 2,
        "output_attn_res_norm.weight": norm * 3,
        "output_attn_res_proj.weight": proj * 3,
    }
    params = convert_pytorch_module_to_maxtext_params(state_dict, hf_config, None)
    for mt_prefix, scale in (
        ("params-decoder-layers_0-self_attention_res", 1),
        ("params-decoder-layers_0-mlp_res", 2),
        ("params-decoder-output_attn_res", 3),
    ):
      np.testing.assert_allclose(params[f"{mt_prefix}_norm-scale"], (norm * scale).numpy())
      np.testing.assert_allclose(params[f"{mt_prefix}_proj-kernel"], (proj * scale).numpy().T)


if __name__ == "__main__":
  unittest.main()
