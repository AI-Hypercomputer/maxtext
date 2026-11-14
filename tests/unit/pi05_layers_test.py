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

"""Unit tests for the pi0.5 building blocks (src/maxtext/models/pi05.py).

References are written with PyTorch, mirroring LeRobot's ``PiGemmaRMSNorm``, transformers' ``GemmaRotaryEmbedding`` /
``apply_rotary_pos_emb``, openpi's ``create_sinusoidal_pos_embedding`` / ``make_att_2d_masks`` and the eager joint
attention, so the tests run without lerobot installed. End-to-end parity against the real LeRobot model is covered by
``~/projects/pi05_ref/mt_parity_pi05.py`` (see ``pi05-base_handover.md``).
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from maxtext.models import pi05

jax.config.update("jax_default_matmul_precision", "highest")


def _torch_rope(x, positions, base):
  """transformers GemmaRotaryEmbedding + apply_rotary_pos_emb (unsqueeze_dim=2 for [B,S,H,D])."""
  head_dim = x.shape[-1]
  inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
  freqs = (inv_freq[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
  emb = torch.cat((freqs, freqs), dim=-1)
  cos, sin = emb.cos()[:, :, None, :], emb.sin()[:, :, None, :]
  x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
  rot = torch.cat((-x2, x1), dim=-1)
  return x * cos + rot * sin


class Pi05LayersTest(unittest.TestCase):

  def test_adarms_matches_pigemma_rmsnorm(self):
    rng = np.random.default_rng(0)
    dim, cond_dim = 64, 32
    norm = pi05.Pi05AdaRMSNorm(dim, cond_dim, 1e-6, jnp.float32, jnp.float32, "highest", rngs=nnx.Rngs(0))
    w = rng.normal(size=(cond_dim, 3 * dim)).astype(np.float32) * 0.1
    b = rng.normal(size=(3 * dim,)).astype(np.float32) * 0.1
    norm.dense.kernel[...] = jnp.asarray(w)
    norm.dense.bias[...] = jnp.asarray(b)
    x = rng.normal(size=(2, 5, dim)).astype(np.float32)
    cond = rng.normal(size=(2, cond_dim)).astype(np.float32)
    out, gate = norm(jnp.asarray(x), jnp.asarray(cond))

    # LeRobot PiGemmaRMSNorm.forward with cond.
    tx, tc = torch.from_numpy(x), torch.from_numpy(cond)
    mod = (tc @ torch.from_numpy(w) + torch.from_numpy(b)).unsqueeze(1)
    scale, shift, tgate = mod.chunk(3, dim=-1)
    normed = tx * torch.rsqrt(torch.mean(tx.square(), dim=-1, keepdim=True) + 1e-6)
    ref = normed * (1 + scale) + shift
    np.testing.assert_allclose(np.asarray(out), ref.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(gate), tgate.numpy(), rtol=1e-5, atol=1e-6)

  def test_rope_matches_hf(self):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 7, 3, 16)).astype(np.float32)
    pos = np.stack([np.arange(7), np.array([0, 1, 2, 2, 2, 900, 901])]).astype(np.int32)
    ours = pi05.apply_rope(jnp.asarray(x), jnp.asarray(pos), 10_000.0)
    ref = _torch_rope(torch.from_numpy(x), torch.from_numpy(pos).long(), 10_000.0)
    np.testing.assert_allclose(np.asarray(ours), ref.numpy(), rtol=1e-5, atol=2e-5)

  def test_sinusoidal_time_embedding(self):
    t = np.array([1.0, 0.3, 0.05], np.float32)
    dim, min_p, max_p = 32, 4e-3, 4.0
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_p * (max_p / min_p) ** fraction
    arg = torch.from_numpy(t).double()[:, None] * (1.0 / period * 2 * math.pi)
    ref = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1).float().numpy()
    ours_host = pi05.sinusoidal_time_embedding(t, dim, min_p, max_p)
    np.testing.assert_allclose(np.asarray(ours_host), ref, atol=1e-7)
    ours_traced = jax.jit(lambda tt: pi05.sinusoidal_time_embedding(tt, dim, min_p, max_p))(jnp.asarray(t))
    np.testing.assert_allclose(np.asarray(ours_traced), ref, atol=2e-4)

  def test_attention_masks_match_make_att_2d_masks(self):
    prefix_pad = np.array([[True] * 5 + [False] * 3, [True] * 8])
    horizon = 4
    # openpi make_att_2d_masks over [prefix (att 0), suffix (att [1, 0, ...])].
    pad = np.concatenate([prefix_pad, np.ones((2, horizon), bool)], axis=1)
    att = np.concatenate([np.zeros((2, 8), int), np.array([[1] + [0] * (horizon - 1)] * 2)], axis=1)
    cum = np.cumsum(att, axis=1)
    full = (cum[:, None, :] <= cum[:, :, None]) & (pad[:, None, :] & pad[:, :, None])
    pb = np.asarray(pi05.prefix_attention_bias(jnp.asarray(prefix_pad))) == 0.0
    sb = np.asarray(pi05.suffix_attention_bias(jnp.asarray(prefix_pad), horizon)) == 0.0
    np.testing.assert_array_equal(pb, full[:, :8, :8])
    np.testing.assert_array_equal(sb, full[:, 8:, :])

  def test_grouped_attention_matches_eager(self):
    rng = np.random.default_rng(2)
    q = rng.normal(size=(2, 6, 4, 8)).astype(np.float32) / math.sqrt(8)
    k = rng.normal(size=(2, 9, 1, 8)).astype(np.float32)
    v = rng.normal(size=(2, 9, 1, 8)).astype(np.float32)
    allowed = rng.random((2, 6, 9)) > 0.3
    allowed[:, :, 0] = True
    bias = np.where(allowed, 0.0, pi05.ATTENTION_MASK_VALUE).astype(np.float32)
    ours = pi05._attend(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(bias))  # pylint: disable=protected-access
    tq = torch.from_numpy(q).transpose(1, 2)
    tk = torch.from_numpy(k).transpose(1, 2).repeat_interleave(4, dim=1)
    tv = torch.from_numpy(v).transpose(1, 2).repeat_interleave(4, dim=1)
    w = torch.softmax(tq @ tk.transpose(2, 3) + torch.from_numpy(bias)[:, None], dim=-1)
    ref = (w @ tv).transpose(1, 2).numpy()
    np.testing.assert_allclose(np.asarray(ours), ref, rtol=1e-5, atol=1e-5)

  def test_prompt_and_state_discretization(self):
    state = np.array([-1.0, -0.999, 0.0, 0.5, 0.999], np.float32)
    bins = pi05.discretize_state(state)
    np.testing.assert_array_equal(bins, np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1)
    self.assertEqual(
        pi05.build_prompt(" pick_up\nthe cup ", state),
        f"Task: pick up the cup, State: {' '.join(map(str, bins))};\nAction: ",
    )


if __name__ == "__main__":
  unittest.main()
