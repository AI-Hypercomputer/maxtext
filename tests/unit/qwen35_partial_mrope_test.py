# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests Qwen3.5 partial multi-dimensional rotary embeddings."""

import unittest
from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp
import numpy as np

from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Qwen3OmniMoeThinkerTextRotaryEmbedding


_HEAD_DIM = 256
_PARTIAL_ROTARY_FACTOR = 0.25
_ROTARY_DIM = 64
_MROPE_SECTION = (11, 11, 10)
_ROPE_THETA = 10_000_000


def _reference_partial_mrope(inputs: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Independent NumPy implementation of the Qwen3.5 partial-MRoPE rule."""
  if positions.ndim == 2:
    positions = np.broadcast_to(positions[np.newaxis, ...], (3,) + positions.shape)

  inv_freq = 1.0 / (_ROPE_THETA ** (np.arange(0, _ROTARY_DIM, 2, dtype=np.float32) / _ROTARY_DIM))
  freqs = positions[..., np.newaxis].astype(np.float32) * inv_freq
  interleaved = np.array(freqs[0], copy=True)
  for dim, offset in enumerate((1, 2), start=1):
    idx = slice(offset, _MROPE_SECTION[dim] * 3, 3)
    interleaved[..., idx] = freqs[dim, ..., idx]

  angles = np.concatenate([interleaved, interleaved], axis=-1)
  cos = np.cos(angles)[:, :, np.newaxis, :]
  sin = np.sin(angles)[:, :, np.newaxis, :]

  rotary, passthrough = np.split(inputs, [_ROTARY_DIM], axis=-1)
  first_half, second_half = np.split(rotary, 2, axis=-1)
  rotate_half = np.concatenate([-second_half, first_half], axis=-1)
  rotated = rotary * cos + rotate_half * sin
  return np.concatenate([rotated, passthrough], axis=-1)


class Qwen35PartialMropeTest(unittest.TestCase):

  def test_matches_reference_and_preserves_unrotated_suffix(self):
    inputs = np.linspace(-1.0, 1.0, num=2 * 4 * 3 * _HEAD_DIM, dtype=np.float32).reshape((2, 4, 3, _HEAD_DIM))
    text_positions = np.broadcast_to(np.arange(4, dtype=np.int32), (2, 4))
    multimodal_positions = np.stack([text_positions, text_positions + 3, text_positions + 7], axis=0)
    embedding = Qwen3OmniMoeThinkerTextRotaryEmbedding(
        min_timescale=1,
        max_timescale=_ROPE_THETA,
        embedding_dims=_HEAD_DIM,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        cast_as_fprop_dtype=False,
        fprop_dtype=jnp.float32,
        mrope_section=_MROPE_SECTION,
        rngs=nnx.Rngs(0),
    )

    self.assertEqual(embedding.rotary_dim, _ROTARY_DIM)
    self.assertEqual(embedding.timescale.shape, (_ROTARY_DIM // 2,))
    for positions in (text_positions, multimodal_positions):
      with self.subTest(position_rank=positions.ndim):
        actual = np.asarray(embedding(jnp.asarray(inputs), jnp.asarray(positions)))
        expected = _reference_partial_mrope(inputs, positions)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(actual[..., _ROTARY_DIM:], inputs[..., _ROTARY_DIM:])

  def test_factor_one_preserves_full_width_mrope_behavior(self):
    inputs = np.linspace(-1.0, 1.0, num=2 * 4 * 3 * _ROTARY_DIM, dtype=np.float32).reshape((2, 4, 3, _ROTARY_DIM))
    positions = np.broadcast_to(np.arange(4, dtype=np.int32), (2, 4))
    embedding = Qwen3OmniMoeThinkerTextRotaryEmbedding(
        min_timescale=1,
        max_timescale=_ROPE_THETA,
        embedding_dims=_ROTARY_DIM,
        partial_rotary_factor=1.0,
        cast_as_fprop_dtype=False,
        fprop_dtype=jnp.float32,
        mrope_section=_MROPE_SECTION,
        rngs=nnx.Rngs(0),
    )

    actual = np.asarray(embedding(jnp.asarray(inputs), jnp.asarray(positions)))
    expected = _reference_partial_mrope(inputs, positions)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_attention_wires_qwen35_partial_factor_into_mrope(self):
    attention = SimpleNamespace(
        config=SimpleNamespace(
            attention_type="global",
            rope_use_scale=False,
            rope_min_timescale=1,
            partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        ),
        qk_rope_head_dim=0,
        head_dim=_HEAD_DIM,
        rope_type="default",
        is_vision=False,
        use_mrope=True,
        rope_max_timescale=_ROPE_THETA,
        dtype=jnp.float32,
        mrope_section=_MROPE_SECTION,
        partial_rotary_factor=None,
        rngs=nnx.Rngs(0),
    )

    embedding = Attention.init_rotary_embedding(attention)

    self.assertIsInstance(embedding, Qwen3OmniMoeThinkerTextRotaryEmbedding)
    self.assertEqual(embedding.head_dim, _HEAD_DIM)
    self.assertEqual(embedding.rotary_dim, _ROTARY_DIM)


if __name__ == "__main__":
  unittest.main()
