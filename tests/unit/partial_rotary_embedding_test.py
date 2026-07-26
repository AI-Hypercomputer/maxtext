# Copyright 2023–2026 Google LLC
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

"""Unit tests for the partial rotary position embedding layer.

The new PartialRotaryEmbedding class is a thin wrapper around
`RotaryEmbedding` that applies RoPE only to the first fraction of the
hidden dimensions.  The tests below exercise the half/fully-rotated
cases and verify basic shift invariance in the same style used by our
existing rotary unit tests.
"""

import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
import numpy as np

from maxtext.layers.embeddings import (
    PartialRotaryEmbedding,
    RotaryEmbedding,
    Gemma4PartialRotaryEmbedding,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class PartialRotaryEmbeddingTest(unittest.TestCase):
  """Tests for the PartialRotaryEmbedding layer."""

  def setUp(self):
    super().setUp()
    # build a simple config and mesh like other embedding tests
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_embeddings",
        enable_checkpointing=False,
    )
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.nnx_rng = nnx.Rngs(params=0)

  def test_partial_rotary_half(self):
    """The first half of the hidden dim should be rotated, the rest passthrough."""
    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 8
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    rope_half = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
        cast_as_fprop_dtype=False,
    )

    y_half = rope_half(inputs, positions)

    rotary_dim = head_dim // 2
    inputs_rot, inputs_pass = inputs[..., :rotary_dim], inputs[..., rotary_dim:]

    rope_full_for_rot_part = RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=rotary_dim,
        rngs=self.nnx_rng,
        cast_as_fprop_dtype=False,
    )
    y_rot_expected = rope_full_for_rot_part(inputs_rot, positions)

    np.testing.assert_allclose(
        y_half[..., :rotary_dim],
        y_rot_expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="First fraction should be rotated.",
    )
    np.testing.assert_allclose(
        y_half[..., rotary_dim:],
        inputs_pass,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Remaining dims should pass through.",
    )

  def test_partial_rotary_full(self):
    """A partial factor of 1.0 is equivalent to the base rotary embedding."""
    batch_size, seq_len, num_heads, head_dim = 1, 4, 4, 8
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    rope_partial = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=1.0,
        rngs=self.nnx_rng,
    )
    y_partial = rope_partial(inputs, positions)

    rope_full = RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        rngs=self.nnx_rng,
    )
    y_full = rope_full(inputs, positions)

    np.testing.assert_allclose(
        y_partial,
        y_full,
        rtol=1e-6,
        atol=1e-6,
        err_msg="PartialRotaryEmbedding with factor=1 should equal full rotary.",
    )

  def test_shift_invariance(self):
    """Verify that rotary attention computed from partial embedding is shift invariant."""
    batch_size, seq_len, num_heads, head_dim = 1, 20, 4, 8
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    rope = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
        cast_as_fprop_dtype=False,
    )

    def get_attn(pos):
      y = rope(inputs, pos)
      return np.einsum("BSNH,BTNH->BSTN", y, y)

    ref_attn = get_attn(positions)
    shifted_attn = get_attn(positions + 3)

    np.testing.assert_allclose(
        ref_attn,
        shifted_attn,
        rtol=1e-6,
        atol=1e-6,
        err_msg="PartialRotaryEmbedding attention should be shift-invariant.",
    )

  def test_snapshot_verification(self):
    """Verify output values against captured snapshot."""
    layer = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)

    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.30078125, 1.3828125, 1.0, 1.0]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class Gemma4PartialRotaryEmbeddingTest(unittest.TestCase):
  """Tests for the Gemma4PartialRotaryEmbedding layer."""

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_embeddings",
        enable_checkpointing=False,
    )
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.nnx_rng = nnx.Rngs(params=0)

  def test_timescale_padding(self):
    """Verify that the unrotated timescale dimensions are padded with inf."""
    head_dim = 16
    partial_rotary_factor = 0.5
    rope = Gemma4PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rngs=self.nnx_rng,
    )
    timescale = rope.timescale
    rotary_dim = int(head_dim * partial_rotary_factor)
    half_rotary_dim = rotary_dim // 2

    self.assertEqual(timescale.shape, (head_dim // 2,))
    np.testing.assert_array_equal(
        timescale[half_rotary_dim:], np.inf, err_msg="Unrotated dimensions should have timescale set to infinity."
    )
    self.assertFalse(
        np.any(np.isinf(timescale[:half_rotary_dim])), msg="Rotated dimensions should have finite timescales."
    )

  def test_gemma4_partial_rotary_layout(self):
    """The features should be interleaved [Rotated Half 1, Unrotated Half 1, Rotated Half 2, Unrotated Half 2]."""
    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 16
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    rope_gemma4 = Gemma4PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
        cast_as_fprop_dtype=False,
    )

    y = rope_gemma4(inputs, positions)

    rotary_dim = head_dim // 2
    half_rotary_dim = rotary_dim // 2
    half_head_dim = head_dim // 2

    inputs_pass1 = inputs[..., half_rotary_dim:half_head_dim]
    inputs_pass2 = inputs[..., half_head_dim + half_rotary_dim :]

    y_pass1 = y[..., half_rotary_dim:half_head_dim]
    y_pass2 = y[..., half_head_dim + half_rotary_dim :]

    np.testing.assert_allclose(
        y_pass1,
        inputs_pass1,
        rtol=1e-6,
        atol=1e-6,
        err_msg="First unrotated part should pass through.",
    )
    np.testing.assert_allclose(
        y_pass2,
        inputs_pass2,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Second unrotated part should pass through.",
    )

  def test_shift_invariance(self):
    """Verify that rotary attention computed from partial embedding is shift invariant."""
    batch_size, seq_len, num_heads, head_dim = 1, 20, 4, 16
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    rope = Gemma4PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=head_dim,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
        cast_as_fprop_dtype=False,
    )

    def get_attn(pos):
      y = rope(inputs, pos)
      return np.einsum("BSNH,BTNH->BSTN", y, y)

    ref_attn = get_attn(positions)
    shifted_attn = get_attn(positions + 3)

    np.testing.assert_allclose(
        ref_attn,
        shifted_attn,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Gemma4PartialRotaryEmbedding attention should be shift-invariant.",
    )

  def test_snapshot_verification(self):
    """Verify output values against captured snapshot."""
    layer = Gemma4PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        partial_rotary_factor=0.5,
        rngs=self.nnx_rng,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)

    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 1.0, 1.38281, 1.0]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class MropePartialRotaryEmbeddingTest(unittest.TestCase):
  """Partial rotary on the MRoPE path (`use_mrope: true`).

  Models such as qwen3.5-35b-a3b and qwen3.5-397b-a17b set both `use_mrope` and
  `partial_rotary_factor`, so the MRoPE rotary layer has to honour the factor the
  same way `PartialRotaryEmbedding` does on the non-MRoPE path.
  """

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_mrope_partial_rotary",
        enable_checkpointing=False,
    )
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.nnx_rng = nnx.Rngs(params=0)
    self.batch, self.seq, self.heads, self.head_dim = 2, 8, 4, 16
    self.inputs = jax.random.normal(
        jax.random.PRNGKey(0), (self.batch, self.seq, self.heads, self.head_dim), dtype=jnp.float32
    )
    self.positions = jnp.broadcast_to(jnp.arange(self.seq, dtype=jnp.int32), (self.batch, self.seq))

  def _layer(self, partial_rotary_factor=None):
    kwargs = {} if partial_rotary_factor is None else {"partial_rotary_factor": partial_rotary_factor}
    return Qwen3OmniMoeThinkerTextRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        embedding_dims=self.head_dim,
        cast_as_fprop_dtype=False,
        mrope_section=(2, 1, 1),
        rngs=self.nnx_rng,
        **kwargs,
    )

  def test_trailing_dims_pass_through(self):
    """Only the leading `head_dim * factor` channels are rotated."""
    factor = 0.25
    rotary_dim = int(self.head_dim * factor)
    outputs = self._layer(factor)(self.inputs, self.positions)

    self.assertEqual(outputs.shape, self.inputs.shape)
    np.testing.assert_array_equal(
        np.asarray(outputs[..., rotary_dim:]),
        np.asarray(self.inputs[..., rotary_dim:]),
        err_msg="Channels beyond rotary_dim must be passed through untouched.",
    )
    self.assertFalse(
        np.allclose(np.asarray(outputs[..., :rotary_dim]), np.asarray(self.inputs[..., :rotary_dim])),
        msg="The leading rotary_dim channels should have been rotated.",
    )

  def test_factor_is_not_ignored(self):
    """A partial factor must not produce the fully rotated result.

    This is the regression: the layer previously ignored the factor and rotated
    every channel, silently disagreeing with the reference implementation for
    every model that configures both MRoPE and a partial factor.
    """
    partial = self._layer(0.25)(self.inputs, self.positions)
    full = self._layer(1.0)(self.inputs, self.positions)
    self.assertFalse(
        np.allclose(np.asarray(partial), np.asarray(full)),
        msg="partial_rotary_factor had no effect on the MRoPE path.",
    )

  def test_default_factor_is_unchanged(self):
    """Callers that do not pass a factor keep the previous full-rotation behaviour."""
    explicit_full = self._layer(1.0)(self.inputs, self.positions)
    default = self._layer()(self.inputs, self.positions)
    np.testing.assert_allclose(np.asarray(default), np.asarray(explicit_full), rtol=1e-6, atol=1e-6)

  def test_matches_non_mrope_partial_rotary(self):
    """With 1D (text-only) positions MRoPE degenerates to ordinary RoPE.

    The two code paths must therefore agree channel for channel, which is what
    makes the MRoPE path safe for text-only training and inference.
    """
    factor = 0.25
    mrope_out = self._layer(factor)(self.inputs, self.positions)
    reference = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=self.head_dim,
        partial_rotary_factor=factor,
        cast_as_fprop_dtype=False,
        rngs=self.nnx_rng,
    )(self.inputs, self.positions)

    np.testing.assert_allclose(np.asarray(mrope_out), np.asarray(reference), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
