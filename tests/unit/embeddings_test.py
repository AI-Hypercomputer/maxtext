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

"""Tests for embeddings.py."""

import os
import sys
import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import embeddings
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class EmbedTest(unittest.TestCase):
  """Tests for Embed."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_features))

  def test_attend(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    query = jnp.ones((batch_size, seq_len, num_features))
    outputs = layer.attend(query)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_embeddings))

  def test_attend_on_embedding_matches_transposed_dot(self):
    """`attend_on_embedding` contracts over the table's feature axis directly.

    Expressing the transpose as dimension numbers instead of materializing
    `table.T` is what lets the input lookup and the tied output head share one
    bf16 cast under `shard_mode: explicit`. The contraction is the same, so the
    logits match; the tolerance only allows a backend a different accumulation
    order.
    """
    table = jax.random.normal(jax.random.PRNGKey(0), (32, 8))
    for query_shape in ((8,), (3, 8), (2, 3, 8)):
      with self.subTest(query_shape=query_shape):
        query = jax.random.normal(jax.random.PRNGKey(1), query_shape)
        expected = jnp.dot(query, jnp.asarray(table, jnp.bfloat16).T, preferred_element_type=jnp.float32)
        got = embeddings.attend_on_embedding(query, table, jnp.float32, self.cfg)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

  def test_attend_on_embedding_weight_gradient_in_table_order(self):
    """The tied head's table gradient comes out in the table's own axis order.

    With `table.T` the gradient is built transposed and handed back through a
    second transpose, which under explicit sharding survives to codegen. The
    gradient itself must be unchanged.
    """
    table = jax.random.normal(jax.random.PRNGKey(0), (32, 8))
    query = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 8))

    def loss(t, attend):
      return jnp.sum(jnp.sin(attend(query, t)))

    got = jax.grad(loss)(table, lambda q, t: embeddings.attend_on_embedding(q, t, jnp.float32, self.cfg))
    expected = jax.grad(loss)(
        table, lambda q, t: jnp.dot(q, jnp.asarray(t, jnp.bfloat16).T, preferred_element_type=jnp.float32)
    )
    self.assertEqual(got.shape, table.shape)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


class RotaryEmbeddingTest(unittest.TestCase):
  """Tests for RotaryEmbedding."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        rngs=self.rngs,
    )

    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])

    outputs = layer(inputs, position=position)

    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.988281, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class LLaMARotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.LLaMARotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        use_scale=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 1.38281, 0.988281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class YarnRotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.996094, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)

  def test_pairwise_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (pairwise=False, interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=False,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # Default YaRN RoPE returns concatenated layout [real0, real1, imag0, imag1];
    # pairwise=True returns interleaved layout [real0, imag0, real1, imag1].
    # Convert default concatenated layout to interleaved layout for comparison.
    expected_interleaved = jnp.stack([default_outputs[..., :2], default_outputs[..., 2:]], axis=-1).reshape(
        default_outputs.shape
    )
    np.testing.assert_allclose(outputs, expected_interleaved, atol=1e-5)

  def test_pairwise_requires_interleave(self):
    with self.assertRaises(ValueError):
      embeddings.YarnRotaryEmbedding(
          embedding_dims=4,
          mesh=self.mesh,
          max_position_embeddings=16384,
          original_max_position_embeddings=4096,
          interleave=False,
          pairwise=True,
          rngs=self.rngs,
      )

  def test_dot_pairwise_requires_pairwise(self):
    with self.assertRaises(ValueError):
      embeddings.YarnRotaryEmbedding(
          embedding_dims=4,
          mesh=self.mesh,
          max_position_embeddings=16384,
          original_max_position_embeddings=4096,
          interleave=True,
          pairwise=False,
          dot_pairwise=True,
          rngs=self.rngs,
      )

  def test_direct_position_freqs_and_dot_pairwise_parity(self):
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.normal(rng, (2, 8, 4, 64), dtype=jnp.bfloat16)
    position = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (2, 8))
    cotangent = jax.random.normal(jax.random.PRNGKey(7), inputs.shape, dtype=jnp.bfloat16)

    ref_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=64,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=True,
        fprop_dtype=jnp.bfloat16,
        rngs=self.rngs,
    )
    ref_out, ref_vjp = jax.vjp(lambda x: ref_layer(x, position=position), inputs)
    (ref_grad,) = ref_vjp(cotangent)

    for direct_freqs, dot_pw in ((True, False), (False, True), (True, True)):
      layer = embeddings.YarnRotaryEmbedding(
          embedding_dims=64,
          mesh=self.mesh,
          max_position_embeddings=16384,
          original_max_position_embeddings=4096,
          interleave=True,
          pairwise=True,
          direct_position_freqs=direct_freqs,
          dot_pairwise=dot_pw,
          fprop_dtype=jnp.bfloat16,
          rngs=self.rngs,
      )
      out, vjp_fn = jax.vjp(lambda x, l=layer: l(x, position=position), inputs)
      (grad,) = vjp_fn(cotangent)
      np.testing.assert_allclose(
          np.asarray(out, dtype=np.float32),
          np.asarray(ref_out, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2,
      )
      np.testing.assert_allclose(
          np.asarray(grad, dtype=np.float32),
          np.asarray(ref_grad, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2,
      )

  def test_direct_position_freqs_non_pairwise_parity(self):
    inputs = jax.random.normal(jax.random.PRNGKey(42), (2, 8, 4, 64), dtype=jnp.float32)
    position = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (2, 8))
    for interleave in (True, False):
      outs = []
      for direct_freqs in (False, True):
        layer = embeddings.YarnRotaryEmbedding(
            embedding_dims=64,
            mesh=self.mesh,
            max_position_embeddings=16384,
            original_max_position_embeddings=4096,
            interleave=interleave,
            direct_position_freqs=direct_freqs,
            fprop_dtype=jnp.float32,
            rngs=self.rngs,
        )
        outs.append(layer(inputs, position=position))
      np.testing.assert_allclose(outs[1], outs[0], rtol=1e-5, atol=1e-5)

  def test_non_interleaved_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=False,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # For all-ones input, the output of non-interleaved RoPE matches interleaved RoPE
    np.testing.assert_allclose(outputs, default_outputs, atol=1e-5)

  def test_explicit_shard_mode_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default shard_mode (AUTO) implementation
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    np.testing.assert_allclose(outputs, default_outputs, atol=1e-5)

  def test_direct_position_freqs_matches_indexing_the_table(self):
    """`direct_position_freqs` computes the rows the table lookup would read, bit for bit.

    Row `p` of `freqs_cis` is `exp(1j * p * inv_freqs)`, so for every in-range position
    the direct computation and the table lookup agree exactly — `arange(max_position_embeddings)[p]`
    is exactly `float(p)` for every representable position — on every rotation path.
    """
    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 128, 2, 64), dtype=jnp.float32)
    for interleave, pairwise in ((True, False), (False, False), (True, True)):
      table_layer, direct_layer = (
          embeddings.YarnRotaryEmbedding(
              embedding_dims=64,
              mesh=self.mesh,
              max_position_embeddings=163840,
              original_max_position_embeddings=4096,
              interleave=interleave,
              pairwise=pairwise,
              direct_position_freqs=direct,
              fprop_dtype=jnp.float32,
              rngs=self.rngs,
          )
          for direct in (False, True)
      )
      self.assertEqual(table_layer.freqs_cis.shape, (163840, 32))
      # Sample the bottom, the middle and the very top of the table.
      for lo in (0, 4096, 163840 - 128):
        with self.subTest(interleave=interleave, pairwise=pairwise, lo=lo):
          position = jnp.arange(lo, lo + 128, dtype=jnp.int32)[jnp.newaxis, :]
          np.testing.assert_array_equal(direct_layer(inputs, position=position), table_layer(inputs, position=position))

  def test_pairwise_explicit_shard_mode_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=True,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (pairwise=False, interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=False,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # Convert default concatenated layout to interleaved layout for comparison
    expected_interleaved = jnp.stack([default_outputs[..., :2], default_outputs[..., 2:]], axis=-1).reshape(
        default_outputs.shape
    )
    np.testing.assert_allclose(outputs, expected_interleaved, atol=1e-5)

  def _complex64_reference(self, layer, inputs, position):
    """The complex64 rotation the default (pairwise=False) path used before the float32 rewrite."""
    freqs = layer._gather_freqs_cis(position)  # pylint: disable=protected-access
    if layer.interleave:
      b, s, n, h = inputs.shape
      pairs = inputs.reshape(b, s, n, h // 2, 2)
      first_half, second_half = pairs[..., 0], pairs[..., 1]
    else:
      first_half, second_half = jnp.split(inputs, 2, axis=-1)
    rotated = jnp.multiply(first_half + 1j * second_half, freqs)
    output = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
    if layer.attention_scaling:
      output = output * (0.1 * np.log(layer.rope_factor) + 1.0)
    return output.astype(layer.fprop_dtype)

  @staticmethod
  def _rotation_is_bitwise():
    """Whether the float32 rotation must be bitwise equal to the complex64 multiply on this backend.

    Both compute (ac - bd, ad + bc) in float32, so they are bitwise equal unless the compiler
    contracts a product and a subtraction into one FMA in one program and not in the other.
    TPU does not; XLA:CPU does on FMA hardware. Run on CPU with XLA_FLAGS=--xla_cpu_max_isa=AVX
    (no FMA) to check bitwise equality there.
    """
    return jax.devices()[0].platform == "tpu" or "--xla_cpu_max_isa=AVX" in os.environ.get("XLA_FLAGS", "")

  def _check_rotation(self, out, ref):
    self.assertEqual(out.dtype, jnp.float32)
    self.assertEqual(ref.dtype, jnp.float32)
    out, ref = np.asarray(out), np.asarray(ref)
    if self._rotation_is_bitwise():
      np.testing.assert_array_equal(out, ref)
    else:
      # An FMA changes ac - bd by at most about one float32 ulp of the products, i.e. of max |ref|.
      np.testing.assert_allclose(out, ref, rtol=0, atol=2**-21 * float(np.max(np.abs(ref))))

  def _default_layer(self, interleave, attention_scaling=False):
    return embeddings.YarnRotaryEmbedding(
        embedding_dims=64,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=interleave,
        attention_scaling=attention_scaling,
        fprop_dtype=jnp.float32,
        rngs=self.rngs,
    )

  def test_default_path_matches_complex64_under_jit(self):
    """Under jit, the float32 cos/sin rotation equals the old complex64 multiply (see _rotation_is_bitwise)."""
    position = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32)[None, :] * 37, (2, 64))
    for dtype in (jnp.float32, jnp.bfloat16):
      inputs = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 4, 64), dtype=dtype)
      for interleave in (True, False):
        for attention_scaling in (False, True):
          with self.subTest(dtype=dtype.__name__, interleave=interleave, attention_scaling=attention_scaling):
            layer = self._default_layer(interleave, attention_scaling)
            out = jax.jit(lambda x, p, l=layer: l(x, position=p))(inputs, position)
            ref = jax.jit(lambda x, p, l=layer: self._complex64_reference(l, x, p))(inputs, position)
            self._check_rotation(out, ref)

  def test_default_path_has_no_complex_multiply(self):
    """The default path rotates in float32, so op-by-op (disable_jit) runs compile no complex64 multiply."""

    def complex_muls(jaxpr):
      found = []
      for eqn in jaxpr.eqns:
        if eqn.primitive.name == "mul" and any(jnp.issubdtype(v.aval.dtype, jnp.complexfloating) for v in eqn.outvars):
          found.append(eqn)
        for param in eqn.params.values():
          sub = getattr(param, "jaxpr", param)
          if hasattr(sub, "eqns"):
            found.extend(complex_muls(sub))
      return found

    for interleave in (True, False):
      layer = embeddings.YarnRotaryEmbedding(
          embedding_dims=64,
          mesh=self.mesh,
          max_position_embeddings=16384,
          original_max_position_embeddings=4096,
          interleave=interleave,
          rngs=self.rngs,
      )
      inputs = jnp.ones((1, 8, 2, 64), dtype=jnp.bfloat16)
      position = jnp.arange(8, dtype=jnp.int32)[None, :]
      closed = jax.make_jaxpr(lambda x, p, l=layer: l(x, position=p))(inputs, position)
      self.assertEqual(complex_muls(closed.jaxpr), [], msg=f"interleave={interleave}")

  def test_default_path_runs_under_disable_jit(self):
    """Op by op (jax.disable_jit), the rotation runs and equals the old complex64 multiply run the same way.

    The eager result is compared with the eager complex64 reference, not with the jit result: under jit
    XLA constant-folds the frequency table (pow, cos, sin) on the host, so the table itself can differ
    from the eager table by an ulp. That gap predates this rotation and does not depend on it.
    """
    inputs = jax.random.normal(jax.random.PRNGKey(5), (2, 16, 4, 64), dtype=jnp.bfloat16)
    position = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (2, 16))
    for interleave in (True, False):
      with self.subTest(interleave=interleave):
        layer = self._default_layer(interleave)
        with jax.disable_jit():
          eager_out = layer(inputs, position=position)
          eager_ref = self._complex64_reference(layer, inputs, position)
        self._check_rotation(eager_out, eager_ref)
        jit_out = jax.jit(lambda x, p, l=layer: l(x, position=p))(inputs, position)
        np.testing.assert_allclose(np.asarray(eager_out), np.asarray(jit_out), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
