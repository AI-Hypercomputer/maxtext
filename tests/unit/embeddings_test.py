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
    bf16 cast under `shard_mode: explicit`. It reassociates the accumulation, so
    the logits agree to float rounding rather than bit for bit.
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


if __name__ == "__main__":
  unittest.main()
