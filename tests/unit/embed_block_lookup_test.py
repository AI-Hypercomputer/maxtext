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

"""Tests for the decode-time embedding block lookup in embeddings.py (env MAXTEXT_EMBED_BLOCK_LOOKUP, default OFF)."""

import os
import sys
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import embeddings
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_OFF_ENV = {embeddings.EMBED_BLOCK_LOOKUP_ENV: "0"}
_ON_ENV = {embeddings.EMBED_BLOCK_LOOKUP_ENV: "1"}


def _bits(x):
  x = np.asarray(x)
  return x.view({2: np.uint16, 4: np.uint32}[x.dtype.itemsize])


class EmbedBlockLookupHelperTest(unittest.TestCase):
  """Tests for the module-level block-lookup helpers."""

  def test_gate_defaults_off(self):
    env = {k: v for k, v in os.environ.items() if k != embeddings.EMBED_BLOCK_LOOKUP_ENV}
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertFalse(embeddings.embed_block_lookup_enabled())
    for value, expected in (("0", False), ("", False), ("yes", False), ("1", True), ("true", True), ("True", True)):
      with mock.patch.dict(os.environ, {embeddings.EMBED_BLOCK_LOOKUP_ENV: value}):
        self.assertEqual(embeddings.embed_block_lookup_enabled(), expected, value)

  def test_max_tokens_default_and_override(self):
    env = {k: v for k, v in os.environ.items() if k != embeddings.EMBED_BLOCK_LOOKUP_MAX_TOK_ENV}
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(embeddings.embed_block_lookup_max_tokens(), 4096)
    with mock.patch.dict(os.environ, {embeddings.EMBED_BLOCK_LOOKUP_MAX_TOK_ENV: "1024"}):
      self.assertEqual(embeddings.embed_block_lookup_max_tokens(), 1024)

  def test_bit_exact_against_stock_gather(self):
    num_rows, num_features = 64, 8
    rng = np.random.default_rng(0)
    ids = jnp.array([[0, 1, 15, 16, 17, 63], [-1, -16, -64, 64, 1000, -1000]], dtype=jnp.int32)
    for dtype in (jnp.bfloat16, jnp.float32):
      table = rng.standard_normal((num_rows, num_features)).astype(np.float32)
      table[3] = -0.0  # a float sum would turn this row into +0.0
      table[5, 0] = np.nan
      table = jnp.asarray(table, dtype)
      expected = table.at[ids].get()
      got = jax.jit(embeddings.embed_block_lookup)(table, ids)
      self.assertEqual(got.dtype, expected.dtype)
      self.assertEqual(got.shape, expected.shape)
      np.testing.assert_array_equal(_bits(got), _bits(expected))

  def test_tokens_per_device(self):
    # pylint: disable=protected-access
    mesh = mock.Mock()
    mesh.shape = {"data": 2, "attn_dp": 4, "model": 8}
    self.assertEqual(embeddings._embed_block_tokens_per_device(jnp.zeros((64, 1), jnp.int32), mesh), (8, 8))
    self.assertEqual(embeddings._embed_block_tokens_per_device(jnp.zeros((3, 3), jnp.int32), mesh), (2, 8))
    self.assertEqual(embeddings._embed_block_tokens_per_device(jnp.zeros((5,), jnp.int32), None), (5, 1))


class EmbedBlockLookupLayerTest(unittest.TestCase):
  """Tests for the block-lookup dispatch inside Embed.__call__."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(
        argv, per_device_batch_size=1.0, run_name="test", enable_checkpointing=False, max_target_length=128
    )
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def _layer(self, num_embeddings=128):
    return embeddings.Embed(
        num_embeddings=num_embeddings, num_features=16, config=self.cfg, mesh=self.mesh, rngs=self.rngs
    )

  def _call(self, layer, inputs, model_mode, env):
    with mock.patch.dict(os.environ, env), mock.patch.object(
        embeddings, "embed_block_lookup", wraps=embeddings.embed_block_lookup
    ) as spy:
      out = layer(inputs, model_mode=model_mode)
    return out, spy.call_count

  def test_off_keeps_stock_lookup(self):
    layer = self._layer()
    inputs = jnp.array([[3], [127]], dtype=jnp.int32)
    _, calls = self._call(layer, inputs, MODEL_MODE_AUTOREGRESSIVE, _OFF_ENV)
    self.assertEqual(calls, 0)

  def test_on_autoregressive_matches_stock(self):
    layer = self._layer()
    inputs = jnp.array([[3], [127]], dtype=jnp.int32)
    stock, _ = self._call(layer, inputs, MODEL_MODE_AUTOREGRESSIVE, _OFF_ENV)
    block, calls = self._call(layer, inputs, MODEL_MODE_AUTOREGRESSIVE, _ON_ENV)
    self.assertEqual(calls, 1)
    np.testing.assert_array_equal(_bits(block), _bits(stock))

  def test_on_keeps_stock_outside_its_scope(self):
    inputs = jnp.array([[3, 4, 5], [6, 7, 8]], dtype=jnp.int32)
    _, calls = self._call(self._layer(), inputs, MODEL_MODE_TRAIN, _ON_ENV)
    self.assertEqual(calls, 0, "train mode must keep the stock lookup")
    _, calls = self._call(self._layer(num_embeddings=100), inputs, MODEL_MODE_AUTOREGRESSIVE, _ON_ENV)
    self.assertEqual(calls, 0, "a table whose rows are not a multiple of 16 must keep the stock lookup")
    capped = dict(_ON_ENV, **{embeddings.EMBED_BLOCK_LOOKUP_MAX_TOK_ENV: "1"})
    _, calls = self._call(self._layer(), inputs, MODEL_MODE_AUTOREGRESSIVE, capped)
    self.assertEqual(calls, 0, "calls above MAXTEXT_EMBED_BLOCK_LOOKUP_MAX_TOK must keep the stock lookup")


if __name__ == "__main__":
  unittest.main()
