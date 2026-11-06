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

"""Tests for independent prefill and decode KV-cache batch sharding."""

import unittest

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.inference import kvcache
from maxtext.utils import maxtext_utils_nnx, sharding


class KVCacheBatchShardingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 2:
      self.skipTest("Requires two devices, including virtual CPU devices")
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("fsdp",))
    self.rules = (
        ("cache_batch_prefill", ()),
        ("cache_batch", ("fsdp",)),
        ("cache_sequence", ()),
        ("cache_heads", ()),
        ("cache_kv", ()),
    )

  def _create_cache(self, model_mode, batch, is_gdn):
    """Construct a cache using the model initialization sharding path."""

    def initialize():
      return kvcache.KVCache(
          max_prefill_length=2,
          max_target_length=4,
          batch=batch,
          key_seq_len=2,
          value_seq_len=2,
          key_heads=1,
          value_heads=1,
          key_head_size=2,
          value_head_size=2,
          dtype=jnp.bfloat16,
          model_mode=model_mode,
          is_gdn=is_gdn,
          conv_kernel_size=2,
          conv_dim=2,
      )

    with jax.set_mesh(self.mesh), nn.logical_axis_rules(self.rules):
      abstract = nnx.eval_shape(initialize)
      _, state = nnx.split(abstract)
      named_sharding = sharding.nnx_construct_named_sharding(state, self.mesh)
      return maxtext_utils_nnx.create_nnx_sharded_model(
          abstract, initialize, mesh=self.mesh, named_sharding=named_sharding
      )

  def test_batch_one_prefill_with_sharded_decode_batch(self):
    for is_gdn in (False, True):
      with self.subTest(is_gdn=is_gdn):
        cache = self._create_cache(MODEL_MODE_PREFILL, batch=1, is_gdn=is_gdn)
        lengths = cache.cached_ar_lengths[...]
        self.assertEqual(lengths.shape, (1,))
        self.assertTrue(lengths.sharding.is_fully_replicated)
        np.testing.assert_array_equal(np.asarray(lengths), [0])

  def test_decode_lengths_remain_batch_sharded(self):
    for is_gdn in (False, True):
      with self.subTest(is_gdn=is_gdn):
        cache = self._create_cache(MODEL_MODE_AUTOREGRESSIVE, batch=2, is_gdn=is_gdn)
        lengths = cache.cached_ar_lengths[...]
        self.assertEqual(lengths.sharding.spec, jax.sharding.PartitionSpec("fsdp"))
        self.assertEqual([s.data.shape for s in lengths.addressable_shards], [(1,), (1,)])
        np.testing.assert_array_equal(np.asarray(lengths), [0, 0])


if __name__ == "__main__":
  unittest.main()
