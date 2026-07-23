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
"""Tests for local Tokamax SplashAttention masks."""

# pylint: disable=protected-access

from absl.testing import absltest
import numpy as np

from maxtext.kernels.tokamax_splash_attention import splash_attention_mask
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info
from maxtext.layers.attention_op import BlockCausalMask


class TokamaxSplashAttentionMaskTest(absltest.TestCase):

  def test_causal_state_grid_uses_block_extents(self):
    mask = splash_attention_mask.CausalMask((8, 8), shard_count=2)

    got = splash_attention_mask_info._causal_state_grid(mask, q_block_size=2, kv_block_size=2)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 2, 1, 0],
            [2, 2, 2, 1],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(got, expected)

  def test_process_causal_mask_keeps_lazy_mask_function(self):
    mask = splash_attention_mask.CausalMask((8, 8), shard_count=2)

    mask_info, mask_function = splash_attention_mask_info.process_mask(
        mask,
        block_shape=(2, 2),
        q_seq_shards=2,
        kv_seq_shards=2,
        return_dynamic_grid=True,
    )

    self.assertIs(mask_function, mask.mask_function)
    self.assertIsNotNone(mask_info.q_sequence)
    self.assertIsNone(mask_info.partial_mask_blocks)
    np.testing.assert_array_equal(mask_info.q_sequence, np.arange(8, dtype=np.int32))

  def test_process_block_causal_mask_keeps_lazy_mask_function(self):
    mask = BlockCausalMask((8, 8), block_size=4)

    mask_info, mask_function = splash_attention_mask_info.process_mask(
        mask,
        block_shape=(2, 2),
        q_seq_shards=1,
        kv_seq_shards=1,
        return_dynamic_grid=True,
    )

    self.assertIs(mask_function, mask.mask_function)
    self.assertIsNotNone(mask_info.q_sequence)
    self.assertIsNone(mask_info.partial_mask_blocks)
    np.testing.assert_array_equal(mask_info.q_sequence, np.arange(8, dtype=np.int32))


if __name__ == "__main__":
  absltest.main()
