# Copyright 2023–2026 Google LLC
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

"""Tests for the NNX UMT5 text encoder (`text_encoder_wan.py`)."""

import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from flax import nnx
from flax.traverse_util import flatten_dict
import numpy as np
import transformers

from maxtext.m3.models.wan import utils_wan
from maxtext.m3.models.wan.text_encoder_wan import UMT5EncoderModel, relative_position_bucket

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

# Tiny UMT5 (umt5-xxl: d_model 4096, 24 layers, 64 heads x 64, d_ff 10240, 32 buckets, max distance 128).
_CONFIG = {
    "vocab_size": 96,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 48,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "layer_norm_epsilon": 1e-6,
}


class UMT5EncoderModelTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.model = UMT5EncoderModel(**_CONFIG, rngs=nnx.Rngs(0))

  def test_output_shape(self):
    input_ids = np.random.default_rng(0).integers(1, _CONFIG["vocab_size"], size=(2, 12))
    out = self.model(input_ids, np.ones_like(input_ids))
    self.assertEqual(out.shape, (2, 12, _CONFIG["d_model"]))
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_padding_does_not_change_valid_tokens(self):
    """Masked (padded) positions must not affect the outputs of the valid ones."""
    length, valid = 12, 5
    rng = np.random.default_rng(1)
    input_ids = rng.integers(1, _CONFIG["vocab_size"], size=(1, length))
    mask = (np.arange(length) < valid)[None].astype(np.int32)
    other_padding = np.where(mask > 0, input_ids, rng.integers(1, _CONFIG["vocab_size"], size=(1, length)))
    out = np.asarray(self.model(input_ids, mask))[:, :valid]
    np.testing.assert_allclose(np.asarray(self.model(other_padding, mask))[:, :valid], out, atol=1e-5, rtol=1e-5)

  def test_params_are_tensor_parallel(self):
    tp = ("data", "fsdp", "context", "tensor")
    sharding = utils_wan.WanSharding.from_rules(utils_wan.yml_rules("text_encoder_logical_axis_rules"))
    model = nnx.eval_shape(lambda: UMT5EncoderModel(**_CONFIG, sharding=sharding, rngs=nnx.Rngs(0)))
    specs = flatten_dict(nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(model, nnx.Param))))
    self.assertEqual(tuple(specs[("token_embedding", "embedding")]), (tp, None))
    self.assertEqual(tuple(specs[("blocks", 0, "attn", "query", "kernel")]), (None, tp))
    self.assertEqual(tuple(specs[("blocks", 0, "attn", "proj_attn", "kernel")]), (tp, None))
    self.assertEqual(tuple(specs[("blocks", 0, "ffn", "wo", "kernel")]), (tp, None))
    self.assertEqual(tuple(specs[("final_norm", "scale")]), (None,))


class RelativePositionBucketTest(unittest.TestCase):

  def test_buckets_in_range_and_signed(self):
    relative_position = np.arange(-64, 65)
    buckets = np.asarray(relative_position_bucket(relative_position, 8, 16))
    self.assertTrue(np.all((buckets >= 0) & (buckets < 8)))
    # Bidirectional: negative and positive offsets use disjoint halves of the buckets.
    self.assertTrue(set(buckets[relative_position < 0]).isdisjoint(set(buckets[relative_position > 0])))

  @unittest.skipIf(torch is None, "needs torch (Hugging Face reference UMT5Attention)")
  def test_matches_hf(self):
    for num_buckets, max_distance in ((32, 128), (8, 16)):
      config = transformers.UMT5Config.from_dict(
          {"relative_attention_num_buckets": num_buckets, "relative_attention_max_distance": max_distance}
      )
      attention = transformers.models.umt5.modeling_umt5.UMT5Attention(config, has_relative_attention_bias=True)
      relative_position = np.arange(-2048, 2049)
      # pylint: disable-next=protected-access
      expected = attention._relative_position_bucket(torch.from_numpy(relative_position)).numpy()
      np.testing.assert_array_equal(relative_position_bucket(relative_position, num_buckets, max_distance), expected)


if __name__ == "__main__":
  unittest.main()
