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

"""DeepSeek-V4.1 tests that need neither PyTorch nor the reference implementation.

The parity tests against the plain-PyTorch reference are in deepseek_v41_vs_reference_test.py.
"""

import sys
import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.configs import pyconfig
from maxtext.layers import engram_deepseek41
from maxtext.models import deepseek41

_FP4_E2M1 = np.array(param_mapping._FP4_E2M1_VALUES, np.float32)  # pylint: disable=protected-access


def _scales(rng, shape):
  return np.exp2(rng.integers(-8, 4, shape)).astype(ml_dtypes.float8_e8m0fnu)


def _expand(scale, rows, cols):
  s = scale.astype(np.float32)
  return np.repeat(np.repeat(s, -(-rows // s.shape[0]), axis=0)[:rows], -(-cols // s.shape[1]), axis=1)[:, :cols]


def _fp8_case(rng, rows, cols, scale_shape):
  weight = (rng.standard_normal((rows, cols)) * 4).astype(ml_dtypes.float8_e4m3fn)
  scale = _scales(rng, scale_shape)
  return weight, scale, weight.astype(np.float32) * _expand(scale, rows, cols)


def _fp4_case(rng, rows, cols, scale_shape):
  codes = rng.integers(0, 16, (rows, cols))
  packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).view(np.int8)
  scale = _scales(rng, scale_shape)
  return packed, scale, _FP4_E2M1[codes] * _expand(scale, rows, cols)


class DequantizeTest(parameterized.TestCase):
  """`param_mapping.dequantize_block_scaled` on numpy, single-device and row-split inputs, against numpy."""

  def _check(self, weight, scale, expected):
    """Checks the numpy, single-device and row-split device paths against `expected`."""
    np.testing.assert_array_equal(
        np.asarray(param_mapping.dequantize_block_scaled((weight, scale)), np.float32), expected
    )
    got = param_mapping.dequantize_block_scaled((jnp.asarray(weight), jnp.asarray(scale)))
    self.assertEqual(got.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got, np.float32), expected)
    # As the streaming loader hands them over: the weight split by rows over the chips, the scales either a full copy
    # or (with one scale row per weight row) split the same way.
    devices = jax.devices()
    for n in sorted({d for d in (2, 4, 8, len(devices)) if 1 < d <= len(devices) and weight.shape[0] % d == 0}):
      w = jax.device_put(weight, hf_streaming_load.choose_load_sharding(weight.shape, weight.dtype, devices[:n], 0))
      for split_scale in (False, True):
        if split_scale and scale.shape[0] != weight.shape[0]:
          continue
        min_bytes = 0 if split_scale else 1 << 62
        s = jax.device_put(
            scale, hf_streaming_load.choose_load_sharding(scale.shape, scale.dtype, devices[:n], min_bytes)
        )
        got = param_mapping.dequantize_block_scaled((w, s))
        self.assertEqual(got.sharding, w.sharding)
        np.testing.assert_array_equal(
            np.asarray(got, np.float32), expected, err_msg=f"{n} chips, split scales {split_scale}"
        )

  @parameterized.named_parameters(
      ("fp8_32x32_blocks", 96, 160, (3, 5)),
      # 256 rows over 2 / 4 / 8 chips: 128 / 64 / 32 rows each, whole scale blocks (the 64-chip v5p `wq_b` case).
      ("fp8_shards_aligned_to_blocks", 256, 96, (8, 3)),
      # 8x32 blocks: over 8 chips each holds 8 weight rows, as many as the scale has rows (the replicated scale must
      # still be indexed by block, e.g. 32x32 blocks over 32 chips).
      ("fp8_rows_per_chip_equal_scale_rows", 64, 64, (8, 2)),
      ("fp8_1x32_rows", 200, 64, (200, 2)),
  )
  def test_fp8(self, rows, cols, scale_shape):
    self._check(*_fp8_case(np.random.default_rng(rows), rows, cols, scale_shape))

  def test_fp4_packed(self):
    self._check(*_fp4_case(np.random.default_rng(0), 64, 128, (64, 4)))


def _reference_hashes(compressed, pad, multipliers, primes, offsets, max_ngram_size, n_heads):
  """The upstream n-gram hash in plain Python integers (int64 semantics; every product fits)."""
  b, s = compressed.shape
  out = np.zeros((b, s, (max_ngram_size - 1) * n_heads), np.int64)
  for i in range(b):
    for t in range(s):
      tokens = [int(compressed[i, t - k]) if t - k >= 0 else pad for k in range(max_ngram_size)]
      rolling = tokens[0] * int(multipliers[0])
      for k in range(1, max_ngram_size):
        rolling ^= tokens[k] * int(multipliers[k])
        for h in range(n_heads):
          col = (k - 1) * n_heads + h
          out[i, t, col] = rolling % int(primes[col]) + int(offsets[col])
  return out


class EngramHashTest(parameterized.TestCase):
  """The uint32-limb engram hash equals the upstream int64 hash, at the real (~16M) bucket sizes."""

  def test_matches_int64_hash(self):
    rng = np.random.default_rng(0)
    vocab, compressed_vocab, layer_ids, max_ngram_size, n_heads, pad_id = 1000, 99092, (1, 14), 4, 8, 2
    token_map = rng.integers(0, compressed_vocab, vocab)
    ids = rng.integers(0, vocab, (2, 12))
    multipliers = engram_deepseek41.compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    primes = engram_deepseek41.engram_primes(layer_ids, max_ngram_size, n_heads, 16_000_000)
    for index, layer_id in enumerate(layer_ids):
      hasher = engram_deepseek41.DeepSeek41NgramHasher(
          token_map, compressed_vocab, pad_id, layer_ids, layer_id, max_ngram_size, n_heads, 16_000_000
      )
      flat = np.array([p for per_ngram in primes[index] for p in per_ngram], np.int64)
      offsets = np.concatenate([[0], np.cumsum(flat)[:-1]])
      want = _reference_hashes(
          token_map[ids], int(token_map[pad_id]), multipliers[index], flat, offsets, max_ngram_size, n_heads
      )
      np.testing.assert_array_equal(np.asarray(hasher(jnp.asarray(ids))), want)


class LayerStructureTest(parameterized.TestCase):
  """The released config's layer roles and the scanned segments built from them."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="deepseek41_structure_test",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        model_name="deepseek4.1-flash",
    )

  def test_layer_specs(self):
    specs = deepseek41.build_layer_specs(self.config)
    self.assertLen(specs, 40)
    self.assertEqual([s.layer_id for s in specs if s.is_kv_source], [2, 8, 14, 20])
    self.assertEqual([s.layer_id for s in specs if s.is_index_source], [2, 8, 14, 20, 24, 28, 32, 36])
    self.assertEqual([s.layer_id for s in specs if s.is_candidate_source], [20])
    # Every layer after the candidate source; the index sources among them restrict their top-k to the candidates.
    self.assertEqual([s.layer_id for s in specs if s.uses_candidates], list(range(21, 40)))
    self.assertEqual([s.layer_id for s in specs if s.has_engram], [1, 14])

  def test_segments(self):
    segments = deepseek41.build_segments(self.config)
    layout = [(seg.groups[0][0], seg.groups[0][-1], seg.repeats) for seg in segments]
    # [0] [1] [2-7]x2 [14-19] [20-23] [24-27]x4: every layer exactly once, in order.
    self.assertEqual(layout, [(0, 0, 1), (1, 1, 1), (2, 7, 2), (14, 19, 1), (20, 23, 1), (24, 27, 4)])
    self.assertEqual([i for seg in segments for group in seg.groups for i in group], list(range(40)))


if __name__ == "__main__":
  unittest.main()
