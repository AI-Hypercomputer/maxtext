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

"""Pre-tokenized C4 ArrayRecord input pipeline for long-context Qwen3.5.

Reads pre-tokenized Qwen3.5 C4 rows matching the MaxText Grain `arrayrecord`
format (`tokenize_train_data=false`): each record is a `tf.train.Example` whose
`text` int64 feature holds `seq_len` token ids. Following MaxText causal LM
semantics, `targets` = `tokens` shifted left by one, and the final position
(which has no next-token target) is excluded from the loss via `segment_ids = 0`.

Global row of `(step, b)` is `(step * global_batch + b) % num_rows`. Every host
builds its addressable shards with `jax.make_array_from_callback`, so it decodes
only the rows its devices hold (and keeps only their CP sequence slices).
"""

from __future__ import annotations

import bisect
import concurrent.futures
import functools

import jax
import numpy as np
import tensorflow as tf

from array_record.python import array_record_module

C4_ARRAY_RECORD_DIR = ""
TOKEN_FEATURE = "text"


def parse_row(record: bytes) -> np.ndarray:
  """Decodes one serialized `tf.train.Example` into its int32 token row."""
  example = tf.train.Example.FromString(record)
  return np.asarray(example.features.feature[TOKEN_FEATURE].int64_list.value, dtype=np.int32)


class ArrayRecordC4:
  """Random-access reader over sharded pre-tokenized C4 ArrayRecord files."""

  def __init__(self, data_dir: str, seq_len: int):
    if not data_dir:
      raise ValueError("data_dir must be provided when using ArrayRecordC4 (--c4_data_dir).")
    pattern = f"{data_dir.rstrip('/')}/*.array_record"
    self._paths = sorted(tf.io.gfile.glob(pattern))
    if not self._paths:
      raise FileNotFoundError(f"No *.array_record files matching {pattern}")
    self._seq_len = seq_len
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._paths)) as pool:
      self._readers = list(pool.map(array_record_module.ArrayRecordReader, self._paths))
    counts = [r.num_records() for r in self._readers]
    self._offsets = np.cumsum([0] + counts).tolist()
    self.num_rows = self._offsets[-1]

  def read_row(self, row: int) -> np.ndarray:
    file_idx = bisect.bisect_right(self._offsets, row) - 1
    tokens = parse_row(self._readers[file_idx].read([row - self._offsets[file_idx]])[0])
    if tokens.shape != (self._seq_len,):
      raise ValueError(f"row {row}: expected {self._seq_len} tokens, got {tokens.shape}")
    return tokens

  def make_batch(
      self, step: int, global_batch: int, sharding: jax.sharding.Sharding
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns sharded global `(tokens, targets, positions, segment_ids)` for `step`."""
    shape = (global_batch, self._seq_len)

    @functools.cache
    def _row(b: int) -> np.ndarray:
      return self.read_row((step * global_batch + b) % self.num_rows)

    def _rows(index) -> np.ndarray:
      b_idx = range(*index[0].indices(global_batch))
      return np.stack([_row(b) for b in b_idx])

    def _tokens(index):
      return _rows(index)[:, index[1]]

    def _targets(index):
      rows = _rows(index)
      shifted = np.concatenate([rows[:, 1:], np.zeros_like(rows[:, :1])], axis=1)
      return shifted[:, index[1]]

    def _positions(index):
      b = len(range(*index[0].indices(global_batch)))
      pos = np.arange(self._seq_len, dtype=np.int32)[index[1]]
      return np.broadcast_to(pos[None, :], (b, pos.size)).copy()

    def _segment_ids(index):
      b = len(range(*index[0].indices(global_batch)))
      seg = np.ones((self._seq_len,), dtype=np.int32)
      seg[-1] = 0  # The last token has no next-token target.
      seg = seg[index[1]]
      return np.broadcast_to(seg[None, :], (b, seg.size)).copy()

    return tuple(
        jax.make_array_from_callback(shape, sharding, fn) for fn in (_tokens, _targets, _positions, _segment_ids)
    )
