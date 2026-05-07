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

"""Unit tests for ``maxtext.input_pipeline.olmo_data``.

Covers test plan items A.1 (index correctness), A.2 (global→local mapping),
A.7 (fingerprint stability), and A.8 (n-gram instance filter). Tests for
the loader paths (Options 1 & 2) live alongside their respective modules.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Tuple

import numpy as np

from maxtext.input_pipeline.olmo_data import (
    OlmoNpyFileEntry,
    build_index,
    compute_fingerprint,
    find_periodic_sequences,
    global_to_local,
    is_clean_instance,
    load_index,
    read_npy_header_from_path,
)


def _write_uint32_npy(tmpdir: str, name: str, values: np.ndarray) -> str:
  """Write a 1-D uint32 array to ``tmpdir/name`` and return the path."""
  assert values.dtype == np.uint32
  assert values.ndim == 1
  path = os.path.join(tmpdir, name)
  np.save(path, values)
  # numpy adds .npy if missing
  return path if path.endswith(".npy") else path + ".npy"


def _make_synthetic_mix(tmpdir: str, sizes: Tuple[int, ...]) -> Tuple[str, ...]:
  """Make a mix of files where file i contains tokens [start_i .. start_i + size_i)."""
  paths = []
  start = 0
  for i, n in enumerate(sizes):
    arr = np.arange(start, start + n, dtype=np.uint32)
    paths.append(_write_uint32_npy(tmpdir, f"file_{i:03d}.npy", arr))
    start += n
  return tuple(paths)


def _stub_reader(spec):
  """Return a header_reader that pretends each path has the given (dtype, shape)."""

  def _reader(path: str):
    return spec[path]

  return _reader


class IndexCorrectnessTest(unittest.TestCase):
  """A.1: index counts and offsets match the underlying files."""

  def test_index_total_counts_match_files(self):
    with tempfile.TemporaryDirectory() as d:
      sizes = (10, 7, 25, 16)
      paths = _make_synthetic_mix(d, sizes)
      seq = 4

      idx = build_index(
          [(p, "label") for p in paths],
          sequence_length=seq,
          tokenizer="test",
      )

      self.assertEqual(idx.sequence_length, seq)
      self.assertEqual(idx.dtype, "uint32")
      self.assertEqual(idx.total_tokens, sum(sizes))
      # Trailing tokens are dropped per OLMo-core convention.
      expected_instances = sum(s // seq for s in sizes)
      self.assertEqual(idx.total_instances, expected_instances)

      cum = 0
      for entry, n_tokens in zip(idx.files, sizes):
        self.assertEqual(entry.n_tokens, n_tokens)
        self.assertEqual(entry.n_instances, n_tokens // seq)
        self.assertEqual(entry.instance_offset, cum)
        cum += entry.n_instances

  def test_round_trip_save_load(self):
    with tempfile.TemporaryDirectory() as d:
      paths = _make_synthetic_mix(d, (32, 64))
      idx = build_index([(p, "lab") for p in paths], sequence_length=8, tokenizer="t")
      out = os.path.join(d, "idx.json")
      idx.save(out)
      restored = load_index(out)
      self.assertEqual(restored.fingerprint, idx.fingerprint)
      self.assertEqual(restored.total_instances, idx.total_instances)
      self.assertEqual(len(restored.files), len(idx.files))
      for a, b in zip(restored.files, idx.files):
        self.assertEqual(a, b)

  def test_dtype_mismatch_raises(self):
    spec = {
        "a.npy": ("uint32", (10,)),
        "b.npy": ("uint16", (10,)),
    }
    with self.assertRaisesRegex(ValueError, "Heterogeneous"):
      build_index(
          [("a.npy", "x"), ("b.npy", "y")],
          sequence_length=4,
          tokenizer="t",
          header_reader=_stub_reader(spec),
      )

  def test_non_1d_raises(self):
    spec = {"a.npy": ("uint32", (10, 2))}
    with self.assertRaisesRegex(ValueError, "1-D"):
      build_index(
          [("a.npy", "x")],
          sequence_length=4,
          tokenizer="t",
          header_reader=_stub_reader(spec),
      )

  def test_empty_paths_raises(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      build_index([], sequence_length=4, tokenizer="t")


class GlobalToLocalTest(unittest.TestCase):
  """A.2: every global instance maps to the expected (file, token offset)."""

  def setUp(self):
    self.spec = {
        "a.npy": ("uint32", (10,)),  # 2 instances at seq=4
        "b.npy": ("uint32", (7,)),  # 1 instance
        "c.npy": ("uint32", (25,)),  # 6 instances
    }
    self.idx = build_index(
        [("a.npy", "x"), ("b.npy", "y"), ("c.npy", "z")],
        sequence_length=4,
        tokenizer="t",
        header_reader=_stub_reader(self.spec),
    )

  def test_first_index_each_file(self):
    self.assertEqual(global_to_local(self.idx, 0), (0, 0))
    self.assertEqual(global_to_local(self.idx, 2), (1, 0))
    self.assertEqual(global_to_local(self.idx, 3), (2, 0))

  def test_last_index_each_file(self):
    # File 0 has 2 instances → last is global 1, local-token 4.
    self.assertEqual(global_to_local(self.idx, 1), (0, 4))
    # File 1 has 1 instance → only index 2.
    # File 2 last instance is global 8, local-token 20.
    self.assertEqual(global_to_local(self.idx, 8), (2, 20))

  def test_full_partition_is_a_function(self):
    # For each global index, the (file, offset) pair is well-defined and
    # within bounds of the file's instance count.
    for i in range(self.idx.total_instances):
      file_idx, tok_off = global_to_local(self.idx, i)
      f = self.idx.files[file_idx]
      local_inst = tok_off // self.idx.sequence_length
      self.assertGreaterEqual(local_inst, 0)
      self.assertLess(local_inst, f.n_instances)

  def test_out_of_range_raises(self):
    with self.assertRaises(IndexError):
      global_to_local(self.idx, -1)
    with self.assertRaises(IndexError):
      global_to_local(self.idx, self.idx.total_instances)

  def test_real_npy_round_trip(self):
    """Build index from real .npy headers, then read instance i and confirm
    its first/last token match what the global index predicts."""
    with tempfile.TemporaryDirectory() as d:
      sizes = (10, 7, 25)
      paths = _make_synthetic_mix(d, sizes)
      seq = 4
      idx = build_index([(p, "lab") for p in paths], sequence_length=seq, tokenizer="t")
      # Recall: file i contains sequential ints starting at sum(sizes[:i]).
      starts = [0]
      for s in sizes:
        starts.append(starts[-1] + s)
      for i in range(idx.total_instances):
        file_idx, tok_off = global_to_local(idx, i)
        path = idx.files[file_idx].path
        arr = np.load(path)
        chunk = arr[tok_off : tok_off + seq]
        # Tokens are sequential ints; the chunk's first value should be the
        # right global token offset.
        global_tok_off_expected = starts[file_idx] + tok_off
        self.assertEqual(int(chunk[0]), global_tok_off_expected)
        self.assertEqual(int(chunk[-1]), global_tok_off_expected + seq - 1)


class FingerprintTest(unittest.TestCase):
  """A.7: the fingerprint changes iff the relevant inputs change."""

  def _entries(self):
    return (
        OlmoNpyFileEntry("a.npy", "x", n_tokens=10, n_instances=2, instance_offset=0),
        OlmoNpyFileEntry("b.npy", "y", n_tokens=8, n_instances=2, instance_offset=2),
    )

  def test_same_inputs_same_fingerprint(self):
    e = self._entries()
    self.assertEqual(
        compute_fingerprint(seq_ := 8, "uint32", "tok", e),
        compute_fingerprint(seq_, "uint32", "tok", e),
    )

  def test_different_seq_changes_fingerprint(self):
    e = self._entries()
    a = compute_fingerprint(8, "uint32", "tok", e)
    b = compute_fingerprint(16, "uint32", "tok", e)
    self.assertNotEqual(a, b)

  def test_different_dtype_changes_fingerprint(self):
    e = self._entries()
    self.assertNotEqual(
        compute_fingerprint(8, "uint32", "tok", e),
        compute_fingerprint(8, "uint16", "tok", e),
    )

  def test_different_tokenizer_changes_fingerprint(self):
    e = self._entries()
    self.assertNotEqual(
        compute_fingerprint(8, "uint32", "alpha", e),
        compute_fingerprint(8, "uint32", "beta", e),
    )

  def test_file_reorder_changes_fingerprint(self):
    a, b = self._entries()
    e1 = (a, b)
    e2 = (b, a)
    self.assertNotEqual(
        compute_fingerprint(8, "uint32", "tok", e1),
        compute_fingerprint(8, "uint32", "tok", e2),
    )

  def test_file_size_change_changes_fingerprint(self):
    a, b = self._entries()
    a2 = OlmoNpyFileEntry(
        a.path,
        a.label,
        n_tokens=a.n_tokens + 1,
        n_instances=a.n_instances,
        instance_offset=a.instance_offset,
    )
    self.assertNotEqual(
        compute_fingerprint(8, "uint32", "tok", (a, b)),
        compute_fingerprint(8, "uint32", "tok", (a2, b)),
    )


class NpyHeaderTest(unittest.TestCase):

  def test_header_round_trip(self):
    with tempfile.TemporaryDirectory() as d:
      path = _write_uint32_npy(d, "h.npy", np.arange(123, dtype=np.uint32))
      dtype, shape = read_npy_header_from_path(path)
      self.assertEqual(dtype, "uint32")
      self.assertEqual(shape, (123,))


class NgramFilterTest(unittest.TestCase):
  """A.8: ``is_clean_instance`` flags excessive periodic repetition."""

  def test_clean_random_input(self):
    rng = np.random.default_rng(0)
    arr = rng.integers(low=0, high=10_000, size=2048, dtype=np.uint32)
    # Random tokens are extremely unlikely to have 32+ repetitions of any
    # period in [1, 13].
    self.assertTrue(is_clean_instance(arr))

  def test_dirty_period_one(self):
    # A run of 100 identical tokens is period=1, times>=32 → dirty.
    arr = np.concatenate(
        [
            np.arange(50, dtype=np.uint32),
            np.full(100, 7, dtype=np.uint32),
            np.arange(50, dtype=np.uint32) + 1000,
        ]
    )
    self.assertFalse(is_clean_instance(arr))

  def test_dirty_period_three(self):
    # Repeat the pattern (1, 2, 3) forty times → period=3, times=40 → dirty.
    pattern = np.array([1, 2, 3], dtype=np.uint32)
    repeats = np.tile(pattern, 40)
    surround = np.array([99, 88, 77, 66, 55], dtype=np.uint32)
    arr = np.concatenate([surround, repeats, surround])
    self.assertFalse(is_clean_instance(arr))

  def test_below_threshold_is_clean(self):
    # 10 repetitions of a 3-gram is well below 32.
    pattern = np.array([1, 2, 3], dtype=np.uint32)
    arr = np.concatenate(
        [
            np.arange(20, dtype=np.uint32) + 100,
            np.tile(pattern, 10),
            np.arange(20, dtype=np.uint32) + 200,
        ]
    )
    self.assertTrue(is_clean_instance(arr))

  def test_period_above_max_ignored(self):
    # Period 50 > default repetition_max_period (13). Even a long repeat is
    # ignored. Construct a 50-token unit repeated 40 times = 2000 tokens.
    rng = np.random.default_rng(1)
    unit = rng.integers(0, 10_000, size=50, dtype=np.uint32)
    arr = np.tile(unit, 40)
    # Use the default config (max_period=13) → should be clean.
    self.assertTrue(is_clean_instance(arr))
    # If we raise max_period to 50 we should now flag it.
    self.assertFalse(is_clean_instance(arr, repetition_max_period=50, repetition_max_count=32))

  def test_find_periodic_sequences_smoke(self):
    arr = np.concatenate(
        [
            np.array([5, 6, 7], dtype=np.uint32),
            np.tile(np.array([1, 2], dtype=np.uint32), 5),
            np.array([8, 9, 10], dtype=np.uint32),
        ]
    )
    matches = list(find_periodic_sequences(arr, max_period=5))
    # Expect at least one match with period=2 covering the [1,2]*5 region.
    found = [m for m in matches if m.period == 2 and m.times >= 3]
    self.assertTrue(found, f"no period=2 matches; got {matches}")
    m = found[0]
    self.assertEqual(arr[m.start : m.end].tolist(), [1, 2] * m.times)


if __name__ == "__main__":
  unittest.main()
