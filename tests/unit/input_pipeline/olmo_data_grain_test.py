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

"""Unit tests for ``maxtext.input_pipeline.olmo_data_grain``.

Covers test plan items A.3-A.7, A.9, A.10:
  A.3  data source __getitem__ returns the right tokens
  A.4  data source shape/dtype contract
  A.5  (deferred to equivalence test once Option 1 lands)
  A.6  sampler partitions the global index space exactly
  A.7  reshuffle determinism + fingerprint
  A.9  checkpoint / restore — get_state() round-trip
  A.10 iteration termination at end-of-epoch and epoch roll-over

All tests use *synthetic* raw-binary token files (matching AI2's headerless
``.npy`` layout) so they're fast and deterministic.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import List, Tuple

import numpy as np

from maxtext.input_pipeline.olmo_data import build_index, OlmoNpyIndex
from maxtext.input_pipeline.olmo_data_grain import (
    NgramFilterTransform,
    OlmoIndexSampler,
    OlmoNpyDataSource,
    ShiftToInputsTargets,
    _combine_seed_epoch,
    make_olmo_grain_data_loader,
)


def _write_raw_uint32(tmpdir: str, name: str, values: np.ndarray) -> str:
  """Write a 1-D uint32 array as raw binary (no .npy header) — matches AI2."""
  assert values.dtype == np.uint32 and values.ndim == 1
  path = os.path.join(tmpdir, name)
  values.tofile(path)
  return path


def _build_synthetic_index(
    tmpdir: str,
    *,
    sizes: Tuple[int, ...],
    sequence_length: int,
) -> Tuple[OlmoNpyIndex, List[str]]:
  """Make ``len(sizes)`` raw-binary uint32 files and an index over them.

  Tokens in file ``i`` are ``[start_i, start_i + size_i)`` so every global
  token has a unique value, making chunk-extraction tests easy to assert.
  """
  paths: List[str] = []
  start = 0
  for i, n in enumerate(sizes):
    arr = np.arange(start, start + n, dtype=np.uint32)
    paths.append(_write_raw_uint32(tmpdir, f"f_{i:03d}.bin", arr))
    start += n

  def _reader(path: str):
    return "uint32", (os.path.getsize(path) // 4,)

  idx = build_index(
      [(p, "lab") for p in paths],
      sequence_length=sequence_length,
      tokenizer="test",
      header_reader=_reader,
  )
  return idx, paths


class DataSourceTest(unittest.TestCase):

  def test_getitem_returns_correct_tokens(self):
    """A.3: instance i covers the tokens we expect from the synthetic mix."""
    with tempfile.TemporaryDirectory() as d:
      sizes = (12, 8, 16)
      seq = 4
      idx, _ = _build_synthetic_index(d, sizes=sizes, sequence_length=seq)
      ds = OlmoNpyDataSource(idx)

      self.assertEqual(len(ds), idx.total_instances)

      # Build the expected concatenation: file0 starts at 0, file1 at 12,
      # file2 at 20.
      starts = [0]
      for s in sizes:
        starts.append(starts[-1] + s)
      # Rebuild a virtual stream that drops trailing remainder per file.
      dropped: List[np.ndarray] = []
      for fi, s in enumerate(sizes):
        full = np.arange(starts[fi], starts[fi] + s, dtype=np.uint32)
        keep = (s // seq) * seq
        dropped.append(full[:keep])
      concat = np.concatenate(dropped)

      # Every global instance i should equal the i-th seq-length window of the
      # concatenated, remainder-dropped stream.
      for i, item in enumerate(ds):
        self.assertEqual(item["tokens"].dtype, np.uint32)
        self.assertEqual(item["tokens"].shape, (seq,))
        np.testing.assert_array_equal(item["tokens"], concat[i * seq : (i + 1) * seq])
        self.assertEqual(item["instance_id"], i)
        # file_id should match the file the instance lives in.
        self.assertGreaterEqual(item["file_id"], 0)
        self.assertLess(item["file_id"], len(sizes))

  def test_returned_token_array_is_safe_to_mutate(self):
    """A.4: the returned ``tokens`` array is a copy, not a view into mmap."""
    with tempfile.TemporaryDirectory() as d:
      idx, _ = _build_synthetic_index(d, sizes=(16,), sequence_length=4)
      ds = OlmoNpyDataSource(idx)
      a = ds[0]["tokens"]
      original = a.copy()
      a[0] = 99999
      b = ds[0]["tokens"]
      np.testing.assert_array_equal(b, original)

  def test_path_remap(self):
    with tempfile.TemporaryDirectory() as d:
      idx, paths = _build_synthetic_index(d, sizes=(8,), sequence_length=4)
      # Move the file so the original path doesn't work, then remap.
      moved = paths[0] + ".moved"
      os.rename(paths[0], moved)

      ds_no_remap = OlmoNpyDataSource(idx)
      with self.assertRaises(FileNotFoundError):
        _ = ds_no_remap[0]

      ds = OlmoNpyDataSource(idx, path_remap={paths[0]: moved})
      self.assertEqual(ds[0]["tokens"].shape, (4,))


class SamplerTest(unittest.TestCase):

  def test_sharding_partition_is_disjoint_and_complete(self):
    """A.6: with shard_count=N, the union of the N hosts' shard_indices
    covers every global index in the per-epoch shuffle exactly once,
    after dropping the trailing remainder."""
    total = 23
    n_shards = 4
    s_full = OlmoIndexSampler(total_instances=total, seed=42, shard_index=0, shard_count=1)
    full = s_full.shuffled_global_indices(seed=42, epoch=0)
    self.assertEqual(len(full), total)

    seen = []
    for shard in range(n_shards):
      s = OlmoIndexSampler(
          total_instances=total,
          seed=42,
          shard_index=shard,
          shard_count=n_shards,
      )
      seen.append(s.shard_indices(seed=42, epoch=0))

    cat = np.concatenate(seen)
    # 23 // 4 = 5 per shard, 4*5 = 20 covered, 3 trailing dropped.
    self.assertEqual(len(cat), n_shards * (total // n_shards))
    # No duplicates.
    self.assertEqual(len(np.unique(cat)), len(cat))
    # All from the global shuffle's first 20 entries.
    np.testing.assert_array_equal(np.sort(cat), np.sort(full[:20]))

  def test_reshuffle_determinism(self):
    """A.7: same (seed, epoch) ⇒ same shuffle; different epoch ⇒ different."""
    s = OlmoIndexSampler(total_instances=100, seed=7)
    a = s.shuffled_global_indices(seed=7, epoch=0)
    b = s.shuffled_global_indices(seed=7, epoch=0)
    np.testing.assert_array_equal(a, b)

    c = s.shuffled_global_indices(seed=7, epoch=1)
    self.assertFalse(np.array_equal(a, c))

    d = s.shuffled_global_indices(seed=8, epoch=0)
    self.assertFalse(np.array_equal(a, d))

  def test_combine_seed_epoch_distinguishes_inputs(self):
    self.assertEqual(_combine_seed_epoch(0, 0), _combine_seed_epoch(0, 0))
    self.assertNotEqual(_combine_seed_epoch(0, 0), _combine_seed_epoch(0, 1))
    self.assertNotEqual(_combine_seed_epoch(0, 0), _combine_seed_epoch(1, 0))

  def test_getitem_emits_each_local_index_per_epoch_then_rolls(self):
    """A.10: ``sampler[i]`` matches the i-th element of this host's shard,
    rolling over to the next epoch's shuffle at i == per_epoch."""
    total = 16
    s = OlmoIndexSampler(
        total_instances=total,
        seed=3,
        shard_index=1,
        shard_count=4,
    )
    expected_per_epoch = total // 4  # = 4
    expected_e0 = list(s.shard_indices(seed=3, epoch=0))
    expected_e1 = list(s.shard_indices(seed=3, epoch=1))

    seen = [int(s[i].record_key) for i in range(expected_per_epoch * 2)]
    self.assertEqual(seen, expected_e0 + expected_e1)

    # Index field equals the input.
    for i in range(expected_per_epoch * 2):
      self.assertEqual(s[i].index, i)

  def test_getitem_negative_raises(self):
    s = OlmoIndexSampler(total_instances=8, seed=0)
    with self.assertRaises(IndexError):
      _ = s[-1]


class SamplerCheckpointTest(unittest.TestCase):
  """A.9: with a __getitem__-style sampler, the only checkpoint state is the
  global step counter (Grain handles persisting that). The sampler itself is
  stateless across runs given the same (seed, shard_options)."""

  def test_resume_index_yields_same_record_key(self):
    """Two independent samplers with the same config + same index produce
    the same RecordMetadata — i.e. there is no hidden mutable state that
    would differ across an in-process restart."""
    s_a = OlmoIndexSampler(total_instances=40, seed=11, shard_count=1, shard_index=0)
    s_b = OlmoIndexSampler(total_instances=40, seed=11, shard_count=1, shard_index=0)
    for i in [0, 1, 2, 7, 39, 40, 79, 80, 1000]:
      self.assertEqual(s_a[i].record_key, s_b[i].record_key)
      self.assertEqual(s_a[i].index, i)

  def test_different_seeds_diverge(self):
    s_a = OlmoIndexSampler(total_instances=40, seed=11, shard_count=1, shard_index=0)
    s_b = OlmoIndexSampler(total_instances=40, seed=12, shard_count=1, shard_index=0)
    keys_a = [s_a[i].record_key for i in range(40)]
    keys_b = [s_b[i].record_key for i in range(40)]
    self.assertNotEqual(keys_a, keys_b)
    self.assertEqual(sorted(keys_a), sorted(keys_b))  # same set, different order


class TransformsTest(unittest.TestCase):

  def test_ngram_filter_marks_clean(self):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 10_000, size=512, dtype=np.uint32)
    out = NgramFilterTransform().map({"tokens": arr, "instance_id": 0, "file_id": 0})
    self.assertTrue(out["instance_mask"])

  def test_ngram_filter_marks_dirty(self):
    arr = np.full(200, 7, dtype=np.uint32)  # period=1, repeats=200, dirty
    out = NgramFilterTransform().map({"tokens": arr, "instance_id": 0, "file_id": 0})
    self.assertFalse(out["instance_mask"])

  def test_shift_to_inputs_targets_clean(self):
    arr = np.arange(8, dtype=np.uint32)
    out = ShiftToInputsTargets().map({"tokens": arr, "instance_id": 0, "file_id": 0, "instance_mask": True})
    # Only rank-2 (batch, seq) tensors are returned — scalar metadata is
    # dropped to satisfy the trainer's 2D sharding contract.
    self.assertEqual(
        set(out.keys()),
        {
            "inputs",
            "targets",
            "inputs_position",
            "inputs_segmentation",
            "targets_segmentation",
        },
    )
    self.assertEqual(out["inputs"].dtype, np.int32)
    self.assertEqual(out["targets"].dtype, np.int32)
    # All outputs have length L (= len(tokens)) so the trainer sees
    # ``max_target_length`` exactly (required by splash attention kernel
    # block-size constraints).
    np.testing.assert_array_equal(out["inputs"], np.arange(8, dtype=np.int32))
    # Targets shifted by 1; last position padded with 0 and masked out below.
    np.testing.assert_array_equal(out["targets"], np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32))
    np.testing.assert_array_equal(out["inputs_position"], np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(out["inputs_segmentation"], np.ones(8, dtype=np.int32))
    # Last target position masked even when instance is clean.
    np.testing.assert_array_equal(out["targets_segmentation"], np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int32))

  def test_shift_to_inputs_targets_dirty_zeros_target_segmentation(self):
    arr = np.arange(8, dtype=np.uint32)
    out = ShiftToInputsTargets().map({"tokens": arr, "instance_id": 0, "file_id": 0, "instance_mask": False})
    # inputs_segmentation still 1 (data is fed to the model); the dirty
    # instance flag zeroes the entire targets_segmentation row.
    np.testing.assert_array_equal(out["inputs_segmentation"], np.ones(8, dtype=np.int32))
    np.testing.assert_array_equal(out["targets_segmentation"], np.zeros(8, dtype=np.int32))


class FactoryTest(unittest.TestCase):

  def test_make_olmo_grain_data_loader_yields_batches(self):
    with tempfile.TemporaryDirectory() as d:
      idx, _ = _build_synthetic_index(d, sizes=(16, 16, 16), sequence_length=4)
      # 12 global instances, 1 host, batch 2 → 6 batches per epoch.
      loader = make_olmo_grain_data_loader(
          idx,
          seed=0,
          batch_size=2,
          shard_index=0,
          shard_count=1,
          apply_ngram_filter=True,
          shift_to_inputs_targets=True,
      )
      batches = []
      for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= 5:
          break
      self.assertEqual(len(batches), 6)
      for b in batches:
        # Inputs and targets are both length seq = 4; targets are shifted
        # within that window with the last position padded + masked.
        self.assertEqual(b["inputs"].shape, (2, 4))
        self.assertEqual(b["targets"].shape, (2, 4))
        self.assertEqual(b["targets_segmentation"].shape, (2, 4))

  def test_two_workers_preserve_record_count(self):
    with tempfile.TemporaryDirectory() as d:
      idx, _ = _build_synthetic_index(d, sizes=(64,), sequence_length=4)
      # 16 instances; batch=4 ⇒ 4 batches/epoch.
      loader = make_olmo_grain_data_loader(
          idx,
          seed=0,
          batch_size=4,
          shard_index=0,
          shard_count=1,
          apply_ngram_filter=False,
          # Disable shift so we can read the raw ``instance_id`` field for
          # this audit-style test. (Production runs always use shift=True.)
          shift_to_inputs_targets=False,
          grain_worker_count=2,
      )
      ids = []
      for i, batch in enumerate(loader):
        ids.extend(batch["instance_id"].tolist())
        if i >= 3:
          break
      self.assertEqual(len(ids), 16)
      self.assertEqual(sorted(ids), list(range(16)))


if __name__ == "__main__":
  unittest.main()
