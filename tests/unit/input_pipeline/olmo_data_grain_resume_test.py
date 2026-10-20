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

"""Resume-via-step-offset tests for the OLMo Grain loader (Option B).

Our :class:`OlmoIndexSampler` is a pure ``__getitem__(idx) → record_key``
function of ``(seed, shard, idx + initial_step)``. That means resume
doesn't need Grain's iterator-state checkpoint; supplying
``initial_step = saved_step * per_host_batch`` at construction time is
enough. These tests pin that contract:

  A) Two loaders with the same seed but different ``initial_step`` produce
     a stream where the late-starter sees exactly the same records the
     early-starter did from that absolute step onward.
  B) The sampler's ``__repr__`` is independent of ``initial_step`` so it
     stays compatible with Grain's repr-based sampler-equality validation
     (i.e. "step offset" and "Grain checkpoint" can coexist).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import List, Tuple

import numpy as np

from maxtext.input_pipeline.olmo_data import build_index, OlmoNpyIndex
from maxtext.input_pipeline.olmo_data_grain import (
    OlmoIndexSampler,
    make_olmo_grain_data_loader,
)


def _write_raw_uint32(tmpdir: str, name: str, values: np.ndarray) -> str:
  assert values.dtype == np.uint32 and values.ndim == 1
  path = os.path.join(tmpdir, name)
  values.tofile(path)
  return path


def _build_synthetic_index(tmpdir: str, *, sizes: Tuple[int, ...], sequence_length: int) -> OlmoNpyIndex:
  """Build a small in-memory OLMo index over raw-binary uint32 token files."""
  paths: List[str] = []
  start = 0
  for i, n in enumerate(sizes):
    arr = np.arange(start, start + n, dtype=np.uint32)
    paths.append(_write_raw_uint32(tmpdir, f"f_{i:03d}.bin", arr))
    start += n

  def _reader(path: str):
    return "uint32", (os.path.getsize(path) // 4,)

  return build_index(
      [(p, "lab") for p in paths],
      sequence_length=sequence_length,
      tokenizer="test",
      header_reader=_reader,
  )


def _take(iterator, n: int):
  return [next(iterator) for _ in range(n)]


def _assert_batch_equal(a: dict, b: dict, msg: str = "") -> None:
  assert set(a.keys()) == set(b.keys()), f"{msg}: key set differs: {sorted(a)} vs {sorted(b)}"
  for k in a:
    np.testing.assert_array_equal(a[k], b[k], err_msg=f"{msg}: batch field {k!r} differs")


class SamplerInitialStepTest(unittest.TestCase):

  def test_offset_matches_unbroken_stream(self):
    """``OlmoIndexSampler(initial_step=N)[i]`` must equal an unbroken
    sampler's record at absolute index ``N + i`` for any (i, N)."""
    s_full = OlmoIndexSampler(total_instances=128, seed=11, shard_count=1)
    for n in [0, 1, 7, 23, 127]:
      s_offset = OlmoIndexSampler(total_instances=128, seed=11, shard_count=1, initial_step=n)
      for i in [0, 1, 5, 17]:
        self.assertEqual(
            s_offset[i].record_key,
            s_full[n + i].record_key,
            msg=f"mismatch at initial_step={n}, i={i}",
        )

  def test_initial_step_does_not_change_repr(self):
    """Grain validates samplers by ``repr(sampler)`` on resume; the offset
    is a runtime cursor, not part of the sampler's identity."""
    a = OlmoIndexSampler(total_instances=64, seed=1, shard_count=1)
    b = OlmoIndexSampler(total_instances=64, seed=1, shard_count=1, initial_step=42)
    self.assertEqual(repr(a), repr(b))

  def test_negative_initial_step_raises(self):
    with self.assertRaises(ValueError):
      OlmoIndexSampler(total_instances=8, seed=0, initial_step=-1)

  def test_offset_crosses_epoch_boundary(self):
    """An offset large enough to roll the epoch must use the next epoch's
    permutation, not wrap inside epoch 0."""
    n = 32
    s = OlmoIndexSampler(total_instances=n, seed=2, shard_count=1)
    # epoch 0 has permutation P0; epoch 1 has P1.
    p0 = s.shard_indices(seed=2, epoch=0)
    p1 = s.shard_indices(seed=2, epoch=1)

    # offset just past end of epoch 0 → first lookups should come from P1.
    s_off = OlmoIndexSampler(total_instances=n, seed=2, shard_count=1, initial_step=n + 3)
    self.assertEqual(s_off[0].record_key, int(p1[3]))
    self.assertEqual(s_off[1].record_key, int(p1[4]))
    # And the canonical first 3 of epoch 1 are NOT visited (because we
    # started 3 records into epoch 1).
    self.assertNotIn(int(p0[0]), [s_off[i].record_key for i in range(3)])


class LoaderResumeViaInitialStepTest(unittest.TestCase):

  def test_loader_offset_matches_unbroken_loader(self):
    """A fresh loader with ``initial_step=K`` produces the same batches as
    an unbroken loader's batches starting at the K-th batch."""
    with tempfile.TemporaryDirectory() as d:
      # 256 tokens at seq=4 → 64 instances. Batch=4 → 16 batches/epoch.
      idx = _build_synthetic_index(d, sizes=(256,), sequence_length=4)
      common_kwargs = {
          "seed": 7,
          "batch_size": 4,
          "shard_index": 0,
          "shard_count": 1,
          "apply_ngram_filter": False,
          "shift_to_inputs_targets": True,
          "grain_worker_count": 0,
      }

      # Reference: 30 batches uninterrupted.
      ref = make_olmo_grain_data_loader(idx, initial_step=0, **common_kwargs)
      it_ref = iter(ref)
      ref_batches = _take(it_ref, 30)

      # Resume: skip the first 15 batches by starting the *sampler* at
      # absolute index 15 * batch_size = 60.
      resumed = make_olmo_grain_data_loader(
          idx,
          initial_step=15 * common_kwargs["batch_size"],
          **common_kwargs,
      )
      it_res = iter(resumed)
      res_batches = _take(it_res, 15)

      for i, (a, b) in enumerate(zip(ref_batches[15:], res_batches)):
        _assert_batch_equal(a, b, msg=f"resumed batch {i}")

  def test_resume_works_across_an_epoch_boundary(self):
    """The harder case: ``initial_step`` lands past the end of epoch 0."""
    with tempfile.TemporaryDirectory() as d:
      # 128 tokens at seq=4 → 32 instances. Batch=4 → 8 batches/epoch.
      idx = _build_synthetic_index(d, sizes=(128,), sequence_length=4)
      common_kwargs = {
          "seed": 3,
          "batch_size": 4,
          "shard_index": 0,
          "shard_count": 1,
          "apply_ngram_filter": False,
          "shift_to_inputs_targets": True,
      }

      # Take 12 batches uninterrupted (covers ~1.5 epochs).
      ref = make_olmo_grain_data_loader(idx, initial_step=0, **common_kwargs)
      ref_batches = _take(iter(ref), 12)

      # Skip 10 batches (= 40 instances, well past epoch boundary at 32).
      resumed = make_olmo_grain_data_loader(
          idx,
          initial_step=10 * common_kwargs["batch_size"],
          **common_kwargs,
      )
      res_batches = _take(iter(resumed), 2)

      for i, (a, b) in enumerate(zip(ref_batches[10:], res_batches)):
        _assert_batch_equal(a, b, msg=f"epoch-cross batch {i}")


class LoaderShardingResumeTest(unittest.TestCase):
  """Regression tests for multi-host sharding (``shard_count > 1``).

  These guard the double-sharding bug: ``grain.DataLoader`` re-shards the
  sampler's index stream when handed a real ``ShardOptions(shard_index,
  shard_count)`` *while* ``OlmoIndexSampler`` already shards internally. That
  strides the sampler index by ``shard_count``, which (a) makes each host read
  only ~1/shard_count of its assigned block and (b) breaks resume because the
  ``initial_step`` offset is added to a strided index. The fix passes
  ``NoSharding`` to the DataLoader so the sampler owns all sharding.

  The pre-existing tests above only used ``shard_count=1`` — the one config
  where the bug cannot appear (stride == 1) — which is why it shipped. These
  exercise ``shard_count > 1`` end-to-end through the real ``grain.DataLoader``.
  """

  @staticmethod
  def _instance_ids(batches):
    ids: List[int] = []
    for b in batches:
      ids.extend(int(x) for x in np.asarray(b["instance_id"]).reshape(-1))
    return ids

  def test_shards_are_disjoint_and_cover_everything(self):
    """One epoch across all shards must read each instance exactly once.

    With the double-shard bug each shard reads only ``n_per/shard_count``
    distinct instances (the rest repeat), so coverage collapses to
    ``1/shard_count`` of the corpus and this fails.
    """
    with tempfile.TemporaryDirectory() as d:
      # 512 tokens at seq=4 → 128 instances; shard_count=4 → 32 per shard.
      idx = _build_synthetic_index(d, sizes=(512,), sequence_length=4)
      shard_count = 4
      batch_size = 4
      n_per = idx.total_instances // shard_count  # 32
      batches_per_epoch = n_per // batch_size  # 8

      seen: List[set] = []
      for si in range(shard_count):
        loader = make_olmo_grain_data_loader(
            idx,
            seed=5,
            batch_size=batch_size,
            shard_index=si,
            shard_count=shard_count,
            apply_ngram_filter=False,
            shift_to_inputs_targets=False,  # keep instance_id in the batch
            grain_worker_count=0,
        )
        ids = self._instance_ids(_take(iter(loader), batches_per_epoch))
        self.assertEqual(len(ids), n_per)
        self.assertEqual(len(set(ids)), n_per, msg=f"shard {si} repeated instances within one epoch (double-shard bug)")
        seen.append(set(ids))

      union = set().union(*seen)
      self.assertEqual(len(union), idx.total_instances, msg="shards do not cover the full corpus exactly once")
      for i in range(shard_count):
        for j in range(i + 1, shard_count):
          self.assertEqual(seen[i] & seen[j], set(), msg=f"shards {i} and {j} overlap")

  def test_resume_matches_unbroken_with_sharding(self):
    """Resume via ``initial_step`` must track the continuous run for shard>1.

    With the bug the resumed stream is offset by a factor of ``shard_count``
    relative to the continuous run, so these batches differ.
    """
    with tempfile.TemporaryDirectory() as d:
      idx = _build_synthetic_index(d, sizes=(512,), sequence_length=4)
      common = {
          "seed": 5,
          "batch_size": 4,
          "shard_index": 2,  # non-zero shard: exercises the stride offset
          "shard_count": 4,
          "apply_ngram_filter": False,
          "shift_to_inputs_targets": True,
          "grain_worker_count": 0,
      }
      ref = make_olmo_grain_data_loader(idx, initial_step=0, **common)
      ref_batches = _take(iter(ref), 12)

      k = 5
      resumed = make_olmo_grain_data_loader(idx, initial_step=k * common["batch_size"], **common)
      res_batches = _take(iter(resumed), 5)

      for i, (a, b) in enumerate(zip(ref_batches[k:], res_batches)):
        _assert_batch_equal(a, b, msg=f"sharded resumed batch {i}")


if __name__ == "__main__":
  unittest.main()
