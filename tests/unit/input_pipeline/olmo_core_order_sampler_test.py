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

"""Unit tests for ``OlmoCoreOrderSampler``: training in OLMo-core's batch order.

The global indices file is built here the way OLMo-core's
``NumpyFSLDataLoader._build_global_indices`` builds it (``PCG64(seed + epoch)``
shuffling ``arange(N, dtype=uint32)`` in place), and OLMo-core's global batch
``k`` is row ``k`` of that permutation reshaped to ``(-1, B)``.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import unittest

import numpy as np

from maxtext.input_pipeline.olmo_data import build_index
from maxtext.input_pipeline.olmo_data_grain import OlmoCoreOrderSampler, make_olmo_grain_data_loader


def _olmo_core_permutation(n: int, *, seed: int, epoch: int = 1) -> np.ndarray:
  indices = np.arange(n, dtype=np.uint32)
  np.random.Generator(np.random.PCG64(seed=seed + epoch)).shuffle(indices)
  return indices


def _olmo_core_batches(perm: np.ndarray, global_batch: int) -> np.ndarray:
  num_batches = len(perm) // global_batch
  return perm[: num_batches * global_batch].reshape(-1, global_batch)


def _write_indices(tmpdir: str, perm: np.ndarray) -> str:
  path = os.path.join(tmpdir, "global_indices.u32")
  perm.astype(np.uint32).tofile(path)
  return path


def _global_batch_from_hosts(path: str, n: int, *, step: int, batch_size: int, shard_count: int, initial_step=0):
  """Instances every host loads at ``step``, as a sorted list."""
  out = []
  for h in range(shard_count):
    sampler = OlmoCoreOrderSampler(
        global_indices_path=path,
        total_instances=n,
        batch_size=batch_size,
        shard_index=h,
        shard_count=shard_count,
        initial_step=initial_step,
    )
    out.extend(sampler[step * batch_size + j].record_key for j in range(batch_size))
  return sorted(out)


class OlmoCoreOrderSamplerTest(unittest.TestCase):

  N = 37  # 4 global batches of 8, with a remainder of 5 that OLMo-core drops.
  SEED = 928_543_231

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.perm = _olmo_core_permutation(self.N, seed=self.SEED)
    self.path = _write_indices(self._tmp.name, self.perm)

  def tearDown(self):
    self._tmp.cleanup()

  def test_every_global_batch_matches_olmo_core(self):
    batches = _olmo_core_batches(self.perm, 8)
    for shard_count, batch_size in ((1, 8), (2, 4), (4, 2), (8, 1)):
      for step, expected in enumerate(batches):
        got = _global_batch_from_hosts(
            self.path,
            self.N,
            step=step,
            batch_size=batch_size,
            shard_count=shard_count,
        )
        self.assertEqual(got, sorted(expected.tolist()), (shard_count, step))

  def test_single_host_reads_the_permutation_in_order(self):
    sampler = OlmoCoreOrderSampler(global_indices_path=self.path, total_instances=self.N, batch_size=8)
    got = [sampler[i].record_key for i in range(32)]
    self.assertEqual(got, self.perm[:32].tolist())

  def test_hosts_take_a_strided_split(self):
    sampler = OlmoCoreOrderSampler(
        global_indices_path=self.path,
        total_instances=self.N,
        batch_size=2,
        shard_index=3,
        shard_count=4,
    )
    # Step 1, slots 0 and 1: positions 8 + 3 + 0*4 and 8 + 3 + 1*4.
    self.assertEqual(
        [sampler[2].record_key, sampler[3].record_key],
        [self.perm[11], self.perm[15]],
    )

  def test_resume_continues_the_same_stream(self):
    full = OlmoCoreOrderSampler(
        global_indices_path=self.path,
        total_instances=self.N,
        batch_size=2,
        shard_index=1,
        shard_count=4,
    )
    resumed = OlmoCoreOrderSampler(
        global_indices_path=self.path,
        total_instances=self.N,
        batch_size=2,
        shard_index=1,
        shard_count=4,
        initial_step=2 * 2,  # Resumed at train step 2, as _make_loader_for_host computes it.
    )
    self.assertEqual(
        [resumed[i].record_key for i in range(4)],
        [full[i].record_key for i in range(4, 8)],
    )

  def test_past_the_epoch_raises(self):
    sampler = OlmoCoreOrderSampler(
        global_indices_path=self.path,
        total_instances=self.N,
        batch_size=2,
        shard_count=4,
    )
    self.assertEqual(sampler.num_global_batches, 4)
    sampler[4 * 2 - 1]  # Last slot of the last full batch.
    with self.assertRaises(IndexError):
      sampler[4 * 2]

  def test_length_mismatch_raises(self):
    with self.assertRaises(ValueError):
      OlmoCoreOrderSampler(global_indices_path=self.path, total_instances=self.N + 1, batch_size=1)

  def test_pickle_round_trip(self):
    sampler = OlmoCoreOrderSampler(global_indices_path=self.path, total_instances=self.N, batch_size=8)
    sampler[0]  # Opens the memmap, which must not be pickled.
    clone = pickle.loads(pickle.dumps(sampler))
    self.assertEqual(clone[5].record_key, sampler[5].record_key)
    self.assertEqual(repr(clone), repr(sampler))


class OlmoCoreOrderLoaderTest(unittest.TestCase):

  def test_loader_yields_olmo_core_batches(self):
    seq = 4
    with tempfile.TemporaryDirectory() as d:
      paths = []
      for i, n in enumerate((40, 36, 52)):  # 10 + 9 + 13 = 32 instances, remainders dropped per file.
        path = os.path.join(d, f"f_{i}.bin")
        np.arange(n, dtype=np.uint32).tofile(path)
        paths.append(path)
      index = build_index(
          [(p, "lab") for p in paths],
          sequence_length=seq,
          tokenizer="test",
          header_reader=lambda p: ("uint32", (os.path.getsize(p) // 4,)),
      )
      self.assertEqual(index.total_instances, 32)
      perm = _olmo_core_permutation(index.total_instances, seed=7)
      indices_path = _write_indices(d, perm)
      expected = _olmo_core_batches(perm, 8)

      for worker_count in (0, 1):
        got = [[] for _ in expected]
        for h in range(2):
          loader = make_olmo_grain_data_loader(
              index,
              seed=0,  # Ignored: the order comes from the file.
              batch_size=4,
              shard_index=h,
              shard_count=2,
              apply_ngram_filter=False,
              shift_to_inputs_targets=False,
              grain_worker_count=worker_count,
              global_indices_path=indices_path,
          )
          step = -1
          for step, batch in enumerate(loader):
            got[step].extend(batch["instance_id"].tolist())
          self.assertEqual(step, len(expected) - 1)  # Stops at the end of the epoch.
        self.assertEqual(
            [sorted(g) for g in got],
            [sorted(e.tolist()) for e in expected],
            worker_count,
        )

  def test_loader_rejects_several_workers(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "f.bin")
      np.arange(64, dtype=np.uint32).tofile(path)
      index = build_index(
          [(path, "lab")],
          sequence_length=4,
          tokenizer="test",
          header_reader=lambda p: ("uint32", (os.path.getsize(p) // 4,)),
      )
      indices_path = _write_indices(d, _olmo_core_permutation(index.total_instances, seed=7))
      with self.assertRaises(ValueError):
        make_olmo_grain_data_loader(
            index,
            seed=0,
            batch_size=2,
            shard_index=0,
            shard_count=1,
            grain_worker_count=2,
            global_indices_path=indices_path,
        )

  def test_loader_resumes_mid_epoch(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "f.bin")
      np.arange(64, dtype=np.uint32).tofile(path)
      index = build_index(
          [(path, "lab")],
          sequence_length=4,
          tokenizer="test",
          header_reader=lambda p: ("uint32", (os.path.getsize(p) // 4,)),
      )
      perm = _olmo_core_permutation(index.total_instances, seed=7)
      indices_path = _write_indices(d, perm)
      expected = _olmo_core_batches(perm, 4)[2:]  # Resumed after 2 steps.
      got = [[] for _ in expected]
      for h in range(2):
        loader = make_olmo_grain_data_loader(
            index,
            seed=0,
            batch_size=2,
            shard_index=h,
            shard_count=2,
            apply_ngram_filter=False,
            shift_to_inputs_targets=False,
            initial_step=2 * 2,  # train step 2 x per-host batch 2, as _make_loader_for_host computes it.
            global_indices_path=indices_path,
        )
        for step, batch in enumerate(loader):
          got[step].extend(batch["instance_id"].tolist())
      self.assertEqual([sorted(g) for g in got], [sorted(e.tolist()) for e in expected])


if __name__ == "__main__":
  unittest.main()
