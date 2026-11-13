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

from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np

from maxtext.experimental.flint40b import data
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types
from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline.protos import feature_pb2
from array_record.python import array_record_module


def _serialize_token_record(
    tokens: Sequence[int] | np.ndarray,
    *,
    feature_name: str = data.TOKEN_FEATURE,
) -> bytes:
  example = example_pb2.Example(
      features=feature_pb2.Features(
          feature={feature_name: feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(x) for x in tokens]))}
      )
  )
  return example.SerializeToString()


def _write_array_record_shards(
    out_dir: str,
    shard_counts: Sequence[int],
    rec_len: int = 64,
) -> tuple[list[str], list[np.ndarray]]:
  shard_paths: list[str] = []
  raw_records: list[np.ndarray] = []
  num_shards = len(shard_counts)
  rec_idx = 0
  for shard_idx, count in enumerate(shard_counts):
    path = os.path.join(out_dir, f"c4-{shard_idx:05d}-of-{num_shards:05d}.array_record")
    writer = array_record_module.ArrayRecordWriter(path, "group_size:1")
    for _ in range(count):
      arr = (rec_idx * 100 + np.arange(1, rec_len + 1, dtype=np.int64)).astype(np.int32)
      raw_records.append(arr)
      writer.write(_serialize_token_record(arr))
      rec_idx += 1
    writer.close()
    shard_paths.append(path)
  return shard_paths, raw_records


def _batch_to_np(
    batch: flint_types.FlintBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  return (
      np.asarray(batch.tokens),
      np.asarray(batch.targets),
      np.asarray(batch.positions),
      np.asarray(batch.segment_ids),
      np.asarray(batch.loss_mask),
  )


class FlintDataTest(absltest.TestCase):

  def test_c4_reader_continuous_across_k_1_to_32(self):
    tmp_dir = self.create_tempdir().full_path
    _, raw_records = _write_array_record_shards(tmp_dir, shard_counts=(10, 12, 10), rec_len=64)
    self.assertLen(raw_records, 32)

    mesh = flint_sharding.create_mesh(flint_config.FlintShardingConfig(fsdp_size=2, cp_size=2, ep_size=2))
    sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
    global_batch_size = 8

    for k in (1, 2, 4, 8, 16, 32):
      seq_len = 64 * k
      expected_num_rows = 32 // k
      with self.subTest(seq_len=seq_len, k=k):
        with data.ArrayRecordC4(tmp_dir, seq_len=seq_len) as reader:
          self.assertEqual(reader.record_len, 64)
          self.assertEqual(reader.chunks_per_row, k)
          self.assertEqual(reader.num_rows, expected_num_rows)

          # Test step=0 and multi-epoch wraparound step=3.
          for step in (0, 3):
            np_batch = reader.make_batch_np(step, global_batch_size)
            jax_batch = reader.make_batch(step, global_batch_size, sharding)
            self.assertIsInstance(jax_batch, flint_types.FlintBatch)
            for leaf in jax_batch:
              self.assertEqual(leaf.dtype, jnp.int32)
              self.assertEqual(leaf.shape, (global_batch_size, seq_len))
              self.assertEqual(leaf.sharding, sharding)

            tok, tgt, pos, seg, mask = _batch_to_np(jax_batch)
            for got, ref in zip((tok, tgt, pos, seg, mask), np_batch, strict=True):
              np.testing.assert_array_equal(got, ref)

            for b in range(global_batch_size):
              row_in_epoch = (step * global_batch_size + b) % reader.num_rows
              expected_tok = np.concatenate([raw_records[row_in_epoch * k + c] for c in range(k)])
              np.testing.assert_array_equal(tok[b], expected_tok)

            for c in range(k):
              c_start = c * 64
              c_end = (c + 1) * 64
              np.testing.assert_array_equal(
                  tgt[:, c_start : c_end - 1],
                  tok[:, c_start + 1 : c_end],
              )
              np.testing.assert_array_equal(
                  tgt[:, c_end - 1],
                  np.zeros(global_batch_size, dtype=np.int32),
              )
              np.testing.assert_array_equal(
                  mask[:, c_start : c_end - 1],
                  np.ones((global_batch_size, 63), dtype=np.int32),
              )
              np.testing.assert_array_equal(
                  mask[:, c_end - 1],
                  np.zeros(global_batch_size, dtype=np.int32),
              )

            np.testing.assert_array_equal(
                mask.sum(axis=1),
                np.full(global_batch_size, k * 63, dtype=np.int64),
            )
            # Every row is one full-length causal sequence.
            np.testing.assert_array_equal(
                pos,
                np.broadcast_to(
                    np.arange(seq_len, dtype=np.int32)[None, :],
                    (global_batch_size, seq_len),
                ),
            )
            np.testing.assert_array_equal(seg, np.ones((global_batch_size, seq_len), dtype=np.int32))

    # Also verify explicit glob pattern resolution matches directory resolution.
    glob_pattern = os.path.join(tmp_dir, "c4-*-of-00003.array_record")
    with (
        data.ArrayRecordC4(tmp_dir, seq_len=128) as dir_reader,
        data.ArrayRecordC4(glob_pattern, seq_len=128) as glob_reader,
    ):
      for got, ref in zip(
          glob_reader.make_batch_np(1, 8),
          dir_reader.make_batch_np(1, 8),
          strict=True,
      ):
        np.testing.assert_array_equal(got, ref)

  def test_c4_and_synthetic_mesh_invariance_across_fsdp_cp_ep(self):
    tmp_dir = self.create_tempdir().full_path
    _write_array_record_shards(tmp_dir, shard_counts=(6, 5, 5), rec_len=64)

    mesh_specs = (
        (1, 1, 1),
        (8, 1, 1),
        (2, 4, 1),
        (4, 1, 2),
        (2, 2, 2),
    )
    shardings: dict[tuple[int, int, int], NamedSharding] = {}
    for fsdp_size, cp_size, ep_size in mesh_specs:
      sh_cfg = flint_config.FlintShardingConfig(fsdp_size=fsdp_size, cp_size=cp_size, ep_size=ep_size)
      devs = jax.devices()[: sh_cfg.num_devices]
      mesh = flint_sharding.create_mesh(sh_cfg, devices=devs)
      shardings[(fsdp_size, cp_size, ep_size)] = NamedSharding(mesh, flint_sharding.batch_pspec())

    ref_sharding = shardings[(1, 1, 1)]
    global_batch_size = 8
    seq_len = 128  # k = 2 chunks of 64; 16 records -> num_rows = 8 per epoch.

    with (
        data.ArrayRecordC4(tmp_dir, seq_len=seq_len, shuffle_seed=None) as seq_reader,
        data.ArrayRecordC4(tmp_dir, seq_len=seq_len, shuffle_seed=42) as shuf_reader,
    ):
      self.assertEqual(seq_reader.num_rows, 8)
      self.assertEqual(shuf_reader.num_rows, 8)

      seq_ep0 = _batch_to_np(seq_reader.make_batch(0, global_batch_size, ref_sharding))
      shuf_ep0 = _batch_to_np(shuf_reader.make_batch(0, global_batch_size, ref_sharding))
      shuf_ep1 = _batch_to_np(shuf_reader.make_batch(1, global_batch_size, ref_sharding))

      # Shuffled epoch 0 differs from sequential order and from epoch 1, while
      # permuting the exact same set of 8 global rows within each epoch.
      self.assertFalse(np.array_equal(shuf_ep0[0], seq_ep0[0]))
      self.assertFalse(np.array_equal(shuf_ep0[0], shuf_ep1[0]))
      self.assertCountEqual(
          [row.tobytes() for row in shuf_ep0[0]],
          [row.tobytes() for row in seq_ep0[0]],
      )
      self.assertCountEqual(
          [row.tobytes() for row in shuf_ep1[0]],
          [row.tobytes() for row in seq_ep0[0]],
      )

      for mesh_dims, sharding in shardings.items():
        with self.subTest(source="c4", mesh=mesh_dims):
          got_seq0 = _batch_to_np(seq_reader.make_batch(0, global_batch_size, sharding))
          got_shuf0 = _batch_to_np(shuf_reader.make_batch(0, global_batch_size, sharding))
          got_shuf1 = _batch_to_np(shuf_reader.make_batch(1, global_batch_size, sharding))
          pref0 = shuf_reader.prefetch_host_rows(0, global_batch_size, sharding)
          got_pref0 = _batch_to_np(shuf_reader.make_batch_from_prefetched(0, pref0, global_batch_size, sharding))
          for got, ref in zip(got_seq0, seq_ep0, strict=True):
            np.testing.assert_array_equal(got, ref)
          for got, ref in zip(got_shuf0, shuf_ep0, strict=True):
            np.testing.assert_array_equal(got, ref)
          for got, ref in zip(got_shuf1, shuf_ep1, strict=True):
            np.testing.assert_array_equal(got, ref)
          for got, ref in zip(got_pref0, shuf_ep0, strict=True):
            np.testing.assert_array_equal(got, ref)

      # Concurrent multi-step prefetch across 8 worker threads stays bit-exact.
      with futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(
                shuf_reader.prefetch_host_rows,
                s,
                global_batch_size,
                shardings[(2, 2, 2)],
            )
            for s in range(8)
        ]
        pref_list = [f.result() for f in futs]
      for s, pref_s in enumerate(pref_list):
        got_s = _batch_to_np(shuf_reader.make_batch_from_prefetched(s, pref_s, global_batch_size, shardings[(2, 2, 2)]))
        ref_s = shuf_reader.make_batch_np(s, global_batch_size)
        for got, ref in zip(got_s, ref_s, strict=True):
          np.testing.assert_array_equal(got, ref)

    vocab_size = 150144
    for pack_num_segments in (1, 2):
      ref_synth = _batch_to_np(
          data.make_synthetic_batch(
              step=3,
              global_batch_size=global_batch_size,
              seq_len=seq_len,
              vocab_size=vocab_size,
              sharding=ref_sharding,
              seed=7,
              pack_num_segments=pack_num_segments,
          )
      )
      next_step_synth = _batch_to_np(
          data.make_synthetic_batch(
              step=4,
              global_batch_size=global_batch_size,
              seq_len=seq_len,
              vocab_size=vocab_size,
              sharding=ref_sharding,
              seed=7,
              pack_num_segments=pack_num_segments,
          )
      )
      tok, tgt, pos, seg, mask = ref_synth
      self.assertFalse(np.array_equal(tok, next_step_synth[0]))
      self.assertGreaterEqual(int(tok.min()), 0)
      self.assertLess(int(tok.max()), vocab_size)
      self.assertFalse(np.array_equal(tok[0], tok[1]))
      self.assertTrue(bool(np.all(seg >= 1)))
      seg_len = seq_len // pack_num_segments
      for c in range(pack_num_segments):
        c_start = c * seg_len
        c_end = (c + 1) * seg_len
        np.testing.assert_array_equal(tgt[:, c_start : c_end - 1], tok[:, c_start + 1 : c_end])
        np.testing.assert_array_equal(tgt[:, c_end - 1], np.zeros(global_batch_size, dtype=np.int32))
        np.testing.assert_array_equal(mask[:, c_end - 1], np.zeros(global_batch_size, dtype=np.int32))
      self.assertEqual(int(mask[0].sum()), pack_num_segments * (seg_len - 1))
      if pack_num_segments == 1:
        np.testing.assert_array_equal(pos[0], np.arange(seq_len, dtype=np.int32))
        np.testing.assert_array_equal(seg[0], np.ones(seq_len, dtype=np.int32))
      else:
        np.testing.assert_array_equal(pos[0], np.tile(np.arange(seg_len, dtype=np.int32), 2))
        np.testing.assert_array_equal(seg[0], np.repeat(np.array([1, 2], dtype=np.int32), seg_len))

      for mesh_dims, sharding in shardings.items():
        with self.subTest(
            source="synthetic",
            pack_num_segments=pack_num_segments,
            mesh=mesh_dims,
        ):
          got_batch = data.make_synthetic_batch(
              step=3,
              global_batch_size=global_batch_size,
              seq_len=seq_len,
              vocab_size=vocab_size,
              sharding=sharding,
              seed=7,
              pack_num_segments=pack_num_segments,
          )
          fsdp_size, cp_size, ep_size = mesh_dims
          expected_local = (
              global_batch_size // (fsdp_size * ep_size),
              seq_len // cp_size,
          )
          self.assertEqual(got_batch.tokens.addressable_shards[0].data.shape, expected_local)
          for got, ref in zip(_batch_to_np(got_batch), ref_synth, strict=True):
            np.testing.assert_array_equal(got, ref)

  def test_single_file_path_and_fail_fast_errors(self):
    tmp_dir = self.create_tempdir().full_path
    shard_paths, raw_records = _write_array_record_shards(tmp_dir, shard_counts=(4,), rec_len=64)
    single_file = shard_paths[0]

    reader = data.ArrayRecordC4(single_file, seq_len=128)
    self.assertEqual(reader.num_rows, 2)
    chunks0 = reader.read_row_chunks(0)
    self.assertEqual(chunks0.shape, (2, 64))
    np.testing.assert_array_equal(chunks0[0], raw_records[0])
    np.testing.assert_array_equal(chunks0[1], raw_records[1])
    reader.close()
    with self.assertRaisesRegex(ValueError, "closed"):
      reader.read_row_chunks(0)

    # Missing directory, empty directory, and unmatched glob raise
    # FileNotFoundError.
    with self.assertRaisesRegex(FileNotFoundError, "No ArrayRecord files"):
      data.ArrayRecordC4(os.path.join(tmp_dir, "missing_subdir"), seq_len=64)

    empty_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(FileNotFoundError, "No ArrayRecord files"):
      data.ArrayRecordC4(empty_dir, seq_len=64)

    with self.assertRaisesRegex(FileNotFoundError, "No ArrayRecord files"):
      data.ArrayRecordC4(os.path.join(tmp_dir, "*.nomatch"), seq_len=64)

    # Invalid arguments & geometry -> ValueError
    with self.assertRaisesRegex(ValueError, "seq_len must be > 0"):
      data.ArrayRecordC4(single_file, seq_len=0)

    with self.assertRaisesRegex(ValueError, "positive multiple"):
      data.ArrayRecordC4(single_file, seq_len=100)

    with self.assertRaisesRegex(ValueError, "Need at least 8 records"):
      data.ArrayRecordC4(single_file, seq_len=512)

    # Corrupted proto bytes or missing/empty 'text' feature -> ValueError
    with self.assertRaisesRegex(ValueError, "empty"):
      data.parse_row(b"")

    with self.assertRaisesRegex(ValueError, "Failed to decode"):
      data.parse_row(b"\xff\xff\xff\x80")

    missing_feature_bytes = _serialize_token_record([1, 2, 3, 4], feature_name="wrong_feature")
    with self.assertRaisesRegex(ValueError, "missing or empty"):
      data.parse_row(missing_feature_bytes)

    bad_feature_path = os.path.join(self.create_tempdir().full_path, "missing_text.array_record")
    writer = array_record_module.ArrayRecordWriter(bad_feature_path, "group_size:1")
    writer.write(missing_feature_bytes)
    writer.close()
    with self.assertRaisesRegex(ValueError, "missing or empty"):
      data.ArrayRecordC4(bad_feature_path, seq_len=4)

    corrupt_path = os.path.join(self.create_tempdir().full_path, "corrupt.array_record")
    writer = array_record_module.ArrayRecordWriter(corrupt_path, "group_size:1")
    writer.write(b"\xff\xff\xff\x80")
    writer.close()
    with self.assertRaisesRegex(ValueError, "Failed to decode"):
      data.ArrayRecordC4(corrupt_path, seq_len=64)

    short_rec_path = os.path.join(self.create_tempdir().full_path, "len1.array_record")
    writer = array_record_module.ArrayRecordWriter(short_rec_path, "group_size:1")
    writer.write(_serialize_token_record([42]))
    writer.close()
    with self.assertRaisesRegex(ValueError, "record_len must be >= 2"):
      data.ArrayRecordC4(short_rec_path, seq_len=1)

    ragged_rec_path = os.path.join(self.create_tempdir().full_path, "ragged.array_record")
    writer = array_record_module.ArrayRecordWriter(ragged_rec_path, "group_size:1")
    writer.write(_serialize_token_record(range(64)))
    writer.write(_serialize_token_record(range(32)))
    writer.close()
    with data.ArrayRecordC4(ragged_rec_path, seq_len=128) as ragged_reader:
      with self.assertRaisesRegex(ValueError, "Record 1 has shape"):
        ragged_reader.read_row_chunks(0)

    neg_rec_path = os.path.join(self.create_tempdir().full_path, "neg_tokens.array_record")
    writer = array_record_module.ArrayRecordWriter(neg_rec_path, "group_size:1")
    writer.write(_serialize_token_record([10, 20, -1, 30]))
    writer.close()
    with self.assertRaisesRegex(ValueError, "Out-of-range token IDs"):
      data.ArrayRecordC4(neg_rec_path, seq_len=4)

    # Synthetic batch fail-fast checks.
    mesh_1 = flint_sharding.create_mesh(
        flint_config.FlintShardingConfig(fsdp_size=1, cp_size=1, ep_size=1),
        devices=jax.devices()[:1],
    )
    sharding_1 = NamedSharding(mesh_1, flint_sharding.batch_pspec())
    with self.assertRaisesRegex(ValueError, "pack_num_segments"):
      data.make_synthetic_batch(0, 2, 64, 256, sharding_1, pack_num_segments=3)
    with self.assertRaisesRegex(ValueError, "step must be >= 0"):
      data.make_synthetic_batch(-1, 2, 64, 256, sharding_1)


if __name__ == "__main__":
  absltest.main()
