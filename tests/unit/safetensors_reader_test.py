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

"""Tests for safetensors_reader, the direct SafeTensors -> sharded jax.Array reader."""

import os

# Simulate 4 devices on CPU so the shardings are real. This only takes effect if JAX
# is not initialized yet.
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

# pylint: disable=wrong-import-position
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import safetensors_reader
import numpy as np
from orbax.checkpoint import v1 as ocp_v1
import safetensors.flax


def _tensors():
  """Name -> (array, partition spec on a 2x2 ("x", "y") mesh), covering every read path."""
  rng = np.random.default_rng(0)

  def normal(shape):
    return rng.standard_normal(shape, dtype=np.float32)

  return {
      "bf16_rows": (jnp.asarray(normal((8, 6))).astype(jnp.bfloat16), P(("x", "y"))),
      "f32_gcd_rows": (jnp.asarray(normal((6, 5))), P("x")),  # 6 rows: 2 pieces, each on 2 chips.
      "i32_full_copy": (jnp.asarray(rng.integers(-100, 100, (7, 3), dtype=np.int32)), P()),
      "f32_scalar": (jnp.asarray(np.float32(3.5)), P()),
      "u8_vector": (jnp.asarray(rng.integers(0, 255, (16,), dtype=np.uint8)), P(("x", "y"))),
      "f32_rows_and_cols": (jnp.asarray(normal((4, 8))), P("x", "y")),  # Splits dim 1 too.
      "bf16_cols_only": (jnp.asarray(normal((3, 8))).astype(jnp.bfloat16), P(None, ("x", "y"))),
      "f16_three_dims": (jnp.asarray(normal((4, 2, 6))).astype(jnp.float16), P("y", None, "x")),
  }


def _mesh():
  return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _write(directory, tensors, num_files=3):
  """Writes the tensors round-robin into `num_files` files."""
  names = sorted(tensors)
  for i in range(num_files):
    path = os.path.join(directory, f"model-{i:05d}-of-{num_files:05d}.safetensors")
    safetensors.flax.save_file({k: tensors[k] for k in names[i::num_files]}, path)


class SafetensorsReaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices.")
    self.mesh = _mesh()
    self.cases = _tensors()
    self.dir = self.create_tempdir().full_path
    _write(self.dir, {k: a for k, (a, _) in self.cases.items()})

  def _request(self, names=None):
    return {
        k: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=NamedSharding(self.mesh, spec))
        for k, (a, spec) in self.cases.items()
        if names is None or k in names
    }

  def test_metadata(self):
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      metadata = reader.metadata
    self.assertEqual(set(metadata), set(self.cases))
    for name, (array, _) in self.cases.items():
      self.assertEqual(metadata[name].shape, array.shape)
      self.assertEqual(metadata[name].dtype, array.dtype)

  @parameterized.named_parameters(
      ("default", safetensors_reader.READ_CHUNK_BYTES, safetensors_reader.READ_THREADS),
      ("tiny_chunks", 7, 8),  # Every block takes many ranged reads, some ending mid-element.
      ("one_thread", 5, 1),
  )
  def test_matches_saved_tensors(self, chunk_bytes, num_threads):
    request = self._request()
    with safetensors_reader.SafetensorsReader(self.dir, num_threads=num_threads, chunk_bytes=chunk_bytes) as reader:
      got = reader.read(request)
    self.assertEqual(list(got), list(request))  # Same order as the request.
    for name, (want, _) in self.cases.items():
      with self.subTest(name):
        self.assertEqual(got[name].dtype, want.dtype)
        self.assertEqual(got[name].sharding, request[name].sharding)
        np.testing.assert_array_equal(np.asarray(got[name]), np.asarray(want))
        # Each TPU chip's piece is exactly its part of the tensor.
        for shard in got[name].addressable_shards:
          np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(want)[shard.index])

  def test_matches_orbax(self):
    """Bit for bit equal to Orbax's SafeTensors loader, with the shardings the streaming loader uses."""
    devices = list(self.mesh.devices.flat)
    request = {
        name: jax.ShapeDtypeStruct(
            a.shape, a.dtype, sharding=hf_streaming_load.choose_load_sharding(a.shape, a.dtype, devices, 0)
        )
        for name, (a, _) in self.cases.items()
    }
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      got = reader.read(request)
    with ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS):
      want = ocp_v1.load(self.dir, request)
    for name in request:
      with self.subTest(name):
        self.assertEqual(got[name].dtype, want[name].dtype)
        self.assertEqual(got[name].sharding, want[name].sharding)
        np.testing.assert_array_equal(np.asarray(got[name]), np.asarray(want[name]))

  def test_neighbouring_tensors_take_one_read_per_file(self):
    """Pieces of all tensors in a file are merged into one byte range, fetched in one ranged read."""
    calls = []
    real = safetensors_reader.SafetensorsReader._read_range  # pylint: disable=protected-access

    def spy(reader, block, offset, length):
      calls.append(block.path)
      return real(reader, block, offset, length)

    with mock.patch.object(safetensors_reader.SafetensorsReader, "_read_range", autospec=True, side_effect=spy):
      with safetensors_reader.SafetensorsReader(self.dir) as reader:
        reader.read(self._request())
    self.assertLen(calls, 3)
    self.assertLen(set(calls), 3)

  def test_reads_a_subset_and_a_single_file(self):
    path = os.path.join(self.dir, "model-00000-of-00003.safetensors")
    with safetensors_reader.SafetensorsReader(path) as reader:
      names = list(reader.metadata)
      got = reader.read(self._request(names[:1]))
    self.assertEqual(list(got), names[:1])
    np.testing.assert_array_equal(np.asarray(got[names[0]]), np.asarray(self.cases[names[0]][0]))

  def test_empty_request(self):
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      self.assertEqual(reader.read({}), {})

  def test_zero_size_tensor(self):
    directory = self.create_tempdir().full_path
    _write(directory, {"empty": jnp.zeros((0, 4), jnp.float32), "x": jnp.ones((4,), jnp.float32)}, num_files=1)
    sharding = NamedSharding(self.mesh, P())
    with safetensors_reader.SafetensorsReader(directory) as reader:
      got = reader.read({"empty": jax.ShapeDtypeStruct((0, 4), jnp.float32, sharding=sharding)})
    self.assertEqual(got["empty"].shape, (0, 4))

  def test_missing_tensor_raises(self):
    sharding = NamedSharding(self.mesh, P())
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      with self.assertRaisesRegex(KeyError, "nope"):
        reader.read({"nope": jax.ShapeDtypeStruct((1,), jnp.float32, sharding=sharding)})

  @parameterized.named_parameters(
      ("wrong_shape", (6, 6), jnp.float32),
      ("wrong_dtype", (6, 5), jnp.bfloat16),
  )
  def test_mismatched_request_raises(self, shape, dtype):
    sharding = NamedSharding(self.mesh, P())
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      with self.assertRaisesRegex(ValueError, "Requested f32_gcd_rows"):
        reader.read({"f32_gcd_rows": jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)})

  def test_request_without_sharding_raises(self):
    with safetensors_reader.SafetensorsReader(self.dir) as reader:
      with self.assertRaisesRegex(ValueError, "without a sharding"):
        reader.read({"f32_gcd_rows": jax.ShapeDtypeStruct((6, 5), jnp.float32)})

  def test_truncated_file_raises(self):
    directory = self.create_tempdir().full_path
    _write(directory, {"x": jnp.ones((64,), jnp.float32)}, num_files=1)
    path = os.path.join(directory, "model-00000-of-00001.safetensors")
    with open(path, "r+b") as f:
      f.truncate(os.path.getsize(path) - 4)
    with self.assertRaisesRegex(ValueError, "truncated"):
      safetensors_reader.SafetensorsReader(directory)

  def test_duplicate_tensor_raises(self):
    directory = self.create_tempdir().full_path
    for i in range(2):
      safetensors.flax.save_file({"x": jnp.ones((4,))}, os.path.join(directory, f"m{i}.safetensors"))
    with self.assertRaisesRegex(ValueError, "Tensor x is in both"):
      safetensors_reader.SafetensorsReader(directory)

  def test_no_files_raises(self):
    with self.assertRaises(FileNotFoundError):
      safetensors_reader.SafetensorsReader(self.create_tempdir().full_path)

  def test_failed_init_shuts_down_read_threads(self):
    directory = self.create_tempdir().full_path
    for i in range(2):
      safetensors.flax.save_file({"x": jnp.ones((4,))}, os.path.join(directory, f"m{i}.safetensors"))
    shutdown = safetensors_reader.concurrent.futures.ThreadPoolExecutor.shutdown
    with mock.patch.object(
        safetensors_reader.concurrent.futures.ThreadPoolExecutor, "shutdown", autospec=True, side_effect=shutdown
    ) as mock_shutdown:
      with self.assertRaisesRegex(ValueError, "Tensor x is in both"):
        safetensors_reader.SafetensorsReader(directory)
    mock_shutdown.assert_called_once()


class StackedReadTest(parameterized.TestCase):
  """Groups of same-shape tensors with tiny per-chip pieces are read whole, then rearranged."""

  def setUp(self):
    """Writes same-shape tensors whose per-chip pieces are small enough to be read whole."""
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices.")
    self.mesh = _mesh()
    self.dir = self.create_tempdir().full_path
    rng = np.random.default_rng(1)
    self.tensors = {f"expert_{i}": jnp.asarray(rng.standard_normal((8, 6), dtype=np.float32)) for i in range(7)}
    self.tensors.update({f"norm_{i}": jnp.asarray(rng.standard_normal((5,), dtype=np.float32)) for i in range(4)})
    self.tensors["other"] = jnp.asarray(rng.standard_normal((4, 4), dtype=np.float32))
    _write(self.dir, self.tensors)

  def _request(self, names, spec):
    sharding = NamedSharding(self.mesh, spec)
    return {n: jax.ShapeDtypeStruct(self.tensors[n].shape, self.tensors[n].dtype, sharding=sharding) for n in names}

  def _check(self, got, request):
    self.assertEqual(list(got), list(request))
    for name in request:
      with self.subTest(name):
        self.assertEqual(got[name].sharding, request[name].sharding)
        np.testing.assert_array_equal(np.asarray(got[name]), np.asarray(self.tensors[name]))
        for shard in got[name].addressable_shards:
          np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(self.tensors[name])[shard.index])

  @parameterized.named_parameters(
      ("split_rows", P(("x", "y")), 256),
      ("split_rows_and_cols", P("x", "y"), 256),
      ("full_copy", P(), 256),
      ("split_in_chunks", P(("x", "y")), 3),  # 7 tensors: chunks of 3, 3 and 1.
  )
  def test_matches_saved_tensors(self, spec, split_chunk):
    names = [f"expert_{i}" for i in range(7)]
    request = self._request(reversed(names), spec)  # Not file order: the result keeps request order.
    with (
        mock.patch.object(safetensors_reader, "_SPLIT_CHUNK", split_chunk),
        safetensors_reader.SafetensorsReader(self.dir, min_piece_bytes=1 << 30) as reader,
    ):
      fetched = reader.fetch(request)
      self.assertLen(fetched.stacks, 1)
      self.assertEqual(fetched.arrays, {})
      got = reader.unpack(fetched)
    self._check(got, request)

  def test_stack_layout(self):
    """7 tensors on 4 chips: padded to 8, 2 consecutive whole tensors per chip."""
    request = self._request([f"expert_{i}" for i in range(7)], P(("x", "y")))
    with safetensors_reader.SafetensorsReader(self.dir, min_piece_bytes=1 << 30) as reader:
      (stack,) = reader.fetch(request).stacks
    self.assertEqual(stack.num_padded, 1)
    self.assertEqual(stack.array.shape, (8, 8, 6))
    for shard in stack.array.addressable_shards:
      self.assertEqual(shard.data.shape, (2, 8, 6))
    want = np.stack([np.asarray(self.tensors[n]) for n in stack.names] + [np.zeros((8, 6), np.float32)])
    np.testing.assert_array_equal(np.asarray(stack.array), want)

  def test_reads_whole_tensors(self):
    """Each tensor is read from storage once, in full, rather than once per piece."""
    request = self._request([f"expert_{i}" for i in range(7)], P(("x", "y")))
    pieces = []
    real = safetensors_reader.SafetensorsReader._whole  # pylint: disable=protected-access

    def spy(reader, name, devices):
      piece = real(reader, name, devices)
      pieces.append(piece)
      return piece

    with mock.patch.object(safetensors_reader.SafetensorsReader, "_whole", autospec=True, side_effect=spy):
      with safetensors_reader.SafetensorsReader(self.dir, min_piece_bytes=1 << 30) as reader:
        reader.read(request)
    self.assertCountEqual([p.name for p in pieces], list(request))
    for piece in pieces:
      self.assertEqual(piece.end - piece.start, 8 * 6 * 4)
      self.assertLen(piece.devices, 1)

  def test_mixed_request(self):
    """Stacked groups, a group of copies, and a lone tensor read by rows, in one request."""
    request = self._request([f"expert_{i}" for i in range(7)], P(("x", "y")))
    request.update(self._request([f"norm_{i}" for i in range(4)], P()))
    request.update(self._request(["other"], P("x")))
    with safetensors_reader.SafetensorsReader(self.dir, min_piece_bytes=1 << 30) as reader:
      fetched = reader.fetch(request)
      self.assertLen(fetched.stacks, 2)
      self.assertEqual(list(fetched.arrays), ["other"])
      got = reader.unpack(fetched)
    self._check(got, request)

  @parameterized.named_parameters(
      ("group_too_small", 1, 1 << 30),  # 1 tensor on 4 chips would mostly be padding.
      ("pieces_big_enough", 7, 0),
  )
  def test_reads_by_rows(self, num, min_piece_bytes):
    request = self._request([f"expert_{i}" for i in range(num)], P(("x", "y")))
    with safetensors_reader.SafetensorsReader(self.dir, min_piece_bytes=min_piece_bytes) as reader:
      fetched = reader.fetch(request)
      self.assertEqual(fetched.stacks, [])
      got = reader.unpack(fetched)
    self._check(got, request)


if __name__ == "__main__":
  absltest.main()
