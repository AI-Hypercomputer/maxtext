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

"""Unit tests for SparseCore ragged_gather_reduce_v2 large-input col_chunk_size capping."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from maxtext.kernels.ragged import ragged_gather_reduce_v2


class RaggedGatherReduceSpmemCapTest(unittest.TestCase):
  """Verifies large-input col_chunk_size caps and col_size divisibility invariants."""

  def test_small_and_boundary_inputs_leave_col_chunk_size_unchanged(self):
    for input_size in (4096, 655360):
      with self.subTest(input_size=input_size):
        self.assertEqual(
            ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(1024, 2048, input_size),  # pylint: disable=protected-access
            1024,
        )

  def test_medium_large_inputs_cap_at_512(self):
    for input_size in (655361, 1048576, 1310720):
      for col_size in (1024, 2048, 3584, 4096):
        with self.subTest(input_size=input_size, col_size=col_size):
          capped = ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(  # pylint: disable=protected-access
              1024, col_size, input_size
          )
          self.assertEqual(capped, 512)
          self.assertEqual(col_size % capped, 0)

  def test_very_large_inputs_cap_at_256(self):
    for input_size in (1310721, 2097152):
      for col_size in (1024, 2048, 3584, 4096):
        with self.subTest(input_size=input_size, col_size=col_size):
          capped = ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(  # pylint: disable=protected-access
              1024, col_size, input_size
          )
          self.assertEqual(capped, 256)
          self.assertEqual(col_size % capped, 0)

  def test_non_power_of_two_col_size_preserves_divisibility(self):
    # col_size = 768 = 6 * 128 -> largest divisor <= 512 is 384; largest divisor <= 256 is 256.
    self.assertEqual(
        ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(768, 768, 700000),  # pylint: disable=protected-access
        384,
    )
    self.assertEqual(
        ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(768, 768, 1500000),  # pylint: disable=protected-access
        256,
    )
    # col_size = 896 = 7 * 128 -> only 128-multiple divisors are 896 and 128.
    self.assertEqual(
        ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(896, 896, 700000),  # pylint: disable=protected-access
        128,
    )
    # Already smaller than cap stays untouched.
    self.assertEqual(
        ragged_gather_reduce_v2._cap_col_chunk_size_for_input_rows(256, 2048, 700000),  # pylint: disable=protected-access
        256,
    )

  def test_ragged_gather_reduce_wires_cap_into_kernel_config(self):
    """Verifies ragged_gather_reduce passes capped col_chunk_size into _Config."""
    captured_cfgs = []

    def _fake_kernel(body_fn, out_type, **kwargs):
      del kwargs
      captured_cfgs.append(body_fn.keywords["cfg"])
      return lambda *args: jnp.zeros(out_type.shape, dtype=out_type.dtype)

    tpu_info = ragged_gather_reduce_v2.pltpu.get_tpu_info_for_chip(ragged_gather_reduce_v2.pltpu.ChipVersion.TPU_7X, 1)
    x = jnp.ones((16, 2048), dtype=jnp.bfloat16)
    indices = jnp.zeros((1310728,), dtype=jnp.int32)
    weights = jnp.ones((1310728,), dtype=jnp.bfloat16)
    valid = jnp.ones((1310728,), dtype=jnp.bool_)

    with (
        mock.patch.object(ragged_gather_reduce_v2.pltpu, "is_tpu_device", return_value=True),
        mock.patch.object(ragged_gather_reduce_v2.pltpu, "get_tpu_info", return_value=tpu_info),
        mock.patch.object(
            ragged_gather_reduce_v2.plsc,
            "VectorSubcoreMesh",
            return_value=mock.MagicMock(core_axis_name="core", subcore_axis_name="subcore"),
        ),
        mock.patch.object(
            ragged_gather_reduce_v2,
            "_preprocess",
            return_value=(indices, jnp.zeros((8,), dtype=jnp.int32), valid),
        ),
        mock.patch.object(ragged_gather_reduce_v2.pl, "kernel", side_effect=_fake_kernel),
    ):
      out = jax.eval_shape(
          lambda: ragged_gather_reduce_v2.ragged_gather_reduce.__wrapped__(
              x, indices, weights, valid, reduce_group_size=8
          )
      )

    self.assertEqual(out.shape, (1310728 // 8, 2048))
    self.assertEqual(len(captured_cfgs), 1)
    self.assertEqual(captured_cfgs[0].col_chunk_size, 256)
    self.assertEqual(captured_cfgs[0].col_size % captured_cfgs[0].col_chunk_size, 0)


if __name__ == "__main__":
  unittest.main()
