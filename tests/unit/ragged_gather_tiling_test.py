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

"""Tests the SparseCore VMEM budget used to tile the ragged gather kernels.

Runs on CPU: `pltpu.get_tpu_info` is replaced by the static hardware table for each chip, so the expected column
sizes follow from `SparseCoreInfo.vmem_capacity_bytes`.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels.ragged import ragged_gather
from maxtext.kernels.ragged import ragged_gather_reduce_v2


def _patch_chip(chip_version: pltpu.ChipVersion):
  info = pltpu.get_tpu_info_for_chip(chip_version, 1)
  return mock.patch.object(pltpu, "get_tpu_info", return_value=info)


class RaggedGatherTilingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # (name, chip, hidden_size, expected col_size)
      ("v6e_h2048", pltpu.ChipVersion.TPU_V6E, 2048, 2048),
      ("v6e_h4096", pltpu.ChipVersion.TPU_V6E, 4096, 4096),
      ("v6e_h8192", pltpu.ChipVersion.TPU_V6E, 8192, 4096),
      ("v5p_h8192", pltpu.ChipVersion.TPU_V5P, 8192, 8192),
      ("tpu7x_h2048", pltpu.ChipVersion.TPU_7X, 2048, 2048),
      ("tpu7x_h7168", pltpu.ChipVersion.TPU_7X, 7168, 7168),
      ("tpu8i_h8192", pltpu.ChipVersion.TPU_8I, 8192, 4096),
      ("tpu8t_h8192", pltpu.ChipVersion.TPU_8T, 8192, 2816),
  )
  def test_gather_col_size_uses_sparsecore_vmem(self, chip, hidden_size, expected):
    with _patch_chip(chip):
      sc_info = pltpu.get_tpu_info().sparse_core
      col_size = ragged_gather.calculate_col_size(hidden_size)
    self.assertEqual(col_size, expected)
    # The uint32 gather buffer for one column block fits in 90% of SparseCore VMEM.
    self.assertLessEqual(sc_info.num_lanes * col_size * 4, sc_info.vmem_capacity_bytes * 0.9)

  @parameterized.named_parameters(
      # (name, chip, col_size, expected col_chunk_size)
      ("v6e_col1024", pltpu.ChipVersion.TPU_V6E, 1024, 1024),
      ("v6e_col3584", pltpu.ChipVersion.TPU_V6E, 3584, 896),
      ("tpu7x_col2048", pltpu.ChipVersion.TPU_7X, 2048, 1024),
      ("tpu8i_col2048", pltpu.ChipVersion.TPU_8I, 2048, 1024),
      ("tpu8t_col4096", pltpu.ChipVersion.TPU_8T, 4096, 1024),
  )
  def test_reduce_col_chunk_size_uses_sparsecore_vmem(self, chip, col_size, expected):
    with _patch_chip(chip):
      sc_info = pltpu.get_tpu_info().sparse_core
      chunk = ragged_gather_reduce_v2._calculate_col_chunk_size(col_size, sc_info.num_lanes)  # pylint: disable=protected-access
    self.assertEqual(chunk, expected)
    # The double-buffered uint32 gather buffer fits in 95% of SparseCore VMEM.
    self.assertLessEqual(sc_info.num_lanes * chunk * 4 * 2, sc_info.vmem_capacity_bytes * 0.95)


if __name__ == "__main__":
  absltest.main()
