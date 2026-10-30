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

"""Tests for shrinking explicit gmm_v2 / tgmm_v2 tile sizes to the VMEM limit."""

import functools

from absl.testing import absltest
import jax.numpy as jnp

from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2

MIB = 2**20
BF16 = gmm_v2.InputConfigs(quant_dtype=None, quant_block_size=-1, dtype=jnp.dtype(jnp.bfloat16))

gmm_bytes = functools.partial(
    gmm_v2.gmm_vmem_bytes,
    lhs_cfgs=BF16,
    rhs_cfgs=BF16,
    out_dtype=jnp.dtype(jnp.bfloat16),
    acc_dtype=jnp.dtype(jnp.float32),
    fuse_act=None,
    has_partial_sum=False,
)
tgmm_bytes = functools.partial(
    tgmm_v2.tgmm_vmem_bytes,
    lhs_cfgs=BF16,
    rhs_cfgs=BF16,
    out_dtype=jnp.dtype(jnp.bfloat16),
    acc_dtype=jnp.dtype(jnp.float32),
    has_partial_sum=False,
)


class FitTilesToVmemTest(absltest.TestCase):

  def test_unchanged_when_tiles_fit(self):
    tiles = gmm_v2.TileSizes(512, 1024, 1024)
    self.assertEqual(gmm_v2.fit_tiles_to_vmem(tiles, gmm_bytes, 64 * MIB), tiles)

  def test_drops_rhs_buffer_before_resizing(self):
    tiles = gmm_v2.TileSizes(512, 1024, 1024)
    limit = gmm_bytes(gmm_v2.TileSizes(512, 1024, 1024, rhs_buffer_count=2))
    self.assertEqual(
        gmm_v2.fit_tiles_to_vmem(tiles, gmm_bytes, limit), gmm_v2.TileSizes(512, 1024, 1024, rhs_buffer_count=2)
    )

  def test_shrinks_to_fit_limit(self):
    tiles = gmm_v2.TileSizes(512, 1024, 1024)
    for vmem_bytes in (gmm_bytes, tgmm_bytes):
      self.assertGreater(vmem_bytes(tiles), 12 * MIB)
      fitted = gmm_v2.fit_tiles_to_vmem(tiles, vmem_bytes, 12 * MIB)
      self.assertLessEqual(vmem_bytes(fitted), 12 * MIB)
      for size in (fitted.tile_m, fitted.tile_k, fitted.tile_n):
        self.assertEqual(size % 128, 0)

  def test_halves_larger_of_tile_n_and_tile_k_first(self):
    fitted = gmm_v2.fit_tiles_to_vmem(gmm_v2.TileSizes(512, 2048, 512), gmm_bytes, 12 * MIB)
    self.assertEqual((fitted.tile_m, fitted.tile_n), (512, 512))
    self.assertLess(fitted.tile_k, 2048)

  def test_not_resizable_only_changes_buffer_count(self):
    tiles = gmm_v2.TileSizes(512, 2048, 2048)
    fitted = gmm_v2.fit_tiles_to_vmem(tiles, gmm_bytes, 8 * MIB, resizable=False)
    self.assertEqual(fitted, gmm_v2.TileSizes(512, 2048, 2048, rhs_buffer_count=2))


if __name__ == "__main__":
  absltest.main()
