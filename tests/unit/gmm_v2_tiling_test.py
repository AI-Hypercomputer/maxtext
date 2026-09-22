# Copyright 2023-2026 Google LLC
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

"""GMM v2 tile sizes must be clamped to the extents they index.

Regression test for a silent-NaN bug. gmm_v2 indexes its operands by tile, so a
tile larger than its dimension walks off the array. Measured on tpu7x with
OLMo 3.5 `tiny`, whose latent (the contracting dim) is 512 against the 1024
default of `wi_tile_fwd_embed_dim`: training aborted on a NaN loss at step 1,
while megablox and tokamax GMM v1 were clean at the identical config. On a
smaller-VMEM part the same over-request surfaces as CompileTimeScopedVmemOom
instead, which is how it stayed hidden.

These are pure arithmetic checks on the clamp helper, so they need no TPU.
"""

import unittest

from maxtext.kernels.megablox.ops import _clamp_tiles


class ClampTilesTest(unittest.TestCase):
  """`_clamp_tiles` shrinks to the operand extent and never grows."""

  def test_olmo35_tiny_wi_gemm_is_clamped(self):
    """The shape that actually produced NaN: latent 512 under a 1024 tile."""
    # wi GEMM: m = tokens*top_k, k = latent 512, n = expert hidden 1024,
    # against base.yml defaults of (512, 1024, 1024).
    self.assertEqual(_clamp_tiles(512, 1024, 1024, 131072, 512, 1024), (512, 512, 1024))

  def test_olmo35_tiny_wo_gemm_is_clamped(self):
    """wo runs the other way: k = expert hidden 1024, n = latent 512."""
    self.assertEqual(_clamp_tiles(512, 1024, 1024, 131072, 1024, 512), (512, 1024, 512))

  def test_tiles_are_never_grown(self):
    """Clamping must only ever shrink; a small tile is legal, just slower."""
    self.assertEqual(_clamp_tiles(128, 128, 128, 131072, 512, 1024), (128, 128, 128))

  def test_exact_fit_is_untouched(self):
    self.assertEqual(_clamp_tiles(512, 512, 1024, 512, 512, 1024), (512, 512, 1024))

  def test_every_axis_clamps_independently(self):
    self.assertEqual(_clamp_tiles(9999, 9999, 9999, 7, 11, 13), (7, 11, 13))

  def test_larger_rungs_are_unaffected(self):
    """medium/large have latents above the default tile, so nothing changes.

    This is why the bug only ever showed on `tiny` and `small`: `medium`'s
    latent is 1280 and `large`'s is 2304, both >= the 1024 default.
    """
    for latent, expert_hidden in ((1280, 2560), (2304, 4608)):
      self.assertEqual(_clamp_tiles(512, 1024, 1024, 131072, latent, expert_hidden), (512, 1024, 1024))


if __name__ == "__main__":
  unittest.main()
