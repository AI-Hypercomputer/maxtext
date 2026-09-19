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

"""Guards the MoE `wo` zero-pad behaviour of the RL rollout weight sync.

RL syncs trainer weights into the vLLM rollout through Tunix's
`_align_per_axis`, which zero-pads MoE MLP weights and repeats everything
else. Tunix's key set omits `wo`, so a `wo` whose intermediate dim the rollout
padded for GMM_v2 took the repeat branch and crashed the job (gemma4-26b
704 -> 1024, qwen3-30b-a3b 768 -> 1024). `maxtext_vllm_rollout` closes the gap
at import time.

These tests assert the resulting *behaviour*, not who fixed it, so they keep
passing once the fix lands upstream in Tunix and the workaround is deleted --
but they fail if the workaround is deleted while Tunix is still missing `wo`.
"""

import os

# Must precede the first JAX import; `_align_per_axis` jits its pad helper.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest  # pylint: disable=wrong-import-position

import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

# Imported for its side effect: the module patches Tunix's MoE key set at
# import time, exactly as it does in production via train_rl.py.
from maxtext.integration.vllm import maxtext_vllm_rollout  # pylint: disable=unused-import,wrong-import-position
from maxtext.integration.vllm.convert_utils import MOE_MLP_WEIGHTS  # pylint: disable=wrong-import-position
from tunix.generate import utils as tunix_utils  # pylint: disable=wrong-import-position

pytestmark = pytest.mark.post_training

# The production shapes at 1/64 scale. What matters is that the pad is not an
# integer multiple of the source dim (1024 % 704 != 0, like 16 % 11 != 0), so
# the repeat branch cannot express it and raises instead of silently
# double-counting.
EXPERTS = 2
EMB = 4
SRC_MOE_DIM = 11
TGT_MOE_DIM = 16


def _wo(dim):
  """A `wo`-shaped ramp: (experts, moe_intermediate, emb)."""
  size = EXPERTS * dim * EMB
  return np.arange(size, dtype=np.float32).reshape(EXPERTS, dim, EMB)


class TunixMoeZeroPadTest(unittest.TestCase):
  """`wo` must reach the rollout zero-padded on its contracting axis."""

  @pytest.mark.cpu_only
  def test_padded_wo_is_zero_padded_not_repeated(self):
    src = _wo(SRC_MOE_DIM)
    tgt_shape = (EXPERTS, TGT_MOE_DIM, EMB)

    out = np.asarray(
        tunix_utils._align_per_axis(  # pylint: disable=protected-access
            src, tgt_shape, None, "decoder.layers_0.moe_block.wo"
        )
    )

    self.assertEqual(out.shape, tgt_shape)
    np.testing.assert_array_equal(out[:, :SRC_MOE_DIM, :], src)
    np.testing.assert_array_equal(out[:, SRC_MOE_DIM:, :], np.zeros((EXPERTS, TGT_MOE_DIM - SRC_MOE_DIM, EMB)))

  @pytest.mark.cpu_only
  def test_non_moe_key_still_takes_the_repeat_branch(self):
    """Negative control: the zero-pad above comes from the key, not the shape.

    Without it, a test that only checked `wo` could pass because Tunix started
    zero-padding everything.
    """
    with self.assertRaises(tunix_utils.ShapeMismatchError):
      tunix_utils._align_per_axis(  # pylint: disable=protected-access
          _wo(SRC_MOE_DIM), (EXPERTS, TGT_MOE_DIM, EMB), None, "decoder.layers_0.self_attention.out"
      )

  @pytest.mark.cpu_only
  def test_tunix_recognizes_every_maxtext_moe_weight(self):
    """Cheap, readable restatement of the invariant the two tests above probe."""
    self.assertTrue(
        MOE_MLP_WEIGHTS.issubset(tunix_utils._MOE_MLP_WEIGHTS),  # pylint: disable=protected-access
        f"Tunix zero-pads {sorted(tunix_utils._MOE_MLP_WEIGHTS)}, "  # pylint: disable=protected-access
        f"missing {sorted(MOE_MLP_WEIGHTS - tunix_utils._MOE_MLP_WEIGHTS)}",  # pylint: disable=protected-access
    )


if __name__ == "__main__":
  unittest.main()
