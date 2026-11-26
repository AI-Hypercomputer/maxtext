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

"""Trains the Shardy MPMD pipeline on Pathways: compile, and feed each step's state to the next.

Shardy MPMD's IFRT IR program compiles only through Pathways' IFRT proxy, so this runs only with JAX_PLATFORMS=proxy
against a Pathways backend of 4 or more TPU devices (e.g. local_pathways.sh on a TPU VM), with shardy-mpmd installed.
"""

import unittest

import jax
import pathwaysutils
import pytest

from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import train

try:
  from maxtext.experimental.mpmd.partir import sdy_mpmd_compat  # pylint: disable=unused-import

  _SKIP = ""
except ModuleNotFoundError as error:  # shardy-mpmd is an optional dependency
  if not (error.name or "").startswith("shardy"):
    raise
  _SKIP = f"needs shardy-mpmd: {error}"


@pytest.mark.tpu_only
@unittest.skipIf(bool(_SKIP), _SKIP)
class PathwaysTrainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if not pathwaysutils.is_pathways_backend_used():
      self.skipTest("needs a Pathways backend (JAX_PLATFORMS=proxy): Shardy MPMD programs compile only there")
    if len(jax.devices()) < 4:
      self.skipTest("needs 4 devices")

  def test_steps_run_and_the_loss_falls(self):
    hparams = hp.Hparams(
        model_name="simple",
        num_layers=4,
        num_stages=2,
        fsdp=2,
        num_microbatches=4,
        per_device_batch_size=4,
        max_target_length=128,
        steps=3,
        warmup_steps=1,
    )
    # A step that rejected its predecessor's output ("Input signature does not match") would raise here.
    result = train.run_training(hparams)
    self.assertEqual(result.stats.num_steps, 3)
    self.assertLess(result.stats.loss, result.stats.first_loss)
    self.assertGreater(result.throughput.tokens_per_second, 0)


if __name__ == "__main__":
  unittest.main()
