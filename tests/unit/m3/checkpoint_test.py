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

"""m3 restores both legacy (`layers_<i>`) and m3 (`layers/layers/<i>`) checkpoints."""

import unittest

import jax.numpy as jnp

from maxtext.m3.core.checkpoint import from_legacy_layer_names, match_checkpoint_layer_names


class CheckpointTest(unittest.TestCase):

  def test_restore_target_follows_checkpoint_layout(self):
    leaf = jnp.ones((2,))
    m3 = {"decoder": {"layers": {"layers": {0: {"w": leaf}}}, "decoder_norm": {"scale": leaf}}}
    legacy = {"decoder": {"layers_0": {"w": leaf}, "decoder_norm": {"scale": leaf}}}
    scanned = {"decoder": {"layers": {"w": leaf}}}

    # Legacy unscanned checkpoint: rename the target, then rename the restored tree back.
    target = match_checkpoint_layer_names(m3, legacy)
    self.assertEqual(set(target["decoder"]), {"layers_0", "decoder_norm"})
    self.assertIs(from_legacy_layer_names(target, m3)["decoder"]["layers"]["layers"][0]["w"], leaf)
    # m3 checkpoints and scanned checkpoints (same layout for both stacks) pass through.
    self.assertIs(match_checkpoint_layer_names(m3, m3), m3)
    self.assertIs(match_checkpoint_layer_names(scanned, scanned), scanned)
    self.assertIs(from_legacy_layer_names(m3, m3)["decoder"]["layers"]["layers"][0]["w"], leaf)
    self.assertIs(from_legacy_layer_names(scanned, scanned), scanned)


if __name__ == "__main__":
  unittest.main()
