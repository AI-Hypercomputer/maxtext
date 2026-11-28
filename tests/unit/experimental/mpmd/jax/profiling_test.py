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

"""CPU tests of profiling.py: what every pipeline profile traces unless told otherwise, and the host names' order.

    python3 -m pytest tests/unit/experimental/mpmd/jax/profiling_test.py -q
"""

import argparse
import unittest

from maxtext.experimental.mpmd.jax import profiling as prof


class ProfilingTest(unittest.TestCase):

  def test_labels_sort_in_stage_then_process_order(self):
    """XProf's order is the names': unpadded below ten, as before; padded to the run's widths past it; a process
    holding no stage last."""
    self.assertEqual((prof.trace_label(1, 16, 2, 32, "h"), prof.trace_label(0, 0, 2, 4, "h")), ("s1-p16-h", "s0-p0-h"))
    self.assertEqual(prof.trace_label(0, 0, 2, 32, "h"), "s0-p00-h")
    labels = [prof.trace_label(s, q, 16, 128, "h") for s in range(16) for q in (8 * s, 8 * s + 1)]
    labels.append(prof.trace_label(None, 127, 16, 128, "h"))
    self.assertEqual(sorted(labels), labels)
    self.assertEqual(labels[4:6], ["s02-p016-h", "s02-p017-h"])
    self.assertEqual([prof.padded(i, 12) for i in (0, 2, 11)], ["00", "02", "11"])
    self.assertEqual([prof.padded(i, 10) for i in (0, 9)], ["0", "9"])

  def test_one_tpu_one_sparsecore_one_tile_by_default(self):
    """One chip a traced process, whatever it holds, and MaxText's profiler told the same."""
    self.assertEqual(prof.CHIPS, 1)
    self.assertEqual(
        prof.profile_options().advanced_configuration,
        {
            "tpu_num_chips_to_profile_per_task": 1,
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        },
    )
    self.assertEqual(
        prof.MAXTEXT_ARGS,
        (
            "enable_tpu_profiling_options=true",
            "tpu_num_chips_to_profile_per_task=1",
            "tpu_num_sparse_cores_to_trace=1",
            "tpu_num_sparse_core_tiles_to_trace=1",
        ),
    )

  def test_positive_int(self):
    """--profile-chips's type (ds_ep_pipeline): a count of one or more."""
    self.assertEqual(prof.positive_int("4"), 4)
    for bad in ("0", "-1"):
      with self.subTest(n=bad), self.assertRaisesRegex(argparse.ArgumentTypeError, f"at least 1, not {bad}"):
        prof.positive_int(bad)
    with self.assertRaises(ValueError):  # argparse reports it as an invalid value
      prof.positive_int("x")

  def test_every_chip(self):
    """--profile-chips all (EVERY_CHIP, the default under Pathways): no chip count, the trims kept."""
    self.assertEqual(prof.chip_count("all"), prof.EVERY_CHIP)
    self.assertEqual(prof.chip_count("2"), 2)
    with self.assertRaises(argparse.ArgumentTypeError):
      prof.chip_count("0")
    self.assertEqual(prof.profile_options(prof.EVERY_CHIP).advanced_configuration, prof.SPARSECORE_TRIM)
    self.assertEqual(prof.profile_options(prof.EVERY_CHIP, platform="cuda").advanced_configuration, prof.GPU_EVENT_CAPS)

  def test_profile_options(self):
    o = prof.profile_options(1, host_tracer_level=2)
    self.assertEqual(
        o.advanced_configuration,
        {
            "tpu_num_chips_to_profile_per_task": 1,
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        },
    )
    self.assertEqual(o.host_tracer_level, 2)
    self.assertEqual(prof.profile_options(4).advanced_configuration["tpu_num_chips_to_profile_per_task"], 4)
    with self.assertRaisesRegex(ValueError, "at least one chip, not 0"):
      prof.profile_options(0)


if __name__ == "__main__":
  unittest.main()
