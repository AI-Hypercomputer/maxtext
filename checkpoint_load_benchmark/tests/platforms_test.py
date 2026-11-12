# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests checkpoint_load_benchmark/bench/platforms.py."""

import os
import sys
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import platforms  # pylint: disable=wrong-import-position


class PlatformProfilesTest(unittest.TestCase):
  """Verifies verified cluster profile values and topology helpers."""

  def test_devices_per_slice(self):
    self.assertEqual(platforms.get_profile("v7x").devices_per_slice("2x2x1"), 8)
    self.assertEqual(platforms.get_profile("v7x").devices_per_slice("2x2x2"), 16)
    self.assertEqual(platforms.get_profile("v7x").devices_per_slice("2x2x4"), 32)
    self.assertEqual(platforms.get_profile("v6e").devices_per_slice("2x4"), 8)
    self.assertEqual(platforms.get_profile("v6e").devices_per_slice("4x4"), 16)
    self.assertEqual(platforms.get_profile("v5e").devices_per_slice("16x16"), 256)

  def test_hosts_per_slice_and_chips_per_host(self):
    self.assertEqual(platforms.get_profile("v7x").hosts_per_slice("2x2x1"), 1)
    self.assertEqual(platforms.get_profile("v6e").hosts_per_slice("2x4"), 1)
    self.assertEqual(platforms.get_profile("v5e").hosts_per_slice("16x16"), 64)

    self.assertEqual(platforms.get_profile("v7x").chips_per_host("2x2x1"), 4)
    self.assertEqual(platforms.get_profile("v6e").chips_per_host("2x4"), 8)
    self.assertEqual(platforms.get_profile("v6e").chips_per_host("2x2"), 4)
    self.assertEqual(platforms.get_profile("v6e").chips_per_host("4x4"), 4)
    self.assertEqual(platforms.get_profile("v5e").chips_per_host("16x16"), 4)

  def test_unknown_platform_or_topology_raises_value_error(self):
    with self.assertRaises(ValueError):
      platforms.get_profile("v99")
    with self.assertRaises(ValueError):
      platforms.get_profile("v6e").devices_per_slice("2x2x1")
    with self.assertRaises(ValueError):
      platforms.get_profile("v5e").hosts_per_slice("2x2")
    with self.assertRaises(ValueError):
      platforms.get_profile("v7x").chips_per_host("16x16")

  def test_mesa_naming_and_cluster_constraints(self):
    for name in ("v7x", "v6e", "v5e"):
      profile = platforms.get_profile(name)
      self.assertEqual(profile.pod_labels.get("app"), "mesa-bench-v2")
      self.assertTrue(profile.hf_token_secret_name.startswith("mesa-"))
      self.assertTrue(profile.hf_token_secret_optional)
      self.assertEqual(profile.gcs_code_dir, "gs://mesa-maxtext/mesa-bench-v2/code")
      self.assertEqual(profile.gcs_data_root, "gs://mesa-maxtext/mesa-bench-v2")

    v7x = platforms.get_profile("v7x")
    self.assertEqual(v7x.kueue_queue, "default")
    self.assertTrue(v7x.kueue_topology_aware_scheduling)
    self.assertEqual(v7x.image_pull_secrets, ("gitimg",))

    v6e = platforms.get_profile("v6e")
    self.assertEqual(v6e.kueue_queue, "default")
    self.assertFalse(v6e.kueue_topology_aware_scheduling)
    self.assertEqual(v6e.priority_class_name, "medium")
    self.assertEqual(v6e.image_pull_secrets, ())

    v5e = platforms.get_profile("v5e")
    self.assertIsNone(v5e.kueue_queue)
    self.assertFalse(v5e.kueue_topology_aware_scheduling)
    self.assertEqual(v5e.service_account_name, "mesa-bench")
    self.assertIsNone(v5e.workload_time_limit_minutes)


if __name__ == "__main__":
  unittest.main()
