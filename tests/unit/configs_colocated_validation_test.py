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

"""Tier A CPU tests for colocated python checkpointing config validation."""

import os
import unittest

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_test_config_path


class ConfigsColocatedValidationTest(unittest.TestCase):
  """Tests validation of checkpoint_storage_device_host_concurrent_gb for colocated Python."""

  def _init_config(self, **kwargs):
    return pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        **kwargs,
    )

  def test_default_config_passes(self):
    cfg = self._init_config()
    self.assertEqual(cfg.pathways_checkpointing_impl, "persistence")
    self.assertEqual(cfg.checkpoint_storage_device_host_concurrent_gb, 8)

  def test_colocated_python_positive_gb_passes(self):
    cfg = self._init_config(
        pathways_checkpointing_impl="colocated_python",
        checkpoint_storage_device_host_concurrent_gb=16,
    )
    self.assertEqual(cfg.pathways_checkpointing_impl, "colocated_python")
    self.assertEqual(cfg.checkpoint_storage_device_host_concurrent_gb, 16)

  def test_colocated_python_none_or_nonpositive_raises(self):
    for invalid_val in (None, 0, -4):
      with self.subTest(invalid_val=invalid_val):
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_storage_device_host_concurrent_gb must be positive",
        ):
          self._init_config(
              pathways_checkpointing_impl="colocated_python",
              checkpoint_storage_device_host_concurrent_gb=invalid_val,
          )

  def test_persistence_with_none_passes(self):
    cfg = self._init_config(
        pathways_checkpointing_impl="persistence",
        checkpoint_storage_device_host_concurrent_gb=None,
    )
    self.assertEqual(cfg.pathways_checkpointing_impl, "persistence")
    self.assertIsNone(cfg.checkpoint_storage_device_host_concurrent_gb)


if __name__ == "__main__":
  unittest.main()
