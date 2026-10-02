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

"""Config tests for the split SparseCore pin flags (EP-only, FSDP-only, FSDP forward-only)."""

import unittest

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path


class MoePinSplitConfigTest(unittest.TestCase):

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="moe_pin_split_test", enable_checkpointing=False, **kw
    )

  def test_defaults_off(self):
    cfg = self._init()
    self.assertFalse(cfg.moe_pin_sparse_core_all_gathers)
    self.assertFalse(cfg.moe_pin_sparse_core_ep_all_gathers)
    self.assertFalse(cfg.moe_pin_sparse_core_fsdp_all_gathers)
    self.assertFalse(cfg.moe_pin_sparse_core_fsdp_all_gathers_fwd_only)

  def test_ep_only_is_accepted(self):
    cfg = self._init(moe_pin_sparse_core_ep_all_gathers=True)
    self.assertTrue(cfg.moe_pin_sparse_core_ep_all_gathers)
    self.assertFalse(cfg.moe_pin_sparse_core_all_gathers)

  def test_fsdp_pin_rejects_two_stage_all_gather(self):
    with self.assertRaises(Exception):
      self._init(moe_pin_sparse_core_fsdp_all_gathers=True, moe_fsdp_use_two_stage_all_gather=True)


if __name__ == "__main__":
  unittest.main()
