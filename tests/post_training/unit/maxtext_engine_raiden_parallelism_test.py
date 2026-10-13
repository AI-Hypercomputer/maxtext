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
"""Tests for the trainer weight synchronizer's RAIDEN_PARALLELISM override."""

import os
import unittest
from unittest import mock

from maxtext.training_engine import maxtext_engine
import pytest

pytestmark = [pytest.mark.post_training]


class RaidenParallelismTest(unittest.TestCase):

  def _parallelism(self, value):
    env = {} if value is None else {"RAIDEN_PARALLELISM": value}
    with mock.patch.dict(os.environ, env, clear=False):
      if value is None:
        os.environ.pop("RAIDEN_PARALLELISM", None)
      return maxtext_engine._raiden_parallelism()  # pylint: disable=protected-access

  def test_unset_keeps_the_default(self):
    self.assertEqual(self._parallelism(None), (4, "default"))

  def test_empty_keeps_the_default(self):
    self.assertEqual(self._parallelism("  "), (4, "default"))

  def test_env_value_is_used(self):
    self.assertEqual(self._parallelism("16"), (16, "env RAIDEN_PARALLELISM"))

  def test_invalid_values_are_rejected(self):
    for raw in ("abc", "0", "-2"):
      with self.subTest(raw=raw), self.assertRaises(ValueError):
        self._parallelism(raw)


if __name__ == "__main__":
  unittest.main()
