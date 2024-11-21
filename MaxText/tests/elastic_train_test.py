#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Tests the elastic related functions in elastic_train.py
"""

import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from MaxText import elastic_train
from pathwaysutils.elastic import manager


class ElasticTrainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("ready_after_0_try", [{0, 1}]),
      ("nothing_available_at_first", [{}, {0, 1}]),
      ("nothing_available_for_a_few_times", [{}, {}, {}, {0, 1}]),
      ("back_and_forth", [{}, {1}, {}, {0}, {}, {1}, {0}, {}, {0}, {0, 1}]),
  )
  def test_wait_for_all_slices(self, slice_availability_side_effect):
    mock_manager = mock.create_autospec(manager.Manager, instance=True)
    mock_manager.total_slice_count = 2
    mock_manager.get_slice_availability.side_effect = slice_availability_side_effect

    mock_sleep = self.enter_context(mock.patch.object(time, "sleep", create_autospec=True))

    elastic_train.wait_for_all_slices(mock_manager)

    self.assertEqual(mock_sleep.call_count, len(slice_availability_side_effect) - 1)


if __name__ == "__main__":
  absltest.main()
