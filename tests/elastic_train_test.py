# Copyright 2023–2025 Google LLC
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
"""
Tests the elastic related functions in elastic_train.py
"""

from unittest import mock
import logging
import os.path
import time

from absl.testing import absltest
from absl.testing import parameterized

import jax

from pathwaysutils.elastic import manager

from MaxText import elastic_train
from maxtext.src.maxtext import max_utils
from maxtext.src.maxtext import pyconfig
from maxtext.src.maxtext.globals import MAXTEXT_PKG_DIR

logging.basicConfig()
logging.getLogger("pathwaysutils.elastic.manager").setLevel(logging.INFO)


class ElasticTrainTest(parameterized.TestCase):

  def tearDown(self):
    """Clean up at the end of the test

    HyperParameters and they must be removed so that they do not impact other unittest
    """
    try:
      del pyconfig.HyperParameters.global_batch_size_to_train_on
      del pyconfig.HyperParameters.global_batch_size_to_load
      del pyconfig.HyperParameters.micro_batch_size_to_train_on
      del pyconfig.HyperParameters.num_slices
    except AttributeError:
      pass

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

  @parameterized.named_parameters(
      ("4_out_of_4_100", {0, 1, 2, 3}, 4, 100, 100),
      ("3_out_of_4_100", {1, 2, 3}, 4, 100, 75),
      ("2_out_of_4_100", {0, 3}, 4, 100, 50),
      ("1_out_of_4_100", {2}, 4, 100, 25),
      ("0_out_of_4_100", {}, 4, 100, 0),
      ("3_out_of_3_100", {0, 1, 2}, 3, 63, 63),
      ("2_out_of_3_100", {0, 2}, 3, 63, 42),
      ("1_out_of_3_100", {1}, 3, 63, 21),
      ("0_out_of_3_100", {}, 3, 63, 0),
  )
  def test_pyconfig_changes(self, good_slice_indices, total_slice_count, base_number, expected_number):

    # Mock max_utils to report that there are 4 slices
    # This is used to set config.num_slices
    self.enter_context(
        mock.patch.object(
            max_utils,
            "get_num_slices",
            return_value=total_slice_count,
            create_autospec=True,
        )
    )

    # Mock jax.devices to report that there is 1 device
    # This is used to compute the micro_batch_size_to_load,
    # micro_batch_size_to_train_on, global_batch_size_to_load,
    # global_batch_size_to_train_on
    # All of those will be equal to per_device_batch_size based on base.yml
    self.enter_context(
        mock.patch.object(
            jax,
            "devices",
            return_value=[
                mock.create_autospec(
                    jax.Device,
                    instance=True,
                ),
            ],
            create_autospec=True,
        )
    )

    # Set checkpoint_period which should be unchanged.
    # Set enable_single_controller to avoid jax.distribute_initialize
    config = pyconfig.initialize(
        argv=[
            "test",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
        ],
        per_device_batch_size=base_number,
        checkpoint_period=1234,
        enable_single_controller=True,
    )

    # Do not set any devices and instead overwrite the total_slice_count and the
    # good_slice_indices directly to avoid code paths that would otherwise need
    # additional mocking
    elastic_manager = elastic_train.elastic_initialize([])
    elastic_manager._total_slice_count = total_slice_count  # pylint: disable=protected-access
    elastic_manager.good_slice_indices = good_slice_indices

    self.assertEqual(config.global_batch_size_to_train_on, expected_number)
    self.assertEqual(config.global_batch_size_to_load, expected_number)
    self.assertEqual(config.micro_batch_size_to_train_on, expected_number)
    self.assertEqual(config.num_slices, len(good_slice_indices))
    self.assertEqual(config.checkpoint_period, 1234)


if __name__ == "__main__":
  absltest.main()
