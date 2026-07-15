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

"""Tests for DiLoCo data loader sharding."""

import sys
import unittest
from unittest.mock import patch
import jax
from jax.sharding import Mesh
import numpy as np
from maxtext.configs import pyconfig
from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator


class DilocoDataLoaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="test",
        global_batch_size_to_load=8,
        max_target_length=128,
        colocated_python_data_input=False,
        expansion_factor_real_data=1.0,
    )

    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), ["data"])
    self.process_indices = [0]

  @patch("maxtext.input_pipeline.grain_data_processing.get_datasets")
  @patch("maxtext.input_pipeline.grain_data_processing._get_pipeline_fn")
  @patch("maxtext.input_pipeline.multihost_dataloading.MultiHostDataLoadIterator")
  def test_grain_loader_sharding_diloco(self, mock_iterator, mock_get_pipeline_fn, mock_get_datasets):
    mock_get_datasets.return_value = unittest.mock.MagicMock()
    mock_get_pipeline_fn.return_value = lambda **kwargs: (lambda dataset: dataset)

    make_grain_train_iterator(self.config, self.mesh, self.process_indices, learner_idx=0, num_learners=2)
    kwargs = mock_get_datasets.call_args[1]
    self.assertEqual(kwargs["dataloading_host_index"], 0)
    self.assertEqual(kwargs["dataloading_host_count"], 2)

    mock_get_datasets.reset_mock()

    make_grain_train_iterator(self.config, self.mesh, self.process_indices, learner_idx=1, num_learners=2)
    kwargs = mock_get_datasets.call_args[1]
    self.assertEqual(kwargs["dataloading_host_index"], 1)
    self.assertEqual(kwargs["dataloading_host_count"], 2)

  @patch("maxtext.input_pipeline.tfds_data_processing.get_datasets")
  @patch("maxtext.input_pipeline.tfds_data_processing.preprocessing_pipeline")
  @patch("maxtext.input_pipeline.multihost_dataloading.MultiHostDataLoadIterator")
  def test_tfds_loader_sharding_diloco(self, mock_iterator, mock_preprocess, mock_get_datasets):
    mock_get_datasets.return_value = unittest.mock.MagicMock()

    make_tfds_train_iterator(self.config, self.mesh, self.process_indices, learner_idx=0, num_learners=2)
    kwargs = mock_get_datasets.call_args[1]
    self.assertEqual(kwargs["dataloading_host_index"], 0)
    self.assertEqual(kwargs["dataloading_host_count"], 2)

    mock_get_datasets.reset_mock()

    make_tfds_train_iterator(self.config, self.mesh, self.process_indices, learner_idx=1, num_learners=2)
    kwargs = mock_get_datasets.call_args[1]
    self.assertEqual(kwargs["dataloading_host_index"], 1)
    self.assertEqual(kwargs["dataloading_host_count"], 2)


if __name__ == "__main__":
  unittest.main()
