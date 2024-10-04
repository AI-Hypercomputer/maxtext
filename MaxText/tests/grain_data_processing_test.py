"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import subprocess
import sys
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import unittest

import pyconfig
from input_pipeline import _grain_data_processing
from input_pipeline import input_pipeline_interface


class GrainDataProcessingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    exit_code = subprocess.call(
        ["bash", "../setup_gcsfuse.sh", "DATASET_GCS_BUCKET=maxtext-dataset", "MOUNT_PATH=/tmp/gcsfuse"]
    )
    if exit_code != 0:
      raise ValueError(f"Running setup_gcsfuse.sh failed with exit code: {exit_code}")

  def setUp(self):
    super().setUp()
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        dataset_type="grain",
        grain_train_files="/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record*",
        tokenizer_path="../assets/tokenizer",
        enable_checkpointing=False,
    )
    self.config = pyconfig.config
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(self.config, self.mesh)
    self.train_iter = self._get_train_iterator()

  def _get_train_iterator(self):
    train_iter, _ = _grain_data_processing.make_grain_iterator(self.config, self.mesh, self.process_indices)
    return train_iter

  def test_train_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    batch = next(self.train_iter)
    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "inputs_position": expected_shape,
            "inputs_segmentation": expected_shape,
            "targets": expected_shape,
            "targets_position": expected_shape,
            "targets_segmentation": expected_shape,
        },
    )

  def test_batch_determinism(self):
    batch1 = next(self.train_iter)
    train_iter = self._get_train_iterator()
    batch2 = next(train_iter)
    self.assertTrue((batch1["inputs"] == batch2["inputs"]).all())
    self.assertTrue((batch1["targets"] == batch2["targets"]).all())
    self.assertTrue((batch1["inputs_segmentation"] == batch2["inputs_segmentation"]).all())
    self.assertTrue((batch1["targets_segmentation"] == batch2["targets_segmentation"]).all())
    self.assertTrue((batch1["inputs_position"] == batch2["inputs_position"]).all())
    self.assertTrue((batch1["targets_position"] == batch2["targets_position"]).all())

  def test_for_loop_repeatable(self):
    def get_first_batch(iterator):
      batch = None
      for batch in iterator:
        break
      return batch

    train_batch1 = get_first_batch(self.train_iter)
    train_batch2 = get_first_batch(self.train_iter)
    self.assertTrue((train_batch1["inputs"] == train_batch2["inputs"]).all())
    self.assertTrue((train_batch1["targets"] == train_batch2["targets"]).all())


if __name__ == "__main__":
  unittest.main()
