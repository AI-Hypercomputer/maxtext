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

import sys
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import unittest

import pyconfig
from input_pipeline import _hf_data_processing
from input_pipeline import input_pipeline_interface


class HfDataProcessingTest(unittest.TestCase):

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
        hf_path="parquet",
        hf_data_dir="",
        hf_data_files="gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet",
        tokenizer_path="google-t5/t5-large",
        enable_checkpointing=False,
    )
    self.config = pyconfig.config
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)

    self.train_ds = self._get_datasets()
    self.train_iter = self._get_preprocessed_datasets()

  def _get_datasets(self):
    train_ds, _ = _hf_data_processing.get_datasets(config=self.config)
    return train_ds

  def _get_preprocessed_datasets(self):
    process_indices = input_pipeline_interface.get_process_loading_real_data(self.config, self.mesh)
    print("Sharding dataset in ", len(process_indices), " shards")
    train_iter, _, _ = _hf_data_processing.preprocess_dataset(
        self.config,
        process_indices.index(jax.process_index()),
        len(process_indices),
        self.mesh,
        self.train_ds,
    )
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
    self.train_ds = self._get_datasets()
    train_iter = self._get_preprocessed_datasets()
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
