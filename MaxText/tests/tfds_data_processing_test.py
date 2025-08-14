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

# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
import unittest

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import tensorflow as tf
import tensorflow_datasets as tfds

from MaxText import pyconfig
from MaxText.globals import PKG_DIR
from MaxText.input_pipeline import _tfds_data_processing
from MaxText.input_pipeline import input_pipeline_interface
from MaxText.max_utils import gcs_bucket_accessible


class TfdsDataProcessingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        tokenizer_path=os.path.join(os.path.dirname(PKG_DIR), "assets", "tokenizer"),
        enable_checkpointing=False,
        eval_interval=10,
    )
    os.environ["TFDS_DATA_DIR"] = config.dataset_path
    self.config = config
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )
    self.read_config = tfds.ReadConfig(
        shuffle_seed=self.config.data_shuffle_seed,
    )
    self.read_config.add_tfds_id = True
    if not gcs_bucket_accessible("max-experiments"):
      return
    self.train_ds = self._get_datasets()
    self.train_iter = _tfds_data_processing.make_tfds_train_iterator(self.config, self.mesh, self.process_indices)
    self.eval_iter = _tfds_data_processing.make_tfds_eval_iterator(self.config, self.mesh, self.process_indices)

  def _get_datasets(self):
    ds_builder = tfds.builder(self.config.dataset_name)
    self.read_config.input_context = tf.distribute.InputContext(
        input_pipeline_id=jax.process_index(),
        num_input_pipelines=jax.process_count(),
    )
    ds = ds_builder.as_dataset(split="train", read_config=self.read_config, shuffle_files=self.config.enable_data_shuffling)

    return ds

  @unittest.skipIf(not gcs_bucket_accessible("max-experiments"), "gs://max-experiments bucket not accessible")
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

  @unittest.skipIf(not gcs_bucket_accessible("max-experiments"), "gs://max-experiments bucket not accessible")
  def test_ds_determinism(self):
    train_ds1 = self.train_ds.batch(64)
    train_ds1 = next(train_ds1.as_numpy_iterator())
    # reset the dataset loading
    train_ds = self._get_datasets()
    train_ds = train_ds.batch(64)
    train_ds2 = next(train_ds.as_numpy_iterator())

    self.assertCountEqual(train_ds1["tfds_id"], train_ds2["tfds_id"])

  @unittest.skipIf(not gcs_bucket_accessible("max-experiments"), "gs://max-experiments bucket not accessible")
  def test_batch_determinism(self):
    batch1 = next(self.train_iter)
    train_iter = _tfds_data_processing.make_tfds_train_iterator(self.config, self.mesh, self.process_indices)
    batch2 = next(train_iter)
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs"], batch2["inputs"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets"], batch2["targets"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs_segmentation"], batch2["inputs_segmentation"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets_segmentation"], batch2["targets_segmentation"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["inputs_position"], batch2["inputs_position"])))
    self.assertTrue(tf.reduce_all(tf.equal(batch1["targets_position"], batch2["targets_position"])))

  @unittest.skipIf(not gcs_bucket_accessible("max-experiments"), "gs://max-experiments bucket not accessible")
  def test_for_loop_repeatable(self):
    def get_first_batch(iterator):
      batch = None
      for batch in iterator:
        break
      return batch

    eval_batch1 = get_first_batch(self.eval_iter)
    eval_batch2 = get_first_batch(self.eval_iter)
    self.assertTrue((eval_batch1["inputs"] == eval_batch2["inputs"]).all())
    self.assertTrue((eval_batch1["targets"] == eval_batch2["targets"]).all())


if __name__ == "__main__":
  unittest.main()
