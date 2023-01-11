# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys
import jax
from jax.experimental.maps import Mesh
from jax.experimental import mesh_utils

import unittest
import tensorflow_datasets as tfds

import pyconfig
import input_pipeline

# By default, XLA presents all the CPU cores as one device. This flag splits up cores in 2 CPU devices.
os.environ["TFDS_DATA_DIR"] = "gs://tensorflow-datasets/datasets"
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
jax.config.update('jax_platform_name', 'cpu')


class InputPipelineTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pyconfig.initialize(sys.argv + ['configs/base.yml', 'run_name=test'], per_device_batch_size=1)
    self.config = pyconfig.config
    self.read_config = tfds.ReadConfig()
    self.read_config.add_tfds_id = True
    self.train_ds, self.eval_ds = self._get_datasets()
    self.train_iter, self.eval_iter, self.predict_iter = self._get_preprocessed_datasets()

  def _get_datasets(self):
    train_ds, eval_ds = input_pipeline.get_datasets(
            config=self.config, read_config = self.read_config)
    return train_ds, eval_ds

  def _get_preprocessed_datasets(self):
    mesh_shape_1d = (len(jax.devices()),)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), self.config.mesh_axes)

    train_iter, eval_iter, test_iter, _ = input_pipeline.preprocess_dataset(
              self.config,
              mesh,
              self.train_ds, self.eval_ds, vocab_path=self.config.vocab_path)
    return train_iter, eval_iter, test_iter

  def test_train_ds(self):
    expected_shape = [2, self.config.max_target_length]
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    batch = next(self.train_iter)
    self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
        'inputs': expected_shape,
        'inputs_position': expected_shape,
        'inputs_segmentation': expected_shape,
        'targets': expected_shape,
        'targets_position': expected_shape,
        'targets_segmentation': expected_shape,
    })


  def test_eval_ds(self):
    expected_shape = [2, self.config.max_eval_target_length]
    batch = next(self.eval_iter)
    self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
       'inputs': expected_shape,
       'targets': expected_shape,
    })


  def test_predict_ds(self):
    expected_shape = [2, self.config.max_predict_length]
    batch = next(self.predict_iter)
    self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
        'inputs': expected_shape,
        'targets': expected_shape,
    })


  def test_ds_determinism(self):
    train_ds1 = self.train_ds.batch(64)
    train_ds1 = next(train_ds1.as_numpy_iterator())
    # reset the dataset loading
    train_ds, _ = self._get_datasets()
    train_ds = train_ds.batch(64)
    train_ds2 = next(train_ds.as_numpy_iterator())

    self.assertCountEqual(train_ds1['tfds_id'], train_ds2['tfds_id'])


if __name__ == '__main__':
  unittest.main()

