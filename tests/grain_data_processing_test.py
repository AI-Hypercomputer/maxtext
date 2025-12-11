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

"""Tests for grain data processing."""

import subprocess
import sys
import os.path
import tempfile
import unittest
import json

import jax
import pytest
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from MaxText import pyconfig
from MaxText.input_pipeline import _grain_data_processing
from MaxText.input_pipeline import input_pipeline_interface
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT, MAXTEXT_REPO_ROOT


class GrainArrayRecordProcessingTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mount_gcsfuse()

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    self.config = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      per_device_batch_size=1,
      run_name="test",
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
      base_output_directory="gs://max-experiments/",
      dataset_type="grain",
      grain_train_files=os.path.join(temp_dir, "gcsfuse", "array-record", "c4", "en", "3.0.1", "c4-train.array_record*"),
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer"),
      enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
      self.config.data_sharding,
      self.config.global_batch_size_to_load,
      self.config.global_batch_size_to_train_on,
      self.config.max_target_length,
      self.mesh,
    )
    self.train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

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
    train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)
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


class GrainArrayRecordProcessingWithMultiSourceBlendingTest(GrainArrayRecordProcessingTest):
  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    # We use the same dataset for testing, but you can use different datasets by changing the file patterns.
    grain_train_files = [
      f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0000*,0.3",
      f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0001*,0.7",
    ]
    grain_train_files = ";".join(grain_train_files)
    self.config = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      per_device_batch_size=1,
      run_name="test",
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
      base_output_directory="gs://max-experiments/",
      dataset_type="grain",
      grain_train_files=grain_train_files,
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer"),
      enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
      self.config.data_sharding,
      self.config.global_batch_size_to_load,
      self.config.global_batch_size_to_train_on,
      self.config.max_target_length,
      self.mesh,
    )
    self.train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainArrayRecordProcessingWithMixtureConfigTest(GrainArrayRecordProcessingTest):
  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    mixture_config = {
      "ds1": {"path": f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0000*", "weight": 0.3},
      "ds2": {"path": f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0001*", "weight": 0.7},
    }
    self.mixture_config_path = os.path.join(temp_dir, "mixture_config.json")
    with open(self.mixture_config_path, "w", encoding="utf-8") as f:
      json.dump(mixture_config, f)

    self.config = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      per_device_batch_size=1,
      run_name="test",
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
      base_output_directory="gs://max-experiments/",
      dataset_type="grain",
      grain_train_mixture_config_path=self.mixture_config_path,
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer"),
      enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
      self.config.data_sharding,
      self.config.global_batch_size_to_load,
      self.config.global_batch_size_to_train_on,
      self.config.max_target_length,
      self.mesh,
    )
    self.train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainArrayRecordAutoTuneTest(GrainArrayRecordProcessingTest):
  """Test grain data processing with auto-tuning enabled (grain_worker_count=-1)."""

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    self.config = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      per_device_batch_size=1,
      run_name="test",
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
      base_output_directory="gs://max-experiments/",
      dataset_type="grain",
      grain_ram_budget_mb=512,
      grain_train_files=os.path.join(temp_dir, "gcsfuse", "array-record", "c4", "en", "3.0.1", "c4-train.array_record*"),
      grain_worker_count=-1,  # Enable auto-tuning
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer"),
      enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
      self.config.data_sharding,
      self.config.global_batch_size_to_load,
      self.config.global_batch_size_to_train_on,
      self.config.max_target_length,
      self.mesh,
    )
    self.train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

  @pytest.mark.skip(
    reason=(
      "Auto-tuning tries multiple numbers of workers during the first few batches "
      "and it affects batch determinism at first."
    )
  )
  def test_batch_determinism(self):
    super().test_batch_determinism()


class GrainParquetProcessingTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mount_gcsfuse()

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    self.config = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      per_device_batch_size=1,
      run_name="test",
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
      base_output_directory="gs://max-experiments/",
      dataset_type="grain",
      grain_file_type="parquet",
      grain_train_files=os.path.join(temp_dir, "gcsfuse", "hf", "c4", "c4-train-00000-of-01637.parquet"),
      grain_worker_count=1,
      grain_per_worker_buffer_size=1,
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer"),
      enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
      self.config.data_sharding,
      self.config.global_batch_size_to_load,
      self.config.global_batch_size_to_train_on,
      self.config.max_target_length,
      self.mesh,
    )
    self.train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

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
    train_iter = _grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)
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
    self.assertTrue((train_batch1["inputs"] == train_batch2["inputs"]).all())  # pytype: disable=unsupported-operands
    self.assertTrue((train_batch1["targets"] == train_batch2["targets"]).all())  # pytype: disable=unsupported-operands


def mount_gcsfuse():
  """
  Mounts a GCS bucket (gs://maxtext-dataset) to a local directory (/tmp/gcsfuse)
  using gcsfuse if not already mounted.
  """
  temp_dir = tempfile.gettempdir()
  mount_path = os.path.join(temp_dir, "gcsfuse")

  # Only mount if the directory is empty or not present
  if not os.path.isdir(mount_path) or not os.listdir(mount_path):
    script_path = os.path.join(MAXTEXT_REPO_ROOT, "setup_gcsfuse.sh")
    if not os.path.isfile(script_path):
      raise FileNotFoundError(script_path)

    exit_code = subprocess.call(
      ["bash", script_path, "DATASET_GCS_BUCKET=maxtext-dataset", f"MOUNT_PATH={os.path.join(temp_dir, 'gcsfuse')}"]
    )
    if exit_code != os.EX_OK:
      raise ValueError(f"Running setup_gcsfuse.sh failed with exit code: {exit_code}")


if __name__ == "__main__":
  mount_gcsfuse()
  unittest.main()
