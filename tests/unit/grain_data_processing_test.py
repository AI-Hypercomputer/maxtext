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

import sys
import os.path
import tempfile
import unittest
import json
import ml_collections
import numpy as np

import jax
import pytest
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from unittest import mock
import grain.python as grain

from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path


class GrainBaseProcessingTest:
  """Base mixin with shared test methods for all grain data processing tests.

  Does not inherit from unittest.TestCase to prevent the test runner from
  discovering and executing it directly. Concrete subclasses must also inherit
  from unittest.TestCase (or a subclass thereof).
  """

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
    train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)
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


class GrainArrayRecordProcessingTest(GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with ArrayRecord format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    decoupled = is_decoupled()

    if decoupled:
      dataset_root = get_test_dataset_path()
      grain_train_files = os.path.join(
          dataset_root,
          "c4",
          "array-record",
          "en",
          "3.0.1",
          "c4-train.array_record-00000-of-01024",
      )
      base_output_directory = get_test_base_output_directory()
    else:
      grain_train_files = os.path.join(
          temp_dir,
          "gcsfuse",
          "array-record",
          "c4",
          "en",
          "3.0.1",
          "c4-train.array_record-00000-of-01024",
      )
      base_output_directory = "gs://max-experiments/"

    config_file = get_test_config_path()
    self.config = pyconfig.initialize(
        [sys.argv[0], config_file],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=base_output_directory,
        dataset_type="grain",
        grain_train_files=grain_train_files,
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
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
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

  def _make_config(self, **overrides):
    """Re-initialize config with base params, applying any overrides."""
    kwargs = {
        "per_device_batch_size": 1,
        "run_name": "test",
        "mesh_axes": ["data"],
        "logical_axis_rules": [["batch", "data"]],
        "data_sharding": ["data"],
        "base_output_directory": self.config.base_output_directory,
        "dataset_type": "grain",
        "grain_train_files": self.config.grain_train_files,
        "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
        "enable_checkpointing": False,
        **overrides,
    }
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)

  @pytest.mark.external_serving  # Skipped in decoupled mode due to rocBLAS scratch buffer TF issues on GPU
  def test_batch_determinism(self):
    super().test_batch_determinism()


class GrainArrayRecordProcessingWithMultiSourceBlendingTest(GrainArrayRecordProcessingTest):

  def setUp(self):
    super().setUp()
    train_files_weighted = ";".join([f"{self.config.grain_train_files},0.3", f"{self.config.grain_train_files},0.7"])
    self.config = self._make_config(grain_train_files=train_files_weighted)
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainArrayRecordProcessingWithMixtureConfigTest(GrainArrayRecordProcessingTest):

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    decoupled = is_decoupled()

    if decoupled:
      dataset_root = get_test_dataset_path()
      mixture_config = {
          "ds1": {
              "path": os.path.join(
                  dataset_root,
                  "c4",
                  "en",
                  "3.0.1",
                  "c4-train.array_record-*",
              ),
              "weight": 0.3,
          },
          "ds2": {
              "path": os.path.join(
                  dataset_root,
                  "c4",
                  "en",
                  "3.0.1",
                  "c4-train.array_record-*",
              ),
              "weight": 0.7,
          },
      }
    else:
      mixture_config = {
          "ds1": {
              "path": f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0000*",
              "weight": 0.3,
          },
          "ds2": {
              "path": f"{temp_dir}/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-0001*",
              "weight": 0.7,
          },
      }
    self.mixture_config_path = os.path.join(temp_dir, "mixture_config.json")
    with open(self.mixture_config_path, "w", encoding="utf-8") as f:
      json.dump(mixture_config, f)

    self.config = self._make_config(grain_train_mixture_config_path=self.mixture_config_path)
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


# TODO(aireenmei): Migrate this test to XLML
@pytest.mark.skip(reason="Flaky test")
class GrainArrayRecordAutoTuneTest(GrainArrayRecordProcessingTest):
  """Test grain data processing with auto-tuning enabled (grain_worker_count=-1)."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(grain_ram_budget_mb=512, grain_worker_count=-1)  # Enable auto-tuning
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

  @pytest.mark.skip(
      reason=(
          "Auto-tuning tries multiple numbers of workers during the first few batches "
          "and it affects batch determinism at first."
      )
  )
  def test_batch_determinism(self):
    super().test_batch_determinism()

  @pytest.mark.skip(reason="Flaky test - see b/475255774.")
  def test_for_loop_repeatable(self):
    super().test_for_loop_repeatable()


class GrainArrayRecordBestFitPackingTest(GrainArrayRecordProcessingTest):
  """Test grain data processing with best_fit packing strategy."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(grain_packing_type="best_fit")
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainParquetProcessingTest(GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with Parquet format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    decoupled = is_decoupled()

    if decoupled:
      dataset_root = get_test_dataset_path()
      grain_train_file = os.path.join(
          dataset_root,
          "hf",
          "c4",
          "c4-train-00000-of-01637.parquet",
      )
      base_output_directory = get_test_base_output_directory()
    else:
      grain_train_file = os.path.join(
          temp_dir,
          "gcsfuse",
          "hf",
          "c4",
          "c4-train-00000-of-01637.parquet",
      )
      base_output_directory = "gs://max-experiments/"

    config_file = get_test_config_path()
    self.config = pyconfig.initialize(
        [sys.argv[0], config_file],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=base_output_directory,
        dataset_type="grain",
        grain_file_type="parquet",
        grain_train_files=grain_train_file,
        grain_worker_count=1,
        grain_per_worker_buffer_size=1,
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
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
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainTFRecordProcessingTest(GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with TFRecord format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.gettempdir()
    decoupled = is_decoupled()

    if decoupled:
      dataset_root = get_test_dataset_path()
      grain_train_file = os.path.join(
          dataset_root,
          "c4",
          "en",
          "3.0.1",
          "c4-train.tfrecord-00000-of-01024",
      )
      base_output_directory = get_test_base_output_directory()
    else:
      grain_train_file = os.path.join(
          temp_dir,
          "gcsfuse",
          "c4",
          "en",
          "3.0.1",
          "c4-train.tfrecord-00000-of-01024",
      )
      base_output_directory = "gs://max-experiments/"

    config_file = get_test_config_path()
    self.config = pyconfig.initialize(
        [sys.argv[0], config_file],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=base_output_directory,
        dataset_type="grain",
        grain_file_type="tfrecord",
        grain_train_files=grain_train_file,
        grain_worker_count=1,
        grain_per_worker_buffer_size=1,
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
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
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)


class GrainSFTPipelineTest(unittest.TestCase):
  """Tests the full SFT preprocessing pipeline end-to-end using dummy data."""

  def setUp(self):
    super().setUp()
    # Create a minimal config to satisfy the pipeline's requirements
    self.config = ml_collections.ConfigDict(
        {
            "grain_file_type": "in_memory",  # Skips arrayrecord parsing
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
            "tokenizer_type": "sentencepiece",
            "add_bos": True,
            "add_eos": True,
            "hf_access_token": "",
            "use_truncation": False,
            "max_target_length": 16,
            "sft_train_on_completion_only": True,
            "packing": False,
            "global_batch_size_to_load": 2,  # Using 2 examples
            "expansion_factor_real_data": 1.0,
            "grain_ram_budget_mb": 512,
            # A very basic chat template for testing purposes
            "chat_template": (
                "{% for message in messages %}" "{{ message['role'] + ': ' + message['content'] + ' ' }}" "{% endfor %}"
            ),
        }
    )

  @mock.patch("maxtext.input_pipeline.input_pipeline_utils.apply_chat_template")
  def test_sft_preprocessing_pipeline(self, mock_apply_chat_template):
    # Fake the exact chunked structure and is_prompt array that MaxText expects
    def fake_apply_chat(element, tokenizer_model, data_column_name):
      # Return hardcoded strings instead of referencing the overwritten dictionary
      element[data_column_name] = ["What is 2+2? ", "It is 4."]
      element["is_prompt"] = [True, False]
      return element

    mock_apply_chat_template.side_effect = fake_apply_chat

    # Create a dummy in-memory dataset
    dummy_data = [{"prompt": "What is 2+2?", "completion": "It is 4."}, {"prompt": "Say hello.", "completion": "Hello!"}]
    dataset = grain.MapDataset.source(dummy_data)
    dataset = dataset.to_iter_dataset()
    data_columns = ["prompt", "completion"]

    # Run pipeline
    pipeline_iterator = grain_data_processing.sft_preprocessing_pipeline(
        dataset=dataset,
        config=self.config,
        data_columns=data_columns,
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    # Get the first batch
    iterator = iter(pipeline_iterator)
    batch = next(iterator)

    # Assert the pipeline output is correct
    self.assertIn("inputs", batch)
    self.assertIn("targets", batch)

    expected_shape = (2, self.config.max_target_length)

    # Check shapes
    self.assertEqual(batch["inputs"].shape, expected_shape)
    self.assertEqual(batch["targets"].shape, expected_shape)

    # Check for masked tokens
    has_masked_tokens = np.any((batch["targets"] == 0) | (batch["targets"] == -1))
    self.assertTrue(has_masked_tokens, "Targets array should contain masked (ignore) IDs for the prompt sections.")


if __name__ == "__main__":
  unittest.main()
