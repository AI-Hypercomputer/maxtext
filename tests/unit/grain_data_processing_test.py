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
import importlib.util
import math
import os.path
import shutil
import tempfile
import unittest
from unittest import mock
import json
import numpy as np

import jax
import pytest
from unittest.mock import patch
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path


class TestTfdsTfrecordPathFallback:
  """Tests TFrecord path construction without reading a dataset."""

  def test_construct_tfds_tfrecord_path(self):
    with patch.object(grain_data_processing.max_logging, "log") as log:
      assert (
          grain_data_processing.construct_tfds_tfrecord_path("  gs://maxtext-dataset/  ", "c4/en:3.0.1", "train")
          == "gs://maxtext-dataset/c4/en/3.0.1/*-train.tfrecord-*"
      )
      log.assert_called_once_with(
          "Automatically constructed Grain TFRecord path from TFDS configuration: "
          "gs://maxtext-dataset/c4/en/3.0.1/*-train.tfrecord-*"
      )

  def test_missing_derived_path_reports_pattern(self, tmp_path):
    pattern = str(tmp_path / "c4/en/3.0.1/*-train.tfrecord-*")

    with pytest.raises(FileNotFoundError, match=r"No files found matching pattern: .*\*-train\.tfrecord-\*"):
      grain_data_processing.find_data_files(pattern)


class GrainBaseProcessingTest:
  """Base mixin with test_train_ds for all grain data processing tests.

  Does not inherit from unittest.TestCase to prevent the test runner from
  discovering and executing it directly. Concrete subclasses must also inherit
  from unittest.TestCase (or a subclass thereof).
  """

  @property
  def train_iter(self):
    cache_key = f"_cached_train_iter_{self.__class__.__name__}"
    if not hasattr(self.__class__, cache_key):
      setattr(
          self.__class__,
          cache_key,
          grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices),
      )
    return getattr(self.__class__, cache_key)

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


class GrainDeterminismMixin:
  """Mixin with determinism tests. Mix into format base classes, not variant subclasses.

  Variant subclasses should inherit only from the setup mixin and GrainBaseProcessingTest
  so that they pick up test_train_ds but not these determinism tests.
  """

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


class _GrainArrayRecordSetup:
  """Private setup mixin for ArrayRecord tests: provides setUp and _make_config.

  No test methods here — inherit this alongside GrainBaseProcessingTest (and
  optionally GrainDeterminismMixin) to compose the desired test surface.
  """

  def setUp(self):
    """Common setup for ArrayRecrd format"""
    super().setUp()
    temp_dir = tempfile.gettempdir()
    decoupled = is_decoupled()

    if decoupled:
      dataset_root = get_test_dataset_path()
      grain_train_files = os.path.join(
          dataset_root,
          "c4",
          "en",
          "3.0.1",
          "c4-train.array_record-00000-of-00008",
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
        max_target_length=128,
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
        "max_target_length": 128,
        **overrides,
    }
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class TestGrainArrayRecordDataSource:
  """Tests ArrayRecord data source construction."""

  @pytest.mark.parametrize(
      ("grain_index_storage_option", "expected_reader_options"),
      [
          (None, None),
          ("in_memory", {"index_storage_option": "in_memory"}),
          ("offloaded", {"index_storage_option": "offloaded"}),
      ],
  )
  def test_index_storage_option_passed_to_arrayrecord_reader(self, grain_index_storage_option, expected_reader_options):
    map_dataset = mock.MagicMock()
    with (
        mock.patch.object(grain_data_processing, "find_data_files", return_value=["data.arrayrecord"]),
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource") as data_source,
        mock.patch.object(grain_data_processing.grain.MapDataset, "source", return_value=map_dataset),
    ):
      grain_data_processing.get_datasets(
          "data.arrayrecord",
          "arrayrecord",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=1,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          grain_index_storage_option=grain_index_storage_option,
          elastic=True,
      )

    data_source.assert_called_once_with(["data.arrayrecord"], reader_options=expected_reader_options)


class GrainArrayRecordProcessingTest(
    _GrainArrayRecordSetup, GrainDeterminismMixin, GrainBaseProcessingTest, unittest.TestCase
):
  """Test grain data processing with ArrayRecord format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path.
  Inherits test_train_ds, test_batch_determinism, and test_for_loop_repeatable.
  Variant subclasses should inherit _GrainArrayRecordSetup + GrainBaseProcessingTest
  directly to get only test_train_ds.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  @pytest.mark.external_serving  # Skipped in decoupled mode due to rocBLAS scratch buffer TF issues on GPU
  def test_batch_determinism(self):
    super().test_batch_determinism()


class GrainArrayRecordProcessingWithMultiSourceBlendingTest(
    _GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase
):

  def setUp(self):
    super().setUp()
    train_files_weighted = ";".join([f"{self.config.grain_train_files},0.3", f"{self.config.grain_train_files},0.7"])
    self.config = self._make_config(grain_train_files=train_files_weighted)


class GrainArrayRecordProcessingWithMixtureConfigTest(_GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase):

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


# TODO(aireenmei): Migrate this test to XLML
@pytest.mark.skip(reason="Flaky test")
class GrainArrayRecordAutoTuneTest(_GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with auto-tuning enabled (grain_worker_count=-1)."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(grain_ram_budget_mb=512, grain_worker_count=-1)  # Enable auto-tuning


class GrainArrayRecordTiktokenTest(_GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with tiktoken tokenizer."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(
        tokenizer_type="tiktoken",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
    )


class GrainArrayRecordHFTokenizerTest(_GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with HuggingFace tokenizer."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(
        tokenizer_type="huggingface",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"),
    )


class GrainArrayRecordBestFitPackingTest(_GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with best_fit packing strategy."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(grain_packing_type="best_fit")


class TestGrainBagzDataSource:
  """Tests Bagz data source construction; runs without the optional bagz package installed."""

  @staticmethod
  def _fake_bagz():
    fake_bagz = mock.MagicMock()
    fake_bagz.LimitsStorage.IN_MEMORY = "IN_MEMORY"
    fake_bagz.LimitsStorage.ON_DISK = "ON_DISK"
    fake_bagz.Reader.Options.return_value.limits_storage = None
    return fake_bagz

  @pytest.mark.parametrize(
      ("grain_index_storage_option", "expected_limits_storage"),
      [(None, None), ("in_memory", "IN_MEMORY"), ("offloaded", "ON_DISK")],
  )
  def test_get_datasets_uses_bagz_reader(self, grain_index_storage_option, expected_limits_storage):
    fake_bagz = self._fake_bagz()
    with (
        mock.patch.dict(sys.modules, {"bagz": fake_bagz}),
        mock.patch.object(grain_data_processing, "find_data_files", return_value=["a.bagz", "b.bagz"]),
        mock.patch.object(grain_data_processing.grain.MapDataset, "source", return_value=mock.MagicMock()) as source,
    ):
      grain_data_processing.get_datasets(
          "*.bagz",
          "bagz",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=1,
          num_epoch=1,
          dataloading_host_index=0,
          dataloading_host_count=1,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          grain_index_storage_option=grain_index_storage_option,
          elastic=True,
      )

    options = fake_bagz.Reader.Options.return_value
    fake_bagz.Reader.assert_called_once_with("a.bagz,b.bagz", options)
    source.assert_called_once_with(fake_bagz.Reader.return_value)
    assert options.limits_storage == expected_limits_storage

  def test_gcs_path_without_bagz_gcs_plugin_raises(self):
    fake_bagz = self._fake_bagz()
    fake_bagz.Reader.side_effect = FileNotFoundError(
        "NOT_FOUND: No file system registered for 'gs://bucket/data-00000-of-00001.bagz'."
    )
    with (
        mock.patch.dict(sys.modules, {"bagz": fake_bagz}),
        pytest.raises(ImportError, match="bagz-gcs"),
    ):
      input_pipeline_utils.make_bagz_data_source(["gs://bucket/data-00000-of-00001.bagz"])

  def test_missing_file_error_is_not_rewritten(self):
    fake_bagz = self._fake_bagz()
    fake_bagz.Reader.side_effect = FileNotFoundError("NOT_FOUND: open: No such file or directory")
    with (
        mock.patch.dict(sys.modules, {"bagz": fake_bagz}),
        pytest.raises(FileNotFoundError, match="No such file"),
    ):
      input_pipeline_utils.make_bagz_data_source(["/data/missing.bagz"])

  def test_comma_in_path_raises(self):
    with (
        mock.patch.dict(sys.modules, {"bagz": self._fake_bagz()}),
        pytest.raises(ValueError, match="must not contain ','"),
    ):
      input_pipeline_utils.make_bagz_data_source(["/data/a,b.bagz"])


def _write_bagz_shards_from_arrayrecord(arrayrecord_path, output_dir, num_shards=2, max_records=2000):
  """Copies the first `max_records` serialized records of an ArrayRecord file into Bagz shards."""
  import bagz  # pylint: disable=import-outside-toplevel

  source = grain_data_processing.grain.ArrayRecordDataSource(arrayrecord_path)
  num_records = min(len(source), max_records)
  per_shard = math.ceil(num_records / num_shards)
  for shard in range(num_shards):
    with bagz.Writer(os.path.join(output_dir, f"c4-train-{shard:05d}-of-{num_shards:05d}.bagz")) as writer:
      for i in range(shard * per_shard, min(num_records, (shard + 1) * per_shard)):
        writer.write(source[i])
  return os.path.join(output_dir, f"c4-train-*-of-{num_shards:05d}.bagz")


class _GrainBagzSetup(_GrainArrayRecordSetup):
  """Private setup mixin for Bagz tests: provides setUp and _make_config.

  Bagz shards are generated from the ArrayRecord test shard, so both formats are exercised with
  identical serialized tf.Example records and no additional test dataset is needed.
  """

  bagz_dir = None
  bagz_pattern = None

  def setUp(self):
    if importlib.util.find_spec("bagz") is None:
      self.skipTest("requires the optional bagz package")
    super().setUp()
    self.arrayrecord_file = self.config.grain_train_files
    cls = type(self)
    if cls.bagz_dir is None:
      cls.bagz_dir = tempfile.mkdtemp(prefix="maxtext_bagz_test_")
      cls.bagz_pattern = _write_bagz_shards_from_arrayrecord(self.arrayrecord_file, cls.bagz_dir)
    self.config = self._make_config(grain_file_type="bagz", grain_train_files=cls.bagz_pattern)

  @classmethod
  def tearDownClass(cls):
    if cls.bagz_dir is not None:
      shutil.rmtree(cls.bagz_dir, ignore_errors=True)
      cls.bagz_dir = None
    super().tearDownClass()


class GrainBagzProcessingTest(_GrainBagzSetup, GrainDeterminismMixin, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with Bagz format.

  Inherits test_train_ds, test_batch_determinism, and test_for_loop_repeatable.
  """

  @pytest.mark.external_serving  # Same as GrainArrayRecordProcessingTest: skipped in decoupled mode.
  def test_batch_determinism(self):
    super().test_batch_determinism()

  def test_bagz_matches_arrayrecord_records(self):
    arrayrecord = grain_data_processing.grain.ArrayRecordDataSource(self.arrayrecord_file)
    # Local glob order is filesystem-dependent; sort so shard 00000 comes first and indices line up.
    bagz_source = input_pipeline_utils.make_bagz_data_source(
        sorted(grain_data_processing.find_data_files(self.config.grain_train_files))
    )
    for i in (0, 1, len(bagz_source) - 1):
      self.assertEqual(bagz_source[i], arrayrecord[i])


class GrainBagzMultiprocessTest(_GrainBagzSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test that bagz.Reader pickles into Grain worker processes."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(grain_file_type="bagz", grain_train_files=self.bagz_pattern, grain_worker_count=2)


class GrainBagzElasticIteratorTest(_GrainBagzSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test Bagz with ElasticIterator and Grain worker processes."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(
        grain_file_type="bagz",
        grain_train_files=self.bagz_pattern,
        grain_use_elastic_iterator=True,
        packing=False,
        grain_worker_count=2,
    )


class _GrainParquetSetup:
  """Private setup mixin for Parquet tests.

  No test methods here — inherit this alongside GrainBaseProcessingTest (and
  optionally GrainDeterminismMixin) to compose the desired test surface.
  """

  def setUp(self):
    """Common setup for Parquet format."""
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

    self.config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
        max_target_length=128,
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
        "grain_file_type": "parquet",
        "grain_train_files": self.config.grain_train_files,
        "grain_worker_count": 1,
        "grain_per_worker_buffer_size": 1,
        "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
        "enable_checkpointing": False,
        "max_target_length": 128,
        **overrides,
    }
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class GrainParquetProcessingTest(_GrainParquetSetup, GrainDeterminismMixin, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with Parquet format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path.
  Inherits test_train_ds, test_batch_determinism, and test_for_loop_repeatable.
  Variant subclasses should inherit _GrainParquetSetup + GrainBaseProcessingTest
  directly to get only test_train_ds.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()


@pytest.mark.external_training
class GrainSFTParquetProcessingTest(_GrainParquetSetup, unittest.TestCase):
  """Tests the SFT pipeline end-to-end using the real ultrachat_200k parquet dataset."""

  def setUp(self):
    super().setUp()
    self.config = self._make_config(
        grain_train_files="gs://maxtext-dataset/hf/ultrachat_200k/train_sft-*.parquet",
        base_output_directory="gs://max-experiments/",
        use_sft=True,
        sft_train_on_completion_only=True,
        train_data_columns=["messages"],
        tokenizer_type="huggingface",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"),
        packing=True,
    )
    self.train_iter = grain_data_processing.make_grain_train_iterator(self.config, self.mesh, self.process_indices)

  def test_train_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    batch = next(self.train_iter)

    # Assert all the required packing and target tensors were generated
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

    # check to see that if prompts are masked, targets will differ from inputs
    has_masked_tokens = np.any(batch["inputs"] != batch["targets"])
    self.assertTrue(bool(has_masked_tokens), "Targets array should differ from inputs array due to prompt masking.")


class _GrainTFRecordSetup:
  """Private setup mixin for TFRecord tests.

  No test methods here — inherit this alongside GrainBaseProcessingTest (and
  optionally GrainDeterminismMixin) to compose the desired test surface.
  """

  def setUp(self):
    """Common setup for TFRecord format"""
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
          "__local_c4_builder-train.tfrecord-00000-of-00008",
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
        max_target_length=128,
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
        "grain_file_type": "tfrecord",
        "grain_train_files": self.config.grain_train_files,
        "grain_worker_count": 1,
        "grain_per_worker_buffer_size": 1,
        "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
        "enable_checkpointing": False,
        "max_target_length": 128,
        **overrides,
    }
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class GrainTFRecordProcessingTest(_GrainTFRecordSetup, GrainDeterminismMixin, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with TFRecord format.

  In decoupled mode, reads directly from GCS. Otherwise, reads from GCSFUSE mounted path.
  Inherits test_train_ds, test_batch_determinism, and test_for_loop_repeatable.
  Variant subclasses should inherit _GrainTFRecordSetup + GrainBaseProcessingTest
  directly to get only test_train_ds.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_config_accepts_tfds_tfrecord_fallback(self):
    config = self._make_config(
        grain_train_files="",
        dataset_path="gs://maxtext-dataset",
        dataset_name="c4/en:3.0.1",
        train_split="train",
        eval_interval=0,
    )
    self.assertEqual(config.grain_train_files, "")


class GrainTFRecordPreTokenizedProcessingTest(_GrainTFRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Test grain data processing with a pre-tokenized TFRecord dataset (tokenize_train_data=False).

  Uses c4/en/3.0.5 validation_tokenized_5662seqs split, which stores token IDs
  in an 'ids' int64 column rather than raw text.
  """

  def setUp(self):
    super().setUp()
    base = get_test_dataset_path() if is_decoupled() else os.path.join(tempfile.gettempdir(), "gcsfuse")
    grain_train_file = os.path.join(base, "c4", "en", "3.0.5", "c4-validation_tokenized_5662seqs.tfrecord-00000-of-00001")
    self.config = self._make_config(
        grain_train_files=grain_train_file,
        tokenize_train_data=False,
        train_data_columns=["ids"],
        packing=False,
    )


class GrainFewerFilesThanHostsTest(_GrainTFRecordSetup, GrainBaseProcessingTest, unittest.TestCase):
  """Tests data loading when file count < dataloading_host_count.

  _GrainTFRecordSetup provides a single TFRecord file. Overriding process_indices
  to [0, 1] makes make_grain_train_iterator use dataloading_host_count=2, simulating
  the undersized scenario without a real multi-host runner. The inherited test_train_ds
  then validates the full batch shape through this code path.
  """

  def setUp(self):
    super().setUp()
    # Simulate 2 dataloading hosts with only 1 file to trigger the
    # fewer-files-than-hosts path in get_datasets via make_grain_train_iterator.
    self.process_indices = [0, 1]

  def test_raises_when_grain_worker_count_exceeds_files_per_host(self):
    # 1 file, 2 hosts (via process_indices=[0,1]) → files_per_host=1;
    # grain_worker_count=2 must raise.
    config = self._make_config(grain_worker_count=2)
    with self.assertRaises(ValueError):
      grain_data_processing.make_grain_train_iterator(config, self.mesh, self.process_indices)


if __name__ == "__main__":
  unittest.main()
