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
import collections
import tempfile
import unittest
import json
import numpy as np

import jax
import pytest
import grain.python as grain
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path


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


class GrainArrayRecordProcessingWithMixtureConfigAndElasticIteratorTest(
    _GrainArrayRecordSetup, GrainBaseProcessingTest, unittest.TestCase
):
  """Test grain data processing with mixture config and elastic iterator enabled."""

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
    self.mixture_config_path = os.path.join(temp_dir, "mixture_config_elastic.json")
    with open(self.mixture_config_path, "w", encoding="utf-8") as f:
      json.dump(mixture_config, f)

    self.config = self._make_config(
        grain_train_mixture_config_path=self.mixture_config_path,
        grain_use_elastic_iterator=True,
        packing=False,
        use_truncation=True,
    )


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


class MixtureHostShardingTest(unittest.TestCase):
  """Every dataloading host must see all mixture domains at the configured weights.

  `grain.MapDataset.mix` picks the source for an index with a deterministic
  apportionment pattern that is periodic in `index % P`, where
  `P = sum(p_i) / gcd(p_i)`. Sharding the *mixed* dataset with
  `[host_index::host_count]` therefore restricts host `h` to the pattern phases
  congruent to `h` modulo `gcd(host_count, P)`, which starves hosts of domains
  whenever that gcd exceeds 1 (and gives each host exactly one domain when `P`
  divides `host_count`). Components must be sharded *before* mixing instead.

  These cases are chosen so that `gcd(host_count, P) > 1`; they pass trivially
  for coprime combinations such as 35 domains over 64 hosts.
  """

  DOMAIN_STRIDE = 1_000_000

  def _components(self, num_domains, per_domain=2048):
    """Domain d yields integers whose `value // DOMAIN_STRIDE` is d."""
    return [grain.MapDataset.source([d * self.DOMAIN_STRIDE + i for i in range(per_domain)]) for d in range(num_domains)]

  def _host_domain_counts(self, num_domains, weights, host_index, host_count, take):
    dataset = grain_data_processing._mix_and_finalize(  # pylint: disable=protected-access
        self._components(num_domains),
        list(weights),
        shuffle=False,
        shuffle_seed=0,
        num_epoch=None,
        dataloading_host_index=host_index,
        dataloading_host_count=host_count,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
    )
    counts = collections.Counter()
    for i, element in enumerate(dataset):
      if i >= take:
        break
      counts[element // self.DOMAIN_STRIDE] += 1
    return counts

  def test_every_host_sees_every_domain(self):
    # (num_domains, host_count) pairs where the mix period aliases with the stride.
    for num_domains, host_count in ((2, 2), (3, 3), (4, 2), (4, 4), (8, 8), (2, 8)):
      weights = [1.0 / num_domains] * num_domains
      for host_index in range(host_count):
        with self.subTest(num_domains=num_domains, host_count=host_count, host_index=host_index):
          counts = self._host_domain_counts(num_domains, weights, host_index, host_count, take=400)
          self.assertEqual(
              set(counts),
              set(range(num_domains)),
              f"host {host_index}/{host_count} saw domains {sorted(counts)} "
              f"but expected all {num_domains}; mixture is sharded per-host incorrectly",
          )

  def test_host_domain_proportions_match_weights(self):
    num_domains, host_count, take = 4, 4, 2000
    weights = [0.1, 0.2, 0.3, 0.4]
    for host_index in range(host_count):
      with self.subTest(host_index=host_index):
        counts = self._host_domain_counts(num_domains, weights, host_index, host_count, take=take)
        total = sum(counts.values())
        for domain, expected in enumerate(weights):
          observed = counts[domain] / total
          self.assertAlmostEqual(
              observed,
              expected,
              delta=0.02,
              msg=f"host {host_index} domain {domain}: observed {observed:.3f} vs expected {expected:.3f}",
          )

  def test_hosts_receive_disjoint_elements(self):
    """Sharding before the mix must still partition the data across hosts."""
    num_domains, host_count, take = 4, 4, 500
    weights = [1.0 / num_domains] * num_domains
    seen = []
    for host_index in range(host_count):
      dataset = grain_data_processing._mix_and_finalize(  # pylint: disable=protected-access
          self._components(num_domains),
          list(weights),
          shuffle=False,
          shuffle_seed=0,
          num_epoch=1,
          dataloading_host_index=host_index,
          dataloading_host_count=host_count,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
      )
      elements = set()
      for i, element in enumerate(dataset):
        if i >= take:
          break
        elements.add(element)
      seen.append(elements)

    for a in range(host_count):
      for b in range(a + 1, host_count):
        self.assertEqual(
            seen[a] & seen[b],
            set(),
            f"hosts {a} and {b} received overlapping elements",
        )

  def test_elastic_returns_unsharded_mapdataset(self):
    """ElasticIterator shards internally, so the mixture must be returned unsharded."""
    dataset = grain_data_processing._mix_and_finalize(  # pylint: disable=protected-access
        self._components(4),
        [0.25] * 4,
        shuffle=False,
        shuffle_seed=0,
        num_epoch=None,
        dataloading_host_index=0,
        dataloading_host_count=4,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        elastic=True,
    )
    self.assertIsInstance(dataset, grain.MapDataset)
    counts = collections.Counter(dataset[i] // self.DOMAIN_STRIDE for i in range(400))
    self.assertEqual(set(counts), {0, 1, 2, 3})

  def test_validate_mixture_elastic_compatibility(self):
    """Verify that ElasticIterator validation rejects aliased shard counts and accepts coprime ones."""
    # 2 equal domains: period P = 2
    mock_config_2_domains = type(
        "Config", (), {"grain_train_files": "file1,0.5;file2,0.5", "grain_train_mixture_config_path": None}
    )()
    # shard_count=1 is safe
    grain_data_processing.validate_mixture_elastic_compatibility(mock_config_2_domains, shard_count=1)
    # shard_count=3 is safe (gcd(2, 3) == 1)
    grain_data_processing.validate_mixture_elastic_compatibility(mock_config_2_domains, shard_count=3)
    # shard_count=2 aliases (gcd(2, 2) == 2) -> raises ValueError
    with self.assertRaises(ValueError):
      grain_data_processing.validate_mixture_elastic_compatibility(mock_config_2_domains, shard_count=2)
    # shard_count=64 aliases (gcd(2, 64) == 2) -> raises ValueError
    with self.assertRaises(ValueError):
      grain_data_processing.validate_mixture_elastic_compatibility(mock_config_2_domains, shard_count=64)

    # 35 equal domains: period P = 35
    patterns = ";".join([f"file{i},1.0" for i in range(35)])
    mock_config_35_domains = type(
        "Config", (), {"grain_train_files": patterns, "grain_train_mixture_config_path": None}
    )()
    # shard_count=64 is safe (gcd(35, 64) == 1) -> passes
    grain_data_processing.validate_mixture_elastic_compatibility(mock_config_35_domains, shard_count=64)
    # shard_count=35 aliases (gcd(35, 35) == 35) -> raises ValueError
    with self.assertRaises(ValueError):
      grain_data_processing.validate_mixture_elastic_compatibility(mock_config_35_domains, shard_count=35)


if __name__ == "__main__":
  unittest.main()
