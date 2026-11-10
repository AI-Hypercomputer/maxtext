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

import functools
import os
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import json
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import grain.python as grain
import pytest
from unittest.mock import patch
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import multihost_dataloading
from maxtext.utils import elastic_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path
from tests.utils.test_helpers import get_test_local_output_directory


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


class MakeElasticIteratorLogicalBatchSizeTest(unittest.TestCase):
  """Tests logical-to-physical batch padding and state checkpointing in `_make_elastic_iterator`."""

  def _dataset(self):
    return (
        grain.MapDataset.range(1000)
        .map(lambda i: {"inputs": np.full((4,), i + 1, np.int32), "inputs_segmentation": np.ones((4,), np.int32)})
        .repeat(None)
    )

  @staticmethod
  def _cfg(physical, logical):
    return SimpleNamespace(
        global_batch_size_to_load=physical,
        logical_global_batch_size_to_load=logical,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
    )

  @staticmethod
  def _read_step(shards):
    """Reads one batch from each shard and separates real from zero-segmentation rows."""
    batches = [next(it) for it in shards]
    rows = [(int(r[0]), bool(np.any(s != 0))) for b in batches for r, s in zip(b["inputs"], b["inputs_segmentation"])]
    return batches, sorted(val for val, is_real in rows if is_real), sum(not is_real for _, is_real in rows)

  def test_non_divisible_colocated_shards_and_resize_state_round_trip(self):
    # Simulates 4 x v5e-32 (32 hosts, L=P=128) <-> 3 x v5e-32 (24 hosts, L=128, P=192; 128 % 24 == 8 != 0).
    def make_shards(physical, count):
      make_it = functools.partial(
          grain_data_processing._make_elastic_iterator,  # pylint: disable=protected-access
          self._dataset(),
          self._cfg(physical, 128),
          lambda dataset: dataset,
          shard_count=count,
      )
      return [iter(make_it(shard_index=h)) for h in range(count)]

    shards_24, shards_32 = make_shards(192, 24), make_shards(128, 32)
    self.addCleanup(lambda: [it.close() for it in shards_24 + shards_32])

    for step in range(2):
      batches, real_ids, zero_rows = self._read_step(shards_24)
      self.assertTrue(all(b["inputs"].shape == (8, 4) for b in batches))
      self.assertEqual((real_ids, zero_rows), (list(range(step * 128 + 1, (step + 1) * 128 + 1)), 64))
      self.assertTrue(all(it.get_state() == {"global_next_index": (step + 1) * 128} for it in shards_24))

    # Scale up to 32 hosts (plain `ElasticIterator` fast path), then scale down to 24 hosts + non-multiple offset.
    for it in shards_32:
      it.set_state(shards_24[0].get_state())
    _, real_ids_32, zero_rows_32 = self._read_step(shards_32)
    self.assertEqual((real_ids_32, zero_rows_32), (list(range(257, 385)), 0))
    self.assertEqual(shards_32[0].get_state(), {"global_next_index": 384})

    for start_index in (384, 10):
      for it in shards_24:
        it.set_state({"global_next_index": start_index})
      _, real_ids, zero_rows = self._read_step(shards_24)
      self.assertEqual((real_ids, zero_rows), (list(range(start_index + 1, start_index + 129)), 64))
      self.assertEqual(shards_24[0].get_state(), {"global_next_index": start_index + 128})

  def test_remote_iterator_with_non_divisible_colocated_shards(self):
    preprocessing_fn = functools.partial(
        grain_data_processing._make_elastic_iterator,  # pylint: disable=protected-access
        config=self._cfg(18, 16),
        preprocessing_fn=lambda dataset: dataset,
    )
    devices = [mock.Mock(process_index=p) for p in (1, 2, 3, 1, 2, 3)]
    remote_iterators = []
    for local_process in (1, 2, 3):
      remote_it = multihost_dataloading.RemoteIterator(
          get_ds_fn=lambda dataloading_host_index, dataloading_host_count: self._dataset(),
          preprocessing_fn=preprocessing_fn,
          global_shape=(18, 4),
          checkpoint_path="",
          elastic=True,
      )
      sharding = SimpleNamespace(
          device_set=set(devices), addressable_devices={d for d in devices if d.process_index == local_process}
      )
      remote_it.build(SimpleNamespace(sharding=sharding))
      remote_iterators.append(remote_it)
    self.addCleanup(lambda: [it.close() for it in remote_iterators])

    batches, real_ids, zero_rows = self._read_step([it.iterator for it in remote_iterators])
    self.assertTrue(all(b["inputs"].shape == (6, 4) for b in batches))
    self.assertEqual((real_ids, zero_rows), (list(range(1, 17)), 2))
    self.assertTrue(all(it.iterator.get_state() == {"global_next_index": 16} for it in remote_iterators))

  def test_make_grain_train_iterator_end_to_end_constant_padded_mode(self):
    live_devices = list(jax.devices())
    num_live = len(live_devices)
    if num_live < 2:
      if not sys.executable:
        self.skipTest("Needs >= 2 devices.")
      self.assertNotEqual(os.environ.get("_MAXTEXT_CONSTANT_PADDED_SUBPROC"), "1")
      env = {
          **os.environ,
          "_MAXTEXT_CONSTANT_PADDED_SUBPROC": "1",
          "JAX_PLATFORMS": "cpu",
          "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip(),
          "PYTHONPATH": os.pathsep.join(filter(None, [*sys.path, os.environ.get("PYTHONPATH", "")])),
      }
      cmd = [sys.executable, os.path.abspath(__file__), f"{type(self).__name__}.{self._testMethodName}"]
      res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
      self.assertEqual(res.returncode, 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
      return

    ar_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets/local_datasets/c4_en_dataset_minimal/c4/en/3.0.1/c4-train.array_record-00000-of-00008",
    )
    all_devices = live_devices + [mock.MagicMock(slice_index=99)]
    with (
        mock.patch("jax.devices", return_value=all_devices),
        mock.patch.object(elastic_utils, "ensure_elastic_manager_initialized"),
        mock.patch.object(elastic_utils, "live_devices", return_value=live_devices),
    ):
      config = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_constant_padded_grain",
          base_output_directory=get_test_local_output_directory(),
          enable_checkpointing=False,
          enable_single_controller=True,
          elastic_enabled=True,
          elastic_global_batch_size_mode="constant_padded",
          per_device_batch_size=1.0,
          dataset_type="grain",
          grain_file_type="arrayrecord",
          grain_train_files=ar_path,
          grain_use_elastic_iterator=True,
          grain_worker_count=0,
          enable_data_shuffling=False,
          packing=False,
          tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
          max_target_length=16,
          mesh_axes=["data"],
          logical_axis_rules=[["batch", "data"]],
          data_sharding=["data"],
      )

    self.assertGreater(config.global_batch_size_to_load, config.logical_global_batch_size_to_load)
    mesh = Mesh(mesh_utils.create_device_mesh((num_live,), live_devices), config.mesh_axes)
    process_indices = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        mesh,
    )
    with mock.patch.object(elastic_utils, "live_devices", return_value=live_devices):
      train_iter = grain_data_processing.make_grain_train_iterator(config, mesh, process_indices)
    self.addCleanup(train_iter.close)

    batch = next(train_iter)
    self.assertEqual(batch["inputs"].shape, (config.global_batch_size_to_load, config.max_target_length))
    in_zero = jnp.all(batch["inputs_segmentation"] == 0, axis=-1)
    tgt_zero = jnp.all(batch["targets_segmentation"] == 0, axis=-1)
    self.assertEqual(int(jnp.sum(~in_zero)), config.logical_global_batch_size_to_load)
    self.assertEqual(int(jnp.sum(in_zero)), config.global_batch_size_to_load - config.logical_global_batch_size_to_load)
    self.assertTrue(bool(jnp.array_equal(in_zero, tgt_zero)))
    self.assertEqual(
        train_iter.local_iterator.get_state(), {"global_next_index": config.logical_global_batch_size_to_load}
    )


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


def _extract_file_instructions(**kwargs):
  return kwargs["file_instructions"]


class ArrayRecordFileInstructionsTest(parameterized.TestCase):
  """Tests colocated-host-sharded ArrayRecord metadata resolution and caching."""

  # pylint: disable=protected-access

  def setUp(self):
    super().setUp()
    grain_data_processing._clear_arrayrecord_metadata_cache()
    self.addCleanup(grain_data_processing._clear_arrayrecord_metadata_cache)

  @staticmethod
  def _resolve_on_hosts(
      data_file_pattern,
      shard_count=1,
      mixture_config_path=None,
      hf_access_token=None,
      max_workers=2,
  ):
    """Simulates colocated hosts counting and exchanging shard counts."""
    controller_loader = grain_data_processing._ColocatedArrayRecordDatasetLoader(
        _extract_file_instructions,
        data_file_pattern,
        mixture_config_path=mixture_config_path,
        hf_access_token=hf_access_token,
        max_workers=max_workers,
    )
    host_loaders = [pickle.loads(pickle.dumps(controller_loader)) for _ in range(shard_count)]
    per_host = [loader.count_local_shards(i, shard_count) for i, loader in enumerate(host_loaders)]
    num_files, _, digest, _ = per_host[0]
    if any((nf, dg) != (num_files, digest) for nf, _, dg, _ in per_host):
      raise ValueError(f"Mismatched shard metadata across hosts: {per_host}")
    combined = [0] * num_files
    for host, (_, _, _, local_counts) in enumerate(per_host):
      combined[host::shard_count] = local_counts
    for loader in host_loaders:
      loader.set_shard_counts(combined)
    return [loader(dataloading_host_index=i, dataloading_host_count=shard_count) for i, loader in enumerate(host_loaders)]

  def test_datasource_uses_file_instructions_without_opening_files(self):
    instructions = [
        grain_data_processing._FileInstruction("/no/such/file-0", 0, 3, 3),
        grain_data_processing._FileInstruction("/no/such/file-1", 0, 5, 5),
    ]
    self.assertLen(grain.ArrayRecordDataSource(instructions), 8)

  def test_caches_counts_and_matches_direct_read_across_shards(self):
    pattern = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets/local_datasets/c4_en_dataset_minimal/c4/en/3.0.1/c4-train.array_record-0000[01]-of-00008",
    )
    shard_paths = sorted(grain_data_processing.find_data_files(pattern))
    self.assertLen(shard_paths, 2)
    shard0_len = len(grain.ArrayRecordDataSource(shard_paths[0]))
    shard1_len = len(grain.ArrayRecordDataSource(shard_paths[1]))
    self.assertGreater(shard0_len, 0)
    self.assertGreater(shard1_len, 0)

    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", wraps=grain_data_processing.find_data_files
        ) as find_files,
        mock.patch.object(
            grain_data_processing.grain, "ArrayRecordDataSource", wraps=grain.ArrayRecordDataSource
        ) as data_source,
    ):
      (first,) = self._resolve_on_hosts(pattern, shard_count=1)
      (second,) = self._resolve_on_hosts(pattern, shard_count=1)

    self.assertEqual(second, first)
    find_files.assert_called_once_with(pattern, hf_access_token=None)
    # Each of the two shards is opened once across both resolutions; removing `_ARRAYRECORD_RECORD_COUNTS` fails here.
    self.assertCountEqual([call.args[0] for call in data_source.call_args_list], shard_paths)

    ds_kwargs = {
        "shuffle": False,
        "shuffle_seed": 0,
        "shuffle_buffer_size": 1,
        "num_epoch": 1,
        "dataloading_host_index": 0,
        "dataloading_host_count": 1,
        "grain_worker_count": 0,
        "grain_num_threads": 1,
        "grain_prefetch_buffer_size": 1,
        "grain_data_source_max_workers": 1,
        "elastic": True,
    }
    direct_ds = grain_data_processing.get_datasets(pattern, "arrayrecord", **ds_kwargs)
    instructed_ds = grain_data_processing.get_datasets(pattern, "arrayrecord", file_instructions=first, **ds_kwargs)
    self.assertLen(instructed_ds, shard0_len + shard1_len)
    self.assertLen(direct_ds, shard0_len + shard1_len)
    # Compare every record across both shards, explicitly crossing `shard0_len - 1 -> shard0_len`.
    self.assertEqual([instructed_ds[i] for i in range(len(direct_ds))], [direct_ds[i] for i in range(len(direct_ds))])

  def test_sharded_counting_across_hosts_and_resize_cache_reuse(self):
    counts_by_file = {"s0": 2, "s1": 3, "s2": 5, "s3": 7, "s4": 11}
    opened = []

    def fake_datasource(arg, **_kwargs):
      opened.append(arg)
      return [None] * counts_by_file[arg]

    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", return_value=["s4", "s1", "s3", "s0", "s2"]
        ) as find_files,
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", side_effect=fake_datasource),
    ):
      per_host = self._resolve_on_hosts("pat", shard_count=3)
      # Each of the 5 shards was opened once in total across the 3 hosts (2 + 2 + 1).
      self.assertCountEqual(opened, list(counts_by_file))
      expected = {
          "pat": tuple(
              grain_data_processing._FileInstruction(f, 0, counts_by_file[f], counts_by_file[f])
              for f in sorted(counts_by_file)
          )
      }
      self.assertEqual(per_host, [expected, expected, expected])

      # Simulate an elastic resize to 2 surviving hosts in the same sidecar process: 0 additional opens or globs!
      opened.clear()
      find_files.reset_mock()
      resized = self._resolve_on_hosts("pat", shard_count=2)
      self.assertEqual(opened, [])
      find_files.assert_not_called()
      self.assertEqual(resized, [expected, expected])

    with self.assertRaisesRegex(ValueError, "Expected pattern and weight pairs"):
      grain_data_processing._parse_arrayrecord_patterns("a,0.3;malformed")

  def test_reuses_grain_data_source_max_workers(self):
    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", side_effect=lambda p, hf_access_token=None: [f"{p}0", f"{p}1"]
        ),
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", return_value=[None] * 4),
        mock.patch.object(
            grain_data_processing.futures,
            "ThreadPoolExecutor",
            wraps=grain_data_processing.futures.ThreadPoolExecutor,
        ) as executor_cls,
    ):
      (resolved,) = self._resolve_on_hosts("a,0.3;b,0.7", shard_count=1, max_workers=7)

    executor_cls.assert_called_once_with(max_workers=7)
    self.assertEqual(
        resolved,
        {
            "a": (
                grain_data_processing._FileInstruction("a0", 0, 4, 4),
                grain_data_processing._FileInstruction("a1", 0, 4, 4),
            ),
            "b": (
                grain_data_processing._FileInstruction("b0", 0, 4, 4),
                grain_data_processing._FileInstruction("b1", 0, 4, 4),
            ),
        },
    )

  def test_controller_does_not_access_files_or_mixture_json(self):
    base_cfg = {
        "global_batch_size_to_load": 4,
        "global_batch_size_to_load_eval": 4,
        "max_target_length": 16,
        "grain_file_type": "arrayrecord",
        "grain_train_files": "",
        "grain_eval_files": "/worker/only/eval.array_record*",
        "grain_train_mixture_config_path": "/worker/only/mixture.json",
        "enable_data_shuffling": False,
        "data_shuffle_seed": 0,
        "grain_shuffle_buffer_size": 1,
        "num_epoch": 1,
        "grain_worker_count": 0,
        "grain_worker_count_eval": 0,
        "grain_num_threads": 1,
        "grain_num_threads_eval": 1,
        "grain_prefetch_buffer_size": 1,
        "grain_prefetch_buffer_size_eval": 1,
        "grain_per_worker_buffer_size": 1,
        "grain_per_worker_buffer_size_eval": 1,
        "grain_data_source_max_workers": 6,
        "grain_use_elastic_iterator": False,
        "hf_access_token": "hf_tok",
        "mmap_npy_split": "",
        "train_data_columns": ("text",),
        "eval_data_columns": ("text",),
        "tokenize_train_data": False,
        "tokenize_eval_data": False,
        "use_sft": False,
        "use_multimodal": False,
        "colocated_python_data_input": True,
        "checkpoint_dir": "",
    }
    mesh = SimpleNamespace(size=4)
    with (
        mock.patch.object(grain_data_processing, "_get_pipeline_fn", return_value=lambda **_kw: object()),
        mock.patch.object(grain_data_processing, "find_data_files", side_effect=AssertionError("Controller globbed")),
        mock.patch.object(
            grain_data_processing.grain, "ArrayRecordDataSource", side_effect=AssertionError("Controller opened shard")
        ),
        mock.patch("builtins.open", side_effect=AssertionError("Controller read worker mixture JSON")),
        mock.patch.object(grain_data_processing.multihost_dataloading, "RemoteIteratorWrapper") as wrapper_cls,
    ):
      grain_data_processing.make_grain_train_iterator(SimpleNamespace(**base_cfg), mesh, [0])
      grain_data_processing.make_grain_eval_iterator(SimpleNamespace(**base_cfg), mesh, [0])
      parquet_cfg = {**base_cfg, "grain_file_type": "parquet", "hf_path": ""}
      grain_data_processing.make_grain_train_iterator(SimpleNamespace(**parquet_cfg), mesh, [0])

    self.assertEqual(wrapper_cls.call_count, 3)
    train_call, eval_call, parquet_call = wrapper_cls.call_args_list
    self.assertIsInstance(
        pickle.loads(pickle.dumps(train_call.args[0])),
        grain_data_processing._ColocatedArrayRecordDatasetLoader,
    )
    self.assertTrue(train_call.kwargs["count_shards"])
    self.assertIsInstance(
        pickle.loads(pickle.dumps(eval_call.args[0])),
        grain_data_processing._ColocatedArrayRecordDatasetLoader,
    )
    self.assertTrue(eval_call.kwargs["count_shards"])
    self.assertNotIsInstance(parquet_call.args[0], grain_data_processing._ColocatedArrayRecordDatasetLoader)
    self.assertFalse(parquet_call.kwargs["count_shards"])

  def test_get_datasets_uses_file_instructions_without_listing_files(self):
    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", side_effect=lambda p, hf_access_token=None: [f"{p}0"]
        ),
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", return_value=[None] * 4),
    ):
      (file_instructions,) = self._resolve_on_hosts("a,0.3;b,0.7", shard_count=1)

    with (
        mock.patch.object(grain_data_processing, "find_data_files", side_effect=AssertionError("Unexpected glob")),
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource") as data_source,
        mock.patch.object(grain_data_processing.grain.MapDataset, "source"),
        mock.patch.object(grain_data_processing.grain.IterDataset, "mix"),
    ):
      grain_data_processing.get_datasets(
          "a,0.3;b,0.7",
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
          file_instructions=file_instructions,
      )

    data_source.assert_has_calls(
        [
            mock.call(file_instructions["a"], reader_options=None),
            mock.call(file_instructions["b"], reader_options=None),
        ],
        any_order=True,
    )

  def test_get_datasets_raises_key_error_on_missing_file_instruction_pattern(self):
    with (
        mock.patch.object(grain_data_processing, "find_data_files", side_effect=AssertionError("Unexpected glob")),
        self.assertRaises(KeyError),
    ):
      grain_data_processing.get_datasets(
          "missing_pattern",
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
          file_instructions={"other_pattern": ()},
      )

  @parameterized.named_parameters(
      ("fewer_files_than_hosts", "a", 4, {"a": (grain_data_processing._FileInstruction("a0", 0, 3, 3),)}),
      ("empty_pattern", "", 2, None),
      ("none_pattern", None, 2, None),
  )
  def test_fewer_files_than_hosts_and_empty_pattern(self, pattern, shard_count, expected):
    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", side_effect=lambda p, hf_access_token=None: [f"{p}0"]
        ),
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", return_value=[None] * 3),
    ):
      resolved = self._resolve_on_hosts(pattern, shard_count=shard_count)
    self.assertEqual(resolved, [expected] * shard_count)

  @parameterized.named_parameters(
      ("single_pattern", "a", None, ["a"]),
      ("semicolon_mixture_with_token", "a,0.3;b,0.7", "hf_test_token", ["a", "b"]),
  )
  def test_resolves_patterns_with_token(self, pattern, hf_token, expected_keys):
    with (
        mock.patch.object(
            grain_data_processing, "find_data_files", side_effect=lambda p, hf_access_token=None: [f"{p}0"]
        ) as find_files,
        mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", return_value=[None] * 3),
    ):
      (resolved,) = self._resolve_on_hosts(pattern, shard_count=1, hf_access_token=hf_token)

    self.assertEqual(
        resolved,
        {key: (grain_data_processing._FileInstruction(f"{key}0", 0, 3, 3),) for key in expected_keys},
    )
    find_files.assert_has_calls([mock.call(key, hf_access_token=hf_token) for key in expected_keys], any_order=True)

  def test_resolves_mixture_json_on_hosts(self):
    with tempfile.TemporaryDirectory() as directory:
      mixture_config_path = os.path.join(directory, "mixture.json")
      with open(mixture_config_path, "w", encoding="utf-8") as f:
        json.dump({"x": {"path": "c", "weight": 1}, "y": {"path": "d", "weight": 2}}, f)
      with (
          mock.patch.object(
              grain_data_processing, "find_data_files", side_effect=lambda p, hf_access_token=None: [f"{p}0"]
          ),
          mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource", return_value=[None] * 2),
      ):
        resolved_per_host = self._resolve_on_hosts("", shard_count=2, mixture_config_path=mixture_config_path)

      resolved = resolved_per_host[0]
      self.assertEqual(
          resolved,
          {
              "c": (grain_data_processing._FileInstruction("c0", 0, 2, 2),),
              "d": (grain_data_processing._FileInstruction("d0", 0, 2, 2),),
          },
      )
      with (
          mock.patch.object(grain_data_processing, "find_data_files", side_effect=AssertionError("Unexpected glob")),
          mock.patch.object(grain_data_processing.grain, "ArrayRecordDataSource") as data_source,
          mock.patch.object(grain_data_processing.grain.MapDataset, "source"),
          mock.patch.object(grain_data_processing.grain.IterDataset, "mix"),
      ):
        grain_data_processing.get_datasets(
            "",
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
            mixture_config_path=mixture_config_path,
            file_instructions=resolved,
        )
      data_source.assert_has_calls(
          [mock.call(resolved["c"], reader_options=None), mock.call(resolved["d"], reader_options=None)],
          any_order=True,
      )


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


class _PadEqualsEosTokenizer:
  """Hugging Face tokenizer stand-in with DeepSeek's pad == eos == 1 and `<｜▁pad▁｜>` at id 2."""

  _VOCAB = {"<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": 1, "<｜▁pad▁｜>": 2}

  def __init__(self):
    self.chat_template = None
    self.bos_token_id = 0
    self.eos_token_id = 1
    self.unk_token_id = None
    self.pad_token = "<｜end▁of▁sentence｜>"

  @property
  def pad_token_id(self):
    return self._VOCAB[self.pad_token]

  def get_vocab(self):
    return dict(self._VOCAB)


class GrainSftPadIdTest(unittest.TestCase):
  """The grain SFT pipeline uses the remapped pad id for prompt masking, padding and the tokenizer."""

  def test_sft_pipeline_uses_remapped_pad_id(self):
    dataset = mock.MagicMock()
    dataset.map.return_value = dataset
    base_tokenizer = SimpleNamespace(tokenizer=_PadEqualsEosTokenizer(), pad_id=1)
    config = SimpleNamespace(sft_train_on_completion_only=True, max_target_length=8, chat_template=None)

    with (
        mock.patch.object(grain_data_processing.data_processing_utils, "parse_and_keep_features", return_value=dataset),
        mock.patch.object(
            grain_data_processing.data_processing_utils, "get_tokenizer_and_pad_id", return_value=(base_tokenizer, 1)
        ),
        mock.patch.object(grain_data_processing.data_processing_utils, "validate_and_configure_sft_columns"),
        mock.patch.object(grain_data_processing.data_processing_utils, "get_local_batch_size", return_value=1),
        mock.patch.object(grain_data_processing.data_processing_utils, "format_and_batch", return_value=dataset) as fmt,
        mock.patch.object(
            grain_data_processing.data_processing_utils, "apply_multiprocessing_and_prefetch", return_value=dataset
        ),
    ):
      grain_data_processing.sft_preprocessing_pipeline(
          dataset, config, ["messages"], tokenize=True, grain_worker_count=0, grain_per_worker_buffer_size=1
      )

    masking = [
        c.args[0] for c in dataset.map.call_args_list if isinstance(c.args[0], input_pipeline_utils.SFTPromptMasking)
    ]
    self.assertEqual(len(masking), 1)
    self.assertEqual(masking[0].unk_id, 2)
    self.assertEqual(fmt.call_args.args[3], 2)
    self.assertEqual(base_tokenizer.pad_id, 2)


class GrainElasticDataReplicaTest(unittest.TestCase):
  """ElasticIterator with several data replicas per process (num_data_replicas_per_process > 1)."""

  def test_each_replica_gets_its_own_batch(self):
    num_replicas, replica_batch, steps = 2, 4, 3
    batches = {}
    for replica in range(num_replicas):
      config = SimpleNamespace(
          global_batch_size_to_load=replica_batch,
          logical_global_batch_size_to_load=None,
          num_data_replicas_per_process=num_replicas,
          data_replica_index=replica,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
      )
      # The shard arguments make_grain_train_iterator passes on one loading process.
      with mock.patch.object(input_pipeline_utils.jax, "process_index", return_value=0):
        shard_index, shard_count = input_pipeline_utils.get_dataloading_shard(config, [0])
      iterator = iter(
          grain_data_processing._make_elastic_iterator(  # pylint: disable=protected-access
              grain.MapDataset.range(64),
              config,
              lambda dataset: dataset,
              shard_index=shard_index,
              shard_count=shard_count,
          )
      )
      batches[replica] = [next(iterator).tolist() for _ in range(steps)]
    for replica in range(num_replicas):
      self.assertEqual([len(b) for b in batches[replica]], [replica_batch] * steps)
    # Together the replicas read each step's global batch of all replicas, without overlap.
    for step in range(steps):
      rows = batches[0][step] + batches[1][step]
      self.assertEqual(sorted(rows), list(range(step * 8, (step + 1) * 8)))


if __name__ == "__main__":
  absltest.main()
