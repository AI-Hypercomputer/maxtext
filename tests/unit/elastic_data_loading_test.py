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

"""Tests for elastic data loading."""

import sys
from unittest import mock

from absl.testing import absltest
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import multihost_dataloading
from maxtext.utils import elastic_utils
from maxtext.utils import maxtext_utils  # pylint: disable=unused-import
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pathwaysutils  # pylint: disable=unused-import


class FakeDevice:
  """Fake Device object."""

  def __init__(
      self,
      device_id,
      slice_index=0,
      process_index=0,
      task_id=0,
      device_kind="tpu",
  ):
    self.id = device_id
    self.slice_index = slice_index
    self.process_index = process_index
    self.task_id = task_id
    self.device_kind = device_kind

  def __repr__(self):
    return (
        f"FakeDevice(device_id={self.id}, slice={self.slice_index},"
        f" process={self.process_index}, kind={self.device_kind})"
    )


class FakeRemoteIterator:
  """Fake Remote Iterator for testing."""

  def __init__(
      self,
      get_ds_fn,
      preprocessing_fn,
      global_shape,
      checkpoint_path,
      elastic=False,
  ):
    self.get_ds_fn = get_ds_fn
    self.preprocessing_fn = preprocessing_fn
    self.global_shape = global_shape
    self.checkpoint_path = checkpoint_path
    self.elastic = elastic
    self.reset()

  def reset(self):
    ds = self.get_ds_fn(
        dataloading_host_index=multihost_dataloading.jax.process_index(),
        dataloading_host_count=multihost_dataloading.jax.process_count(),
    )
    dataloader = self.preprocessing_fn(ds)
    self.iterator = dataloader.as_numpy_iterator()


class ElasticDataLoadingTest(absltest.TestCase):
  """Tests for elastic data loading."""

  def setUp(self):
    super().setUp()
    # Save global state
    self.original_elastic_manager = elastic_utils.elastic_manager

  def tearDown(self):
    elastic_utils.elastic_manager = self.original_elastic_manager
    super().tearDown()

  @mock.patch(
      "maxtext.src.maxtext.input_pipeline.multihost_dataloading.RemoteIterator",
      FakeRemoteIterator,
  )
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing._get_pipeline_fn")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.ElasticIterator")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.get_datasets")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.jax")
  @mock.patch("maxtext.src.maxtext.input_pipeline.input_pipeline_interface.jax")
  @mock.patch("maxtext.src.maxtext.configs.types.jax")
  @mock.patch("maxtext.src.maxtext.utils.elastic_utils.jax")
  @mock.patch("maxtext.src.maxtext.utils.maxtext_utils.jax")
  @mock.patch("maxtext.src.maxtext.utils.elastic_utils.pathwaysutils")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading.jax")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading._colocated_cpu_devices")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading._colocated_cpu_mesh")
  def test_elastic_data_loading_active_processes(
      self,
      mock_colocated_cpu_mesh,
      mock_colocated_cpu_devices,
      mock_multihost_jax,
      mock_pathwaysutils,
      mock_utils_jax,
      mock_elastic_jax,
      mock_types_jax,
      mock_interface_jax,
      mock_grain_jax,
      mock_get_datasets,
      mock_elastic_iterator,
      mock_get_pipeline_fn,
  ):

    # Setup 2 slices, 2 processes per slice, 1 device per process.
    # Slice 0 (active): process 0, 1
    # Slice 1 (inactive): process 2, 3
    devices = [
        FakeDevice(device_id=0, slice_index=0, process_index=0, task_id=0),
        FakeDevice(device_id=1, slice_index=0, process_index=1, task_id=1),
        FakeDevice(device_id=2, slice_index=1, process_index=2, task_id=2),
        FakeDevice(device_id=3, slice_index=1, process_index=3, task_id=3),
    ]

    # Mock JAX in different modules
    for mock_jax in [
        mock_grain_jax,
        mock_interface_jax,
        mock_types_jax,
        mock_elastic_jax,
        mock_utils_jax,
        mock_multihost_jax,
    ]:

      mock_jax.devices.return_value = devices
      mock_jax.process_count.return_value = 4
      # We simulate running on process 0
      mock_jax.process_index.return_value = 0
      mock_jax.local_devices.return_value = [devices[0]]

    # Mock pathwaysutils
    mock_pathwaysutils.is_pathways_backend_used.return_value = True

    # Mock elastic manager
    fake_manager = mock.MagicMock()
    fake_manager.active_slice_indices = {0}
    fake_manager.slice_to_devices = {
        0: [devices[0], devices[1]],
        1: [devices[2], devices[3]],
    }
    fake_manager.all_slice_indices = {0, 1}
    fake_manager.total_slice_count = 2
    elastic_utils.elastic_manager = fake_manager

    # Mock get_datasets to avoid GCS access
    mock_get_datasets.return_value = (mock.MagicMock(), mock.MagicMock())

    # Mock _get_pipeline_fn to return a dummy function
    mock_get_pipeline_fn.return_value = lambda dataset, *args, **kwargs: dataset

    # Mock ElasticIterator to return a dummy dataloader
    mock_dataloader = mock.MagicMock()
    mock_iterator = mock.MagicMock()
    mock_dataloader.as_numpy_iterator.return_value = mock_iterator
    mock_elastic_iterator.return_value = mock_dataloader

    # Mock NamedSharding.devices_indices_map
    mock_sharding_instance = mock_interface_jax.sharding.NamedSharding.return_value
    mock_sharding_instance.devices_indices_map.return_value = {
        devices[0]: (slice(0, 1), slice(0, 128)),
        devices[1]: (slice(1, 2), slice(0, 128)),
    }

    # Setup config
    # We need to initialize config with elastic_enabled=True
    # We use a dummy argv
    argv = [
        sys.argv[0],
        get_test_config_path(),
        "elastic_enabled=True",
        "per_device_batch_size=1",
        "dataset_type=grain",
        "grain_train_files=gs://dummy/train",
        "run_name=test",
        "colocated_python_data_input=True",  # to trigger RemoteIterator
        "grain_use_elastic_iterator=True",
        "enable_single_controller=True",
        "packing=False",
        "mesh_axes=['data']",
        "logical_axis_rules=[['batch', 'data']]",
        "data_sharding=['data']",
    ]

    # We need to mock jax in pyconfig as well because it calls
    # maybe_initialize_jax_distributed_system
    with mock.patch("maxtext.src.maxtext.configs.pyconfig.jax") as mock_pyconfig_jax:
      mock_pyconfig_jax.process_count.return_value = 4
      mock_pyconfig_jax.process_index.return_value = 0
      config = pyconfig.initialize(argv)

    # Create CPU devices and mesh for colocated python
    cpu_devices = [
        FakeDevice(
            device_id=10,
            slice_index=0,
            process_index=0,
            task_id=0,
            device_kind="cpu",
        ),
        FakeDevice(
            device_id=11,
            slice_index=0,
            process_index=1,
            task_id=1,
            device_kind="cpu",
        ),
    ]
    cpu_mesh = Mesh(np.array(cpu_devices), config.mesh_axes)
    mock_colocated_cpu_devices.return_value = [cpu_devices[0]]
    mock_colocated_cpu_mesh.return_value = cpu_mesh

    # Create mesh
    # Mesh has 2 devices (active ones)
    mesh_devices = np.array([devices[0], devices[1]])
    mesh = Mesh(mesh_devices, config.mesh_axes)

    # Call get_process_loading_real_data directly to verify it
    process_indices_train = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        mesh,
    )
    self.assertIn(0, process_indices_train)
    self.assertEqual(process_indices_train, [0, 1])

    # Call create_data_iterator
    # This should not raise ValueError
    _, _ = input_pipeline_interface.create_data_iterator(config, mesh)

    # Verify that ElasticIterator was called with shard_count=2 and
    # shard_index=0
    mock_elastic_iterator.assert_called_once()
    _, called_kwargs = mock_elastic_iterator.call_args
    shard_options = called_kwargs.get("shard_options")
    self.assertEqual(shard_options.shard_index, 0)
    self.assertEqual(shard_options.shard_count, 2)

  @mock.patch(
      "maxtext.src.maxtext.input_pipeline.multihost_dataloading.RemoteIterator",
      FakeRemoteIterator,
  )
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing._get_pipeline_fn")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.ElasticIterator")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.get_datasets")
  @mock.patch("maxtext.src.maxtext.input_pipeline.grain_data_processing.jax")
  @mock.patch("maxtext.src.maxtext.input_pipeline.input_pipeline_interface.jax")
  @mock.patch("maxtext.src.maxtext.configs.types.jax")
  @mock.patch("maxtext.src.maxtext.utils.elastic_utils.jax")
  @mock.patch("maxtext.src.maxtext.utils.maxtext_utils.jax")
  @mock.patch("maxtext.src.maxtext.utils.elastic_utils.pathwaysutils")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading.jax")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading._colocated_cpu_devices")
  @mock.patch("maxtext.src.maxtext.input_pipeline.multihost_dataloading._colocated_cpu_mesh")
  def test_elastic_data_loading_with_expansion_factor(
      self,
      mock_colocated_cpu_mesh,
      mock_colocated_cpu_devices,
      mock_multihost_jax,
      mock_pathwaysutils,
      mock_utils_jax,
      mock_elastic_jax,
      mock_types_jax,
      mock_interface_jax,
      mock_grain_jax,
      mock_get_datasets,
      mock_elastic_iterator,
      mock_get_pipeline_fn,
  ):

    # Setup 2 slices, 2 processes per slice, 1 device per process.
    devices = [
        FakeDevice(device_id=0, slice_index=0, process_index=0, task_id=0),
        FakeDevice(device_id=1, slice_index=0, process_index=1, task_id=1),
        FakeDevice(device_id=2, slice_index=1, process_index=2, task_id=2),
        FakeDevice(device_id=3, slice_index=1, process_index=3, task_id=3),
    ]

    # Mock JAX in different modules
    for mock_jax in [
        mock_grain_jax,
        mock_interface_jax,
        mock_types_jax,
        mock_elastic_jax,
        mock_utils_jax,
        mock_multihost_jax,
    ]:
      mock_jax.devices.return_value = devices
      mock_jax.process_count.return_value = 4
      mock_jax.process_index.return_value = 0
      mock_jax.local_devices.return_value = [devices[0]]

    # Mock pathwaysutils
    mock_pathwaysutils.is_pathways_backend_used.return_value = True

    # Mock elastic manager
    fake_manager = mock.MagicMock()
    fake_manager.active_slice_indices = {0}
    fake_manager.slice_to_devices = {
        0: [devices[0], devices[1]],
        1: [devices[2], devices[3]],
    }
    fake_manager.all_slice_indices = {0, 1}
    fake_manager.total_slice_count = 2
    elastic_utils.elastic_manager = fake_manager

    # Mock get_datasets to avoid GCS access
    mock_get_datasets.return_value = (mock.MagicMock(), mock.MagicMock())

    # Mock _get_pipeline_fn to return a dummy function
    mock_get_pipeline_fn.return_value = lambda dataset, *args, **kwargs: dataset

    # Mock ElasticIterator to return a dummy dataloader
    mock_dataloader = mock.MagicMock()
    mock_iterator = mock.MagicMock()
    mock_dataloader.as_numpy_iterator.return_value = mock_iterator
    mock_elastic_iterator.return_value = mock_dataloader

    # Mock NamedSharding.devices_indices_map for expansion_factor=2
    # Global batch size to load will be 4.
    mock_sharding_instance = mock_interface_jax.sharding.NamedSharding.return_value
    mock_sharding_instance.devices_indices_map.return_value = {
        devices[0]: (slice(0, 2), slice(0, 128)),
        devices[1]: (slice(2, 4), slice(0, 128)),
    }

    # Setup config with expansion_factor_real_data=2
    argv = [
        sys.argv[0],
        get_test_config_path(),
        "elastic_enabled=True",
        "per_device_batch_size=1",
        "dataset_type=grain",
        "grain_train_files=gs://dummy/train",
        "run_name=test",
        "colocated_python_data_input=True",
        "grain_use_elastic_iterator=True",
        "enable_single_controller=True",
        "packing=False",
        "mesh_axes=['data']",
        "logical_axis_rules=[['batch', 'data']]",
        "data_sharding=['data']",
        "expansion_factor_real_data=2",
    ]

    with mock.patch("maxtext.src.maxtext.configs.pyconfig.jax") as mock_pyconfig_jax:
      mock_pyconfig_jax.process_count.return_value = 4
      mock_pyconfig_jax.process_index.return_value = 0
      config = pyconfig.initialize(argv)

    # Create CPU devices and mesh for colocated python
    cpu_devices = [
        FakeDevice(
            device_id=10,
            slice_index=0,
            process_index=0,
            task_id=0,
            device_kind="cpu",
        ),
        FakeDevice(
            device_id=11,
            slice_index=0,
            process_index=1,
            task_id=1,
            device_kind="cpu",
        ),
    ]
    cpu_mesh = Mesh(np.array(cpu_devices), config.mesh_axes)
    mock_colocated_cpu_devices.return_value = [cpu_devices[0]]
    mock_colocated_cpu_mesh.return_value = cpu_mesh

    # Create mesh
    mesh_devices = np.array([devices[0], devices[1]])
    mesh = Mesh(mesh_devices, config.mesh_axes)

    # Call get_process_loading_real_data directly to verify it
    process_indices_train = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        mesh,
    )
    self.assertEqual(process_indices_train, [0])

    # Call create_data_iterator
    _, _ = input_pipeline_interface.create_data_iterator(config, mesh)

    # Verify that ElasticIterator was called with shard_count=1 and
    # shard_index=0
    mock_elastic_iterator.assert_called_once()
    _, called_kwargs = mock_elastic_iterator.call_args
    shard_options = called_kwargs.get("shard_options")
    self.assertEqual(shard_options.shard_index, 0)
    self.assertEqual(shard_options.shard_count, 1)


if __name__ == "__main__":

  absltest.main()
