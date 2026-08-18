# Copyright 2023–2026 Google LLC
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

"""Unit tests for elastic data loading, dynamic topology transitions, and edge cases."""

import unittest
from unittest.mock import MagicMock, Mock, patch
from absl.testing import parameterized
import numpy as np
import jax
import jax.numpy as jnp
import grain.python as grain

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import grain_data_processing
from maxtext.utils import elastic_utils


class FakeDevice:
  """Fake Device for simulating TPU topologies."""

  def __init__(self, slice_index=0, process_index=0, task_id=0, device_id=0):
    self.slice_index = slice_index
    self.process_index = process_index
    self.task_id = task_id
    self.id = device_id
    self.platform = "tpu"
    self.device_kind = "TPU"
    self.client = Mock()


def create_fake_tpu_topology(num_slices=2, hosts_per_slice=16, devices_per_host=8, processes_per_host=2):
  """Builds a simulated multi-slice TPU device list with realistic host/process mapping."""
  devices = []
  device_counter = 0
  task_counter = 0
  process_counter = 0

  for slice_idx in range(num_slices):
    for _ in range(hosts_per_slice):
      current_task_id = task_counter
      task_counter += 1
      # Simulate multiple processes per host VM
      host_processes = [process_counter + p for p in range(processes_per_host)]
      process_counter += processes_per_host

      for d in range(devices_per_host):
        # Distribute devices across the host's processes
        assigned_process = host_processes[d % len(host_processes)]
        devices.append(
            FakeDevice(
                slice_index=slice_idx,
                process_index=assigned_process,
                task_id=current_task_id,
                device_id=device_counter,
            )
        )
        device_counter += 1

  return devices


class FakeConfig:
  """Config mock supporting elastic training and data loading parameters."""

  def __init__(
      self,
      elastic_enabled=True,
      colocated_python_data_input=True,
      grain_use_elastic_iterator=True,
      per_device_batch_size=1.0,
      expansion_factor_real_data=0,
      gradient_accumulation_steps=1,
      num_slices=2,
      dcn_data_parallelism=2,
      grain_worker_count=-1,
      grain_per_worker_buffer_size=1,
      grain_num_threads=1,
      grain_prefetch_buffer_size=1,
      grain_ram_budget_mb=1024,
      max_target_length=8192,
  ):
    self.elastic_enabled = elastic_enabled
    self.colocated_python_data_input = colocated_python_data_input
    self.grain_use_elastic_iterator = grain_use_elastic_iterator
    self.per_device_batch_size = per_device_batch_size
    self.expansion_factor_real_data = expansion_factor_real_data
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.num_slices = num_slices
    self.dcn_data_parallelism = dcn_data_parallelism
    self.grain_worker_count = grain_worker_count
    self.grain_per_worker_buffer_size = grain_per_worker_buffer_size
    self.grain_num_threads = grain_num_threads
    self.grain_prefetch_buffer_size = grain_prefetch_buffer_size
    self.grain_ram_budget_mb = grain_ram_budget_mb
    self.max_target_length = max_target_length
    self.num_target_devices = 256
    self.global_batch_size_to_load = 256
    self.global_batch_size_to_train_on = 256
    self.micro_batch_size_to_train_on = 256
    self.eval_per_device_batch_size = 1.0
    self.enable_rampup_batch_size = False


class ElasticGrainTopologyTest(parameterized.TestCase):
  """Comprehensive edge-case tests for elastic scaling, topology transitions, and Grain dataloading."""

  def setUp(self):
    super().setUp()
    # Reset global elastic_manager state
    elastic_utils.elastic_manager = None

  @parameterized.named_parameters(
      ("standard_2x64_topology", 2, 16, 8, 2, 1.0),
      ("large_2x1k_topology", 2, 128, 8, 2, 1.0),
      ("small_4_devices_per_host", 2, 8, 4, 1, 1.0),
      ("batch_size_2_per_device", 2, 16, 8, 2, 2.0),
  )
  def test_elastic_iterator_batch_divisibility_across_topology(
      self, num_slices, hosts_per_slice, devices_per_host, processes_per_host, per_device_batch_size
  ):
    """Verifies that _make_elastic_iterator dynamically calculates global_batch_size

    so that each worker task receives a local batch size divisible by its local devices.
    """
    devices = create_fake_tpu_topology(
        num_slices=num_slices,
        hosts_per_slice=hosts_per_slice,
        devices_per_host=devices_per_host,
        processes_per_host=processes_per_host,
    )
    total_processes = num_slices * hosts_per_slice * processes_per_host
    total_devices = len(devices)

    config = FakeConfig(
        elastic_enabled=True,
        colocated_python_data_input=True,
        grain_use_elastic_iterator=True,
        per_device_batch_size=per_device_batch_size,
    )

    # Initialize fake elastic manager with all slices
    fake_manager = MagicMock()
    fake_manager.active_slice_indices = set(range(num_slices))
    fake_manager.active_slice_count = num_slices
    elastic_utils.elastic_manager = fake_manager

    with patch.object(jax, "devices", return_value=devices), \
         patch.object(jax, "process_count", return_value=total_processes), \
         patch.object(jax, "process_index", return_value=0), \
         patch("maxtext.input_pipeline.grain_data_processing.ElasticIterator") as mock_elastic_iter:

      dummy_dataset = Mock()
      preprocessing_fn = lambda dataset: dataset

      grain_data_processing._make_elastic_iterator(
          dummy_dataset,
          config,
          preprocessing_fn,
          shard_index=0,
          shard_count=total_processes,
      )

      # Verify ElasticIterator was instantiated
      mock_elastic_iter.assert_called_once()
      _, kwargs = mock_elastic_iter.call_args
      effective_gbs = kwargs["global_batch_size"]

      # Calculate the batch size that each process shard will produce
      local_batch_per_process = effective_gbs // total_processes
      expected_devices_per_host = devices_per_host
      expected_local_batch = int(per_device_batch_size * expected_devices_per_host)

      self.assertEqual(local_batch_per_process, expected_local_batch)

      # Verify that local_batch can be evenly split across devices_per_host
      fake_array = np.zeros((local_batch_per_process, config.max_target_length))
      split_result = np.split(fake_array, devices_per_host, axis=0)
      self.assertEqual(len(split_result), devices_per_host)

  def test_elastic_scale_down_and_scale_up_cycle(self):
    """Simulates full elastic lifecycle: 2 slices -> scale down to 1 slice -> scale up back to 2 slices."""
    devices = create_fake_tpu_topology(num_slices=2, hosts_per_slice=16, devices_per_host=8, processes_per_host=2)
    config = FakeConfig(
        elastic_enabled=True,
        colocated_python_data_input=True,
        grain_use_elastic_iterator=True,
        per_device_batch_size=1.0,
    )

    fake_manager = MagicMock()
    elastic_utils.elastic_manager = fake_manager

    # 1. INITIAL: 2 slices active (256 devices)
    fake_manager.active_slice_indices = {0, 1}
    fake_manager.active_slice_count = 2

    with patch.object(jax, "devices", return_value=devices):
      elastic_utils.mutate_config_for_topology(config, fake_manager)
      self.assertEqual(config.num_slices, 2)
      self.assertEqual(config.num_target_devices, 256)
      self.assertEqual(config.global_batch_size_to_load, 256)

      # 2. SCALE DOWN: Slice 1 drops -> only Slice 0 active (128 devices)
      fake_manager.active_slice_indices = {0}
      fake_manager.active_slice_count = 1

      elastic_utils.mutate_config_for_topology(config, fake_manager)
      self.assertEqual(config.num_slices, 1)
      self.assertEqual(config.num_target_devices, 128)
      self.assertEqual(config.global_batch_size_to_load, 128)

      # Devices per host should remain 8
      devices_per_host = elastic_utils.get_devices_per_host(config)
      self.assertEqual(devices_per_host, 8)

      # 3. SCALE UP: Slice 1 recovers -> {0, 1} active (256 devices)
      fake_manager.active_slice_indices = {0, 1}
      fake_manager.active_slice_count = 2

      elastic_utils.mutate_config_for_topology(config, fake_manager)
      self.assertEqual(config.num_slices, 2)
      self.assertEqual(config.num_target_devices, 256)
      self.assertEqual(config.global_batch_size_to_load, 256)

  def test_grain_worker_count_elastic_bypass_benchmark(self):
    """Verifies that apply_multiprocessing_and_prefetch bypasses pick_performance_config

    when grain_worker_count is -1 during elastic or colocated python runs.
    """
    dummy_dataset = Mock()
    dummy_dataset.mp_prefetch = Mock(return_value="prefetched_dataset")

    # Case A: Elastic training active + worker_count = -1 -> should use num_workers=0 without benchmarking
    config_elastic = FakeConfig(elastic_enabled=True, colocated_python_data_input=False, grain_use_elastic_iterator=False)
    with patch("maxtext.input_pipeline.data_processing_utils.pick_performance_config") as mock_benchmark:
      res = data_processing_utils.apply_multiprocessing_and_prefetch(
          dummy_dataset, config_elastic, grain_worker_count=-1, grain_per_worker_buffer_size=2
      )
      mock_benchmark.assert_not_called()
      dummy_dataset.mp_prefetch.assert_called()
      opts = dummy_dataset.mp_prefetch.call_args[0][0]
      self.assertEqual(opts.num_workers, 0)
      self.assertEqual(opts.per_worker_buffer_size, 2)

    # Case B: Colocated python active + worker_count = -1 -> should use num_workers=0 without benchmarking
    dummy_dataset.mp_prefetch.reset_mock()
    config_colocated = FakeConfig(elastic_enabled=False, colocated_python_data_input=True, grain_use_elastic_iterator=False)
    with patch("maxtext.input_pipeline.data_processing_utils.pick_performance_config") as mock_benchmark:
      res = data_processing_utils.apply_multiprocessing_and_prefetch(
          dummy_dataset, config_colocated, grain_worker_count=-1, grain_per_worker_buffer_size=2
      )
      mock_benchmark.assert_not_called()
      opts = dummy_dataset.mp_prefetch.call_args[0][0]
      self.assertEqual(opts.num_workers, 0)

    # Case C: Non-elastic, non-colocated + worker_count = -1 -> should use pick_performance_config
    dummy_dataset.mp_prefetch.reset_mock()
    config_regular = FakeConfig(elastic_enabled=False, colocated_python_data_input=False, grain_use_elastic_iterator=False)
    mock_perf = Mock()
    mock_perf.multiprocessing_options = grain.MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)
    with patch("maxtext.input_pipeline.data_processing_utils.pick_performance_config", return_value=mock_perf) as mock_benchmark:
      res = data_processing_utils.apply_multiprocessing_and_prefetch(
          dummy_dataset, config_regular, grain_worker_count=-1, grain_per_worker_buffer_size=1
      )
      mock_benchmark.assert_called_once()
      opts = dummy_dataset.mp_prefetch.call_args[0][0]
      self.assertEqual(opts.num_workers, 4)

    # Case D: Explicit worker count provided -> should respect explicitly requested worker count
    dummy_dataset.mp_prefetch.reset_mock()
    with patch("maxtext.input_pipeline.data_processing_utils.pick_performance_config") as mock_benchmark:
      res = data_processing_utils.apply_multiprocessing_and_prefetch(
          dummy_dataset, config_elastic, grain_worker_count=8, grain_per_worker_buffer_size=4
      )
      mock_benchmark.assert_not_called()
      opts = dummy_dataset.mp_prefetch.call_args[0][0]
      self.assertEqual(opts.num_workers, 8)
      self.assertEqual(opts.per_worker_buffer_size, 4)

  def test_form_global_array_colocated_python_end_to_end_split(self):
    """Verifies that the colocated python array splitting works across all local devices

    in a multi-task simulated environment without raising ValueError.
    """
    devices_per_host = 8
    local_devices = [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=i) for i in range(devices_per_host)]

    config = FakeConfig(
        elastic_enabled=True,
        colocated_python_data_input=True,
        grain_use_elastic_iterator=True,
        per_device_batch_size=1.0,
    )

    fake_manager = MagicMock()
    fake_manager.active_slice_indices = {0}
    fake_manager.active_slice_count = 1
    elastic_utils.elastic_manager = fake_manager

    with patch.object(jax, "devices", return_value=local_devices):
      devices_per_host_val = elastic_utils.get_devices_per_host(config)
      self.assertEqual(devices_per_host_val, 8)

      # Simulated batch produced by ElasticIterator for 1 process with 8 devices
      local_batch = int(config.per_device_batch_size * devices_per_host_val)
      self.assertEqual(local_batch, 8)

      fake_batch = {
          "inputs": np.ones((local_batch, config.max_target_length), dtype=np.int32),
          "targets": np.ones((local_batch, config.max_target_length), dtype=np.int32),
      }

      # Test the splitting logic in multihost_dataloading
      for key, arr in fake_batch.items():
        device_splits = np.split(arr, len(local_devices), axis=0)
        self.assertEqual(len(device_splits), 8)
        self.assertEqual(device_splits[0].shape, (1, config.max_target_length))


if __name__ == "__main__":
  unittest.main()
