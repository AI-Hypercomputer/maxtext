# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional
from jax._src import xla_bridge
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

# TODO(mattdavidow) # Does QR always align worker 0 to coordinator?
# Test on the big 5 (qr,gke) x (single,multi) + borg

def get_metadata(key):
  import requests  # pytype: disable=import-error
  import time  # pytype: disable=import-error
  # Based on https://github.com/tensorflow/tensorflow/pull/40317
  gce_metadata_endpoint = 'http://' + os.environ.get(
      'GCE_METADATA_IP', 'metadata.google.internal')

  retry_count = 0
  retrySeconds = 0.500
  api_resp = None

  while retry_count < 6:
    api_resp = requests.get(
        f'{gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/{key}',
        headers={'Metadata-Flavor': 'Google'})
    if api_resp.status_code == 200:
      break
    retry_count += 1
    time.sleep(retrySeconds)

  if api_resp is None:
    raise RuntimeError(f"Getting metadata['{key}'] failed for 6 tries")
  return api_resp.text

def get_tpu_env_value(key):
  def get_tpu_env_value_from_metadata(key):
    tpu_env_data = get_metadata('tpu-env')
    key_value_pairs = tpu_env_data.split('\n')
    for key_value_pair in key_value_pairs:
      # Typical line is <MEGASCALE_NUM_SLICES: '2'>
      if ':' in key_value_pair:
        key_value_split = key_value_pair.split(':')
        row_key, value = key_value_split[0].strip(), key_value_split[1]
        if row_key == key:
          return value.strip().strip("'")
    return None

  value = os.environ.get(key, None)
  return value if value is not None else get_tpu_env_value_from_metadata(key)

def is_multislice_env():
  return get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS') is not None

def is_gke_env():
  return os.environ.get("TPU_WORKER_ID", None) is not None

def get_gce_worker_endpoints() -> str:
    return get_metadata('worker-network-endpoints').split(',')

class SingleSliceGceTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and not is_multislice_env() and not is_gke_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    return get_gce_worker_endpoints()[0].split(':')[2] + ':8476'

  @classmethod
  def get_process_count(cls) -> int:
    return len(get_gce_worker_endpoints())

  @classmethod
  def get_process_id(cls) -> int:
    return int(get_metadata('agent-worker-number'))

  @classmethod
  def get_local_process_id(cls) -> Optional[int]:
    return None

class MultisliceGceTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_multislice_env() and not is_gke_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    coordinator_address = get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS')
    # Use a different port for the jax coordinator than the MXLA coordinator.
    coordinator_address = coordinator_address.split(':')[0] + ':8476'
    return coordinator_address

  @classmethod
  def get_process_count(cls) -> int:
    processes_per_slice = cls._get_process_count_per_slice()
    num_slices = get_metadata('MEGASCALE_NUM_SLICES')
    return processes_per_slice * num_slices

  @classmethod
  def get_process_id(cls) -> int:
    process_id_in_slice = int(get_metadata('agent-worker-number'))
    slice_id = get_metadata('MEGASCALE_SLICE_ID')
    processes_per_slice = cls._get_process_count_per_slice()
    return process_id_in_slice + slice_id * processes_per_slice

  @classmethod
  def get_local_process_id(cls) -> Optional[int]:
    return None

  @staticmethod
  def _get_process_count_per_slice() -> Optional[int]:
    return len(get_gce_worker_endpoints())

class GkeTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_gke_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    coordinator_address = get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS')
    # Use a different port for the jax coordinator than the MXLA coordinator.
    coordinator_address = coordinator_address.split(':')[0] + ':8476'
    return coordinator_address

  @classmethod
  def get_process_count(cls) -> int:
    processes_per_slice = cls._get_process_count_per_slice()
    num_slices = int(os.environ.get('MEGASCALE_NUM_SLICES'))
    return processes_per_slice * num_slices

  @classmethod
  def get_process_id(cls) -> int:
    process_id_in_slice = int(os.environ.get('TPU_WORKER_ID'))
    slice_id = int(os.environ.get('MEGASCALE_SLICE_ID'))
    processes_per_slice = cls._get_process_count_per_slice()
    return process_id_in_slice + slice_id * processes_per_slice

  @classmethod
  def get_local_process_id(cls) -> Optional[int]:
    return None

  @staticmethod
  def _get_process_count_per_slice() -> Optional[int]:
    return len(os.environ.get('TPU_WORKER_HOSTNAMES').split(','))