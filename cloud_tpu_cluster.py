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
import re
from typing import Optional
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm


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
      # Typical line is MEGASCALE_NUM_SLICES: '2'
      if ':' in key_value_pair:
        row_key, value = re.split(':', key_value_pair, 1)
        row_key = row_key.strip()
        if row_key == key:
          return value.strip().strip("'")
    return None

  value = os.environ.get(key, None)
  return value if value is not None else get_tpu_env_value_from_metadata(key)

def is_gce_env():
  worker_number_string = get_metadata('agent-worker-number')
  try:
    worker_number = int(worker_number_string)
    return True
  except:
    return False

def is_multislice_gce_env():
  return is_gce_env() and get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS') is not None

def is_gke_env():
  return os.environ.get("TPU_WORKER_HOSTNAMES", None) is not None

def get_gce_worker_endpoints() -> str:
  return get_metadata('worker-network-endpoints').split(',')

class SingleSliceGceTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_gce_env() and not is_multislice_gce_env()

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
    return running_in_cloud_tpu_vm and is_multislice_gce_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    coordinator_address = get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS')
    # Use a different port for the jax coordinator than the MXLA coordinator,
    # which is set to 8080 in multislice GCE.
    coordinator_address = coordinator_address.split(':')[0] + ':8476'
    return coordinator_address

  @classmethod
  def get_process_count(cls) -> int:
    processes_per_slice = cls._get_process_count_per_slice()
    num_slices = int(get_tpu_env_value('MEGASCALE_NUM_SLICES'))
    return processes_per_slice * num_slices

  @classmethod
  def get_process_id(cls) -> int:
    process_id_in_slice = cls._get_process_id_in_slice()
    slice_id = int(get_tpu_env_value('MEGASCALE_SLICE_ID'))
    processes_per_slice = cls._get_process_count_per_slice()
    return process_id_in_slice + slice_id * processes_per_slice

  @classmethod
  def get_local_process_id(cls) -> Optional[int]:
    return None

  @staticmethod
  def _get_process_count_per_slice() -> Optional[int]:
    return len(get_gce_worker_endpoints())

  @staticmethod
  def _get_process_id_in_slice() -> Optional[int]:
    return int(get_metadata('agent-worker-number'))

class GkeTpuCluster(MultisliceGceTpuCluster):
  # This class handles both single and multislice GKE as the environment
  # variables are set the same in both cases.
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_gke_env()

  @staticmethod
  def _get_process_count_per_slice() -> Optional[int]:
    tpu_worker_hostnames = str(os.environ.get('TPU_WORKER_HOSTNAMES', None))
    return len(tpu_worker_hostnames.split(','))

  @staticmethod
  def _get_process_id_in_slice() -> Optional[int]:
    return int(os.environ.get('TPU_WORKER_ID'))
