"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Common Max Utils needed by multiple modules"""
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import checkpointing
import common_types
import functools
import time
import json
import optax
import os
import psutil
import socket
import subprocess
from etils import epath
from collections.abc import Sequence
import collections
from pathlib import Path
from typing import Any, Tuple

import max_logging


import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager


import yaml
import flax
from flax.training import train_state
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from tensorboardX import writer
from google.cloud import storage

HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"

# pylint: disable=too-many-positional-arguments


def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind=memory_kind), t)


def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(lambda t: t.astype(dst) if t.dtype == src else t, nest)


def find_nans_and_infs(pytree):
  def finder(x):
    return jnp.any(jnp.isinf(x) | jnp.isnan(x))

  bad_pytree = jax.tree_util.tree_map(finder, pytree)
  return jax.tree_util.tree_flatten(bad_pytree)


def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), x, initializer=0.0))


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def calculate_total_params_per_chip(params):
  """Calculate total paramsper chip."""

  def calculate_leaf_params_per_chip(arr):
    shard = arr.addressable_shards[0]
    return np.prod(shard.data.shape)

  params_sizes_per_chip = jax.tree_util.tree_map(calculate_leaf_params_per_chip, params)
  total_parameters_per_chip = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes_per_chip)
  return total_parameters_per_chip


def calculate_bytes_from_pytree(params):
  params_bytes = jax.tree_util.tree_map(lambda x: x.nbytes, params)
  total_bytes = jax.tree_util.tree_reduce(lambda x, y: x + y, params_bytes)
  return total_bytes


def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes / num_params


def initialize_summary_writer(config):
  summary_writer_path = os.path.join(config.tensorboard_dir, config.run_name)
  return writer.SummaryWriter(summary_writer_path) if jax.process_index() == 0 else None


def close_summary_writer(summary_writer):
  if jax.process_index() == 0:
    summary_writer.close()


def add_config_to_summary_writer(config, summary_writer):
  """Writes config params to tensorboard"""
  if jax.process_index() == 0:
    for key, value in config.get_keys().items():
      add_text_to_summary_writer(key, str(value), summary_writer)


def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary"""
  if jax.process_index() == 0:
    summary_writer.add_text(key, value)


def write_config_raw_keys_for_gcs(raw_keys):
  """Writes config raw keys to GCS"""
  if not raw_keys["save_config_to_gcs"] or jax.process_index() != 0:
    return
  max_logging.log("Writing config to GCS...")

  raw_keys_dict = dict(raw_keys)
  filename = "config.yml"
  with open(filename, "w", encoding="utf8") as config_for_gcs:
    yaml.dump(raw_keys_dict, config_for_gcs)
  config_for_gcs.close()

  gcs_filename = os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], filename)
  max_logging.log(f"Moving file {filename} to GCS...")
  upload_blob(gcs_filename, filename)
  max_logging.log(f"File {filename} moved successfully!")


def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key


def add_trailing_slash(path):
  if not path.endswith("/"):
    return path + "/"
  return path


def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)


def upload_dump(local_dir, target_dir, module_name=None, delete_local_after=True, all_host_upload=False):
  """Uploads a directory to a GCS location, with an optional filter"""
  if not all_host_upload and jax.process_index() != 0:
    return
  storage_client = storage.Client()
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(target_dir)
  bucket = storage_client.get_bucket(bucket_name)
  if all_host_upload:
    hostname = socket.gethostname()  # Alternatively can use jax.process_id()
    prefix_name = os.path.join(prefix_name, hostname)
    target_dir = os.path.join(target_dir, hostname)
  max_logging.log(f"Uploading HLO Dump to {target_dir}...")
  for root, _, files in os.walk(local_dir):
    for file in files:
      if module_name and module_name not in file:
        continue
      else:
        max_logging.log(f"Uploading {file}")
      local_path = os.path.join(root, file)
      relative_path = os.path.relpath(local_path, local_dir)
      blob_name = os.path.join(prefix_name, relative_path)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(local_path)
  max_logging.log(f"HLO Dump Uploaded to {target_dir}!")
  if delete_local_after:
    shutil.rmtree(local_dir)


def gcs_path_exists(file_path):
  """Checks if a GCS file_path exits."""
  try:
    storage_client = storage.Client()

    bucket_name, file_name = parse_gcs_bucket_and_prefix(file_path)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    return blob.exists()
  except Exception as e:
    print(f"Error while accessing {file_path} from GCE: {str(e)}")
    return False


def gcs_list_directories(directory_path):
  """
  Lists "directories" (prefixes one level down) within a GCS "directory".

  Args:
      directory_path: The prefix representing the parent "directory".

  Returns:
      A list of "directory" names (prefixes).
  """
  storage_client = storage.Client()

  bucket_name, directory_prefix = parse_gcs_bucket_and_prefix(directory_path)

  bucket = storage_client.bucket(bucket_name)

  # Ensures the prefix has a trailing slash to simulate a directory
  if not directory_prefix.endswith("/"):
    directory_prefix += "/"

  # Use list_blobs with a delimiter to get "directories"
  delimiter = "/"
  blobs = bucket.list_blobs(prefix=directory_prefix, delimiter=delimiter)

  directories = []
  # Iterate through the blobs and extract the "directories"
  for page in blobs.pages:
    for prefix in page.prefixes:
      path_obj = Path(prefix)

      directory = path_obj.name

      directories.append(directory)

  return directories


def read_json_from_gcs(file_path):
  try:
    storage_client = storage.Client()

    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_prefix)

    json_string = blob.download_as_string()

    data = json.loads(json_string)

    return data
  except Exception as e:
    print(f"Error reading JSON file from GCS: {str(e)}")
    return None


def write_dict_to_gcs_json(data_dict, file_path):
  """
  Writes a Python dictionary to a JSON file in GCS.

  Args:
    data_dict: The Python dictionary to write
    file_path: GCS path (Bucket + blob) to create the json file
  """
  try:
    storage_client = storage.Client()

    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_prefix)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(data_dict, indent=4)

    # Upload the JSON string to GCS
    blob.upload_from_string(json_string, content_type="application/json")
  except Exception as e:
    print(f"Failed to write json file at {file_path} with error: {str(e)}")


def maybe_initialize_jax_distributed_system(raw_keys):
  """The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
  indirection in MaxText to avoid breaking the call sites unnecessarily.

  Currently jax.distributed.initialize() fully works as expected!

  For CPUs, we call jax.distributed.initialize() explicitly, with the specified arguments.
  """
  if raw_keys["skip_jax_distributed_system"]:
    max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
    return
  if raw_keys["inference_benchmark_test"]:
    # Disable initialization for inference benmark test.
    return
  if raw_keys["compile_topology"]:
    # Don't initialize jax distributed with AOT compilation
    return
  if is_gpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for GPU backend...")
    initialize_jax_for_gpu(raw_keys)
    max_logging.log("Jax distributed system initialized on GPU!")
  elif is_cpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for CPU backend...")
    initialize_jax_for_cpu(raw_keys)
    max_logging.log("Jax distributed system initialized on CPUs!")
  elif (
      raw_keys["enable_checkpointing"]
      and raw_keys["async_checkpointing"]
      and raw_keys["compile_topology_num_slices"] == -1
      and not raw_keys["enable_single_controller"]
  ) or raw_keys["hardware"] == "gpu_multiprocess":
    max_logging.log("Attempting to initialize the jax distributed system...")
    if not raw_keys["enable_emergency_checkpoint"]:
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
    else:
      initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys)
    max_logging.log("Jax distributed system initialized!")


def initialize_jax_for_gpu(raw_keys):
  """Jax distributed initialize for GPUs."""
  if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
    coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{coordinator_port}",
        num_processes=int(os.getenv("NNODES")),
        process_id=int(os.getenv("NODE_RANK")),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )
    max_logging.log(f"JAX global devices: {jax.devices()}")


def initialize_jax_for_cpu(raw_keys):
  """Jax distributed initialize for CPUs. Includes retries until the coordinator is ready."""
  coordinator_ip_address = get_coordinator_ip_address()
  coordinator_address = coordinator_ip_address + ":1234"  # JAX coordinator port used in XPK
  # Env variables to be set in XPK or otherwise
  job_index = int(os.environ.get("JOB_INDEX"))
  job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
  processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
  pid = job_index * processes_in_job + job_completion_index
  max_logging.log(f" Jax process id is {pid} ")
  # Explicit initialize is needed only for CPUs
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      process_id=pid,
      num_processes=int(os.environ.get("JAX_PROCESS_COUNT")),
      initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
  )


def _wait_for_file_to_disappear(f, timeout=300):
  for _ in range(timeout):
    if not f.exists():
      return True
    time.sleep(1)
  return False


def _extract_step(f):
  # The base file name is formatted as {job_name}-s{step}-n{node_rank}-g{gpu_rank}
  return f.rsplit("-", 3)[1][1:]


def _block_and_proces_restore_dir(directory, timeout=300):
  """Block until a file ending with `.restore` appears, then extract the step number and rename
  the directory using the step number.
  """
  WORD = ".restore"
  for _ in range(timeout):
    files = os.listdir(directory)
    for f in files:
      if f.endswith(WORD):
        step = _extract_step(f)
        if step != "0":
          os.rename(epath.Path(directory) / f, epath.Path(directory) / step)
          max_logging.log(f"Found a restore directory at step {step} and renamed it to {epath.Path(directory) / step}.")
        else:
          max_logging.log("Found a restore directory at step 0, skipping renaming.")
        return
    time.sleep(1)
  max_logging.log(f"{timeout} seconds have passed but no .restore file was found.")


def initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys):
  """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
  The information required to initialize JAX distributed runtime will be written by GKE to
  the local checkpoint directory. This function retrieves that information and initializes
  JAX distributed runtime.
  """
  process_id, coordinator_address = _retrieve_jax_init_info(raw_keys)

  if process_id != "" and coordinator_address != "":
    max_logging.log(
        f"Using {process_id} as the process_id and {coordinator_address} as the"
        " coordinator_address to initialize JAX distributed runtime..."
    )
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=int(process_id),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )
    if raw_keys["use_replicator_service"]:
      REPLICATOR_FILE = "replicator.yaml"
      TEMP_FILE = REPLICATOR_FILE + ".tmp"
      replicator_file = epath.Path(raw_keys["local_checkpoint_directory"]) / REPLICATOR_FILE
      if not _wait_for_file_to_disappear(replicator_file):
        max_logging.log("There is existing replicator.yaml which did not disappear in time.")
      else:
        max_logging.log("replicator.yaml no longer exists, creating new replicator.yaml.")
      TEMP_FILE = REPLICATOR_FILE + ".tmp"
      temp_file = epath.Path(raw_keys["local_checkpoint_directory"]) / TEMP_FILE
      num_slices = get_num_slices(raw_keys)
      num_nodes = jax.process_count()
      nodes_per_slice = num_nodes // num_slices
      max_logging.log(f"num_slices: {num_slices}, num_nodes: {num_nodes}, nodes_per_slice: {nodes_per_slice}")
      node_rank = jax.process_index()
      peer_ranks = []
      for i in range(num_slices):
        peer = node_rank % nodes_per_slice + i * nodes_per_slice
        if peer != node_rank:
          peer_ranks.append(peer)
      run_name = raw_keys["run_name"]
      if run_name == "":
        run_name = os.environ.get("JOBSET_NAME")  # using XPK default

      replicator_yaml = f"""job-name: {run_name}
      framework: orbax
      assume-data-parallelism: {num_slices}
      node-rank: {node_rank}
      nodes: {num_nodes}
      workers-per-node: 1
      peer-ranks: {peer_ranks}
      backup-interval-minutes: {raw_keys["replicator_backup_interval_minutes"]}"""

      temp_file.write_text("\n".join([l.strip() for l in replicator_yaml.split("\n")]))
      os.rename(temp_file, replicator_file)
      if not _wait_for_file_to_disappear(replicator_file):
        max_logging.log("The newly created replicator.yaml was not deleted in time.")
      else:
        max_logging.log("The newly created replicator.yaml was deleted, moving forward.")
      _block_and_proces_restore_dir(raw_keys["local_checkpoint_directory"])
  else:
    max_logging.log(
        "Initializing JAX distributed runtime without args when emergency checkpointing is"
        " enabled. This should not happen and your workload may have unexpected behavior."
    )
    jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])

  ocp.multihost.initialize_runtime_to_distributed_ids()
  ocp.multihost.initialize_distributed_to_device_ids()


def _retrieve_jax_init_info(raw_keys):
  """Retrieve JAX init info from a local file."""
  JAX_INIT_INFO_FILE = "jax-init-info.txt"
  local_jax_init_info_file = epath.Path(raw_keys["local_checkpoint_directory"]) / JAX_INIT_INFO_FILE
  # Allow time for the JAX init info file to be populated by GKE. This is needed because the file is
  # only populated when the worker with process id of 0 is determined. After a disruption, although some
  # workers might be up and running, the init info file won't be populated until the node with process id
  # of 0 is known and this could take time. Using 900 seconds for now and it needs to be increased if the
  # "repair" time is longer.
  for i in range(900):
    if local_jax_init_info_file.exists():
      return local_jax_init_info_file.read_text().split("\n")[:2]
    max_logging.log(f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, sleeping for 1 second before retrying...")
    time.sleep(1)
  max_logging.log(
      f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds," "returning empty process id and coordinator address."
  )
  return "", ""


def get_num_slices(raw_keys):
  """Calculate num_slices based on number of devices."""
  if raw_keys["hardware"] == "cpu":
    max_logging.log(" Setting num_slices=1 for CPU hardware type")
    return 1
  if int(raw_keys["compile_topology_num_slices"]) > 0:
    return raw_keys["compile_topology_num_slices"]
  else:
    devices = jax.devices()
    try:
      return 1 + max(d.slice_index for d in devices)
    except (ValueError, AttributeError):
      return 1


def is_cpu_backend(raw_keys):
  """Determine whether Maxtext is intended to run on a CPU backend."""
  return raw_keys["hardware"] == "cpu"


def is_gpu_backend(raw_keys):
  """Determine whether Maxtext is intended to run on a GPU backend."""
  return raw_keys["hardware"] == "gpu"


def get_coordinator_ip_address():
  """Get coordinator IP Address with retries"""
  coordinator_address = ""
  coordinator_ip_address = ""
  if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        coordinator_ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        max_logging.log(
            f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying..."
        )
        lookup_attempt += 1
        time.sleep(5)
  max_logging.log(f"Coordinator IP address: {coordinator_ip_address}")
  return coordinator_ip_address


def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert (
        parallelism_vals.count(-1) == 1
    ), f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert (
        determined_val >= 1 and determined_val.is_integer
    ), f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"
  assert np.prod(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match"
      f" the product of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
  )

  return parallelism_vals


def reshape_mesh_to_rings(a, strategy):
  """Reshape device mesh to rings for 64x4 or 32x8 mesh shape"""
  b = []
  if strategy == HYBRID_RING_64X4:
    for i in range(8):
      b.append([])
      for j in range(8):
        a_i = i * 2
        a_j = j * 2
        # forms a ring of size 4
        b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    b = np.array(b)
    b = np.reshape(b, (64, 4))
  elif strategy == HYBRID_RING_32X8:
    for i in range(8):
      b.append([])
      for j in range(4):
        a_i = i * 2
        a_j = j * 4
        # forms a ring of size 8
        b[i].append(
            [
                a[a_i, a_j],
                a[a_i, a_j + 1],
                a[a_i, a_j + 2],
                a[a_i, a_j + 3],
                a[a_i + 1, a_j + 3],
                a[a_i + 1, a_j + 2],
                a[a_i + 1, a_j + 1],
                a[a_i + 1, a_j],
            ]
        )
    b = np.array(b)
    b = np.reshape(b, (32, 8))
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is not implemented.")
  return b


def create_custom_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any],
    custom_strategy: str,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
) -> np.ndarray:
  """Custom device mesh for 64x4 ici parallelism"""
  assert len(devices) % 256 == 0, f"This custom mesh is not valid for {len(devices)} devices"
  attr = "process_index" if process_is_granule else "slice_index"
  if not hasattr(devices[0], attr):
    raise ValueError(f"Device {devices[0]} does not have attribute {attr}. See" " `process_is_granule` option.")
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())] if should_sort_granules_by_key else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(f"Number of slices {len(granules)} must equal the product of " f"dcn_mesh_shape {dcn_mesh_shape}")
  per_granule_meshes = [
      mesh_utils.create_device_mesh(
          [16, 16],
          granule,
          allow_split_physical_axes=False,
      )
      for granule in granules
  ]

  per_granule_meshes = [np.reshape(reshape_mesh_to_rings(x, custom_strategy), mesh_shape) for x in per_granule_meshes]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh


def is_valid_custom_mesh(ici_parallelism, strategy):
  """Checks if the given strategy and ICI parallelism are valid."""
  if not strategy:
    return False

  valid_strategies = {
      HYBRID_RING_64X4: [1, 4, 64],
      HYBRID_RING_32X8: [1, 8, 32],
  }

  if strategy in valid_strategies:
    if sorted(set(ici_parallelism)) == valid_strategies[strategy]:
      return True
    else:
      raise ValueError(f"Invalid custom_mesh:{strategy} chosen for ICI mesh shape {ici_parallelism}")
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is invalid.")


def optimize_mesh_for_tpu_v6e(mesh, devices):
  """Apply transformations to the mesh to optimize for TPU v6e"""
  if devices[0].device_kind != "TPU v6 lite":
    return mesh
  num_devices = len(devices)
  mesh_is_1d_ring = num_devices in mesh.shape
  if not mesh_is_1d_ring:
    return mesh
  # check that the physical topology is 2x4
  device_coords = [d.coords for d in devices]
  coord_size = len(device_coords[0])
  max_coords = tuple(max(dc[i] for dc in device_coords) for i in range(coord_size))
  min_coords = tuple(min(dc[i] for dc in device_coords) for i in range(coord_size))
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))
  if dims != (2, 4, 1):
    return mesh
  axis_idx = mesh.shape.index(num_devices)
  new_mesh = np.moveaxis(mesh, axis_idx, 0)
  new_mesh[4:] = new_mesh[-1:3:-1]
  new_mesh = np.moveaxis(new_mesh, 0, axis_idx)
  max_logging.log("Optimized the mesh for TPU v6e")
  return new_mesh


def create_device_mesh(config, devices=None):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas"""
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  num_slices = 1 if config.inference_benchmark_test else config.num_slices
  num_devices_per_slice = num_devices // num_slices

  multi_slice_env = num_slices > 1

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(config.ici_parallelism.copy(), num_devices_per_slice, "ICI")

  allow_split_physical_axes = config.allow_split_physical_axes if config.allow_split_physical_axes else False

  if multi_slice_env:
    dcn_parallelism = fill_unspecified_mesh_axes(config.dcn_parallelism.copy(), num_slices, "DCN")
    if is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = create_custom_device_mesh(ici_parallelism, dcn_parallelism, devices, config.custom_mesh)
    else:
      mesh = mesh_utils.create_hybrid_device_mesh(
          ici_parallelism,
          dcn_parallelism,
          devices,
          allow_split_physical_axes=allow_split_physical_axes,
      )
  else:
    if allow_split_physical_axes:
      if is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
        mesh = mesh_utils.create_device_mesh(
            [16, 16],
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=False,
        )
        mesh = reshape_mesh_to_rings(mesh, config.custom_mesh)
        mesh = np.reshape(mesh, ici_parallelism)
      else:
        mesh = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
      mesh = mesh_utils.create_device_mesh(
          ici_parallelism,
          devices,
      )
      if config.optimize_mesh_for_tpu_v6e:
        mesh = optimize_mesh_for_tpu_v6e(mesh, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")

  return mesh


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def init_decode_state(apply_fn, params):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})  # type: ignore
  return state


def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
  return state


def init_initial_state(model, tx, config, is_training, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, is_training, key
  """
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  model_vars = model.init(
      {"params": key, "dropout": key, "aqt": key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)


def apply_lora_on_base_params(base_params, lora_params, lora_scale_factor=1.0):
  """
  Apply the LoRA weights on the base weights of the model using formula:
                W_new = W + BA, where
                    W_new is the new weights with LoRA applied
                    W is the base model weights
                    B is lora_b adapter weights
                    A is lora_a adapter weights

  Here both the base_params and lora_params are PyTrees of same structure depending
  on the base model. The leaf nodes of lora_params are only not-None if it is the target
  module for lora in its config.
  """

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      return base_weight + jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor
    else:
      return base_weight  # Keep the base weight if no Lora update

  def apply_lora_recursively(base_params, lora_params, module_name):
    for name, param in lora_params.items():
      if isinstance(param, dict):
        apply_lora_recursively(base_params[name], param, f"{module_name}.{name}")
      elif param is not None:
        if name not in ["lora_a.kernel", "lora_b.kernel"]:
          raise ValueError(f"Unexpected non-lora specific weights ({module_name}.{name}) found in the lora_params")

        lora_b = lora_params["lora_a.kernel"]
        lora_a = lora_params["lora_b.kernel"]

        base = base_params["kernel"]

        base_params["kernel"] = lora_update_or_base(base, lora_a, lora_b)
        break

  apply_lora_recursively(base_params, lora_params, "")


def setup_decode_state(model, config, rng, mesh, checkpoint_manager):
  """Setup decode state by loading params from a checkpoint.
  Args:
    model: the flax model to initialize
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: Checkpoint manager

  Returns:
    state: state with decode params loaded from the checkpoint
    state_mesh_annotations: the mesh annotations for the state
  """
  if not config.load_parameters_path:
    # generate random params
    max_logging.log("No decode checkpoint specified - generating random weights.")
    state, state_mesh_annotations, _, _ = setup_initial_state(
        model, None, None, config, rng, mesh, checkpoint_manager, False
    )
  else:
    # Load params from checkpoint
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    unboxed_abstract_state, state_mesh_annotations, _ = get_abstract_state(model, None, config, rng, mesh, False)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(config.load_parameters_path, unboxed_abstract_state.params)
    state = init_decode_state(None, params)

  state = unbox_logicallypartioned(state)
  return state, state_mesh_annotations


def load_adapter(config, base_abstract_state_params, adapter_config_path, adapter_weights_path):
  """
  Load the LoRA weights into a PyTree and return it.
  """
  # Load LoRA weights
  lora_params = None
  lora_config = None
  if adapter_config_path:
    if adapter_config_path.startswith("gs://"):
      lora_config = read_json_from_gcs(adapter_config_path)
    else:
      with open(adapter_config_path, "r") as f:
        lora_config = json.load(f)

    if lora_config is None:
      max_logging.log(f"Failed to read lora_config from {adapter_config_path}.")
      return None, None

    if not gcs_path_exists(adapter_weights_path + "/commit_success.txt"):
      max_logging.log(f"Failed to read lora_weights from {adapter_weights_path}.")
      return None, None

    lora_state, _ = get_lora_abstract_state(base_abstract_state_params, lora_config)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      lora_params = checkpointing.load_params_from_path(adapter_weights_path, lora_state.params)

  return lora_params, lora_config


def setup_training_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager):
  is_training = True
  return setup_initial_state(
      model,
      data_iterator,
      tx,
      config,
      rng,
      mesh,
      checkpoint_manager,
      is_training,
  )


def setup_initial_state(
    model,
    data_iterator,
    tx,
    config,
    rng,
    mesh,
    checkpoint_manager,
    is_training=True,
):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    is_training: True to initialize training state, False for decode state

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
      model, tx, config, rng, mesh, is_training
  )

  # Initialization
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    restored, raw_params = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        unboxed_abstract_state,
        config.enable_single_replica_ckpt_restoring,
        config.dataset_type,
    )

    if restored:
      if isinstance(
          checkpoint_manager,
          (
              emergency_checkpoint_manager.CheckpointManager,
              emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
          ),
      ):
        state = restored
      else:
        if "iter" in restored and restored["iter"] is not None:
          data_iterator.local_iterator = restored["iter"]
        state = restored["items"]
    else:
      init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training)
      init_state_partial.__name__ = "initialize_state"
      # pylint: disable=not-callable
      state = jax.jit(
          init_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings,
      )(rng)
      if raw_params:  # If we loaded a partial state, we need to merge it.
        state = state.replace(params=raw_params)

  state = unbox_logicallypartioned(state)

  return state, state_mesh_annotations, state_mesh_shardings, data_iterator


def setup_initial_lora_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager, lora_adapter_path):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    lora_adapter_path: Path of the LoRA adapter which is expected to have
        `adapter_config.json` and adapter weights

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  lora_state = None
  lora_state_annotations = None
  lora_config = None

  if lora_adapter_path:
    max_logging.log(f"Setting initial state of LoRA with lora_adapter_path = {lora_adapter_path}")
    unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
        model, tx, config, rng, mesh, True
    )

    lora_config_path = lora_adapter_path + config.lora_config_file_name

    lora_config = read_json_from_gcs(lora_config_path)

    lora_state, lora_state_annotations = get_lora_abstract_state(unboxed_abstract_state.params, lora_config)

    lora_weights_path = lora_adapter_path + "/0/items"

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      restored_lora, raw_lora_params = checkpointing.load_state_if_possible(
          checkpoint_manager,
          data_iterator,
          lora_weights_path,
          config.load_full_state_path,
          lora_state,
          config.enable_single_replica_ckpt_restoring,
          config.dataset_type,
      )

      if restored_lora:
        raise NotImplementedError("This codepath is not implemented for LoRA adapters yet.")
      else:
        lora_state = lora_state.replace(params=raw_lora_params)
        lora_state = unbox_logicallypartioned(lora_state)

  return lora_config, lora_state, lora_state_annotations


# Learning Rate Schedule
# -----------------------------------------------------------------------------


def create_learning_rate_schedule(config):
  """Creates a warmup and cosine decay learning rate schedule:
  We take inspiration from Llama2's learning rate (LR) schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  Learning rate schedule has either two or three parts:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Cosine from [learning_rate] to [learning_rate * cosine_learning_rate_final_fraction] until learning_rate_schedule_steps
  3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
  The zero learning rate section can be used to more accurately measure the fully trained model's performance.
  """

  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = (step) / len_steps
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr

    return schedule

  lr = config.learning_rate
  cos_final_lr = lr * config.cosine_learning_rate_final_fraction

  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  cos_steps = config.learning_rate_schedule_steps - warmup_steps
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
  cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
  constant_schedule = optax.constant_schedule(0.0)

  pieces = [warmup_schedule, cos_schedule]
  boundaries = [
      warmup_steps,
      warmup_steps + cos_steps,
  ]

  if constant_zero_steps > 0:
    pieces.append(constant_schedule)
    boundaries.append(warmup_steps + cos_steps + constant_zero_steps)

  return optax.join_schedules(pieces, boundaries)


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.
  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.
  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.
  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray],
    Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,  # pytype: disable=bad-return-type  #jax-ndarray
      log_softmax,
      log_z,
  )


def _cross_entropy_with_logits_bwd(
    res: Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (
      jnp.asarray(g_logits, logits.dtype),
      jnp.asarray(g_targets, targets.dtype),
      jnp.array(0.0),
  )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


def get_abstract_state(model, tx, config, rng, mesh, is_training=True):
  """Get a shaped abstraction of the state (including optimizer)"""
  init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training, rng)

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)

  state_logical_annotations = nn.get_partition_spec(abstract_state)

  state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)
  if is_training and config.optimizer_memory_host_offload:
    opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.opt_state)
    params = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.params)
    state_mesh_shardings = state_mesh_shardings.replace(opt_state=opt_state, params=params)

  abstract_sharded_state = jax.jit(init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings).eval_shape()

  unboxed_abstract_sharded_state = unbox_logicallypartioned(abstract_sharded_state)
  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return (
      unboxed_abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
  )


def get_lora_abstract_state(base_abstract_params, lora_config):
  """
  Generates an abstract state representing only the LoRA parameters,
  inferring sharding information from the base parameters.

  Args:
    base_abstract_params: A PyTree containing jax.ShapeDtypeStruct objects
                            representing the abstract state of the base model
                            parameters. This includes sharding information.
    lora_config: A config of the Lora adapter that includes details about the
                 Lora like rank, or target_modules on which lora is implemented.

  Returns:
    A TrainState object representing the abstract state of the LoRA parameters, including
    inferred sharding information.
  """
  other_lora_format_to_jax_format = {
      "q_proj": "self_attention.query",
      "k_proj": "self_attention.key",
      "v_proj": "self_attention.value",
      "o_proj": "self_attention.out",
  }

  lora_target_modules = lora_config["target_modules"]
  lora_target_modules = [other_lora_format_to_jax_format.get(s, s) for s in lora_target_modules]

  lora_rank = int(lora_config["r"])

  lora_abstract_params = {}

  def get_lora_param_shape(base_array_shape, lora_rank, lora_module):
    base_array_dimensions = len(base_array_shape)

    if base_array_dimensions > 4:
      raise ValueError(
          f"Encountered unexpected shape={base_array_shape} of array in base params. Array dimensions > 4 not supported."
      )

    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
      lora_a_shape = base_array_shape[:-2] + (lora_rank,)
      lora_b_shape = (lora_rank,) + base_array_shape[1:]
    elif lora_module in ["self_attention.out"]:
      lora_a_shape = base_array_shape[:-1] + (lora_rank,)
      if base_array_dimensions == 4:
        lora_b_shape = (lora_rank, base_array_shape[1], base_array_shape[-1])
      else:
        lora_b_shape = (lora_rank, base_array_shape[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")

    return lora_a_shape, lora_b_shape

  def get_lora_param_sharding(base_param_sharding, lora_module):
    if base_param_sharding is None:  # Base parameter is replicated
      return None, None  # Replicate LoRA parameters as well

    base_sharding_pspec_size = len(base_param_sharding.spec)

    if base_sharding_pspec_size > 4:
      raise ValueError(f"Encountered unexpected size of PartitionSpec in sharding. Size > 4 is not supported")

    base_mesh = base_param_sharding.mesh
    base_memory_kind = base_param_sharding.memory_kind
    base_pspec = base_param_sharding.spec

    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
      lora_a_pspec_tuple = base_pspec[:-2] + ((),)
      lora_a_pspec = jax.sharding.PartitionSpec(*lora_a_pspec_tuple)

      lora_b_pspec_tuple = ((),) + base_pspec[1:]
      lora_b_pspec = jax.sharding.PartitionSpec(*lora_b_pspec_tuple)

    elif lora_module in ["self_attention.out"]:
      lora_a_pspec_tuple = base_pspec[:-1] + ((),)
      lora_a_pspec = jax.sharding.PartitionSpec(*lora_a_pspec_tuple)
      if base_sharding_pspec_size == 4:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[1], base_pspec[-1])
      else:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")

    lora_a_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_a_pspec, memory_kind=base_memory_kind)
    lora_b_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_b_pspec, memory_kind=base_memory_kind)

    return lora_a_sharding, lora_b_sharding

  def module_is_target_module(module, target_modules):
    """Checks if any of the target_modules is part of the current module which represents an array.

    Args:
      module: A string where nested dictionary keys are concatenated to make a path of the internal most kernel/scale arrays.
      target_modules: A list of strings which represents the target_modules on which lora is applied.

    Return:
      The matched target_module, if that is found in the current module path, None otherwise.
    """
    for target_module in target_modules:
      if target_module in module:
        return target_module
    return None

  def add_lora_params(lora_params, module_name, base_params, lora_rank, lora_target_modules):
    for name, param in base_params.items():
      if isinstance(param, dict):
        lora_params[name] = {}
        add_lora_params(lora_params[name], f"{module_name}.{name}", param, lora_rank, lora_target_modules)
      else:
        if name not in ["kernel", "scale", "embedding"]:
          raise ValueError(f"Unexpected key={name} exists in the abstract params of base model.")

        if not isinstance(param, jax.ShapeDtypeStruct):
          raise ValueError(f"Unexpected type found in the abstract params of the base model.")

        lora_a_key = f"lora_a.kernel"
        lora_b_key = f"lora_b.kernel"

        target_module = module_is_target_module(module_name, lora_target_modules)

        if target_module is not None:
          lora_a_shape, lora_b_shape = get_lora_param_shape(param.shape, lora_rank, target_module)
          base_dtype = param.dtype
          lora_a_sharding, lora_b_sharding = get_lora_param_sharding(param.sharding, target_module)

          lora_params[lora_a_key] = jax.ShapeDtypeStruct(shape=lora_a_shape, dtype=base_dtype, sharding=lora_a_sharding)

          lora_params[lora_b_key] = jax.ShapeDtypeStruct(shape=lora_b_shape, dtype=base_dtype, sharding=lora_b_sharding)
        else:
          lora_params[name] = None

  def get_lora_annotations(lora_abstract_params):
    return jax.tree_util.tree_map(lambda x: x.sharding.spec, lora_abstract_params)

  add_lora_params(lora_abstract_params, "", base_abstract_params, lora_rank, lora_target_modules)

  unboxed_abstract_lora_state = train_state.TrainState(
      step=0, apply_fn=None, params=lora_abstract_params, tx=None, opt_state={}
  )

  lora_state_mesh_annotations = train_state.TrainState(
      step=0, apply_fn=None, params=get_lora_annotations(lora_abstract_params), tx=None, opt_state={}
  )

  return unboxed_abstract_lora_state, lora_state_mesh_annotations


def get_prefill_kv_cache_annotations(model, config, rng, mesh):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.global_batch_size_to_load,
        config.max_prefill_predict_length,
    )

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        model_mode=common_types.MODEL_MODE_PREFILL,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def get_kv_cache_annotations(model, config, rng, mesh):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.global_batch_size_to_load,
        1,
    )

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def print_pytree_shape(print_str, ptree):
  print("\n")
  print(print_str)
  print(jax.tree_util.tree_map(lambda x: x.shape, ptree))


def print_model_vars(print_str, model_vars):
  for k in model_vars:
    print(f"{print_str} key{k}:")
    print(f"\t {model_vars[k]}")


def get_project():
  """Get project"""
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split("\n")
  if len(project_outputs) < 1 or project_outputs[-1] == "":
    max_logging.log("You must specify config.vertex_tensorboard_project or set 'gcloud config set project <project>'")
    return None
  return project_outputs[-1]


def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_util.tree_map(delete_leaf, p)


def summarize_pytree_data(params, name="Params", raw=False):
  """Generate basic metrics of a given Pytree."""
  num_params, total_param_size, avg_param_size = summarize_size_from_pytree(params)
  if not raw:
    num_params_in_billions = num_params / 1e9
    total_param_size_in_gb = total_param_size / 1e9
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params_in_billions:.3f} billion \n"
        f"\tTotal memory usage: {total_param_size_in_gb:.3f} GB \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  else:
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params:.3f} \n"
        f"\tTotal memory usage: {total_param_size:.3f} bytes \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  return num_params, total_param_size, avg_param_size


def save_quantized_checkpoint_if_configured(config, params):
  assert config.quantization, "quantization must be configured"
  if config.save_quantized_params_path:
    checkpointing.save_params_to_path(config.save_quantized_params_path, params)
  else:
    "Skipping saving quantized checkpoint as save_quantized_params_path is null."


def print_mem_stats(label: str):
  max_logging.log(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      max_logging.log(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tMemstats unavailable, error: {ex}")


def print_cpu_ram_stats(label: str):
  max_logging.log(f"\nRAMstats: {label}:")
  try:
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    total = round(ram.total / 2**30, 2)
    available = round(ram.available / 2**30, 2)
    used = round(ram.used / 2**30, 2)

    max_logging.log(f"\tUsing (GB) {used} / {total} ({used/total:%}) -->  Available:{available}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tRAM stats unavailable, error: {ex}")


def print_system_information():
  """Print system information of the current environment.
  Note that this will initialize the JAX backend."""
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.lib.xla_bridge.get_backend().platform_version}")


def permute_to_match_maxtext_rope(arr):
  """Permutes the Huggingface Rope to match the MaxText logic."""
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  evens, odds = np.split(arr, 2, axis=arr.ndim - 1)  # pylint: disable=W0632
  x = np.empty_like(arr)
  x[..., ::2] = evens
  x[..., 1::2] = odds
  return x


def unpermute_from_match_maxtext_rope(arr, model_size):
  """
  Function to get the RoPE values in correct ordering
  """
  if model_size[:8] != "llama3.1":
    return arr
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)
