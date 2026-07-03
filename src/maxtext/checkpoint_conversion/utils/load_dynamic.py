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

"""Dynamic loading of HuggingFace checkpoints during training/eval workloads directly in the target format.

This module allows loading HuggingFace checkpoints (in Safetensors format)
directly during MaxText training or evaluation runs, performing on-the-fly sharded
restore and CPU/TPU transformations. This avoids offline pre-conversion steps
and prevents host OOM.

Usage:
  To load Hugging Face checkpoints directly, configure the following flags:
  1. `source_checkpoint_layout`: Set to `"safetensors_dynamic"`.
  2. `load_parameters_path`: Set to the source path of the Hugging Face checkpoint.

Examples:
  A. Load from a Google Cloud Storage (GCS) directory containing `.safetensors`:
     ```
     python3 maxtext/trainers/pre_train/train.py \
         maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         load_parameters_path="gs://my-bucket/path/to/safetensors_directory/"
     ```

  B. Load directly from the Hugging Face Hub (automatically cached to GCS):
     ```
     python3 maxtext/trainers/pre_train/train.py \
         maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         load_parameters_path="hf://meta-llama/Meta-Llama-3-8B" \
         hf_access_token="<your_token>" \
         base_output_directory="gs://my-bucket/output/"
     ```

  C. Load from Hugging Face Hub using automatic model_name resolution:
     ```
     python3 maxtext/trainers/pre_train/train.py \
         maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         load_parameters_path="" \
         hf_access_token="<your_token>" \
         base_output_directory="gs://my-bucket/output/"
     ```

Note:
  - Hugging Face weights from HF Hub are cached to `base_output_directory`.
  - When loading from Hugging Face Hub, `base_output_directory` must start with
    "gs://" and `hf_access_token` is required if downloading gated models.
"""

import concurrent.futures
import multiprocessing
import os
import random
import time

from flax import nnx
import flax.traverse_util
from google.cloud import storage
import huggingface_hub
import jax
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import tensor_handling
from maxtext.utils import gcs_utils
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils


HF_MODEL_CONFIGS = hf_model_configs.HF_MODEL_CONFIGS
get_hf_loading_function = tensor_handling.get_hf_loading_function


def build_gcs_cache_worker(fpath, gcs_cache_dir, hf_access_token):
  """Caches a file from Hugging Face to a GCS bucket cache directory.

  Args:
    fpath: The full remote file path on the Hugging Face virtual file system
      (e.g., "meta-llama/Meta-Llama-3-8B/model-00001-of-00004.safetensors").
    gcs_cache_dir: The destination directory in GCS.
    hf_access_token: The access token for Hugging Face.
  """
  fs = huggingface_hub.HfFileSystem(token=hf_access_token)
  time.sleep(random.uniform(0.0, 5.0))

  bucket_name, blob_prefix = gcs_utils.parse_gcs_bucket_and_prefix(gcs_cache_dir)
  blob_name = os.path.join(blob_prefix, os.path.basename(fpath))

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)

  if blob.exists():
    max_logging.log(f"[Worker] Cache hit for {os.path.basename(fpath)}.")
    return

  t0 = time.time()
  max_retries = 5
  for attempt in range(max_retries):
    try:
      with fs.open(fpath, "rb") as remote_f:
        blob.chunk_size = 1024 * 1024 * 32  # 32MB chunks
        blob.upload_from_file(remote_f, client=storage_client)
      print(
          f"[Worker] Cached {os.path.basename(fpath)} in" f" {time.time() - t0:.1f}s",
          flush=True,
      )
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      if attempt < max_retries - 1:
        max_logging.log(
            f"Error fetching {fpath} to GCS: {e}. Retrying in 15 seconds..." f" (Attempt {attempt+1}/{max_retries})"
        )
        time.sleep(15)
      else:
        max_logging.log(f"Failed to fetch {fpath} to GCS after {max_retries} attempts.")
        raise


def get_hf_config_and_mappings(maxtext_config):
  """Gets HF config and parameter mapping based on the MaxText config."""
  model_key = maxtext_config.model_name
  if "-Instruct" in model_key:
    model_key = model_key.replace("-Instruct", "")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()

  param_map_mt_to_hf = param_mapping.PARAM_MAPPING[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers
  )
  hook_fn_map_mt = param_mapping.HOOK_FNS[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers, saving_to_hf=False
  )
  return param_map_mt_to_hf, hook_fn_map_mt


def load_sharded_hf_state(path):
  """Loads HF state with maximal sharding across TPU mesh to avoid host OOM.

  Args:
    path: A directory path (either local or GCS starting with gs://) containing
      the .safetensors files (e.g., "gs://my-bucket/hf_cache/model_id" or
      "/path/to/safetensors_directory/"). If a Hugging Face Hub ID was used,
      it should already be cached/downloaded to GCS before calling this
      function.

  Returns:
    The loaded Hugging Face state dictionary mapping parameter names to
    JAX arrays.
  """
  t0 = time.time()
  context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
  with context:
    metadata = ocp_v1.pytree_metadata(path)
    simple_abstract_state = metadata.metadata

    # Distributed Sharded Download: Tell JAX to shard the HF Safetensors download
    # across the entire TPU mesh to avoid Host OOM.
    current_global_devices = jax.devices()
    shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state, devices=current_global_devices)

    def combine_sharding(sds, single_sharding):
      return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=single_sharding)

    sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)

    max_logging.log("Reading raw Safetensors into memory (Distributed Sharded GCS Download)...")
    hf_state = ocp_v1.load_pytree(path, sharded_abstract_state)
    max_logging.log(f"load_sharded_hf_state took {time.time() - t0:.2f}s")
    return hf_state


def transform_hf_state_to_mt_state(hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config):
  """Transforms HF state into MaxText state by applying param mappings and mathematical hooks."""
  t0 = time.time()

  def tensor_getter(key):
    return hf_state.pop(key)

  flat_target = flax.traverse_util.flatten_dict(target_tree, sep=None)
  flat_restored = flat_target.copy()

  # Create a lookup mapping from stringified/joined path to the original tuple path
  path_str_to_tuple = {".".join(map(str, path)): path for path in flat_target}

  mapped_count = 0
  keys_missed = []
  max_logging.log("Starting fast in-memory Distributed Transformations...")

  for mt_key, hf_source in param_map_mt_to_hf.items():
    mt_name = mt_key.replace("params-", "").replace("-", ".")

    # Determine the correct key in path_str_to_tuple
    check_name = mt_name
    if check_name not in path_str_to_tuple:
      if f"params.{mt_name}" in path_str_to_tuple:
        check_name = f"params.{mt_name}"
      elif mt_key.replace("-", ".") in path_str_to_tuple:
        check_name = mt_key.replace("-", ".")

    if check_name not in path_str_to_tuple:
      keys_missed.append(mt_name)
      continue

    target_path = path_str_to_tuple[check_name]
    target_leaf = flat_target[target_path]
    hook_fn = hook_fn_map_mt.get(mt_key)

    load_fn = get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_leaf,
        maxtext_config,
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    flat_restored[target_path] = load_fn()

    max_logging.log(f"Transformed {check_name} from {hf_source} in {time.time() - t_layer:.4f}s")
    mapped_count += 1

  if mapped_count == 0:
    max_logging.log(f"All transformations missed! Sample missed mt_names: {keys_missed[:5]}")
    max_logging.log(f"Sample flat_target keys: {list(path_str_to_tuple.keys())[:5]}")

  max_logging.log(f"Successfully mapped {mapped_count} parameters.")
  restored_params = flax.traverse_util.unflatten_dict(flat_restored, sep=None)

  if "params" in restored_params:
    restored_params = restored_params["params"]

  max_logging.log(f"transform_hf_state_to_mt_state took {time.time() - t0:.2f}s")

  return {"params": restored_params}


def write_gcs_latch(gcs_cache_dir):
  """Host 0 writes the GCS latch file to signal that caching is complete."""
  storage_client = storage.Client()
  bucket_name = gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[0]
  blob_prefix = (
      gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[1]
      if "/" in gcs_cache_dir.replace("gs://", "")
      else ""
  )
  latch_blob_name = os.path.join(blob_prefix, "download_complete.txt")
  latch_blob = storage_client.bucket(bucket_name).blob(latch_blob_name)
  latch_blob.upload_from_string("complete")
  max_logging.log(f"Host 0 wrote dynamic GCS download latch file: {gcs_cache_dir}/download_complete.txt")


def wait_on_gcs_latch(gcs_cache_dir, host_id):
  """Hosts 1-255 wait for Host 0 in standard CPU sleep loop to prevent JAX collective hang timeout abort."""
  storage_client = storage.Client()
  bucket_name = gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[0]
  blob_prefix = (
      gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[1]
      if "/" in gcs_cache_dir.replace("gs://", "")
      else ""
  )
  latch_blob_name = os.path.join(blob_prefix, "download_complete.txt")
  latch_blob = storage_client.bucket(bucket_name).blob(latch_blob_name)

  max_logging.log(f"Host {host_id} polling GCS latch at {gcs_cache_dir}/download_complete.txt...")
  t_poll_start = time.time()
  last_logged_min = 0
  while not latch_blob.exists():
    time.sleep(10)
    elapsed_min = int(time.time() - t_poll_start) // 60
    if elapsed_min > last_logged_min:
      last_logged_min = elapsed_min
      # only log every minute to avoid spamming logs.
      max_logging.log(f"Host {host_id} still waiting for Host 0 download latch...")

  max_logging.log(f"Host {host_id} detected GCS download complete latch!")


def jax_devices_barrier(name="dynamic_hf_download_complete"):
  """Synchronizes all hosts/devices using standard JAX multihost sync_global_devices."""
  host_id = jax.process_index()
  max_logging.log(f"Host {host_id} aligning device clocks via JAX sync_global_devices...")
  jax.experimental.multihost_utils.sync_global_devices(name)


def _execute_gcs_download(gcs_cache_dir, files, maxtext_config):
  """Checks GCS cache and executes parallel downloads to shared GCS."""
  t_gcs_start = time.time()
  max_logging.log("Dynamic HF Hub Fast DL: Host 0 is downloading to shared GCS" f" Cache: {gcs_cache_dir}")

  # List existing blobs to avoid spawning processes for already cached
  # files
  storage_client = storage.Client()
  bucket_name = gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[0]
  blob_prefix = (
      gcs_cache_dir.replace("gs://", "").split("/", maxsplit=1)[1]
      if "/" in gcs_cache_dir.replace("gs://", "")
      else ""
  )

  existing_blobs = {blob.name for blob in storage_client.list_blobs(bucket_name, prefix=blob_prefix)}

  files_to_download = []
  for fpath in files:
    expected_blob_name = os.path.join(blob_prefix, os.path.basename(fpath))
    if expected_blob_name not in existing_blobs:
      files_to_download.append(fpath)

  if files_to_download:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=32, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
      futures = [
          executor.submit(
              build_gcs_cache_worker,
              fpath,
              gcs_cache_dir,
              maxtext_config.hf_access_token,
          )
          for fpath in files_to_download
      ]

      while futures:
        done, futures = concurrent.futures.wait(futures, timeout=10)

        # Raise any exceptions if a worker failed
        for f in done:
          f.result()

  t_gcs_end = time.time()
  max_logging.log(
      f"GCS caching complete in {t_gcs_end - t_gcs_start:.2f}s."
      f" Downloaded {len(files_to_download)} missing files."
  )


def sync_dynamic_caching(gcs_cache_dir, files, host_id, maxtext_config):
  """Coordinate downloading files on Host 0 and polling status on Hosts 1-255."""
  sync_via_jax = getattr(maxtext_config, "safetensors_sync_via_jax", False)

  if sync_via_jax:
    # Option 1: Baseline JAX barrier (Host 0 downloads, others wait inside JAX barrier)
    if host_id == 0:
      _execute_gcs_download(gcs_cache_dir, files, maxtext_config)

    max_logging.log(f"Host {host_id} waiting for GCS cache at {gcs_cache_dir} to be populated by Host 0...")
    jax_devices_barrier()
    max_logging.log(f"Host {host_id} detected GCS cache is ready!")
  else:
    # Option 2: GCS latch file polling (Host 0 downloads and writes latch, others poll via CPU sleep)
    if host_id == 0:
      _execute_gcs_download(gcs_cache_dir, files, maxtext_config)
      write_gcs_latch(gcs_cache_dir)
    else:
      wait_on_gcs_latch(gcs_cache_dir, host_id)

    # Finally, align clocks across all devices via brief JAX barrier
    jax_devices_barrier()
    max_logging.log(f"Host {host_id} detected GCS cache is ready!")


def load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config):
  """Main entry point to dynamically build and load safetensors into MaxText format.

  Splits execution into:
  1. Deriving Mappings
  2. Loading Sharded arrays directly to TPUs
  3. Processing the transformations natively on TPUs
  """
  if maxtext_config is None:
    raise ValueError("maxtext_config must be provided for safetensors_dynamic loading.")

  model_name = maxtext_config.model_name
  if "-Instruct" in model_name:
    model_name = model_name.replace("-Instruct", "")

  if not path:
    if model_name not in maxtext_globals.HF_IDS:
      raise ValueError(f"Unsupported model name for automatic HF repo resolution: {model_name}.")
    path = maxtext_globals.HF_IDS[model_name]

  if path.startswith("hf://"):
    path = path[5:]

  if not path.startswith("gs://") and not os.path.isdir(path):
    fs = huggingface_hub.HfFileSystem(token=maxtext_config.hf_access_token)
    repo_id = path

    files = fs.glob(f"{repo_id}/*.safetensors")

    host_id = jax.process_index()

    if hasattr(maxtext_config, "base_output_directory") and maxtext_config.base_output_directory.startswith("gs://"):
      gcs_cache_dir = f"{maxtext_config.base_output_directory}/hf_cache/{repo_id.replace('/', '_')}"
      path = gcs_cache_dir

      # Only Host 0 downloads while Hosts 1-255 wait using HTTP polling on coordinator
      sync_dynamic_caching(gcs_cache_dir, files, host_id, maxtext_config)

    else:
      raise ValueError("base_output_directory with gs:// prefix is required for " "huggingface downloads.")

  t_total = time.time()
  param_map_mt_to_hf, hook_fn_map_mt = get_hf_config_and_mappings(maxtext_config)
  max_logging.log(f"[1/3] Mappings derived in {time.time() - t_total:.2f}s")

  if isinstance(abstract_unboxed_pre_state, nnx.State):
    # In NNX, abstract_unboxed_pre_state contains both model and optimizer states.
    # We only want to target and transform the model variables.
    model_state = getattr(abstract_unboxed_pre_state, "model", None)
    if model_state is not None:
      target_tree = (
          model_state.to_pure_dict()
          if hasattr(model_state, "to_pure_dict")
          else model_state
      )
    else:
      target_tree = abstract_unboxed_pre_state.to_pure_dict()
  else:
    # In Linen, params is only the model parameters.
    target_tree = abstract_unboxed_pre_state.params

  t1 = time.time()
  hf_state = load_sharded_hf_state(path)
  max_logging.log(f"[2/3] Distributed Sharded GCS load completed in {time.time() - t1:.2f}s")

  t2 = time.time()
  # Transform Hugging Face weight tensors on-the-fly into MaxText format
  # in-memory. This is done in-memory on each host, sharded across the mesh.
  restored_params = transform_hf_state_to_mt_state(
      hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
  )
  max_logging.log(f"[3/3] CPU Transformations completed in {time.time() - t2:.2f}s")
  max_logging.log(f"Total safetensors_dynamic duration: {time.time() - t_total:.2f}s")

  if restored_params and "params" in restored_params:
    restored_params = restored_params["params"]

  def _filter_shape_dtype_structs(d):
    if not isinstance(d, dict):
      return d
    res = {}
    for k, v in d.items():
      if isinstance(v, dict):
        sub = _filter_shape_dtype_structs(v)
        if sub:
          res[k] = sub
      elif not isinstance(v, jax.ShapeDtypeStruct):
        res[k] = v
    return res

  restored_params = _filter_shape_dtype_structs(restored_params)

  return None, restored_params
