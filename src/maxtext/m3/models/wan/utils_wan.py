# Copyright 2023–2025 Google LLC
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

"""Wan model utilities: activations, hardware, mesh, precision, GCS and profiler helpers."""

from enum import Enum
import json
import os

import jax
from jax.experimental import mesh_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils

_ACTIVATIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": jax.nn.gelu,
    "mish": jax.nn.mish,
}


def get_activation(name: str):
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


# ==============================================================================
# Wan Model Hardware, Mesh, GCS, Profiler, and Configuration Utilities
# ==============================================================================


# --- Hardware & TPU Generation Detection ---
class TpuType(Enum):
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6_LITE = "v6e"
  TPU_7X = "v7x"
  UNKNOWN = "unknown"


def get_tpu_type() -> TpuType:
  """Detects the current TPU hardware generation."""
  try:
    device_kind = jax.devices()[0].device_kind
    if "7x" in device_kind:
      return TpuType.TPU_7X
    elif "v6 lite" in device_kind:
      return TpuType.TPU_V6_LITE
    else:
      return TpuType.UNKNOWN
  except Exception:  # pylint: disable=broad-exception-caught
    # Any failure to query devices means "unknown TPU type".
    return TpuType.UNKNOWN


def safe_getattr(obj, name, default=None):
  try:
    return getattr(obj, name)
  except (AttributeError, KeyError):
    return default


# --- Attention & Dimension Constants ---
try:
  from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

  BlockSizes = splash_attention_kernel.BlockSizes
except (ImportError, AttributeError):
  BlockSizes = None


# Physical axis names for device meshes.
CONTEXT = "context"

# Logical axis names for diffusion attention.
SELF_ATTN_HEAD = "activation_self_attn_heads"
SELF_ATTN_Q_LENGTH = "activation_self_attn_q_length"
SELF_ATTN_KV_LENGTH = "activation_self_attn_kv_length"
CROSS_ATTN_HEAD = "activation_cross_attn_heads"
CROSS_ATTN_Q_LENGTH = "activation_cross_attn_q_length"
CROSS_ATTN_KV_LENGTH = "activation_cross_attn_kv_length"


# --- Mesh & Precision Utilities ---
def get_precision(config):
  if not hasattr(config, "precision") or not config.precision:
    return None
  return jax.lax.Precision[config.precision]


def get_flash_block_sizes(config):
  """Build splash-attention BlockSizes from config.flash_block_sizes (dict or JSON string)."""
  if not hasattr(config, "flash_block_sizes") or not config.flash_block_sizes:
    return None
  if isinstance(config.flash_block_sizes, str):
    flash_block_sizes = json.loads(config.flash_block_sizes)
  else:
    flash_block_sizes = config.flash_block_sizes
  if flash_block_sizes and BlockSizes is not None:
    # The fused backward kernel computes dq inside the dkv kernel: the separate dq block sizes must be unset.
    fused = str(flash_block_sizes.get("use_fused_bwd_kernel", False)).lower() in ("true", "1")
    return BlockSizes(
        block_q=flash_block_sizes["block_q"],
        block_kv_compute=flash_block_sizes["block_kv_compute"],
        block_kv=flash_block_sizes["block_kv"],
        block_q_dkv=flash_block_sizes["block_q_dkv"],
        block_kv_dkv=flash_block_sizes["block_kv_dkv"],
        block_kv_dkv_compute=flash_block_sizes["block_kv_dkv_compute"],
        block_q_dq=None if fused else flash_block_sizes["block_q_dq"],
        block_kv_dq=None if fused else flash_block_sizes["block_kv_dq"],
        use_fused_bwd_kernel=fused,
    )
  return None


def device_put_replicated(x, sharding):
  """Although the name indicates replication, this function can be used

  to also shard an array based on sharding.
  """
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=import-outside-toplevel

  arr = getattr(x, "value", x)
  shd = getattr(sharding, "value", sharding)
  if hasattr(shd, "devices") and not hasattr(shd, "device_set"):
    shd = NamedSharding(shd, P())
  res = jax.make_array_from_callback(arr.shape, shd, lambda index: arr[index])
  if hasattr(x, "set_value"):
    x.set_value(res)
    return x
  return res


def create_device_mesh(config, devices=None, logging=True):
  """Creates a device mesh supporting 4D mesh (data, fsdp, context, tensor)."""
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max(d.slice_index for d in devices)
  except Exception:  # pylint: disable=broad-exception-caught
    # Devices without slice_index (e.g. CPU) are treated as a single slice.
    num_slices = 1
  num_devices_per_slice = num_devices // num_slices
  if logging:
    max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")

  multi_slice_env = num_slices > 1
  config_keys = config.get_keys() if hasattr(config, "get_keys") else config
  if "dcn_context_parallelism" in config_keys and "ici_context_parallelism" in config_keys:
    dcn_parallelism = [
        config.dcn_data_parallelism,
        config.dcn_fsdp_parallelism,
        config.dcn_context_parallelism,
        config.dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_context_parallelism,
        config.ici_tensor_parallelism,
    ]
  else:
    dcn_parallelism = [
        getattr(config, "dcn_data_parallelism", 1),
        getattr(config, "dcn_fsdp_parallelism", 1),
        getattr(config, "dcn_tensor_parallelism", 1),
    ]
    ici_parallelism = [
        getattr(config, "ici_data_parallelism", 1),
        getattr(config, "ici_fsdp_parallelism", 1),
        getattr(config, "ici_tensor_parallelism", 1),
    ]

  ici_parallelism = max_utils.fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, "ICI")
  allow_split = getattr(config, "allow_split_physical_axes", False)
  if multi_slice_env:
    dcn_parallelism = max_utils.fill_unspecified_mesh_axes(dcn_parallelism, num_slices, "DCN")
    mesh = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism,
        dcn_parallelism,
        devices,
        allow_split_physical_axes=allow_split,
    )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism, devices, allow_split_physical_axes=allow_split)

  if logging:
    max_logging.log(f"Decided on mesh: {mesh}")

  return mesh


# --- File, GCS, and Prompt Utilities ---
def upload_file_to_gcs(output_dir: str, file_path: str, subdir: str = ""):
  """Uploads one generated file to {output_dir}/{subdir}/, logging failures."""
  try:
    # Optional dependency, only needed when uploading to GCS.
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""
    destination_blob_name = os.path.normpath(os.path.join(folder_name, subdir, os.path.basename(file_path))).lstrip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    max_logging.log(f"Uploading {file_path} to {bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(file_path)
    max_logging.log(f"Upload complete {file_path}.")
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Best-effort upload: log and continue.
    max_logging.log(f"An error occurred: {e}")


def delete_file(file_path: str):
  """Removes a local file, e.g. after it has been uploaded to GCS."""
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_gcs_output_path(config):
  """Return the gs:// output path for this run, or None if output_dir is not on GCS."""
  output_dir = getattr(config, "output_dir", "")
  if not output_dir or not output_dir.startswith("gs://"):
    return None
  run_name = getattr(config, "run_name", "")
  return os.path.join(output_dir, run_name) if run_name else output_dir


def load_prompts(prompt_file_path: str, default_prompt: str = "") -> list:
  """Load non-empty, non-comment prompts from a local file or http(s) URL."""
  if not prompt_file_path:
    if not default_prompt:
      raise ValueError("Neither prompt_file nor prompt was specified.")
    return [default_prompt]
  if prompt_file_path.startswith("http://") or prompt_file_path.startswith("https://"):
    import requests  # pylint: disable=import-outside-toplevel

    response = requests.get(prompt_file_path, timeout=60)
    response.raise_for_status()
    raw_lines = response.text.splitlines()
  else:
    if not os.path.isfile(prompt_file_path):
      raise FileNotFoundError(f"Prompt file not found at local path: {prompt_file_path}")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()
  prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
  if not prompts:
    if default_prompt:
      return [default_prompt]
    raise ValueError(f"Prompt file '{prompt_file_path}' contains no valid non-empty prompts.")
  return prompts


def chunk_and_pad(items: list, batch_size: int):
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}")
  for i in range(0, len(items), batch_size):
    chunk = items[i : i + batch_size]
    actual_len = len(chunk)
    padded_chunk = chunk + [chunk[-1]] * (batch_size - actual_len) if actual_len < batch_size else chunk
    yield i, padded_chunk, actual_len


# --- Profiler ---
def profiler_enabled(config):
  return bool(getattr(config, "enable_profiler", False)) and jax.process_index() == 0


class Profiler:
  """Starts/stops a JAX profiler trace and uploads it to GCS when configured."""

  def __init__(self, config, session_name=None):
    self.config = config
    self.session_name = session_name
    self._active = None

  def start(self):
    """Start a JAX trace if profiling is enabled on this process."""
    if not profiler_enabled(self.config):
      return
    log_dir = getattr(self.config, "tensorboard_dir", "")
    if not log_dir or log_dir.startswith("gs://"):
      run_name = getattr(self.config, "run_name", "run")
      log_dir = os.path.join("/tmp/profiler_traces", run_name)
    if self.session_name:
      log_dir = os.path.join(log_dir, self.session_name)
    os.makedirs(log_dir, exist_ok=True)
    max_logging.log(f"Starting profiler trace in: {log_dir}")
    jax.profiler.start_trace(log_dir)
    self._active = "jax"

  def stop(self):
    """Stop the active trace and upload it to GCS if tensorboard_dir is a gs:// path."""
    if self._active == "jax":  # pylint: disable=too-many-nested-blocks
      jax.profiler.stop_trace()
      trace_dir = getattr(self.config, "tensorboard_dir", "")
      if trace_dir.startswith("gs://"):
        run_name = getattr(self.config, "run_name", "run")
        local_dir = os.path.join("/tmp/profiler_traces", run_name)
        if os.path.exists(local_dir):
          try:
            # Optional dependency, only needed when uploading traces.
            from google.cloud import storage  # pylint: disable=import-outside-toplevel

            path_without_scheme = trace_dir.removeprefix("gs://")
            parts = path_without_scheme.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            for root, _, files in os.walk(local_dir):
              for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_dir)
                blob_name = os.path.join(prefix, rel_path)
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                max_logging.log(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}")
          except Exception as e:  # pylint: disable=broad-exception-caught
            # Best-effort upload: never fail the run because of trace upload.
            max_logging.log(f"Failed to upload profiler traces: {e}")
    self._active = None
