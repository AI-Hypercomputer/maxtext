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

By default the whole checkpoint is read (`load_sharded_hf_state`) and then converted
(`transform_hf_state_to_mt_state`), so HBM holds the HF checkpoint and the MaxText weights at
once. With `safetensors_dynamic_streaming=True` (experimental) the checkpoint is instead streamed
a few decoder layers at a time (see `hf_streaming_load`): each read call's tensors (about 4 GiB
per host) are read straight into HBM, converted on the TPU chips, and written into the sharded
MaxText weights.

Usage:
  To load Hugging Face checkpoints directly, configure the following flags:
  1. `source_checkpoint_layout`: Set to `"safetensors_dynamic"`.
  2. `load_parameters_path`: Set to the source path of the Hugging Face checkpoint.
  3. Optional: `safetensors_dynamic_streaming=True` to stream the checkpoint layer by layer.

Examples:
  A. Load from a Google Cloud Storage (GCS) directory containing `.safetensors`:
     ```
     python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         load_parameters_path="gs://my-bucket/path/to/safetensors_directory/"
     ```

  B. Load directly from the Hugging Face Hub (automatically cached to GCS). The repo must match
     `model_name`:
     ```
     python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         load_parameters_path="hf://meta-llama/Llama-3.1-8B" \
         base_output_directory="gs://my-bucket/output/"
     ```

  C. Load from the Hugging Face Hub repo listed in `HF_IDS[model_name]`. Leave
     `load_parameters_path` unset; an empty value fails config validation.
     ```
     python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
         run_name=my_run \
         model_name=llama3.1-8b \
         source_checkpoint_layout="safetensors_dynamic" \
         base_output_directory="gs://my-bucket/output/"
     ```

Note:
  - Weights from the Hugging Face Hub are cached under `<base_output_directory>/hf_cache/`, so
    `base_output_directory` must start with "gs://".
  - `hf_access_token` is required if downloading gated models.
  - Loaded weights use `weight_dtype`, regardless of the checkpoint dtype. Block-scaled `FP8`
    (`*_scale_inv` / `*.scale`) and packed `MXFP4` (`*.scale`) weights (e.g., official DeepSeek-V3 /
    V3.2 / V4 Hugging Face checkpoints) are dequantized to `bfloat16` on the fly before casting to
    `weight_dtype`.
  - Supported by both training restore (`load_state_if_possible`) and `from_pretrained`.
  - The Pathways backend is not supported.
"""

import collections
import concurrent.futures
import functools
import math
import multiprocessing
import os
import random
import time

from flax import nnx
import flax.traverse_util
import huggingface_hub
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from maxtext.checkpoint_conversion.utils import dequantize
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import safetensors_reader
from maxtext.checkpoint_conversion.utils import tensor_handling
from maxtext.common.gcloud_stub import gcs_storage
from maxtext.utils import gcs_utils
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
import numpy as np
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils


# Route GCS through the decoupling helper so this module imports cleanly in
# decoupled environments where google-cloud-storage is intentionally absent.
storage = gcs_storage()

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


def _leading_axis_shardings(abstract_state, devices, num_groups=None):
  """Shards each tensor along its leading axis only, so each process reads one contiguous byte range.

  A safetensors tensor is stored row-major, so a shard is contiguous in the file only if it splits the
  leading axis alone. A sharding that splits an inner axis makes every process read many small strided
  ranges, and Orbax then reads whole file spans instead: each host reads almost the whole checkpoint.

  Each leading axis is split across all devices if it divides evenly, else across processes (copies on a
  process's local devices), else not at all (only small tensors such as norms and routers in practice).

  Args:
    abstract_state: Tree of `jax.ShapeDtypeStruct` from the checkpoint metadata.
    devices: All devices.
    num_groups: If set, splits the devices into this many groups by id instead of one per process. For
      tests on one process.

  Returns:
    A tree of `NamedSharding` matching `abstract_state`.
  """
  if num_groups is None:
    by_process = {}
    for device in devices:
      by_process.setdefault(device.process_index, []).append(device)
    groups = [sorted(group, key=lambda d: d.id) for _, group in sorted(by_process.items())]
  else:
    ordered = sorted(devices, key=lambda d: d.id)
    size = len(ordered) // num_groups
    groups = [ordered[i * size : (i + 1) * size] for i in range(num_groups)]
  if len({len(group) for group in groups}) != 1 or sum(map(len, groups)) != len(devices):
    raise ValueError(f"Cannot split {len(devices)} devices into {len(groups)} groups of equal size.")
  mesh = Mesh(np.array(groups), ("process", "local"))
  num_processes, num_local = mesh.devices.shape

  def leading_axis_sharding(sds):
    if not sds.shape:
      return NamedSharding(mesh, PartitionSpec())
    rows = sds.shape[0]
    if rows % (num_processes * num_local) == 0:
      return NamedSharding(mesh, PartitionSpec(("process", "local")))
    if num_processes > 1 and rows % num_processes == 0:
      return NamedSharding(mesh, PartitionSpec("process"))
    return NamedSharding(mesh, PartitionSpec())

  if isinstance(abstract_state, dict):
    return {
        key: (
            NamedSharding(mesh, PartitionSpec())
            if dequantize.is_companion_scale_key(key, abstract_state)
            else jax.tree.map(leading_axis_sharding, value)
        )
        for key, value in abstract_state.items()
    }
  return jax.tree.map(leading_axis_sharding, abstract_state)


_FALLBACK_MAX_BYTES_PER_HOST = 4 << 30


def _load_sharded_via_safetensors_reader(path, devices, num_groups=None, max_bytes_per_host=_FALLBACK_MAX_BYTES_PER_HOST):
  """Loads a SafeTensors checkpoint via `SafetensorsReader` when Orbax rejects header dtypes (e.g. `F8_E8M0`)."""
  multi_process = num_groups is not None or jax.process_count() > 1
  num_hosts = num_groups if num_groups is not None else max(jax.process_count(), 1)
  with safetensors_reader.SafetensorsReader(path) as reader:
    abstract_state = dict(reader.metadata)
    shardings = (
        _leading_axis_shardings(abstract_state, devices, num_groups)
        if multi_process
        else _single_process_shardings(abstract_state, devices)
    )
    request = {
        key: jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings[key])
        for key, sds in abstract_state.items()
    }
    if not request or max_bytes_per_host is None:
      return reader.read(request)

    def _host_bytes(key: str) -> int:
      sds = abstract_state[key]
      nbytes = math.prod(sds.shape) * np.dtype(sds.dtype).itemsize
      spec = shardings[key].spec if isinstance(shardings[key], NamedSharding) else ()
      return nbytes // num_hosts if (multi_process and spec and spec[0] is not None) else nbytes

    # Read in bounded deterministic batches so every host calls `reader.read` (`fetch` + `unpack`)
    # in lockstep without allocating host/HBM buffers for all checkpoint shards at once.
    hf_state = {}
    batch_keys, batch_bytes = [], 0
    for key in request:
      size = _host_bytes(key)
      if batch_keys and batch_bytes + size > max_bytes_per_host:
        hf_state.update(reader.read({k: request[k] for k in batch_keys}))
        batch_keys, batch_bytes = [], 0
      batch_keys.append(key)
      batch_bytes += size
    if batch_keys:
      hf_state.update(reader.read({k: request[k] for k in batch_keys}))
    return hf_state


def _single_process_shardings(abstract_state, devices):
  """Constructs single-process shardings while keeping quantized weights axis-0 sharded and scales replicated."""
  if not isinstance(abstract_state, dict):
    return sharding_utils.construct_maximal_shardings(abstract_state, devices=devices)
  quant_or_scale_keys = {
      key
      for key, sds in abstract_state.items()
      if dequantize.is_companion_scale_key(key, abstract_state)
      or (
          dequantize.is_quantized_weight_dtype(getattr(sds, "dtype", None))
          and dequantize.find_scale_key(key, abstract_state) is not None
      )
  }
  if not quant_or_scale_keys:
    return sharding_utils.construct_maximal_shardings(abstract_state, devices=devices)
  leading = _leading_axis_shardings({k: abstract_state[k] for k in quant_or_scale_keys}, devices)
  other_state = {k: v for k, v in abstract_state.items() if k not in quant_or_scale_keys}
  other = sharding_utils.construct_maximal_shardings(other_state, devices=devices) if other_state else {}
  return {k: leading[k] if k in leading else other[k] for k in abstract_state}


def load_sharded_hf_state(path, num_groups=None):
  """Loads HF state sharded across the TPU mesh to avoid host OOM.

  On one process, each tensor is sharded as much as possible (except block-quantized weights and
  their companion scales, which stay leading-axis sharded and replicated so per-shard dequantization
  never splits inner dimensions). On several processes, each tensor is sharded along its leading
  axis only; see `_leading_axis_shardings`.

  This holds the whole HF checkpoint in HBM at once. With `safetensors_dynamic_streaming=True`,
  `load_safetensors_dynamic_state` streams it with `hf_streaming_load` instead.
  Note: When Orbax's SafeTensors reader rejects headers with unsupported dtypes such as `F8_E8M0`
  (e.g. DeepSeek-V4 block scales), this function falls back to `SafetensorsReader`.

  Args:
    path: A directory path (either local or GCS starting with gs://) containing
      the .safetensors files (e.g., "gs://my-bucket/hf_cache/model_id" or
      "/path/to/safetensors_directory/"). If a Hugging Face Hub ID was used,
      it should already be cached/downloaded to GCS before calling this
      function.
    num_groups: If set, shards as on several processes, with the devices split into this many groups by
      id. For tests on one process.

  Returns:
    The loaded Hugging Face state dictionary mapping parameter names to
    JAX arrays.
  """
  t0 = time.time()
  multi_process = num_groups is not None or jax.process_count() > 1
  current_global_devices = jax.devices()
  # Leading-axis shards are contiguous, so never read whole spans: with more than 1024 tensors in a file
  # (fine-grained MoE), Orbax's automatic choice would otherwise read whole spans again.
  safetensors_options = ocp_v1.options.SafetensorsOptions(max_over_read_ratio=1.0) if multi_process else None
  context = ocp_v1.Context(
      checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS, safetensors_options=safetensors_options
  )
  with context, safetensors_reader.no_gcs_read_ahead():
    try:
      metadata = ocp_v1.pytree_metadata(path)
    except ValueError as e:
      if "Unsupported dtype in SafeTensors header" not in str(e):
        raise
      max_logging.log(f"Orbax SafeTensors metadata rejected header ({e}); reading with SafetensorsReader...")
      hf_state = _load_sharded_via_safetensors_reader(path, current_global_devices, num_groups)
      max_logging.log(f"load_sharded_hf_state (SafetensorsReader fallback) took {time.time() - t0:.2f}s")
      return hf_state
    simple_abstract_state = metadata.metadata

    # Distributed Sharded Download: Tell JAX to shard the HF Safetensors download
    # across the entire TPU mesh to avoid Host OOM.
    if multi_process:
      shardings = _leading_axis_shardings(simple_abstract_state, current_global_devices, num_groups)
    else:
      shardings = _single_process_shardings(simple_abstract_state, current_global_devices)

    def combine_sharding(sds, single_sharding):
      return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=single_sharding)

    sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)

    max_logging.log("Reading raw Safetensors into memory (Distributed Sharded GCS Download)...")
    hf_state = ocp_v1.load_pytree(path, sharded_abstract_state)
    max_logging.log(f"load_sharded_hf_state took {time.time() - t0:.2f}s")
    return hf_state


# Prefix renames from the older multimodal checkpoint layout (e.g. `google/gemma-3-4b-it`) to the names
# transformers uses after loading, which the param mappings use. Same as transformers' llava renames.
_LEGACY_HF_PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head.", "lm_head."),
    ("vision_tower.", "model.vision_tower."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
)


def _hf_keys_in(hf_source):
  """Yields every HF key in a (possibly nested) list or tuple of HF keys."""
  if isinstance(hf_source, (list, tuple)):
    for item in hf_source:
      yield from _hf_keys_in(item)
  elif hf_source is not None:
    yield hf_source


def _atomic_hf_sources(hf_source):
  """Yields every atomic `str` or `tuple` HF source inside a (possibly nested) list."""
  if isinstance(hf_source, list):
    for item in hf_source:
      yield from _atomic_hf_sources(item)
  elif hf_source is not None:
    yield hf_source


def rename_legacy_hf_keys(hf_state, param_map_mt_to_hf):
  """Renames HF tensors stored under a legacy prefix to the names the mapping uses.

  A tensor is renamed only if the mapping does not use its stored name but uses the renamed one
  (or if it is a companion `*_scale_inv` / `*.scale` tensor of a mapped quantized weight), so
  checkpoints already in the mapping's layout are unchanged.
  """
  mapped = set(_hf_keys_in(list(param_map_mt_to_hf.values())))

  def should_rename(old_key: str, new_key: str) -> bool:
    if new_key in mapped:
      return True
    new_weight = dequantize.companion_weight_key(new_key)
    old_weight = dequantize.companion_weight_key(old_key)
    if new_weight is None or new_weight not in mapped:
      return False
    w_val = hf_state.get(new_weight, hf_state.get(old_weight))
    return dequantize.is_quantized_weight_dtype(getattr(w_val, "dtype", None))

  for key in [k for k in hf_state if k not in mapped]:
    for old, new in _LEGACY_HF_PREFIXES:
      if key.startswith(old):
        candidate = new + key[len(old) :]
        if should_rename(key, candidate):
          hf_state[candidate] = hf_state.pop(key)
          break
  return hf_state


def transform_hf_state_to_mt_state(hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config):
  """Transforms HF state into MaxText state by applying param mappings and mathematical hooks.

  `hf_streaming_load` (used with `safetensors_dynamic_streaming=True`) must produce the same weights.
  """
  t0 = time.time()

  # An HF tensor may feed several MaxText leaves (gemma4 maps `experts.gate_up_proj` to both `wi_0`
  # and `wi_1`), so each tensor is dropped only after its last active use.
  flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")
  flat_restored = flat_target.copy()

  mapped_count = 0
  keys_missed = []
  max_logging.log("Starting fast in-memory Distributed Transformations...")

  def target_name(mt_key):
    mt_name = mt_key.replace("params-", "").replace("-", ".")
    candidates = [mt_name, f"params.{mt_name}", mt_key.replace("-", ".")]
    # A key of another variable collection (DeepSeek `MoEBiasVar-...`, `Tid2EidVar-...`) names a
    # leaf that the weights state holds without the collection.
    if tensor_handling.is_variable_collection_key(mt_key):
      candidates.append(mt_key.partition("-")[2].replace("-", "."))
    return next((name for name in candidates if name in flat_target), None)

  # DeepSeek-V3 maps its gate bias both as a weight (`params-...`) and in `MoEBiasVar`, and both keys
  # name the same leaf. The collection key stacks it along the right axis, so the other is skipped.
  collection_targets = {
      target_name(k)
      for k in param_map_mt_to_hf
      if not isinstance(k, tuple) and tensor_handling.is_variable_collection_key(k)
  }

  active_entries = []
  for mt_key, hf_source in param_map_mt_to_hf.items():
    # A tuple of MaxText keys is a composite key: one HF source split into several leaves.
    mt_keys = mt_key if isinstance(mt_key, tuple) else (mt_key,)
    check_names = [target_name(k) for k in mt_keys]
    if None in check_names:
      keys_missed.extend(k.replace("params-", "").replace("-", ".") for k, n in zip(mt_keys, check_names) if n is None)
      continue
    if (
        not isinstance(mt_key, tuple)
        and not tensor_handling.is_variable_collection_key(mt_key)
        and check_names[0] in collection_targets
    ):
      continue
    active_entries.append((mt_key, mt_keys, check_names, hf_source))

  remaining_uses = collections.Counter(key for *_, hf_source in active_entries for key in _hf_keys_in([hf_source]))

  def tensor_getter(key):
    remaining_uses[key] -= 1
    return hf_state.pop(key) if remaining_uses[key] <= 0 else hf_state[key]

  fp8_block_size = dequantize.resolve_fp8_block_size(maxtext_config)

  mapped_hf_keys = {key for *_, hf_source in active_entries for key in _hf_keys_in([hf_source])}
  scale_for = {}
  scale_uses = collections.Counter()
  for mt_key, _, check_names, hf_source in active_entries:
    target_dtypes = [
        flat_target[name].dtype if hasattr(flat_target[name], "dtype") else np.float32 for name in check_names
    ]
    for source in _atomic_hf_sources(hf_source):
      for key, scale_key in dequantize.resolve_source_scales(
          source, target_dtypes, hf_state, mapped_hf_keys, mt_key
      ).items():
        scale_for[key] = scale_key
        scale_uses[scale_key] += 1

  def dequantizing_getter(key):
    weight = tensor_getter(key)
    if key in scale_for:
      scale_key = scale_for[key]
      scale_uses[scale_key] -= 1
      scale = hf_state.pop(scale_key) if scale_uses[scale_key] <= 0 else hf_state[scale_key]
      return dequantize.dequantize_weight(weight, scale, block_size=fp8_block_size)
    return weight

  for mt_key, mt_keys, check_names, hf_source in active_entries:
    leaves = [flat_target[name] for name in check_names]
    hook_fn = hook_fn_map_mt.get(mt_key)
    if isinstance(mt_key, tuple):
      target_leaf = tensor_handling.composite_target_leaf(leaves)
      hook_fn = tensor_handling.composite_hook_fn(hook_fn, len(leaves))
    else:
      target_leaf = leaves[0]

    load_fn = get_hf_loading_function(
        hf_source,
        dequantizing_getter,
        hook_fn,
        target_leaf,
        maxtext_config,
        mt_keys[0],
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    loaded = load_fn()
    if isinstance(mt_key, tuple):
      for i, (name, leaf) in enumerate(zip(check_names, leaves)):
        part = loaded[..., i]
        flat_restored[name] = jax.device_put(part, leaf.sharding) if hasattr(leaf, "sharding") else part
    else:
      flat_restored[check_names[0]] = loaded

    max_logging.log(f"Transformed {', '.join(check_names)} from {hf_source} in {time.time() - t_layer:.4f}s")
    mapped_count += 1

  if mapped_count == 0:
    max_logging.log(f"All transformations missed! Sample missed mt_names: {keys_missed[:5]}")
    max_logging.log(f"Sample flat_target keys: {list(flat_target.keys())[:5]}")

  max_logging.log(f"Successfully mapped {mapped_count} parameters.")
  restored = flax.traverse_util.unflatten_dict(flat_restored, sep=".")

  max_logging.log(f"transform_hf_state_to_mt_state took {time.time() - t0:.2f}s")

  # A Linen tree carries the `params` collection, and maybe others beside it (e.g. `MoEBiasVar`
  # and `Tid2EidVar`): return it as is. An NNX pure dict has bare weights: wrap them once.
  return restored if "params" in restored else {"params": restored}


def load_safetensors_dynamic_state(path, abstract_params, maxtext_config):
  """Loads HF safetensors and converts them into MaxText weights on device.

  By default, reads every tensor sharded across `jax.devices()`, then maps, dequantizes (when stored
  as block-scaled `FP8` or packed `MXFP4`), transforms, and stacks the tensors into the target
  layout. With `maxtext_config.safetensors_dynamic_streaming`, streams the checkpoint a few decoder
  layers at a time instead (see `hf_streaming_load`).

  Args:
    path: The safetensors source: a `gs://` or local directory, or an HF repo ID with or without the
      `hf://` prefix. If empty, `HF_IDS[maxtext_config.model_name]` is used. An HF repo is first
      copied to `<base_output_directory>/hf_cache/` by host 0.
    abstract_params: The target weights: Linen's `params` collection or the NNX params state, not the
      full train state.
    maxtext_config: The MaxText config.

  Returns:
    A tuple `(None, {"params": weights})`. Each weight has the dtype and sharding of its leaf in
    `abstract_params`.

  Raises:
    ValueError: If `maxtext_config` is None, if `path` is empty and `HF_IDS` has no entry for the
      model, or if an HF repo is used without a `gs://` `base_output_directory`.
  """
  if maxtext_config is None:
    raise ValueError("maxtext_config must be provided for safetensors_dynamic loading.")

  model_name = maxtext_config.model_name

  if not path:
    # Use the exact name: -Instruct variants have their own HF_IDS entries, and stripping the suffix
    # would load the base model's weights.
    if model_name not in maxtext_globals.HF_IDS:
      raise ValueError(
          f"No HF repo for model_name={model_name} in HF_IDS; set load_parameters_path to the safetensors"
          " location (a gs:// or local directory, or hf://<repo>)."
      )
    path = maxtext_globals.HF_IDS[model_name]

  if path.startswith("hf://"):
    path = path[5:]

  if not path.startswith("gs://") and not os.path.isdir(path):
    repo_id = path
    host_id = jax.process_index()

    if hasattr(maxtext_config, "base_output_directory") and maxtext_config.base_output_directory.startswith("gs://"):
      gcs_cache_dir = f"{maxtext_config.base_output_directory}/hf_cache/{repo_id.replace('/', '_')}"
      path = gcs_cache_dir

      # Only Host 0 downloads to the shared GCS cache
      if host_id == 0:
        max_logging.log("Dynamic HF Hub Fast DL: Host 0 is downloading to shared GCS" f" Cache: {gcs_cache_dir}")
        t_gcs_start = time.time()
        # Only host 0 calls the HF API: anonymous calls from every host of a large job are rate-limited (HTTP 429).
        fs = huggingface_hub.HfFileSystem(token=maxtext_config.hf_access_token)
        files = fs.glob(f"{repo_id}/*.safetensors")

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

      # Global barrier: all hosts wait for Host 0 to finish downloading to the
      # shared GCS bucket
      max_logging.log(f"Host {host_id} waiting for GCS cache at {gcs_cache_dir} to be" " populated by Host 0...")
      jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")
      max_logging.log(f"Host {host_id} detected GCS cache is ready!")

    else:
      raise ValueError("base_output_directory with gs:// prefix is required for " "huggingface downloads.")

  t_total = time.time()
  param_map_mt_to_hf, hook_fn_map_mt = get_hf_config_and_mappings(maxtext_config)
  streaming = getattr(maxtext_config, "safetensors_dynamic_streaming", False) is True
  num_steps = 2 if streaming else 3
  max_logging.log(f"[1/{num_steps}] Mappings derived in {time.time() - t_total:.2f}s")

  target_tree = abstract_params.to_pure_dict() if isinstance(abstract_params, nnx.State) else abstract_params

  if streaming:
    t1 = time.time()
    # Read, convert, and place the weights a few decoder layers at a time. The next
    # read call is fetched while the current one is converted, so at most two calls
    # of HF tensors (about 4 GiB per host each) are in host RAM / HBM at once.
    restored_params = hf_streaming_load.load_hf_params_streaming(
        path,
        target_tree,
        param_map_mt_to_hf,
        hook_fn_map_mt,
        maxtext_config,
        rename_hf_keys=functools.partial(rename_legacy_hf_keys, param_map_mt_to_hf=param_map_mt_to_hf),
    )
    max_logging.log(f"[2/2] Streamed and converted HF weights in {time.time() - t1:.2f}s")
  else:
    t1 = time.time()
    hf_state = rename_legacy_hf_keys(load_sharded_hf_state(path), param_map_mt_to_hf)
    max_logging.log(f"[2/3] Distributed Sharded GCS load completed in {time.time() - t1:.2f}s")

    t2 = time.time()
    # Transform Hugging Face weight tensors on-the-fly into MaxText format
    # in-memory. This is done in-memory on each host, sharded across the mesh.
    restored_params = transform_hf_state_to_mt_state(
        hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
    )
    max_logging.log(f"[3/3] CPU Transformations completed in {time.time() - t2:.2f}s")
  max_logging.log(f"Total safetensors_dynamic duration: {time.time() - t_total:.2f}s")

  return None, restored_params
