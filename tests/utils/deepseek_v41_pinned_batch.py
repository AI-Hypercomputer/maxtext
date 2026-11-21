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

"""Test-side 16-row pinned-batch NPZ export and sharded iterator wrapper for DeepSeek-V4.1."""

from collections.abc import Mapping
import hashlib
import os
import pathlib
from typing import Any

import jax
from jax.experimental import multihost_utils
import numpy as np

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils import sharding


def _gather_host_array(arr: Any, flatten_diloco: bool = False) -> np.ndarray:
  """Fetches `arr` onto host memory without calling `process_allgather` when addressable."""
  if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
    arr = multihost_utils.process_allgather(arr, tiled=True)
  host_arr = np.asarray(jax.device_get(arr))
  if flatten_diloco and host_arr.ndim >= 2:
    host_arr = host_arr.reshape(-1, *host_arr.shape[2:])
  return host_arr


def _to_global_sharded_array(host_np_arr: np.ndarray, shd: jax.sharding.NamedSharding) -> jax.Array:
  """Places host slices onto addressable devices and constructs a global sharded `jax.Array`."""
  g_shape = host_np_arr.shape
  idx_map = shd.addressable_devices_indices_map(g_shape)
  single_dev_arrays = [jax.device_put(host_np_arr[idx_map[d]], d) for d in shd.addressable_devices]
  return jax.make_array_from_single_device_arrays(g_shape, shd, single_dev_arrays)


def export_pinned_batch(batch: Any, path: str | os.PathLike[str]) -> pathlib.Path:
  """Writes the first 16 global rows of `batch` to `path` as an atomic `.npz` archive."""
  if isinstance(batch, Mapping):
    data: Mapping[str, Any] = batch
  elif hasattr(batch, "batch") and isinstance(batch.batch, Mapping):
    data = batch.batch
  else:
    data = next(iter(batch))

  export_path = os.fspath(path).strip()
  if not export_path:
    raise ValueError("export_pinned_batch requires a non-empty path.")
  if not export_path.endswith(".npz"):
    export_path = f"{export_path}.npz"

  ref_key = "tokens" if "tokens" in data else "inputs"
  flatten_diloco = np.ndim(data[ref_key]) == 3

  if "tokens" in data:
    g_tokens = _gather_host_array(data["tokens"], flatten_diloco=flatten_diloco)
  else:
    g_inputs = _gather_host_array(data["inputs"], flatten_diloco=flatten_diloco)
    g_targets = _gather_host_array(data["targets"], flatten_diloco=flatten_diloco)
    g_tokens = np.concatenate([g_inputs, g_targets[:, -1:]], axis=1)

  if g_tokens.shape[0] < 16:
    raise ValueError(f"export_pinned_batch requires at least 16 rows, got {g_tokens.shape[0]}.")
  g_tokens = g_tokens[:16]

  if "batch_positions" in data:
    g_positions = _gather_host_array(data["batch_positions"], flatten_diloco=flatten_diloco)[:16]
  else:
    g_inp_pos = _gather_host_array(data["inputs_position"], flatten_diloco=flatten_diloco)[:16]
    if "targets_position" in data:
      g_last_pos = _gather_host_array(data["targets_position"], flatten_diloco=flatten_diloco)[:16, -1:]
    else:
      g_last_pos = g_inp_pos[:, -1:] + 1
    g_positions = np.concatenate([g_inp_pos, g_last_pos], axis=1)

  seg_key = "segmentation" if "segmentation" in data else "inputs_segmentation"
  g_segmentation = _gather_host_array(data[seg_key], flatten_diloco=flatten_diloco)[:16]
  g_targets_seg = _gather_host_array(data["targets_segmentation"], flatten_diloco=flatten_diloco)[:16]
  g_images = (
      _gather_host_array(data["images"], flatten_diloco=flatten_diloco)[:16]
      if "images" in data and data["images"] is not None
      else None
  )

  if jax.process_index() == 0:
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    save_dict: dict[str, np.ndarray] = {
        "tokens": g_tokens,
        "batch_positions": g_positions,
        "segmentation": g_segmentation,
        "targets_segmentation": g_targets_seg,
    }
    if g_images is not None:
      save_dict["images"] = g_images
    tmp_export_path = f"{export_path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_export_path, **save_dict)
    os.replace(tmp_export_path, export_path)

  return pathlib.Path(export_path)


class PinnedBatchIterator:
  """Loads a 16-row `.npz` batch verified by SHA-256 and tiles it onto `mesh` shardings."""

  batch: dict[str, Any]

  def __init__(
      self,
      inner_or_config: Any,
      mesh: Any,
      npz_path: str | os.PathLike[str],
      sha256: str = "",
      is_training: bool | None = None,
  ):
    if hasattr(inner_or_config, "config") and not hasattr(inner_or_config, "global_batch_size_to_load"):
      self.inner = inner_or_config
      config = inner_or_config.config
      default_is_training = bool(getattr(inner_or_config, "is_training", True))
    else:
      self.inner = None
      config = inner_or_config
      default_is_training = True

    resolved_is_training = default_is_training if is_training is None else bool(is_training)
    self.config = config
    self.mesh = mesh
    self.is_training = resolved_is_training

    load_path = os.fspath(npz_path).strip()
    if not load_path:
      raise ValueError("PinnedBatchIterator requires a non-empty npz_path.")
    expected_sha = (sha256 or "").strip()
    if not expected_sha:
      raise ValueError("PinnedBatchIterator requires a non-empty sha256.")
    if not os.path.exists(load_path):
      raise FileNotFoundError(f"Pinned batch NPZ file not found: {load_path}")
    with open(load_path, "rb") as f:
      actual_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_sha.lower() != expected_sha.lower():
      raise ValueError(
          f"Pinned batch SHA-256 fail-closed mismatch: expected {expected_sha}, got {actual_sha} for {load_path}"
      )

    if resolved_is_training:
      micro_batch = getattr(config, "micro_batch_size_to_train_on", None)
      if micro_batch is not None and micro_batch != 16:
        raise AssertionError(f"PinnedBatchIterator requires micro_batch_size_to_train_on == 16, got {micro_batch}")

    batch_size = config.global_batch_size_to_load if resolved_is_training else config.global_batch_size_to_load_eval
    replica_index, num_replicas = input_pipeline_utils.get_data_replica(config)
    all_replicas_batch_size = batch_size * num_replicas
    if all_replicas_batch_size < 16:
      raise AssertionError(f"PinnedBatchIterator requires loaded batch_size >= 16, got {all_replicas_batch_size}")

    with np.load(load_path) as loaded:
      if "tokens" in loaded:
        np_tokens = loaded["tokens"]
        np_inputs = np_tokens[:, :-1]
        np_targets = np_tokens[:, 1:]
      else:
        np_inputs = loaded["inputs"]
        np_targets = loaded["targets"]

      if "batch_positions" in loaded or "positions" in loaded:
        np_pos_full = loaded["batch_positions"] if "batch_positions" in loaded else loaded["positions"]
        np_inputs_pos = np_pos_full[:, :-1]
        np_targets_pos = np_pos_full[:, 1:]
      else:
        np_inputs_pos = loaded["inputs_position"]
        np_targets_pos = loaded["targets_position"] if "targets_position" in loaded else (np_inputs_pos + 1)

      np_segmentation = loaded["segmentation"] if "segmentation" in loaded else loaded["inputs_segmentation"]
      np_targets_seg = loaded["targets_segmentation"]
      np_images = loaded["images"] if "images" in loaded else None

    if np_inputs.shape[0] != 16:
      raise ValueError(f"PinnedBatchIterator expects 16 rows, got {np_inputs.shape[0]}")

    rows = slice(replica_index * batch_size, (replica_index + 1) * batch_size)
    tile_factor = (all_replicas_batch_size + 15) // 16
    np_inputs = np.tile(np_inputs, (tile_factor, 1))[:all_replicas_batch_size][rows]
    np_targets = np.tile(np_targets, (tile_factor, 1))[:all_replicas_batch_size][rows]
    np_inputs_pos = np.tile(np_inputs_pos, (tile_factor, 1))[:all_replicas_batch_size][rows]
    np_targets_pos = np.tile(np_targets_pos, (tile_factor, 1))[:all_replicas_batch_size][rows]
    np_segmentation = np.tile(np_segmentation, (tile_factor, 1))[:all_replicas_batch_size][rows].copy()
    np_targets_seg = np.tile(np_targets_seg, (tile_factor, 1))[:all_replicas_batch_size][rows].copy()
    if np_images is not None:
      np_images = np.tile(np_images, (tile_factor,) + (1,) * (np_images.ndim - 1))[:all_replicas_batch_size][rows]

    if resolved_is_training and getattr(config, "logical_global_batch_size_to_load", None) is not None:
      logical_bs = config.logical_global_batch_size_to_load
      np_segmentation[logical_bs:] = 0
      np_targets_seg[logical_bs:] = 0

    rules = None if resolved_is_training else config.logical_axis_rules_for_eval
    data_shd = sharding.get_input_data_sharding(config, mesh, rules=rules)
    enable_diloco = bool(getattr(config, "enable_diloco", False))
    img_shd = sharding.get_ds41_image_sharding(data_shd, enable_diloco)

    if enable_diloco:
      num_diloco = int(config.num_diloco_replicas)
      if batch_size % num_diloco != 0:
        raise ValueError(f"Batch dimension {batch_size} is not divisible by num_diloco_replicas {num_diloco}.")
      per_island = batch_size // num_diloco
      np_inputs = np_inputs.reshape(num_diloco, per_island, *np_inputs.shape[1:])
      np_targets = np_targets.reshape(num_diloco, per_island, *np_targets.shape[1:])
      np_inputs_pos = np_inputs_pos.reshape(num_diloco, per_island, *np_inputs_pos.shape[1:])
      np_targets_pos = np_targets_pos.reshape(num_diloco, per_island, *np_targets_pos.shape[1:])
      np_segmentation = np_segmentation.reshape(num_diloco, per_island, *np_segmentation.shape[1:])
      np_targets_seg = np_targets_seg.reshape(num_diloco, per_island, *np_targets_seg.shape[1:])
      if np_images is not None:
        np_images = np_images.reshape(num_diloco, per_island, *np_images.shape[1:])

    batch_out: dict[str, Any] = {
        "inputs": _to_global_sharded_array(np_inputs, data_shd),
        "inputs_position": _to_global_sharded_array(np_inputs_pos, data_shd),
        "inputs_segmentation": _to_global_sharded_array(np_segmentation, data_shd),
        "targets": _to_global_sharded_array(np_targets, data_shd),
        "targets_position": _to_global_sharded_array(np_targets_pos, data_shd),
        "targets_segmentation": _to_global_sharded_array(np_targets_seg, data_shd),
    }
    if np_images is not None:
      batch_out["images"] = _to_global_sharded_array(np_images, img_shd)
    self.batch = batch_out

  def reset(self) -> None:
    if self.inner is not None and hasattr(self.inner, "reset"):
      self.inner.reset()

  def __iter__(self) -> "PinnedBatchIterator":
    return self

  def __next__(self) -> dict[str, Any]:
    return self.batch
