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

"""Input pipeline for synthetic dataset."""

from collections.abc import Callable
from typing import Any

import hashlib
import os

import numpy as np

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import multihost_dataloading
from maxtext.configs import pyconfig
from maxtext.utils.diloco_sharding import reshape_first_axis_with_diloco
from maxtext.utils import sharding


def _make_packed_segment_ids(batch_size: int, seq_len: int, max_segments_per_seq: int) -> jnp.ndarray:
  """Creates synthetic segment IDs simulating packed sequences.

  Splits each sequence into a random number (2..max_segments_per_seq) of
  varying-length segments, assigning each a sequential integer segment ID
  starting from 1 via a fully vectorized JAX implementation.

  Args:
    batch_size: Number of sequences.
    seq_len: Total sequence length.
    max_segments_per_seq: Maximum number of segments per sequence (>= 2).

  Returns:
    Integer array [batch_size, seq_len] with sequential segment IDs starting at 1.
  """
  if max_segments_per_seq < 2 or seq_len < 2:
    return jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  max_segs = min(max_segments_per_seq, seq_len)
  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key, 2)

  # Sample random number of segments per row in [2, max_segs].
  n_segs = jax.random.randint(k1, shape=(batch_size, 1), minval=2, maxval=max_segs + 1)
  num_splits = n_segs - 1

  # Assign random uniform scores to positions 1..seq_len-1 and rank them.
  scores = jax.random.uniform(k2, shape=(batch_size, seq_len - 1))
  ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)

  # A position is a split boundary if its rank < num_splits for that row.
  split_mask = ranks < num_splits

  # Position 0 is always the start of segment 1.
  first_pos = jnp.ones((batch_size, 1), dtype=jnp.bool_)
  is_boundary = jnp.concatenate([first_pos, split_mask], axis=-1)

  # Cumulative sum across sequence length gives sequential segment IDs: 1, 2, ...
  return jnp.cumsum(is_boundary, axis=-1, dtype=jnp.int32)


class SyntheticDataIterator:
  """Creates a synthetic data iterator for performance testing work"""

  data_generator: Callable[[pyconfig.HyperParameters, tuple[Any, ...]], dict]

  def __init__(self, config, mesh, is_training: bool = True):
    self.mesh = mesh
    self.config = config
    batch_size = config.global_batch_size_to_load if is_training else config.global_batch_size_to_load_eval
    rules = None if is_training else config.logical_axis_rules_for_eval
    data_pspec_shardings = sharding.get_input_data_sharding(config, mesh, rules=rules)

    is_ds41 = (
        bool(getattr(config, "use_multimodal", False))
        and getattr(getattr(config, "vision_encoder_block", None), "value", getattr(config, "vision_encoder_block", None))
        == "deepseek41"
    )

    if is_ds41:
      if getattr(config, "enable_diloco", False) and len(data_pspec_shardings.spec) >= 2:
        img_spec = jax.sharding.PartitionSpec(
            data_pspec_shardings.spec[0], data_pspec_shardings.spec[1], None, None, None
        )
      elif len(data_pspec_shardings.spec) >= 1:
        img_spec = jax.sharding.PartitionSpec(data_pspec_shardings.spec[0], None, None, None)
      else:
        img_spec = jax.sharding.PartitionSpec()
      img_shd = jax.sharding.NamedSharding(mesh, img_spec)
      out_shardings = {
          "inputs": data_pspec_shardings,
          "inputs_position": data_pspec_shardings,
          "inputs_segmentation": data_pspec_shardings,
          "targets": data_pspec_shardings,
          "targets_position": data_pspec_shardings,
          "targets_segmentation": data_pspec_shardings,
          "images": img_shd,
      }
    else:
      out_shardings = data_pspec_shardings

    self.data_generator = jax.jit(
        SyntheticDataIterator.raw_generate_synthetic_data, out_shardings=out_shardings, static_argnums=0
    )

    # When several data replicas share this process (threaded DiLoCo), `global_batch_size_to_load` is one replica's
    # batch. Build the batch of all replicas and keep this replica's rows, so replica i gets the same rows as replica i
    # of SPMD DiLoCo, which splits one global batch along its first axis. With one replica this is the full batch.
    replica_index, num_replicas = input_pipeline_utils.get_data_replica(config)
    all_replicas_batch_size = batch_size * num_replicas
    tokens = jax.random.randint(
        jax.random.PRNGKey(0),
        (all_replicas_batch_size, config.max_target_length + 1),
        0,
        config.vocab_size,
        dtype=jnp.int32,
    )

    sequence_positions = jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
    batch_positions = jnp.broadcast_to(sequence_positions, (all_replicas_batch_size, config.max_target_length + 1))
    if config.packing:
      segmentation = _make_packed_segment_ids(
          all_replicas_batch_size,
          config.max_target_length,
          config.max_segments_per_seq,
      )
    else:
      segmentation = jnp.ones((all_replicas_batch_size, config.max_target_length), dtype=jnp.int32)
    rows = slice(replica_index * batch_size, (replica_index + 1) * batch_size)
    tokens, batch_positions, segmentation = tokens[rows], batch_positions[rows], segmentation[rows]
    if is_training and config.logical_global_batch_size_to_load is not None:
      # With `elastic_global_batch_size_mode='constant_padded'`, mask out rows beyond the logical global batch.
      segmentation = segmentation.at[config.logical_global_batch_size_to_load :].set(0)

    targets_segmentation = segmentation
    images = None
    if is_ds41:
      vocab_size = int(config.vocab_size)
      image_token_id = int(config.ds41_image_token_id)
      if image_token_id <= 0 or image_token_id >= vocab_size:
        raise ValueError(
            f"DeepSeek-V4.1 multimodal synthetic data requires 0 < ds41_image_token_id < vocab_size "
            f"({vocab_size}), got ds41_image_token_id={image_token_id}."
        )
      if bool(config.packing) and int(config.max_segments_per_seq) >= 2:
        raise ValueError(
            "DeepSeek-V4.1 multimodal synthetic data uses a single fixed-offset image span per sequence and "
            f"requires max_segments_per_seq < 2 when packing=True, got max_segments_per_seq={config.max_segments_per_seq}."
        )
      num_images = max(1, int(config.max_num_images_per_example))
      n_vit_h, n_vit_w = int(config.ds41_vision_grid[0]), int(config.ds41_vision_grid[1])
      ds = int(config.ds41_vision_downsample_ratio)
      span_len = ((n_vit_h + (-n_vit_h) % ds) // ds) * (((n_vit_w + (-n_vit_w) % ds) // ds) + 1) + 2
      total_span_len = num_images * span_len
      if 8 + total_span_len >= int(config.max_target_length):
        raise ValueError(
            f"DeepSeek-V4.1 multimodal synthetic data requires max_target_length > 8 + num_images * span_len "
            f"({8 + total_span_len}), got max_target_length={config.max_target_length}."
        )
      tokens = jnp.where(tokens == image_token_id, (image_token_id + 1) % vocab_size, tokens)
      tokens = tokens.at[:, 8 : 8 + total_span_len].set(image_token_id)
      seg_shape = (batch_size, config.max_target_length)
      segmentation = jnp.ones(seg_shape, dtype=jnp.int32)
      targets_segmentation = jnp.ones(seg_shape, dtype=jnp.int32).at[:, 7 : 7 + total_span_len].set(0)
      if is_training and getattr(config, "logical_global_batch_size_to_load", None) is not None:
        segmentation = segmentation.at[config.logical_global_batch_size_to_load :].set(0)
        targets_segmentation = targets_segmentation.at[config.logical_global_batch_size_to_load :].set(0)
      patch_dim = int(config.num_channels_for_vit) * (int(config.patch_size_for_vit) ** 2)
      images = jax.random.normal(
          jax.random.PRNGKey(1),
          (batch_size, num_images, n_vit_h * n_vit_w, patch_dim),
          dtype=jnp.float32,
      )

    if "DS41_EXPORT_PINNED_BATCH_NPZ" in os.environ and os.environ["DS41_EXPORT_PINNED_BATCH_NPZ"].strip():
      export_path = os.environ["DS41_EXPORT_PINNED_BATCH_NPZ"].strip()

      def _gather_global(arr):
        if arr is None:
          return None
        np_arr = np.asarray(jax.device_get(arr))
        if jax.process_count() > 1:
          return multihost_utils.process_allgather(np_arr, tiled=True)
        return np_arr

      g_tokens = _gather_global(tokens)[:16]
      g_positions = _gather_global(batch_positions)[:16]
      g_segmentation = _gather_global(segmentation)[:16]
      g_targets_seg = _gather_global(targets_segmentation)[:16]
      g_images = _gather_global(images)[:16] if images is not None else None

      if jax.process_index() == 0:
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        save_dict = {
            "tokens": g_tokens,
            "batch_positions": g_positions,
            "segmentation": g_segmentation,
            "targets_segmentation": g_targets_seg,
        }
        if g_images is not None:
          save_dict["images"] = g_images
        if not export_path.endswith(".npz"):
          export_path_actual = f"{export_path}.npz"
        else:
          export_path_actual = export_path
        tmp_export_path = f"{export_path_actual}.tmp.{os.getpid()}.npz"
        np.savez(tmp_export_path, **save_dict)
        os.replace(tmp_export_path, export_path_actual)

    if "DS41_PINNED_BATCH_NPZ" in os.environ and os.environ["DS41_PINNED_BATCH_NPZ"].strip():
      load_path = os.environ["DS41_PINNED_BATCH_NPZ"].strip()
      if not os.path.exists(load_path):
        raise FileNotFoundError(f"DS41_PINNED_BATCH_NPZ file not found: {load_path}")
      with open(load_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
      if "DS41_PINNED_BATCH_SHA256" in os.environ and os.environ["DS41_PINNED_BATCH_SHA256"].strip():
        expected_sha = os.environ["DS41_PINNED_BATCH_SHA256"].strip()
        if actual_sha != expected_sha:
          raise ValueError(
              f"DS41_PINNED_BATCH_SHA256 fail-closed mismatch: expected {expected_sha}, got {actual_sha} for {load_path}"
          )
      micro_batch = config.micro_batch_size_to_train_on
      if micro_batch is not None and micro_batch != 16:
        raise AssertionError(f"DS41_PINNED_BATCH_NPZ requires micro_batch_size_to_train_on == 16, got {micro_batch}")
      if config.global_batch_size_to_load < 16:
        raise AssertionError(
            f"DS41_PINNED_BATCH_NPZ requires global_batch_size_to_load >= 16, got {config.global_batch_size_to_load}"
        )
      with np.load(load_path) as loaded:
        np_tokens = loaded["tokens"]
        np_positions = loaded["batch_positions"] if "batch_positions" in loaded else loaded["positions"]
        np_segmentation = loaded["segmentation"]
        np_targets_seg = loaded["targets_segmentation"]
        np_images = loaded["images"] if "images" in loaded else None

      if np_tokens.shape[0] != 16:
        raise ValueError(f"DS41_PINNED_BATCH_NPZ expects 16 rows, got {np_tokens.shape[0]}")

      g_load = config.global_batch_size_to_load
      tile_factor = (g_load + 15) // 16
      np_tokens = np.tile(np_tokens, (tile_factor, 1))[:g_load]
      np_positions = np.tile(np_positions, (tile_factor, 1))[:g_load]
      np_segmentation = np.tile(np_segmentation, (tile_factor, 1))[:g_load]
      np_targets_seg = np.tile(np_targets_seg, (tile_factor, 1))[:g_load]
      if np_images is not None:
        np_images = np.tile(np_images, (tile_factor,) + (1,) * (np_images.ndim - 1))[:g_load]

      def _to_global_sharded_array(host_np_arr, shd):
        if host_np_arr is None:
          return None
        g_shape = host_np_arr.shape
        idx_map = shd.addressable_devices_indices_map(g_shape)
        single_dev_arrays = [jax.device_put(host_np_arr[idx_map[d]], d) for d in shd.addressable_devices]
        return jax.make_array_from_single_device_arrays(g_shape, shd, single_dev_arrays)

      img_shd = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(data_pspec_shardings.spec[0], None, None, None)
      )
      tokens = _to_global_sharded_array(np_tokens, data_pspec_shardings)
      batch_positions = _to_global_sharded_array(np_positions, data_pspec_shardings)
      segmentation = _to_global_sharded_array(np_segmentation, data_pspec_shardings)
      targets_segmentation = _to_global_sharded_array(np_targets_seg, data_pspec_shardings)
      images = _to_global_sharded_array(np_images, img_shd) if np_images is not None else None

    self.data = (tokens, batch_positions, segmentation, targets_segmentation, images)

  def reset(self):
    pass  # Synthetic data is stateless; nothing to reset.

  def __iter__(self):
    return self

  def __next__(self):
    with self.mesh:
      return self.data_generator(self.config, self.data)  # pylint: disable=not-callable

  @staticmethod
  def raw_generate_synthetic_data(config: pyconfig.HyperParameters, data):
    """Generates a single batch of synthetic data"""
    tokens, positions, segmentation, targets_segmentation, images = data

    output = {}
    output["inputs"] = tokens[:, :-1]
    output["inputs_position"] = positions[:, :-1]
    output["inputs_segmentation"] = segmentation
    output["targets"] = tokens[:, 1:]
    output["targets_position"] = positions[:, 1:]
    output["targets_segmentation"] = targets_segmentation
    if images is not None:
      output["images"] = images
    if config.enable_diloco:
      output = reshape_first_axis_with_diloco(config.num_diloco_replicas, output)
    return output


class PlaceHolderDataIterator:
  """Creates a place holder synthetic data iterator for loading on subset of hosts"""

  def __init__(self, config: pyconfig.HyperParameters, mesh, is_training: bool = True):
    self.mesh = mesh
    dataset = PlaceHolderDataIterator.get_place_holder_synthetic_data(config, is_training=is_training)
    self.data_generator = multihost_dataloading.MultiHostDataLoadIterator(dataset, self.mesh)

  def __iter__(self):
    return self.data_generator

  def __next__(self):
    return next(self.data_generator)

  def reset(self):
    pass

  @staticmethod
  def get_place_holder_synthetic_data(config: pyconfig.HyperParameters, is_training: bool = True):
    """fill negative value in synthetic data

    Args:
      config: hyperparameters, used for the loaded batch size and the sequence length.
      is_training: whether this placeholder stands in for the train iterator. The eval
        iterator loads `global_batch_size_to_load_eval` rows instead, and every host has
        to contribute the same number of rows to the global batch.
    """
    global_batch_size_to_load = config.global_batch_size_to_load if is_training else config.global_batch_size_to_load_eval
    batch_size = global_batch_size_to_load // jax.process_count()
    neg_ones = np.full((batch_size, config.max_target_length), -1, dtype=np.int32)
    batch = {
        "inputs": neg_ones,
        "inputs_position": neg_ones,
        "inputs_segmentation": neg_ones,
        "targets": neg_ones,
        "targets_position": neg_ones,
        "targets_segmentation": neg_ones,
    }

    def infinite_iterator():
      while True:
        yield batch

    return infinite_iterator()
