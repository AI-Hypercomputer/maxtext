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

import numpy as np

import jax
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
  # Pin threefry2x32: base.yml defaults prng_impl="unsafe_rbg", whose HLO rng-bit-generator
  # partitions per shard on Pathways TPU and diverges between shard_mode="auto" and "explicit".
  key = jax.random.key(42, impl="threefry2x32")
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


def _pad_replica_blocks(rows: jnp.ndarray, num_replicas: int, batch_size: int) -> jnp.ndarray:
  """Zero-pads each replica's block of `rows` to `batch_size` rows.

  Args:
    rows: Array [num_replicas * logical_batch_size, ...] holding the rows of each replica consecutively.
    num_replicas: Number of data replicas.
    batch_size: Rows per replica after padding (>= logical_batch_size).

  Returns:
    Array [num_replicas * batch_size, ...]; replica i's rows start at row i * batch_size, followed by zero rows.
  """
  logical_batch_size = rows.shape[0] // num_replicas
  if logical_batch_size == batch_size:
    return rows
  blocks = rows.reshape(num_replicas, logical_batch_size, *rows.shape[1:])
  blocks = jnp.pad(
      blocks,
      [(0, 0), (0, batch_size - logical_batch_size)] + [(0, 0)] * (rows.ndim - 1),
  )
  return blocks.reshape(num_replicas * batch_size, *rows.shape[1:])


def _validate_ds41_synthetic_config(config) -> tuple[int, int, int, int, int, int]:
  """Validates DeepSeek-V4.1 synthetic multimodal config and returns geometry tuple."""
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
  if bool(getattr(config, "ds41_dynamic_vision_grid", False)):
    p_max = (ds**2) * (int(config.ds41_vision_max_n_token) - 3)
    if n_vit_h * n_vit_w > p_max:
      raise ValueError(
          f"ds41_vision_grid {(n_vit_h, n_vit_w)} ({n_vit_h * n_vit_w} patches) exceeds dynamic P_max={p_max}."
      )
  span_len = ((n_vit_h + (-n_vit_h) % ds) // ds) * (((n_vit_w + (-n_vit_w) % ds) // ds) + 1) + 2
  total_span_len = num_images * span_len
  if 8 + total_span_len >= int(config.max_target_length):
    raise ValueError(
        f"DeepSeek-V4.1 multimodal synthetic data requires max_target_length > 8 + num_images * span_len "
        f"({8 + total_span_len}), got max_target_length={config.max_target_length}."
    )
  return vocab_size, image_token_id, num_images, n_vit_h, n_vit_w, total_span_len


class SyntheticDataIterator:
  """Creates a synthetic data iterator for performance testing work"""

  data_generator: Callable[..., dict[str, Any]]
  batch: dict[str, Any]

  def __init__(self, config, mesh, is_training: bool = True):
    self.mesh = mesh
    self.config = config
    self.is_training = is_training
    rules = None if is_training else config.logical_axis_rules_for_eval
    data_pspec_shardings = sharding.get_input_data_sharding(config, mesh, rules=rules)
    if sharding.is_ds41_multimodal(config):
      _validate_ds41_synthetic_config(config)
      ds41_keys: tuple[str, ...] = (
          "inputs",
          "inputs_position",
          "inputs_segmentation",
          "targets",
          "targets_position",
          "targets_segmentation",
          "images",
      )
      if bool(getattr(config, "ds41_dynamic_vision_grid", False)):
        ds41_keys = ds41_keys + ("image_masks", "image_grids")
      out_shardings: Any = sharding.get_input_batch_sharding(
          config,
          data_pspec_shardings,
          batch_keys=ds41_keys,
      )
    else:
      out_shardings = data_pspec_shardings
    # Generate the (static) synthetic batch once, directly in its sharded layout. Building it unsharded on
    # local_devices()[0] each step cost 1.5-12 GiB of HBM on that device and stalled step 0 on 512+ chips.
    mesh_ctx = jax.set_mesh(self.mesh) if isinstance(self.mesh, jax.sharding.Mesh) else self.mesh
    with mesh_ctx:
      self.data_generator = jax.jit(
          SyntheticDataIterator.raw_generate_synthetic_data,
          in_shardings=(),
          out_shardings=out_shardings,
          static_argnums=(0, 1),
      )
      self.batch = self.data_generator(self.config, is_training)  # pylint: disable=not-callable
      jax.block_until_ready(self.batch)

  def reset(self):
    pass  # Synthetic data is stateless; nothing to reset.

  def __iter__(self):
    return self

  def __next__(self):
    return self.batch

  @staticmethod
  def raw_generate_synthetic_data(config: pyconfig.HyperParameters, is_training: bool = True):
    """Generates a single batch of synthetic data"""
    batch_size = config.global_batch_size_to_load if is_training else config.global_batch_size_to_load_eval
    # When several data replicas share this process (threaded DiLoCo), `global_batch_size_to_load` is one replica's
    # batch. Build the batch of all replicas and keep this replica's rows, so replica i gets the same rows as replica i
    # of SPMD DiLoCo, which splits one global batch along its first axis. With one replica this is the full batch.
    replica_index, num_replicas = input_pipeline_utils.get_data_replica(config)
    all_replicas_batch_size = batch_size * num_replicas
    # With a constant-global-batch `elastic_global_batch_size_mode`, only the logical rows carry data: they are generated
    # exactly as they would be for the unpadded batch of the full job, so a padded run trains on the same rows, and
    # the rows beyond them are zero with zero segmentation, which the loss masks out.
    logical_batch_size = batch_size
    if is_training and config.logical_global_batch_size_to_load is not None:
      logical_batch_size = min(config.logical_global_batch_size_to_load, batch_size)
    # Generate the logical rows of every replica with the shapes the unpadded full job uses, so they are bit-identical
    # to its rows, then zero-pad each replica's block to `batch_size` rows (segmentation 0 masks the padding rows).
    logical_rows = logical_batch_size * num_replicas
    # Pin threefry2x32: base.yml defaults prng_impl="unsafe_rbg", whose HLO rng-bit-generator
    # partitions per shard on Pathways TPU and diverges between shard_mode="auto" and "explicit".
    tokens = jax.random.randint(
        jax.random.key(0, impl="threefry2x32"),
        (logical_rows, config.max_target_length + 1),
        0,
        config.vocab_size,
        dtype=jnp.int32,
    )

    sequence_positions = jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
    batch_positions = jnp.broadcast_to(sequence_positions, (all_replicas_batch_size, config.max_target_length + 1))
    rows = slice(replica_index * batch_size, (replica_index + 1) * batch_size)

    images = None
    image_masks = None
    image_grids = None
    if sharding.is_ds41_multimodal(config):
      vocab_size, image_token_id, num_images, n_vit_h, n_vit_w, total_span_len = _validate_ds41_synthetic_config(config)
      tokens = jnp.where(tokens == image_token_id, (image_token_id + 1) % vocab_size, tokens)
      tokens = tokens.at[:, 8 : 8 + total_span_len].set(image_token_id)
      tokens = _pad_replica_blocks(tokens, num_replicas, batch_size)[rows]
      batch_positions = batch_positions[rows]

      logical_seg_shape = (logical_rows, config.max_target_length)
      segmentation = _pad_replica_blocks(jnp.ones(logical_seg_shape, dtype=jnp.int32), num_replicas, batch_size)[rows]
      targets_segmentation = _pad_replica_blocks(
          jnp.ones(logical_seg_shape, dtype=jnp.int32).at[:, 7 : 7 + total_span_len].set(0),
          num_replicas,
          batch_size,
      )[rows]

      patch_dim = int(config.num_channels_for_vit) * (int(config.patch_size_for_vit) ** 2)
      valid_patches = jax.random.normal(
          jax.random.key(1, impl="threefry2x32"),
          (logical_rows, num_images, n_vit_h * n_vit_w, patch_dim),
          dtype=jnp.float32,
      )
      if bool(getattr(config, "ds41_dynamic_vision_grid", False)):
        ds = int(config.ds41_vision_downsample_ratio)
        p_max = (ds**2) * (int(config.ds41_vision_max_n_token) - 3)
        pad_patches = p_max - (n_vit_h * n_vit_w)
        valid_patches = jnp.pad(valid_patches, [(0, 0), (0, 0), (0, pad_patches), (0, 0)])
        image_masks = _pad_replica_blocks(
            jnp.ones((logical_rows, num_images), dtype=jnp.int32), num_replicas, batch_size
        )[rows]
        logical_grids = jnp.broadcast_to(jnp.array([n_vit_h, n_vit_w], dtype=jnp.int32), (logical_rows, num_images, 2))
        image_grids = _pad_replica_blocks(logical_grids, num_replicas, batch_size)[rows]
      images = _pad_replica_blocks(valid_patches, num_replicas, batch_size)[rows]
    else:
      tokens = _pad_replica_blocks(tokens, num_replicas, batch_size)
      if config.packing:
        segmentation = _make_packed_segment_ids(
            logical_rows,
            config.max_target_length,
            config.max_segments_per_seq,
        )
        segmentation = _pad_replica_blocks(segmentation, num_replicas, batch_size)
      else:
        segmentation = jnp.ones((all_replicas_batch_size, config.max_target_length), dtype=jnp.int32)
      tokens, batch_positions, segmentation = (
          tokens[rows],
          batch_positions[rows],
          segmentation[rows],
      )
      if logical_batch_size < batch_size:
        segmentation = segmentation.at[logical_batch_size:].set(0)
      targets_segmentation = segmentation

    output: dict[str, Any] = {}
    output["inputs"] = tokens[:, :-1]
    output["inputs_position"] = batch_positions[:, :-1]
    output["inputs_segmentation"] = segmentation
    output["targets"] = tokens[:, 1:]
    output["targets_position"] = batch_positions[:, 1:]
    output["targets_segmentation"] = targets_segmentation
    if images is not None:
      output["images"] = images
    if image_masks is not None:
      output["image_masks"] = image_masks
    if image_grids is not None:
      output["image_grids"] = image_grids
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
