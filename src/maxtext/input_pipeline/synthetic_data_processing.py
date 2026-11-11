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

  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec_shardings = sharding.get_input_data_sharding(config, mesh)
    self.data_generator = jax.jit(
        SyntheticDataIterator.raw_generate_synthetic_data, out_shardings=data_pspec_shardings, static_argnums=0
    )

    is_olmo_mm = getattr(config, "use_multimodal", False) and (
        getattr(getattr(config, "vision_encoder_block", None), "value", str(getattr(config, "vision_encoder_block", "")))
        == "olmo35"
    )
    tok_upper = (
        min(config.vocab_size, int(config.olmo_image_patch_token_id)) if is_olmo_mm else config.vocab_size
    )
    tokens = jax.random.randint(
        jax.random.PRNGKey(0),
        (config.global_batch_size_to_load, config.max_target_length + 1),
        0,
        tok_upper,
        dtype=jnp.int32,
    )
    if is_olmo_mm:
      img_sz = config.image_size_for_vit
      img_h, img_w = tuple(img_sz) if isinstance(img_sz, (list, tuple)) else (int(img_sz), int(img_sz))
      patch = config.patch_size_for_vit
      pool_h, pool_w = (img_h // patch + 1) // 2, (img_w // patch + 1) // 2
      n_crops = int(config.olmo_max_crops_per_seq)
      n_pooled = (
          int(config.olmo_max_pooled_patches_per_seq)
          if getattr(config, "olmo_max_pooled_patches_per_seq", 0) > 0
          else n_crops * pool_h * pool_w
      )
      n_active_pooled = min(n_pooled, max(1, config.max_target_length // 4))
      tokens = tokens.at[:, :n_active_pooled].set(int(config.olmo_image_patch_token_id))

    sequence_positions = jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
    batch_positions = jnp.broadcast_to(
        sequence_positions, (config.global_batch_size_to_load, config.max_target_length + 1)
    )
    if config.packing:
      segmentation = _make_packed_segment_ids(
          config.global_batch_size_to_load,
          config.max_target_length,
          config.max_segments_per_seq,
      )
    else:
      segmentation = jnp.ones((config.global_batch_size_to_load, config.max_target_length), dtype=jnp.int32)
    self.data = (tokens, batch_positions, segmentation)

  def reset(self):
    pass  # Synthetic data is stateless; nothing to reset.

  def __iter__(self):
    return self

  def __next__(self):
    # self.data never changes, so every step's batch is the same array. Reusing the first one skips the
    # per-step reshard (about 30 ms on a v4-128 at seq 4096), which a real loader's prefetch would hide.
    cached = getattr(self, "_batch", None)
    if self.config.synthetic_data_reuse_batch and cached is not None:
      return cached
    with self.mesh:
      batch = self.data_generator(self.config, self.data)  # pylint: disable=not-callable
    if self.config.synthetic_data_reuse_batch:
      self._batch = batch
    return batch

  @staticmethod
  def raw_generate_synthetic_data(config: pyconfig.HyperParameters, data):
    """Generates a single batch of synthetic data"""
    tokens, positions, segmentation = data

    output = {}
    output["inputs"] = tokens[:, :-1]
    output["inputs_position"] = positions[:, :-1]
    output["inputs_segmentation"] = segmentation
    output["targets"] = tokens[:, 1:]
    output["targets_position"] = positions[:, 1:]
    output["targets_segmentation"] = segmentation
    if getattr(config, "use_multimodal", False) and (
        getattr(getattr(config, "vision_encoder_block", None), "value", str(getattr(config, "vision_encoder_block", "")))
        == "olmo35"
    ):
      b_load = config.global_batch_size_to_load
      img_sz = config.image_size_for_vit
      img_h, img_w = tuple(img_sz) if isinstance(img_sz, (list, tuple)) else (int(img_sz), int(img_sz))
      patch = config.patch_size_for_vit
      channels = config.num_channels_for_vit
      n_patches = (img_h // patch) * (img_w // patch)
      pool_h, pool_w = (img_h // patch + 1) // 2, (img_w // patch + 1) // 2
      n_crops = int(config.olmo_max_crops_per_seq)
      n_pooled = (
          int(config.olmo_max_pooled_patches_per_seq)
          if getattr(config, "olmo_max_pooled_patches_per_seq", 0) > 0
          else n_crops * pool_h * pool_w
      )
      pool_size = int(getattr(config, "olmo_connector_pool_size", 4))
      n_active_pooled = min(n_pooled, max(1, config.max_target_length // 4))
      output["images"] = jax.random.normal(
          jax.random.PRNGKey(1),
          (b_load, n_crops, n_patches, channels * patch * patch),
          dtype=config.dtype_mm,
      )
      active_idx = (
          jnp.arange(n_active_pooled * pool_size, dtype=jnp.int32) % max(1, n_crops * n_patches)
      ).reshape(1, n_active_pooled, pool_size)
      if n_pooled > n_active_pooled:
        pad_idx = jnp.full((1, n_pooled - n_active_pooled, pool_size), -1, dtype=jnp.int32)
        base_idx = jnp.concatenate([active_idx, pad_idx], axis=1)
      else:
        base_idx = active_idx
      output["pooled_patches_idx"] = jnp.broadcast_to(base_idx, (b_load, n_pooled, pool_size))
      if bool(getattr(config, "olmo_use_loss_weights", False)):
        patch_tok = int(config.olmo_image_patch_token_id)
        output["loss_weights"] = (output["targets"] != patch_tok).astype(jnp.float32)
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
    ds_type = getattr(getattr(config, "dataset_type", None), "value", str(getattr(config, "dataset_type", "")))
    if ds_type == "olmo_grain":
      batch.pop("targets_position", None)
    if config.use_multimodal and (
        getattr(getattr(config, "vision_encoder_block", None), "value", str(getattr(config, "vision_encoder_block", "")))
        == "olmo35"
    ):
      img_sz = config.image_size_for_vit
      img_h, img_w = tuple(img_sz) if isinstance(img_sz, (list, tuple)) else (int(img_sz), int(img_sz))
      patch = config.patch_size_for_vit
      channels = config.num_channels_for_vit
      n_crops = int(getattr(config, "olmo_max_crops_per_seq", 64))
      n_patches = (img_h // patch) * (img_w // patch)
      pool_h, pool_w = (img_h // patch + 1) // 2, (img_w // patch + 1) // 2
      n_pooled = (
          int(config.olmo_max_pooled_patches_per_seq)
          if getattr(config, "olmo_max_pooled_patches_per_seq", 0) > 0
          else n_crops * pool_h * pool_w
      )
      pool_size = int(getattr(config, "olmo_connector_pool_size", 4))
      mm_np_dtype = jnp.dtype(config.dtype_mm)
      batch["images"] = np.zeros((batch_size, n_crops, n_patches, channels * patch * patch), dtype=mm_np_dtype)
      batch["pooled_patches_idx"] = np.full((batch_size, n_pooled, pool_size), -1, dtype=np.int32)
      if bool(getattr(config, "olmo_use_loss_weights", False)):
        batch["loss_weights"] = np.zeros((batch_size, config.max_target_length), dtype=np.float32)
      if ds_type == "olmo_mm":
        batch["router_token_mask"] = np.zeros((batch_size, config.max_target_length), dtype=np.int32)

    def infinite_iterator():
      while True:
        yield batch

    return infinite_iterator()
