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
from maxtext.utils import sharding


def _make_packed_segment_ids(batch_size: int, seq_len: int, max_segments_per_seq: int) -> jnp.ndarray:
  """Creates synthetic segment IDs simulating packed sequences.

  Splits each sequence into a random number (2..max_segments_per_seq) of
  varying-length segments, assigning each a sequential integer segment ID
  starting from 1. Segment boundaries are randomized per row so the same
  segment IDs can be reused across the batch dimension without cross-row
  correlation.

  Args:
    batch_size: Number of sequences.
    seq_len: Total sequence length.
    max_segments_per_seq: Maximum number of segments per sequence (>= 2).

  Returns:
    Integer array [batch_size, seq_len] where 0 = padding.
  """
  if max_segments_per_seq < 2:
    return jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  key = jax.random.PRNGKey(42)
  segments = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  for b in range(batch_size):
    key, subkey = jax.random.split(key)
    n_segs = jax.random.randint(subkey, (1,), 2, max_segments_per_seq + 1)[0]
    # Random split points (sorted) in range [1, seq_len-1)
    key, subkey = jax.random.split(key)
    splits = jax.random.randint(subkey, (n_segs - 1,), 1, seq_len)
    splits = jnp.sort(splits)
    splits = jnp.concatenate([jnp.array([0], dtype=jnp.int32), splits, jnp.array([seq_len], dtype=jnp.int32)])
    seg_ids = jnp.arange(1, n_segs + 1)
    row = jnp.repeat(seg_ids, splits[1:] - splits[:-1], total_repeat_length=seq_len)
    segments = segments.at[b].set(row)
  return segments


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

    tokens = jax.random.randint(
        jax.random.PRNGKey(0),
        (config.global_batch_size_to_load, config.max_target_length + 1),
        0,
        config.vocab_size,
        dtype=jnp.int32,
    )

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
    with self.mesh:
      return self.data_generator(self.config, self.data)  # pylint: disable=not-callable

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
    return output


class PlaceHolderDataIterator:
  """Creates a place holder synthetic data iterator for loading on subset of hosts"""

  def __init__(self, config: pyconfig.HyperParameters, mesh):
    self.mesh = mesh
    dataset = PlaceHolderDataIterator.get_place_holder_synthetic_data(config)
    self.data_generator = multihost_dataloading.MultiHostDataLoadIterator(dataset, self.mesh)

  def __iter__(self):
    return self.data_generator

  def __next__(self):
    return next(self.data_generator)

  def reset(self):
    pass

  @staticmethod
  def get_place_holder_synthetic_data(config: pyconfig.HyperParameters):
    """fill negative value in synthetic data"""
    batch_size = config.global_batch_size_to_load // jax.process_count()
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
