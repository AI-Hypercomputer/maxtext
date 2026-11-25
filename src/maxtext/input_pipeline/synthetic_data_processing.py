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

"""Input pipeline for synthetic dataset.

`SyntheticDataIterator` is the static batch of random token ids text training runs on. `SyntheticMultimodalDataSource`
is its multimodal counterpart at the dataset level: rows of random text and random media (images of random size and
count) in the raw form a real multimodal SFT dataset provides, so the same model-specific processing turns them into
training batches.
"""

from collections.abc import Callable
from typing import Any

import grain.python as grain
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
  blocks = jnp.pad(blocks, [(0, 0), (0, batch_size - logical_batch_size)] + [(0, 0)] * (rows.ndim - 1))
  return blocks.reshape(num_replicas * batch_size, *rows.shape[1:])


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
    # Generate the (static) synthetic batch once, directly in its sharded layout. Building it unsharded on
    # local_devices()[0] each step cost 1.5-12 GiB of HBM on that device and stalled step 0 on 512+ chips.
    mesh_ctx = jax.set_mesh(self.mesh) if isinstance(self.mesh, jax.sharding.Mesh) else self.mesh
    with mesh_ctx:
      self.data_generator = jax.jit(
          SyntheticDataIterator.raw_generate_synthetic_data,
          in_shardings=(),
          out_shardings=data_pspec_shardings,
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
    tokens = _pad_replica_blocks(tokens, num_replicas, batch_size)

    sequence_positions = jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
    batch_positions = jnp.broadcast_to(sequence_positions, (all_replicas_batch_size, config.max_target_length + 1))
    if config.packing:
      segmentation = _make_packed_segment_ids(
          logical_rows,
          config.max_target_length,
          config.max_segments_per_seq,
      )
      segmentation = _pad_replica_blocks(segmentation, num_replicas, batch_size)
    else:
      segmentation = jnp.ones((all_replicas_batch_size, config.max_target_length), dtype=jnp.int32)
    rows = slice(replica_index * batch_size, (replica_index + 1) * batch_size)
    tokens, batch_positions, segmentation = tokens[rows], batch_positions[rows], segmentation[rows]
    if logical_batch_size < batch_size:
      segmentation = segmentation.at[logical_batch_size:].set(0)

    output = {}
    output["inputs"] = tokens[:, :-1]
    output["inputs_position"] = batch_positions[:, :-1]
    output["inputs_segmentation"] = segmentation
    output["targets"] = tokens[:, 1:]
    output["targets_position"] = batch_positions[:, 1:]
    output["targets_segmentation"] = segmentation
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


########## Multimodal synthetic dataset

# The dataset is infinite in practice; grain's index sampler only needs an upper bound.
_NUM_SYNTHETIC_RECORDS = 10_000_000_000
# Words of random letters stand in for text: tokenized by any tokenizer, they contain no special tokens.
_PROMPT_WORDS = 24
_MIN_WORD_LENGTH, _MAX_WORD_LENGTH = 2, 8


def synthetic_modalities(config) -> list[str]:
  """The media modalities `SyntheticMultimodalDataSource` generates for `config`.

  Images when `use_multimodal` and `synthetic_num_images_per_example_range` allows at least one image. Future modalities
  (video, audio, ...) add their trigger here and their generator to `_MEDIA_GENERATORS`.
  """
  modalities = []
  if config.use_multimodal and config.synthetic_num_images_per_example_range[1] > 0:
    modalities.append("image")
  return modalities


def generate_synthetic_images(rng: np.random.Generator, config, largest: bool = False) -> list[np.ndarray]:
  """Draws the images of one example: random uint8 RGB arrays of shape (H, W, 3) of random size and count.

  The count, height and width (independent, so any aspect ratio) are uniform over the inclusive
  `synthetic_num_images_per_example_range` and `synthetic_image_{height,width}_range`. With `largest`, every range
  yields its maximum (the largest example the config can produce).
  """
  min_images, max_images = config.synthetic_num_images_per_example_range
  min_height, max_height = config.synthetic_image_height_range
  min_width, max_width = config.synthetic_image_width_range
  if largest:
    min_images, min_height, min_width = max_images, max_height, max_width
  images = []
  for _ in range(int(rng.integers(min_images, max_images + 1))):
    height = int(rng.integers(min_height, max_height + 1))
    width = int(rng.integers(min_width, max_width + 1))
    # `rng.bytes` is the fastest way to fill the pixels.
    images.append(np.frombuffer(rng.bytes(height * width * 3), dtype=np.uint8).reshape(height, width, 3))
  return images


# Media generators by modality: `(rng, config, largest) -> the media of one example`, in the raw form a real dataset
# provides them (uint8 RGB arrays for images). The row of an example holds each modality's media under the modality
# name.
_MEDIA_GENERATORS: dict[str, Callable[[np.random.Generator, Any, bool], list]] = {"image": generate_synthetic_images}


class SyntheticMultimodalDataSource(grain.RandomAccessDataSource):
  """A synthetic multimodal SFT dataset: rows of random text and random media, like the rows of a real dataset.

  Each row holds a `prompt` and a `completion` of random words (the completion has enough words to fill any
  `max_target_length` once tokenized and truncated) and, per modality of `synthetic_modalities(config)`, the media of
  the example under the modality name (e.g. `image`: a list of uint8 RGB arrays). The model-specific processing (chat
  format, tokenization, media preprocessing) is left to the consuming pipeline, exactly as for a real dataset. Row
  `index` is a pure function of `(seed, index)`, so every host and worker produces the same row for the same index.
  """

  TEXT_COLUMNS = ("prompt", "completion")

  def __init__(self, config, seed: int):
    self.config = config
    self.seed = seed
    self.modalities = synthetic_modalities(config)
    if not self.modalities:
      raise ValueError(
          "SyntheticMultimodalDataSource needs a media modality: use_multimodal with a"
          " synthetic_num_images_per_example_range that allows images."
      )

  def __len__(self):
    return _NUM_SYNTHETIC_RECORDS

  @staticmethod
  def _words(rng: np.random.Generator, num_words: int) -> str:
    lengths = rng.integers(_MIN_WORD_LENGTH, _MAX_WORD_LENGTH + 1, size=num_words)
    letters = rng.integers(ord("a"), ord("z") + 1, size=int(lengths.sum()), dtype=np.uint8).tobytes().decode()
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return " ".join(letters[start : start + length] for start, length in zip(starts, lengths))

  def generate(self, rng: np.random.Generator, largest: bool = False) -> dict[str, Any]:
    """Draws one row from `rng`; with `largest`, the largest row the config can produce (see the generators)."""
    row = {
        "prompt": self._words(rng, _PROMPT_WORDS),
        "completion": self._words(rng, self.config.max_target_length),
    }
    for modality in self.modalities:
      row[modality] = _MEDIA_GENERATORS[modality](rng, self.config, largest)
    return row

  def __getitem__(self, index):
    return self.generate(np.random.default_rng([self.seed, int(index)]))
