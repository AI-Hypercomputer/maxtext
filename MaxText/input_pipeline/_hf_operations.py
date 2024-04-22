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

"""Operations used by HuggingFace input pipeline"""

from typing import Generic, Iterator, TypeVar, Union
from collections import defaultdict
import jax
import jaxtyping as jt
from jax import tree_util
import dataclasses
import numpy as np

_T = TypeVar("_T")

def tokenization(example, tokenizer, max_length):
  """Tokenize dataset"""
  return tokenizer(example["text"], truncation=True, max_length=max_length)

def normalize_features(example, key):
  """Extract text column from data and rename as inputs and targets"""
  return {
      'inputs': example[key],
      'targets': example[key],
  }

def shift(example):
  example['inputs'] = [0] + example['inputs'][:-1]
  return example



def pack_in_batch(batch, max_len=2048):

  def _pad_and_refine(current_pack, max_len):
    targets_nonzero = current_pack['targets'] != 0
    current_pack['inputs_segmentation'] *= targets_nonzero
    current_pack['targets_segmentation'] *= targets_nonzero
    for k,v in current_pack.items():
      current_pack[k] = v + [0] * (max_len-len(v))
    return current_pack

  packed_text = []
  current_pack = {'inputs':[], 'targets':[],
                  'inputs_segmentation':[], 'targets_segmentation': [], 
                  'inputs_position': [], 'targets_position': []}
  current_pack_len = 0
  current_segmentation = 0
  #print(f"{batch=}")

  for text in batch:
    #print(f"{text=}")
    new_len = current_pack_len + len(text['inputs']) + 1
    if new_len <= max_len:
      current_segmentation += 1
      current_pack['inputs'] += [0] + text['inputs']
      current_pack['targets'] += [0] + text['targets']
      current_pack['inputs_segmentation'] += [current_segmentation] * (len(text['inputs'])+1)
      current_pack['targets_segmentation'] += [current_segmentation] * (len(text['targets'])+1)
      current_pack['inputs_position'] += [i for i in range(len(text['inputs'])+1)]
      current_pack['targets_position'] += [i for i in range(len(text['targets'])+1)]
      current_pack_len = new_len
    else:
      current_pack = _pad_and_refine(current_pack, max_len)
      packed_text.append(current_pack)
      current_segmentation=1
      current_pack = {'inputs': [0] + text['inputs'],
                        'targets': [0] + text['targets'], 
                        'inputs_segmentation': [current_segmentation] * (len(text['inputs'])+1),
                        'targets_segmentation': [current_segmentation] * (len(text['targets'])+1),
                        'inputs_position': [i for i in range(len(text['inputs'])+1)],
                        'targets_position': [i for i in range(len(text['targets'])+1)]
                      } # Start a new pack
      current_pack_len = len(text['inputs']) + 1

  # Handle the last pack (might not reach max_len)
  if current_pack:
    current_pack = _pad_and_refine(current_pack, max_len)
    packed_text.append(current_pack)

  return packed_text

def group_in_batch(batch):
    # new_batch = {'inputs':np.array([]), 'targets':np.array([]), 
    #             'inputs_segmentation':np.array([]), 'targets_segmentation': np.array([]), 
    #             'inputs_position': np.array([]), 'targets_position': np.array([])}
    new_batch = defaultdict(list)
    for x in batch:
        for k, v in x.items():
            new_batch[k].append(v)

    return {
        'inputs': np.array(new_batch['inputs']),
        'targets': np.array(new_batch['targets']),
        'inputs_segmentation': np.array(new_batch['inputs_segmentation']),
        'targets_segmentation': np.array(new_batch['targets_segmentation']),
        'inputs_position': np.array(new_batch['inputs_position']),
        'targets_position': np.array(new_batch['targets_position']),
    }

class _PackedBatch:
  """Class to represent a batch of packed examples.
  Adopted from grain/_src/python/experimental/example_packing/packing.py,
  which provides an implementation for example packing in pure python"""

  def __init__(
      self,
      element_for_shapes: jt.PyTree[np.ndarray],
      batch_size: int,
      length_struct: jt.PyTree[int],
      shift_inputs: bool = True,
  ):
    self._batch_size = batch_size
    self._length_struct = length_struct
    self._shift_inputs = shift_inputs

    # Define the main buffers we will pack the data into.
    def make_packed_buffer(length: int, input_arr: np.ndarray):
      return np.zeros(
        shape=(batch_size, length, *input_arr.shape[1:]),  # (B, T, ...)
        dtype=input_arr.dtype,
      )

    self._batch = jax.tree.map(
        make_packed_buffer, length_struct, element_for_shapes
    )

    def make_packed_aux_info(length: int):
      return np.zeros(shape=(batch_size, length), dtype=np.int32)

    self._segmentations = jax.tree.map(make_packed_aux_info, length_struct)
    self._positions = jax.tree.map(make_packed_aux_info, length_struct)

    # Tracks the next empty position to insert an example for each row
    # in the batch, for each feature in features_to_pack.
    self._first_free_cell_per_row = jax.tree.map(
      lambda _: np.zeros(batch_size, dtype=np.int32), length_struct
    )

    # Tracks the number of examples already packed into row of the batch. Used
    # to fill the segmentation values for each feature.
    self._num_examples_per_row = [0 for _ in range(batch_size)]

  def _tree_flatten(self, structure):
    return tree_util.tree_flatten(structure)[0]

  def _shift_right(self, x, axis=1):
    """Shift the input to the right by padding and slicing on axis."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    slices = [slice(None),] * len(x.shape)
    slices[axis] = slice(0, -1)
    padded = np.pad(
      x,
      pad_widths,
      mode='constant',
      constant_values=x.dtype.type(0)
      )
    return padded[tuple(slices)]

  def get_packed_shifted_batch(self):
    """Reformat packed output and apply shift"""
    targets_nonzero = self._batch['targets'] != 0
    self._batch['inputs_segmentation'] = self._segmentations['inputs'] * targets_nonzero
    self._batch['inputs_position'] = self._positions['inputs']
    self._batch['targets_segmentation'] = self._segmentations['targets'] * targets_nonzero
    self._batch['targets_position'] = self._positions['targets']
    if self._shift_inputs:
      self._batch['inputs'] = self._shift_right(self._batch['inputs'])
      self._batch['inputs'] *= self._batch['inputs_segmentation'] == self._shift_right(self._batch['inputs_segmentation'])

    return self._batch

  def _can_add_at_row(self, element: jt.PyTree[np.ndarray]) -> int:
    """Returns the index of the first row which fits element, or -1 if none."""
    element_feature_lengths = jax.tree.map(len, element)

    # Check no feature exceeds max length
    length_exceeded = jax.tree.map(
      lambda feature_length, max_length: feature_length > max_length,
      element_feature_lengths,
      self._length_struct,
    )
    if any(self._tree_flatten(length_exceeded)):
      raise ValueError(
        "Inputs to PackAndBatchOperation must be truncated to max length."
      )

    # For each row, check whether the total length after adding the current
    # element would exceed max feature lengths.
    def _feature_will_fit(feature_length, first_free_cell, max_length):
      return feature_length + first_free_cell <= max_length

    is_row_free_struct = jax.tree.map(
      _feature_will_fit,
      element_feature_lengths,
      self._first_free_cell_per_row,
      self._length_struct,
    )

    ## Pick first row (if exists) where element can be added.
    for i in range(self._batch_size):
      row_is_free_per_feature = [
        free[i] for free in self._tree_flatten(is_row_free_struct)
      ]
      if all(row_is_free_per_feature):
        return i
    return -1

  def add_element_to_batch(
      self, element: jt.PyTree[np.ndarray], row: int
  ) -> None:
    """Adds element to current batch at the specified row."""
    # Apply updates to each feature.
    for per_feature_data in zip(
      self._tree_flatten(element),
      self._tree_flatten(self._batch),
      self._tree_flatten(self._segmentations),
      self._tree_flatten(self._positions),
      self._tree_flatten(self._first_free_cell_per_row),
    ):
      value, batch_value, segmentations, positions, first_free_cell_per_row = (
        per_feature_data
      )
      # Update batch value, segmentations, and positions.
      start = first_free_cell_per_row[row]
      end = first_free_cell_per_row[row] + len(value)
      batch_value[row][start:end] = value
      segmentations[row][start:end] = self._num_examples_per_row[row] + 1
      positions[row][start:end] = np.arange(end - start)
      # Update first_free_cell_per_row.
      first_free_cell_per_row[row] += len(value)

    self._num_examples_per_row[row] += 1

  def try_add_to_batch(self, element: jt.PyTree[np.ndarray]) -> bool:
    """Finds a row in the batch at which element can be added."""
    if (row_idx := self._can_add_at_row(element)) == -1:
      return False
    self.add_element_to_batch(element, row_idx)
    # self._last_record_metadata = element.metadata.remove_record_key()
    return True

@dataclasses.dataclass
class PackAndBatchOperation(Generic[_T]):
  """Perform pack-shift-batch
  Adopted from grain/_src/python/experimental/example_packing/packing.py,
  which provides an implementation for example packing in pure python
  """
  length_struct: jt.PyTree[int]
  batch_size: int
  # We don't know input shapes and corresponding buffer shapes until __call__.
  _cur_batch: Union[_PackedBatch, None] = None
  shift_inputs: bool = True

  def __call__(
    self, input_iterator: Iterator[_T]
  ) -> Iterator[tuple[_T, _T, _T]]:
    for element in input_iterator:
      # Use `element` to set dtypes + trailing dimensions.
      if self._cur_batch is None:  # pytype: disable=attribute-error
        self._cur_batch = _PackedBatch(
          element, self.batch_size, self.length_struct,
          shift_inputs=self.shift_inputs
        )

      # Try adding element to the current packed batch.
      element_added_to_batch = self._cur_batch.try_add_to_batch(element)

      # When we have a full batch, yield the current packed data,
      # and then start a new batch with this element.
      if not element_added_to_batch:
        yield self._cur_batch.get_packed_shifted_batch()  # Main yield
        self._cur_batch = _PackedBatch(
          element, self.batch_size, self.length_struct
        )
        self._cur_batch.try_add_to_batch(element)

    # Final batch
    yield self._cur_batch.get_packed_shifted_batch()

class TransformedDataset:
  """Apply operation to the dataset and return an iterator"""
  def __init__(self, transform, input_iterator):
    self.transform = transform
    self.input_iterator = input_iterator

  def _apply_transform(self):
    for r in self.transform(self.input_iterator):
      yield r

  def __iter__(self):
    return self._apply_transform()
