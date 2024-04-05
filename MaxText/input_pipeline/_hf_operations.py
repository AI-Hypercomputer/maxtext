from typing import Generic, Iterator, TypeVar, Union, cast
import jax
import jaxtyping as jt
from jax import tree_util
import dataclasses
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer, GemmaTokenizer

_T = TypeVar("_T")

def load_tokenizer(tokenizer_loader, tokenizer_path, add_bos, add_eos, max_length, token):
    def choose_tokenizer_loader(tokenizer_loader):
        if tokenizer_loader == "AutoTokenizer":
            return AutoTokenizer
        elif tokenizer_loader == "LlamaTokenizer":
            return LlamaTokenizer
        elif tokenizer_loader == "GemmaTokenizer":
            return GemmaTokenizer
        else:
            raise ValueError(f"Unknown tokenizer_loader {tokenizer_loader}")

    loader = choose_tokenizer_loader(tokenizer_loader)
    return loader.from_pretrained(tokenizer_path,
                        add_bos_token=add_bos,
                        add_eos_token=add_eos,
                        model_max_length=max_length,
                        token=token)

def tokenization(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length)

def normalize_features(example, key):
    return {
        'inputs': example[key],
        'targets': example[key],
    }

class _PackedBatch:
  """Class to represent a batch of packed examples."""

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

    # For determinism, the metadata.index for the packed batch must match
    # metadata.index of the _last_ included input example.
    #self._last_record_metadata = None
  def tree_flatten(self, structure):
    return tree_util.tree_flatten(structure)[0]

  def shift_right(self, x, axis=1):
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

  def get_packed_batch(self):
    # assert self._last_record_metadata is not None
    # return record.Record(
    #     metadata=cast(record.RecordMetadata, self._last_record_metadata),
    #     data=(self._batch, self._segmentations, self._positions),
    # )
    # set input and target segmentation to 0 when target element is 0
    targets_nonzero = self._batch['targets'] != 0
    self._batch['inputs_segmentation'] = self._segmentations['inputs'] * targets_nonzero
    self._batch['inputs_position'] = self._positions['inputs']
    self._batch['targets_segmentation'] = self._segmentations['targets'] * targets_nonzero
    self._batch['targets_position'] = self._positions['targets']
    if self._shift_inputs:
      # padding_array = np.array([self.shift_padding_token])
      # padding_array = np.broadcast_to(padding_array, (self._batch['inputs'].shape[0], padding_array.shape[0]))
      # self._batch['inputs'] = np.concatenate((a, self._batch['inputs'][:,:-1]), axis=1)
      self._batch['inputs'] = self.shift_right(self._batch['inputs'])
      self._batch['inputs'] *= self._batch['inputs_segmentation'] == self.shift_right(self._batch['inputs_segmentation'])

    return self._batch

    #return (self._batch, self._segmentations, self._positions)

  def _can_add_at_row(self, element: jt.PyTree[np.ndarray]) -> int:
    """Returns the index of the first row which fits element, or -1 if none."""
    element_feature_lengths = jax.tree.map(len, element)

    # Check no feature exceeds max length
    length_exceeded = jax.tree.map(
        lambda feature_length, max_length: feature_length > max_length,
        element_feature_lengths,
        self._length_struct,
    )
    if any(self.tree_flatten(length_exceeded)):
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
          free[i] for free in self.tree_flatten(is_row_free_struct)
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
        self.tree_flatten(element),
        self.tree_flatten(self._batch),
        self.tree_flatten(self._segmentations),
        self.tree_flatten(self._positions),
        self.tree_flatten(self._first_free_cell_per_row),
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
        yield self._cur_batch.get_packed_batch()  # Main yield
        self._cur_batch = _PackedBatch(
            element, self.batch_size, self.length_struct
        )
        self._cur_batch.try_add_to_batch(element)

    # Final batch
    yield self._cur_batch.get_packed_batch()

class TransformedDataset:
    def __init__(self, transform, input_iterator):
        self.transform = transform
        self.input_iterator = input_iterator

    def _apply_transform(self):
        for r in self.transform(self.input_iterator):
          yield r

    def __iter__(self):
        return self._apply_transform()
