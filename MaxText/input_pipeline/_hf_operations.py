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
import collections
from collections.abc import Iterator, Sequence
from concurrent import futures
from typing import Generic, TypeVar, Union
from collections import defaultdict
import jax
import jaxtyping as jt
from jax import tree_util
import dataclasses
import numpy as np
import datasets
from datasets import distributed
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from threading import current_thread
import max_logging
from grain import python_lazy_dataset
from grain import python as grain

_T = TypeVar("_T")

class HuggingFaceLazyDatasetIterator(python_lazy_dataset.LazyDatasetIterator):

  def __init__(
      self,
      #datasets: Sequence[datasets.IterableDataset],
      dataset: datasets.IterableDataset,
      read_options: grain.ReadOptions,
  ):
    #self._datasets = tuple(datasets)
    self._dataset = dataset
    self._read_options = read_options
    self._next_index = 0
    self._iterator = None
  #   self._transform()
    
  # def _transform(self):
  #   for ds in self._datasets:
      

  def _make_iterator(self) -> Iterator:
    """Reads data in a round-robin fashion."""
    # We use a thread pool to read elements and add them to a buffer in the
    # background.
    # The main thread simply gets elements from the buffer and waits for them
    # to be available.
    index = 0
    #iterators = tuple(iter(dataset) for dataset in self._datasets)
    data_iter = iter(self._dataset)
    
    def prefetch_element():
      # print(f"{current_thread().name=}")
      # idx = int(current_thread().name.split("_")[1])
      return next(data_iter)
      #return next(iterators[idx])

    buffer = collections.deque()
    buffer_size = self._read_options.prefetch_buffer_size
    # threads = tuple(Thread(target = prefetch_element, args=(i), name="thread-{i}")
    #                 for i in len(self._datasets) )

    #for i in range(buffer_size):

    #with futures.ThreadPoolExecutor(self._read_options.num_threads) as executor:
    for i in range(buffer_size):
      #buffer.append(executor.submit(prefetch_element, i % len(self._datasets)))
      buffer.append(prefetch_element())
      #print("in buffer_size loop", i)
      #buffer.append(executor.submit(prefetch_element, i))
    while True:
      try:
        element = buffer.popleft()
        #element = buffer.popleft().result()
      except (IndexError, StopIteration):
        # End of sampler.
        return
      if index == self._next_index:
        yield element
        self._next_index += 1
      # buffer.append(
      #     executor.submit(prefetch_element,
      #                     buffer_size + index))
      # buffer.append(
      #     executor.submit(prefetch_element,
      #                     (buffer_size + index) % len(self._datasets)))
      buffer.append(prefetch_element())
      index += 1

  def __next__(self):
    if self._iterator is None:
      self._iterator = self._make_iterator()
    return next(self._iterator)

  def get_state(self):
    return {"next_index": self._next_index}

  def set_state(self, state):
    self._next_index = state["next_index"]


class HuggingFaceLazyIterDataset(python_lazy_dataset.LazyIterDataset):

  def __init__(self,
                dataset: datasets.IterableDataset,
                tokenizer,
                max_length,
                read_options: grain.ReadOptions | None = None):
    super().__init__()
    self._dataset = dataset
    self._tokenizer = tokenizer
    self._max_length = max_length
    self._read_options = (
        grain.ReadOptions() if read_options is None else read_options)
    self._slice = slice(0, self._dataset.n_shards, 1)
    #self._transform()

  # def _transform(self):
  #   self._dataset = self._dataset.map(tokenization, batched=True,
  #                   fn_kwargs={"tokenizer": self._tokenizer, "max_length": self._max_length-1})
  #   self._dataset = self._dataset.select_columns(["input_ids"])

  def __iter__(self):
    # datasets = tuple(
    #     distributed.split_dataset_by_node(self._dataset, i, self._read_options.num_threads)
    #     for i in range(self._read_options.num_threads)
    #     # distributed.split_dataset_by_node(self._dataset, i, self._dataset.n_shards)
    #     # for i in range(self._dataset.n_shards)[self._slice]
    # )
    return HuggingFaceLazyDatasetIterator(self._dataset, self._read_options)

  def set_parent_maps_slice(self, sl: slice) -> None:
    super().set_parent_maps_slice(sl)
    self._slice = sl


class HuggingFaceDataLoader(python_lazy_dataset.DataLoader):

  def __init__(self,
                dataset: datasets.Dataset | datasets.IterableDataset,
                tokenizer,
                max_length,
                multiprocessing_options: grain.MultiprocessingOptions | None = None,
                read_options: grain.ReadOptions | None = None):
    if isinstance(dataset, datasets.Dataset):
      super().__init__(
          lazy_ds=python_lazy_dataset.SourceLazyMapDataset(dataset),
          multiprocessing_options=multiprocessing_options, read_options=read_options)
    elif isinstance(dataset, datasets.IterableDataset):
      super().__init__(
          lazy_ds=HuggingFaceLazyIterDataset(dataset, tokenizer, max_length, read_options),
          multiprocessing_options=multiprocessing_options,
          read_options=read_options)
    else:
      raise ValueError(f'Unknown {dataset=}.')


def tokenization(example, tokenizer, max_length):
  """Tokenize dataset"""
  return tokenizer(example["text"], truncation=True, max_length=max_length)

class HFDataSource:
  def __init__(self, dataset, dataloading_host_index, dataloading_host_count, num_threads, num_worker, add_bos=True, add_eos=True):
    #self.dataset_length = dataset.info.splits['train'].num_examples
    #dataset = split_dataset_by_node(dataset, world_size=jax.process_count(), rank=jax.process_index())

    # self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_path,
    #                                         add_bos_token=add_bos,
    #                                         add_eos_token=add_eos,
    #                                         model_max_length=max_length)

    # dataset = dataset.map(tokenization, batched=True,
    #                     fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length-1})
    # dataset = dataset.select_columns(["input_ids"])
    self.dataset = dataset
    self.num_threads = num_threads
    self.num_worker = num_worker
    self.n_shards = dataset.n_shards
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range (self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x)
                          for x in self.dataset_shards]
    # self.datasets = tuple(split_dataset_by_node(dataset, world_size=self.n_shards,
    #                                             rank=dataloading_host_index * dataloading_host_count + i)
    #                       for i in range(self.num_threads))
    # self.datasets = tuple(split_dataset_by_node(dataset, world_size=self.world_size, rank=i)
    #                       for i in range(self.world_size))
    #self.count = [0] * num_threads
    #self.current_dataset_idx = None
    self.data_iters = None

  def _update_shard(self, idx):
    max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx} from shard {self.dataset_shards[idx]}")
    self.dataset_shards[idx] += self.dataloading_host_count * self.num_threads
    max_logging.log(f"New shard is {self.dataset_shards[idx]}")
    if self.dataset_shards[idx] > self.n_shards:
      raise ValueError(f"Run out of shards, shard {self.dataset_shards[idx]} is not available")
    self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
    self.data_iters[idx] = iter(self.datasets[idx])


  def __len__(self):
    return 10_000_000_000
    #return self
    #return len(self.dataset)

  def __getitem__(self, index):
    if self.data_iters is None:
      self.data_iters = [iter(x) for x in self.datasets]
      #self.current_dataset_idx = -1

    idx = int(current_thread().name.split("_")[1])

    # self.count[idx] += 1
    # if self.count[idx]>100:
    #   self._update_shard(idx)
    #   self.count[idx] = 0
    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)

    #return data
    #self.current_dataset_idx = (self.current_dataset_idx + 1) % self.world_size
    #return next(self.data_iters[self.current_dataset_idx])

      #self.current_dataset_idx = -1
    # if self.num_worker != 0:
    #   self.current_dataset_idx = (index % (self.num_thread * self.num_worker)) // self.num_worker
    # else:
    #   self.current_dataset_idx = index % self.num_thread
    # #max_logging.log(f"{self.current_dataset_idx=}")
    # return next(self.data_iters[self.current_dataset_idx])

def normalize_features(example, key):
  """Extract text column from data and rename as inputs and targets"""
  return {
      'inputs': example[key],
      'targets': example[key],
  }

def shift(example):
  example['inputs'] = [0] + example['targets'][:-1]
  return example

def group_batch(batch):
    return {k: [v] for k, v in batch.items()}

def unbatch(batch):
    new_batch = defaultdict(list)
    keys = batch.keys()
    for values in zip(*batch.values()):
        ex = {k: v for k, v in zip(keys, values)}
        new_batch["inputs"].extend(ex["inputs"])
        new_batch["targets"].extend(ex["targets"])
        new_batch["inputs_segmentation"].extend(ex["inputs_segmentation"])
        new_batch["targets_segmentation"].extend(ex["targets_segmentation"])
        new_batch["inputs_position"].extend(ex["inputs_position"])
        new_batch["targets_position"].extend(ex["targets_position"])
    return new_batch

def pack_in_batch_hf(batch, max_len=2048):

  def _pad_and_refine(current_pack, max_len):
    targets_nonzero = current_pack['targets'] != 0
    current_pack['inputs_segmentation'] *= targets_nonzero
    current_pack['targets_segmentation'] *= targets_nonzero
    for k,v in current_pack.items():
      current_pack[k] = v + [0] * (max_len-len(v))
    return current_pack

  def _add_current_pack(packed_text, current_pack):
    for key in current_pack.keys():
      packed_text[key].append(current_pack[key])

  # packed_text = {'inputs':[], 'targets':[],
  #                 'inputs_segmentation':[], 'targets_segmentation': [], 
  #                 'inputs_position': [], 'targets_position': []}
  packed_text = defaultdict(list)

  current_pack = {'inputs':[], 'targets':[],
                  'inputs_segmentation':[], 'targets_segmentation': [], 
                  'inputs_position': [], 'targets_position': []}
  current_pack_len = 0
  current_segmentation = 0
  #print(f"{batch=}")

  for inputs, targets in zip(batch["inputs"], batch["targets"]):
    #print(f"{text=}")
    new_len = current_pack_len + len(inputs) + 1
    if new_len <= max_len:
      current_segmentation += 1
      current_pack['inputs'] += [0] + inputs
      current_pack['targets'] += [0] + targets
      current_pack['inputs_segmentation'] += [current_segmentation] * (len(inputs)+1)
      current_pack['targets_segmentation'] += [current_segmentation] * (len(targets)+1)
      current_pack['inputs_position'] += [i for i in range(len(inputs)+1)]
      current_pack['targets_position'] += [i for i in range(len(targets)+1)]
      current_pack_len = new_len
    else:
      current_pack = _pad_and_refine(current_pack, max_len)
      _add_current_pack(packed_text, current_pack)
      current_segmentation=1
      current_pack = {'inputs': [0] + inputs,
                        'targets': [0] + targets, 
                        'inputs_segmentation': [current_segmentation] * (len(inputs)+1),
                        'targets_segmentation': [current_segmentation] * (len(targets)+1),
                        'inputs_position': [i for i in range(len(inputs)+1)],
                        'targets_position': [i for i in range(len(targets)+1)]
                      } # Start a new pack
      current_pack_len = len(inputs) + 1
    # Handle the last pack (might not reach max_len)
  if current_pack:
    current_pack = _pad_and_refine(current_pack, max_len)
    _add_current_pack(packed_text, current_pack)

  return packed_text

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
