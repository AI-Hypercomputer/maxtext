# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MaxText Offline Inference Engine

Features 
- Continous batching 
- Prefill packing
- Single and multihost with TP

Example usage: 

    max_engine = MaxEngine(config, jax.devices())

    offline_engine = OfflineEngine(
        engine=max_engine, params=None, enable_batch_prefill=True
    )

    input_data = [jax.numpy.arange(80),
                    jax.numpy.arange(90),
                    jax.numpy.arange(100),
                    ]

    results = offline_engine.batch_inference(input_data)

    for tokens in results:
        text = offline_engine.tokenizer.decode(tokens)
        print(f"Output: {text}")

Notes: 
    - Prefill packing is only supported with scan_layers=False
    - DO NOT add print statements in the inference loop as it will
      introduce non-deterministic behavior due to the background
      detokenization thread. 

TODO(wenxindongwork): 
    - support DP 
"""

from typing import Any, List, Tuple, Callable, Optional, Dict, Union
from collections import defaultdict
import dataclasses
import functools
import logging
import os
import queue
import signal
import threading
import traceback
import jax
from jetstream.engine import engine_api
from enum import Enum
# pylint: disable=no-name-in-module
from MaxText.maxengine import MaxEngine
from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor

DecodeState = Any
Params = Any

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InputData:
  """Class for storing input data and its metadata.

  Attributes:
      id: Unique identifier for this input
      tokens: JAX array containing the tokenized input
      true_length: Actual length of the input before padding
  """

  id: str
  tokens: jax.Array
  true_length: int


@dataclasses.dataclass
class EventCounter:
  """Class for tracking statistics during inference.

  Attributes:
      input: Number of input sequences processed
      prefill: Number of prefill operations completed
      decode: Number of decode operations completed
      detokenize: Number of sequences completely detokenized
  """

  input: int
  prefill: int
  decode: int
  detokenize: int


class JetThread(threading.Thread):
  """Thread class with exception handling to prevent silent failures."""

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      # Kill the process if a thread encounters an error
      os.kill(os.getpid(), signal.SIGKILL)


class PrefillType(Enum):
  DEFAULT = "default"
  BATCH = "batch"


class PrefillHelper:
  """
  This class abstracts the details of two prefill methods (default vs batch)
  and provides a common interface for prefill operations.
  """

  def __init__(self, type: PrefillType, engine: MaxEngine, batch_prefill_max_batch_size: int = 16):
    """Initialize the PrefillHelper.

    Args:
        type: The type of prefill processor to use ("default" or "batch")
        engine: The MaxEngine instance to use for prefill operations
    """
    self._type = type
    self.engine = engine
    if type == PrefillType.DEFAULT:
      self._processor = PrefillProcessor(engine)
    elif type == PrefillType.BATCH:
      self._batch_processor = BatchedPrefillProcessor(engine=engine, max_batch_size=batch_prefill_max_batch_size)
      # Also create a standard processor for fallback cases
      self._processor = PrefillProcessor(engine)
    else:
      raise ValueError(f"Invalid type: {type}")

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: int,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      max_length: int,
      prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None],
  ) -> None:
    """Process an input through the appropriate prefill processor.

    This method routes the input to either the default processor or the
    batch processor based on configuration and input characteristics.

    Args:
        model_params: Model parameters for inference
        decode_state: Current decoder state
        decode_slot: The decode slot index to use for this input
        input_id: Unique identifier for this input
        input_tokens_padded: Padded token array for the input
        input_true_length: Actual length of the input before padding
        max_length: Maximum prefill length allowed
        prefill_done: Callback function called when prefill completes
    """
    padded_length = len(input_tokens_padded)
    # Use default processor if configured or if input is already at max length
    if self._type == PrefillType.DEFAULT or padded_length == max_length:
      first_token, decode_state = self._processor.process(
          model_params,
          decode_state,
          decode_slot,
          input_tokens_padded,
          input_true_length,
      )
      prefill_done([(first_token, decode_slot)], [input_id], decode_state)
    # Use batch processor for inputs that can benefit from prefill packing
    elif self._type == PrefillType.BATCH:
      self._batch_processor.process(
          model_params,
          decode_state,
          decode_slot,
          input_id,
          input_tokens_padded[:input_true_length],
          padded_length,
          max_length,
          prefill_done,
      )

  def finalize(
      self,
      model_params: Params,
      decode_state: DecodeState,
      prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None],
  ) -> None:
    """Finalize prefill operations, flushing any pending inputs.

    Args:
        model_params: Model parameters for inference
        decode_state: Current decoder state
        prefill_done: Callback function called when prefill completes
    """
    if self._type == PrefillType.DEFAULT:
      # No finalization needed for default processor
      pass
    elif self._type == PrefillType.BATCH:
      # Flush any remaining inputs in the batch processor
      self._batch_processor.flush(model_params, decode_state, prefill_done)


class OfflineEngine:
  """Class for handling offline inference on batches of inputs.

  The logic is the following:
  1. Process input one at a time
  2. Pad the input data to the nearest power of 2
  3. Prefill input and insert kv cache
  4. Keep on processing inputs and prefilling until
  there are enough samples to do batch decoding
  5. Decode until at least one of the samples is finished,
  so there is room for new samples
  6. Prefill to fill up the newly emptied decode slots
  7. Repeat step 5 and 6 until all samples are finished
  8. A background thread is used to detokenize the results
  9. Return the results

  Prefill Packing:
      When enable_batch_prefill is True, the prefill processor
      will pack multiple inputs into a single sequence before
      doing the prefill.

      There are multiple buckets for packed sequences, where each bucket
      contains inputs with same padded length. Only inputs with same
      padded length will be packed together.

      It is important to sort the inputs by padded length so that the
      buckets fill up quickly.

      When a decode slot is freed up, the prefill processor will add the
      sequence to the bucket. If the bucket is full, the bucket will be
      prefilled.

      E.g.
      Bucket for length 64: [...seq1, ...seq2, ...seq3, ...seq4]
      Bucket for length 128: [...seq1, ...seq2]
      Bucket for length 256: [...seq1]
  """

  def __init__(
      self,
      engine: MaxEngine,
      params: Optional[Params] = None,
      enable_batch_prefill: bool = False,
      min_decode_steps: int = 10,
      tokenizer: Any = None,
      prefill_lengths: Union[List[int], str] = "auto",
      batch_prefill_max_batch_size: int = 16,
  ):
    """
    Args:
        engine: MaxEngine instance for running inference
        params: Model parameters (loaded from engine if None)
        enable_batch_prefill: Whether to use prefill packing.
            config.scan_layers must be False if this is True
        min_decode_steps: Number of decode steps to perform
            before checking for completion.
        tokenizer: Tokenizer instance for encoding/decoding text
        prefill_lengths: List of expected prefill lengths, or "auto" to
            automatically determine appropriate lengths from the engine
            config. Input sequences will be padded to the nearest length
            in this list.

    """
    # Engine and parameters
    self.engine = engine
    self.params = params
    self.min_decode_steps = min_decode_steps
    self.enable_batch_prefill = enable_batch_prefill
    self.batch_size = engine.max_concurrent_decodes
    self.max_prefill_length = engine.config.max_prefill_predict_length
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    self.tokenizer = tokenizer
    self.validate_config()

    if prefill_lengths == "auto":
      # Create buckets: [0, 64], (64, 128], (128, 256], ..., [max_length//2, max_length]
      self.prefill_lengths = [2**i for i in range(6, max(6, (self.max_prefill_length - 1).bit_length()) + 1)]
    else:
      self.prefill_lengths = sorted(prefill_lengths)

    # Prefill processing setup
    if enable_batch_prefill:
      self.prefill = PrefillHelper(PrefillType.BATCH, self.engine, batch_prefill_max_batch_size)
    else:
      self.prefill = PrefillHelper(PrefillType.DEFAULT, self.engine)

    # State management
    self.detokenize_backlog = queue.Queue(maxsize=100)
    self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
    self.slot_to_id = {}  # Maps decode slots to input ids
    self.empty_slots = []  # Available decode slots
    self.decode_state = None
    self.res = None  # Results storage

    # Compiled functions
    self.generate_fn = None

    self._init()

  def _init(self):
    """Initialize the inference components.

    Loads model parameters and tokenizer if not provided.
    """

    # Load model parameters if not provided
    if self.params is None:
      self.params = self.engine.load_params()

    # Create tokenizer if not provided
    if self.tokenizer is None:
      tokenizer_params = self.engine.get_tokenizer()
      self.tokenizer = self.engine.build_tokenizer(tokenizer_params)

    # Initialize decode state
    self.generate_fn, self.params, init_decode_state_fn = self.engine.aot_compile(self.params, pass_rng_shape=False)
    self.decode_state = init_decode_state_fn(None)

  def batch_inference(
      self,
      data: Union[List[InputData], List[jax.Array]],
      data_is_padded: bool = False,
      desc: str = "",
  ) -> Dict[str, List[int]]:
    """Run inference on a batch of inputs.

    Args:
        data: List of InputData objects containing input sequences
        desc: Description string for logging

    Returns:
        Dictionary mapping input ids to output token sequences
    """
    # Prepare input data (pad, and sort if using batch prefill)
    data_is_jax_array = isinstance(data[0], jax.Array)
    data = self.prepare_data(data, data_is_padded)

    # Reset state
    self.counter = EventCounter(input=len(data), prefill=0, decode=0, detokenize=0)
    self.empty_slots = list(range(self.batch_size))
    self.slot_to_id = {}
    self.res = defaultdict(list)

    # Start detokenization thread
    detokenize_thread = JetThread(
        target=functools.partial(
            self.detokenize,
        ),
        name="detokenize",
    )
    detokenize_thread.start()

    # Process each input
    for row in data:
      # 1. Wait for an empty slot
      while not self.empty_slots:
        self.decode()

      # 2. Get an available slot
      slot = self.empty_slots.pop()
      # 3. Prefill and insert kv cache
      self.prefill.process(
          model_params=self.params,
          decode_state=self.decode_state,
          decode_slot=slot,
          input_id=row.id,
          input_tokens_padded=row.tokens,
          input_true_length=row.true_length,
          max_length=self.max_prefill_length,
          prefill_done=self.prefill_done,
      )

    # 4. Flush any pending inputs in batch prefill mode
    self.prefill.finalize(self.params, self.decode_state, self.prefill_done)

    # 5. Continue decoding until all sequences are complete
    while self.slot_to_id:
      self.decode()

    # Wait for detokenization to complete
    detokenize_thread.join()

    # Log completion statistics
    log.info(
        "summary-%s-prefills-%d-decodes-%d-detokens-%d completed.",
        desc,
        self.counter.prefill,
        self.counter.decode,
        self.counter.detokenize,
    )

    if data_is_jax_array:
      res = [self.res[input_data.id] for input_data in data]
      return res

    return self.res

  def sort_data(self, data: List[InputData]) -> List[InputData]:
    """Sort input data by padded length. This helps with batch prefilling by
    filling up the buckets quickly.
    """
    data_dict = defaultdict(list)
    padded_lengths = []

    # Group inputs by padded length
    for row in data:
      padded_lengths.append(row.tokens.shape[0])
      data_dict[row.tokens.shape[0]].append(row)

    data = []
    for padded_len in sorted(set(padded_lengths)):
      log.info("padded len: %d, num: %d", padded_len, len(data_dict[padded_len]))
      data += data_dict[padded_len]

    return data

  def pad_data(self, data: List[InputData]) -> List[InputData]:
    """For each input, pad it to the next length in self.prefill_lengths
    that is greater than or equal to its true length.
    """
    padded_data = []

    for item in data:
      # Find the smallest prefill length that can accommodate this input
      target_length = None
      for length in self.prefill_lengths:
        if length >= item.true_length:
          target_length = length
          break

      # If no suitable length found, use the maximum prefill length
      if target_length is None:
        target_length = self.max_prefill_length

      if len(item.tokens) < target_length:
        # Pad with zeros
        padded_tokens = jax.numpy.zeros(target_length, dtype=item.tokens.dtype)
        padded_tokens = padded_tokens.at[: item.true_length].set(item.tokens[: item.true_length])
      else:
        # Input is too long, truncate to max_prefill_length
        padded_tokens = item.tokens[:target_length]

      # Create new InputData with padded tokens
      padded_data.append(InputData(id=item.id, tokens=padded_tokens, true_length=item.true_length))

    return padded_data

  def prepare_data(self, data: List[InputData], data_is_padded: bool = False) -> List[InputData]:
    """Prepare input data for inference by padding and optionally sorting."""
    if isinstance(data[0], jax.Array):
      data = [InputData(id=i, tokens=array, true_length=len(array)) for i, array in enumerate(data)]

    if not data_is_padded:
      data = self.pad_data(data)

    # Sort data by length if using batch prefill
    if self.enable_batch_prefill:
      # Sort data by length when doing batch prefilling so
      # buckets fill up quickly
      return self.sort_data(data)

    return data

  def emit_token(self, prompt_id, token):
    """Adds the token to the results for the specified prompt ID and
    determines if generation should terminate.

    Args:
        prompt_id: ID of the prompt
        token: Token to emit

    Returns:
        True if this token signals the end of generation, False otherwise
    """
    already_reached_eos = len(self.res[prompt_id]) > 0 and self.res[prompt_id][-1] == self.tokenizer.eos_id

    if not already_reached_eos:
      self.res[prompt_id].append(token)

    return token == self.tokenizer.eos_id

  def prefill_done(self, prefill_result, prompt_ids, decode_state):
    """Callback function called when prefill completes.
    This function adds the prefill tokens to the detokenization queue,
    which manages the token emission and decode slot evictions.

    Args:
        prefill_result: List of (token, slot) tuples
        prompt_ids: List of prompt IDs
        decode_state: Updated decode state
    """
    # Update decode state
    self.decode_state = decode_state

    # Process each prefill result
    for i, (first_token, slot) in enumerate(prefill_result):
      self.counter.prefill += 1
      self.slot_to_id[slot] = prompt_ids[i]

      # Add token to detokenization queue
      self.detokenize_backlog.put((first_token, True, prompt_ids[i], slot), block=True)

  def decode(self):
    """Run decode steps on current decoder state.

    Performs `self.min_decode_steps` decode operations
    and puts results in the detokenization queue.
    """
    for i in range(self.min_decode_steps):
      # Generate next tokens
      self.decode_state, result_tokens = self.generate_fn(self.params, self.decode_state, None)
      # Add results to detokenization queue
      self.detokenize_backlog.put((result_tokens.convert_to_numpy(), False, 0, 0), block=True)

    self.counter.decode += 1

  def detokenize(self):
    """Detokenize results and manage decode slots.

    Runs in a background thread to process tokens from
    the detokenization queue, emit tokens, and free up
    decode slots when sequences complete.
    """
    while self.counter.detokenize < self.counter.input:

      newly_empty = []

      # Get next item from queue with timeout
      result_tokens, is_first_token, row_id, slot = self.detokenize_backlog.get()

      # Process generated tokens
      if is_first_token:
        first_token, is_valid, length = result_tokens.data[0]
        should_terminate = self.emit_token(row_id, first_token)

        # Free up slot if terminated
        if should_terminate:
          newly_empty.append(slot)
      else:
        for slot, id_ in self.slot_to_id.items():
          token, is_valid, length = result_tokens.data[slot]
          should_terminate = False
          if is_valid:
            should_terminate = self.emit_token(id_, token.item())

          # Free up slot if terminated or max length reached
          if should_terminate or length >= self.max_decode_length:
            newly_empty.append(slot)

      # Update decode slots
      for slot in newly_empty:
        self.counter.detokenize += 1
        del self.slot_to_id[slot]
        self.empty_slots.append(slot)

  def validate_config(self):
    if self.enable_batch_prefill and self.engine.config.scan_layers:
      raise ValueError("scan_layers must be False if enable_batch_prefill is True")
