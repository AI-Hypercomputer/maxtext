# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Tuple, Callable # Duplicate Callable here
import dataclasses
from collections import defaultdict
import jax
from jax import numpy as jnp
from jax.experimental import layout
import numpy as np
import queue
import os
import functools
import threading
import traceback
import signal
import random
# import uuid # Add if you want to generate UUIDs for request_id

from jetstream.engine import engine_api

import logging
# pylint: disable=no-name-in-module
from MaxText.maxengine import MaxEngine
from MaxText.maxengine import set_engine_vars_from_base_engine
from MaxText.prefill_packing import PrefillProcessor # Used if not paged_direct
from MaxText.prefill_packing import BatchedPrefillProcessor # Used if not paged_direct

DecodeState = Any
Params = Any

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array
  true_length: int


@dataclasses.dataclass
class EventCounter:
  input: int
  prefill: int
  decode: int
  detokenize: int


class JetThread(threading.Thread):

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


class PrefillHelper:
  """Helper class to manage prefill related code and provide a unified interface."""

  def __init__(self, type: str, engine: MaxEngine):
    self.engine = engine
    self._original_type = type # Store the requested type

    # Explicitly check for paged attention to modify behavior
    if self.engine.config.attention == "paged":
      log.info(
          "Paged attention is enabled. Overriding prefill strategy to 'paged_direct'."
      )
      self._type = "paged_direct"
    else:
      self._type = type

    log.info(f"PrefillHelper initialized with effective type: {self._type}")

    if self._type == "default":
      self._processor = PrefillProcessor(engine)
    elif self._type == "batch":
      self._batch_processor = BatchedPrefillProcessor(engine=engine, max_batch_size=16)
      # Fallback for batch processor might still use PrefillProcessor
      self._processor = PrefillProcessor(engine)
    elif self._type == "dummy":
      pass
    elif self._type == "paged_direct":
      # No specialized processor instance needed for paged_direct;
      # we will call engine methods directly.
      pass
    else:
      raise ValueError(f"Invalid effective prefill type: {self._type}")

  def aot_compile(
      self,
      max_length: int,
      params: Params,
      params_layout: layout.Layout,
      decode_state_layout: layout.Layout,
      decode_state_shape: jax.ShapeDtypeStruct,
  ) -> None:
    if self._type == "paged_direct":
      log.info(
          "Skipping PrefillHelper AOT for 'paged_direct' type. "
          "Relying on internal JIT of engine.prefill and engine.insert."
      )
      return

    if max_length > 4096: # This check can remain for other types
        raise ValueError(f"Max length for AOT exceeds 4096. {max_length=}")

    if self._type == "default":
      # length buckets = (0, 64], (64, 128], (128, 256], ...
      buckets = [2**i for i in range(6, max(6, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        self._processor.aot_compile(params, bucket)
    elif self._type == "batch":
      # length buckets = (0, 128], (128, 256], (256, 512], ...
      buckets = [2**i for i in range(7, max(7, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        # Calculate a reasonable num_prompts for AOT compilation
        # This could be a fixed small set, e.g., 1 and a few typical values.
        # For simplicity, using a small range or a typical value.
        # The original code was: for num_prompts in range(1, 2 * max_length // bucket):
        # This can lead to many compilations. Let's cap it or use representative values.
        # Example: compile for num_prompts = 1 and a typical packing number like 4.
        for num_prompts_aot in [1, min(4, 2 * max_length // bucket if bucket > 0 else 1)]:
            if num_prompts_aot > 0 :
                 self._batch_processor.aot_compile(params, bucket, max_length, num_prompts_aot)

      # for fallback for BatchedPrefillProcessor
      for bucket_fallback in [max_length]: # Fallback for single, large prompts
          self._processor.aot_compile(params, bucket_fallback)
    elif self._type == "dummy":
        pass # No AOT for dummy
    else: # Should not happen due to constructor logic
        raise RuntimeError(f"Unexpected type in aot_compile: {self._type}")


  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: str, # Changed from int to str to match InputData.id
      input_tokens_padded: jax.Array,
      input_true_length: int,
      max_length: int, # Passed from OfflineInference
      prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[str], DecodeState], None],
  ) -> None:
    # request_id_uuid = uuid.uuid4() # Example if you want unique request IDs
    request_id_for_engine = None # MaxEngine methods handle None for request_id

    if self._type == "paged_direct":
      log.debug(f"PrefillHelper ('paged_direct'): Processing slot={decode_slot}, id={input_id}")
      # Directly call engine prefill and insert. These are JITted methods in MaxEngine.
      prefill_output_prefix, result_tokens = self.engine.prefill(
          params=model_params,
          padded_tokens=input_tokens_padded,
          true_length=input_true_length,
          slot=decode_slot,
          request_id=request_id_for_engine
      )
      new_decode_state = self.engine.insert(
          prefix=prefill_output_prefix,
          decode_state=decode_state,
          slot=decode_slot,
          request_id=request_id_for_engine
      )
      # Ensure input_id list matches prefill_done signature
      prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
      return

    # Fallback to original logic for other types
    # print(f"PrefillHelper: {self._type=}") # Original debug print
    padded_length = len(input_tokens_padded)
    if self._type == "default":
      # Assumes PrefillProcessor.process is corrected to pass slot
      result_tokens, new_decode_state = self._processor.process(
          model_params, decode_state, decode_slot, input_tokens_padded, input_true_length
      )
      prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
    elif self._type == "batch":
      if padded_length == max_length: # Original condition from offline_inference for fallback
        # fallback to default mode (PrefillProcessor)
        result_tokens, new_decode_state = self._processor.process(
            model_params, decode_state, decode_slot, input_tokens_padded, input_true_length
        )
        prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
      else:
        self._batch_processor.process(
            model_params=model_params,
            decode_state=decode_state,
            decode_slot=decode_slot,
            input_id=input_id, # BatchedPrefillProcessor expects input_id
            input_prompt=input_tokens_padded[:input_true_length],
            input_padding=padded_length, # This is max_length for the bucket
            capacity=max_length, # This is the 'capacity' of the prefill bucket
            prefill_done=prefill_done,
        )
    elif self._type == "dummy":
      log.debug("PrefillHelper ('dummy'): Dummy prefill")
      # Create a dummy ResultTokens matching expected structure if necessary
      # For simplicity, keeping original dummy behavior
      # This might need adjustment based on how `prefill_done` uses the token.
      dummy_result_tokens_data = engine_api.ResultTokens(
          data=np.array([[123, 1, 0]]), # token_id, valid, length
          tokens_idx=(0,1),
          valid_idx=(1,2),
          length_idx=(2,3),
          samples_per_slot=1
      )
      prefill_done([(dummy_result_tokens_data, decode_slot)], [str(input_id)], decode_state)
    else:
        raise RuntimeError(f"Unexpected type in process: {self._type}")


  def finalize(
      self,
      model_params: Params,
      decode_state: DecodeState,
      prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[str], DecodeState], None],
  ) -> None:
    if self._type == "paged_direct":
      # Nothing to flush for direct calls
      pass
      return

    # Original finalize logic
    if self._type == "default":
      pass # PrefillProcessor has no explicit finalize/flush
    elif self._type == "batch":
      self._batch_processor.flush(model_params, decode_state, prefill_done)
    elif self._type == "dummy":
      pass # No finalization for dummy
    else:
        raise RuntimeError(f"Unexpected type in finalize: {self._type}")


class OfflineInference:

  def __init__(self, engine: engine_api.Engine, params, base_engine: engine_api.Engine, enable_batch_prefill: bool):
    self.live = False
    self.engine = engine
    self.decode_state = None
    self.decode_state_executable = None # Corrected: was _decode_state_executable
    if params is None:
      self.relayout_params = True
      params = engine.load_params()
    else:
      self.relayout_params = False
      rng = jax.random.PRNGKey(0)
      set_engine_vars_from_base_engine(engine, base_engine, rng)
    self.params = params

    self.dummy = False # User's debug flag

    # Determine prefill strategy based on paged attention and flags
    effective_prefill_type = "default" # Default
    if self.dummy:
        effective_prefill_type = "dummy"
    # Paged attention check now happens inside PrefillHelper constructor
    # We still pass the user's preference (enable_batch_prefill or default)
    elif enable_batch_prefill:
        effective_prefill_type = "batch"
    # else it remains "default"

    # PrefillHelper will internally adjust to "paged_direct" if engine.config.attention == "paged"
    self.prefill = PrefillHelper(effective_prefill_type, self.engine)

    self.batch_size = engine.max_concurrent_decodes
    self.max_prefill_length = engine.config.max_prefill_predict_length
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    self.tokenizer = engine.build_tokenizer(engine.get_tokenizer())

    self._cached_generate = None
    self.detokenize_backlog = queue.Queue(10)

    # self._decode_state_executable = None # Already initialized above

  def init_decode_state(self):
    if self.decode_state is None:
      assert self.decode_state_executable is not None, "Decode state executable is none" # Corrected: was _decode_state_executable
      self.decode_state = self.decode_state_executable(None) # Assuming it takes one arg like rng

  def warmup(self, max_length, warmup_samples):
    # AOT compile the main generation function
    self._cached_generate, self.params, self.decode_state_executable = self.engine.aot_compile( # Corrected: was _decode_state_executable
        self.params, pass_rng_shape=False # Assuming False for typical inference
    )

    self.init_decode_state() # Initialize decode state after getting the executable

    # AOT compile prefill related components via PrefillHelper
    # PrefillHelper's aot_compile will now handle different types, including 'paged_direct'
    self.prefill.aot_compile(
        max_length, # This is the bucket size for AOT, not necessarily self.max_prefill_length
        self.params,
        self.engine.param_layouts, # Assuming these are accessible post engine.aot_compile
        self.engine.decode_state_layouts, # Assuming these are accessible
        self.engine.decode_state_shapes # Assuming these are accessible
    )

    if warmup_samples: # Only run batch_inference if there are samples
        self.batch_inference(warmup_samples, desc="warmup")
    else:
        log.info("Skipping warmup batch_inference as no warmup_samples provided.")

  def batch_inference_with_callback(
      self,
      data: List[InputData],
      emit_first_token: Callable[[str, int], bool],
      emit_token: Callable[[str, int], bool],
      desc: str,
  ):
    """callback is a function that takes id and token. It will be called once per output token."""

    empty_slots = list(range(self.batch_size))
    # Ensure slot_to_id correctly maps slots (int) to original input IDs (str)
    slot_to_id: dict[int, str] = {}


    counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
    dummy_length = 1 # For dummy mode generate

    # prefill_done callback signature matches PrefillHelper's expectation
    def prefill_done_callback(
        prefill_results: List[Tuple[engine_api.ResultTokens, int]],
        ids: List[str], # List of input_ids (strings)
        current_decode_state: DecodeState
    ):
      nonlocal self # Not strictly needed if only accessing self.decode_state
      nonlocal counter
      self.decode_state = current_decode_state # Update shared decode_state
      for i in range(len(prefill_results)):
        result_tokens_for_sample, slot_for_sample = prefill_results[i]
        id_for_sample = ids[i]
        counter.prefill += 1
        log.debug(f"Prefill done for id={id_for_sample} in slot={slot_for_sample} (Total prefills: {counter.prefill})")
        # Put (ResultTokens, is_first_token, original_input_id, slot_where_prefilled)
        self.detokenize_backlog.put((result_tokens_for_sample, True, id_for_sample, slot_for_sample), block=True)

    def decode():
      nonlocal self, dummy_length, counter # Ensure all needed variables are nonlocal if required by nesting
      counter.decode += 1
      if self.dummy:
        log.info("Dummy generate step")
        res_data = []
        for _ in range(self.batch_size):
             # token_id, is_valid, current_length
             res_data.append([random.randint(1,1000), 1, dummy_length])
        res = engine_api.ResultTokens(
            data=np.array(res_data), tokens_idx=(0, 1), valid_idx=(1, 2),
            length_idx=(2, 3), samples_per_slot=1 # Assuming one sample per slot for dummy
        )
        dummy_length += 1
        result_tokens_obj = res
        # Note: self.decode_state is not updated here in dummy mode for simplicity.
        # A more complete dummy mode might need to update a dummy decode_state.
      else:
        # ALWAYS call self.engine.generate (the Python public API method).
        # This method handles its own Python-level state updates (like self.page_state)
        # and then calls its internal JITted kernel (_generate_jit).
        # The _cached_generate from engine.aot_compile() is NOT used here.
        log.debug("Calling self.engine.generate (Python wrapper) for decode step.")
        self.decode_state, result_tokens_obj = self.engine.generate(
            params=self.params, 
            decode_state=self.decode_state, 
            sampler=None # Pass sampler if it's used, None otherwise
            # rng is handled by self.engine.generate if needed
        )
      
      # Send the result (from either dummy or actual generation) to the detokenize backlog
      self.detokenize_backlog.put((result_tokens_obj, False, "N/A", -1), block=True)


    def detokenize():
      nonlocal self # Not strictly needed
      nonlocal slot_to_id
      nonlocal empty_slots
      nonlocal counter
      while self.live or not self.detokenize_backlog.empty(): # Process queue even if not live
        try:
            # Timeout to prevent indefinite blocking if self.live becomes false and queue is empty
            result_data, is_first_token, row_id_str, prefill_slot = self.detokenize_backlog.get(block=True, timeout=0.1)
        except queue.Empty:
            if not self.live: # If not live and queue is empty, exit
                break
            continue # If live, continue trying to get from queue

        # Convert to numpy if it's not already (JAX arrays might be device-local)
        if hasattr(result_data, 'convert_to_numpy'):
            actual_result_tokens = result_data.convert_to_numpy()
        else: # Assuming it's already numpy or a compatible structure
            actual_result_tokens = result_data


        if is_first_token:
          # This ResultTokens object is from prefill, containing one sample.
          # Data shape: (samples_per_slot, fields) e.g. (1,3)
          # Accessing the first (and only) sample's token
          first_token_id = actual_result_tokens.data[0, actual_result_tokens.tokens_idx[0]].item()
          should_terminate = emit_first_token(row_id_str, first_token_id)
          if not should_terminate:
            slot_to_id[prefill_slot] = row_id_str # Track active request
            log.debug(f"Detokenize: First token for id={row_id_str} in slot={prefill_slot}. Tracking.")
          else:
            empty_slots.append(prefill_slot) # Slot is immediately free
            self.engine.free_resource(prefill_slot) # Free paged attention resources
            counter.detokenize += 1 # Count as completed
            log.debug(f"Detokenize: First token for id={row_id_str} (slot {prefill_slot}) was EOS or max_len. Request finished.")
          self.detokenize_backlog.task_done() # Mark task as done
          continue

        # This is a result from a decode step, containing tokens for all active slots
        # actual_result_tokens.data is expected to be a NumPy array [batch_size, num_fields]
        newly_empty_slots_this_step = []
        active_slots_before_decode = list(slot_to_id.keys()) # Iterate over slots active before this decode result

        for current_slot_idx in active_slots_before_decode:
            if current_slot_idx not in slot_to_id: # Slot might have been freed by a previous iteration
                continue

            current_input_id = slot_to_id[current_slot_idx]
            
            # Ensure current_slot_idx is within bounds of the data array
            if current_slot_idx >= actual_result_tokens.data.shape[0]:
                log.warning(f"Detokenize: slot index {current_slot_idx} out of bounds for decode result data shape {actual_result_tokens.data.shape}. Skipping.")
                continue

            token_val, is_valid_val, length_val = (
                actual_result_tokens.data[current_slot_idx, actual_result_tokens.tokens_idx[0]].item(),
                actual_result_tokens.data[current_slot_idx, actual_result_tokens.valid_idx[0]].item(),
                actual_result_tokens.data[current_slot_idx, actual_result_tokens.length_idx[0]].item()
            )

            log.debug(f"Detokenize (decode step): slot={current_slot_idx}, id={current_input_id}, token={token_val}, valid={is_valid_val}, length={length_val}")

            should_finish_request = False
            if is_valid_val: # Process only if the token is valid for this slot
              should_finish_request = emit_token(current_input_id, token_val)

            # Check for termination conditions
            if should_finish_request or length_val >= self.max_decode_length:
              if current_slot_idx not in newly_empty_slots_this_step: # Avoid double-adding
                  newly_empty_slots_this_step.append(current_slot_idx)
              log.debug(f"Detokenize: Request for id={current_input_id} (slot {current_slot_idx}) finished. Length={length_val}.")


        for slot_to_free in newly_empty_slots_this_step:
          if slot_to_free in slot_to_id: # Check if still tracked
            del slot_to_id[slot_to_free] # Stop tracking
            empty_slots.append(slot_to_free) # Add to available slots
            self.engine.free_resource(slot_to_free) # Free paged attention resources
            counter.detokenize += 1 # Increment completed count
            log.debug(f"Detokenize: Slot {slot_to_free} freed up. Total detokenized: {counter.detokenize}")
        
        self.detokenize_backlog.task_done() # Mark task as done

      # After loop, if not live, ensure any remaining backlog items are processed (covered by loop condition)


    detokenize_thread = JetThread(
        target=detokenize, # Removed functools.partial as not needed
        name="detokenize_thread", # More descriptive name
    )

    counter.input = len(data)
    self.live = True

    detokenize_thread.start()

    # Main input processing loop
    for row_idx, row_data in enumerate(data):
      # Wait for a free slot
      while not empty_slots:
        if not slot_to_id and self.detokenize_backlog.empty() and not self.live : # All processed
             break
        log.debug("MainLoop: No empty slots, triggering decode cycle.")
        if slot_to_id: # Only decode if there are active requests
            decode()
        else: # No active requests, but backlog might have first tokens, or waiting for new inputs
            # If backlog is also empty, this means we are waiting for inputs or everything finished.
            # Adding a small sleep to prevent busy-waiting if decode() is not called.
            threading.Event().wait(0.01) # Small wait to yield control
      
      if not self.live and not empty_slots: # If processing is shutting down
          break

      if not empty_slots: # Should not happen if self.live is true and loop continues
          log.warning("MainLoop: Still no empty slots after decode cycle, but proceeding. This might indicate an issue.")
          # If this happens, it means decode didn't free up a slot, which is problematic.
          # Forcing a slot by picking one might be bad, but for now, let's see.
          # This path implies something is wrong with slot management or termination.
          # Forcing a break or raising an error might be better.
          if slot_to_id: # If there are active slots, decode again
              decode()
              continue # Retry getting an empty slot
          else: # No active slots, but also no empty_slots, very weird state.
              log.error("MainLoop: Inconsistent state - no empty slots and no active_slots. Breaking.")
              break


      # Get a free slot
      # It's possible empty_slots is momentarily empty if detokenize is slow
      # to add them back after a decode. Add a small wait with timeout.
      current_empty_slot = -1
      try:
          current_empty_slot = empty_slots.pop(0) # Get first available slot
      except IndexError:
          log.warning("MainLoop: empty_slots list was empty, waiting briefly.")
          threading.Event().wait(0.05) # wait a bit for detokenize thread
          if not empty_slots:
              log.error("MainLoop: Failed to get an empty slot even after waiting. Terminating this request.")
              # Potentially skip this request or error out
              # For now, continue to next request, this one will be dropped
              continue 
          current_empty_slot = empty_slots.pop(0)


      log.debug(f"MainLoop: Processing input id={row_data.id} in slot={current_empty_slot}")

      # Submit prefill request
      # The 'max_length' here refers to the padded length of the input_tokens for this specific prefill operation
      # which corresponds to a bucket size or max_prefill_length for single prefills.
      # Using self.max_prefill_length seems appropriate if not using bucketing here.
      # The PrefillHelper's `process` method takes `max_length` which is used by BatchedPrefillProcessor
      # for 'capacity' and 'input_padding'. For 'default' or 'paged_direct', it's less critical
      # but should align with the padding of `input_tokens_padded`.
      # Assuming `row_data.tokens.shape[0]` is the padded length for this item.
      self.prefill.process(
          self.params,
          self.decode_state,
          current_empty_slot,
          row_data.id,
          row_data.tokens, # Padded tokens for this specific input
          row_data.true_length,
          row_data.tokens.shape[0], # Pass the actual padded length of this item as 'max_length' for this call
          prefill_done_callback
      )

    # After all inputs are submitted, finalize any batched prefills
    log.info("MainLoop: All inputs submitted. Finalizing prefills...")
    self.prefill.finalize(self.params, self.decode_state, prefill_done_callback)

    # Wait for all active slots to complete decoding
    log.info("MainLoop: Waiting for active requests to complete decoding...")
    while slot_to_id: # While there are still active requests
      if self.detokenize_backlog.empty() and not self.live: # Graceful shutdown check
          log.warning("MainLoop: Shutting down with active slots but empty backlog.")
          break
      log.debug(f"MainLoop: Waiting for {len(slot_to_id)} active slots. Triggering decode.")
      decode()
      # Add a small delay to allow detokenize thread to process and free slots
      # This helps prevent decode() from being called too rapidly if detokenize is slower.
      # However, if detokenize is stuck, this won't help.
      # The primary control should be `while not empty_slots` at the beginning of the loop.
      # threading.Event().wait(0.01) # Optional small delay

    log.info("MainLoop: All active slots processed or finalized.")
    self.live = False # Signal detokenize thread to stop trying to get new items indefinitely
    # detokenize_backlog.join() # Wait for all items in queue to be processed by task_done()
    log.info("MainLoop: Waiting for detokenize thread to finish processing backlog...")
    detokenize_thread.join() # Wait for the thread to terminate

    log.info(
        "Summary for '%s': Prefills=%d, Decodes=%d, Detokenized_Requests=%d (Total Inputs: %d).",
        desc, counter.prefill, counter.decode, counter.detokenize, counter.input
    )


  def batch_inference(self, data: List[InputData], desc="") -> dict[str, List[int]]:
    """data is list of obj with id, tokens, and true length"""
    # Sorting data by padded length - this is a form of bucketing for batching.
    # If paged attention is on and we are using 'paged_direct', this sorting might be less critical
    # for prefill performance, but doesn't harm.
    data_dict = defaultdict(list)
    log.info("Sorting input data by padded length...")
    for row in data:
      data_dict[row.tokens.shape[0]].append(row)

    # Example of combining smaller buckets into larger ones (optional optimization)
    # if 64 in data_dict and 128 in data_dict: # Check if keys exist
    #    data_dict[128].extend(data_dict.pop(64, []))

    sorted_data: List[InputData] = []
    # Process in increasing order of padded length, or a specific order
    # Using a fixed set of expected padded lengths for ordering
    # This should ideally come from config or be more dynamic.
    # Assuming these are the relevant bucket sizes used for padding.
    # For paged_direct, actual padding of row.tokens matters.
    # For PrefillProcessor, it AOTs for specific buckets.
    expected_padded_lengths = sorted(data_dict.keys()) # Process existing padded lengths in order

    for padded_len in expected_padded_lengths:
      log.info(f"Padded length bucket: {padded_len}, Number of items: {len(data_dict[padded_len])}")
      random.shuffle(data_dict[padded_len]) # Shuffle within a bucket
      sorted_data.extend(data_dict[padded_len])
    log.info("Finished organizing input data.")

    # Output dictionary
    res: dict[str, List[int]] = defaultdict(list) # Maps input_id (str) to list of token_ids

    # Callback for first token
    def emit_first_token_callback(id_str: str, token: int) -> bool:
      nonlocal res
      # Log first token received
      log.debug(f"Output: ID '{id_str}', First Token: {token}")
      if token == self.tokenizer.eos_id:
        log.debug(f"Output: ID '{id_str}' received EOS as first token.")
        # No need to add to res[id_str] if it's EOS immediately
        return True # Terminate if EOS
      res[id_str].append(token)
      return False # Continue decoding

    # Callback for subsequent tokens
    def emit_token_callback(id_str: str, token: int) -> bool:
      nonlocal res
      # Ensure list exists for id_str (should be created by first_token_callback)
      if id_str not in res and token != self.tokenizer.eos_id :
          log.warning(f"Output: ID '{id_str}' missing from results dict but received token {token}. This might indicate an issue if it's not the first token.")
          # If it's truly not the first and not EOS, append.
          # However, emit_first_token_callback should handle the first.
          # This path is defensive.
          # res[id_str].append(token)

      # Append token if not EOS, or if EOS and list is empty (first token was EOS)
      # The main logic is to stop appending *after* EOS.
      if not res[id_str] or res[id_str][-1] != self.tokenizer.eos_id:
          res[id_str].append(token)

      if token == self.tokenizer.eos_id:
        log.debug(f"Output: ID '{id_str}' received EOS token.")
        return True # Terminate if EOS
      return False # Continue decoding

    self.batch_inference_with_callback(
        sorted_data,
        emit_first_token=emit_first_token_callback,
        emit_token=emit_token_callback,
        desc=desc
    )
    return dict(res) # Convert defaultdict to dict for return