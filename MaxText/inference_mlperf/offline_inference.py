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

from typing import Any, Callable, List, Tuple
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

from jetstream.engine import engine_api

import logging
# pylint: disable=no-name-in-module
from MaxText.maxengine import MaxEngine # Assuming this is your MaxEngine class
from MaxText.maxengine import set_engine_vars_from_base_engine
from MaxText.prefill_packing import PrefillProcessor
from MaxText.prefill_packing import BatchedPrefillProcessor

from MaxText import profiler
import pstats

# Imports for timing and logging
import time
import pandas as pd

# Global list to store performance log entries
performance_log = []

DecodeState = Any
Params = Any
log = logging.getLogger(__name__)

# --- Helper function to add PageManager stats to log entries ---
def _add_pagemanager_stats_to_log(log_entry: dict, engine_instance: MaxEngine):
    if hasattr(engine_instance, 'config') and getattr(engine_instance.config, 'attention', None) == "paged" and \
       hasattr(engine_instance, 'page_state') and engine_instance.page_state is not None:
        try:
            # Ensure attributes exist before trying to access .item()
            if hasattr(engine_instance.page_state, 'stat_total_pages_allocated'):
                log_entry["pg_alloc_cum"] = engine_instance.page_state.stat_total_pages_allocated.item()
            if hasattr(engine_instance.page_state, 'stat_total_pages_released'):
                log_entry["pg_release_cum"] = engine_instance.page_state.stat_total_pages_released.item()
            if hasattr(engine_instance.page_state, 'stat_total_find_page_calls'):
                log_entry["pg_find_cum"] = engine_instance.page_state.stat_total_find_page_calls.item()
        except AttributeError as e:
            log.debug(f"PageState missing stat attributes: {e}")
            log_entry["pg_stats_error"] = "PageState missing stat attributes"
        except Exception as e: # Catch other potential errors like .item() on non-JAX array
            log.debug(f"Error accessing PageState stat attributes: {e}")
            log_entry["pg_stats_error"] = str(e)
    return log_entry


class Stats(pstats.Stats):
    # list the tuple indices and directions for sorting,
    # along with some printable description
    sort_arg_dict_default = {
      "calls"             : (((1,-1),                ), "call count"),
      "ncalls"            : (((1,-1),                ), "call count"),
      "cumtime"           : (((4,-1),                ), "cumulative time"),
      "cumulative"        : (((4,-1),                ), "cumulative time"),
      "file"              : (((6, 1),                ), "file name"),
      "filename"          : (((6, 1),                ), "file name"),
      "line"              : (((7, 1),                ), "line number"),
      "module"            : (((6, 1),                ), "file name"),
      "name"              : (((8, 1),                ), "function name"),
      "nfl"               : (((8, 1),(6, 1),(7, 1),), "name/file/line"),
      "pcalls"            : (((0,-1),                ), "primitive call count"),
      "stdname"           : (((9, 1),                ), "standard name"),
      "time"              : (((2,-1),                ), "internal time"),
      "tottime"           : (((2,-1),                ), "internal time"),
      "cumulativepercall": (((5,-1),                ), "cumulative time per call"),
      "totalpercall"      : (((3,-1),                ), "total time per call"),
      }

    def sort_stats(self, *field):
      if not field:
        self.fcn_list = 0
        return self
      if len(field) == 1 and isinstance(field[0], int):
        # Be compatible with old profiler
        field = [ {-1: "stdname",
                   0:  "calls",
                   1:  "time",
                   2:  "cumulative"}[field[0]] ]
      elif len(field) >= 2:
        for arg in field[1:]:
          if type(arg) != type(field[0]): # pylint: disable=unidiomatic-typecheck
            raise TypeError("Can't have mixed argument type")

      sort_arg_defs = self.get_sort_arg_defs()

      sort_tuple = ()
      self.sort_type = ""
      connector = ""
      for word in field:
        if isinstance(word, pstats.SortKey):
          word = word.value
        sort_tuple = sort_tuple + sort_arg_defs[word][0]
        self.sort_type += connector + sort_arg_defs[word][1]
        connector = ", "

      stats_list = []
      # Use self.stats.items() correctly
      for func, (cc, nc, tt, ct, _) in self.stats.items(): # Removed 'callers' as it's not used
        if nc == 0:
          npc = 0.0
        else:
          npc = float(tt)/nc

        if cc == 0:
          cpc = 0.0
        else:
          cpc = float(ct)/cc

        stats_list.append((cc, nc, tt, npc, ct, cpc) + func +
                          (pstats.func_std_string(func), func))

      stats_list.sort(key=pstats.cmp_to_key(pstats.TupleComp(sort_tuple).compare))

      self.fcn_list = fcn_list = []
      for tuple_val in stats_list:
        fcn_list.append(tuple_val[-1])
      return self


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
  free_resource: int # Added for tracking free_resource calls


class JetThread(threading.Thread):
  def run(self):
    try:
      super().run()
    except Exception as e:
      print(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


class PrefillHelper:
  """Helper class to manage prefill related code and provide a unified interface."""
  def __init__(self, prefill_strategy: str, engine: MaxEngine): # Renamed 'type' to 'prefill_strategy'
    self.engine = engine
    self._original_type = prefill_strategy

    self._type = "direct" # Overriding to 'direct' as per original logic in user's file
    log.info(f"PrefillHelper initialized with effective prefill strategy: {self._type} (Original: {self._original_type})")

    if self._type == "default":
      self._processor = PrefillProcessor(engine)
    elif self._type == "batch":
      self._batch_processor = BatchedPrefillProcessor(engine=engine, max_batch_size=16) # Example batch size
      self._processor = PrefillProcessor(engine) # Fallback processor
    elif self._type == "dummy":
      pass # No specific processor needed for dummy
    elif self._type == "direct":
      pass # No specific processor needed for direct calls to engine
    else:
      raise ValueError(f"Invalid effective prefill type: {self._type}")

  def aot_compile(
      self,
      max_length: int,
      params: Params,
      # Removed params_layout, decode_state_layout, decode_state_shape as they were not used when type is 'direct'
  ) -> None:
    if self._type == "direct":
      log.info(
          "Skipping PrefillHelper AOT for 'direct' type. "
          "Relying on internal JIT of engine.prefill and engine.insert."
      )
      return

    # The following logic would apply if _type was not overridden to 'direct'
    if max_length > 4096: # Example check from original file
      raise ValueError(f"Max length for AOT exceeds 4096. {max_length=}")

    if self._type == "default":
      buckets = [2**i for i in range(6, max(6, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        self._processor.aot_compile(params, bucket)
    elif self._type == "batch":
      # Logic from original file
      buckets = [2**i for i in range(7, max(7, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        for num_prompts_aot in [1, min(4, 2 * max_length // bucket if bucket > 0 else 1)]:
          if num_prompts_aot > 0:
            self._batch_processor.aot_compile(params, bucket, max_length, num_prompts_aot)
      for bucket_fallback in [max_length]: # Fallback for lengths not fitting batch buckets
        self._processor.aot_compile(params, bucket_fallback)
    elif self._type == "dummy":
      pass # No AOT for dummy
    else:
      raise RuntimeError(f"Unexpected type in aot_compile: {self._type}")


  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: str,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      max_seq_len_for_op: int, # Renamed from max_length for clarity (specific to this operation)
      prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[str], DecodeState], None],
  ) -> None:
    request_id_for_engine = None # Assuming None as per original logic

    if self._type == "direct":
      log.debug(f"PrefillHelper ('direct'): Processing slot={decode_slot}, id={input_id}")

      # log_entry_base = { # Base info for log entries related to this processing step
      #     "timestamp": time.time(), # Python timestamp for ordering
      #     "request_id": str(input_id),
      #     "slot": decode_slot,
      #     "input_true_length": input_true_length,
      # }

      # --- TIMING engine.prefill ---
      # start_time_prefill = time.monotonic()
      prefill_output_prefix, result_tokens = self.engine.prefill(
          params=model_params,
          padded_tokens=input_tokens_padded,
          true_length=input_true_length,
          slot=decode_slot,
          request_id=request_id_for_engine,
      )
      # jax.block_until_ready(prefill_output_prefix)
      # jax.block_until_ready(result_tokens)
      # end_time_prefill = time.monotonic()
      # prefill_duration_ms = (end_time_prefill - start_time_prefill) * 1000

      # prefill_log_entry = {**log_entry_base, "event_type": "engine_prefill", "duration_ms": prefill_duration_ms}
      # _add_pagemanager_stats_to_log(prefill_log_entry, self.engine) # Add PageManager stats
      # performance_log.append(prefill_log_entry)
      # log.debug(f"Timed engine.prefill for id={input_id}: {prefill_duration_ms:.2f} ms")

      # --- TIMING engine.insert ---
      # start_time_insert = time.monotonic()
      new_decode_state = self.engine.insert(
          prefix=prefill_output_prefix,
          decode_state=decode_state,
          slot=decode_slot,
          request_id=request_id_for_engine
      )
      # jax.block_until_ready(new_decode_state)
      # end_time_insert = time.monotonic()
      # insert_duration_ms = (end_time_insert - start_time_insert) * 1000

      # PageManager stats reflect state *after* prefill, which is when insert operates on its result.
      # insert_log_entry = {**log_entry_base, "event_type": "engine_insert", "duration_ms": insert_duration_ms}
      # _add_pagemanager_stats_to_log(insert_log_entry, self.engine)
      # performance_log.append(insert_log_entry)
      # log.debug(f"Timed engine.insert for id={input_id}: {insert_duration_ms:.2f} ms")

      prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
      return

    # Logic for other prefill types (if _type override is removed)
    padded_length = input_tokens_padded.shape[-1] # Correct way to get length
    if self._type == "default":
      result_tokens, new_decode_state = self._processor.process(
          model_params, decode_state, decode_slot, input_tokens_padded, input_true_length
      )
      prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
    elif self._type == "batch":
      if padded_length == max_seq_len_for_op: # Compare with max_seq_len for this op
        result_tokens, new_decode_state = self._processor.process( # Fallback
            model_params, decode_state, decode_slot, input_tokens_padded, input_true_length
        )
        prefill_done([(result_tokens, decode_slot)], [str(input_id)], new_decode_state)
      else:
        self._batch_processor.process(
            model_params=model_params,
            decode_state=decode_state,
            decode_slot=decode_slot,
            input_id=input_id,
            input_prompt=input_tokens_padded[:input_true_length], # Pass unpadded prompt
            input_padding=padded_length, # Length of the padded input
            capacity=max_seq_len_for_op, # Max capacity for this batch op
            prefill_done=prefill_done,
        )
    elif self._type == "dummy":
      log.debug("PrefillHelper ('dummy'): Dummy prefill")
      dummy_result_tokens_data = engine_api.ResultTokens(
          data=np.array([[123, 1, 0]]), # Example data
          tokens_idx=(0, 1), valid_idx=(1, 2), length_idx=(2, 3), samples_per_slot=1,
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
    if self._type == "direct":
      return # Nothing to finalize for direct calls
    if self._type == "default":
      pass # Default processor might not need explicit finalize
    elif self._type == "batch":
      self._batch_processor.flush(model_params, decode_state, prefill_done)
    elif self._type == "dummy":
      pass # Nothing to finalize for dummy
    else:
      raise RuntimeError(f"Unexpected type in finalize: {self._type}")


class OfflineInference:
  def __init__(self, engine: MaxEngine, params, base_engine: MaxEngine, enable_batch_prefill: bool):
    self.live = False
    self.engine = engine
    self.decode_state = None
    self.decode_state_executable = None # For JITted init_decode_state
    if params is None:
      self.relayout_params = True # Assuming this flag is used elsewhere
      params = engine.load_params() # engine must have load_params method
    else:
      self.relayout_params = False
      rng = jax.random.PRNGKey(0) # Example RNG key
      # Ensure set_engine_vars_from_base_engine handles base_engine=None if it can be
      if base_engine:
          set_engine_vars_from_base_engine(engine, base_engine, rng)
    self.params = params

    self.dummy = False # For enabling dummy prefill/decode

    # Determine prefill strategy
    effective_prefill_strategy = "default"
    if self.dummy:
      effective_prefill_strategy = "dummy"
    elif enable_batch_prefill: # This was a parameter to __init__
      effective_prefill_strategy = "batch"
    # The PrefillHelper itself overrides to "direct" in this version of user's code.

    self.prefill = PrefillHelper(effective_prefill_strategy, self.engine)

    self.batch_size = engine.max_concurrent_decodes
    self.max_prefill_length = engine.config.max_prefill_predict_length
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    self.tokenizer = engine.build_tokenizer(engine.get_tokenizer())

    self._cached_generate = None # May not be used if engine.generate is directly called
    self.detokenize_backlog = queue.Queue(maxsize=self.batch_size * 2) # Increased maxsize

  def init_decode_state(self):
    if self.decode_state is None:
      assert self.decode_state_executable is not None, "Decode state executable is not initialized. Call warmup first."
      # Assuming decode_state_executable takes an optional RNG if needed for init
      self.decode_state = self.decode_state_executable(None) # Or pass an RNG key if required

  def warmup(self, max_length: int, warmup_samples: List[InputData]):
    # AOT compile generate and init_decode_state
    # Assuming engine.aot_compile returns (generate_exe, updated_params, decode_state_init_exe)
    generate_exe, aot_compiled_params, decode_state_init_exe = self.engine.aot_compile(
        self.params, pass_rng_shape=False # Or True if your engine needs it
    )
    # self._cached_generate = generate_exe # Store if planning to call this specific executable
    self.params = aot_compiled_params # Update params to AOT compiled/sharded version
    self.decode_state_executable = decode_state_init_exe

    self.init_decode_state() # Initialize decode state using the JITted function

    # AOT compile prefill parts (delegated to PrefillHelper)
    # PrefillHelper's aot_compile needs params, but not necessarily layouts/shapes from engine if it handles them
    self.prefill.aot_compile(max_length, self.params)

    if warmup_samples:
      log.info("Starting warmup batch_inference...")
      # Clear performance_log before warmup if it's global and you only want post-warmup data
      # global performance_log
      # performance_log.clear()
      self.batch_inference(warmup_samples, desc="warmup")
      log.info("Warmup batch_inference finished.")
      # Optionally clear again if you want to be absolutely sure no warmup data remains
      # performance_log.clear()
    else:
      log.info("Skipping warmup batch_inference as no warmup_samples provided.")

  def batch_inference_with_callback(
      self,
      data: List[InputData],
      emit_first_token: Callable[[str, int], bool],
      emit_token: Callable[[str, int], bool],
      desc: str,
  ):
    empty_slots = list(range(self.batch_size))
    slot_to_id: dict[int, str] = {}
    counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0, free_resource=0)
    dummy_length = 1 # For dummy mode

    def prefill_done_callback(
        prefill_results: List[Tuple[engine_api.ResultTokens, int]],
        ids: List[str],
        current_decode_state: DecodeState,
    ):
      nonlocal self, counter # 'self' is implicitly available
      self.decode_state = current_decode_state # Update engine's decode state
      for i in range(len(prefill_results)):
        result_tokens_for_sample, slot_for_sample = prefill_results[i]
        id_for_sample = ids[i]
        counter.prefill += 1
        log.debug(f"Prefill done for id={id_for_sample} in slot={slot_for_sample} (Total prefills: {counter.prefill})")
        self.detokenize_backlog.put((result_tokens_for_sample, True, id_for_sample, slot_for_sample), block=True)

    def decode():
      nonlocal self, dummy_length, counter, slot_to_id # 'self' is implicitly available
      counter.decode += 1
      active_slots_count_before_decode = len(slot_to_id)
      result_tokens_obj_for_backlog = None # To store result from either dummy or actual generate

      if self.dummy:
        log.info("Dummy generate step")
        res_data = []
        for _ in range(self.batch_size): # Dummy result for all potential slots
          res_data.append([random.randint(1, 1000), 1, dummy_length])
        res = engine_api.ResultTokens(
            data=np.array(res_data),
            tokens_idx=(0, 1), valid_idx=(1, 2), length_idx=(2, 3), samples_per_slot=1,
        )
        dummy_length += 1
        result_tokens_obj_for_backlog = res
        # No specific PageManager stats to log for dummy decode, but could log the event
        # log_entry = {
        #     "timestamp": time.time(), "event_type": "engine_generate_dummy",
        #     "duration_ms": 0, # Dummy op is fast
        #     "active_slots": active_slots_count_before_decode,
        #     "decode_step_count": counter.decode,
        # }
        # performance_log.append(log_entry)
      else:
        log.debug(f"Calling self.engine.generate for decode step {counter.decode} with {active_slots_count_before_decode} active slots.")
        # start_time_generate = time.monotonic()
        # Assuming self.engine.generate is the JAX-JITted or wrapper function
        generated_decode_state, result_tokens_obj_engine = self.engine.generate(
            params=self.params,
            decode_state=self.decode_state, # This state is updated by the engine
            sampler=None, # Assuming sampler is not used or handled internally
        )
        # jax.block_until_ready(generated_decode_state) # Ensure JAX ops complete
        # jax.block_until_ready(result_tokens_obj_engine)
        # end_time_generate = time.monotonic()
        # generate_duration_ms = (end_time_generate - start_time_generate) * 1000

        self.decode_state = generated_decode_state # Persist the updated state
        result_tokens_obj_for_backlog = result_tokens_obj_engine

        # log_entry = {
        #     "timestamp": time.time(), "event_type": "engine_generate",
        #     "duration_ms": generate_duration_ms,
        #     "active_slots": active_slots_count_before_decode,
        #     "decode_step_count": counter.decode,
        # }
        # _add_pagemanager_stats_to_log(log_entry, self.engine) # Add PageManager stats
        # performance_log.append(log_entry)
        # log.debug(f"Timed engine.generate (decode step {counter.decode}): {generate_duration_ms:.2f} ms")

      self.detokenize_backlog.put((result_tokens_obj_for_backlog, False, "N/A", -1), block=True)

    def detokenize():
      nonlocal self, slot_to_id, empty_slots, counter # 'self' is implicitly available
      while self.live or not self.detokenize_backlog.empty():
        try:
          result_data, is_first_token, row_id_str_from_q, prefill_slot_from_q = self.detokenize_backlog.get(block=True, timeout=0.1)
        except queue.Empty:
          if not self.live: # If not live and queue is empty, exit
            break
          continue # If live, continue waiting for items

        actual_result_tokens = result_data.convert_to_numpy() if hasattr(result_data, "convert_to_numpy") else result_data

        if is_first_token:
          # This path is for results from prefill
          # actual_result_tokens.data should be shaped [1, num_features] for prefill of one sample
          first_token_id = actual_result_tokens.data[0, actual_result_tokens.tokens_idx[0]].item()
          should_terminate_request = emit_first_token(row_id_str_from_q, first_token_id)

          if not should_terminate_request:
            slot_to_id[prefill_slot_from_q] = row_id_str_from_q # Track this slot
            log.debug(f"Detokenize: First token for id={row_id_str_from_q} in slot={prefill_slot_from_q}. Tracking.")
          else: # Terminate: EOS as first token or other condition met
            empty_slots.append(prefill_slot_from_q)
            # --- TIMING engine.free_resource ---
            # start_free_time = time.monotonic()
            self.engine.free_resource(prefill_slot_from_q)
            # jax.block_until_ready() # If free_resource has significant JAX ops to sync
            # end_free_time = time.monotonic()
            # free_duration_ms = (end_free_time - start_free_time) * 1000
            counter.free_resource +=1
            counter.detokenize += 1 # Count as a detokenized (completed) request

            # log_entry = {
            #     "timestamp": time.time(), "event_type": "engine_free_resource",
            #     "duration_ms": free_duration_ms, "slot": prefill_slot_from_q,
            #     "freed_after": "prefill_first_token_eos", "request_id": row_id_str_from_q
            # }
            # _add_pagemanager_stats_to_log(log_entry, self.engine) # Add PageManager stats
            # performance_log.append(log_entry)
            # log.debug(f"Detokenize & Freed: Slot {prefill_slot_from_q} for {row_id_str_from_q} (EOS on first token). Total detokenized req: {counter.detokenize}")
          self.detokenize_backlog.task_done()
          continue

        # This path is for results from decode (autoregressive steps)
        newly_empty_slots_this_step = []
        active_slots_before_decode_processing = list(slot_to_id.keys()) # Iterate over a copy

        for current_slot_idx in active_slots_before_decode_processing:
          if current_slot_idx not in slot_to_id: # Slot might have been freed by another thread/path (less likely here)
            continue

          current_input_id = slot_to_id[current_slot_idx]
          if current_slot_idx >= actual_result_tokens.data.shape[0]:
            log.warning(
                f"Detokenize: slot index {current_slot_idx} out of bounds for decode result data shape {actual_result_tokens.data.shape}. Skipping."
            )
            continue

          token_val = actual_result_tokens.data[current_slot_idx, actual_result_tokens.tokens_idx[0]].item()
          is_valid_val = actual_result_tokens.data[current_slot_idx, actual_result_tokens.valid_idx[0]].item()
          length_val = actual_result_tokens.data[current_slot_idx, actual_result_tokens.length_idx[0]].item()

          log.debug(
              f"Detokenize (decode step): slot={current_slot_idx}, id={current_input_id}, token={token_val}, valid={is_valid_val}, length={length_val}"
          )
          should_finish_request_this_step = False
          if is_valid_val: # Process token if valid
            should_finish_request_this_step = emit_token(current_input_id, token_val)

          # Check for termination conditions (EOS token handled by emit_token OR max decode length reached)
          if should_finish_request_this_step or length_val >= self.max_decode_length:
            if current_slot_idx not in newly_empty_slots_this_step:
              newly_empty_slots_this_step.append(current_slot_idx)
            log.debug(
                f"Detokenize: Request for id={current_input_id} (slot {current_slot_idx}) finished. Length={length_val}."
            )

        # Free resources for slots that finished in this decode step
        for slot_to_free in newly_empty_slots_this_step:
          if slot_to_free in slot_to_id: # Ensure it's still tracked
            request_id_being_freed = slot_to_id.pop(slot_to_free) # Remove and get ID
            empty_slots.append(slot_to_free)

            # --- TIMING engine.free_resource ---
            # start_free_time = time.monotonic()
            self.engine.free_resource(slot_to_free)
            # jax.block_until_ready() # If needed
            # end_free_time = time.monotonic()
            # free_duration_ms = (end_free_time - start_free_time) * 1000
            counter.free_resource +=1
            counter.detokenize += 1 # Increment for completed request

            # log_entry = {
            #     "timestamp": time.time(), "event_type": "engine_free_resource",
            #     "duration_ms": free_duration_ms, "slot": slot_to_free,
            #     "freed_after": "decode_eos_or_max_len", "request_id": request_id_being_freed
            # }
            # _add_pagemanager_stats_to_log(log_entry, self.engine) # Add PageManager stats
            # performance_log.append(log_entry)
            log.debug(f"Detokenize & Freed: Slot {slot_to_free} for {request_id_being_freed}. Total detokenized req: {counter.detokenize}")

        self.detokenize_backlog.task_done()

    # --- Start Detokenize Thread ---
    detokenize_thread = JetThread(target=detokenize, name="detokenize_thread")
    counter.input = len(data) # Initialize input counter
    self.live = True
    detokenize_thread.start()

    # --- Main Request Processing Loop ---
    for row_idx, row_data in enumerate(data):
      while not empty_slots: # Wait for an empty slot
        if not slot_to_id and self.detokenize_backlog.empty() and not self.live:
          # No active slots, backlog empty, and processing is winding down
          log.info("MainLoop: No active slots, empty backlog, and not live. Likely finishing.")
          break
        log.debug(f"MainLoop: No empty slots. Active slots: {len(slot_to_id)}. Detokenize backlog: {self.detokenize_backlog.qsize()}. Waiting...")
        if slot_to_id: # If there are active slots, trigger a decode to free them up
          decode()
        else: # No active slots, but backlog might still be processing or waiting for new inputs
          threading.Event().wait(0.01) # Short wait to avoid busy loop

      if not self.live and not empty_slots and not slot_to_id : # More robust exit condition
          log.info("MainLoop: Exiting input processing loop as not live and no slots available or active.")
          break

      if not empty_slots: # If still no slots after trying to decode
          log.warning(f"MainLoop: Still no empty slots after checks for request id={row_data.id}. This may indicate a bottleneck or issue.")
          # Depending on desired behavior, could skip, error, or wait longer.
          # For now, let's try one more decode if slots are active, then wait briefly if it's truly stuck.
          if slot_to_id:
            decode()
            threading.Event().wait(0.05) # Wait a bit longer after forced decode
            if not empty_slots:
                log.error(f"MainLoop: Critical - No empty slot for id={row_data.id} after extended wait. Skipping request.")
                counter.input -=1 # Adjust count as this one is skipped.
                continue # Skip this input
          else: # No active slots, implies system might be stuck if backlog isn't clearing.
            log.error(f"MainLoop: Critical - No empty slots and no active requests for id={row_data.id}. Skipping request.")
            counter.input -=1
            continue


      current_empty_slot = empty_slots.pop(0)
      log.debug(f"MainLoop: Processing input id={row_data.id} in slot={current_empty_slot} (Request {row_idx + 1}/{len(data)})")
      self.prefill.process(
          model_params=self.params,
          decode_state=self.decode_state,
          decode_slot=current_empty_slot,
          input_id=row_data.id,
          input_tokens_padded=row_data.tokens,
          input_true_length=row_data.true_length,
          max_seq_len_for_op=row_data.tokens.shape[0], # Max sequence length for this specific prefill operation
          prefill_done=prefill_done_callback,
      )

    log.info("MainLoop: All inputs submitted to PrefillHelper. Finalizing any batched prefills...")
    self.prefill.finalize(self.params, self.decode_state, prefill_done_callback)

    log.info("MainLoop: Waiting for all active requests to complete decoding...")
    while slot_to_id: # While there are still requests being processed
      if self.detokenize_backlog.empty() and not self.live:
        log.warning("MainLoop: Exiting decode wait loop - not live and empty backlog, but slots still active. This might be an issue.")
        break
      log.debug(f"MainLoop: Waiting for {len(slot_to_id)} active slots to complete. Triggering decode. Backlog: {self.detokenize_backlog.qsize()}")
      decode()
      if slot_to_id and self.detokenize_backlog.empty(): # Prevent busy loop if decode doesn't immediately clear slot but backlog is empty
          threading.Event().wait(0.005)


    log.info("MainLoop: All active slots processed or finalized.")
    self.live = False # Signal detokenize thread to finish up
    log.info("MainLoop: Waiting for detokenize thread to finish processing any remaining backlog...")
    detokenize_thread.join(timeout=10.0) # Add a timeout to join
    if detokenize_thread.is_alive():
        log.warning("MainLoop: Detokenize thread did not finish within timeout.")

    log.info(
        "Summary for '%s': Prefills=%d, Decodes=%d, Detokenized_Requests=%d, Freed_Resources=%d (Total Inputs Submitted: %d).",
        desc,
        counter.prefill,
        counter.decode,
        counter.detokenize,
        counter.free_resource,
        counter.input, # This reflects inputs processed or attempted
    )

  def batch_inference(self, data: List[InputData], desc="") -> dict[str, List[int]]:
    # Original cProfile setup commented out
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    random.seed(99) 

    # Create a mutable copy of the data to shuffle
    shuffled_data = list(data)
    random.shuffle(shuffled_data)
    log.info(f"Input data shuffled. Processing {len(shuffled_data)} items randomly.")

    res: dict[str, List[int]] = defaultdict(list)

    # Define callbacks for token emission
    def emit_first_token_callback(id_str: str, token: int) -> bool:
      nonlocal res
      log.debug(f"Output: ID '{id_str}', First Token: {token}")
      if token == self.tokenizer.eos_id: # Assuming self.tokenizer is set
        log.debug(f"Output: ID '{id_str}' received EOS as first token. Terminating.")
        return True # Terminate if EOS is the first token
      res[id_str].append(token)
      return False # Continue generation

    def emit_token_callback(id_str: str, token: int) -> bool:
      nonlocal res
      # Ensure the list for id_str exists
      if id_str not in res and token != self.tokenizer.eos_id:
        # This case should ideally not happen if emit_first_token_callback populated it,
        # unless first token was EOS. Log a warning if a non-EOS token arrives for an untracked ID.
        log.warning(
            f"Output: ID '{id_str}' missing from results dict but received non-EOS token {token}. Appending."
        )
        res[id_str].append(token) # Start new list
      elif id_str in res:
          # Append token if it's not EOS or if it is EOS and the last token wasn't also EOS (avoid multiple EOS)
        if not res[id_str] or res[id_str][-1] != self.tokenizer.eos_id:
            res[id_str].append(token)
      # If id_str not in res and token is EOS, do nothing (already handled or was EOS on first token)

      if token == self.tokenizer.eos_id:
        log.debug(f"Output: ID '{id_str}' received EOS token. Terminating.")
        return True # Terminate on EOS
      return False # Continue generation

    self.batch_inference_with_callback(
        shuffled_data, # Use the shuffled data
        emit_first_token=emit_first_token_callback,
        emit_token=emit_token_callback,
        desc=desc
    )

    # Original cProfile and stats printing commented out
    # profiler.disable()
    # stats_obj = Stats(profiler) # Renamed from 'stats' to avoid conflict if any
    # print("\n---- Stats sorted by total time (tottime) ----")
    # stats_obj.sort_stats('tottime').print_stats(50)
    # print("\n---- Stats sorted by number of calls (ncalls) ----")
    # stats_obj.sort_stats('ncalls').print_stats(50)
    # print("\n---- Stats sorted by cumulative time (cumtime) ----")
    # stats_obj.sort_stats('cumtime').print_stats(50)


    # Save performance log
    if desc != "warmup": # Avoid saving log for warmup runs unless specifically desired
        try:
            if performance_log:
                df_perf = pd.DataFrame(performance_log) # Create DataFrame from global list
                # Sanitize desc for filename
                safe_desc = "".join(c if c.isalnum() else "_" for c in desc) if desc else "run"
                log_filename_timestamp = time.strftime('%Y%m%d_%H%M%S')
                log_filename = f"performance_log_{safe_desc}_{log_filename_timestamp}.csv"
                df_perf.to_csv(log_filename, index=False)
                log.info(f"Performance log saved to {log_filename}")
                # Optionally clear the log if batch_inference might be called multiple times
                # and you want separate logs per call rather than one cumulative log.
                # performance_log.clear()
            else:
                log.info(f"No performance data logged for run: {desc}")
        except Exception as e:
            log.error(f"Failed to save performance log for run '{desc}': {e}", exc_info=True)

    return dict(res)