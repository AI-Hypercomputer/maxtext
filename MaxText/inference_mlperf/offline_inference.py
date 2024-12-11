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

from typing import Callable, List
import dataclasses
from collections import defaultdict
import jax
from jax import numpy as jnp
import numpy as np
import queue
import os
import functools
import threading
import traceback
import signal
import time

from jetstream.engine import engine_api

import logging
# pylint: disable=no-name-in-module
from maxengine import set_engine_vars_from_base_engine

log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOGLEVEL", "INFO"))


@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array
  true_length: int


class ProcessingError(Exception):
  """Custom exception for processing errors."""
  pass


class JetThread(threading.Thread):
  def run(self):
    try:
      super().run()
    except Exception as e:
      log.error(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      raise


class OfflineInference:
  def __init__(self, engine: engine_api.Engine, params, base_engine: engine_api.Engine):
    self.engine = engine
    self.decode_state = None
    if params is None:
      params = engine.load_params()
    else:
      rng = jax.random.PRNGKey(0)
      set_engine_vars_from_base_engine(engine, base_engine, rng)
    self.params = params

    self.batch_size = engine.max_concurrent_decodes
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    metadata = engine.get_tokenizer()
    self.tokenizer = engine.build_tokenizer(metadata)
    self.dummy = False

    self._cached_pref = {}
    self._cached_generate = None
    self.detokenize_backlog = queue.Queue(10)
    self._shutdown = threading.Event()
    self._error = None
    
    # Configurable timeouts
    self.activity_timeout = float(os.getenv("ACTIVITY_TIMEOUT", "1800"))  # 30 minutes default
    self.queue_timeout = float(os.getenv("QUEUE_TIMEOUT", "300"))  # 5 minutes default
    
    # Monitoring counters
    self._total_tokens_processed = 0
    self._last_activity_time = time.time()
    self._processing_stats = {
      'prefills': 0,
      'decodes': 0,
      'tokens_processed': 0,
    }

  def _update_activity(self):
    """Update last activity timestamp and log if significant time has passed."""
    current_time = time.time()
    elapsed = current_time - self._last_activity_time
    if elapsed > 60:  # Log if more than a minute has passed
      log.info(f"Activity resumed after {elapsed:.1f} seconds")
    self._last_activity_time = current_time

  def init_decode_state(self):
    if self.decode_state is None:
      self.decode_state = self.engine.init_decode_state()

  def warmup(self, max_length, warmup_samples):
    log.info("Starting warmup...")
    self.init_decode_state()
    interesting_buckets = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    for length in interesting_buckets:
      if length > max_length:
        break
      log.info(f"Compiling prefill: {length}")
      input_data = jax.ShapeDtypeStruct((length,), jnp.dtype("int32"))
      self._cached_pref[length] = (
        jax.jit(self._prefill_insert, donate_argnums=(4,))
        .lower(self.params, tokens=input_data, slot=0, true_length=length - 1, decode_state=self.decode_state)
        .compile()
      )
    self.batch_inference(warmup_samples, desc="warmup")
    self._cached_generate = (
      jax.jit(self.engine.generate, donate_argnums=(1,))
      .lower(self.params, self.decode_state)
      .compile()
    )
    log.info("Warmup completed successfully")

  def _prefill_insert(self, params, tokens, slot, true_length, decode_state):
    """Return decode state."""
    try:
      self._update_activity()
      prefill_result, first_token = self.engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
      )
      decode_state = self.engine.insert(prefill_result, decode_state, slot=slot)
      self._processing_stats['prefills'] += 1
      return first_token, decode_state
    except Exception as e:
      log.error(f"Error in prefill_insert: {e}")
      raise ProcessingError(f"Prefill failed: {e}") from e

  def decode_batch(self):
    """Execute generation step with batched JAX operations."""
    if self.dummy:
      log.debug("Running dummy decode")
      return

    try:
      self._update_activity()
      gen_fn = self.engine.generate if self._cached_generate is None else self._cached_generate
      result_tokens_l = []
      
      # Do 5 JAX operations in a batch
      for i in range(5):
        self.decode_state, result_tokens = gen_fn(self.params, self.decode_state)
        result_tokens_l.append(result_tokens)
        self._update_activity()  # Update after each operation in case they're slow
      
      # Convert and queue results
      results = []
      for result_tokens in result_tokens_l:
        results.append(result_tokens.convert_to_numpy())
      
      # Update monitoring
      self._processing_stats['decodes'] += 5
      
      return results
      
    except Exception as e:
      log.error(f"Error in decode: {e}")
      self._shutdown.set()
      self._error = e
      raise

  def process_decode_results(self, results):
    """Process decode results with timeouts."""
    try:
      for result_tokens in results:
        self._update_activity()
        self.detokenize_backlog.put(
          (result_tokens, False, 0, 0),
          block=True,
          timeout=self.queue_timeout
        )
        log.debug(f"Decode: Successfully queued result. Queue size: {self.detokenize_backlog.qsize()}")
    except queue.Full:
      log.error("Queue full in decode - possible deadlock")
      self._shutdown.set()
      raise ProcessingError("Detokenization queue full")

  def detokenize(self, slot_to_id, empty_slots, emit_first_token, emit_token):
    """Process tokens with enhanced monitoring and error handling."""
    last_log_time = time.time()
    tokens_processed_since_last_log = 0

    while not self._shutdown.is_set():
      try:
        # Periodic status logging
        current_time = time.time()
        if current_time - last_log_time > 10:  # Log every 10 seconds
          log.info(f"Detokenize status - Queue size: {self.detokenize_backlog.qsize()}, "
                  f"Active slots: {len(slot_to_id)}, "
                  f"Tokens processed in last 10s: {tokens_processed_since_last_log}")
          last_log_time = current_time
          tokens_processed_since_last_log = 0

        # Get next item with timeout
        try:
          result_tokens, is_first_token, row_id, _slot = self.detokenize_backlog.get(
            block=True,
            timeout=self.queue_timeout
          )
          self._update_activity()
        except queue.Empty:
          if len(slot_to_id) == 0:
            log.info("No more active slots and queue empty, finishing detokenize")
            break
          continue

        newly_empty = []
        
        if is_first_token:
          first_token = result_tokens.data[0][0].item()
          should_terminate = emit_first_token(row_id, first_token)
          if not should_terminate:
            slot_to_id[_slot] = row_id
          else:
            empty_slots.append(_slot)
          continue

        # Process each slot
        for slot, id_ in list(slot_to_id.items()):
          token, is_valid, length = result_tokens.data[slot]
          should_finish = False
          if is_valid:
            should_finish = emit_token(id_, token.item())
            tokens_processed_since_last_log += 1
            self._total_tokens_processed += 1
          
          if should_finish or length >= self.max_decode_length:
            newly_empty.append(slot)
            log.info(f"Slot {slot} finished (length: {length})")

        # Update slots
        for slot in newly_empty:
          del slot_to_id[slot]
          empty_slots.append(slot)
          log.debug(f"Freed slot {slot}, {len(empty_slots)} now available")

      except Exception as e:
        log.error(f"Error in detokenize: {e}")
        self._shutdown.set()
        self._error = e
        raise

  def batch_inference_with_callback(
    self,
    data: List[InputData],
    emit_first_token: Callable[[str, int], bool],
    emit_token: Callable[[str, int], bool],
    desc: str,
  ):
    """Process batch with improved error handling."""
    empty_slots = list(range(self.batch_size))
    slot_to_id = {}
    num_prefills = {}
    num_decodes = 0
    start_time = time.time()

    def prefill(slot, tokens, true_length):
      if self.dummy:
        return 123

      prefill_fn = self._prefill_insert
      if (cached := self._cached_pref.get(len(tokens))) is not None:
        prefill_fn = cached

      first_token, self.decode_state = prefill_fn(
        self.params, tokens=tokens, slot=slot, true_length=true_length, decode_state=self.decode_state
      )
      return first_token

    self._shutdown.clear()
    self._error = None
    
    detokenize_thread = JetThread(
      target=functools.partial(
        self.detokenize,
        slot_to_id,
        empty_slots,
        emit_first_token,
        emit_token
      ),
      name="detokenize"
    )
    detokenize_thread.start()

    try:
      for row in data:
        if time.time() - self._last_activity_time > self.activity_timeout:
          log.error(f"No activity detected for {self.activity_timeout:.1f} seconds")
          raise ProcessingError(f"No activity detected for {self.activity_timeout:.1f} seconds")

        while not empty_slots and not self._shutdown.is_set():
          num_decodes += 1
          log.info(f"decode-{desc}-{num_decodes} (queue size: {self.detokenize_backlog.qsize()})")
          
          # Get batch of 5 results and process immediately
          results = self.decode_batch()
          if results:
            self.process_decode_results(results)

        if self._shutdown.is_set():
          if self._error:
            raise self._error
          raise ProcessingError("Processing terminated due to error")

        # Prefill handling
        num_tokens = len(row.tokens)
        num_prefills[num_tokens] = num_prefills.get(num_tokens, 0) + 1
        
        log.info(
          f"prefill-{desc} stats: prefills={num_prefills} "
          f"tokens={num_tokens} length={row.true_length} "
          f"empty_slots={len(empty_slots)} decodes={num_decodes}"
        )
        
        slot = empty_slots.pop()
        first_token = prefill(slot, row.tokens, row.true_length)
        
        # Handle first token
        try:
          self.detokenize_backlog.put(
            (first_token, True, row.id, slot),
            block=True,
            timeout=self.queue_timeout
          )
        except queue.Full:
          raise ProcessingError("Queue full during prefill")

      # Process remaining slots
      while slot_to_id and not self._shutdown.is_set():
        log.info(f"Finishing remaining {len(slot_to_id)} slots")
        num_decodes += 1
        
        # Get and process batch of 5 results
        results = self.decode_batch()
        if results:
          self.process_decode_results(results)

    except Exception as e:
      log.error(f"Error during batch processing: {e}")
      self._shutdown.set()
      self._error = e
      raise
    finally:
      self._shutdown.set()
      
      detokenize_thread.join(timeout=60)
      if detokenize_thread.is_alive():
        log.error("Detokenize thread failed to terminate")
      
      elapsed_time = time.time() - start_time
      log.info(
        f"Batch complete: {desc} "
        f"prefills={num_prefills} "
        f"decodes={num_decodes} "
        f"tokens={self._total_tokens_processed} "
        f"time={elapsed_time:.2f}s "
        f"tokens/sec={self._total_tokens_processed/elapsed_time:.2f}"
      )

  def batch_inference(self, data: List[InputData], desc=""):
    """Batch inference with result collection."""
    res = defaultdict(list)

    def callback(id_, token):
      if token == self.tokenizer.eos_id:
        log.info(f"EOS token for id {id_}")
      if not res[id_] or res[id_][-1] != self.tokenizer.eos_id:
        res[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback, desc=desc)
    return res