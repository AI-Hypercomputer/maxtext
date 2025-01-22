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

from jetstream.engine import engine_api

import logging
# pylint: disable=no-name-in-module
from maxengine import set_engine_vars_from_base_engine

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array
  true_length: int


class JetThread(threading.Thread):

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


class OfflineInference:

  def __init__(self, engine: engine_api.Engine, params, base_engine: engine_api.Engine, enable_batch_prefill: bool):
    self.live = False
    self.engine = engine
    self.decode_state = None
    if params is None:
      params = engine.load_params()
    else:
      rng = jax.random.PRNGKey(0)
      set_engine_vars_from_base_engine(engine, base_engine, rng)
    self.params = params

    self.enable_batch_prefill = enable_batch_prefill
    self.batch_size = engine.max_concurrent_decodes
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    metadata = engine.get_tokenizer()
    self.tokenizer = engine.build_tokenizer(metadata)
    self.dummy = False

    self._cached_pref = {}
    self._cached_pref_batch = {}
    self._cached_generate = None
    self.detokenize_backlog = queue.Queue(10)
    self.prefill_buckets = defaultdict(list)

  def init_decode_state(self):
    if self.decode_state is None:
      self.decode_state = self.engine.init_decode_state()

  def warmup(self, max_length, warmup_samples):
    self.init_decode_state()
    interesting_buckets = [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]
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
      if length == 64 or length == 1024:
        continue
      log.info(f"Compiling batched prefill: {length}")
      input_data_batch = jax.ShapeDtypeStruct((max_length,), jnp.dtype("int32"))
      num_prompts = max_length // length
      self._cached_pref_batch[length] = (
          jax.jit(
              self._prefill_insert_batch,
              static_argnames=(
                  "num_prompts",
                  "padded_length",
              ),
              donate_argnames=("decode_state",),
          )
          .lower(
              self.params,
              tokens=input_data_batch,
              slots=jnp.arange(0, 8, dtype=int),
              num_prompts=num_prompts,
              decoder_positions=jnp.arange(0, max_length, dtype=int),
              decoder_segment_ids=jnp.ones(max_length, dtype=int),
              start_pos=jnp.arange(0, max_length, 128, dtype=int),
              padded_length=length,
              true_lengths=jnp.full(8, length, dtype=int),
              decode_state=self.decode_state,
          )
          .compile()
      )
    self._cached_generate = (
        jax.jit(self.engine.generate, donate_argnums=(1,)).lower(self.params, self.decode_state).compile()
    )
    self.batch_inference(warmup_samples, desc="warmup")

  def _prefill_insert(self, params, tokens, slot, true_length, decode_state):
    """return decodestate."""
    padded_len = tokens.shape[0]
    prefill_result, first_token = self.engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = self.engine.insert(prefill_result, decode_state, slot)
    return first_token, decode_state

  def _prefill_insert_batch(
      self,
      params,
      tokens,
      slots,
      num_prompts,
      decoder_positions,
      decoder_segment_ids,
      start_pos,
      padded_length,
      true_lengths,
      decode_state,
  ):
    """return decodestate."""
    cache, prefill_results, first_tokens = self.engine.prefill_concat(
        params=params,
        padded_tokens=tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        start_pos=start_pos,
        true_lengths=true_lengths,
        num_prompts=num_prompts,
    )
    decode_state = self.engine.insert_partial(
        prefill_results,
        decode_state,
        cache,
        slots,
        num_prompts=num_prompts,
        start_indices=start_pos,
        seq_len=padded_length,
    )
    return first_tokens, decode_state

  def batch_inference_with_callback(
      self,
      data: List[InputData],
      emit_first_token: Callable[[str, int], bool],
      emit_token: Callable[[str, int], bool],
      desc: str,
  ):
    """callback is a function that takes id and token. It will be called once per output

    token.
    """

    def prefill(prefill_bucket, prefill_len):
      nonlocal self
      if self.dummy:
        log.info("dummy prefill")
        return 123
      if not self.enable_batch_prefill or prefill_len in (64, 1024) or prefill_len * len(prefill_bucket) != 1024:
        prefill_result = []
        prefill_fn = self._prefill_insert
        if (cached := self._cached_pref.get(prefill_len)) is not None:
          prefill_fn = cached
        for slot, row in prefill_bucket:
          first_token, self.decode_state = prefill_fn(
              self.params, tokens=row.tokens, slot=slot, true_length=row.true_length, decode_state=self.decode_state
          )
          prefill_result.append((first_token, slot, row))
        return prefill_result
      else:
        prefill_fn = self._prefill_insert_batch
        if (cached := self._cached_pref_batch.get(prefill_len)) is not None:
          prefill_fn = cached
        positions = np.concatenate([np.arange(0, row.tokens.shape[0]) for (slot, row) in prefill_bucket])
        positions = jnp.array(positions)

        sequence_indicators = []
        for idx, (slot, row) in enumerate(prefill_bucket):
          zero_to_n = np.arange(0, row.tokens.shape[0])
          ones_to_keep = zero_to_n < row.true_length
          one_d_output = (zero_to_n < row.true_length).astype(int) * (idx * 2 + 1) + (zero_to_n >= row.true_length).astype(
              int
          ) * (idx + 1) * 2
          sequence_indicators.append(one_d_output)
        sequence_indicator = jnp.array(np.concatenate(sequence_indicators))

        tokens = jnp.concat([row.tokens for (slot, row) in prefill_bucket])

        slots = [slot for (slot, row) in prefill_bucket]
        true_lengths = [row.true_length for (slot, row) in prefill_bucket]
        start_pos = np.cumsum([0] + [row.tokens.shape[0] for (slot, row) in prefill_bucket])[:-1]
        start_pos = start_pos.tolist()

        # pad slots to keep static shape of jitted function input
        def pad_num_prompts_len_array(array_to_pad, pad_len):
          if len(array_to_pad) < pad_len:
            array_to_pad.extend([0] * (pad_len - len(array_to_pad)))
          return jnp.array(array_to_pad)

        slots = pad_num_prompts_len_array(slots, 8)
        true_lengths = pad_num_prompts_len_array(true_lengths, 8)
        start_pos = pad_num_prompts_len_array(start_pos, 8)
        # this lowered function has static input for num_prompts and padded_length
        first_tokens, self.decode_state = prefill_fn(
            self.params,
            tokens=tokens,
            slots=slots,
            decoder_positions=positions,
            decoder_segment_ids=sequence_indicator,
            start_pos=start_pos,
            true_lengths=true_lengths,
            decode_state=self.decode_state,
        )  # pytype: disable=missing-parameter
        prefill_result = [(first_tokens[idx], slot, row) for (idx, (slot, row)) in enumerate(prefill_bucket)]

        return prefill_result

    empty_slots = list(range(self.batch_size))
    slot_to_id = {}
    num_prefills = {}
    num_decodes = 0

    dummy_length = 1

    def decode():
      nonlocal self
      nonlocal dummy_length
      if self.dummy:
        log.info("Dummy generate")
        res = engine_api.ResultTokens(
            data=np.array([[123, 1, dummy_length]] * self.batch_size),
            tokens_idx=(0, 0),
            valid_idx=(0, 0),
            length_idx=(0, 0),
            samples_per_slot=(0, 0),
        )
        dummy_length += 1
        self.decode_state, result_tokens = self.decode_state, res
      else:
        gen_fn = self.engine.generate
        if self._cached_generate is not None:
          gen_fn = self._cached_generate
        result_tokens_l = []
        for i in range(5):
          self.decode_state, result_tokens = gen_fn(self.params, self.decode_state)
          result_tokens_l.append(result_tokens)
      for i in range(5):
        # result_tokens.copy_to_host_async()
        result_tokens = result_tokens_l[i].convert_to_numpy()
        self.detokenize_backlog.put((result_tokens, False, 0, 0), block=True)
        # log.info(f"Decode put result {i} to queue")

    def detokenize():
      nonlocal self
      nonlocal slot_to_id
      nonlocal empty_slots
      while self.live:
        # log.info("Detokenize start")
        newly_empty = []
        result_tokens, is_first_token, row_id, _slot = self.detokenize_backlog.get(block=True)
        # result_tokens = result_tokens.convert_to_numpy()
        # log.info("Detokenize get from queue")
        if is_first_token:
          first_token = result_tokens.data[0][0].item()
          should_terminate = emit_first_token(row_id, first_token)
          if not should_terminate:
            slot_to_id[_slot] = row_id
          else:
            empty_slots.append(_slot)
          continue
        for slot, id_ in slot_to_id.items():
          token, is_valid, length = result_tokens.data[slot]
          log.debug(f"slot is {slot}, length is {length}")
          should_finish = False
          if is_valid:
            should_finish = emit_token(id_, token.item())
          if should_finish or length >= self.max_decode_length:
            newly_empty.append(slot)
            log.debug(f"Detokenize free up {slot}, length {length}")
        # Add slots of those that are empty to empty
        for slot in newly_empty:
          del slot_to_id[slot]
          empty_slots.append(slot)
        if newly_empty and self.detokenize_backlog.qsize() == 0 and len(slot_to_id.items()) == 0:
          break

    detokenize_thread = JetThread(
        target=functools.partial(
            detokenize,
        ),
        name="detokenize",
    )
    self.live = True
    detokenize_thread.start()
    total_num_prefills = 0
    for row in data:
      while not empty_slots:
        # If slots are all full, decode until there are free slots
        # to insert
        num_decodes += 1
        log.info(f"decode-{desc}-{num_decodes}")
        decode()
      # do one insert
      num_tokens = len(row.tokens)
      num_prefills[num_tokens] = 1 if num_tokens not in num_prefills else num_prefills[num_tokens] + 1
      log.info(
          f"prefill-{desc}-{num_prefills} num_prefills {sum(num_prefills.values())} num_tokens {num_tokens} true_length {row.true_length} num_empty_slots {len(empty_slots)} num_decodes {num_decodes}"
      )
      total_num_prefills += 1
      log.info(f"Total num prefill: {total_num_prefills}")
      slot = empty_slots.pop()
      # directly prefill prompts with 64 or less tokens, and with 1024 tokens
      if num_tokens in (64, 1024) or not self.enable_batch_prefill:
        first_token, slot, row = prefill([(slot, row)], num_tokens)[0]
        self.detokenize_backlog.put((first_token, True, row.id, slot), block=True)
        continue
      self.prefill_buckets[num_tokens].append((slot, row))
      prefill_buckets_len = {k: len(self.prefill_buckets[k]) for k in self.prefill_buckets}
      log.debug(f"prefill buckets {prefill_buckets_len}")
      if len(self.prefill_buckets[num_tokens]) * num_tokens == 1024:
        prefill_results = prefill(self.prefill_buckets[num_tokens], num_tokens)
        for first_token, slot, row in prefill_results:
          log.debug(f"Put row of len {row.tokens.shape[0]} true length {row.true_length} slot {slot} to detokenize backlog")
          self.detokenize_backlog.put((first_token, True, row.id, slot), block=True)
        self.prefill_buckets[num_tokens] = []
    # For leftover requests in buckets at the end of computation, do prefill individually.
    for num_tokens in self.prefill_buckets.keys():
      prefill_results = prefill(self.prefill_buckets[num_tokens], num_tokens)
      for first_token, slot, row in prefill_results:
        log.debug(f"Put row of len {row.tokens.shape[0]} true length {row.true_length} slot {slot} to detokenize backlog")
        self.detokenize_backlog.put((first_token, True, row.id, slot), block=True)
    self.prefill_buckets = defaultdict(list)
    while slot_to_id:
      log.debug(f"decode-{desc}-{num_decodes} num_filled_slots {len(slot_to_id)}")
      num_decodes += 1
      decode()

    self.live = False
    detokenize_thread.join()
    log.info(f"summary-{desc}-prefills-{num_prefills}-decodes-{num_decodes} completed.")

  def batch_inference(self, data: List[InputData], desc=""):
    """data is list of obj with id, tokens, and true length"""
    res = defaultdict(list)

    def callback(id_, token):
      nonlocal res
      if token == self.tokenizer.eos_id:
        log.debug(f"res[{id_}] eos")
      if not res[id_] or res[id_][-1] != self.tokenizer.eos_id:
        res[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback, desc=desc)
    return res
