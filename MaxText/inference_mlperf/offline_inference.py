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
import random

from jetstream.engine import engine_api

import logging
# pylint: disable=no-name-in-module
from maxengine import set_engine_vars_from_base_engine
from maxengine import PrefillPackingFeature
from maxengine import BatchInput

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
    self.decode_state_executable = None
    if params is None:
      self.relayout_params = True
      params = engine.load_params()
    else:
      self.relayout_params = False
      rng = jax.random.PRNGKey(0)
      set_engine_vars_from_base_engine(engine, base_engine, rng)
    self.params = params

    self.enable_batch_prefill = enable_batch_prefill
    self.batch_size = engine.max_concurrent_decodes
    self.max_prefill_length = engine.config.max_prefill_predict_length
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    metadata = engine.get_tokenizer()
    self.tokenizer = engine.build_tokenizer(metadata)
    self.dummy = False

    self._cached_pref = {}
    self._cached_pref_batch = {}
    self._cached_generate = None
    self.detokenize_backlog = queue.Queue(10)
    self.prefill_buckets = defaultdict(list)

    self._decode_state_executable = None

    self.feature = PrefillPackingFeature(engine)

  def init_decode_state(self):
    if self.decode_state is None:
      assert self._decode_state_executable != None, "Decode state executable is none"
      self.decode_state = self._decode_state_executable(None)

  def warmup(self, max_length, warmup_samples):

    self._cached_generate, self.params, self._decode_state_executable = self.engine.aot_compile(
        self.params, pass_rng_shape=False
    )

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
    log.info("prefill packing: begin aot compile")
    self.feature.aot_compile(self.params, max_length, interesting_buckets)
    log.info("prefill packing: end aot compile")

    self.batch_inference(warmup_samples, desc="warmup")

  def _prefill_insert(self, params, tokens, slot, true_length, decode_state):
    """return decodestate."""
    padded_len = tokens.shape[0]
    prefill_result, first_token = self.engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = self.engine.insert(prefill_result, decode_state, slot)
    return first_token, decode_state

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

    def prefill(prefill_bucket, prefill_len) -> None:
      nonlocal self
      if self.dummy:
        log.info("dummy prefill")
        return

      input_ids = []
      input_prompts = []
      decode_slots = []
      for slot, row in prefill_bucket:
        decode_slots.append(slot)
        input_ids.append(row.id)
        input_prompts.append(row.tokens[:row.true_length])
        if row.true_length > prefill_len:
          log.info(f"length: {row.true_length} padding: {len(row.tokens)} prefill_len: {prefill_len}")
      batch_input = BatchInput(input_ids, input_prompts, prefill_len)

      prefill_result, self.decode_state = self.feature.prefill_and_insert_batch(
        self.params, batch_input, decode_slots, self.decode_state)

      for i in range(batch_input.count):
        # TODO: does slot in prefill_bucket and prefill_result match here ?
        _, row = prefill_bucket[i]
        first_token, slot = prefill_result[i]
        log.debug(f"Put row of len {row.tokens.shape[0]} true length {row.true_length} slot {slot} to detokenize backlog")
        self.detokenize_backlog.put((first_token, True, row.id, slot), block=True)

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
        else:
          assert False, "no generate fn"
        result_tokens_l = []
        for i in range(10):
          self.decode_state, result_tokens = gen_fn(self.params, self.decode_state, None)
          result_tokens_l.append(result_tokens)
      for i in range(10):
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
        log.debug(f"decode-{desc}-{num_decodes}")
        decode()
      # do one insert
      padded_len = len(row.tokens)
      num_prefills[padded_len] = 1 if padded_len not in num_prefills else num_prefills[padded_len] + 1
      log.debug(
          f"prefill-{desc}-{num_prefills} num_prefills {sum(num_prefills.values())} padded_len {padded_len} true_length {row.true_length} num_empty_slots {len(empty_slots)} num_decodes {num_decodes}"
      )
      total_num_prefills += 1
      log.debug(f"Total num prefill: {total_num_prefills}")
      slot = empty_slots.pop()
      # directly prefill prompts
      if not self.enable_batch_prefill:
        prefill([(slot, row)], padded_len)
        continue

      if len(self.prefill_buckets[padded_len // 2]) != 0:
        prefill(self.prefill_buckets[padded_len // 2], padded_len // 2)
        self.prefill_buckets[padded_len // 2] = []
      if padded_len == self.max_prefill_length:
        prefill([(slot, row)], padded_len)
        continue
      if padded_len == 64:
        row.tokens = jnp.concat([row.tokens, jnp.zeros(64, dtype=int)])
        padded_len = 128

      self.prefill_buckets[padded_len].append((slot, row))
      prefill_buckets_len = {k: len(self.prefill_buckets[k]) for k in self.prefill_buckets}
      log.debug(f"prefill buckets {prefill_buckets_len}")
      if len(self.prefill_buckets[padded_len]) * padded_len >= self.max_prefill_length:
        total_true_len = sum([row.true_length for (slot, row) in self.prefill_buckets[padded_len]])
        # Can't hold another buffer, prefill right away
        if total_true_len > self.max_prefill_length - padded_len // 2 and total_true_len <= self.max_prefill_length:
          log.debug(
              f"Normal batch {padded_len} total padded len {len(self.prefill_buckets[padded_len]) * padded_len} total true len {total_true_len}"
          )
          prefill(self.prefill_buckets[padded_len], padded_len)
          self.prefill_buckets[padded_len] = []
        # Already overloading, left over the last and do prefill
        elif total_true_len > self.max_prefill_length:
          log.debug(
              f"Overloading {padded_len} total padded len {len(self.prefill_buckets[padded_len]) * padded_len} total true len {total_true_len}"
          )
          current = self.prefill_buckets[padded_len][-1]
          prefill(self.prefill_buckets[padded_len][:-1], padded_len)
          self.prefill_buckets[padded_len] = [current]
    # For leftover requests in buckets at the end of computation, do prefill individually.
    for padded_len in self.prefill_buckets.keys():
      prefill(self.prefill_buckets[padded_len], padded_len)
    self.prefill_buckets = defaultdict(list)
    while slot_to_id:
      log.debug(f"decode-{desc}-{num_decodes} num_filled_slots {len(slot_to_id)}")
      num_decodes += 1
      decode()

    self.live = False
    detokenize_thread.join()
    log.info(f"summary-{desc}-prefills-{num_prefills}-decodes-{num_decodes} completed.")

  def batch_inference(self, data: List[InputData], desc="") -> dict[str, List[int]]:
    """data is list of obj with id, tokens, and true length"""
    data_dict = defaultdict(list)
    log.info("sorting data")
    for row in data:
      # log.info(f"row shape: {row.tokens.shape}")
      data_dict[row.tokens.shape[0]].append(row)
    data_dict[128] += data_dict[64]
    data_dict[64] = []
    data = []
    for padded_len in [128, 256, 512, 1024]:
      log.info(f"padded len: {padded_len}, num: {len(data_dict[padded_len])}")
      random.shuffle(data_dict[padded_len])
      data += data_dict[padded_len]
    log.info("finished sorting data")

    res = defaultdict(list)
    def callback(id_, token):
      nonlocal res
      if token == self.tokenizer.eos_id:
        log.debug(f"res[{id_}] eos")
        log.info(self.tokenizer.decode(res[id_]))
      if not res[id_] or res[id_][-1] != self.tokenizer.eos_id:
        res[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback, desc=desc)
    return res
