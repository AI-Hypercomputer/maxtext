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
log.setLevel(os.getenv("LOGLEVEL", "INFO"))


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

  def __init__(self, engine: engine_api.Engine, params, base_engine: engine_api.Engine):
    self.live = False
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

  def init_decode_state(self):
    if self.decode_state is None:
      self.decode_state = self.engine.init_decode_state()

  def warmup(self, max_length, warmup_samples):
    self.init_decode_state()
    interesting_buckets = [
        32,
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
    self.batch_inference(warmup_samples, desc="warmup")
    self._cached_generate = (
        jax.jit(self.engine.generate, donate_argnums=(1,)).lower(self.params, self.decode_state).compile()
    )

  def _prefill_insert(self, params, tokens, slot, true_length, decode_state):
    """return decodestate."""
    prefill_result, first_token = self.engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = self.engine.insert(prefill_result, decode_state, slot=slot)
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

    def prefill(slot, tokens, true_length):
      nonlocal self
      if self.dummy:
        log.info("dummy prefill")
        return 123

      prefill_fn = self._prefill_insert
      if (cached := self._cached_pref.get(len(tokens))) is not None:
        prefill_fn = cached

      first_token, self.decode_state = prefill_fn(
          self.params, tokens=tokens, slot=slot, true_length=true_length, decode_state=self.decode_state
      )
      return first_token

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
            log.info(f"Detokenize free up {slot}, length {length}")
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
    for row in data:
      while not empty_slots:
        # If slots are all full, decode until there are free slots
        # to insert
        num_decodes += 1
        log.info(f"decode-{desc}-{num_decodes}")
        decode()
      # do one insert
      num_tokens = len(row.tokens)
      num_prefills[num_tokens] = 0 if num_tokens not in num_prefills else num_prefills[num_tokens] + 1
      log.info(
          f"prefill-{desc}-{num_prefills} num_prefills {sum(num_prefills.values())} num_tokens {num_tokens} true_length {row.true_length} num_empty_slots {len(empty_slots)} num_decodes {num_decodes}"
      )
      slot = empty_slots.pop()
      first_token = prefill(slot, row.tokens, row.true_length)
      self.detokenize_backlog.put((first_token, True, row.id, slot), block=True)

    while slot_to_id:
      log.info(f"decode-{desc}-{num_decodes} num_filled_slots {len(slot_to_id)}")
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
        log.info(f"res[{id_}] eos")
      if not res[id_] or res[id_][-1] != self.tokenizer.eos_id:
        res[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback, desc=desc)
    return res
