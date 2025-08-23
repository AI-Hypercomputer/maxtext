# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Offline inference for mlperf """

from collections import defaultdict
from typing import Any, Callable
import dataclasses
import functools
import logging
import os
import queue
import random
import signal
import threading
import traceback

import numpy as np

import jax
from jax.experimental import layout

from jetstream.engine import engine_api

# pylint: disable=no-name-in-module
from MaxText.maxengine import MaxEngine
from MaxText.maxengine import set_engine_vars_from_base_engine
from MaxText.prefill_packing import PrefillProcessor
from MaxText.prefill_packing import BatchedPrefillProcessor

DecodeState = Any
Params = Any
PRNGKeyType = Any

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

  def __init__(self, kind: str, engine: MaxEngine):
    self._type = kind
    self.engine = engine
    if self._type == "default":
      self._processor = PrefillProcessor(engine)
    elif self._type == "batch":
      self._batch_processor = BatchedPrefillProcessor(engine=engine, max_batch_size=16)
      self._processor = PrefillProcessor(engine)  # for fallback
    elif self._type == "dummy":
      pass
    else:
      raise ValueError(f"Invalid type: {self._type}")

  def aot_compile(
      self,
      max_length: int,
      params: Params,
      params_layout: layout.Format,
      decode_state_layout: layout.Format,
      decode_state_shape: jax.ShapeDtypeStruct,
  ) -> None:
    """Ahead-of-Time compile"""
    if max_length > 4096:
      raise ValueError(f"Max length exceeds 4096. {max_length=}")
    if self._type == "default":
      # length buckets = (0, 64], (64, 128], (128, 256], ...
      # lengths has at least one bucket.
      # bucket limits are aligned to exponential of 2.
      # the last bucket is the only one that can hold `max_length`.
      buckets = [2**i for i in range(6, max(6, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        self._processor.aot_compile(params, bucket)
    elif self._type == "batch":
      # length buckets = (0, 128], (128, 256], (256, 512], ...
      # lengths has at least one item.
      # each item aligns to exponential of 2.
      # the last bucket is the only one that can hold `max_length`.
      buckets = [2**i for i in range(7, max(7, (max_length - 1).bit_length()) + 1)]
      for bucket in buckets:
        for num_prompts in range(1, 2 * max_length // bucket):
          self._batch_processor.aot_compile(params, bucket, max_length, num_prompts)
      # for fallback
      for bucket in [max_length]:
        self._processor.aot_compile(params, bucket)
    else:
      assert self._type == "dummy", f"type: {self._type}"

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: int,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      max_length: int,
      prefill_done: Callable[[list[tuple[engine_api.ResultTokens, int]], list[int], DecodeState], None],
      rng: PRNGKeyType,
  ) -> None:
    """Prefill helper process runner"""
    padded_length = len(input_tokens_padded)
    if self._type == "default":
      first_token, decode_state = self._processor.process(
          model_params, decode_state, decode_slot, input_tokens_padded, input_true_length, rng
      )
      prefill_done([(first_token, decode_slot)], [input_id], decode_state)
    elif self._type == "batch":
      if padded_length == max_length:
        # fallback to default mode
        first_token, decode_state = self._processor.process(
            model_params, decode_state, decode_slot, input_tokens_padded, input_true_length, rng
        )
        prefill_done([(first_token, decode_slot)], [input_id], decode_state)
      else:
        self._batch_processor.process(
            model_params=model_params,
            decode_state=decode_state,
            decode_slot=decode_slot,
            input_id=input_id,
            input_prompt=input_tokens_padded[:input_true_length],
            input_padding=padded_length,
            capacity=max_length,
            prefill_done=prefill_done,
        )
    else:
      assert self._type == "dummy", f"type: {self._type}"
      log.debug("dummy prefill")
      prefill_done([(123, decode_slot)], [input_id], decode_state)

  def finalize(
      self,
      model_params: Params,
      decode_state: DecodeState,
      prefill_done: Callable[[list[tuple[engine_api.ResultTokens, int]], list[int], DecodeState], None],
  ) -> None:
    """Finalize helper process"""
    if self._type == "default":
      pass
    elif self._type == "batch":
      self._batch_processor.flush(model_params, decode_state, prefill_done)
    else:
      assert self._type == "dummy", f"type: {self._type}"


class OfflineInference:
  """Offline inference for mlperf"""

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

    self.dummy = False

    if self.dummy:
      self.prefill = PrefillHelper("dummy", self.engine)
    elif enable_batch_prefill:
      self.prefill = PrefillHelper("batch", self.engine)
    else:
      self.prefill = PrefillHelper("default", self.engine)
    self.batch_size = engine.max_concurrent_decodes
    self.max_prefill_length = engine.config.max_prefill_predict_length
    self.max_decode_length = engine.config.max_target_length - engine.config.max_prefill_predict_length
    self.tokenizer = engine.build_tokenizer(engine.get_tokenizer())

    self._cached_generate = None
    self.detokenize_backlog = queue.Queue(10)

    self._decode_state_executable = None

  def init_decode_state(self):
    """Instantiate decode state"""
    if self.decode_state is None:
      assert self._decode_state_executable is not None, "Decode state executable is none"
      self.decode_state = self._decode_state_executable(None)

  def warmup(self, max_length, warmup_samples):
    """Warmup (cache, AoT compile, batch_inference)"""
    self._cached_generate, self.params, self._decode_state_executable = self.engine.aot_compile(
        self.params, pass_rng_shape=False
    )

    self.init_decode_state()

    self.prefill.aot_compile(
        max_length,
        self.params,
        self.engine.param_layouts,
        self.engine.decode_state_layouts,
        self.engine.decode_state_shapes,
    )

    self.batch_inference(warmup_samples, desc="warmup")

  def batch_inference_with_callback(
      self,
      data: list[InputData],
      emit_first_token: Callable[[str, int], bool],
      emit_token: Callable[[str, int], bool],
      desc: str,
  ):
    """callback is a function that takes id and token. It will be called once per output

    token.
    """

    empty_slots = list(range(self.batch_size))
    slot_to_id = {}

    counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
    dummy_length = 1

    rng = jax.random.PRNGKey(1234)
    rng, _ = jax.random.split(rng)

    def prefill_done(prefill_result, ids, decode_state):
      nonlocal self
      nonlocal counter
      self.decode_state = decode_state
      for i, (first_token, slot_) in enumerate(prefill_result):
        counter.prefill += 1
        log.debug("prefill done: slot=%d (%d)", slot_, counter.prefill)
        self.detokenize_backlog.put((first_token, True, ids[i], slot_), block=True)

    def decode():
      nonlocal self
      nonlocal dummy_length
      nonlocal counter
      counter.decode += 1
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
          # log.info("Decode put result %d to queue", i)

    def detokenize():
      nonlocal self
      nonlocal slot_to_id
      nonlocal empty_slots
      nonlocal counter
      while self.live and counter.detokenize < counter.input:
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
          log.debug("slot is %s, length is %d", slot, length)
          should_finish = False
          if is_valid:
            should_finish = emit_token(id_, token.item())
          if should_finish or length >= self.max_decode_length:
            newly_empty.append(slot)
            counter.detokenize += 1
            log.debug("Detokenize free up %s, length %d", slot, length)
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

    counter.input = len(data)
    self.live = True

    detokenize_thread.start()
    for row in data:
      # Do decode until there are free slots
      while not empty_slots:
        decode()
      slot = empty_slots.pop()

      # Do prefill when there are free slots
      self.prefill.process(
          self.params,
          self.decode_state,
          slot,
          int(row.id),
          row.tokens,
          row.true_length,
          self.max_prefill_length,
          prefill_done,
          rng,
      )
    self.prefill.finalize(self.params, self.decode_state, prefill_done)

    while slot_to_id:
      decode()

    self.live = False
    detokenize_thread.join()
    log.info(
        "summary-%s-prefills-%d-decodes-%d-detokens-%d completed.",
        desc,
        counter.prefill,
        counter.decode,
        counter.detokenize,
    )

  def batch_inference(self, data: list[InputData], desc="") -> dict[str, list[int]]:
    """data is list of obj with id, tokens, and true length"""
    data_dict = defaultdict(list)
    log.info("sorting data")
    for row in data:
      data_dict[row.tokens.shape[0]].append(row)
    data_dict[128] += data_dict[64]
    data_dict[64] = []
    data = []
    for padded_len in [128, 256, 512, 1024]:
      log.info("padded len: %d, num: %d", padded_len, len(data_dict[padded_len]))
      random.shuffle(data_dict[padded_len])
      data += data_dict[padded_len]
    log.info("finished sorting data")
    res = defaultdict(list)

    def callback(id_, token):
      nonlocal res
      if token == self.tokenizer.eos_id:
        log.debug("res[%d] eos", id_)
      if not res[id_] or res[id_][-1] != self.tokenizer.eos_id:
        res[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback, desc=desc)
    return res
