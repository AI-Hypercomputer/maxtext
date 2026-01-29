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

"""Implementation of Prefill Packing feature"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jetstream.engine import engine_api

from MaxText.maxengine import MaxEngine

import warnings
import logging

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any

log = logging.getLogger(__name__)


class PrefillBucket:
  """Manage a list of prefill requests."""

  def __init__(self, capacity: int):
    # decode slots
    self.slots = []
    # input row ids
    self.row_ids = []
    # prompt token ids, a list of jax.Array
    self.token_ids = []

    if capacity <= 0:
      raise ValueError("capacity must be a positive number.")
    self.capacity = capacity
    self.length = 0
    self.count = 0

  def clear(self) -> None:
    """Clear bucket."""
    self.slots = []
    self.row_ids = []
    self.token_ids = []
    self.length = 0
    self.count = 0

  def try_add(self, slot: int, row_id: int, token_ids: jax.Array) -> bool:
    """Try to add a prefill prompt to bucket.

    Returns False if bucket doesn't have enough space.
    Raise ValueError if prompt length exceeds capacity."""
    if len(token_ids) > self.capacity:
      raise ValueError(f"Prefill length exceeds capacity. prefill length: {len(token_ids)}, capacity: {self.capacity}")
    if len(token_ids) > self.unallocated():
      return False
    self.slots.append(slot)
    self.row_ids.append(row_id)
    self.token_ids.append(token_ids)
    self.length += len(token_ids)
    self.count += 1
    return True

  def add(self, slot: int, row_id: int, token_ids: jax.Array) -> None:
    """Adds a prefill prompt to bucket.

    Raise ValueError if fails.
    """
    if not self.try_add(slot, row_id, token_ids):
      raise ValueError(
          f"Not enough space. prefill length: {len(token_ids)}, unallocated length: {self.capacity - self.length}"
      )

  def is_empty(self) -> bool:
    return self.count == 0

  def unallocated(self) -> int:
    return self.capacity - self.length


class PrefillProcessor:
  """A wrapper around MaxEngine prefill and insert API."""

  def __init__(self, engine: MaxEngine):
    self.engine = engine
    self.process_func = {}

  def aot_compile(self, params: Params, input_padding: int):
    """Ahead-of-time compile prefill processing routines."""

    return self._process_compiled(params, input_padding)

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      rng: PRNGKeyType,
      return_prompt_logp: bool = False,
  ) -> tuple[engine_api.ResultTokens, DecodeState]:
    """Process a new input."""

    process_fn = self._process_compiled(model_params, len(input_tokens_padded), return_prompt_logp)
    return process_fn(
        model_params, input_tokens_padded, decode_slot, input_true_length, decode_state, rng, return_prompt_logp
    )

  def _process_compiled(self, params: Params, padded_length: int, return_prompt_logp: bool = False):
    """Ahead-of-time compilation wrapper of _process()."""

    if padded_length not in self.process_func:
      log.info("compile prefill process(%d)", padded_length)
      self.process_func[(padded_length, return_prompt_logp)] = (
          jax.jit(
              self._process,
              in_shardings=(self.engine.param_layouts, None, None, None, self.engine.decode_state_layouts, None),
              out_shardings=(
                  None,
                  self.engine.decode_state_layouts,
              ),
              donate_argnames=("decode_state"),
              static_argnames=("return_prompt_logp",),
          )
          .lower(
              params,
              jax.ShapeDtypeStruct((padded_length,), jnp.dtype("int32")),
              jax.ShapeDtypeStruct((), int),
              jax.ShapeDtypeStruct((), int),
              self.engine.decode_state_shapes,
              jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32")),
              return_prompt_logp,
          )
          .compile(compiler_options=None)
      )
    return self.process_func[(padded_length, return_prompt_logp)]

  def _process(
      self,
      params: Params,
      tokens: jax.Array,
      slot: int,
      true_length: int,
      decode_state: DecodeState,
      rng: PRNGKeyType,
      return_prompt_logp: bool = False,
  ) -> tuple[engine_api.ResultTokens, DecodeState]:
    """Prefill and insert a request."""

    prefill_result, first_token = self.engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length, rng=rng, return_prompt_logp=return_prompt_logp
    )
    decode_state = self.engine.insert(prefill_result, decode_state, slot)
    if return_prompt_logp:
      decode_state["prompt_logp"] = prefill_result["prompt_logp"]
    return first_token, decode_state


class BatchedPrefillProcessor:
  """A wrapper around the APIs used by MaxEngine to do prefill and insert, provides prefill packing feature."""

  def __init__(self, engine: MaxEngine, max_batch_size: int, auto_layout_supported: bool = True):
    self.engine = engine
    self.process_batch_func = {}
    self.buckets = {}
    self.max_batch_size = max_batch_size
    self.auto_layout_supported = auto_layout_supported
    self.jitted_process_batch = jax.jit(
        self._process_batch,
        static_argnames=("num_prompts", "padded_length", "return_prompt_logp"),
        donate_argnames=("decode_state"),
    )

  def aot_compile(
      self, params: Params, input_padding: int, capacity: int, num_prompts: int, return_prompt_logp: bool = False
  ):
    """Ahead-of-time compile prefill processing routines."""

    return self._process_batch_compiled(params, input_padding, capacity, num_prompts, return_prompt_logp)

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: int,
      input_prompt: jax.Array,
      input_padding: int,
      capacity: int,
      prefill_done: Callable[[list[tuple[engine_api.ResultTokens, int]], list[int], DecodeState], None],
      return_prompt_logp: bool = False,
  ) -> None:
    """Process a new input.

    This may trigger MaxEngine prefill API call."""
    length = len(input_prompt)
    if length > capacity or length > input_padding:
      raise ValueError(
          f"Prefill length exceeds limit. prefill length: {length} padding: {input_padding} capacity: {capacity}"
      )

    bucket = self.buckets.setdefault(input_padding, PrefillBucket(capacity))
    if len(input_prompt) > bucket.unallocated():
      prefill_result, decode_state = self._process_bucket(
          model_params, bucket, input_padding, decode_state, return_prompt_logp
      )
      if prefill_done:
        prefill_done(prefill_result, bucket.row_ids, decode_state)
      bucket.clear()
    bucket.add(decode_slot, input_id, input_prompt)

    log.debug(
        "prefill: slot=%d, id=%d, length=%d, padding=%d, capacity=%d, unallocated=%d",
        decode_slot,
        input_id,
        length,
        input_padding,
        capacity,
        bucket.unallocated(),
    )

  def flush(
      self,
      model_params: Params,
      decode_state: DecodeState,
      prefill_done: Callable[[list[tuple[engine_api.ResultTokens, int]], list[int], DecodeState], None],
      return_prompt_logp: bool = False,
  ) -> None:
    """Process all remaining items in buckets."""

    for input_padding, bucket in self.buckets.items():
      if not bucket.is_empty():
        prefill_result, decode_state = self._process_bucket(
            model_params, bucket, input_padding, decode_state, return_prompt_logp
        )
        if prefill_done:
          prefill_done(prefill_result, bucket.row_ids, decode_state)
        bucket.clear()

  def _process_bucket(
      self,
      model_params: Params,
      bucket: PrefillBucket,
      input_padding: int,
      decode_state: DecodeState,
      return_prompt_logp: bool = False,
  ) -> tuple[list[tuple[engine_api.ResultTokens, int]], DecodeState]:
    """Process all items in a bucket."""
    # pylint: disable=import-outside-toplevel
    from maxtext.inference.offline_engine import PrefillResult  # type: ignore

    slots = bucket.slots
    lengths = [len(prompt) for prompt in bucket.token_ids]
    offsets = np.cumsum([0] + lengths)[:-1].tolist()

    tok_ids = bucket.token_ids
    tok_ids.append(jnp.zeros(bucket.unallocated(), dtype=jnp.int32))
    tok_ids = jnp.concat(tok_ids)

    pos_ids = []
    for length in lengths:
      pos_ids.append(np.arange(length, dtype=int))
    pos_ids.append(np.arange(bucket.unallocated(), dtype=int))
    pos_ids = jnp.array(np.concatenate(pos_ids))

    seg_ids = []
    for i in range(bucket.count):
      seg_ids.append(np.full(lengths[i], i * 2 + 1, dtype=int))
    seg_ids.append(np.zeros(bucket.unallocated(), dtype=int))
    seg_ids = jnp.array(np.concatenate(seg_ids))

    # Use padding below to keep static shape of jitted function input.
    def zero_padded(arr: list[int], padding: int):
      if len(arr) < padding:
        arr.extend([0] * (padding - len(arr)))
      return jnp.array(arr)

    slots = zero_padded(slots, self.max_batch_size)
    offsets_jax = zero_padded(offsets, self.max_batch_size)
    lengths_jax = zero_padded(lengths, self.max_batch_size)
    if not self.auto_layout_supported:
      first_tokens, decode_state = self.jitted_process_batch(
          model_params,
          tok_ids,
          slots,
          bucket.count,
          pos_ids,
          seg_ids,
          offsets_jax,
          input_padding,
          lengths_jax,
          decode_state,
          return_prompt_logp,
      )
    else:
      prefill_fn = self._process_batch_compiled(
          model_params, input_padding, bucket.capacity, bucket.count, return_prompt_logp
      )
      first_tokens, decode_state = prefill_fn(
          model_params, tok_ids, slots, pos_ids, seg_ids, offsets_jax, lengths_jax, decode_state, return_prompt_logp
      )

    prefill_result = []
    prompt_logp_numpy = None
    if return_prompt_logp:
      prompt_logp_numpy = np.array(decode_state["prompt_logp"])
    for i in range(bucket.count):
      if return_prompt_logp:
        prompt_logp = prompt_logp_numpy[:, offsets[i] : offsets[i] + lengths[i]]
        prefill_result.append(PrefillResult(first_tokens[i], bucket.slots[i], prompt_logp))
      else:
        prefill_result.append(PrefillResult(first_tokens[i], bucket.slots[i], None))
    return prefill_result, decode_state

  def _process_batch_compiled(
      self, params: Params, padded_length: int, capacity: int, num_prompts: int, return_prompt_logp: bool
  ):
    """Ahead-of-time compilation wrapper of _process_batch()."""

    if (padded_length, num_prompts) not in self.process_batch_func:
      log.info("compile prefill process_batch{(%d, %d)} capacity=%d", padded_length, num_prompts, capacity)
      self.process_batch_func[(padded_length, num_prompts, return_prompt_logp)] = (
          jax.jit(
              self._process_batch,
              in_shardings=(
                  self.engine.param_layouts,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  self.engine.decode_state_layouts,
              ),
              out_shardings=(None, self.engine.decode_state_layouts),
              static_argnames=(
                  "num_prompts",
                  "padded_length",
                  "return_prompt_logp",
              ),
              donate_argnames=("decode_state"),
          )
          .lower(
              params,
              jax.ShapeDtypeStruct((capacity,), jnp.dtype("int32")),
              jnp.arange(0, self.max_batch_size, dtype=int),
              num_prompts,
              jnp.arange(0, capacity, dtype=int),
              jnp.ones(capacity, dtype=int),
              jnp.arange(0, capacity, capacity // self.max_batch_size, dtype=int),
              padded_length,
              jnp.full(self.max_batch_size, padded_length, dtype=int),
              self.engine.decode_state_shapes,
              return_prompt_logp,
          )
          .compile(compiler_options=None)
      )
    return self.process_batch_func[(padded_length, num_prompts, return_prompt_logp)]

  def _process_batch(  # pylint: disable=too-many-positional-arguments
      self,
      params: Params,
      tokens: jax.Array,
      slots: jax.Array,
      num_prompts: int,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array,
      start_pos: jax.Array,
      padded_length: int,
      true_lengths: jax.Array,
      decode_state: DecodeState,
      return_prompt_logp: bool = False,
  ) -> tuple[list[engine_api.ResultTokens], DecodeState]:
    """Prefill and insert a packed request."""

    cache, prefix_state, first_tokens = self.engine.prefill_concat(
        params=params,
        padded_tokens=tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        start_pos=start_pos,
        true_lengths=true_lengths,
        num_prompts=num_prompts,
        return_prompt_logp=return_prompt_logp,
    )
    decode_state = self.engine.insert_partial(
        prefix=prefix_state,
        decode_state=decode_state,
        cache=cache,
        slots=slots,
        start_indices=start_pos,
        num_prompts=num_prompts,
        seq_len=padded_length,
    )
    if return_prompt_logp:
      decode_state["prompt_logp"] = prefix_state["prompt_logp"]
    return first_tokens, decode_state
