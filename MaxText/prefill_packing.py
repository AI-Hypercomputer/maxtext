#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Implementation of Prefill Packing feature"""

import common_types
import functools
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Callable

from flax.linen import partitioning as nn_partitioning

import jax
import jax.numpy as jnp
import numpy as np

from jetstream.engine import engine_api

import max_utils
import inference_utils
from maxengine import MaxEngine

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
      raise ValueError(f"Not enough space. prefill length: {len(token_ids)}, unallocated length: {self.capacity - self.length}")

  def is_empty(self) -> bool:
    return self.count == 0

  def unallocated(self) -> int:
    return self.capacity - self.length


class PrefillProcessor:
  """A wrapper around MaxEngine prefill and insert API."""

  def __init__(self, engine: MaxEngine):
    self.engine = engine
    self.process_func = {}

  def aot_compile(
    self,
    params: Params,
    input_padding: int
  ):
    """Ahead-of-time compile prefill processing routines."""

    return self._process_compiled(params, input_padding)

  def process(
    self,
    model_params: Params,
    decode_state: DecodeState,
    decode_slot: int,
    input_tokens_padded: jax.Array,
    input_true_length: int
  ) -> Tuple[engine_api.ResultTokens, DecodeState]:
    """Process a new input."""

    process_fn = self._process_compiled(model_params, len(input_tokens_padded))
    return process_fn(
        model_params,
        input_tokens_padded,
        decode_slot,
        input_true_length,
        decode_state)

  def _process_compiled(
    self,
    params: Params,
    padded_length: int
  ):
    """Ahead-of-time compilation wrapper of _process()."""

    if padded_length not in self.process_func:
      log.info(f"compile prefill process({padded_length})")
      self.process_func[padded_length] = (
        jax.jit(
            self._process,
            in_shardings=(self.engine.param_layouts, None, None, None, self.engine.decode_state_layouts),
            out_shardings=(None, self.engine.decode_state_layouts,),
            donate_argnames=("decode_state"),
        )
        .lower(
            params,
            jax.ShapeDtypeStruct((padded_length,), jnp.dtype("int32")),
            jax.ShapeDtypeStruct((), int),
            jax.ShapeDtypeStruct((), int),
            self.engine.decode_state_shapes
        )
        .compile(compiler_options=None)
      )
    return self.process_func[padded_length]

  def _process(
    self,
    params: Params,
    tokens: jax.Array,
    slot: int,
    true_length: int,
    decode_state: DecodeState,
  ) -> Tuple[engine_api.ResultTokens, DecodeState]:
    """Prefill and insert a request."""

    prefill_result, first_token = self.engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = self.engine.insert(prefill_result, decode_state, slot)
    return first_token, decode_state


class BatchedPrefillProcessor:
  """A wrapper around the APIs used by MaxEngine to do prefill and insert, provides prefill packing feature."""

  def __init__(self, engine: MaxEngine):
    self.engine = engine
    self.process_batch_func = {}
    self.buckets = {}

  def aot_compile(
    self,
    params: Params,
    input_padding: int,
    capacity: int,
    num_prompts: int
  ):
    """Ahead-of-time compile prefill processing routines."""

    return self._process_batch_compiled(params, input_padding, capacity, num_prompts)

  def process(
    self,
    model_params: Params,
    decode_state: DecodeState,
    decode_slot: int,
    input_id: int,
    input_prompt: jax.Array,
    input_padding: int,
    capacity: int,
    prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None]
  ) -> None:
    """Process a new input.
    
    This may trigger MaxEngine prefill API call."""

    length = len(input_prompt)
    if (length > capacity or length > input_padding):
      raise ValueError(f"Prefill length exceeds limit. prefill length: {length} padding: {input_padding} capacity: {capacity}")

    bucket = self.buckets.setdefault(input_padding, PrefillBucket(capacity))
    if len(input_prompt) > bucket.unallocated():
      prefill_result, decode_state = self._process_bucket(model_params, bucket, input_padding, decode_state)
      if prefill_done:
        prefill_done(prefill_result, bucket.row_ids, decode_state)
      bucket.clear()
    bucket.add(decode_slot, input_id, input_prompt)

    log.debug(f"prefill: slot={decode_slot} id={input_id}, length={length}, padding={input_padding}, {capacity=}, unallocated={bucket.unallocated()}")

  def flush(
    self,
    model_params: Params,
    decode_state: DecodeState,
    prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None]
  ) -> None:
    """Process all remaining items in buckets."""

    for input_padding in self.buckets.keys():
      bucket = self.buckets[input_padding]
      if not bucket.is_empty():
        prefill_result, decode_state = self._process_bucket(model_params, bucket, input_padding, decode_state)
        if prefill_done:
          prefill_done(prefill_result, bucket.row_ids, decode_state)
        bucket.clear()

  def _process_bucket(
    self,
    model_params: Params,
    bucket: PrefillBucket,
    input_padding: int,
    decode_state: DecodeState
  ) -> Tuple[List[Tuple[engine_api.ResultTokens, int]], DecodeState]:
    """Process all items in a bucket."""

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

    slots   = zero_padded(slots,   16)
    offsets = zero_padded(offsets, 16)
    lengths = zero_padded(lengths, 16)

    prefill_fn = self._process_batch_compiled(model_params,
                                              input_padding,
                                              bucket.capacity,
                                              bucket.count)
    first_tokens, decode_state = prefill_fn(
        model_params,
        tok_ids,
        slots,
        pos_ids,
        seg_ids,
        offsets,
        lengths,
        decode_state,
    )

    prefill_result = []
    for i in range(bucket.count):
      prefill_result.append((first_tokens[i], bucket.slots[i]))
    return prefill_result, decode_state

  def _process_batch_compiled(
    self,
    params: Params,
    padded_length: int,
    capacity: int,
    num_prompts: int
  ):
    """Ahead-of-time compilation wrapper of _process_batch()."""

    if (padded_length, num_prompts) not in self.process_batch_func:
      log.info(f"compile prefill process_batch{(padded_length, num_prompts)} {capacity=}")
      self.process_batch_func[(padded_length, num_prompts)] = (
        jax.jit(
            self._process_batch,
            in_shardings=(self.engine.param_layouts, None, None, None, None, None, None, self.engine.decode_state_layouts),
            out_shardings=(None, self.engine.decode_state_layouts),
            static_argnames=("num_prompts", "padded_length"),
            donate_argnames=("decode_state"),
        )
        .lower(
            params,
            jax.ShapeDtypeStruct((capacity,), jnp.dtype("int32")),
            jnp.arange(0, 16, dtype=int),
            num_prompts,
            jnp.arange(0, capacity, dtype=int),
            jnp.ones(capacity, dtype=int),
            jnp.arange(0, capacity, 64, dtype=int),
            padded_length,
            jnp.full(16, padded_length, dtype=int),
            self.engine.decode_state_shapes,
        )
        .compile(compiler_options=None)
      )
    return self.process_batch_func[(padded_length, num_prompts)]

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
  ) -> Tuple[List[engine_api.ResultTokens], DecodeState]:
    """Prefill and insert a packed request."""

    cache, prefix_state, first_tokens = self._prefill_batch(
        params=params,
        padded_tokens=tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        start_pos=start_pos,
        true_lengths=true_lengths,
        num_prompts=num_prompts,
    )
    decode_state = self._insert_batch(
        prefix_state=prefix_state,
        cache=cache,
        start_indices=start_pos,
        seq_len=padded_length,
        num_prompts=num_prompts,
        slots=slots,
        decode_state=decode_state,
    )
    return first_tokens, decode_state

  @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_prompts",))
  def _prefill_batch(
      self,
      *,
      params: Params,
      padded_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array,
      start_pos: jax.Array,
      true_lengths: jax.Array,
      num_prompts: int,
      rng: Optional[PRNGKeyType] = None,
  ) -> Tuple[Any, PackedPrefix, List[engine_api.ResultTokens]]:
    """Computes a kv-cache for a packed request."""

    if rng is None:
      rng = jax.random.PRNGKey(0)

    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
    decoder_positions = jnp.expand_dims(decoder_positions, 0)
    decoder_segment_ids = jnp.expand_dims(decoder_segment_ids, 0)
    rng, new_rng = jax.random.split(rng)
    with self.engine._mesh, nn_partitioning.axis_rules(self.engine.config.logical_axis_rules):
      flat_logits, new_vars = self.engine.model.apply(
          params,
          input_tokens,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": new_rng},
          mutable=["cache"],
      )

    cache = new_vars["cache"]
    cache = self.engine._maybe_stack_prefill_result_cache(cache)

    def process_packed_logits_and_caches(packed_flat_logits, idx):
      next_pos = jnp.full((1, 1), true_lengths[idx], dtype=jnp.int32)
      generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
      selected_logits = jax.lax.dynamic_slice(
          packed_flat_logits,
          (0, start_pos[idx] + true_lengths[idx] - 1, 0),
          (packed_flat_logits.shape[0], 1, packed_flat_logits.shape[2]),
      )
      selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.engine.replicated_sharding)
      first_generated_token = inference_utils.sampling(
          selected_logits,
          rng,
          self.engine.config.decode_sampling_strategy,
          topk=self.engine.config.decode_sampling_top_k,
          nucleus_topp=self.engine.config.decode_sampling_nucleus_p,
          temperature=self.engine.config.decode_sampling_temperature,
      )
      all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
      result = engine_api.ResultTokens(
          data=jnp.concatenate((first_generated_token, all_valid, generated_tokens), axis=1),
          # Tokens are shape [batch, speculations], so when we concatenate
          # tokens, validity and length along their index 1 dimension then they
          # occupy 0:speculations.
          tokens_idx=(0, 1),
          # Validity occupies the same amount of space, but next in line.
          valid_idx=(1, 2),
          # And lengths is rank 1.
          length_idx=(2, 3),
          samples_per_slot=1,
      )
      return {
          "logits": selected_logits,
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": first_generated_token,
      }, result

    prefill_results = defaultdict(list)
    first_tokens = []
    for idx in range(num_prompts):
      prefill_result, first_token = process_packed_logits_and_caches(flat_logits, idx)
      for k, v in prefill_result.items():
        prefill_results[k].append(v)
      first_tokens.append(first_token)
    prefill_results = {k: jnp.stack(v) for k, v in prefill_results.items()}
    return cache, prefill_results, first_tokens

  @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_prompts", "seq_len",), donate_argnames=("prefix_state", "decode_state",))
  def _insert_batch(
      self,
      *,
      prefix_state: PackedPrefix,
      cache: Any,
      start_indices: jax.Array,
      seq_len: int,
      num_prompts: int,
      slots: jax.Array,
      decode_state: DecodeState,
  ) -> DecodeState:
    """Insert computed kv-cache."""
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix_state)
    cache_unboxed = max_utils.unbox_logicallypartioned(cache)
    cache_unboxed = self.engine._maybe_unstack_prefill_result_cache(cache_unboxed)
    start_idx = 0
    slot = slots[0]

    def copy(path, partial_cache, full_cache, annotations):
      path_key = path[-1].key
      if path_key in [
          "cache_ar_index",
          "cached_ar_key",
          "cached_ar_value",
          "cached_ar_key_scale",
          "cached_ar_value_scale",
      ]:
        return full_cache  # we don't even zero these out because we can mask them out.

      batch_idx = -1
      if "cache_batch" in annotations:
        batch_idx = annotations.index("cache_batch")
      elif "cache_scale_batch" in annotations:
        batch_idx = annotations.index("cache_scale_batch")

      if batch_idx < 0:
        raise ValueError(f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}")

      if path_key == "cache_ar_segment_id":
        ### goal: zero this out in case there is existing data
        zeros = jnp.zeros((1, self.engine.config.max_target_length - self.engine.config.max_prefill_predict_length), dtype=jnp.int32)
        return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      elif path_key == "cache_prefill_segment_id":
        zeros = jnp.zeros((1, self.engine.config.max_prefill_predict_length), dtype=jnp.int32)
        ## zero out in case prefill cache is too small to cover
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
        # In case partial_cache is too small to slice at the given index, pad it with an extra seqlen
        if i == num_prompts - 1:
          pad = jnp.zeros((1, seq_len), dtype=int)
          partial_cache = jnp.concatenate([partial_cache, pad], axis=1)
        ## copy prefill cache
        partial_cache = jax.lax.dynamic_slice(partial_cache, (0, start_idx), (1, seq_len))
        partial_cache = (partial_cache == partial_cache[0, 0]).astype(int)
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
        return full_cache
      elif path_key == "cached_ar_lengths":
        return full_cache.at[slot].set(0)
      elif path_key in [
          "cached_prefill_key",
          "cached_prefill_value",
          "cached_prefill_key_scale",
          "cached_prefill_value_scale",
      ]:
        seqlen_index = self.engine.config.prefill_cache_axis_order.split(",").index("1")
        start_indices = [0, 0, 0, 0]
        start_indices[seqlen_index] = start_idx
        slice_size = list(partial_cache.shape)
        slice_size[seqlen_index] = seq_len

        slice_size = tuple(slice_size)
        # Same as in prefill_segment_id processing
        if i == num_prompts - 1:
          pad = jnp.zeros(slice_size, dtype=partial_cache.dtype)
          partial_cache = jnp.concatenate([partial_cache, pad], axis=seqlen_index)
        partial_cache = jax.lax.dynamic_slice(partial_cache, start_indices, slice_size)

        return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
      else:
        raise ValueError(f"We don't have a strategy for inserting {path_key}")

    inserted_cache = decode_state["cache"]
    inserted_logits = decode_state["logits"]
    inserted_next_pos = decode_state["next_pos"]
    inserted_generated_tokens = decode_state["generated_tokens"]
    inserted_tokens = decode_state["tokens"]

    for i in range(num_prompts):
      start_idx = start_indices[i]
      slot = slots[i]
      inserted_cache = jax.tree_util.tree_map_with_path(copy, cache_unboxed, inserted_cache, self.engine.kv_cache_annotations_named)
      inserted_logits = jax.lax.dynamic_update_index_in_dim(inserted_logits, unboxed_prefix["logits"][i, ...], slot, 0)
      inserted_next_pos = jax.lax.dynamic_update_index_in_dim(inserted_next_pos, unboxed_prefix["next_pos"][i, ...], slot, 0)
      inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
          inserted_generated_tokens,
          unboxed_prefix["generated_tokens"][i, ...],
          slot,
          0,
      )
      inserted_tokens = jax.lax.dynamic_update_index_in_dim(inserted_tokens, unboxed_prefix["tokens"][i, ...], slot, 0)

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.engine.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.engine.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.engine.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.engine.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.engine.kv_cache_shardings)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
    }
