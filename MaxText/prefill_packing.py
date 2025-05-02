# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Prefill Packing feature"""

from typing import Any, List, Tuple, Callable # Duplicate Callable

import jax
import jax.numpy as jnp
import numpy as np

from jetstream.engine import engine_api

from MaxText.maxengine import MaxEngine # Assuming MaxEngine is correctly importable

import warnings
import logging
# import uuid # If request_id were to be generated here

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any # Matches MaxEngine.ExistingPrefix.cache or MaxEngine.prefill return
# PackedPrefix refers to the 'prefix_state' from prefill_concat, not used by PrefillProcessor
Params = Any
PRNGKeyType = Any

log = logging.getLogger(__name__)


class PrefillBucket:
  """Manage a list of prefill requests."""
  # ... (no changes needed here) ...
  def __init__(self, capacity: int):
    self.slots = []
    self.row_ids = []
    self.token_ids = []
    if capacity <= 0:
      raise ValueError("capacity must be a positive number.")
    self.capacity = capacity
    self.length = 0
    self.count = 0

  def clear(self) -> None:
    self.slots = []
    self.row_ids = []
    self.token_ids = []
    self.length = 0
    self.count = 0

  def try_add(self, slot: int, row_id: int, token_ids: jax.Array) -> bool:
    if len(token_ids) > self.capacity:
      raise ValueError(f"Prefill length exceeds capacity. prefill length: {len(token_ids)}, capacity: {self.capacity}")
    if len(token_ids) > self.unallocated():
      return False
    self.slots.append(slot)
    self.row_ids.append(row_id) # row_id can be str or int, consistent with BatchedPrefillProcessor
    self.token_ids.append(token_ids)
    self.length += len(token_ids)
    self.count += 1
    return True

  def add(self, slot: int, row_id: int, token_ids: jax.Array) -> None:
    if not self.try_add(slot, row_id, token_ids):
      raise ValueError(f"Not enough space. prefill length: {len(token_ids)}, unallocated length: {self.capacity - self.length}")

  def is_empty(self) -> bool:
    return self.count == 0

  def unallocated(self) -> int:
    return self.capacity - self.length


class PrefillProcessor:
  """A wrapper around MaxEngine prefill and insert API for single (non-packed) prefills."""

  def __init__(self, engine: MaxEngine):
    self.engine = engine
    self.process_func: dict[int, Callable[..., Tuple[engine_api.ResultTokens, DecodeState]]] = {}


  def aot_compile(
    self,
    params: Params,
    input_padding: int # This is the 'padded_length' for which to compile
  ):
    """Ahead-of-time compile prefill processing routines for a specific padded length."""
    # This will get or create the compiled version of _process for the given padding.
    self._process_compiled(params, input_padding)
    # No explicit return needed, compilation is a side effect.

  def process(
    self,
    model_params: Params,
    decode_state: DecodeState,
    decode_slot: int,
    input_tokens_padded: jax.Array,
    input_true_length: int
  ) -> Tuple[engine_api.ResultTokens, DecodeState]:
    """Process a new input by prefilling and inserting it."""
    padded_length = len(input_tokens_padded)
    # Get or compile the JITted function for this padded_length
    compiled_process_fn = self._process_compiled(model_params, padded_length)
    
    # Call the JITted function
    return compiled_process_fn(
        model_params,
        input_tokens_padded, # 'tokens' arg for _process
        decode_slot,         # 'slot' arg for _process
        input_true_length,   # 'true_length' arg for _process
        decode_state         # 'decode_state' arg for _process
    )

  def _process_compiled(
    self,
    params: Params, # Abstract params for shape inference during .lower()
    padded_length: int
  ) -> Callable[..., Tuple[engine_api.ResultTokens, DecodeState]]:
    """
    Gets or compiles the JITted _process function for a given padded_length.
    """
    if padded_length not in self.process_func:
      log.info(f"Compiling PrefillProcessor._process for padded_length={padded_length}")
      
      # Define abstract shapes for the arguments of _process
      # params: Use self.engine.param_layouts if available and params is abstract,
      #         otherwise, JAX infers from the concrete `params` value.
      # tokens: Shape (padded_length,), dtype int32
      # slot: Scalar int, JAX treats as 0-dim array.
      # true_length: Scalar int.
      # decode_state: From self.engine.decode_state_shapes.

      abstract_params_shape = self.engine.abstract_params # Or derive from params if concrete
      abstract_tokens_shape = jax.ShapeDtypeStruct((padded_length,), jnp.int32)
      abstract_slot_shape = jax.ShapeDtypeStruct((), jnp.int32) # Python int becomes int32[()]
      abstract_true_length_shape = jax.ShapeDtypeStruct((), jnp.int32)
      # decode_state_shapes should be a pytree matching the decode_state structure
      abstract_decode_state_shape = self.engine.decode_state_shapes


      self.process_func[padded_length] = (
          jax.jit(
              self._process,
              # Shardings should match the JITted function's arguments
              in_shardings=(
                  self.engine.param_layouts,       # for params
                  None,                           # for tokens (typically replicated or data-parallel)
                  None,                           # for slot (scalar, replicated)
                  None,                           # for true_length (scalar, replicated)
                  self.engine.decode_state_layouts # for decode_state
              ),
              # Output shardings for (ResultTokens, DecodeState)
              # ResultTokens is usually small and replicated. DecodeState has its own layout.
              out_shardings=(None, self.engine.decode_state_layouts),
              donate_argnames=("decode_state",),
          )
          .lower( # Provide abstract representations for lowering
              abstract_params_shape, # Should be params passed to aot_compile or a representative abstract version
              abstract_tokens_shape,
              abstract_slot_shape,
              abstract_true_length_shape,
              abstract_decode_state_shape
          )
          .compile(compiler_options=None) # Add XLA flags if needed
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
    """
    JITted function to perform prefill and insert for a single request.
    This is the function that had the original error.
    """
    log.debug(f"PrefillProcessor._process JIT CALL: slot={slot}, true_length={true_length}, tokens_shape={tokens.shape}")
    
    # request_id can be None if not specifically tracked at this level by MaxEngine's JITted paths
    request_id_for_engine = None

    # Correctly call engine.prefill with slot as a keyword argument
    prefill_output_prefix, result_tokens = self.engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        slot=slot, # Pass the slot
        request_id=request_id_for_engine
    )
    
    # Correctly call engine.insert with slot as a keyword argument
    # The 'prefix' arg to engine.insert is the first element of engine.prefill's return tuple
    new_decode_state = self.engine.insert(
        prefix=prefill_output_prefix, # This is the 'Prefix' type for the KV cache
        decode_state=decode_state,
        slot=slot, # Pass the slot
        request_id=request_id_for_engine
    )
    
    return result_tokens, new_decode_state


class BatchedPrefillProcessor:
  """A wrapper around the APIs used by MaxEngine to do prefill and insert, provides prefill packing feature."""
  # ... (If this class is used, its calls to engine.prefill_concat and engine.insert_partial
  # also need to be checked for correctness regarding paged attention, though they don't directly take a single 'slot'
  # in the same way. `prefill_concat` does not take `slot` or `page_state`. `insert_partial` does not take `page_state`.)
  # For now, focusing on PrefillProcessor as it was in the error path and is simpler.
  # The key is that MaxEngine's own methods (`prefill`, `prefill_concat`, `insert`, `insert_partial`, `generate`)
  # must correctly handle `page_state` when paged attention is active.
  # The wrappers like PrefillProcessor must call these engine methods correctly.

# ... (rest of prefill_packing.py) ...
  def __init__(self, engine: MaxEngine, max_batch_size: int):
    self.engine = engine
    self.process_batch_func = {}
    self.buckets = {}
    self.max_batch_size = max_batch_size

  def aot_compile(
    self,
    params: Params,
    input_padding: int, # Max length of individual prompts in the packed batch
    capacity: int,      # Total capacity of the packed sequence
    num_prompts: int    # Number of prompts being packed
  ):
    """Ahead-of-time compile prefill processing routines."""
    # This will get or compile the JITted _process_batch function.
    self._process_batch_compiled(params, input_padding, capacity, num_prompts)

  def process(
    self,
    model_params: Params,
    decode_state: DecodeState,
    decode_slot: int, # Slot for this individual request before packing
    input_id: Any,    # ID of the input request
    input_prompt: jax.Array, # Unpadded prompt tokens
    input_padding: int, # Max length of individual items this bucket handles (from PrefillHelper)
    capacity: int,      # Max capacity of the packed sequence this bucket targets (from PrefillHelper)
    prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[Any], DecodeState], None]
  ) -> None:
    """Process a new input, potentially batching it for prefill."""
    length = len(input_prompt)
    if (length > capacity or length > input_padding): # input_padding is max item length for this bucket
      raise ValueError(f"Prefill length exceeds limit. Prefill length: {length}, Item padding for bucket: {input_padding}, Packed capacity: {capacity}")

    # Key for buckets should be based on what defines a compatible batch:
    # Here, (item_padding, total_capacity) could define a bucket type.
    bucket_key = (input_padding, capacity)
    bucket = self.buckets.setdefault(bucket_key, PrefillBucket(capacity))

    # If current prompt doesn't fit, process the current bucket first
    if not bucket.try_add(decode_slot, input_id, input_prompt): # try_add checks against bucket.capacity
        log.debug(f"BatchedPrefill: Bucket for key {bucket_key} is full or cannot fit item of length {len(input_prompt)}. Processing bucket.")
        # Note: _process_bucket uses input_padding as the `padded_length` for insert_partial's seq_len.
        prefill_results_list, decode_state = self._process_bucket(model_params, bucket, input_padding, decode_state)
        if prefill_done:
            prefill_done(prefill_results_list, bucket.row_ids, decode_state)
        bucket.clear()
        # Add the item that didn't fit to the now empty bucket
        if not bucket.try_add(decode_slot, input_id, input_prompt):
             # This should not happen if bucket was just cleared and item length is valid
             raise RuntimeError(f"Failed to add item to cleared bucket. Item length {len(input_prompt)}, Bucket capacity {bucket.capacity}")


    log.debug(f"BatchedPrefill: Added to bucket {bucket_key} - slot={decode_slot}, id={input_id}, length={length}. Bucket status: count={bucket.count}, filled_length={bucket.length}, unallocated={bucket.unallocated()}")


  def flush(
    self,
    model_params: Params,
    decode_state: DecodeState,
    prefill_done: Callable[[List[Tuple[engine_api.ResultTokens, int]], List[Any], DecodeState], None]
  ) -> None:
    """Process all remaining items in all buckets."""
    log.debug("BatchedPrefill: Flushing all buckets.")
    for bucket_key, bucket in self.buckets.items():
      if not bucket.is_empty():
        item_padded_length_for_bucket = bucket_key[0] # This was 'input_padding' for this bucket type
        log.debug(f"BatchedPrefill: Processing bucket for key {bucket_key} (item_pad_len={item_padded_length_for_bucket}). Count: {bucket.count}")
        prefill_results_list, decode_state = self._process_bucket(model_params, bucket, item_padded_length_for_bucket, decode_state)
        if prefill_done:
          prefill_done(prefill_results_list, bucket.row_ids, decode_state)
        bucket.clear()
    self.buckets.clear() # Clear all bucket structures

  def _process_bucket(
    self,
    model_params: Params,
    bucket: PrefillBucket,
    item_padded_length: int, # Max length of individual items in this bucket (used for insert_partial's seq_len)
    decode_state: DecodeState
  ) -> Tuple[List[Tuple[engine_api.ResultTokens, int]], DecodeState]:
    """Process all items in a single bucket using prefill_concat."""
    if bucket.is_empty():
        return [], decode_state

    slots_for_this_batch = bucket.slots
    true_lengths_for_this_batch = [len(prompt) for prompt in bucket.token_ids]
    # Offsets for packing tokens into a single sequence
    current_offsets = np.cumsum([0] + true_lengths_for_this_batch)[:-1].tolist()

    # Concatenate all tokens, adding padding at the end to fill bucket.capacity
    concatenated_tokens_list = list(bucket.token_ids) # Make a mutable copy
    if bucket.unallocated() > 0 :
        concatenated_tokens_list.append(jnp.zeros(bucket.unallocated(), dtype=jnp.int32))
    concatenated_tokens = jnp.concatenate(concatenated_tokens_list)

    # Create positions and segment IDs for the packed sequence
    positions_list = []
    for length in true_lengths_for_this_batch:
      positions_list.append(np.arange(length, dtype=np.int32))
    if bucket.unallocated() > 0:
      positions_list.append(np.arange(bucket.unallocated(), dtype=np.int32)) # Positions for padding
    concatenated_positions = jnp.concatenate(positions_list)

    segment_ids_list = []
    # Using 1-based, odd segment IDs as in original code example, or 0-based if preferred.
    # Ensure segment IDs correctly distinguish prompts.
    for i, length in enumerate(true_lengths_for_this_batch):
      segment_ids_list.append(np.full(length, i + 1, dtype=np.int32)) # Segments 1, 2, ... N
    if bucket.unallocated() > 0:
      segment_ids_list.append(np.zeros(bucket.unallocated(), dtype=np.int32)) # Padding segment 0
    concatenated_segment_ids = jnp.concatenate(segment_ids_list)


    # Pad batch metadata (slots, offsets, true_lengths) to self.max_batch_size for JIT
    def zero_pad_list_to_array(int_list: list[int], target_size: int, dtype=jnp.int32) -> jax.Array:
      padded_list = list(int_list) # Copy
      if len(padded_list) < target_size:
        padded_list.extend([0] * (target_size - len(padded_list)))
      elif len(padded_list) > target_size: # Should not happen if max_batch_size is enforced for bucket.count
          padded_list = padded_list[:target_size]
      return jnp.array(padded_list, dtype=dtype)

    # Number of actual prompts in this batch
    current_num_prompts = bucket.count

    padded_slots_for_jit = zero_pad_list_to_array(slots_for_this_batch, self.max_batch_size)
    padded_offsets_for_jit = zero_pad_list_to_array(current_offsets, self.max_batch_size)
    padded_true_lengths_for_jit = zero_pad_list_to_array(true_lengths_for_this_batch, self.max_batch_size)


    # Get the JITted function for processing this batch configuration
    # Note: 'item_padded_length' is the max length of an item this bucket was designed for.
    # This is used as 'padded_length' for insert_partial's seq_len.
    # 'bucket.capacity' is the total length of the concatenated sequence.
    compiled_batch_fn = self._process_batch_compiled(
        model_params,
        item_padded_length, # This is 'padded_length' for _process_batch static arg
        bucket.capacity,    # This is 'capacity' for creating abstract shapes
        current_num_prompts # This is 'num_prompts' for _process_batch static arg
    )

    # Call the JITted batch processing function
    # The `tokens` fed to `prefill_concat` should be `concatenated_tokens` (total length `bucket.capacity`)
    # `decoder_positions` and `decoder_segment_ids` also match this.
    # `start_pos` are the `padded_offsets_for_jit`.
    # `true_lengths` are `padded_true_lengths_for_jit`.
    # `num_prompts` is `current_num_prompts`.
    # `padded_length` for `insert_partial` (passed to `_process_batch`) is `item_padded_length`.
    
    # Ensure all JAX arrays are used
    list_of_result_tokens, new_decode_state = compiled_batch_fn(
        model_params,
        concatenated_tokens,       # tokens (shape: bucket.capacity)
        padded_slots_for_jit,      # slots (shape: max_batch_size)
        # num_prompts is static: current_num_prompts
        concatenated_positions,    # decoder_positions (shape: bucket.capacity)
        concatenated_segment_ids,  # decoder_segment_ids (shape: bucket.capacity)
        padded_offsets_for_jit,    # start_pos (shape: max_batch_size) - these are offsets into the concat sequence
        # padded_length is static: item_padded_length
        padded_true_lengths_for_jit, # true_lengths of individual prompts (shape: max_batch_size)
        decode_state,
        # Static arguments passed implicitly via .lower() and partial eval:
        # num_prompts=current_num_prompts,
        # padded_length=item_padded_length
    )

    # Structure the results for the callback
    # list_of_result_tokens is expected to be a list of ResultTokens, one for each of the current_num_prompts
    final_prefill_results = []
    for i in range(current_num_prompts):
      # ResultTokens for i-th prompt, to be inserted into its original slot (bucket.slots[i])
      final_prefill_results.append((list_of_result_tokens[i], bucket.slots[i]))
      
    return final_prefill_results, new_decode_state


  def _process_batch_compiled(
    self,
    params: Params,      # Abstract params for shape inference
    item_padded_length: int, # Max length of individual items (static for JIT) -> 'padded_length' for _process_batch
    capacity: int,       # Total capacity of packed sequence (for abstract shapes)
    num_prompts: int     # Number of prompts in the batch (static for JIT)
  ) -> Callable[..., Tuple[List[engine_api.ResultTokens], DecodeState]]:
    """
    Gets or compiles the JITted _process_batch function.
    'item_padded_length' becomes the static 'padded_length' arg for _process_batch.
    'num_prompts' is also a static arg for _process_batch.
    """
    compile_key = (item_padded_length, num_prompts, capacity) # Add capacity to key if it changes abstract shapes

    if compile_key not in self.process_batch_func:
      log.info(f"Compiling BatchedPrefillProcessor._process_batch for item_pad_len={item_padded_length}, num_prompts={num_prompts}, capacity={capacity}")

      # Abstract shapes for .lower()
      abstract_params_shape = self.engine.abstract_params # Or derive from params
      # Tokens for prefill_concat (total capacity)
      abstract_concat_tokens_shape = jax.ShapeDtypeStruct((capacity,), jnp.int32)
      # Batch metadata arrays (padded to max_batch_size)
      abstract_batch_meta_shape = jax.ShapeDtypeStruct((self.max_batch_size,), jnp.int32)
      # Decoder positions/segments (total capacity)
      abstract_concat_decode_aux_shape = jax.ShapeDtypeStruct((capacity,), jnp.int32)
      abstract_decode_state_shape = self.engine.decode_state_shapes
      
      # Static arguments for jax.jit
      static_argnames_for_jit = ("num_prompts_static", "padded_length_static")


      # Wrapper to match JIT signature if needed, or pass static args directly
      # to jax.jit if the function signature matches.
      # _process_batch takes (params, tokens, slots, num_prompts_static, ..., decode_state)
      # We need to ensure the call to .lower() matches this.

      partial_fn = functools.partial(self._process_batch,
                                     num_prompts_static=num_prompts,
                                     padded_length_static=item_padded_length)


      self.process_batch_func[compile_key] = (
          jax.jit(
              partial_fn, # JIT the partially evaluated function
              # Static arguments are now bound in partial_fn
              # in_shardings needs to match the non-static args of partial_fn
              in_shardings=(
                  self.engine.param_layouts, # params
                  None,                       # concatenated_tokens
                  None,                       # padded_slots_for_jit
                  None,                       # concatenated_positions
                  None,                       # concatenated_segment_ids
                  None,                       # padded_offsets_for_jit
                  None,                       # padded_true_lengths_for_jit
                  self.engine.decode_state_layouts # decode_state
              ),
              # Output: List[ResultTokens] (usually replicated), DecodeState
              out_shardings=(None, self.engine.decode_state_layouts),
              donate_argnames=("decode_state",),
          )
          .lower( # Provide abstract shapes for the non-static arguments of partial_fn
              abstract_params_shape,
              abstract_concat_tokens_shape,    # for concatenated_tokens
              abstract_batch_meta_shape,       # for padded_slots_for_jit
              abstract_concat_decode_aux_shape,# for concatenated_positions
              abstract_concat_decode_aux_shape,# for concatenated_segment_ids
              abstract_batch_meta_shape,       # for padded_offsets_for_jit
              abstract_batch_meta_shape,       # for padded_true_lengths_for_jit
              abstract_decode_state_shape
          )
          .compile(compiler_options=None)
      )
    return self.process_batch_func[compile_key]

  def _process_batch( # pylint: disable=too-many-arguments # Expected for this internal JITted fn
      self,
      # Dynamic arguments:
      params: Params,
      concatenated_tokens: jax.Array,    # Shape: (capacity,)
      padded_slots: jax.Array,           # Shape: (max_batch_size,) - original slots for these prompts
      concatenated_positions: jax.Array, # Shape: (capacity,)
      concatenated_segment_ids: jax.Array,# Shape: (capacity,)
      padded_offsets: jax.Array,         # Shape: (max_batch_size,) - offsets into concatenated_tokens
      padded_true_lengths: jax.Array,    # Shape: (max_batch_size,) - true lengths of individual prompts
      decode_state: DecodeState,
      # Static arguments (bound by functools.partial before JIT):
      num_prompts_static: int,
      padded_length_static: int # This is item_padded_length, used for insert_partial's seq_len
  ) -> Tuple[List[engine_api.ResultTokens], DecodeState]:
    """
    JITted function to perform packed prefill and insert.
    """
    log.debug(
        f"BatchedPrefillProcessor._process_batch JIT CALL: "
        f"num_prompts_static={num_prompts_static}, item_padded_length_static={padded_length_static}, "
        f"concat_tokens_shape={concatenated_tokens.shape}"
    )

    # Call engine.prefill_concat
    # It expects num_prompts to be passed dynamically if it's not static in prefill_concat itself.
    # MaxEngine.prefill_concat has num_prompts as a static arg in its JIT decorator.
    # So, the num_prompts_static here must match how prefill_concat is JITted or called.
    
    # Assuming MaxEngine.prefill_concat is jitted with num_prompts as a static argument
    # We need to ensure that the compilation of prefill_concat matches this num_prompts_static.
    # This implies prefill_concat might need to be compiled for each num_prompts value.
    # For now, assume MaxEngine.prefill_concat handles this.
    
    # The 'params' for prefill_concat
    # 'padded_tokens' is 'concatenated_tokens'
    # 'decoder_positions' is 'concatenated_positions'
    # 'decoder_segment_ids' is 'concatenated_segment_ids'
    # 'start_pos' is 'padded_offsets' (these are the starts of each prompt in the concat sequence)
    # 'true_lengths' is 'padded_true_lengths' (lengths of each prompt)
    # 'num_prompts' is 'num_prompts_static'

    # If engine.prefill_concat itself is not JITted with num_prompts as static,
    # but expects it as a dynamic Python int that then becomes static for its *internal* model call,
    # then this is fine. Let's check MaxEngine.prefill_concat:
    # @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_prompts",))
    # `self` is arg 0. `num_prompts` is static. This means when we *call* prefill_concat,
    # num_prompts must be a compile-time constant for that call.
    # This is achieved here because `num_prompts_static` is static for `_process_batch`.

    packed_cache, packed_prefix_state, list_of_first_tokens = self.engine.prefill_concat(
        params=params,
        padded_tokens=concatenated_tokens,
        decoder_positions=concatenated_positions,
        decoder_segment_ids=concatenated_segment_ids,
        start_pos=padded_offsets, # start_pos for each prompt in the concatenated sequence
        true_lengths=padded_true_lengths, # true_lengths of each prompt
        num_prompts=num_prompts_static # Must be static for the JITted prefill_concat
    )

    # Call engine.insert_partial
    # `prefix` is `packed_prefix_state`
    # `cache` is `packed_cache`
    # `slots` are `padded_slots` (original slots to insert into)
    # `start_indices` are `padded_offsets` (where each prompt started in the packed cache)
    # `num_prompts` is `num_prompts_static`
    # `seq_len` for insert_partial is `padded_length_static` (max len of individual item in this bucket)
    # This is also static for engine.insert_partial's JIT.

    new_decode_state = self.engine.insert_partial(
        prefix=packed_prefix_state,
        decode_state=decode_state,
        cache=packed_cache,
        slots=padded_slots, # Original slots where these KVs should go
        start_indices=padded_offsets, # Where each prompt's cache data starts in packed_cache
        num_prompts=num_prompts_static, # Static for insert_partial's JIT
        seq_len=padded_length_static   # Static for insert_partial's JIT
    )

    return list_of_first_tokens, new_decode_state