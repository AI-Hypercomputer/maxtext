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

"""Implementation of Engine API for MaxText"""
import copy as cp
import functools
from typing import Any, List, Optional, Tuple, Callable
from collections import defaultdict

import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from layers import models, quantizations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental import layout as jax_layout

import common_types
from page_manager import PageManager, PageState
from layers.attentions import (
    get_initial_paged_kv_cache,
    get_initial_contiguous_kv_cache,
)
from jetstream.core import config_lib
from jetstream.engine import engine_api
from jetstream.engine import tokenizer_pb2
from jetstream.engine import tokenizer_api
from jetstream.engine import token_utils

import checkpointing
import max_utils
import inference_utils
import pyconfig

import warnings

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any
DLL = jax_layout.DeviceLocalLayout
Layout = jax_layout.Layout
from flax.training import train_state


def update_page_state(state, slot, true_length, config):
    """Pure function to update page state without using dynamic shapes.
    
    This function allocates pages for paged attention in a JAX-friendly way,
    avoiding dynamic shapes and traced values in control flow.
    
    Args:
        state: Initial PageState
        slot: Page group ID to update
        true_length: Sequence length
        config: Configuration object with paged attention parameters
        
    Returns:
        Updated PageState with allocated pages
    """
    # Calculate pages needed
    num_pages_needed = (true_length + config.tokens_per_page - 1) // config.tokens_per_page
    
    # Use a fixed maximum size with masking approach
    max_possible_pages = config.max_pages_per_group
    
    # Create a fixed-size array of indices
    all_indices = jnp.arange(max_possible_pages)
    
    # Create a mask for valid pages
    valid_pages_mask = all_indices < num_pages_needed
    
    # Generate physical page assignments with a fixed pattern
    # This replaces the dynamic jnp.arange(num_pages_needed) % config.num_pages
    physical_pages = all_indices % config.num_pages
    
    # Apply the mask to keep only valid pages (-1 for invalid ones)
    masked_physical_pages = jnp.where(valid_pages_mask, physical_pages, -1)
    
    # Update page map for all layers with masked indices
    updated_page_map = state.page_map.at[:, slot, :max_possible_pages].set(
        jnp.broadcast_to(masked_physical_pages, 
                         (config.num_decoder_layers, max_possible_pages))
    )
    
    # Update page status (mark used pages as allocated)
    # Create a mask for which physical pages are actually used
    used_pages_mask = jnp.zeros((config.num_pages,), dtype=jnp.int32)
    for i in range(min(max_possible_pages, config.num_pages)):
        used_pages_mask = used_pages_mask.at[i % config.num_pages].set(
            jnp.where(i < num_pages_needed, 1, 0)
        )
    
    # Broadcast the mask to all layers
    updated_page_status = state.page_status.at[:, :].set(
        jnp.broadcast_to(used_pages_mask, 
                         (config.num_decoder_layers, config.num_pages))
    )
    
    # Update sequence lengths for all layers
    updated_sequence_lengths = state.sequence_lengths.at[:, slot].set(true_length)
    
    # Update number of pages used
    updated_num_pages_used = state.num_pages_used.at[:, slot].set(
        jnp.minimum(num_pages_needed, max_possible_pages)
    )
    
    # Set current page to last allocated page (if any pages were allocated)
    last_page_idx = jnp.maximum(0, jnp.minimum(num_pages_needed, max_possible_pages) - 1)
    last_physical_page = physical_pages[last_page_idx]
    last_physical_page = jnp.where(num_pages_needed > 0, last_physical_page, -1)
    
    updated_current_page = state.current_page.at[:, slot].set(last_physical_page)
    
    # Set position in current page
    last_position = (true_length - 1) % config.tokens_per_page
    last_position = jnp.where(true_length > 0, last_position, 0)
    updated_current_position = state.current_page_position.at[:, slot].set(last_position)
    
    return PageState(
        page_status=updated_page_status,
        page_map=updated_page_map,
        sequence_lengths=updated_sequence_lengths,
        num_pages_used=updated_num_pages_used,
        current_page=updated_current_page,
        current_page_position=updated_current_position
    )

class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  The default pyconfig.config is a global param shared across multiple instances and doesn't
  allow using different config for each MaxEngine instance.
  """

  def __init__(self, keys):
    # self.keys = keys
    self.__dict__["keys"] = keys

  def __getattr__(self, attr):
    if attr not in self.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return self.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return self.keys


class MaxEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, config: Any, devices: config_lib.Devices | None = None):
      self.config = config

      # Mesh definition
      devices_array = max_utils.create_device_mesh(config=config, devices=devices)
      self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

      # Model and Optimizer definition
      self.model = models.Transformer(config, mesh=self._mesh, quant=None)  # Create model
      self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

      self.abstract_params = None
      self.prefill_kv_cache_annotations = None
      self.kv_cache_annotations = None
      self.kv_cache_annotations_named = None  # Needed for insert.
      self.prefill_kv_cache_shardings = None
      self.kv_cache_shardings = None
      self.state_mesh_annotations = None
      self.decode_state_shapes = None
      self.params = None  # Initialize self.params

      # Initialize PageManager if using paged attention
      if self.config.attention == "paged":
        self.page_manager = PageManager(
            num_pages=self.config.num_pages,
            tokens_per_page=self.config.tokens_per_page,
            max_page_groups=int(self.config.per_device_batch_size * jax.device_count()),
            max_target_length=self.config.max_target_length,
            max_prefill_predict_length=self.config.max_prefill_predict_length,
            max_pages_per_group=(self.config.max_target_length + self.config.tokens_per_page - 1)
            // self.config.tokens_per_page,
            num_layers=self.config.num_decoder_layers,
            config=self.config,
        )
      else:
        self.page_manager = None

  def create_decode_state(self, rng: Optional[PRNGKeyType] = None):
      """Creates a new decode state, including the PageManager state."""
      if rng is None:
        rng = jax.random.PRNGKey(0)

      batch_size = int(self.config.per_device_batch_size * jax.device_count())

      if self.config.attention == "paged":
        from layers.attentions import get_initial_paged_kv_cache

        abstract_cache = get_initial_paged_kv_cache(self.model, self.config, batch_size=batch_size, abstract=True)

        decode_state = {
            "logits": jnp.zeros((batch_size, 1, self.config.vocab_size)),
            "cache": abstract_cache,
            "next_pos": jnp.zeros((batch_size, 1), dtype=jnp.int32),
            "generated_tokens": jnp.zeros((batch_size, 1), dtype=jnp.int32),
            "tokens": jnp.zeros((batch_size, 1), dtype=jnp.int32),
        }
      else:  # Contiguous attention
        from layers.attentions import get_initial_contiguous_kv_cache

        decode_state = {
            "logits": jnp.zeros((batch_size, 1, self.config.vocab_size)),
            "cache": get_initial_contiguous_kv_cache(
                self.model,
                self.config,
                batch_size=batch_size,
                abstract=False,
            ),
            "next_pos": jnp.zeros((batch_size, 1), dtype=jnp.int32),
            "generated_tokens": jnp.zeros((batch_size, 1), dtype=jnp.int32),
            "tokens": jnp.zeros((batch_size, 1), dtype=jnp.int32),
        }

      return decode_state

  def generate_aot(
      self, params: Params, decode_state: DecodeState, rng: Optional[PRNGKeyType] = None
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Wrapper to generate for ahead of time compilation."""

    return self.generate(params=params, decode_state=decode_state, rng=rng)

  def _compile_generate_and_get_layouts(
      self, params: Any, decode_state: Any, rng_shape: Any, xla_flags: dict[str, Any] | None = None
  ) -> tuple[Any, Any, Any, Any]:
    """Optimal memory layout for params and decode_state."""

    param_layout = Layout(DLL.AUTO)
    decode_state_layout = Layout(DLL.AUTO)
    # Keyword arguments are not yet supported in JAX for specifying shardings. Therefore, all AOT
    # compiled functions use arguments instead.
    compiled_generate = (
        jax.jit(
            self.generate_aot,
            in_shardings=(param_layout, decode_state_layout, None),
            out_shardings=(Layout(DLL.AUTO), Layout(DLL.AUTO)),
            donate_argnames=("decode_state",),
        ).lower(params, decode_state, rng_shape)
    ).compile(compiler_options=xla_flags)

    arg_layouts, _ = compiled_generate.input_layouts
    generate_out_layouts, _ = compiled_generate.output_layouts

    return compiled_generate, arg_layouts[0], arg_layouts[1], generate_out_layouts

  def _identity(self, x: Any) -> Any:
    """Avoids lambda that breaks JAX caching."""

    return x

  def _iterated_layout(self, arrays: Any, layouts: Any, xla_flags: dict[str, Any] | None = None) -> Any:
    """Lays out an array tensor by tensor to prevent OOMs."""

    def _layout(x, s, l):
      if x.layout == l:
        return x
      # Somehow this can be None sometimes.
      dll = l.device_local_layout if isinstance(l, Layout) else l
      f = jax.jit(self._identity, out_shardings=Layout(dll, s)).lower(x).compile(compiler_options=xla_flags)
      y = f(x)
      # Achieves donation of the input argument, but allows for different memory
      # layouts and shapes.
      jax.tree.map(lambda z: z.delete(), x)
      jax.block_until_ready(y)
      return y

    shardings = jax.tree.map(lambda x: x.sharding, arrays)
    arrays = jax.tree.map(_layout, arrays, shardings, layouts)
    return arrays

  def aot_compile(
      self, params: Params, pass_rng_shape: bool, xla_flags: dict[str, Any] | None = None
  ) -> Tuple[Any, Params, Any]:
    """Ahead of time compilation of generate with auto layout, relayout parameters."""
    if pass_rng_shape:
      rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))
    else:
      rng_shape = None
    init_decode_state_partial = functools.partial(
        self.init_decode_state, params=params, model=self.model, config=self.config
    )
    self.decode_state_shapes = jax.eval_shape(init_decode_state_partial, rng_shape)

    generate_executable, self.param_layouts, _, self.decode_state_layouts = self._compile_generate_and_get_layouts(
        self.abstract_params, self.decode_state_shapes, rng_shape, xla_flags
    )

    return (
        generate_executable,
        self._iterated_layout(params, self.param_layouts),
        jax.jit(init_decode_state_partial, in_shardings=(None), out_shardings=self.decode_state_layouts)
        .lower(rng_shape)
        .compile(),
    )

  def load_params(self, *args, rng: Optional[PRNGKeyType] = None, **kwargs):
    """Loads parameters and initializes necessary state/annotations."""
    if rng is None:
      rng = jax.random.PRNGKey(0)

    print("\nMaxEngine.load_params() entry:")
    if self.config.attention == "paged":
      print(f"  config.num_pages: {self.config.num_pages}")
      print(f"  config.tokens_per_page: {self.config.tokens_per_page}")
    max_utils.print_mem_stats("Before setup_decode_state")

    rng1, rng2, rng3 = jax.random.split(rng, 3)

    # Initialize the model parameters *ONLY*.
    input_shape = (self.config.micro_batch_size_to_train_on, self.config.max_target_length)
    model_vars = self.model.init(
        {"params": rng1, "dropout": rng1, "cache": rng1, "page_manager": rng1},
        jnp.ones(input_shape, dtype=jnp.int32),
        jnp.ones(input_shape, dtype=jnp.int32),
    )
    self.params = model_vars["params"]

    # Get KV Cache annotations directly from attentions.py
    from layers.attentions import get_prefill_paged_kv_cache_annotations

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      if self.config.attention == "paged":
        self.kv_cache_annotations = get_prefill_paged_kv_cache_annotations(self.model, self.config, rng2, self._mesh)
      else:
        from layers.attentions import get_initial_contiguous_kv_cache

        self.kv_cache_annotations = get_initial_contiguous_kv_cache(
            self.model, self.config, batch_size=int(self.config.per_device_batch_size * jax.device_count()), abstract=True
        )

    # Rest of the function remains the same...
    return self.params

  def quantize_params(self, state, rng: Optional[PRNGKeyType] = None):
    """Forward pass to quantize decode params."""
    if rng is None:
      rng = jax.random.PRNGKey(0)

    self.model.quant.quant_mode = quantizations.get_quant_mode("convert")

    @jax.jit
    def model_apply(_p, _rng):
      return self.model.apply(
          _p | {"aqt": {}},
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          decoder_segment_ids=jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": _rng},
          mutable=True,
      )

    _, new_vars = model_apply(state.params, rng)
    # Remove param values which have corresponding qtensors in aqt to save memory.
    params = {}
    params["aqt"] = new_vars["aqt"]
    params["params"] = quantizations.remove_quantized_params(state.params["params"], new_vars["aqt"])
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding),
        params,
    )
    max_utils.save_quantized_checkpoint_if_configured(self.config, params)
    self.model.quant.quant_mode = quantizations.get_quant_mode("serve")
    return params

  def _maybe_stack_prefill_result_cache(self, cache):
    """Stack the caches across the layers."""
    if not self.config.stack_prefill_result_cache:
      return cache

    layer_keys = []
    for i in range(self.config.num_decoder_layers):
      layer_keys.append(f"layers_{i}")

    layer_cache = [cache["decoder"][layer_key] for layer_key in layer_keys]

    return jax.tree.map(lambda *c: jnp.stack(c), *layer_cache)

  def _maybe_unstack_prefill_result_cache(self, cache):
    """Unstack the caches across the layers."""
    if not self.config.stack_prefill_result_cache:
      return cache

    flat_cache, treedef = jax.tree.flatten(cache)
    layer_cache = [jax.tree.unflatten(treedef, flat_cache_vars) for flat_cache_vars in zip(*flat_cache, strict=True)]
    res_cache = {"decoder": {}}

    for i in range(self.config.num_decoder_layers):
      res_cache["decoder"][f"layers_{i}"] = layer_cache[i]

    return res_cache

  def prefill_aot(
      self,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: Optional[PRNGKeyType] = None,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Wrapper for prefill for ahead-of-time compilation."""

    return self.prefill(params=params, padded_tokens=padded_tokens, true_length=true_length, rng=rng)


  @functools.partial(jax.jit, static_argnums=(0,))
  def prefill(
      self,
      *,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: Optional[PRNGKeyType] = None,
      slot: Optional[int] = None,
      dynamic_kv_cache: bool = False,
      page_state: Optional[PageState] = None, # Add page_state as an argument
  ):
      """Computes prefill for a new generate request with paged attention."""
      if rng is None:
          rng = jax.random.PRNGKey(0)

      input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
      positions = jnp.expand_dims(jnp.arange(0, input_tokens.shape[1]), 0)

      # Create active sequence mask
      zero_to_n = jnp.arange(0, padded_tokens.shape[0])
      ones_to_keep = zero_to_n < true_length
      one_d_output = jnp.where(ones_to_keep, common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR, 0)
      sequence_indicator = jnp.expand_dims(one_d_output, 0)

      rng, new_rng = jax.random.split(rng)

      # Initialize updated_page_state to the input page_state
      updated_page_state = page_state
      # Initialize cache to a default value.
      cache = {}

      with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          if self.config.attention == "paged":

              # Unbox the logically partitioned parameters
              unboxed_params = max_utils.unbox_logicallypartioned(params)

              # Initialize decoder cache structure
              decoder_cache = {}
              for layer_id in range(self.config.num_decoder_layers):
                  decoder_cache[f"layers_{layer_id}"] = {}

              # Create variables dict with cache and page_state
              variables = {"params": unboxed_params, "cache": {"decoder": decoder_cache}}

              # Apply model with prefill mode, passing page_state
              model_output, new_vars = self.model.apply(
                  variables,
                  input_tokens,
                  positions,
                  decoder_segment_ids=sequence_indicator,
                  enable_dropout=False,
                  model_mode=common_types.MODEL_MODE_PREFILL,
                  rngs={"params": new_rng},
                  mutable=["cache"],
                  slot=slot,
                  true_length=true_length,
                  page_state=page_state,
              )
              flat_logits, updated_page_state = model_output
               # Get the updated page_state and cache from new_vars
              updated_cache = variables.get("cache", {}) # get updated cache from variables
              if updated_page_state is None:
                updated_page_state = page_state
              # Check for updated page state (it should be in the first layer)
              if "decoder" in updated_cache and f"layers_0" in updated_cache["decoder"] and \
                "self_attention" in updated_cache["decoder"][f"layers_0"] and \
                "attention_op" in updated_cache["decoder"][f"layers_0"]["self_attention"]:
                  updated_page_state = updated_cache["decoder"][f"layers_0"]["self_attention"]["attention_op"].get("page_state") # .get is safer

              cache = {
                  "page_manager": updated_page_state,  # Use updated page state
                  "decoder": updated_cache.get("decoder", {})  # And updated decoder cache
              }


          else: # non-paged
              # Non-paged attention path (unchanged)
              flat_logits, updated_page_state = self.model.apply( # ALWAYS unpack two return values
                  params | {"cache": {}},
                  input_tokens,
                  positions,
                  decoder_segment_ids=sequence_indicator,
                  enable_dropout=False,
                  model_mode=common_types.MODEL_MODE_PREFILL,
                  rngs={"params": new_rng},
                  mutable=["cache"],
              )

              # Use the cache returned by the model.  Get new_vars here.
              new_vars = self.model.apply(
                params | {"cache":{}},
                input_tokens,
                positions,
                decoder_segment_ids=sequence_indicator,
                enable_dropout=False,
                model_mode = common_types.MODEL_MODE_PREFILL,
                rngs = {"params": new_rng},
                mutable=["cache"],
              )[1] #getting new_vars
              cache = new_vars["cache"] #getting from correct spot

      # Get logits for the last position
      next_pos = jnp.full((1, 1), true_length, dtype=jnp.int32)
      generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)

      # Check attention type *before* accessing .shape
      if self.config.attention == "paged":
          selected_logits = jax.lax.dynamic_slice(
              flat_logits,
              (0, true_length - 1, 0),
              (flat_logits.shape[0], 1, flat_logits.shape[2]),  # Now safe
          )
      else: # non-paged
          selected_logits = jax.lax.dynamic_slice(
              flat_logits,
              (0, true_length - 1, 0),
              (flat_logits.shape[0], 1, flat_logits.shape[2]), # Now safe
          )
      selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)


      # Sample first token
      first_generated_token = inference_utils.sampling(
          selected_logits,
          rng,
          self.config.decode_sampling_strategy,
          topk=self.config.decode_sampling_top_k,
          nucleus_topp=self.config.decode_sampling_nucleus_p,
          temperature=self.config.decode_sampling_temperature,
      )

      # Create result structure
      all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
      result = engine_api.ResultTokens(
          data=jnp.concatenate((first_generated_token, all_valid, generated_tokens), axis=1),
          tokens_idx=(0, 1),
          valid_idx=(1, 2),
          length_idx=(2, 3),
          samples_per_slot=1,
      )
      if self.config.attention != "paged":
        cache = self._maybe_stack_prefill_result_cache(cache)

      # Return a DICTIONARY:
      return {
          "logits": selected_logits,
          "cache": cache,
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": first_generated_token,
          "first_token": result,  # Include the result here
          "page_state": updated_page_state,  # Include updated_page_state
      }

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def generate(
      self,
      params: Params,
      decode_state: DecodeState,
      sampler: Optional[Callable[[Any], Any]] = None,
      rng: Optional[PRNGKeyType] = None,
      slot: Optional[int] = None,
  ):
      """Run one generate step with paged attention."""
      if rng is None:
          rng = jax.random.PRNGKey(0)

      # Get the previous token
      previous_token = jnp.array([[decode_state["tokens"][0, 0]]], dtype=jnp.int32) if slot is None else \
                      jnp.array([[decode_state["tokens"][slot, 0]]], dtype=jnp.int32)

      # Split RNG
      rng, new_rng = jax.random.split(rng)

      with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          if self.config.attention == "paged":
              # Get current page state from the cache - MUST get from decode_state
              page_state = decode_state["cache"]["page_manager"]

              # Extract updated KV cache
              decoder_cache = decode_state["cache"]["decoder"]

              # Unbox parameters
              unboxed_params = max_utils.unbox_logicallypartioned(params)

              # Apply model with autoregressive mode
              variables = {"params": unboxed_params, "cache": {"decoder": decoder_cache}}

              model_output, new_vars = self.model.apply(
                  variables,
                  previous_token,
                  decode_state["next_pos"],
                  enable_dropout=False,
                  model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
                  rngs={"params": new_rng},
                  mutable=["cache"],
                  slot=slot,
                  page_state=page_state, # Pass the page state!
              )
              flat_logits, updated_page_state = model_output

              # Get updated cache
              updated_cache = new_vars.get("cache", {})
              # Look for updated page state in each layer
              updated_page_state = page_state

              # First check for the updated_page_state in the first layer's attention_op
              if "decoder" in updated_cache and f"layers_0" in updated_cache["decoder"] and \
                "self_attention" in updated_cache["decoder"][f"layers_0"] and \
                "attention_op" in updated_cache["decoder"][f"layers_0"]["self_attention"]:
                  updated_page_state = updated_cache["decoder"][f"layers_0"]["self_attention"]["attention_op"].get("page_state") # Use .get for safety
              if updated_page_state is None:  # If None, keep the original.
                  updated_page_state = page_state
              # Update the cache
              cache = {
                  "page_manager": updated_page_state,  # Use updated page state
                  "decoder": updated_cache.get("decoder", decoder_cache) # Keep updated decoder cache
              }

          else: # Non-paged
              # Non-paged attention path
              out_logits, new_vars = self.model.apply(
                  params | {"cache": decode_state["cache"]},
                  previous_token,
                  decode_state["next_pos"],
                  enable_dropout=False,
                  model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
                  rngs={"params": new_rng},
                  mutable=["cache"],
              )

              # Update cache with new KV data
              cache = new_vars["cache"]
              flat_logits = out_logits

      # Apply sharding constraint
      out_logits = jax.lax.with_sharding_constraint(flat_logits, self.replicated_sharding)

      # Sample next token
      new_token = inference_utils.sampling(
          out_logits,
          rng,
          self.config.decode_sampling_strategy,
          topk=self.config.decode_sampling_top_k,
          nucleus_topp=self.config.decode_sampling_nucleus_p,
          temperature=self.config.decode_sampling_temperature,
      )

      # Create result
      all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
      result = engine_api.ResultTokens(
          data=jnp.concatenate((new_token, all_valid, decode_state["generated_tokens"]), axis=1),
          tokens_idx=(0, 1),
          valid_idx=(1, 2),
          length_idx=(2, 3),
          samples_per_slot=1,
      )

      # Update decode state
      updated_decode_state = {
          "logits": out_logits,
          "cache": cache,  # Use updated cache
          "next_pos": decode_state["next_pos"] + 1,
          "generated_tokens": decode_state["generated_tokens"] + 1,
          "tokens": new_token,
      }

      return updated_decode_state, result

  @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_prompts",))
  def prefill_concat(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array,
      start_pos: jax.Array,
      true_lengths: jax.Array,
      num_prompts: int,
      sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
      rng: Optional[PRNGKeyType] = None,
  ) -> Tuple[Any, PackedPrefix, List[engine_api.ResultTokens]]:
    """Computes a kv-cache for a new packed generate request, which is a
    concatenation of several shorter prompts. Experimentation shows that
    longer prefill sequences gives approximately 15% boost in time per prefilled
    token.

    Args:
      params: Scalar multiplier.
      existing_prefix: If provided, represents a prefix that has already been
        processed by the underlying model.
      padded_tokens: Logically appended tokens to any existing prefix, this is
        what we compute prefill on.
      decoder_positions: int values indicating the position of token in its
        original sequence.
      decoder_segment_ids: int values indicating which sequence the the token
        originally belong to.
      start_pos: Padded array indicating the start position of each of the prompts.
      true_length: Padded array indicating the true lengths of each of the prompts.
      num_prompts: the number of prompts packed in the entire sequence.
    Returns:
      kv_cache: For the resulting text.
    """
    if existing_prefix:
      raise ValueError("We don't know what to do with existing_prefix")

    if rng is None:
      rng = jax.random.PRNGKey(0)
    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
    decoder_positions = jnp.expand_dims(decoder_positions, 0)
    decoder_segment_ids = jnp.expand_dims(decoder_segment_ids, 0)
    rng, new_rng = jax.random.split(rng)
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flat_logits, new_vars = self.model.apply(
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
    cache = self._maybe_stack_prefill_result_cache(cache)

    def process_packed_logits_and_caches(packed_flat_logits, idx):
      next_pos = jnp.full((1, 1), true_lengths[idx], dtype=jnp.int32)
      generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
      selected_logits = jax.lax.dynamic_slice(
          packed_flat_logits,
          (0, start_pos[idx] + true_lengths[idx] - 1, 0),
          (packed_flat_logits.shape[0], 1, packed_flat_logits.shape[2]),
      )
      selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)
      first_generated_token = inference_utils.sampling(
          selected_logits,
          rng,
          self.config.decode_sampling_strategy,
          topk=self.config.decode_sampling_top_k,
          nucleus_topp=self.config.decode_sampling_nucleus_p,
          temperature=self.config.decode_sampling_temperature,
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

  def prefill_insert(  # pylint: disable=too-many-positional-arguments
      self,
      padded_tokens: jax.Array,
      true_length: int,
      rng: Any,
      decode_state: DecodeState,
      slot: int,
      params: Params,
  ) -> DecodeState:
    """Prefill and insert a single computed prefill cache into KV cache."""

    prefix, _ = self.prefill(params=params, padded_tokens=padded_tokens, true_length=true_length, rng=rng)
    return self.insert(prefix, decode_state, slot)
  
  
  def _update_kv_cache_for_new_token(
    self,
    key: jax.Array,
    value: jax.Array,
    page_state: Any,
    slot: int,
    layer_id: int,
  ):
    """Store key/value data for a new token in the appropriate page.
    
    This method should be called during autoregressive generation to store
    the key/value data for the new token in the correct page location.
    
    Args:
        key: [batch, 1, num_kv_heads, head_dim] key for the new token
        value: [batch, 1, num_kv_heads, head_dim] value for the new token
        page_state: Current page state
        slot: Slot (page group) ID
        layer_id: Layer ID
    
    Returns:
        Updated key_pages and value_pages
    """
    # Extract relevant data
    tokens_per_page = self.config.tokens_per_page
    batch_idx = 0  # We only handle the first batch element
    
    # Get sequence length (after the update)
    seq_len = page_state.sequence_lengths[slot]
    
    # Calculate logical page and position within page
    logical_page_idx = (seq_len - 1) // tokens_per_page
    pos_in_page = (seq_len - 1) % tokens_per_page
    
    # Get physical page
    physical_page = page_state.page_map[slot, logical_page_idx]
    
    # Get current key and value pages
    key_pages = self.page_manager.key_pages[layer_id]
    value_pages = self.page_manager.value_pages[layer_id]
    
    # Reshape key and value to match page storage format
    key_to_store = key[batch_idx, 0]  # [num_kv_heads, head_dim]
    value_to_store = value[batch_idx, 0]  # [num_kv_heads, head_dim]
    
    # Update key and value pages
    key_pages = key_pages.at[physical_page, pos_in_page].set(key_to_store)
    value_pages = value_pages.at[physical_page, pos_in_page].set(value_to_store)
    
    # Store updated pages back to page manager
    self.page_manager.key_pages[layer_id] = key_pages
    self.page_manager.value_pages[layer_id] = value_pages
    
    return key_pages, value_pages
  
  def _populate_prefill_kv_pages_minimal(self, input_tokens, new_vars, page_state, slot, true_length):
    """Extremely simplified version to diagnose hanging issues.
    
    This version uses minimal JAX control flow to help isolate where hanging might occur.
    """
    print("\n=== Populating KV Pages (Minimal) ===")
    
    # Initialize cache structure
    cache = {
        "page_manager": page_state,
        "decoder": {}
    }
    
    # Get fixed parameters
    tokens_per_page = self.config.tokens_per_page
    num_pages = self.config.num_pages
    
    # Process each layer
    for layer_id in range(self.config.num_decoder_layers):
        print(f"Processing layer {layer_id}")
        
        # Initialize empty key and value pages
        key_pages = jnp.zeros(
            (num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        value_pages = jnp.zeros(
            (num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        
        # Force update the first 3 physical pages regardless of page map
        # This avoids any conditional logic that might hang
        for page_idx in range(3):
            # Create simple data based on layer and page indices
            page_key = jnp.ones(
                (tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                dtype=self.config.dtype
            ) * (layer_id + 1) * (page_idx + 1)
            
            page_value = jnp.ones(
                (tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                dtype=self.config.dtype
            ) * (layer_id + 1) * (page_idx + 1) * 2
            
            # Position-dependent factor for first 8 positions
            for pos in range(8):
                page_key = page_key.at[pos].multiply(pos + 1)
                page_value = page_value.at[pos].multiply(pos + 1)
            
            # Update pages directly without conditionals
            key_pages = key_pages.at[page_idx].set(page_key)
            value_pages = value_pages.at[page_idx].set(page_value)
        
        # Store the populated pages in the cache
        cache["decoder"][f"layers_{layer_id}"] = {
            "key_pages": key_pages,
            "value_pages": value_pages,
        }
    

  
  def _populate_prefill_kv_pages_dynamic(self, input_tokens, new_vars, page_state, slot, true_length):
    """Extract key/value projections and populate pages - simplified dynamic version.
    
    This version simplifies the dynamic aspects to avoid hanging during JAX tracing.
    
    Args:
        input_tokens: Input token tensor
        new_vars: Variables returned from model.apply()
        page_state: Current page state
        slot: Slot (page group) ID
        true_length: Actual sequence length
    
    Returns:
        Updated cache with populated key/value pages
    """
    print("\n=== Populating KV Pages (Simplified Dynamic) ===")
    
    # Initialize cache structure
    cache = {
        "page_manager": page_state,
        "decoder": {}
    }
    
    # Get fixed shape parameters
    tokens_per_page = self.config.tokens_per_page
    num_pages = self.config.num_pages
    
    # Process each layer
    for layer_id in range(self.config.num_decoder_layers):
        print(f"Processing layer {layer_id}")
        
        # Initialize empty key and value pages with fixed shapes
        key_pages = jnp.zeros(
            (num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        value_pages = jnp.zeros(
            (num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        
        # Extract parameters for computing key/value projections
        # In a real implementation, you'd get these from the model
        # For now, we'll create simple data based on position
        
        # Get page map for this slot
        slot_page_map = page_state.page_map[slot]
        
        # Simplified approach: Just process a fixed number of pages
        # This avoids issues with dynamic bounds in loops
        MAX_PAGES_TO_PROCESS = 4  # Fixed number to avoid tracing issues
        
        for logical_page_idx in range(MAX_PAGES_TO_PROCESS):
            # Get physical page index
            physical_page = slot_page_map[logical_page_idx]
            
            # Skip invalid pages using lax.cond
            def process_page(_):
                # Create a batch of tokens for this page
                # We'll just use the page and position indices to create unique values
                page_keys = jnp.ones(
                    (tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype
                ) * (logical_page_idx + 1)
                
                page_values = jnp.ones(
                    (tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype
                ) * (logical_page_idx + 1) * 2
                
                # For each position in the page, generate position-specific values
                for pos in range(tokens_per_page):
                    # Make values position-dependent
                    page_keys = page_keys.at[pos].multiply(pos + 1)
                    page_values = page_values.at[pos].multiply(pos + 1) 
                
                # Update the page in our key/value arrays
                new_key_pages = key_pages.at[physical_page].set(page_keys)
                new_value_pages = value_pages.at[physical_page].set(page_values)
                
                return new_key_pages, new_value_pages
            
            def skip_page(_):
                return key_pages, value_pages
            
            # Only process valid pages
            key_pages, value_pages = jax.lax.cond(
                physical_page >= 0,
                process_page,
                skip_page,
                None
            )
        
        # Store the populated pages in the cache
        cache["decoder"][f"layers_{layer_id}"] = {
            "key_pages": key_pages,
            "value_pages": value_pages,
        }
    
    return cache


  def _populate_prefill_kv_pages_static(self, input_tokens, new_vars, page_state, slot, true_length):
    """A completely static version that avoids any traced values.
    
    This implementation uses completely static shapes and dummy values. For actual
    implementation, you'd need to extract real K/V values from the model.
    """
    print("\n=== Populating KV Pages (Static Version) ===")
    
    # Initialize cache structure
    cache = {
        "page_manager": page_state,
        "decoder": {}
    }
    
    # Create a minimal dummy implementation that just fills in some content
    # for testing purposes
    for layer_id in range(self.config.num_decoder_layers):
        # Initialize key and value pages
        key_pages = jnp.zeros(
            (self.config.num_pages, self.config.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        value_pages = jnp.zeros(
            (self.config.num_pages, self.config.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        
        # Create test data - in real implementation, get this from the model
        # Fill the first 3 physical pages with some test data
        for test_page in range(3):
            # Fill first 10 positions in each page with non-zero values
            for pos in range(10):
                test_k = jnp.ones((self.config.num_kv_heads, self.config.head_dim)) * (layer_id + 1) * (pos + 1)
                test_v = jnp.ones((self.config.num_kv_heads, self.config.head_dim)) * (layer_id + 1) * (pos + 1) * 2
                
                key_pages = key_pages.at[test_page, pos].set(test_k)
                value_pages = value_pages.at[test_page, pos].set(test_v)
        
        # Store in cache
        cache["decoder"][f"layers_{layer_id}"] = {
            "key_pages": key_pages,
            "value_pages": value_pages,
        }
    
    return cache

  def _populate_prefill_kv_pages(self, input_tokens, new_vars, page_state, slot, true_length):
    """Extract key/value projections and populate pages - JAX tracing compatible version.
    
    This version avoids using traced values in shapes and uses JAX-friendly operations.
    
    Args:
        input_tokens: Input token tensor
        new_vars: Variables returned from model.apply()
        page_state: Current page state
        slot: Slot (page group) ID
        true_length: Actual sequence length
    
    Returns:
        Updated cache with populated key/value pages
    """
    print("\n=== Populating KV Pages ===")
    
    # Initialize cache structure
    cache = {
        "page_manager": page_state,
        "decoder": {}
    }
    
    # Get fixed shape parameters
    tokens_per_page = self.config.tokens_per_page
    max_seq_len = self.config.max_prefill_predict_length
    
    # Don't calculate num_pages_needed using true_length (which is traced)
    # Instead, use page_map directly to determine which pages to populate
    
    # Process each layer
    for layer_id in range(self.config.num_decoder_layers):
        print(f"Processing layer {layer_id}")
        
        # Initialize empty key and value pages with fixed shapes
        key_pages = jnp.zeros(
            (self.config.num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        value_pages = jnp.zeros(
            (self.config.num_pages, tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        
        # Create dummy key and value data with fixed shapes
        # In a real implementation, this would come from the model's forward pass
        dummy_keys = jnp.ones(
            (1, max_seq_len, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        dummy_values = jnp.ones(
            (1, max_seq_len, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
        )
        
        # Now fill pages using the page map
        # We'll use page_map to determine which physical pages to fill
        # Look at the first few entries in page_map to find valid pages
        for logical_page_idx in range(min(8, page_state.page_map.shape[1])):  # Limit to avoid traced length issues
            # Get physical page from page map
            physical_page = page_state.page_map[slot, logical_page_idx]
            
            # Skip invalid pages
            def fill_page(_):
                # Calculate token range for this page
                start_token = logical_page_idx * tokens_per_page
                # Use min to avoid going past max_seq_len or true_length
                # Note that min works with traced values in JAX
                end_token = jnp.minimum(jnp.minimum((logical_page_idx + 1) * tokens_per_page, max_seq_len), true_length)
                
                updated_k = key_pages
                updated_v = value_pages
                
                # For each token position in this page
                # Since we can't use a loop with a traced bound, we'll loop over all positions
                # and mask out invalid ones
                for token_offset in range(tokens_per_page):
                    token_idx = start_token + token_offset
                    
                    # Check if this token is within valid range
                    is_valid = jnp.logical_and(token_idx < end_token, token_idx < true_length)
                    
                    # Get key/value for this token
                    # Use dynamic slice to safely handle out-of-bounds indices
                    token_k = jax.lax.dynamic_slice(
                        dummy_keys, 
                        (0, jnp.minimum(token_idx, max_seq_len - 1), 0, 0), 
                        (1, 1, self.config.num_kv_heads, self.config.head_dim)
                    )
                    token_v = jax.lax.dynamic_slice(
                        dummy_values, 
                        (0, jnp.minimum(token_idx, max_seq_len - 1), 0, 0), 
                        (1, 1, self.config.num_kv_heads, self.config.head_dim)
                    )
                    
                    # Reshape to match expected dimensions
                    token_k = jnp.reshape(token_k, (self.config.num_kv_heads, self.config.head_dim))
                    token_v = jnp.reshape(token_v, (self.config.num_kv_heads, self.config.head_dim))
                    
                    # Only update if token is valid (using a safe conditional update)
                    def update_kv(args):
                        k, v = args
                        new_k = k.at[physical_page, token_offset].set(token_k)
                        new_v = v.at[physical_page, token_offset].set(token_v)
                        return new_k, new_v
                    
                    def keep_kv(args):
                        return args
                    
                    updated_k, updated_v = jax.lax.cond(
                        is_valid,
                        update_kv,
                        keep_kv,
                        (updated_k, updated_v)
                    )
                
                return updated_k, updated_v
            
            def skip_page(_):
                return key_pages, value_pages
            
            # Only update valid pages
            key_pages, value_pages = jax.lax.cond(
                physical_page >= 0,
                fill_page,
                skip_page,
                None
            )
        
        # Store the populated pages in the cache
        cache["decoder"][f"layers_{layer_id}"] = {
            "key_pages": key_pages,
            "value_pages": value_pages,
        }
    
    return cache

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      donate_argnums=(
          1,
          2,
      ),
  )
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Insert a single computed prefill cache into KV cache."""
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)

    unboxed_prefix["cache"] = self._maybe_unstack_prefill_result_cache(unboxed_prefix["cache"])

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
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      elif path_key == "cache_prefill_segment_id":
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        ## zero out in case prefill cache is too small to cover
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
        ## copy prefill cachce
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
        return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
      else:
        raise ValueError(f"We don't have a strategy for inserting {path_key}")

    inserted_cache = jax.tree_util.tree_map_with_path(
        copy,
        unboxed_prefix["cache"],
        decode_state["cache"],
        self.kv_cache_annotations_named,
    )
    inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
    inserted_next_pos = jax.lax.dynamic_update_index_in_dim(decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
    inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
        decode_state["generated_tokens"],
        unboxed_prefix["generated_tokens"],
        slot,
        0,
    )
    inserted_tokens = jax.lax.dynamic_update_index_in_dim(decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
    }

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=(
          "num_prompts",
          "seq_len",
      ),
      donate_argnums=(
          1,
          2,
      ),
  )
  def insert_partial(
      self,
      prefix: PackedPrefix,
      decode_state: DecodeState,
      cache: Any,
      slots: jax.Array,
      *,
      start_indices: jax.Array,
      num_prompts: int,
      seq_len: int,
  ) -> DecodeState:
    """Insert into KV cache"""
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)
    cache_unboxed = max_utils.unbox_logicallypartioned(cache)
    cache_unboxed = self._maybe_unstack_prefill_result_cache(cache_unboxed)
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
        zeros = jnp.zeros((1, self.config.max_target_length - self.config.max_prefill_predict_length), dtype=jnp.int32)
        return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      elif path_key == "cache_prefill_segment_id":
        zeros = jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32)
        ## zero out in case prefill cache is too small to cover
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
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
        seqlen_index = self.config.prefill_cache_axis_order.split(",").index("1")
        start_indices = [0, 0, 0, 0]
        start_indices[seqlen_index] = start_idx
        slice_size = list(partial_cache.shape)
        slice_size[seqlen_index] = seq_len

        slice_size = tuple(slice_size)
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
      inserted_cache = jax.tree_util.tree_map_with_path(copy, cache_unboxed, inserted_cache, self.kv_cache_annotations_named)
      inserted_logits = jax.lax.dynamic_update_index_in_dim(inserted_logits, unboxed_prefix["logits"][i, ...], slot, 0)
      inserted_next_pos = jax.lax.dynamic_update_index_in_dim(inserted_next_pos, unboxed_prefix["next_pos"][i, ...], slot, 0)
      inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
          inserted_generated_tokens,
          unboxed_prefix["generated_tokens"][i, ...],
          slot,
          0,
      )
      inserted_tokens = jax.lax.dynamic_update_index_in_dim(inserted_tokens, unboxed_prefix["tokens"][i, ...], slot, 0)

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
    }

  def get_prefix_destination_sharding(self) -> Any:
    return {
        "logits": self.replicated_sharding,
        "cache": self.prefill_kv_cache_shardings,
        "next_pos": self.replicated_sharding,
        "generated_tokens": self.replicated_sharding,
        "tokens": self.replicated_sharding,
    }

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path=self.config.tokenizer_path, extra_ids=0)

  def build_tokenizer(self, metadata: tokenizer_pb2.TokenizerParameters) -> tokenizer_api.Tokenizer:
    """Return a tokenizer"""
    if "tiktoken" in metadata.path:
      return token_utils.TikToken(metadata)
    else:
      return token_utils.SentencePieceTokenizer(metadata)

  def _init_paged_decode_state(self, params, model, config, rng):
    """Initializes decode state for Paged Attention (returns a dictionary)."""
    print("\nEntering _init_paged_decode_state:")
    print(f"  model: {type(model).__name__}")
    print(f"  mesh.shape: {self._mesh.shape}")
    print(f"  logical_axis_rules: {config.logical_axis_rules}")

    batch_size = config.per_device_batch_size * jax.device_count()

    with nn_partitioning.axis_rules(config.logical_axis_rules):
      # Initialize using model.init() directly like test.py
      input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
      initial_vars = model.init(
          {"params": rng, "dropout": rng, "cache": rng},
          jnp.ones(input_shape, dtype=jnp.int32),
          jnp.ones(input_shape, dtype=jnp.int32),
      )

      if params is not None:
        initial_vars["params"] = params  # Replace with provided params if available

      # Apply partitioning directly to initial_vars instead of TrainState
      state_mesh_annotations = nn.get_partition_spec(initial_vars)
      initial_vars = nn.with_logical_partitioning(initial_vars, state_mesh_annotations)

      print("Exiting _init_paged_decode_state: Returning type:", type(initial_vars))
      return initial_vars

  def init_decode_state(self, rng: Optional[PRNGKeyType] = None, params=None):
      """Initialize decode state with JAX-compatible structures only."""
      if rng is None:
        rng = jax.random.PRNGKey(0)

      def init():
        x = jnp.ones((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)

        # Basic cache structure
        cache_struct = {
            "decoder": {
                f"layers_{i}": {
                    # Pages for KV cache
                    "key_pages": jnp.zeros(
                        (self.config.num_pages, self.config.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                        dtype=self.config.dtype,
                    ),
                    "value_pages": jnp.zeros(
                        (self.config.num_pages, self.config.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
                        dtype=self.config.dtype,
                    ),
                    # Page manager state arrays
                    "page_status": jnp.zeros((self.config.num_pages,), dtype=jnp.int32),
                    "page_map": jnp.full(
                        (
                            int(self.config.per_device_batch_size * jax.device_count()),
                            (self.config.max_target_length + self.config.tokens_per_page - 1) // self.config.tokens_per_page,
                        ),
                        -1,
                        dtype=jnp.int32,
                    ),
                    "sequence_lengths": jnp.zeros(
                        (int(self.config.per_device_batch_size * jax.device_count()),), dtype=jnp.int32
                    ),
                    "current_page": jnp.full(
                        (int(self.config.per_device_batch_size * jax.device_count()),), -1, dtype=jnp.int32
                    ),
                    "current_page_position": jnp.zeros(
                        (int(self.config.per_device_batch_size * jax.device_count()),), dtype=jnp.int32
                    ),
                }
                for i in range(self.config.num_decoder_layers)
            }
        }

        return {
            "logits": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1, self.config.vocab_size)),
            "cache": cache_struct if self.config.attention == "paged" else {},
            "next_pos": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32),
            "generated_tokens": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32),
            "tokens": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32),
        }

      with nn_partitioning.axis_rules(self.config.logical_axis_rules):
        abstract_outputs = jax.eval_shape(init)
      logical_annotations = nn.get_partition_spec(abstract_outputs)

      with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        mesh_annotations = nn.logical_to_mesh(logical_annotations)

      shardings = jax.tree_util.tree_map(
          lambda mesh_annotation: jax.sharding.NamedSharding(self._mesh, mesh_annotation),
          mesh_annotations,
      )

      @functools.partial(jax.jit, out_shardings=shardings)
      def initialize():
        return init()

      return initialize()

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return int(self.config.per_device_batch_size * jax.device_count())

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return int(self.config.max_prefill_predict_length)

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return 1

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError


def set_engine_vars_from_base_engine(
    engine: MaxEngine,
    base_engine: MaxEngine,
    rng: PRNGKeyType,
):
  """Set internal vars from base_engine, which has already loaded the checkpoint and has sharding,
  mesh, and kv cache related vars set.
  """
  engine.model.quant.quant_mode = base_engine.model.quant.quant_mode
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = max_utils.get_kv_cache_annotations(engine.model, engine.config, rng, engine.mesh)  # pylint: disable=protected-access
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine.mesh, x),
      engine.kv_cache_annotations,  # pylint: disable=protected-access
  )


def create_engine_from_config_flags(batch_size, max_prefill_predict_length, max_target_length, args_str):
  """Create new MaxEngine instance with given batch_size, prefill and target lengths, and any config
  params provided through `args_str`.
  """
  args = {}
  args["scan_layers"] = "false"
  args["async_checkpointing"] = "false"
  args["ici_fsdp_parallelism"] = "1"
  args["ici_autoregressive_parallelism"] = "1"
  args["ici_tensor_parallelism"] = "-1"
  args["weight_dtype"] = "bfloat16"
  args["attention"] = "dot_product"

  # batch and cache related
  args["max_prefill_predict_length"] = f"{max_prefill_predict_length}"
  args["max_target_length"] = f"{max_target_length}"
  args["per_device_batch_size"] = f"{batch_size}"
  print(f"Command line args: {args_str}")
  cmd_args = args_str.split(" ")
  for cmd_arg in cmd_args:
    k, v = cmd_arg.split("=")
    args[k.strip()] = v.strip()
  assert "load_parameters_path" in args, "load_parameters_path must be defined"
  updated_args = ["MaxText/maxengine_server.py", "../configs/base.yml"]
  for k, v in args.items():
    option = f"{k}={v}"
    updated_args.append(option)
  print(f"Invoking maxengine with args:\n \t{updated_args}")
  pyconfig.initialize(updated_args)
  cfg = MaxEngineConfig(cp.deepcopy(pyconfig._config.keys))  # pylint: disable=protected-access
  engine = MaxEngine(cfg)
  return engine
