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

"""Implementation of Engine API for MaxText."""

from collections import defaultdict
from typing import Any, Callable
import functools
import os.path
import uuid
import warnings

from jax.experimental.layout import Format
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp

if jax.__version_info__ >= (0, 6, 3):
  from jax.experimental.layout import Layout as DLL  # type: ignore
else:
  from jax.experimental.layout import DeviceLocalLayout as DLL  # type: ignore

from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
import flax

from MaxText import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.models import models
from maxtext.layers import quantizations
from maxtext.inference import inference_utils
from maxtext.inference.page_manager import PageManager, PageState
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import lora_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.common.gcloud_stub import jetstream, is_decoupled
from maxtext.common.common_types import MODEL_MODE_PREFILL, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE

config_lib, engine_api, token_utils, tokenizer_api, _token_params_ns = jetstream()
TokenizerParameters = getattr(_token_params_ns, "TokenizerParameters", object)  # type: ignore[assignment]
TokenizerType = getattr(_token_params_ns, "TokenizerType", object)  # type: ignore[assignment]


warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any


# TODO(yuyanpeng): Should import ExistingPrefix from jetstream.engine.engine_api
@struct.dataclass
class ExistingPrefix:
  """Represents a prefix that has already been processed."""

  #: The kv-cache for the prefix get from model params cache.
  cache: Any
  #: The tokens that have already been processed without padding.
  common_prefix_tokens: jax.Array


class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  TODO: evaluate the need for this given the restructured pyconfig.py
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


_BaseEngine = engine_api.Engine if (not is_decoupled() and hasattr(engine_api, "Engine")) else object


class MaxEngine(_BaseEngine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, config: Any, devices: Any | None = None):
    self.config = config

    # Mesh definition
    devices_array = maxtext_utils.create_device_mesh(config=config, devices=devices)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    # Model and Optimizer definition
    quant = quantizations.configure_quantization(config)
    self.model = models.transformer_as_linen(config, mesh=self._mesh, quant=quant, model_mode=MODEL_MODE_PREFILL)
    self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

    self.abstract_params = None
    self.prefill_kv_cache_annotations = None
    self.kv_cache_annotations = None
    self.kv_cache_annotations_named = None
    self.prefill_kv_cache_shardings = None
    self.kv_cache_shardings = None
    self.state_mesh_annotations = None
    self.decode_state_shapes = None
    self.decode_state_layouts = None
    self.param_layouts = None
    self.rng = None

    # Initialize page manager and page state
    self.page_manager = None
    self.page_state = None
    if self.config.attention == "paged":
      self.page_manager = PageManager(self.config)
      self.page_state = self.page_manager.get_initial_page_state()

  def print_stats(self, label: str):
    max_utils.print_mem_stats(label)
    max_utils.print_cpu_ram_stats(label)

  def generate_aot(
      self, params: Params, decode_state: DecodeState, rng: PRNGKeyType | None = None
  ):  # returns (new_decode_state, result_tokens)
    """Wrapper to generate for ahead of time compilation."""

    return self.generate(params=params, decode_state=decode_state, rng=rng)

  def _compile_generate_and_get_layouts(
      self, params: Any, decode_state: Any, rng_shape: Any, xla_flags: dict[str, Any] | None = None
  ) -> tuple[Any, Any, Any, Any]:
    """Optimal memory layout for params and decode_state."""

    param_layout = Format(DLL.AUTO)
    decode_state_layout = Format(DLL.AUTO)
    # Keyword arguments are not yet supported in JAX for specifying shardings. Therefore, all AOT
    # compiled functions use arguments instead.
    compiled_generate = (
        jax.jit(
            self.generate_aot,
            in_shardings=(param_layout, decode_state_layout, None),
            out_shardings=(Format(DLL.AUTO), Format(DLL.AUTO)),
            donate_argnames=("decode_state",),
        ).lower(params, decode_state, rng_shape)
    ).compile(compiler_options=xla_flags)

    arg_layouts, _ = compiled_generate.input_formats
    generate_out_layouts, _ = compiled_generate.output_formats

    return compiled_generate, arg_layouts[0], arg_layouts[1], generate_out_layouts

  def _identity(self, x: Any) -> Any:
    """Avoids lambda that breaks JAX caching."""

    return x

  def _iterated_layout(self, arrays: Any, layouts: Any, xla_flags: dict[str, Any] | None = None) -> Any:
    """Lays out an array tensor by tensor to prevent OOMs."""

    def _layout(x, s, l):
      if x.format == l:
        return x
      # Somehow this can be None sometimes.
      dll = (l.layout if jax.__version_info__ >= (0, 6, 3) else l.device_local_layout) if isinstance(l, Format) else l
      f = jax.jit(self._identity, out_shardings=Format(dll, s)).lower(x).compile(compiler_options=xla_flags)
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
  ) -> tuple[Any, Params, Any]:
    """Ahead of time compilation of generate with auto layout, relayout parameters."""
    if pass_rng_shape:
      rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))
    else:
      rng_shape = None
    self.decode_state_shapes = jax.eval_shape(self.init_decode_state, rng_shape)

    generate_executable, self.param_layouts, _, self.decode_state_layouts = self._compile_generate_and_get_layouts(
        self.abstract_params, self.decode_state_shapes, rng_shape, xla_flags
    )
    return (
        generate_executable,
        self._iterated_layout(params, self.param_layouts),
        jax.jit(self.init_decode_state, in_shardings=(None), out_shardings=self.decode_state_layouts)
        .lower(rng_shape)
        .compile(),
    )

  def load_params(self, *args, params=None, rng: PRNGKeyType | None = None, **kwargs) -> Params:
    """Load Parameters from GCS or reshard given Parameters"""
    # pylint: disable=unused-argument

    if rng is None:
      rng = jax.random.PRNGKey(0)

    if self.model.quant and self.config.checkpoint_is_quantized:
      print("Loading from the quantized checkpoint...")
      self.model.quant.quant_mode = quantizations.get_quant_mode("serve")

    rng1, rng2, rng3 = jax.random.split(rng, 3)
    if params:
      print("Resharding given params")
      _, self.state_mesh_annotations, state_mesh_shardings = maxtext_utils.get_abstract_state(
          self.model, None, self.config, rng, self._mesh, False
      )
      # reshard given params based on shardings from config in MaxEngine
      params = jax.device_put(params, state_mesh_shardings.params)
      state = maxtext_utils.init_decode_state(None, params)
      state = max_utils.unbox_logicallypartioned(state)
    else:
      state, self.state_mesh_annotations = maxtext_utils.setup_decode_state(
          self.model, self.config, rng1, self._mesh, None
      )
    # pylint: disable=isinstance-second-argument-not-valid-type
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
        if isinstance(x, jax.Array)
        else None,
        state.params,
    )

    self.prefill_kv_cache_annotations = maxtext_utils.get_prefill_kv_cache_annotations(
        self.model, self.config, rng2, self._mesh, self.page_state
    )
    self.prefill_kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x),
        self.prefill_kv_cache_annotations,
    )

    if self.config.stack_prefill_result_cache:
      # Add extra axis for the axis generated by the stack.
      self.prefill_kv_cache_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(None, *x.spec)),
          self.prefill_kv_cache_shardings,
      )
      self.prefill_kv_cache_shardings = self.prefill_kv_cache_shardings["decoder"]["layers_0"]

    self.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations(
        self.model, self.config, rng2, self._mesh, self.page_state
    )
    self.kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x),
        self.kv_cache_annotations,
    )

    if self.model.quant and not self.config.checkpoint_is_quantized:
      params = self.quantize_params(state, rng3)
    else:
      params = state.params

    self.print_stats("After load_params")

    return params

  def load_single_adapter(self, adapter_path):
    """
    Load Single adapter from adapter_path.
    Expect adapter_config.json and LoRA adapter weights at this path within subdirectory `/0/items`.
    """
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights_path = os.path.join(adapter_path, "0", "items")

    params, config = lora_utils.load_adapter(self.config, self.abstract_params, adapter_config_path, adapter_weights_path)

    if config is None:
      raise ValueError(f"Failed to read lora_config from {adapter_config_path}")

    if params is None:
      raise ValueError(f"Failed to read lora_config from {adapter_config_path}")

    config["adapter_path"] = adapter_weights_path

    self.print_stats("After load_single_adapter.")

    return params, config

  def apply_adapter(self, base_params, adapter_config, adapter_params):
    """Apply the adapter params on the base params."""

    lora_rank = int(adapter_config["r"])
    lora_scale_factor = float(adapter_config["lora_alpha"]) / lora_rank
    lora_utils.apply_lora_on_base_params(base_params, adapter_params, lora_scale_factor)

  def unapply_adapter(self, base_params, adapter_config, adapter_params):
    """Unapply the adapter params from the merged params to get back the base params."""

    lora_rank = int(adapter_config["r"])
    lora_scale_factor = float(adapter_config["lora_alpha"]) / lora_rank
    lora_utils.unapply_lora_from_base_params(base_params, adapter_params, lora_scale_factor)

  def quantize_params(self, state, rng: PRNGKeyType | None = None):
    """Forward pass to quantize decode params."""
    if rng is None:
      rng = jax.random.PRNGKey(0)

    self.model.quant.quant_mode = quantizations.get_quant_mode("convert")

    @jax.jit
    def model_apply(_p, _rng):
      image_shape = mm_processor.get_dummy_image_shape_for_init(
          model_name=self.config.model_name,
          batch_size=self.config.micro_batch_size_to_train_on,
      )
      audio_shape = mm_processor.get_dummy_audio_shape_for_init(self.config)
      return self.model.apply(
          _p | {"aqt": {}},
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          encoder_images=jnp.ones(image_shape, dtype=jnp.float32) if self.config.use_multimodal else None,
          # encoder_image_masks indicates valid tiles if image tiling + padding is used in vision encoder input.
          encoder_image_masks=jnp.ones(image_shape[:2], dtype=jnp.int32)
          if self.config.use_multimodal and "llama4" in self.config.model_name
          else None,
          encoder_audios=jnp.ones(audio_shape, dtype=jnp.float32) if self.config.use_audio else None,
          decoder_segment_ids=jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
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
    maxtext_utils.save_quantized_checkpoint_if_configured(self.config, params)
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

  def prefill_aot(  # pylint: disable=too-many-positional-arguments
      self,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: PRNGKeyType | None = None,
  ):  # returns (new_prefix, result_tokens)
    """Wrapper for prefill for ahead-of-time compilation."""

    return self.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        rng=rng,
    )

  @functools.partial(
      jax.jit, static_argnums=(0,), static_argnames=("return_prompt_logp", "algorithm", "topk", "nucleus_topp")
  )
  def _prefill_jit(
      self,
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      positions: jax.Array | None = None,
      mrope_deltas: jax.Array | None = None,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      audio_values: jax.Array | None = None,
      audio_masks: jax.Array | None = None,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      slot: int | None = None,
      page_state: PageState | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (new_prefix, result_tokens)
    """Performs a JIT-compiled prefill operation on a sequence of tokens.

    This function processes an input sequence (prompt) through the model to compute
    the key-value cache (KV cache). It supports chunked prefilling, where a long
    prompt can be processed in multiple calls by passing an `existing_prefix`.
    After processing the tokens, it samples the first token to begin the
    autoregressive decoding process.

    Args:
      params: The model parameters.
      existing_prefix: An optional `ExistingPrefix` containing the KV cache and
        tokens of a previously processed chunk. Used for chunked prefilling.
      padded_tokens: The input token sequence, padded to a fixed length.
      images: Optional input images for multimodal models.
      image_masks: Optional image masks for multimodal models with tiled images.
      audio_values: Optional input audio for multimodal models.
      audio_masks: Optional audio masks for multimodal models (currently unused).
      true_length: The actual length of `padded_tokens` before padding.
      sampler: A callable for custom sampling logic (currently unused).
      rng: JAX random number generator key for sampling.
      slot: The batch slot index for this request, used for paged attention.
      page_state: The current state of the paged attention manager.
      return_prompt_logp: If True, calculates and returns the log probabilities
        of the prompt tokens.
      algorithm: The sampling algorithm to use (e.g., 'greedy', 'composite').
        Overrides the engine's default.
      topk: The value for top-k sampling. Overrides the engine's default.
      nucleus_topp: The value for top-p (nucleus) sampling. Overrides the
        engine's default.
      temperature: The sampling temperature. Overrides the engine's default.

    Returns:
      A tuple containing:
        - A prefix dictionary with the computed KV cache, the logits for the
          next token, the first sampled token, and other metadata.
        - An `engine_api.ResultTokens` object containing the single sampled
          token to begin decoding.
    """

    start_position = 0
    previous_chunk = None
    input_params = params
    if existing_prefix is not None:
      if not self.use_chunked_prefill:
        raise ValueError("Using chunked prefill is needed for existing_prefix.")
      input_params = params | {"cache": existing_prefix.cache}
      start_position = existing_prefix.common_prefix_tokens.shape[0]
      # TODO(yuyanpeng): rename previous_chunk
      previous_chunk = jnp.expand_dims(existing_prefix.common_prefix_tokens, 0)

    full_true_length = start_position + true_length

    input_tokens = jnp.expand_dims(padded_tokens, 0)

    if positions is not None:
      if positions.ndim == 2:
        positions = jnp.expand_dims(positions, 1)
    else:
      positions = jnp.expand_dims(jnp.arange(start_position, start_position + input_tokens.shape[1]), 0)

    if self.config.use_multimodal and images is not None:
      if images.ndim == 3:
        # For Gemma3 single image, add batch and image count dimensions
        images = images[jnp.newaxis, jnp.newaxis, ...]
        image_masks = image_masks[jnp.newaxis, jnp.newaxis, ...] if image_masks is not None else None
      elif images.ndim == 4:
        # add batch dimension
        images = images[jnp.newaxis, ...]
        image_masks = image_masks[jnp.newaxis, ...] if image_masks is not None else None

    # sequence_indicator will be concatenated to existing_prefix decoder_segment_ids
    start_to_n = jnp.arange(start_position, start_position + input_tokens.shape[1])
    ones_to_keep = start_to_n < full_true_length
    one_d_output = ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    rng, new_rng = jax.random.split(rng)
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flat_logits, new_vars = self.model.apply(
          input_params,
          input_tokens,
          positions,
          encoder_images=images,
          encoder_image_masks=image_masks,
          encoder_audios=audio_values,
          decoder_segment_ids=sequence_indicator,
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
          rngs={"params": new_rng},
          mutable=["cache"],
          previous_chunk=previous_chunk,
          true_length=true_length,
          slot=slot,
          page_state=page_state,
      )
    if return_prompt_logp:
      prompt_logp = inference_utils.prompt_logprobs_from_prefill(flat_logits, input_tokens, true_length)
    else:
      prompt_logp = None

    generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits,
        (0, true_length - 1, 0),
        (flat_logits.shape[0], 1, flat_logits.shape[2]),
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    # sampling first token
    rng, new_rng = jax.random.split(rng)
    first_generated_token = inference_utils.sampling(
        selected_logits,
        new_rng,
        algorithm if algorithm is not None else self.config.decode_sampling_strategy,
        topk=topk if topk is not None else self.config.decode_sampling_top_k,
        nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
        temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
    )

    all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
    if self.config.return_log_prob:
      token_logp = inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token)
    else:
      token_logp = jnp.zeros(first_generated_token.shape, dtype=jnp.float32)
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
        log_prob=token_logp,
        samples_per_slot=1,
    )

    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)

    if mrope_deltas is not None:
      next_pos = jnp.full((1, 1), full_true_length, dtype=jnp.int32) + mrope_deltas
    else:
      next_pos = jnp.full((1, 1), full_true_length, dtype=jnp.int32)

    return {
        "logits": selected_logits,
        "cache": cache,
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_token,
        "prompt_logp": prompt_logp,
        "token_logp": token_logp,
    }, result

  # Public non-JIT prefill method that updates page state
  def prefill(
      self,  # pytype: disable=signature-mismatch
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      positions: jax.Array | None = None,
      mrope_deltas: jax.Array | None = None,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      audio_values: jax.Array | None = None,
      audio_masks: jax.Array | None = None,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      request_id: uuid.UUID | None = None,  # pylint: disable=unused-argument
      slot: int | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (new_prefix, result_tokens)
    """Public API for prefill that updates page state outside JIT."""
    # Update page state before JIT call
    if self.config.attention == "paged" and self.page_manager is not None and self.page_state is not None:
      self.page_state = self.page_manager.update_prefill_pages(  # pytype: disable=attribute-error
          page_state=self.page_state,
          page_group_id=slot,
          true_length=true_length,
      )

    # Sample rng before JIT call
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    # Call JIT-compiled version with current state
    return self._prefill_jit(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padded_tokens,
        positions=positions,
        mrope_deltas=mrope_deltas,
        images=images,
        image_masks=image_masks,
        audio_values=audio_values,
        audio_masks=audio_masks,
        sampler=sampler,
        true_length=true_length,
        page_state=self.page_state,  # Pass current page state
        slot=slot,
        rng=rng,
        return_prompt_logp=return_prompt_logp,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  def prefill_multisampling_aot(  # pylint: disable=too-many-positional-arguments
      self,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      sampler: Callable[[Any], Any] | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (new_prefix, result_tokens)
    """Wrapper for multi-sampling prefill for ahead-of-time compilation."""
    return self.prefill_multisampling(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
        rng=rng,
        num_samples=num_samples,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  def prefill_multisampling(
      self,  # pytype: disable=signature-mismatch
      *,
      # pylint: disable=arguments-differ
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (new_prefix, result_tokens)
    """Public API for prefill multisampling."""

    # Sample rng before JIT call
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    # Call JIT-compiled version
    return self._prefill_multisampling_jit(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
        rng=rng,
        num_samples=num_samples,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

  @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("num_samples", "algorithm", "topk", "nucleus_topp"))
  def _prefill_multisampling_jit(
      self,
      *,
      params: Params,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      num_samples: int = 1,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ) -> tuple[Prefix, Any]:
    """Computes a kv-cache for a new generate request.

    With multi-sampling, the engine will generate multiple first tokens in the
    prefilling stage. The number of tokens is specified by num_samples.
    """

    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
    positions = jnp.expand_dims(jnp.arange(0, input_tokens.shape[1]), 0)

    zero_to_n = jnp.arange(0, padded_tokens.shape[0])
    ones_to_keep = zero_to_n < true_length
    one_d_output = ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    rng, new_rng = jax.random.split(rng)
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flat_logits, new_vars = self.model.apply(
          params,
          input_tokens,
          positions,
          decoder_segment_ids=sequence_indicator,
          enable_dropout=False,
          model_mode=MODEL_MODE_PREFILL,
          rngs={"params": new_rng},
          mutable=["cache"],
      )

    next_pos = jnp.full((1, 1), true_length, dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits,
        (0, true_length - 1, 0),
        (flat_logits.shape[0], 1, flat_logits.shape[2]),
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    # sampling first tokens
    first_generated_tokens = []
    token_logps = [] if self.config.return_log_prob else None
    for _ in range(num_samples):
      rng, new_rng = jax.random.split(rng)
      first_generated_token = inference_utils.sampling(
          selected_logits,
          new_rng,
          algorithm if algorithm is not None else self.config.decode_sampling_strategy,
          topk=topk if topk is not None else self.config.decode_sampling_top_k,
          nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
          temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
      )
      first_generated_tokens.append(first_generated_token)
      if self.config.return_log_prob:
        # pytype: disable=attribute-error
        token_logps.append(inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token))
    first_generated_tokens = jnp.concatenate(first_generated_tokens, axis=0)
    if self.config.return_log_prob:
      token_logps = jnp.concatenate(token_logps, axis=0)

    all_valid = jnp.ones((num_samples, 1), dtype=jnp.int8)
    generated_tokens = jnp.zeros((num_samples, 1), dtype=jnp.int32)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((first_generated_tokens, all_valid, generated_tokens), axis=1),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, 1),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(1, 2),
        # And lengths is rank 1.
        length_idx=(2, 3),
        log_prob=token_logps,
        samples_per_slot=num_samples,
    )

    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)

    return {
        "logits": selected_logits,
        "cache": cache,
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_tokens,
    }, result

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("num_prompts", "return_prompt_logp", "algorithm", "topk", "nucleus_topp"),
  )
  def prefill_concat(
      self,
      *,
      params: Params,
      existing_prefix: ExistingPrefix | None = None,
      padded_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array,
      start_pos: jax.Array,
      true_lengths: jax.Array,
      num_prompts: int,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      return_prompt_logp: bool = False,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (maybe_batch, packed_prefix, list_of_result_tokens)
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
          model_mode=MODEL_MODE_PREFILL,
          rngs={"params": new_rng},
          mutable=["cache"],
      )
    cache = new_vars["cache"]
    cache = self._maybe_stack_prefill_result_cache(cache)
    if return_prompt_logp:
      prompt_logp = inference_utils.prompt_logprobs_from_packed_prefill(
          flat_logits, input_tokens, decoder_positions, decoder_segment_ids, true_lengths
      )
    else:
      prompt_logp = None

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
          algorithm if algorithm is not None else self.config.decode_sampling_strategy,
          topk=topk if topk is not None else self.config.decode_sampling_top_k,
          nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
          temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
      )
      all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
      if self.config.return_log_prob:
        token_logp = inference_utils.log_prob_of_chosen_token(selected_logits, first_generated_token)
      else:
        token_logp = None
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
          log_prob=token_logp,
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
    prefill_results["prompt_logp"] = prompt_logp
    return cache, prefill_results, first_tokens

  # Public non-JIT generate method that updates page state
  def generate(
      self,
      params: Params,
      decode_state: DecodeState,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (decode_state, result_tokens)
    """Public API for generate that updates page state outside JIT."""

    # Update page state before JIT call
    if self.page_manager is not None and self.page_state is not None:
      self.page_state = self.page_manager.update_decode_pages(self.page_state)

    # Sample rng before JIT call
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    # Call JIT-compiled version with current state
    new_state, result = self._generate_jit(
        params=params,
        decode_state=decode_state,
        sampler=sampler,
        page_state=self.page_state,
        rng=rng,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )

    return max_utils.unbox_logicallypartioned(new_state), result

  @functools.partial(
      jax.jit, static_argnums=(0,), donate_argnums=(2,), static_argnames=("algorithm", "topk", "nucleus_topp")
  )
  def _generate_jit(
      self,
      params: Params,
      decode_state: DecodeState,
      *,
      sampler: Callable[[Any], Any] | None = None,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      page_state: PageState | None = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):  # returns (decode_state, result_tokens)
    """Performs a single, JIT-compiled autoregressive decoding step.

    This function takes the current decoding state, which includes the KV cache
    for the sequence generated so far, and generates the next token. It uses the
    model to predict logits for the next token based on the previous token and
    then samples from that distribution.

    Args:
      params: The model parameters.
      decode_state: The current state of the decoding process, containing the
        KV cache, the previously generated token, the current position, etc.
        This argument is donated to save memory.
      sampler: A callable for custom sampling logic (currently unused).
      rng: JAX random number generator key for sampling.
      page_state: The current state of the paged attention manager.
      algorithm: The sampling algorithm to use (e.g., 'greedy', 'composite').
        Overrides the engine's default.
      topk: The value for top-k sampling. Overrides the engine's default.
      nucleus_topp: The value for top-p (nucleus) sampling. Overrides the
        engine's default.
      temperature: The sampling temperature. Overrides the engine's default.

    Returns:
      A tuple containing:
        - The updated `DecodeState` with the new KV cache, new token, and
          incremented position, ready for the next decoding step.
        - An `engine_api.ResultTokens` object containing the newly generated
          token and its metadata.
    """

    previous_token = decode_state["tokens"]
    rng, new_rng = jax.random.split(rng)
    # run one step generation
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out_logits, new_vars = self.model.apply(
          params | {"cache": decode_state["cache"]},
          previous_token,
          decode_state["next_pos"],
          enable_dropout=False,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"params": new_rng},
          mutable=["cache"],
          page_state=page_state,
      )
    out_logits = jax.lax.with_sharding_constraint(out_logits, self.replicated_sharding)
    new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], self.kv_cache_shardings)
    # sampling tokens
    rng, new_rng = jax.random.split(rng)
    new_token = inference_utils.sampling(
        out_logits,
        new_rng,
        algorithm if algorithm is not None else self.config.decode_sampling_strategy,
        topk=topk if topk is not None else self.config.decode_sampling_top_k,
        nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
        temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
    )
    all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
    if self.config.return_log_prob:
      token_logp = inference_utils.log_prob_of_chosen_token(out_logits, new_token)
    else:
      token_logp = jnp.zeros(new_token.shape, dtype=jnp.float32)

    # Increment index by 1 as prefill returns first token
    next_pos = decode_state["next_pos"] + 1
    generated_tokens = decode_state["generated_tokens"] + 1

    result = engine_api.ResultTokens(
        data=jnp.concatenate((new_token, all_valid, generated_tokens), axis=1),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, 1),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(1, 2),
        # And lengths is rank 1.
        length_idx=(2, 3),
        log_prob=token_logp,
        samples_per_slot=1,
    )

    return {
        "logits": out_logits,
        "cache": new_cache,
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": new_token,
        "token_logp": token_logp,
    }, result

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      donate_argnums=(
          1,
          2,
      ),
  )
  def bulk_insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slots: list[int],
  ) -> DecodeState:
    """Insert a single computed prefill cache into multiple slots in KV cache."""
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

      for slot in slots:
        if path_key == "cache_ar_segment_id":
          ### goal: zero this out in case there is existing data
          s = list(full_cache.shape)
          s[batch_idx] = 1
          zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
          full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
        elif path_key == "cache_prefill_segment_id":
          s = list(full_cache.shape)
          s[batch_idx] = 1
          zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
          ## zero out in case prefill cache is too small to cover
          full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
          ## copy prefill cachce
          full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
        elif path_key == "cached_ar_lengths":
          full_cache = full_cache.at[slot].set(0)
        elif path_key in [
            "cached_prefill_key",
            "cached_prefill_value",
            "cached_prefill_key_scale",
            "cached_prefill_value_scale",
        ]:
          full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
        else:
          raise ValueError(f"We don't have a strategy for inserting {path_key}")

      return full_cache

    inserted_cache = jax.tree_util.tree_map_with_path(
        copy,
        unboxed_prefix["cache"],
        decode_state["cache"],
        self.kv_cache_annotations_named,
    )

    for i, slot in enumerate(slots):
      decode_state["logits"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["logits"], unboxed_prefix["logits"], slot, 0
      )
      decode_state["next_pos"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0
      )
      decode_state["generated_tokens"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["generated_tokens"],
          jnp.expand_dims(unboxed_prefix["generated_tokens"][i], axis=0),
          slot,
          0,
      )
      decode_state["tokens"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["tokens"],
          jnp.expand_dims(unboxed_prefix["tokens"][i], axis=0),
          slot,
          0,
      )
      decode_state["token_logp"] = jax.lax.dynamic_update_index_in_dim(
          decode_state["token_logp"],
          jnp.expand_dims(unboxed_prefix["token_logp"][i], axis=0),
          slot,
          0,
      )

    inserted_logits = jax.lax.with_sharding_constraint(decode_state["logits"], self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(
        decode_state["generated_tokens"], self.replicated_sharding
    )
    inserted_next_pos = jax.lax.with_sharding_constraint(decode_state["next_pos"], self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(decode_state["tokens"], self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(decode_state["token_logp"], self.replicated_sharding)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnames=("prefix", "decode_state"))
  def _insert_jit(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
      request_id: uuid.UUID | None = None,  # pylint: disable=unused-argument
      page_state_in: PageState | None = None,
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
        return full_cache

      batch_idx = -1
      if "cache_batch" in annotations:
        batch_idx = annotations.index("cache_batch")
      elif "cache_scale_batch" in annotations:
        batch_idx = annotations.index("cache_scale_batch")

      if batch_idx < 0:
        raise ValueError(f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}")

      if path_key == "cache_ar_segment_id":
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      elif path_key == "cache_prefill_segment_id":
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        # zero out in case prefill cache is too small to cover
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
        # copy prefill cache
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

    if self.config.attention == "paged" and self.page_state is not None:

      def _copy_paged(path, prefix_cache, decode_state_cache):
        path_key = path[-1].key
        if path_key in ["key_pages", "value_pages"]:
          page_map_for_slot = page_state_in.page_map[slot]  # pytype: disable=attribute-error
          num_pages_to_copy = page_state_in.num_pages_used[slot]  # pytype: disable=attribute-error

          def _update_pages(prefix_page_idx, state):
            decode_state_pages, prefix_pages, current_page_map = state
            prefix_page = jax.lax.dynamic_index_in_dim(prefix_pages, prefix_page_idx, axis=1)
            dest_page_idx = current_page_map[prefix_page_idx]
            decode_state_pages = jax.lax.dynamic_update_slice_in_dim(
                decode_state_pages, prefix_page, dest_page_idx, axis=1
            )
            return decode_state_pages, prefix_pages, current_page_map

          decode_state_cache, _, _ = jax.lax.fori_loop(
              0,
              num_pages_to_copy,
              _update_pages,
              (decode_state_cache, prefix_cache, page_map_for_slot),
          )
          return decode_state_cache
        else:
          raise ValueError(f"We don't have a strategy for inserting {path_key} for paged attention.")

      inserted_cache = jax.tree_util.tree_map_with_path(
          _copy_paged,
          unboxed_prefix["cache"],
          decode_state["cache"],
      )
    else:
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
    inserted_token_logp = jax.lax.dynamic_update_index_in_dim(
        decode_state["token_logp"], unboxed_prefix["token_logp"], slot, 0
    )

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(inserted_token_logp, self.replicated_sharding)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
      request_id: uuid.UUID | None = None,
  ) -> DecodeState:
    """Non-JIT wrapper for inserting prefill cache."""

    current_page_state = None
    if self.config.attention == "paged" and self.page_manager is not None:
      if self.page_state is None:
        self.page_state = self.page_manager.get_initial_page_state()
      current_page_state = self.page_state

    updated_decode_state = self._insert_jit(
        prefix=prefix,
        decode_state=decode_state,
        slot=slot,
        page_state_in=current_page_state,
    )

    # Update the PageState after the JIT call
    if self.config.attention == "paged" and self.page_manager is not None and self.page_state is not None:
      new_has_active_page = self.page_state.has_active_page.at[slot].set(True)
      self.page_state = self.page_state.replace(has_active_page=new_has_active_page)
    return updated_decode_state

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
        seqlen_index = self.config.prefill_cache_axis_order.split(",").index("1")
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
    inserted_token_logp = decode_state["token_logp"]

    for i in range(num_prompts):
      start_idx = start_indices[i]
      slot = slots[i]
      inserted_cache = jax.tree_util.tree_map_with_path(
          copy, cache_unboxed, inserted_cache, self.kv_cache_annotations_named
      )
      inserted_logits = jax.lax.dynamic_update_index_in_dim(inserted_logits, unboxed_prefix["logits"][i, ...], slot, 0)
      inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
          inserted_next_pos, unboxed_prefix["next_pos"][i, ...], slot, 0
      )
      inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
          inserted_generated_tokens,
          unboxed_prefix["generated_tokens"][i, ...],
          slot,
          0,
      )
      inserted_tokens = jax.lax.dynamic_update_index_in_dim(inserted_tokens, unboxed_prefix["tokens"][i, ...], slot, 0)
      inserted_token_logp = jax.lax.dynamic_update_index_in_dim(
          inserted_token_logp, unboxed_prefix["token_logp"][i, ...], slot, 0
      )

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)
    inserted_token_logp = jax.lax.with_sharding_constraint(inserted_token_logp, self.replicated_sharding)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
        "token_logp": inserted_token_logp,
    }

  def release_pages(self, slot: int):
    """Releases pages associated with a specific slot (page group) via the PageManager."""
    if self.config.attention != "paged" or self.page_manager is None or self.page_state is None:
      print(f"Warning: release_pages called for slot {slot} but paged attention is not configured or state is missing.")
      return
    new_page_state = self.page_manager.release_pages(
        page_state=self.page_state, page_group_id=slot
    )  # pytype: disable=attribute-error
    self.page_state = new_page_state

  def get_prefix_destination_sharding(self) -> Any:
    return {
        "logits": self.replicated_sharding,
        "cache": self.prefill_kv_cache_shardings,
        "next_pos": self.replicated_sharding,
        "generated_tokens": self.replicated_sharding,
        "tokens": self.replicated_sharding,
        "token_logp": self.replicated_sharding,
    }

  def get_tokenizer(self) -> Any:
    """Return tokenizer parameters; requires JetStream when decoupled.

    When DECOUPLE_GCLOUD is FALSE we provide a clear error instead of failing
    cryptically on attribute access.
    """
    token_params_is_stub = getattr(_token_params_ns, "_IS_STUB", False)
    engine_api_is_stub = getattr(engine_api, "_IS_STUB", False)
    if is_decoupled() and (token_params_is_stub or engine_api_is_stub):
      raise RuntimeError(
          "JetStream disabled by DECOUPLE_GCLOUD=TRUE or stubbed; get_tokenizer is unsupported. "
          "Unset DECOUPLE_GCLOUD or install JetStream to enable tokenizer functionality."
      )
    try:
      tokenizer_type_val = TokenizerType.DESCRIPTOR.values_by_name[self.config.tokenizer_type].number
      return TokenizerParameters(
          path=self.config.tokenizer_path,
          tokenizer_type=tokenizer_type_val,
          access_token=self.config.hf_access_token,
          use_chat_template=self.config.use_chat_template,
          extra_ids=0,
      )
    except KeyError as _:
      raise KeyError(f"Unsupported tokenizer type: {self.config.tokenizer_type}") from None

  def build_tokenizer(self, metadata: Any):  # return type depends on JetStream
    """Return a tokenizer"""
    token_params_is_stub = getattr(_token_params_ns, "_IS_STUB", False)
    engine_api_is_stub = getattr(engine_api, "_IS_STUB", False)
    if is_decoupled() and (token_params_is_stub or engine_api_is_stub):
      raise RuntimeError(
          "JetStream disabled by DECOUPLE_GCLOUD=TRUE or stubbed; build_tokenizer is unsupported. "
          "Unset DECOUPLE_GCLOUD or install JetStream to enable tokenizer functionality."
      )
    if metadata.tokenizer_type == TokenizerType.tiktoken:
      return token_utils.TikToken(metadata)
    elif metadata.tokenizer_type == TokenizerType.sentencepiece:
      return token_utils.SentencePieceTokenizer(metadata)
    elif metadata.tokenizer_type == TokenizerType.huggingface:
      tokenizer_model = token_utils.HuggingFaceTokenizer(metadata)
      if tokenizer_model.tokenizer.pad_token_id is None:
        if tokenizer_model.tokenizer.unk_token_id is not None:
          tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.unk_token_id
        else:
          print(f"Warning: setting pad_token_id to eos_token_id:{tokenizer_model.tokenizer.eos_token_id}")
          tokenizer_model.tokenizer.pad_token_id = tokenizer_model.tokenizer.eos_token_id
      return tokenizer_model
    else:
      raise ValueError(f"Unsupported tokenizer type: {metadata.tokenizer_type}")

  def init_decode_state(
      self,
      *args,  # pylint: disable=unused-argument
      rng: PRNGKeyType | None = None,
      **kwargs,  # pylint: disable=unused-argument
  ) -> DecodeState:
    """Initialises any state which a generation step transforms."""
    if rng is None:
      rng = jax.random.PRNGKey(0)
    page_state = None
    if self.config.attention == "paged" and self.page_manager is not None:
      page_state = self.page_manager.get_initial_page_state()  # pytype: disable=attribute-error

    # pylint: disable=unused-argument
    def init(abstract_params, page_state):
      x = jnp.ones(
          (int(self.config.per_device_batch_size * self.mesh.size), 1),
          dtype=jnp.int32,
      )
      dummy_image = jnp.ones(
          mm_processor.get_dummy_image_shape_for_init(
              model_name=self.config.model_name, batch_size=self.config.per_device_batch_size
          ),
          dtype=jnp.int32,
      )
      dummy_audio = jnp.ones(
          mm_processor.get_dummy_audio_shape_for_init(self.config),
          dtype=jnp.float32,
      )
      _, cache = self.model.apply(
          abstract_params,
          x,
          x,
          encoder_images=dummy_image if self.config.use_multimodal else None,
          encoder_audios=dummy_audio if self.config.use_audio else None,
          enable_dropout=False,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"params": rng},
          mutable=["cache"],
          page_state=page_state,
          slot=0,
      )

      next_pos = jnp.zeros(
          (int(self.config.per_device_batch_size * self.mesh.size), 1),
          dtype=jnp.int32,
      )
      generated_tokens = jnp.zeros(
          (int(self.config.per_device_batch_size * self.mesh.size), 1),
          dtype=jnp.int32,
      )
      tokens = jnp.zeros(
          (int(self.config.per_device_batch_size * self.mesh.size), 1),
          dtype=jnp.int32,
      )
      token_logp = jnp.zeros(
          (int(self.config.per_device_batch_size * self.mesh.size), 1),
          dtype=jnp.float32,
      )
      return {
          "logits": jnp.zeros(
              (
                  int(self.config.per_device_batch_size * self.mesh.size),
                  1,
                  self.config.vocab_size,
              )
          ),
          "cache": cache["cache"],
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": tokens,
          "token_logp": token_logp,
      }

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      abstract_outputs = jax.eval_shape(init, self.abstract_params, page_state)
    logical_annotations = nn.get_partition_spec(abstract_outputs)

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mesh_annotations = nn.logical_to_mesh(logical_annotations)

    shardings = jax.tree_util.tree_map(
        lambda mesh_annotation: jax.sharding.NamedSharding(self._mesh, mesh_annotation),
        mesh_annotations,
    )

    @functools.partial(jax.jit, out_shardings=shardings)
    def initialize():
      return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

    init_state = initialize()
    cache = init_state["cache"]

    def is_lp(k):
      return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

    self.kv_cache_annotations_named = jax.tree_util.tree_map(lambda x: tuple(x.names), cache, is_leaf=is_lp)
    zeroed = max_utils.unbox_logicallypartioned(init_state)
    return zeroed

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return int(self.config.per_device_batch_size * self.mesh.size)

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return int(self.config.max_prefill_predict_length)

  @property
  def use_chunked_prefill(self) -> bool:
    """Whether to use chunked prefill."""
    return self.config.use_chunked_prefill

  @property
  def prefill_chunk_size(self) -> int:
    """Prefill chunk size."""
    return int(self.config.prefill_chunk_size)

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
  if base_engine.model.quant:
    engine.model.quant.quant_mode = base_engine.model.quant.quant_mode
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations(engine.model, engine.config, rng, engine.mesh)  # pylint: disable=protected-access
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine.mesh, x),
      engine.kv_cache_annotations,  # pylint: disable=protected-access
  )


def create_engine_from_config_flags(
    maxengine_config_filepath, batch_size, max_prefill_predict_length, max_target_length, args_str
):
  """Create new MaxEngine instance with given batch_size, prefill and target lengths, and any config
  params provided through `args_str`.
  """
  # batch and cache related
  args = {
      "scan_layers": "false",
      "async_checkpointing": "false",
      "ici_fsdp_parallelism": "1",
      "ici_autoregressive_parallelism": "1",
      "ici_tensor_parallelism": "-1",
      "weight_dtype": "bfloat16",
      "attention": "dot_product",
      "max_prefill_predict_length": f"{max_prefill_predict_length}",
      "max_target_length": f"{max_target_length}",
      "per_device_batch_size": f"{batch_size}",
  }

  print(f"Command line args: {args_str}")
  cmd_args = args_str.split(" ")
  for cmd_arg in cmd_args:
    if cmd_arg:
      k, v = cmd_arg.split("=")
      args[k.strip()] = v.strip()
  assert "load_parameters_path" in args, "load_parameters_path must be defined"
  if maxengine_config_filepath is None:
    maxengine_config_filepath = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
  updated_args = [
      os.path.join(MAXTEXT_PKG_DIR, "inference", "maxengine", "maxengine_server.py"),
      maxengine_config_filepath,
  ]
  for k, v in args.items():
    option = f"{k}={v}"
    updated_args.append(option)
  print(f"Invoking maxengine with args:\n \t{updated_args}")
  cfg = pyconfig.initialize(updated_args)
  engine = MaxEngine(cfg)
  return engine
