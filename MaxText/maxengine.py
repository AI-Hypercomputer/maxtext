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
from typing import Any, Optional, Tuple

import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax import struct

from layers import models, quantizations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import common_types
from jetstream.engine import engine_api
from jetstream.engine import tokenizer_pb2
from jetstream.engine import tokenizer_api
from jetstream.engine import token_utils

import max_utils
import inference_utils
import pyconfig


Prefix = Any
Params = Any


@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generated_token: jax.Array


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

  def __init__(self, config):
    self.config = config
    self.rng = jax.random.PRNGKey(0)

    # Mesh definition
    devices_array = max_utils.create_device_mesh(config)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    # Model and Optimizer definition
    quant = quantizations.configure_quantization(config)
    self.model = models.Transformer(config, mesh=self._mesh, quant=quant)
    self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

    self.abstract_params = None
    self.kv_cache_annotations = None
    self.kv_cache_annotations_named = None
    self.kv_cache_shardings = None
    self.state_mesh_annotations = None

  def load_params(self, *args, **kwargs) -> Params:
    """Load Parameters, typically from GCS"""
    # pylint: disable=unused-argument

    if self.model.quant and self.config.checkpoint_is_quantized:
      print("Loading from the quantized checkpoint...")
      self.model.quant.quant_mode = quantizations.get_quant_mode("serve")

    state, self.state_mesh_annotations = max_utils.setup_decode_state(self.model, self.config, self.rng, self._mesh, None)
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), state.params
    )
    self.kv_cache_annotations = max_utils.get_kv_cache_annotations(self.model, self.config, self.rng, self._mesh)
    self.kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x), self.kv_cache_annotations
    )

    if self.model.quant and not self.config.checkpoint_is_quantized:
      params = self.quantize_params(state)
    else:
      params = state.params
    max_utils.print_mem_stats("After load_params")
    return params

  def quantize_params(self, state):
    """Forward pass to quantize decode params."""
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

    _, new_vars = model_apply(state.params, self.rng)
    # Remove param values which have corresponding qtensors in aqt to save memory.
    params = {}
    params["aqt"] = new_vars["aqt"]
    params["params"] = quantizations.remove_quantized_params(state.params["params"], new_vars["aqt"])
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), params
    )
    max_utils.save_quantized_checkpoint_if_configured(self.config, params)
    self.model.quant.quant_mode = quantizations.get_quant_mode("serve")
    return params

  @functools.partial(jax.jit, static_argnums=(0,))
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Computes a kv-cache for a new generate request.

    Args:
      params: Scalar multiplier.
      existing_prefix: If provided, represents a prefix that has already been
        processed by the underlying model.
      padded_tokens: Logically appended tokens to any existing prefix, this is
        what we compute prefill on.
      true_length: The real length of the tokens, pre-pad.
    Returns:
      kv_cache: For the resulting text.
    """
    if existing_prefix:
      raise ValueError("We don't know what to do with existing_prefix")

    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
    positions = jnp.expand_dims(jnp.arange(0, input_tokens.shape[1]), 0)

    zero_to_n = jnp.arange(0, padded_tokens.shape[0])
    ones_to_keep = zero_to_n < true_length
    one_d_output = ones_to_keep * common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flat_logits, new_vars = self.model.apply(
          params,
          input_tokens,
          positions,
          decoder_segment_ids=sequence_indicator,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": self.rng},
          mutable=["cache"],
      )

    next_pos = jnp.full((1, 1), true_length, dtype=jnp.int32)
    generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits, (0, true_length - 1, 0), (flat_logits.shape[0], 1, flat_logits.shape[2])
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    # sampling first token
    first_generated_token = inference_utils.sampling(
        selected_logits,
        self.rng,
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
        "cache": new_vars["cache"],
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_token,
    }, result

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def generate(self, params: Params, decode_state: DecodeState) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Run one generate step"""
    previous_token = decode_state["tokens"]

    # run one step generation
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out_logits, new_vars = self.model.apply(
          params | {"cache": decode_state["cache"]},
          previous_token,
          decode_state["next_pos"],
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"params": self.rng},
          mutable=["cache"],
      )

    out_logits = jax.lax.with_sharding_constraint(out_logits, self.replicated_sharding)
    new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], self.kv_cache_shardings)

    # sampling tokens
    new_token = inference_utils.sampling(
        out_logits,
        self.rng,
        self.config.decode_sampling_strategy,
        topk=self.config.decode_sampling_top_k,
        nucleus_topp=self.config.decode_sampling_nucleus_p,
        temperature=self.config.decode_sampling_temperature,
    )

    all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((new_token, all_valid, decode_state["generated_tokens"]), axis=1),
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
        "logits": out_logits,
        "cache": new_cache,
        "next_pos": decode_state["next_pos"] + 1,
        "generated_tokens": decode_state["generated_tokens"] + 1,
        "tokens": new_token,
    }, result

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
    """Insert into KV cache"""
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)

    def copy(path, partial_cache, full_cache, annotations):
      path_key = path[-1].key
      if path_key in ["cache_ar_index", "cached_ar_key", "cached_ar_value", "cached_ar_key_scale", "cached_ar_value_scale"]:
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
        copy, unboxed_prefix["cache"], decode_state["cache"], self.kv_cache_annotations_named
    )
    inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
    inserted_next_pos = jax.lax.dynamic_update_index_in_dim(decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
    inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
        decode_state["generated_tokens"], unboxed_prefix["generated_tokens"], slot, 0
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

  def get_prefix_destination_sharding(self) -> Any:
    return jax.sharding.NamedSharding(mesh=self.mesh, spec=jax.sharding.PartitionSpec())

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path=self.config.tokenizer_path, extra_ids=0)

  def build_tokenizer(self, metadata: tokenizer_pb2.TokenizerParameters) -> tokenizer_api.Tokenizer:
    """Return a tokenizer"""
    if "tiktoken" in metadata.path:
      return token_utils.TikToken(metadata)
    else:
      return token_utils.SentencePieceTokenizer(metadata)

  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    """Initialises any state which a generation step transforms."""

    # pylint: disable=unused-argument
    def init(abstract_params):
      x = jnp.ones(
          (int(self.config.per_device_batch_size * jax.device_count()), self.config.max_prefill_predict_length),
          dtype=jnp.int32,
      )
      _, cache = self.model.apply(
          abstract_params,
          x,
          x,
          decoder_segment_ids=jnp.zeros(x.shape, dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": self.rng},
          mutable=["cache"],
      )

      next_pos = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      generated_tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      return {
          "logits": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1, self.config.vocab_size)),
          "cache": cache["cache"],
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": tokens,
      }

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      abstract_outputs = jax.eval_shape(init, self.abstract_params)
    logical_annotations = nn.get_partition_spec(abstract_outputs)

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mesh_annotations = nn.logical_to_mesh(logical_annotations)

    shardings = jax.tree_util.tree_map(
        lambda mesh_annotation: jax.sharding.NamedSharding(self._mesh, mesh_annotation), mesh_annotations
    )

    @functools.partial(jax.jit, out_shardings=shardings)
    def initialize():
      return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

    cache = initialize()["cache"]

    def is_lp(k):
      return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

    self.kv_cache_annotations_named = jax.tree_util.tree_map(lambda x: tuple(x.names), cache, is_leaf=is_lp)
    del cache
    zeroed = max_utils.unbox_logicallypartioned(initialize())
    return zeroed

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


def set_engine_vars_from_base_engine(engine: engine_api.Engine, base_engine: engine_api.Engine):
  """Set internal vars from base_engine, which has already loaded the checkpoint and has sharding,
  mesh, and kv cache related vars set.
  """
  engine.model.quant.quant_mode = base_engine.model.quant.quant_mode
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = max_utils.get_kv_cache_annotations(engine.model, engine.config, engine.rng, engine._mesh)  # pylint: disable=protected-access
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine._mesh, x), engine.kv_cache_annotations  # pylint: disable=protected-access
  )


def create_engine_from_config_flags(batch_size, max_prefill_predict_length, max_target_length, args_str):
  """Create new MaxEngine instance with given batch_size, prefill and target lengths, and any config
  params provided through `args_str`.
  """
  args = []
  args.append("scan_layers=false")
  args.append("async_checkpointing=false")
  args.append("ici_fsdp_parallelism=1")
  args.append("ici_autoregressive_parallelism=1")
  args.append("ici_tensor_parallelism=-1")
  args.append("weight_dtype=bfloat16")
  args.append("attention=dot_product")
  # args.append("")

  # batch and cache related
  args.append(f"max_prefill_predict_length={max_prefill_predict_length}")
  args.append(f"max_target_length={max_target_length}")
  args.append(f"per_device_batch_size={batch_size}")

  args_str = "MaxText/maxengine_server.py configs/base.yml " + args_str
  cmd_args = [b for b in args_str.split(" ") if len(b) > 0]
  # Add at the end to cmd args override the default values set above
  args.extend(cmd_args)

  pyconfig.initialize(args)
  cfg = MaxEngineConfig(cp.deepcopy(pyconfig._config.keys))  # pylint: disable=protected-access
  engine = MaxEngine(cfg)
  return engine
