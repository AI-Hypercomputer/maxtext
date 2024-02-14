"""Simple test engine for the API described in go/inference_engine_v2_api.

Contains simple functions that we can hand calculate the desired outcome of.

Prefill: Doubles the sequence by multiplying it with an integer weight.
Insert: Writes this sequence into a cache row.
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight.

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""
import sys
sys.path.append("/home/rwitten/disaggregation/")

import functools
from typing import Any, Optional, Tuple

from flax import struct
import jax
from layers import models, quantizations

import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import os

import common_types
from inference_engine import engine_api
from inference_engine import tokenizer_pb2

import max_utils
import inference_utils


Prefix = Any
Params = Any



@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""
  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array


class TestEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  Wiz efficient serving infrastructure.
  """

  def __init__(self, config):
    self.config = config
    self.rng = jax.random.PRNGKey(0)

    # Mesh definition
    devices_array = max_utils.create_device_mesh(config)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    # Model and Optimizer definition
    quant = quantizations.configure_quantization(config)
    self.model = models.Transformer(config, mesh = self._mesh, quant=quant)
    self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

  def load_params(self) -> Params:
    state, self.state_mesh_annotations = max_utils.setup_decode_state(
      self.model, self.config, self.rng, self._mesh, None
    )
    self.abstract_params = jax.tree_map(lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), state.params)
    self.kv_cache_annotations = max_utils.get_kv_cache_annotations(self.model, self.config, self.rng, self._mesh)
    return state.params


  @functools.partial(jax.jit, static_argnums=(0,))
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Prefix:
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
  
    input = jnp.expand_dims(padded_tokens, 0) # [BATCh, SEQUENCE]
    positions = jnp.expand_dims(jnp.arange(0, input.shape[1]), 0)

    
    zero_to_n = jnp.arange(0, padded_tokens.shape[0])
    ones_to_keep = zero_to_n < true_length
    one_d_output = ones_to_keep * common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    flat_logits, new_vars = self.model.apply(
      {
          "params": params
      },
      input,
      positions,
      decoder_segment_ids=sequence_indicator,
      enable_dropout=False,
      model_mode=common_types.MODEL_MODE_PREFILL,
      rngs={'params': self.rng},
      mutable=["cache"]
    )

    next_pos = jnp.full((1,1), true_length, dtype = jnp.int32)
    selected_logits = jax.lax.dynamic_slice(flat_logits, (0, true_length-1,0), (flat_logits.shape[0], 1, flat_logits.shape[2]))
    return {"logits" : selected_logits, "cache" : new_vars['cache'], "next_pos" : next_pos}

  

  @functools.partial(jax.jit, static_argnums=(0,))
  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    previous_logits = decode_state['logits']

    new_token = inference_utils.sampling(previous_logits, self.rng, self.config.decode_sampling_strategy,\
                                       topk=self.config.decode_sampling_top_k, nucleus_topp=self.config.decode_sampling_nucleus_p,
                                       temperature=self.config.decode_sampling_temperature)
    out_logits, new_vars = self.model.apply(
      {
          "params": params,
          "cache": decode_state['cache']
      },
      new_token,
      decode_state['next_pos'],
      enable_dropout=False,
      model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
      rngs={'params': self.rng},
      mutable=["cache"]
    )

    result = engine_api.ResultTokens(
        data=jnp.concatenate((new_token, decode_state["next_pos"]), axis=1),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, 1),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(1, 1),
        # And lengths is rank 1.
        length_idx=(1, 2), 
        samples_per_slot=1,
    )

    return {"logits" : out_logits, "cache" : new_vars["cache"], "next_pos" : decode_state["next_pos"]+1}, result

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    target_idx = 0 if self.config.scan_layers == False else self.config.param_scan_axis
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)

    def copy(partial_cache, full_cache):
      return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, target_idx)
        
    inserted_cache = jax.tree_map(copy, unboxed_prefix['cache'], decode_state['cache'])
    inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state['logits'], unboxed_prefix['logits'], slot, 0)
    inserted_next_pos = jax.lax.dynamic_update_index_in_dim(decode_state['next_pos'], unboxed_prefix['next_pos'], slot, 0)

    return {'logits' : inserted_logits, 'cache' : inserted_cache, 'next_pos' : inserted_next_pos }

  def get_prefix_destination_sharding(self) -> Any:
    return jax.sharding.NamedSharding(
        mesh=self.mesh, spec=jax.sharding.PartitionSpec()
    )

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path=os.path.join(self.config.assets_path, self.config.vocab_relative_path), extra_ids=0)

  def init_decode_state(self) -> DecodeState:
    """Initialises any state which a generation step transforms."""

    def init(abstract_params):
      x = jnp.ones( (int(self.config.per_device_batch_size * jax.device_count()), self.config.max_prefill_predict_length),
                   dtype=jnp.int32)
      _, cache = self.model.apply(
        {
            "params": abstract_params
        },
        x,
        x,
        decoder_segment_ids=jnp.zeros(x.shape, dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR,
        enable_dropout=False,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={'params': self.rng},
        mutable=["cache"]
      )

      return {"logits" : jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1, self.config.vocab_size)),
              "cache" : cache["cache"],
              "next_pos" : jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
              }

    abstract_outputs = jax.eval_shape(init, self.abstract_params)
    logical_annotations = nn.get_partition_spec(abstract_outputs)

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mesh_annotations = nn.logical_to_mesh(logical_annotations)
    
    shardings = jax.tree_map(lambda mesh_annotation : jax.sharding.NamedSharding(self._mesh, mesh_annotation), mesh_annotations)
    
    @functools.partial(jax.jit, out_shardings = shardings)
    def initialize():
      return jax.tree_map( lambda x : jnp.zeros(x.shape, x.dtype), abstract_outputs)
    
    zeroed = max_utils.unbox_logicallypartioned(initialize())
    return zeroed

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return self.config.per_device_batch_size * jax.device_count()

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return self.max_prefill_predict_length

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
