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
from jax.experimental import mesh_utils
import jax.numpy as jnp

from inference_engine import engine_api
from inference_engine import tokenizer_pb2


Params = jax.Array  # [1,].
Prefix = jax.Array  # [batch,] of strings with different lengths.


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

  def __init__(self, batch_size: int, cache_length: int, weight: float):
    self.prefill_cache_batch = batch_size
    self.generate_cache_batch = batch_size
    self.cache_length = cache_length
    self.weight = weight
    self._mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((1, 1, 1), jax.devices()), ('x', 'y', 'z')
    )

  def load_params(self) -> Params:
    """Loads model weights."""
    # An integer, used to multiply inputs.
    return jnp.array([self.weight], dtype=jnp.float32)

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
    if existing_prefix is not None:
      raise NotImplementedError
    del true_length
    assert padded_tokens.ndim == 1
    # Wait to simulate model step time.
    fake_size = 4096
    fake_work = jnp.ones((fake_size, fake_size)) @ jnp.ones(
        (fake_size, fake_size)
    )
    # Do some fake work that isn't eliminated by dead code elimination (DCE).
    params = params + fake_work.mean() - fake_work.mean()
    return padded_tokens[None, :] * params

  @functools.partial(jax.jit, static_argnums=(0,))
  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel."""
    prefill_cache, generate_cache, generate_cache_index, generate_lengths = (
        decode_state.prefill_cache,
        decode_state.generate_cache,
        decode_state.generate_cache_index,
        decode_state.generate_lengths,
    )
    # Sum each row of prefill cache and generate cache to produce new timestep,
    # multiply by params.
    l_iota = jax.lax.broadcasted_iota(
        jnp.int32,
        (self.generate_cache_batch, self.cache_length),
        dimension=1,
    )

    # The generate cache should be circular and right aligned.
    # TODO(sholto): Do we need a left aligned one to test spec sampling?
    # Don't need the + 1 you normally would, because we don't provide a
    # token from prefill in the dummy.
    # This iota and masking is to allow for a cicular cache.
    length_mask = (
        -(l_iota - generate_cache_index + 1) % self.cache_length
    ) <= generate_lengths[:, None]
    length_masked_gen_cache = generate_cache * length_mask
    new_timestep = (
        prefill_cache.sum(axis=-1)
        + (length_masked_gen_cache.sum(axis=-1) / params)
    )[:, jnp.newaxis]
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        generate_cache, new_timestep, start_index=generate_cache_index, axis=1
    )
    generate_cache_index = (generate_cache_index + 1) % self.cache_length
    # Wait to simulate model step time.
    fake_size = 4096
    fake_work = jnp.ones((fake_size, fake_size)) @ jnp.ones(
        (fake_size, fake_size)
    )
    # Do some fake work that isn't eliminated by dead code elimination (DCE).
    generate_cache = generate_cache + fake_work.mean() - fake_work.mean()
    new_lengths = generate_lengths + 1
    speculations = new_timestep.shape[1]
    # Concatenates the tokens, their validity and the lengths of each sequence
    # into one tensor so that copy operations are faster on pathways.
    token_data = jnp.concatenate(
        [new_timestep, jnp.ones_like(new_timestep), new_lengths[:, None]],
        axis=-1,
    )
    return DecodeState(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_cache_index=generate_cache_index,
        generate_lengths=new_lengths,
    ), engine_api.ResultTokens(
        data=token_data.astype(jnp.int32),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=self.generate_cache_batch // self.prefill_cache_batch,
    )

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Adds `prefix` into `decode_state` at `slot`."""
    # [B, T], [T,] -> [B, T]
    prefill_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.prefill_cache, prefix, slot, axis=0
    )
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_cache, jnp.zeros((1, self.cache_length)),
        slot, axis=0
    )
    samples_per_slot = self.generate_cache_batch // self.prefill_cache_batch
    generate_lengths = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_lengths,
        jnp.zeros((samples_per_slot), dtype=jnp.int32),
        slot * samples_per_slot,
        axis=0,
    )
    return decode_state.replace(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_lengths=generate_lengths,
    )

  def get_prefix_destination_sharding(self) -> Any:
    return jax.sharding.NamedSharding(
        mesh=self.mesh, spec=jax.sharding.PartitionSpec()
    )

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path='test', extra_ids=0)

  def init_decode_state(self) -> DecodeState:
    """Initialises any state which a generation step transforms."""
    return DecodeState(
        prefill_cache=jnp.zeros(
            (self.prefill_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache=jnp.zeros(
            (self.generate_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache_index=0,
        generate_lengths=jnp.zeros(
            (self.generate_cache_batch), dtype=jnp.int32
        ),
    )

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return self.prefill_cache_batch

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return self.cache_length

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return self.generate_cache_batch // self.max_concurrent_decodes

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError
