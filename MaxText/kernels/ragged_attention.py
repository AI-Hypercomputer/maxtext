"""Kernels for ragged attention."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import common_types

from jax.experimental import shard_map


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
BATCH = common_types.BATCH
shard_map = shard_map.shard_map


@functools.partial(jax.jit, static_argnames=["mask_value"])
def mqa_reference(
    q: jax.Array,       
    k: jax.Array,       
    v: jax.Array,       
    lengths: jax.Array, 
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Multi query attention reference.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, head_dim] jax.Array.
    v: A [batch_size, seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """

  # loads the entire batch to its full size each time
  # expects all the entries to be left-aligned, but not necessarily any padding

  # computes the logits by multiplying q and k
  logits = jnp.einsum(
      "bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32)
  )

  # Creates the mask based on the sequence length of each entry in `lengths` in k
  mask = jnp.arange(k.shape[1])[None] < lengths[:, None]        

  # Change the logits by making all non-used entries a very negative number
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None]   

  # Get logit info
  logits_max = logits.max(axis=-1)                              
  unnormalized = jnp.exp(logits - logits_max[..., None])        
  denominator = unnormalized.sum(axis=-1)                       
  o = (                                                         
      jnp.einsum("bht,btd->bhd", unnormalized.astype(v.dtype), v)
      / denominator[..., None]
  )
  return o, (logits_max, denominator)


def ragged_flash_attention_kernel(
    lengths_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    *,
    bk: int,
    mask_value: float,
):
  """Pallas kernel for flash attention."""
  b, i = pl.program_id(0), pl.program_id(1)

  @pl.when(i == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  length = lengths_ref[b]

  @pl.when(i * bk < length)
  def run():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m_prev, l_prev = m_ref[...], l_ref[...]

    # multiply q and k
    qk = lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    
    mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    o_curr_times_l_curr = jnp.dot(s_curr, v)

    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

    m_ref[...], l_ref[...] = m_next, l_next_safe
    o_ref[...] = (
        (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
    ).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=["bk", "mask_value"])
def ragged_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *, 
    bk: int = 128,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  batch_size, num_heads, head_dim = q.shape 
  assert lengths.shape == (batch_size,)
  assert lengths.dtype == jnp.int32
  seq_len = k.shape[1]  

  def compute_ragged_block_indices(b, i, lengths_ref):
    length = lengths_ref[b]
    not_done = i * bk < length

    # Compare current batch index(?) to see if we are at the last or not
    am_last_batch = b == batch_size - 1

    # Compute the last good block based on the length of the current batch index and the block size
    last_good_block = lax.div(length, bk) - 1

    # Compute next batch index based on 1) if we are at the end of the current batch index's length, 
    #                               and 2) if we are the last batch index or not 
    b_next = jnp.where(not_done, b, jnp.where(am_last_batch, b, b + 1))

    # Compute next block index based on if we are processing the last block of the current batch index's length or not
    i_next = jnp.where(not_done, i, jnp.where(am_last_batch, last_good_block, 0))
    return b_next, i_next, 0

  out, m, l = pl.pallas_call(
      functools.partial(
          ragged_flash_attention_kernel,
          bk=bk,
          mask_value=mask_value,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec(lambda b, i, _: (b, 0, 0), (None, num_heads, head_dim)),
              pl.BlockSpec(compute_ragged_block_indices, (None, bk, head_dim)),
              pl.BlockSpec(compute_ragged_block_indices, (None, bk, head_dim)),
          ],
          out_specs=[
              pl.BlockSpec(lambda b, i, _: (b, 0, 0), (None, num_heads, head_dim)),
              pl.BlockSpec(lambda b, i, _: (b, 0, 0), (None, num_heads, head_dim)),
              pl.BlockSpec(lambda b, i, _: (b, 0, 0), (None, num_heads, head_dim)),
          ],
          grid=(batch_size, seq_len // bk),
      ),
      mosaic_params=dict(dimension_semantics=("parallel", "arbitrary")),
      out_shape=[
          q,
          jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
      ],
  )(lengths, q, k, v)
  return out, (m[..., 0], l[..., 0])
