from absl import flags
# flags.FLAGS.xla_enable_transpose_trace = False
from absl.testing import absltest
# from google3.learning.deepmind.jax.statix import statix

import jax
import jax.numpy as jnp
from jax import lax
import random as pyrandom
from jax import random
from jax.experimental import pallas as pl
from jax.experimental import shard_map as shmap
from jax.experimental.pallas import tpu as pltpu


import datetime
import functools
import numpy as np
import time

######################################

NUM_HEADS = 16
HEAD_DIM = 256
dtype = jnp.bfloat16
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
BATCH = 'activation_batch'
BLOCK_SIZE = 256
shard_map = shmap.shard_map


#######################################
seed = 0

def _assert_allclose(a, b, **kwargs):
  if a.dtype == jnp.bfloat16:
    a = a.astype(np.float32)
  if b.dtype == jnp.bfloat16:
    b = b.astype(np.float32)
  np.testing.assert_allclose(a, b, **kwargs)

def profile_loop(f, iters=10):
  '''Simple utility to profile a function for multiple runs'''
  for _ in range(iters):
    x = f()
  x[0].block_until_ready()

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
  logits = jnp.einsum(
      "bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  mask = jnp.arange(k.shape[1])[None] < lengths[:, None]
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None]
  logits_max = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - logits_max[..., None])
  denominator = unnormalized.sum(axis=-1)
  o = (
      jnp.einsum("bht,btd->bhd", unnormalized.astype(v.dtype), v)
      / denominator[..., None]
  )
  return o, (logits_max, denominator)


@pltpu.trace('trace.ragged_flash_attention_kernel')
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
  @pltpu.trace('trace.ragged_flash_attention_kernel.init')
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  length = lengths_ref[b]

  @pl.when(i * bk < length)
  @pltpu.trace('trace.ragged_flash_attention_kernel.run')
  def run():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m_prev, l_prev = m_ref[...], l_ref[...]

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
    bk: int = BLOCK_SIZE,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  batch_size, num_heads, head_dim = q.shape
  assert lengths.shape == (batch_size,)
  assert lengths.dtype == jnp.int32
  seq_len = k.shape[1]

  def _compute_ragged_block_indices(b, i, lengths_ref):
    length = lengths_ref[b]
    not_done = i * bk < length
    am_last_batch = b == batch_size - 1
    last_good_block = lax.div(length, bk) - 1

    b_next = jnp.where(not_done, b, jnp.where(am_last_batch, b, b + 1))
    i_next = jnp.where(
        not_done, i, jnp.where(am_last_batch, last_good_block, 0)
    )
    return b_next, i_next

  def kv_index_map(b, i, lengths_ref):
    b_next, i_next = _compute_ragged_block_indices(b, i, lengths_ref)
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
              pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
              pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
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

def run_ref_mqa(batch_size, seq_len, cache_entry_lengths, num_heads, head_dim, warmup=100, iters=1000):
  key = random.PRNGKey(seed)
  k1, k2, k3 = random.split(key, 3)

  q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=dtype)
  k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=dtype)
  v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=dtype)

  profile_loop(lambda : mqa_reference(q, k, v, cache_entry_lengths), iters=warmup)

  start_time = time.perf_counter()
  profile_loop(lambda : mqa_reference(q, k, v, cache_entry_lengths), iters=iters)
  end_time = time.perf_counter()
  return (end_time - start_time) / iters / 1000
  # print(f"ref_mqa: Mean time to run iteration: {(end_time - start_time) * 1000 / iters:.3f}ms, "
  #       f"{iters}, {end_time-start_time}")

def run_ragged_mqa(batch_size, seq_len, cache_entry_lengths, num_heads, head_dim, bk=256, warmup=100, iters=1000):
  key = random.PRNGKey(seed)
  k1, k2, k3 = random.split(key, 3)

  q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=dtype)
  k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=dtype)
  v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=dtype)

  profile_loop(lambda : ragged_mqa(q, k, v, cache_entry_lengths, bk=bk), iters=warmup)

  start_time = time.perf_counter()
  profile_loop(lambda : ragged_mqa(q, k, v, cache_entry_lengths, bk=bk), iters=iters)
  end_time = time.perf_counter()
  return (end_time - start_time) / iters / 1000
  # print(f"ragged_mqa: Mean time to run iteration: {(end_time - start_time) * 1000 / iters:.3f}ms, "
  #       f"{iters}, {end_time-start_time}")


def run_vmapped_ref_mqa(batch_size, seq_len, cache_entry_lengths, num_heads, head_dim, warmup=100, iters=1000):
  key = random.PRNGKey(seed)
  k1, k2, k3 = random.split(key, 3)
  q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=dtype)
  k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=dtype)
  v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=dtype)

  q = jnp.expand_dims(q, axis=1)  # (b,n,d) -> (b,1,n,d)
  q = jnp.reshape(q, (batch_size, num_heads, head_dim))

  k = jnp.expand_dims(k, axis=2)  # (b,s,d) -> (b,s,1,d)
  v = jnp.expand_dims(v, axis=2)  # (b,s,d) -> (b,s,1,d)

  k *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)
  v *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)

  vmap_mqa_ref = jax.jit(jax.vmap(mqa_reference, in_axes=[None, 2, 2, None]))

  profile_loop(lambda : vmap_mqa_ref(q, k, v, cache_entry_lengths), iters=warmup)

  start_time = time.perf_counter()
  profile_loop(lambda : vmap_mqa_ref(q, k, v, cache_entry_lengths), iters=iters)
  end_time = time.perf_counter()
  return (end_time - start_time) / iters / 1000


def run_vmapped_ragged_mqa(batch_size, seq_len, cache_entry_lengths, num_heads, head_dim, warmup=100, iters=1000):
  key = random.PRNGKey(seed)
  k1, k2, k3 = random.split(key, 3)
  q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=dtype)
  k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=dtype)
  v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=dtype)

  q = jnp.expand_dims(q, axis=1)  # (b,n,d) -> (b,1,n,d)
  q = jnp.reshape(q, (batch_size, num_heads, head_dim))

  k = jnp.expand_dims(k, axis=2)  # (b,s,d) -> (b,s,1,d)
  v = jnp.expand_dims(v, axis=2)  # (b,s,d) -> (b,s,1,d)

  k *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)
  v *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)

  k = jnp.swapaxes(k, 1, 2)
  v = jnp.swapaxes(v, 1, 2)

  vmap_ragged_mqa = jax.jit(jax.vmap(ragged_mqa, in_axes=[None, 1, 1, None]))
  profile_loop(lambda : vmap_ragged_mqa(q, k, v, cache_entry_lengths), iters=warmup)

  start_time = time.perf_counter()
  profile_loop(lambda : vmap_ragged_mqa(q, k, v, cache_entry_lengths), iters=iters)
  end_time = time.perf_counter()
  return (end_time - start_time) / iters / 1000


def create_cache_entries(batch_size, entry_len_mean, entry_len_stddev):
  entries = []
  for _ in range(batch_size):
    entries.append(int(pyrandom.gauss(entry_len_mean, entry_len_stddev))) 
  return jnp.array(entries, dtype=jnp.int32)


def get_cache_usage(cache_entries, block_size, max_cache_length):
  blocks_used = int(sum((cache_entries // block_size) + 1))
  total_cache_blocks = cache_entries.shape[0] * max_cache_length // block_size
  block_usage = blocks_used / total_cache_blocks

  total_tokens = int(sum(cache_entries))
  token_usage = total_tokens / (cache_entries.shape[0] * max_cache_length)
  return block_usage, token_usage


# block_sizes = [256]

max_cache_lengths = [2048]
batch_sizes = [4, 8, 16, 24,] #32, 48, 64, 96, 128]
# block_sizes = [256]


print("batch_size, max_cache_length, block_size, mean_length, block_usage, token_usage, dtype, ref_mqa_ms, ragged_mqa_ms, vmapped_ref_mqa_ms, vmapped_ragged_mqa_ms")
for mcl in max_cache_lengths:
  # for bk in block_sizes: 
  # desired_lengths = [BLOCK_SIZE * i for i in range(1, (mcl // BLOCK_SIZE) + 1, 1)]
  desired_lengths = [BLOCK_SIZE]
  for bs in batch_sizes: 
    for dl in desired_lengths:
      cache_entry_lengths = jnp.array([128, 250, 260, 500] * int(bs // 4), dtype=jnp.int32)
      # cache_entry_lengths = create_cache_entries(bs, dl, 0)
      bu, tu = get_cache_usage(cache_entry_lengths, bs, mcl)
      a = run_ref_mqa(bs, mcl, cache_entry_lengths, NUM_HEADS, HEAD_DIM)
      b = run_ragged_mqa(bs, mcl, cache_entry_lengths, NUM_HEADS, HEAD_DIM)
      c = run_vmapped_ref_mqa(bs, mcl, cache_entry_lengths, NUM_HEADS, HEAD_DIM)
      d = run_vmapped_ragged_mqa(bs, mcl, cache_entry_lengths, NUM_HEADS, HEAD_DIM)
      print(f"{bs}, {mcl}, {BLOCK_SIZE}, {dl}, {bu:.3}, {tu:.3}, {a:.3}, {b:.3}, {c:.3}, {d:.3}")
