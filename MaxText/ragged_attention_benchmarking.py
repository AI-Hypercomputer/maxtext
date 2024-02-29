"""Tests for ragged_attention."""

from absl.testing import absltest
import datetime
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from kernels.ragged_attention import ragged_mqa, mqa_reference

seed = 0
batch_size = 96
num_heads = 16
seq_len = 2048
head_dim = 256
dtype = jnp.bfloat16
lengths_size = 128
lengths_of_cache_entries = jnp.array([lengths_size] * batch_size, dtype=jnp.int32)
block_size = 128
profiling_out_dir = f"/home/rwitten/profiling/"

def xprof_profile_loop(f, funcname, warmups=100, runs=1000):
  '''Simple utility to profile a function for multiple runs'''
  jax.block_until_ready(f()) #warm it up!
  for _ in range(warmups):
    x = f()
  jax.block_until_ready(x)

  profile_trace_location = profiling_out_dir + f"{runs}runs_{lengths_size}lengths_{block_size}bk_{seq_len}seqlen_bs{batch_size}/{funcname}"
  print(f"Saving profile trace to {profile_trace_location}")
  jax.profiler.start_trace(profile_trace_location)
  s = datetime.datetime.now()
  for _ in range(runs):
    y = f()
  jax.block_until_ready(y)
  e = datetime.datetime.now()

  jax.profiler.stop_trace()

  print(f"{funcname} time:", (e-s).total_seconds())



class RaggedAttentionTest(absltest.TestCase):

  def _assert_allclose(self, a, b, **kwargs):
    if a.dtype == jnp.bfloat16:
      a = a.astype(np.float32)
    if b.dtype == jnp.bfloat16:
      b = b.astype(np.float32)
    np.testing.assert_allclose(a, b, **kwargs)

  def test_ragged_mqa(self):
    key = random.PRNGKey(seed)
    k1, k2, k3 = random.split(key, 3)
    q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=dtype) 
    k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=dtype)  
    v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=dtype)  

    # Dummy profiling due to issues caused to the first profiling section
    jax.profiler.start_trace(f"/home/rwitten/profiling/dummy")
    jax.profiler.stop_trace()

    # xprof_profile_loop(lambda : mqa_reference(q, k, v, lengths_of_cache_entries), "mqa_ref")
    # xprof_profile_loop(lambda : ragged_mqa(q, k, v, lengths_of_cache_entries, bk=bk), "ragged_mqa")
    # out_mqa_ref, (logits_mqa_ref, denom_mqa_ref) = mqa_reference(q, k, v, lengths_of_cache_entries)
    # out_ragged_mqa, (logits_ragged_mqa, denom_ragged_mqa) = ragged_mqa(q, k, v, lengths_of_cache_entries, bk=bk)

    q_mha = jnp.expand_dims(q, axis=1)  # (b,n,d) -> (b,1,n,d)
    q_mha = jnp.reshape(q_mha, (batch_size, num_heads, head_dim))

    k_mha = jnp.expand_dims(k, axis=2)  # (b,s,d) -> (b,s,1,d)
    v_mha = jnp.expand_dims(v, axis=2)  # (b,s,d) -> (b,s,1,d) 

    k_mha *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)
    v_mha *= jnp.ones((batch_size, seq_len, num_heads, head_dim), dtype=dtype)  # (b,s,1,d) -> (b,s,n,d)

    # Call vmapped reference MQA
    vmap_mqa_ref = jax.jit(jax.vmap(mqa_reference, in_axes=[None, 2, 2, None]),static_argnames=["mask_value"])

    xprof_profile_loop(lambda : vmap_mqa_ref(q_mha, k_mha, v_mha, lengths_of_cache_entries), "vmap_mqa_ref")
    # vmap_mqa_ref, (vmap_mqa_ref_logits, vmap_mqa_ref_denom) = vmap_mqa_ref(q_mha, k_mha, v_mha, lengths_of_cache_entries) 

    k_mha = jnp.swapaxes(k_mha, 1, 2)   # 4, 8, 128, 256
    v_mha = jnp.swapaxes(v_mha, 1, 2)   # 4, 8, 128, 256

    # Call vmapped ragged attention MQA
    vmap_ragged = jax.jit(jax.vmap(ragged_mqa, in_axes=[None, 1, 1, None]), static_argnames=["bk", "mask_value"])

    xprof_profile_loop(lambda : vmap_ragged(q_mha, k_mha, v_mha, lengths_of_cache_entries), "vmap_ragged_mqa")
    # vmap_ragged_output, (vmap_ragged_logits, vmap_ragged_denom) = vmap_ragged(q_mha, k_mha, v_mha, lengths_of_cache_entries)

    # Compare the pallas ragged MQA kernel results with that of the MHA implementation
    # self._assert_allclose(out_mqa_ref, vmap_mqa_ref[0], atol=3e-3, rtol=3e-3)
    # self._assert_allclose(out_mqa_ref, vmap_ragged_output[0], atol=3e-3, rtol=3e-3)

if __name__ == "__main__":
  absltest.main()
