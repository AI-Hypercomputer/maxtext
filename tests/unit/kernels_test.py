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

"""Tests for kernels."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
from maxtext.kernels.attention.ragged_attention import ragged_gqa, ragged_mha, ragged_mqa, reference_gqa, reference_mha, reference_mqa
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as tokamax_splash_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as tokamax_splash_mask
import numpy as np
import pytest


class RaggedAttentionTest(unittest.TestCase):
  """Tests for ragged attention kernel."""

  batch_size = 4
  num_kv_heads = 8
  num_query_heads = 32
  max_prefill_predict_length = 256
  max_target_length = 512
  head_dim = 128

  dtype = jnp.float32

  @pytest.mark.tpu_only
  def test_ragged_mqa(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(k2, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    v = jax.random.normal(k3, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, _ = ragged_mqa(q, k, v, lengths)
    reference_out, _, _ = reference_mqa(q, k, v, lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )
    self.assertTrue(
        jnp.average(abs(ragged_out - reference_out)) < 1e-2,
        msg=f"Avg difference: {jnp.average(abs(ragged_out - reference_out))} > 1e-2",
    )

  @pytest.mark.tpu_only
  def test_ragged_mha(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.num_query_heads, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(
        k2, (self.batch_size, self.max_target_length, self.num_query_heads, self.head_dim), dtype=self.dtype
    )
    v = jax.random.normal(
        k3, (self.batch_size, self.max_target_length, self.num_query_heads, self.head_dim), dtype=self.dtype
    )
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, ragged_denom = ragged_mha(q, k, v, lengths)
    ragged_out = ragged_out / ragged_denom
    reference_out, _, _ = reference_mha(q, k, v, lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )
    self.assertTrue(
        jnp.average(abs(ragged_out - reference_out)) < 1e-2,
        msg=f"Avg difference: {jnp.average(abs(ragged_out - reference_out))} > 1e-2",
    )

  @pytest.mark.tpu_only
  def test_ragged_gqa(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.num_query_heads, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(
        k2, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype
    )
    v = jax.random.normal(
        k3, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype
    )
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, ragged_denom = ragged_gqa(q, k, v, lengths)
    ragged_out = ragged_out / ragged_denom
    reference_out, _, _ = reference_gqa(jnp.squeeze(q), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2), lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )
    self.assertTrue(
        jnp.average(abs(ragged_out - reference_out)) < 1e-2,
        msg=f"Avg difference: {jnp.average(abs(ragged_out - reference_out))} > 1e-2",
    )


class RaggedAttentionCpuTest(unittest.TestCase):
  """Tests for ragged attention kernel on CPU (interpret mode)."""

  batch_size = 2  # Smaller size for faster CPU interpretation
  num_kv_heads = 2
  num_query_heads = 4
  max_target_length = 32  # Smaller size for CPU
  head_dim = 32  # Smaller size for CPU

  dtype = jnp.float32

  def test_ragged_mqa_cpu(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(k2, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    v = jax.random.normal(k3, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, _ = ragged_mqa(q, k, v, lengths, block_size=16, interpret=True)
    reference_out, _, _ = reference_mqa(q, k, v, lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )

  def test_ragged_mha_cpu(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.num_query_heads, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(
        k2, (self.batch_size, self.max_target_length, self.num_query_heads, self.head_dim), dtype=self.dtype
    )
    v = jax.random.normal(
        k3, (self.batch_size, self.max_target_length, self.num_query_heads, self.head_dim), dtype=self.dtype
    )
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, ragged_denom = ragged_mha(q, k, v, lengths, block_size=16, interpret=True)
    ragged_out = ragged_out / ragged_denom
    reference_out, _, _ = reference_mha(q, k, v, lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )

  def test_ragged_gqa_cpu(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (self.batch_size, 1, self.num_query_heads, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(
        k2, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype
    )
    v = jax.random.normal(
        k3, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype
    )
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, _, ragged_denom = ragged_gqa(q, k, v, lengths, block_size=16, interpret=True)
    ragged_out = ragged_out / ragged_denom
    reference_out, _, _ = reference_gqa(jnp.squeeze(q), jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2), lengths)
    self.assertTrue(
        jnp.max(abs(ragged_out - reference_out)) < 1.5e-1,
        msg=f"Max difference: {jnp.max(abs(ragged_out - reference_out))} > 1e-1",
    )


class SplashAttentionDkvMegacoreCpuTest(unittest.TestCase):
  """Tests for the splash attention dkv backward kv-head group split on CPU (interpret mode)."""

  num_kv_heads = 2
  num_query_heads = 4
  seq_len = 256  # Two blocks per grid dimension at the default block size of 128
  head_dim = 128

  dtype = jnp.float32

  def _grads(self, bwd_dkv_megacore, num_kv_heads, interpret=True):
    """Returns (dq, dk, dv) of the causal splash kernel for the given flag and KV head count."""
    config = dataclasses.replace(
        tokamax_splash_kernel.SplashConfig.get_default(),
        interpret=interpret,
        bwd_dkv_megacore=bwd_dkv_megacore,
    )
    mask = tokamax_splash_mask.CausalMask((self.seq_len, self.seq_len))
    kernel = tokamax_splash_kernel.make_splash_mha_single_device(mask, config=config)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k1, (self.num_query_heads, self.seq_len, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(k2, (num_kv_heads, self.seq_len, self.head_dim), dtype=self.dtype)
    v = jax.random.normal(k3, (num_kv_heads, self.seq_len, self.head_dim), dtype=self.dtype)
    return jax.grad(lambda q, k, v: kernel(q, k, v).sum(), argnums=(0, 1, 2))(q, k, v)

  def test_megacore_grads_match_single_core_gqa(self):
    reference = self._grads(bwd_dkv_megacore=False, num_kv_heads=self.num_kv_heads)
    megacore = self._grads(bwd_dkv_megacore=True, num_kv_heads=self.num_kv_heads)
    for expected, actual in zip(reference, megacore):
      np.testing.assert_array_equal(expected, actual)

  def test_megacore_grads_match_single_core_mha(self):
    reference = self._grads(bwd_dkv_megacore=False, num_kv_heads=self.num_query_heads)
    megacore = self._grads(bwd_dkv_megacore=True, num_kv_heads=self.num_query_heads)
    for expected, actual in zip(reference, megacore):
      np.testing.assert_array_equal(expected, actual)

  def test_megacore_rejects_single_kv_head(self):
    with self.assertRaisesRegex(ValueError, "more than one KV head"):
      self._grads(bwd_dkv_megacore=True, num_kv_heads=1)

  @pytest.mark.tpu_only
  def test_megacore_grads_match_single_core_gqa_tpu(self):
    reference = self._grads(bwd_dkv_megacore=False, num_kv_heads=self.num_kv_heads, interpret=False)
    megacore = self._grads(bwd_dkv_megacore=True, num_kv_heads=self.num_kv_heads, interpret=False)
    for expected, actual in zip(reference, megacore):
      np.testing.assert_array_equal(expected, actual)

  def test_megacore_rejects_mqa(self):
    config = dataclasses.replace(
        tokamax_splash_kernel.SplashConfig.get_default(),
        interpret=True,
        bwd_dkv_megacore=True,
    )
    mask = tokamax_splash_mask.CausalMask((self.seq_len, self.seq_len))
    kernel = tokamax_splash_kernel.make_splash_mqa_single_device(mask, config=config)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k1, (self.num_query_heads, self.seq_len, self.head_dim), dtype=self.dtype)
    k = jax.random.normal(k2, (self.seq_len, self.head_dim), dtype=self.dtype)
    v = jax.random.normal(k3, (self.seq_len, self.head_dim), dtype=self.dtype)
    with self.assertRaisesRegex(ValueError, "more than one KV head"):
      jax.grad(lambda q, k, v: kernel(q, k, v).sum(), argnums=(0, 1, 2))(q, k, v)


if __name__ == "__main__":
  unittest.main()
