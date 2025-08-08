"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for kernels. """

import unittest

import pytest

import numpy as np

import jax
import jax.numpy as jnp

from MaxText.kernels.ragged_attention import ragged_mqa, reference_mqa, ragged_mha, reference_mha, ragged_gqa, reference_gqa


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


if __name__ == "__main__":
  unittest.main()
