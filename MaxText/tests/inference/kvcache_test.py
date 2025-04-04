"""Copyright 2025 Google LLC.

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

import unittest

import common_types
from inference import kvcache

import jax
from jax import random
import jax.numpy as jnp


class MlaKVCacheTest(unittest.TestCase):
  """Tests for MLA KVCache."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)
    self.batchsize = 8
    self.prefill_len = 100
    self.target_len = 196
    self.dtype = jnp.bfloat16
    self.kv_lora_rank = 512
    self.k_rope_head_dim = 64

  def test_update_kv_cache(self):
    test_module = kvcache.MlaKVCache(self.prefill_len, self.target_len, self.dtype)
    low_rank_main = jnp.ones((self.batchsize, self.prefill_len, self.kv_lora_rank), dtype=self.dtype) * 0.02
    key_rope = (
        jnp.ones(
            (self.batchsize, self.prefill_len, 1, self.k_rope_head_dim),
            dtype=self.dtype,
        )
        * 0.03
    )
    decoder_segment_ids = None
    model_mode = common_types.MODEL_MODE_PREFILL
    variables = test_module.init(
        {"params": self.rng},
        low_rank_main,
        key_rope,
        decoder_segment_ids,
        model_mode,
    )

    # Prefill step. Inits all cache variables but populates only prefill
    # variables
    _, new_vars = test_module.apply(
        variables,
        low_rank_main,
        key_rope,
        decoder_segment_ids,
        model_mode,
        rngs={"params": random.PRNGKey(0)},
        mutable=True,
    )
    prefill_low_rank_main = jnp.transpose(
        new_vars["cache"]["cached_prefill_key"].value,
        test_module.key_axis_order,
    )
    prefill_key_rope = jnp.transpose(
        new_vars["cache"]["cached_prefill_value"].value,
        test_module.key_axis_order,
    )
    ar_low_rank_main = jnp.transpose(new_vars["cache"]["cached_ar_key"].value, test_module.key_axis_order)
    ar_key_rope = jnp.transpose(new_vars["cache"]["cached_ar_value"].value, test_module.key_axis_order)

    # Ensure prefill cache variables have correct shapes and values
    self.assertEqual(
        prefill_low_rank_main.shape,
        (self.batchsize, self.prefill_len, 1, self.kv_lora_rank),
    )
    self.assertEqual(
        prefill_key_rope.shape,
        (self.batchsize, self.prefill_len, 1, self.k_rope_head_dim),
    )
    self.assertEqual(prefill_low_rank_main[0][0][0][0], low_rank_main[0][0][0])
    self.assertEqual(prefill_key_rope[0][0][0][0], key_rope[0][0][0][0])

    # Ensure ar cache variables are initialized with right shape and 0 values
    self.assertEqual(
        ar_low_rank_main.shape,
        (self.batchsize, (self.target_len - self.prefill_len), 1, self.kv_lora_rank),
    )
    self.assertEqual(
        ar_key_rope.shape,
        (self.batchsize, (self.target_len - self.prefill_len), 1, self.k_rope_head_dim),
    )
    self.assertEqual(ar_low_rank_main[0][0][0][0], 0.0)
    self.assertEqual(ar_key_rope[0][0][0][0], 0.0)

    # Autoregressive step. Prefill remains same but updates autoregressive
    # variables
    model_mode = common_types.MODEL_MODE_AUTOREGRESSIVE
    low_rank_main_1 = jnp.ones((self.batchsize, 1, self.kv_lora_rank), dtype=self.dtype) * 0.04
    key_rope_1 = jnp.ones((self.batchsize, 1, 1, self.k_rope_head_dim), dtype=self.dtype) * 0.05
    _, new_vars = test_module.apply(
        new_vars,
        low_rank_main_1,
        key_rope_1,
        decoder_segment_ids,
        model_mode,
        rngs={"params": random.PRNGKey(0)},
        mutable=True,
    )
    prefill_low_rank_main = jnp.transpose(
        new_vars["cache"]["cached_prefill_key"].value,
        test_module.key_axis_order,
    )
    prefill_key_rope = jnp.transpose(
        new_vars["cache"]["cached_prefill_value"].value,
        test_module.key_axis_order,
    )
    ar_low_rank_main = jnp.transpose(new_vars["cache"]["cached_ar_key"].value, test_module.key_axis_order)
    ar_key_rope = jnp.transpose(new_vars["cache"]["cached_ar_value"].value, test_module.key_axis_order)

    # Ensure prefill cache variables are same as before
    self.assertEqual(
        prefill_low_rank_main.shape,
        (self.batchsize, self.prefill_len, 1, self.kv_lora_rank),
    )
    self.assertEqual(
        prefill_key_rope.shape,
        (self.batchsize, self.prefill_len, 1, self.k_rope_head_dim),
    )
    self.assertEqual(prefill_low_rank_main[0][0][0][0], low_rank_main[0][0][0])
    self.assertEqual(prefill_key_rope[0][0][0][0], key_rope[0][0][0][0])

    # Ensure ar cache variables have correct shapes and are correctly populated
    ar_cache_len = self.target_len - self.prefill_len
    self.assertEqual(
        ar_low_rank_main.shape,
        (self.batchsize, ar_cache_len, 1, self.kv_lora_rank),
    )
    self.assertEqual(
        ar_key_rope.shape,
        (self.batchsize, ar_cache_len, 1, self.k_rope_head_dim),
    )
    self.assertEqual(ar_low_rank_main[0][0][0][0], low_rank_main_1[0][0][0])
    self.assertEqual(ar_key_rope[0][0][0][0], key_rope_1[0][0][0][0])
