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

""" Tests for the maxengine """

import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

import common_types
from jetstream.engine import token_utils
from layers import models
from layers import quantizations
import max_utils
from maxengine import MaxEngine
import prefix_cache
import pyconfig


class MaxEngineTest(unittest.TestCase):
  """Tests for MaxEngine."""

  # TODO: add unit test for the MaxEngine.

  def setUp(self):
    super().setUp()
    self.cfg = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    init_kwargs = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "base_num_decoder_layers": 2,
        "attention": "dot_product",
        "max_target_length": 16,
        "base_emb_dim": 256,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "max_prefill_predict_length": 4,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        **init_kwargs,
    )
    return config

  def get_data(self):
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(self.rng, s, 0, self.cfg.vocab_size)

    decoder_segment_ids = jax.numpy.zeros(s) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    decoder_positions = jnp.stack(
        [jnp.arange(self.cfg.max_target_length, dtype=jnp.int32) for _ in range(self.cfg.global_batch_size_to_train_on)]
    )

    return ids, decoder_segment_ids, decoder_positions

  def test_stack_and_unstack_prefill_cache(self):
    config = pyconfig.initialize(
        [None, "configs/base.yml"],
        enable_checkpointing=False,
        stack_prefill_result_cache=True,
    )
    engine = MaxEngine(config, jax.devices())
    num_layers = engine.config.num_decoder_layers
    input = {
        "decoder": {},
    }
    for i in range(num_layers):
      input["decoder"][f"layers_{i}"] = {
          "a": jnp.ones((1, 10)),
          "b": jnp.ones((1, 9)),
      }

    expected_stacked = {
        "a": jnp.ones((num_layers, 1, 10)),
        "b": jnp.ones((num_layers, 1, 9)),
    }
    got_stacked = engine._maybe_stack_prefill_result_cache(input)
    jax.tree.map(np.testing.assert_array_equal, got_stacked, expected_stacked)

    got_unstacked = engine._maybe_unstack_prefill_result_cache(got_stacked)
    jax.tree.map(np.testing.assert_array_equal, got_unstacked, input)

  def test_basic_prefill(self):
    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    model = models.Transformer(config=self.cfg, mesh=mesh, quant=quant)
    ids, decoder_segment_ids, decoder_positions = self.get_data()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng}, ids, decoder_positions, decoder_segment_ids, enable_dropout=False
    )
    input_tokens = jnp.array([1, 306, 5360, 304, 0, 0, 0, 0])
    true_length = 4
    engine = MaxEngine(self.cfg, jax.devices())
    prefill_result, first_token = engine.prefill(
        params=transformer_vars, padded_tokens=input_tokens, true_length=true_length
    )

    self.assertEqual(prefill_result["generated_tokens"], jnp.array([0]))
    # test default strategy is gready which choose only one next token
    self.assertEqual(prefill_result["tokens"].size, 1)
    self.assertNotEqual(prefill_result["tokens"], jnp.array([0]))
    self.assertTrue(jnp.array_equal(first_token.data.size, 3))

  def test_chunked_prefill(self):
    config = pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        max_prefill_predict_length=8,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        model_call_mode="inference",
    )

    engine = MaxEngine(config)
    params = engine.load_params()

    tokens = jnp.array([1, 306, 5360, 304, 306, 5360, 304])
    padding_tokens = jnp.array([1, 306, 5360, 304, 306, 5360, 304, 0])
    true_length = 7
    prefill_length = 8
    chunk_size = 4
    assert tokens.shape[0] == true_length
    assert padding_tokens.shape[0] == prefill_length

    chunked_padded_tokens, chunked_true_lengths, chunked_positions = token_utils.chunk_and_pad_tokens(
        tokens,
        bos_id=1,
        pad_id=0,
        is_bos=False,
        prefill_lengths=[prefill_length],
        max_prefill_length=prefill_length,
        chunk_size=chunk_size,
        jax_padding=True,
    )
    # prefill_length // chunk_size = 2
    assert len(chunked_padded_tokens) == 2

    # Prefill without chunked
    expected_prefill_result, expected_first_token = engine.prefill(
        params=params, padded_tokens=padding_tokens, true_length=true_length
    )

    chunked_prefill_result = None
    chunked_first_token = None
    for chunk_num, _ in enumerate(chunked_padded_tokens):
      if chunked_prefill_result is None:
        chunked_prefill_result, chunked_first_token = engine.prefill(
            params=params,
            padded_tokens=chunked_padded_tokens[chunk_num],
            true_length=chunked_true_lengths[chunk_num],
            positions=chunked_positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=padding_tokens,
            previous_chunk=chunked_prefill_result,
        )
      else:
        chunked_prefill_result, chunked_first_token = engine.prefill(
            params=params | {"cache": chunked_prefill_result["cache"]},
            padded_tokens=chunked_padded_tokens[chunk_num],
            true_length=chunked_true_lengths[chunk_num],
            positions=chunked_positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=padding_tokens,
            previous_chunk=chunked_prefill_result,
        )
      chunked_prefill_result["true_length_array"] = jnp.expand_dims(
          jnp.arange(0, chunk_num * chunk_size + chunked_true_lengths[chunk_num]), 0
      )

    # Delete extra contents only used in chunked prefill
    assert chunked_prefill_result is not None
    del chunked_prefill_result["true_length_array"]
    assert jax.tree.map(np.array_equal, expected_prefill_result, chunked_prefill_result)
    assert jax.tree.map(np.array_equal, expected_first_token, chunked_first_token)

  def test_prefix_cache_with_chunked_prefill(self):
    prefill_lengths = [4, 8]
    max_prefill_length = 8

    # Init Model
    config = self.init_pyconfig(
        max_target_length=max_prefill_length * 2,
        max_prefill_predict_length=max_prefill_length,
        model_call_mode="inference",
    )

    engine = MaxEngine(config)
    params = engine.load_params()

    # Prepare prefix tokens and longer tokens sharing common prefix
    prefix_tokens = np.array([1, 306, 5360, 304, 306, 123])
    longer_tokens = np.array([1, 306, 5360, 304, 306, 5361, 304])
    prefix_key = tuple(prefix_tokens.tolist())
    longer_key = tuple(longer_tokens.tolist())
    padding_prefix_tokens, prefix_true_length = token_utils.pad_tokens(
        prefix_tokens,
        bos_id=1,
        pad_id=0,
        is_bos=False,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
    )
    padding_longer_tokens, longer_true_length = token_utils.pad_tokens(
        longer_tokens,
        bos_id=1,
        pad_id=0,
        is_bos=False,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
    )
    assert prefix_true_length == 6
    assert longer_true_length == 7

    # Prefill the prefix tokens and save the prefix cache
    prefix_prefill_result, _ = engine.prefill(
        params=params, padded_tokens=padding_prefix_tokens, true_length=prefix_true_length
    )
    cache_value = prefix_cache.Value(
        prefix=prefix_prefill_result["cache"],
        true_length=prefix_true_length,
        padded_length=padding_prefix_tokens.shape[0],
        tokens=tuple(padding_prefix_tokens.tolist()),
    )
    prefix_cache_inst = prefix_cache.PrefixCache(
        hbm_bytes=cache_value.prefix_size_bytes, dram_bytes=cache_value.prefix_size_bytes
    )
    assert prefix_cache_inst.save(key=prefix_key, value=cache_value)

    # Fetch and load longest common prefix key
    common_key = prefix_cache_inst.fetch_longest_common_prefix_key(key=longer_key)
    assert common_key is not None
    assert common_key == prefix_key
    common_prefix_length = 0
    for prefix_token, longer_token in zip(prefix_key, longer_key):
      if prefix_token == longer_token:
        common_prefix_length += 1
      else:
        break
    assert common_prefix_length == 5
    loaded_prefix = prefix_cache_inst.load(common_key)
    assert loaded_prefix is not None

    # Prepare chunked prefill for suffix tokens
    suffix_tokens = longer_tokens[common_prefix_length:]
    padding_suffix_tokens, suffix_true_length = token_utils.pad_tokens(
        suffix_tokens,
        bos_id=1,
        pad_id=0,
        is_bos=False,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
    )
    positions = jnp.expand_dims(jnp.arange(common_prefix_length, common_prefix_length + padding_suffix_tokens.shape[0]), 0)
    assert suffix_true_length == 2
    assert padding_suffix_tokens.shape[0] == 4
    previous_chunk = {"true_length_array": jnp.expand_dims(jnp.arange(0, common_prefix_length), 0)}

    # Prefill suffix tokens and compare with the whole longer tokens without prefix cache
    longer_prefix, longer_first_token = engine.prefill(
        params=params | {"cache": loaded_prefix.prefix},
        padded_tokens=padding_suffix_tokens,
        true_length=suffix_true_length,
        positions=positions,
        complete_prompt_true_length=longer_true_length,
        complete_padded_prompt=longer_tokens,
        previous_chunk=previous_chunk,
    )

    expected_prefix, expected_first_token = engine.prefill(
        params=params,
        padded_tokens=padding_longer_tokens,
        true_length=longer_true_length,
    )

    assert jax.tree.map(np.array_equal, expected_prefix, longer_prefix)
    assert jax.tree.map(np.array_equal, expected_first_token, longer_first_token)


if __name__ == "__main__":
  unittest.main()
