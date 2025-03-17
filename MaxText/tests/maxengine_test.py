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
import pyconfig


class MaxEngineTest(unittest.TestCase):
  """Tests for MaxEngine."""

  # TODO: add unit test for the MaxEngine.

  def setUp(self):
    super().setUp()
    self.cfg = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    config = pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        **kwargs,
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
    next_pos = 0
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
      chunked_prefill_result["next_pos"] = jnp.full((1, 1), next_pos + chunked_true_lengths[chunk_num], dtype=jnp.int32)
      next_pos = next_pos + chunked_true_lengths[chunk_num]

    # Delete extra contents only used in chunked prefill
    assert chunked_prefill_result is not None
    del chunked_prefill_result["true_length_array"]
    assert jax.tree.map(np.array_equal, expected_prefill_result, chunked_prefill_result)
    assert jax.tree.map(np.array_equal, expected_first_token, chunked_first_token)


if __name__ == "__main__":
  unittest.main()
