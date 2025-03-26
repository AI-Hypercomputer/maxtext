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
import os.path

import pytest

import jax
from jax import numpy as jnp
import numpy as np

from MaxText import common_types
from MaxText import pyconfig
from MaxText import max_utils
from MaxText.constants import PKG_ROOT
from MaxText.maxengine import MaxEngine
from MaxText.layers import quantizations
from MaxText.layers import models

Mesh = jax.sharding.Mesh


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
        [sys.argv[0], os.path.join(PKG_ROOT, "configs", "base.yml")],
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
        [None, os.path.join(PKG_ROOT, "configs", "base.yml")],
        enable_checkpointing=False,
        stack_prefill_result_cache=True,
    )
    engine = MaxEngine(config, jax.devices())
    num_layers = engine.config.num_decoder_layers
    input_d = {
        "decoder": {},
    }
    for i in range(num_layers):
      input_d["decoder"][f"layers_{i}"] = {
          "a": jnp.ones((1, 10)),
          "b": jnp.ones((1, 9)),
      }

    expected_stacked = {
        "a": jnp.ones((num_layers, 1, 10)),
        "b": jnp.ones((num_layers, 1, 9)),
    }
    got_stacked = engine._maybe_stack_prefill_result_cache(input_d)
    jax.tree.map(np.testing.assert_array_equal, got_stacked, expected_stacked)

    got_unstacked = engine._maybe_unstack_prefill_result_cache(got_stacked)
    jax.tree.map(np.testing.assert_array_equal, got_unstacked, input_d)

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

  @pytest.mark.tpu_only
  def test_chunked_prefill(self):
    """Test identical result between chunked prefill with single and multiple chunked.

    The return value in kv_cache_prefill function without chunked prefill is key and value itself.
    Although the value of key and value are the same as stored in the KVCache without quantization,
    the prefill still produce slightly different result while using multiple TPU devices due to unknown reasons.
    Cannot test exactly identical between enabled / disable chunked prefill.
    Only test enable chunked prefill.

    The test could not pass on GPU.
    """
    prefill_length = 8
    tokens = jnp.array([1, 11, 22, 33, 444, 555, 666])
    padding_tokens = jnp.array([1, 11, 22, 33, 444, 555, 666, 0])
    true_length = tokens.shape[0]
    chunk_size = 4
    assert padding_tokens.shape[0] == prefill_length

    # jetstream.engine.token_utils:chunk_and_pad_tokens could produce the similar results
    chunked_padded_tokens, chunked_true_lengths, chunked_positions = (
        [jnp.array([1, 11, 22, 33]), jnp.array([444, 555, 666, 0])],
        [4, 3],
        [jnp.array([[0, 1, 2, 3]]), jnp.array([[4, 5, 6, 7]])],
    )

    # prefill_length // chunk_size = 2
    assert len(chunked_padded_tokens) == 2
    assert chunked_padded_tokens[0].shape[0] == chunk_size

    # Model with chunked prefill
    config = self.init_pyconfig(
        use_chunked_prefill="true",
        max_target_length=prefill_length * 4,
        max_prefill_predict_length=prefill_length * 2,
        model_call_mode="inference",
        capacity_factor=1,
        decoder_block="mistral",
        scan_layers=False,
    )
    engine = MaxEngine(config)
    params = engine.load_params()

    # Chunk size >= length of tokens
    one_chunk_prefill_result, one_chunk_first_token = engine.prefill(
        params=params,
        padded_tokens=padding_tokens,
        true_length=true_length,
        complete_padded_prompt=padding_tokens,
        complete_prompt_true_length=true_length,
        previous_chunk=None,
        positions=jax.numpy.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )

    # Two chunks
    two_chunk_prefill_result = None
    two_chunk_first_token = None
    for chunk_num, _ in enumerate(chunked_padded_tokens):
      if two_chunk_prefill_result is None:
        two_chunk_prefill_result, two_chunk_first_token = engine.prefill(
            params=params,
            padded_tokens=chunked_padded_tokens[chunk_num],
            true_length=chunked_true_lengths[chunk_num],
            positions=chunked_positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=padding_tokens,
            previous_chunk=two_chunk_prefill_result,
        )
      else:
        two_chunk_prefill_result, two_chunk_first_token = engine.prefill(
            params=params | {"cache": two_chunk_prefill_result["cache"]},
            padded_tokens=chunked_padded_tokens[chunk_num],
            true_length=chunked_true_lengths[chunk_num],
            positions=chunked_positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=padding_tokens,
            previous_chunk=two_chunk_prefill_result,
        )
      two_chunk_prefill_result["true_length_array"] = jnp.expand_dims(
          jnp.arange(0, chunk_num * chunk_size + chunked_true_lengths[chunk_num]), 0
      )

    # Delete extra contents only used in chunked prefill
    assert two_chunk_prefill_result is not None
    del two_chunk_prefill_result["true_length_array"]

    assert jax.tree.all(jax.tree.map(jnp.array_equal, one_chunk_prefill_result, two_chunk_prefill_result))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, one_chunk_first_token, two_chunk_first_token))


if __name__ == "__main__":
  unittest.main()
