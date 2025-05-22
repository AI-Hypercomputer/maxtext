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

import functools
import pytest
import sys
import unittest
import os.path

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from MaxText import maxtext_utils
from MaxText import pyconfig, maxengine
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.globals import PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.maxengine import MaxEngine
from MaxText import inference_utils # Added for BatchedPrefillProcessor


class MaxEngineTest(unittest.TestCase):
  """Tests for MaxEngine."""

  # TODO: add unit test for the MaxEngine.

  def setUp(self):
    super().setUp()
    self.cfg = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    """init pyconfig"""
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
        "return_log_prob": True,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config

  def get_data(self):
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(self.rng, s, 0, self.cfg.vocab_size)

    decoder_segment_ids = jax.numpy.zeros(s) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    decoder_positions = jnp.stack(
        [jnp.arange(self.cfg.max_target_length, dtype=jnp.int32) for _ in range(self.cfg.global_batch_size_to_train_on)]
    )

    return ids, decoder_segment_ids, decoder_positions

  def test_stack_and_unstack_prefill_cache(self):
    config = pyconfig.initialize(
        [None, os.path.join(PKG_DIR, "configs", "base.yml")],
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
    # pylint: disable=protected-access
    got_stacked = engine._maybe_stack_prefill_result_cache(input_d)
    jax.tree.map(np.testing.assert_array_equal, got_stacked, expected_stacked)

    # pylint: disable=protected-access
    got_unstacked = engine._maybe_unstack_prefill_result_cache(got_stacked)
    jax.tree.map(np.testing.assert_array_equal, got_unstacked, input_d)

  def test_basic_prefill(self):
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
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
    self.assertEqual(first_token.log_prob.shape, (1, 1))

  def test_basic_decode(self):
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    model = models.Transformer(config=self.cfg, mesh=mesh, quant=quant)
    ids, decoder_segment_ids, decoder_positions = self.get_data()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng}, ids, decoder_positions, decoder_segment_ids, enable_dropout=False
    )
    input_tokens = jnp.array([1, 306, 5360, 304])
    engine = MaxEngine(self.cfg, jax.devices())
    params = engine.load_params(params=transformer_vars)
    decode_state = engine.init_decode_state()
    prefill_result, _ = engine.prefill(params=params, padded_tokens=input_tokens, true_length=4)
    decode_state = engine.insert(prefill_result, decode_state, slot=0)
    decode_state, result_token = engine.generate(params=params, decode_state=decode_state)

    self.assertEqual(result_token.log_prob.ndim, 2)
    self.assertEqual(result_token.log_prob.shape[1], 1)
    self.assertEqual(result_token.data.ndim, 2)
    self.assertEqual(result_token.data.shape[1], 3)

  def test_packed_prefill_segment_id_preservation(self):
    """Tests that segment IDs are preserved during packed prefill."""
    # 1. Setup Configuration
    # Prompt lengths: p1=3, p2=4. Capacity = 3+4+2 = 9.
    # max_prefill_predict_length must be >= capacity.
    # per_device_batch_size is 1 for this test.
    config_kwargs = {
        "max_prefill_predict_length": 20,
        "max_target_length": 30, # Must be > max_prefill_predict_length
        "num_decoder_layers": 1, # Simplify cache structure
        "per_device_batch_size": 1,
        "vocab_size": 1024,
        "attention": "dot_product", # Non-paged attention for simplicity
        "scan_layers": False,
        "ici_tensor_parallelism": -1, # Run on a single device for testing
        "dataset_path": "", # Avoid issues with dataset loading
        "tokenizer_path": "", # Avoid issues with tokenizer loading
        "load_parameters_path": "", # No real params needed
    }
    self.cfg = self.init_pyconfig(**config_kwargs)
    
    # Use only one device for this test to simplify batch dimension
    devices = jax.devices()[:1]
    mesh = Mesh(maxtext_utils.create_device_mesh(self.cfg, devices), self.cfg.mesh_axes)

    # 2. Initialize Engine and State
    self.engine = MaxEngine(self.cfg, devices=devices) # Pass devices here
    
    # Create mock parameters
    # For Transformer, model_vars['params'] is what setup_decode_state and load_params expect.
    # We need a minimal structure that allows init_decode_state and prefill_concat to run.
    quant = quantizations.configure_quantization(self.cfg)
    model_for_init = models.Transformer(config=self.cfg, mesh=mesh, quant=quant)
    # Dummy inputs for model.init
    # Shape: (global_batch_size, max_target_length)
    # global_batch_size = per_device_batch_size * num_devices = 1 * 1 = 1
    dummy_ids = jnp.zeros((1, self.cfg.max_target_length), dtype=jnp.int32)
    dummy_positions = jnp.zeros((1, self.cfg.max_target_length), dtype=jnp.int32)
    dummy_segment_ids = jnp.zeros((1, self.cfg.max_target_length), dtype=jnp.int32)

    # Initialize abstract_vars which contains the 'params' pytree
    abstract_vars = model_for_init.init(
        {"params": self.rng, "aqt": self.rng}, 
        dummy_ids, 
        dummy_positions, 
        dummy_segment_ids, 
        enable_dropout=False
    )
    
    # load_params initializes abstract_params, kv_cache_annotations etc.
    # It expects the 'params' part of the variables.
    self.params = self.engine.load_params(params=abstract_vars, rng=self.rng) # Pass abstract_vars directly
    self.decode_state = self.engine.init_decode_state(rng=self.rng)

    # 3. Initialize BatchedPrefillProcessor
    bpp = inference_utils.BatchedPrefillProcessor(self.engine, max_batch_size=2)

    # 4. Test Data
    prompt1_tokens = jnp.array([10, 11, 12], dtype=jnp.int32)
    prompt2_tokens = jnp.array([20, 21, 22, 23], dtype=jnp.int32)
    len1 = len(prompt1_tokens)
    len2 = len(prompt2_tokens)
    capacity = len1 + len2 + 2  # 3 + 4 + 2 = 9

    # 5. Process Prompts
    processed_results = []
    # 'decode_state' will be updated by the callback
    # Need to capture the state potentially modified by prefill_concat and insert_partial
    
    # Keep a reference to the test's decode_state that can be updated
    # by the callback.
    shared_decode_state_list = [self.decode_state] 

    def dummy_prefill_done(results, row_ids, updated_decode_state_from_bpp):
      nonlocal processed_results # Use nonlocal if Python 3, or a mutable object for Python 2
      processed_results.extend(results)
      shared_decode_state_list[0] = updated_decode_state_from_bpp

    # Process prompt1
    bpp.process(self.params, shared_decode_state_list[0], decode_slot=0, input_id=0,
                input_prompt=prompt1_tokens, input_padding=capacity, capacity=capacity,
                prefill_done=dummy_prefill_done)
    # Process prompt2
    bpp.process(self.params, shared_decode_state_list[0], decode_slot=0, input_id=1, 
                input_prompt=prompt2_tokens, input_padding=capacity, capacity=capacity,
                prefill_done=dummy_prefill_done)
    
    # Flush to force processing of the bucket
    bpp.flush(self.params, shared_decode_state_list[0], prefill_done=dummy_prefill_done)
    
    # Update self.decode_state with the one modified by BPP
    self.decode_state = shared_decode_state_list[0]

    # 6. Retrieve segment IDs
    # Path: decode_state['cache']['decoder']['layers_0']['self_attention']['cache_prefill_segment_id']
    # Shape is (per_device_batch_size, max_prefill_predict_length)
    # Here per_device_batch_size is 1.
    cached_segment_ids_full = self.decode_state['cache']['decoder']['layers_0']['self_attention']['cache_prefill_segment_id']
    slot_0_segment_ids = cached_segment_ids_full[0] # We are using slot 0

    # 7. Assertions
    # Prompt 1 (len1=3) should have ID 1
    np.testing.assert_array_equal(slot_0_segment_ids[0:len1], 1,
                                  "Prompt 1 segment IDs mismatch")
    # Prompt 2 (len2=4) should have ID 3 (BPP uses odd numbers for segment IDs)
    np.testing.assert_array_equal(slot_0_segment_ids[len1:len1+len2], 3,
                                  "Prompt 2 segment IDs mismatch")
    # Padding within capacity (capacity=9) should have ID 0
    np.testing.assert_array_equal(slot_0_segment_ids[len1+len2:capacity], 0,
                                  "Padding within capacity mismatch")
    # Padding beyond capacity up to max_prefill_predict_length (20) should also be 0
    np.testing.assert_array_equal(slot_0_segment_ids[capacity:self.cfg.max_prefill_predict_length], 0,
                                  "Padding beyond capacity mismatch")

  @pytest.mark.skip(reason="Can only pass on CPU.")
  def test_chunked_prefill(self):
    """Test identical result between chunked prefill with single and multiple chunked.

    The return value in kv_cache_prefill function is key and value itself.
    Although the value of key and value are the same as stored in the KVCache without quantization,
    the prefill still produce slightly different result while using multiple TPU devices or GPU due to unknown reasons.
    """

    prefill_length = 8
    tokens = jnp.array([1, 11, 22, 33, 444, 555, 666])
    padding_tokens = jnp.array([1, 11, 22, 33, 444, 555, 666, 0])
    true_length = tokens.shape[0]
    assert padding_tokens.shape[0] == prefill_length

    def array_equal_valid_tokens(x, y, *, compare_length):
      if len(x.shape) > 1:
        # containing sequence
        if x.shape[0] > 1:
          # Assume batch size is 1
          # sequence is the first axis for kv cache
          return jnp.array_equal(x[:compare_length], y[:compare_length])
        else:
          # sequence is the second axis for decoder segment id
          return jnp.array_equal(x[:, :compare_length], y[:, :compare_length])
      else:
        # single integer
        return jnp.array_equal(x, y)

    model_config_args = {
        "max_target_length": prefill_length * 4,
        "max_prefill_predict_length": prefill_length * 2,
        "model_call_mode": "inference",
        "capacity_factor": -1,
        "decoder_block": "mistral",
        "scan_layers": False,
        "per_device_batch_size": 1.0,
    }

    # Model without chunked prefill
    config = self.init_pyconfig(
        use_chunked_prefill=False,
        **model_config_args,
    )
    engine = MaxEngine(config)
    params = engine.load_params()
    expected_prefill_result, expected_first_token = engine.prefill(
        params=params,
        padded_tokens=padding_tokens,
        true_length=true_length,
    )

    # Model with chunked prefill
    config = self.init_pyconfig(
        use_chunked_prefill=True,
        **model_config_args,
    )
    engine = MaxEngine(config)
    params = engine.load_params()

    # One chunk
    one_chunk_prefill_result, one_chunk_first_token = engine.prefill(
        params=params,
        padded_tokens=padding_tokens,
        true_length=true_length,
    )

    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            one_chunk_prefill_result,
            expected_prefill_result,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            one_chunk_first_token,
            expected_first_token,
        )
    )

    # Two chunks
    two_chunk_prefill_result = None
    two_chunk_first_token = None
    existing_prefix = None

    two_chunk_prefill_result, two_chunk_first_token = engine.prefill(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padding_tokens[:4],
        true_length=4,
    )

    existing_prefix = maxengine.ExistingPrefix(
        cache=two_chunk_prefill_result["cache"], common_prefix_tokens=padding_tokens[:4]
    )
    two_chunk_prefill_result, two_chunk_first_token = engine.prefill(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padding_tokens[4:],
        true_length=3,
    )

    # Delete extra contents only used in chunked prefill
    assert two_chunk_prefill_result is not None

    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            one_chunk_prefill_result,
            two_chunk_prefill_result,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            one_chunk_first_token,
            two_chunk_first_token,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            expected_prefill_result,
            two_chunk_prefill_result,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            functools.partial(array_equal_valid_tokens, compare_length=true_length),
            expected_first_token,
            two_chunk_first_token,
        )
    )


if __name__ == "__main__":
  unittest.main()
