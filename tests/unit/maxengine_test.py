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

""" Tests for the maxengine """

import functools
import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_PREFILL
from maxtext.layers import quantizations
from maxtext.inference.maxengine import maxengine
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest

# All tests except test_chunked_prefill work in both TPU and CPU env.
# So no need for pytestmark = [pytest.mark.external_serving]

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
        [sys.argv[0], get_test_config_path()],
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
        [None, get_test_config_path()],
        enable_checkpointing=False,
        stack_prefill_result_cache=True,
    )
    engine = maxengine.MaxEngine(config, jax.devices())
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
    model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_PREFILL)
    ids, decoder_segment_ids, decoder_positions = self.get_data()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
    )
    input_tokens = jnp.array([1, 306, 5360, 304, 0, 0, 0, 0])
    true_length = 4
    engine = maxengine.MaxEngine(self.cfg, jax.devices())
    prefill_result, first_token = engine.prefill(
        params=transformer_vars, padded_tokens=input_tokens, true_length=true_length
    )

    self.assertEqual(prefill_result["generated_tokens"], jnp.array([0]))
    # test default strategy is gready which choose only one next token
    self.assertEqual(prefill_result["tokens"].size, 1)
    self.assertNotEqual(prefill_result["tokens"], jnp.array([0]))
    self.assertTrue(jnp.array_equal(first_token.data.size, 3))
    self.assertEqual(first_token.log_prob.shape, (1, 1))

  def test_diverse_beam_search_init(self):
    """Test that decode state initialization correctly handles diverse beam search."""
    num_beams = 4
    num_groups = 2
    cfg = self.init_pyconfig(
        decode_sampling_strategy="diverse_beam_search",
        decode_num_beams=num_beams,
        decode_num_beam_groups=num_groups,
    )
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params()
    decode_state = engine.init_decode_state()

    # The KV cache is stored in decode_state["cache"]
    # We need to find the batch dimension in the cache.
    # From maxengine.py: x = jnp.ones((total_batch_size, 1), dtype=jnp.int32)
    # total_batch_size = batch * num_beams

    # Let's explore one layer's cache to check the shape.
    # The cache structure is typically nested: decoder -> layers_i -> ...
    # {
    #   "decoder": {
    #     "layers_0": { "key": jax.Array, "value": jax.Array },
    #     "layers_1": { "key": jax.Array, "value": jax.Array },
    #     # ... and so on for every layer of the model
    #   }
    # }
    def check_cache_shape(x):
      # Check if x is a JAX array and has at least 3 dimensions - We only check
      # arrays that have enough dimensions to be actual KV caches (usually 4
      # dimensions: Batch, Seq, Heads, Dim).
      if isinstance(x, jax.Array) and x.ndim >= 3:
        # We expect one of the dimensions to be local_batch * num_beams
        # local_batch = per_device_batch_size * devices_per_replica (usually 1 in this test)
        expected_total_batch = int(cfg.per_device_batch_size * jax.local_device_count() * num_beams)
        # We use assertIn instead of x.shape[0] == expected_total_batch because
        # the cache is often permuted due to axis_order config.
        self.assertIn(expected_total_batch, x.shape)
    # jax.tree.map drills down to the leaf level and verify each element.
    jax.tree.map(check_cache_shape, decode_state["cache"])

  def test_num_beams_ignored_when_not_dbs(self):
    """Test that num_beams config is ignored when not using diverse beam search."""
    num_beams = 4
    # Set num_beams but keep greedy decoding (default)
    cfg = self.init_pyconfig(
        decode_sampling_strategy="greedy",
        decode_num_beams=num_beams,
    )
    engine = maxengine.MaxEngine(cfg, jax.devices())
    engine.load_params()
    decode_state = engine.init_decode_state()

    def check_cache_shape(x):
      if isinstance(x, jax.Array) and x.ndim >= 3:
        # Without DBS, total_batch_size should be 1 (per_device_batch_size * devices)
        expected_total_batch = int(cfg.per_device_batch_size * jax.local_device_count())
        # The batch dimension (usually the first one) should NOT be scaled by num_beams
        # We check x.shape[0] or whichever dimension we expect to be batch.
        # Given (2, 1, 4), if 2 is the batch (e.g. from local_device_count), it matches.
        # self.assertEqual(x.shape[0], expected_total_batch)
        self.assertIn(expected_total_batch, x.shape)

    jax.tree.map(check_cache_shape, decode_state["cache"])

  def test_reorder_cache(self):
    """Test that reorder_cache correctly gathers parent memories."""
    cfg = self.init_pyconfig()
    engine = maxengine.MaxEngine(cfg, jax.devices())

    # Create a dummy cache with 4 slots (e.g., 1 user * 4 beams)
    # Each slot has a unique value (index 0..3)
    dummy_cache = {
        "layer0": jnp.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
        "layer1": jnp.array([[[10.0]], [[11.0]], [[12.0]], [[13.0]]]),
    }

    # Winning parents: [3, 2, 2, 0]
    # Slot 0 wants parent 3
    # Slot 1 wants parent 2
    # Slot 2 wants parent 2 (branching!)
    # Slot 3 wants parent 0
    parent_indices = jnp.array([[3], [2], [2], [0]])

    reordered = engine.reorder_cache(dummy_cache, parent_indices)

    # Verify layer0: should be [3.0, 2.0, 2.0, 0.0]
    expected0 = jnp.array([[[3.0]], [[2.0]], [[2.0]], [[0.0]]])
    self.assertTrue(jnp.array_equal(reordered["layer0"], expected0))

    # Verify layer1: should be [13.0, 12.0, 12.0, 10.0]
    expected1 = jnp.array([[[13.0]], [[12.0]], [[12.0]], [[10.0]]])
    self.assertTrue(jnp.array_equal(reordered["layer1"], expected1))

  def test_diverse_beam_search_decode_step(self):
    """Test that decode_step handles DBS branching correctly."""
    num_beams = 2
    cfg = self.init_pyconfig(
        decode_sampling_strategy="diverse_beam_search",
        decode_num_beams=num_beams,
        decode_num_beam_groups=1,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    quant = quantizations.configure_quantization(cfg)
    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_PREFILL)
    
    # Initialize dummy variables to get params
    ids, decoder_segment_ids, decoder_positions = self.get_data()
    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
    )
    
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=transformer_vars)
    decode_state = engine.init_decode_state()

    # Manually ensure is_dbs is in the state (as prefill would do)
    decode_state["is_dbs"] = jnp.array([True])
    # Add dummy cumulative logprobs for DBS
    total_batch = int(cfg.per_device_batch_size * jax.local_device_count() * num_beams)
    decode_state["cumulative_logprobs"] = jnp.zeros((total_batch, 1))

    rng = jax.random.PRNGKey(0)
    # Run one decode step
    # This will trigger jax.lax.cond and use inference_utils.sampling_dbs
    new_state, result = engine.generate(
        params,
        decode_state,
        rng=rng,
    )

    # Verify that the state was updated
    self.assertIn("tokens", new_state)
    self.assertIn("cache", new_state)
    # Total batch size should be maintained
    self.assertEqual(new_state["tokens"].shape[0], total_batch)

  def test_basic_decode(self):
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_PREFILL)
    ids, decoder_segment_ids, decoder_positions = self.get_data()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
    )
    input_tokens = jnp.array([1, 306, 5360, 304])
    engine = maxengine.MaxEngine(self.cfg, jax.devices())
    params = engine.load_params(params=transformer_vars)
    decode_state = engine.init_decode_state()
    prefill_result, _ = engine.prefill(params=params, padded_tokens=input_tokens, true_length=4)
    decode_state = engine.insert(prefill_result, decode_state, slot=0)
    decode_state, result_token = engine.generate(params=params, decode_state=decode_state)

    self.assertEqual(result_token.log_prob.ndim, 2)
    self.assertEqual(result_token.log_prob.shape[1], 1)
    self.assertEqual(result_token.data.ndim, 2)
    self.assertEqual(result_token.data.shape[1], 3)

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
    engine = maxengine.MaxEngine(config)
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
    engine = maxengine.MaxEngine(config)
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
