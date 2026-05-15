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

"""Tests for the maxengine"""

import functools
import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from maxtext.configs import pyconfig
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_PREFILL
from maxtext.layers import quantizations

pytest.importorskip("jetstream", reason="jetstream not installed")
from maxtext.inference.maxengine import maxengine
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.external_serving]


@pytest.mark.integration_test
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
        "base_emb_dim": 32,
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

  def _init_nnx_pyconfig(self, **kwargs):
    """init_pyconfig with NNX flags on."""
    return self.init_pyconfig(pure_nnx=True, enable_nnx=True, pure_nnx_decoder=True, **kwargs)

  def _build_nnx_params(self, cfg, mesh):
    """Materialize an NNX Transformer and return its nnx.Param state."""
    _create_model = model_creation_utils.get_nnx_create_model_fn(cfg, mesh=mesh, model_mode=MODEL_MODE_PREFILL)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = _create_model()
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    return params_state

  def test_init_nnx(self):
    """NNX engine init exposes graphdef + abstract Transformer."""
    cfg = self._init_nnx_pyconfig()
    engine = maxengine.MaxEngine(cfg, jax.devices())
    self.assertIsNotNone(engine.graphdef)
    self.assertIsNotNone(engine.model)
    self.assertEqual(type(engine.model).__name__, "Transformer")

  def test_basic_prefill_nnx(self):
    """NNX prefill returns a Linen-shape result dict with finite values."""
    cfg = self._init_nnx_pyconfig()
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    params_state = self._build_nnx_params(cfg, mesh)

    input_tokens = jnp.array([1, 306, 5360, 304, 0, 0, 0, 0])
    true_length = 4
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=input_tokens, true_length=true_length)

    self.assertEqual(prefill_result["generated_tokens"], jnp.array([0]))
    self.assertEqual(prefill_result["tokens"].size, 1)
    self.assertTrue(jnp.array_equal(first_token.data.size, 3))
    self.assertEqual(first_token.log_prob.shape, (1, 1))
    self.assertIn("cache", prefill_result)
    self.assertIsInstance(prefill_result["cache"], dict)
    # Catch silent NaN/inf from a bad nnx.merge or cache round-trip.
    self.assertTrue(jnp.all(jnp.isfinite(prefill_result["logits"])))
    cache_leaves, _ = jax.tree.flatten(prefill_result["cache"])
    for leaf in cache_leaves:
      self.assertTrue(jnp.all(jnp.isfinite(leaf)), msg=f"non-finite cache leaf, shape={leaf.shape}")
    # scan_layers=True (default in test config) ⇒ leading axis is num_decoder_layers.
    for leaf in cache_leaves:
      self.assertEqual(leaf.shape[0], cfg.num_decoder_layers, msg=f"layer-axis mismatch, got shape={leaf.shape}")

  def test_basic_decode_nnx(self):
    """NNX prefill → insert → 4 generate steps. Verifies next_pos advances and logits stay finite."""
    cfg = self._init_nnx_pyconfig()
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    params_state = self._build_nnx_params(cfg, mesh)

    input_tokens = jnp.array([1, 306, 5360, 304])
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    decode_state = engine.init_decode_state()
    prefill_result, _ = engine.prefill(params=params, padded_tokens=input_tokens, true_length=4)
    decode_state = engine.insert(prefill_result, decode_state, slot=0)

    # 4 steps is enough to catch off-by-one cache pointer bugs.
    initial_next_pos = int(decode_state["next_pos"][0, 0])
    for step in range(4):
      decode_state, result_token = engine.generate(params=params, decode_state=decode_state)
      self.assertEqual(result_token.log_prob.ndim, 2)
      self.assertEqual(result_token.log_prob.shape[1], 1)
      self.assertEqual(result_token.data.ndim, 2)
      self.assertEqual(result_token.data.shape[1], 3)
      self.assertTrue(jnp.all(jnp.isfinite(decode_state["logits"])))
      self.assertEqual(
          int(decode_state["next_pos"][0, 0]),
          initial_next_pos + step + 1,
          msg=f"next_pos didn't advance at step {step}",
      )

  def test_quantize_passes_gate_for_nnx(self):
    """pure_nnx + quantization (convert-on-load) reaches the actual machinery in train mode."""
    # checkpoint_is_quantized defaults to False — full-precision on disk, AQT
    # quantizes per-forward against the loaded kernel (train mode).
    cfg = self._init_nnx_pyconfig(quantization="int8")
    engine = maxengine.MaxEngine(cfg, jax.devices())
    self.assertEqual(engine._nnx_quant_mode_str, "train")  # pylint: disable=protected-access
    try:
      engine.load_params(rng=self.rng)
    except NotImplementedError as e:
      self.fail(f"convert-on-load path should not raise NotImplementedError; got: {e}")
    except Exception:  # pylint: disable=broad-except
      pass  # any other failure (e.g. checkpoint not found) is fine for this test

  def test_load_pre_quantized_nnx_passes_quant_gate(self):
    """pure_nnx + quantization + checkpoint_is_quantized=True clears the load gate."""
    cfg = self._init_nnx_pyconfig(quantization="int8", checkpoint_is_quantized=True)
    engine = maxengine.MaxEngine(cfg, jax.devices())
    self.assertEqual(engine._nnx_quant_mode_str, "serve")  # pylint: disable=protected-access
    try:
      engine.load_params(rng=self.rng)
    except NotImplementedError as e:
      self.fail(f"checkpoint_is_quantized=True path should not raise NotImplementedError; got: {e}")
    except Exception:  # pylint: disable=broad-except
      pass  # any other failure (e.g. checkpoint not found) is fine for this test

  def test_quantized_prefill_nnx_train_mode(self):
    """End-to-end: NNX prefill with quantization=int8 + checkpoint_is_quantized=False.

    TRAIN-mode AQT layers quantize per-forward against the loaded full-precision
    kernel; output must be finite and shape-valid. This is the real numerical
    verification that the convert-on-load path produces a usable model.
    """
    cfg = self._init_nnx_pyconfig(quantization="int8")
    self.assertFalse(cfg.checkpoint_is_quantized)
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    params_state = self._build_nnx_params(cfg, mesh)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    self.assertEqual(engine._nnx_quant_mode_str, "train")  # pylint: disable=protected-access
    params = engine.load_params(params=params_state)
    input_tokens = jnp.array([1, 306, 5360, 304, 0, 0, 0, 0])
    prefill_result, _ = engine.prefill(params=params, padded_tokens=input_tokens, true_length=4)
    self.assertTrue(jnp.all(jnp.isfinite(prefill_result["logits"])))

  def test_lora_load_single_adapter_reaches_loader_on_nnx(self):
    """pure_nnx + LoRA: load_single_adapter dispatches to the NNX loader.

    With a nonexistent path the loader raises FileNotFoundError (not
    NotImplementedError, which would mean the dispatch never reached the loader).
    """
    cfg = self._init_nnx_pyconfig()
    engine = maxengine.MaxEngine(cfg, jax.devices())
    with self.assertRaises(FileNotFoundError):
      engine.load_single_adapter("/nonexistent/adapter/path")

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
