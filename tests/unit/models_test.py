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
"""Unit tests for models.py — covering routing, guards, and trivial methods."""

import sys
import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import (
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
)
from maxtext.configs import pyconfig
from maxtext.layers import nnx_wrappers
from maxtext.models.models import Transformer, TransformerLinen, TransformerLinenPure, transformer_as_linen
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(enable_nnx=True, pure_nnx=True, pure_nnx_decoder=True, **kwargs):
  extra = get_decoupled_parallelism_overrides()
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
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
      enable_nnx=enable_nnx,
      pure_nnx=pure_nnx,
      pure_nnx_decoder=pure_nnx_decoder,
      **kwargs,
      **extra,
  )


def _make_mesh(config):
  return Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)


# ---------------------------------------------------------------------------
# transformer_as_linen routing
# ---------------------------------------------------------------------------


class TestTransformerAsLinenRouting(unittest.TestCase):
  """Tests that transformer_as_linen() returns the correct type based on enable_nnx."""

  def test_returns_transformer_linen_pure_when_enable_nnx_false(self):
    """enable_nnx=False → TransformerLinenPure."""
    config = _make_config(enable_nnx=False, pure_nnx=False, pure_nnx_decoder=False)
    mesh = _make_mesh(config)
    model = transformer_as_linen(config, mesh, quant=None)
    self.assertIsInstance(model, TransformerLinenPure)

  def test_returns_transformer_linen_when_enable_nnx_true(self):
    """enable_nnx=True → TransformerLinen (an nnx_wrappers.ToLinen subclass)."""
    config = _make_config(enable_nnx=True)
    mesh = _make_mesh(config)
    model = transformer_as_linen(config, mesh, quant=None)
    self.assertIsInstance(model, TransformerLinen)
    self.assertIsInstance(model, nnx_wrappers.ToLinen)

  def test_name_kwarg_forwarded(self):
    """Optional name kwarg is accepted without error."""
    config = _make_config(enable_nnx=False, pure_nnx=False, pure_nnx_decoder=False)
    mesh = _make_mesh(config)
    model = transformer_as_linen(config, mesh, quant=None, name="my_transformer")
    self.assertIsInstance(model, TransformerLinenPure)

  def test_model_mode_forwarded_to_linen_pure(self):
    """model_mode is forwarded when enable_nnx=False."""
    config = _make_config(enable_nnx=False, pure_nnx=False, pure_nnx_decoder=False)
    mesh = _make_mesh(config)
    model = transformer_as_linen(config, mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
    self.assertEqual(model.model_mode, MODEL_MODE_PREFILL)


# ---------------------------------------------------------------------------
# TransformerLinenPure — __call__ guard
# ---------------------------------------------------------------------------


class TestTransformerLinenPureCallGuard(unittest.TestCase):
  """Tests the autoregressive + segment_ids ValueError guard in TransformerLinenPure."""

  def setUp(self):
    self.config = _make_config(enable_nnx=False, pure_nnx=False, pure_nnx_decoder=False)
    self.mesh = _make_mesh(self.config)
    self.rng = jax.random.PRNGKey(0)

  def _make_inputs(self):
    bs = self.config.global_batch_size_to_train_on
    seq = self.config.max_target_length
    ids = jax.random.randint(self.rng, (bs, seq), 0, self.config.vocab_size)
    positions = jnp.arange(seq)[None].repeat(bs, axis=0)
    segment_ids = jnp.ones((bs, seq)) * DECODING_ACTIVE_SEQUENCE_INDICATOR
    return ids, positions, segment_ids

  def test_raises_value_error_autoregressive_with_segment_ids(self):
    """Passing decoder_segment_ids in autoregressive mode must raise ValueError."""
    model = transformer_as_linen(self.config, self.mesh, quant=None)
    ids, positions, segment_ids = self._make_inputs()

    # Init first with train mode
    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
    )

    with self.assertRaises(ValueError, msg="autoregressive decoding"):
      model.apply(
          transformer_vars,
          ids,
          positions,
          decoder_segment_ids=segment_ids,  # non-None → triggers guard
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          enable_dropout=False,
          rngs={"aqt": self.rng},
      )


# ---------------------------------------------------------------------------
# TransformerLinen — apply with non-default model_mode
# ---------------------------------------------------------------------------


class TestTransformerLinenApply(unittest.TestCase):
  """Tests TransformerLinen.apply() and init() with explicit model_mode."""

  def setUp(self):
    self.config = _make_config(enable_nnx=True)
    self.mesh = _make_mesh(self.config)
    self.rng = jax.random.PRNGKey(0)

  def _make_inputs(self):
    bs = self.config.global_batch_size_to_train_on
    seq = self.config.max_target_length
    ids = jax.random.randint(self.rng, (bs, seq), 0, self.config.vocab_size)
    positions = jnp.arange(seq)[None].repeat(bs, axis=0)
    segment_ids = jnp.ones((bs, seq)) * DECODING_ACTIVE_SEQUENCE_INDICATOR
    return ids, positions, segment_ids

  def test_apply_with_prefill_model_mode(self):
    """TransformerLinen.apply with model_mode=PREFILL should return logits."""
    model = transformer_as_linen(self.config, self.mesh, quant=None)
    ids, positions, segment_ids = self._make_inputs()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
    )

    logits = jax.eval_shape(
        lambda: model.apply(
            transformer_vars,
            ids,
            positions,
            segment_ids,
            enable_dropout=False,
            model_mode=MODEL_MODE_PREFILL,
            rngs={"aqt": self.rng},
        )
    )
    # Logits shape: (batch, seq, vocab)
    self.assertEqual(logits.shape[0], ids.shape[0])
    self.assertEqual(logits.shape[1], ids.shape[1])
    self.assertEqual(logits.shape[2], self.config.vocab_size)

  def test_raises_value_error_autoregressive_with_segment_ids(self):
    """TransformerLinen also raises ValueError for autoregressive + segment_ids."""
    model = transformer_as_linen(self.config, self.mesh, quant=None)
    ids, positions, segment_ids = self._make_inputs()

    transformer_vars = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
    )

    with self.assertRaises(ValueError):
      model.apply(
          transformer_vars,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          enable_dropout=False,
          rngs={"aqt": self.rng},
      )


# ---------------------------------------------------------------------------
# Transformer (NNX) — trivial methods and call guard
# ---------------------------------------------------------------------------


class TestTransformerNNXMethods(unittest.TestCase):
  """Tests for the NNX Transformer class's trivial methods and guards."""

  def setUp(self):
    self.config = _make_config(enable_nnx=True)
    self.mesh = _make_mesh(self.config)

  def _create_abstract_model(self):
    def create_fn():
      rngs = maxtext_utils_nnx.create_nnx_rngs(self.config, is_training=True)
      return Transformer(self.config, self.mesh, quant=None, rngs=rngs)

    return nnx.eval_shape(create_fn)

  def test_no_op_returns_none(self):
    """Transformer.no_op() is a no-op and returns None."""
    model = self._create_abstract_model()
    result = model.no_op(1, 2, key="value")
    self.assertIsNone(result)

  def test_init_cache_returns_true(self):
    """Transformer.init_cache() always returns True."""
    model = self._create_abstract_model()
    result = model.init_cache(cache_size=128, batch_size=2, dtype=jnp.float32)
    self.assertTrue(result)

  def test_init_cache_default_dtype(self):
    """init_cache works with default dtype parameter."""
    model = self._create_abstract_model()
    result = model.init_cache(cache_size=64, batch_size=1)
    self.assertTrue(result)

  def test_call_raises_value_error_autoregressive_with_segment_ids(self):
    """NNX Transformer.__call__ raises ValueError for autoregressive + segment_ids."""
    model = self._create_abstract_model()

    bs = self.config.global_batch_size_to_train_on
    seq = self.config.max_target_length
    ids = jnp.ones((bs, seq), dtype=jnp.int32)
    positions = jnp.arange(seq)[None].repeat(bs, axis=0)
    segment_ids = jnp.ones((bs, seq), dtype=jnp.int32)

    with self.assertRaises(ValueError, msg="autoregressive decoding"):
      model(
          ids,
          positions,
          decoder_segment_ids=segment_ids,  # non-None → triggers guard
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

  def test_segment_ids_none_does_not_trigger_guard(self):
    """Guard condition: decoder_segment_ids is not None AND autoregressive.
    When decoder_segment_ids is None, the guard must not fire."""
    model = self._create_abstract_model()

    bs = self.config.global_batch_size_to_train_on
    seq = self.config.max_target_length
    ids = jnp.ones((bs, seq), dtype=jnp.int32)
    positions = jnp.arange(seq)[None].repeat(bs, axis=0)

    # Call the model directly; the guard fires before any JAX computation.
    # With decoder_segment_ids=None the guard evaluates to False.
    # Any subsequent error is a computation error, NOT the guard — we only catch
    # the guard's specific ValueError message.
    try:
      model(ids, positions, decoder_segment_ids=None, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    except ValueError as e:
      if "autoregressive decoding" in str(e):
        self.fail(f"Guard ValueError raised unexpectedly when segment_ids is None: {e}")
    except Exception:  # pylint: disable=broad-exception-caught
      pass  # Computation errors after the guard are expected for an abstract model


if __name__ == "__main__":
  unittest.main()
