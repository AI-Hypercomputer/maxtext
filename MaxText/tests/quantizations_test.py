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

""" Tests for the quantizations """
import unittest
import os.path
import sys
import pytest

import numpy as np

import jax
from jax import numpy as jnp
from jax import random, lax
from jax.sharding import Mesh

from flax import linen as nn

from aqt.jax.v2 import aqt_tensor

from MaxText.globals import PKG_DIR, tpu_present
from MaxText import pyconfig
from MaxText.layers import quantizations
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR

_QUERY_REGEX = ".*/query"
_VALUE_REGEX = ".*/value"


class QuantTestModule(nn.Module):
  """Test module for einsum."""

  quantization: quantizations.AqtQuantization

  @nn.compact
  def __call__(self, inputs):
    identity = jnp.identity(2, dtype=inputs.dtype)
    einsum = jnp.einsum
    dot_general = lax.dot_general
    if self.quantization:
      einsum = self.quantization.einsum()
      dot_general_cls = self.quantization.dot_general_cls()
      dot_general = dot_general_cls()
    res_einsum = einsum("bc,ab->ac", inputs, identity)
    res_dg = dot_general(inputs, inputs, (((), ()), ((), ())), precision=None)
    return res_einsum, res_dg


def _configure_quantization(quant_str="", quant_cfg_path="", mode_str="train", replicate_scale=False):
  config = pyconfig.initialize(
      [None, os.path.join(PKG_DIR, "configs", "base.yml")],
      enable_checkpointing=False,
      quantization=quant_str,
      quant_cfg_path=quant_cfg_path,
      replicate_quant_scale=replicate_scale,
  )
  quant = quantizations.configure_quantization(config, mode_str)
  return quant


def _apply(quant_str=""):
  quant = _configure_quantization(quant_str)
  test_module = QuantTestModule(quant)
  rng = random.PRNGKey(0)
  variables = test_module.init({"params": rng}, jnp.ones((2, 2)))
  inputs = jnp.ones((2, 2))
  res_einsum, res_dg = test_module.apply(variables, inputs, rngs={"params": random.PRNGKey(0)})
  return inputs, res_einsum, res_dg


class QuantizationTest(unittest.TestCase):
  """Tests for quantization."""

  def test_in_quant_mode(self):
    quant = _configure_quantization(quant_str="int8", mode_str="convert")
    self.assertTrue(quantizations.in_convert_mode(quant))
    self.assertFalse(quantizations.in_serve_mode(quant))

  def test_configure_quantization_is_null(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="", mode_str=quant_mode)
      self.assertEqual(quant, None)

  def test_configure_quantization_replicate_scale(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="int8", mode_str=quant_mode)
      self.assertEqual(quant.replicate_scale, False)

    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="int8", mode_str=quant_mode, replicate_scale=True)
      self.assertEqual(quant.replicate_scale, True)

  def test_configure_quantization_is_int8(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="int8", mode_str=quant_mode)
      self.assertNotEqual(quant, None)

  @pytest.mark.tpu_only  # b/421002974
  def test_aqt_quantization(self):
    # Without quantization
    inputs, res_einsum, res_dg = _apply()
    self.assertTrue(jnp.array_equal(inputs, res_einsum))
    self.assertEqual(res_einsum.dtype, np.dtype(np.float32))
    self.assertTrue(jnp.array_equal(inputs, res_dg[0][0]))
    self.assertEqual(res_dg.dtype, np.dtype(np.float32))

    # With int8 quantization
    inputs, res_einsum, res_dg = _apply(quant_str="int8")
    self.assertTrue(jnp.greater(jnp.max(inputs), jnp.max(res_einsum)))
    self.assertEqual(res_einsum.dtype, np.dtype(np.float32))
    self.assertTrue(jnp.greater(jnp.max(inputs), jnp.max(res_dg[0][0])))
    # self.assertEqual(res_dg.dtype, np.dtype(np.float32))

  def test_mixed_precision_config_int8w(self):
    quant = _configure_quantization(
        quant_str="intmp", quant_cfg_path=os.path.join(PKG_DIR, "configs", "quantization", "int8_weight_only.json")
    )
    self.assertTrue(isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 1)
    # pylint: disable=unsupported-membership-test
    self.assertTrue(quantizations.DEFAULT in quant.quant_dg)
    quant_cfg, _ = quant.quant_dg[quantizations.DEFAULT]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.dtype, None)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 8)

  def test_mixed_precision_config_scale(self):
    quant = _configure_quantization(
        quant_str="intmp",
        quant_cfg_path=os.path.join(PKG_DIR, "configs", "quantization", "dense_llm_weight_only_scale.json"),
    )
    self.assertTrue(isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 7)
    # pylint: disable=unsupported-membership-test
    self.assertTrue(quantizations.DEFAULT in quant.quant_dg)
    quant_cfg, _ = quant.quant_dg[quantizations.DEFAULT]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.dtype, None)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 8)
    quant_cfg, _ = quant.quant_dg[_QUERY_REGEX]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.dtype, None)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 4)

  def test_mixed_precision_config_subchannel(self):
    quant = _configure_quantization(
        quant_str="intmp", quant_cfg_path=os.path.join(PKG_DIR, "configs", "quantization", "dense_llm_subchannel.json")
    )
    self.assertTrue(isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 7)
    # pylint: disable=unsupported-membership-test
    self.assertTrue(quantizations.DEFAULT in quant.quant_dg)
    quant_cfg, tile_size = quant.quant_dg[quantizations.DEFAULT]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.bits, 8)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 8)
    self.assertEqual(tile_size, -1)
    quant_cfg, tile_size = quant.quant_dg[_QUERY_REGEX]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.bits, 8)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 4)
    self.assertEqual(tile_size, 128)

    quant_cfg, tile_size = quant.quant_dg[_VALUE_REGEX]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.bits, 8)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 4)
    self.assertEqual(tile_size, -1)

  def test_remove_quantized_params(self):
    _params = {
        "decoder": {
            "decoder_norm": {"scale": 1.0},
            "layers": {
                "mlp": {"wi_0": {"kernel": 1.0}, "wi_1": {"kernel": 1.0}, "wo": {"kernel": 1.0}},
                "self_attention": {
                    "key": {"kernel": 1.0},
                },
            },
            "logits_dense": {"kernel": 1.0},
        },
    }
    _aqt_vars = {
        "decoder": {
            "layers": {
                "mlp": {
                    "wi_0": {
                        "AqtDotGeneral_0": {
                            "qrhs": {"frozen": aqt_tensor.QTensor(qvalue=[1.1, 1.0], scale=[1.0], scale_t=[1.0], bias=1.0)}
                        }
                    },
                    "wi_1": {
                        "AqtDotGeneral_0": {
                            "qrhs": {"frozen": aqt_tensor.QTensor(qvalue=[1.1, 1.0], scale=[1.0], scale_t=[1.0], bias=1.0)}
                        }
                    },
                    "wo": {
                        "AqtDotGeneral_0": {
                            "qrhs": {"frozen": aqt_tensor.QTensor(qvalue=[1.1, 1.0], scale=[1.0], scale_t=[1.0], bias=1.0)}
                        }
                    },
                },
                "self_attention": {
                    "key": {
                        "AqtDotGeneral_0": {
                            "qrhs": {"frozen": aqt_tensor.QTensor(qvalue=[1.1, 1.0], scale=[1.0], scale_t=[1.0], bias=1.0)}
                        }
                    }
                },
            }
        }
    }
    _expected = {
        "decoder": {
            "decoder_norm": {"scale": 1.0},
            "layers": {
                "mlp": {"wi_0": {"kernel": {}}, "wi_1": {"kernel": {}}, "wo": {"kernel": {}}},
                "self_attention": {
                    "key": {"kernel": {}},
                },
            },
            "logits_dense": {"kernel": 1.0},
        }
    }
    result = quantizations.remove_quantized_params(_params, _aqt_vars)
    self.assertEqual(_expected, result)


class QuantTest(unittest.TestCase):
  """Tests for quantized model correctness."""

  def setUp(self):
    self.cfg = self.init_pyconfig()
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.inputs = jnp.ones((4, 16))
    self.rng = jax.random.PRNGKey(0)
    self.rtol = 5e-1
    self.atol = 5e-1

  def init_pyconfig(self, **kwargs):
    """Initialize MaxText pyconfig."""
    init_kwargs = {
        "run_name": "test",
        "dataset_type": "synthetic",
        "enable_checkpointing": False,
        "enable_goodput_recording": False,
        "steps": 1,
        "per_device_batch_size": 1,
        "use_qwix_quantization": True,
        "skip_jax_distributed_system": True,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config

  def get_data(self):
    """Get data."""
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(self.rng, s, 0, self.cfg.vocab_size)

    decoder_segment_ids = jax.numpy.zeros(s) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    decoder_positions = jnp.stack(
        [jnp.arange(self.cfg.max_target_length, dtype=jnp.int32) for _ in range(self.cfg.global_batch_size_to_train_on)]
    )
    return ids, decoder_segment_ids, decoder_positions

  def pytree_allclose(self, a, b, *, tolerance=0.01):
    """Return True if every pair of leaves is all-close."""
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return all(jnp.abs(y - x).mean() / jnp.abs(x).mean() < tolerance for x, y in zip(leaves_a, leaves_b))

  def quantization_config(self, quant):
    """Run forward pass and backward pass for quantized model and compare with base model."""
    cfg = self.init_pyconfig(quantization=quant)
    model = train_utils.create_model(self.cfg, self.mesh)
    qt_model = train_utils.create_model(cfg, self.mesh)

    ids, decoder_segment_ids, decoder_positions = self.get_data()
    var = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
    )
    quantized_vars = qt_model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
    )

    def loss_base(params, inputs):
      logits = model.apply({"params": params}, *inputs, enable_dropout=False, rngs={"params": self.rng})
      return jnp.mean((logits) ** 2)

    def loss_quant(params, inputs):
      logits = qt_model.apply({"params": params}, *inputs, enable_dropout=False, rngs={"params": self.rng})
      return jnp.mean((logits) ** 2)

    # Compute gradients w.r.t. both models
    grads_base = jax.grad(loss_base)(var["params"], (ids, decoder_positions, decoder_segment_ids))
    grads_quant = jax.grad(loss_quant)(quantized_vars["params"], (ids, decoder_positions, decoder_segment_ids))

    logits = model.apply(
        {"params": var["params"]},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"params": self.rng},
    )
    quant_logits = qt_model.apply(
        {"params": quantized_vars["params"]},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"params": self.rng},
    )
    print(f"relative error in logits: {jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean()}")
    assert jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean() < 2e-1

    self.assertTrue(self.pytree_allclose(grads_base, grads_quant, tolerance=5e-1))

  @pytest.mark.tpu_only
  @unittest.skipIf(not tpu_present, "TPU only test")
  def test_int8_quantization(self):
    self.quantization_config("int8")

  @pytest.mark.tpu_only
  @unittest.skipIf(not tpu_present, "TPU only test")
  def test_fp8_quantization(self):
    self.quantization_config("fp8")

  @pytest.mark.tpu_only
  @unittest.skipIf(not tpu_present, "TPU only test")
  def test_fp8_full_quantization(self):
    self.quantization_config("fp8_full")


if __name__ == "__main__":
  unittest.main()
