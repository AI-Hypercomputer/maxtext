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

"""Tests for the quantizations"""

import functools
import os.path
import sys
from typing import Any
import unittest
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.flax import aqt_flax
from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh
from MaxText import pyconfig
from maxtext.common.gcloud_stub import is_decoupled
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.globals import MAXTEXT_PKG_DIR
from maxtext.kernels.megablox import gmm
from MaxText.layers import nnx_wrappers, quantizations
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest

_QUERY_REGEX = ".*/query"
_VALUE_REGEX = ".*/value"
MAXTEXT_PKG_DIR = os.path.join("src", MAXTEXT_PKG_DIR)


class QuantTestModule(nnx.Module):
  """Test module for einsum."""

  def __init__(
      self,
      quantization: quantizations.AqtQuantization,
      data_type: Any,
      rngs: nnx.Rngs,
  ):
    self.quantization = quantization
    self.identity = jnp.identity(2, dtype=data_type)
    self.einsum = None
    self.dot_general = None

    if self.quantization:
      quant_dg, is_tiled, tiling_fn = None, False, None
      if isinstance(self.quantization.quant_dg, dict):
        quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
      else:
        quant_dg, is_tiled, tiling_fn = self.quantization.quant_dg, False, None
      rhs_axis_metadata_wrapper = None
      if self.quantization.quant_mode == aqt_flax.QuantMode.CONVERT:
        rhs_axis_metadata_wrapper = None
      else:
        rhs_axis_metadata_wrapper = functools.partial(
            quantizations._rhs_axis_metadata_wrapper,
            mesh_axes=(),
            is_tiled=is_tiled,
            replicate_scale=self.quantization.replicate_scale,
        )

      aqt_dg_cls = aqt_flax.AqtDotGeneral(
          quant_dg,
          rhs_quant_mode=self.quantization.quant_mode,
          lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
          rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
          use_legacy_freezer=False,
          tiling_fn=tiling_fn,
      )
      aqt_dg_cls_nnx = nnx_wrappers.ToNNX(aqt_dg_cls, rngs=nnx.Rngs(params=0))
      aqt_einsum = aqt_flax.AqtEinsum(
          cfg=quant_dg,
          rhs_quant_mode=self.quantization.quant_mode,
          lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
          rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
          use_legacy_freezer=False,
          tiling_fn=tiling_fn,
      )
      aqt_einsum_nnx = nnx_wrappers.ToNNX(aqt_einsum, rngs=nnx.Rngs(params=0))

      self.einsum = nnx.data(aqt_einsum_nnx)
      self.dot_general = nnx.data(aqt_dg_cls_nnx)
    else:
      self.einsum = jnp.einsum
      self.dot_general = lax.dot_general

  def __call__(self, inputs):
    res_einsum = self.einsum("bc,ab->ac", inputs, self.identity)
    res_dg = self.dot_general(
        inputs, inputs, (((), ()), ((), ())), precision=None
    )
    return res_einsum, res_dg


def _configure_quantization(
    quant_str="", quant_cfg_path="", mode_str="train", replicate_scale=False
):
  config = pyconfig.initialize(
      [None, get_test_config_path()],
      enable_checkpointing=False,
      quantization=quant_str,
      quant_cfg_path=quant_cfg_path,
      replicate_quant_scale=replicate_scale,
  )
  quant = quantizations.configure_quantization(config, mode_str)
  return quant


def _apply(quant_str=""):
  rngs = nnx.Rngs(params=0)
  inputs = jnp.ones((2, 2))
  data_type = inputs.dtype
  quant = _configure_quantization(quant_str)
  test_module = QuantTestModule(quant, data_type, rngs)
  res_einsum, res_dg = test_module(inputs)
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
      quant = _configure_quantization(
          quant_str="int8", mode_str=quant_mode, replicate_scale=True
      )
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
        quant_str="intmp",
        quant_cfg_path=os.path.join(
            MAXTEXT_PKG_DIR, "configs", "quantization", "int8_weight_only.json"
        ),
    )
    self.assertTrue(
        isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 1
    )
    # pylint: disable=unsupported-membership-test
    self.assertTrue(quantizations.DEFAULT in quant.quant_dg)
    quant_cfg, _ = quant.quant_dg[quantizations.DEFAULT]
    self.assertEqual(quant_cfg.fwd.dg_quantizer.lhs.numerics.dtype, None)
    self.assertEqual(quant_cfg.fwd.dg_quantizer.rhs.numerics.bits, 8)

  def test_mixed_precision_config_scale(self):
    quant = _configure_quantization(
        quant_str="intmp",
        quant_cfg_path=os.path.join(
            MAXTEXT_PKG_DIR,
            "configs",
            "quantization",
            "dense_llm_weight_only_scale.json",
        ),
    )
    self.assertTrue(
        isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 7
    )
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
        quant_str="intmp",
        quant_cfg_path=os.path.join(
            MAXTEXT_PKG_DIR,
            "configs",
            "quantization",
            "dense_llm_subchannel.json",
        ),
    )
    self.assertTrue(
        isinstance(quant.quant_dg, dict) and len(quant.quant_dg) == 7
    )
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
                "mlp": {
                    "wi_0": {"kernel": 1.0},
                    "wi_1": {"kernel": 1.0},
                    "wo": {"kernel": 1.0},
                },
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
                            "qrhs": {
                                "frozen": aqt_tensor.QTensor(
                                    qvalue=[1.1, 1.0],
                                    scale=[1.0],
                                    scale_t=[1.0],
                                    bias=1.0,
                                )
                            }
                        }
                    },
                    "wi_1": {
                        "AqtDotGeneral_0": {
                            "qrhs": {
                                "frozen": aqt_tensor.QTensor(
                                    qvalue=[1.1, 1.0],
                                    scale=[1.0],
                                    scale_t=[1.0],
                                    bias=1.0,
                                )
                            }
                        }
                    },
                    "wo": {
                        "AqtDotGeneral_0": {
                            "qrhs": {
                                "frozen": aqt_tensor.QTensor(
                                    qvalue=[1.1, 1.0],
                                    scale=[1.0],
                                    scale_t=[1.0],
                                    bias=1.0,
                                )
                            }
                        }
                    },
                },
                "self_attention": {
                    "key": {
                        "AqtDotGeneral_0": {
                            "qrhs": {
                                "frozen": aqt_tensor.QTensor(
                                    qvalue=[1.1, 1.0],
                                    scale=[1.0],
                                    scale_t=[1.0],
                                    bias=1.0,
                                )
                            }
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
                "mlp": {
                    "wi_0": {"kernel": {}},
                    "wi_1": {"kernel": {}},
                    "wo": {"kernel": {}},
                },
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
    # Conditionally set ici_fsdp_parallelism to match device count in decoupled mode
    extra_args = (
        {"ici_fsdp_parallelism": jax.device_count()} if is_decoupled() else {}
    )
    init_kwargs = (
        {
            "run_name": "test",
            "dataset_type": "synthetic",
            "enable_checkpointing": False,
            "enable_goodput_recording": False,
            "steps": 1,
            "per_device_batch_size": 1,
            "use_qwix_quantization": True,
            "skip_jax_distributed_system": True,
            "base_emb_dim": 1024,
            "base_num_query_heads": 8,
            "base_num_kv_heads": 8,
            "base_mlp_dim": 4096,
            "base_num_decoder_layers": 12,
        }
        | kwargs
        | extra_args
    )
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **init_kwargs,
    )
    return config

  def get_data(self):
    """Get data."""
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(self.rng, s, 0, self.cfg.vocab_size)

    decoder_segment_ids = (
        jax.numpy.zeros(s) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )
    decoder_positions = jnp.stack([
        jnp.arange(self.cfg.max_target_length, dtype=jnp.int32)
        for _ in range(self.cfg.global_batch_size_to_train_on)
    ])
    return ids, decoder_segment_ids, decoder_positions

  def pytree_allclose(self, a, b, *, tolerance=0.01):
    """Return True if every pair of leaves is all-close."""
    leaves_a, leaves_b = jax.tree_util.tree_leaves(
        a
    ), jax.tree_util.tree_leaves(b)
    return all(
        jnp.abs(y - x).mean() / (jnp.abs(x).mean() + 1e-8) < tolerance
        for x, y in zip(leaves_a, leaves_b)
    )

  def print_grad_diff(self, a, b):
    """Print the key path and relative error for each leaf in two gradient PyTrees."""

    def format_key_path(keys):
      return "/".join(str(k) for k in keys)

    def compare_fn(path, x, y):
      rel_error = jnp.abs(y - x).mean() / (jnp.abs(x).mean() + 1e-8)
      print(f"{format_key_path(path)}: relative error = {rel_error}")

    jax.tree_util.tree_map_with_path(compare_fn, a, b)

  def quantization_config(
      self, quant, logits_tolerance=2e-1, grad_tolerance=5e-1
  ):
    """Run forward pass and backward pass for quantized model and compare with base model."""
    cfg = self.init_pyconfig(quantization=quant)
    model = model_creation_utils.create_model(self.cfg, self.mesh)
    qt_model = model_creation_utils.create_model(cfg, self.mesh)

    ids, decoder_segment_ids, decoder_positions = self.get_data()
    var = model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        mutable=True,
    )
    quantized_vars = qt_model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        mutable=True,
    )

    def loss_base(all_vars, inputs):
      logits, _ = model.apply(
          all_vars,
          *inputs,
          enable_dropout=False,
          rngs={"params": self.rng},
          mutable=True,
      )
      return jnp.mean((logits) ** 2)

    def loss_quant(all_vars, inputs):
      logits, _ = qt_model.apply(
          all_vars,
          *inputs,
          enable_dropout=False,
          rngs={"params": self.rng},
          mutable=True,
      )
      return jnp.mean((logits) ** 2)

    # Compute gradients w.r.t. both models
    grads_base = jax.grad(loss_base)(
        var, (ids, decoder_positions, decoder_segment_ids)
    )
    grads_quant = jax.grad(loss_quant)(
        quantized_vars, (ids, decoder_positions, decoder_segment_ids)
    )

    logits, _ = model.apply(
        var,
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"params": self.rng},
        mutable=True,
    )
    quant_logits, _ = qt_model.apply(
        quantized_vars,
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"params": self.rng},
        mutable=True,
    )
    print(
        "relative error in logits:"
        f" {jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean()}"
    )
    assert (
        jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean()
        < logits_tolerance
    )
    self.print_grad_diff(grads_base["params"], grads_quant["params"])
    self.assertTrue(
        self.pytree_allclose(
            grads_base["params"],
            grads_quant["params"],
            tolerance=grad_tolerance,
        )
    )

  @pytest.mark.tpu_only
  def test_int8_quantization(self):
    self.quantization_config("int8")

  @pytest.mark.tpu_only
  def test_fp8_quantization(self):
    self.quantization_config("fp8")

  @pytest.mark.tpu_only
  def test_fp8_full_quantization(self):
    self.quantization_config("fp8_full")

  @pytest.mark.gpu_only
  @pytest.mark.external_serving
  def test_fp8_gpu_quantization(self):
    self.quantization_config("fp8_gpu", grad_tolerance=1.0)

  @pytest.mark.gpu_only
  @pytest.mark.external_serving
  def test_fp8_nanoo_quantization(self):
    self.quantization_config("fp8_nanoo", grad_tolerance=1.0)

  @pytest.mark.skip(reason="No runner with GPU arch >= 89 is available")
  @pytest.mark.gpu_only
  def test_fp8_te_fp8_delayedscaling_quantization(self):
    self.quantization_config("te_fp8_delayedscaling", grad_tolerance=1.0)

  @pytest.mark.skip(reason="No runner with GPU arch >= 89 is available")
  @pytest.mark.gpu_only
  def test_fp8_te_fp8_currentscaling_quantization(self):
    self.quantization_config("te_fp8_currentscaling", grad_tolerance=1.0)

  @pytest.mark.skip(reason="No runner with GPU arch >= 100 is available")
  @pytest.mark.gpu_only
  def test_fp8_te_mxfp8_quantization(self):
    self.quantization_config("te_mxfp8", grad_tolerance=1.0)

  @pytest.mark.skip(reason="No runner with GPU arch >= 100 is available")
  @pytest.mark.gpu_only
  def test_fp8_te_nvfp4_quantization(self):
    self.quantization_config("te_nvfp4", grad_tolerance=1.0)


@pytest.mark.parametrize(
    "group_sizes,k,n,tiling,dtype",
    [
        # m = sum(group_sizes) must be divisible by tm (first element of tiling)
        ([3, 5], 6, 4, (1, 1, 1), jnp.int8),  # m = 8, tm = 8
    ],
)
@pytest.mark.tpu_only
def test_gmm_kernel(group_sizes, k, n, tiling, dtype):
  """Smoke-test + correctness check for the grouped matrix-multiply kernel.

  For each group i, gmm should compute

      lhs[start_i:end_i, :]  @  rhs[i]
  and stitch the results back together along rows.
  """
  group_sizes = jnp.array(group_sizes, dtype=jnp.int32)
  m = int(group_sizes.sum())

  key = jax.random.key(0)
  key, k1, k2 = jax.random.split(key, 3)

  lhs = jax.random.normal(k1, (m, k), dtype=jnp.float32)
  rhs = jax.random.normal(k2, (group_sizes.size, k, n), dtype=jnp.float32)

  # ---- run the Pallas kernel ------------------------------------------------
  base_out = gmm(
      lhs,
      rhs,
      group_sizes,
      tiling=tiling,  # small tiles so the shapes above work
      interpret=True,  # avoids device-specific compilation in CI
      lhs_quantize_dtype=None,
      rhs_quantize_dtype=None,
  ).block_until_ready()

  quant_out = gmm(
      lhs,
      rhs,
      group_sizes,
      tiling=tiling,  # small tiles so the shapes above work
      interpret=True,  # avoids device-specific compilation in CI
      lhs_quantize_dtype=dtype,
      rhs_quantize_dtype=dtype,
  ).block_until_ready()

  assert jnp.abs(quant_out - base_out).mean() / jnp.abs(base_out).mean() < 2e-1


if __name__ == "__main__":
  unittest.main()
