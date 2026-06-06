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

import sys
import unittest
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from flax import nnx
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.kernels.megablox.ops import gmm
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path
import pytest


def _configure_quantization(quant_str="", mode_str="train"):
  config = pyconfig.initialize(
      [None, get_test_config_path()],
      enable_checkpointing=False,
      quantization=quant_str,
  )
  quant = quantizations.configure_quantization(config, mode_str)
  return quant


class QuantizationTest(unittest.TestCase):
  """Tests for quantization."""

  def test_configure_quantization_is_null(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="", mode_str=quant_mode)
      self.assertEqual(quant, None)


class QuantizationConfigValidationTest(unittest.TestCase):
  """Tests for quantization configuration validation."""

  def test_use_qwix_quantization_default(self):
    # Verify that use_qwix_quantization defaults to True when initializing config
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantization="int8",
    )
    self.assertTrue(config.use_qwix_quantization)

  def test_unsupported_quantization_without_qwix(self):
    # Verify that setting use_qwix_quantization=False with non-native/non-TE quantization raises ValueError
    with self.assertRaisesRegex(ValueError, "is unsupported because legacy AQT has been completely removed"):
      pyconfig.initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          quantization="int8",
          use_qwix_quantization=False,
      )

  def test_supported_quantization_without_qwix(self):
    # Verify that setting use_qwix_quantization=False with native FP8 or TE is allowed and does not raise
    for quant_type in ["fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling"]:
      config = pyconfig.initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          quantization=quant_type,
          use_qwix_quantization=False,
      )
      self.assertFalse(config.use_qwix_quantization)


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
        "base_emb_dim": 16,
        "base_num_query_heads": 1,
        "base_num_kv_heads": 1,
        "base_mlp_dim": 16,
        "base_num_decoder_layers": 1,
        "max_target_length": 16,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
    return all(jnp.abs(y - x).mean() / (jnp.abs(x).mean() + 1e-8) < tolerance for x, y in zip(leaves_a, leaves_b))

  def print_grad_diff(self, a, b):
    """Print the key path and relative error for each leaf in two gradient PyTrees."""

    def format_key_path(keys):
      return "/".join(str(k) for k in keys)

    def compare_fn(path, x, y):
      rel_error = jnp.abs(y - x).mean() / (jnp.abs(x).mean() + 1e-8)
      print(f"{format_key_path(path)}: relative error = {rel_error}")

    jax.tree_util.tree_map_with_path(compare_fn, a, b)

  def quantization_config(self, quant, logits_tolerance=2e-1, grad_tolerance=5e-1):
    """Run forward pass and backward pass for quantized model and compare with base model."""
    # pylint: disable=protected-access
    cfg = self.init_pyconfig(quantization=quant)
    qt_model = model_creation_utils.create_model(cfg, self.mesh)

    ids, decoder_segment_ids, decoder_positions = self.get_data()

    if not hasattr(self.__class__, "_cached_base_results"):
      model = model_creation_utils.create_model(self.cfg, self.mesh)
      var = model.init(
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

      grads_base = jax.grad(loss_base)(var, (ids, decoder_positions, decoder_segment_ids))
      logits, _ = model.apply(
          var,
          ids,
          decoder_positions,
          decoder_segment_ids,
          enable_dropout=False,
          rngs={"params": self.rng},
          mutable=True,
      )
      self.__class__._cached_base_results = (grads_base, logits)

    grads_base, logits = self.__class__._cached_base_results

    quantized_vars = qt_model.init(
        {"params": self.rng, "aqt": self.rng, "dropout": self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        mutable=True,
    )

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
    grads_quant = jax.grad(loss_quant)(quantized_vars, (ids, decoder_positions, decoder_segment_ids))

    quant_logits, _ = qt_model.apply(
        quantized_vars,
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"params": self.rng},
        mutable=True,
    )
    print("relative error in logits:" f" {jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean()}")
    assert jnp.abs(quant_logits - logits).mean() / jnp.abs(logits).mean() < logits_tolerance
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
    self.quantization_config("fp8_gpu", grad_tolerance=1.5)

  @pytest.mark.gpu_only
  @pytest.mark.external_serving
  def test_fp8_nanoo_quantization(self):
    self.quantization_config("fp8_nanoo", grad_tolerance=1.5)

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
  # pylint: disable=undefined-variable
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


class QuantizationCoverageTest(unittest.TestCase):
  """Explicit tests to ensure 100% test coverage of all quantization paths."""

  def test_configure_quantization_paths(self):
    # Test all configure_quantization paths on CPU (instantiation only)
    config_fp8 = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantization="fp8",
        use_qwix_quantization=True,
    )
    quant_fp8 = quantizations.configure_quantization(config_fp8, "train")
    self.assertIsNotNone(quant_fp8)

    config_nanoo = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantization="nanoo_fp8",
        use_qwix_quantization=True,
    )
    quant_nanoo = quantizations.configure_quantization(config_nanoo, "train")
    self.assertIsNotNone(quant_nanoo)

    # Only run TE quantization config test if transformer_engine is installed
    try:
      import transformer_engine  # pylint: disable=unused-import,import-outside-toplevel

      has_te = True
    except ImportError:
      has_te = False

    if has_te:
      config_te = pyconfig.initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          quantization="te_fp8_delayedscaling",
          use_qwix_quantization=True,
      )
      quant_te = quantizations.configure_quantization(config_te, "train")
      self.assertIsNotNone(quant_te)

  def test_configure_kv_quant(self):
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantize_kvcache=False,
    )
    # Should not raise
    quantizations.configure_kv_quant(config)

    config_fail = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantize_kvcache=True,
    )
    with self.assertRaises(ValueError):
      quantizations.configure_kv_quant(config_fail)

  def test_moe_quantization_coverage(self):
    # Instantiates RoutedMoE on CPU to cover the AQT-free parameter initialization path in moe.py
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        quantization="int8",
        use_qwix_quantization=True,
        num_experts=2,
        base_emb_dim=8,
        base_mlp_dim=8,
        base_moe_mlp_dim=8,  # Required positive base value to derive positive moe_mlp_dim
        parameter_memory_host_offload=True,  # Cover the parameter offloading paths in linears.py
    )

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    quant = quantizations.configure_quantization(config, "train")

    with mesh:
      moe_layer = moe.RoutedMoE(
          config=config,
          num_experts=config.num_experts,
          num_experts_per_tok=1,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("expert", "embed_moe", "heads"),
          rngs=nnx.Rngs(0),
          quant=quant,
      )

      # In Flax NNX, parameters are fully initialized during instantiation.
      self.assertIsNotNone(moe_layer.gate.kernel)

      # Execute a forward pass to cover DenseGeneral.__call__, RoutedMoE.__call__,
      # sparse_matmul, and the custom quant_einsum wrapper in moe.py
      inputs = jnp.ones((2, 4, 8), dtype=jnp.float32)
      outputs, _, _ = moe_layer(inputs)
      self.assertEqual(outputs.shape, (2, 4, 8))

  def test_quantization_fallbacks(self):
    # Cover the fallback return None path in _get_quant_config when an unsupported scheme is passed
    config_invalid = pyconfig.initialize(
        [None, get_test_config_path()],
        quantization="int4",
    )
    self.assertIsNone(quantizations.configure_quantization(config_invalid))

    # Cover the implicit return None path in configure_kv_quant when quantize_kvcache is False
    config_no_kv = pyconfig.initialize(
        [None, get_test_config_path()],
        quantize_kvcache=False,
    )
    self.assertIsNone(quantizations.configure_kv_quant(config_no_kv))

  def test_dense_general_parameter_offload_coverage(self):
    # Covers parameter_memory_host_offload paths in linears.py
    from maxtext.layers import linears

    dense_layer = linears.DenseGeneral(
        in_features_shape=8,
        out_features_shape=8,
        parameter_memory_host_offload=True,
        rngs=nnx.Rngs(0),
    )
    inputs = jnp.ones((2, 8), dtype=jnp.float32)
    outputs = dense_layer(inputs)
    self.assertEqual(outputs.shape, (2, 8))

  def test_configure_quantization_batch_split_schedule(self):
    # Covers use_batch_split_schedule path in quantizations.py
    config_bs = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        use_batch_split_schedule=True,
        quantization="fp8_full",
        use_qwix_quantization=False,
    )
    quant = quantizations.configure_quantization(config_bs, "train")
    self.assertIsInstance(quant, quantizations.QwixQuantization)

    config_bs_manual = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        use_batch_split_schedule=True,
        quantization="fp8_full",
        use_manual_quantization=True,
        use_qwix_quantization=False,
    )
    quant_manual = quantizations.configure_quantization(config_bs_manual, "train")
    self.assertIsNone(quant_manual)

  def test_moe_gemma4_coverage(self):
    # Covers GEMMA4 routing and expert scale fusion paths in moe.py
    from maxtext.layers import moe

    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        decoder_block="gemma4",
        model_call_mode="inference",
        fuse_expert_scales=True,
        num_experts=2,
        base_emb_dim=8,
        base_mlp_dim=8,
        base_moe_mlp_dim=8,
    )
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    with mesh:
      moe_layer = moe.RoutedMoE(
          config=config,
          num_experts=config.num_experts,
          num_experts_per_tok=1,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("expert", "embed_moe", "heads"),
          rngs=nnx.Rngs(0),
      )
      inputs = jnp.ones((2, 4, 8), dtype=jnp.float32)
      outputs, _, _ = moe_layer(inputs)
      self.assertEqual(outputs.shape, (2, 4, 8))


if __name__ == "__main__":
  unittest.main()
