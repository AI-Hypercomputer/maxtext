# Copyright 2026 Google LLC
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

"""Tests for linears.py."""

import sys
import unittest
from unittest.mock import MagicMock
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.layers import linears
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class UtilsTest(unittest.TestCase):
  """Tests for utility functions in linears.py."""

  def test_normalize_axes(self):
    self.assertEqual(linears.normalize_axes((1, 2), 4), (1, 2))
    self.assertEqual(linears.normalize_axes((-1, -2), 4), (3, 2))
    self.assertEqual(linears.normalize_axes((0, -1), 3), (0, 2))

  def test_canonicalize_tuple(self):
    self.assertEqual(linears.canonicalize_tuple(1), (1,))
    self.assertEqual(linears.canonicalize_tuple((1, 2)), (1, 2))
    self.assertEqual(linears.canonicalize_tuple([1, 2]), (1, 2))

  # pylint: disable=protected-access
  def test_convert_to_activation_function(self):
    lin_fn = linears._convert_to_activation_function("linear")
    x = jnp.array([1.0, 2.0])
    np.testing.assert_array_equal(lin_fn(x), x)

    relu_fn = linears._convert_to_activation_function("relu")
    x = jnp.array([-1.0, 2.0])
    np.testing.assert_array_equal(relu_fn(x), jnp.array([0.0, 2.0]))

    # Test with callable
    def dummy_fn(x):
      return x + 1

    self.assertEqual(linears._convert_to_activation_function(dummy_fn), dummy_fn)

    with self.assertRaises(ValueError):
      linears._convert_to_activation_function(123)


class DenseGeneralTest(unittest.TestCase):
  """Tests for DenseGeneral."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

  def test_basic_call(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))

  def test_bias(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        use_bias=True,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))
    self.assertIsNotNone(layer.bias)

  def test_slice_bounds(self):
    batch_size = 2
    in_features = 4
    n_heads = 2
    head_dim = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(n_heads, head_dim),
        use_bias=True,
        rngs=self.rngs,
    )

    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_features))
    full_output = layer(inputs)

    slice1 = layer(inputs, slice_bounds=(0, 3))
    slice2 = layer(inputs, slice_bounds=(3, head_dim))

    self.assertEqual(slice1.shape, (batch_size, n_heads, 3))
    self.assertEqual(slice2.shape, (batch_size, n_heads, head_dim - 3))

    np.testing.assert_allclose(slice1, full_output[..., :3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(slice2, full_output[..., 3:], rtol=1e-5, atol=1e-5)

  def test_slice_bounds_with_quantization(self):
    batch_size = 2
    in_features = 4
    n_heads = 2
    head_dim = 8

    mock_quant = MagicMock()
    mock_quant.quant_mode = None
    # Configure mock for ToNNX initialization
    mock_cls = mock_quant.dot_general_cls.return_value
    mock_instance = mock_cls.return_value
    mock_instance.init_with_output.return_value = (MagicMock(), {})
    mock_instance.apply.return_value = (MagicMock(), {})

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(n_heads, head_dim),
        use_bias=True,
        quant=mock_quant,
        rngs=self.rngs,
    )

    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_features))

    with self.assertRaisesRegex(ValueError, "sliced contraction is only supported when quant is None"):
      layer(inputs, slice_bounds=(0, 3))

  def test_slice_bounds_invalid(self):
    batch_size = 2
    in_features = 4
    n_heads = 2
    head_dim = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(n_heads, head_dim),
        use_bias=True,
        rngs=self.rngs,
    )

    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_features))

    with self.assertRaisesRegex(ValueError, "slice_bounds .* must be valid and within"):
      layer(inputs, slice_bounds=(3, 2))

    with self.assertRaisesRegex(ValueError, "slice_bounds .* must be valid and within"):
      layer(inputs, slice_bounds=(-1, 4))

    with self.assertRaisesRegex(ValueError, "slice_bounds .* must be valid and within"):
      layer(inputs, slice_bounds=(0, 100))

  def _run_dense_test(self, axis, in_feat_shape, expected_shape):
    batch_size = 2
    seq_len = 3
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_feat_shape,
        out_features_shape=out_features,
        axis=axis,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, expected_shape)

  def test_axis_neg_1(self):
    self._run_dense_test(-1, 4, (2, 3, 8))

  def test_axis_1(self):
    self._run_dense_test(1, 3, (2, 4, 8))

  def test_axis_0(self):
    self._run_dense_test(0, 2, (3, 4, 8))

  def test_dequantize_weight_scalar(self):
    w = jnp.ones((4, 8), dtype=jnp.float8_e4m3fn)
    scale = jnp.array(0.5, dtype=jnp.float32)
    w_dequant = linears.dequantize_weight(w, scale, compute_dtype=jnp.bfloat16)
    self.assertEqual(w_dequant.shape, (4, 8))
    self.assertEqual(w_dequant.dtype, jnp.bfloat16)
    np.testing.assert_allclose(w_dequant, np.full((4, 8), 0.5, dtype=np.float32), rtol=1e-3)

  def test_dequantize_weight_per_channel(self):
    w = jnp.ones((4, 8), dtype=jnp.float8_e4m3fn)
    scale = jnp.arange(1, 9, dtype=jnp.float32)
    w_dequant = linears.dequantize_weight(w, scale, compute_dtype=jnp.bfloat16)
    self.assertEqual(w_dequant.shape, (4, 8))
    expected = np.tile(np.arange(1, 9, dtype=np.float32), (4, 1))
    np.testing.assert_allclose(w_dequant, expected, rtol=1e-3)

  def test_dequantize_weight_block_wise(self):
    w = jnp.ones((4, 8), dtype=jnp.float8_e4m3fn)
    scale = jnp.array([[2.0, 3.0], [4.0, 5.0]], dtype=jnp.float32)  # 2x2 blocks of size 2x4
    w_dequant = linears.dequantize_weight(w, scale, compute_dtype=jnp.bfloat16)
    self.assertEqual(w_dequant.shape, (4, 8))
    expected = np.block([[np.full((2, 4), 2.0), np.full((2, 4), 3.0)], [np.full((2, 4), 4.0), np.full((2, 4), 5.0)]])
    np.testing.assert_allclose(w_dequant, expected, rtol=1e-3)

  def test_fp8_e4m3fn_dense_general(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        weight_dtype=jnp.float8_e4m3fn,
        dtype=jnp.bfloat16,
        rngs=self.rngs,
    )

    self.assertEqual(layer.kernel[...].dtype, jnp.float8_e4m3fn)
    self.assertIsNotNone(layer.kernel_scale)
    self.assertEqual(layer.kernel_scale[...].shape, ())

    inputs = jnp.ones((batch_size, in_features), dtype=jnp.bfloat16)
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))
    self.assertEqual(outputs.dtype, jnp.bfloat16)

  def test_fp8_e5m2_dense_general(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        weight_dtype=jnp.float8_e5m2,
        dtype=jnp.bfloat16,
        rngs=self.rngs,
    )

    self.assertEqual(layer.kernel[...].dtype, jnp.float8_e5m2)
    self.assertIsNotNone(layer.kernel_scale)

    inputs = jnp.ones((batch_size, in_features), dtype=jnp.bfloat16)
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))
    self.assertEqual(outputs.dtype, jnp.bfloat16)

  def test_fp8_block_wise_scale(self):
    in_features = 128
    out_features = 256

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        weight_dtype=jnp.float8_e4m3fn,
        dtype=jnp.bfloat16,
        block_size=64,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )

    self.assertEqual(layer.kernel[...].shape, (128, 256))
    self.assertEqual(layer.kernel_scale[...].shape, (2, 4))
    self.assertEqual(layer.scale_axes, ("embed", "mlp"))

    inputs = jnp.ones((2, in_features), dtype=jnp.bfloat16)
    outputs = layer(inputs)
    self.assertEqual(outputs.shape, (2, out_features))

  def test_kernel_scale_sharding_inference(self):
    # Test block scale sharding
    layer_block = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=128,
        weight_dtype=jnp.float8_e4m3fn,
        block_size=32,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )
    self.assertEqual(layer_block.scale_axes, ("embed", "mlp"))

    # Test scalar scale sharding
    layer_scalar = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=128,
        weight_dtype=jnp.float8_e4m3fn,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )
    self.assertEqual(layer_scalar.scale_axes, ())

    # Test per-channel scale sharding
    layer_channel = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=128,
        weight_dtype=jnp.float8_e4m3fn,
        scale_shape=(1, 128),
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )
    self.assertEqual(layer_channel.scale_axes, (None, "mlp"))

  def test_fp8_slice_bounds(self):
    in_features = 64
    out_features = 128
    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        weight_dtype=jnp.float8_e4m3fn,
        dtype=jnp.bfloat16,
        block_size=32,
        rngs=self.rngs,
    )
    inputs = jnp.ones((2, in_features), dtype=jnp.bfloat16)
    sliced_outputs = layer(inputs, slice_bounds=(0, 32))
    self.assertEqual(sliced_outputs.shape, (2, 32))
    self.assertEqual(sliced_outputs.dtype, jnp.bfloat16)

    full_outputs = layer(inputs)
    np.testing.assert_allclose(sliced_outputs, full_outputs[:, :32], rtol=1e-3, atol=1e-3)

  def test_dequantize_weight_block_divisibility(self):
    w = jnp.ones((64, 128), dtype=jnp.float8_e4m3fn)
    scale_valid = jnp.ones((2, 4), dtype=jnp.float32) * 2.0
    dequant = linears.dequantize_weight(w, scale_valid, compute_dtype=jnp.bfloat16)
    self.assertEqual(dequant.shape, (64, 128))
    self.assertEqual(dequant.dtype, jnp.bfloat16)
    np.testing.assert_allclose(dequant, 2.0, rtol=1e-3, atol=1e-3)

    scale_invalid = jnp.ones((3, 5), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      linears.dequantize_weight(w, scale_invalid, compute_dtype=jnp.bfloat16)

  def test_deepseek_v4_grouped_linear_fp8(self):
    layer = linears.DeepSeekV4GroupedLinear(
        in_features_per_group=16,
        out_features=128,
        n_groups=4,
        weight_dtype=jnp.float8_e4m3fn,
        dtype=jnp.bfloat16,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )
    self.assertIsNotNone(layer.kernel_scale)
    self.assertEqual(layer.scale_axes, ())
    inputs = jnp.ones((2, 4, 16), dtype=jnp.bfloat16)
    out = layer(inputs)
    self.assertEqual(out.shape, (2, 4, 32))
    self.assertEqual(out.dtype, jnp.bfloat16)


class MlpBlockTest(unittest.TestCase):
  """Tests for MlpBlock."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0, dropout=1)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": False,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    layer = linears.MlpBlock(
        config=self.cfg,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, intermediate_dim))

  def test_fused_mlp(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": True,
    }
    argv = [sys.argv[0], get_test_config_path()]
    cfg_fused = pyconfig.initialize(argv, **config_arguments)

    layer = linears.MlpBlock(
        config=cfg_fused,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, 1, intermediate_dim))

  @pytest.mark.tpu_only
  def test_dense_fsdp_two_stage_all_gather_tpu_only(self):
    """`dense_fsdp_use_two_stage_all_gather` is a pure resharding and must not change MLP numerics.

    The flag replaces the single combined 2-axis (fsdp x fsdp_transpose) kernel all-gather with two
    single-axis gathers separated by an `optimization_barrier`. That changes only the weight's layout,
    not its values, so on a 2D-FSDP mesh the MLP output must match the default single-all-gather path.
    Covers both the non-fused (2D `wi`/`wo` kernels) and fused (3D `wi` kernel) MLP paths.
    """
    # Imperative skip: calling jax.device_count() in a decorator would force JAX init during collection.
    if jax.device_count() != 4:
      self.skipTest("Dense FSDP two-stage all-gather test requires exactly 4 devices")

    in_features = 128
    intermediate_dim = 256
    batch_size = 4
    seq_len = 8

    def build_and_run(use_two_stage, fused_mlp):
      config_arguments = {
          "per_device_batch_size": 1.0,
          "run_name": "test",
          "enable_checkpointing": False,
          "max_target_length": seq_len,
          "dtype": "bfloat16",
          "fused_mlp": fused_mlp,
          "ici_fsdp_parallelism": 2,
          "ici_fsdp_transpose_parallelism": 2,
          "dense_fsdp_use_two_stage_all_gather": use_two_stage,
      }
      argv = [sys.argv[0], get_test_config_path()]
      cfg = pyconfig.initialize(argv, **config_arguments)

      devices_array = maxtext_utils.create_device_mesh(cfg)
      mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
      # Fresh, fixed-seed Rngs yields deterministic, identical weights across both builds.
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        layer = linears.MlpBlock(
            config=cfg,
            mesh=mesh,
            in_features=in_features,
            intermediate_dim=intermediate_dim,
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        inputs = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, seq_len, in_features), dtype=cfg.dtype)
        return nnx.jit(lambda m, x: m(x))(layer, inputs)

    for fused_mlp in (False, True):
      with self.subTest(fused_mlp=fused_mlp):
        reference = build_and_run(use_two_stage=False, fused_mlp=fused_mlp)
        two_stage = build_and_run(use_two_stage=True, fused_mlp=fused_mlp)

        self.assertEqual(two_stage.shape, (batch_size, seq_len, in_features))
        self.assertTrue(
            jnp.allclose(reference, two_stage, rtol=1e-2, atol=1e-2, equal_nan=False),
            msg=(
                f"two-stage all-gather changed MLP output (fused_mlp={fused_mlp}): max abs diff="
                f"{jnp.max(jnp.abs(reference.astype(jnp.float32) - two_stage.astype(jnp.float32)))}"
            ),
        )


if __name__ == "__main__":
  unittest.main()
