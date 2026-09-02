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

import functools
import sys
import unittest
from unittest.mock import MagicMock, patch
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import ShardMode
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

  def _make_kernel_ordered_pair(self, in_features_shape, out_features_shape, axis=-1, use_bias=False):
    """Builds a default layer and a weight_grad_in_kernel_order twin sharing weights.

    Both store the kernel identically and share the RNG stream, which is exactly
    the property that makes the flag checkpoint-compatible.
    """
    kwargs = {
        "in_features_shape": in_features_shape,
        "out_features_shape": out_features_shape,
        "axis": axis,
        "use_bias": use_bias,
        "shard_mode": ShardMode.EXPLICIT,
    }
    default = linears.DenseGeneral(**kwargs, rngs=nnx.Rngs(params=0))
    ordered = linears.DenseGeneral(**kwargs, weight_grad_in_kernel_order=True, rngs=nnx.Rngs(params=0))
    return default, ordered

  def test_weight_grad_in_kernel_order_leaves_kernel_untouched(self):
    default, ordered = self._make_kernel_ordered_pair(4, 8)
    # No checkpoint impact is the point of this flag, so the stored array must be
    # bit-identical, not merely the same shape.
    self.assertEqual(default.kernel[...].shape, ordered.kernel[...].shape)
    self.assertEqual(default.kernel_axes, ordered.kernel_axes)
    np.testing.assert_array_equal(default.kernel[...], ordered.kernel[...])

  def test_weight_grad_in_kernel_order_forward_matches(self):
    cases = ((4, 8, -1), (4, (2, 8), -1), ((2, 3), 8, (1, 2)))
    for in_features_shape, out_features_shape, axis in cases:
      with self.subTest(in_features_shape=in_features_shape, out_features_shape=out_features_shape):
        default, ordered = self._make_kernel_ordered_pair(in_features_shape, out_features_shape, axis=axis)
        shape = (2, 3) if axis == -1 else (5,)
        shape += (in_features_shape,) if isinstance(in_features_shape, int) else tuple(in_features_shape)
        inputs = jax.random.normal(jax.random.PRNGKey(0), shape)
        np.testing.assert_array_equal(default(inputs), ordered(inputs))

  def test_weight_grad_in_kernel_order_gradients_match(self):
    """Reordering the contraction must not change the gradient beyond float noise."""
    cases = ((4, 8, -1), (4, (2, 8), -1), ((2, 3), 8, (1, 2)))
    for in_features_shape, out_features_shape, axis in cases:
      with self.subTest(in_features_shape=in_features_shape, out_features_shape=out_features_shape):
        default, ordered = self._make_kernel_ordered_pair(in_features_shape, out_features_shape, axis=axis)
        shape = (2, 3) if axis == -1 else (5,)
        shape += (in_features_shape,) if isinstance(in_features_shape, int) else tuple(in_features_shape)
        inputs = jax.random.normal(jax.random.PRNGKey(0), shape)

        def loss(layer, x):
          return jnp.sum(jnp.sin(layer(x)))

        grad_default = nnx.grad(loss)(default, inputs).kernel[...]
        grad_ordered = nnx.grad(loss)(ordered, inputs).kernel[...]
        self.assertEqual(grad_ordered.shape, ordered.kernel[...].shape)
        np.testing.assert_allclose(grad_default, grad_ordered, rtol=1e-5, atol=1e-6)

        grad_x_default = jax.grad(functools.partial(loss, default))(inputs)
        grad_x_ordered = jax.grad(functools.partial(loss, ordered))(inputs)
        np.testing.assert_allclose(grad_x_default, grad_x_ordered, rtol=1e-5, atol=1e-6)

  def test_weight_grad_in_kernel_order_lm_head_shape(self):
    """The LM head's 2-D kernel is the only shape the config flag enables.

    Reordering the contraction reassociates the sum, so the gradient agrees to
    float rounding rather than exactly. The stored kernel -- and therefore the
    initialization -- is untouched, so the difference stays at rounding level
    instead of changing the training trajectory.
    """
    for dtype in (jnp.float32, jnp.bfloat16):
      with self.subTest(dtype=jnp.dtype(dtype).name):
        default, ordered = self._make_kernel_ordered_pair(8, 32)
        inputs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 8), dtype)
        weights = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 32), dtype)

        def loss(layer, x, weights=weights):
          return jnp.sum((layer(x) * weights).astype(jnp.float32))

        grad_default = nnx.grad(loss)(default, inputs).kernel[...]
        grad_ordered = nnx.grad(loss)(ordered, inputs).kernel[...]
        scale = float(jnp.abs(grad_default).max())
        self.assertLess(float(jnp.abs(grad_default - grad_ordered).max()), 1e-5 * scale)

  def test_weight_grad_in_kernel_order_composes_with_remat(self):
    # The LM head sits outside the scanned stack, but training still wraps it in
    # remat, and a custom_vjp that breaks under checkpointing is unusable.
    default, ordered = self._make_kernel_ordered_pair(4, 8)
    inputs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))

    def grads(layer):
      return nnx.grad(nnx.remat(lambda l, x: jnp.sum(jnp.sin(l(x)))))(layer, inputs).kernel[...]

    np.testing.assert_allclose(grads(default), grads(ordered), rtol=1e-5, atol=1e-6)

  def test_weight_grad_in_kernel_order_ignored_under_auto(self):
    # Under auto sharding XLA folds the transpose itself, so the custom_vjp buys
    # nothing and must not take over the dot.
    layer = linears.DenseGeneral(
        in_features_shape=4,
        out_features_shape=8,
        shard_mode=ShardMode.AUTO,
        weight_grad_in_kernel_order=True,
        rngs=nnx.Rngs(params=0),
    )
    inputs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    reference = linears.DenseGeneral(
        in_features_shape=4, out_features_shape=8, shard_mode=ShardMode.AUTO, rngs=nnx.Rngs(params=0)
    )
    np.testing.assert_array_equal(layer(inputs), reference(inputs))

  def test_weight_grad_in_kernel_order_yields_to_quantization(self):
    # The custom_vjp has to own the whole dot, so it must step aside for a
    # quantized one rather than wrap it.
    mock_quant = MagicMock()
    mock_quant.quant_mode = None
    mock_instance = mock_quant.dot_general_cls.return_value.return_value
    mock_instance.init_with_output.return_value = (MagicMock(), {})
    mock_instance.apply.return_value = (MagicMock(), {})

    layer = linears.DenseGeneral(
        in_features_shape=4,
        out_features_shape=8,
        shard_mode=ShardMode.EXPLICIT,
        quant=mock_quant,
        weight_grad_in_kernel_order=True,
        rngs=self.rngs,
    )
    inputs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    with patch.object(linears, "_dot_general_kernel_ordered_grad") as spy:
      layer(inputs)
    spy.assert_not_called()

  def test_weight_grad_in_kernel_order_emits_no_transpose(self):
    """The mechanism itself: no transpose on the weight-gradient path."""
    default, ordered = self._make_kernel_ordered_pair(4, 8)
    inputs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))

    def count_transposes(layer):
      def loss(l, x):
        return jnp.sum(jnp.sin(l(x)))

      text = nnx.jit(nnx.grad(loss)).lower(layer, inputs).as_text()
      return text.count("stablehlo.transpose")

    self.assertEqual(count_transposes(ordered), 0)
    self.assertGreater(count_transposes(default), 0)

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
