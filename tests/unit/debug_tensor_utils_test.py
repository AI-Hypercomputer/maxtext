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

# pylint: disable=protected-access
"""Unit tests for debug_tensor_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.utils import debug_tensor_interceptors
from maxtext.utils import debug_tensor_utils
import numpy as np


class DummyConfig:
  """Mock config for testing debug_tensor filtering."""

  def __init__(
      self,
      debug_tensor_distribution=False,
      debug_tensor_distribution_layers="all",
      debug_tensor_distribution_step_interval=1,
  ):
    self.debug_tensor_distribution = debug_tensor_distribution
    self.debug_tensor_distribution_layers = debug_tensor_distribution_layers
    self.debug_tensor_distribution_step_interval = debug_tensor_distribution_step_interval


class DebugTensorUtilsTest(parameterized.TestCase):

  def test_compute_stats_accuracy(self):
    np.random.seed(42)
    data = np.random.randn(10, 20).astype(np.float32)
    x = jnp.array(data)

    stats = debug_tensor_utils._compute_stats(x)

    np.testing.assert_allclose(float(stats["mean"]), float(np.mean(data)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["std"]), float(np.std(data)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["min"]), float(np.min(data)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["max"]), float(np.max(data)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["l2_norm"]), float(np.linalg.norm(data)), rtol=1e-5)

    expected_pcts = np.percentile(data, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
    np.testing.assert_allclose(float(stats["p01"]), float(expected_pcts[0]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p05"]), float(expected_pcts[1]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p25"]), float(expected_pcts[2]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p50"]), float(expected_pcts[3]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p75"]), float(expected_pcts[4]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p95"]), float(expected_pcts[5]), rtol=1e-4)
    np.testing.assert_allclose(float(stats["p99"]), float(expected_pcts[6]), rtol=1e-4)

    self.assertEqual(int(stats["nans"]), 0)
    self.assertEqual(int(stats["infs"]), 0)

  def test_compute_stats_nan_and_inf_detection(self):
    data = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    x = jnp.array(data)

    stats = debug_tensor_utils._compute_stats(x)
    self.assertEqual(int(stats["nans"]), 2)
    self.assertEqual(int(stats["infs"]), 2)

  def test_vjp_gradient_preservation(self):
    x = jnp.array([-2.0, -1.0, 0.5, 2.0, 3.0], dtype=jnp.float32)

    def f_instrumented(val):
      d = debug_tensor_utils.debug_tensor(val, "layer/activation", step=0, enabled=True)
      return jnp.sum(d**3)

    def f_reference(val):
      return jnp.sum(val**3)

    grad_instrumented = jax.grad(f_instrumented)(x)
    grad_reference = jax.grad(f_reference)(x)

    np.testing.assert_array_equal(grad_instrumented, grad_reference)
    np.testing.assert_array_equal(grad_instrumented, 3.0 * (x**2))

  def test_compute_stats_scalar(self):
    scalar = jnp.array(42.0, dtype=jnp.float32)
    stats = debug_tensor_utils._compute_stats(scalar)

    self.assertEqual(float(stats["mean"]), 42.0)
    self.assertEqual(float(stats["std"]), 0.0)
    self.assertEqual(float(stats["min"]), 42.0)
    self.assertEqual(float(stats["max"]), 42.0)
    self.assertEqual(float(stats["l2_norm"]), 42.0)
    self.assertEqual(float(stats["p50"]), 42.0)
    self.assertEqual(int(stats["nan_count"]), 0)
    self.assertEqual(int(stats["inf_count"]), 0)
    self.assertEqual(int(stats["nans"]), 0)
    self.assertEqual(int(stats["infs"]), 0)

  def test_zero_overhead_when_disabled(self):
    def f_instrumented_bool(x):
      y = x * 2.0
      y = debug_tensor_utils.debug_tensor(y, "my_tensor", enabled=False)
      return y + 1.0

    def f_instrumented_none(x):
      y = x * 2.0
      y = debug_tensor_utils.debug_tensor(y, "my_tensor", enabled=None)
      return y + 1.0

    def f_instrumented_cfg_disabled(x):
      y = x * 2.0
      cfg = DummyConfig(debug_tensor_distribution=False)
      y = debug_tensor_utils.debug_tensor(y, "my_tensor", enabled=cfg)
      return y + 1.0

    def f_plain(x):
      y = x * 2.0
      return y + 1.0

    x_dummy = jnp.zeros((4, 8), dtype=jnp.float32)
    jaxpr_plain = jax.make_jaxpr(f_plain)(x_dummy)
    jaxpr_bool = jax.make_jaxpr(f_instrumented_bool)(x_dummy)
    jaxpr_none = jax.make_jaxpr(f_instrumented_none)(x_dummy)
    jaxpr_cfg = jax.make_jaxpr(f_instrumented_cfg_disabled)(x_dummy)

    self.assertEqual(len(jaxpr_bool.eqns), len(jaxpr_plain.eqns))
    self.assertEqual(str(jaxpr_bool.jaxpr), str(jaxpr_plain.jaxpr))

    self.assertEqual(len(jaxpr_none.eqns), len(jaxpr_plain.eqns))
    self.assertEqual(str(jaxpr_none.jaxpr), str(jaxpr_plain.jaxpr))

    self.assertEqual(len(jaxpr_cfg.eqns), len(jaxpr_plain.eqns))
    self.assertEqual(str(jaxpr_cfg.jaxpr), str(jaxpr_plain.jaxpr))

  def test_should_debug_tensor_filtering(self):
    # None config
    self.assertFalse(debug_tensor_utils.should_debug_tensor(None, "test"))

    # Disabled config
    cfg_disabled = DummyConfig(debug_tensor_distribution=False)
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_disabled, "test"))

    # Enabled all layers
    cfg_all = DummyConfig(debug_tensor_distribution=True, debug_tensor_distribution_layers="all")
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_all, "attn/query_proj"))
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_all, "mlp/out_proj"))

    # Layer filtering
    cfg_filtered = DummyConfig(
        debug_tensor_distribution=True,
        debug_tensor_distribution_layers="attn,mlp",
    )
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_filtered, "attn/query_proj"))
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_filtered, "decoder/mlp/out_proj"))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_filtered, "norm/rms_norm"))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_filtered, "embed/token_embeddings"))

    # Numeric layer index filtering
    cfg_layer_idx = DummyConfig(
        debug_tensor_distribution=True,
        debug_tensor_distribution_layers="0,1",
    )
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_layer_idx, "decoder/layers_0/attn/query"))
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_layer_idx, "decoder/layers_1/moe/router"))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_layer_idx, "decoder/layers_2/attn/query"))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_layer_idx, "decoder/layers_10/attn/query"))

    # Step interval filtering
    cfg_interval = DummyConfig(
        debug_tensor_distribution=True,
        debug_tensor_distribution_layers="all",
        debug_tensor_distribution_step_interval=5,
    )
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_interval, "attn/query", step=0))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_interval, "attn/query", step=1))
    self.assertFalse(debug_tensor_utils.should_debug_tensor(cfg_interval, "attn/query", step=4))
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_interval, "attn/query", step=5))
    self.assertTrue(debug_tensor_utils.should_debug_tensor(cfg_interval, "attn/query", step=10))

  def test_debug_tensor_with_config(self):
    cfg_enabled = DummyConfig(debug_tensor_distribution=True)
    cfg_disabled = DummyConfig(debug_tensor_distribution=False)

    x = jnp.ones((2, 4), dtype=jnp.float32)

    # Enabled via config object
    out_enabled = debug_tensor_utils.debug_tensor(x, "tensor_a", step=0, enabled=cfg_enabled)
    np.testing.assert_array_equal(out_enabled, x)

    # Disabled via config object
    out_disabled = debug_tensor_utils.debug_tensor(x, "tensor_b", step=0, enabled=cfg_disabled)
    np.testing.assert_array_equal(out_disabled, x)

    # debug_tensor_from_config helper
    out_helper = debug_tensor_utils.debug_tensor_from_config(x, "tensor_c", cfg_enabled, step=0)
    np.testing.assert_array_equal(out_helper, x)

  def test_debug_telemetry_scope_activation(self):
    cfg_disabled = DummyConfig(debug_tensor_distribution=False)
    cfg_enabled = DummyConfig(debug_tensor_distribution=True)

    self.assertFalse(debug_tensor_interceptors.is_debug_telemetry_active())

    # Disabled scope -> should stay inactive
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_disabled, step=1):
      self.assertFalse(debug_tensor_interceptors.is_debug_telemetry_active())

    # Enabled scope -> should be active inside, inactive outside
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=3):
      self.assertTrue(debug_tensor_interceptors.is_debug_telemetry_active())
      active_cfg, active_step = debug_tensor_interceptors.get_active_telemetry_context()
      self.assertEqual(active_cfg, cfg_enabled)
      self.assertEqual(active_step, 3)

    self.assertFalse(debug_tensor_interceptors.is_debug_telemetry_active())

  def test_linen_interceptor_with_flax_module(self):
    class SimpleSubmodule(nn.Module):
      """Mock Linen submodule for testing."""

      @nn.compact
      def __call__(self, x):
        return x * 2.0

    class SimpleModel(nn.Module):
      """Mock Linen root model for testing."""

      @nn.compact
      def __call__(self, x):
        sub = SimpleSubmodule(name="sub")
        return sub(x) + 1.0

    cfg_enabled = DummyConfig(debug_tensor_distribution=True)
    x = jnp.ones((2, 3), dtype=jnp.float32)
    model = SimpleModel()

    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=0):
      out, _ = model.init_with_output(jax.random.PRNGKey(0), x)

    np.testing.assert_allclose(np.array(out), np.ones((2, 3)) * 3.0)

  def test_wrap_nnx_module_for_debug(self):
    class SubNNX(nnx.Module):
      """Mock NNX leaf module for testing."""

      def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))

      def __call__(self, x):
        return jnp.dot(x, self.w.value)

    class RootNNX(nnx.Module):
      """Mock NNX root module for testing."""

      def __init__(self, rngs: nnx.Rngs):
        self.sub = SubNNX(rngs)

      def __call__(self, x):
        return self.sub(x)

    rngs = nnx.Rngs(params=0)
    root = RootNNX(rngs)
    cfg_enabled = DummyConfig(debug_tensor_distribution=True)

    wrapped = debug_tensor_interceptors.wrap_nnx_module_for_debug(
        root, parent_path="decoder/layers_0", step=0, config=cfg_enabled
    )
    x = jnp.ones((2, 4), dtype=jnp.float32)
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=0):
      out = wrapped(x)
    self.assertEqual(out.shape, (2, 4))

  def test_wrap_nnx_moe_routing_methods(self):
    class DummyMoE(nnx.Module):
      """Mock NNX MoE module for testing."""

      def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))

      def get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None):
        return gate_logits * 2.0, jnp.zeros_like(gate_logits, dtype=jnp.int32)

      def reshape_and_update_weights(self, weights, indices):
        return weights * 1.5

      def __call__(self, x):
        weights, indices = self.get_topk(x, x)
        comb_weights = self.reshape_and_update_weights(weights, indices)
        return jnp.dot(comb_weights, self.w.value), jnp.array(0.05, dtype=jnp.float32)

    rngs = nnx.Rngs(params=0)
    moe = DummyMoE(rngs)
    cfg_enabled = DummyConfig(debug_tensor_distribution=True)

    wrapped = debug_tensor_interceptors.wrap_nnx_module_for_debug(
        moe, parent_path="decoder/layers_0/moe", step=0, config=cfg_enabled
    )
    x = jnp.ones((2, 4), dtype=jnp.float32)
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=0):
      out, lb = wrapped(x)
    self.assertEqual(out.shape, (2, 4))
    self.assertAlmostEqual(float(lb), 0.05, places=5)

  def test_dynamic_step_and_scope_switching(self):
    class SimpleNNX(nnx.Module):
      """Mock NNX simple module for testing."""

      def __call__(self, x):
        return x * 3.0

    node = SimpleNNX()
    cfg_interval = DummyConfig(
        debug_tensor_distribution=True,
        debug_tensor_distribution_step_interval=5,
    )
    wrapped = debug_tensor_interceptors.wrap_nnx_module_for_debug(node, parent_path="layer_0", config=cfg_interval)
    x = jnp.ones((2, 2), dtype=jnp.float32)

    # Step 0: should be active (0 % 5 == 0)
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_interval, step=0):
      out0 = wrapped(x)
      np.testing.assert_array_equal(out0, x * 3.0)

    # Step 1: should be active scope, but filtered out by step interval
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_interval, step=1):
      out1 = wrapped(x)
      np.testing.assert_array_equal(out1, x * 3.0)

    # Outside scope: should be inactive
    out_out = wrapped(x)
    np.testing.assert_array_equal(out_out, x * 3.0)

  def test_linen_interceptor_auxiliary_outputs(self):
    class MoeLinen(nn.Module):
      """Mock Linen MoE module for testing."""

      @nn.compact
      def __call__(self, x):
        return x * 2.0, jnp.array(0.01, dtype=jnp.float32)

    class AttnLinen(nn.Module):
      """Mock Linen attention module for testing."""

      @nn.compact
      def __call__(self, x):
        return x * 1.5, jnp.zeros((2, 4), dtype=jnp.float32)

    cfg = DummyConfig(debug_tensor_distribution=True)
    moe_mod = MoeLinen(name="moe_layer")
    attn_mod = AttnLinen(name="self_attention")

    x = jnp.ones((2, 4), dtype=jnp.float32)
    with debug_tensor_interceptors.debug_telemetry_scope(cfg, step=0):
      moe_out, _ = moe_mod.init_with_output(jax.random.PRNGKey(0), x)
      attn_out, _ = attn_mod.init_with_output(jax.random.PRNGKey(1), x)

    np.testing.assert_array_equal(moe_out[0], x * 2.0)
    self.assertAlmostEqual(float(moe_out[1]), 0.01, places=5)
    np.testing.assert_array_equal(attn_out[0], x * 1.5)
    np.testing.assert_array_equal(attn_out[1], jnp.zeros((2, 4)))

  def test_debug_tensor_moe_routing_expert_weights(self):
    # Shape: (batch=2, seq=4, num_experts=8)
    np.random.seed(42)
    weights = np.random.uniform(0.0, 1.0, size=(2, 4, 8)).astype(np.float32)
    x = jnp.array(weights)
    cfg_enabled = DummyConfig(debug_tensor_distribution=True)

    out = debug_tensor_utils.debug_tensor(x, "decoder/layer_0/moe/router_weights", step=0, enabled=cfg_enabled)
    np.testing.assert_array_equal(out, x)


if __name__ == "__main__":
  absltest.main()
