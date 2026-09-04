# Copyright 2023–2026 Google LLC
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
"""Mixture of Experts (MoE) tests."""

import functools
import re
from types import SimpleNamespace
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import ml_dtypes
import numpy as np
import qwix
import qwix.pallas as qpl
from jax.sharding import Mesh, PartitionSpec as P
from maxtext.common.common_types import Config, DType
from maxtext.configs import pyconfig
from maxtext.configs import types as maxtext_types
from maxtext.kernels.megablox import ops as mblx_ops
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers.moe import _all_gather_quantized_payload, _moe_combine_psum_scatter
from maxtext.layers.initializers import NdInitializer, nd_dense_init, variable_to_logically_partitioned
from maxtext.layers.quantizations import Fp8Quantization, WeightQuantConfig, configure_quantization
from maxtext.utils import max_logging, maxtext_utils, sharding, sparsecore
from maxtext.utils.sharding import remove_expert_from_partition_spec
from tests.unit.moe_megatron_aux_loss_test import _tiny_deepseek_config
from tests.utils.test_helpers import get_test_config_path
from tests.utils import linen_wrappers
import pytest


def compare_tree(a, b, relative_norm_diff_threshold=1e-02):
  """
  Compute the relative norm difference between two pytrees and enforce threshold.
  """
  # the first key is customized prefix define by user.
  # the rest of the keys are extracted from the path tuple.
  leaves_a, tree_def_a = jax.tree_util.tree_flatten_with_path(a)
  leaves_b, tree_def_b = jax.tree_util.tree_flatten_with_path(b)

  if tree_def_a != tree_def_b:
    raise ValueError("Reference and actual pytrees must have the same structure.")

  log_lines = ["\nCALCULATE DIFF"]
  num_failed = 0
  for (path, val_a), (_, val_b) in zip(leaves_a, leaves_b):
    path = "-".join([k.key for k in path if hasattr(k, "key")]) or "root"

    assert val_a.dtype in (jnp.float32, jnp.bfloat16, jnp.float16)
    val_a = val_a.astype(jnp.float32)
    val_b = val_b.astype(jnp.float32)

    max_abs_diff = jnp.max(jnp.abs(val_a - val_b))
    norm_a = jnp.linalg.norm(val_a)
    relative_norm_diff = jnp.linalg.norm(val_a - val_b) / norm_a if norm_a > 0 else max_abs_diff

    # Negated comparison (not <) rather than >=. NaN must count as a failure.
    failed = not relative_norm_diff < relative_norm_diff_threshold
    num_failed += failed
    log_lines.append(
        f"[{'FAIL' if failed else 'PASS'}] {path} | "
        f"max_abs_diff: {max_abs_diff:.3e} | "
        f"relative_norm_diff: {relative_norm_diff:.3e}"
    )
  diff_summary = "\n".join(log_lines)
  if num_failed:
    raise AssertionError(f"{diff_summary}\n{num_failed}/{len(leaves_a)} leaves exceed {relative_norm_diff_threshold=}.")
  max_logging.log(diff_summary)
  return diff_summary


def assert_moe_close(actual, expected, dtype):
  """Asserts that the actual and expected MoE outputs are close."""
  assert np.isfinite(actual).all(), "Actual output contains NaNs or Infs!"

  if jax.default_backend() == "tpu" or dtype == jnp.bfloat16:
    # TPU float32 (which is downcasted/accumulates differently) and bfloat16
    # both exhibit accumulation drift, especially on newer hardware like v7x.
    rtol, atol = 2e-2, 1e-2
  else:
    rtol, atol = 1e-5, 1e-6

  max_diff = float(np.max(np.abs(actual - expected)))
  rms_expected = float(np.sqrt(np.mean(np.square(expected))))
  max_logging.debug(
      f"\n[assert_moe_close] dtype={dtype}, max_diff={max_diff:.6f}, max_diff/RMS={max_diff/rms_expected:.6f}"
  )

  np.testing.assert_allclose(
      np.array(actual, dtype=np.float32), np.array(expected, dtype=np.float32), rtol=rtol, atol=atol, equal_nan=False
  )


class TokenDroppingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="token_dropping_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        capacity_factor=2,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_generate_masks(self):
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    # expert_capacity_in_batch = (4 * 2 / 8) * 2 = 2
    top_k_indices = jnp.array(
        [
            [[0, 5], [0, 4], [1, 0], [3, 5]],
            [[1, 2], [4, 1], [5, 0], [7, 1]],
            [[6, 2], [2, 3], [4, 2], [1, 2]],
            [[4, 1], [0, 7], [5, 0], [4, 7]],
        ]
    )
    softmax_probs = jnp.array(
        [
            [
                [0.20, 0, 0, 0, 0, 0.80, 0, 0],
                [0.68, 0, 0, 0, 0.32, 0, 0, 0],
                [0.22, 0.78, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.32, 0, 0.68, 0, 0],
            ],
            [
                [0, 0.26, 0.74, 0, 0, 0, 0, 0],
                [0, 0.79, 0, 0, 0.21, 0, 0, 0],
                [0.89, 0, 0, 0, 0, 0.11, 0, 0],
                [0, 0.11, 0, 0, 0, 0, 0, 0.89],
            ],
            [
                [0, 0, 0.26, 0, 0, 0, 0.74, 0],
                [0, 0, 0.88, 0.12, 0, 0, 0, 0],
                [0, 0, 0.17, 0, 0.83, 0, 0, 0],
                [0, 0.35, 0.65, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0.47, 0, 0, 0.53, 0, 0, 0],
                [0.36, 0, 0, 0, 0, 0, 0, 0.64],
                [0.15, 0, 0, 0, 0, 0.85, 0, 0],
                [0, 0, 0, 0, 0.18, 0, 0, 0.82],
            ],
        ]
    )

    # As expert_capacity_in_batch=2, so updated softmax_probs become (4 tokens were dropped):
    # softmax_probs = jnp.array([[[0.20, 0, 0, 0, 0, 0.80, 0, 0],
    #                             [0.68, 0, 0, 0, 0.32, 0, 0, 0],
    #                             [0, 0.78, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0.32, 0, 0.68, 0, 0]],
    #                            [[0, 0.26, 0.74, 0, 0, 0, 0, 0],
    #                             [0, 0.79, 0, 0, 0.21, 0, 0, 0],
    #                             [0.89, 0, 0, 0, 0, 0.11, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0, 0.89]],
    #                            [[0, 0, 0.26, 0, 0, 0, 0.74, 0],
    #                             [0, 0, 0.88, 0.12, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0.83, 0, 0, 0],
    #                             [0, 0.35, 0, 0, 0, 0, 0, 0]],
    #                            [[0, 0.47, 0, 0, 0.53, 0, 0, 0],
    #                             [0.36, 0, 0, 0, 0, 0, 0, 0.64],
    #                             [0.15, 0, 0, 0, 0, 0.85, 0, 0],
    #                             [0, 0, 0, 0, 0.18, 0, 0, 0.82]]])

    # shape of dispatch_mask & combine_mask: (batch_size, seq_len, num_experts, expert_capacity_per_batch)
    expected_combine_mask = jnp.array(
        [
            [
                [[0.2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.8, 0], [0, 0], [0, 0]],
                [[0, 0.68], [0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.78, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0.68], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.26, 0], [0.74, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0.79], [0, 0], [0, 0], [0.21, 0], [0, 0], [0, 0], [0, 0]],
                [[0.89, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.11, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.89, 0]],
            ],
            [
                [[0, 0], [0, 0], [0.26, 0], [0, 0], [0, 0], [0, 0], [0.74, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0.88], [0.12, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0.83, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.35, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.47, 0], [0, 0], [0, 0], [0.53, 0], [0, 0], [0, 0], [0, 0]],
                [[0.36, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.64, 0]],
                [[0, 0.15], [0, 0], [0, 0], [0, 0], [0, 0], [0.85, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0.18], [0, 0], [0, 0], [0, 0.82]],
            ],
        ],
        dtype=jnp.float32,
    )
    expected_dispatch_mask = expected_combine_mask.astype(bool)
    actual_dispatch_mask, actual_combine_mask = self.model.generate_masks(top_k_indices, softmax_probs)

    self.assertTrue((expected_dispatch_mask == actual_dispatch_mask).all())
    assert_moe_close(actual_combine_mask, expected_combine_mask, self.cfg.dtype)


class MlpBlockTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="mlp_block_init_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        capacity_factor=2,
    )
    self.rng = jax.random.PRNGKey(42)
    quant = Fp8Quantization()
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.model = linen_wrappers.to_linen(
        linears.MlpBlock,
        mesh=Mesh(devices_array, self.config.mesh_axes),
        config=self.config,
        in_features=2,
        intermediate_dim=2,
        activations=["silu", "linear"],
        intermediate_dropout_rate=0.0,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        name="mlp",
        quant=quant,
        use_bias=True,
    )

  @pytest.mark.external_serving
  def test_init(self):
    x = jnp.array([1.0, 2.0]).reshape((1, 1, 2))  # TODO(bug): need reshape due to error
    self.model.init({"params": self.rng, "dropout": self.rng}, x)


class DeepSeekRoutingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="deepseek_routing_test",
        enable_checkpointing=False,
        decoder_block="deepseek",
        dtype="bfloat16",
        max_target_length=2,
        max_prefill_predict_length=1,
        per_device_batch_size=1,
        n_routing_groups=4,
        topk_routing_group=2,
        num_experts=16,
        num_experts_per_tok=4,
        sparse_matmul=True,
        base_moe_mlp_dim=1024,
        base_mlp_dim=1024,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_deepseek_routing(self):
    # shape as [batch, sequence, num_experts] = [1,2,16]
    gate_logits = jnp.array(
        [
            [
                [0.20, 0.10, 0.05, 0.10, 0.10, 0.60, 0.30, 0.10, 0.80, 0.01, 0.01, 0.01, 0.05, 0.80, 0.20, 0.10],
                [0.68, 0.20, 0.06, 0.03, 0.32, 0.10, 0.05, 0.02, 0.65, 0.20, 0.04, 0.01, 0.32, 0.10, 0.05, 0.02],
            ]
        ]
    )
    pre_bias_logits = gate_logits - 0.5

    # 4 groups of 1st token:
    #  [0.20, 0.10, 0.05, 0.10] - sum top2 = 0.7
    #  [0.10, 0.60, 0.30, 0.10] - sum top2 = 0.9 (selected group) - index from 4 to 7
    #  [0.80, 0.01, 0.01, 0.01] - sum top2 = 0.81
    #  [0.05, 0.80, 0.20, 0.10] - sum top2 = 1.0 (selected group) - index from 12 to 15
    #
    # 4 groups of 2nd token
    #  [0.68, 0.20, 0.06, 0.03] - sum top2 = 0.88 (selected group) - index from 0 to 3
    #  [0.32, 0.10, 0.05, 0.02] - sum top2 = 0.42
    #  [0.65, 0.20, 0.04, 0.01] - sum top2 = 0.85 (selected group) - index from 8 to 11
    #  [0.32, 0.10, 0.05, 0.02] - sum top2 = 0.42
    #
    # From selected groups to choice top4 for each token
    expected_top_k_indices = jnp.array([[[13, 5, 6, 14], [0, 8, 1, 9]]])
    expected_top_k_weights = jnp.take_along_axis(pre_bias_logits, expected_top_k_indices, axis=-1)
    actual_top_k_weights, actual_top_k_indices = self.model.deepseek_routing(gate_logits, pre_bias_logits)
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_indices, actual_top_k_indices, rtol=1e-05, atol=1e-05, equal_nan=False)
    )
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_weights, actual_top_k_weights, rtol=1e-05, atol=1e-05, equal_nan=False)
    )

  def test_take_along_last_axis_dense_vjp_matches_take_along_axis(self):
    # Distinct top-k indices, as the router produces: forward and gradient must match the stock path exactly.
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 32), dtype=jnp.bfloat16)
    _, indices = jax.lax.top_k(logits, 4)
    cotangent = jax.random.normal(jax.random.PRNGKey(1), (2, 64, 4), dtype=jnp.float32)

    def loss(x, take):
      return jnp.sum(take(x).astype(jnp.float32) * cotangent)

    def stock(x):
      return jnp.take_along_axis(x, indices, axis=-1)

    def dense(x):
      return moe.take_along_last_axis_dense_vjp(x, indices, x.shape[-1])

    np.testing.assert_array_equal(np.asarray(stock(logits)), np.asarray(dense(logits)))
    np.testing.assert_array_equal(
        np.asarray(jax.grad(lambda x: loss(x, stock))(logits)),
        np.asarray(jax.grad(lambda x: loss(x, dense))(logits)),
    )
    lowered = jax.jit(jax.grad(lambda x: loss(x, dense))).lower(logits).as_text()
    self.assertNotIn("scatter", lowered)

  def test_deepseek_routing_topk_matmul_vjp(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="deepseek_routing_test",
        enable_checkpointing=False,
        decoder_block="deepseek",
        dtype="bfloat16",
        max_target_length=2,
        max_prefill_predict_length=1,
        per_device_batch_size=1,
        n_routing_groups=4,
        topk_routing_group=2,
        num_experts=16,
        num_experts_per_tok=4,
        sparse_matmul=True,
        base_moe_mlp_dim=1024,
        base_mlp_dim=1024,
        router_topk_matmul_vjp=True,
    )
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )
    gate_logits = jax.random.normal(jax.random.PRNGKey(2), (2, 8, 16), dtype=jnp.float32)
    pre_bias_logits = gate_logits - 0.5
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 4), dtype=jnp.float32)

    def loss(pre, m):
      w, _ = m.deepseek_routing(gate_logits, pre)
      return jnp.sum(w * cotangent)

    w_ref, i_ref = self.model.deepseek_routing(gate_logits, pre_bias_logits)
    w_new, i_new = model.deepseek_routing(gate_logits, pre_bias_logits)
    np.testing.assert_array_equal(np.asarray(i_ref), np.asarray(i_new))
    np.testing.assert_array_equal(np.asarray(w_ref), np.asarray(w_new))
    np.testing.assert_array_equal(
        np.asarray(jax.grad(loss)(pre_bias_logits, self.model)),
        np.asarray(jax.grad(loss)(pre_bias_logits, model)),
    )

  def test_deepseek_bias_updates(self):
    num_experts = 4
    rate = 0.01
    # total tokens = 16, average tokens = 16 / 4 = 4
    # expert 0 assigned 5 tokens --> overload, update: -0.01
    # expert 1 assigned 4 tokens --> same, update: 0.0
    # expert 2 assigned 3 tokens --> underload, update: 0.01
    # expert 3 assigned 4 tokens --> same, update: 0.0
    # [batch, sequence, top_k] = [2, 4, 2]
    top_k_indices = jnp.array([[[0, 1], [3, 0], [3, 1], [1, 0]], [[0, 3], [2, 0], [1, 2], [2, 3]]])
    expected_updates = jnp.array([-0.01, 0.0, 0.01, 0.0])
    actual_updates = moe.calculate_load_balance_updates(top_k_indices, num_experts, rate)

    assert_moe_close(actual_updates, expected_updates, jnp.float32)

  def test_batch_axis_names(self):
    # pylint: disable=protected-access
    self.assertIsNone(moe._batch_axis_names(None))
    self.assertEqual(moe._batch_axis_names(P("data", "context")), ("data", "context"))
    self.assertEqual(moe._batch_axis_names(P("stage", "data")), ("data",))
    self.assertEqual(
        moe._batch_axis_names(P(("data", "fsdp", "context_usp_ulysses", "expert"), "context")),
        ("data", "fsdp", "context_usp_ulysses", "expert", "context"),
    )

  def test_filter_axis_names(self):
    # pylint: disable=protected-access
    axes = moe._batch_axis_names(P(("data", "fsdp", "expert"), "context"))
    # When exclude_axes is a string:
    self.assertEqual(
        moe._filter_axis_names(axes, "expert"),
        ("data", "fsdp", "context"),
    )
    # When exclude_axes is a tuple:
    self.assertEqual(
        moe._filter_axis_names(axes, ("context", "expert")),
        ("data", "fsdp"),
    )
    # When all axes are filtered, result is None:
    only_ep = moe._batch_axis_names(P("expert", None))
    self.assertIsNone(moe._filter_axis_names(only_ep, "expert"))
    # When axis_names is None:
    self.assertIsNone(moe._filter_axis_names(None, "expert"))
    # When exclude_axes is None:
    self.assertEqual(moe._filter_axis_names(axes, None), axes)

  def test_deepseek_bias_updates_multi_slice_psum_reduction(self):
    """Verifies calculate_load_balance_updates reduces counts across mesh axes via real psum all-reduce."""
    num_experts, rate = 4, 0.01
    shard0_indices = jnp.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.int32).reshape((2, 4, 2))
    shard1_indices = jnp.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=jnp.int32).reshape((2, 4, 2))

    # Without psum, the two shards compute conflicting local updates:
    assert_moe_close(
        moe.calculate_load_balance_updates(shard0_indices, num_experts, rate),
        jnp.array([0.01, -0.01, 0.0, -0.01]),
        jnp.float32,
    )
    assert_moe_close(
        moe.calculate_load_balance_updates(shard1_indices, num_experts, rate),
        jnp.array([0.0, 0.01, 0.0, -0.01]),
        jnp.float32,
    )

    # Global counts after psum across data/fsdp shards:
    # total tokens = 6 + 6 + 8 + 12 = 32, average tokens per expert = 32 / 4 = 8
    # expert 0: 6 tokens  --> underload (8 - 6 = +2 > 0), update: +0.01
    # expert 1: 6 tokens  --> underload (8 - 6 = +2 > 0), update: +0.01
    # expert 2: 8 tokens  --> balanced  (8 - 8 =  0 == 0), update:  0.0
    # expert 3: 12 tokens --> overload  (8 - 12 = -4 < 0), update: -0.01
    expected_updates = jnp.array([0.01, 0.01, 0.0, -0.01])

    devices = jax.devices()
    n_dev = min(len(devices), 2)
    mesh = Mesh(np.array(devices[:n_dev]).reshape(n_dev, 1), axis_names=("data", "fsdp"))
    top_k_indices = jnp.concatenate([shard0_indices, shard1_indices], axis=0)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P(("data", "fsdp"), None, None),
        out_specs=P(),
        check_vma=False,
    )
    def compute_updates(indices):
      return moe.calculate_load_balance_updates(indices, num_experts, rate, axis_names=("data", "fsdp"))

    actual_updates = compute_updates(top_k_indices)
    assert_moe_close(actual_updates, expected_updates, jnp.float32)


class MoeLoopBlock(nnx.Module):
  """Reference implementation from https://github.com/mistralai/mistral-inference.
  This is not included anymore in our repo, due to a limitation of for-loop implementation in sharding.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      inputs_shape: tuple[int, ...],
      num_experts: int,
      num_experts_per_tok: int,
      kernel_init: NdInitializer,
      kernel_axes: tuple[str, ...],
      rngs: nnx.Rngs,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.bfloat16,
  ):
    self.config = config
    self.mesh = mesh
    self.inputs_shape = inputs_shape
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.gate = moe.GateLogit(
        in_features_shape=self.inputs_shape[-1],
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    for k in range(self.num_experts):
      expert_module = linears.MlpBlock(
          config=self.config,
          mesh=self.mesh,
          in_features=self.inputs_shape[-1],
          intermediate_dim=self.config.mlp_dim,
          activations=["silu", "linear"],
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=dtype,
          weight_dtype=weight_dtype,
          rngs=rngs,
      )
      setattr(self, f"mlp_{k}", expert_module)

  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = self.gate(inputs)[0]
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    mlp_lnx = jnp.zeros_like(inputs)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    for k in range(self.num_experts):
      weights_exp = jnp.sum(jnp.multiply(selected_experts == k, weights), axis=-1)
      getattr(self, f"mlp_{k}")
      mlp_lnx_exp = getattr(self, f"mlp_{k}")(inputs, deterministic=deterministic)
      mlp_lnx_exp = nn.with_logical_constraint(mlp_lnx_exp, ("activation_batch", "activation_length", "activation_embed"))
      mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
      mlp_lnx += mlp_lnx_exp

    return mlp_lnx


def get_moe_loop(
    config: Config,
    mesh: Mesh,
    inputs_shape: tuple[int, ...],
    num_experts: int,
    num_experts_per_tok: int,
    kernel_init: NdInitializer,
    kernel_axes: tuple[str, ...],
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.bfloat16,
):
  """Creates a MoeLoopBlock Linen module."""
  module = nnx_wrappers.to_linen(
      MoeLoopBlock,
      config=config,
      mesh=mesh,
      inputs_shape=inputs_shape,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      weight_dtype=weight_dtype,
      dtype=dtype,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


@pytest.mark.parametrize(
    ("expert_parallelism", "batch_partition"),
    (
        (1, None),
        (2, ("fsdp", "expert")),
    ),
)
def test_sparse_matmul_repairs_batch_specs_only_without_expert_parallelism(expert_parallelism, batch_partition):
  """Sparse MoE only replicates batches when expert routing remains local."""
  fake_moe = SimpleNamespace(
      config=SimpleNamespace(
          shard_exp_on_fsdp=False,
          use_2d_fsdp_sharding=False,
          shard_embed_moe_on_fsdp=False,
          model_name="qwen3.5-35b-a3b",
          check_vma=False,
          moe_fsdp_use_two_stage_all_gather=False,
          moe_pin_sparse_core_all_gathers=False,
          moe_flat_fsdp_weights=False,
          moe_dropless_fallback=None,
          load_balance_loss_weight=0.0,
      ),
      mesh=SimpleNamespace(
          axis_names=("diloco", "fsdp", "expert"), shape={"diloco": 2, "fsdp": 32, "expert": expert_parallelism}
      ),
      rngs=object(),
      get_expert_parallelism_size=lambda: expert_parallelism,
      _expert_parallelism_name="expert",
  )
  original_batch_partition = "fsdp" if expert_parallelism == 1 else ("fsdp", "expert")
  fake_moe._logical_to_mesh_axes = lambda logical_axes: P(  # pylint: disable=protected-access
      *(original_batch_partition if axis == "activation_batch" else None for axis in logical_axes)
  )
  fake_moe._maybe_shard_with_pspec = lambda value, _pspec, **_kwargs: value  # pylint: disable=protected-access
  fake_moe.sow = lambda *_args, **_kwargs: None

  inputs = SimpleNamespace(shape=(4, 1024, 2048))
  gate_logits = SimpleNamespace(shape=(4, 1024, 256))
  w0 = SimpleNamespace(shape=(256, 2048, 512))
  w1 = SimpleNamespace(shape=(256, 2048, 512))
  wo = SimpleNamespace(shape=(256, 512, 2048))
  captured = {}

  def fake_shard_map(function, *, mesh, in_specs, out_specs, check_vma):
    del function, mesh, check_vma
    captured["in_specs"] = in_specs
    captured["out_specs"] = out_specs
    return lambda x, *_args: (x, None, None, jnp.bool_(False), jnp.bool_(False), None)

  with mock.patch.object(jax, "shard_map", side_effect=fake_shard_map):
    output, _, _ = moe.RoutedMoE.sparse_matmul(
        fake_moe,
        inputs,
        gate_logits,
        None,
        w0,
        w1,
        wo,
        None,
        None,
        None,
    )

  assert output is inputs
  assert captured["in_specs"][0] == P(batch_partition, None, None)
  assert captured["in_specs"][1] == P(batch_partition, None, None)
  assert captured["in_specs"][2] is None
  assert captured["in_specs"][9] is None
  assert captured["out_specs"][0] == P(batch_partition, None, None)
  assert captured["out_specs"][3] == P(batch_partition)


# The TC ragged sort tests' moe_mlp_dim (256) is smaller than the default 1024 gmm_v2 mlp tiles.
_GMM_V2_SMALL_MLP_TILES = {f"{w}_tile_{p}_mlp_dim": 256 for w in ("wi", "wo") for p in ("fwd", "dlhs", "drhs")}


class RoutedMoeTest(parameterized.TestCase):
  """Routed Mixture of Experts test."""

  @parameterized.parameters(False, True)
  def test_permute_direct_token_gather_matches_repeat_and_sort(self, compute_gradient):
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    selected_experts = jnp.array(
        [
            [[3, 1], [0, 2], [1, 3]],
            [[2, 0], [3, 2], [0, 1]],
        ],
        dtype=jnp.int32,
    )
    weights = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
    gate_logits = jnp.zeros((2, 3, 4), dtype=jnp.float32)

    def permute(x, use_direct_token_gather):
      routed_moe = SimpleNamespace(
          config=SimpleNamespace(
              decoder_block=None,
              load_balance_loss_weight=0.0,
              moe_use_direct_token_gather=use_direct_token_gather,
              num_experts=4,
              use_ragged_sort=False,
              use_ring_of_experts=False,
          ),
          dtype=jnp.float32,
          is_hash_routing=False,
          num_experts=4,
          num_experts_per_tok=2,
          get_expert_parallelism_size=lambda: 1,
          get_topk=lambda *_args, **_kwargs: (weights, selected_experts),
          should_update_load_balance=lambda: False,
      )
      return moe.RoutedMoE.permute(routed_moe, x, gate_logits, gate_logits)

    if compute_gradient:
      cotangent = jnp.arange(48, dtype=jnp.float32).reshape(12, 4)
      expected = jax.grad(lambda x: jnp.sum(permute(x, False)[0] * cotangent))(inputs)
      result = jax.grad(lambda x: jnp.sum(permute(x, True)[0] * cotangent))(inputs)
      np.testing.assert_array_equal(result, expected)
    else:
      expected = permute(inputs, False)
      result = permute(inputs, True)
      for actual_value, expected_value in zip(result, expected):
        if expected_value is None:
          self.assertIsNone(actual_value)
        else:
          np.testing.assert_array_equal(actual_value, expected_value)

  def get_expected_output(self, rng, hidden_states, cfg, mesh):
    """Retrieve expected output from Routed Mixture of Experts."""
    model = get_moe_loop(
        config=cfg,
        mesh=mesh,
        inputs_shape=hidden_states.shape,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=jnp.float32,
        weight_dtype=cfg.weight_dtype,
    )
    variables = model.init(
        {"params": rng, "dropout": rng},
        jax.random.normal(
            rng, (int(cfg.per_device_batch_size) * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim)
        ),
    )

    output = jax.jit(model.apply)(variables, hidden_states.astype(jnp.float32))  # pylint: disable=not-callable
    return variables, output.astype(cfg.dtype)

  def get_moe_output(self, variables, hidden_states, cfg, mesh):
    """retrieve expected output from MoE"""
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    # convert format of parameters
    kernel = variables["params"]["gate"]["kernel"].value
    kernel = kernel.astype(cfg.weight_dtype)

    exp_wi_0 = []
    exp_wi_1 = []
    exp_wo = []

    for i in range(cfg.num_experts):
      tmp_wi_0 = variables["params"][f"mlp_{i}"]["wi_0"]["kernel"].value
      tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wi_1 = variables["params"][f"mlp_{i}"]["wi_1"]["kernel"].value
      tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wo = variables["params"][f"mlp_{i}"]["wo"]["kernel"].value
      tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))

      exp_wi_0.append(tmp_wi_0)
      exp_wi_1.append(tmp_wi_1)
      exp_wo.append(tmp_wo)

    wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
    wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
    wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)

    moe_variables = {"params": {"gate": {"kernel": kernel}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo}}

    output = jax.jit(model.apply)(moe_variables, hidden_states)  # pylint: disable=not-callable
    return output

  @pytest.mark.tpu_only
  def test_megablox(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_ragged_dot(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_ragged_dot_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_dense(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_dense_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="float32",
        megablox=False,
        sparse_matmul=False,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_moe_emb_chunking_random_routing(self):
    cfg_chunked = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_chunking_rr_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        num_moe_emb_chunks=4,
        use_ring_of_experts=True,
        ici_expert_parallelism=4,
        use_random_routing=True,
        mlp_bias=True,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    cfg_non_chunked = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_non_chunking_rr_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        num_moe_emb_chunks=0,
        ici_expert_parallelism=4,
        use_ring_of_experts=True,
        use_random_routing=True,
        mlp_bias=True,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg_chunked.per_device_batch_size) * device_count, cfg_chunked.max_target_length, cfg_chunked.base_emb_dim),
        dtype=cfg_chunked.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg_chunked)
    mesh = Mesh(devices_array, cfg_chunked.mesh_axes)

    moe_chunked = moe.RoutedMoE(
        config=cfg_chunked,
        num_experts=cfg_chunked.num_experts,
        num_experts_per_tok=cfg_chunked.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_chunked.dtype,
        rngs=nnx.Rngs(params=rng_model),
    )

    moe_non_chunked = moe.RoutedMoE(
        config=cfg_non_chunked,
        num_experts=cfg_non_chunked.num_experts,
        num_experts_per_tok=cfg_non_chunked.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_non_chunked.dtype,
        rngs=nnx.Rngs(params=rng_model),
    )

    moe_non_chunked.gate.kernel.value = moe_chunked.gate.kernel.value
    moe_non_chunked.wi_0.value = moe_chunked.wi_0.value
    moe_non_chunked.wi_1.value = moe_chunked.wi_1.value
    moe_non_chunked.wo.value = moe_chunked.wo.value

    chunked_out, _, _ = moe_chunked(hidden_states)
    non_chunked_out, _, _ = moe_non_chunked(hidden_states)

    self.assertTrue(jax.numpy.allclose(chunked_out, non_chunked_out, rtol=1e-01, atol=1e-01, equal_nan=False))

  @pytest.mark.tpu_only
  @pytest.mark.skip(reason="Correctness fails after adding EP. (b/540041424)")
  def test_moe_emb_chunking_gmm_v2(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_chunking_gmm_v2_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        ici_expert_parallelism=4,
        num_moe_emb_chunks=4,
        use_ring_of_experts=True,
        mlp_bias=True,
        per_device_batch_size=1,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_megablox_expert_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_expert_parallelism=4,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_ring_of_expert_and_tensor_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_ring_ep_tp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        ici_tensor_parallelism=2,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  def _run_ragged_sort_loss_and_grad(
      self,
      use_ring_of_experts: bool,
      ragged_buffer_factor: float = -1.0,
      ragged_gather_fallback: bool = False,
      ragged_gather_reduce_fallback: bool = False,
      ragged_sort_use_single_sparsecore: bool = False,
  ):
    """Loss and gradient correctness for the use_ragged_sort flag.

    Compares an EP run with use_ragged_sort=True against the same
    configuration with use_ragged_sort=False, sharing the same model variables
    and inputs. Both the scalar loss and the full pytree of parameter
    gradients must match within bf16 tolerance.
    """

    def _build_cfg(use_ragged_sort: bool):
      # Disable the buffer factor (-1.0) for the non-ragged sort baseline
      effective_buffer_factor = ragged_buffer_factor if use_ragged_sort else -1.0
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name=(f"moe_block_use_ragged_sort_{use_ragged_sort}" f"_ring_{use_ring_of_experts}_test"),
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          override_model_config=True,
          base_emb_dim=7168,  # we want emb dim being multiple of 1024 for fully using the kernel
          base_mlp_dim=256,
          base_moe_mlp_dim=256,
          dtype="bfloat16",
          megablox=True,
          sparse_matmul=True,
          per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
          ici_expert_parallelism=2,
          use_ring_of_experts=use_ring_of_experts,
          max_target_length=128,
          float32_gate_logits=True,
          use_ragged_sort=use_ragged_sort,
          ragged_buffer_factor=effective_buffer_factor,
          ragged_gather_fallback=ragged_gather_fallback,
          ragged_gather_reduce_fallback=ragged_gather_reduce_fallback,
          ragged_sort_use_single_sparsecore=ragged_sort_use_single_sparsecore,
      )

    def _build_model(cfg, mesh):
      return linen_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )

    def _loss_and_grad(model, variables, hidden_states):
      def loss_fn(params, x):
        out, lb_loss, _ = model.apply({"params": params}, x)
        loss = jnp.mean(out.astype(jnp.float32) ** 2)
        if lb_loss is not None:
          loss = loss + lb_loss.astype(jnp.float32)
        return loss

      return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))(variables["params"], hidden_states)

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()

    # Reference run: use_ragged_sort=False.
    cfg_ref = _build_cfg(use_ragged_sort=False)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg_ref.per_device_batch_size) * device_count, cfg_ref.max_target_length, cfg_ref.base_emb_dim),
        dtype=cfg_ref.dtype,
    )
    devices_array_ref = maxtext_utils.create_device_mesh(cfg_ref)
    mesh_ref = Mesh(devices_array_ref, cfg_ref.mesh_axes)
    model_ref = _build_model(cfg_ref, mesh_ref)
    with jax.set_mesh(mesh_ref), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      variables = model_ref.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      loss_ref, (grads_ref, x_grad_ref) = _loss_and_grad(model_ref, variables, hidden_states)

    # Target run: use_ragged_sort=True, sharing variables with the reference.
    cfg_rs = _build_cfg(use_ragged_sort=True)
    devices_array_rs = maxtext_utils.create_device_mesh(cfg_rs)
    mesh_rs = Mesh(devices_array_rs, cfg_rs.mesh_axes)
    model_rs = _build_model(cfg_rs, mesh_rs)
    with jax.set_mesh(mesh_rs), nn_partitioning.axis_rules(cfg_rs.logical_axis_rules):
      loss_rs, (grads_rs, x_grad_rs) = _loss_and_grad(model_rs, variables, hidden_states)

    # Loss correctness.
    self.assertTrue(
        jnp.allclose(loss_rs, loss_ref, rtol=1e-2, atol=1e-2),
        msg=f"Loss mismatch: ragged={loss_rs} ref={loss_ref}",
    )

    # Hidden-state gradient correctness. This is the cotangent that flows
    # through `ring_ragged_sort`'s custom_vjp backward (the kernel under
    # test). Without checking this, DCE removes the bwd entirely.
    self.assertEqual(x_grad_ref.shape, x_grad_rs.shape, "Hidden-state grad shape mismatch")
    x_atol = 8e-2 * jnp.max(jnp.abs(x_grad_ref.astype(jnp.float32)))
    self.assertTrue(
        jnp.allclose(x_grad_rs.astype(jnp.float32), x_grad_ref.astype(jnp.float32), rtol=1e-2, atol=x_atol),
        msg=(
            "Hidden-state gradient mismatch: max abs diff="
            f"{jnp.max(jnp.abs(x_grad_rs.astype(jnp.float32) - x_grad_ref.astype(jnp.float32)))}"
        ),
    )

    # Gradient correctness across the full pytree.
    leaves_ref, treedef_ref = jax.tree_util.tree_flatten(grads_ref)
    leaves_rs, treedef_rs = jax.tree_util.tree_flatten(grads_rs)
    self.assertEqual(treedef_ref, treedef_rs, "Gradient pytree structures differ")
    for i, (g_ref, g_rs) in enumerate(zip(leaves_ref, leaves_rs)):
      self.assertEqual(g_ref.shape, g_rs.shape, f"Grad shape mismatch at leaf {i}")
      # Scaled to the leaf: a fixed atol passes a leaf whose whole gradient is below it.
      atol = 8e-2 * jnp.max(jnp.abs(g_ref.astype(jnp.float32)))
      self.assertTrue(
          jnp.allclose(g_rs.astype(jnp.float32), g_ref.astype(jnp.float32), rtol=1e-2, atol=atol),
          msg=(
              f"Gradient mismatch at leaf {i} (shape={g_ref.shape}): "
              f"max abs diff={jnp.max(jnp.abs(g_rs.astype(jnp.float32) - g_ref.astype(jnp.float32)))}, {atol=}"
          ),
      )

  def _run_tc_ragged_sort_loss_and_grad(
      self,
      ragged_buffer_factor: float = 1.5,
      ref_overrides: dict | None = None,
      common_overrides: dict | None = None,
      randomize_biases: bool = False,
      check_jaxprs=None,
      quantization_rule: list | None = None,
      exact: bool = False,
      **tc_overrides,
  ):
    """Loss and gradient correctness for the moe_tc_ragged_sort flag and its options.

    Compares a ring-of-experts EP run with the TensorCore ragged sort (moe_tc_ragged_sort=True plus
    `tc_overrides`) against a reference run, sharing model variables and inputs. By default the reference is the
    SparseCore ragged sort with the same ragged buffer; `ref_overrides` changes it (e.g. to the TC sort without
    the option under test). `common_overrides` apply to both runs. All variants truncate each shard's slots to
    the same buffer, so loss, hidden-state gradient and parameter gradients must match within bf16 tolerance.
    `randomize_biases` replaces the zero-initialized biases with random values (for mlp_bias=True).
    `check_jaxprs(ref_jaxpr, tc_jaxpr)`, if given, is called with the printed loss-and-grad jaxprs of both runs
    (e.g. to check that a layout option actually changed the collectives).
    `quantization_rule`, if given, qwix-quantizes both models with these rules. `exact` requires loss and all
    gradients to be bit-identical instead of within bf16 tolerance.
    """

    def _build_cfg(moe_tc_ragged_sort: bool):
      overrides = {"moe_tc_ragged_sort": moe_tc_ragged_sort, **(common_overrides or {})}
      overrides.update(tc_overrides if moe_tc_ragged_sort else (ref_overrides or {}))
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name=f"moe_block_tc_ragged_sort_{moe_tc_ragged_sort}_test",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          override_model_config=True,
          base_emb_dim=7168,
          base_mlp_dim=256,
          base_moe_mlp_dim=256,
          dtype="bfloat16",
          megablox=True,
          sparse_matmul=True,
          per_device_batch_size=4,
          ici_expert_parallelism=2,
          use_ring_of_experts=True,
          max_target_length=128,
          float32_gate_logits=True,
          use_ragged_sort=True,
          ragged_buffer_factor=ragged_buffer_factor,
          **overrides,
      )

    def _build_model(cfg, mesh):
      if not quantization_rule:
        # Same RoutedMoE as QuantizedMoeTest; with quantization=fp8_full it also applies the Qwix FP8 gmm rule.
        return QuantizedMoeTest._build_and_quantize_moe_model(cfg, mesh)  # pylint: disable=protected-access
      model = linen_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )
      return qwix.quantize_model(model, qwix.QtProvider(quantization_rule))

    def _loss_and_grad_fn(model):
      def loss_fn(params, x):
        out, lb_loss, _ = model.apply({"params": params}, x)
        loss = jnp.mean(out.astype(jnp.float32) ** 2)
        if lb_loss is not None:
          loss = loss + lb_loss.astype(jnp.float32)
        return loss

      return jax.value_and_grad(loss_fn, argnums=(0, 1))

    def _loss_and_grad(model, variables, hidden_states):
      return jax.jit(_loss_and_grad_fn(model))(variables["params"], hidden_states)

    def _jaxpr(model, variables, hidden_states):
      return str(jax.make_jaxpr(_loss_and_grad_fn(model))(variables["params"], hidden_states))

    rng_model, rng_hidden_states = jax.random.split(jax.random.PRNGKey(2345))

    # Reference run: SparseCore ragged sort.
    cfg_ref = _build_cfg(moe_tc_ragged_sort=False)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg_ref.per_device_batch_size) * jax.device_count(), cfg_ref.max_target_length, cfg_ref.base_emb_dim),
        dtype=cfg_ref.dtype,
    )
    mesh_ref = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    model_ref = _build_model(cfg_ref, mesh_ref)
    with jax.set_mesh(mesh_ref), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      variables = model_ref.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      if randomize_biases:
        # Biases are zero-initialized; make them nonzero so the mlp_bias path is actually checked.
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(variables)
        keys = jax.random.split(jax.random.PRNGKey(7), len(path_leaves))
        variables = jax.tree_util.tree_unflatten(
            treedef,
            [
                (0.1 * jax.random.normal(k, x.shape, jnp.float32)).astype(x.dtype)
                if "bias" in jax.tree_util.keystr(path)
                else x
                for k, (path, x) in zip(keys, path_leaves)
            ],
        )
      loss_ref, (grads_ref, x_grad_ref) = _loss_and_grad(model_ref, variables, hidden_states)
      jaxpr_ref = _jaxpr(model_ref, variables, hidden_states) if check_jaxprs else None

    # Target run: TensorCore ragged sort, sharing variables with the reference.
    cfg_tc = _build_cfg(moe_tc_ragged_sort=True)
    mesh_tc = Mesh(maxtext_utils.create_device_mesh(cfg_tc), cfg_tc.mesh_axes)
    model_tc = _build_model(cfg_tc, mesh_tc)
    with jax.set_mesh(mesh_tc), nn_partitioning.axis_rules(cfg_tc.logical_axis_rules):
      loss_tc, (grads_tc, x_grad_tc) = _loss_and_grad(model_tc, variables, hidden_states)
      if check_jaxprs:
        check_jaxprs(jaxpr_ref, _jaxpr(model_tc, variables, hidden_states))

    # An all-zero gradient (e.g. an fp8 bwd quantization that flushed the cotangent) would pass trivially.
    self.assertGreater(
        float(jnp.max(jnp.abs(x_grad_ref.astype(jnp.float32)))), 0.0, "Reference hidden-state gradient is all zero"
    )
    if exact:
      np.testing.assert_array_equal(np.asarray(loss_tc), np.asarray(loss_ref), err_msg="Loss mismatch")
      np.testing.assert_array_equal(
          np.asarray(x_grad_tc.astype(jnp.float32)),
          np.asarray(x_grad_ref.astype(jnp.float32)),
          err_msg="Hidden-state gradient mismatch",
      )
      leaves_ref, treedef_ref = jax.tree_util.tree_flatten_with_path(grads_ref)
      leaves_tc, treedef_tc = jax.tree_util.tree_flatten_with_path(grads_tc)
      self.assertEqual(treedef_ref, treedef_tc, "Gradient pytree structures differ")
      for (path, g_ref), (_, g_tc) in zip(leaves_ref, leaves_tc):
        np.testing.assert_array_equal(
            np.asarray(g_tc.astype(jnp.float32)),
            np.asarray(g_ref.astype(jnp.float32)),
            err_msg=f"Gradient mismatch at {jax.tree_util.keystr(path)}",
        )
      return

    self.assertTrue(
        jnp.allclose(loss_tc, loss_ref, rtol=1e-2, atol=1e-2),
        msg=f"Loss mismatch: tc={loss_tc} sc={loss_ref}",
    )

    # The hidden-state cotangent flows through the TC sort's custom_vjp backward (the TC gather-reduce).
    self.assertEqual(x_grad_ref.shape, x_grad_tc.shape, "Hidden-state grad shape mismatch")
    x_ref32 = x_grad_ref.astype(jnp.float32)
    x_tc32 = x_grad_tc.astype(jnp.float32)
    x_atol = 8e-2 * jnp.max(jnp.abs(x_ref32))
    self.assertTrue(
        jnp.allclose(x_tc32, x_ref32, rtol=1e-2, atol=x_atol),
        msg=f"Hidden-state gradient mismatch: max abs diff={jnp.max(jnp.abs(x_tc32 - x_ref32))}",
    )

    leaves_ref, treedef_ref = jax.tree_util.tree_flatten(grads_ref)
    leaves_tc, treedef_tc = jax.tree_util.tree_flatten(grads_tc)
    self.assertEqual(treedef_ref, treedef_tc, "Gradient pytree structures differ")
    for i, (g_ref, g_tc) in enumerate(zip(leaves_ref, leaves_tc)):
      self.assertEqual(g_ref.shape, g_tc.shape, f"Grad shape mismatch at leaf {i}")
      g_ref32 = g_ref.astype(jnp.float32)
      g_tc32 = g_tc.astype(jnp.float32)
      atol = 8e-2 * jnp.max(jnp.abs(g_ref32))
      self.assertTrue(
          jnp.allclose(g_tc32, g_ref32, rtol=1e-2, atol=atol),
          msg=(
              f"Gradient mismatch at leaf {i} (shape={g_ref.shape}): "
              f"max abs diff={jnp.max(jnp.abs(g_tc32 - g_ref32))}, {atol=}"
          ),
      )

  def _run_topk_before_ep_all_gather_loss_and_grad(self, **overrides):
    """moe_topk_before_ep_all_gather=True matches the gathered-logits top-k in loss and gradients."""

    def _build_cfg(topk_before: bool):
      kwargs = {
          "enable_checkpointing": False,
          "model_name": "mixtral-8x7b",
          "override_model_config": True,
          "base_emb_dim": 512,
          "base_mlp_dim": 256,
          "base_moe_mlp_dim": 256,
          "dtype": "bfloat16",
          "weight_dtype": "float32",
          "megablox": False,
          "sparse_matmul": True,
          "per_device_batch_size": 4,
          "ici_expert_parallelism": 2,
          "use_ring_of_experts": True,
          "max_target_length": 64,
          "float32_gate_logits": True,
          "load_balance_loss_weight": 0.01,
          "use_ragged_sort": True,
          # CPU ragged_dot needs the truncated (local-expert) group sizes, so no dropless default here.
          "ragged_buffer_factor": 1.5,
          "ragged_gather_fallback": True,
          "ragged_gather_reduce_fallback": True,
          "moe_topk_before_ep_all_gather": topk_before,
      }
      kwargs.update(overrides)
      return pyconfig.initialize(
          [None, get_test_config_path()], run_name=f"moe_topk_before_ep_ag_{topk_before}", **kwargs
      )

    def _loss_and_grad(cfg, variables, hidden_states):
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = linen_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )

      def loss_fn(params, x):
        out, lb_loss, _ = model.apply({"params": params}, x)
        return jnp.mean(out.astype(jnp.float32) ** 2) + lb_loss.astype(jnp.float32), lb_loss

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        if variables is None:
          variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, hidden_states)
        (loss, lb_loss), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(
            variables["params"], hidden_states
        )
      return variables, loss, lb_loss, grads

    cfg_ref = _build_cfg(topk_before=False)
    hidden_states = jax.random.uniform(
        jax.random.PRNGKey(2345),
        (int(cfg_ref.per_device_batch_size) * jax.device_count(), cfg_ref.max_target_length, cfg_ref.base_emb_dim),
        dtype=cfg_ref.dtype,
    )
    variables, loss_ref, lb_ref, grads_ref = _loss_and_grad(cfg_ref, None, hidden_states)
    _, loss_new, lb_new, grads_new = _loss_and_grad(_build_cfg(topk_before=True), variables, hidden_states)

    # Same routing; only the bf16 accumulation order of the per-sequence probability mean differs.
    np.testing.assert_allclose(lb_new, lb_ref, rtol=1e-3)
    np.testing.assert_allclose(loss_new, loss_ref, rtol=1e-3)
    leaves_ref, treedef_ref = jax.tree_util.tree_flatten(grads_ref)
    leaves_new, treedef_new = jax.tree_util.tree_flatten(grads_new)
    self.assertEqual(treedef_ref, treedef_new)
    for g_ref, g_new in zip(leaves_ref, leaves_new):
      g_ref, g_new = np.asarray(g_ref, np.float32), np.asarray(g_new, np.float32)
      np.testing.assert_allclose(g_new, g_ref, rtol=1e-2, atol=1e-2 * float(np.max(np.abs(g_ref))))

  @pytest.mark.tpu_only
  def test_topk_before_ep_all_gather_loss_and_grad(self):
    self._run_topk_before_ep_all_gather_loss_and_grad()

  @pytest.mark.tpu_only
  def test_topk_before_ep_all_gather_loss_and_grad_token_chunks(self):
    self._run_topk_before_ep_all_gather_loss_and_grad(num_moe_token_chunks=2)

  @pytest.mark.tpu_only
  def test_topk_before_ep_all_gather_loss_and_grad_dropping_buffer(self):
    # Drops tokens, so the dropped set must match too.
    self._run_topk_before_ep_all_gather_loss_and_grad(ragged_buffer_factor=0.5)

  @pytest.mark.tpu_only
  def test_topk_before_ep_all_gather_loss_and_grad_sparse_core(self):
    self._run_topk_before_ep_all_gather_loss_and_grad(
        base_emb_dim=7168,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=True,
        ragged_buffer_factor=-1.0,
        ragged_gather_fallback=False,
        ragged_gather_reduce_fallback=False,
    )

  def _moe_routing_maps_cfg(self, moe_routing_maps: str, **overrides):
    """Tiny ring-of-experts ragged-sort MoE config (EP2) for the moe_routing_maps test."""
    kwargs = {
        "enable_checkpointing": False,
        "model_name": "mixtral-8x7b",
        "override_model_config": True,
        "base_emb_dim": 512,
        "base_mlp_dim": 256,
        "base_moe_mlp_dim": 256,
        "dtype": "bfloat16",
        "weight_dtype": "float32",
        "megablox": False,
        "sparse_matmul": True,
        "per_device_batch_size": 4,
        "ici_expert_parallelism": 2,
        "use_ring_of_experts": True,
        "max_target_length": 64,
        "float32_gate_logits": True,
        "load_balance_loss_weight": 0.01,
        "use_ragged_sort": True,
        "ragged_buffer_factor": 1.5,
        "ragged_gather_fallback": True,
        "ragged_gather_reduce_fallback": True,
        "moe_routing_maps": moe_routing_maps,
        **overrides,
    }
    if kwargs.get("moe_tc_ragged_weights_on_activation"):
      kwargs["moe_tc_routing_checkpoint"] = moe_routing_maps
    return pyconfig.initialize([None, get_test_config_path()], run_name=f"moe_routing_maps_{moe_routing_maps}", **kwargs)

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("sparsecore_ragged_sort", {}),
      ("tc_ragged_sort", {"moe_tc_ragged_sort": True}),
      # moe_tc_routing_checkpoint also saves the per-buffer-row routing weights.
      ("tc_ragged_sort_weights_on_activation", {"moe_tc_ragged_sort": True, "moe_tc_ragged_weights_on_activation": True}),
      # Window routing also saves the group sizes.
      (
          "tc_ragged_sort_window_weights_on_activation",
          {"moe_tc_ragged_sort": True, "moe_tc_window_routing": True, "moe_tc_ragged_weights_on_activation": True},
      ),
  )
  def test_moe_routing_maps_remat_loss_and_grad(self, overrides):
    """moe_routing_maps=moe_tc_routing_checkpoint=device under a remat saving only those names: bit-identical results."""

    def _loss_and_grad(cfg, variables, hidden_states):
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = linen_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )

      # Same remat as the custom policy with only moe_routing_maps=moe_tc_routing_checkpoint=device.
      @functools.partial(
          jax.checkpoint,
          policy=jax.checkpoint_policies.save_only_these_names("moe_routing_maps", "moe_tc_routing_checkpoint"),
      )
      def apply(params, x):
        return model.apply({"params": params}, x)

      def loss_fn(params, x):
        out, lb_loss, _ = apply(params, x)
        return jnp.mean(out.astype(jnp.float32) ** 2) + lb_loss.astype(jnp.float32), lb_loss

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        if variables is None:
          variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, hidden_states)
        step = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
        (loss, lb_loss), grads = step(variables["params"], hidden_states)
      return variables, loss, lb_loss, grads

    cfg_ref = self._moe_routing_maps_cfg("remat", **overrides)
    hidden_states = jax.random.uniform(
        jax.random.PRNGKey(2345),
        (int(cfg_ref.per_device_batch_size) * jax.device_count(), cfg_ref.max_target_length, cfg_ref.base_emb_dim),
        dtype=cfg_ref.dtype,
    )
    variables, loss_ref, lb_ref, grads_ref = _loss_and_grad(cfg_ref, None, hidden_states)
    _, loss_new, lb_new, grads_new = _loss_and_grad(
        self._moe_routing_maps_cfg("device", **overrides), variables, hidden_states
    )

    # Saved values equal the recomputed ones, so every value is bit-identical.
    np.testing.assert_array_equal(loss_new, loss_ref)
    np.testing.assert_array_equal(lb_new, lb_ref)
    leaves_ref, treedef_ref = jax.tree_util.tree_flatten(grads_ref)
    leaves_new, treedef_new = jax.tree_util.tree_flatten(grads_new)
    self.assertEqual(treedef_ref, treedef_new)
    for g_ref, g_new in zip(leaves_ref, leaves_new):
      np.testing.assert_array_equal(np.asarray(g_new), np.asarray(g_ref))

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_ring_of_experts(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=True)

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_ring_of_experts_ragged_buffer(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=True, ragged_buffer_factor=1.5)

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_ring_of_experts_fallback(self):
    self._run_ragged_sort_loss_and_grad(
        use_ring_of_experts=True, ragged_gather_fallback=True, ragged_gather_reduce_fallback=True
    )

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_no_ring_of_experts(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=False)

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_no_ring_of_experts_ragged_buffer(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=False, ragged_buffer_factor=1.5)

  @pytest.mark.tpu_only
  def test_ragged_sort_loss_and_grad_no_ring_of_experts_fallback(self):
    self._run_ragged_sort_loss_and_grad(
        use_ring_of_experts=False, ragged_gather_fallback=True, ragged_gather_reduce_fallback=True
    )

  @pytest.mark.tpu_only
  def test_ragged_sort_single_sparsecore_ring_of_experts(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=True, ragged_sort_use_single_sparsecore=True)

  @pytest.mark.tpu_only
  def test_ragged_sort_single_sparsecore_no_ring_of_experts(self):
    self._run_ragged_sort_loss_and_grad(use_ring_of_experts=False, ragged_sort_use_single_sparsecore=True)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_loss_and_grad(self):
    self._run_tc_ragged_sort_loss_and_grad()

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_loss_and_grad_truncated_buffer(self):
    # A buffer smaller than the balanced load, so shards drop slots exactly as the SparseCore sort does.
    self._run_tc_ragged_sort_loss_and_grad(ragged_buffer_factor=0.75)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_loss_and_grad_flatten(self):
    self._run_tc_ragged_sort_loss_and_grad(moe_tc_ragged_flatten_block_size=256)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_loss_and_grad_no_mask_padding(self):
    self._run_tc_ragged_sort_loss_and_grad(moe_tc_ragged_mask_padding=False)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_loss_and_grad_small_blocks(self):
    self._run_tc_ragged_sort_loss_and_grad(moe_tc_ragged_gather_block_size=128, moe_tc_ragged_reduce_block_size=128)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_weights_on_activation(self):
    self._run_tc_ragged_sort_loss_and_grad(moe_tc_ragged_weights_on_activation=True)

  @pytest.mark.tpu_only
  def test_tc_ragged_sort_weights_on_activation_truncated_flatten(self):
    self._run_tc_ragged_sort_loss_and_grad(
        ragged_buffer_factor=0.75, moe_tc_ragged_weights_on_activation=True, moe_tc_ragged_flatten_block_size=256
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(("weights_on_output", False), ("weights_on_activation", True))
  def test_tc_ragged_sort_mlp_bias(self, weights_on_activation):
    """With mlp_bias the wo bias must also be weighted by the routing weight (incl. weights on the activation)."""
    self._run_tc_ragged_sort_loss_and_grad(
        common_overrides={"mlp_bias": True},
        randomize_biases=True,
        moe_tc_ragged_weights_on_activation=weights_on_activation,
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("default", 1.5, {}),
      ("truncated", 0.75, {}),
      ("weights_on_activation", 1.5, {"moe_tc_ragged_weights_on_activation": True}),
      ("accumulate_wi_dlhs", 1.5, {"moe_accumulate_wi_dlhs": True}),
      ("dlhs_transpose_rhs", 1.5, {"moe_gmm_v2_dlhs_transpose_rhs": True}),
      ("accumulate_chunk_wgrad", 1.5, {"moe_accumulate_chunk_wgrad": True, "num_moe_token_chunks": 2}),
      (
          "all_gmm_options",
          1.5,
          {
              "moe_accumulate_wi_dlhs": True,
              "moe_gmm_v2_dlhs_transpose_rhs": True,
              "moe_accumulate_chunk_wgrad": True,
              "num_moe_token_chunks": 2,
              "moe_tc_ragged_weights_on_activation": True,
          },
      ),
  )
  def test_tc_ragged_sort_3d_gmm(self, ragged_buffer_factor, extra):
    """moe_tc_ragged_3d_gmm (3D-layout gmm_v2 / tgmm_v2) matches the TC sort with the 2D gmm_v2 kernels."""
    self._run_tc_ragged_sort_loss_and_grad(
        ragged_buffer_factor=ragged_buffer_factor,
        common_overrides={**_GMM_V2_SMALL_MLP_TILES, "use_tokamax_gmm": True, "use_gmm_v2": True},
        ref_overrides={"moe_tc_ragged_sort": True, **extra},
        moe_tc_ragged_3d_gmm=True,
        **extra,
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("default", 1.5, {}),
      ("truncated_weights_on_activation", 0.75, {"moe_tc_ragged_weights_on_activation": True}),
      ("rowwise_combine_bwd", 1.5, {"moe_quantize_combine_bwd_method": "rowwise"}),
      ("fixed_combine_bwd", 1.5, {"moe_quantize_combine_bwd_method": "fixed,0.01"}),
  )
  def test_tc_ragged_sort_3d_dispatch(self, ragged_buffer_factor, extra):
    """moe_tc_ragged_3d_dispatch (3D-layout token all-gather / combine) matches the 2D dispatch."""
    common = {
        **_GMM_V2_SMALL_MLP_TILES,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "moe_tc_ragged_sort": True,
        "moe_tc_ragged_3d_gmm": True,
    }
    # emb 7168 = 56 x 128: the token all-gather and the combine reduce-scatter (and their transposes in the
    # backward pass) must run on (..., 56, 128) tensors with the flag, and on (..., 7168) ones without it.
    collective_3d = re.compile(r"\[(?:\d+,)*56,128\](?:\{[^}]*\})? = (all_gather|reduce_scatter)\[")

    def check_jaxprs(jaxpr_ref, jaxpr_tc):
      self.assertEqual(set(collective_3d.findall(jaxpr_tc)), {"all_gather", "reduce_scatter"})
      self.assertEqual(collective_3d.findall(jaxpr_ref), [])

    self._run_tc_ragged_sort_loss_and_grad(
        ragged_buffer_factor=ragged_buffer_factor,
        common_overrides={**common, **extra},
        ref_overrides={},
        check_jaxprs=check_jaxprs,
        moe_tc_ragged_3d_dispatch=True,
    )

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("prequant_before_unsort", {"moe_bwd_prequant_before_unsort": True}),
      ("combine_bwd_direct_qarray", {"moe_bwd_prequant_before_unsort": True, "moe_combine_bwd_direct_qarray": True}),
      ("3d_dlhs_partial_sum", {"moe_3d_dlhs_partial_sum": True}),
      ("no_mask_padding", {"moe_bwd_prequant_before_unsort": True, "moe_tc_ragged_mask_padding": False}),
      ("prequant_window_routing", {"moe_bwd_prequant_before_unsort": True, "moe_tc_window_routing": True}),
      ("fp8_prequant_before_unsort", {"fp8": True, "moe_bwd_prequant_before_unsort": True}),
      (
          "fp8_all",
          {
              "fp8": True,
              "moe_bwd_prequant_before_unsort": True,
              "moe_combine_bwd_direct_qarray": True,
              "moe_3d_dlhs_partial_sum": True,
          },
      ),
  )
  def test_tc_ragged_sort_3d_dispatch_bwd_fusions(self, flags):
    """The fused TC unsort / EP combine backward and the 3D DLHS partial_sum match the unfused 3D dispatch."""
    common = {
        **_GMM_V2_SMALL_MLP_TILES,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "moe_tc_ragged_sort": True,
        "moe_tc_ragged_3d_gmm": True,
        "moe_tc_ragged_3d_dispatch": True,
        "moe_tc_ragged_weights_on_activation": True,
        "moe_accumulate_wi_dlhs": True,
        "moe_tc_ragged_mask_padding": flags.get("moe_tc_ragged_mask_padding", True),
        # Small TC blocks keep the gather / gather-reduce VMEM scratch (~4 / ~7 MiB at emb 7168) within the
        # default scoped VMEM of every TPU generation; the default 1024-row gather block needs ~29 MiB.
        "moe_tc_ragged_gather_block_size": 128,
        "moe_tc_ragged_reduce_block_size": 128,
    }
    if flags.get("fp8"):
      if pltpu.get_tpu_info().fp8_ops_per_second == 0:
        # gmm_v2 only applies a static lhs scale when it can quantize lhs for an fp8 matmul.
        self.skipTest("A fixed act scale in gmm_v2 needs fp8 matmul hardware.")
      # FP8 with fixed per-tensor scales (the only scales the fused combine supports): exercises the fused fp8
      # QArray TC gather / EP all-gather. The bwd bound is sized to this test's ~1e-8 output cotangents
      # (mean-reduced loss) so e5m2 does not flush them to zero. The router is not covered by the Qwix gmm rule,
      # and float32_gate_logits=True requires quantize_router_proj=False.
      common.update(
          quantization="fp8_full",
          use_qwix_quantization=True,
          weight_quantization_calibration_method="fixed,-224,224",
          act_quantization_calibration_method="fixed,-224,224",
          bwd_quantization_calibration_method="fixed,1e-5",
          quantize_router_proj=False,
      )
    self._run_tc_ragged_sort_loss_and_grad(
        ragged_buffer_factor=0.75,
        common_overrides=common,
        ref_overrides={},
        **{k: v for k, v in flags.items() if k not in ("moe_tc_ragged_mask_padding", "fp8")},
    )

  def _fp8_bwd_fusion_overrides(self, act_calibration_method: str, disable_channelwise_axes: bool):
    """3D TC dispatch overrides plus an fp8 qwix rule (fixed weight / bwd scales) for the fused-unsort tests."""
    common = {
        **_GMM_V2_SMALL_MLP_TILES,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "moe_tc_ragged_sort": True,
        "moe_tc_ragged_3d_gmm": True,
        "moe_tc_ragged_3d_dispatch": True,
        "moe_tc_ragged_weights_on_activation": True,
        "moe_tc_ragged_gather_block_size": 128,
        "moe_tc_ragged_reduce_block_size": 128,
        # No config quantization: the test applies its own qwix rule (which may disable channelwise axes, unlike
        # the config's rule), and use_qwix_quantization routes the gmm through it.
        "use_qwix_quantization": True,
        "weight_quantization_calibration_method": "fixed,-224,224",
        "act_quantization_calibration_method": act_calibration_method,
        # Sized to the ~1e-8 output cotangents (mean-reduced loss) so e5m2 does not flush them to zero.
        "bwd_quantization_calibration_method": "fixed,1e-5",
    }
    rule = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            weight_calibration_method=common["weight_quantization_calibration_method"],
            act_calibration_method=act_calibration_method,
            bwd_calibration_method=common["bwd_quantization_calibration_method"],
            disable_channelwise_axes=disable_channelwise_axes,
            op_names=("gmm", "ragged_dot"),
        )
    ]
    return common, rule

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("fixed_act", "fixed,-224,224", False, {}),
      ("fixed_act_no_channelwise", "fixed,-224,224", True, {}),
      ("absmax_act_no_channelwise", "absmax", True, {}),
      ("absmax_act_window_routing", "absmax", True, {"moe_tc_window_routing": True}),
  )
  def test_tc_ragged_sort_3d_dispatch_bwd_prequant_fp8(self, act_calibration_method, disable_channelwise_axes, flags):
    """With a fixed fp8 bwd scale, quantizing the cotangent before the TC unsort-gather is bit-identical to after."""
    if act_calibration_method.startswith("fixed") and pltpu.get_tpu_info().fp8_ops_per_second == 0:
      # gmm_v2 only applies a static lhs scale when it can quantize lhs for an fp8 matmul.
      self.skipTest("A fixed act scale in gmm_v2 needs fp8 matmul hardware.")
    common, rule = self._fp8_bwd_fusion_overrides(act_calibration_method, disable_channelwise_axes)
    self._run_tc_ragged_sort_loss_and_grad(
        ragged_buffer_factor=0.75,
        common_overrides=common,
        ref_overrides={},
        quantization_rule=rule,
        exact=True,
        moe_bwd_prequant_before_unsort=True,
        **flags,
    )

  _COMBINE = ("expert", 128)

  @parameterized.named_parameters(
      ("per_row_bwd", {"bwd": "absmax"}, False, None, "requires a per-tensor bwd scale"),
      ("per_tensor_dynamic_bwd", {"bwd": "absmax"}, True, None, None),
      ("per_tensor_dynamic_weight_act", {"weight": "absmax", "act": "absmax"}, True, None, None),
      ("dynamic_bwd_with_combine", {"bwd": "absmax"}, True, _COMBINE, "requires a fixed bwd_calibration_method"),
      ("dynamic_weight_with_combine", {"weight": "absmax"}, True, _COMBINE, "requires a fixed weight_calibration_method"),
      ("dynamic_act_with_combine", {"act": "absmax"}, True, _COMBINE, "requires a fixed act_calibration_method"),
      ("fixed_with_combine", {}, False, _COMBINE, None),
  )
  def test_tc_unsort_bwd_rejects_dynamic_scales(self, dynamic, no_channelwise, combine, error):
    """The fused unsort backward refuses scales it can't apply in token order or before the EP all-gather."""
    calibration = {name: dynamic.get(name, "fixed,-224,224") for name in ("weight", "act", "bwd")}
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method=calibration["weight"],
        act_calibration_method=calibration["act"],
        bwd_calibration_method=calibration["bwd"],
        disable_channelwise_axes=no_channelwise,
        op_names=("gmm",),
    )
    cfg = mblx_ops.TcUnsortCfg(
        unpadded_cap=8, num_out_tokens=4, topk=2, blocks=(128, 0, True, True), mask_padding=True, combine_rs_cfg=combine
    )
    lhs, rhs = jnp.zeros((8, 128), jnp.bfloat16), jnp.zeros((2, 128, 128), jnp.bfloat16)
    if error is None:
      mblx_ops._check_tc_unsort_bwd_supported(lhs, rhs, rule, cfg)  # pylint: disable=protected-access
    else:
      with self.assertRaisesRegex(NotImplementedError, error):
        mblx_ops._check_tc_unsort_bwd_supported(lhs, rhs, rule, cfg)  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("same_dtype", jnp.bfloat16, jnp.bfloat16, True),
      ("scale_dtype_mismatch", jnp.bfloat16, jnp.float32, False),
  )
  def test_lhs_rhs_share_fixed_scale_requires_same_scale_dtype(self, lhs_dtype, rhs_dtype, expected):
    """A fixed scale is built in the quantized array's dtype, so scales of different dtypes are not shared."""
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        weight_calibration_method="fixed,-200,200",
        act_calibration_method="fixed,-200,200",
        op_names=("gmm",),
    )
    lhs = qpl.quantize(
        jnp.ones((8, 128), lhs_dtype), jnp.float8_e4m3fn, channelwise_axes=[], calibration_method="fixed,-200,200"
    )
    rhs = qpl.quantize(
        jnp.ones((2, 128, 128), rhs_dtype), jnp.float8_e4m3fn, channelwise_axes=[], calibration_method="fixed,-200,200"
    )
    self.assertEqual(mblx_ops._lhs_rhs_share_fixed_scale(rule, lhs, rhs), expected)  # pylint: disable=protected-access

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("overflow", 0.1, True),
      ("dropless", -1.0, False),
  )
  def test_ragged_sort_overflow_detection(self, ragged_buffer_factor, expect_overflow):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"moe_overflow_detection_{ragged_buffer_factor}",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=7168,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        max_target_length=128,
        float32_gate_logits=True,
        use_ragged_sort=True,
        ragged_buffer_factor=ragged_buffer_factor,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      _, mutated = model.apply({"params": variables["params"]}, hidden_states, mutable=["intermediates"])

    has_overflow = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_has_overflow")
    self.assertTrue(has_overflow, "Expected a moe_has_overflow intermediate to be sown.")
    any_overflow = bool(jnp.any(jnp.array([jnp.any(x) for x in has_overflow])))

    if expect_overflow:
      self.assertTrue(any_overflow, "Expected has_overflow=True with a tiny ragged_buffer_factor.")
    else:
      self.assertFalse(any_overflow, "Expected has_overflow=False with a dropless (worst-case) buffer.")

  def _build_retry_test_mesh(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_retry_test_probe",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        ici_expert_parallelism=2,
    )
    return Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)

  def _build_retry_test_model(
      self,
      mesh,
      ragged_buffer_factor,
      moe_dropless_fallback: str | None = None,
      force_dropless: bool = False,
  ):
    """Builds a mixtral-8x7b RoutedMoE with the given ragged buffer/retry settings."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=(f"moe_retry_test_{ragged_buffer_factor}_{moe_dropless_fallback}_{force_dropless}"),
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=7168,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        max_target_length=128,
        float32_gate_logits=True,
        use_ragged_sort=True,
        ragged_buffer_factor=ragged_buffer_factor,
        moe_dropless_fallback=moe_dropless_fallback,
    )
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
        force_dropless=force_dropless,
    )
    return cfg, model

  def _out_loss_and_grad(self, model, params, hidden_states):
    def loss_fn(p):
      out, lb_loss, _ = model.apply({"params": p}, hidden_states)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      return loss + (lb_loss.astype(jnp.float32) if lb_loss is not None else 0.0), out

    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
    return out, loss, grads

  @pytest.mark.tpu_only
  def test_force_dropless_matches_dropless(self):
    """A tiny buffer + force_dropless=True matches dropless output and gradients; without it, they diverge."""
    mesh = self._build_retry_test_mesh()
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()

    cfg_dropless, model_dropless = self._build_retry_test_model(mesh, ragged_buffer_factor=-1.0)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (
            int(cfg_dropless.per_device_batch_size) * device_count,
            cfg_dropless.max_target_length,
            cfg_dropless.base_emb_dim,
        ),
        dtype=cfg_dropless.dtype,
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      variables = model_dropless.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      out_dropless, _, grads_dropless = self._out_loss_and_grad(model_dropless, variables["params"], hidden_states)

    _, model_retry = self._build_retry_test_model(
        mesh, ragged_buffer_factor=0.1, moe_dropless_fallback="step", force_dropless=True
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_retry, _, grads_retry = self._out_loss_and_grad(model_retry, variables["params"], hidden_states)

    _, model_no_retry = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_no_retry, _, _ = self._out_loss_and_grad(model_no_retry, variables["params"], hidden_states)

    # Sanity check: the buffer must actually force drops without the flag, else this test proves nothing.
    self.assertFalse(
        jnp.allclose(out_no_retry.astype(jnp.float32), out_dropless.astype(jnp.float32), rtol=1e-2, atol=1e-2),
        msg="moe_dropless_fallback=None unexpectedly matches dropless -- buffer isn't forcing an overflow.",
    )

    assert_moe_close(out_retry, out_dropless, cfg_dropless.dtype)
    for g_retry, g_dropless in zip(jax.tree_util.tree_leaves(grads_retry), jax.tree_util.tree_leaves(grads_dropless)):
      assert_moe_close(g_retry, g_dropless, cfg_dropless.dtype)

  @pytest.mark.tpu_only
  def test_step_dropless_fallback_asymmetric_shard_overflow(self):
    """Overflow flag and replay must fire correctly when only one EP shard overflows, not both."""
    mesh = self._build_retry_test_mesh()
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()

    cfg_dropless, model_dropless = self._build_retry_test_model(mesh, ragged_buffer_factor=-1.0)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (
            int(cfg_dropless.per_device_batch_size) * device_count,
            cfg_dropless.max_target_length,
            cfg_dropless.base_emb_dim,
        ),
        dtype=cfg_dropless.dtype,
    )
    # num_experts=8, ici_expert_parallelism=2 -> shard 0 owns experts [0, 4).
    # Route every token to experts 0 and 1: shard 0 gets everything and
    # overflows a tiny buffer; shard 1 gets nothing and never overflows.
    forced_routed_experts = jnp.broadcast_to(
        jnp.arange(cfg_dropless.num_experts_per_tok, dtype=jnp.int32),
        hidden_states.shape[:2] + (cfg_dropless.num_experts_per_tok,),
    )

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      variables = model_dropless.init(
          {"params": rng_model, "dropout": rng_model}, hidden_states, forced_routed_experts=forced_routed_experts
      )
      out_dropless, _, _ = model_dropless.apply(
          {"params": variables["params"]}, hidden_states, forced_routed_experts=forced_routed_experts
      )

    _, model_no_retry = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_no_retry, _, _ = model_no_retry.apply(
          {"params": variables["params"]}, hidden_states, forced_routed_experts=forced_routed_experts
      )
    self.assertFalse(
        jnp.allclose(out_no_retry.astype(jnp.float32), out_dropless.astype(jnp.float32), rtol=1e-2, atol=1e-2),
        msg="moe_dropless_fallback=None unexpectedly matches dropless -- forced routing isn't overflowing shard 0.",
    )

    _, model_retry = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1, moe_dropless_fallback="step")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      _, mutated = model_retry.apply(
          {"params": variables["params"]},
          hidden_states,
          forced_routed_experts=forced_routed_experts,
          mutable=["intermediates"],
      )
      has_overflow = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_has_overflow")
      self.assertTrue(has_overflow, "Expected a moe_has_overflow intermediate to be sown.")
      self.assertTrue(
          bool(jnp.any(jnp.array([jnp.any(x) for x in has_overflow]))),
          "Expected the reduced overflow flag to be True when only shard 0 overflows.",
      )

    # Replay with force_dropless=True matches dropless output.
    _, model_replay = self._build_retry_test_model(
        mesh, ragged_buffer_factor=0.1, moe_dropless_fallback="step", force_dropless=True
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_replay, _, _ = model_replay.apply(
          {"params": variables["params"]}, hidden_states, forced_routed_experts=forced_routed_experts
      )
    assert_moe_close(out_replay, out_dropless, cfg_dropless.dtype)

  @pytest.mark.tpu_only
  def test_layer_dropless_fallback_matches_dropless(self):
    """A tiny buffer + in-layer fallback matches dropless output and gradients in one pass (no replay)."""
    mesh = self._build_retry_test_mesh()
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()

    cfg_dropless, model_dropless = self._build_retry_test_model(mesh, ragged_buffer_factor=-1.0)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (
            int(cfg_dropless.per_device_batch_size) * device_count,
            cfg_dropless.max_target_length,
            cfg_dropless.base_emb_dim,
        ),
        dtype=cfg_dropless.dtype,
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      variables = model_dropless.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      out_dropless, _, grads_dropless = self._out_loss_and_grad(model_dropless, variables["params"], hidden_states)

    _, model_no_fallback = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_no_fallback, _, _ = self._out_loss_and_grad(model_no_fallback, variables["params"], hidden_states)
    self.assertFalse(
        jnp.allclose(out_no_fallback.astype(jnp.float32), out_dropless.astype(jnp.float32), rtol=1e-2, atol=1e-2),
        msg="ragged_buffer_factor=0.1 unexpectedly matches dropless -- buffer isn't forcing an overflow.",
    )

    _, model_fallback = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1, moe_dropless_fallback="layer")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      out_fallback, _, grads_fallback = self._out_loss_and_grad(model_fallback, variables["params"], hidden_states)
      _, mutated = model_fallback.apply({"params": variables["params"]}, hidden_states, mutable=["intermediates"])
    took_fallback = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_dropless_fallback")
    self.assertTrue(bool(jnp.any(jnp.array([jnp.any(x) for x in took_fallback]))), "Expected the dropless branch.")
    # The fallback absorbed the overflow, so moe_has_overflow must not report dropped tokens.
    has_overflow = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_has_overflow")
    self.assertFalse(bool(jnp.any(jnp.array([jnp.any(x) for x in has_overflow]))), "No tokens should be dropped.")

    assert_moe_close(out_fallback, out_dropless, cfg_dropless.dtype)
    for g_fallback, g_dropless in zip(
        jax.tree_util.tree_leaves(grads_fallback), jax.tree_util.tree_leaves(grads_dropless)
    ):
      assert_moe_close(g_fallback, g_dropless, cfg_dropless.dtype)

  @pytest.mark.tpu_only
  def test_layer_dropless_fallback_asymmetric_shard_overflow(self):
    """The predicate must fire (on every device) when only one EP shard overflows, and match dropless."""
    mesh = self._build_retry_test_mesh()
    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()

    cfg_dropless, model_dropless = self._build_retry_test_model(mesh, ragged_buffer_factor=-1.0)
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (
            int(cfg_dropless.per_device_batch_size) * device_count,
            cfg_dropless.max_target_length,
            cfg_dropless.base_emb_dim,
        ),
        dtype=cfg_dropless.dtype,
    )
    # num_experts=8, ici_expert_parallelism=2 -> shard 0 owns experts [0, 4); route everything there.
    forced_routed_experts = jnp.broadcast_to(
        jnp.arange(cfg_dropless.num_experts_per_tok, dtype=jnp.int32),
        hidden_states.shape[:2] + (cfg_dropless.num_experts_per_tok,),
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      variables = model_dropless.init(
          {"params": rng_model, "dropout": rng_model}, hidden_states, forced_routed_experts=forced_routed_experts
      )
      out_dropless, _, _ = model_dropless.apply(
          {"params": variables["params"]}, hidden_states, forced_routed_experts=forced_routed_experts
      )

    _, model_fallback = self._build_retry_test_model(mesh, ragged_buffer_factor=0.1, moe_dropless_fallback="layer")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_dropless.logical_axis_rules):
      (out_fallback, _, _), mutated = model_fallback.apply(
          {"params": variables["params"]},
          hidden_states,
          forced_routed_experts=forced_routed_experts,
          mutable=["intermediates"],
      )
    took_fallback = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_dropless_fallback")
    self.assertTrue(bool(jnp.any(jnp.array([jnp.any(x) for x in took_fallback]))), "Expected the dropless branch.")
    # The fallback absorbed the overflow, so moe_has_overflow must not report dropped tokens.
    has_overflow = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_has_overflow")
    self.assertFalse(bool(jnp.any(jnp.array([jnp.any(x) for x in has_overflow]))), "No tokens should be dropped.")
    assert_moe_close(out_fallback, out_dropless, cfg_dropless.dtype)

  @pytest.mark.tpu_only
  def test_moe_fsdp_two_stage_parallelism_tpu_only(self):
    # Use an imperative skip inside the test method instead of a static decorator.
    # Calling jax.device_count() in @unittest.skipIf would force JAX initialization
    # during PyTest's collection phase, locking the TPU or conflicting with CUDA.
    if jax.device_count() != 4:
      self.skipTest("FSDP two stage parallelism test requires exactly 4 devices")

    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_fsdp_parallelism=2,
        ici_fsdp_transpose_parallelism=2,
        moe_fsdp_use_two_stage_all_gather=True,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_shard_embed_moe_on_fsdp(self):
    if jax.device_count() != 4:
      self.skipTest("shard_embed_moe_on_fsdp test requires exactly 4 devices")

    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_shard_embed_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        per_device_batch_size=4,
        ici_fsdp_parallelism=4,
        shard_embed_moe_on_fsdp=True,
        max_target_length=128,
        float32_gate_logits=True,
        quantize_router_proj=False,
        quantization="fp8_full",
        use_qwix_quantization=True,
        weight_quantization_calibration_method="fixed,-224,224",
        act_quantization_calibration_method="fixed,-224,224",
        bwd_quantization_calibration_method="absmax",
    )

    def get_fp8_full_qwix_rule_for_test(config):
      return [
          qwix.QtRule(
              module_path=".*",
              weight_qtype=jnp.float8_e4m3fn,
              act_qtype=jnp.float8_e4m3fn,
              bwd_qtype=jnp.float8_e5m2,
              weight_calibration_method=config.weight_quantization_calibration_method,
              act_calibration_method=config.act_quantization_calibration_method,
              bwd_calibration_method=config.bwd_quantization_calibration_method,
              op_names=("gmm", "ragged_dot"),
          ),
      ]

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    # Instantiate QAG-quantized model with shard_embed_moe_on_fsdp
    model_qag = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.dtype,
    )
    quantization_rule = get_fp8_full_qwix_rule_for_test(cfg)
    quantization_provider = qwix.QtProvider(quantization_rule)
    model_qag = qwix.quantize_model(model_qag, quantization_provider)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      # Initialize reference unfused model to get initial weights
      _, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)

      variables_qag = model_qag.init({"params": rng_model, "dropout": rng_model}, hidden_states.astype(jnp.float32))

      output_qag, _, _ = jax.jit(model_qag.apply)(variables_qag, hidden_states.astype(jnp.float32))

      self.assertEqual(output_qag.shape, expected_output.shape)

  @pytest.mark.tpu_only
  def test_megablox_context_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_cp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        ici_context_parallelism=4,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_megablox_expert_context_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_cp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_context_parallelism=2,
        ici_expert_parallelism=2,
        packing=False,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  @pytest.mark.tpu_only
  def test_megablox_expert_tensor_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_tp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_tensor_parallelism=2,
        ici_expert_parallelism=2,
        max_target_length=128,
        float32_gate_logits=True,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      assert_moe_close(actual_output, expected_output, cfg.dtype)

  def test_random_routing(self):
    bs, seq_len, num_experts, num_experts_per_tok = 12, 1024, 8, 2
    rng = jax.random.PRNGKey(0)
    rng, logits_key = jax.random.split(rng)
    gate_logits = jax.random.normal(logits_key, (bs, seq_len, num_experts))

    rng, run_key = jax.random.split(rng)
    _, top_k_indices = moe.random_routing(run_key, gate_logits, num_experts_per_tok)

    flat_indices = top_k_indices.flatten()
    counts = jnp.bincount(flat_indices, length=num_experts)
    expected_count = bs * seq_len * num_experts_per_tok // num_experts
    tol = 0.05

    lower_bound = expected_count - expected_count * tol
    upper_bound = expected_count + expected_count * tol
    is_with_tolerance = (counts >= lower_bound) & (counts <= upper_bound)
    self.assertTrue(is_with_tolerance.all())

  def test_local_permute_no_offset(self):
    """Tests local_permute with is_offset=False across multiple shards."""
    num_experts = 8
    num_shards = 4
    experts_per_shard = num_experts // num_shards  # 2 experts per shard

    # Global group sizes for each of the 8 experts
    # Expert 0 gets 0 token, Expert 1 gets 1, ..., Expert 7 gets 7 tokens.
    global_group_sizes = jnp.arange(num_experts)
    total_assignments = jnp.sum(global_group_sizes)

    original_inputs = jnp.arange(total_assignments * 5, dtype=jnp.int32).reshape(total_assignments, 5)

    # Calculate the cumulative sum of global group sizes to determine shard input slices
    global_group_sizes_cumsum = jnp.cumsum(global_group_sizes)

    shard_start_indices = jnp.concatenate(
        [jnp.array([0]), global_group_sizes_cumsum[:-experts_per_shard:experts_per_shard]]
    )
    shard_end_indices = global_group_sizes_cumsum[experts_per_shard - 1 :: experts_per_shard]

    #               *****Expected outputs****
    # Shard 0: tokens for global experts 0, 1 (0+1=1 tokens)
    #  expected_local_group_size: [0, 1]
    #  expected_sorted_inputs: original_inputs[:0+1]
    #  expected_sorted_indices: [0]]
    #  expected_sorted_experts_ids: [1]
    # Shard 1: tokens for global experts 2, 3 (2+3=5 tokens)
    #  expected_local_group_size: [2, 3]
    #  expected_sorted_inputs: original_inputs[1:1+2+3]
    #  expected_sorted_indices: [0,1,2,3,4]
    #  expected_sorted_experts_ids: [0]*2 + [1]*3
    # Shard 2: tokens for global experts 4, 5 (4+5=9 tokens)
    #  expected_local_group_size: [4, 5]
    #  expected_sorted_inputs: original_inputs[6:6+4+5]
    #  expected_sorted_indices: [0,1,2,3,4,5,6,7,8]
    #  expected_sorted_experts_ids: [0]*4 + [1]*5
    # Shard 3: tokens for global experts 6, 7 (6+7=13 tokens)
    #  expected_local_group_size: [6, 7]
    #  expected_sorted_inputs: original_inputs[15:15+13]
    #  expected_sorted_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12]
    #  expected_sorted_experts_ids: [0]*6 + [1]*7
    for shard_index in range(num_shards):
      # Determine the input slice for the current shard
      start_idx = shard_start_indices[shard_index]
      end_idx = shard_end_indices[shard_index]
      inputs_shard = original_inputs[start_idx:end_idx]
      shard_total_tokens = end_idx - start_idx

      # Get the global group sizes relevant to this shard's experts
      global_group_sizes_for_shard = global_group_sizes[
          shard_index * experts_per_shard : (shard_index + 1) * experts_per_shard
      ]

      # Get the actual local_permute outputs.
      sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids = moe.RoutedMoE.local_permute(
          inputs_shard,
          global_group_sizes[None, :],
          experts_per_shard,
          shard_index,
          use_custom_sort_vjp=False,
          is_offset=False,
      )

      # Calculate expected outputs for the current shard
      expected_local_group_size = global_group_sizes_for_shard
      # With is_offset=False, input is assumed pre-sorted by expert, so sorted_inputs is the input itself.
      expected_sorted_inputs = inputs_shard
      # Indices are relative to inputs_shard, and since it's already sorted, they are just arange.
      expected_sorted_indices = jnp.arange(shard_total_tokens)
      # Local expert IDs: repeat local expert index (0, 1, ...) by its count
      expected_sorted_experts_ids = jnp.repeat(
          jnp.arange(experts_per_shard), expected_local_group_size, total_repeat_length=shard_total_tokens
      )

      self.assertTrue(
          jnp.array_equal(sorted_inputs, expected_sorted_inputs), f"Shard {shard_index}: sorted_inputs mismatch"
      )
      self.assertTrue(
          jnp.array_equal(sorted_indices, expected_sorted_indices), f"Shard {shard_index}: sorted_indices mismatch"
      )
      self.assertTrue(
          jnp.array_equal(local_group_size, expected_local_group_size), f"Shard {shard_index}: local_group_size mismatch"
      )
      self.assertTrue(
          jnp.array_equal(sorted_experts_ids, expected_sorted_experts_ids),
          f"Shard {shard_index}: sorted_experts_ids mismatch",
      )

  def test_local_permute_offset(self):
    experts_per_group = 2
    expert_groups = 4  # aka number of expert shards.
    num_experts = 8

    # Global group sizes for each of the 8 experts
    # Each entry i specifies the number of tokens assigned to expert i.
    simple_group_sizes = jnp.arange(8)
    manual_global_group_sizes = jnp.array([0, 0, 1, 1, 2, 0, 2, 2])
    for global_expert_counts in [simple_group_sizes, manual_global_group_sizes]:
      for shard_id in range(expert_groups):
        # Unpermuted data. shape: (sum(global_expert_counts), 5)
        x = jnp.tile(jnp.arange(1, jnp.sum(global_expert_counts) + 1).reshape(-1, 1), (1, 5))

        # The number of expert IDs assigned to each expert shard.
        local_group_sizes = jnp.sum(jnp.reshape(global_expert_counts, (expert_groups, experts_per_group)), axis=-1)

        # Expert assignments corresponding to each entry of x.
        # NOTE: It is assumed that x is sorted in order of expert ID (because it is previously
        # passed through permute()), so expert_assignments just repeats the expert ID using counts from
        # global_expert_counts.
        expert_assignments = jnp.repeat(jnp.arange(0, num_experts), repeats=global_expert_counts)

        # Offset for the start of each shard (aka expert group). Offset for shard i is the sum
        # of the number of tokens assigned to all shards (local_group_size) before i.
        input_offsets = jnp.concatenate((jnp.array([0]), jnp.cumsum(local_group_sizes)[:-1]))

        # Actual results of local_permute().
        permuted_x, local_sorted_indices, local_expert_counts, local_expert_assignments = moe.RoutedMoE.local_permute(
            x,
            global_expert_counts[None, :],
            experts_per_group,
            shard_index=shard_id,
            use_custom_sort_vjp=False,
            is_offset=True,
            global_sorted_experts=expert_assignments,
        )

        # permuted_x should be equivalent to slicing x at the input offset for that shard.
        assert jnp.all(
            permuted_x[: local_group_sizes[shard_id]]
            == x[input_offsets[shard_id] : input_offsets[shard_id] + local_group_sizes[shard_id]]
        ), f"Local permuted rows do not match their unpermuted original rows for shard_id={shard_id}"

        # local_sorted_indices should match the indices of the slice from x corresponding to this shard.
        # That can be computed by taking all of the indices between the input_offset for the current shard
        # until the last index belonging to the current shard (i.e. input_offset[shard_id] + local_group_sizes[shard_id]).
        assert jnp.all(
            local_sorted_indices[: local_group_sizes[shard_id]]
            == jnp.arange(input_offsets[shard_id], input_offsets[shard_id] + local_group_sizes[shard_id])
        ), (
            "Local permuted row indices do not match their respective unpermuted indices in the "
            f"original inputs for shard_id={shard_id}!"
        )

        # local_expert_counts should correspond to slicing experts_per_group values from global_expert_counts
        # for the shard_id.
        assert jnp.all(
            local_expert_counts == global_expert_counts[shard_id * experts_per_group : (shard_id + 1) * experts_per_group]
        ), "Local permuted group sizes do not match the respective unpermuted expert bincounts for shard_id={shard_id}."

        # local_expert_assignments should correspond to taking a slice out of expert_assignments.
        # The slice size is shard i's size (local_group_sizes[i]]) and the slice should start
        # at input_offsets[i].
        assert jnp.all(
            local_expert_assignments[: local_group_sizes[shard_id]]
            == jnp.mod(
                expert_assignments[input_offsets[shard_id] : input_offsets[shard_id] + local_group_sizes[shard_id]],
                experts_per_group,
            )
        ), (
            "Local permuted expert assignments to not match the expected unpermuted expert assignments "
            f"for shard_id={shard_id}."
        )

  def test_get_all_to_all_params_sharded_batch(self):
    num_expert_parallelism_sharded = 4

    # all_group_sizes[i, j] = num inputs batch_shard i sends to expert_shard j
    all_group_sizes_sharded = jnp.array([[1, 2, 0, 3], [4, 0, 1, 2], [0, 3, 2, 1], [2, 1, 4, 0]], dtype=jnp.int32)

    # The offset for the current batch shard (row) to send inputs to a particular expert
    # shard (column), will be the cumulative number of tokens sent to all previous experts.
    # Example: batch shard 1
    # all_group_sizes_sharded[1] = [4, 0, 1, 2]:
    # input_offsets = [0, 4, 4+0, 4+0+1] = [0, 4, 4, 5]
    expected_input_offsets_sharded = jnp.array([[0, 1, 3, 3], [0, 4, 4, 5], [0, 0, 3, 5], [0, 2, 3, 7]], dtype=jnp.int32)

    # The number of tokens that the current batch shard (row) sends to each expert shard (columns)
    # is recorded in all_group_sizes_sharded[row].
    expected_send_sizes_sharded = jnp.array([[1, 2, 0, 3], [4, 0, 1, 2], [0, 3, 2, 1], [2, 1, 4, 0]], dtype=jnp.int32)

    # The offset at which each expert shard (column) will receive the current batch shard's (row)
    # input is the cumulative number of tokens received by all previous batch shards (rows)
    # for that expert.
    # (batch shard 0) output_offsets = [0, 0, 0, 0]
    # (batch shard 1) output_offsets = [0+1, 0+2, 0+0, 0+3])
    # (batch shard 2) output_offsets = [0+1+4, 0+2+0, 0+0+1, 0+3+2])
    # ...
    expected_output_offsets_sharded = jnp.array([[0, 0, 0, 0], [1, 2, 0, 3], [5, 2, 1, 5], [5, 5, 3, 6]], dtype=jnp.int32)

    # The number of inputs a particular expert shard (col) receives from all of the batch_shards (rows).
    # Example: expert shard 1
    # Receives 2 from batch_shard 0, 0 from batch_shard 1, 3 from batch_shard 2, 1 from batch_shard 3
    # Which is the same as all_group_sizes_sharded[:, 1].
    expected_recv_sizes_sharded = jnp.array([[1, 4, 0, 2], [2, 0, 3, 1], [0, 1, 2, 4], [3, 2, 1, 0]], dtype=jnp.int32)

    for expert_shard_id in range(num_expert_parallelism_sharded):
      exp_in_off = expected_input_offsets_sharded[expert_shard_id]
      exp_send_sz = expected_send_sizes_sharded[expert_shard_id]
      exp_out_off = expected_output_offsets_sharded[expert_shard_id]
      exp_recv_sz = expected_recv_sizes_sharded[expert_shard_id]

      in_off, send_sz, out_off, recv_sz = moe.RoutedMoE.get_all_to_all_params(
          all_group_sizes_sharded, expert_shard_id, num_expert_parallelism_sharded, is_batch_sharded=True
      )
      self.assertTrue(
          jnp.array_equal(in_off, exp_in_off), f"Sharded Batch: Input offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(send_sz, exp_send_sz), f"Sharded Batch: Send sizes mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(out_off, exp_out_off), f"Sharded Batch: Output offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(recv_sz, exp_recv_sz), f"Sharded Batch: Receive sizes mismatch for shard {expert_shard_id}"
      )

  def test_get_all_to_all_params_unsharded_batch(self):
    """Tests get_all_to_all_params with a simple hard-coded example using 4 expert shards."""
    num_expert_parallelism_unsharded = 4

    # group_sizes_unsharded[i] = num inputs that each expert_shard i is responsible for.
    group_sizes_unsharded = jnp.array([6, 7, 6, 7], dtype=jnp.int32)

    # Each expert shard will send their data starting at index 0.
    expected_input_offsets_unsharded_template = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    # Each expert shard will send the amount of data they are responsible for
    # (indicated by group_sizes_unsharded).
    expected_send_sizes_unsharded_per_shard = jnp.array(
        [[6, 6, 6, 6], [7, 7, 7, 7], [6, 6, 6, 6], [7, 7, 7, 7]], dtype=jnp.int32
    )

    # When the batches are fully replicated (unsharded) then each batch will receive expert i's
    # data at the cumulative sum of the amount of input received from all previous experts.
    # (batch shard 0) output_offsets = [0, 0, 0, 0]
    # (batch shard 1) output_offsets = [0+6, 0+6, 0+6, 0+6])
    # (batch shard 2) output_offsets = [0+6+7, 0+6+7, 0+6+7, 0+6+7])
    # Which is just the cumulative sum of 0 and group_sizes_unsharded.
    expected_output_offsets_unsharded_per_shard = jnp.array(
        [[0, 0, 0, 0], [6, 6, 6, 6], [13, 13, 13, 13], [19, 19, 19, 19]], dtype=jnp.int32
    )

    # Each (replicated) batch shard will the amount of data from each expert specified by
    # group_sizes_unsharded.
    expected_recv_sizes_unsharded_template = jnp.array([6, 7, 6, 7], dtype=jnp.int32)

    for expert_shard_id in range(num_expert_parallelism_unsharded):
      exp_in_off = expected_input_offsets_unsharded_template
      exp_send_sz = expected_send_sizes_unsharded_per_shard[expert_shard_id]
      exp_out_off = expected_output_offsets_unsharded_per_shard[expert_shard_id]
      exp_recv_sz = expected_recv_sizes_unsharded_template

      in_off, send_sz, out_off, recv_sz = moe.RoutedMoE.get_all_to_all_params(
          group_sizes_unsharded, expert_shard_id, num_expert_parallelism_unsharded, is_batch_sharded=False
      )
      self.assertTrue(
          jnp.array_equal(in_off, exp_in_off), f"Unsharded Batch: Input offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(send_sz, exp_send_sz), f"Unsharded Batch: Send sizes mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(out_off, exp_out_off), f"Unsharded Batch: Output offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(recv_sz, exp_recv_sz), f"Unsharded Batch: Receive sizes mismatch for shard {expert_shard_id}"
      )

  def test_ragged_buffer_balanced(self):
    ragged_buffer_factor = 1.0
    local_batch = 32768
    ep_degree = 4  # unused for ragged_factor>0
    num_experts_per_tok = 8  # unused for ragged_factor>0
    global_experts = 256  # unused for ragged_factor>0

    expected_ragged_buffer = 32768
    actual_ragged_buffer = moe.RoutedMoE.get_ragged_buffer_size(
        local_batch, ep_degree, global_experts, num_experts_per_tok, ragged_buffer_factor
    )
    self.assertEqual(expected_ragged_buffer, actual_ragged_buffer)

  def test_ragged_buffer_larger(self):
    ragged_buffer_factor = 2.0
    local_batch = 32768
    ep_degree = 4  # unused for ragged_factor>0
    num_experts_per_tok = 8  # unused for ragged_factor>0
    global_experts = 256  # unused for ragged_factor>0

    expected_ragged_buffer = 65536
    actual_ragged_buffer = moe.RoutedMoE.get_ragged_buffer_size(
        local_batch, ep_degree, global_experts, num_experts_per_tok, ragged_buffer_factor
    )
    self.assertEqual(expected_ragged_buffer, actual_ragged_buffer)

  def test_small_ep_worst_case(self):
    ragged_buffer_factor = -1.0  # Not using ragged_buffer_factor
    local_batch = 32768
    num_experts_per_tok = 8
    global_experts = 256
    ep_degree = 4

    expected_ragged_buffer = 131072  # local_batch * ep_degree
    actual_ragged_buffer = moe.RoutedMoE.get_ragged_buffer_size(
        local_batch, ep_degree, global_experts, num_experts_per_tok, ragged_buffer_factor
    )
    self.assertEqual(expected_ragged_buffer, actual_ragged_buffer)

  def test_large_ep_worst_case(self):
    ragged_buffer_factor = -1.0  # Not using ragged_buffer_factor
    local_batch = 32768
    num_experts_per_tok = 8
    global_experts = 256
    ep_degree = 128

    expected_ragged_buffer = 1048576  # (32768) * (global_exp / top_k)
    actual_ragged_buffer = moe.RoutedMoE.get_ragged_buffer_size(
        local_batch, ep_degree, global_experts, num_experts_per_tok, ragged_buffer_factor
    )
    self.assertEqual(expected_ragged_buffer, actual_ragged_buffer)


class QuantizedMoeTest(parameterized.TestCase):
  """Tests for quantized Mixture of Experts (MoE) execution and gradients."""

  @staticmethod
  def _build_and_quantize_moe_model(cfg: Config, mesh: Mesh):
    """Instantiates and optionally applies Qwix FP8 quantization rules to RoutedMoE."""
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    if cfg.quantization:
      if not (cfg.quantization == "fp8_full" and cfg.use_qwix_quantization):
        raise ValueError("Only fp8_full with qwix quantization is supported for MoE testing.")

      quantization_rule = [
          qwix.QtRule(
              module_path=".*",
              weight_qtype=jnp.float8_e4m3fn,
              act_qtype=jnp.float8_e4m3fn,
              bwd_qtype=jnp.float8_e5m2,
              weight_calibration_method=cfg.weight_quantization_calibration_method,
              act_calibration_method=cfg.act_quantization_calibration_method,
              bwd_calibration_method=cfg.bwd_quantization_calibration_method,
              op_names=("gmm", "ragged_dot"),
          ),
      ]
      model = qwix.quantize_model(model, qwix.QtProvider(quantization_rule))

    return model

  def _run_moe_loss_and_grad(
      self,
      cfg: Config,
      rng_model: jax.Array,
      hidden_states: jax.Array,
  ) -> dict[str, jax.Array]:
    """Executes forward pass, loss calculation, and backward gradients for RoutedMoE."""
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = self._build_and_quantize_moe_model(cfg, mesh)

    def loss_fn(params, x):
      out, lb_loss, _ = model.apply({"params": params}, x)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init({"params": rng_model, "dropout": rng_model}, hidden_states)
      (_, output), (grads, x_grad) = loss_and_grad(variables["params"], hidden_states)

    return {
        "output": output,
        "state_grad": x_grad,
        "var_grad": grads,
    }

  @parameterized.named_parameters(
      {
          "testcase_name": f"{base_name}_ep{ici_expert_parallelism}" + ("_qag" if shard_embed_moe_on_fsdp else ""),
          "quantization": quantization,
          "use_tokamax_gmm": use_tokamax_gmm,
          "use_gmm_v2": use_gmm_v2,
          "wa_static": wa_static,
          "ici_expert_parallelism": ici_expert_parallelism,
          "shard_embed_moe_on_fsdp": shard_embed_moe_on_fsdp,
      }
      for (
          base_name,
          quantization,
          use_tokamax_gmm,
          use_gmm_v2,
          wa_static,
          ici_expert_parallelism,
          shard_embed_moe_on_fsdp,
      ) in [
          ("megablox_bf16", "", False, False, False, 1, False),
          ("megablox_fp8_dynamic", "fp8_full", False, False, False, 1, False),
          ("megablox_fp8_static", "fp8_full", False, False, True, 1, False),
          ("tokamax_v1_bf16", "", True, False, False, 1, False),
          ("tokamax_v1_fp8_dynamic", "fp8_full", True, False, False, 1, False),
          ("tokamax_v1_fp8_static", "fp8_full", True, False, True, 1, False),
          ("tokamax_v2_bf16", "", True, True, False, 1, False),
          ("tokamax_v2_fp8_dynamic", "fp8_full", True, True, False, 1, False),
          ("tokamax_v2_fp8_static", "fp8_full", True, True, True, 1, False),
          ("tokamax_v2_bf16", "", True, True, False, 4, False),
          ("tokamax_v2_fp8_dynamic", "fp8_full", True, True, False, 4, False),
          ("tokamax_v2_fp8_static", "fp8_full", True, True, True, 4, False),
          ("tokamax_v2_fp8_static", "fp8_full", True, True, True, 1, True),
      ]
  )
  @pytest.mark.skip_on_tpu7x  # TODO(b/543017989): Investigate correctness failures
  @pytest.mark.tpu_only
  def test_gmm_grad_equivalence(
      self,
      quantization: str,
      use_tokamax_gmm: bool,
      use_gmm_v2: bool,
      wa_static: bool,
      ici_expert_parallelism: int,
      shard_embed_moe_on_fsdp: bool = False,
      **kwargs,
  ):
    calibration_method = "fixed,-224,224" if wa_static else "absmax"
    rng_model, rng_hidden_states = jax.random.split(jax.random.PRNGKey(2345))

    def _build_cfg(
        sparse_matmul,
        quantization,
        ici_expert_parallelism,
        megablox=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        shard_embed_moe_on_fsdp=False,
    ):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="gmm_grad_equivalence_test",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          weight_dtype="float32",
          dtype="bfloat16",
          per_device_batch_size=2,
          max_target_length=256,
          float32_gate_logits=True,
          quantize_router_proj=False,
          ici_expert_parallelism=ici_expert_parallelism,
          sparse_matmul=sparse_matmul,
          megablox=megablox,
          use_tokamax_gmm=use_tokamax_gmm,
          use_gmm_v2=use_gmm_v2,
          shard_embed_moe_on_fsdp=shard_embed_moe_on_fsdp,
          quantization=quantization,
          use_qwix_quantization=True,
          weight_quantization_calibration_method=calibration_method,
          act_quantization_calibration_method=calibration_method,
          bwd_quantization_calibration_method="absmax",
          wi_tile_fwd_batch_seq=128,
          wi_tile_dlhs_batch_seq=128,
          wi_tile_dlhs_embed_dim=256,
          wi_tile_drhs_batch_seq=128,
          wo_tile_fwd_batch_seq=128,
          wo_tile_fwd_embed_dim=256,
          wo_tile_dlhs_batch_seq=128,
          wo_tile_dlhs_mlp_dim=256,
          wo_tile_drhs_batch_seq=128,
      )

    # Reference run: dense matmul, no quantization, EP=1, no weight all-gather sharding.
    # shard_embed_moe_on_fsdp stays False here: it requires static weight quantization,
    # which the unquantized reference does not have.
    cfg_ref = _build_cfg(
        sparse_matmul=False,
        quantization="",
        ici_expert_parallelism=1,
        megablox=False,
        use_tokamax_gmm=False,
        use_gmm_v2=False,
    )
    # Use normal distribution to generate realistic variances and negative values
    # to guarantee the quantization scale != 1.0, which catches scale-dropping bugs.
    hidden_states = jax.random.normal(
        rng_hidden_states,
        (
            int(cfg_ref.per_device_batch_size) * jax.device_count(),
            cfg_ref.max_target_length,
            cfg_ref.base_emb_dim,
        ),
        dtype=cfg_ref.dtype,
    )
    tree_ref = self._run_moe_loss_and_grad(cfg_ref, rng_model, hidden_states)

    # Target run: custom configuration (sparse, GMM, EP, quantization)
    cfg_tgt = _build_cfg(
        sparse_matmul=True,
        quantization=quantization,
        ici_expert_parallelism=ici_expert_parallelism,
        megablox=True,
        use_tokamax_gmm=use_tokamax_gmm,
        use_gmm_v2=use_gmm_v2,
        shard_embed_moe_on_fsdp=shard_embed_moe_on_fsdp,
    )
    tree_tgt = self._run_moe_loss_and_grad(cfg_tgt, rng_model, hidden_states)

    compare_tree(tree_ref, tree_tgt, 0.22 if quantization else 0.012)

  @parameterized.named_parameters(
      {
          "testcase_name": (f"{'ragged_sort' if use_ragged_sort else 'default_sort'}"),
          "use_ragged_sort": use_ragged_sort,
      }
      for use_ragged_sort in [False, True]
  )
  @pytest.mark.tpu_only
  def test_moe_quantize_token_all_gather(
      self,
      use_ragged_sort: bool,
  ):
    """Tests numerical equivalence of MoeBlock with moe_quantize_token_all_gather."""
    ici_expert_parallelism = 4
    calibration_method = "fixed,-224,224"
    rng_model, rng_hidden_states = jax.random.split(jax.random.PRNGKey(42))

    def _build_cfg(moe_quantize_token_all_gather: bool):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="moe_quantize_token_all_gather_test",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          weight_dtype="float32",
          dtype="bfloat16",
          per_device_batch_size=1,
          max_target_length=128,
          float32_gate_logits=True,
          quantize_router_proj=False,
          ici_expert_parallelism=ici_expert_parallelism,
          sparse_matmul=True,
          megablox=False,
          use_tokamax_gmm=True,
          use_gmm_v2=True,
          use_ring_of_experts=True,
          use_ragged_sort=use_ragged_sort,
          mlp_bias=True,
          moe_quantize_token_all_gather=moe_quantize_token_all_gather,
          quantization="fp8_full",
          use_qwix_quantization=True,
          weight_quantization_calibration_method=calibration_method,
          act_quantization_calibration_method=calibration_method,
          bwd_quantization_calibration_method="absmax",
          wi_tile_fwd_batch_seq=128,
          wi_tile_dlhs_batch_seq=128,
          wi_tile_dlhs_embed_dim=256,
          wi_tile_drhs_batch_seq=128,
          wo_tile_fwd_batch_seq=128,
          wo_tile_fwd_embed_dim=256,
          wo_tile_dlhs_batch_seq=128,
          wo_tile_dlhs_mlp_dim=256,
          wo_tile_drhs_batch_seq=128,
      )

    cfg_ref = _build_cfg(moe_quantize_token_all_gather=False)
    hidden_states = jax.random.normal(
        rng_hidden_states,
        (
            int(cfg_ref.per_device_batch_size) * jax.device_count(),
            cfg_ref.max_target_length,
            cfg_ref.base_emb_dim,
        ),
        dtype=cfg_ref.dtype,
    )
    tree_ref = self._run_moe_loss_and_grad(cfg_ref, rng_model, hidden_states)

    cfg_tgt = _build_cfg(moe_quantize_token_all_gather=True)
    tree_tgt = self._run_moe_loss_and_grad(cfg_tgt, rng_model, hidden_states)

    compare_tree(tree_ref, tree_tgt, relative_norm_diff_threshold=0.22)

  @parameterized.named_parameters(
      {"testcase_name": "rowwise", "bwd_method": "rowwise"},
      {"testcase_name": "fixed", "bwd_method": "fixed,0.01"},
  )
  @pytest.mark.tpu_only
  def test_moe_quantize_combine_bwd_method(
      self,
      bwd_method: str,
  ):
    """Tests numerical equivalence of MoeBlock with moe_quantize_combine_bwd_method."""
    ici_expert_parallelism = 4
    calibration_method = "fixed,-224,224"
    rng_model, rng_hidden_states = jax.random.split(jax.random.PRNGKey(42))

    def _build_cfg(bwd_method_val: str):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="moe_quantize_combine_bwd_test",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          weight_dtype="float32",
          dtype="bfloat16",
          per_device_batch_size=1,
          max_target_length=128,
          float32_gate_logits=True,
          quantize_router_proj=False,
          ici_expert_parallelism=ici_expert_parallelism,
          sparse_matmul=True,
          megablox=False,
          use_tokamax_gmm=True,
          use_gmm_v2=True,
          use_ring_of_experts=True,
          use_ragged_sort=True,
          mlp_bias=True,
          moe_quantize_combine_bwd_method=bwd_method_val,
          quantization="fp8_full",
          use_qwix_quantization=True,
          weight_quantization_calibration_method=calibration_method,
          act_quantization_calibration_method=calibration_method,
          bwd_quantization_calibration_method="absmax",
          wi_tile_fwd_batch_seq=128,
          wi_tile_dlhs_batch_seq=128,
          wi_tile_dlhs_embed_dim=256,
          wi_tile_drhs_batch_seq=128,
          wo_tile_fwd_batch_seq=128,
          wo_tile_fwd_embed_dim=256,
          wo_tile_dlhs_batch_seq=128,
          wo_tile_dlhs_mlp_dim=256,
          wo_tile_drhs_batch_seq=128,
      )

    cfg_ref = _build_cfg(bwd_method_val="")
    hidden_states = jax.random.normal(
        rng_hidden_states,
        (
            int(cfg_ref.per_device_batch_size) * jax.device_count(),
            cfg_ref.max_target_length,
            cfg_ref.base_emb_dim,
        ),
        dtype=cfg_ref.dtype,
    )
    tree_ref = self._run_moe_loss_and_grad(cfg_ref, rng_model, hidden_states)

    cfg_tgt = _build_cfg(bwd_method)
    tree_tgt = self._run_moe_loss_and_grad(cfg_tgt, rng_model, hidden_states)

    compare_tree(tree_ref, tree_tgt, relative_norm_diff_threshold=0.25)


class TcRaggedSortConfigTest(parameterized.TestCase):
  """moe_tc_ragged_sort is rejected unless it can run (truncated-buffer ring-of-experts ragged sort)."""

  _CONFIG = {
      "run_name": "tc_ragged_sort_config_test",
      "num_experts": 8,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 64,
      "override_logical_axis_rules": True,
      "ici_expert_parallelism": 2,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "ragged_buffer_factor": 1.5,
  }

  def test_accepts_ring_of_experts_ragged_sort(self):
    self.assertTrue(maxtext_types.MaxTextConfig(**self._CONFIG, moe_tc_ragged_sort=True).moe_tc_ragged_sort)

  @parameterized.named_parameters(
      ("no_ring_of_experts", {"use_ring_of_experts": False}, "requires use_ring_of_experts=True"),
      # ragged_buffer_factor > 0 already requires use_ragged_sort elsewhere.
      ("no_ragged_sort", {"use_ragged_sort": False, "ragged_buffer_factor": -1.0}, "requires use_ragged_sort=True"),
      ("dropless_buffer", {"ragged_buffer_factor": -1.0}, "requires ragged_buffer_factor > 0.0"),
  )
  def test_rejects_unsupported(self, overrides, msg):
    with self.assertRaisesRegex(ValueError, re.escape("moe_tc_ragged_sort=True " + msg)):
      maxtext_types.MaxTextConfig(**{**self._CONFIG, **overrides}, moe_tc_ragged_sort=True)

  _CONFIG_3D = {**_CONFIG, "moe_tc_ragged_sort": True, "use_tokamax_gmm": True, "use_gmm_v2": True}

  def test_accepts_3d_gmm(self):
    self.assertTrue(maxtext_types.MaxTextConfig(**self._CONFIG_3D, moe_tc_ragged_3d_gmm=True).moe_tc_ragged_3d_gmm)

  def test_accepts_3d_gmm_with_gmm_options(self):
    cfg = maxtext_types.MaxTextConfig(
        **self._CONFIG_3D,
        moe_tc_ragged_3d_gmm=True,
        moe_accumulate_wi_dlhs=True,
        moe_accumulate_chunk_wgrad=True,
        moe_gmm_v2_dlhs_transpose_rhs=True,
    )
    self.assertTrue(cfg.moe_tc_ragged_3d_gmm)

  @parameterized.named_parameters(
      ("no_gmm_v2", {"use_gmm_v2": False}, "requires use_tokamax_gmm=True and use_gmm_v2=True"),
      ("mlp_bias", {"mlp_bias": True}, "does not support: mlp_bias"),
      ("emb_chunks", {"num_moe_emb_chunks": 2}, "does not support: num_moe_emb_chunks > 0"),
  )
  def test_3d_gmm_rejects_unsupported(self, overrides, msg):
    with self.assertRaisesRegex(ValueError, re.escape("moe_tc_ragged_3d_gmm=True " + msg)):
      maxtext_types.MaxTextConfig(**{**self._CONFIG_3D, **overrides}, moe_tc_ragged_3d_gmm=True)

  _BWD_PREQUANT_FP8 = {
      "moe_tc_ragged_3d_gmm": True,
      "moe_tc_ragged_3d_dispatch": True,
      "moe_tc_ragged_weights_on_activation": True,
      "moe_bwd_prequant_before_unsort": True,
      "quantization": "fp8_full",
      "use_qwix_quantization": True,
      "quantize_router_proj": False,
      "weight_quantization_calibration_method": "fixed,-224,224",
      "act_quantization_calibration_method": "fixed,-224,224",
      "bwd_quantization_calibration_method": "fixed,-224,224",
  }

  @parameterized.parameters(False, True)
  def test_bwd_prequant_accepts_fixed_calibration(self, combine):
    cfg = maxtext_types.MaxTextConfig(**self._CONFIG_3D, **self._BWD_PREQUANT_FP8, moe_combine_bwd_direct_qarray=combine)
    self.assertTrue(cfg.moe_bwd_prequant_before_unsort)

  @parameterized.product(name=["weight", "act", "bwd"], combine=[False, True])
  def test_bwd_prequant_rejects_dynamic_calibration(self, name, combine):
    overrides = {
        **self._BWD_PREQUANT_FP8,
        "moe_combine_bwd_direct_qarray": combine,
        f"{name}_quantization_calibration_method": "absmax",
    }
    with self.assertRaisesRegex(ValueError, re.escape(f"got {name}_quantization_calibration_method='absmax'")):
      maxtext_types.MaxTextConfig(**self._CONFIG_3D, **overrides)

  def test_bwd_prequant_without_quantization_skips_calibration_check(self):
    overrides = {**self._BWD_PREQUANT_FP8, "quantization": "", "act_quantization_calibration_method": "absmax"}
    self.assertTrue(maxtext_types.MaxTextConfig(**self._CONFIG_3D, **overrides).moe_bwd_prequant_before_unsort)

  def test_combine_bwd_direct_qarray_rejects_combine_bwd_method(self):
    with self.assertRaisesRegex(ValueError, re.escape("moe_quantize_combine_bwd_method='rowwise' has no effect")):
      maxtext_types.MaxTextConfig(
          **self._CONFIG_3D,
          moe_tc_ragged_3d_gmm=True,
          moe_tc_ragged_3d_dispatch=True,
          moe_tc_ragged_weights_on_activation=True,
          moe_bwd_prequant_before_unsort=True,
          moe_combine_bwd_direct_qarray=True,
          moe_quantize_combine_bwd_method="rowwise",
      )

  def test_accepts_3d_dispatch(self):
    cfg = maxtext_types.MaxTextConfig(**self._CONFIG_3D, moe_tc_ragged_3d_gmm=True, moe_tc_ragged_3d_dispatch=True)
    self.assertTrue(cfg.moe_tc_ragged_3d_dispatch)

  _DISPATCH_REQUIRES = "moe_tc_ragged_3d_dispatch=True requires moe_tc_ragged_sort=True and moe_tc_ragged_3d_gmm=True"

  @parameterized.named_parameters(
      ("no_3d_gmm", {}, _DISPATCH_REQUIRES),
      ("no_tc_sort", {"moe_tc_ragged_sort": False, "moe_tc_ragged_3d_gmm": True}, _DISPATCH_REQUIRES),
      # The 3D gmm requirements (here mlp_bias) apply to the 3D dispatch too.
      (
          "mlp_bias",
          {"moe_tc_ragged_3d_gmm": True, "mlp_bias": True},
          "moe_tc_ragged_3d_gmm=True does not support: mlp_bias",
      ),
  )
  def test_3d_dispatch_rejects_unsupported(self, overrides, msg):
    with self.assertRaisesRegex(ValueError, re.escape(msg)):
      maxtext_types.MaxTextConfig(**{**self._CONFIG_3D, **overrides}, moe_tc_ragged_3d_dispatch=True)

  @parameterized.named_parameters(("device", "device"), ("offload", "offload"))
  def test_accepts_routing_weights(self, location):
    cfg = maxtext_types.MaxTextConfig(
        **self._CONFIG_3D, moe_tc_ragged_weights_on_activation=True, moe_tc_routing_checkpoint=location
    )
    self.assertEqual(cfg.moe_tc_routing_checkpoint, location)

  def test_window_routing_requires_tc_ragged_sort(self):
    msg = "moe_tc_window_routing=True requires moe_tc_ragged_sort=True."
    with self.assertRaisesRegex(ValueError, re.escape(msg)):
      maxtext_types.MaxTextConfig(**{**self._CONFIG_3D, "moe_tc_ragged_sort": False}, moe_tc_window_routing=True)

  @parameterized.named_parameters(
      ("no_weights_on_activation", {}),
      ("no_tc_sort", {"moe_tc_ragged_sort": False, "moe_tc_ragged_weights_on_activation": True}),
  )
  def test_routing_weights_rejects_unsupported(self, overrides):
    msg = "moe_tc_routing_checkpoint=device requires moe_tc_ragged_sort=True and moe_tc_ragged_weights_on_activation=True"
    with self.assertRaisesRegex(ValueError, re.escape(msg)):
      maxtext_types.MaxTextConfig(**{**self._CONFIG_3D, **overrides}, moe_tc_routing_checkpoint="device")


class GetRaggedBufferFactorTest(parameterized.TestCase):
  """Tests that RoutedMoE.get_ragged_buffer_factor picks eval_ragged_buffer_factor only under eval axis rules."""

  TRAIN_RULES = (("activation_batch", ("data", "fsdp")),)
  EVAL_RULES = (("activation_batch", ("data",)),)

  def _factor(self, eval_ragged_buffer_factor, eval_rules, active_rules):
    config = SimpleNamespace(
        ragged_buffer_factor=1.5,
        eval_ragged_buffer_factor=eval_ragged_buffer_factor,
        logical_axis_rules=self.TRAIN_RULES,
        logical_axis_rules_for_eval=eval_rules,
    )
    with nn_partitioning.axis_rules(active_rules):
      return moe.RoutedMoE.get_ragged_buffer_factor(SimpleNamespace(config=config))

  @parameterized.named_parameters(
      ("eval_rules_with_override", 3.0, EVAL_RULES, EVAL_RULES, 3.0),
      ("eval_rules_worst_case", -1.0, EVAL_RULES, EVAL_RULES, -1.0),
  )
  def test_get_ragged_buffer_factor(self, eval_factor, eval_rules, active_rules, expected):
    self.assertEqual(self._factor(eval_factor, eval_rules, active_rules), expected)

  @parameterized.named_parameters(
      ("train_rules", TRAIN_RULES, 4.0),
      ("eval_rules", EVAL_RULES, 4.0),
  )
  def test_graphdef_override_wins(self, active_rules, expected):
    # A per-graphdef ragged_buffer_factor_override > 0 (first-phase or eval graphdef) wins over both
    # ragged_buffer_factor and eval_ragged_buffer_factor; an override <= 0 is ignored.
    config = SimpleNamespace(
        ragged_buffer_factor=1.5,
        eval_ragged_buffer_factor=-1.0,
        logical_axis_rules=self.TRAIN_RULES,
        logical_axis_rules_for_eval=self.EVAL_RULES,
    )
    with nn_partitioning.axis_rules(active_rules):
      self.assertEqual(
          moe.RoutedMoE.get_ragged_buffer_factor(SimpleNamespace(config=config, ragged_buffer_factor_override=4.0)),
          expected,
      )
      unset = moe.RoutedMoE.get_ragged_buffer_factor(SimpleNamespace(config=config, ragged_buffer_factor_override=0.0))
    self.assertEqual(unset, 1.5 if active_rules == self.TRAIN_RULES else -1.0)


class DroplessFallbackNumChunksTest(parameterized.TestCase):
  """Tests RoutedMoE.get_dropless_fallback_num_chunks sizing of the in-layer dropless branch."""

  def _chunks(self, ep, factor, seq_len, n_chunks=1, num_experts=256, top_k=8):
    config = SimpleNamespace(num_experts=num_experts)
    module = SimpleNamespace(
        config=config,
        num_experts_per_tok=top_k,
        get_expert_parallelism_size=lambda: ep,
        get_ragged_buffer_factor=lambda: factor,
    )
    return moe.RoutedMoE.get_dropless_fallback_num_chunks(module, seq_len, n_chunks)

  @parameterized.named_parameters(
      # Dropless buffer = EP * balanced -> EP / factor chunks.
      ("dsv3_ep16_factor4", 16, 4.0, 4096, 1, 4),
      ("dsv3_ep32_factor4", 32, 4.0, 4096, 1, 8),
      # EP > E/top_k: the dropless buffer is still EP (not min(EP, E/top_k) = 32) times balanced.
      ("dsv3_ep64_factor4", 64, 4.0, 4096, 1, 16),
      # Fast path already chunked 2x -> fallback needs 2x more chunks.
      ("fast_path_chunked", 32, 4.0, 4096, 2, 16),
      # Target 32/3 -> 11 does not divide 4096; the next divisor is 16.
      ("rounds_up_to_divisor", 32, 3.0, 4096, 1, 16),
      # Buffer already >= worst case: overflow is impossible, no fallback branch.
      ("buffer_covers_worst_case", 4, 4.0, 4096, 1, None),
  )
  def test_num_chunks(self, ep, factor, seq_len, n_chunks, expected):
    self.assertEqual(self._chunks(ep, factor, seq_len, n_chunks), expected)


class GetEinsumTest(parameterized.TestCase):
  """Tests for the quantized einsums RoutedMoE.get_einsum hands to dense_matmul."""

  def _make_moe(self, quant):
    """Builds a small RoutedMoE on the dense_matmul path with the given quantization."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="get_einsum_test",
        enable_checkpointing=False,
        decoder_block="mixtral",
        num_experts=4,
        num_experts_per_tok=2,
        base_emb_dim=64,
        base_mlp_dim=32,
        base_moe_mlp_dim=32,
        dtype="float32",
        weight_dtype="float32",
        megablox=False,
        sparse_matmul=False,
        max_target_length=8,
        per_device_batch_size=1,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    return moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=jnp.float32,
        quant=quant,
        rngs=nnx.Rngs(0),
    )

  def _quantization(self, quantization):
    """Returns the quantization object the config string maps to."""
    return configure_quantization(
        pyconfig.initialize(
            [None, get_test_config_path()],
            enable_checkpointing=False,
            quantization=quantization,
        )
    )

  def test_fp8_einsum_is_bound(self):
    model = self._make_moe(Fp8Quantization())
    einsum_fn = model.get_einsum(einsum_name=moe.WI_0)
    result = einsum_fn("ab,bc->ac", jnp.ones((2, 3)), jnp.ones((3, 4)))
    self.assertEqual(result.shape, (2, 4))

  def test_aqt_einsum_is_bound(self):
    model = self._make_moe(self._quantization("int8"))
    self.assertIsNone(model.quant_einsums)
    einsum_fn = model.get_einsum(einsum_name=moe.WI_0)
    result = einsum_fn("ab,bc->ac", jnp.ones((2, 3)), jnp.ones((3, 4)))
    self.assertEqual(result.shape, (2, 4))

  def test_unregistered_quant_einsum_name_raises(self):
    model = self._make_moe(Fp8Quantization())
    einsum_fn = model.get_einsum(einsum_name="not_registered")
    with self.assertRaises(ValueError) as ctx:
      einsum_fn("ab,bc->ac", jnp.ones((2, 3)), jnp.ones((3, 4)))
    self.assertIn("not_registered", str(ctx.exception))
    self.assertIn("Available names", str(ctx.exception))

  @parameterized.named_parameters(
      ("fp8", "fp8", jnp.float8_e4m3fn),
      ("nanoo_fp8", "nanoo_fp8", jnp.float8_e4m3fnuz),
  )
  def test_fp8_einsum_quantizes_both_operands(self, quantization, e4m3_dtype):
    """The bridged einsum is the plain one with both operands cast to the scheme's e4m3."""
    model = self._make_moe(self._quantization(quantization))
    lhs = jax.random.normal(jax.random.PRNGKey(0), (4, 16), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(1), (16, 8), dtype=jnp.float32)

    actual = model.get_einsum(einsum_name=moe.WI_0)("ab,bc->ac", lhs, rhs)

    # The scaling factors start at 1 and are only updated on the backward pass, so a forward
    # call on a freshly built layer quantizes by a plain cast.
    quantized = jnp.einsum(
        "ab,bc->ac", lhs.astype(e4m3_dtype).astype(jnp.float32), rhs.astype(e4m3_dtype).astype(jnp.float32)
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(quantized))
    self.assertFalse(np.array_equal(np.asarray(actual), np.asarray(jnp.einsum("ab,bc->ac", lhs, rhs))))

  @parameterized.named_parameters(("fp8", "fp8"), ("nanoo_fp8", "nanoo_fp8"))
  def test_quantized_dense_matmul_tracks_unquantized(self, quantization):
    """A quantized MoE layer follows the same layer run unquantized, to within e4m3."""
    reference = self._make_moe(None)
    model = self._make_moe(self._quantization(quantization))
    copy_weights(reference, model)

    inputs = jax.random.normal(jax.random.PRNGKey(42), (1, 8, reference.config.base_emb_dim), dtype=jnp.float32)
    expected, _, _ = reference(inputs)
    actual, _, _ = model(inputs)

    self.assertTrue(np.isfinite(actual).all())
    # e4m3 keeps three mantissa bits, and the layer is quantized at each of wi_0, wi_1 and wo,
    # so the agreement is loose; it is the same threshold the qwix MoE test above uses.
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    self.assertLess(relative_error, 0.22)
    self.assertFalse(np.allclose(actual, expected))


class SparseMatmulQuantizationTest(parameterized.TestCase):
  """The dtypes RoutedMoE hands the grouped matmul on the sparse_matmul path."""

  @parameterized.named_parameters(
      ("fp8", "fp8", False, jnp.float8_e4m3fn),
      ("nanoo_fp8", "nanoo_fp8", False, jnp.float8_e4m3fnuz),
      ("int8", "int8", False, jnp.int8),
      ("unquantized", "", False, None),
      # Under qwix a non-fp8_full scheme declares no gmm rule, so the gmm is left alone.
      ("fp8_under_qwix", "fp8", True, None),
      ("int8_under_qwix", "int8", True, None),
  )
  def test_gmm_quantize_dtypes(self, quantization, use_qwix_quantization, expected_dtype):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="sparse_matmul_quantization_test",
        enable_checkpointing=False,
        decoder_block="mixtral",
        num_experts=4,
        num_experts_per_tok=2,
        base_emb_dim=64,
        base_mlp_dim=32,
        base_moe_mlp_dim=32,
        dtype="float32",
        weight_dtype="float32",
        sparse_matmul=True,
        megablox=True,
        max_target_length=8,
        per_device_batch_size=1,
        quantization=quantization,
        use_qwix_quantization=use_qwix_quantization,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=jnp.float32,
        quant=configure_quantization(cfg),
        rngs=nnx.Rngs(0),
    )

    calls = []

    def record_gmm(**kwargs):
      calls.append(kwargs)
      return jnp.zeros((kwargs["lhs"].shape[0], kwargs["rhs"].shape[-1]), dtype=kwargs["preferred_element_type"])

    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 8, cfg.base_emb_dim), dtype=jnp.float32)
    with mock.patch.object(moe.mblx, "gmm", record_gmm):
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model(inputs)

    self.assertNotEmpty(calls)
    for kwargs in calls:
      self.assertEqual(kwargs["lhs_quantize_dtype"], expected_dtype)
      self.assertEqual(kwargs["rhs_quantize_dtype"], expected_dtype)


def make_moe(cfg, mesh, intermediate_dim: int = 2048, weight_quant=None):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=intermediate_dim,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      weight_quant=weight_quant,
      rngs=nnx.Rngs(params=0),
  )


def copy_weights(src_model, dst_model):
  """Copy wi_0, wi_1, wo, and gate weights from src to dst."""
  dst_model.wi_0 = src_model.wi_0
  dst_model.wi_1 = src_model.wi_1
  dst_model.wo = src_model.wo
  dst_model.gate = src_model.gate


def copy_weights_prefused(src_model, dst_model):
  """Copy weights from a split-weight model into a prefuse_moe_weights=True model.

  Concatenates src wi_0 and wi_1 along the last axis to produce the fused wi.
  """
  wi_fused = jnp.concatenate([src_model.wi_0[...], src_model.wi_1[...]], axis=-1)
  dst_model.wi = nnx.Param(wi_fused)
  dst_model.wo = src_model.wo
  dst_model.gate = src_model.gate


@pytest.mark.tpu_only
@pytest.mark.post_training
class FusedMoeTPUTest(unittest.TestCase):
  """Tests for fused_moe_matmul (vllm_rpa path) in RoutedMoE."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._B = 1  # per-device batch size
    self._S = 16  # sequence length

  def setUp(self):
    super().setUp()
    try:
      import tpu_inference  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
      self.skipTest("Fused MoE tests require tpu-inference package to be installed.")
    self.rng = jax.random.PRNGKey(42)

    # Dense reference config (no vllm, einsum-based)
    self.dense_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_dense_ref",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=False,
        megablox=False,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    dense_devices = maxtext_utils.create_device_mesh(self.dense_cfg)
    self.dense_mesh = Mesh(dense_devices, self.dense_cfg.mesh_axes)
    self.dense_model = make_moe(self.dense_cfg, self.dense_mesh)

    # vllm_rpa fused config
    self.fused_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml")],
        run_name="fused_moe_vllm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    fused_devices = maxtext_utils.create_device_mesh(self.fused_cfg)
    self.fused_mesh = Mesh(fused_devices, self.fused_cfg.mesh_axes)
    self.fused_model = make_moe(self.fused_cfg, self.fused_mesh)
    copy_weights(self.dense_model, self.fused_model)

  def _inputs(self):
    return jax.random.normal(self.rng, (self._B, self._S, self.dense_cfg.base_emb_dim), dtype=jnp.bfloat16)

  def test_fused_vs_dense_softmax(self):
    """fused_moe_matmul agrees with dense_matmul under softmax routing."""
    inputs = self._inputs()

    dense_out, _, _ = self.dense_model(inputs)
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    np.testing.assert_allclose(
        np.array(dense_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_vs_sparse_softmax(self):
    """fused_moe_matmul agrees with sparse_matmul (Megablox) under softmax routing."""
    sparse_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_sparse_ref",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    sparse_devices = maxtext_utils.create_device_mesh(sparse_cfg)
    sparse_mesh = Mesh(sparse_devices, sparse_cfg.mesh_axes)
    sparse_model = make_moe(sparse_cfg, sparse_mesh)
    copy_weights(self.dense_model, sparse_model)

    inputs = self._inputs()
    with nn_partitioning.axis_rules(sparse_cfg.logical_axis_rules):
      sparse_out, _, _ = sparse_model(inputs)
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    np.testing.assert_allclose(
        np.array(sparse_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_output_shape_and_dtype(self):
    """Output shape is (B, S, D), dtype matches cfg.dtype, and losses are None."""
    inputs = self._inputs()
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    expected_shape = (self._B, self._S, self.fused_cfg.base_emb_dim)
    self.assertEqual(fused_out.shape, expected_shape)
    self.assertEqual(fused_out.dtype, self.fused_cfg.dtype)
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_vs_dense_renormalize(self):
    """fused_moe_matmul agrees with dense_matmul when norm_topk_prob=True."""
    dense_renorm_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_dense_renorm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=False,
        megablox=False,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        norm_topk_prob=True,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    dense_renorm_devices = maxtext_utils.create_device_mesh(dense_renorm_cfg)
    dense_renorm_mesh = Mesh(dense_renorm_devices, dense_renorm_cfg.mesh_axes)
    dense_renorm_model = make_moe(dense_renorm_cfg, dense_renorm_mesh)

    fused_renorm_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml")],
        run_name="fused_moe_vllm_renorm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        norm_topk_prob=True,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    fused_renorm_devices = maxtext_utils.create_device_mesh(fused_renorm_cfg)
    fused_renorm_mesh = Mesh(fused_renorm_devices, fused_renorm_cfg.mesh_axes)
    fused_renorm_model = make_moe(fused_renorm_cfg, fused_renorm_mesh)
    copy_weights(dense_renorm_model, fused_renorm_model)

    inputs = self._inputs()
    dense_out, _, _ = dense_renorm_model(inputs)
    fused_out, lb_loss, bias_updates = fused_renorm_model(inputs)

    np.testing.assert_allclose(
        np.array(dense_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_prefused_vs_dense_softmax(self):
    """prefuse_moe_weights=True agrees with dense_matmul under softmax routing."""
    prefused_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml")],
        run_name="fused_moe_vllm_prefused",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        prefuse_moe_weights=True,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    prefused_devices = maxtext_utils.create_device_mesh(prefused_cfg)
    prefused_mesh = Mesh(prefused_devices, prefused_cfg.mesh_axes)
    prefused_model = make_moe(prefused_cfg, prefused_mesh)
    copy_weights_prefused(self.dense_model, prefused_model)

    inputs = self._inputs()
    dense_out, _, _ = self.dense_model(inputs)
    prefused_out, lb_loss, bias_updates = prefused_model(inputs)

    np.testing.assert_allclose(
        np.array(dense_out, dtype=np.float32),
        np.array(prefused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_prefused_vs_sparse_softmax(self):
    """prefuse_moe_weights=True agrees with sparse_matmul (Megablox) under softmax routing."""
    sparse_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_sparse_ref2",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        ici_expert_parallelism=jax.device_count(),
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    sparse_devices = maxtext_utils.create_device_mesh(sparse_cfg)
    sparse_mesh = Mesh(sparse_devices, sparse_cfg.mesh_axes)
    sparse_model = make_moe(sparse_cfg, sparse_mesh)
    copy_weights(self.dense_model, sparse_model)

    prefused_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml")],
        run_name="fused_moe_vllm_prefused2",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        ici_expert_parallelism=jax.device_count(),
        prefuse_moe_weights=True,
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    prefused_devices = maxtext_utils.create_device_mesh(prefused_cfg)
    prefused_mesh = Mesh(prefused_devices, prefused_cfg.mesh_axes)
    prefused_model = make_moe(prefused_cfg, prefused_mesh)
    copy_weights_prefused(self.dense_model, prefused_model)

    inputs = self._inputs()
    with nn_partitioning.axis_rules(sparse_cfg.logical_axis_rules):
      sparse_out, _, _ = sparse_model(inputs)
    prefused_out, lb_loss, bias_updates = prefused_model(inputs)

    np.testing.assert_allclose(
        np.array(sparse_out, dtype=np.float32),
        np.array(prefused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)


def _quantize_moe_weight_blockwise(rng, shape, block_size):
  """Absmax-quantizes synthetic (E, K, N) MoE weight per (block_size, block_size) tile."""
  e, k, n = shape
  kb, nb = k // block_size, n // block_size
  w = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
  w_blocks = w.reshape(e, kb, block_size, nb, block_size)
  scale = np.max(np.abs(w_blocks), axis=(2, 4)) / 448.0
  scale = np.where(scale == 0, 1.0, scale)
  scale_full = np.repeat(np.repeat(scale, block_size, axis=1), block_size, axis=2)
  w_q = np.clip(np.round(w / scale_full), -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn)
  return w_q, scale.astype(np.float32)


@pytest.mark.tpu_only
@pytest.mark.post_training
class SparseMoeNativeGmmPerChannelAxisTest(unittest.TestCase):
  """Tests that K-axis per-channel scales fall back to dequantize in RoutedMoE."""

  def test_k_axis_only_scale_falls_back_to_dequantize(self):
    block_size = 128
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_native_gmm_k_axis_scale",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="float8_e4m3fn",
        weight_block_size=block_size,
        quantization="serve_fp8_weight",
        sparse_matmul=True,
        use_gmm_v2=True,
        use_tokamax_gmm=True,
        megablox=True,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=16,
        per_device_batch_size=1,
    )
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        weight_dtype=jnp.float8_e4m3fn,
        quant=configure_quantization(cfg),
        rngs=nnx.Rngs(params=0),
    )

    e, embed, moe_mlp = model.num_experts, model.moe_expert_input_dim, model.intermediate_dim
    rng = np.random.default_rng(5)
    wi_0_q, wi_0_scale_full = _quantize_moe_weight_blockwise(rng, (e, embed, moe_mlp), block_size)
    wi_1_q, wi_1_scale_full = _quantize_moe_weight_blockwise(rng, (e, embed, moe_mlp), block_size)
    wo_q, wo_scale_full = _quantize_moe_weight_blockwise(rng, (e, moe_mlp, embed), block_size)
    # Collapse block-wise grid to K-axis per-channel scale (channel_axis == 0).
    model.wi_0[...] = jnp.asarray(wi_0_q)
    model.wi_0_scale[...] = jnp.asarray(np.max(wi_0_scale_full, axis=2, keepdims=True))
    model.wi_1[...] = jnp.asarray(wi_1_q)
    model.wi_1_scale[...] = jnp.asarray(np.max(wi_1_scale_full, axis=2, keepdims=True))
    model.wo[...] = jnp.asarray(wo_q)
    model.wo_scale[...] = jnp.asarray(np.max(wo_scale_full, axis=2, keepdims=True))

    inputs = jax.random.normal(jax.random.PRNGKey(11), (1, 16, embed), dtype=jnp.bfloat16)
    native_quant = model.quant
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      native_out, _, _ = model(inputs)
      model.quant = None  # native_gmm requires isinstance(self.quant, ServeFp8WeightQuantization)
      dequant_out, _, _ = model(inputs)
    model.quant = native_quant

    native_np = np.array(native_out, dtype=np.float32)
    dequant_np = np.array(dequant_out, dtype=np.float32)
    self.assertTrue(np.all(np.isfinite(native_np)))
    # Output should match since K-axis scale falls back to dequantize.
    np.testing.assert_allclose(native_np, dequant_np, rtol=1e-5, atol=1e-5)


@pytest.mark.tpu_only
@pytest.mark.post_training
class SparseMoeNativeGmmPerTensorTest(unittest.TestCase):
  """Covers the per_tensor branch of _maybe_native_gmm_weight (untested by the
  other native-gmm tests). Needs weight_block_size=None for a genuine 1D
  per-expert scale allocation.
  """

  def test_per_tensor_scale_matches_dequantize(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_native_gmm_per_tensor_scale",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="float8_e4m3fn",
        weight_block_size=None,
        quantization="serve_fp8_weight",
        sparse_matmul=True,
        use_gmm_v2=True,
        use_tokamax_gmm=True,
        megablox=True,
        ici_expert_parallelism=jax.device_count(),
        log_config=False,
        max_target_length=16,
        per_device_batch_size=1,
    )
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        weight_dtype=jnp.float8_e4m3fn,
        quant=configure_quantization(cfg),
        rngs=nnx.Rngs(params=0),
    )

    e, embed, moe_mlp = model.num_experts, model.moe_expert_input_dim, model.intermediate_dim
    rng = np.random.default_rng(3)

    def _quantize_per_tensor(shape):
      w = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
      scale = np.max(np.abs(w), axis=tuple(range(1, w.ndim))) / 448.0  # true 1D (E,)
      scale = np.where(scale == 0, 1.0, scale)
      w_q = np.clip(np.round(w / scale[:, None, None]), -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn)
      return w_q, scale.astype(np.float32)

    wi_0_q, wi_0_scale = _quantize_per_tensor((e, embed, moe_mlp))
    wi_1_q, wi_1_scale = _quantize_per_tensor((e, embed, moe_mlp))
    wo_q, wo_scale = _quantize_per_tensor((e, moe_mlp, embed))
    self.assertEqual(model.wo_scale[...].shape, (e,))  # confirms a true 1D per-expert allocation
    model.wi_0[...] = jnp.asarray(wi_0_q)
    model.wi_0_scale[...] = jnp.asarray(wi_0_scale)
    model.wi_1[...] = jnp.asarray(wi_1_q)
    model.wi_1_scale[...] = jnp.asarray(wi_1_scale)
    model.wo[...] = jnp.asarray(wo_q)
    model.wo_scale[...] = jnp.asarray(wo_scale)

    inputs = jax.random.normal(jax.random.PRNGKey(11), (1, 16, embed), dtype=jnp.bfloat16)
    native_quant = model.quant
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      native_out, _, _ = model(inputs)
      model.quant = None
      dequant_out, _, _ = model(inputs)
    model.quant = native_quant

    native_np = np.array(native_out, dtype=np.float32)
    dequant_np = np.array(dequant_out, dtype=np.float32)
    self.assertTrue(np.all(np.isfinite(native_np)))
    relerr = float(np.max(np.abs(native_np - dequant_np)) / (np.max(np.abs(dequant_np)) + 1e-12))
    self.assertLess(relerr, 0.1, f"native (per_tensor) vs dequantize relerr={relerr:.3e}")


@pytest.mark.tpu_only
@pytest.mark.post_training
class FusedMoeNativeFp8Test(unittest.TestCase):
  """Tests native FP8 through fused_moe_matmul -- the real vllm_rpa rollout
  MoE path, gated separately from sparse_matmul+gmm_v2 (native_fused_gmm vs.
  native_gmm in moe.py).
  """

  def _make_model(self, cfg):
    """Builds a RoutedMoE on a real 2-axis ("data", "model") mesh."""
    # fused_moe_func's shard_map hardcodes tpu_inference's ("data", "model")
    # axis names, not MaxText's training-mesh convention -- build the same
    # 2-axis mesh real rollout uses.
    devices = np.array(jax.devices()).reshape(-1, 1)  # (data=N, model=1)
    mesh = Mesh(devices, ("data", "model"))
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        # RoutedMoE.__init__'s weight_dtype is a separate constructor arg from
        # cfg.weight_dtype -- omitting it silently builds a non-fp8 model.
        weight_dtype=jnp.float8_e4m3fn,
        quant=configure_quantization(cfg),
        rngs=nnx.Rngs(params=0),
    )
    return model, mesh

  def _assign_synthetic_fp8_weights(self, model, block_size, seed=7):
    """Assigns block-wise quantized weights matching a real weight_block_size=128
    checkpoint layout (not a collapsed per-tensor/per-channel special case)."""
    e, embed, moe_mlp = model.num_experts, model.moe_expert_input_dim, model.intermediate_dim
    rng = np.random.default_rng(seed)
    wi_0_q, wi_0_scale = _quantize_moe_weight_blockwise(rng, (e, embed, moe_mlp), block_size)
    wi_1_q, wi_1_scale = _quantize_moe_weight_blockwise(rng, (e, embed, moe_mlp), block_size)
    wo_q, wo_scale = _quantize_moe_weight_blockwise(rng, (e, moe_mlp, embed), block_size)
    model.wi_0[...] = jnp.asarray(wi_0_q)
    model.wi_0_scale[...] = jnp.asarray(wi_0_scale)
    model.wi_1[...] = jnp.asarray(wi_1_q)
    model.wi_1_scale[...] = jnp.asarray(wi_1_scale)
    model.wo[...] = jnp.asarray(wo_q)
    model.wo_scale[...] = jnp.asarray(wo_scale)
    return embed

  def _assign_synthetic_fused_fp8_weights(self, model, block_size, seed=7):
    """Like _assign_synthetic_fp8_weights, but for prefuse_moe_weights=True:
    assigns directly to the single fused model.wi/model.wi_scale (shape
    (E, embed, 2*moe_mlp)), matching what a real prefused rollout checkpoint
    -- and the fused_native_scale branch in RoutedMoE.__call__ -- actually
    reads."""
    e, embed, moe_mlp = model.num_experts, model.moe_expert_input_dim, model.intermediate_dim
    rng = np.random.default_rng(seed)
    wi_q, wi_scale = _quantize_moe_weight_blockwise(rng, (e, embed, 2 * moe_mlp), block_size)
    wo_q, wo_scale = _quantize_moe_weight_blockwise(rng, (e, moe_mlp, embed), block_size)
    model.wi[...] = jnp.asarray(wi_q)
    model.wi_scale[...] = jnp.asarray(wi_scale)
    model.wo[...] = jnp.asarray(wo_q)
    model.wo_scale[...] = jnp.asarray(wo_scale)
    return embed

  def _make_config(self, run_name, **overrides):
    """Base vllm_rpa + serve_fp8_weight config, block_size=128."""
    block_size = 128
    kwargs = dict(  # pylint: disable=use-dict-literal
        run_name=run_name,
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="float8_e4m3fn",
        weight_block_size=block_size,
        quantization="serve_fp8_weight",
        attention="vllm_rpa",
        # EP=1: real expert parallelism hits an unrelated pre-existing bug in
        # this tpu_inference version's ragged_gather kernel.
        ici_expert_parallelism=1,
        log_config=False,
        max_target_length=16,
        per_device_batch_size=1,
    )
    kwargs.update(overrides)
    return pyconfig.initialize([None, get_test_config_path()], **kwargs), block_size

  def test_native_matches_dequantize_baseline(self):
    """Native FP8 through fused_moe_matmul should closely match the existing
    dequantize-then-requantize baseline."""
    cfg, block_size = self._make_config("fused_moe_native_fp8")
    model, _ = self._make_model(cfg)
    embed = self._assign_synthetic_fp8_weights(model, block_size)

    inputs = jax.random.normal(jax.random.PRNGKey(11), (1, 16, embed), dtype=jnp.bfloat16)
    native_quant = model.quant
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      native_out, _, _ = model(inputs)
      model.quant = None
      dequant_out, _, _ = model(inputs)
    model.quant = native_quant

    native_np = np.array(native_out, dtype=np.float32)
    dequant_np = np.array(dequant_out, dtype=np.float32)
    self.assertTrue(np.all(np.isfinite(native_np)))
    relerr = float(np.max(np.abs(native_np - dequant_np)) / (np.max(np.abs(dequant_np)) + 1e-12))
    # Loose tolerance: native skips fused_moe_func's own activation
    # requantization, so the two paths aren't bit-identical.
    self.assertLess(relerr, 0.1, f"native vs dequantize relerr={relerr:.3e}")

  def test_native_matches_dequantize_baseline_prefused(self):
    """Same as test_native_matches_dequantize_baseline, but with
    prefuse_moe_weights=True -- the actual real vLLM serving configuration
    (weight_converter.py fuses wi_0/wi_1 -> wi before rollout sees it), which
    exercises the fused_native_scale branch and prepare_fused_gmm_scale on a
    real block-quantized (not collapsed) scale shape."""
    cfg, block_size = self._make_config("fused_moe_native_fp8_prefused", prefuse_moe_weights=True)
    model, _ = self._make_model(cfg)
    embed = self._assign_synthetic_fused_fp8_weights(model, block_size)

    inputs = jax.random.normal(jax.random.PRNGKey(13), (1, 16, embed), dtype=jnp.bfloat16)
    native_quant = model.quant
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      native_out, _, _ = model(inputs)
      model.quant = None
      dequant_out, _, _ = model(inputs)
    model.quant = native_quant

    native_np = np.array(native_out, dtype=np.float32)
    dequant_np = np.array(dequant_out, dtype=np.float32)
    self.assertTrue(np.all(np.isfinite(native_np)))
    relerr = float(np.max(np.abs(native_np - dequant_np)) / (np.max(np.abs(dequant_np)) + 1e-12))
    self.assertLess(relerr, 0.1, f"native vs dequantize relerr={relerr:.3e}")

  def test_sparse_matmul_gmm_v2_flags_dont_crash_under_vllm_rpa(self):
    """Regression test: sparse_matmul=True + use_gmm_v2=True with
    attention=vllm_rpa used to wrap kernels in a QArray that crashed
    fused_moe_matmul's jnp.concatenate. vllm_rpa always wins the dispatch,
    so this should behave like sparse_matmul=False."""
    cfg, block_size = self._make_config(
        "fused_moe_sparse_matmul_guard",
        sparse_matmul=True,
        use_gmm_v2=True,
        use_tokamax_gmm=True,
        megablox=True,
    )
    model, _ = self._make_model(cfg)
    embed = self._assign_synthetic_fp8_weights(model, block_size)
    inputs = jax.random.normal(jax.random.PRNGKey(12), (1, 16, embed), dtype=jnp.bfloat16)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      out, lb_loss, bias_updates = model(inputs)  # must not raise
    self.assertTrue(np.all(np.isfinite(np.array(out, dtype=np.float32))))
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)


@pytest.mark.parametrize(
    "model_name,flag",
    [
        ("mixtral-8x7b", True),  # flag on: expert stays on E, peeled off the batch dim
        ("mixtral-8x22b", True),  # flag on
        ("mixtral-8x7b", False),  # flag off (default): batch dim keeps 'expert'
    ],
)
def test_moe_dispatch_keeps_expert_on_expert_dim(model_name, flag):
  """Regression guard for the MoE dispatch/MLP expert-parallel sharding.

  The expert (E) dim is always sharded by the 'expert' mesh axis (via activation_exp).
  With moe_dispatch_no_expert_sharding the batch (B) dim must NOT also take 'expert'
  (which would double-map two tensor dims onto one mesh axis and force an FSDP-style
  fallback instead of expert-parallel AllToAll); with the flag off, the default keeps
  'expert' on the batch dim. Mirrors dense_matmul's axis selection.
  """
  cfg = pyconfig.initialize(
      [None, get_test_config_path()],
      run_name=f"moe_shard_{model_name}_{flag}",
      enable_checkpointing=False,
      model_name=model_name,
      moe_dispatch_no_expert_sharding=flag,
  )
  rules = cfg.logical_axis_rules

  def _as_set(entry):
    if entry is None:
      return set()
    return {entry} if isinstance(entry, str) else set(entry)

  # Mirror _maybe_shard_moe_dispatch: resolve E and batch dims independently (so the shared
  # 'expert' axis isn't deduped off E), then peel 'expert' from the batch dim when the flag is set.
  e_spec = nn_partitioning.logical_to_mesh_axes(("activation_exp",), rules=rules)
  b_spec = nn_partitioning.logical_to_mesh_axes(("activation_batch_moe",), rules=rules)
  if cfg.moe_dispatch_no_expert_sharding:
    b_spec = remove_expert_from_partition_spec(b_spec, dims_to_peel=(0,))

  e_axes, b_axes = _as_set(e_spec[0]), _as_set(b_spec[0])
  assert "expert" in e_axes, "expert dim must be sharded by the 'expert' mesh axis"
  if cfg.moe_dispatch_no_expert_sharding:
    assert "expert" not in b_axes, "flag on: the batch dim must not take 'expert'"
  else:
    assert "expert" in b_axes, "flag off (default): the batch dim keeps 'expert' (activation_batch_moe)"


def test_remove_expert_from_partition_spec():
  """remove_expert_from_partition_spec peels 'expert' only from the requested dims."""
  spec = jax.sharding.PartitionSpec
  assert remove_expert_from_partition_spec(spec("expert", ("data", "fsdp", "expert"), None), dims_to_peel=(1,)) == spec(
      "expert", ("data", "fsdp"), None
  )
  assert remove_expert_from_partition_spec(spec("expert", "expert", None), dims_to_peel=(1,)) == spec(
      "expert", None, None
  )
  assert remove_expert_from_partition_spec(spec(("expert",), None), dims_to_peel=(0, 1)) == spec(None, None)


def test_moe_dispatch_no_expert_sharding_dense_forward():
  """The moe_dispatch_no_expert_sharding peel path runs in dense_matmul (capacity_factor>0)."""
  cfg = pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="moe_dense_no_exp_fwd",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      dtype="bfloat16",
      megablox=False,
      sparse_matmul=False,
      capacity_factor=1.0,
      moe_dispatch_no_expert_sharding=True,
      per_device_batch_size=1,
      max_target_length=16,
  )
  devices = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices, cfg.mesh_axes)
  model = make_moe(cfg, mesh)
  inputs = jax.random.normal(
      jax.random.PRNGKey(0),
      (int(cfg.per_device_batch_size) * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim),
      dtype=jnp.bfloat16,
  )
  with mesh:
    out, _, _ = model(inputs)
  assert out.shape == inputs.shape


@pytest.mark.tpu_only
class FusedMlpMoETest(unittest.TestCase):
  """Tests that prefuse_moe_weights=True and prefuse_moe_weights=False produce identical outputs for MoE."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._B = 1
    self._S = 16

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.ref_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_mlp_moe_ref",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        prefuse_moe_weights=False,
        ici_expert_parallelism=jax.device_count(),
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    ref_devices = maxtext_utils.create_device_mesh(self.ref_cfg)
    self.ref_mesh = Mesh(ref_devices, self.ref_cfg.mesh_axes)
    self.ref_model = make_moe(self.ref_cfg, self.ref_mesh)

  def _inputs(self):
    return jax.random.normal(self.rng, (self._B, self._S, self.ref_cfg.base_emb_dim), dtype=jnp.bfloat16)

  def test_prefuse_moe_weights_matches_unfused(self):
    """prefuse_moe_weights=True output matches prefuse_moe_weights=False with sparse_matmul (Megablox)."""
    fused_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_mlp_moe_fused",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        prefuse_moe_weights=True,
        ici_expert_parallelism=jax.device_count(),
        max_target_length=self._S,
        per_device_batch_size=self._B,
    )
    fused_devices = maxtext_utils.create_device_mesh(fused_cfg)
    fused_mesh = Mesh(fused_devices, fused_cfg.mesh_axes)
    fused_model = make_moe(fused_cfg, fused_mesh)
    copy_weights_prefused(self.ref_model, fused_model)

    inputs = self._inputs()
    ref_out, _, _ = self.ref_model(inputs)
    fused_out, _, _ = fused_model(inputs)

    np.testing.assert_allclose(
        np.array(ref_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


class MoePinSparseCoreAllGathersTest(unittest.TestCase):
  """Tests for moe_pin_sparse_core_all_gathers config and model construction."""

  def test_config_moe_pin_sparse_core_all_gathers(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_moe_pin_sc",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        moe_pin_sparse_core_all_gathers=True,
        moe_fsdp_all_gather_sparse_core_id=0,
        moe_ep_all_gather_sparse_core_id=1,
    )
    self.assertTrue(cfg.moe_pin_sparse_core_all_gathers)
    self.assertEqual(cfg.moe_fsdp_all_gather_sparse_core_id, 0)
    self.assertEqual(cfg.moe_ep_all_gather_sparse_core_id, 1)

  def test_moe_pin_sparse_core_with_two_stage_all_gather_raises(self):
    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="test_moe_pin_sc_two_stage_error",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          moe_pin_sparse_core_all_gathers=True,
          moe_fsdp_use_two_stage_all_gather=True,
      )

  def test_moe_model_init_with_pin_sparse_core(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_moe_pin_sc_model",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        moe_pin_sparse_core_all_gathers=True,
        moe_fsdp_all_gather_sparse_core_id=0,
        moe_ep_all_gather_sparse_core_id=1,
        per_device_batch_size=1,
        max_target_length=16,
    )
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh)
    self.assertTrue(model.config.moe_pin_sparse_core_all_gathers)

  # `use_ring_of_experts` (required by `moe_quantize_token_all_gather`) is only a
  # valid config when the EP rank is > 1, so this test needs a multi-device mesh.
  @pytest.mark.tpu_only
  def test_moe_pin_sparse_core_with_quantize_token_all_gather_init(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_moe_pin_sc_quant_model",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=False,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        ici_expert_parallelism=4,
        use_ring_of_experts=True,
        quantization="fp8_full",
        use_qwix_quantization=True,
        weight_quantization_calibration_method="fixed,-224,224",
        act_quantization_calibration_method="fixed,-224,224",
        moe_pin_sparse_core_all_gathers=True,
        moe_quantize_token_all_gather=True,
        moe_fsdp_all_gather_sparse_core_id=0,
        moe_ep_all_gather_sparse_core_id=1,
        per_device_batch_size=1,
        max_target_length=16,
    )
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh)
    self.assertTrue(model.config.moe_pin_sparse_core_all_gathers)
    self.assertTrue(model.config.moe_quantize_token_all_gather)


class RoutedMoEFp8Test(parameterized.TestCase):
  """Unit tests for RoutedMoE FP8 weight storage and dynamic dequantization scales."""

  def _make_fp8_cfg(self, prefuse_moe_weights=True, weight_block_size=None):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fp8_moe_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        weight_dtype="float8_e4m3fn",
        dtype="bfloat16",
        base_emb_dim=128,
        base_mlp_dim=64,
        base_moe_mlp_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        prefuse_moe_weights=prefuse_moe_weights,
        weight_block_size=weight_block_size,
        megablox=False,
        sparse_matmul=False,
        per_device_batch_size=1,
        max_target_length=8,
    )

  def test_fp8_prefused_per_expert_scale_init(self):
    """Verifies 1D per-expert scale initialization and sharding when prefuse_moe_weights=True."""
    cfg = self._make_fp8_cfg(prefuse_moe_weights=True, weight_block_size=None)
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh, intermediate_dim=cfg.base_moe_mlp_dim)

    self.assertIsNotNone(model.wi_scale)
    self.assertIsNotNone(model.wo_scale)
    self.assertIsNone(model.wi_0_scale)
    self.assertIsNone(model.wi_1_scale)

    # 1D scale shape per expert
    self.assertEqual(model.wi_scale.shape, (cfg.num_experts,))
    self.assertEqual(model.wo_scale.shape, (cfg.num_experts,))
    self.assertEqual(model.wi_scale.dtype, jnp.float32)
    self.assertEqual(model.wo_scale.dtype, jnp.float32)

    # Sharding must be 1D (expert axis only) to avoid rank mismatch with 3D kernel axes
    self.assertEqual(model.wi_scale_axes, (model.wi_kernel_axes[0],))
    self.assertEqual(model.wo_scale_axes, (model.wo_kernel_axes[0],))

  def test_fp8_prefused_block_scale_init(self):
    """Verifies 3D block-grid scale initialization and sharding when block scaling is configured."""
    block_size = [64, 32]
    cfg = self._make_fp8_cfg(prefuse_moe_weights=True, weight_block_size=block_size)
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh, intermediate_dim=cfg.base_moe_mlp_dim)

    # wi has input_dim=128, fused_intermediate_dim=128 (64*2)
    # in_blocks = 128 // 64 = 2; out_blocks = 128 // 32 = 4
    expected_wi_shape = (cfg.num_experts, 128 // block_size[0], (cfg.base_moe_mlp_dim * 2) // block_size[1])
    # wo has in_dim=64, out_dim=128
    # out_blocks = 64 // 32 = 2; in_blocks = 128 // 64 = 2
    expected_wo_shape = (cfg.num_experts, cfg.base_moe_mlp_dim // block_size[1], 128 // block_size[0])

    self.assertEqual(model.wi_scale.shape, expected_wi_shape)
    self.assertEqual(model.wo_scale.shape, expected_wo_shape)
    self.assertEqual(model.wi_scale_axes, model.wi_kernel_axes)
    self.assertEqual(model.wo_scale_axes, model.wo_kernel_axes)

  def test_fp8_unfused_scale_init(self):
    """Verifies scale initialization and sharding when prefuse_moe_weights=False."""
    cfg = self._make_fp8_cfg(prefuse_moe_weights=False, weight_block_size=None)
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh, intermediate_dim=cfg.base_moe_mlp_dim)

    self.assertIsNone(model.wi_scale)
    self.assertIsNotNone(model.wi_0_scale)
    self.assertIsNotNone(model.wi_1_scale)
    self.assertIsNotNone(model.wo_scale)

    self.assertEqual(model.wi_0_scale.shape, (cfg.num_experts,))
    self.assertEqual(model.wi_1_scale.shape, (cfg.num_experts,))
    self.assertEqual(model.wo_scale.shape, (cfg.num_experts,))

    self.assertEqual(model.wi_scale_axes, (model.wi_kernel_axes[0],))
    self.assertEqual(model.wo_scale_axes, (model.wo_kernel_axes[0],))

  def test_fp8_forward_pass_dequantization(self):
    """Verifies dynamic dequantization during MoE forward pass produces valid bfloat16 outputs."""
    cfg = self._make_fp8_cfg(prefuse_moe_weights=True, weight_block_size=None)
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    model = make_moe(cfg, mesh, intermediate_dim=cfg.base_moe_mlp_dim)

    inputs = jax.random.normal(
        jax.random.PRNGKey(0),
        (cfg.per_device_batch_size, cfg.max_target_length, cfg.base_emb_dim),
        dtype=jnp.bfloat16,
    )
    with jax.set_mesh(mesh):
      out, _, _ = model(inputs)

    self.assertEqual(out.shape, inputs.shape)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertTrue(np.all(np.isfinite(out)))

  def test_fp8_moe_with_weight_quant_config(self):
    """Verifies RoutedMoE initialization when explicit weight_quant is provided."""
    cfg = self._make_fp8_cfg(prefuse_moe_weights=True, weight_block_size=None)
    devices = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices, cfg.mesh_axes)
    wq = WeightQuantConfig(
        quant_type="fp8",
        weight_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.float32,
        block_size=[64, 32],
    )
    model = make_moe(cfg, mesh, intermediate_dim=cfg.base_moe_mlp_dim, weight_quant=wq)
    self.assertIsNotNone(model.weight_quant)
    self.assertEqual(model.weight_quant.block_size, [64, 32])
    expected_wi_shape = (cfg.num_experts, 128 // 64, (cfg.base_moe_mlp_dim * 2) // 32)
    self.assertEqual(model.wi_scale.shape, expected_wi_shape)

  @parameterized.named_parameters(
      {"testcase_name": "rowwise", "bwd_method": "rowwise", "max_mse": 0.05},
      {"testcase_name": "static", "bwd_method": "fixed,57344", "max_mse": 0.1},
  )
  def test_moe_combine_psum_scatter_bwd(self, bwd_method: str, max_mse: float):
    """Verifies _moe_combine_psum_scatter custom VJP on 3D tensors."""
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 32), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 32), dtype=jnp.bfloat16)

    with (
        mock.patch.object(jax.lax, "psum_scatter", side_effect=lambda v, *a, **k: v),
        mock.patch.object(jax.lax, "all_gather", side_effect=lambda v, *a, **k: v),
    ):
      out_ref, vjp_ref = jax.vjp(
          lambda v: _moe_combine_psum_scatter(v, "expert", scatter_dimension=0, tiled=False, bwd_method=""), x
      )
      (d_ref,) = vjp_ref(cotangent)

      out_q, vjp_q = jax.vjp(
          lambda v: _moe_combine_psum_scatter(v, "expert", scatter_dimension=0, tiled=False, bwd_method=bwd_method), x
      )
      (d_q,) = vjp_q(cotangent)

    np.testing.assert_allclose(out_q, out_ref, rtol=1e-5, atol=1e-5)
    self.assertEqual(d_q.dtype, jnp.bfloat16)
    self.assertEqual(d_q.shape, cotangent.shape)
    mse = jnp.mean((d_ref.astype(jnp.float32) - d_q.astype(jnp.float32)) ** 2)
    self.assertLess(float(mse), max_mse)

  @parameterized.named_parameters(
      {"testcase_name": "rowwise", "bwd_method": "rowwise", "scale_shape": (2, 4, 1, 1)},
      {"testcase_name": "static", "bwd_method": "fixed,57344", "scale_shape": None},
  )
  def test_moe_combine_psum_scatter_bwd_3d_dispatch_layout(self, bwd_method: str, scale_shape):
    """The (batch, seq, emb // 128, 128) combine of moe_tc_ragged_3d_dispatch quantizes its backward per token.

    The gathered gradient must match the (batch, seq, emb) layout exactly, with one 'rowwise' scale per token.
    """
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 1024), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 1024), dtype=jnp.bfloat16)
    gathered_shapes = []

    def fake_all_gather(v, *a, **k):
      gathered_shapes.append(v.shape)
      return v

    def combine_vjp(v, ct):
      _, vjp = jax.vjp(lambda u: _moe_combine_psum_scatter(u, "expert", 0, False, bwd_method), v)
      return vjp(ct)[0]

    with (
        mock.patch.object(jax.lax, "psum_scatter", side_effect=lambda v, *a, **k: v),
        mock.patch.object(jax.lax, "all_gather", side_effect=fake_all_gather),
    ):
      d_3d = combine_vjp(x, cotangent)
      gathered_shapes.clear()
      d_4d = combine_vjp(x.reshape(2, 4, 8, 128), cotangent.reshape(2, 4, 8, 128))

    self.assertEqual(d_4d.shape, (2, 4, 8, 128))
    np.testing.assert_array_equal(np.asarray(d_4d.reshape(d_3d.shape), np.float32), np.asarray(d_3d, np.float32))
    expected_shapes = [(2, 4, 8, 128)] if scale_shape is None else [scale_shape, (2, 4, 8, 128)]
    self.assertEqual(gathered_shapes, expected_shapes)

  def test_all_gather_quantized_payload_rowwise_requires_token_axis(self):
    x = jnp.ones((2, 4, 8, 128), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "requires gathering along a token axis"):
      _all_gather_quantized_payload(x, "expert", axis=2, tiled=True, method="rowwise", num_feature_axes=2)


class LoadBalanceLossDensityProbTest(unittest.TestCase):
  """RoutedMoE.load_balance_loss with a precomputed per-sequence density_prob."""

  def test_density_prob_matches_probs(self):
    fake = SimpleNamespace(num_experts=8, num_experts_per_tok=2, config=SimpleNamespace(load_balance_loss_weight=0.01))
    logits = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 8), dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    _, top_k_indices = jax.lax.top_k(logits, 2)

    def from_probs(x):
      return moe.RoutedMoE.load_balance_loss(fake, top_k_indices, jax.nn.softmax(x, axis=-1))

    def from_density_prob(x):
      density_prob = jnp.mean(jax.nn.softmax(x, axis=-1), axis=1)
      return moe.RoutedMoE.load_balance_loss(fake, top_k_indices, None, density_prob=density_prob)

    np.testing.assert_allclose(from_density_prob(logits), from_probs(logits), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(from_density_prob)(logits), jax.grad(from_probs)(logits), rtol=1e-5, atol=1e-10)
    # The density_prob of a batch concatenation is the concatenation of per-shard density_probs.
    halves = [jnp.mean(probs[:2], axis=1), jnp.mean(probs[2:], axis=1)]
    np.testing.assert_allclose(
        moe.RoutedMoE.load_balance_loss(fake, top_k_indices, None, density_prob=jnp.concatenate(halves)),
        moe.RoutedMoE.load_balance_loss(fake, top_k_indices, probs),
        rtol=1e-6,
    )


class RequiredRaggedBufferFactorTest(unittest.TestCase):
  """RoutedMoE.required_ragged_buffer_factor: the factor at which the fullest shard's buffer is exactly full."""

  def _required(self, group_sizes, bsz_times_seq_len, num_ep, expert_shard_id):
    fake = SimpleNamespace(config=SimpleNamespace(num_experts=len(group_sizes)), num_experts_per_tok=2, mesh=None)
    return float(
        moe.RoutedMoE.required_ragged_buffer_factor(
            fake, jnp.array(group_sizes, dtype=jnp.int32), bsz_times_seq_len, num_ep, expert_shard_id
        )
    )

  def test_matches_buffer_size_boundary(self):
    # 8 tokens, top-2, EP=2: balanced_size = (8 // 2) * 2 = 8 rows per shard. Shard 1 owns experts 2, 3 with
    # 6 + 6 = 12 tokens, so it needs factor 12 / 8 = 1.5; shard 0 (1 + 1 = 2 tokens) needs 0.25.
    group_sizes = [1, 1, 6, 6]
    self.assertAlmostEqual(self._required(group_sizes, 8, 2, 1), 1.5)
    self.assertAlmostEqual(self._required(group_sizes, 8, 2, 0), 0.25)
    # The required factor is the smallest that keeps get_ragged_buffer_size >= the shard's token count.
    self.assertEqual(moe.RoutedMoE.get_ragged_buffer_size(8, 2, 4, 2, 1.5), 12)
    self.assertLess(moe.RoutedMoE.get_ragged_buffer_size(8, 2, 4, 2, 1.49), 12)


def _flat_fsdp_config(**overrides):
  """Tiny DeepSeek config on the tokamax gmm_v2 fp8 path with the explicit weight all-gather over FSDP."""
  return _tiny_deepseek_config(
      "flat_fsdp_" + "_".join(f"{k}{v}" for k, v in sorted(overrides.items())),
      ici_expert_parallelism=2,
      ici_fsdp_parallelism=jax.device_count() // 2,
      shard_embed_moe_on_fsdp=True,
      use_tokamax_gmm=True,
      use_gmm_v2=True,
      quantization="fp8_full",
      use_qwix_quantization=True,
      weight_quantization_calibration_method="fixed,-224,224",
      act_quantization_calibration_method="fixed,-224,224",
      bwd_quantization_calibration_method="fixed,-1,1",
      # The tiny config sets float32_gate_logits=True, which is rejected with a quantized router projection.
      quantize_router_proj=False,
      # gmm_v2 only quantizes the lhs (which the fixed act scale requires) when the contraction dim is at least the
      # MXU width; with smaller K it dequantizes the rhs instead and rejects the lhs scale.
      base_emb_dim=256,
      base_moe_mlp_dim=256,
      **overrides,
  )


@pytest.mark.tpu_only
class FlatFsdpWeightsGraphTest(unittest.TestCase):
  """RoutedMoE with moe_flat_fsdp_weights builds an abstract model the training path can shard."""

  def test_abstract_model_and_shardings(self):
    if jax.device_count() < 4 or jax.device_count() % 2:
      self.skipTest("Needs an even number (>= 4) of devices for expert parallelism 2 and FSDP >= 2.")
    cfg = _flat_fsdp_config(moe_flat_fsdp_weights=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    # As get_abstract_state_nnx: eval_shape under the axis rules (no global mesh), shardings resolved by MaxText.
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = nnx.eval_shape(
          lambda: moe.RoutedMoE(
              config=cfg,
              num_experts=cfg.num_experts,
              num_experts_per_tok=cfg.num_experts_per_tok,
              mesh=mesh,
              kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
              kernel_axes=("embed", "mlp"),
              intermediate_dim=cfg.moe_mlp_dim,
              dtype=cfg.dtype,
              rngs=nnx.Rngs(0),
          )
      )
      # The qwix fix-up in Linen-wrapped init walks the NNX graph, which descends into the kernel axes.
      self.assertIn(("wi_kernel_axes", 0, 0), [path for path, _ in nnx.iter_graph(model)])
      _, abs_state, _ = nnx.split(model, nnx.Param, ...)
      shardings = sharding.nnx_construct_named_sharding(abs_state, mesh)

    num_experts, emb, mlp = cfg.num_experts, cfg.base_emb_dim, cfg.moe_mlp_dim
    for name, shape in (
        ("wi_0", (num_experts * emb, mlp)),
        ("wi_1", (num_experts * emb, mlp)),
        ("wo", (num_experts * mlp, emb)),
    ):
      self.assertEqual(abs_state[name].shape, shape, name)
      self.assertEqual(shardings[name].get_value().spec[0], ("expert", "fsdp"), name)


@pytest.mark.tpu_only
class FlatFsdpWeightsParityTest(unittest.TestCase):
  """RoutedMoE with flat 2D (expert, fsdp) expert weights matches the 3D layout."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4 or jax.device_count() % 2:
      self.skipTest("Needs an even number (>= 4) of devices for expert parallelism 2 and FSDP >= 2.")

  def _run(self, **overrides):
    """Runs RoutedMoE forward and backward under fp8 gmm rules, so both layouts take the explicit weight QAG.

    The block runs as an NNX module, as in training: Linen-wrapped params are unboxed with Flax's logical-axis
    resolution, which does not know CompoundLogicalAxis.

    Returns:
      (output, lb_loss, grads, params), with grads and params as pure dicts.
    """
    cfg = _flat_fsdp_config(**overrides)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method=cfg.weight_quantization_calibration_method,
        act_calibration_method=cfg.act_quantization_calibration_method,
        bwd_calibration_method=cfg.bwd_quantization_calibration_method,
        op_names=("gmm", "ragged_dot"),
    )
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (cfg.per_device_batch_size * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim)
    )
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      # As create_model: built under the axis rules (no global mesh), shardings resolved by MaxText.
      model = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          rngs=nnx.Rngs(7),
      )
      # Fixed calibrations keep no quant stats, so the eager init call qwix makes is not needed.
      model = qwix.quantize_model(model, qwix.QtProvider([rule]), inputs, skip_nnx_init=True)
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      shardings = sharding.nnx_construct_named_sharding(params, mesh)
      params = jax.tree.map(jax.device_put, params, shardings)

      def loss_fn(params):
        out, lb_loss, _ = nnx.merge(graphdef, params, rest)(inputs)
        return jnp.mean(out**2) + lb_loss, (out, lb_loss)

      with jax.set_mesh(mesh):
        (_, (out, lb_loss)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
    return out, lb_loss, nnx.to_pure_dict(grads), nnx.to_pure_dict(params)

  def test_weights_loss_and_grads_match(self):
    out_ref, lb_ref, grads_ref, params_ref = self._run(moe_flat_fsdp_weights=False)
    out, lb, grads, params = self._run(moe_flat_fsdp_weights=True)

    # Same init: each flat [E * rows, cols] weight is the 3D [E, rows, cols] one viewed as 2D.
    for (path, p), p_ref in zip(
        jax.tree_util.tree_leaves_with_path(params), jax.tree_util.tree_leaves(params_ref), strict=True
    ):
      np.testing.assert_array_equal(np.reshape(p, p_ref.shape), p_ref, err_msg=jax.tree_util.keystr(path))

    np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lb, lb_ref, rtol=1e-6)
    for (path, g), g_ref in zip(
        jax.tree_util.tree_leaves_with_path(grads), jax.tree_util.tree_leaves(grads_ref), strict=True
    ):
      np.testing.assert_allclose(
          np.reshape(g, g_ref.shape), g_ref, rtol=1e-4, atol=1e-6, err_msg=jax.tree_util.keystr(path)
      )


# Meshes exercised by SparseCoreOffloadTest, spelled out so that enabling one
# axis does not leave `ici_fsdp_parallelism` at its `-1` default.
_FSDP = {"ici_fsdp_parallelism": -1, "ici_expert_parallelism": 1}
_EP = {"ici_fsdp_parallelism": 1, "ici_expert_parallelism": -1}
# The ragged gather/reduce kernels partition the hidden dimension across
# SparseCore lanes, so they need a realistically wide embedding to be legal.
_EP_RAGGED = {**_EP, "use_ragged_sort": True, "base_emb_dim": 4096}


@pytest.mark.tpu_only
class SparseCoreOffloadTest(parameterized.TestCase):
  """Tests for `moe_sparse_core_offload_targets`.

  Moving an op to the SparseCore is a scheduling hint, so each target must (a)
  actually annotate ops in the compiled HLO and (b) leave the loss and every
  parameter gradient unchanged. The backward pass is the interesting half: the
  offload is applied to real collectives inside the `sparse_matmul` shard_map,
  and a collective whose transpose rule drops a reduction would still produce a
  correct forward value.
  """

  BASE_CONFIG = {
      "enable_checkpointing": False,
      "model_name": "mixtral-8x7b",
      "override_model_config": True,
      "base_emb_dim": 512,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "dtype": "bfloat16",
      "megablox": True,
      "sparse_matmul": True,
      "per_device_batch_size": 1,
      "max_target_length": 64,
      "float32_gate_logits": True,
  }

  def setUp(self):
    super().setUp()
    if not sparsecore.has_sparse_core():
      self.skipTest("Requires a TPU with a SparseCore (v5p, v6e, tpu7x or newer).")

  def _loss_and_grads(self, run_name, **overrides):
    """Returns `(loss, param_grads, hlo_text)` for one MoE config."""
    cfg = pyconfig.initialize([None, get_test_config_path()], run_name=run_name, **{**self.BASE_CONFIG, **overrides})
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )
    inputs = jax.random.uniform(
        jax.random.PRNGKey(1),
        (int(cfg.per_device_batch_size) * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    def loss_fn(params, x):
      output, load_balance_loss, _ = model.apply({"params": params}, x)
      loss = jnp.mean(output.astype(jnp.float32) ** 2)
      if load_balance_loss is not None:
        loss += load_balance_loss.astype(jnp.float32)
      return loss

    def init():
      return model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      var_shardings = nn.logical_to_mesh_sharding(
          nn.get_partition_spec(jax.eval_shape(init)), mesh, cfg.logical_axis_rules
      )
      variables = jax.jit(init, out_shardings=var_shardings)()
      # Constraining the gradients back to the parameter sharding mirrors a real
      # train step, which is what makes the weight-gradient collectives appear.
      step = jax.jit(jax.value_and_grad(loss_fn), out_shardings=(None, var_shardings["params"]))
      hlo_text = step.lower(variables["params"], inputs).compile().as_text()
      loss, grads = jax.block_until_ready(step(variables["params"], inputs))
    return float(loss), grads, hlo_text

  @parameterized.named_parameters(
      ("fsdp_all_gather", _FSDP, sparsecore.FSDP_ALL_GATHER),
      ("ep_collectives", _EP, sparsecore.EP_COLLECTIVES),
      ("ragged_sort", _EP_RAGGED, sparsecore.RAGGED_SORT),
      ("all_targets", _EP_RAGGED, "all"),
  )
  def test_offload_is_numerically_transparent(self, parallelism, targets):
    """Enabling a target annotates the HLO without changing loss or gradients."""
    ref_loss, ref_grads, ref_hlo = self._loss_and_grads(
        f"sc_offload_ref_{targets}", moe_sparse_core_offload_targets="", **parallelism
    )
    loss, grads, hlo = self._loss_and_grads(
        f"sc_offload_{targets}", moe_sparse_core_offload_targets=targets, **parallelism
    )

    annotation = '_xla_compute_type="sparseoffload"'
    self.assertEqual(ref_hlo.count(annotation), 0, "The baseline must not offload anything to the SparseCore.")
    self.assertGreater(hlo.count(annotation), 0, f"Offload target {targets!r} annotated no ops.")

    self.assertAlmostEqual(loss, ref_loss, places=6)
    diff_summary = compare_tree(ref_grads, grads, relative_norm_diff_threshold=1e-5)
    max_logging.log("\n" + diff_summary)

  def test_disabled_offload_leaves_the_hlo_untouched(self):
    """Backward compatibility: the default config compiles to the same HLO as before."""
    _, _, hlo = self._loss_and_grads("sc_offload_disabled", ici_fsdp_parallelism=-1)
    self.assertEqual(hlo.count('_xla_compute_type="sparseoffload"'), 0)


if __name__ == "__main__":
  absltest.main()
