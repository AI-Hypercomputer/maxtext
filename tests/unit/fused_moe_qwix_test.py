# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the qwix boundary around tpu-inference's fused MoE kernel."""

import sys
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import qwix.pallas as qpl

from maxtext.common import common_types as ctypes
from maxtext.layers import moe
from maxtext.layers import quantizations


def _assert_error_within(actual, ref, bound):
  """Every |actual - ref| stays under `bound`, which broadcasts against the arrays."""
  err = np.abs(np.asarray(actual - ref))
  np.testing.assert_array_less(err, np.broadcast_to(np.asarray(bound) + 1e-6, err.shape))


def _fp8_rule(**kwargs):
  return qwix.QtRule(
      module_path=".*",
      weight_qtype=jnp.float8_e4m3fn,
      act_qtype=jnp.float8_e4m3fn,
      op_names=("dot_general", "gmm", "ragged_dot"),
      **kwargs,
  )


class QuantizeWeightForFusedMoeTest(unittest.TestCase):
  """quantize_weight_for_fused_moe produces what the fused kernel takes."""

  def setUp(self):
    super().setUp()
    self.num_experts, self.in_dim, self.out_dim = 4, 256, 128
    self.kernel = jax.random.normal(jax.random.key(0), (self.num_experts, self.in_dim, self.out_dim), jnp.bfloat16)

  def test_no_rule_keeps_weight(self):
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, None)
    self.assertIs(weight, self.kernel)
    self.assertIsNone(scale)

  def test_rule_without_weight_qtype_keeps_weight(self):
    rule = qwix.QtRule(module_path=".*", act_qtype=jnp.float8_e4m3fn)
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, rule)
    self.assertIs(weight, self.kernel)
    self.assertIsNone(scale)

  def test_unsupported_qtype_keeps_weight(self):
    rule = qwix.QtRule(module_path=".*", weight_qtype=jnp.float8_e5m2)
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, rule)
    self.assertIs(weight, self.kernel)
    self.assertIsNone(scale)

  def test_fp8_channelwise(self):
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, _fp8_rule())
    self.assertEqual(weight.dtype, jnp.float8_e4m3fn)
    self.assertEqual(weight.shape, self.kernel.shape)
    # one scale per (expert, output column), laid out as the kernel expects
    self.assertEqual(scale.shape, (self.num_experts, 1, 1, self.out_dim))
    self.assertEqual(scale.dtype, jnp.float32)
    dequantized = weight.astype(jnp.float32) * scale[:, 0]
    ref = self.kernel.astype(jnp.float32)
    # fp8 e4m3 keeps 3 mantissa bits: the error is bounded by half an ulp of the column max
    col_max = jnp.max(jnp.abs(ref), axis=1, keepdims=True)
    _assert_error_within(dequantized, ref, col_max / 16)

  def test_fp8_blockwise_from_tile_size(self):
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, _fp8_rule(tile_size=64))
    self.assertEqual(weight.dtype, jnp.float8_e4m3fn)
    self.assertEqual(scale.shape, (self.num_experts, self.in_dim // 64, 1, self.out_dim))
    blocks = weight.astype(jnp.float32).reshape(self.num_experts, self.in_dim // 64, 64, self.out_dim)
    dequantized = (blocks * scale).reshape(self.kernel.shape)
    ref = self.kernel.astype(jnp.float32)
    block_max = jnp.max(jnp.abs(ref.reshape(blocks.shape)), axis=2, keepdims=True)
    _assert_error_within(dequantized.reshape(blocks.shape), ref.reshape(blocks.shape), block_max / 16)

  def test_int8_channelwise(self):
    rule = qwix.QtRule(module_path=".*", weight_qtype=jnp.int8, op_names=("gmm",))
    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, rule)
    self.assertEqual(weight.dtype, jnp.int8)
    self.assertEqual(scale.shape, (self.num_experts, 1, 1, self.out_dim))
    dequantized = weight.astype(jnp.float32) * scale[:, 0]
    ref = self.kernel.astype(jnp.float32)
    col_max = jnp.max(jnp.abs(ref), axis=1, keepdims=True)
    _assert_error_within(dequantized, ref, col_max / 127)

  def test_rejects_non_3d_weight(self):
    with self.assertRaises(ValueError):
      quantizations.quantize_weight_for_fused_moe(self.kernel[0], _fp8_rule())


class WithoutQwixInterceptionTest(unittest.TestCase):
  """The boundary hides the qwix rule (and its op rewriting) from whatever runs inside."""

  def test_rule_visible_outside_and_hidden_inside(self):
    seen = {}

    class Layer(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((32, 64), jnp.bfloat16))

      def __call__(self, x):
        seen["outside"] = quantizations.get_fused_moe_rule()

        def opaque(y):
          seen["inside"] = quantizations.get_fused_moe_rule()
          return jnp.dot(y, self.w[...], preferred_element_type=jnp.float32)

        out = quantizations.without_qwix_interception(opaque)(x)
        seen["after"] = quantizations.get_fused_moe_rule()
        return out

    x = jax.random.normal(jax.random.key(0), (8, 32), jnp.bfloat16)
    layer = qwix.quantize_model(Layer(), qwix.QtProvider([_fp8_rule()]), x)
    jaxpr = str(jax.make_jaxpr(layer)(x))

    self.assertIsNotNone(seen["outside"])
    self.assertIsNone(seen["inside"])
    self.assertIsNotNone(seen["after"])
    # the dot inside the boundary is left alone: no fp8 casts anywhere in the trace
    self.assertNotIn("float8", jaxpr)


class FusedMoeMatmulTest(unittest.TestCase):
  """RoutedMoE.fused_moe_matmul hands the kernel pre-quantized weights, outside qwix."""

  num_experts, top_k, emb_dim, mlp_dim, tokens = 4, 2, 64, 128, 16

  def _fake_tpu_inference(self, calls):
    """Stand-in for the tpu-inference modules fused_moe_matmul imports."""

    def fused_moe_func(**kwargs):
      calls.append(dict(kwargs, rule_inside=quantizations.get_fused_moe_rule()))
      return jnp.zeros((kwargs["hidden_states"].shape[0], self.emb_dim), jnp.bfloat16)

    envs = types.SimpleNamespace(
        ENABLE_RS_KERNEL=False, USE_GMM_FUSED_RS_KERNEL=False, ONEHOT_MOE_PERMUTE_THRESHOLD=0, VLLM_MOE_CHUNK_SIZE=0
    )
    pkg = types.ModuleType("tpu_inference")
    pkg.envs = envs
    layers = types.ModuleType("tpu_inference.layers")
    common = types.ModuleType("tpu_inference.layers.common")
    gmm = types.ModuleType("tpu_inference.layers.common.fused_moe_gmm")
    gmm.fused_moe_func = fused_moe_func
    return {
        "tpu_inference": pkg,
        "tpu_inference.envs": envs,
        "tpu_inference.layers": layers,
        "tpu_inference.layers.common": common,
        "tpu_inference.layers.common.fused_moe_gmm": gmm,
    }

  def _make_layer(self, calls):
    test = self
    config = types.SimpleNamespace(
        mlp_activations=("silu",),
        routed_score_func="softmax",
        norm_topk_prob=True,
        decoder_block=ctypes.DecoderBlockType.MIXTRAL,
    )

    class Layer(nnx.Module):
      """Drives RoutedMoE.fused_moe_matmul with just the attributes it reads."""

      def __init__(self):
        keys = jax.random.split(jax.random.key(0), 3)
        self.w0 = nnx.Param(jax.random.normal(keys[0], (test.num_experts, test.emb_dim, test.mlp_dim), jnp.bfloat16))
        self.w1 = nnx.Param(jax.random.normal(keys[1], (test.num_experts, test.emb_dim, test.mlp_dim), jnp.bfloat16))
        self.wo = nnx.Param(jax.random.normal(keys[2], (test.num_experts, test.mlp_dim, test.emb_dim), jnp.bfloat16))
        self.config = config
        self.num_experts = test.num_experts
        self.num_experts_per_tok = test.top_k
        self.mesh = None

      def get_expert_parallelism_size(self):
        return 1

      def __call__(self, inputs, gate_logits):
        with mock.patch.dict(sys.modules, test._fake_tpu_inference(calls)):
          out, _, _ = moe.RoutedMoE.fused_moe_matmul(self, inputs, gate_logits, self.wo[...], self.w0[...], self.w1[...])
        return out

    return Layer()

  def _inputs(self):
    inputs = jax.random.normal(jax.random.key(1), (1, self.tokens, self.emb_dim), jnp.bfloat16)
    gate_logits = jax.random.normal(jax.random.key(2), (1, self.tokens, self.num_experts), jnp.float32)
    return inputs, gate_logits

  def test_unquantized_without_qwix(self):
    calls = []
    inputs, gate_logits = self._inputs()
    self._make_layer(calls)(inputs, gate_logits)
    (call,) = calls
    self.assertEqual(call["w1"].dtype, jnp.bfloat16)
    self.assertEqual(call["w1"].shape, (self.num_experts, self.emb_dim, 2 * self.mlp_dim))
    self.assertIsNone(call["w1_scale"])
    self.assertIsNone(call["w2_scale"])

  def test_fp8_rule_prequantizes_weights_outside_qwix(self):
    calls = []
    inputs, gate_logits = self._inputs()
    layer = qwix.quantize_model(self._make_layer(calls), qwix.QtProvider([_fp8_rule()]), inputs, gate_logits)
    calls.clear()
    layer(inputs, gate_logits)
    (call,) = calls
    # weights arrive quantized, with scales in the kernel's [E, blocks, 1, N] layout
    self.assertEqual(call["w1"].dtype, jnp.float8_e4m3fn)
    self.assertEqual(call["w2"].dtype, jnp.float8_e4m3fn)
    self.assertEqual(call["w1_scale"].shape, (self.num_experts, 1, 1, 2 * self.mlp_dim))
    self.assertEqual(call["w2_scale"].shape, (self.num_experts, 1, 1, self.emb_dim))
    # and the kernel itself runs outside qwix's interception
    self.assertIsNone(call["rule_inside"])

  def test_dot_general_only_rule_keeps_experts_unquantized(self):
    calls = []
    inputs, gate_logits = self._inputs()
    rule = qwix.QtRule(
        module_path=".*", weight_qtype=jnp.float8_e4m3fn, act_qtype=jnp.float8_e4m3fn, op_names=("dot_general",)
    )
    layer = qwix.quantize_model(self._make_layer(calls), qwix.QtProvider([rule]), inputs, gate_logits)
    calls.clear()
    layer(inputs, gate_logits)
    (call,) = calls
    self.assertEqual(call["w1"].dtype, jnp.bfloat16)
    self.assertIsNone(call["w1_scale"])
    self.assertIsNone(call["rule_inside"])


if __name__ == "__main__":
  unittest.main()
