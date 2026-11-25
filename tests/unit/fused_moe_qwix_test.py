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

# pylint: disable=missing-class-docstring,missing-function-docstring,unbalanced-tuple-unpacking

"""Tests for RoutedMoE.fused_moe_matmul (tpu-inference's fused MoE kernel) and the qwix boundary around it."""

import sys
import types
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest
import qwix

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.integration.vllm.convert_utils import compute_padded_moe_mlp_dim
from maxtext.kernels.megablox.pallas_mosaic_tpu_v2_gmm_kernel import swigluoai
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

# A GPT-OSS MoE small enough for CPU. Its MLP dim, like GPT-OSS's 2880, needs padding for GMM_v2.
_GPT_OSS_EMB_DIM, _GPT_OSS_MLP_DIM, _GPT_OSS_NUM_EXPERTS, _GPT_OSS_TOP_K, _GPT_OSS_TOKENS = 64, 96, 4, 2, 8

# Mesh axes tpu-inference shards the fused kernel's MLP dim over in tensor-parallel mode
# (ShardingAxisNameBase.MLP_TENSOR in tpu_inference/layers/common/sharding.py).
_TPU_INFERENCE_MLP_TENSOR_AXES = ("attn_dp", "attn_dp_expert", "expert", "model", "dcp", "pcp")


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

  def test_rule_with_missing_attributes(self):
    """getattr fallback handles rules missing tile_size or calibration_method."""

    class MinimalRule:
      weight_qtype = jnp.float8_e4m3fn

    weight, scale = quantizations.quantize_weight_for_fused_moe(self.kernel, MinimalRule())
    self.assertEqual(weight.dtype, jnp.float8_e4m3fn)
    self.assertEqual(scale.shape, (self.num_experts, 1, 1, self.out_dim))


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


def _fake_tpu_inference(fused_moe_func):
  """Stand-in for the tpu-inference modules fused_moe_matmul imports, with `fused_moe_func` as the kernel."""
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


def _emulated_fused_moe_func(
    *,
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    gating_output,
    topk,
    renormalize,
    mesh,
    use_ep,
    activation,
    scoring_fn,
    **unused_kwargs,
):
  """Pure-JAX stand-in for tpu-inference's unquantized `fused_moe_func`.

  It routes like the kernel (score function over all experts, top-k, renormalization) and, in
  tensor-parallel mode, gives every MLP shard its contiguous chunk of `w1` and `w1_bias`, whose first
  half is the gate and second half the up projection: the layout gmm_v2 reads.
  """
  del unused_kwargs
  assert w1_scale is None and w2_scale is None
  if activation != "swigluoai" or scoring_fn != "softmax":
    raise NotImplementedError(f"Unsupported {activation=}, {scoring_fn=}.")
  weights, experts = jax.lax.top_k(jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1), topk)
  if renormalize:
    weights /= weights.sum(axis=-1, keepdims=True)

  num_shards = 1 if use_ep else int(np.prod([mesh.shape.get(axis, 1) for axis in _TPU_INFERENCE_MLP_TENSOR_AXES]))
  chunk = w1.shape[-1] // num_shards
  x = hidden_states.astype(jnp.float32)
  expert_out = 0.0
  for shard in range(num_shards):
    cols = slice(shard * chunk, (shard + 1) * chunk)
    h = jnp.einsum("td,edn->ten", x, w1[..., cols].astype(jnp.float32))
    if w1_bias is not None:
      h += w1_bias[:, 0, cols]
    act = swigluoai(h[..., : chunk // 2], h[..., chunk // 2 :])
    rows = slice(shard * chunk // 2, (shard + 1) * chunk // 2)
    expert_out += jnp.einsum("ten,end->ted", act, w2[:, rows].astype(jnp.float32))
  if w2_bias is not None:
    expert_out += w2_bias[:, 0]
  selected = jnp.take_along_axis(expert_out, experts[..., None], axis=1)
  return jnp.einsum("tkd,tk->td", selected, weights).astype(hidden_states.dtype)


def _make_moe(cfg, mesh):
  """A RoutedMoE built like GPT-OSS's decoder layer builds it."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    return moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )


def _mesh(cfg, **axis_sizes):
  """A mesh over the first devices with `axis_sizes` (e.g. model=2) and every other axis of size 1."""
  shape = [axis_sizes.get(name, 1) for name in cfg.mesh_axes]
  return Mesh(np.asarray(jax.devices()[: int(np.prod(shape))]).reshape(shape), cfg.mesh_axes)


def _randomize_biases(layer, seed):
  """The default bias init is zeros, which would not test where the biases end up."""
  keys = iter(jax.random.split(jax.random.key(seed), 4))
  for bias in (layer.wi_0_bias, layer.wi_1_bias, layer.wo_bias, layer.gate.bias):
    value = bias[...]
    bias.set_value(jax.device_put(jax.random.normal(next(keys), value.shape, value.dtype), value.sharding))


def _load_like_vllm(src, dst, dst_cfg, dst_mesh):
  """Loads the unpadded expert weights of `src` into the padded vLLM layer `dst` like the vLLM loader.

  That is `model_creation_utils.from_pretrained`: it restores the checkpoint sharded like the model,
  then `_fuse_moe_weights` builds a prefused `wi`, one [gate | up] chunk per MLP shard, and
  `_align_checkpoint_to_model_shapes` zero-pads every shard of `wi_0` / `wi_1` / `wo` along the MLP dim.
  The biases keep their unpadded shape.
  """
  with jax.set_mesh(dst_mesh), nn_partitioning.axis_rules(dst_cfg.logical_axis_rules):

    def sharding(logical_axes):
      return NamedSharding(dst_mesh, dst._logical_to_mesh_axes(logical_axes))  # pylint: disable=protected-access

    def target(name, logical_axes):
      value = getattr(dst, name)[...]
      return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding(logical_axes))

    def restore(name, logical_axes):
      # Like a checkpoint saved in the serving weight dtype.
      return jax.device_put(getattr(src, name)[...].astype(dst.wo[...].dtype), sharding(logical_axes))

    def align(name, logical_axes):
      return model_creation_utils._align_checkpoint_to_model_shapes(  # pylint: disable=protected-access
          restore(name, logical_axes), target(name, logical_axes), logical_axes
      )

    if dst_cfg.prefuse_moe_weights:
      checkpoint = {"wi_0": restore("wi_0", dst.wi_kernel_axes), "wi_1": restore("wi_1", dst.wi_kernel_axes)}
      dst.wi.set_value(
          model_creation_utils._fuse_moe_weights(  # pylint: disable=protected-access
              checkpoint, {"wi": target("wi", dst.wi_kernel_axes)}, dst_cfg
          )["wi"]
      )
    else:
      dst.wi_0.set_value(align("wi_0", dst.wi_kernel_axes))
      dst.wi_1.set_value(align("wi_1", dst.wi_kernel_axes))
    dst.wo.set_value(align("wo", dst.wo_kernel_axes))

  def copy(src_variable, dst_variable):
    value = src_variable[...].astype(dst_variable[...].dtype)
    dst_variable.set_value(jax.device_put(value, NamedSharding(dst_mesh, P())))

  for name in ("wi_0_bias", "wi_1_bias", "wo_bias"):
    copy(getattr(src, name), getattr(dst, name))
  copy(src.gate.kernel, dst.gate.kernel)
  copy(src.gate.bias, dst.gate.bias)


class FusedMoeMatmulTest(parameterized.TestCase):
  """RoutedMoE.fused_moe_matmul hands the kernel pre-quantized weights, outside qwix."""

  num_experts, top_k, emb_dim, mlp_dim, tokens = 4, 2, 64, 128, 16

  def _recording_tpu_inference(self, calls):
    """tpu-inference stand-in whose kernel records its arguments (and the qwix rule it sees) in `calls`."""

    def fused_moe_func(**kwargs):
      calls.append(dict(kwargs, rule_inside=quantizations.get_fused_moe_rule()))
      return jnp.zeros((kwargs["hidden_states"].shape[0], self.emb_dim), jnp.bfloat16)

    return _fake_tpu_inference(fused_moe_func)

  def _make_layer(self, calls, return_routed_experts=False, mesh=None):
    test = self
    config = types.SimpleNamespace(
        mlp_activations=("silu",),
        routed_score_func="softmax",
        norm_topk_prob=True,
        decoder_block=ctypes.DecoderBlockType.MIXTRAL,
        return_routed_experts=return_routed_experts,
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
        self.mesh = mesh

      def get_expert_parallelism_size(self):
        return 1

      def __call__(self, inputs, gate_logits):
        with mock.patch.dict(sys.modules, test._recording_tpu_inference(calls)):
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
    self.assertFalse(call["scatter_results"])

  def test_return_routed_experts_records_selected_experts(self):
    calls = []
    inputs, gate_logits = self._inputs()
    layer = self._make_layer(calls, return_routed_experts=True)
    layer(inputs, gate_logits)
    self.assertTrue(hasattr(layer, "selected_experts"))
    self.assertEqual(layer.selected_experts[...].shape, (self.tokens, self.top_k))

  def test_routing_is_not_recorded_by_default(self):
    """Serving pays for neither the duplicate top_k nor the intermediate it feeds."""
    calls = []
    inputs, gate_logits = self._inputs()
    layer = self._make_layer(calls)
    layer(inputs, gate_logits)
    self.assertFalse(hasattr(layer, "selected_experts"))
    # the kernel does its own routing, so the layer's trace holds no second top_k
    jaxpr = str(jax.make_jaxpr(layer)(inputs, gate_logits))
    self.assertNotIn("top_k", jaxpr)

  def test_scatter_results_reflects_dp_mesh_size(self):
    """Any data-parallel attention axis above size 1 makes the kernel reduce-scatter."""
    inputs, gate_logits = self._inputs()
    # each DP axis on its own, then together, against a mesh that declares none of them
    cases = [
        ({"data": 1, "attn_dp": 1, "attn_dp_expert": 1}, False),
        ({"expert": 8, "model": 4}, False),
        ({"data": 2, "attn_dp": 1, "attn_dp_expert": 1}, True),
        ({"data": 1, "attn_dp": 2, "attn_dp_expert": 1}, True),
        ({"data": 1, "attn_dp": 1, "attn_dp_expert": 4}, True),
        ({"data": 2, "attn_dp": 2, "attn_dp_expert": 2, "model": 8}, True),
    ]
    for shape, expected in cases:
      with self.subTest(shape=shape):
        calls = []
        mesh = types.SimpleNamespace(shape=shape)
        self._make_layer(calls, mesh=mesh)(inputs, gate_logits)
        self.assertEqual(calls[0]["scatter_results"], expected)

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

  # GPT-OSS needs four things from this path: its clamped SwiGLU ("swigluoai"), its expert biases, its MLP dim
  # (2880) padded so every tensor-parallel shard satisfies the GMM_v2 kernel's alignment, and valid shardings for
  # those biases under vLLM's logical axis rules.

  def _model_config(self, model_name, config_file="base.yml", **overrides):
    return pyconfig.initialize(
        [None, get_test_config_path(config_file)],
        run_name="fused_moe_matmul_test",
        model_name=model_name,
        enable_checkpointing=False,
        log_config=False,
        **overrides,
    )

  def _small_gpt_oss_config(self, config_file, **overrides):
    return self._model_config(
        "gpt-oss-20b",
        config_file,
        override_model_config=True,
        base_emb_dim=_GPT_OSS_EMB_DIM,
        base_mlp_dim=_GPT_OSS_MLP_DIM,
        base_moe_mlp_dim=_GPT_OSS_MLP_DIM,
        num_experts=_GPT_OSS_NUM_EXPERTS,
        num_experts_per_tok=_GPT_OSS_TOP_K,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        max_target_length=_GPT_OSS_TOKENS,
        per_device_batch_size=1,
        **overrides,
    )

  def test_fused_moe_activation_gpt_oss_uses_swigluoai(self):
    # mlp_activations[0] is "sigmoid", which the kernel rejects.
    self.assertEqual(moe.fused_moe_activation(self._model_config("gpt-oss-20b")), "swigluoai")

  def test_fused_moe_activation_other_models_keep_their_activation(self):
    self.assertEqual(moe.fused_moe_activation(self._model_config("mixtral-8x7b")), "silu")

  def test_fused_moe_activation_rejects_another_gpt_oss_clamp_limit(self):
    # The kernel hardcodes GPT-OSS's limit, so a different one cannot silently run with 7.0.
    cfg = self._model_config("gpt-oss-20b", override_model_config=True, mlp_activations_limit=10.0)
    with self.assertRaisesRegex(NotImplementedError, "mlp_activations_limit"):
      moe.fused_moe_activation(cfg)

  def _gate_and_up_biases(self, num_experts=2, mlp_dim=12):
    gate = jnp.arange(num_experts * mlp_dim, dtype=jnp.float32).reshape(num_experts, mlp_dim)
    return gate + 1.0, -(gate + 1.0)

  @parameterized.parameters((1, 12), (1, 16), (2, 16), (4, 16), (4, 24))
  def test_fuse_moe_biases_per_shard_layout_matches_fused_kernel(self, num_shards, padded_mlp_dim):
    """A bias is a kernel with one input row, so `_fuse_moe_weights` must produce the same layout."""
    gate, up = self._gate_and_up_biases()
    mesh = jax.sharding.AbstractMesh((num_shards,), ("model",))
    target = jax.ShapeDtypeStruct(
        (gate.shape[0], 1, 2 * padded_mlp_dim), jnp.float32, sharding=NamedSharding(mesh, P(None, None, "model"))
    )
    expected = model_creation_utils._fuse_moe_weights(  # pylint: disable=protected-access
        {"wi_0": gate[:, None, :], "wi_1": up[:, None, :]}, {"wi": target}, types.SimpleNamespace(attention="vllm_rpa")
    )["wi"]

    actual = moe.fuse_moe_biases_for_kernel(gate, up, 2 * padded_mlp_dim, num_shards, per_shard_layout=True)

    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_array_equal(actual, expected)

  def test_fuse_moe_biases_concatenated_layout(self):
    """Unfused `wi_0` / `wi_1` (each padded per shard) are concatenated whole by `fused_moe_matmul`."""
    gate, up = self._gate_and_up_biases()
    actual = moe.fuse_moe_biases_for_kernel(gate, up, 32, num_shards=2, per_shard_layout=False)

    def pad_shards(bias):
      return np.concatenate([np.pad(shard, ((0, 0), (0, 2))) for shard in np.split(np.asarray(bias), 2, axis=-1)], -1)

    expected = np.concatenate([pad_shards(gate), pad_shards(up)], axis=-1)[:, None, :]
    np.testing.assert_array_equal(actual, expected)

  def test_fuse_moe_biases_rejects_unshardable_dims(self):
    gate, up = self._gate_and_up_biases()
    with self.assertRaises(ValueError):
      moe.fuse_moe_biases_for_kernel(gate, up, 32, num_shards=5, per_shard_layout=True)

  @parameterized.named_parameters(
      ("unfused", 1, 1, False),
      ("prefused", 1, 1, True),
      ("prefused_tp2", 2, 1, True),
      ("prefused_tp4", 4, 1, True),
      # The expert-parallel kernel reads gate/up as two whole halves, which is how unfused wi_0/wi_1 arrive.
      ("unfused_tp2_ep2", 2, 2, False),
  )
  def test_gpt_oss_matches_dense(self, tp, ep, prefuse):
    """GPT-OSS through `fused_moe_matmul` matches the dense MaxText implementation (the kernel is emulated)."""
    if jax.device_count() < tp * ep:
      self.skipTest(f"Needs {tp * ep} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={tp * ep}).")
    dense_cfg = self._small_gpt_oss_config("base.yml", sparse_matmul=False, megablox=False)
    dense = _make_moe(dense_cfg, _mesh(dense_cfg, fsdp=1))
    _randomize_biases(dense, seed=1)

    # What the vLLM adapter sets up for GPT-OSS: the MLP dim is padded for GMM_v2 on every shard.
    padded_mlp_dim = compute_padded_moe_mlp_dim(_GPT_OSS_MLP_DIM, tp)
    self.assertGreater(padded_mlp_dim, _GPT_OSS_MLP_DIM)
    fused_cfg = self._small_gpt_oss_config(
        "inference/vllm.yml",
        ici_tensor_parallelism=tp,
        ici_expert_parallelism=ep,
        padded_base_moe_mlp_dim=padded_mlp_dim,
        prefuse_moe_weights=prefuse,
    )
    fused_mesh = _mesh(fused_cfg, model=tp, expert=ep)
    fused = _make_moe(fused_cfg, fused_mesh)
    _load_like_vllm(dense, fused, fused_cfg, fused_mesh)

    inputs = jax.random.normal(jax.random.key(2), (1, _GPT_OSS_TOKENS, _GPT_OSS_EMB_DIM), jnp.float32)
    with nn_partitioning.axis_rules(dense_cfg.logical_axis_rules):
      expected, _, _ = dense(inputs)
    calls = []

    def fused_moe_func(**kwargs):
      calls.append(kwargs)
      return _emulated_fused_moe_func(**kwargs)

    with (
        mock.patch.dict(sys.modules, _fake_tpu_inference(fused_moe_func)),
        jax.set_mesh(fused_mesh),
        nn_partitioning.axis_rules(fused_cfg.logical_axis_rules),
    ):
      actual, lb_loss, bias_updates = fused(jax.device_put(inputs, NamedSharding(fused_mesh, P())))

    self.assertLen(calls, 1)
    call = calls[0]
    self.assertEqual(call["activation"], "swigluoai")
    self.assertEqual(call["w1_bias"].shape, (_GPT_OSS_NUM_EXPERTS, 1, 2 * padded_mlp_dim))
    self.assertEqual(call["w2_bias"].shape, (_GPT_OSS_NUM_EXPERTS, 1, _GPT_OSS_EMB_DIM))
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_gpt_oss_vllm_model_shards_biases_like_kernels(self):
    """The GPT-OSS model builds under vLLM's logical axis rules, as `MaxTextForCausalLM` builds it."""
    cfg = self._model_config(
        "gpt-oss-20b",
        "inference/vllm.yml",
        override_model_config=True,
        base_emb_dim=_GPT_OSS_EMB_DIM,
        base_mlp_dim=_GPT_OSS_MLP_DIM,
        base_moe_mlp_dim=_GPT_OSS_MLP_DIM,
        base_num_query_heads=8,
        base_num_kv_heads=2,
        head_dim=16,
        base_num_decoder_layers=2,
        vocab_size=512,
        num_experts=_GPT_OSS_NUM_EXPERTS,
        num_experts_per_tok=_GPT_OSS_TOP_K,
        padded_base_moe_mlp_dim=256,
        prefuse_moe_weights=True,
        max_target_length=_GPT_OSS_TOKENS,
        per_device_batch_size=1,
    )
    mesh = _mesh(cfg)
    # tpu-inference creates the model under its mesh, so every Variable's sharding becomes a NamedSharding.
    # That rejects a spec that maps one mesh axis to two dimensions, which vLLM's rules produced for the
    # expert biases' ("exp", "activation_mlp").
    with jax.set_mesh(mesh):
      _, abstract_model = model_creation_utils.create_nnx_abstract_model(
          cfg, mesh, model_mode=ctypes.MODEL_MODE_AUTOREGRESSIVE
      )

    moe_block = abstract_model.decoder.layers_0.GptOssMlp
    wi_spec = moe_block.wi.get_value().sharding.spec
    for bias in (moe_block.wi_0_bias, moe_block.wi_1_bias):
      # Sharded like the expert kernels: experts, then the MLP dim.
      self.assertEqual(tuple(bias.get_value().sharding.spec), (wi_spec[0], wi_spec[-1]))


@pytest.mark.tpu_only
@pytest.mark.post_training
class GptOssFusedMoeTPUTest(parameterized.TestCase):
  """GPT-OSS through the real tpu-inference kernel matches the dense MaxText implementation."""

  def setUp(self):
    super().setUp()
    try:
      import tpu_inference  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
      self.skipTest("Fused MoE tests require tpu-inference package to be installed.")

  def _config(self, config_file, **overrides):
    # A smaller GPT-OSS MoE. Its MLP dim (360, an eighth of 2880) still needs padding on every shard.
    return pyconfig.initialize(
        [None, get_test_config_path(config_file)],
        run_name="gpt_oss_fused_moe_tpu",
        model_name="gpt-oss-20b",
        override_model_config=True,
        base_emb_dim=256,
        base_mlp_dim=360,
        base_moe_mlp_dim=360,
        num_experts=16,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        enable_checkpointing=False,
        log_config=False,
        max_target_length=16,
        per_device_batch_size=1,
        **overrides,
    )

  @parameterized.named_parameters(("expert_parallel", "expert", False), ("tensor_parallel", "model", True))
  def test_matches_dense(self, axis, prefuse):
    num_devices = jax.device_count()
    dense_cfg = self._config("base.yml", sparse_matmul=False, megablox=False)
    dense = _make_moe(dense_cfg, _mesh(dense_cfg, fsdp=1))
    _randomize_biases(dense, seed=1)

    tp = num_devices if axis == "model" else 1
    fused_cfg = self._config(
        "inference/vllm.yml",
        ici_tensor_parallelism=tp,
        ici_expert_parallelism=num_devices // tp,
        padded_base_moe_mlp_dim=compute_padded_moe_mlp_dim(dense_cfg.mlp_dim, tp),
        prefuse_moe_weights=prefuse,
    )
    fused_mesh = _mesh(fused_cfg, **{axis: num_devices})
    fused = _make_moe(fused_cfg, fused_mesh)
    _load_like_vllm(dense, fused, fused_cfg, fused_mesh)

    inputs = jax.random.normal(jax.random.key(2), (1, 16, dense_cfg.emb_dim), jnp.bfloat16)
    with nn_partitioning.axis_rules(dense_cfg.logical_axis_rules):
      expected, _, _ = dense(inputs)
    with jax.set_mesh(fused_mesh), nn_partitioning.axis_rules(fused_cfg.logical_axis_rules):
      actual, _, _ = fused(jax.device_put(inputs, NamedSharding(fused_mesh, P())))

    # bf16 everywhere: compare the whole output rather than element by element. A missing bias or the
    # wrong activation is off by O(1) relative error.
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-2)


if __name__ == "__main__":
  unittest.main()
