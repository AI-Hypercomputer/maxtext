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

"""Tests GPT-OSS MoE on the fused tpu-inference kernel path (attention=vllm_rpa).

GPT-OSS needs four things from that path: its clamped SwiGLU ("swigluoai"), its expert biases, its MLP
dim (2880) padded so every tensor-parallel shard satisfies the GMM_v2 kernel's alignment, and valid
shardings for those biases under vLLM's logical axis rules.
"""

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

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.convert_utils import compute_padded_moe_mlp_dim
from maxtext.kernels.megablox.pallas_mosaic_tpu_v2_gmm_kernel import swigluoai
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

# Mesh axes tpu-inference shards the fused kernel's MLP dim over in tensor-parallel mode
# (ShardingAxisNameBase.MLP_TENSOR in tpu_inference/layers/common/sharding.py).
_TPU_INFERENCE_MLP_TENSOR_AXES = ("attn_dp", "attn_dp_expert", "expert", "model", "dcp", "pcp")


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


def _fake_tpu_inference(fused_moe_func):
  """Stand-in for the tpu-inference modules `fused_moe_matmul` imports."""
  envs = types.SimpleNamespace(
      ENABLE_RS_KERNEL=False, USE_GMM_FUSED_RS_KERNEL=False, ONEHOT_MOE_PERMUTE_THRESHOLD=0, VLLM_MOE_CHUNK_SIZE=0
  )
  pkg = types.ModuleType("tpu_inference")
  pkg.envs = envs
  gmm = types.ModuleType("tpu_inference.layers.common.fused_moe_gmm")
  gmm.fused_moe_func = fused_moe_func
  return {
      "tpu_inference": pkg,
      "tpu_inference.envs": envs,
      "tpu_inference.layers": types.ModuleType("tpu_inference.layers"),
      "tpu_inference.layers.common": types.ModuleType("tpu_inference.layers.common"),
      "tpu_inference.layers.common.fused_moe_gmm": gmm,
  }


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
              checkpoint, {"wi": target("wi", dst.wi_kernel_axes)}
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


class FusedMoeActivationTest(unittest.TestCase):
  """`fused_moe_activation` names the kernel activation equivalent to `apply_ffn_activation`."""

  def _config(self, **overrides):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="fused_moe_activation", enable_checkpointing=False, **overrides
    )

  def test_gpt_oss_uses_swigluoai(self):
    # mlp_activations[0] is "sigmoid", which the kernel rejects.
    self.assertEqual(moe.fused_moe_activation(self._config(model_name="gpt-oss-20b")), "swigluoai")

  def test_other_models_keep_their_activation(self):
    self.assertEqual(moe.fused_moe_activation(self._config(model_name="mixtral-8x7b")), "silu")

  def test_gpt_oss_with_another_clamp_limit_is_rejected(self):
    # The kernel hardcodes GPT-OSS's limit, so a different one cannot silently run with 7.0.
    cfg = self._config(model_name="gpt-oss-20b", override_model_config=True, mlp_activations_limit=10.0)
    with self.assertRaisesRegex(NotImplementedError, "mlp_activations_limit"):
      moe.fused_moe_activation(cfg)


class FuseMoeBiasesForKernelTest(parameterized.TestCase):
  """`fuse_moe_biases_for_kernel` lays the biases out exactly like the vLLM loader lays out `wi`."""

  num_experts, mlp_dim = 2, 12

  def _biases(self):
    gate = jnp.arange(self.num_experts * self.mlp_dim, dtype=jnp.float32).reshape(self.num_experts, self.mlp_dim)
    return gate + 1.0, -(gate + 1.0)

  @parameterized.parameters((1, 12), (1, 16), (2, 16), (4, 16), (4, 24))
  def test_per_shard_layout_matches_fused_kernel(self, num_shards, padded_mlp_dim):
    """A bias is a kernel with one input row, so `_fuse_moe_weights` must produce the same layout."""
    gate, up = self._biases()
    mesh = jax.sharding.AbstractMesh((num_shards,), ("model",))
    target = jax.ShapeDtypeStruct(
        (self.num_experts, 1, 2 * padded_mlp_dim), jnp.float32, sharding=NamedSharding(mesh, P(None, None, "model"))
    )
    expected = model_creation_utils._fuse_moe_weights(  # pylint: disable=protected-access
        {"wi_0": gate[:, None, :], "wi_1": up[:, None, :]}, {"wi": target}
    )["wi"]

    actual = moe.fuse_moe_biases_for_kernel(gate, up, 2 * padded_mlp_dim, num_shards, per_shard_layout=True)

    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_array_equal(actual, expected)

  def test_concatenated_layout(self):
    """Unfused `wi_0` / `wi_1` (each padded per shard) are concatenated whole by `fused_moe_matmul`."""
    gate, up = self._biases()
    actual = moe.fuse_moe_biases_for_kernel(gate, up, 32, num_shards=2, per_shard_layout=False)

    def pad_shards(bias):
      return np.concatenate([np.pad(shard, ((0, 0), (0, 2))) for shard in np.split(np.asarray(bias), 2, axis=-1)], -1)

    expected = np.concatenate([pad_shards(gate), pad_shards(up)], axis=-1)[:, None, :]
    np.testing.assert_array_equal(actual, expected)

  def test_rejects_unshardable_dims(self):
    gate, up = self._biases()
    with self.assertRaises(ValueError):
      moe.fuse_moe_biases_for_kernel(gate, up, 32, num_shards=5, per_shard_layout=True)


class GptOssFusedMoeTest(parameterized.TestCase):
  """GPT-OSS through `fused_moe_matmul` matches the dense MaxText implementation.

  The kernel is emulated (see `_emulated_fused_moe_func`), so this covers what MaxText hands it.
  """

  emb_dim, mlp_dim, num_experts, top_k, tokens = 64, 96, 4, 2, 8

  def _config(self, config_file, **overrides):
    return pyconfig.initialize(
        [None, get_test_config_path(config_file)],
        run_name="gpt_oss_fused_moe",
        model_name="gpt-oss-20b",
        override_model_config=True,
        base_emb_dim=self.emb_dim,
        base_mlp_dim=self.mlp_dim,
        base_moe_mlp_dim=self.mlp_dim,
        num_experts=self.num_experts,
        num_experts_per_tok=self.top_k,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        enable_checkpointing=False,
        log_config=False,
        max_target_length=self.tokens,
        per_device_batch_size=1,
        **overrides,
    )

  @parameterized.named_parameters(
      ("unfused", 1, 1, False),
      ("prefused", 1, 1, True),
      ("prefused_tp2", 2, 1, True),
      ("prefused_tp4", 4, 1, True),
      # The expert-parallel kernel reads gate/up as two whole halves, which is how unfused wi_0/wi_1 arrive.
      ("unfused_tp2_ep2", 2, 2, False),
  )
  def test_matches_dense(self, tp, ep, prefuse):
    if jax.device_count() < tp * ep:
      self.skipTest(f"Needs {tp * ep} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={tp * ep}).")
    dense_cfg = self._config("base.yml", sparse_matmul=False, megablox=False)
    dense = _make_moe(dense_cfg, _mesh(dense_cfg, fsdp=1))
    _randomize_biases(dense, seed=1)

    # What the vLLM adapter sets up for GPT-OSS: the 96 MLP dim is padded for GMM_v2 on every shard.
    padded_mlp_dim = compute_padded_moe_mlp_dim(self.mlp_dim, tp)
    self.assertGreater(padded_mlp_dim, self.mlp_dim)
    fused_cfg = self._config(
        "inference/vllm.yml",
        ici_tensor_parallelism=tp,
        ici_expert_parallelism=ep,
        padded_base_moe_mlp_dim=padded_mlp_dim,
        prefuse_moe_weights=prefuse,
    )
    fused_mesh = _mesh(fused_cfg, model=tp, expert=ep)
    fused = _make_moe(fused_cfg, fused_mesh)
    _load_like_vllm(dense, fused, fused_cfg, fused_mesh)

    inputs = jax.random.normal(jax.random.key(2), (1, self.tokens, self.emb_dim), jnp.float32)
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
    self.assertEqual(call["w1_bias"].shape, (self.num_experts, 1, 2 * padded_mlp_dim))
    self.assertEqual(call["w2_bias"].shape, (self.num_experts, 1, self.emb_dim))
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


class GptOssVllmModelShardingTest(unittest.TestCase):
  """The GPT-OSS model can be created with vLLM's logical axis rules, as `MaxTextForCausalLM` creates it."""

  def test_abstract_model_has_valid_shardings(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml")],
        run_name="gpt_oss_vllm_model_sharding",
        model_name="gpt-oss-20b",
        override_model_config=True,
        base_emb_dim=64,
        base_mlp_dim=96,
        base_moe_mlp_dim=96,
        base_num_query_heads=8,
        base_num_kv_heads=2,
        head_dim=16,
        base_num_decoder_layers=2,
        vocab_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        padded_base_moe_mlp_dim=256,
        prefuse_moe_weights=True,
        enable_checkpointing=False,
        log_config=False,
        max_target_length=8,
        per_device_batch_size=1,
    )
    mesh = _mesh(cfg)
    # tpu-inference creates the model under its mesh, so every Variable's sharding becomes a NamedSharding.
    # That rejects a spec that maps one mesh axis to two dimensions, which vLLM's rules produced for the
    # expert biases' ("exp", "activation_mlp").
    with jax.set_mesh(mesh):
      _, abstract_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)

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
    return pyconfig.initialize(
        [None, get_test_config_path(config_file)],
        run_name="gpt_oss_fused_moe_tpu",
        model_name="gpt-oss-20b",
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
