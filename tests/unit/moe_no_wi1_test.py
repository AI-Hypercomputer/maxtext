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
"""Tests for non-gated (relu2, no wi_1) MoE experts with Nemotron-H routing."""

import unittest

from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_EMB, _EXPERTS, _TOPK, _MOE_DIM, _SHARED = 16, 8, 2, 8, 2
_BATCH, _SEQ = 2, 4


def _config(sparse_matmul: bool, return_routed_experts: bool = False):
  # override_model_config: this model's yml, when present, must not fight the tiny shapes below.
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="moe_no_wi1_test",
      enable_checkpointing=False,
      model_name="nemotron3-nano-omni-30b-a3b",
      override_model_config=True,
      decoder_block="nemotron3",
      scan_layers=False,
      base_num_decoder_layers=1,
      hybrid_override_pattern="E",
      base_emb_dim=_EMB,
      base_num_query_heads=2,
      base_num_kv_heads=1,
      head_dim=8,
      vocab_size=32,
      num_experts=_EXPERTS,
      num_experts_per_tok=_TOPK,
      base_moe_mlp_dim=_MOE_DIM,
      base_mlp_dim=_MOE_DIM,
      shared_experts=_SHARED,
      mlp_activations=["relu2"],
      mlp_bias=False,
      routed_score_func="sigmoid",
      routed_bias=True,
      routed_scaling_factor=2.5,
      norm_topk_prob=True,
      n_routing_groups=1,
      topk_routing_group=1,
      float32_gate_logits=True,
      float32_weight_sum=True,
      routed_bias_update_rate=0.0,
      load_balance_loss_weight=0.0,
      sparse_matmul=sparse_matmul,
      megablox=False,
      capacity_factor=-1.0,
      return_routed_experts=return_routed_experts,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      per_device_batch_size=_BATCH,
      max_target_length=_SEQ,
      max_prefill_predict_length=_SEQ,
  )


def _make_moe(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  return moe.RoutedAndSharedMoE(
      config=cfg,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      rngs=nnx.Rngs(params=0),
      weight_dtype=jnp.float32,
      dtype=jnp.float32,
  )


def _weights(layer):
  """Router, expert and shared-expert weights as numpy arrays."""
  routed = layer.routed_moe
  return {
      "gate": np.asarray(routed.gate.kernel[...]),
      "bias": np.asarray(routed.gate.bias[...]),
      "wi": np.asarray(routed.wi_0[...]),
      "wo": np.asarray(routed.wo[...]),
      "shared_wi": np.asarray(layer.shared_experts.wi.kernel[...]),
      "shared_wo": np.asarray(layer.shared_experts.wo.kernel[...]),
  }


def _relu2(x):
  return np.square(np.maximum(x, 0.0))


def _reference(x, w, forced=None):
  """Nemotron-H MoE: sigmoid scores, bias picks the top-k, unbiased weights normalized x 2.5, relu2 experts."""
  x = x.astype(np.float64)
  scores = 1.0 / (1.0 + np.exp(-(x @ w["gate"])))
  top_k = np.argsort(-(scores + w["bias"]), axis=-1)[..., :_TOPK] if forced is None else forced
  weights = np.take_along_axis(scores, top_k, axis=-1)
  weights = weights / (weights.sum(-1, keepdims=True) + 1e-20) * 2.5
  routed = np.zeros_like(x)
  for k in range(_TOPK):
    up = np.einsum("bsm,bsmh->bsh", x, w["wi"][top_k[..., k]])
    down = np.einsum("bsh,bshm->bsm", _relu2(up), w["wo"][top_k[..., k]])
    routed += weights[..., k : k + 1] * down
  shared = _relu2(x @ w["shared_wi"]) @ w["shared_wo"]
  return routed + shared, top_k


class _AttentionOverride:
  """Config view with a different attention kernel name, to reach the vLLM fused-MoE branches."""

  def __init__(self, cfg, attention):
    self._cfg = cfg
    self.attention = attention

  def __getattr__(self, name):
    return getattr(self._cfg, name)


class Relu2ActivationTest(unittest.TestCase):

  def test_relu2(self):
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_array_equal(
        np.asarray(linears._convert_to_activation_function("relu2")(x)),  # pylint: disable=protected-access
        np.array([0.0, 0.0, 0.0, 0.25, 9.0], dtype=np.float32),
    )


class NonGatedMoeTest(parameterized.TestCase):

  def _setup(self, sparse_matmul, return_routed_experts=False):
    """Tiny relu2 RoutedAndSharedMoE with spread fp32 router biases, and its input."""
    cfg = _config(sparse_matmul, return_routed_experts)
    layer = _make_moe(cfg)
    routed = layer.routed_moe
    self.assertFalse(routed.has_wi_1)
    self.assertIsNone(routed.wi_1)
    # Spread router biases (about the sigmoid range) so the bias changes the top-k and no two experts tie.
    bias = np.linspace(-0.6, 0.6, _EXPERTS, dtype=np.float32)[np.random.default_rng(1).permutation(_EXPERTS)]
    routed.gate.bias[...] = jnp.asarray(bias)
    self.assertEqual(routed.gate.bias[...].dtype, jnp.float32)
    x = np.random.default_rng(0).normal(size=(_BATCH, _SEQ, _EMB)).astype(np.float32)
    return layer, x

  @parameterized.named_parameters(("dense_matmul", False), ("sparse_matmul", True))
  def test_matches_numpy_reference(self, sparse_matmul):
    layer, x = self._setup(sparse_matmul)
    w = _weights(layer)
    expected, top_k = _reference(x, w)
    unbiased_top_k = np.argsort(-(x.astype(np.float64) @ w["gate"]), axis=-1)[..., :_TOPK]
    self.assertFalse(np.array_equal(np.sort(top_k, -1), np.sort(unbiased_top_k, -1)), "bias must change the top-k")
    out, _, _ = layer(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(("dense_matmul", False), ("sparse_matmul", True))
  def test_forced_free_top_k_is_identical(self, sparse_matmul):
    layer, x = self._setup(sparse_matmul)
    _, top_k = _reference(x, _weights(layer))
    free, _, _ = layer(jnp.asarray(x))
    forced, _, _ = layer(jnp.asarray(x), forced_routed_experts=jnp.asarray(top_k, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(free), np.asarray(forced))

  @parameterized.named_parameters(("dense_matmul", False), ("sparse_matmul", True))
  def test_return_routed_experts_records_own_top_k(self, sparse_matmul):
    layer, x = self._setup(sparse_matmul, return_routed_experts=True)
    w = _weights(layer)
    _, top_k = _reference(x, w)
    # Force the lowest-scoring experts, a disjoint set from the router's own choice.
    scores = 1.0 / (1.0 + np.exp(-(x.astype(np.float64) @ w["gate"])))
    forced = np.argsort(scores + w["bias"], axis=-1)[..., :_TOPK].astype(np.int32)
    self.assertFalse(np.any(forced[..., :, None] == top_k[..., None, :]))
    out, _, _ = layer(jnp.asarray(x), forced_routed_experts=jnp.asarray(forced))
    expected, _ = _reference(x, w, forced=forced)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)
    recorded = np.asarray(layer.routed_moe.selected_experts.get_value())
    np.testing.assert_array_equal(np.sort(recorded, -1), np.sort(top_k, -1))

  def test_rejects_fused_paths(self):
    cfg = _config(sparse_matmul=False)
    with self.assertRaisesRegex(NotImplementedError, "unfused dense_matmul and sparse_matmul"):
      _make_moe(_AttentionOverride(cfg, "vllm_rpa"))
    layer = _make_moe(cfg)
    layer.routed_moe.config = _AttentionOverride(cfg, "vllm_rpa")
    with self.assertRaisesRegex(NotImplementedError, "fused MoE path needs gated experts"):
      layer.routed_moe(jnp.zeros((_BATCH, _SEQ, _EMB), jnp.float32))


if __name__ == "__main__":
  unittest.main()
