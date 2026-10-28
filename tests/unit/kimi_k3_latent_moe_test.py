# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Kimi-K3 latent MoE block and SiTU dense MLP (`layers/latent_moe.py`).

The sharding/shape/guard tests and the hermetic forward/gradient tests (against a pure-JAX oracle
written from HF `KimiMLP` / `KimiSparseMoeBlock`) run everywhere. The HF parity tests compare against
Moonshot's PyTorch modeling code and skip unless torch, transformers and einops are installed and the
HF reference is checked out at `kimi-k3-hf-reference/` (see `tests/utils/kimi_k3_reference.py`).
"""

import contextlib
import importlib
import sys
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.latent_moe import KimiDenseMLP, KimiLatentMoEBlock, KimiMxfp4RoutedExperts, ROUTER_KERNEL_AXES
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, reference_available
from tests.utils.test_helpers import get_test_config_path

try:
  import einops  # pylint: disable=unused-import
  import torch
  import transformers  # pylint: disable=unused-import

  from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params, load_params_into_nnx

  _HAVE_TORCH_DEPS = True
except ImportError:
  _HAVE_TORCH_DEPS = False

requires_reference = unittest.skipUnless(
    _HAVE_TORCH_DEPS and reference_available(),
    "needs torch, transformers, einops and the Kimi-K3 HF reference checkout at " + KIMI_K3_REFERENCE_DIR,
)

_REFERENCE_PACKAGE = "_kimi_k3_ref"
_REFERENCE_MODULES = ("modeling_kimi_linear", "configuration_kimi_k3")


@contextlib.contextmanager
def _hf_reference():
  """Imports the HF reference modules, then removes them from `sys.modules`/`sys.path` again.

  The modeling file is imported as a submodule of a synthetic package rooted at the checkout, so the
  upstream relative import (`from .configuration_kimi_k3 import ...`) resolves; the checkout is also put
  on `sys.path` for copies that use a top-level import instead. The configuration module returned is
  the one the modeling module actually uses. The reference has its own fallbacks for the optional `fla`
  kernels and transformers helpers, so no stubs are needed.
  """
  saved_path = list(sys.path)
  names = (_REFERENCE_PACKAGE,) + _REFERENCE_MODULES + tuple(f"{_REFERENCE_PACKAGE}.{m}" for m in _REFERENCE_MODULES)
  preexisting = {name: sys.modules[name] for name in names if name in sys.modules}
  package = types.ModuleType(_REFERENCE_PACKAGE)
  package.__path__ = [KIMI_K3_REFERENCE_DIR]
  sys.modules[_REFERENCE_PACKAGE] = package
  sys.path.insert(0, KIMI_K3_REFERENCE_DIR)
  try:
    modeling = importlib.import_module(f"{_REFERENCE_PACKAGE}.modeling_kimi_linear")
    yield modeling, sys.modules[modeling.KimiLinearConfig.__module__]
  finally:
    sys.path[:] = saved_path
    for name in names:
      sys.modules.pop(name, None)
    sys.modules.update(preexisting)


# Pure-JAX oracle, written directly from HF `SituAndMul` / `KimiMLP` / `KimiMoEGate` / `KimiSparseMoeBlock`
# (independent of `latent_moe.py` / `moe.py`), so forward values and VJPs are checked without torch.
_SITU_BETA, _SITU_LINEAR_BETA = 4.0, 25.0


def _oracle_situ_and_mul(gate, up):
  g = gate.astype(jnp.float32)
  u = up.astype(jnp.float32)
  return _SITU_BETA * jnp.tanh(g / _SITU_BETA) * jax.nn.sigmoid(g) * (_SITU_LINEAR_BETA * jnp.tanh(u / _SITU_LINEAR_BETA))


def _oracle_dense_mlp(p, x):
  """HF `KimiMLP` with `hidden_act="situ"`; MaxText kernels are [in, out]."""
  return _oracle_situ_and_mul(x @ p["wi_0"], x @ p["wi_1"]) @ p["wo"]


def _oracle_latent_moe(p, x, *, top_k, scaling, eps):
  """HF `KimiSparseMoeBlock` (latent MoE + RMSNorm + shared experts) on [B, S, H] inputs."""
  b, s, h = x.shape
  tokens = x.reshape(-1, h)
  scores = jax.nn.sigmoid(tokens @ p["gate"])  # [T, E]
  # The correction bias only affects selection; the weights are the unbiased scores.
  _, idx = jax.lax.top_k(jax.lax.stop_gradient(scores + p["gate_bias"]), top_k)
  w = jnp.take_along_axis(scores, idx, axis=-1)
  w = w / (w.sum(-1, keepdims=True) + 1e-20) * scaling
  latent = tokens @ p["down"]  # [T, L]
  w0, w1, wo = p["e_wi_0"][idx], p["e_wi_1"][idx], p["e_wo"][idx]  # [T, K, ...]
  act = _oracle_situ_and_mul(jnp.einsum("tl,tkli->tki", latent, w0), jnp.einsum("tl,tkli->tki", latent, w1))
  y = jnp.einsum("tk,tkl->tl", w, jnp.einsum("tki,tkil->tkl", act, wo))
  y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps) * p["norm"]
  y = y @ p["up"]
  shared = {"wi_0": p["s_wi_0"], "wi_1": p["s_wi_1"], "wo": p["s_wo"]}
  return (y + _oracle_dense_mlp(shared, tokens)).reshape(b, s, h)


def _dense_mlp_params(mlp):
  return {name: getattr(mlp, name).kernel.get_value() for name in ("wi_0", "wi_1", "wo")}


def _latent_moe_params(block):
  moe_block = block.routed_experts
  shared = _dense_mlp_params(block.shared_expert)
  return {
      "gate": moe_block.gate.kernel.get_value(),
      "gate_bias": moe_block.gate.bias.get_value(),
      "down": block.routed_expert_down_proj.kernel.get_value(),
      "e_wi_0": moe_block.wi_0.get_value(),
      "e_wi_1": moe_block.wi_1.get_value(),
      "e_wo": moe_block.wo.get_value(),
      "norm": block.routed_expert_norm.scale.get_value(),
      "up": block.routed_expert_up_proj.kernel.get_value(),
      "s_wi_0": shared["wi_0"],
      "s_wi_1": shared["wi_1"],
      "s_wo": shared["wo"],
  }


def _randomize(module, seed, scale=0.3):
  """Replaces every param/router bias with N(0, scale^2) noise (norm scale 1 + noise), in place."""
  key = jax.random.PRNGKey(seed)
  for path, var in nnx.state(module, (nnx.Param, moe.MoEBiasVar)).flat_state():
    key, sub = jax.random.split(key)
    value = var.get_value()
    noise = scale * jax.random.normal(sub, value.shape, jnp.float32)
    if path[-1] == "scale":
      noise = 1.0 + noise
    var.set_value(noise.astype(value.dtype))


class KimiK3LatentMoETest(unittest.TestCase):
  """Tests validating KimiDenseMLP, KimiLatentMoEBlock and the mxfp4 routed experts' router."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 64
    self.routed_expert_hidden_size = 32
    self.moe_intermediate_size = 16
    self.shared_intermediate_size = 32
    self.num_experts = 8
    self.top_k = 2
    self.num_shared_experts = 2
    self.rms_norm_eps = 1e-5
    # Non-trivial so the DeepSeek-V3 `routed_scaling_factor` path is exercised.
    self.routed_scaling_factor = 2.5

    self.hf_config = {
        "hidden_size": self.hidden_size,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.top_k,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "shared_intermediate_size": self.shared_intermediate_size,
        "num_shared_experts": self.num_shared_experts,
        "rms_norm_eps": self.rms_norm_eps,
        "first_k_dense_replace": 1,
        "full_attn_layers": [1],
    }

  def _make_config(self, **overrides):
    """A real kimi-k3 HyperParameters at this test's toy MoE dims (RoutedMoE needs one)."""
    kwargs = {
        "override_model_config": True,
        "base_emb_dim": self.hidden_size,
        "base_num_decoder_layers": 2,
        "full_attn_layers": [1],
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "head_dim": 16,
        "vocab_size": 256,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "base_mlp_dim": 128,
        "base_moe_mlp_dim": self.moe_intermediate_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "shared_intermediate_size": self.shared_intermediate_size,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.top_k,
        "shared_experts": self.num_shared_experts,
        "num_shared_experts": self.num_shared_experts,
        "routed_scaling_factor": self.routed_scaling_factor,
        "normalization_layer_epsilon": self.rms_norm_eps,
        "mlp_activations": ["situ_gate", "situ_linear"],
        # Experts on the latent, router on the hidden state. Megablox is a TPU Pallas kernel;
        # on CPU the sparse path uses `jax.lax.ragged_dot`.
        "moe_expert_input_dim": self.routed_expert_hidden_size,
        "moe_router_input_dim": self.hidden_size,
        "megablox": False,
        "max_target_length": self.seq_len,
        "max_position_embeddings": self.seq_len,
        "original_max_position_embeddings": self.seq_len,
        "per_device_batch_size": self.batch_size,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "scan_layers": False,
        "enable_checkpointing": False,
    }
    kwargs.update(overrides)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **kwargs)

  def _make_block(self, **config_overrides):
    """A KimiLatentMoEBlock built from a real toy kimi-k3 config."""
    cfg = self._make_config(**config_overrides)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    block = KimiLatentMoEBlock(
        hidden_size=self.hidden_size,
        num_experts=self.num_experts,
        top_k=self.top_k,
        routed_expert_hidden_size=self.routed_expert_hidden_size,
        moe_intermediate_size=self.moe_intermediate_size,
        num_shared_experts=self.num_shared_experts,
        shared_intermediate_size=self.shared_intermediate_size,
        routed_scaling_factor=self.routed_scaling_factor,
        rms_norm_eps=self.rms_norm_eps,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        mesh=mesh,
        routed_experts_weight_format=cfg.routed_experts_weight_format,
        config=cfg,
        rngs=nnx.Rngs(0),
    )
    return block, cfg, mesh

  def _make_mxfp4_experts(self):
    """`KimiMxfp4RoutedExperts` at this test's dims (intermediate 32: MXFP4 groups are 32 wide)."""
    cfg = self._make_config(
        routed_experts_weight_format="mxfp4",
        model_call_mode="inference",
        moe_intermediate_size=32,
        base_moe_mlp_dim=32,
        shared_intermediate_size=32 * self.num_shared_experts,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    experts = KimiMxfp4RoutedExperts(
        config=cfg,
        mesh=mesh,
        num_experts=self.num_experts,
        num_experts_per_tok=self.top_k,
        intermediate_dim=32,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=ROUTER_KERNEL_AXES,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(1),
    )
    return experts, cfg

  def test_mxfp4_router_matches_routed_moe(self):
    """The mxfp4 experts' routing equals moe.RoutedMoE's (same GateLogit params, same top-k)."""
    block, _, _ = self._make_block()
    experts, _ = self._make_mxfp4_experts()
    rng = np.random.default_rng(0)
    kernel = rng.standard_normal((self.hidden_size, self.num_experts)).astype(np.float32) / 8
    bias = rng.standard_normal((self.num_experts,)).astype(np.float32) / 10
    for gate in (block.routed_experts.gate, experts.gate):
      gate.kernel[...] = jnp.asarray(kernel)
      gate.bias[...] = jnp.asarray(bias)  # a `moe.MoEBiasVar` for kimi-k3

    x = jnp.asarray(rng.standard_normal((self.batch_size * self.seq_len, self.hidden_size)).astype(np.float32))
    idx, weight = experts.route(x)
    logits, pre_bias = block.routed_experts.gate(x)
    want_weight, want_idx = block.routed_experts.get_topk(logits, pre_bias)

    np.testing.assert_array_equal(np.asarray(idx), np.asarray(want_idx))
    np.testing.assert_allclose(np.asarray(weight), np.asarray(want_weight), rtol=1e-6, atol=1e-6)
    # Renormalized top-k weights, scaled by `routed_scaling_factor` (DeepSeek-V3 routing).
    np.testing.assert_allclose(np.asarray(weight).sum(-1), self.routed_scaling_factor, rtol=1e-5)

  def test_routed_expert_params_have_sharding_axes(self):
    """Router and routed-expert params carry logical sharding axes (else they replicate per chip).

    MXFP4 packed codes / scales leave the quantized contraction axis (-2) unsharded.
    """

    def sharding_of(module, var_type):
      return {
          "/".join(map(str, path)): var.get_metadata().get("out_sharding")
          for path, var in nnx.state(module, var_type).flat_state()
      }

    with self.subTest(weight_format="bf16"):
      block, _, _ = self._make_block()
      self.assertIsInstance(block.routed_experts, moe.RoutedMoE)
      self.assertEqual(
          sharding_of(block.routed_experts, nnx.Param),
          {
              "gate/kernel": ("embed_router", None),
              "wi_0": ("exp", "embed_moe", "mlp_moe"),
              "wi_1": ("exp", "embed_moe", "mlp_moe"),
              "wo": ("exp", "mlp_moe", "embed_moe"),
          },
      )
      # The router bias is a `MoEBiasVar` (as for DeepSeek-V3 / Kimi-K2): same `gate/bias` path, but outside
      # `nnx.Param`, so the optimizer (and AdamW weight decay) never touches it.
      self.assertIsInstance(block.routed_experts.gate.bias, moe.MoEBiasVar)
      self.assertEqual(sharding_of(block.routed_experts, moe.MoEBiasVar), {"gate/bias": (None,)})
      # Router reads the full hidden state; experts run on the latent.
      self.assertEqual(block.routed_experts.gate.kernel.shape, (self.hidden_size, self.num_experts))
      self.assertEqual(
          block.routed_experts.wi_0.shape, (self.num_experts, self.routed_expert_hidden_size, self.moe_intermediate_size)
      )

    with self.subTest(weight_format="mxfp4"):
      experts, _ = self._make_mxfp4_experts()
      self.assertEqual(
          sharding_of(experts, nnx.Param),
          {
              "gate/kernel": ("embed_router", None),
              **{f"{n}_{s}": ("exp", None, "mlp_moe") for n in ("wi_0", "wi_1") for s in ("packed", "scale")},
              **{f"wo_{s}": ("exp", None, "embed_moe") for s in ("packed", "scale")},
          },
      )
      self.assertEqual(sharding_of(experts, moe.MoEBiasVar), {"gate/bias": (None,)})

  def test_rejects_load_balance_loss_it_would_drop(self):
    """The block returns only the MoE output, so a config that needs its aux loss fails fast."""
    with self.assertRaisesRegex(NotImplementedError, "load-balance loss"):
      self._make_block(load_balance_loss_weight=0.01)

  def test_forwards_out_sharding(self):
    """`out_sharding` reaches both `routed_expert_up_proj` and `shared_expert.wo`."""
    block, _, mesh = self._make_block()
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
    dense_cls = type(block.routed_expert_up_proj)
    orig_call = dense_cls.__call__
    seen = {}

    def spy_call(mod, x, *args, **kwargs):
      if mod is block.routed_expert_up_proj:
        seen["up"] = kwargs.get("out_sharding")
      elif mod is block.shared_expert.wo:
        seen["shared_wo"] = kwargs.get("out_sharding")
      return orig_call(mod, x, *args, **kwargs)

    x = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    with mesh, mock.patch.object(dense_cls, "__call__", spy_call):
      _ = block(x, out_sharding=sharding)
    self.assertEqual(seen, {"up": sharding, "shared_wo": sharding})

  def test_routed_expert_norm_latent_axis_unsharded(self):
    """The norm keeps the latent axis replicated, like `routed_expert_down_proj` / `routed_expert_up_proj`."""
    block, _, _ = self._make_block()
    self.assertEqual(block.routed_expert_norm.scale.get_metadata().get("out_sharding"), (None,))
    self.assertEqual(block.routed_expert_down_proj.kernel.get_metadata().get("out_sharding"), ("embed", None))
    self.assertEqual(block.routed_expert_up_proj.kernel.get_metadata().get("out_sharding"), (None, "embed"))

  def test_inherits_shard_mode_and_matmul_precision_from_config(self):
    """Omitted `shard_mode` / `matmul_precision` fall back to the config's, for every dense projection."""
    block, cfg, _ = self._make_block()
    for dense in (block.routed_expert_down_proj, block.routed_expert_up_proj, block.shared_expert.wo):
      self.assertEqual(dense.matmul_precision, cfg.matmul_precision)
      self.assertEqual(dense.shard_mode, cfg.shard_mode)

  def test_config_guards(self):
    """Routing / activation settings the block does not implement are rejected at construction."""
    for overrides, pattern in (
        ({"routed_score_func": "softmax"}, "routed_score_func"),
        ({"routed_bias": False}, "routed_bias"),
        ({"mlp_activations": ["silu", "linear"]}, "mlp_activations"),
    ):
      with self.subTest(**{k: str(v) for k, v in overrides.items()}):
        with self.assertRaisesRegex(ValueError, pattern):
          self._make_block(**overrides)

  def test_rejects_non_3d_inputs(self):
    block, _, mesh = self._make_block()
    with mesh, self.assertRaisesRegex(ValueError, "batch, seq_len, hidden_size"):
      block(jnp.ones((self.batch_size * self.seq_len, self.hidden_size), jnp.float32))

  def test_dense_mlp_checkpoint_names(self):
    """`KimiDenseMLP` tags its activations with MaxText's MLP remat names."""
    mlp = KimiDenseMLP(in_features=self.hidden_size, intermediate_dim=32, rngs=nnx.Rngs(0))
    jaxpr = str(jax.make_jaxpr(mlp)(jnp.ones((1, 2, self.hidden_size), jnp.float32)))
    for name in ("mlpwi_0", "mlpwi_1", "mlpwo"):
      self.assertIn(f"name={name}", jaxpr)

  def test_kimi_dense_mlp_matches_oracle(self):
    """Layer-0 form (`out_features=None`): forward and parameter/input VJPs match the pure-JAX oracle."""
    mlp = KimiDenseMLP(in_features=self.hidden_size, intermediate_dim=96, matmul_precision="highest", rngs=nnx.Rngs(1))
    _randomize(mlp, seed=1)
    self.assertEqual(mlp.wo.kernel.shape, (96, self.hidden_size))
    x = 3.0 * jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, self.hidden_size))
    cot = jax.random.normal(jax.random.PRNGKey(3), x.shape)
    graphdef, params = nnx.split(mlp, nnx.Param)

    def loss(params, x):
      return jnp.sum(nnx.merge(graphdef, params)(x) * cot)

    def loss_ref(p, x):
      return jnp.sum(_oracle_dense_mlp(p, x) * cot)

    with jax.default_matmul_precision("highest"):
      np.testing.assert_allclose(mlp(x), _oracle_dense_mlp(_dense_mlp_params(mlp), x), rtol=1e-5, atol=1e-5)
      g_params, g_x = jax.grad(loss, argnums=(0, 1))(params, x)
      r_params, r_x = jax.grad(loss_ref, argnums=(0, 1))(_dense_mlp_params(mlp), x)
    np.testing.assert_allclose(g_x, r_x, rtol=1e-4, atol=1e-4)
    for name in ("wi_0", "wi_1", "wo"):
      np.testing.assert_allclose(g_params[name]["kernel"].get_value(), r_params[name], rtol=1e-4, atol=1e-4)

  def test_kimi_dense_mlp_bfloat16_rounds_once(self):
    """bf16: gate * up is computed in fp32 and rounded once (HF `SituAndMul`), and stays close to fp32."""
    mlp32 = KimiDenseMLP(in_features=self.hidden_size, intermediate_dim=96, rngs=nnx.Rngs(4))
    _randomize(mlp32, seed=4)
    mlp16 = KimiDenseMLP(in_features=self.hidden_size, intermediate_dim=96, dtype=jnp.bfloat16, rngs=nnx.Rngs(4))
    nnx.update(mlp16, nnx.state(mlp32, nnx.Param))
    x = 3.0 * jax.random.normal(jax.random.PRNGKey(5), (self.batch_size, self.seq_len, self.hidden_size))
    x16 = x.astype(jnp.bfloat16)
    gate, up = mlp16.wi_0(x16), mlp16.wi_1(x16)
    self.assertEqual(gate.dtype, jnp.bfloat16)
    expected = mlp16.wo(_oracle_situ_and_mul(gate, up).astype(jnp.bfloat16))
    y16 = mlp16(x16)
    self.assertEqual(y16.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y16, np.float32), np.asarray(expected, np.float32))
    y32 = np.asarray(mlp32(x))
    rel_err = np.linalg.norm(np.asarray(y16, np.float32) - y32) / np.linalg.norm(y32)
    self.assertLess(rel_err, 2e-2)

  def test_routed_moe_situ_bfloat16_rounds_once(self):
    """RoutedMoE's SiTU expert activation multiplies in fp32 and casts once, matching HF `SituAndMul`."""
    block, _, _ = self._make_block(dtype="bfloat16")
    k0, k1 = jax.random.split(jax.random.PRNGKey(6))
    w0 = (10.0 * jax.random.normal(k0, (64, 48))).astype(jnp.bfloat16)
    w1 = (40.0 * jax.random.normal(k1, (64, 48))).astype(jnp.bfloat16)
    out = block.routed_experts.apply_ffn_activation(w0, w1)
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(_oracle_situ_and_mul(w0, w1).astype(jnp.bfloat16), np.float32)
    )

  def test_kimi_latent_moe_block_matches_oracle(self):
    """Forward and parameter/input VJPs match the pure-JAX oracle, on both RoutedMoE dispatch paths."""
    x = jax.random.normal(jax.random.PRNGKey(7), (self.batch_size, self.seq_len, self.hidden_size))
    cot = jax.random.normal(jax.random.PRNGKey(8), x.shape)
    oracle = jax.tree_util.Partial(
        _oracle_latent_moe, top_k=self.top_k, scaling=self.routed_scaling_factor, eps=self.rms_norm_eps
    )
    for dispatch, overrides in (("sparse_ragged_dot", {"sparse_matmul": True}), ("dense", {"sparse_matmul": False})):
      with self.subTest(dispatch=dispatch):
        block, _, mesh = self._make_block(**overrides)
        _randomize(block, seed=9)
        ref_params = _latent_moe_params(block)
        graphdef, params, rest = nnx.split(block, nnx.Param, ...)

        def loss(params, x, graphdef=graphdef, rest=rest):
          return jnp.sum(nnx.merge(graphdef, params, rest)(x) * cot)

        def loss_ref(p, x):
          return jnp.sum(oracle(p, x) * cot)

        with mesh, jax.default_matmul_precision("highest"):
          y = block(x)
          y_ref = oracle(ref_params, x)
          g_params, g_x = jax.grad(loss, argnums=(0, 1))(params, x)
          r_params, r_x = jax.grad(loss_ref, argnums=(0, 1))(ref_params, x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(g_x, r_x, rtol=1e-4, atol=1e-4)
        got = {
            "gate": g_params["routed_experts"]["gate"]["kernel"],
            "down": g_params["routed_expert_down_proj"]["kernel"],
            "e_wi_0": g_params["routed_experts"]["wi_0"],
            "e_wi_1": g_params["routed_experts"]["wi_1"],
            "e_wo": g_params["routed_experts"]["wo"],
            "norm": g_params["routed_expert_norm"]["scale"],
            "up": g_params["routed_expert_up_proj"]["kernel"],
            "s_wi_0": g_params["shared_expert"]["wi_0"]["kernel"],
            "s_wi_1": g_params["shared_expert"]["wi_1"]["kernel"],
            "s_wo": g_params["shared_expert"]["wo"]["kernel"],
        }
        for name, grad in got.items():
          with self.subTest(param=name):
            self.assertGreater(float(jnp.abs(r_params[name]).max()), 0.0)
            np.testing.assert_allclose(grad.get_value(), r_params[name], rtol=1e-4, atol=1e-4)

  def _load_block(self, block, converted, prefix):
    """Strictly loads `converted` into `block`, including the `MoEBiasVar` router bias."""
    loaded = load_params_into_nnx(block, converted, prefix)
    self.assertIn(f"{prefix}-routed_experts-gate-bias", loaded)
    self.assertEqual(set(loaded), {k for k in converted if k.startswith(prefix + "-")})
    self.assertEqual(len(loaded), len(nnx.state(block, (nnx.Param, moe.MoEBiasVar)).flat_state()))

  @requires_reference
  def test_kimi_dense_mlp_parity(self):
    """Verifies Layer 0 Dense MLP parity against PyTorch KimiMLP with 4x expansion."""
    test_hidden_size = 128
    test_intermediate_size = 512
    seq_len = 16
    with _hf_reference() as (modeling_kimi_linear, configuration_kimi_k3):
      torch.manual_seed(42)
      pt_config = configuration_kimi_k3.KimiLinearConfig(
          hidden_size=test_hidden_size,
          intermediate_size=test_intermediate_size,
          hidden_act="situ",
          activation_situ_beta=4.0,
          activation_situ_linear_beta=25.0,
      )
      pt_mlp = modeling_kimi_linear.KimiMLP(pt_config)
      pt_mlp.eval()

    test_hf_config = dict(self.hf_config)
    test_hf_config["hidden_size"] = test_hidden_size
    test_hf_config["intermediate_size"] = test_intermediate_size
    prefix = "params-decoder-layers_0-mlp"
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_mlp.state_dict(), test_hf_config, maxtext_config=None, prefix_filter=prefix + "-"
    )

    jax_mlp = KimiDenseMLP(
        in_features=test_hidden_size,
        intermediate_dim=test_intermediate_size,
        matmul_precision="highest",
        rngs=nnx.Rngs(0),
    )
    # Strict: also checks the layer-0 HF -> MaxText mapping.
    load_params_into_nnx(jax_mlp, converted_params, prefix)

    x_torch = torch.randn(self.batch_size, seq_len, test_hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_mlp(x_torch).numpy()

    y_jax = np.array(jax_mlp(jnp.array(x_torch.numpy())))
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

  @requires_reference
  def test_kimi_latent_moe_block_parity(self):
    """Verifies Latent MoE block parity against PyTorch KimiSparseMoeBlock."""
    with _hf_reference() as (modeling_kimi_linear, configuration_kimi_k3):
      torch.manual_seed(42)
      pt_config = configuration_kimi_k3.KimiLinearConfig(
          hidden_size=self.hidden_size,
          intermediate_size=128,
          num_experts=self.num_experts,
          num_experts_per_token=self.top_k,
          routed_expert_hidden_size=self.routed_expert_hidden_size,
          moe_intermediate_size=self.moe_intermediate_size,
          num_shared_experts=self.num_shared_experts,
          shared_intermediate_size=self.shared_intermediate_size,
          moe_renormalize=True,
          routed_scaling_factor=self.routed_scaling_factor,
          latent_moe_use_norm=True,
          rms_norm_eps=self.rms_norm_eps,
          hidden_act="situ",
          activation_situ_beta=4.0,
          activation_situ_linear_beta=25.0,
      )
      pt_moe = modeling_kimi_linear.KimiSparseMoeBlock(pt_config)
      pt_moe.eval()
    # A non-zero correction bias, so a mis-mapped or unloaded router bias changes expert selection.
    # A non-trivial norm scale, so a mis-mapped `routed_expert_norm-scale` changes the output.
    with torch.no_grad():
      pt_moe.gate.e_score_correction_bias.normal_(std=0.5)
      pt_moe.routed_expert_norm.weight.uniform_(0.5, 1.5)

    prefix = "params-decoder-layers_1-mlp"
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_moe.state_dict(), self.hf_config, maxtext_config=None, prefix_filter=prefix + "-"
    )

    x_torch = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_moe(x_torch).numpy()

    x_jax = jnp.array(x_torch.numpy())
    # Both RoutedMoE dispatch paths: sorted/grouped (`ragged_dot` on CPU) and dense einsum.
    for dispatch, overrides in (("sparse_ragged_dot", {"sparse_matmul": True}), ("dense", {"sparse_matmul": False})):
      with self.subTest(dispatch=dispatch):
        jax_moe, _, mesh = self._make_block(**overrides)
        self._load_block(jax_moe, converted_params, prefix)
        with mesh, jax.default_matmul_precision("highest"):
          y_jax = np.array(jax_moe(x_jax))
        np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
