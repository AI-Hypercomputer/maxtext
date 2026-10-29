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

"""Composition tests for the Kimi-K3 decoder (`decoder_block: kimi_linear`).

These need no PyTorch or HF reference, so they run in CI. They cover:

  1. `kimi-k3.yml` building a `HyperParameters` through the real config pipeline, the
     released 93-layer layout, and the guards for unsupported modes (packing, scan_layers,
     pipeline parallelism, autoregressive decode).
  2. `KimiK3DecoderLayer` running forward and backward passes, threading the AttnRes
     `block_residual` highway across a heterogeneous KDA/MLA, dense/MoE stack.
  3. The emitted parameter tree matching the `param_mapping.py` naming contract.
  4. Kimi-K3's MLA layers being NoPE (HF `mla_use_nope: true`).

The model is shrunk to toy dimensions: this is a wiring/shape test, not a parity test.
Numerical parity against the PyTorch reference is in `kimi_k3_decoder_vs_reference_test.py`.
"""

import os
import sys
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.common.common_types import AttentionType, DecoderBlockType, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla, nnx_decoders
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils.test_helpers import get_test_config_path

_KIMI_K3_YML = os.path.join(MAXTEXT_CONFIGS_DIR, "models", "kimi-k3.yml")

# Toy stack: 6 layers. `full_attn_layers` from the real config contains 3, so
# layer 3 is the single MLA layer and layers 0,1,2,4,5 are KDA. Layer 0 is the
# only dense-MLP layer (`first_num_dense_layers: 1`). `attn_res_block_size` is
# lowered to 2 so boundaries land at layers 0, 2 and 4, producing three
# checkpointed blocks and exercising the concatenate path more than once.
_NUM_LAYERS = 6
_ATTN_RES_BLOCK_SIZE = 2
_EXPECTED_MLA_LAYERS = {3}
_EXPECTED_DENSE_LAYERS = {0}
_SEQ_LEN = 8
_EMB_DIM = 128


def _toy_config(**overrides):
  """Builds a real HyperParameters for kimi-k3 at toy scale."""
  kwargs = {
      # Shrinking a real model config to toy scale requires opting in explicitly.
      "override_model_config": True,
      # Core dims.
      "base_emb_dim": _EMB_DIM,
      "base_num_decoder_layers": _NUM_LAYERS,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 32,
      "vocab_size": 256,
      # MLA low-rank dims.
      "q_lora_rank": 32,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 8,
      "v_head_dim": 16,
      # MLP / MoE.
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 64,
      "moe_intermediate_size": 64,
      "shared_intermediate_size": 64,
      "routed_expert_hidden_size": 64,
      "num_experts": 8,
      "num_experts_per_tok": 2,
      # Routed experts (moe.RoutedMoE): experts on the latent, router on the hidden state;
      # megablox is a TPU kernel, so CPU uses the `ragged_dot` sparse path.
      "moe_expert_input_dim": 64,
      "moe_router_input_dim": _EMB_DIM,
      "megablox": False,
      # Sequence. Keep original == max so MLA's YaRN mscale rescaling stays off.
      "max_target_length": _SEQ_LEN,
      "max_position_embeddings": _SEQ_LEN,
      "original_max_position_embeddings": _SEQ_LEN,
      "max_prefill_predict_length": _SEQ_LEN // 2,
      "per_device_batch_size": 1,
      # Kimi-K3 is unscanned and unpacked only for now.
      "scan_layers": False,
      "packing": False,
      "enable_checkpointing": False,
      "attn_res_block_size": _ATTN_RES_BLOCK_SIZE,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **kwargs)


def _build_model(cfg):
  return model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))


@nnx.jit
def _forward(model, tokens, positions):
  # Jitted: eager op-by-op dispatch of the KDA/MoE stack is several times slower on CPU.
  return model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)


def _toy_inputs():
  tokens = (jnp.arange(_SEQ_LEN, dtype=jnp.int32) % 7 + 1)[None, :]
  positions = jnp.broadcast_to(jnp.arange(_SEQ_LEN), tokens.shape)
  return tokens, positions


class KimiK3ConfigTest(unittest.TestCase):
  """Config pipeline, released layout and config-time guards."""

  def test_config_builds_through_pydantic_pipeline(self):
    """`kimi-k3.yml` must survive the real config pipeline, not just yaml.safe_load."""
    cfg = _toy_config()
    self.assertEqual(cfg.decoder_block, DecoderBlockType.KIMI_LINEAR)
    self.assertEqual(cfg.num_decoder_layers, _NUM_LAYERS)
    self.assertEqual(cfg.first_num_dense_layers, 1)
    self.assertEqual(cfg.attn_res_block_size, _ATTN_RES_BLOCK_SIZE)
    # Kimi ships no rope_scaling; the mscale softmax rescale must stay disabled.
    self.assertFalse(cfg.max_position_embeddings > cfg.original_max_position_embeddings)

  def test_packing_is_rejected(self):
    """KDA ignores segment ids, so packed inputs must fail at config time."""
    with self.assertRaisesRegex(ValueError, "packing=False"):
      _toy_config(packing=True)

  def test_context_parallelism_is_rejected(self):
    """KDA has no cross-shard state composition or load-balance unpermute, so CP > 1 must fail."""
    for overrides in (
        {"ici_context_parallelism": 2},
        {"dcn_context_parallelism": 2},
        {"ici_context_usp_ulysses_parallelism": 2},
        {"dcn_context_usp_ulysses_parallelism": 2},
        {"ici_context_autoregressive_parallelism": 2},
        {"dcn_context_autoregressive_parallelism": 2},
        {"context_sharding": "expert", "ici_expert_parallelism": 2},
    ):
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(ValueError, "context parallelism"):
          _toy_config(**overrides)

  def test_layer_spec_matches_full_model_architecture(self):
    """The real 93-layer arithmetic: 24 MLA layers, 1 dense layer."""
    full_attn_layers = [3 + 4 * i for i in range(23)] + [92]
    spec = kimi_k3.build_kimi_layer_spec(93, full_attn_layers, 1)

    self.assertEqual(len(spec), 93)
    mla = [i for i, (is_linear, _) in enumerate(spec) if not is_linear]
    dense = [i for i, (_, is_moe) in enumerate(spec) if not is_moe]
    self.assertEqual(len(mla), 24)
    self.assertEqual(dense, [0])
    # Layer 0: KDA + dense. Layer 3 and the tail layer 92: MLA + MoE.
    self.assertEqual(spec[0], (True, False))
    self.assertEqual(spec[3], (False, True))
    self.assertEqual(spec[92], (False, True))


class KimiK3ReleasedLayoutTest(unittest.TestCase):
  """Structural properties of the released layout in `configs/models/kimi-k3.yml`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(_KIMI_K3_YML, "rt", encoding="utf8") as f:
      cls.yml = yaml.safe_load(f)

  def test_full_attn_layers_are_consistent_with_the_cycle_fields(self):
    """`full_attn_layers` must be the 3xKDA+1xMLA cycle, plus the trailing extra MLA."""
    interval = self.yml["inhomogeneous_layer_cycle_interval"]  # 3 KDA + 1 MLA
    self.assertEqual(interval, 4)
    num_cycles = (self.yml["base_num_decoder_layers"] - 1) // interval
    cycled = [c * interval + interval - 1 for c in range(num_cycles)]
    self.assertEqual(self.yml["full_attn_layers"], cycled + [self.yml["base_num_decoder_layers"] - 1])
    # 23 cycles cover 92 layers; layer 92 is the odd one out and is full attention.
    self.assertEqual(num_cycles * interval, self.yml["base_num_decoder_layers"] - 1)

  def test_last_attn_res_block_is_partial(self):
    """93 layers / block 12 -> 8 blocks, the last covering only 9 layers."""
    n, bs = self.yml["base_num_decoder_layers"], self.yml["attn_res_block_size"]
    self.assertEqual((n, bs), (93, 12))
    self.assertEqual((n + bs - 1) // bs, 8)
    self.assertEqual(n % bs, 9)


class KimiK3FullAttnLayerResolutionTest(unittest.TestCase):
  """Index-base contract of `param_mapping._resolve_kimi_k3_full_attn_layers`."""

  def test_hf_linear_attn_config_is_one_indexed_and_authoritative(self):
    cfg = {"num_hidden_layers": 8, "full_attn_layers": [0], "linear_attn_config": {"full_attn_layers": [4, 8]}}
    self.assertEqual(param_mapping._resolve_kimi_k3_full_attn_layers(cfg), {3, 7})  # pylint: disable=protected-access

  def test_top_level_is_zero_indexed(self):
    cfg = {"num_hidden_layers": 4, "full_attn_layers": [1]}
    self.assertEqual(param_mapping._resolve_kimi_k3_full_attn_layers(cfg), {1})  # pylint: disable=protected-access

  def test_out_of_range_is_rejected(self):
    with self.assertRaises(ValueError):
      param_mapping._resolve_kimi_k3_full_attn_layers(  # pylint: disable=protected-access
          {"num_hidden_layers": 4, "full_attn_layers": [4]}
      )
    with self.assertRaises(ValueError):
      param_mapping._resolve_kimi_k3_full_attn_layers(  # pylint: disable=protected-access
          {"num_hidden_layers": 4, "linear_attn_config": {"full_attn_layers": [0]}}
      )

  def test_mapping_and_hooks_agree_on_layer_types(self):
    """Layer 3 must be MLA in both the name map and the hook map for an HF-style config."""
    cfg = {"num_hidden_layers": 4, "num_experts": 2, "linear_attn_config": {"full_attn_layers": [4]}}
    mapping = param_mapping.PARAM_MAPPING["kimi-k3"](cfg, None, scan_layers=False)
    hooks = param_mapping.HOOK_FNS["kimi-k3"](cfg, None, scan_layers=False, saving_to_hf=False)
    self.assertIn("params-decoder-layers_3-self_attention-wq_a-kernel", mapping)
    self.assertNotIn("params-decoder-layers_3-self_attention-q_proj-kernel", mapping)
    self.assertIn("params-decoder-layers_3-self_attention-wq_b-kernel", hooks)
    self.assertIn("params-decoder-layers_2-self_attention-q_conv1d-kernel", hooks)


class KimiK3DecoderGuardTest(unittest.TestCase):
  """Unsupported modes must fail loudly instead of building a silently wrong model."""

  def test_scan_layers_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "scan_layers"):
      _build_model(_toy_config(scan_layers=True))

  def test_pipeline_parallelism_is_rejected(self):
    # A real pipeline mesh needs several devices; the guard only reads these two fields.
    fake_decoder = types.SimpleNamespace(
        is_kimi_linear=True,
        config=types.SimpleNamespace(using_pipeline_parallelism=True, scan_layers=False),
    )
    with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
      nnx_decoders.NNXDecoder._init_decoder_layers(fake_decoder, None, None, None)  # pylint: disable=protected-access

  def test_layer_structure_args_are_required(self):
    cfg = _toy_config()
    with self.assertRaises(TypeError):
      kimi_k3.KimiK3DecoderLayer(  # pylint: disable=missing-kwoa
          config=cfg, model_mode=MODEL_MODE_TRAIN, mesh=None, rngs=nnx.Rngs(0)
      )


class KimiK3DecoderTest(unittest.TestCase):
  """Forward/backward smoke tests for the composed Kimi-K3 decoder (one shared toy model)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.cfg = _toy_config()
    cls.model = _build_model(cls.cfg)

  def test_decoder_composes_expected_layer_types(self):
    """Each layer index must get the right attention and MLP variant."""
    for i in range(_NUM_LAYERS):
      layer = getattr(self.model.decoder, f"layers_{i}")
      with self.subTest(layer=i):
        self.assertEqual(layer.is_linear_attn, i not in _EXPECTED_MLA_LAYERS)
        self.assertEqual(layer.is_moe, i not in _EXPECTED_DENSE_LAYERS)

  def test_mla_layers_are_nope(self):
    """Kimi-K3 sets `mla_use_nope: true`; MLA must not apply rotary embeddings."""
    for i in sorted(_EXPECTED_MLA_LAYERS):
      layer = getattr(self.model.decoder, f"layers_{i}")
      self.assertTrue(layer.self_attention.is_nope_layer, f"MLA layer {i} must be a NoPE layer")

  def test_mla_lora_norms_use_1e_6_epsilon(self):
    """HF `KimiMLAAttention` uses `eps=1e-6` on `q_a_layernorm`/`kv_a_layernorm` while `MLA` defaults to config eps."""
    self.assertEqual(self.cfg.normalization_layer_epsilon, 1e-5)
    for i in sorted(_EXPECTED_MLA_LAYERS):
      layer = getattr(self.model.decoder, f"layers_{i}")
      self.assertEqual(layer.pre_self_attention_layer_norm.epsilon, 1e-5)
      self.assertEqual(layer.self_attention.q_norm.epsilon, 1e-6)
      self.assertEqual(layer.self_attention.kv_norm.epsilon, 1e-6)

    mla_layer = getattr(self.model.decoder, f"layers_{min(_EXPECTED_MLA_LAYERS)}").self_attention
    default_mla = attention_mla.MLA(
        config=self.cfg,
        num_query_heads=self.cfg.num_query_heads,
        num_kv_heads=self.cfg.num_kv_heads,
        head_dim=self.cfg.head_dim,
        max_target_length=self.cfg.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        attention_kernel=self.cfg.attention,
        attention_type=AttentionType(self.cfg.attention_type),
        inputs_q_shape=(1, _SEQ_LEN, _EMB_DIM),
        inputs_kv_shape=(1, _SEQ_LEN, _EMB_DIM),
        mesh=mla_layer.mesh,
        dtype=self.cfg.dtype,
        weight_dtype=self.cfg.weight_dtype,
        q_lora_rank=self.cfg.q_lora_rank,
        kv_lora_rank=self.cfg.kv_lora_rank,
        qk_nope_head_dim=self.cfg.qk_nope_head_dim,
        qk_rope_head_dim=self.cfg.qk_rope_head_dim,
        v_head_dim=self.cfg.v_head_dim,
        is_nope_layer=True,
        mla_use_output_gate=True,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(default_mla.q_norm.epsilon, 1e-5)
    self.assertEqual(default_mla.kv_norm.epsilon, 1e-5)

  def test_forward_pass_produces_finite_logits(self):
    """One full forward pass through the AttnRes highway."""
    tokens, positions = _toy_inputs()
    logits = _forward(self.model, tokens, positions)
    self.assertEqual(logits.shape, (1, _SEQ_LEN, self.cfg.vocab_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))), "logits contain NaN/Inf")

  def test_remat_does_not_change_forward(self):
    """remat_policy='full' (default) and 'none' take different code paths; outputs must match."""
    self.assertEqual(self.cfg.remat_policy, "full")
    no_remat = _build_model(_toy_config(remat_policy="none"))
    nnx.update(no_remat, nnx.state(self.model, nnx.Param))
    tokens, positions = _toy_inputs()
    ref = _forward(self.model, tokens, positions)
    out = _forward(no_remat, tokens, positions)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5)

  def test_gradients_are_finite(self):
    """A backward pass through the remat'd stack yields finite, non-trivial gradients."""
    tokens, positions = _toy_inputs()

    def loss_fn(model):
      logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)
      return jnp.mean(jax.nn.log_softmax(logits.astype(jnp.float32))[..., 0])

    grads = nnx.jit(nnx.grad(loss_fn))(self.model)
    leaves = jax.tree_util.tree_leaves(grads)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))), "non-finite gradient")
    self.assertGreater(sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves), 0.0)

  def test_autoregressive_decode_is_rejected_for_kda(self):
    """KDA state is not cached between decode steps yet, so decode must raise."""
    layer = self.model.decoder.layers_1
    self.assertTrue(layer.is_linear_attn)
    x = jnp.zeros((1, 1, _EMB_DIM), jnp.float32)
    with self.assertRaises(NotImplementedError):
      layer(
          x,
          jnp.ones((1, 1), jnp.int32),
          jnp.zeros((1, 1), jnp.int32),
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

  def test_param_tree_matches_param_mapping_contract(self):
    """Emitted parameter paths must match what param_mapping.py expects (exact path suffixes)."""
    state = nnx.state(self.model, nnx.Param)

    def _key_str(p):
      for attr in ("key", "name", "idx"):
        if hasattr(p, attr):
          return str(getattr(p, attr))
      return str(p)

    paths = set()
    for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]:
      parts = [_key_str(p) for p in path]
      if parts and parts[-1] == "value":
        parts = parts[:-1]
      paths.add("-".join(parts))

    def _assert_param(expected):
      self.assertTrue(
          any(p == expected or p.endswith("-" + expected) for p in paths),
          f"missing parameter {expected!r}",
      )

    # Per-layer AttnRes + norm contract, checked on a dense/KDA layer (0) and the
    # MoE/MLA layer (3).
    for i in (0, 3):
      for name in (
          "pre_self_attention_layer_norm-scale",
          "post_self_attention_layer_norm-scale",
          "self_attention_res_norm-scale",
          "self_attention_res_proj-kernel",
          "mlp_res_norm-scale",
          "mlp_res_proj-kernel",
      ):
        _assert_param(f"decoder-layers_{i}-{name}")

    # Decoder-level output AttnRes.
    _assert_param("decoder-output_attn_res_norm-scale")
    _assert_param("decoder-output_attn_res_proj-kernel")


if __name__ == "__main__":
  unittest.main()
