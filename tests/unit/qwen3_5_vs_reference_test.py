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

"""Numerical reference tests for the Qwen3.5 TEXT decoder composition.

The Qwen3.5 text decoder is a hybrid stack: every ``inhomogeneous_layer_cycle_interval``-th
layer is a full-attention layer and the rest are GatedDeltaNet (linear-attention) layers
(see ``Qwen3_5DecoderLayer`` in ``maxtext/models/qwen3_5.py`` line 146 and
``Qwen3_5ScannableBlock`` which builds one full cycle of these layers).

``Qwen3_5GatedDeltaNet`` / ``Qwen3_5FullAttention`` / ``Qwen3_5SparseMoEBlock`` are
unmodified subclasses of their Qwen3-Next equivalents, whose per-op numerical correctness
is already covered by ``tests/unit/qwen3_next_vs_reference_test.py``. This file therefore
does not re-test the shared GDN/attention/MoE math. Instead it validates the
*Qwen3.5-specific composition*:

  * a whole ``Qwen3_5DecoderLayer`` forward at a full-attention layer index and at a
    GatedDeltaNet layer index, against a PyTorch reference assembled from the reused
    Qwen3-Next ``_PT`` reference modules, and
  * the ``Qwen3_5ScannableBlock`` hybrid layer-type selection wiring over one full cycle,
    both structurally (which layers are full-attention vs. GDN) and numerically.

Run on TPU v6e with float32 + matmul_precision=highest for tight tolerances::

    python -m pytest tests/unit/qwen3_5_vs_reference_test.py -v
"""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ScannableBlock,
    Qwen3_5FullAttention,
    Qwen3_5GatedDeltaNet,
)
from tests.utils.test_helpers import get_test_config_path

# Reuse the hand-written Qwen3-Next PyTorch reference modules directly: Qwen3.5's layers
# are unmodified subclasses, so these references apply without change.
from tests.unit.qwen3_next_vs_reference_test import (
    Qwen3NextRMSNorm_PT,
    Qwen3NextGatedDeltaNet_PT,
    Qwen3NextFullAttention_PT,
    Qwen3NextSparseMoeBlock_PT,
    Qwen3NextRotaryEmbedding_PT,
    create_causal_mask_PT,
)

torch.set_grad_enabled(False)


# ----------------------------------------------------------------------
# Config + PyTorch-config construction
# ----------------------------------------------------------------------
def build_test_config(cycle_interval: int = 4):
  """Builds a small float32 Qwen3.5 text config for tight-tolerance reference checks.

  Dimensions mirror ``tests/unit/qwen3_next_vs_reference_test.py`` so the reused
  Qwen3-Next ``_PT`` references apply directly. In particular the GatedDeltaNet
  key/value head counts are kept equal (the head-ratio math is GDN-internal and out of
  scope for this composition test), which is the case the ``_PT`` GDN reference handles.
  """
  return pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=qwen3_5_text_ref_test",
          "dtype=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "float32_logits=True",
          "decoder_block=qwen3_5",
          "attention=dot_product",
          f"inhomogeneous_layer_cycle_interval={cycle_interval}",
          # Model dimensions
          "base_emb_dim=128",
          "base_num_query_heads=4",
          "base_num_kv_heads=4",
          "head_dim=32",
          # Gated Delta Net dims (equal key/value heads: see docstring)
          "gdn_num_value_heads=4",
          "gdn_num_key_heads=4",
          "gdn_key_head_dim=32",
          "gdn_value_head_dim=32",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          "use_qk_norm_in_gdn=True",
          "normalization_layer_epsilon=1e-6",
          # MoE dims (small number of experts)
          "base_mlp_dim=256",
          "num_experts=8",
          "num_experts_per_tok=2",
          "base_moe_mlp_dim=256",
          "norm_topk_prob=True",
          "shard_exp_on_fsdp=False",
          "mlp_activations=['silu', 'linear']",
          "dropout_rate=0.0",
          # dense_matmul path in MoE matches the reference (sparse path is numerically off).
          "sparse_matmul=False",
          "skip_jax_distributed_system=True",
          # Full-attention layer settings
          "attention_bias=False",
          "rope_max_timescale=10000.0",
          "partial_rotary_factor=0.25",
      ]
  )


def make_pt_gdn_moe_config(cfg):
  """SimpleNamespace consumed by the GatedDeltaNet / MoE PyTorch references."""
  return SimpleNamespace(
      hidden_size=cfg.emb_dim,
      gdn_num_value_heads=cfg.gdn_num_value_heads,
      gdn_num_key_heads=cfg.gdn_num_key_heads,
      gdn_key_head_dim=cfg.gdn_key_head_dim,
      gdn_value_head_dim=cfg.gdn_value_head_dim,
      gdn_conv_kernel_dim=cfg.gdn_conv_kernel_dim,
      hidden_act="silu",
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      gdn_chunk_size=cfg.gdn_chunk_size,
      use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
      moe_intermediate_size=cfg.moe_mlp_dim,
      shared_expert_intermediate_size=cfg.moe_mlp_dim,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      norm_topk_prob=cfg.norm_topk_prob,
  )


def make_pt_attention_config(cfg):
  """SimpleNamespace consumed by the FullAttention PyTorch reference + rotary embedding."""
  return SimpleNamespace(
      hidden_size=cfg.emb_dim,
      num_attention_heads=cfg.num_query_heads,
      head_dim=cfg.head_dim,
      num_key_value_heads=cfg.num_kv_heads,
      attention_bias=False,
      rms_norm_eps=cfg.normalization_layer_epsilon,
      rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_max_timescale},
      max_position_embeddings=cfg.max_target_length,
      attention_dropout=cfg.dropout_rate,
      partial_rotary_factor=cfg.partial_rotary_factor,
  )


# ----------------------------------------------------------------------
# Weight-copy helpers (PyTorch reference -> JAX module).
# These consolidate the per-sublayer mappings that appear inline in
# tests/unit/qwen3_next_vs_reference_test.py so both tests below can share them.
# ----------------------------------------------------------------------
def copy_rmsnorm(pt_norm, jax_norm):
  """Copies a Qwen3NextRMSNorm weight (PT -> JAX)."""
  nnx.update(jax_norm, {"scale": nnx.Param(jnp.array(pt_norm.weight.detach().numpy()))})


def copy_gated_delta_net(pt_gdn, jax_gdn, cfg):
  """Copies GatedDeltaNet weights, reordering qkvz/ba projections into the JAX layout."""
  conv1d_weight_pt = pt_gdn.conv1d.weight.detach().numpy()
  # PT (out, in/groups, kw) -> JAX (kw, in/groups, out); depthwise => (kw, 1, C).
  conv1d_weight_jax = np.transpose(conv1d_weight_pt, (2, 1, 0))

  in_features = cfg.emb_dim
  h_k = cfg.gdn_num_key_heads
  d_k = cfg.gdn_key_head_dim
  h_v = cfg.gdn_num_value_heads
  d_v = cfg.gdn_value_head_dim
  key_dim = h_k * d_k
  value_dim = h_v * d_v
  v_per_k = h_v // h_k

  qkvz_pt = pt_gdn.in_proj_qkvz.weight.T.detach().numpy()
  q_w = qkvz_pt[:, :key_dim].reshape(in_features, h_k, d_k)
  k_w = qkvz_pt[:, key_dim : 2 * key_dim].reshape(in_features, h_k, d_k)
  v_w = qkvz_pt[:, 2 * key_dim : 2 * key_dim + value_dim].reshape(in_features, h_k, v_per_k * d_v)
  z_w = qkvz_pt[:, 2 * key_dim + value_dim :].reshape(in_features, h_k, v_per_k * d_v)
  reordered_qkvz = np.concatenate([q_w, k_w, v_w, z_w], axis=-1).reshape(in_features, -1)

  ba_pt = pt_gdn.in_proj_ba.weight.T.detach().numpy()
  b_w = ba_pt[:, :h_v].reshape(in_features, h_k, v_per_k)
  a_w = ba_pt[:, h_v:].reshape(in_features, h_k, v_per_k)
  reordered_ba = np.concatenate([b_w, a_w], axis=-1).reshape(in_features, -1)

  params = {
      "in_proj_qkvz": {"kernel": nnx.Param(jnp.array(reordered_qkvz))},
      "in_proj_ba": {"kernel": nnx.Param(jnp.array(reordered_ba))},
      "conv1d": {"kernel": nnx.Param(jnp.array(conv1d_weight_jax))},
      "A_log": nnx.Param(jnp.array(pt_gdn.A_log.detach().numpy())),
      "dt_bias": nnx.Param(jnp.array(pt_gdn.dt_bias.detach().numpy())),
      "norm": {"weight": nnx.Param(jnp.array(pt_gdn.norm.weight.detach().numpy()))},
      "out_proj": {"kernel": nnx.Param(jnp.array(pt_gdn.out_proj.weight.T.detach().numpy()))},
  }
  nnx.update(jax_gdn, params)


def copy_full_attention(pt_attn, jax_attention, cfg):
  """Copies FullAttention weights into the JAX ``attentions.Attention`` submodule."""
  sd = pt_attn.state_dict()

  pt_q = sd["q_proj.weight"].T.numpy().reshape(cfg.emb_dim, cfg.num_query_heads, cfg.head_dim * 2)
  nnx.update(jax_attention.query, {"kernel": nnx.Param(jnp.array(pt_q))})

  pt_k = sd["k_proj.weight"].T.numpy().reshape(cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)
  nnx.update(jax_attention.key, {"kernel": nnx.Param(jnp.array(pt_k))})

  pt_v = sd["v_proj.weight"].T.numpy().reshape(cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)
  nnx.update(jax_attention.value, {"kernel": nnx.Param(jnp.array(pt_v))})

  nnx.update(jax_attention.out, {"kernel": nnx.Param(jnp.array(sd["o_proj.weight"].T.numpy()))})

  if jax_attention.query_norm is not None:
    nnx.update(jax_attention.query_norm, {"weight": nnx.Param(jnp.array(sd["q_norm.weight"].numpy()))})
  if jax_attention.key_norm is not None:
    nnx.update(jax_attention.key_norm, {"weight": nnx.Param(jnp.array(sd["k_norm.weight"].numpy()))})


def copy_sparse_moe(pt_moe, jax_moe, cfg):
  """Copies SparseMoE (routed + shared experts + gates) weights, transposing as needed."""
  pt_experts = pt_moe.experts
  stacked_gate_proj = torch.stack([e.gate_proj.weight.T for e in pt_experts])
  stacked_up_proj = torch.stack([e.up_proj.weight.T for e in pt_experts])
  stacked_down_proj = torch.stack([e.down_proj.weight.T for e in pt_experts])

  jax_params = {
      "routed_experts": {
          "gate": {"kernel": nnx.Param(jnp.array(pt_moe.gate.weight.T.detach().numpy()))},
          "wi_0": nnx.Param(jnp.array(stacked_gate_proj.detach().numpy())),
          "wi_1": nnx.Param(jnp.array(stacked_up_proj.detach().numpy())),
          "wo": nnx.Param(jnp.array(stacked_down_proj.detach().numpy())),
      },
      "shared_expert": {
          "wi": {
              "0": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.gate_proj.weight.T.detach().numpy()))},
              "1": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.up_proj.weight.T.detach().numpy()))},
          },
          "wo": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.down_proj.weight.T.detach().numpy()))},
      },
      "shared_expert_gate": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert_gate.weight.T.detach().numpy()))},
  }
  if not cfg.fused_mlp:
    jax_params["shared_expert"] = {
        "wi_0": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.gate_proj.weight.T.detach().numpy()))},
        "wi_1": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.up_proj.weight.T.detach().numpy()))},
        "wo": {"kernel": nnx.Param(jnp.array(pt_moe.shared_expert.down_proj.weight.T.detach().numpy()))},
    }
  nnx.update(jax_moe, jax_params)


# ----------------------------------------------------------------------
# PyTorch reference for a single Qwen3.5 decoder layer (mirrors Qwen3_5DecoderLayer).
# ----------------------------------------------------------------------
class Qwen3_5DecoderLayerReference:
  """Assembles the reused ``_PT`` modules into one hybrid decoder layer.

  This mirrors ``Qwen3_5DecoderLayer.__call__``: input norm -> attention (GDN or full) ->
  residual -> post-attention norm -> MoE -> residual. It is deliberately *not* an
  ``nn.Module`` subclass; it just holds the reference submodules and runs them.
  """

  def __init__(self, cfg, layer_idx, pt_gdn_moe_cfg, pt_attn_cfg):
    self.is_full_attention = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    self.input_layernorm = Qwen3NextRMSNorm_PT(cfg.emb_dim, epsilon=cfg.normalization_layer_epsilon).eval()
    self.post_attention_layernorm = Qwen3NextRMSNorm_PT(cfg.emb_dim, epsilon=cfg.normalization_layer_epsilon).eval()
    self.mlp = Qwen3NextSparseMoeBlock_PT(pt_gdn_moe_cfg).eval()

    if self.is_full_attention:
      self.attention = Qwen3NextFullAttention_PT(pt_attn_cfg, layer_idx=layer_idx).eval()
      self.rotary_emb = Qwen3NextRotaryEmbedding_PT(pt_attn_cfg)
    else:
      self.attention = Qwen3NextGatedDeltaNet_PT(pt_gdn_moe_cfg).eval()
      # The GDN reference reads a couple of fields off ``.config`` internally.
      self.attention.config = cfg
      self.rotary_emb = None

  def forward(self, hidden_states, position_ids=None, attention_mask=None):
    """Runs the reference decoder layer forward pass."""
    residual = hidden_states
    normed = self.input_layernorm(hidden_states)
    if self.is_full_attention:
      cos, sin = self.rotary_emb(normed, position_ids)
      attn_out = self.attention(normed, (cos, sin), attention_mask=attention_mask)
    else:
      attn_out = self.attention(normed)
    hidden_states = residual + attn_out

    residual = hidden_states
    normed = self.post_attention_layernorm(hidden_states)
    mlp_out, _ = self.mlp(normed)
    return residual + mlp_out

  def copy_weights_into(self, jax_layer, cfg):
    """Copies this reference's weights into a matching JAX ``Qwen3_5DecoderLayer``."""
    copy_rmsnorm(self.input_layernorm, jax_layer.input_layernorm)
    copy_rmsnorm(self.post_attention_layernorm, jax_layer.post_attention_layernorm)
    copy_sparse_moe(self.mlp, jax_layer.mlp, cfg)
    if self.is_full_attention:
      # jax_layer.attention is Qwen3_5FullAttention; its .attention is attentions.Attention.
      copy_full_attention(self.attention, jax_layer.attention.attention, cfg)
    else:
      copy_gated_delta_net(self.attention, jax_layer.attention, cfg)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestQwen3_5TextDecoder(unittest.TestCase):
  """Validates the Qwen3.5 text-decoder composition against PyTorch references."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    np.random.seed(0)
    self.cfg = build_test_config(cycle_interval=4)
    self.pt_gdn_moe_cfg = make_pt_gdn_moe_config(self.cfg)
    self.pt_attn_cfg = make_pt_attention_config(self.cfg)

    devices = np.array(jax.devices())
    num_devices = len(devices)
    self.batch_size = max(8, num_devices)
    self.seq_len = 128
    self.hidden_size = self.cfg.emb_dim

    mesh_shape = [1] * len(self.cfg.mesh_axes)
    mesh_shape[self.cfg.mesh_axes.index("data")] = num_devices
    self.mesh = Mesh(devices.reshape(mesh_shape), self.cfg.mesh_axes)
    self.nnx_rngs = nnx.Rngs(jax.random.PRNGKey(0))

  def _make_inputs(self):
    """Shared random inputs (PT + JAX views) plus causal mask / segment ids / positions."""
    hidden_np = np.random.randn(self.batch_size, self.seq_len, self.hidden_size).astype(np.float32)
    hidden_pt = torch.from_numpy(hidden_np)
    hidden_jax = jnp.array(hidden_np)

    position_ids_pt = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).repeat(self.batch_size, 1)
    positions_jax = jnp.array(position_ids_pt.numpy())
    segment_ids_jax = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    attn_mask_pt = create_causal_mask_PT(self.seq_len, self.seq_len)[None, None, :, :]
    return hidden_pt, hidden_jax, position_ids_pt, positions_jax, segment_ids_jax, attn_mask_pt

  def _run_decoder_layer(self, layer_idx):
    """Builds a JAX Qwen3_5DecoderLayer at ``layer_idx`` and compares to the PT reference."""
    ref = Qwen3_5DecoderLayerReference(self.cfg, layer_idx, self.pt_gdn_moe_cfg, self.pt_attn_cfg)

    jax_layer = Qwen3_5DecoderLayer(
        config=self.cfg,
        mesh=self.mesh,
        model_mode="train",
        layer_idx=layer_idx,
        quant=None,
        rngs=self.nnx_rngs,
    )
    ref.copy_weights_into(jax_layer, self.cfg)

    hidden_pt, hidden_jax, position_ids_pt, positions_jax, segment_ids_jax, attn_mask_pt = self._make_inputs()

    expected = ref.forward(hidden_pt, position_ids=position_ids_pt, attention_mask=attn_mask_pt).numpy()

    @jax.jit
    def run_jax(x, seg, pos):
      out, _ = jax_layer(x, seg, pos, True, "train")
      return out

    actual = np.asarray(run_jax(hidden_jax, segment_ids_jax, positions_jax))

    self.assertEqual(expected.shape, actual.shape, "Decoder-layer output shape mismatch")
    np.testing.assert_allclose(
        expected,
        actual,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Qwen3_5DecoderLayer (layer_idx={layer_idx}) does not match PyTorch reference!",
    )

  def test_decoder_layer_full_attention(self):
    """layer_idx=3 -> (3+1) % 4 == 0 -> full-attention branch."""
    self.assertTrue((3 + 1) % self.cfg.inhomogeneous_layer_cycle_interval == 0)
    self._run_decoder_layer(layer_idx=3)

  def test_decoder_layer_gated_delta_net(self):
    """layer_idx=0 -> GatedDeltaNet (linear-attention) branch."""
    self.assertFalse((0 + 1) % self.cfg.inhomogeneous_layer_cycle_interval == 0)
    self._run_decoder_layer(layer_idx=0)

  def test_scannable_block_layer_selection(self):
    """Structural check: the block's layer cycle selects full-attention only at the Nth layer."""
    block = Qwen3_5ScannableBlock(
        config=self.cfg,
        mesh=self.mesh,
        model_mode="train",
        quant=None,
        rngs=self.nnx_rngs,
    )
    interval = self.cfg.inhomogeneous_layer_cycle_interval
    for i in range(interval):
      layer = getattr(block, f"layer_{i}")
      is_full = (i + 1) % interval == 0
      expected_cls = Qwen3_5FullAttention if is_full else Qwen3_5GatedDeltaNet
      self.assertIsInstance(
          layer.attention,
          expected_cls,
          f"layer_{i} attention type mismatch: expected {expected_cls.__name__}",
      )

  def test_scannable_block_matches_reference(self):
    """Numerical check of the full hybrid cycle: 3 GDN layers then 1 full-attention layer."""
    interval = self.cfg.inhomogeneous_layer_cycle_interval
    block = Qwen3_5ScannableBlock(
        config=self.cfg,
        mesh=self.mesh,
        model_mode="train",
        quant=None,
        rngs=self.nnx_rngs,
    )

    # Build one PT reference layer per cycle index and copy its weights into the block.
    refs = []
    for i in range(interval):
      ref = Qwen3_5DecoderLayerReference(self.cfg, i, self.pt_gdn_moe_cfg, self.pt_attn_cfg)
      ref.copy_weights_into(getattr(block, f"layer_{i}"), self.cfg)
      refs.append(ref)

    hidden_pt, hidden_jax, position_ids_pt, positions_jax, segment_ids_jax, attn_mask_pt = self._make_inputs()

    # Reference: run the layers sequentially, matching Qwen3_5ScannableBlock.__call__.
    x_pt = hidden_pt
    for ref in refs:
      x_pt = ref.forward(x_pt, position_ids=position_ids_pt, attention_mask=attn_mask_pt)
    expected = x_pt.numpy()

    @jax.jit
    def run_jax(x, seg, pos):
      out, _ = block(x, seg, pos, True, "train")
      return out

    actual = np.asarray(run_jax(hidden_jax, segment_ids_jax, positions_jax))

    self.assertEqual(expected.shape, actual.shape, "ScannableBlock output shape mismatch")
    np.testing.assert_allclose(
        expected,
        actual,
        rtol=2e-4,
        atol=2e-4,
        err_msg="Qwen3_5ScannableBlock hybrid cycle does not match PyTorch reference!",
    )


if __name__ == "__main__":
  unittest.main()
