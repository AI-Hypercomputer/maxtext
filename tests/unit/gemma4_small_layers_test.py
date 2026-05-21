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

"""Torch-parity tests for Gemma 4 small (E2B / E4B) text-side layers.

Companion to ``gemma4_layers_test.py`` (which covers the vision side). Compares
the MaxText implementation of:

  - ``Gemma4SmallPLE`` against torch's ``Gemma4TextModel.get_per_layer_inputs`` /
    ``project_per_layer_inputs`` pipeline,
  - ``maxtext.layers.attentions.Attention`` (with and without ``share_kv_layer``)
    against torch's ``Gemma4TextAttention``,

using random-input forward passes with weights copied from torch.
"""

import os
import unittest

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.common.common_types import AttentionType
from maxtext.layers.attentions import Attention
from maxtext.models import gemma4_small
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from tests.utils.multimodal_test_utils import assert_all_close_jax_torch, copy_rmsnorm_weights

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention as TorchGemma4TextAttention,
    Gemma4TextModel as TorchGemma4TextModel,
    Gemma4TextRotaryEmbedding as TorchGemma4TextRotaryEmbedding,
)


torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Test config — small E2B-shaped variant, dimensions trimmed to keep tests fast.
# ---------------------------------------------------------------------------

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

# Small dimensions that still exercise GQA, KV sharing, partial RoPE.
_NUM_LAYERS = 6
_NUM_KV_SHARED = 3  # last 3 of 6 layers share K/V
_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 1
_HEAD_DIM = 32
_GLOBAL_HEAD_DIM = 64
_HIDDEN_SIZE = 128
_PLE_DIM = 32
_VOCAB = 256


def _build_jax_config():
  """Builds a MaxText config that mirrors the gemma4-e2b decoder block."""
  return pyconfig.initialize(
      ["", _BASE_CONFIG_PATH],
      model_name="gemma4-e2b",
      scan_layers=False,
      use_multimodal=False,
      override_model_config=True,
      # Override shapes for fast tests:
      base_num_decoder_layers=_NUM_LAYERS,
      base_num_query_heads=_NUM_Q_HEADS,
      base_num_kv_heads=_NUM_KV_HEADS,
      base_emb_dim=_HIDDEN_SIZE,
      base_mlp_dim=4 * _HIDDEN_SIZE,
      head_dim=_HEAD_DIM,
      global_head_dim=_GLOBAL_HEAD_DIM,
      vocab_size=_VOCAB,
      vocab_size_per_layer_input=_VOCAB,
      hidden_size_per_layer_input=_PLE_DIM,
      num_kv_shared_layers=_NUM_KV_SHARED,
      max_target_length=64,
      max_prefill_predict_length=8,
      attention="dot_product",  # avoid splash on CPU
      dtype="float32",
      weight_dtype="float32",
      float32_qk_product=True,
      float32_logits=True,
      matmul_precision="highest",
      dropout_rate=0.0,
  )


def _build_torch_text_config(jax_config):
  """Builds a torch ``Gemma4TextConfig`` that matches the JAX config shapes."""
  # Period-5 pattern → with 6 layers: [L, L, L, L, G, L]. We keep it simple and
  # reuse the e2b pattern; layer 4 is the GLOBAL layer in this test.
  # Put the only GLOBAL layer in the non-shared region so we can exercise a
  # global donor; the rest are sliding. With num_kv_shared=3 and 6 layers, the
  # non-shared region is indices 0..2.
  layer_types = ["sliding_attention"] * jax_config.num_decoder_layers
  layer_types[2] = "full_attention"
  return Gemma4TextConfig(
      hidden_size=jax_config.emb_dim,
      intermediate_size=jax_config.mlp_dim,
      num_hidden_layers=jax_config.num_decoder_layers,
      num_attention_heads=jax_config.num_query_heads,
      num_key_value_heads=jax_config.num_kv_heads,
      num_global_key_value_heads=jax_config.num_kv_heads,
      head_dim=jax_config.head_dim,
      global_head_dim=jax_config.global_head_dim,
      vocab_size=jax_config.vocab_size,
      vocab_size_per_layer_input=jax_config.vocab_size_per_layer_input,
      hidden_size_per_layer_input=jax_config.hidden_size_per_layer_input,
      num_kv_shared_layers=jax_config.num_kv_shared_layers,
      layer_types=layer_types,
      sliding_window=jax_config.sliding_window_size,
      max_position_embeddings=64,
      rms_norm_eps=jax_config.normalization_layer_epsilon,
      rope_parameters={
          "full_attention": {
              "partial_rotary_factor": jax_config.global_rope_proportion,
              "rope_theta": float(jax_config.global_rope_max_timescale),
              "rope_type": "proportional",
          },
          "sliding_attention": {
              "rope_theta": float(jax_config.local_rope_max_timescale),
              "rope_type": "default",
          },
      },
      attention_dropout=0.0,
      attention_bias=False,
      use_bidirectional_attention=None,
      attention_k_eq_v=False,
      use_double_wide_mlp=jax_config.use_double_wide_mlp,
      tie_word_embeddings=True,
  )


# ---------------------------------------------------------------------------
# Weight copying helpers (torch -> JAX).
# ---------------------------------------------------------------------------


def _copy_attention_weights(torch_attn, jax_attn, *, num_q_heads, num_kv_heads, head_dim, hidden_size):
  """Copy a torch Gemma4TextAttention's weights into a JAX ``Attention`` module."""
  q_w = torch_attn.q_proj.weight.detach().cpu().numpy()  # (num_q*hd, H)
  o_w = torch_attn.o_proj.weight.detach().cpu().numpy()  # (H, num_q*hd)
  jax_attn.query.kernel.value = jnp.asarray(q_w.T.reshape(hidden_size, num_q_heads, head_dim))
  jax_attn.out.kernel.value = jnp.asarray(o_w.T.reshape(num_q_heads, head_dim, hidden_size))
  copy_rmsnorm_weights(torch_attn.q_norm, jax_attn.query_norm)

  if not torch_attn.is_kv_shared_layer:
    k_w = torch_attn.k_proj.weight.detach().cpu().numpy()  # (num_kv*hd, H)
    v_w = torch_attn.v_proj.weight.detach().cpu().numpy()  # (num_kv*hd, H)
    jax_attn.key.kernel.value = jnp.asarray(k_w.T.reshape(hidden_size, num_kv_heads, head_dim))
    jax_attn.value.kernel.value = jnp.asarray(v_w.T.reshape(hidden_size, num_kv_heads, head_dim))
    copy_rmsnorm_weights(torch_attn.k_norm, jax_attn.key_norm)
    # torch v_norm has with_scale=False (no learnable param), so nothing to copy.


def _make_jax_attention(jax_config, mesh, *, attention_type, num_kv_heads, head_dim, share_kv_layer):
  """Construct a JAX ``Attention`` module configured like a gemma4-small layer."""
  seq_len = jax_config.max_target_length
  dummy_shape = (1, seq_len, jax_config.emb_dim)
  partial_rotary_factor = (
      jax_config.global_rope_proportion if attention_type == AttentionType.GLOBAL else jax_config.local_rope_proportion
  )
  rope_max_timescale = (
      jax_config.global_rope_max_timescale
      if attention_type == AttentionType.GLOBAL
      else jax_config.local_rope_max_timescale
  )
  return Attention(
      config=jax_config,
      num_query_heads=jax_config.num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      max_target_length=seq_len,
      max_prefill_predict_length=jax_config.max_prefill_predict_length,
      attention_kernel="dot_product",
      inputs_q_shape=dummy_shape,
      inputs_kv_shape=dummy_shape,
      mesh=mesh,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      dropout_rate=0.0,
      float32_qk_product=True,
      float32_logits=True,
      attention_type=attention_type,
      sliding_window_size=jax_config.sliding_window_size,
      use_qk_norm=True,
      use_v_norm=True,
      query_pre_attn_scalar=1.0,
      rope_max_timescale=rope_max_timescale,
      partial_rotary_factor=partial_rotary_factor,
      share_kv_layer=share_kv_layer,
      rngs=nnx.Rngs(0),
  )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class Gemma4SmallParityBase(unittest.TestCase):
  """Common config + mesh setup for Gemma 4 small parity tests."""

  @classmethod
  def setUpClass(cls):
    np.random.seed(42)
    torch.manual_seed(42)
    cls.jax_config = _build_jax_config()
    cls.torch_config = _build_torch_text_config(cls.jax_config)
    cls.torch_config._attn_implementation = "eager"  # pylint: disable=protected-access
    cls.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))


class TestGemma4SmallPLE(Gemma4SmallParityBase):
  """Parity for the Per-Layer-Embedding block."""

  def test_ple_matches_torch(self):
    torch_model = TorchGemma4TextModel(self.torch_config)
    torch_model.eval()

    jax_ple = gemma4_small.Gemma4SmallPLE(
        config=self.jax_config,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )

    # Copy torch weights into JAX (token-identity embedding, projection, norm).
    jax_ple.embed_tokens_per_layer.value = jnp.asarray(torch_model.embed_tokens_per_layer.weight.detach().cpu().numpy())
    jax_ple.per_layer_model_projection.kernel.value = jnp.asarray(
        torch_model.per_layer_model_projection.weight.detach().cpu().numpy().T
    )
    copy_rmsnorm_weights(torch_model.per_layer_projection_norm, jax_ple.per_layer_projection_norm)

    batch_size, seq_len = 2, 8
    input_ids_np = np.random.randint(0, self.torch_config.vocab_size_per_layer_input, size=(batch_size, seq_len))
    inputs_embeds_np = np.random.randn(batch_size, seq_len, self.torch_config.hidden_size).astype(np.float32)

    torch_pli = torch_model.get_per_layer_inputs(torch.from_numpy(input_ids_np).long(), None)
    torch_out = torch_model.project_per_layer_inputs(torch.from_numpy(inputs_embeds_np), torch_pli)

    jax_out = jax_ple(jnp.asarray(input_ids_np), jnp.asarray(inputs_embeds_np))

    assert_all_close_jax_torch(jax_out, torch_out, rtol=1e-4, atol=1e-4, error_msg="Gemma4SmallPLE output differs")


class TestGemma4SmallAttentionDonor(Gemma4SmallParityBase):
  """Parity for a non-shared (donor) gemma4-small attention layer."""

  def _run_for_layer(self, layer_idx, attention_type, head_dim):
    """Forward-parity check for a single non-shared attention layer."""
    torch_attn = TorchGemma4TextAttention(self.torch_config, layer_idx=layer_idx).eval()
    self.assertFalse(torch_attn.is_kv_shared_layer)

    jax_attn = _make_jax_attention(
        self.jax_config,
        self.mesh,
        attention_type=attention_type,
        num_kv_heads=self.jax_config.num_kv_heads,
        head_dim=head_dim,
        share_kv_layer=False,
    )
    _copy_attention_weights(
        torch_attn,
        jax_attn,
        num_q_heads=self.jax_config.num_query_heads,
        num_kv_heads=self.jax_config.num_kv_heads,
        head_dim=head_dim,
        hidden_size=self.jax_config.emb_dim,
    )

    batch_size, seq_len = 1, 8
    hidden_np = np.random.randn(batch_size, seq_len, self.jax_config.emb_dim).astype(np.float32)
    positions_np = np.broadcast_to(np.arange(seq_len), (batch_size, seq_len)).astype(np.int32)

    torch_rope = TorchGemma4TextRotaryEmbedding(self.torch_config).eval()
    cos, sin = torch_rope(
        torch.from_numpy(hidden_np), torch.from_numpy(positions_np).long(), layer_type=torch_attn.layer_type
    )
    # Make attention mask: lower-triangular for causal; torch eager expects float mask.
    mask = torch.full((batch_size, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1)
    torch_out, _ = torch_attn(
        hidden_states=torch.from_numpy(hidden_np),
        position_embeddings=(cos, sin),
        attention_mask=mask,
        shared_kv_states={},
    )

    jax_out, _ = jax_attn(
        jnp.asarray(hidden_np),
        jnp.asarray(hidden_np),
        inputs_positions=jnp.asarray(positions_np),
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
    )
    assert_all_close_jax_torch(
        jax_out, torch_out, rtol=1e-3, atol=1e-3, error_msg=f"Donor attn parity (layer {layer_idx}) differs"
    )

  def test_sliding_donor_matches_torch(self):
    # Layer 0 is a sliding (non-shared) layer.
    self._run_for_layer(0, AttentionType.LOCAL_SLIDING, self.jax_config.head_dim)

  def test_global_donor_matches_torch(self):
    # Layer 2 is the only GLOBAL layer in our 6-layer config (and is non-shared).
    self._run_for_layer(2, AttentionType.GLOBAL, self.jax_config.global_head_dim)


class TestGemma4SmallAttentionShared(Gemma4SmallParityBase):
  """Parity for a KV-shared (consumer) gemma4-small attention layer."""

  def test_shared_layer_matches_torch(self):
    # Donor for the shared sliding layers: last non-shared sliding layer.
    # In our 6-layer / 3-shared config, first_shared=3 and the non-shared
    # region is [L, L, G] (indices 0, 1, 2). Layer 1 is the sliding donor;
    # layer 3 is the first shared sliding layer.
    donor_idx, shared_idx = 1, 3
    head_dim = self.jax_config.head_dim
    attention_type = AttentionType.LOCAL_SLIDING

    # Build torch donor + shared layers and run forward to capture donor K/V.
    torch_donor = TorchGemma4TextAttention(self.torch_config, layer_idx=donor_idx).eval()
    torch_shared = TorchGemma4TextAttention(self.torch_config, layer_idx=shared_idx).eval()
    self.assertFalse(torch_donor.is_kv_shared_layer)
    self.assertTrue(torch_shared.is_kv_shared_layer)

    batch_size, seq_len = 1, 8
    hidden_np = np.random.randn(batch_size, seq_len, self.jax_config.emb_dim).astype(np.float32)
    positions_np = np.broadcast_to(np.arange(seq_len), (batch_size, seq_len)).astype(np.int32)

    torch_rope = TorchGemma4TextRotaryEmbedding(self.torch_config).eval()
    cos, sin = torch_rope(
        torch.from_numpy(hidden_np), torch.from_numpy(positions_np).long(), layer_type=torch_donor.layer_type
    )
    mask = torch.full((batch_size, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1)

    # Run torch donor first so it populates shared_kv_states for the consumer.
    shared_kv_states = {}
    torch_donor(
        hidden_states=torch.from_numpy(hidden_np),
        position_embeddings=(cos, sin),
        attention_mask=mask,
        shared_kv_states=shared_kv_states,
    )
    self.assertIn(torch_donor.layer_type, shared_kv_states)
    torch_shared_out, _ = torch_shared(
        hidden_states=torch.from_numpy(hidden_np),
        position_embeddings=(cos, sin),
        attention_mask=mask,
        shared_kv_states=shared_kv_states,
    )

    # Build JAX shared attention layer (share_kv_layer=True).
    jax_shared = _make_jax_attention(
        self.jax_config,
        self.mesh,
        attention_type=attention_type,
        num_kv_heads=self.jax_config.num_kv_heads,
        head_dim=head_dim,
        share_kv_layer=True,
    )
    # Copy Q-side weights (Q proj, Q norm, O proj) — K/V modules don't exist.
    _copy_attention_weights(
        torch_shared,
        jax_shared,
        num_q_heads=self.jax_config.num_query_heads,
        num_kv_heads=self.jax_config.num_kv_heads,
        head_dim=head_dim,
        hidden_size=self.jax_config.emb_dim,
    )

    # Translate torch donor K / V (shape [B, num_kv_heads, S, head_dim]) into
    # JAX layout ([B, S, num_kv_heads, head_dim]).
    donor_k, donor_v = shared_kv_states[torch_donor.layer_type]
    jax_shared_key = jnp.asarray(donor_k.detach().cpu().numpy().transpose(0, 2, 1, 3))
    jax_shared_value = jnp.asarray(donor_v.detach().cpu().numpy().transpose(0, 2, 1, 3))

    jax_shared_out, _ = jax_shared(
        jnp.asarray(hidden_np),
        jnp.asarray(hidden_np),
        inputs_positions=jnp.asarray(positions_np),
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        shared_key=jax_shared_key,
        shared_value=jax_shared_value,
    )

    assert_all_close_jax_torch(
        jax_shared_out, torch_shared_out, rtol=1e-3, atol=1e-3, error_msg="Shared-layer attn parity differs"
    )


if __name__ == "__main__":
  unittest.main()
