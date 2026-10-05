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

"""End-to-end composition smoke tests for the Kimi-K3 decoder (Track A).

These tests exercise the parts that the per-layer unit tests structurally cannot:

  1. `kimi-k3.yml` surviving the real Pydantic config pipeline. The g5 config test
     only `yaml.safe_load`s the file as a dict, so nothing previously proved that
     `decoder_block: "kimi_linear"` builds a `HyperParameters`.
  2. `KimiK3DecoderLayer` actually running a forward pass, threading the AttnRes
     `block_residual` highway across a heterogeneous KDA/MLA, Dense/MoE stack.
  3. The emitted parameter tree matching the `param_mapping.py` naming contract,
     checked against a really-built model rather than by inspection.
  4. Kimi-K3's MLA layers being NoPE (HF `mla_use_nope: true`).

The model is shrunk to toy dimensions; this is a wiring/shape test, not a parity
test. Numerical parity against the reference lives in the g1-g6 suites.
"""

import sys
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


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


def _toy_config():
  """Builds a real HyperParameters for kimi-k3 at toy scale."""
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      model_name="kimi-k3",
      # Shrinking a real model config to toy scale requires opting in explicitly.
      override_model_config=True,
      # Core dims.
      base_emb_dim=_EMB_DIM,
      base_num_decoder_layers=_NUM_LAYERS,
      base_num_query_heads=4,
      base_num_kv_heads=4,
      head_dim=32,
      vocab_size=256,
      # MLA low-rank dims.
      q_lora_rank=32,
      kv_lora_rank=16,
      qk_nope_head_dim=16,
      qk_rope_head_dim=8,
      v_head_dim=16,
      # MLP / MoE.
      base_mlp_dim=64,
      base_moe_mlp_dim=64,
      moe_intermediate_size=64,
      shared_intermediate_size=64,
      routed_expert_hidden_size=64,
      num_experts=8,
      num_experts_per_tok=2,
      # Sequence. Keep original == max so MLA's YaRN mscale rescaling stays off.
      max_target_length=_SEQ_LEN,
      max_position_embeddings=_SEQ_LEN,
      original_max_position_embeddings=_SEQ_LEN,
      max_prefill_predict_length=_SEQ_LEN // 2,
      per_device_batch_size=1,
      # Kimi-K3 is unscanned only for now.
      scan_layers=False,
      enable_checkpointing=False,
      attn_res_block_size=_ATTN_RES_BLOCK_SIZE,
  )


class KimiK3CompositionTest(unittest.TestCase):
  """Smoke tests for the composed Kimi-K3 decoder."""

  def test_config_builds_through_pydantic_pipeline(self):
    """`kimi-k3.yml` must survive the real config pipeline, not just yaml.safe_load."""
    cfg = _toy_config()
    self.assertEqual(cfg.decoder_block, DecoderBlockType.KIMI_LINEAR)
    self.assertEqual(cfg.num_decoder_layers, _NUM_LAYERS)
    self.assertEqual(cfg.first_num_dense_layers, 1)
    self.assertEqual(cfg.attn_res_block_size, _ATTN_RES_BLOCK_SIZE)
    # Kimi ships no rope_scaling; the mscale softmax rescale must stay disabled.
    self.assertFalse(cfg.max_position_embeddings > cfg.original_max_position_embeddings)

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

  def test_decoder_composes_expected_layer_types(self):
    """Each layer index must get the right attention and MLP variant."""
    cfg = _toy_config()
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    decoder = model.decoder

    for i in range(_NUM_LAYERS):
      layer = getattr(decoder, f"layers_{i}")
      with self.subTest(layer=i):
        self.assertEqual(layer.is_linear_attn, i not in _EXPECTED_MLA_LAYERS)
        self.assertEqual(layer.is_moe, i not in _EXPECTED_DENSE_LAYERS)

  def test_mla_layers_are_nope(self):
    """Kimi-K3 sets `mla_use_nope: true`; MLA must not apply rotary embeddings."""
    cfg = _toy_config()
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    for i in sorted(_EXPECTED_MLA_LAYERS):
      layer = getattr(model.decoder, f"layers_{i}")
      self.assertTrue(
          layer.self_attention.is_nope_layer,
          f"MLA layer {i} must be a NoPE layer",
      )

  def test_forward_pass_produces_finite_logits(self):
    """One full forward pass through the AttnRes highway."""
    cfg = _toy_config()
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    tokens = jnp.ones((1, _SEQ_LEN), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(_SEQ_LEN), tokens.shape)
    logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape[0], 1)
    self.assertEqual(logits.shape[1], _SEQ_LEN)
    self.assertEqual(logits.shape[-1], cfg.vocab_size)
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))), "logits contain NaN/Inf")

  def test_param_tree_matches_param_mapping_contract(self):
    """Emitted parameter paths must match what param_mapping.py expects."""
    cfg = _toy_config()
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    state = nnx.state(model, nnx.Param)

    def _key_str(p):
      if hasattr(p, "key"):
        return str(p.key)
      if hasattr(p, "name"):
        return str(p.name)
      if hasattr(p, "idx"):
        return str(p.idx)
      return str(p)

    paths = {"-".join(_key_str(p) for p in path) for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]}

    def _assert_prefix(expected):
      self.assertTrue(
          any(expected in p for p in paths),
          f"missing parameter matching {expected!r}",
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
        _assert_prefix(f"layers_{i}-{name}")

    # Decoder-level output AttnRes.
    _assert_prefix("output_attn_res_norm-scale")
    _assert_prefix("output_attn_res_proj-kernel")


if __name__ == "__main__":
  unittest.main()
