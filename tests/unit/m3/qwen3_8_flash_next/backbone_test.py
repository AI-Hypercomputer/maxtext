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

"""Unit tests for Qwen3.8-Flash-Next backbone, scannable cycles, and registry."""

from typing import Any
import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.m3 import models as m3_models  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import scannable  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
import numpy as np
import pytest

torch: Any
configuration_qwen4_exp: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import configuration_qwen4_exp  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  torch = None
  configuration_qwen4_exp = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


def _get_mini_config(
    scan_layers: bool = False,
    dtype: str = "float32",
    weight_dtype: str = "float32",
    **kwargs,
):
  """Builds a test config initialized from qwen3.8-flash-next-mini.yml."""
  return pyconfig.initialize(
      [None, f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml"],
      model_name="qwen3.8-flash-next-mini",
      override_model_config=True,
      scan_layers=scan_layers,
      packing=False,
      dtype=dtype,
      weight_dtype=weight_dtype,
      run_name="backbone_test",
      **kwargs,
  )


def _copy_hf_weights_to_jax(hf_model, jax_model):
  """Copies weights from PyTorch Qwen4ExpForCausalLM to Qwen3_8FlashNextModel."""
  d, pm = jax_model.decoder, hf_model.model

  def to_j(t, tr=False):
    return jnp.array(t.detach().cpu().numpy().T if tr else t.detach().cpu().numpy())

  def cp(dst, src, *keys, tr=True, attr="kernel"):
    for k in keys:
      getattr(getattr(dst, k), attr)[...] = to_j(getattr(src, k).weight, tr)

  d.token_embedder.embedding[...] = to_j(pm.embed_tokens.weight)
  jax_model.lm_head.kernel[...] = to_j(hf_model.lm_head.weight, True)
  cp(d.hyper_connection_mixer, pm.hyper_connection_mixer, "input_mix_weight_down", "input_mix_weight_up")
  cp(d.hyper_connection_mixer, pm.hyper_connection_mixer, "hc_norm", tr=False, attr="scale")
  for i in range(d.num_layers):
    jl, pl = getattr(d, f"layers_{i}"), pm.layers[i]
    for jh, ph in (
        (jl.attn_hyper_connection, pl.attn_hyper_connection),
        (jl.mlp_hyper_connection, pl.mlp_hyper_connection),
    ):
      cp(jh, ph, "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
      cp(jh, ph, "hc_norm", tr=False, attr="scale")
    if jl.ple is not None:
      jl.ple.ple_embedding.ngram_embedding.embedding[...] = to_j(pl.ple.ple_embedding.ngram_embedding.weight)
      jl.ple.conv1d.kernel[...] = jnp.array(pl.ple.conv1d.weight.detach().cpu().permute(2, 1, 0).numpy())
      cp(jl.ple, pl.ple, "norm_key", "norm_query", "norm_conv", tr=False, attr="scale")
      cp(jl.ple, pl.ple, "key_proj", "value_proj")
    if jl.layer_type == "linear_attention":
      cp(jl.linear_attn, pl.linear_attn, "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
      jl.linear_attn.A_log[...], jl.linear_attn.dt_bias[...] = to_j(pl.linear_attn.A_log), to_j(pl.linear_attn.dt_bias)
      cp(jl.linear_attn, pl.linear_attn, "norm", tr=False, attr="scale")
      jl.linear_attn.conv1d.kernel[...] = jnp.array(pl.linear_attn.conv1d.weight.detach().cpu().permute(2, 1, 0).numpy())
    else:
      cp(jl.self_attn, pl.self_attn, "q_proj", "k_proj", "v_proj", "o_proj")
      cp(jl.self_attn, pl.self_attn, "q_norm", "k_norm", tr=False, attr="scale")
      cp(jl.self_attn.indexer, pl.self_attn.indexer, "q_layernorm", "k_layernorm", tr=False, attr="scale")
      cp(jl.self_attn.indexer, pl.self_attn.indexer, "index_qk_proj")
    for d_l, w in (
        (jl.mlp.gate.linear, pl.mlp.gate.weight),
        (jl.mlp.shared_expert_gate, pl.mlp.shared_expert_gate.weight),
    ):
      d_l.kernel[...] = to_j(w, True)
    for k in ("gate_up_proj", "down_proj"):
      getattr(jl.mlp.experts, k)[...] = jnp.swapaxes(to_j(getattr(pl.mlp.experts, k)), -1, -2)
    cp(jl.mlp.shared_expert, pl.mlp.shared_expert, "gate_proj", "up_proj", "down_proj")


def _make_matching_hf_config(c) -> Any:
  """Constructs matching HuggingFace configuration from MaxText mini config."""
  if configuration_qwen4_exp is None:
    raise RuntimeError("transformers qwen4_exp not available")
  l = int(c.num_decoder_layers)
  rp = {
      "rope_theta": float(c.rope_max_timescale),
      "partial_rotary_factor": float(c.partial_rotary_factor),
      "mrope_interleaved": True,
      "mrope_section": list(c.mrope_section),
  }
  return configuration_qwen4_exp.Qwen4ExpTextConfig(
      vocab_size=int(c.vocab_size),
      hidden_size=int(c.emb_dim),
      num_hidden_layers=l,
      head_dim=int(c.head_dim),
      num_attention_heads=int(c.num_query_heads),
      num_key_value_heads=int(c.num_kv_heads),
      hc_count=int(c.hc_count),
      shared_expert_intermediate_size=int(c.mlp_dim),
      moe_intermediate_size=int(c.moe_mlp_dim),
      attention_bias=False,
      num_experts=int(c.num_experts),
      num_experts_per_tok=int(c.num_experts_per_tok),
      hc_lowrank=int(c.hc_lowrank),
      norm_topk_prob=bool(c.norm_topk_prob),
      output_gate_type=str(c.gdn_output_gate_activation),
      attention_dropout=0.0,
      layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (l // 4),
      rope_parameters=rp,
      ple_layer_ids=[2],
      linear_num_key_heads=int(c.gdn_num_key_heads),
      linear_num_value_heads=int(c.gdn_num_value_heads),
      linear_key_head_dim=int(c.gdn_key_head_dim),
      linear_value_head_dim=int(c.gdn_value_head_dim),
      linear_conv_kernel_dim=int(c.gdn_conv_kernel_dim),
      indexer_compress_ratio=int(c.index_select_block_size),
      indexer_n_heads=int(c.index_n_heads),
      indexer_kv_heads=int(c.index_kv_heads),
      indexer_head_dim=int(c.index_head_dim),
      indexer_budget=int(c.index_select_budget),
      rms_norm_eps=float(c.normalization_layer_epsilon),
      seed=int(c.ple_seed),
      ple_embed_dim=int(c.ple_embed_dim),
      ple_conv_kernel_size=int(c.ple_conv_kernel_size),
      ngram_size=int(c.ngram_size),
      heads_per_ngram=int(c.heads_per_ngram),
      ngram_vocab_size_base=int(c.ngram_vocab_size_base),
      eos_token_id=int(c.ple_eos_token_id),
      make_ngram_vocab_size_divisible_by=int(c.make_ngram_vocab_size_divisible_by),
  )


@pytest.mark.cpu_only
class Qwen3_8BackboneTest(absltest.TestCase):
  """Backbone tests for scanned/unscanned parity and PyTorch reference match."""

  def test_scanned_vs_unscanned_parity_2_cycles(self):
    """Verifies scanned vs unscanned parity, 2-segment inputs, and registry dispatch."""
    cu, cs = (_get_mini_config(scan_layers=s, base_num_decoder_layers=8) for s in (False, True))
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    self.assertIsInstance(mu, modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel)
    self.assertTrue({"qwen3.8-flash-next", "qwen3.8-flash-next-mini"}.issubset(m3_models.MODEL_REGISTRY))
    scannable.copy_unscanned_to_scanned(mu, ms)
    toks = jnp.array([[3, 17, 42, 88, 3, 50, 61, 99]], dtype=jnp.int32)
    pos, cmask = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (1, 8)), jnp.ones((1, 8), dtype=jnp.int32)
    (lu, au), (ls, as_) = (m(toks, decoder_positions=pos, decoder_segment_ids=cmask, return_aux=True) for m in (mu, ms))
    np.testing.assert_allclose(lu, ls, atol=1e-5, rtol=1e-5)
    jax.tree.map(lambda u, s: np.testing.assert_allclose(u, s, atol=1e-5, rtol=1e-5), au, as_)
    s2 = cmask.at[:, 4:].set(2)
    np.testing.assert_allclose(mu(toks, decoder_segment_ids=s2), ms(toks, decoder_segment_ids=s2), atol=1e-5, rtol=1e-5)

    logits_hs, pre_mixer = mu(toks, decoder_positions=pos, decoder_segment_ids=cmask, return_hidden_states=True)
    self.assertEqual(pre_mixer.shape, (1, 8, int(cu.hc_count) * int(cu.emb_dim)))
    np.testing.assert_allclose(
        mu.lm_head(mu.decoder.hyper_connection_mixer.mix_input(pre_mixer)), logits_hs, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(logits_hs, lu, atol=1e-5, rtol=1e-5)

  def test_scanned_vs_unscanned_bfloat16_parity(self):
    """Verifies scanned vs unscanned parity under bfloat16."""
    cu = _get_mini_config(
        scan_layers=False,
        base_num_decoder_layers=4,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        num_experts=2,
        num_experts_per_tok=2,
        index_select_budget=64,
    )
    cs = _get_mini_config(
        scan_layers=True,
        base_num_decoder_layers=4,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        num_experts=2,
        num_experts_per_tok=2,
        index_select_budget=64,
    )
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    scannable.copy_unscanned_to_scanned(mu, ms)
    toks = jnp.array([[3, 17, 42, 88]], dtype=jnp.int32)
    pos, cmask = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (1, 4)), jnp.ones((1, 4), dtype=jnp.int32)
    out_u = mu(toks, decoder_positions=pos, decoder_segment_ids=cmask)
    out_s = ms(toks, decoder_positions=pos, decoder_segment_ids=cmask)
    self.assertEqual(out_u.dtype, jnp.bfloat16)
    self.assertEqual(out_s.dtype, jnp.bfloat16)
    self.assertTrue(jnp.all(jnp.isfinite(out_u)))
    self.assertTrue(jnp.all(jnp.isfinite(out_s)))
    np.testing.assert_allclose(out_u, out_s, atol=1e-1, rtol=1e-1)

  def test_scanned_cycles_fail_fast_validation(self):
    """Verifies fail-fast validation in scanned cycle construction."""
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, ple_layer_id=2)
      m3_models.create_model(c, mesh=None)
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, inhomogeneous_layer_cycle_interval=2)
      m3_models.create_model(c, mesh=None)
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, base_num_decoder_layers=6)
      m3_models.create_model(c, mesh=None)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_full_model_weight_copy_parity(self):
    """Verifies fp32 forward parity against HuggingFace Qwen4ExpForCausalLM."""
    self.assertIsNotNone(torch)
    self.assertIsNotNone(modeling_qwen4_exp)
    self.assertIsNotNone(configuration_qwen4_exp)
    cu, cs = (_get_mini_config(scan_layers=s, base_num_decoder_layers=8) for s in (False, True))
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    torch.manual_seed(0)
    hf_model = modeling_qwen4_exp.Qwen4ExpForCausalLM(_make_matching_hf_config(cu)).eval()
    with torch.no_grad():
      for layer in hf_model.model.layers:
        if hasattr(layer, "ple") and layer.ple is not None:
          layer.ple.conv1d.weight.normal_(std=0.02)
      for mod in hf_model.modules():
        if type(mod).__name__.endswith("RMSNorm") and hasattr(mod, "weight") and mod.weight is not None:
          mod.weight.normal_(std=0.02)
    _copy_hf_weights_to_jax(hf_model, mu)
    scannable.copy_unscanned_to_scanned(mu, ms)
    input_ids = np.array([[10, 20, 3, 40, 50, 60, 3, 70], [5, 3, 15, 25, 35, 45, 55, 65]], dtype=np.int64)
    hf_masks = {}
    for idx in (3, 7):
      hf_model.model.layers[idx].self_attn.indexer.register_forward_hook(
          lambda _, __, o, i=idx: hf_masks.__setitem__(i, o)
      )
    pt_out = hf_model(torch.from_numpy(input_ids), output_router_logits=True)
    (lu, au), (ls, as_) = (m(jnp.array(input_ids, dtype=jnp.int32), return_aux=True) for m in (mu, ms))
    for out in (lu, ls):
      np.testing.assert_allclose(np.array(out), pt_out.logits.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)
    for i, topk in enumerate(
        torch.topk(l.view(2, 8, -1), k=2, dim=-1).indices.cpu().numpy() for l in pt_out.router_logits
    ):
      for aux in (au, as_):
        np.testing.assert_array_equal(np.array(aux[i]["router_indices"]), topk)
    for l_idx in (3, 7):
      mask, jax_b = hf_masks[l_idx].detach().cpu().numpy(), np.array(au[l_idx]["selected_block_indices"])
      mask = mask if mask.dtype == bool else (mask == 0.0)
      for b, q in ((b, q) for b in range(2) for q in range(3, 8)):
        hf_sel = {m for m in range((q + 1) // 4) if np.all(mask[b, 0, q, m * 4 : (m + 1) * 4])}
        self.assertEqual(hf_sel, {int(idx) for idx in jax_b[b, q] if idx >= 0})


if __name__ == "__main__":
  absltest.main()
