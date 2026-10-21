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

"""Unit tests for Qwen3.8-Flash-Next MTP draft layer and integrated loss module.

Reference: SGLang qwen4_exp_mtp.py L78-85 & L104-116 and
transformers.models.qwen4_exp.modeling_qwen4_exp.
"""
# pylint: disable=no-name-in-module

import types
from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.m3.models.qwen38_flash_next import loss
from maxtext.m3.models.qwen38_flash_next import modeling_qwen38_flash_next
from maxtext.m3.models.qwen38_flash_next import mtp
from maxtext.m3.models.qwen38_flash_next import scannable

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


def _create_mini_config(
    num_layers: int = 4, mtp_num_layers: int = 1, scan_layers: bool = False, dtype=jnp.float32, **kwargs: Any
) -> types.SimpleNamespace:
  """Creates a mini-twin config SimpleNamespace for fast MTP and loss unit testing."""
  base = {
      "emb_dim": 64,
      "num_decoder_layers": num_layers,
      "vocab_size": 128,
      "eos_token_id": 127,
      "ple_eos_token_id": 127,
      "hc_count": 4,
      "hc_lowrank": 16,
      "scan_layers": scan_layers,
      "dtype": dtype,
      "weight_dtype": dtype,
      "normalization_layer_epsilon": 1e-6,
      "inhomogeneous_layer_cycle_interval": 4,
      "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 4,
      "gdn_key_head_dim": 16,
      "gdn_value_head_dim": 16,
      "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 16,
      "num_query_heads": 4,
      "num_kv_heads": 1,
      "head_dim": 16,
      "partial_rotary_factor": 0.25,
      "rope_max_timescale": 1e4,
      "mrope_section": (1, 1, 0),
      "index_n_heads": 2,
      "index_kv_heads": 1,
      "index_head_dim": 16,
      "index_select_budget": 32,
      "index_select_block_size": 4,
      "moe_mlp_dim": 32,
      "mlp_dim": 32,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "norm_topk_prob": True,
      "routed_scaling_factor": 1.0,
      "use_ple_layer": False,
      "heads_per_ngram": 2,
      "ple_seed": 0,
      "ple_kernel_size": 3,
      "split_ngram_parts": 2,
      "ple_hash_seed": 0,
      "ngram_vocab_size_base": 500,
      "mtp_num_layers": mtp_num_layers,
      "mtp_num_streams": 4,
      "mtp_loss_scaling_factor": 0.1,
      "load_balance_loss_weight": 0.001,
      "gdn_output_gate_activation": "silu",
      "ple_layer_ids": [],
  }
  base.update(kwargs)
  return types.SimpleNamespace(**base)


def _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_layer, jx_mixer):
  """Copies decoder layer and mixer parameters from PyTorch to JAX modules."""

  def to_j(t, tr=False):
    return jnp.array(t.detach().cpu().numpy().T if tr else t.detach().cpu().numpy())

  def cp(dst, src, *keys, tr=True, attr="kernel"):
    for k in keys:
      getattr(getattr(dst, k), attr).value = to_j(getattr(src, k).weight, tr)

  for jh, ph in (
      (jx_layer.attn_hyper_connection, pt_layer.attn_hyper_connection),
      (jx_layer.mlp_hyper_connection, pt_layer.mlp_hyper_connection),
  ):
    cp(jh, ph, "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
    cp(jh, ph, "hc_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_proj", "k_proj", "v_proj", "o_proj")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_norm", "k_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "q_layernorm", "k_layernorm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "index_qk_proj")
  for d_l, w in (
      (jx_layer.mlp.gate.linear, pt_layer.mlp.gate.weight),
      (jx_layer.mlp.shared_expert_gate, pt_layer.mlp.shared_expert_gate.weight),
  ):
    d_l.kernel.value = to_j(w, True)
  for k in ("gate_up_proj", "down_proj"):
    getattr(jx_layer.mlp.experts, k).value = jnp.swapaxes(to_j(getattr(pt_layer.mlp.experts, k)), -1, -2)
  cp(jx_layer.mlp.shared_expert, pt_layer.mlp.shared_expert, "gate_proj", "up_proj", "down_proj")
  cp(jx_mixer, pt_mixer, "input_mix_weight_down", "input_mix_weight_up")
  cp(jx_mixer, pt_mixer, "hc_norm", tr=False, attr="scale")


@pytest.mark.cpu_only
class Qwen38MTPLossTest(absltest.TestCase):

  def test_mtp_block_shapes_and_parity(self):
    cfg = _create_mini_config()
    block = mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    self.assertEqual((block.pre_fc_norm_embedding.scale.shape, block.pre_fc_norm_hidden.scale.shape), ((64,), (256,)))
    self.assertEqual((block.fc_embedding.kernel.shape, block.fc_hidden.kernel.shape), ((64, 64), (64, 64)))
    b, s, c, d = 2, 8, 4, 64
    hc_hidden = jax.random.normal(jax.random.PRNGKey(1), (b, s, c * d))
    next_embeds = jax.random.normal(jax.random.PRNGKey(2), (b, s, d))
    out = block(hc_hidden, next_embeds)
    assert isinstance(out, jax.Array)
    self.assertEqual(out.shape, (b, s, d))
    e_proj = block.fc_embedding(block.pre_fc_norm_embedding(next_embeds))
    h_proj = block.fc_hidden(block.pre_fc_norm_hidden(hc_hidden).reshape(b, s, c, d))
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(b, s, c * d)
    ref_out = block.hyper_connection_mixer.mix_input(block.layers_0(h_mtp_in, return_aux=False))
    np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_weight_copy_parity(self):
    assert torch is not None and modeling_qwen4_exp is not None and configuration_qwen4_exp is not None
    cfg, (b, s, c, dim) = _create_mini_config(), (2, 4, 4, 64)
    rp = {
        "rope_theta": float(cfg.rope_max_timescale),
        "partial_rotary_factor": float(cfg.partial_rotary_factor),
        "mrope_interleaved": True,
        "mrope_section": list(cfg.mrope_section),
    }
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=dim,
        num_hidden_layers=4,
        head_dim=cfg.head_dim,
        num_attention_heads=cfg.num_query_heads,
        num_key_value_heads=cfg.num_kv_heads,
        hc_count=c,
        shared_expert_intermediate_size=cfg.mlp_dim,
        moe_intermediate_size=cfg.moe_mlp_dim,
        attention_bias=False,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        hc_lowrank=cfg.hc_lowrank,
        norm_topk_prob=cfg.norm_topk_prob,
        output_gate_type=cfg.gdn_output_gate_activation,
        attention_dropout=0.0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters=rp,
        ple_layer_ids=[],
        linear_num_key_heads=cfg.gdn_num_key_heads,
        linear_num_value_heads=cfg.gdn_num_value_heads,
        linear_key_head_dim=cfg.gdn_key_head_dim,
        linear_value_head_dim=cfg.gdn_value_head_dim,
        linear_conv_kernel_dim=cfg.gdn_conv_kernel_dim,
        indexer_compress_ratio=cfg.index_select_block_size,
        indexer_n_heads=cfg.index_n_heads,
        indexer_kv_heads=cfg.index_kv_heads,
        indexer_head_dim=cfg.index_head_dim,
        indexer_budget=cfg.index_select_budget,
        rms_norm_eps=cfg.normalization_layer_epsilon,
        seed=cfg.ple_seed,
    )
    torch.manual_seed(42)
    pt_norm_emb, pt_norm_hid = (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm(d, group_size=None, eps=1e-6) for d in (dim, c * dim)
    )
    pt_fc_emb, pt_fc_hid = (torch.nn.Linear(dim, dim, bias=False) for _ in range(2))
    pt_layer = modeling_qwen4_exp.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=3)
    pt_mixer = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False)

    with torch.no_grad():
      for p in (*pt_layer.parameters(), *pt_mixer.parameters()):
        p.normal_(std=0.02)

    jx_block = mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    jx_block.pre_fc_norm_embedding.scale.value = jnp.array(pt_norm_emb.weight.detach().cpu().numpy())
    jx_block.pre_fc_norm_hidden.scale.value = jnp.array(pt_norm_hid.weight.detach().cpu().numpy())
    jx_block.fc_embedding.kernel.value = jnp.array(pt_fc_emb.weight.detach().cpu().numpy().T)
    jx_block.fc_hidden.kernel.value = jnp.array(pt_fc_hid.weight.detach().cpu().numpy().T)
    _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_block.layers_0, jx_block.hyper_connection_mixer)

    pt_hc, pt_nxt = torch.randn(b, s, c * dim), torch.randn(b, s, dim)
    pt_fused = pt_fc_hid(pt_norm_hid(pt_hc).view(b, s, c, dim)) + pt_fc_emb(pt_norm_emb(pt_nxt)).unsqueeze(-2)
    pt_mtp_in = pt_fused.view(b, s, c * dim)
    cos, sin = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)(
        pt_mtp_in, torch.arange(s).unsqueeze(0).expand(b, -1)
    )
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    pt_out = pt_mixer(pt_layer(pt_mtp_in, position_embeddings=(cos, sin), attention_mask=pt_mask)).detach().cpu().numpy()

    jx_out = np.array(
        jx_block(
            jnp.array(pt_hc.numpy()),
            jnp.array(pt_nxt.numpy()),
            positions=jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s)),
        )
    )
    self.assertGreater(np.max(np.abs(pt_out)), 0.0)
    np.testing.assert_allclose(jx_out, pt_out, rtol=1e-5, atol=1e-5)

  def test_checkpoint_pinned_shape_contract(self):
    cfg = _create_mini_config(
        emb_dim=2560,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
        mrope_section=(11, 11, 10),
        index_head_dim=128,
        num_query_heads=24,
        num_kv_heads=2,
        index_n_heads=4,
        index_kv_heads=1,
        num_experts=512,
        num_experts_per_tok=10,
        moe_mlp_dim=640,
        mlp_dim=640,
        use_ple_layer=True,
    )
    shapes = nnx.eval_shape(lambda: mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0)))
    self.assertIsNone(shapes.layers_0.ple)
    self.assertEqual((shapes.fc_embedding.kernel.shape, shapes.fc_hidden.kernel.shape), ((2560, 2560), (2560, 2560)))
    self.assertEqual(
        (shapes.pre_fc_norm_embedding.scale.shape, shapes.pre_fc_norm_hidden.scale.shape), ((2560,), (10240,))
    )
    self.assertEqual(
        (
            shapes.hyper_connection_mixer.hc_norm.scale.shape,
            shapes.hyper_connection_mixer.input_mix_weight_down.kernel.shape,
            shapes.hyper_connection_mixer.input_mix_weight_up.kernel.shape,
        ),
        ((10240,), (10240, 320), (320, 10240)),
    )
    self.assertEqual(
        (shapes.layers_0.self_attn.q_norm.scale.shape, shapes.layers_0.self_attn.k_norm.scale.shape), ((256,), (256,))
    )
    self.assertEqual(
        (
            shapes.layers_0.self_attn.indexer.q_layernorm.scale.shape,
            shapes.layers_0.self_attn.indexer.k_layernorm.scale.shape,
        ),
        ((128,), (128,)),
    )
    self.assertEqual(
        (
            shapes.layers_0.mlp.experts.gate_up_proj.shape,
            shapes.layers_0.mlp.experts.down_proj.shape,
            shapes.layers_0.mlp.gate.linear.kernel.shape,
            shapes.layers_0.mlp.shared_expert_gate.kernel.shape,
            shapes.layers_0.mlp.shared_expert.gate_proj.kernel.shape,
            shapes.layers_0.mlp.shared_expert.up_proj.kernel.shape,
            shapes.layers_0.mlp.shared_expert.down_proj.kernel.shape,
        ),
        (
            (512, 2560, 1280),
            (512, 640, 2560),
            (2560, 512),
            (2560, 1),
            (2560, 640),
            (2560, 640),
            (640, 2560),
        ),
    )

  def test_phase_gating(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    self.assertIsNotNone(model.mtp)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, 8), 0, 128)
    self.assertEqual(model(tokens, return_mtp=False).shape, (2, 8, 128))
    logits, mtp_logits = model(tokens, return_mtp=True)
    self.assertEqual((logits.shape, mtp_logits.shape), ((2, 8, 128), (2, 8, 128)))
    logits_aux, aux_dict = model(tokens, return_mtp=True, return_aux=True)
    self.assertEqual(logits_aux.shape, (2, 8, 128))
    self.assertTrue({"layer_aux", "mtp_logits", "mtp_aux"}.issubset(aux_dict))
    self.assertIn("router_probs", aux_dict["mtp_aux"])

  def test_cross_entropy_loss(self):
    logits, targets = jnp.zeros((2, 4, 8), dtype=jnp.float32), jnp.zeros((2, 4), dtype=jnp.int32)
    np.testing.assert_allclose(loss.cross_entropy_loss(logits, targets), np.log(8.0), rtol=1e-5)
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_allclose(loss.cross_entropy_loss(logits, targets, mask=mask), np.log(8.0), rtol=1e-5)
    with self.assertRaises(ValueError):
      loss.cross_entropy_loss(jnp.array([]), jnp.array([]))
    with self.assertRaises(ValueError):
      loss.cross_entropy_loss(jnp.zeros((2, 4, 8)), jnp.zeros((2, 5), dtype=jnp.int32))
    with self.assertRaises(ValueError):
      loss.cross_entropy_loss(jnp.zeros((2, 4, 8)), jnp.zeros((2, 4), dtype=jnp.int32), mask=jnp.zeros((2, 5)))

  def test_router_load_balancing_loss(self):
    b, s, e = 2, 4, 4
    r_probs, r_idx = jnp.full((b, s, e), 1.0 / e, dtype=jnp.float32), jnp.tile(
        jnp.array([0, 1], dtype=jnp.int32), (b, s, 1)
    )
    aux = [{"router_probs": r_probs, "router_indices": r_idx}]
    lb_loss = loss.router_load_balancing_loss(aux, num_experts=e)
    self.assertTrue(jnp.isfinite(lb_loss) and lb_loss > 0.0)
    for bad_aux in (None, [], [{"bad": 1}]):
      with self.assertRaises(ValueError):
        loss.router_load_balancing_loss(bad_aux, 4)

  def test_router_load_balancing_loss_hf_equivalence(self):
    b, s, e, k = 2, 8, 4, 2
    rng = np.random.default_rng(42)
    layer_aux = []
    for _ in range(3):
      logits = rng.standard_normal((b, s, e)).astype(np.float32)
      probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
      indices = np.argsort(-probs, axis=-1)[:, :, :k].astype(np.int32)
      layer_aux.append({"router_probs": jnp.array(probs), "router_indices": jnp.array(indices)})
    mtp_logits = rng.standard_normal((b, s, e)).astype(np.float32)
    mtp_probs = np.exp(mtp_logits) / np.sum(np.exp(mtp_logits), axis=-1, keepdims=True)
    mtp_indices = np.argsort(-mtp_probs, axis=-1)[:, :, :k].astype(np.int32)
    mtp_aux = {"router_probs": jnp.array(mtp_probs), "router_indices": jnp.array(mtp_indices)}

    mask = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.float32)
    mtp_mask = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))

    aux_dict = {"layer_aux": layer_aux, "mtp_aux": mtp_aux}
    jax_loss = loss.router_load_balancing_loss(aux_dict, num_experts=e, mask=mask, mtp_mask=mtp_mask)

    items = [(lay, mask) for lay in layer_aux] + [(mtp_aux, mtp_mask)]
    tokens_sum = np.zeros(e, dtype=np.float64)
    probs_sum = np.zeros(e, dtype=np.float64)
    total_tokens = 0.0
    for l_aux, l_mask in items:
      rp = np.array(l_aux["router_probs"], dtype=np.float64)
      ri = np.array(l_aux["router_indices"])
      lm = np.array(l_mask, dtype=np.float64)
      one_hot = np.zeros((*ri.shape, e), dtype=np.float64)
      np.put_along_axis(one_hot, np.expand_dims(ri, -1), 1.0, axis=-1)
      tokens_sum += np.sum(one_hot * lm[:, :, None, None], axis=(0, 1, 2))
      probs_sum += np.sum(rp * lm[:, :, None], axis=(0, 1))
      total_tokens += np.sum(lm)

    ref_loss = e * np.sum((tokens_sum / total_tokens) * (probs_sum / total_tokens))
    np.testing.assert_allclose(float(jax_loss), ref_loss, rtol=1e-5, atol=1e-5)

  def test_compute_total_loss_and_gradients(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    inp, tgt = (jax.random.randint(jax.random.PRNGKey(k), (2, 16), 0, 128) for k in (4, 5))
    (total_l, metrics), grads = nnx.value_and_grad(
        lambda m: loss.compute_total_loss(m, inp, tgt, router_aux_coeff=0.001, mtp_loss_coeff=0.1, use_mtp=True),
        has_aux=True,
    )(model)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertTrue({"total_loss", "primary_loss", "router_loss", "mtp_loss"}.issubset(metrics))
    self.assertGreater(jnp.sum(jnp.abs(grads.lm_head.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.mtp.fc_embedding.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.decoder.layers_0.mlp.experts.gate_up_proj[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.decoder.layers_0.mlp.gate.linear.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.mtp.layers_0.mlp.gate.linear.kernel[...])), 0.0)

  def test_compute_total_loss_single_token(self):
    cfg = _create_mini_config(num_layers=4)
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jnp.array([[5]], dtype=jnp.int32)
    total_l, mets = loss.compute_total_loss(model, toks, toks, use_mtp=True)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertEqual(float(mets["mtp_loss"]), 0.0)

  def test_compute_total_loss_scanned(self):
    cfg_unscanned = _create_mini_config(num_layers=4, scan_layers=False)
    cfg_scanned = _create_mini_config(num_layers=4, scan_layers=True)
    unscanned_model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg_unscanned, rngs=nnx.Rngs(params=0))
    scanned_model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg_scanned, rngs=nnx.Rngs(params=42))
    scannable.copy_unscanned_to_scanned(unscanned_model, scanned_model)
    if unscanned_model.mtp is not None and scanned_model.mtp is not None:
      nnx.update(scanned_model.mtp, nnx.state(unscanned_model.mtp))
    toks = jax.random.randint(jax.random.PRNGKey(6), (2, 8), 0, 128)
    loss_u, mets_u = loss.compute_total_loss(unscanned_model, toks, toks, router_aux_coeff=0.001, use_mtp=True)
    loss_s, mets_s = loss.compute_total_loss(scanned_model, toks, toks, router_aux_coeff=0.001, use_mtp=True)
    self.assertTrue(jnp.isfinite(loss_s))
    np.testing.assert_allclose(loss_u, loss_s, rtol=1e-5, atol=1e-5)
    for k, val_u in mets_u.items():
      np.testing.assert_allclose(val_u, mets_s[k], rtol=1e-5, atol=1e-5)

  def test_packed_document_boundary_mtp_loss(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks, seg_ids = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32), jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)
    toks_modified = jnp.array([[10, 20, 99, 40]], dtype=jnp.int32)
    _, m1 = loss.compute_total_loss(model, toks, toks, segment_ids=seg_ids, use_mtp=True)
    _, m2 = loss.compute_total_loss(model, toks_modified, toks_modified, segment_ids=seg_ids, use_mtp=True)
    np.testing.assert_allclose(float(m1["mtp_loss"]), float(m2["mtp_loss"]), rtol=1e-5)
    _, u1 = loss.compute_total_loss(model, toks, toks, use_mtp=True)
    _, u2 = loss.compute_total_loss(model, toks_modified, toks_modified, use_mtp=True)
    self.assertNotEqual(float(u1["mtp_loss"]), float(u2["mtp_loss"]))

  def test_document_boundary_next_tokens_masking(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32)
    seg_ids = jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)
    logits, mtp_logits = model(toks, decoder_segment_ids=seg_ids, return_mtp=True)
    self.assertEqual(logits.shape, (1, 4, 128))
    self.assertEqual(mtp_logits.shape, (1, 4, 128))
    same_seg = (seg_ids != 0) & (seg_ids == jnp.pad(seg_ids[:, 1:], ((0, 0), (0, 1))))
    self.assertTrue(bool(same_seg[0, 0]))
    self.assertFalse(bool(same_seg[0, 1]))
    self.assertFalse(bool(same_seg[0, 2]))
    self.assertFalse(bool(same_seg[0, 3]))

  def test_validations_and_error_handling(self):
    with self.assertRaises(ValueError):
      modeling_qwen38_flash_next.Qwen38FlashNextModel(_create_mini_config(mtp_num_layers=2), rngs=nnx.Rngs(params=0))
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(_create_mini_config(), rngs=nnx.Rngs(params=0))
    toks = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    with self.assertRaises(ValueError):
      model(toks, return_mtp=True, return_hidden_states=True)

    block = mtp.Qwen38MTPBlock(_create_mini_config(), rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      block(jnp.zeros((2, 4, 100)), jnp.zeros((2, 4, 64)))
    with self.assertRaises(ValueError):
      block(jnp.zeros((2, 4, 256)), jnp.zeros((2, 4, 100)))

    cfg_no_aux = _create_mini_config()
    delattr(cfg_no_aux, "load_balance_loss_weight")
    model_no_aux = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg_no_aux, rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      loss.compute_total_loss(model_no_aux, toks, toks)

    cfg_no_mtp = _create_mini_config()
    delattr(cfg_no_mtp, "mtp_loss_scaling_factor")
    model_no_mtp = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg_no_mtp, rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      loss.compute_total_loss(model_no_mtp, toks, toks, router_aux_coeff=0.001)


if __name__ == "__main__":
  absltest.main()
