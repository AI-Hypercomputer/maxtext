# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tier 1 unit, parity, and invariant tests for GLM-5.3-Flash model and registry."""

# pylint: disable=protected-access
from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import os
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.utils import model_creation_utils
import numpy as np
import pytest

try:
  import torch
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

try:
  HAS_TOKAMAX_KDA = "TEST_SRCDIR" in os.environ or importlib.util.find_spec("tokamax._src.ops.kda") is not None
except (ImportError, ValueError):
  HAS_TOKAMAX_KDA = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TOKAMAX_KDA, reason="tokamax KDA not available"),
]


def _make_cfg(
    num_hidden_layers: int = 7,
    scan_layers: bool = True,
    dtype: str = "float32",
    weight_dtype: str = "float32",
):
  """Constructs matching mini Glm5Config and HF Glm5NextTextConfig."""
  kws = (
      "hidden_size=64 intermediate_size=128 moe_intermediate_size=32"
      " n_routed_experts=8 num_experts_per_tok=2 q_lora_rank=32 kv_lora_rank=16"
      " num_attention_heads=4 num_key_value_heads=4 qk_nope_head_dim=16"
      " v_head_dim=16 index_n_heads=8 index_head_dim=16 index_topk=8"
      " linear_num_key_heads=4 linear_num_value_heads=4"
      " linear_key_head_dim=16 linear_value_head_dim=16"
  )
  args = {k: int(v) for k, v in (s.split("=") for s in kws.split())}
  args.update(
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      kda_implementation="xla",
  )
  args.update(dtype=dtype, weight_dtype=weight_dtype, vocab_size=256)
  c = glm5_config.Glm5Config.mini_config(**args)
  hf_args = {x: args[x] for x in args if not x.startswith("linear_") and x != "qk_head_dim"}
  hf_k = (
      "vocab_size pad_token_id num_hidden_layers _attn_implementation"
      " index_kpool linear_num_heads linear_head_dim linear_conv_kernel_dim"
  ).split()
  hf_args.update(zip(hf_k, (256, 0, num_hidden_layers, "eager", 4, 4, 16, 4)))
  hf_args["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (num_hidden_layers - 3)
  hf_args["layer_types"], hf_args["indexer_types"] = list(c.layer_types), list(c.indexer_types)
  return c, (hf_cfg.Glm5NextTextConfig(**hf_args) if HAS_TORCH else None)


@contextlib.contextmanager
def _fp64_mode():
  """Context manager promoting float32 casts in JAX and Torch/HF to float64."""
  orig_jax_x64 = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(jnp, "float32", jnp.float64))
      if HAS_TORCH:
        stack.enter_context(mock.patch.object(torch, "float32", torch.float64))
        stack.enter_context(mock.patch.object(torch.Tensor, "float", torch.Tensor.double))
        stack.enter_context(
            mock.patch.object(
                hf_moe_integ,
                "_can_use_grouped_mm",
                lambda *args, **kwargs: False,
            )
        )
      yield
  finally:
    jax.config.update("jax_enable_x64", orig_jax_x64)


def _copy_hf_to_jax(hf_m, hf_head, jax_m, cfg):
  """Copies weights from HF Glm5NextTextModel and lm_head into JAX Glm5Model."""

  def _w(x):
    return jnp.array(x.detach().numpy())

  def _t(x):
    return _w(x).T

  def _set_proj(dst, src, names):
    for p in names:
      getattr(dst, f"{p}_proj").kernel.value = _t(getattr(src, f"{p}_proj").weight)

  def _copy_decoder(hf_l, jax_l):
    jax_l.attn_hc.fn.value, jax_l.ffn_hc.fn.value = _t(hf_l.attn_hc.fn), _t(hf_l.ffn_hc.fn)
    for hc, hf_hc in (
        (jax_l.attn_hc, hf_l.attn_hc),
        (jax_l.ffn_hc, hf_l.ffn_hc),
    ):
      hc.base.value, hc.scale.value = _w(hf_hc.base), _w(hf_hc.scale)
    jax_l.input_layernorm.weight.value = _w(hf_l.input_layernorm.weight)
    jax_l.post_attention_layernorm.weight.value = _w(hf_l.post_attention_layernorm.weight)
    if jax_l.is_moe:
      _set_proj(
          jax_l.mlp.shared_experts,
          hf_l.mlp.shared_experts,
          ("gate", "up", "down"),
      )
      jax_l.mlp.gate.weight.value = _t(hf_l.mlp.gate.weight)
      jax_l.mlp.gate.e_score_correction_bias.value = _w(hf_l.mlp.gate.e_score_correction_bias)
      hf_gu = _w(hf_l.mlp.experts.gate_up_proj)
      inter = cfg.moe_intermediate_size
      jax_l.mlp.experts.gate_up_proj.value = hf_gu.reshape(hf_gu.shape[0], 2, inter, -1).transpose(0, 3, 1, 2)
      jax_l.mlp.experts.down_proj.value = _w(hf_l.mlp.experts.down_proj).swapaxes(1, 2)
    else:
      _set_proj(jax_l.mlp, hf_l.mlp, ("gate", "up", "down"))
    hf_a, ja = hf_l.self_attn, jax_l.self_attn
    if jax_l.block_type == "linear_attention":
      _set_proj(ja, hf_a, ("q", "k", "v", "b", "o", "g_a", "g_b"))
      _set_proj(ja, hf_a.forget_gate, ("f_a", "f_b"))
      cw = _w(hf_a.conv1d.weight)
      k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
      for i, c in enumerate(("q", "k", "v")):
        getattr(ja, f"{c}_conv1d").weight.value = cw[i * k_dim : (i + 1) * k_dim]
      ja.A_log.value = _w(hf_a.forget_gate.A_log)
      ja.dt_bias.value = _w(hf_a.forget_gate.dt_bias)
      ja.o_norm.weight.value = _w(hf_a.o_norm.weight)
    else:
      _set_proj(ja, hf_a, ("q_a", "q_b", "kv_b", "o"))
      ja.kv_a_proj_with_mqa.kernel.value = _t(hf_a.kv_a_proj_with_mqa.weight)
      ja.q_a_layernorm.weight.value = _w(hf_a.q_a_layernorm.weight)
      ja.kv_a_layernorm.weight.value = _w(hf_a.kv_a_layernorm.weight)
      for p in ("wq_b", "wk", "weights_proj"):
        getattr(ja.indexer, p).kernel.value = _t(getattr(hf_a.indexer, p).weight)
      ja.indexer.k_norm.weight.value = _w(hf_a.indexer.k_norm.weight)
      ja.indexer.k_norm.bias.value = _w(hf_a.indexer.k_norm.bias)
      ja.indexer.index_kpool_compress_ape.value = _w(hf_a.indexer.index_kpool_compress_ape)
      ja.indexer.index_kpool_compress_gate.value = _w(hf_a.indexer.index_kpool_compress_gate)

  jax_m.embed_tokens.embedding.value, jax_m.norm.weight.value = _w(hf_m.embed_tokens.weight), _w(hf_m.norm.weight)
  jax_m.lm_head.kernel.value = _t(hf_head.weight)
  for l in range(cfg.num_hidden_layers):
    sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(l, cfg.first_k_dense_replace, cfg.num_hidden_layers)
    r = nnx.Rngs(0)
    if sec in ("prefix", "suffix"):
      _copy_decoder(hf_m.layers[l], getattr(jax_m, f"{sec}_layers")[b_idx])
    elif sec == "super_block":
      slot = getattr(jax_m.super_block, f"slot_{s_idx}")
      temp_l = modeling_glm5_3_flash.Glm5DecoderLayer(cfg, cfg.first_k_dense_replace + s_idx, rngs=r)
      _copy_decoder(hf_m.layers[l], temp_l)
      jax.tree.map(
          lambda tp, sp, idx=b_idx: setattr(sp, "value", sp.value.at[idx].set(tp.value)),
          nnx.state(temp_l),
          nnx.state(slot),
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5ModelTest(parameterized.TestCase):
  """Parity, scan/remat equivalence, and registry dispatch tests for Glm5Model."""

  def setUp(self):
    super().setUp()
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _assert_allclose(self, a, b, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)

  def _calc_e3(self, actual, expected, name: str = "", dtype=np.float32) -> tuple[float, float, float, float, float]:
    """Calculates E3 metrics (frob_rel, max_rel_inf, cos, norm_b, ref_inf) between actual and expected."""
    a, b = np.asarray(actual, dtype=dtype), np.asarray(expected, dtype=dtype)
    ref_inf, norm_a, norm_b = (
        float(np.max(np.abs(b))),
        float(np.linalg.norm(a)),
        float(np.linalg.norm(b)),
    )
    self.assertGreater(norm_a, 0.0, msg=f"{name}: actual norm is zero")
    self.assertGreater(norm_b, 0.0, msg=f"{name}: expected norm is zero")
    return (
        float(np.linalg.norm(a - b)) / norm_b,
        float(np.max(np.abs(a - b))) / max(1.0, ref_inf),
        float(np.sum(a * b) / (norm_a * norm_b)),
        norm_b,
        ref_inf,
    )

  def _assert_e3(self, a: np.ndarray, b: np.ndarray, name: str = "") -> tuple[float, float, float]:
    """Asserts parity between actual and expected using E3 metric suite."""
    frob_rel, max_rel_inf, cos, norm_b, ref_inf = self._calc_e3(a, b, name, dtype=np.float32)
    a_f32, b_f32 = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    np.testing.assert_allclose(
        a_f32,
        b_f32,
        atol=1e-5 * max(1.0, ref_inf),
        rtol=1e-5,
        err_msg=f"Mismatch in {name}",
    )
    diff_norm = float(np.linalg.norm(a_f32 - b_f32))
    self.assertLessEqual(
        diff_norm,
        1e-5 * max(1.0, norm_b),
        f"Frobenius mismatch in {name}: {diff_norm:.3e}",
    )
    self.assertGreater(cos, 0.99999, f"Cosine mismatch in {name}: {cos}")
    return (frob_rel, max_rel_inf, cos)

  def _run_parity_pass(
      self,
      num_layers: int,
      cfg: glm5_config.Glm5Config,
      hf_m: torch.nn.Module,
      hf_head: torch.nn.Module,
      ids: torch.Tensor,
      cot: torch.Tensor,
      *,
      is_fp64: bool = False,
  ):
    """Runs one forward and backward parity pass between HF and JAX models."""
    np_dt = np.float64 if is_fp64 else np.float32
    jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    _copy_hf_to_jax(hf_m, hf_head, jax_m, cfg)

    orig_kda = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    try:
      hf_logits = hf_head(hf_m(input_ids=ids).last_hidden_state)
      cot_jax = jnp.array(cot.detach().cpu().numpy())
      (hf_logits * cot).sum().backward()
    finally:
      hf_model.chunk_kimi_delta_attention = orig_kda

    tokens, np_hf = jnp.array(ids.numpy()), hf_logits.detach().cpu().numpy()
    out_aux = jax_m.forward_with_aux(tokens)
    l_frob, l_inf, l_cos, _, _ = self._calc_e3(out_aux.logits, np_hf, name="forward_logits", dtype=np_dt)

    wrt = nnx.DiffState(0, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias))
    _, grads = nnx.value_and_grad(lambda m: jnp.sum(m(tokens) * cot_jax), argnums=wrt)(jax_m)

    hf_grad_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    with torch.no_grad():
      for p in list(hf_m.parameters()) + list(hf_head.parameters()):
        p.copy_(p.grad if p.grad is not None else torch.zeros_like(p))
    _copy_hf_to_jax(hf_m, hf_head, hf_grad_m, cfg)

    (
        worst_frob,
        worst_frob_name,
        worst_inf,
        worst_inf_name,
        min_cos,
        min_cos_name,
    ) = (-1.0, "", -1.0, "", 2.0, "")
    leaf_data = []
    leaves_jax = jax.tree_util.tree_leaves_with_path(nnx.state(grads, nnx.Param))
    leaves_hf = jax.tree_util.tree_leaves_with_path(nnx.state(hf_grad_m, nnx.Param))
    self.assertEqual(len(leaves_jax), len(leaves_hf))
    for (p_jax, g_jax), (_, g_hf) in zip(leaves_jax, leaves_hf, strict=True):
      p_str, v_jax, v_hf = (
          jax.tree_util.keystr(p_jax),
          np.asarray(g_jax),
          np.asarray(g_hf),
      )
      if "indexer" in p_str:
        self.assertTrue(np.all(v_jax == 0.0), msg=f"Indexer grad non-zero: {p_str}")
      else:
        g_frob, g_inf, g_cos, _, _ = self._calc_e3(v_jax, v_hf, name=p_str, dtype=np_dt)
        leaf_data.append((p_str, v_jax, v_hf))
        worst_frob, worst_frob_name = max((worst_frob, worst_frob_name), (g_frob, p_str))
        worst_inf, worst_inf_name = max((worst_inf, worst_inf_name), (g_inf, p_str))
        min_cos, min_cos_name = min((min_cos, min_cos_name), (g_cos, p_str))

    tag = "E3_FP64_METRICS" if is_fp64 else "E3_METRICS"
    print(
        f"[{tag} num_layers={num_layers}] logits: frob_rel={l_frob:.3e},"
        f" max_rel_inf={l_inf:.3e}, cos={l_cos:.8f} | grads ({len(leaf_data)}"
        f" params): worst_frob_rel={worst_frob:.3e} ({worst_frob_name}),"
        f" worst_max_rel_inf={worst_inf:.3e} ({worst_inf_name}),"
        f" min_cos={min_cos:.8f} ({min_cos_name})"
    )

    if is_fp64:
      self.assertLessEqual(l_frob, 1e-10, f"fp64 logits frob_rel mismatch: {l_frob}")
      for (p_jax, g_jax), (_, g_hf) in zip(leaves_jax, leaves_hf, strict=True):
        p_str, jg, hg = (
            jax.tree_util.keystr(p_jax),
            np.asarray(g_jax),
            np.asarray(g_hf),
        )
        if "indexer" in p_str:
          self.assertTrue(
              np.array_equal(jg, np.zeros_like(jg)),
              msg=f"Indexer grad non-zero in fp64: {p_str}",
          )
        else:
          norm_b = float(np.linalg.norm(hg))
          self.assertGreater(
              float(np.linalg.norm(jg)),
              0.0,
              msg=f"JAX grad zero in fp64: {p_str}",
          )
          self.assertGreater(norm_b, 0.0, msg=f"HF grad zero in fp64: {p_str}")
          self.assertLessEqual(
              float(np.linalg.norm(jg - hg)) / norm_b,
              1e-10,
              f"fp64 grad mismatch in {p_str}",
          )
      return None

    return jax_m, tokens, np_hf, out_aux.logits, leaf_data, grads

  @parameterized.named_parameters(
      ("layers5_fp64", 5, True),
      ("layers5_fp32", 5, False),
      ("layers7_fp64", 7, True),
      ("layers7_fp32", 7, False),
      ("layers13_fp64", 13, True),
      ("layers13_fp32", 13, False),
  )
  def test_hf_parity_and_backward(self, num_layers, is_fp64):
    """Verifies forward and backward parity between HF and JAX Glm5Model."""
    cfg_f32, hf_c = _make_cfg(
        num_hidden_layers=num_layers,
        scan_layers=True,
        dtype="float32",
        weight_dtype="float32",
    )
    cfg_f64, _ = _make_cfg(
        num_hidden_layers=num_layers,
        scan_layers=True,
        dtype="float64",
        weight_dtype="float64",
    )

    torch.manual_seed(42)
    hf_m_f32 = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head_f32 = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    with torch.no_grad():
      for n, p in list(hf_m_f32.named_parameters()) + list(hf_m_f32.named_buffers()):
        if p.is_floating_point():
          if "layernorm" in n or "norm" in n:
            if n.endswith(".bias"):
              p.zero_()
            else:
              p.fill_(1.0)
          elif "A_log" in n:
            p.copy_(torch.log(torch.rand_like(p) + 0.1))
          elif "dt_bias" in n:
            p.uniform_(-2.0, -0.5)
          else:
            p.normal_(0.0, 0.02)
      hf_head_f32.weight.normal_(0.0, 0.02)

    seq_len = 32 if num_layers == 5 else 20
    ids = torch.randint(1, 200, (1, seq_len), dtype=torch.long)
    cot_f32 = torch.randn(1, seq_len, hf_c.vocab_size, dtype=torch.float32)

    if not isinstance(is_fp64, bool):
      raise ValueError(f"Unknown is_fp64 value: {is_fp64!r}")
    if is_fp64:
      hf_m_f64 = copy.deepcopy(hf_m_f32).to(torch.float64)
      hf_head_f64 = copy.deepcopy(hf_head_f32).to(torch.float64)
      cot_f64 = cot_f32.detach().clone().to(torch.float64)
      with _fp64_mode():
        self._run_parity_pass(
            num_layers,
            cfg_f64,
            hf_m_f64,
            hf_head_f64,
            ids,
            cot_f64,
            is_fp64=True,
        )
      return

    jax_m, tokens, np_hf, aux_logits, leaf_data, grads = self._run_parity_pass(
        num_layers, cfg_f32, hf_m_f32, hf_head_f32, ids, cot_f32, is_fp64=False
    )
    self._assert_e3(aux_logits, np_hf, "forward_aux_logits")
    self._assert_e3(
        np.asarray(jax_m(tokens), dtype=np.float32),
        np_hf,
        "forward_call_logits",
    )
    for path_str, val_jax, val_hf in leaf_data:
      self._assert_e3(val_jax, val_hf, path_str)
    bias_leaves = jax.tree.leaves(nnx.state(grads, modeling_glm5_3_flash.Glm5RouterBias))
    self.assertNotEmpty(bias_leaves)
    for b_grad in bias_leaves:
      np.testing.assert_allclose(np.array(b_grad), 0.0, atol=0.0, rtol=0.0)

  @parameterized.named_parameters(
      ("unscanned", "unscanned"),
      ("remat", "remat"),
  )
  def test_scanned_unscanned_and_remat_parity(self, variant):
    """Verifies bit-level forward and per-param gradient parity on 13-layer model."""
    cfg_s, _ = _make_cfg(num_hidden_layers=13, scan_layers=True)
    m_scan = modeling_glm5_3_flash.Glm5Model(cfg_s, rngs=nnx.Rngs(42), place_on_devices=False)
    target_rngs = nnx.Rngs(42)
    if variant == "unscanned":
      cfg_u, _ = _make_cfg(num_hidden_layers=13, scan_layers=False)
      m_target = modeling_glm5_3_flash.Glm5Model(cfg_u, rngs=target_rngs, place_on_devices=False)
    elif variant == "remat":
      pol = jax.checkpoint_policies.nothing_saveable
      m_target = modeling_glm5_3_flash.apply_glm5_remat(
          modeling_glm5_3_flash.Glm5Model(cfg_s, rngs=target_rngs, place_on_devices=False),
          policy=pol,
      )
    else:
      raise ValueError(f"Unknown variant: {variant!r}")

    for lyr in (
        [getattr(m_scan.super_block, f"slot_{s}") for s in range(4)]
        + list(m_scan.prefix_layers)
        + list(m_scan.suffix_layers)
    ):
      if lyr and lyr.is_moe:
        b = lyr.mlp.gate.e_score_correction_bias
        b.value = jnp.array(np.random.randn(*b.value.shape).astype(np.float32))

    nnx.update(m_target, nnx.state(m_scan))

    ids = torch.randint(1, 200, (1, 20), dtype=torch.long)
    tokens = jnp.array(ids.numpy())

    def loss_fn(m):
      o = m.forward_with_aux(tokens)
      return o.logits.mean() + o.aux["moe_lb_loss"], (
          o.logits,
          o.aux["moe_lb_loss"],
      )

    (_, (logits_s, lb_loss_s)), grads_s = nnx.value_and_grad(loss_fn, has_aux=True)(m_scan)
    leaves_s = jax.tree.leaves(nnx.state(grads_s, nnx.Param))

    (_, (logits_m, lb_loss_m)), grads_m = nnx.value_and_grad(loss_fn, has_aux=True)(m_target)
    self._assert_allclose(logits_s, logits_m, atol=1e-5, rtol=1e-5)
    self._assert_allclose(lb_loss_s, lb_loss_m, atol=1e-5, rtol=1e-5)
    self.assertGreater(float(lb_loss_s), 0.0)
    leaves_m = jax.tree.leaves(nnx.state(grads_m, nnx.Param))
    for ps, pm in zip(leaves_s, leaves_m, strict=True):
      self._assert_allclose(ps, pm, atol=1e-5, rtol=1e-5)

  def test_right_padded_parity(self):
    """Verifies parity on right-padded sequences using decoder_segment_ids."""
    num_layers = 5
    cfg, hf_c = _make_cfg(num_hidden_layers=num_layers, scan_layers=True)
    jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    _copy_hf_to_jax(hf_m, hf_head, jax_m, cfg)

    seq_len = 24
    valid_len_0 = 16
    valid_len_1 = 11
    ids_0 = torch.cat(
        [
            torch.randint(1, 200, (1, valid_len_0), dtype=torch.long),
            torch.zeros((1, seq_len - valid_len_0), dtype=torch.long),
        ],
        dim=1,
    )
    ids_1 = torch.cat(
        [
            torch.randint(1, 200, (1, valid_len_1), dtype=torch.long),
            torch.zeros((1, seq_len - valid_len_1), dtype=torch.long),
        ],
        dim=1,
    )
    ids = torch.cat([ids_0, ids_1], dim=0)
    attn_mask = torch.tensor(
        [
            [1] * valid_len_0 + [0] * (seq_len - valid_len_0),
            [1] * valid_len_1 + [0] * (seq_len - valid_len_1),
        ],
        dtype=torch.long,
    )

    orig_kda = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    try:
      with torch.no_grad():
        hf_logits = hf_head(hf_m(input_ids=ids, attention_mask=attn_mask).last_hidden_state)
    finally:
      hf_model.chunk_kimi_delta_attention = orig_kda

    tokens = jnp.array(ids.numpy())
    segment_ids = jnp.array(attn_mask.numpy(), dtype=jnp.int32)
    out_aux = jax_m.forward_with_aux(tokens, decoder_segment_ids=segment_ids)
    out_unpadded_0 = jax_m.forward_with_aux(tokens[0:1, :valid_len_0])
    out_unpadded_1 = jax_m.forward_with_aux(tokens[1:2, :valid_len_1])

    hf_valid_0 = hf_logits[0:1, :valid_len_0, :].detach().cpu().numpy()
    jax_valid_0 = np.array(out_aux.logits[0:1, :valid_len_0, :])
    self._assert_e3(jax_valid_0, hf_valid_0, "right_padded_row0")

    hf_valid_1 = hf_logits[1:2, :valid_len_1, :].detach().cpu().numpy()
    jax_valid_1 = np.array(out_aux.logits[1:2, :valid_len_1, :])
    self._assert_e3(jax_valid_1, hf_valid_1, "right_padded_row1")

    self._assert_allclose(
        out_aux.logits[0:1, :valid_len_0, :],
        out_unpadded_0.logits,
        atol=1e-5,
        rtol=1e-5,
    )
    self._assert_allclose(
        out_aux.logits[1:2, :valid_len_1, :],
        out_unpadded_1.logits,
        atol=1e-5,
        rtol=1e-5,
    )
    self.assertGreater(float(out_aux.aux["moe_lb_loss"]), 0.0)

  def test_moe_lb_loss_parity_against_hf(self):
    """Verifies moe_lb_loss matches HF load_balancing_loss_func.

    Tests with output_router_logits=True for padded and unpadded inputs.
    """
    num_layers = 7
    cfg_s, hf_c = _make_cfg(num_hidden_layers=num_layers, scan_layers=True)
    cfg_u, _ = _make_cfg(num_hidden_layers=num_layers, scan_layers=False)
    hf_c.output_router_logits = True
    torch.manual_seed(123)
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    with torch.no_grad():
      for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers()):
        if p.is_floating_point():
          if "e_score_correction_bias" in n:
            p.normal_(0.0, 0.5)
          elif "layernorm" in n or "norm" in n:
            if n.endswith(".bias"):
              p.zero_()
            else:
              p.fill_(1.0)
          elif "A_log" in n:
            p.uniform_(-2.0, -0.5)
          else:
            p.normal_(0.0, 0.02)

    m_scan = modeling_glm5_3_flash.Glm5Model(cfg_s, rngs=nnx.Rngs(0), place_on_devices=False)
    m_unscan = modeling_glm5_3_flash.Glm5Model(cfg_u, rngs=nnx.Rngs(0), place_on_devices=False)
    _copy_hf_to_jax(hf_m, hf_head, m_scan, cfg_s)
    _copy_hf_to_jax(hf_m, hf_head, m_unscan, cfg_u)

    ids_pt = torch.randint(1, 200, (2, 8), dtype=torch.long)
    tokens_jax = jnp.asarray(ids_pt.numpy(), dtype=jnp.int32)
    seg_pad_jax = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.int32)
    mask_pad_pt = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    orig_fn = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    try:
      for seg_jax, attn_pt in ((None, None), (seg_pad_jax, mask_pad_pt)):
        with torch.no_grad():
          hf_out = hf_m(
              input_ids=ids_pt,
              attention_mask=attn_pt,
              output_router_logits=True,
              return_dict=True,
          )
          hf_lb_loss = hf_model.load_balancing_loss_func(
              hf_out.router_logits,
              hf_c.n_routed_experts,
              hf_c.num_experts_per_tok,
              attn_pt,
          )
        for m_jax in (m_scan, m_unscan):
          out_jax = m_jax.forward_with_aux(tokens_jax, decoder_segment_ids=seg_jax)
          self._assert_allclose(
              out_jax.aux["moe_lb_loss"],
              float(hf_lb_loss.item()),
              atol=1e-5,
              rtol=1e-5,
          )
    finally:
      hf_model.chunk_kimi_delta_attention = orig_fn


class Glm5ModelPureJaxTest(parameterized.TestCase):
  """Pure-JAX tests for Glm5Model that run without PyTorch."""

  def setUp(self):
    super().setUp()
    np.random.seed(42)

  def _assert_allclose(self, a, b, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)

  def test_registry_dispatch_and_invariants(self):
    """Verifies registry dispatch, debug stats generation, and factory validation."""
    self.assertIs(
        m3_models.MODEL_REGISTRY["glm5.3-flash"],
        m3_models.create_glm5_3_flash_model,
    )
    self.assertEqual(
        m3_models.create_glm5_3_flash_model.__module__,
        m3_models.__name__,
    )
    cfg = dataclasses.replace(_make_cfg(num_hidden_layers=5)[0], enable_debug_stats=True)
    m = m3_models.create_model(cfg, None, rngs=nnx.Rngs(0))
    self.assertIsInstance(m, modeling_glm5_3_flash.Glm5Model)
    stats = m.forward_with_aux(jnp.array([[5, 6, 7]])).aux["debug_stats"]
    self.assertTrue({"hidden_norm_mean", "logits_max", "logits_min"}.issubset(stats))

    cases = (
        ({}, TypeError),
        ({"rngs": None}, ValueError),
        ({"rngs": nnx.Rngs(0), "quant_mode_str": "fp8"}, ValueError),
        ({"rngs": nnx.Rngs(0), "place_on_devices": False}, TypeError),
        ({"rngs": nnx.Rngs(0), "bad_kwarg": 1}, TypeError),
    )
    for kwargs, err in cases:
      with self.assertRaises(err):
        m3_models.create_model(cfg, None, **kwargs)

    m3 = modeling_glm5_3_flash.Glm5Model(
        dataclasses.replace(_make_cfg(num_hidden_layers=3)[0]),
        rngs=nnx.Rngs(0),
        place_on_devices=False,
    )
    self.assertEqual(
        float(m3.forward_with_aux(jnp.array([[5, 6, 7]])).aux["moe_lb_loss"]),
        0.0,
    )

    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh = jax.sharding.Mesh(devices, axis_names=("data", "fsdp", "tensor", "expert"))
    with jax.sharding.set_mesh(mesh):
      m_mesh = modeling_glm5_3_flash.Glm5Model(cfg, mesh=mesh, rngs=nnx.Rngs(0), place_on_devices=True)
      self.assertIsInstance(m_mesh.sharding, glm5_config.Glm5Sharding)
      self.assertFalse(m_mesh.sharding.use_tp)
      self.assertEqual(
          m_mesh.embed_tokens.embedding.out_sharding,
          jax.sharding.PartitionSpec(None, "fsdp"),
      )
      self.assertEqual(m_mesh.embed_tokens.embedding.sharding.mesh, mesh)
      out_mesh = m_mesh.forward_with_aux(jnp.array([[5, 6, 7]]))
      self.assertEqual(out_mesh.logits.shape, (1, 3, cfg.vocab_size))

    sh_tp = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      m_tp = modeling_glm5_3_flash.Glm5Model(cfg, sharding=sh_tp, rngs=nnx.Rngs(0), place_on_devices=True)
      self.assertTrue(m_tp.sharding.use_tp)
      self.assertEqual(
          m_tp.embed_tokens.embedding.out_sharding,
          jax.sharding.PartitionSpec("tensor", "fsdp"),
      )
      self.assertEqual(m_tp.embed_tokens.embedding.sharding.mesh, mesh)

    argv = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=glm5.3-flash",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        "base_emb_dim=128",
        "base_mlp_dim=128",
        "base_moe_mlp_dim=64",
        "first_num_dense_layers=2",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=16",
        "vocab_size=256",
        "num_experts=8",
        "num_experts_per_tok=2",
        "use_m3_model=true",
    ]
    raw_cfg = pyconfig.initialize(argv)
    m_created = model_creation_utils.create_model(raw_cfg, None, rngs=nnx.Rngs(0), quant_mode_str="train")
    self.assertIsInstance(m_created, modeling_glm5_3_flash.Glm5Model)
    self.assertEqual(m_created.config.num_hidden_layers, 7)
    self.assertEqual(m_created.config.hidden_size, 128)
    self.assertEqual(m_created.config.n_routed_experts, 8)
    self.assertEqual(m_created.config.moe_intermediate_size, 64)
    self.assertEqual(m_created.config.first_k_dense_replace, 2)
    self.assertEqual(m_created.config.rms_norm_eps, 1e-5)
    self.assertEqual(m_created.config.routed_scaling_factor, 2.5)
    self.assertEqual(m_created.config.scoring_func, "sigmoid")
    self.assertLen(m_created.prefix_layers, 2)
    self.assertIsNotNone(m_created.super_block)

    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.apply_glm5_remat(m, policy=True)
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.Glm5Model(cfg, rngs=None, place_on_devices=False)
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0))  # pylint: disable=missing-kwoa
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)

  def test_bfloat16_precision(self):
    """Verifies model execution with dtype=bfloat16 and weight_dtype=float32 on 13-layer model."""
    cfg, _ = _make_cfg(
        num_hidden_layers=13,
        scan_layers=True,
        dtype="bfloat16",
        weight_dtype="float32",
    )
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    self.assertIsNotNone(m.super_block)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    out = m.forward_with_aux(tokens)
    self.assertEqual(out.logits.dtype, jnp.bfloat16)
    self.assertEqual(out.hidden_states.dtype, jnp.bfloat16)
    self.assertEqual(m.embed_tokens.embedding.value.dtype, jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out.logits))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out.hidden_states))))

    # Assert lax.scan carry dtypes on m.super_block
    init_streams = jnp.zeros((1, 4, cfg.mhc_expansion_rate, cfg.hidden_size), dtype=jnp.bfloat16)
    final_streams, _, (tokens_sum, prob_sum, total_rows), _, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        m.super_block,
        init_streams,
        scan_layers=True,
        return_router_logits=False,
        return_aux=False,
    )
    self.assertEqual(final_streams.dtype, jnp.bfloat16)
    self.assertEqual(tokens_sum.dtype, jnp.float32)
    self.assertEqual(prob_sum.dtype, jnp.float32)
    self.assertEqual(total_rows.dtype, jnp.float32)

    wrt = nnx.DiffState(0, nnx.Param)
    loss_val, grads = nnx.value_and_grad(lambda model: jnp.sum(model(tokens).astype(jnp.float32)), argnums=wrt)(m)
    self.assertTrue(bool(jnp.isfinite(loss_val)))
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    self.assertNotEmpty(leaves)
    for g in leaves:
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

  def test_model_on_single_device_mesh(self):
    """Verifies Glm5Model instantiation and execution on 1-device MaxText mesh."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    cfg = _make_cfg(num_hidden_layers=7, scan_layers=True)[0]
    tokens = jnp.array([[5, 6, 7]], dtype=jnp.int32)

    # 1-device MaxText mesh ("data", "fsdp", "tensor", "expert")
    mesh_mt = jax.sharding.Mesh(devices, axis_names=("data", "fsdp", "tensor", "expert"))
    with jax.sharding.set_mesh(mesh_mt):
      m_mt = modeling_glm5_3_flash.Glm5Model(cfg, mesh=mesh_mt, rngs=nnx.Rngs(0), place_on_devices=True)
      self.assertIsNotNone(m_mt.super_block)
      self.assertIsInstance(m_mt.sharding, glm5_config.Glm5Sharding)
      self.assertFalse(m_mt.sharding.use_tp)
      self.assertEqual(m_mt.sharding.tp_axis, "tensor")
      self.assertEqual(m_mt.sharding.ep_axis, "expert")
      out_mt = m_mt.forward_with_aux(tokens)
      self.assertEqual(out_mt.logits.shape, (1, 3, cfg.vocab_size))
      wrt = nnx.DiffState(0, nnx.Param)
      loss_val, grads = nnx.value_and_grad(lambda model: jnp.sum(model(tokens)), argnums=wrt)(m_mt)
      self.assertTrue(bool(jnp.isfinite(loss_val)))
      grad_leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
      self.assertNotEmpty(grad_leaves)
      for g in grad_leaves:
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    # Verify create_nnx_abstract_model with raw_cfg_7layers without set_mesh active
    argv_7layers = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=glm5.3-flash",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_moe_mlp_dim=32",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=16",
        "vocab_size=256",
        "num_experts=8",
        "num_experts_per_tok=2",
        "use_m3_model=true",
    ]
    raw_cfg_7layers = pyconfig.initialize(argv_7layers)
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(raw_cfg_7layers, mesh=mesh_mt)
    self.assertIsNotNone(abstract_model.super_block)
    self.assertIsNone(abstract_model.super_block.slot_0.mlp.shared_experts.gate_proj.kernel.out_sharding[0])
    self.assertEqual(abstract_model.mesh, mesh_mt)

    # Test default mesh=None path (T30)
    _, abstract_model_default = model_creation_utils.create_nnx_abstract_model(raw_cfg_7layers)
    self.assertIsNotNone(abstract_model_default.super_block)
    self.assertIsInstance(abstract_model_default.mesh, jax.sharding.Mesh)

    bad_mesh = jax.sharding.Mesh(devices.reshape((1,)), axis_names=("bogus",))
    with self.assertRaises(ValueError):
      glm5_config.Glm5Sharding.from_mesh(bad_mesh)

  def test_moe_lb_loss_all_padding_zeros(self):
    """Verifies all-padding decoder_segment_ids produces moe_lb_loss == 0.0 (not NaN) (T31)."""
    b, s = 2, 8
    tokens = jnp.ones((b, s), dtype=jnp.int32)
    segment_ids = jnp.zeros((b, s), dtype=jnp.int32)
    for scan_layers in (True, False):
      cfg, _ = _make_cfg(num_hidden_layers=7, scan_layers=scan_layers)
      model = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
      out = model.forward_with_aux(tokens, decoder_segment_ids=segment_ids)
      moe_lb = out.aux["moe_lb_loss"]
      self.assertTrue(
          bool(jnp.isfinite(moe_lb)),
          f"Expected finite moe_lb_loss for scan_layers={scan_layers}, got" f" {moe_lb}",
      )
      self.assertEqual(float(moe_lb), 0.0)

  def test_jit_forward_with_aux(self):
    """Verifies jax.jit compilation of forward_with_aux returning Glm5Output."""
    cfg, _ = _make_cfg(num_hidden_layers=5, scan_layers=True)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    tokens = jnp.array([[5, 10, 15, 20]], dtype=jnp.int32)
    out_eager = m.forward_with_aux(tokens)

    jit_fn = jax.jit(lambda model, x: model.forward_with_aux(x))
    out_jit = jit_fn(m, tokens)
    self.assertIsInstance(out_jit, modeling_glm5_3_flash.Glm5Output)
    self._assert_allclose(out_jit.logits, out_eager.logits, atol=1e-5, rtol=1e-5)

  def test_return_aux_harness_contract(self):
    for scan_layers in (True, False):
      cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=scan_layers)
      r = nnx.Rngs(0)
      m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
      tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
      logits_only = m(tokens, return_aux=False)
      logits_pair, aux = m(tokens, return_aux=True)

      self.assertIsInstance(logits_only, jax.Array)
      self.assertNotIsInstance(logits_only, tuple)
      self.assertIsInstance(logits_pair, jax.Array)
      np.testing.assert_array_equal(np.array(logits_only), np.array(logits_pair))

      self.assertIn("layer_aux", aux)
      self.assertIn("router_indices", aux)
      self.assertIn("routing_weights", aux)
      self.assertIn("router_probs", aux)
      self.assertIn("selected_block_indices", aux)
      self.assertIn("moe_lb_loss", aux)
      self.assertIs(modeling_glm5_3_flash.Glm5ForCausalLM, modeling_glm5_3_flash.Glm5Model)

      self.assertEqual(aux["router_indices"].shape, (8, 1, 4, cfg.num_experts_per_tok))
      self.assertEqual(aux["routing_weights"].shape, (8, 1, 4, cfg.num_experts_per_tok))
      self.assertEqual(aux["router_probs"].shape, (8, 1, 4, cfg.n_routed_experts))
      out_w = cfg.index_topk + (cfg.index_kpool - 1 if cfg.index_kpool_always_select_tail else 0)
      self.assertEqual(aux["selected_block_indices"].shape, (2, 1, 4, out_w))

      self.assertEqual(aux["router_indices"].dtype, jnp.int32)
      self.assertEqual(aux["routing_weights"].dtype, jnp.float32)
      self.assertEqual(aux["router_probs"].dtype, jnp.float32)
      self.assertEqual(aux["selected_block_indices"].dtype, jnp.int32)

      for key in [
          "router_indices",
          "routing_weights",
          "router_probs",
          "selected_block_indices",
      ]:
        stacked = np.stack([np.asarray(a[key]) for a in aux["layer_aux"] if key in a], axis=0)
        np.testing.assert_array_equal(stacked, np.asarray(aux[key]))

  def test_generic_maxtext_kwargs_validation(self):
    cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=False)
    r = nnx.Rngs(0)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
    tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)

    # Accepted kwargs (None or False)
    out = m(
        tokens,
        enable_dropout=False,
        decoder_target_tokens=None,
        decoder_target_mask=None,
        encoder_input_tokens=None,
    )
    self.assertIsInstance(out, jax.Array)

    # Unsupported active kwargs raise ValueError
    with self.assertRaisesRegex(ValueError, "Dropout is not supported"):
      m(tokens, enable_dropout=True)
    with self.assertRaisesRegex(ValueError, "decoder_target_tokens is not supported"):
      m(tokens, decoder_target_tokens=tokens)
    with self.assertRaisesRegex(ValueError, "decoder_target_mask is not supported"):
      m(tokens, decoder_target_mask=tokens)
    with self.assertRaisesRegex(ValueError, "Encoder inputs are not supported"):
      m(tokens, encoder_input_tokens=tokens)

    # Unrecognized kwargs raise TypeError
    with self.assertRaises(TypeError):
      m(tokens, return_mtp=True)
    with self.assertRaises(TypeError):
      m(tokens, unknown_kwarg=123)
    with self.assertRaises(TypeError):
      m.forward_with_aux(tokens, foo="bar")

  def test_f49_heterogeneous_slot_alignment(self):
    for scan_layers in (False, True):
      cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=scan_layers)
      r = nnx.Rngs(0)
      m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
      tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
      _, aux = m(tokens, return_aux=True)

      self.assertEqual(len(aux["layer_aux"]), 11)
      self.assertNotIn("router_indices", aux["layer_aux"][0])
      self.assertNotIn("selected_block_indices", aux["layer_aux"][0])

      self.assertIn("router_indices", aux["layer_aux"][3])
      self.assertIn("selected_block_indices", aux["layer_aux"][3])
      self.assertIn("router_indices", aux["layer_aux"][7])
      self.assertIn("selected_block_indices", aux["layer_aux"][7])

      for i in [4, 5, 6, 8, 9, 10]:
        self.assertIn("router_indices", aux["layer_aux"][i])
        self.assertNotIn("selected_block_indices", aux["layer_aux"][i])

    # Binding Condition 2: Non-periodic layer_types raises ValueError at super-block init time
    bad_lt = list(cfg.layer_types)
    bad_lt[7] = "linear_attention"
    bad_cfg = dataclasses.replace(cfg, layer_types=tuple(bad_lt))
    with self.assertRaisesRegex(ValueError, "Non-periodic schedule at layer"):
      modeling_glm5_3_flash.create_stacked_super_block(bad_cfg, 2, rngs=nnx.Rngs(0), place_on_devices=False)

  def test_zero_mla_layers_none_selected_block_indices(self):
    cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=False)
    cfg = dataclasses.replace(cfg, layer_types=tuple("linear_attention" for _ in range(11)))
    r = nnx.Rngs(0)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
    tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
    _, aux = m(tokens, return_aux=True)
    self.assertIsNone(aux["selected_block_indices"])

  def test_use_indexer_false_none_selected_block_indices(self):
    cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=False)
    cfg = dataclasses.replace(cfg, use_indexer=False)
    r = nnx.Rngs(0)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
    tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
    _, aux = m(tokens, return_aux=True)
    self.assertIsNone(aux["selected_block_indices"])
    for layer_aux in aux["layer_aux"]:
      self.assertNotIn("selected_block_indices", layer_aux)

  def test_multiple_full_attention_slots_per_super_block(self):
    for scan_layers in (False, True):
      layer_types = (
          ("linear_attention",) * 3
          + (
              "linear_attention",
              "full_attention",
              "linear_attention",
              "full_attention",
          )
          + (
              "linear_attention",
              "full_attention",
              "linear_attention",
              "full_attention",
          )
      )
      cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=scan_layers)
      cfg = dataclasses.replace(cfg, layer_types=layer_types)
      r = nnx.Rngs(0)
      m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=r, place_on_devices=False)
      tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
      _, aux = m(tokens, return_aux=True)
      self.assertIsNotNone(aux["selected_block_indices"])
      # 2 super-blocks * 2 full_attention slots = 4 full_attention layers in super blocks
      self.assertEqual(aux["selected_block_indices"].shape[0], 4)
      for idx in [4, 6, 8, 10]:
        self.assertIn("selected_block_indices", aux["layer_aux"][idx])
      for idx in [0, 1, 2, 3, 5, 7, 9]:
        self.assertNotIn("selected_block_indices", aux["layer_aux"][idx])

  def test_unstacked_per_layer_reference_parity_seq_len_gt_index_topk(self):
    tokens = jnp.array(np.random.RandomState(42).randint(0, 64, size=(1, 32)))

    def _slice_block(b_idx: int, state: Any) -> Any:
      return jax.tree_util.tree_map(
          lambda x: (x.replace(value=x.value[b_idx]) if isinstance(x, nnx.Variable) else x),
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )

    for scan_layers in (True, False):
      cfg = _make_cfg(num_hidden_layers=11, scan_layers=scan_layers)[0]
      m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(123), place_on_devices=False)
      out, aux = m(tokens, return_aux=True)

      hidden = jnp.broadcast_to(
          m.embed_tokens(tokens.astype(jnp.int32))[:, :, None, :],
          (
              tokens.shape[0],
              tokens.shape[1],
              cfg.mhc_expansion_rate,
              cfg.hidden_size,
          ),
      )
      ref_layer_aux: list[dict[str, Any]] = []

      for lyr_idx, lyr in enumerate(m.prefix_layers):
        hidden, _, r_idx, idx_topk, r_w, r_p = lyr(hidden, return_aux=True)
        layer_dict: dict[str, Any] = {}
        if lyr_idx >= cfg.first_k_dense_replace:
          layer_dict["router_indices"] = r_idx
          layer_dict["routing_weights"] = r_w
          layer_dict["router_probs"] = r_p
        if cfg.layer_types[lyr_idx] == "full_attention":
          layer_dict["selected_block_indices"] = idx_topk
        ref_layer_aux.append(layer_dict)

      for b in range(2):
        sb_b = modeling_glm5_3_flash.Glm5SuperBlock(cfg, rngs=nnx.Rngs(999))
        nnx.update(sb_b, _slice_block(b, nnx.state(m.super_block)))
        for s in range(4):
          slot = getattr(sb_b, f"slot_{s}")
          hidden, _, r_idx, idx_topk, r_w, r_p = slot(hidden, return_aux=True)
          slot_dict: dict[str, Any] = {
              "router_indices": r_idx,
              "routing_weights": r_w,
              "router_probs": r_p,
          }
          if slot.layer_type == "full_attention":
            slot_dict["selected_block_indices"] = idx_topk
          ref_layer_aux.append(slot_dict)

      ref_logits = m.lm_head(
          m.norm(jnp.mean(hidden.astype(jnp.float32), axis=2).astype(m.norm.weight.value.dtype))
      ).astype(jnp.float32)

      np.testing.assert_allclose(np.array(out), np.array(ref_logits), atol=1e-5, rtol=1e-5)
      self.assertEqual(len(aux["layer_aux"]), 11)
      self.assertEqual(len(ref_layer_aux), 11)

      for i in range(11):
        self.assertEqual(set(aux["layer_aux"][i].keys()), set(ref_layer_aux[i].keys()))
        if "selected_block_indices" in ref_layer_aux[i]:
          np.testing.assert_array_equal(
              np.array(aux["layer_aux"][i]["selected_block_indices"]),
              np.array(ref_layer_aux[i]["selected_block_indices"]),
          )
        if "router_indices" in ref_layer_aux[i]:
          np.testing.assert_array_equal(
              np.array(aux["layer_aux"][i]["router_indices"]),
              np.array(ref_layer_aux[i]["router_indices"]),
          )
          np.testing.assert_allclose(
              np.array(aux["layer_aux"][i]["routing_weights"]),
              np.array(ref_layer_aux[i]["routing_weights"]),
              atol=1e-5,
              rtol=1e-5,
          )
          np.testing.assert_allclose(
              np.array(aux["layer_aux"][i]["router_probs"]),
              np.array(ref_layer_aux[i]["router_probs"]),
              atol=1e-5,
              rtol=1e-5,
          )

      self.assertFalse(
          np.array_equal(
              np.array(aux["layer_aux"][3]["selected_block_indices"]),
              np.array(aux["layer_aux"][7]["selected_block_indices"]),
          )
      )

  def test_return_aux_use_indexer_false_and_forced_none_guard(self):
    for num_layers, scan_layers in ((5, False), (11, False), (11, True)):
      cfg, _ = _make_cfg(num_hidden_layers=num_layers, scan_layers=scan_layers)
      cfg = dataclasses.replace(cfg, use_indexer=False)
      m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
      tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
      logits, aux = m(tokens, return_aux=True)
      self.assertEqual(logits.shape, (1, 4, cfg.vocab_size))
      self.assertIsNone(aux["selected_block_indices"])
      for layer_dict in aux["layer_aux"]:
        self.assertNotIn("selected_block_indices", layer_dict)

    # Positive control 1: use_indexer=True with forced-None idx (unscanned)
    cfg, _ = _make_cfg(num_hidden_layers=5, scan_layers=False)
    cfg = dataclasses.replace(cfg, use_indexer=True)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
    # Mock Glm5Indexer.__call__ around single call to force idx=None when use_indexer=True per crew_coder_standards.md
    with mock.patch.object(modeling_glm5_3_flash.Glm5Indexer, "__call__", return_value=None):
      with self.assertRaisesRegex(RuntimeError, "has full_attention but idx is None"):
        m(tokens, return_aux=True)

    # Positive control 2: use_indexer=True with forced-None idx (scanned)
    cfg, _ = _make_cfg(num_hidden_layers=11, scan_layers=True)
    cfg = dataclasses.replace(cfg, use_indexer=True)
    m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    tokens = jax.random.randint(jax.random.key(1), (1, 4), 0, cfg.vocab_size)
    with mock.patch.object(modeling_glm5_3_flash.Glm5Indexer, "__call__", return_value=None):
      with self.assertRaisesRegex(
          RuntimeError,
          "has full_attention with use_indexer=True but sb_idx.*is None",
      ):
        m(tokens, return_aux=True)


if __name__ == "__main__":
  absltest.main()
