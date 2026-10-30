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

"""Parity, concatenation order, checkpoint coverage, and training gate tests for GLM-5.3-Flash Layer-45 MTP."""

# pylint: disable=protected-access

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
import gc
import importlib.util
import json
import math
import os
from typing import Any
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_train
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from tests.utils import glm5_test_utils
from tests.utils import zero_gen_logits_checker
import numpy as np
import pytest

try:
  from maxtext.m3.models.glm5_3_flash import glm5_ckpt
  import torch
  import transformers
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

try:
  HAS_TUNIX = "TEST_SRCDIR" in os.environ or importlib.util.find_spec("tunix") is not None
except (ImportError, ValueError):
  HAS_TUNIX = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

_RUN_REAL_MTP = flags.DEFINE_bool(
    "run_real_mtp",
    False,
    "Whether to run real-weight Layer-45 MTP test.",
)
_MINI_SLICE_DIR = flags.DEFINE_string(
    "mini_slice_dir",
    glm5_test_utils.MINI_SLICE_DIR,
    "Path to mini-slice directory containing model.safetensors and" " config.json.",
    allow_override=True,
)

_TorchModuleBase = torch.nn.Module if HAS_TORCH else object


class HfMtpReference(_TorchModuleBase):
  """PyTorch reference for GLM-5.3-Flash Layer-45 MTP block using HF submodules."""

  def __init__(self, config: hf_cfg.Glm5NextTextConfig):
    super().__init__()
    h = config.hidden_size
    self.enorm = hf_model.Glm5NextTextRMSNorm(h, eps=config.rms_norm_eps)
    self.hnorm = hf_model.Glm5NextTextRMSNorm(h, eps=config.rms_norm_eps)
    self.eh_proj = torch.nn.Linear(2 * h, h, bias=False)
    self.input_layernorm = hf_model.Glm5NextTextRMSNorm(h, eps=config.rms_norm_eps)
    self.self_attn = hf_model.Glm5NextTextAttention(config, layer_idx=0)
    self.post_attention_layernorm = hf_model.Glm5NextTextRMSNorm(h, eps=config.rms_norm_eps)
    self.mlp = hf_model.Glm5NextTextMoE(config)
    self.shared_head_norm = hf_model.Glm5NextTextRMSNorm(h, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      next_token_ids: torch.Tensor,
      embed_tokens: torch.nn.Module,
      lm_head: torch.nn.Module,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes next-token MTP logits using shared embed_tokens and lm_head."""
    if attention_mask is None:
      attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    emb = embed_tokens(next_token_ids)
    # Pinned concatenation order: Embedding FIRST, Hidden state SECOND
    x = self.eh_proj(torch.cat([self.enorm(emb), self.hnorm(hidden_states)], dim=-1))
    attn_out = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)[0]
    y = x + attn_out
    mlp_out = self.mlp(self.post_attention_layernorm(y))
    z = y + mlp_out
    return lm_head(self.shared_head_norm(z))


def _make_cfg(
    num_hidden_layers: int = 5,
    num_nextn_predict_layers: int = 1,
    scan_layers: bool = False,
    dtype: str = "float32",
    weight_dtype: str = "float32",
) -> tuple[glm5_config.Glm5Config, hf_cfg.Glm5NextTextConfig]:
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
      num_nextn_predict_layers=num_nextn_predict_layers,
      scan_layers=scan_layers,
      kda_implementation="xla",
      dtype=dtype,
      weight_dtype=weight_dtype,
      vocab_size=256,
  )
  c = glm5_config.Glm5Config.mini_config(**args)
  hf_args = {x: args[x] for x in args if not x.startswith("linear_") and x not in {"qk_head_dim", "scan_layers"}}
  hf_k = (
      "vocab_size pad_token_id num_hidden_layers _attn_implementation"
      " index_kpool linear_num_heads linear_head_dim linear_conv_kernel_dim"
  ).split()
  hf_args.update(zip(hf_k, (256, 0, num_hidden_layers, "eager", 4, 4, 16, 4)))
  hf_args["mlp_layer_types"] = ["dense"] * min(3, num_hidden_layers) + ["sparse"] * max(0, num_hidden_layers - 3)
  hf_args["layer_types"], hf_args["indexer_types"] = list(c.layer_types), list(c.indexer_types)
  return c, hf_cfg.Glm5NextTextConfig(**hf_args)


def _init_hf(hf_m: torch.nn.Module, hf_head: torch.nn.Module, hf_mtp: torch.nn.Module) -> None:
  """Initializes PyTorch HF backbone, lm_head, and MTP modules with deterministic non-zero weights."""
  with torch.no_grad():
    for n, p in (
        list(hf_m.named_parameters())
        + list(hf_m.named_buffers())
        + list(hf_mtp.named_parameters())
        + list(hf_mtp.named_buffers())
    ):
      if p.is_floating_point():
        if "layernorm" in n or "norm" in n:
          if n.endswith(".bias"):
            p.zero_()
          else:
            p.fill_(1.0)
        elif n.endswith(".scale"):
          p.fill_(1.0)
        elif n.endswith(".fn"):
          p.normal_(0.0, 0.15)
        elif "embed_tokens.weight" in n:
          p.normal_(0.0, 0.02)
          p[:, 0] = 0.06
        elif "weights_proj" in n:
          p.normal_(0.0, 0.002)
          p[:, 0] = 0.25
        elif "e_score_correction_bias" in n:
          p.copy_(torch.linspace(-0.05, 0.05, p.numel(), dtype=p.dtype).reshape(p.shape))
        elif "A_log" in n:
          p.uniform_(-3.5, -1.5)
        elif "dt_bias" in n:
          p.uniform_(-2.0, -0.5)
        else:
          p.normal_(0.0, 0.02)
    hf_head.weight.normal_(0.0, 0.02)


def _copy_mtp_hf_to_jax(hf_mtp: HfMtpReference, jax_mtp: modeling_glm5_3_flash.Glm5Mtp) -> None:
  """Transfers PyTorch MTP reference weights into JAX Glm5Mtp module."""

  def _w(x):
    return jnp.array(x.detach().numpy())

  def _t(x):
    return _w(x).T

  jax_mtp.enorm.weight.value = _w(hf_mtp.enorm.weight)
  jax_mtp.hnorm.weight.value = _w(hf_mtp.hnorm.weight)
  jax_mtp.eh_proj.kernel.value = _t(hf_mtp.eh_proj.weight)
  jax_mtp.shared_head_norm.weight.value = _w(hf_mtp.shared_head_norm.weight)
  jax_mtp.input_layernorm.weight.value = _w(hf_mtp.input_layernorm.weight)
  jax_mtp.post_attention_layernorm.weight.value = _w(hf_mtp.post_attention_layernorm.weight)

  hf_a, ja = hf_mtp.self_attn, jax_mtp.self_attn
  for p in ("q_a", "q_b", "kv_b", "o"):
    getattr(ja, f"{p}_proj").kernel.value = _t(getattr(hf_a, f"{p}_proj").weight)
  ja.kv_a_proj_with_mqa.kernel.value = _t(hf_a.kv_a_proj_with_mqa.weight)
  ja.q_a_layernorm.weight.value = _w(hf_a.q_a_layernorm.weight)
  ja.kv_a_layernorm.weight.value = _w(hf_a.kv_a_layernorm.weight)
  for p in ("wq_b", "wk", "weights_proj"):
    getattr(ja.indexer, p).kernel.value = _t(getattr(hf_a.indexer, p).weight)
  ja.indexer.k_norm.weight.value = _w(hf_a.indexer.k_norm.weight)
  ja.indexer.k_norm.bias.value = _w(hf_a.indexer.k_norm.bias)
  ja.indexer.index_kpool_compress_ape.value = _w(hf_a.indexer.index_kpool_compress_ape)
  ja.indexer.index_kpool_compress_gate.value = _w(hf_a.indexer.index_kpool_compress_gate)

  hf_moe_mod = hf_mtp.mlp
  for p in ("gate", "up", "down"):
    getattr(jax_mtp.mlp.shared_experts, f"{p}_proj").kernel.value = _t(
        getattr(hf_moe_mod.shared_experts, f"{p}_proj").weight
    )
  jax_mtp.mlp.gate.weight.value = _t(hf_moe_mod.gate.weight)
  jax_mtp.mlp.gate.e_score_correction_bias.value = _w(hf_moe_mod.gate.e_score_correction_bias)

  # [E, H, 2, I] layout for gate_up_proj
  cfg = jax_mtp.config
  gu_w = (
      _w(hf_moe_mod.experts.gate_up_proj)
      .transpose(0, 2, 1)
      .reshape(cfg.n_routed_experts, cfg.hidden_size, 2, cfg.moe_intermediate_size)
  )
  jax_mtp.mlp.experts.gate_up_proj.value = gu_w
  jax_mtp.mlp.experts.down_proj.value = _w(hf_moe_mod.experts.down_proj).swapaxes(1, 2)


def _copy_backbone_hf_to_jax(
    hf_m: torch.nn.Module,
    hf_head: torch.nn.Module,
    jax_m: modeling_glm5_3_flash.Glm5Model,
    cfg: glm5_config.Glm5Config,
) -> None:
  """Transfers PyTorch backbone weights into JAX Glm5Model."""

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
      gu_w = (
          _w(hf_l.mlp.experts.gate_up_proj)
          .transpose(0, 2, 1)
          .reshape(
              cfg.n_routed_experts,
              cfg.hidden_size,
              2,
              cfg.moe_intermediate_size,
          )
      )
      jax_l.mlp.experts.gate_up_proj.value = gu_w
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

  layer_rngs = nnx.Rngs(0)
  for l in range(cfg.num_hidden_layers):
    sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(l, cfg.first_k_dense_replace, cfg.num_hidden_layers)
    if sec in ("prefix", "suffix"):
      _copy_decoder(hf_m.layers[l], getattr(jax_m, f"{sec}_layers")[b_idx])
    elif sec == "super_block":
      slot = getattr(jax_m.super_block, f"slot_{s_idx}")
      temp_l = modeling_glm5_3_flash.Glm5DecoderLayer(cfg, 3 + s_idx, rngs=layer_rngs)
      _copy_decoder(hf_m.layers[l], temp_l)
      jax.tree.map(
          lambda tp, sp, idx=b_idx: setattr(sp, "value", sp.value.at[idx].set(tp.value)),
          nnx.state(temp_l),
          nnx.state(slot),
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5MtpTest(parameterized.TestCase):
  """Numerical parity, concatenation order, checkpoint coverage, and training tests for Glm5Mtp."""

  def setUp(self):
    super().setUp()
    if _RUN_REAL_MTP.value and self._testMethodName != "test_real_weight_layer_45_mtp":
      self.skipTest("Skipping hermetic test when --run_real_mtp=true")
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)
    self._orig_matmul_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")

  def tearDown(self):
    jax.config.update("jax_default_matmul_precision", self._orig_matmul_precision)
    jax.clear_caches()
    super().tearDown()

  @parameterized.named_parameters(
      ("fp32_e3", False),
      ("fp64_strict", True),
  )
  def test_forward_and_backward_all_param_parity(self, is_fp64: bool):
    """Test 1: Forward + all-parameter backward parity vs PyTorch reference."""
    dt_str = "float64" if is_fp64 else "float32"
    cfg, hf_c = _make_cfg(
        num_hidden_layers=5,
        num_nextn_predict_layers=1,
        scan_layers=False,
        dtype=dt_str,
        weight_dtype=dt_str,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    hf_m = hf_model.Glm5NextTextModel(hf_c).eval().to(torch.float32)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch.float32)
    hf_mtp = HfMtpReference(hf_c).eval().to(torch.float32)
    _init_hf(hf_m, hf_head, hf_mtp)

    b_sz, seq_len = 4, 16
    ids_np = np.random.randint(1, cfg.vocab_size, (b_sz, seq_len))
    targets_np = np.random.randint(1, cfg.vocab_size, (b_sz, seq_len))
    cot_np = np.random.randn(b_sz, seq_len, cfg.vocab_size).astype(np.float32)

    if is_fp64:
      hf_m = hf_m.to(torch.float64)
      hf_head = hf_head.to(torch.float64)
      hf_mtp = hf_mtp.to(torch.float64)
      ids_pt = torch.from_numpy(ids_np)
      targets_pt = torch.from_numpy(targets_np)
      cot_pt = torch.from_numpy(cot_np).to(torch.float64)
      cot_j = jnp.array(cot_np, dtype=jnp.float64)
    else:
      ids_pt = torch.from_numpy(ids_np)
      targets_pt = torch.from_numpy(targets_np)
      cot_pt = torch.from_numpy(cot_np)
      cot_j = jnp.array(cot_np, dtype=jnp.float32)

    cm = glm5_test_utils._fp64_mode() if is_fp64 else contextlib.nullcontext()
    with cm:
      jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
      self.assertIsNotNone(jax_m.mtp)
      _copy_backbone_hf_to_jax(hf_m, hf_head, jax_m, cfg)
      _copy_mtp_hf_to_jax(hf_mtp, jax_m.mtp)

      # Ensure KDA recurrence is used in PyTorch for clean autograd,
      # and disable grouped_mm so PyTorch MoE expert matmul uses standard float32 matmul.
      with mock.patch.object(hf_moe_integ, "_can_use_grouped_mm", return_value=False):
        orig_kda = hf_model.chunk_kimi_delta_attention
        hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
        try:
          hf_last_hidden = hf_m(input_ids=ids_pt).last_hidden_state
          hf_mtp_logits = hf_mtp(hf_last_hidden, targets_pt, hf_m.embed_tokens, hf_head)
          (hf_mtp_logits * cot_pt).sum().backward()
        finally:
          hf_model.chunk_kimi_delta_attention = orig_kda

      ids_j = jnp.array(ids_np)
      targets_j = jnp.array(targets_np)

      out_aux = jax_m.forward_with_aux(ids_j)
      h_i = out_aux.hidden_states
      jax_mtp_logits, _, _ = jax_m.mtp(
          h_i,
          targets_j,
          embed_tokens=jax_m.embed_tokens,
          lm_head=jax_m.lm_head,
          decoder_segment_ids=None,
      )

      # 1. Forward MTP logits parity
      np_hf = hf_mtp_logits.detach().cpu().numpy()
      glm5_test_utils._assert_e3(np.asarray(jax_mtp_logits), np_hf, name="mtp_logits", is_fp64=is_fp64)

      # 2. Backward parity across all parameters
      def loss_fn(m):
        out = m.forward_with_aux(ids_j)
        l, _, _ = m.mtp(
            out.hidden_states,
            targets_j,
            embed_tokens=m.embed_tokens,
            lm_head=m.lm_head,
            decoder_segment_ids=None,
        )
        return jnp.sum(l * cot_j)

      _, grads = nnx.value_and_grad(loss_fn)(jax_m)

      # Construct matching PyTorch gradient container
      hf_grad_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
      with torch.no_grad():
        for p in list(hf_m.parameters()) + list(hf_head.parameters()):
          p.copy_(p.grad if p.grad is not None else torch.zeros_like(p))
        for p in hf_mtp.parameters():
          p.copy_(p.grad if p.grad is not None else torch.zeros_like(p))
      _copy_backbone_hf_to_jax(hf_m, hf_head, hf_grad_m, cfg)
      _copy_mtp_hf_to_jax(hf_mtp, hf_grad_m.mtp)

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
          glm5_test_utils._assert_e3(v_jax, v_hf, name=p_str, is_fp64=is_fp64)

  def test_eh_proj_concatenation_order_negative_test(self):
    """Test 2: Swapping [enorm(E); hnorm(h)] -> [hnorm(h); enorm(E)] produces large mismatch (> 1e-2)."""
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1)
    jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(42), place_on_devices=False)

    b_sz, seq_len = 2, 8
    tokens = jax.random.randint(jax.random.PRNGKey(0), (b_sz, seq_len), 1, cfg.vocab_size)
    next_tokens = jax.random.randint(jax.random.PRNGKey(1), (b_sz, seq_len), 1, cfg.vocab_size)

    out = jax_m.forward_with_aux(tokens)
    h_i = out.hidden_states

    # Canonical order: embedding first, hidden state second
    normal_logits, _, _ = jax_m.mtp(
        h_i,
        next_tokens,
        embed_tokens=jax_m.embed_tokens,
        lm_head=jax_m.lm_head,
        decoder_segment_ids=None,
    )

    # Omitting lm_head or decoder_segment_ids raises TypeError per M2
    with self.assertRaises(TypeError):
      jax_m.mtp(  # pylint: disable=no-value-for-parameter
          h_i,
          next_tokens,
          embed_tokens=jax_m.embed_tokens,
          decoder_segment_ids=None,
      )
    with self.assertRaises(TypeError):
      jax_m.mtp(  # pylint: disable=no-value-for-parameter
          h_i,
          next_tokens,
          embed_tokens=jax_m.embed_tokens,
          lm_head=jax_m.lm_head,
      )

    # Swapped order: hidden state first, embedding second
    emb = jax_m.embed_tokens(next_tokens)
    swapped_x = jax_m.mtp.eh_proj(jnp.concatenate([jax_m.mtp.hnorm(h_i), jax_m.mtp.enorm(emb)], axis=-1))
    y = swapped_x + jax_m.mtp.self_attn(jax_m.mtp.input_layernorm(swapped_x))
    mlp_out, _ = jax_m.mtp.mlp(jax_m.mtp.post_attention_layernorm(y))
    z = y + mlp_out
    swapped_logits = jax_m.lm_head(jax_m.mtp.shared_head_norm(z).astype(jax_m.lm_head.kernel.dtype))

    diff = float(jnp.max(jnp.abs(normal_logits - swapped_logits)))
    self.assertGreater(
        diff,
        1e-2,
        "Concatenation order swap did not produce expected mismatch (> 1e-2):" f" got {diff}",
    )

  def test_layers_45_checkpoint_coverage(self):
    """Test 3: 100% layers.45.* checkpoint key coverage, restoration, and validation."""
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1)
    jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)

    # Build full synthetic Hub dict containing all backbone parameters and layers.45.* keys
    hub_state = {}
    h = cfg.hidden_size
    v = cfg.vocab_size
    hub_state["model.language_model.embed_tokens.weight"] = torch.randn(v, h)
    hub_state["model.language_model.norm.weight"] = torch.randn(h)
    hub_state["lm_head.weight"] = torch.randn(v, h)

    k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    f_dim = cfg.linear_key_head_dim
    g_dim = cfg.linear_key_head_dim
    nh = cfg.num_attention_heads
    q_nope = cfg.qk_nope_head_dim
    v_hd = cfg.v_head_dim
    mhc_dim = cfg.mhc_expansion_rate * h
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size

    # Backbone layers 0..4
    for l in range(cfg.num_hidden_layers):
      lp = f"model.language_model.layers.{l}."
      for pre in ("attn", "ffn"):
        hub_state[f"{lp}hc_{pre}_fn"] = torch.randn(24, mhc_dim)
        hub_state[f"{lp}hc_{pre}_base"] = torch.randn(24)
        hub_state[f"{lp}hc_{pre}_scale"] = torch.randn(3)
      hub_state[f"{lp}input_layernorm.weight"] = torch.randn(h)
      hub_state[f"{lp}post_attention_layernorm.weight"] = torch.randn(h)

      lt = cfg.layer_types[l]
      if lt == "linear_attention":
        for p in ("q", "k"):
          hub_state[f"{lp}self_attn.{p}_proj.weight"] = torch.randn(k_dim, h)
        hub_state[f"{lp}self_attn.v_proj.weight"] = torch.randn(v_dim, h)
        hub_state[f"{lp}self_attn.b_proj.weight"] = torch.randn(cfg.linear_num_value_heads, h)
        hub_state[f"{lp}self_attn.f_a_proj.weight"] = torch.randn(f_dim, h)
        hub_state[f"{lp}self_attn.f_b_proj.weight"] = torch.randn(v_dim, f_dim)
        hub_state[f"{lp}self_attn.g_a_proj.weight"] = torch.randn(g_dim, h)
        hub_state[f"{lp}self_attn.g_b_proj.weight"] = torch.randn(v_dim, g_dim)
        hub_state[f"{lp}self_attn.o_proj.weight"] = torch.randn(h, v_dim)
        hub_state[f"{lp}self_attn.A_log"] = torch.randn(cfg.linear_num_value_heads)
        hub_state[f"{lp}self_attn.dt_bias"] = torch.randn(v_dim)
        hub_state[f"{lp}self_attn.o_norm.weight"] = torch.randn(cfg.linear_value_head_dim)
        hub_state[f"{lp}self_attn.q_conv1d.weight"] = torch.randn(k_dim, 1, cfg.linear_conv_kernel_dim)
        hub_state[f"{lp}self_attn.k_conv1d.weight"] = torch.randn(k_dim, 1, cfg.linear_conv_kernel_dim)
        hub_state[f"{lp}self_attn.v_conv1d.weight"] = torch.randn(v_dim, 1, cfg.linear_conv_kernel_dim)
      else:
        hub_state[f"{lp}self_attn.q_a_proj.weight"] = torch.randn(cfg.q_lora_rank, h)
        hub_state[f"{lp}self_attn.q_a_layernorm.weight"] = torch.randn(cfg.q_lora_rank)
        hub_state[f"{lp}self_attn.q_b_proj.weight"] = torch.randn(nh * q_nope, cfg.q_lora_rank)
        hub_state[f"{lp}self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(cfg.kv_lora_rank, h)
        hub_state[f"{lp}self_attn.kv_a_layernorm.weight"] = torch.randn(cfg.kv_lora_rank)
        hub_state[f"{lp}self_attn.kv_b_proj.weight"] = torch.randn(nh * (q_nope + v_hd), cfg.kv_lora_rank)
        hub_state[f"{lp}self_attn.o_proj.weight"] = torch.randn(h, nh * v_hd)
        hub_state[f"{lp}self_attn.indexer.wq_b.weight"] = torch.randn(
            cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank
        )
        hub_state[f"{lp}self_attn.indexer.wk.weight"] = torch.randn(cfg.index_head_dim, h)
        hub_state[f"{lp}self_attn.indexer.weights_proj.weight"] = torch.randn(cfg.index_n_heads, h)
        hub_state[f"{lp}self_attn.indexer.k_norm.weight"] = torch.randn(cfg.index_head_dim)
        hub_state[f"{lp}self_attn.indexer.k_norm.bias"] = torch.randn(cfg.index_head_dim)
        hub_state[f"{lp}self_attn.indexer.index_kpool_compress_ape"] = torch.randn(cfg.index_kpool, cfg.index_head_dim)
        hub_state[f"{lp}self_attn.indexer.index_kpool_compress_gate"] = torch.randn(cfg.index_head_dim, h)

      is_moe = l >= cfg.first_k_dense_replace
      if not is_moe:
        for p in ("gate", "up"):
          hub_state[f"{lp}mlp.{p}_proj.weight"] = torch.randn(inter, h)
        hub_state[f"{lp}mlp.down_proj.weight"] = torch.randn(h, inter)
      else:
        for p in ("gate", "up"):
          hub_state[f"{lp}mlp.shared_experts.{p}_proj.weight"] = torch.randn(moe_inter * cfg.n_shared_experts, h)
        hub_state[f"{lp}mlp.shared_experts.down_proj.weight"] = torch.randn(h, moe_inter * cfg.n_shared_experts)
        hub_state[f"{lp}mlp.gate.weight"] = torch.randn(cfg.n_routed_experts, h)
        hub_state[f"{lp}mlp.gate.e_score_correction_bias"] = torch.randn(cfg.n_routed_experts)
        for e in range(cfg.n_routed_experts):
          for p in ("gate", "up"):
            hub_state[f"{lp}mlp.experts.{e}.{p}_proj.weight"] = torch.randn(moe_inter, h)
          hub_state[f"{lp}mlp.experts.{e}.down_proj.weight"] = torch.randn(h, moe_inter)

    # MTP layer (layers.45.*)
    mtp_p = f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."
    hub_state[f"{mtp_p}enorm.weight"] = torch.randn(h)
    hub_state[f"{mtp_p}hnorm.weight"] = torch.randn(h)
    hub_state[f"{mtp_p}eh_proj.weight"] = torch.randn(h, 2 * h)
    hub_state[f"{mtp_p}shared_head.norm.weight"] = torch.randn(h)
    hub_state[f"{mtp_p}input_layernorm.weight"] = torch.randn(h)
    hub_state[f"{mtp_p}post_attention_layernorm.weight"] = torch.randn(h)

    hub_state[f"{mtp_p}self_attn.q_a_proj.weight"] = torch.randn(cfg.q_lora_rank, h)
    hub_state[f"{mtp_p}self_attn.q_a_layernorm.weight"] = torch.randn(cfg.q_lora_rank)
    hub_state[f"{mtp_p}self_attn.q_b_proj.weight"] = torch.randn(nh * q_nope, cfg.q_lora_rank)
    hub_state[f"{mtp_p}self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(cfg.kv_lora_rank, h)
    hub_state[f"{mtp_p}self_attn.kv_a_layernorm.weight"] = torch.randn(cfg.kv_lora_rank)
    hub_state[f"{mtp_p}self_attn.kv_b_proj.weight"] = torch.randn(nh * (q_nope + v_hd), cfg.kv_lora_rank)
    hub_state[f"{mtp_p}self_attn.o_proj.weight"] = torch.randn(h, nh * v_hd)
    hub_state[f"{mtp_p}self_attn.indexer.wq_b.weight"] = torch.randn(
        cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank
    )
    hub_state[f"{mtp_p}self_attn.indexer.wk.weight"] = torch.randn(cfg.index_head_dim, h)
    hub_state[f"{mtp_p}self_attn.indexer.weights_proj.weight"] = torch.randn(cfg.index_n_heads, h)
    hub_state[f"{mtp_p}self_attn.indexer.k_norm.weight"] = torch.randn(cfg.index_head_dim)
    hub_state[f"{mtp_p}self_attn.indexer.k_norm.bias"] = torch.randn(cfg.index_head_dim)
    hub_state[f"{mtp_p}self_attn.indexer.index_kpool_compress_ape"] = torch.randn(cfg.index_kpool, cfg.index_head_dim)
    hub_state[f"{mtp_p}self_attn.indexer.index_kpool_compress_gate"] = torch.randn(cfg.index_head_dim, h)

    for p in ("gate", "up"):
      hub_state[f"{mtp_p}mlp.shared_experts.{p}_proj.weight"] = torch.randn(moe_inter * cfg.n_shared_experts, h)
    hub_state[f"{mtp_p}mlp.shared_experts.down_proj.weight"] = torch.randn(h, moe_inter * cfg.n_shared_experts)
    hub_state[f"{mtp_p}mlp.gate.weight"] = torch.randn(cfg.n_routed_experts, h)
    hub_state[f"{mtp_p}mlp.gate.e_score_correction_bias"] = torch.randn(cfg.n_routed_experts)
    for e in range(cfg.n_routed_experts):
      for p in ("gate", "up"):
        hub_state[f"{mtp_p}mlp.experts.{e}.{p}_proj.weight"] = torch.randn(moe_inter, h)
      hub_state[f"{mtp_p}mlp.experts.{e}.down_proj.weight"] = torch.randn(h, moe_inter)

    # 1. Successful restoration with ignore_mtp_layer=False
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=False)
    np.testing.assert_allclose(
        np.asarray(jax_m.mtp.enorm.weight.value),
        hub_state[f"{mtp_p}enorm.weight"].numpy(),
    )
    np.testing.assert_allclose(
        np.asarray(jax_m.mtp.eh_proj.kernel.value),
        hub_state[f"{mtp_p}eh_proj.weight"].numpy().T,
    )

    # 2. Missing layers.45.* parameter raises ValueError
    missing_state = dict(hub_state)
    del missing_state[f"{mtp_p}eh_proj.weight"]
    with self.assertRaisesRegex(ValueError, "Required parameter missing from HF state"):
      glm5_ckpt.load_hf_state_into_model(jax_m, missing_state, ignore_mtp_layer=False)

    # 3. Unmapped layers.45.* parameter raises ValueError
    unmapped_state = dict(hub_state)
    unmapped_state[f"{mtp_p}unmapped_proj.weight"] = torch.randn(h, h)
    with self.assertRaisesRegex(ValueError, "Unmapped key in hf_tensors"):
      glm5_ckpt.load_hf_state_into_model(jax_m, unmapped_state, ignore_mtp_layer=False)

    # 4. ignore_mtp_layer=False on model without MTP raises ValueError
    cfg_no_mtp, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=0)
    m_no_mtp = modeling_glm5_3_flash.Glm5Model(cfg_no_mtp, rngs=nnx.Rngs(0), place_on_devices=False)
    with self.assertRaisesRegex(ValueError, "requires model.mtp to be initialized"):
      glm5_ckpt.load_hf_state_into_model(m_no_mtp, hub_state, ignore_mtp_layer=False)

    # 5. ignore_mtp_layer=True on model WITH MTP (num_nextn_predict_layers > 0) raises ValueError
    with self.assertRaisesRegex(
        ValueError,
        "ignore_mtp_layer=True cannot be used when model.mtp is initialized",
    ):
      glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)

  def test_mtp_indexer_type_and_layer_idx(self):
    """Test 4: Verifies MTP self-attn layer_idx is 45 with mtp_indexer_type='full', rejecting 'shared'."""
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1)
    jax_m = modeling_glm5_3_flash.Glm5Model(cfg, rngs=nnx.Rngs(0), place_on_devices=False)
    self.assertIsNotNone(jax_m.mtp)
    self.assertEqual(jax_m.mtp.self_attn.layer_idx, glm5_config.MTP_LAYER_INDEX)
    self.assertEqual(jax_m.mtp.self_attn.indexer_type, "full")

    # mtp_indexer_type != 'full' rejected by Glm5Config.__post_init__
    with self.assertRaisesRegex(ValueError, "Shared indexer unsupported"):
      dataclasses.replace(cfg, mtp_indexer_type="shared")

    # indexer_type != 'full' rejected by Glm5DsaMlaAttention.__init__
    with self.assertRaisesRegex(ValueError, "Shared indexer unsupported"):
      modeling_glm5_3_flash.Glm5DsaMlaAttention(
          cfg,
          layer_idx=glm5_config.MTP_LAYER_INDEX,
          indexer_type="shared",
          rngs=nnx.Rngs(0),
      )

    # Glm5DsaMlaAttention requires keyword-only indexer_type
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.Glm5DsaMlaAttention(cfg, layer_idx=0, rngs=nnx.Rngs(0))  # pylint: disable=missing-kwoa

  def test_mtp_training_gate(self):
    """Test 5: 10 @nnx.jit training steps with MTP loss, verifying loss decrease and frozen params."""
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1, scan_layers=False)
    object.__setattr__(cfg, "remat_policy", "none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(101))

    tx = glm5_train.create_glm5_optimizer(learning_rate=2e-2, weight_decay=0.1)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    seq_len = 8
    np.random.seed(777)
    train_tokens = np.random.randint(1, cfg.vocab_size, (1, seq_len))
    train_batch = {
        "inputs": jnp.array(train_tokens),
        "targets": jnp.array(train_tokens),
        "targets_segmentation": jnp.ones((1, seq_len), dtype=jnp.int32),
        "inputs_segmentation": jnp.ones((1, seq_len), dtype=jnp.int32),
    }

    # Snapshot frozen states before training
    params_before = nnx.state(model, nnx.Param)
    indexer_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_before)
        if "indexer" in jax.tree_util.keystr(p)
    }
    self.assertNotEmpty(indexer_before)

    router_bias_before = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    self.assertNotEmpty(router_bias_before)

    @nnx.jit
    def step_fn(m: Any, opt: Any, b: Mapping[str, Any]):
      def loss_fn(mod):
        loss, aux = modeling_glm5_3_flash.glm5_loss_fn(mod, is_train=True, **b)
        return loss, aux

      (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
      opt.update(m, grads)
      return loss, aux, grads

    ce_losses = []
    mtp_losses = []
    for step in range(10):
      loss, aux, grads = step_fn(model, optimizer, train_batch)
      self.assertTrue(jnp.isfinite(loss), f"Loss non-finite at step {step}: {loss}")
      self.assertTrue(jnp.isfinite(aux["ce_loss"]), f"CE loss non-finite at step {step}")
      self.assertTrue(jnp.isfinite(aux["mtp_loss"]), f"MTP loss non-finite at step {step}")
      self.assertTrue(
          jnp.isfinite(aux["moe_lb_loss"]),
          f"MoE LB loss non-finite at step {step}",
      )
      self.assertTrue(
          jnp.isfinite(aux["mtp_moe_lb_loss"]),
          f"MTP MoE LB loss non-finite at step {step}",
      )
      ce_losses.append(float(aux["ce_loss"]))
      mtp_losses.append(float(aux["mtp_loss"]))

      # Verify gradients: zero on all indexers (backbone and MTP) on boundary steps
      if step in (0, 9):
        grad_leaves = {jax.tree_util.keystr(p): np.asarray(g) for p, g in jax.tree.leaves_with_path(grads)}
        for k, g_val in grad_leaves.items():
          self.assertTrue(
              np.all(np.isfinite(g_val)),
              f"Gradient non-finite in {k} at step {step}",
          )
          if "indexer" in k:
            self.assertTrue(
                np.all(g_val == 0.0),
                f"Indexer grad non-zero in {k} at step {step}",
            )

    # Verify directional decrease in both main CE loss and MTP loss
    self.assertLess(
        ce_losses[-1],
        ce_losses[0],
        f"CE loss did not decrease: {ce_losses[0]} -> {ce_losses[-1]}",
    )
    self.assertLess(
        mtp_losses[-1],
        mtp_losses[0],
        f"MTP loss did not decrease: {mtp_losses[0]} -> {mtp_losses[-1]}",
    )

    # Verify is_train=False gates out MTP loss
    _, eval_aux = modeling_glm5_3_flash.glm5_loss_fn(model, is_train=False, **train_batch)
    self.assertEqual(float(eval_aux["mtp_loss"]), 0.0)

    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.compute_glm5_loss(  # pylint: disable=missing-kwoa
          model,
          tokens=train_batch["inputs"],
          decoder_target_tokens=train_batch["targets"],
          decoder_target_mask=train_batch["targets_segmentation"],
      )
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.glm5_loss_fn(model, is_train=1, **train_batch)
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.compute_glm5_loss(
          model,
          tokens=train_batch["inputs"],
          decoder_target_tokens=train_batch["targets"],
          decoder_target_mask=train_batch["targets_segmentation"],
          is_train=1,
      )

    # Verify bit-for-bit identity on indexer parameters
    params_after = nnx.state(model, nnx.Param)
    indexer_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(params_after)
        if "indexer" in jax.tree_util.keystr(p)
    }
    for k, val_before in indexer_before.items():
      self.assertTrue(
          np.array_equal(val_before, indexer_after[k]),
          f"Indexer parameter drifted during training: {k}",
      )

    # Verify bit-for-bit identity on router bias (Glm5RouterBias)
    router_bias_after = {
        jax.tree_util.keystr(p): np.asarray(v)
        for p, v in jax.tree.leaves_with_path(nnx.state(model, modeling_glm5_3_flash.Glm5RouterBias))
    }
    for k, val_before in router_bias_before.items():
      self.assertTrue(
          np.array_equal(val_before, router_bias_after[k]),
          f"Router bias drifted during training: {k}",
      )

  def test_mtp_sharding_fsdp_mesh(self):
    """Test 6: Verifies Glm5Mtp creates and runs on FSDP mesh without duplicate axis error."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))
    sharding = glm5_config.Glm5Sharding.from_mesh(mesh)
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1)
    with jax.sharding.set_mesh(mesh):
      model = modeling_glm5_3_flash.Glm5Model(
          cfg,
          mesh=mesh,
          sharding=sharding,
          rngs=nnx.Rngs(0),
          place_on_devices=False,
      )
      self.assertIsNotNone(model.mtp)

      tokens = jnp.ones((1, 8), dtype=jnp.int32)
      targets = jnp.ones((1, 8), dtype=jnp.int32)
      out = model.forward_with_aux(tokens)
      mtp_logits, _, _ = model.mtp(
          out.hidden_states,
          targets,
          embed_tokens=model.embed_tokens,
          lm_head=model.lm_head,
          decoder_segment_ids=None,
      )
      self.assertEqual(mtp_logits.shape, (1, 8, cfg.vocab_size))

  def test_mtp_vocab_tiling_loss_and_gradient_parity(self):
    """Verifies num_vocab_tiling=1 and 2 produce identical MTP loss, gradients, and engine integration (P2-56)."""
    cfg, _ = _make_cfg(num_hidden_layers=5, num_nextn_predict_layers=1, scan_layers=False)
    object.__setattr__(cfg, "remat_policy", "none")
    model = modeling_glm5_3_flash.create_glm5_3_flash_model(cfg, rngs=nnx.Rngs(42))

    tokens = jnp.array([[2, 4, 6, 8, 10, 12, 14, 16]])
    target_tokens = jnp.array([[4, 6, 8, 10, 12, 14, 16, 18]])
    target_mask = jnp.array([[1, 1, 1, 1, 0, 1, 1, 0]], dtype=jnp.int32)

    # 1. Untiled (num_vocab_tiling=1)
    object.__setattr__(model.config, "num_vocab_tiling", 1)
    (loss_1, aux_1), grads_1 = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.compute_glm5_loss(m, tokens, target_tokens, target_mask, is_train=True),
        has_aux=True,
    )(model)

    # 2. Tiled (num_vocab_tiling=2)
    object.__setattr__(model.config, "num_vocab_tiling", 2)
    (loss_2, aux_2), grads_2 = nnx.value_and_grad(
        lambda m: modeling_glm5_3_flash.compute_glm5_loss(m, tokens, target_tokens, target_mask, is_train=True),
        has_aux=True,
    )(model)

    # Reset
    object.__setattr__(model.config, "num_vocab_tiling", 1)

    np.testing.assert_allclose(float(loss_1), float(loss_2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["mtp_loss"]), float(aux_2["mtp_loss"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(aux_1["ce_loss"]), float(aux_2["ce_loss"]), rtol=1e-5, atol=1e-5)

    # Verify all gradients match, including mtp.* parameters
    g1_leaves = jax.tree.leaves(grads_1)
    g2_leaves = jax.tree.leaves(grads_2)
    self.assertNotEmpty(g1_leaves)
    self.assertEqual(len(g1_leaves), len(g2_leaves))
    for g1, g2 in zip(g1_leaves, g2_leaves):
      np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-4, atol=1e-4)

    # 3. Engine integration with glm5_engine_loss_fn
    if HAS_TUNIX:
      batch = {
          "inputs": tokens,
          "targets": target_tokens,
          "targets_segmentation": target_mask,
      }
      metric_train, aux_train = glm5_train.glm5_engine_loss_fn(model, cfg, batch, None, None, is_train=True)
      expected_total_train, _ = modeling_glm5_3_flash.glm5_loss_fn(
          model,
          inputs=tokens,
          targets=target_tokens,
          targets_segmentation=target_mask,
          is_train=True,
      )
      metric_eval, aux_eval = glm5_train.glm5_engine_loss_fn(model, cfg, batch, None, None, is_train=False)
      train_loss_val = float(metric_train.unreduced_sum / metric_train.denominator)
      eval_loss_val = float(metric_eval.unreduced_sum / metric_eval.denominator)
      np.testing.assert_allclose(
          train_loss_val,
          float(expected_total_train),
          rtol=1e-6,
          atol=1e-6,
      )
      np.testing.assert_allclose(
          eval_loss_val,
          float(aux_eval["ce_loss"] + cfg.moe_lb_loss_weight * aux_eval["moe_lb_loss"]),
          rtol=1e-6,
          atol=1e-6,
      )
      self.assertGreater(float(aux_train["mtp_loss"]), 0.0)
      self.assertGreater(float(aux_train["moe_lb_loss"]), 0.0)

      # is_train=False gates out MTP loss (mtp_loss == 0.0)
      self.assertEqual(float(aux_eval["mtp_loss"]), 0.0)
      self.assertEqual(float(aux_eval["mtp_moe_lb_loss"]), 0.0)

  def test_real_weight_layer_45_mtp(self):
    """Test 7: Loads real Layer-45 weights and verifies next-next-token prediction accuracy."""
    slice_path = os.path.join(_MINI_SLICE_DIR.value, "model.safetensors")
    if not _RUN_REAL_MTP.value or not os.path.exists(slice_path):
      self.skipTest(f"Real mini-slice not requested or not found at {slice_path}")

    cfg_path = os.path.join(_MINI_SLICE_DIR.value, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
      hf_data = json.load(f)
    text_cfg = hf_data["text_config"]
    lin_cfg = text_cfg["linear_attn_config"]

    c = glm5_config.Glm5Config(
        vocab_size=text_cfg["vocab_size"],
        hidden_size=text_cfg["hidden_size"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        first_k_dense_replace=text_cfg["first_k_dense_replace"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        q_lora_rank=text_cfg["q_lora_rank"],
        kv_lora_rank=text_cfg["kv_lora_rank"],
        qk_nope_head_dim=text_cfg["qk_nope_head_dim"],
        qk_rope_head_dim=text_cfg["qk_rope_head_dim"],
        v_head_dim=text_cfg["v_head_dim"],
        index_n_heads=text_cfg["index_n_heads"],
        index_head_dim=text_cfg["index_head_dim"],
        index_topk=text_cfg["index_topk"],
        index_kpool=text_cfg["index_kpool"],
        intermediate_size=text_cfg["intermediate_size"],
        moe_intermediate_size=text_cfg["moe_intermediate_size"],
        n_routed_experts=text_cfg["n_routed_experts"],
        num_experts_per_tok=text_cfg["num_experts_per_tok"],
        linear_num_key_heads=lin_cfg["num_heads"],
        linear_num_value_heads=lin_cfg["num_heads"],
        linear_key_head_dim=lin_cfg["head_dim"],
        linear_value_head_dim=lin_cfg["head_dim"],
        linear_conv_kernel_dim=lin_cfg["short_conv_kernel_size"],
        mlp_clamp_limit=text_cfg["swiglu_limit"],
        routed_scaling_factor=text_cfg["routed_scaling_factor"],
        rms_norm_eps=text_cfg["rms_norm_eps"],
        linear_lower_bound=lin_cfg["gate_lower_bound"],
        mhc_expansion_rate=text_cfg["hc_mult"],
        mhc_sinkhorn_iters=text_cfg["hc_sinkhorn_iters"],
        mhc_eps=text_cfg["hc_eps"],
        n_group=text_cfg["n_group"],
        topk_group=text_cfg["topk_group"],
        norm_topk_prob=text_cfg["norm_topk_prob"],
        scoring_func=text_cfg["scoring_func"],
        n_shared_experts=text_cfg["n_shared_experts"],
        index_kpool_always_select_tail=text_cfg["index_kpool_always_select_tail"],
        layer_types=tuple(
            "full_attention" if lt in ("deepseek_sparse_attention", "indexed_attention") else lt
            for lt in text_cfg["layer_types"][: text_cfg["num_hidden_layers"]]
        ),
        indexer_types=tuple(text_cfg["indexer_types"][: text_cfg["num_hidden_layers"]]),
        num_nextn_predict_layers=1,
        scan_layers=False,
        kda_implementation="xla",
        dtype="bfloat16",
        weight_dtype="bfloat16",
    )

    model = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    lazy_map = glm5_test_utils._LazySafetensorsMapping(slice_path)
    glm5_ckpt.load_hf_state_into_model(model, lazy_map, ignore_mtp_layer=False)
    lazy_map.close()
    del lazy_map
    gc.collect()

    tokenizer = transformers.AutoTokenizer.from_pretrained(_MINI_SLICE_DIR.value, trust_remote_code=True)
    corpus_text = "\n\n".join([zero_gen_logits_checker.GOLDEN_BATCH_TEXT] * 3)
    raw_tokens = tokenizer.encode(corpus_text, add_special_tokens=False)
    seq_len = 514
    self.assertGreaterEqual(len(raw_tokens), seq_len)
    tokens = np.array(raw_tokens[:seq_len], dtype=np.int32)[None, :]

    in_tokens = jnp.asarray(tokens[:, :-2])
    mtp_in_tokens = jnp.asarray(tokens[:, 1:-1])
    ground_truth = tokens[:, 2:]

    out = model.forward_with_aux(in_tokens)
    mtp_logits, _, _ = model.mtp(
        out.hidden_states,
        mtp_in_tokens,
        embed_tokens=model.embed_tokens,
        lm_head=model.lm_head,
        decoder_segment_ids=None,
    )
    predicted_tokens = np.asarray(jnp.argmax(mtp_logits, axis=-1))

    # Control arm with zero hidden states (P2-57)
    mtp_logits_zero, _, _ = model.mtp(
        jnp.zeros_like(out.hidden_states),
        mtp_in_tokens,
        embed_tokens=model.embed_tokens,
        lm_head=model.lm_head,
        decoder_segment_ids=None,
    )
    predicted_tokens_zero = np.asarray(jnp.argmax(mtp_logits_zero, axis=-1))

    del out, mtp_logits, mtp_logits_zero, in_tokens, mtp_in_tokens, model
    jax.clear_caches()
    gc.collect()

    pos_correct = (predicted_tokens == ground_truth).reshape(-1)
    zero_correct = (predicted_tokens_zero == ground_truth).reshape(-1)
    n_total = int(ground_truth.size)
    acc_post = float(np.sum(pos_correct)) / float(n_total)
    acc_zero = float(np.sum(zero_correct)) / float(n_total)
    n_pos_only = int(np.sum(pos_correct & ~zero_correct))
    n_zero_only = int(np.sum(~pos_correct & zero_correct))
    n_disc = n_pos_only + n_zero_only
    p_val = sum(math.comb(n_disc, k) for k in range(n_pos_only, n_disc + 1)) * (0.5**n_disc) if n_disc > 0 else 1.0

    print(
        f"[REAL_MTP] n={n_total} acc_post={acc_post:.4f}"
        f" acc_zero={acc_zero:.4f} margin={acc_post - acc_zero:.4f}"
        f" pos_only={n_pos_only} zero_only={n_zero_only} n_disc={n_disc}"
        f" p_val={p_val:.6e}",
        flush=True,
    )
    self.assertEqual(n_total, 512)
    self.assertGreater(n_pos_only, n_zero_only)
    self.assertLess(p_val, 0.01)


if __name__ == "__main__":
  absltest.main()
