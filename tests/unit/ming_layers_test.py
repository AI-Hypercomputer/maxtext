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

"""Layer-wise tests: MaxText Ming (BailingMoeV3 audio) vs the publisher's HuggingFace remote code.

The HF reference is the `inclusionAI/Ming-3.0-audio-tiny` remote code. Its KDA kernels are rebound to
fla's naive PyTorch references, and gated RMSNorm and short convolution use test-local PyTorch
implementations (`_load_remote_modules`). This avoids relying on version-specific CPU fallbacks in
fla. All tests run on CPU in float32 with small layer dimensions and explicit weight copies.

Audio shape, token handling, and prompt tests also run locally without fetching the HF reference.

No checkpoints, logits or data files are needed: weights are random.
Only the HF remote code is fetched (`MING_HF_PATH` may point at a local HF directory).

Explicit runs require PyTorch, Transformers, fla, and the HF remote code's dependencies
(including `openai-whisper` and `torchaudio`). Missing dependencies or unavailable reference code
raise setup errors rather than skipping the comparisons. Set `HF_HUB_OFFLINE=1` to use cached code.

The talker (text -> speech) tests at the end of the file compare against the HF `talker/` modules
(`MING_TALKER_HF_PATH`, or `$MING_HF_PATH/talker` for a local snapshot) and skip without them.

Run:
  JAX_PLATFORMS=cpu python3 -m pytest tests/unit/ming_layers_test.py
"""

import os
import tempfile
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

import torch
from torch import nn
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MING_HF_MODEL_ID = "inclusionAI/Ming-3.0-audio-tiny"
HF_PATH = os.environ.get("MING_HF_PATH", MING_HF_MODEL_ID)


class TorchShortConvolution(nn.Conv1d):
  """CPU depthwise causal convolution for the HF KDA reference (fixed-length batches)."""

  def __init__(self, hidden_size, kernel_size, bias=False, activation="silu"):
    super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)
    if activation not in (None, "silu", "swish"):
      raise ValueError(f"Unsupported activation: {activation}")
    self.activation = activation

  def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):  # pylint: disable=arguments-renamed
    """Returns output and optional last W inputs, updating a supplied cache in place."""
    if cu_seqlens is not None:
      raise NotImplementedError("Variable-length batches are not supported by the CPU reference")
    width = self.kernel_size[0]
    inputs = x.transpose(1, 2)
    if cache is None:
      history = inputs.new_zeros(inputs.shape[0], inputs.shape[1], width)
    else:
      history = cache
    combined = torch.cat((history, inputs), dim=-1)
    # Drop the oldest history token so each output sees exactly W causal inputs.
    out = super().forward(combined[..., 1:]).transpose(1, 2)
    if self.activation is not None:
      out = torch.nn.functional.silu(out)
    final_state = combined[..., -width:].clone()
    if cache is not None:
      cache.copy_(final_state)
      final_state = cache
    return out, final_state if output_final_state else None


class TorchRMSNormGated(nn.Module):
  """`fla.modules.FusedRMSNormGated` reference: rmsnorm(x) * weight * act(g), float32 math."""

  def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kwargs):
    del kwargs
    super().__init__()
    self.eps = eps
    self.activation = activation
    self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

  def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    """Returns rmsnorm(x) * weight * act(g)."""
    del residual, prenorm, residual_in_fp32
    xf, gf = x.float(), g.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.weight is not None:
      y = y * self.weight.float()
    y = y * torch.sigmoid(gf) if self.activation == "sigmoid" else y * gf * torch.sigmoid(gf)
    return y.to(x.dtype)


def _torch_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,  # pylint: disable=invalid-name
    dt_bias=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    lower_bound=None,
    cu_seqlens=None,
    scale=None,
    **kwargs,
):
  """CPU replacement for `fla.ops.kda.chunk_kda` / `fused_recurrent_kda` built from fla's naive references."""
  from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate  # pylint: disable=import-outside-toplevel
  from fla.ops.kda.naive import naive_recurrent_kda  # pylint: disable=import-outside-toplevel

  del kwargs  # e.g. safe_gate: only selects a kernel variant, same math.
  assert cu_seqlens is None, "varlen is not supported by the CPU reference"
  if use_gate_in_kernel:
    if lower_bound is not None:
      g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)
    else:
      g = naive_kda_gate(g, A_log, dt_bias)
  if use_qk_l2norm_in_kernel:  # fla l2norm: x / sqrt(sum(x^2) + 1e-6)
    q, k = (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + 1e-6) for x in (q, k))
  return naive_recurrent_kda(
      q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
  )


def _load_remote_modules(model_path):
  """Returns the publisher's `(text_module, audio_module)` remote-code modules with CPU kernels rebound."""
  text_cls = get_class_from_dynamic_module("modeling_bailing_moe_v3.BailingMoeV3Model", model_path)
  text_mod = sys.modules[text_cls.__module__]
  text_mod.ShortConvolution = TorchShortConvolution
  text_mod.FusedRMSNormGated = TorchRMSNormGated
  text_mod.chunk_kda = _torch_kda
  text_mod.fused_recurrent_kda = _torch_kda
  return text_mod, None


TEXT_MOD, AUDIO_MOD = None, None

pytestmark = [
    pytest.mark.cpu_only,
]

from flax import nnx  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from jax.sharding import Mesh  # pylint: disable=wrong-import-position

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN  # pylint: disable=wrong-import-position
from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.models import ming  # pylint: disable=wrong-import-position
from maxtext.multimodal import processor_ming  # pylint: disable=wrong-import-position
from maxtext.utils import maxtext_utils  # pylint: disable=wrong-import-position
from maxtext.utils.globals import MAXTEXT_REPO_ROOT  # pylint: disable=wrong-import-position

BASE_CONFIG = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

# Small but structurally faithful dimensions.
EMB, HEADS, HEAD_DIM = 128, 4, 32
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 64, 32, 32, 16, 32
MLP, MOE_MLP, EXPERTS, TOPK, GROUPS, TOPK_GROUPS = 256, 64, 16, 4, 4, 2
VOCAB = 1000
A_D, A_HEADS, A_FFN, A_LAYERS = 64, 4, 256, 2
ATOL, RTOL = 2e-4, 2e-4


def make_config(model_mode_overrides=None, **kwargs):
  """Small, structurally faithful Ming config (CPU, float32)."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="ming-3.0-audio-tiny",
      override_model_config=True,
      base_emb_dim=EMB,
      base_num_query_heads=HEADS,
      base_num_kv_heads=HEADS,
      head_dim=HEAD_DIM,
      q_lora_rank=Q_LORA,
      kv_lora_rank=KV_LORA,
      qk_nope_head_dim=NOPE,
      qk_rope_head_dim=ROPE,
      v_head_dim=V_DIM,
      base_mlp_dim=MLP,
      base_moe_mlp_dim=MOE_MLP,
      num_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      n_routing_groups=GROUPS,
      topk_routing_group=TOPK_GROUPS,
      vocab_size=VOCAB,
      base_num_decoder_layers=4,
      d_model_for_audio=A_D,
      encoder_attention_heads_for_audio=A_HEADS,
      encoder_ffn_dim_for_audio=A_FFN,
      encoder_layers_for_audio=A_LAYERS,
      output_dim_for_audio=EMB,
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=16,
      scan_layers=False,
      attention="dot_product",
      sparse_matmul=False,
      capacity_factor=-1,
      matmul_precision="highest",
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      dropout_rate=0.0,
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  overrides.update(model_mode_overrides or {})
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


def make_hf_text_config():
  """HF BailingMoeV3Config with the same small dimensions as `make_config`."""
  cfg = TEXT_MOD.BailingMoeV3Config(
      vocab_size=VOCAB,
      hidden_size=EMB,
      intermediate_size=MLP,
      num_hidden_layers=4,
      num_attention_heads=HEADS,
      num_key_value_heads=HEADS,
      head_dim=HEAD_DIM,
      rms_norm_eps=1e-6,
      num_experts=EXPERTS,
      num_shared_experts=1,
      num_experts_per_tok=TOPK,
      n_group=GROUPS,
      topk_group=TOPK_GROUPS,
      moe_intermediate_size=MOE_MLP,
      moe_shared_expert_intermediate_size=MOE_MLP,
      first_k_dense_replace=1,
      routed_scaling_factor=2.5,
      layer_group_size=4,
      kv_lora_rank=KV_LORA,
      q_lora_rank=Q_LORA,
      qk_rope_head_dim=ROPE,
      v_head_dim=V_DIM,
      qk_nope_head_dim=NOPE,
      rope_interleave=True,
      rope_theta=6_000_000,
      max_position_embeddings=131072,
      gated_attention_proj_granularity_type="head_wise",
      no_kda_lora=True,
      kda_safe_gate=True,
      kda_lower_bound=-5,
      short_conv_kernel_size=4,
      num_nextn_predict_layers=0,
      pad_token_id=0,
      eos_token_id=1,
  )
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def make_mesh(config):
  return Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)


def t2j(t):
  return jnp.asarray(t.detach().float().cpu().numpy())


def randomize_(module, seed=0, std=0.1):
  """Fills every HF parameter/buffer with random values so no weight copy can silently be skipped."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in list(module.named_parameters()) + list(module.named_buffers()):
      if "inv_freq" in name:
        continue
      if name.endswith("A_log"):
        p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16, generator=g)))
      elif "norm" in name or "_ln" in name or name.endswith("ln_post.weight"):
        p.copy_(1.0 + std * torch.randn(p.shape, generator=g))
      else:
        p.copy_(std * torch.randn(p.shape, generator=g))


def assert_close(jax_out, torch_out, atol=ATOL, rtol=RTOL, msg=""):
  np.testing.assert_allclose(
      np.asarray(jax_out, dtype=np.float32), torch_out.detach().float().numpy(), atol=atol, rtol=rtol, err_msg=msg
  )


# -----------------------------------------------------------------------------
# Weight copy helpers (the same rules are used by checkpoint conversion)
# -----------------------------------------------------------------------------


def set_dense(jax_dense, torch_linear, out_shape=None, in_shape=None):
  w = t2j(torch_linear.weight).T  # [in, out]
  if in_shape is not None or out_shape is not None:
    w = w.reshape(tuple(in_shape or (w.shape[0],)) + tuple(out_shape or (w.shape[-1],)))
  jax_dense.kernel.value = w
  if getattr(torch_linear, "bias", None) is not None and getattr(jax_dense, "bias", None) is not None:
    jax_dense.bias.value = t2j(torch_linear.bias)


def set_conv1d(jax_conv, torch_conv):
  jax_conv.kernel.value = jnp.transpose(t2j(torch_conv.weight), (2, 1, 0))  # [out, in, k] -> [k, in, out]
  if torch_conv.bias is not None:
    jax_conv.bias.value = t2j(torch_conv.bias)


def copy_kda(hf, mt):
  for n in ("q_proj", "k_proj", "v_proj", "f_proj", "b_proj", "g_proj", "o_proj"):
    set_dense(getattr(mt, n), getattr(hf, n))
  for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
    set_conv1d(getattr(mt, n), getattr(hf, n))
  mt.A_log.value = t2j(hf.A_log)
  mt.dt_bias.value = t2j(hf.dt_bias)
  mt.o_norm.scale.value = t2j(hf.o_norm.weight)


def copy_mla(hf, mt):
  set_dense(mt.wq_a, hf.q_a_proj)
  mt.q_norm.scale.value = t2j(hf.q_a_layernorm.weight)
  set_dense(mt.wq_b, hf.q_b_proj, out_shape=(HEADS, NOPE + ROPE))
  set_dense(mt.wkv_a, hf.kv_a_proj_with_mqa)
  mt.kv_norm.scale.value = t2j(hf.kv_a_layernorm.weight)
  set_dense(mt.wkv_b, hf.kv_b_proj, out_shape=(HEADS, NOPE + V_DIM))
  mt.out.kernel.value = t2j(hf.dense.weight).T.reshape(HEADS, V_DIM, EMB)
  set_dense(mt.g_proj, hf.g_proj)


def copy_mlp(hf, mt):
  set_dense(mt.wi_0, hf.gate_proj)
  set_dense(mt.wi_1, hf.up_proj)
  set_dense(mt.wo, hf.down_proj)


def copy_moe(hf, mt):
  routed = mt.MoeBlock_0
  routed.gate.kernel.value = t2j(hf.gate.weight).T
  routed.gate.bias.value = t2j(hf.gate.expert_bias)
  routed.wi_0.value = jnp.stack([t2j(e.gate_proj.weight).T for e in hf.experts])
  routed.wi_1.value = jnp.stack([t2j(e.up_proj.weight).T for e in hf.experts])
  routed.wo.value = jnp.stack([t2j(e.down_proj.weight).T for e in hf.experts])
  copy_mlp(hf.shared_experts, mt.shared_experts)


def copy_decoder_layer(hf, mt):
  """Copies one HF BailingMoeV3DecoderLayer into a MingDecoderLayer."""
  mt.pre_self_attention_layer_norm.scale.value = t2j(hf.input_layernorm.weight)
  mt.post_self_attention_layer_norm.scale.value = t2j(hf.post_attention_layernorm.weight)
  if mt.is_mla:
    copy_mla(hf.attention, mt.self_attention)
  else:
    copy_kda(hf.attention, mt.kda)
  if mt.is_moe:
    copy_moe(hf.mlp, mt.moe_block)
  else:
    copy_mlp(hf.mlp, mt.mlp)


def set_layernorm(jax_ln, torch_ln):
  jax_ln.scale.value = t2j(torch_ln.weight)
  jax_ln.bias.value = t2j(torch_ln.bias)


def copy_whisper(hf_audio, mt_enc):
  """Copies the HF whisper encoder into MingWhisperEncoder."""
  set_conv1d(mt_enc.conv1, hf_audio.conv1)
  set_conv1d(mt_enc.conv2, hf_audio.conv2)
  mt_enc.positional_embedding.value = t2j(hf_audio.positional_embedding)
  for i, blk in enumerate(hf_audio.blocks):
    m = getattr(mt_enc, f"blocks_{i}")
    set_layernorm(m.attn_ln, blk.attn_ln)
    set_layernorm(m.mlp_ln, blk.mlp_ln)
    for n in ("query", "key", "value", "out"):
      set_dense(getattr(m.attn, n), getattr(blk.attn, n))
    set_dense(m.mlp_fc1, blk.mlp[0])
    set_dense(m.mlp_fc2, blk.mlp[2])
  set_layernorm(mt_enc.ln_post, hf_audio.ln_post)


def copy_projector(hf_proj, mt_proj):
  convs = [m for m in hf_proj if isinstance(m, torch.nn.Conv1d)]
  for i, c in enumerate(convs):
    set_conv1d(getattr(mt_proj, f"conv_{i}"), c)
  set_dense(mt_proj.linear, hf_proj[-1])


def causal_mask(batch, seq):
  m = torch.full((seq, seq), float("-inf")).triu(1)
  return m[None, None].expand(batch, 1, seq, seq)


class MingTextLayersTest(unittest.TestCase):
  """KDA, gated MLA, MoE, MLP and full decoder layers vs HF."""

  @classmethod
  def setUpClass(cls):
    global TEXT_MOD, AUDIO_MOD
    try:
      TEXT_MOD, AUDIO_MOD = _load_remote_modules(HF_PATH)
    except (ImportError, OSError) as exc:
      raise RuntimeError(
          f"Cannot load Ming HF reference from {HF_PATH!r}: {exc}. "
          "Install the missing reference dependencies or set MING_HF_PATH to a local HF model directory."
      ) from exc
    torch.set_grad_enabled(False)
    cls.hf_cfg = make_hf_text_config()
    cls.config = make_config()
    cls.mesh = make_mesh(cls.config)
    cls.rope = TEXT_MOD.BailingMoeV3RotaryEmbedding(cls.hf_cfg)

  def setUp(self):
    np.random.seed(0)
    torch.manual_seed(0)

  def _inputs(self, batch=2, seq=12, seed=0):
    x = np.random.RandomState(seed).randn(batch, seq, EMB).astype(np.float32)
    return torch.tensor(x), jnp.asarray(x)

  def _layer(self, idx, model_mode=MODEL_MODE_TRAIN, config=None):
    config = config or self.config
    return ming.MingDecoderLayer(
        config=config, model_mode=model_mode, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1), layer_idx=idx
    )

  def test_layer_pattern_matches_hf(self):
    full = make_config(base_num_decoder_layers=24)
    for i in range(24):
      hf_layer_type = "attention" if ((i + 1) % 4 == 0 or i >= 24 // 4 * 4) else "linear_attention"
      self.assertEqual(ming.is_mla_layer(i, full), hf_layer_type == "attention", i)
      self.assertEqual(ming.is_moe_layer(i, full), i >= 1, i)

  def test_kda_matches_hf(self):
    hf = TEXT_MOD.BailingMoeV3KimiDeltaAttention(self.hf_cfg, layer_idx=0).eval()
    randomize_(hf, seed=1)
    mt = ming.MingKimiDeltaAttention(self.config, self.mesh, MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    copy_kda(hf, mt)
    xt, xj = self._inputs(seq=80)  # > 64 exercises HF's chunk mode
    ref, _, _ = hf(xt)
    out = mt(xj, model_mode=MODEL_MODE_TRAIN)
    assert_close(out, ref, msg="KDA (chunk mode)")
    xt, xj = self._inputs(seq=9, seed=3)  # <= 64 exercises HF's fused_recurrent mode
    ref, _, _ = hf(xt)
    assert_close(mt(xj, model_mode=MODEL_MODE_TRAIN), ref, msg="KDA (recurrent mode)")

  def test_mla_matches_hf(self):
    hf = TEXT_MOD.BailingMoeV3MultiLatentAttention(self.hf_cfg, layer_idx=3).eval()
    randomize_(hf, seed=2)
    layer = self._layer(3)
    mt = layer.self_attention
    copy_mla(hf, mt)
    batch, seq = 2, 12
    xt, xj = self._inputs(batch, seq)
    pos = torch.arange(seq)[None].expand(batch, seq)
    ref, _, _ = hf(xt, position_embeddings=self.rope(xt, pos), attention_mask=causal_mask(batch, seq))
    out, _ = mt(
        xj,
        xj,
        jnp.asarray(pos.numpy()),
        decoder_segment_ids=jnp.ones((batch, seq), jnp.int32),
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )
    assert_close(out, ref, msg="gated MLA")

  def test_dense_mlp_matches_hf(self):
    hf = TEXT_MOD.BailingMoeV3MLP(self.hf_cfg, intermediate_size=MLP).eval()
    randomize_(hf, seed=3)
    layer = self._layer(0)
    copy_mlp(hf, layer.mlp)
    xt, xj = self._inputs()
    assert_close(layer.mlp(xj, deterministic=True), hf(xt), msg="dense MLP")

  def test_moe_matches_hf(self):
    hf = TEXT_MOD.BailingMoeV3SparseMoeBlock(self.hf_cfg, layer_idx=1).eval()
    randomize_(hf, seed=4, std=0.5)
    layer = self._layer(1)
    copy_moe(hf, layer.moe_block)
    xt, xj = self._inputs()
    ref, _ = hf(xt)
    out, _, _ = layer.moe_block(xj)
    assert_close(out, ref, msg="MoE")

  def _check_decoder_layer(self, idx, seed):
    """Compares HF and MaxText decoder layer `idx` with random weights."""
    hf = TEXT_MOD.BailingMoeV3DecoderLayer(self.hf_cfg, layer_idx=idx).eval()
    randomize_(hf, seed=seed)
    mt = self._layer(idx)
    copy_decoder_layer(hf, mt)
    batch, seq = 2, 12
    xt, xj = self._inputs(batch, seq, seed=seed)
    pos = torch.arange(seq)[None].expand(batch, seq)
    ref = hf(xt, attention_mask=causal_mask(batch, seq), position_ids=pos, position_embeddings=self.rope(xt, pos))[0]
    out, _ = mt(xj, jnp.ones((batch, seq), jnp.int32), jnp.asarray(pos.numpy()), True, MODEL_MODE_TRAIN)
    assert_close(out, ref, msg=f"decoder layer {idx}")

  def test_decoder_layer_kda_dense(self):
    self._check_decoder_layer(0, seed=5)

  def test_decoder_layer_kda_moe(self):
    self._check_decoder_layer(1, seed=6)

  def test_decoder_layer_mla_moe(self):
    self._check_decoder_layer(3, seed=7)

  def test_kda_cached_decode_matches_full_forward(self):
    """Right-padded prefill + token-by-token AR decode reproduces the full (train-mode) KDA output."""
    hf = TEXT_MOD.BailingMoeV3KimiDeltaAttention(self.hf_cfg, layer_idx=0).eval()
    randomize_(hf, seed=8)
    total, prompt = 10, 6
    cfg = self.config
    full = ming.MingKimiDeltaAttention(cfg, self.mesh, MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    pre = ming.MingKimiDeltaAttention(cfg, self.mesh, MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
    copy_kda(hf, full)
    copy_kda(hf, pre)
    _, xj = self._inputs(1, total, seed=9)
    ref = full(xj, model_mode=MODEL_MODE_TRAIN)

    # Prefill padded to max_prefill_predict_length with garbage after `prompt` tokens.
    pad = cfg.max_prefill_predict_length
    xp = jnp.concatenate([xj[:, :prompt], 3.0 * jnp.ones((1, pad - prompt, EMB))], axis=1)
    seg = (jnp.arange(pad) < prompt).astype(jnp.int32)[None]
    out_p = pre(xp, model_mode=MODEL_MODE_PREFILL, decoder_segment_ids=seg)
    np.testing.assert_allclose(out_p[:, :prompt], ref[:, :prompt], atol=ATOL, rtol=RTOL)

    ar = ming.MingKimiDeltaAttention(cfg, self.mesh, MODEL_MODE_AUTOREGRESSIVE, rngs=nnx.Rngs(0))
    copy_kda(hf, ar)
    rec, conv = pre.cache.get_gdn_states()
    ar.cache.update_gdn_states(rec, conv)
    for t in range(prompt, total):
      o = ar(xj[:, t : t + 1], model_mode=MODEL_MODE_AUTOREGRESSIVE)
      np.testing.assert_allclose(o[:, 0], ref[:, t], atol=ATOL, rtol=RTOL, err_msg=f"AR step {t}")


class TorchShortConvolutionTest(unittest.TestCase):
  """Checks causal alignment and channel isolation in the CPU reference."""

  def test_causal_depthwise_convolution(self):
    conv = TorchShortConvolution(2, 3, bias=True, activation=None)
    with torch.no_grad():
      conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]))
      conv.bias.copy_(torch.tensor([0.5, -0.5]))
    x = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]])
    out, state = conv(x)
    torch.testing.assert_close(out, torch.tensor([[[3.5, 59.5], [8.5, 169.5], [14.5, 319.5], [20.5, 469.5]]]))
    self.assertIsNone(state)

  def test_cached_chunks_match_full_sequence(self):
    for width in (1, 4):
      with self.subTest(width=width):
        conv = TorchShortConvolution(2, width, activation="silu")
        x = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2) / 10
        full, _ = conv(x)
        prefix, cache = conv(x[:, :2], output_final_state=True)
        outputs = [prefix]
        for t in range(2, 6):
          out, updated = conv(x[:, t : t + 1], cache=cache, output_final_state=True)
          self.assertIs(updated, cache)
          outputs.append(out)
        torch.testing.assert_close(torch.cat(outputs, dim=1), full)
        torch.testing.assert_close(cache, x[:, -width:].transpose(1, 2))


# Audio shape and token handling checks (no publisher HF setup required).
@pytest.mark.parametrize("length", [1, 17, 2999, 3000, 3001, 6001])
def test_projected_token_count(length):
  """Preprocessing and encoder/projector agree at chunk and stride boundaries."""
  expected = sum(
      ming.projector_out_length(ming.whisper_conv_out_length(chunk), 3)
      for chunk in ming.audio_chunk_lengths(length, 3000)
  )
  assert processor_ming.get_num_audio_tokens(length) == expected


def test_audio_token_expansion_and_offsets():
  output = SimpleNamespace(audio_lengths=np.array([3], dtype=np.int32))
  pad = processor_ming.AUDIO_PAD_ID
  tokens = np.array([11, pad, 12], dtype=np.int32)
  np.testing.assert_array_equal(processor_ming.add_extra_tokens_for_ming(tokens, output), [11, pad, pad, pad, 12])
  assert processor_ming.get_audio_offsets_ming(output) == 2
  with pytest.raises(ValueError, match="1D array"):
    processor_ming.add_extra_tokens_for_ming(tokens[None], output)
  with pytest.raises(ValueError, match="Found 0"):
    processor_ming.add_extra_tokens_for_ming(np.array([11, 12]), output)


@pytest.mark.parametrize("output", [None, SimpleNamespace(), SimpleNamespace(audio_lengths=None)])
def test_audio_tokens_without_lengths(output):
  tokens = np.array([11, 12], dtype=np.int32)
  assert processor_ming.get_audio_offsets_ming(output) == 0
  assert processor_ming.add_extra_tokens_for_ming(tokens, output) is tokens


@pytest.mark.parametrize("shape", [(4, 128), (0, 4, 128)])
def test_audio_encoder_rejects_invalid_shape(shape):
  encoder = ming.MingWhisperEncoder(make_config(), None, rngs=nnx.Rngs(0))
  with pytest.raises(ValueError, match="expected mel features"):
    encoder(jnp.zeros(shape))


def test_audio_encoder_and_projector_preserve_batch_order():
  config = make_config(ming_audio_chunk_size=16)
  encoder = ming.MingWhisperEncoder(config, None, rngs=nnx.Rngs(0))
  projector = ming.MingAudioProjector(config, rngs=nnx.Rngs(1))
  features = jnp.arange(2 * 17 * 128, dtype=jnp.float32).reshape((2, 17, 128)) / 1000

  batched = projector(encoder(features), mel_length=features.shape[1])
  separate = jnp.concatenate(
      [projector(encoder(features[i : i + 1]), mel_length=features.shape[1]) for i in range(features.shape[0])],
      axis=0,
  )

  assert batched.shape[0] == 2
  np.testing.assert_allclose(np.asarray(batched), np.asarray(separate), atol=1e-5, rtol=1e-5)


def test_prompt_places_audio_before_text():
  prompt = processor_ming.reformat_prompt_ming("transcribe", num_audios=1)
  assert prompt.index(processor_ming.AUDIO_PAD_TOKEN) < prompt.index("transcribe")
  assert processor_ming.reformat_prompt_ming("<|audio|>transcribe", num_audios=1) == prompt


class MingAudioTest(unittest.TestCase):
  """Whisper encoder + projector and audio preprocessing vs HF."""

  def create_tempdir_path(self):
    tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(tmp.cleanup)
    return tmp.name

  @classmethod
  def setUpClass(cls):
    torch.set_grad_enabled(False)
    pytest.importorskip("whisper")
    pytest.importorskip("torchaudio")
    global TEXT_MOD, AUDIO_MOD  # pylint: disable=global-statement
    audio_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3_audio.BailingMoeV3AudioForConditionalGeneration", HF_PATH
    )
    AUDIO_MOD = sys.modules[audio_cls.__module__]
    text_module = sys.modules[AUDIO_MOD.BailingMoeV3Model.__module__]
    TEXT_MOD = text_module
    text_module.FusedRMSNormGated = TorchRMSNormGated
    text_module.chunk_kda = _torch_kda
    text_module.fused_recurrent_kda = _torch_kda
    cls.config = make_config()
    cls.mesh = make_mesh(cls.config)

  def _hf_audio_model(self, seed=0):
    """A small BailingMoeV3AudioForConditionalGeneration (only audio parts are exercised)."""
    audio_cfg = {
        "n_mels": 128,
        "n_audio_ctx": 3000,
        "n_audio_state": A_D,
        "n_audio_head": A_HEADS,
        "n_audio_layer": A_LAYERS,
        "downsampler_kernel_size_list": [3, 3, 3],
        "downsampler_stride_list": [2, 2, 2],
    }
    text_cfg = make_hf_text_config().to_dict()
    text_cfg["num_hidden_layers"] = 1
    cfg = AUDIO_MOD.BailingMoeV3AudioConfig(text_config=text_cfg, audio_config=audio_cfg, norm_query_embeds=True)
    cfg.text_config._attn_implementation = "eager"  # pylint: disable=protected-access
    model = AUDIO_MOD.BailingMoeV3AudioForConditionalGeneration(cfg).eval()
    # whisper blocks hardcode mlp hidden = 4 * n_state
    self.assertEqual(model.model.audio.blocks[0].mlp[0].out_features, A_FFN)
    randomize_(model.model.audio, seed=seed)
    randomize_(model.linear_proj_audio, seed=seed + 1)
    return model

  def _check_audio(self, mel_length, seed):
    """Compares HF and MaxText audio features for random mel input of `mel_length` frames."""
    hf = self._hf_audio_model(seed)
    enc = ming.MingWhisperEncoder(self.config, self.mesh, rngs=nnx.Rngs(0))
    proj = ming.MingAudioProjector(self.config, rngs=nnx.Rngs(0))
    copy_whisper(hf.model.audio, enc)
    copy_projector(hf.linear_proj_audio, proj)
    mel = np.random.RandomState(seed).randn(1, mel_length, 128).astype(np.float32)
    ref = hf._get_audio_features(torch.tensor(mel), torch.tensor([[mel_length]]))  # pylint: disable=protected-access
    out = proj(enc(jnp.asarray(mel)), mel_length=mel_length)
    self.assertEqual(out.shape[1], processor_ming.get_num_audio_tokens(mel_length))
    assert_close(out, ref, atol=5e-4, rtol=5e-4, msg=f"audio features (mel_length={mel_length})")

  def test_audio_encoder_single_chunk(self):
    self._check_audio(248, seed=10)

  def test_audio_encoder_multi_chunk(self):
    # 2 chunks: 3000 + 457 mel frames; the short chunk is zero-padded without a mask, like HF.
    self._check_audio(3457, seed=11)

  def test_resample_matches_torchaudio(self):
    import torchaudio  # pylint: disable=import-outside-toplevel

    wav = np.random.RandomState(0).randn(2, 24000 + 123).astype(np.float32)
    ref = torchaudio.functional.resample(torch.tensor(wav), 24000, 16000).numpy()
    out = processor_ming.resample(wav, 24000, 16000)
    self.assertEqual(out.shape, ref.shape)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-4)

  def test_preprocessing_matches_hf_processor(self):
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    # Synthetic 2.5 s stereo clip at 24 kHz (exercises resampling to 16 kHz and channel averaging).
    t = np.arange(int(2.5 * 24000)) / 24000
    rng = np.random.RandomState(0)
    wav = np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 1234 * t)], axis=1)
    wav = (0.3 * wav + 0.01 * rng.randn(*wav.shape)).astype(np.float32)
    audio_file = os.path.join(self.create_tempdir_path(), "synthetic.wav")
    sf.write(audio_file, wav, 24000, subtype="FLOAT")

    proc = AutoProcessor.from_pretrained(HF_PATH, trust_remote_code=True, use_fast=False)
    prompt = "Please answer the question in the audio."
    messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_file}, {"type": "text", "text": prompt}]}]
    hf_inputs = proc(messages)

    audio_values, _, num_tokens = processor_ming.preprocess_audio_ming(audio_file)
    np.testing.assert_allclose(audio_values, hf_inputs["audio_feats"].numpy(), atol=2e-4, rtol=1e-4)

    hf_prompt = proc.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    mt_prompt = processor_ming.reformat_prompt_ming(prompt, num_audios=1)
    self.assertEqual(mt_prompt, hf_prompt)
    ids = np.asarray(proc.tokenizer(mt_prompt)["input_ids"], dtype=np.int32)

    class _Out:
      audio_lengths = np.asarray([num_tokens])

    expanded = processor_ming.add_extra_tokens_for_ming(ids, _Out())
    np.testing.assert_array_equal(expanded, hf_inputs["input_ids"][0].numpy())


class MingScanTest(unittest.TestCase):
  """`scan_layers=True` (unrolled prefix + scanned MingScannableBlocks) matches the unscanned stack."""

  def test_scan_layout(self):
    self.assertEqual(ming.scan_layout(make_config(base_num_decoder_layers=24)), (4, 5, 24))
    self.assertEqual(ming.scan_layout(make_config(base_num_decoder_layers=14)), (4, 2, 12))
    self.assertEqual(ming.scan_layout(make_config(base_num_decoder_layers=4)), (4, 0, 4))

  def test_scanned_matches_unscanned(self):
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from maxtext.layers import moe  # pylint: disable=import-outside-toplevel
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    # 4 unrolled, 2 scanned blocks, 2 unrolled.
    common = {"base_num_decoder_layers": 14, "use_multimodal": False, "use_audio": False}
    cfg_u = make_config(scan_layers=False, **common)
    cfg_s = make_config(scan_layers=True, **common)
    mesh = make_mesh(cfg_u)
    mu = model_creation_utils.from_config(cfg_u, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    ms = model_creation_utils.from_config(cfg_s, mesh=mesh, rngs=nnx.Rngs(params=1, dropout=1))
    num_prefix, num_blocks, _ = ming.scan_layout(cfg_s)
    cycle = cfg_s.inhomogeneous_layer_cycle_interval

    var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
    src = {tuple(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(mu, var_filter))}
    dst_state = nnx.state(ms, var_filter)
    new_flat = []
    for path, var in nnx.to_flat_state(dst_state):
      key = tuple(map(str, path))
      if "scanned_blocks" in key:
        i = key.index("scanned_blocks")
        j = int(key[i + 1].split("_")[1])
        slices = [
            src[key[:i] + (f"layers_{num_prefix + b * cycle + j}",) + key[i + 2 :]].get_value() for b in range(num_blocks)
        ]
        value = jnp.stack(slices, axis=0)
        if isinstance(var, nnx.Param) and value.ndim > cfg_s.param_scan_axis:
          value = jnp.moveaxis(value, 0, cfg_s.param_scan_axis)
      else:
        value = src[key].get_value()
      self.assertEqual(value.shape, var.get_value().shape, key)
      new_flat.append((path, var.replace(value)))
    nnx.update(ms, nnx.from_flat_state(new_flat))

    batch, seq = 1, 12
    ids = jnp.asarray(np.random.RandomState(0).randint(0, VOCAB, (batch, seq)), jnp.int32)
    pos = jnp.arange(seq, dtype=jnp.int32)[None]
    seg = jnp.ones((batch, seq), jnp.int32)
    outs = []
    for m, c in ((mu, cfg_u), (ms, cfg_s)):
      with mesh, nn_partitioning.axis_rules(c.logical_axis_rules):
        outs.append(m(decoder_input_tokens=ids, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False))
    np.testing.assert_allclose(np.asarray(outs[1]), np.asarray(outs[0]), atol=1e-4, rtol=1e-4)


# -----------------------------------------------------------------------------
# Talker (text -> speech)
# -----------------------------------------------------------------------------
#
# Compares the CFM DiT (with classifier-free guidance), the Euler sampler, the aggregator and the AudioVAE
# decoder with the publisher's `talker/` PyTorch modules (needs `x-transformers` and `diffusers`). The talker
# LLM is MaxText's regular Qwen2 decoder and is covered by a separate end-to-end HF parity check.
# The reference defaults to `$MING_HF_PATH/talker` for a local snapshot; `MING_TALKER_HF_PATH` overrides it
# and `MING_TALKER_EXTRA_PATH` may point at a side directory with the extra packages. Without a reference
# the comparisons are skipped; the frontend tests always run.

_LOCAL_TALKER_PATH = os.path.join(HF_PATH, "talker")
TALKER_HF_PATH = os.environ.get("MING_TALKER_HF_PATH") or (
    _LOCAL_TALKER_PATH if os.path.isdir(_LOCAL_TALKER_PATH) else ""
)
TALKER_EXTRA_PATH = os.environ.get("MING_TALKER_EXTRA_PATH", "")

# Talker: small but structurally faithful dimensions.
TALKER_EMB, TALKER_LATENT = 32, 8
TALKER_DIT_HIDDEN, TALKER_DIT_HEADS, TALKER_DIT_LAYERS = 64, 4, 2
(
    TALKER_VAE_HIDDEN,
    TALKER_VAE_HEADS,
    TALKER_VAE_KV_HEADS,
    TALKER_VAE_MLP,
    TALKER_VAE_LAYERS,
    TALKER_VAE_WINDOW,
    TALKER_VAE_HOP,
) = (64, 4, 2, 128, 2, 8, 16)


def make_talker_config(**kwargs):
  """Small, structurally faithful Ming talker config (CPU, float32)."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="ming-3.0-audio-tiny-talker",
      override_model_config=True,
      base_emb_dim=TALKER_EMB,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      head_dim=8,
      base_mlp_dim=64,
      base_num_decoder_layers=1,
      vocab_size=128,
      ming_talker_latent_dim=TALKER_LATENT,
      ming_talker_speaker_dim=16,
      ming_talker_dit_hidden_size=TALKER_DIT_HIDDEN,
      ming_talker_dit_num_layers=TALKER_DIT_LAYERS,
      ming_talker_dit_num_heads=TALKER_DIT_HEADS,
      ming_talker_aggregator_hidden_size=TALKER_DIT_HIDDEN,
      ming_talker_aggregator_num_layers=TALKER_DIT_LAYERS,
      ming_talker_aggregator_num_heads=TALKER_DIT_HEADS,
      ming_vae_hidden_size=TALKER_VAE_HIDDEN,
      ming_vae_num_layers=TALKER_VAE_LAYERS,
      ming_vae_num_heads=TALKER_VAE_HEADS,
      ming_vae_num_kv_heads=TALKER_VAE_KV_HEADS,
      ming_vae_mlp_dim=TALKER_VAE_MLP,
      ming_vae_sliding_window=TALKER_VAE_WINDOW,
      ming_vae_hop_length=TALKER_VAE_HOP,
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=16,
      scan_layers=False,
      matmul_precision="highest",
      dtype="float32",
      weight_dtype="float32",
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


class MingTalkerFrontendTest(unittest.TestCase):
  """Reference-free checks of the text frontend."""

  def test_ip_instruction(self):
    instruction = processor_ming.build_ip_instruction("灵小柔")
    self.assertEqual(
        instruction,
        '{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null,'
        ' "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null,'
        ' "ENV": null, "SNR": null}, "IP": "灵小柔"}]}',
    )

  def test_prompt_language(self):
    default = processor_ming.TALKER_TTS_PROMPT
    self.assertEqual(processor_ming.build_tts_prompt(None), default)
    self.assertEqual(processor_ming.build_tts_prompt("en"), default)
    self.assertEqual(processor_ming.build_tts_prompt("ja"), default + "Target language: Japanese.\n")

  def test_split_text(self):
    self.assertEqual(processor_ming.split_text("Hello world"), ["Hello world."])
    frags = processor_ming.split_text("One two three. Four five six! " * 4, max_length=40)
    self.assertTrue(all(len(f) <= 40 for f in frags))
    self.assertEqual(" ".join(frags), ("One two three. Four five six! " * 4).strip())

  def test_trim_trailing_silence(self):
    sr = 1000
    speech = np.concatenate([0.5 * np.ones(250), np.zeros(1000), 0.5 * np.ones(30)])
    # Frames of 100: last loud frame is frame 2 (samples 200-299); keep 0.3 s after it; remainder 80 samples dropped.
    out = processor_ming.trim_trailing_silence(speech[:1250], sr)
    self.assertEqual(out.shape[0], 300 + 300)
    self.assertEqual(processor_ming.trim_trailing_silence(np.zeros(50), sr).shape[0], 50)

  def test_timesteps(self):
    t = np.asarray([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], np.float32) / 32
    np.testing.assert_allclose(ming.epss_timesteps(10), 1 - np.cos(np.pi / 2 * t), atol=1e-6)


@pytest.mark.skipif(
    not TALKER_HF_PATH, reason="no talker reference: set MING_TALKER_HF_PATH or a local MING_HF_PATH snapshot"
)
class MingTalkerLayersTest(unittest.TestCase):
  """MaxText talker heads vs the publisher's PyTorch modules with identical random weights."""

  @classmethod
  def setUpClass(cls):
    for path in (TALKER_EXTRA_PATH, TALKER_HF_PATH):
      if path and path not in sys.path:
        sys.path.insert(0, path)
    from talker_module import aggregator, cfm, dit  # pylint: disable=import-outside-toplevel,import-error
    from AudioVAE import vae_modules  # pylint: disable=import-outside-toplevel,import-error

    cls.hf_dit, cls.hf_cfm, cls.hf_agg, cls.hf_vae = dit, cfm, aggregator, vae_modules
    torch.manual_seed(0)
    cls.config = make_talker_config()
    cls.heads = ming.MingTalkerHeads(cls.config, rngs=nnx.Rngs(0))

  # ---------------------------------------------------------------- helpers
  def t2j(self, t):
    return jnp.asarray(t.detach().float().cpu().numpy())

  def randomize(self, module, seed, std=0.1):
    """Fills HF parameters with random values; RoPE buffers are bf16-rounded like the checkpoint."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
      for name, p in list(module.named_parameters()) + list(module.named_buffers()):
        if "inv_freq" in name:
          # The publisher's checkpoint stores bf16-rounded RoPE frequencies.
          p.copy_(p.to(torch.bfloat16).float())
        elif name.endswith("window"):
          continue
        elif "norm" in name:
          p.copy_(1.0 + std * torch.randn(p.shape, generator=g))
        else:
          p.copy_(std * torch.randn(p.shape, generator=g))

  def set_dense(self, mt, hf):
    mt.kernel.value = self.t2j(hf.weight).T
    if getattr(hf, "bias", None) is not None:
      mt.bias.value = self.t2j(hf.bias)

  def set_norm(self, mt, hf):
    mt.scale.value = self.t2j(hf.weight)

  def copy_blocks(self, mt, hf_blocks):
    for i, blk in enumerate(hf_blocks):
      m = getattr(mt, f"blocks_{i}")
      self.set_norm(m.norm1, blk.norm1)
      self.set_norm(m.norm2, blk.norm2)
      for n in ("to_q", "to_k", "to_v"):
        self.set_dense(getattr(m.attn, n), getattr(blk.attn, n))
      self.set_dense(m.attn.to_out, blk.attn.to_out[0])
      self.set_dense(m.mlp_in, blk.mlp.ff[0][0])
      self.set_dense(m.mlp_out, blk.mlp.ff[2])

  def make_dit(self):
    """Builds a random HF DiT and copies its weights into the MaxText CFM DiT."""
    hf = self.hf_dit.DiT(
        in_channels=TALKER_LATENT,
        hidden_size=TALKER_DIT_HIDDEN,
        depth=TALKER_DIT_LAYERS,
        num_heads=TALKER_DIT_HEADS,
        mlp_ratio=4,
        llm_cond_dim=TALKER_EMB,
        attn_backend="torch",
        attn_mask_enabled=False,
        dropout=0.0,
    ).eval()
    self.randomize(hf, 1)
    mt = self.heads.cfm
    self.set_dense(mt.x_embedder, hf.x_embedder)
    self.set_dense(mt.c_embedder, hf.c_embedder.cond_embedder)
    self.set_dense(mt.time_fc1, hf.t_embedder.time_mlp[0])
    self.set_dense(mt.time_fc2, hf.t_embedder.time_mlp[2])
    self.copy_blocks(mt, hf.blocks)
    self.set_norm(mt.final_norm, hf.final_layer.norm_final)
    self.set_dense(mt.final_linear, hf.final_layer.linear)
    return hf, mt

  def assert_close(self, mt_out, hf_out, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(np.asarray(mt_out, np.float32), hf_out.detach().float().numpy(), atol=atol, rtol=rtol)

  # ---------------------------------------------------------------- tests
  def test_dit_rope_matches_checkpoint_buffers(self):
    ref = (1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))).to(torch.bfloat16).float()
    np.testing.assert_array_equal(ming.dit_rope_freqs(2, 64)[1, ::2], ref.numpy())
    hann = torch.hann_window(3528).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(ming.talker_bf16_round(ming.periodic_hann_window(3528)), hann)

  def test_dit_forward_with_cfg(self):
    hf, mt = self.make_dit()
    g = torch.Generator().manual_seed(2)
    x, hist = torch.randn(2, 1, TALKER_LATENT, generator=g), torch.randn(2, 1, TALKER_LATENT, generator=g)
    cond = torch.randn(2, 1, TALKER_EMB, generator=g)
    t = torch.tensor(0.3)
    with torch.no_grad():
      pred, null_pred = torch.chunk(hf.forward_with_cfg(x, t, cond, hist), 2, dim=0)
      expected = pred + (pred - null_pred) * 2.0
    out = mt.forward_with_cfg(self.t2j(x), jnp.float32(0.3), self.t2j(cond), self.t2j(hist), 2.0)
    self.assert_close(out, expected)

  def test_cfm_sample(self):
    hf_dit, mt = self.make_dit()
    hf = self.hf_cfm.CFM(hf_dit, steps=10)
    g = torch.Generator().manual_seed(3)
    noise, hist = torch.randn(1, 1, TALKER_LATENT, generator=g), torch.randn(1, 1, TALKER_LATENT, generator=g)
    cond = torch.randn(1, 1, TALKER_EMB, generator=g)
    sde = torch.randn(10, 1, 1, TALKER_LATENT, generator=g)
    timesteps = self.hf_cfm.get_epss_timesteps(10, "cpu", torch.float32)
    for sigma, temperature in ((0.25, 0.0), (0.25, 0.5)):
      with torch.no_grad():
        expected = hf.sample(cond, hist, noise, timesteps, torch.tensor([2.0, sigma, temperature]), sde)
      out = ming.cfm_sample(
          mt, self.t2j(cond), self.t2j(hist), self.t2j(noise), 10, 2.0, sigma, temperature, self.t2j(sde)
      )
      self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

  def test_aggregator(self):
    hf = self.hf_agg.Aggregator(
        in_channels=TALKER_LATENT,
        hidden_size=TALKER_DIT_HIDDEN,
        depth=TALKER_DIT_LAYERS,
        num_heads=TALKER_DIT_HEADS,
        mlp_ratio=4,
        llm_input_dim=TALKER_EMB,
        attn_backend="torch",
        attn_mask_enabled=False,
        dropout=0.0,
    ).eval()
    self.randomize(hf, 4)
    mt = self.heads.aggregator
    mt.word_embedder.value = self.t2j(hf.word_embedder.weight)
    self.set_dense(mt.x_embedder, hf.x_embedder)
    self.copy_blocks(mt, hf.blocks)
    self.set_norm(mt.final_norm, hf.final_layer.norm_final)
    self.set_dense(mt.final_linear, hf.final_layer.linear)
    latent = torch.randn(3, 1, TALKER_LATENT, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
      expected = hf(latent)
    self.assert_close(mt(self.t2j(latent)), expected)

  def test_vae_decoder(self):
    backbone = {
        "hidden_size": TALKER_VAE_HIDDEN,
        "intermediate_size": TALKER_VAE_MLP,
        "num_hidden_layers": TALKER_VAE_LAYERS,
        "num_attention_heads": TALKER_VAE_HEADS,
        "num_key_value_heads": TALKER_VAE_KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "sliding_window": TALKER_VAE_WINDOW,
        "use_sliding_window": True,
        "max_window_layers": 0,
        "max_position_embeddings": 4096,
        "vocab_size": 1,
        "is_causal": True,
        "use_cache": False,
        "_attn_implementation": "eager",
        "attn_implementation": "eager",
        "hidden_act": "silu",
        "model_type": "qwen2",
    }
    hf = self.hf_vae.Decoder(
        decoder_args=backbone, output_dim=TALKER_VAE_HOP, latent_dim=TALKER_LATENT, patch_size=4
    ).eval()
    self.randomize(hf, 6, std=0.05)
    mt = self.heads.audio_vae
    self.set_dense(mt.fc1, hf.fc1)
    for i, layer in enumerate(hf.decoder.layers):
      m = getattr(mt, f"layers_{i}")
      self.set_norm(m.input_layernorm, layer.input_layernorm)
      self.set_norm(m.post_attention_layernorm, layer.post_attention_layernorm)
      for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        self.set_dense(getattr(m, n), getattr(layer.self_attn, n))
      for n in ("gate_proj", "up_proj", "down_proj"):
        self.set_dense(getattr(m, n), getattr(layer.mlp, n))
    self.set_norm(mt.norm, hf.decoder.norm)
    self.set_dense(mt.head_out, hf.head.out)
    # 6 latents -> 24 frames > sliding window 8, so the window mask is exercised.
    latents = torch.randn(1, 6, TALKER_LATENT, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
      expected, _, _ = hf.low_level_reconstruct(
          latents, use_cache=False, stream_state=(None, None, None), last_chunk=True
      )
    out = mt(self.t2j(latents))
    self.assertEqual(out.shape, (1, 6 * 4 * TALKER_VAE_HOP))
    self.assert_close(out, expected[:, 0], atol=5e-4, rtol=5e-3)
    # Edge-padding the latents (shape bucketing in ming_talker_decode) leaves the real samples unchanged.
    padded = jnp.pad(self.t2j(latents), ((0, 0), (0, 5), (0, 0)), mode="edge")
    padded_out = mt(padded, jnp.int32(6))
    np.testing.assert_allclose(np.asarray(padded_out[:, : out.shape[1]]), np.asarray(out), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
