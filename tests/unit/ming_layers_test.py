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

The HF reference is the `inclusionAI/Ming-3.0-audio-tiny` remote code, loaded unmodified through
`trust_remote_code`. Its KDA layers call GPU-only `flash-linear-attention` (fla) kernels; on CPU fla's
fallbacks reject the in-kernel gate arguments and `FusedRMSNormGated` needs a GPU, so those symbols are
rebound to fla's own PyTorch references (`_load_remote_modules`). Everything else, including fla's
`ShortConvolution`, runs as published. All tests run on CPU in float32 with small layer dimensions and
explicit weight copies.

No checkpoints, logits or data files are needed: weights are random and the audio clip is synthesized.
Only the HF remote code and tokenizer are fetched (`MING_HF_PATH` may point at a local HF directory).

Run (`openai-whisper` and `torchaudio` must be importable, e.g. via PYTHONPATH):
  JAX_PLATFORMS=cpu python3 -m pytest tests/unit/ming_layers_test.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pytest

try:
  import torch
  from torch import nn
  from transformers import AutoProcessor
  from transformers.dynamic_module_utils import get_class_from_dynamic_module
  import whisper  # pylint: disable=unused-import  # imported by the remote audio modeling code

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

MING_HF_MODEL_ID = "inclusionAI/Ming-3.0-audio-tiny"
HF_PATH = os.environ.get("MING_HF_PATH", MING_HF_MODEL_ID)

if HAS_TORCH:

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
  audio_cls = get_class_from_dynamic_module(
      "modeling_bailing_moe_v3_audio.BailingMoeV3AudioForConditionalGeneration", model_path
  )
  text_mod, audio_mod = sys.modules[text_cls.__module__], sys.modules[audio_cls.__module__]
  # The audio module imports BailingMoeV3Model via a relative import; patch the module it resolves to.
  for mod in (text_mod, sys.modules[audio_mod.BailingMoeV3Model.__module__]):
    mod.FusedRMSNormGated = TorchRMSNormGated  # pylint: disable=possibly-used-before-assignment
    mod.chunk_kda = _torch_kda
    mod.fused_recurrent_kda = _torch_kda
  return text_mod, audio_mod


try:
  TEXT_MOD, AUDIO_MOD = _load_remote_modules(HF_PATH)
  HAS_REF = True
except Exception:  # pylint: disable=broad-exception-caught
  HAS_REF = False

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(not HAS_REF, reason="Ming HF remote code / torch / whisper not available"),
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


class MingAudioTest(unittest.TestCase):
  """Whisper encoder + projector and audio preprocessing vs HF."""

  def create_tempdir_path(self):
    tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(tmp.cleanup)
    return tmp.name

  @classmethod
  def setUpClass(cls):
    torch.set_grad_enabled(False)
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


if __name__ == "__main__":
  unittest.main()
