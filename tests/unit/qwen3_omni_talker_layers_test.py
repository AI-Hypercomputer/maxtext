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

"""Parity tests for the Qwen3-Omni talker stack (talker MoE decoder, code predictor, code2wav) vs. HF PyTorch.

All tests use tiny layer sizes with weights copied from the PyTorch reference and compare outputs with
``assert_allclose``.
"""

import hashlib
import json
import os
import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import qwen3_omni_talker, qwen3_omni_code2wav
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

jax.config.update("jax_enable_x64", False)
torch.manual_seed(0)

# Tiny talker shapes shared by all tests.
EMB = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
MROPE_SECTION = [4, 2, 2]  # sums to HEAD_DIM // 2
N_LAYERS = 2
N_EXPERTS = 8
TOP_K = 2
MOE_DIM = 32
SHARED_EXPERTS = 2  # shared expert width = 2 * MOE_DIM = 64
VOCAB = 48
THINKER_HIDDEN = 40
PROJ_DIM = 36
NUM_CODE_GROUPS = 4
CP_LAYERS = 2
CP_VOCAB = 24
CP_MLP = 40
C2W_HIDDEN = 32
C2W_LAYERS = 2
C2W_HEADS = 4
C2W_MLP = 40
C2W_WINDOW = 3
C2W_CODEBOOK = 12
C2W_NQ = NUM_CODE_GROUPS
C2W_UPSAMPLE_RATES = [2, 3]
C2W_UPSAMPLING_RATIOS = [2]
C2W_DECODER_DIM = 32
SEQ = 8
MAX_PREFILL = 24  # talker prefill budget = thinker max_prefill_predict_length + 16
MAX_TARGET = 32


def make_config(**overrides):
  """Thinker config (`qwen3-omni-30b-a3b`) with tiny speech-stack sizes; HF-named `talker_*` etc. fields."""
  base = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
  kwargs = {
      "run_name": "qwen3_omni_talker_test",
      "model_name": "qwen3-omni-30b-a3b",
      "override_model_config": True,
      "skip_jax_distributed_system": True,
      "enable_checkpointing": False,
      "enable_audio_output": True,
      "mrope_section": MROPE_SECTION,
      "talker_hidden_size": EMB,
      "talker_intermediate_size": PROJ_DIM,
      "talker_num_hidden_layers": N_LAYERS,
      "talker_num_attention_heads": HEADS,
      "talker_num_key_value_heads": KV_HEADS,
      "talker_head_dim": HEAD_DIM,
      "talker_num_experts": N_EXPERTS,
      "talker_num_experts_per_tok": TOP_K,
      "talker_moe_intermediate_size": MOE_DIM,
      "talker_shared_expert_intermediate_size": SHARED_EXPERTS * MOE_DIM,
      "talker_vocab_size": VOCAB,
      "thinker_hidden_size": THINKER_HIDDEN,
      "num_code_groups": NUM_CODE_GROUPS,
      "code_predictor_hidden_size": EMB,
      "code_predictor_num_hidden_layers": CP_LAYERS,
      "code_predictor_num_attention_heads": HEADS,
      "code_predictor_num_key_value_heads": KV_HEADS,
      "code_predictor_head_dim": HEAD_DIM,
      "code_predictor_intermediate_size": CP_MLP,
      "code_predictor_vocab_size": CP_VOCAB,
      "code2wav_hidden_size": C2W_HIDDEN,
      "code2wav_num_hidden_layers": C2W_LAYERS,
      "code2wav_num_attention_heads": C2W_HEADS,
      "code2wav_num_key_value_heads": C2W_HEADS,
      "code2wav_intermediate_size": C2W_MLP,
      "code2wav_sliding_window": C2W_WINDOW,
      "code2wav_codebook_size": C2W_CODEBOOK,
      "code2wav_num_quantizers": C2W_NQ,
      "code2wav_upsample_rates": C2W_UPSAMPLE_RATES,
      "code2wav_upsampling_ratios": C2W_UPSAMPLING_RATIOS,
      "code2wav_decoder_dim": C2W_DECODER_DIM,
      "talker_max_new_tokens": MAX_TARGET - MAX_PREFILL,
      "per_device_batch_size": 1,
      "max_prefill_predict_length": MAX_PREFILL - 16,
      "max_target_length": MAX_TARGET,
      "attention": "dot_product",
      "matmul_precision": "highest",
      "dtype": "float32",
      "weight_dtype": "float32",
      "float32_logits": True,
      "float32_qk_product": True,
      "megablox": False,
      "sparse_matmul": True,
      "scan_layers": False,
      "dropout_rate": 0.0,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], base], **kwargs)


def torch_talker_text_config():
  return Qwen3OmniMoeTalkerTextConfig(
      hidden_size=EMB,
      intermediate_size=PROJ_DIM,
      num_hidden_layers=N_LAYERS,
      num_attention_heads=HEADS,
      num_key_value_heads=KV_HEADS,
      head_dim=HEAD_DIM,
      num_experts=N_EXPERTS,
      num_experts_per_tok=TOP_K,
      moe_intermediate_size=MOE_DIM,
      shared_expert_intermediate_size=SHARED_EXPERTS * MOE_DIM,
      norm_topk_prob=True,
      vocab_size=VOCAB,
      rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0, "mrope_section": MROPE_SECTION},
  )


def torch_code_predictor_config():
  return Qwen3OmniMoeTalkerCodePredictorConfig(
      hidden_size=EMB,
      intermediate_size=CP_MLP,
      num_hidden_layers=CP_LAYERS,
      num_attention_heads=HEADS,
      num_key_value_heads=KV_HEADS,
      head_dim=HEAD_DIM,
      vocab_size=CP_VOCAB,
      num_code_groups=NUM_CODE_GROUPS,
      rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
  )


def torch_talker_config():
  return Qwen3OmniMoeTalkerConfig(
      text_config=torch_talker_text_config().to_dict(),
      code_predictor_config=torch_code_predictor_config().to_dict(),
      num_code_groups=NUM_CODE_GROUPS,
      thinker_hidden_size=THINKER_HIDDEN,
      spatial_merge_size=2,
  )


def torch_code2wav_config():
  return Qwen3OmniMoeCode2WavConfig(
      codebook_size=C2W_CODEBOOK,
      hidden_size=C2W_HIDDEN,
      intermediate_size=C2W_MLP,
      num_hidden_layers=C2W_LAYERS,
      num_attention_heads=C2W_HEADS,
      num_key_value_heads=C2W_HEADS,
      sliding_window=C2W_WINDOW,
      num_quantizers=C2W_NQ,
      upsample_rates=C2W_UPSAMPLE_RATES,
      upsampling_ratios=C2W_UPSAMPLING_RATIOS,
      decoder_dim=C2W_DECODER_DIM,
      rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
  )


def to_np(t):
  return t.detach().cpu().float().numpy()


def set_param(param, value):
  param.value = jnp.asarray(value, dtype=param.value.dtype)


# ----------------------------------------------------------------------------- weight copying


def copy_linear(mt_dense, pt_linear, target_shape=None):
  w = to_np(pt_linear.weight).T
  if target_shape is not None:
    w = w.reshape(target_shape)
  set_param(mt_dense.kernel, w)
  if getattr(pt_linear, "bias", None) is not None and hasattr(mt_dense, "bias") and mt_dense.bias is not None:
    set_param(mt_dense.bias, to_np(pt_linear.bias))


def copy_qwen3_attention(mt_layer, pt_attn):
  """Copies HF Qwen3-style attention (+ qk norm) into a MaxText ``AttentionWithNorm`` layer."""
  attn = mt_layer.self_attention
  heads, kv_heads, head_dim = attn.num_query_heads, attn.num_kv_heads, attn.head_dim
  emb = pt_attn.q_proj.weight.shape[1]
  copy_linear(attn.query, pt_attn.q_proj, (emb, heads, head_dim))
  copy_linear(attn.key, pt_attn.k_proj, (emb, kv_heads, head_dim))
  copy_linear(attn.value, pt_attn.v_proj, (emb, kv_heads, head_dim))
  set_param(attn.out.kernel, to_np(pt_attn.o_proj.weight).T.reshape(heads, head_dim, emb))
  set_param(attn.query_norm.scale, to_np(pt_attn.q_norm.weight))
  set_param(attn.key_norm.scale, to_np(pt_attn.k_norm.weight))


def copy_talker_layer(mt_layer, pt_layer):
  """Copies HF talker attention, normalization, and expert weights into MaxText."""
  set_param(mt_layer.pre_self_attention_layer_norm.scale, to_np(pt_layer.input_layernorm.weight))
  set_param(mt_layer.post_self_attention_layer_norm.scale, to_np(pt_layer.post_attention_layernorm.weight))
  copy_qwen3_attention(mt_layer, pt_layer.self_attn)
  moe = pt_layer.mlp
  set_param(mt_layer.moe_block.gate.kernel, to_np(moe.gate.weight).T)
  gate_up = to_np(moe.experts.gate_up_proj)  # (E, 2I, H)
  down = to_np(moe.experts.down_proj)  # (E, H, I)
  inter = gate_up.shape[1] // 2
  set_param(mt_layer.moe_block.wi_0, np.transpose(gate_up[:, :inter, :], (0, 2, 1)))
  set_param(mt_layer.moe_block.wi_1, np.transpose(gate_up[:, inter:, :], (0, 2, 1)))
  set_param(mt_layer.moe_block.wo, np.transpose(down, (0, 2, 1)))
  copy_linear(mt_layer.shared_expert.wi_0, moe.shared_expert.gate_proj)
  copy_linear(mt_layer.shared_expert.wi_1, moe.shared_expert.up_proj)
  copy_linear(mt_layer.shared_expert.wo, moe.shared_expert.down_proj)
  copy_linear(mt_layer.shared_expert_gate, moe.shared_expert_gate)


def copy_resize_mlp(mt_mlp, pt_mlp):
  copy_linear(mt_mlp.linear_fc1, pt_mlp.linear_fc1)
  copy_linear(mt_mlp.linear_fc2, pt_mlp.linear_fc2)


def copy_talker(mt_talker, pt_talker):
  set_param(mt_talker.codec_embedding.embedding, to_np(pt_talker.model.codec_embedding.weight))
  copy_resize_mlp(mt_talker.text_projection, pt_talker.text_projection)
  copy_resize_mlp(mt_talker.hidden_projection, pt_talker.hidden_projection)
  for i, pt_layer in enumerate(pt_talker.model.layers):
    copy_talker_layer(getattr(mt_talker, f"layers_{i}"), pt_layer)
  set_param(mt_talker.decoder_norm.scale, to_np(pt_talker.model.norm.weight))
  copy_linear(mt_talker.codec_head, pt_talker.codec_head)


def copy_code_predictor(mt_cp, pt_cp):
  """Copies HF code predictor embeddings, decoder layers, and heads into MaxText."""
  for g, emb in enumerate(pt_cp.model.codec_embedding):
    set_param(getattr(mt_cp, f"codec_embedding_{g}").embedding, to_np(emb.weight))
  for i, pt_layer in enumerate(pt_cp.model.layers):
    mt_layer = getattr(mt_cp, f"layers_{i}")
    set_param(mt_layer.pre_self_attention_layer_norm.scale, to_np(pt_layer.input_layernorm.weight))
    set_param(mt_layer.post_self_attention_layer_norm.scale, to_np(pt_layer.post_attention_layernorm.weight))
    copy_qwen3_attention(mt_layer, pt_layer.self_attn)
    copy_linear(mt_layer.mlp.wi_0, pt_layer.mlp.gate_proj)
    copy_linear(mt_layer.mlp.wi_1, pt_layer.mlp.up_proj)
    copy_linear(mt_layer.mlp.wo, pt_layer.mlp.down_proj)
  set_param(mt_cp.decoder_norm.scale, to_np(pt_cp.model.norm.weight))
  for g, head in enumerate(pt_cp.lm_head):
    copy_linear(getattr(mt_cp, f"lm_head_{g}"), head)


def copy_code2wav(mt_c2w, pt_c2w):
  """Copies the HF vocoder state dict leaf-by-leaf; MaxText keeps the PyTorch layouts so names map 1:1."""
  mt_params = dict(nnx.to_flat_state(nnx.state(mt_c2w, nnx.Param)))
  mt_by_name = {}
  for path, var in mt_params.items():
    parts = [str(p) for p in path if str(p) != "value"]
    mt_by_name[".".join(parts)] = var
  for hf_name, tensor in pt_c2w.state_dict().items():
    name = hf_name
    name = name.replace("pre_transformer.layers.", "pre_transformer.layers_")
    name = name.replace("_layer_scale.scale", "_layer_scale")
    name = name.replace("input_layernorm.weight", "input_layernorm").replace(
        "post_attention_layernorm.weight", "post_attention_layernorm"
    )
    name = name.replace("pre_transformer.norm.weight", "pre_transformer.norm")
    name = name.replace("code_embedding.weight", "code_embedding")
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
      name = name.replace(f"{proj}.weight", proj)
    # upsample.i.j -> upsample_i_j ; decoder.i -> decoder_i ; block.i -> block_i
    import re  # pylint: disable=import-outside-toplevel

    name = re.sub(r"upsample\.(\d+)\.(\d+)", r"upsample_\1_\2", name)
    name = re.sub(r"decoder\.(\d+)", r"decoder_\1", name)
    name = re.sub(r"block\.(\d+)", r"block_\1", name)
    assert name in mt_by_name, f"{hf_name} -> {name} not found in MaxText code2wav params"
    set_param(mt_by_name[name], to_np(tensor))


# ----------------------------------------------------------------------------- helpers


def causal_mask(batch, seq):
  mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)
  return mask[None, None].expand(batch, 1, seq, seq)


class TalkerTestBase(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.config = make_config()
    devices_array = maxtext_utils.create_device_mesh(cls.config)
    cls.mesh = Mesh(devices_array, cls.config.mesh_axes)


class TestTalkerDecoder(TalkerTestBase):
  """Talker MoE decoder (prefill logits + normed hidden) vs HF ``Qwen3OmniMoeTalkerForConditionalGeneration``."""

  def setUp(self):
    torch.manual_seed(1)
    self.pt = Qwen3OmniMoeTalkerForConditionalGeneration(torch_talker_config()).eval()
    for p in self.pt.parameters():
      torch.nn.init.normal_(p, std=0.2)
    with self.mesh:
      self.mt = qwen3_omni_talker.Qwen3OmniTalker(
          self.config, self.mesh, None, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
      )
    copy_talker(self.mt, self.pt)

  def test_projections_match(self):
    x = torch.randn(2, 5, THINKER_HIDDEN)
    pt_text = to_np(self.pt.text_projection(x))
    pt_hidden = to_np(self.pt.hidden_projection(x))
    mt_text = np.asarray(self.mt.text_projection(jnp.asarray(x.numpy())))
    mt_hidden = np.asarray(self.mt.hidden_projection(jnp.asarray(x.numpy())))
    np.testing.assert_allclose(mt_text, pt_text, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mt_hidden, pt_hidden, rtol=1e-5, atol=1e-5)

  def test_prefill_logits_match(self):
    batch = 1
    embeds = torch.randn(batch, SEQ, EMB)
    # 3-D MRoPE positions with distinct temporal/height/width values.
    pos = torch.stack(
        [
            torch.arange(SEQ),
            torch.arange(SEQ) // 2,
            torch.arange(SEQ) % 3,
        ]
    )[
        :, None, :
    ].expand(3, batch, SEQ)
    with torch.no_grad():
      out = self.pt.model(inputs_embeds=embeds, position_ids=pos, attention_mask=torch.ones(batch, SEQ, dtype=torch.long))
      pt_hidden = to_np(out.last_hidden_state)
      pt_logits = to_np(self.pt.codec_head(out.last_hidden_state))

    positions = jnp.asarray(pos.permute(1, 2, 0).numpy(), dtype=jnp.int32)  # (batch, seq, 3)
    segment_ids = jnp.ones((batch, SEQ), dtype=jnp.int32)
    with self.mesh:
      mt_logits, mt_hidden = self.mt(jnp.asarray(embeds.numpy()), positions, segment_ids, model_mode=MODEL_MODE_TRAIN)
    np.testing.assert_allclose(np.asarray(mt_hidden), pt_hidden, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(mt_logits), pt_logits, rtol=2e-3, atol=2e-3)

  def test_prefill_then_autoregressive_matches_full_forward(self):
    """KV-cached decode (PREFILL then AUTOREGRESSIVE on one AR-mode module, as the pipeline does) equals a full forward."""
    config = make_config(per_device_batch_size=1.0 / jax.device_count())
    with self.mesh:
      mt_ar = qwen3_omni_talker.Qwen3OmniTalker(
          config, self.mesh, None, model_mode=MODEL_MODE_AUTOREGRESSIVE, rngs=nnx.Rngs(0)
      )
    copy_talker(mt_ar, self.pt)
    batch = mt_ar.config.micro_batch_size_to_train_on
    total = MAX_PREFILL + 3
    embeds = jax.random.normal(jax.random.PRNGKey(3), (batch, total, EMB), dtype=jnp.float32)
    positions_1d = jnp.broadcast_to(jnp.arange(total, dtype=jnp.int32)[None, :], (batch, total))
    positions_3d = jnp.broadcast_to(positions_1d[..., None], (batch, total, 3))

    with self.mesh:
      full_logits, _ = self.mt(embeds, positions_3d, jnp.ones((batch, total), jnp.int32), model_mode=MODEL_MODE_TRAIN)
      prefill_logits, _ = mt_ar(
          embeds[:, :MAX_PREFILL],
          positions_3d[:, :MAX_PREFILL],
          jnp.ones((batch, MAX_PREFILL), jnp.int32),
          model_mode=MODEL_MODE_PREFILL,
      )
      ar_logits = []
      for t in range(MAX_PREFILL, total):
        logits_t, _ = mt_ar(embeds[:, t : t + 1], positions_1d[:, t : t + 1], None, model_mode=MODEL_MODE_AUTOREGRESSIVE)
        ar_logits.append(np.asarray(logits_t[:, 0]))
    np.testing.assert_allclose(np.asarray(prefill_logits), np.asarray(full_logits[:, :MAX_PREFILL]), rtol=2e-3, atol=2e-3)
    for i, t in enumerate(range(MAX_PREFILL, total)):
      np.testing.assert_allclose(ar_logits[i], np.asarray(full_logits[:, t]), rtol=2e-3, atol=2e-3)


class TestCodePredictor(TalkerTestBase):
  """Code predictor hidden states + per-group heads vs HF."""

  def setUp(self):
    torch.manual_seed(2)
    self.pt = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(torch_code_predictor_config()).eval()
    for p in self.pt.parameters():
      torch.nn.init.normal_(p, std=0.2)
    with self.mesh:
      self.mt = qwen3_omni_talker.Qwen3OmniCodePredictor(self.config, self.mesh, None, rngs=nnx.Rngs(0))
    copy_code_predictor(self.mt, self.pt)  # standalone module; inside the model it is `talker.code_predictor`

  def test_hidden_and_heads_match(self):
    batch = 2
    embeds = torch.randn(batch, NUM_CODE_GROUPS, EMB)
    with torch.no_grad():
      pt_hidden = to_np(self.pt.model(inputs_embeds=embeds).last_hidden_state)
      pt_heads = [to_np(head(torch.from_numpy(pt_hidden))) for head in self.pt.lm_head]
    with self.mesh:
      mt_hidden = self.mt(jnp.asarray(embeds.numpy()))
      mt_heads = [np.asarray(self.mt.head(g, mt_hidden)) for g in range(NUM_CODE_GROUPS - 1)]
    np.testing.assert_allclose(np.asarray(mt_hidden), pt_hidden, rtol=2e-3, atol=2e-3)
    for g in range(NUM_CODE_GROUPS - 1):
      np.testing.assert_allclose(mt_heads[g], pt_heads[g], rtol=2e-3, atol=2e-3)

  def test_embeddings_match(self):
    codes = torch.randint(0, CP_VOCAB, (2, 1))
    for g in range(NUM_CODE_GROUPS - 1):
      pt = to_np(self.pt.model.codec_embedding[g](codes))
      mt = np.asarray(self.mt.embed_residual(g, jnp.asarray(codes.numpy())))
      np.testing.assert_allclose(mt, pt, rtol=1e-6, atol=1e-6)


class TestCode2Wav(TalkerTestBase):
  """Code2Wav vocoder waveform vs HF ``Qwen3OmniMoeCode2Wav``."""

  def setUp(self):
    torch.manual_seed(3)
    self.pt = Qwen3OmniMoeCode2Wav(torch_code2wav_config()).eval()
    for name, p in self.pt.named_parameters():
      if name.endswith("alpha") or name.endswith("beta"):
        torch.nn.init.normal_(p, std=0.3)
      elif p.ndim > 1:
        torch.nn.init.normal_(p, std=0.1)
    with self.mesh:
      self.mt = qwen3_omni_code2wav.Qwen3OmniCode2Wav(self.config, self.mesh, rngs=nnx.Rngs(0))
    copy_code2wav(self.mt, self.pt)

  def test_waveform_matches(self):
    codes = torch.randint(0, C2W_CODEBOOK, (1, C2W_NQ, 11))
    with torch.no_grad():
      pt_wav = to_np(self.pt(codes))
    mt_wav = np.asarray(self.mt(jnp.asarray(codes.numpy())))
    self.assertEqual(mt_wav.shape, pt_wav.shape)
    np.testing.assert_allclose(mt_wav, pt_wav, rtol=1e-3, atol=1e-3)

  def test_chunked_decode_matches(self):
    codes = torch.randint(0, C2W_CODEBOOK, (1, C2W_NQ, 13))
    with torch.no_grad():
      pt_wav = to_np(self.pt.chunked_decode(codes, chunk_size=5, left_context_size=2))
    mt_wav = np.asarray(self.mt.chunked_decode(jnp.asarray(codes.numpy()), chunk_size=5, left_context_size=2))
    self.assertEqual(mt_wav.shape, pt_wav.shape)
    np.testing.assert_allclose(mt_wav, pt_wav, rtol=1e-3, atol=1e-3)


class TestTransformerSpeechStack(TalkerTestBase):
  """`models.Transformer` exposes the speech stack under the HF sub-module names when enable_audio_output is set."""

  def test_param_layout(self):
    from maxtext.models import models  # pylint: disable=import-outside-toplevel

    def build(config):
      with self.mesh:
        return models.Transformer(config, self.mesh, None, model_mode=MODEL_MODE_AUTOREGRESSIVE, rngs=nnx.Rngs(0))

    tiny_thinker = {
        "accept_hidden_layer": 1,
        "base_emb_dim": EMB,
        "base_num_query_heads": HEADS,
        "base_num_kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "base_num_decoder_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "base_mlp_dim": MOE_DIM,
        "base_moe_mlp_dim": MOE_DIM,
        "vocab_size": 256,
        "use_multimodal": False,
        "use_audio": False,
    }
    model = nnx.eval_shape(lambda: build(make_config(**tiny_thinker)))
    paths = {"/".join(str(p) for p in path) for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    # Recorded before the refactor: every checkpoint leaf must retain its path, shape, and dtype.
    manifest = {
        "/".join(map(str, path)): {"shape": list(value.shape), "dtype": str(value.dtype)}
        for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    self.assertEqual(len(manifest), 184)
    self.assertEqual(
        hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "880627c093ac0daad64acb0bc9faa507133dc1ba20b0baa3c0e79fd4e69886e0",
    )
    self.assertTrue(any(p.startswith("talker/layers_0/moe_block/") for p in paths))
    self.assertTrue(any(p.startswith("talker/code_predictor/lm_head_0/") for p in paths))
    self.assertTrue(any(p.startswith("code2wav/pre_transformer/layers_0/") for p in paths))
    self.assertTrue(any(p.startswith("talker/codec_head/kernel") for p in paths))
    # Thinker-only models carry no speech-stack params, so existing checkpoints keep loading.
    text_model = nnx.eval_shape(lambda: build(make_config(enable_audio_output=False, **tiny_thinker)))
    text_paths = {"/".join(str(p) for p in path) for path, _ in nnx.to_flat_state(nnx.state(text_model, nnx.Param))}
    self.assertFalse(any(p.startswith(("talker/", "code2wav/")) for p in text_paths))


if __name__ == "__main__":
  unittest.main()
