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

"""Qwen3-Omni audio output: talker, residual code predictor, and Code2Wav waveform decoder.

This module implements the *audio output* half of Qwen3-Omni (HF `Qwen3OmniMoeForConditionalGeneration`):

* ``Qwen3OmniTalker``: a 20-layer MoE decoder (Qwen3-MoE style attention with a sigmoid-gated shared expert)
  that consumes projected thinker embeddings/hidden states and predicts the first codec code group.
* ``Qwen3OmniCodePredictor``: a small dense decoder that autoregressively predicts the remaining
  ``num_code_groups - 1`` residual codec codes of a frame.
* ``Qwen3OmniCode2Wav``: a convolutional vocoder that turns codec frames into a 24 kHz waveform.

``models.Transformer`` instantiates ``talker`` (which owns ``code_predictor``) and ``code2wav`` when
``config.enable_audio_output`` is set, mirroring the HF module layout (``talker.*``, ``talker.code_predictor.*``,
``code2wav.*``) so one MaxText checkpoint holds thinker and speech stack. Architecture fields come from the
nested ``audio_output_config`` model settings; the talker derives a MaxText
sub-config from the thinker config so it can reuse ``AttentionWithNorm``, ``RoutedMoE``, ``MlpBlock`` and
``Qwen3DecoderLayer`` (KV caching / sharding follow the rest of MaxText). The code predictor sees at most
``num_code_groups`` positions per frame, so it runs cache-free in ``MODEL_MODE_TRAIN`` over a fixed-size window.
"""
# pylint: disable=arguments-differ
# pylint: disable=attribute-defined-outside-init

import math

from flax import nnx
from pydantic import BaseModel, ConfigDict, Field
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Array, Config, DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import initializers as max_initializers
from maxtext.layers.embeddings import Embed
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.moe import RoutedMoE
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import AttentionWithNorm, Qwen3DecoderLayer


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
class Qwen3OmniTalkerConfig(BaseModel):
  """Validated Qwen3-Omni talker configuration."""

  model_config = ConfigDict(extra="forbid")

  hidden_size: int = Field(description="Talker hidden size.")
  intermediate_size: int = Field(description="Talker dense MLP size (also the thinker->talker projection MLP).")
  num_hidden_layers: int = Field(description="Talker decoder layers.")
  num_attention_heads: int = Field(description="Talker query heads.")
  num_key_value_heads: int = Field(description="Talker KV heads.")
  head_dim: int = Field(description="Talker head dim.")
  num_experts: int = Field(description="Talker routed experts.")
  num_experts_per_tok: int = Field(description="Talker experts per token.")
  moe_intermediate_size: int = Field(description="Talker routed-expert MLP size.")
  shared_expert_intermediate_size: int = Field(description="Talker shared-expert MLP size.")
  norm_topk_prob: bool = Field(description="Renormalize talker top-k routing weights.")
  vocab_size: int = Field(description="Talker codec vocabulary (codebook 0 + control ids).")
  rope_theta: float = Field(description="Talker RoPE theta (interleaved MRoPE, thinker mrope_section).")
  rms_norm_eps: float = Field(description="Talker RMSNorm epsilon.")
  max_new_tokens: int = Field(description="Maximum codec frames (12.5 per second of audio).")
  do_sample: bool = Field(description="Sample talker codes instead of greedy decoding.")
  top_k: int = Field(description="Top-k for talker sampling.")
  top_p: float = Field(description="Top-p for talker sampling.")
  temperature: float = Field(description="Temperature for talker sampling.")
  repetition_penalty: float = Field(description="Repetition penalty on generated talker codes.")


class Qwen3OmniCodePredictorConfig(BaseModel):
  """Validated Qwen3-Omni code_predictor configuration."""

  model_config = ConfigDict(extra="forbid")

  hidden_size: int = Field(description="Code predictor hidden size.")
  intermediate_size: int = Field(description="Code predictor MLP size.")
  num_hidden_layers: int = Field(description="Code predictor layers.")
  num_attention_heads: int = Field(description="Code predictor query heads.")
  num_key_value_heads: int = Field(description="Code predictor KV heads.")
  head_dim: int = Field(description="Code predictor head dim.")
  vocab_size: int = Field(description="Residual codebook size.")
  rope_theta: float = Field(description="Code predictor RoPE theta.")


class Qwen3OmniCode2WavConfig(BaseModel):
  """Validated Qwen3-Omni code2wav configuration."""

  model_config = ConfigDict(extra="forbid")

  dtype: str = Field(description="Compute/param dtype of the code2wav vocoder.")
  codebook_size: int = Field(description="Code2wav codebook size per quantizer.")
  hidden_size: int = Field(description="Code2wav transformer hidden size.")
  intermediate_size: int = Field(description="Code2wav MLP size.")
  num_hidden_layers: int = Field(description="Code2wav transformer layers.")
  num_attention_heads: int = Field(description="Code2wav attention heads.")
  num_key_value_heads: int = Field(description="Code2wav KV heads.")
  sliding_window: int = Field(description="Code2wav sliding attention window.")
  rope_theta: float = Field(description="Code2wav RoPE theta.")
  rms_norm_eps: float = Field(description="Code2wav RMSNorm epsilon.")
  layer_scale_initial_scale: float = Field(description="Code2wav layer-scale init.")
  num_quantizers: int = Field(description="Code2wav number of quantizers.")
  upsample_rates: list[int] = Field(description="Code2wav decoder upsample rates.")
  upsampling_ratios: list[int] = Field(description="Code2wav pre-decoder upsampling ratios.")
  decoder_dim: int = Field(description="Code2wav decoder channel dim.")


class Qwen3OmniAudioConfig(BaseModel):
  """Validated Qwen3-Omni audio output configuration."""

  model_config = ConfigDict(extra="forbid")

  talker: Qwen3OmniTalkerConfig
  code_predictor: Qwen3OmniCodePredictorConfig
  code2wav: Qwen3OmniCode2WavConfig
  num_code_groups: int = Field(
      description="Codec code groups per frame (1 from the talker, rest from the code predictor)."
  )
  thinker_hidden_size: int = Field(description="Thinker hidden size fed to the talker projections.")
  accept_hidden_layer: int = Field(description="Thinker layer whose output feeds the talker at multimodal positions.")
  codec_pad_id: int = Field(description="Codec pad id.")
  codec_bos_id: int = Field(description="Codec bos id.")
  codec_eos_token_id: int = Field(description="Codec eos id (ends speech generation).")
  codec_nothink_id: int = Field(description="Codec no-think id.")
  codec_think_bos_id: int = Field(description="Codec think-bos id.")
  codec_think_eos_id: int = Field(description="Codec think-eos id.")
  speaker_id: dict[str, int] = Field(description="Speaker name -> codec speaker id.")
  tts_pad_token_id: int = Field(description="TTS pad token id (thinker vocab).")
  tts_bos_token_id: int = Field(description="TTS bos token id (thinker vocab).")
  tts_eos_token_id: int = Field(description="TTS eos token id (thinker vocab).")
  im_start_token_id: int = Field(description="`<|im_start|>` id.")
  user_token_id: int = Field(description="`user` role token id.")
  assistant_token_id: int = Field(description="`assistant` role token id.")
  speaker: str = Field(description="Speaker voice for audio generation.")


def get_audio_config(config: Config) -> Qwen3OmniAudioConfig:
  """Validates backend-owned settings without adding model fields to the global schema."""
  return Qwen3OmniAudioConfig.model_validate(config.audio_output_config)


def derive_config(config: Config, **overrides) -> Config:
  """Returns a copy of a MaxText ``HyperParameters`` config with the given (already-derived) fields overridden.

  Pydantic validators are not re-run, so callers must override both the ``base_*`` and derived fields when needed
  (e.g. ``base_emb_dim`` and ``emb_dim``).
  """
  pydantic_config = getattr(config, "_pydantic_config", None)
  if pydantic_config is None:
    raise ValueError("derive_config expects a pyconfig.HyperParameters config.")
  return pyconfig.HyperParameters(pydantic_config.model_copy(update=overrides))


def talker_config(config: Config) -> Config:
  """Derives the MaxText config of the talker MoE decoder from the thinker config.

  The talker decodes one speech stream alongside the thinker: its prefill covers the user turn plus a 9-token
  assistant header (<= thinker prefill + 16). Its autoregressive budget is
  ``audio_output_config.talker.max_new_tokens`` frames.
  """
  audio = get_audio_config(config)
  max_prefill = config.max_prefill_predict_length + 16
  shared_experts = audio.talker.shared_expert_intermediate_size // audio.talker.moe_intermediate_size
  if shared_experts * audio.talker.moe_intermediate_size != audio.talker.shared_expert_intermediate_size:
    raise ValueError("talker_shared_expert_intermediate_size must be a multiple of talker_moe_intermediate_size.")
  return derive_config(
      config,
      decoder_block=DecoderBlockType.QWEN3_MOE,
      base_emb_dim=audio.talker.hidden_size,
      emb_dim=audio.talker.hidden_size,
      base_num_query_heads=audio.talker.num_attention_heads,
      num_query_heads=audio.talker.num_attention_heads,
      base_num_kv_heads=audio.talker.num_key_value_heads,
      num_kv_heads=audio.talker.num_key_value_heads,
      head_dim=audio.talker.head_dim,
      base_num_decoder_layers=audio.talker.num_hidden_layers,
      num_decoder_layers=audio.talker.num_hidden_layers,
      base_mlp_dim=audio.talker.moe_intermediate_size,
      mlp_dim=audio.talker.moe_intermediate_size,
      base_moe_mlp_dim=audio.talker.moe_intermediate_size,
      moe_mlp_dim=audio.talker.moe_intermediate_size,
      num_experts=audio.talker.num_experts,
      num_experts_per_tok=audio.talker.num_experts_per_tok,
      shared_experts=shared_experts,
      norm_topk_prob=audio.talker.norm_topk_prob,
      vocab_size=audio.talker.vocab_size,
      normalization_layer_epsilon=audio.talker.rms_norm_eps,
      rope_max_timescale=int(audio.talker.rope_theta),
      use_qk_norm=True,
      use_mrope=True,
      max_prefill_predict_length=max_prefill,
      max_target_length=max_prefill + audio.talker.max_new_tokens,
      # Single speech stream; the KV cache batch follows the thinker's decode batch (see talker pipeline).
      use_multimodal=False,
      use_audio=False,
      mtp_num_layers=0,
      scan_layers=False,
  )


def code_predictor_config(config: Config) -> Config:
  """Derives the config for the dense code-predictor decoder layers from the thinker config."""
  audio = get_audio_config(config)
  num_positions = audio.num_code_groups
  return derive_config(
      config,
      decoder_block=DecoderBlockType.QWEN3,
      base_emb_dim=audio.code_predictor.hidden_size,
      emb_dim=audio.code_predictor.hidden_size,
      base_num_query_heads=audio.code_predictor.num_attention_heads,
      num_query_heads=audio.code_predictor.num_attention_heads,
      base_num_kv_heads=audio.code_predictor.num_key_value_heads,
      num_kv_heads=audio.code_predictor.num_key_value_heads,
      head_dim=audio.code_predictor.head_dim,
      base_mlp_dim=audio.code_predictor.intermediate_size,
      mlp_dim=audio.code_predictor.intermediate_size,
      base_num_decoder_layers=audio.code_predictor.num_hidden_layers,
      num_decoder_layers=audio.code_predictor.num_hidden_layers,
      num_experts=1,
      num_experts_per_tok=1,
      shared_experts=0,
      vocab_size=audio.code_predictor.vocab_size,
      normalization_layer_epsilon=audio.talker.rms_norm_eps,
      rope_max_timescale=int(audio.code_predictor.rope_theta),
      use_qk_norm=True,
      # 1-D RoPE: MRoPE with identical positions on all three axes is exactly HF rotate-half RoPE.
      use_mrope=True,
      use_multimodal=False,
      use_audio=False,
      mtp_num_layers=0,
      scan_layers=False,
      # The code predictor runs cache-free over a fixed window of `num_code_groups` positions.
      max_target_length=num_positions,
      max_prefill_predict_length=num_positions // 2,
  )


# -----------------------------------------------------------------------------
# Talker
# -----------------------------------------------------------------------------
class Qwen3OmniTalkerResizeMLP(nnx.Module):
  """HF ``Qwen3OmniMoeTalkerResizeMLP``: Linear(+bias) -> SiLU -> Linear(+bias)."""

  def __init__(self, config: Config, in_features: int, intermediate_dim: int, out_features: int, *, rngs: nnx.Rngs):
    common = {
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "use_bias": True,
        "matmul_precision": config.matmul_precision,
        "kernel_init": max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        "rngs": rngs,
    }
    self.linear_fc1 = DenseGeneral(in_features, intermediate_dim, kernel_axes=("embed", "mlp"), **common)
    self.linear_fc2 = DenseGeneral(intermediate_dim, out_features, kernel_axes=("mlp", "embed"), **common)

  def __call__(self, x: Array) -> Array:
    return self.linear_fc2(jax.nn.silu(self.linear_fc1(x)))


class Qwen3OmniTalkerDecoderLayer(AttentionWithNorm):
  """Talker decoder layer: Qwen3 attention + routed MoE + sigmoid-gated shared expert.

  Mirrors HF ``Qwen3OmniMoeTalkerDecoderLayer`` / ``Qwen3OmniMoeTalkerTextSparseMoeBlock``:
  ``out = routed_experts(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)``.
  The shared expert width is ``shared_experts * moe_mlp_dim`` (768 = 2 * 384 for Qwen3-Omni-30B-A3B).
  """

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant, rngs: nnx.Rngs):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.moe_block = RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )
    self.shared_expert = MlpBlock(
        config=config,
        mesh=mesh,
        in_features=config.emb_dim,
        intermediate_dim=config.shared_experts * config.moe_mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.shared_expert_gate = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=1,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: None | Array,
      decoder_positions: None | Array,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | Array = None,
      attention_metadata=None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    routed_output, _, _ = self.moe_block(hidden_states, out_sharding=self.out_sharding)
    shared_output = self.shared_expert(
        hidden_states,
        deterministic=deterministic,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )
    shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_states).astype(jnp.float32)).astype(shared_output.dtype)
    mlp_lnx = routed_output + shared_gate * shared_output
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)
    layer_output = intermediate_inputs + mlp_lnx
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)
    return layer_output, kv_cache


class Qwen3OmniTalker(nnx.Module):
  """Talker (HF ``Qwen3OmniMoeTalkerForConditionalGeneration``): MoE decoder, projections, codec head, code predictor.

  ``__call__`` takes *input embeddings* (not token ids) because the talker is driven by projected thinker
  states and summed codec embeddings; use ``codec_embedding`` to embed codec ids. Built from the *thinker*
  config (see ``talker_config``).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant,
      *,
      model_mode: str,
      rngs: nnx.Rngs,
  ):
    audio = get_audio_config(config)
    self.thinker_config = config
    cfg = talker_config(config)
    self.config = cfg
    self.mesh = mesh
    self.codec_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        config=cfg,
        mesh=mesh,
        dtype=cfg.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        rngs=rngs,
    )
    self.text_projection = Qwen3OmniTalkerResizeMLP(
        cfg,
        audio.thinker_hidden_size,
        audio.talker.intermediate_size,
        cfg.emb_dim,
        rngs=rngs,
    )
    self.hidden_projection = Qwen3OmniTalkerResizeMLP(
        cfg,
        audio.thinker_hidden_size,
        audio.talker.intermediate_size,
        cfg.emb_dim,
        rngs=rngs,
    )
    self.num_layers = cfg.num_decoder_layers
    for i in range(self.num_layers):
      setattr(
          self,
          f"layers_{i}",
          Qwen3OmniTalkerDecoderLayer(cfg, mesh, model_mode, quant, rngs),
      )
    self.decoder_norm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.codec_head = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=cfg.vocab_size,
        use_bias=False,
        dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "vocab"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.code_predictor = Qwen3OmniCodePredictor(config, mesh, quant, rngs=rngs)

  def layers(self):
    return [getattr(self, f"layers_{i}") for i in range(self.num_layers)]

  def __call__(
      self,
      inputs_embeds: Array,
      decoder_positions: Array,
      decoder_segment_ids: None | Array = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = True,
  ) -> tuple[Array, Array]:
    """Runs the talker decoder.

    Args:
      inputs_embeds: ``[batch, seq, emb_dim]`` talker input embeddings.
      decoder_positions: ``[batch, seq, 3]`` MRoPE positions (or ``[batch, seq]``).
      decoder_segment_ids: ``[batch, seq]`` segment ids (prefill/train) or ``None`` (autoregressive).
      model_mode: MaxText model mode controlling the KV cache behaviour.
      deterministic: Disables dropout.

    Returns:
      ``(logits [batch, seq, vocab], normed_hidden [batch, seq, emb_dim])``. The normed hidden state is
      what the code predictor consumes (HF ``hidden_states[-1]`` == ``last_hidden_state``).
    """
    y = inputs_embeds.astype(self.config.dtype)
    for layer in self.layers():
      y, _ = layer(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
    hidden = self.decoder_norm(y)
    logits = self.codec_head(hidden.astype(self.codec_head.dtype))
    return logits.astype(jnp.float32), hidden


# -----------------------------------------------------------------------------
# Code predictor
# -----------------------------------------------------------------------------
class Qwen3OmniCodePredictor(nnx.Module):
  """Residual codec-code predictor (HF ``Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration``).

  Frame layout over the ``num_code_groups`` positions it sees:
  position 0 = talker normed hidden, position 1 = ``talker.codec_embedding(code_0)``,
  position ``i + 2`` = ``codec_embedding_i(code_{i+1})``. Logits for ``code_{i+1}`` come from ``lm_head_i``
  applied at position ``i + 1``.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant, *, rngs: nnx.Rngs):
    audio = get_audio_config(config)
    self.thinker_config = config
    self.mesh = mesh
    cp_cfg = code_predictor_config(config)
    self.config = cp_cfg
    self.num_residual_groups = audio.num_code_groups - 1
    self.num_layers = cp_cfg.num_decoder_layers
    for i in range(self.num_residual_groups):
      setattr(
          self,
          f"codec_embedding_{i}",
          Embed(
              num_embeddings=cp_cfg.vocab_size,
              num_features=cp_cfg.emb_dim,
              config=cp_cfg,
              mesh=mesh,
              dtype=cp_cfg.dtype,
              embedding_init=jax.nn.initializers.normal(stddev=1.0),
              rngs=rngs,
          ),
      )
    for i in range(self.num_layers):
      setattr(
          self,
          f"layers_{i}",
          Qwen3DecoderLayer(cp_cfg, mesh, MODEL_MODE_TRAIN, quant, rngs),
      )
    self.decoder_norm = RMSNorm(
        num_features=cp_cfg.emb_dim,
        dtype=cp_cfg.dtype,
        weight_dtype=cp_cfg.weight_dtype,
        shard_mode=cp_cfg.shard_mode,
        kernel_axes=("norm",),
        epsilon=cp_cfg.normalization_layer_epsilon,
        rngs=rngs,
    )
    for i in range(self.num_residual_groups):
      setattr(
          self,
          f"lm_head_{i}",
          DenseGeneral(
              in_features_shape=cp_cfg.emb_dim,
              out_features_shape=cp_cfg.vocab_size,
              use_bias=False,
              dtype=jnp.float32 if cp_cfg.logits_dot_in_fp32 else cp_cfg.dtype,
              weight_dtype=cp_cfg.weight_dtype,
              kernel_axes=("embed", "vocab"),
              matmul_precision=cp_cfg.matmul_precision,
              rngs=rngs,
          ),
      )

  def embed_residual(self, group: int, codes: Array) -> Array:
    """Embeds residual code group ``group + 1`` (``codes`` holds ``code_{group+1}``)."""
    return getattr(self, f"codec_embedding_{group}")(codes)

  def head(self, group: int, hidden: Array) -> Array:
    lm_head = getattr(self, f"lm_head_{group}")
    return lm_head(hidden.astype(lm_head.dtype)).astype(jnp.float32)

  def __call__(self, inputs_embeds: Array, deterministic: bool = True) -> Array:
    """Runs the dense decoder over the full ``[batch, num_code_groups, emb_dim]`` window; returns normed hidden."""
    batch, seq, _ = inputs_embeds.shape
    positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))
    segment_ids = jnp.ones((batch, seq), dtype=jnp.int32)
    y = inputs_embeds.astype(self.config.dtype)
    for i in range(self.num_layers):
      y, _ = getattr(self, f"layers_{i}")(y, segment_ids, positions, deterministic, MODEL_MODE_TRAIN)
    return self.decoder_norm(y)


# -----------------------------------------------------------------------------
# Code2Wav
# -----------------------------------------------------------------------------
def _param(rngs: nnx.Rngs, shape, dtype, init=None, scale: float = 0.02) -> nnx.Param:
  if init is None:
    value = jax.random.normal(rngs.params(), shape, dtype) * scale
  else:
    value = jnp.full(shape, init, dtype)
  return nnx.Param(value)


class ParamGroup(nnx.Module):
  """A named bag of ``nnx.Param`` leaves (e.g. ``conv.weight`` / ``conv.bias``) mirroring an HF submodule."""

  def __init__(self, **params: nnx.Param):
    for name, value in params.items():
      setattr(self, name, value)

  def __getattr__(self, name: str) -> nnx.Param:
    # Attributes are set dynamically in __init__; this only runs on misses and keeps linters from flagging no-member.
    raise AttributeError(f"{type(self).__name__} has no parameter {name!r}")


class Qwen3OmniCausalConv1d(nnx.Module):
  """HF ``Qwen3OmniMoeCausalConvNet``: left-padded (causal) 1-D convolution. Layout ``[batch, channels, time]``.

  Kernel is stored in PyTorch layout ``[out_channels, in_channels // groups, kernel_size]``.
  """

  def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1, *, dtype, rngs):
    self.stride = stride
    self.dilation = dilation
    self.groups = groups
    self.effective_kernel_size = (kernel_size - 1) * dilation + 1
    self.padding = self.effective_kernel_size - stride
    self.conv = ParamGroup(
        weight=_param(rngs, (out_channels, in_channels // groups, kernel_size), dtype),
        bias=_param(rngs, (out_channels,), dtype, init=0.0),
    )

  def __call__(self, x: Array) -> Array:
    length = x.shape[-1]
    n_frames = (length - self.effective_kernel_size + self.padding) / self.stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.effective_kernel_size - self.padding)
    extra_padding = ideal_length - length
    x = jnp.pad(x, ((0, 0), (0, 0), (self.padding, extra_padding)))
    weight = self.conv.weight.get_value().astype(x.dtype)
    y = lax.conv_general_dilated(
        x,
        weight,
        window_strides=(self.stride,),
        padding=[(0, 0)],
        rhs_dilation=(self.dilation,),
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=self.groups,
        precision=lax.Precision.HIGHEST,
    )
    return y + self.conv.bias.get_value().astype(y.dtype)[None, :, None]


class Qwen3OmniCausalTransConv1d(nnx.Module):
  """HF ``Qwen3OmniMoeCausalTransConvNet``: transposed conv with symmetric trimming of ``kernel - stride``.

  Kernel is stored in PyTorch layout ``[in_channels, out_channels, kernel_size]``.
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, *, dtype, rngs):
    self.stride = stride
    self.kernel_size = kernel_size
    pad = kernel_size - stride
    self.left_pad = math.ceil(pad)
    self.right_pad = self.left_pad
    self.conv = ParamGroup(
        weight=_param(rngs, (in_channels, out_channels, kernel_size), dtype),
        bias=_param(rngs, (out_channels,), dtype, init=0.0),
    )

  def __call__(self, x: Array) -> Array:
    weight = self.conv.weight.get_value().astype(x.dtype)
    # PyTorch ConvTranspose1d(in, out) is the input-gradient of Conv1d(out, in) whose weight is the same
    # `[in, out, k]` tensor read in OIH layout; `transpose_kernel=True` flips/swaps it accordingly.
    y = lax.conv_transpose(
        x,
        weight,
        strides=(self.stride,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        transpose_kernel=True,
        precision=lax.Precision.HIGHEST,
    )
    y = y + self.conv.bias.get_value().astype(y.dtype)[None, :, None]
    return y[..., self.left_pad : y.shape[-1] - self.right_pad]


class Qwen3OmniSnakeBeta(nnx.Module):
  """SnakeBeta activation: ``x + 1/(exp(beta)+eps) * sin(x * exp(alpha))^2`` with per-channel ``alpha``/``beta``."""

  def __init__(self, channels, *, dtype, rngs):
    self.alpha = _param(rngs, (channels,), dtype, init=0.0)
    self.beta = _param(rngs, (channels,), dtype, init=0.0)
    self.no_div_by_zero = 1e-9

  def __call__(self, x: Array) -> Array:
    alpha = jnp.exp(self.alpha.get_value().astype(x.dtype))[None, :, None]
    beta = jnp.exp(self.beta.get_value().astype(x.dtype))[None, :, None]
    return x + (1.0 / (beta + self.no_div_by_zero)) * jnp.square(jnp.sin(x * alpha))


class Qwen3OmniConvNeXtBlock(nnx.Module):
  """HF ``Qwen3OmniMoeConvNeXtBlock``: depthwise causal conv -> LayerNorm -> MLP(GELU) -> layer scale -> residual."""

  def __init__(self, dim, *, dtype, rngs):
    self.dwconv = Qwen3OmniCausalConv1d(dim, dim, kernel_size=7, groups=dim, dilation=1, dtype=dtype, rngs=rngs)
    self.norm = ParamGroup(weight=_param(rngs, (dim,), dtype, init=1.0), bias=_param(rngs, (dim,), dtype, init=0.0))
    self.pwconv1 = ParamGroup(weight=_param(rngs, (4 * dim, dim), dtype), bias=_param(rngs, (4 * dim,), dtype, init=0.0))
    self.pwconv2 = ParamGroup(weight=_param(rngs, (dim, 4 * dim), dtype), bias=_param(rngs, (dim,), dtype, init=0.0))
    self.gamma = _param(rngs, (dim,), dtype, init=1e-6)
    self.eps = 1e-6

  def __call__(self, x: Array) -> Array:
    residual = x
    h = self.dwconv(x)
    h = jnp.swapaxes(h, 1, 2)  # [B, T, C]
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    h = (h - mean) * lax.rsqrt(var + self.eps)
    h = h * self.norm.weight.get_value().astype(h.dtype) + self.norm.bias.get_value().astype(h.dtype)
    h = h @ self.pwconv1.weight.get_value().astype(h.dtype).T + self.pwconv1.bias.get_value().astype(h.dtype)
    h = jax.nn.gelu(h, approximate=False)
    h = h @ self.pwconv2.weight.get_value().astype(h.dtype).T + self.pwconv2.bias.get_value().astype(h.dtype)
    h = self.gamma.get_value().astype(h.dtype) * h
    return residual + jnp.swapaxes(h, 1, 2)


class Qwen3OmniCode2WavTransformerLayer(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavTransformerLayer``: sliding-window causal attention + SwiGLU MLP with layer scale."""

  def __init__(self, config: Config, *, dtype, rngs):
    audio = get_audio_config(config)
    hidden = audio.code2wav.hidden_size
    self.num_heads = audio.code2wav.num_attention_heads
    self.num_kv_heads = audio.code2wav.num_key_value_heads
    self.head_dim = hidden // self.num_heads
    self.sliding_window = audio.code2wav.sliding_window
    self.rope_theta = audio.code2wav.rope_theta
    self.eps = audio.code2wav.rms_norm_eps
    inter = audio.code2wav.intermediate_size
    self.self_attn = ParamGroup(
        q_proj=_param(rngs, (self.num_heads * self.head_dim, hidden), dtype),
        k_proj=_param(rngs, (self.num_kv_heads * self.head_dim, hidden), dtype),
        v_proj=_param(rngs, (self.num_kv_heads * self.head_dim, hidden), dtype),
        o_proj=_param(rngs, (hidden, self.num_heads * self.head_dim), dtype),
    )
    self.mlp = ParamGroup(
        gate_proj=_param(rngs, (inter, hidden), dtype),
        up_proj=_param(rngs, (inter, hidden), dtype),
        down_proj=_param(rngs, (hidden, inter), dtype),
    )
    self.input_layernorm = _param(rngs, (hidden,), dtype, init=1.0)
    self.post_attention_layernorm = _param(rngs, (hidden,), dtype, init=1.0)
    self.self_attn_layer_scale = _param(rngs, (hidden,), dtype, init=audio.code2wav.layer_scale_initial_scale)
    self.mlp_layer_scale = _param(rngs, (hidden,), dtype, init=audio.code2wav.layer_scale_initial_scale)

  def _rms_norm(self, x: Array, weight: nnx.Param) -> Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
    return weight.get_value().astype(x.dtype) * x32.astype(x.dtype)

  def _rope(self, x: Array) -> Array:
    """HF rotate-half RoPE over ``[B, T, H, D]`` with positions ``arange(T)``."""
    seq = x.shape[1]
    dim = x.shape[-1]
    inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.arange(seq, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[None, :, None, :]
    sin = jnp.sin(emb)[None, :, None, :]
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (x.astype(jnp.float32) * cos + rotated * sin).astype(x.dtype)

  def __call__(self, x: Array) -> Array:
    batch, seq, _ = x.shape
    residual = x
    h = self._rms_norm(x, self.input_layernorm)
    dt = h.dtype
    q = (h @ self.self_attn.q_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_heads, self.head_dim)
    k = (h @ self.self_attn.k_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_kv_heads, self.head_dim)
    v = (h @ self.self_attn.v_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_kv_heads, self.head_dim)
    q = self._rope(q)
    k = self._rope(k)
    rep = self.num_heads // self.num_kv_heads
    if rep > 1:
      k = jnp.repeat(k, rep, axis=2)
      v = jnp.repeat(v, rep, axis=2)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * (self.head_dim**-0.5)
    q_idx = jnp.arange(seq)[:, None]
    k_idx = jnp.arange(seq)[None, :]
    allowed = (k_idx <= q_idx) & (k_idx > q_idx - self.sliding_window)
    scores = jnp.where(allowed[None, None], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(dt)
    attn = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(batch, seq, self.num_heads * self.head_dim)
    attn = attn @ self.self_attn.o_proj.get_value().astype(dt).T
    x = residual + self.self_attn_layer_scale.get_value().astype(dt) * attn

    residual = x
    h = self._rms_norm(x, self.post_attention_layernorm)
    mlp = self.mlp
    gate = h @ mlp.gate_proj.get_value().astype(dt).T
    up = h @ mlp.up_proj.get_value().astype(dt).T
    m = (jax.nn.silu(gate) * up) @ mlp.down_proj.get_value().astype(dt).T
    return residual + self.mlp_layer_scale.get_value().astype(dt) * m


class Qwen3OmniCode2WavTransformer(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavTransformerModel``: stacked sliding-window layers + final RMSNorm."""

  def __init__(self, config: Config, *, dtype, rngs):
    audio = get_audio_config(config)
    self.num_layers = audio.code2wav.num_hidden_layers
    self.eps = audio.code2wav.rms_norm_eps
    for i in range(self.num_layers):
      setattr(
          self,
          f"layers_{i}",
          Qwen3OmniCode2WavTransformerLayer(config, dtype=dtype, rngs=rngs),
      )
    self.norm = _param(rngs, (audio.code2wav.hidden_size,), dtype, init=1.0)

  def __call__(self, hidden: Array) -> Array:
    for i in range(self.num_layers):
      hidden = getattr(self, f"layers_{i}")(hidden)
    h32 = hidden.astype(jnp.float32)
    h32 = h32 * lax.rsqrt(jnp.mean(jnp.square(h32), axis=-1, keepdims=True) + self.eps)
    return self.norm.get_value().astype(hidden.dtype) * h32.astype(hidden.dtype)


class Qwen3OmniCode2WavDecoderResidualUnit(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavDecoderResidualUnit``."""

  def __init__(self, dim, dilation, *, dtype, rngs):
    self.act1 = Qwen3OmniSnakeBeta(dim, dtype=dtype, rngs=rngs)
    self.conv1 = Qwen3OmniCausalConv1d(dim, dim, kernel_size=7, dilation=dilation, dtype=dtype, rngs=rngs)
    self.act2 = Qwen3OmniSnakeBeta(dim, dtype=dtype, rngs=rngs)
    self.conv2 = Qwen3OmniCausalConv1d(dim, dim, kernel_size=1, dtype=dtype, rngs=rngs)

  def __call__(self, x: Array) -> Array:
    return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class Qwen3OmniCode2WavDecoderBlock(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavDecoderBlock``: SnakeBeta -> CausalTransConv(2r, r) -> 3 residual units."""

  def __init__(self, config: Config, layer_idx: int, *, dtype, rngs):
    audio = get_audio_config(config)
    in_dim = audio.code2wav.decoder_dim // 2**layer_idx
    out_dim = audio.code2wav.decoder_dim // 2 ** (layer_idx + 1)
    rate = audio.code2wav.upsample_rates[layer_idx]
    blocks = [
        Qwen3OmniSnakeBeta(in_dim, dtype=dtype, rngs=rngs),
        Qwen3OmniCausalTransConv1d(in_dim, out_dim, 2 * rate, rate, dtype=dtype, rngs=rngs),
    ] + [Qwen3OmniCode2WavDecoderResidualUnit(out_dim, d, dtype=dtype, rngs=rngs) for d in (1, 3, 9)]
    self.num_blocks = len(blocks)
    for i, block in enumerate(blocks):
      setattr(self, f"block_{i}", block)

  def __call__(self, x: Array) -> Array:
    for i in range(self.num_blocks):
      x = getattr(self, f"block_{i}")(x)
    return x


class Qwen3OmniCode2Wav(nnx.Module):
  """HF ``Qwen3OmniMoeCode2Wav`` vocoder: ``[batch, num_quantizers, T]`` codes -> ``[batch, 1, T * total_upsample]``."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    audio = get_audio_config(config)
    self.config = config
    dtype = jnp.dtype(audio.code2wav.dtype)
    self.dtype = dtype
    hidden = audio.code2wav.hidden_size
    self.codebook_size = audio.code2wav.codebook_size
    self.num_quantizers = audio.code2wav.num_quantizers
    self.total_upsample = int(math.prod(audio.code2wav.upsample_rates) * math.prod(audio.code2wav.upsampling_ratios))
    self.code_embedding = _param(rngs, (self.codebook_size * self.num_quantizers, hidden), dtype)
    self.num_transformer_layers = audio.code2wav.num_hidden_layers
    self.pre_transformer = Qwen3OmniCode2WavTransformer(config, dtype=dtype, rngs=rngs)
    self.num_upsample = len(audio.code2wav.upsampling_ratios)
    for i, factor in enumerate(audio.code2wav.upsampling_ratios):
      setattr(
          self,
          f"upsample_{i}_0",
          Qwen3OmniCausalTransConv1d(hidden, hidden, factor, factor, dtype=dtype, rngs=rngs),
      )
      setattr(
          self,
          f"upsample_{i}_1",
          Qwen3OmniConvNeXtBlock(hidden, dtype=dtype, rngs=rngs),
      )
    decoder = [Qwen3OmniCausalConv1d(hidden, audio.code2wav.decoder_dim, 7, dtype=dtype, rngs=rngs)]
    for i in range(len(audio.code2wav.upsample_rates)):
      decoder.append(Qwen3OmniCode2WavDecoderBlock(config, i, dtype=dtype, rngs=rngs))
    output_dim = audio.code2wav.decoder_dim // 2 ** len(audio.code2wav.upsample_rates)
    decoder += [
        Qwen3OmniSnakeBeta(output_dim, dtype=dtype, rngs=rngs),
        Qwen3OmniCausalConv1d(output_dim, 1, 7, dtype=dtype, rngs=rngs),
    ]
    self.num_decoder_blocks = len(decoder)
    for i, block in enumerate(decoder):
      setattr(self, f"decoder_{i}", block)

  def __call__(self, codes: Array) -> Array:
    if codes.shape[1] != self.num_quantizers:
      raise ValueError(f"Expected {self.num_quantizers} layers of codes, got {codes.shape[1]}")
    offset = (jnp.arange(self.num_quantizers, dtype=codes.dtype) * self.codebook_size)[None, :, None]
    table = self.code_embedding.get_value()
    hidden = jnp.take(table, codes + offset, axis=0).astype(self.dtype).mean(axis=1)  # [B, T, D]
    hidden = self.pre_transformer(hidden)
    hidden = jnp.swapaxes(hidden, 1, 2)  # [B, D, T]
    for i in range(self.num_upsample):
      hidden = getattr(self, f"upsample_{i}_0")(hidden)
      hidden = getattr(self, f"upsample_{i}_1")(hidden)
    wav = hidden
    for i in range(self.num_decoder_blocks):
      wav = getattr(self, f"decoder_{i}")(wav)
    return jnp.clip(wav, -1.0, 1.0)

  def chunked_decode(self, codes: Array, chunk_size: int = 300, left_context_size: int = 25) -> Array:
    """Decodes long code sequences in chunks with left context (mirrors HF ``chunked_decode``)."""
    wavs = []
    start_index = 0
    total = codes.shape[-1]
    while start_index < total:
      end_index = min(start_index + chunk_size, total)
      context_size = left_context_size if start_index - left_context_size > 0 else start_index
      codes_chunk = codes[..., start_index - context_size : end_index]
      wav_chunk = self(codes_chunk)
      wavs.append(wav_chunk[..., context_size * self.total_upsample :])
      start_index = end_index
    return jnp.concatenate(wavs, axis=-1)
