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

"""Qwen3-Omni Talker: speech-token generation stack (talker MoE decoder and code predictor).

This module implements the *audio output* half of Qwen3-Omni (HF `Qwen3OmniMoeForConditionalGeneration`):

* ``Qwen3OmniTalker``: a 20-layer MoE decoder (Qwen3-MoE style attention with a sigmoid-gated shared expert)
  that consumes projected thinker embeddings/hidden states and predicts the first codec code group.
* ``Qwen3OmniCodePredictor``: a small dense decoder that autoregressively predicts the remaining
  ``num_code_groups - 1`` residual codec codes of a frame.
* The waveform decoder is implemented in ``qwen3_omni_code2wav``.

``models.Transformer`` instantiates ``talker`` (which owns ``code_predictor``) and ``code2wav`` when
``config.enable_audio_output`` is set, mirroring the HF module layout (``talker.*``, ``talker.code_predictor.*``,
``code2wav.*``) so one MaxText checkpoint holds thinker and speech stack. Architecture fields come from the
``talker_*`` / ``code_predictor_*`` / ``code2wav_*`` config entries (HF names); the talker derives a MaxText
sub-config from the thinker config so it can reuse ``AttentionWithNorm``, ``RoutedMoE``, ``MlpBlock`` and
``Qwen3DecoderLayer`` (KV caching / sharding follow the rest of MaxText). The code predictor sees at most
``num_code_groups`` positions per frame, so it runs cache-free in ``MODEL_MODE_TRAIN`` over a fixed-size window.
"""
# pylint: disable=arguments-differ
# pylint: disable=attribute-defined-outside-init

from flax import nnx
import jax
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
  assistant header (<= thinker prefill + 16) and its autoregressive budget is ``talker_max_new_tokens`` frames.
  """
  max_prefill = config.max_prefill_predict_length + 16
  shared_experts = config.talker_shared_expert_intermediate_size // config.talker_moe_intermediate_size
  if shared_experts * config.talker_moe_intermediate_size != config.talker_shared_expert_intermediate_size:
    raise ValueError("talker_shared_expert_intermediate_size must be a multiple of talker_moe_intermediate_size.")
  return derive_config(
      config,
      decoder_block=DecoderBlockType.QWEN3_MOE,
      base_emb_dim=config.talker_hidden_size,
      emb_dim=config.talker_hidden_size,
      base_num_query_heads=config.talker_num_attention_heads,
      num_query_heads=config.talker_num_attention_heads,
      base_num_kv_heads=config.talker_num_key_value_heads,
      num_kv_heads=config.talker_num_key_value_heads,
      head_dim=config.talker_head_dim,
      base_num_decoder_layers=config.talker_num_hidden_layers,
      num_decoder_layers=config.talker_num_hidden_layers,
      base_mlp_dim=config.talker_moe_intermediate_size,
      mlp_dim=config.talker_moe_intermediate_size,
      base_moe_mlp_dim=config.talker_moe_intermediate_size,
      moe_mlp_dim=config.talker_moe_intermediate_size,
      num_experts=config.talker_num_experts,
      num_experts_per_tok=config.talker_num_experts_per_tok,
      shared_experts=shared_experts,
      norm_topk_prob=config.talker_norm_topk_prob,
      vocab_size=config.talker_vocab_size,
      normalization_layer_epsilon=config.talker_rms_norm_eps,
      rope_max_timescale=int(config.talker_rope_theta),
      use_qk_norm=True,
      use_mrope=True,
      max_prefill_predict_length=max_prefill,
      max_target_length=max_prefill + config.talker_max_new_tokens,
      # Single speech stream; the KV cache batch follows the thinker's decode batch (see talker pipeline).
      use_multimodal=False,
      use_audio=False,
      mtp_num_layers=0,
      scan_layers=False,
  )


def code_predictor_config(config: Config) -> Config:
  """Derives the config for the dense code-predictor decoder layers from the thinker config."""
  num_positions = config.num_code_groups
  return derive_config(
      config,
      decoder_block=DecoderBlockType.QWEN3,
      base_emb_dim=config.code_predictor_hidden_size,
      emb_dim=config.code_predictor_hidden_size,
      base_num_query_heads=config.code_predictor_num_attention_heads,
      num_query_heads=config.code_predictor_num_attention_heads,
      base_num_kv_heads=config.code_predictor_num_key_value_heads,
      num_kv_heads=config.code_predictor_num_key_value_heads,
      head_dim=config.code_predictor_head_dim,
      base_mlp_dim=config.code_predictor_intermediate_size,
      mlp_dim=config.code_predictor_intermediate_size,
      base_num_decoder_layers=config.code_predictor_num_hidden_layers,
      num_decoder_layers=config.code_predictor_num_hidden_layers,
      num_experts=1,
      num_experts_per_tok=1,
      shared_experts=0,
      vocab_size=config.code_predictor_vocab_size,
      normalization_layer_epsilon=config.talker_rms_norm_eps,
      rope_max_timescale=int(config.code_predictor_rope_theta),
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

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant, *, model_mode: str, rngs: nnx.Rngs):
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
        cfg, config.thinker_hidden_size, config.talker_intermediate_size, cfg.emb_dim, rngs=rngs
    )
    self.hidden_projection = Qwen3OmniTalkerResizeMLP(
        cfg, config.thinker_hidden_size, config.talker_intermediate_size, cfg.emb_dim, rngs=rngs
    )
    self.num_layers = cfg.num_decoder_layers
    for i in range(self.num_layers):
      setattr(self, f"layers_{i}", Qwen3OmniTalkerDecoderLayer(cfg, mesh, model_mode, quant, rngs))
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
    self.thinker_config = config
    self.mesh = mesh
    cp_cfg = code_predictor_config(config)
    self.config = cp_cfg
    self.num_residual_groups = config.num_code_groups - 1
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
      setattr(self, f"layers_{i}", Qwen3DecoderLayer(cp_cfg, mesh, MODEL_MODE_TRAIN, quant, rngs))
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
