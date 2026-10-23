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

"""Qwen3-Omni speech generation pipeline (thinker outputs -> talker codes -> waveform).

Mirrors HF ``Qwen3OmniMoeForConditionalGeneration.generate`` after the thinker has produced its text:

1. Build the talker prefill sequence from the thinker prompt (user part: projected thinker embeddings /
   ``accept_hidden_layer`` hidden states) and the assistant header + first generated token (with codec special tokens).
2. Autoregressively generate codec frames: the talker predicts code group 0, the code predictor fills the
   remaining residual groups, and the summed codec embeddings (+ the next thinker text embedding) feed the
   next talker step. Stops at ``codec_eos_token_id`` or after ``talker_max_new_tokens`` frames.
3. Decode ``[num_code_groups, T]`` codes to a 24 kHz waveform with Code2Wav and write a WAV file.

The talker and code2wav are sub-modules of ``models.Transformer`` (``enable_audio_output=true``), so they share the
thinker's mesh, params and checkpoint. The talker runs with MaxText KV caches (PREFILL then AUTOREGRESSIVE mode) at
the thinker's decode batch size (one stream is broadcast to every row; row 0 is used).
"""

import dataclasses
import functools
import sys
import time
from typing import Any, Sequence

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.inference.audio_generation import AudioGenerationRequest, AudioGenerationResult
from maxtext.inference.component_runtime import GenerationContext
from maxtext.multimodal.audio_utils import write_wav
from maxtext.multimodal import processor_qwen3_omni
from maxtext.utils import max_logging

AUDIO_SAMPLE_RATE = 24000
# Talker ids in the last `_NUM_SUPPRESSED_TOKENS` slots are control tokens; HF never samples them (except eos).
_NUM_SUPPRESSED_TOKENS = 1024
_CODE2WAV_CHUNK_SIZE = 300
_CODE2WAV_LEFT_CONTEXT = 25


@dataclasses.dataclass
class TalkerPrefillInputs:
  """Talker prefill sequence for one stream."""

  inputs_embeds: jax.Array  # [1, true_length, emb]
  input_ids: np.ndarray  # [1, true_length] thinker-vocab ids used for MRoPE
  trailing_text_hidden: jax.Array  # [1, n_trailing, emb]
  tts_pad_embed: jax.Array  # [1, 1, emb]


class Qwen3OmniTalkerPipeline:
  """Drives the talker / code predictor / code2wav sub-modules of a loaded MaxText Qwen3-Omni model."""

  def __init__(self, config, context: GenerationContext):
    """Uses loaded components supplied by the shared runtime; never loads a second checkpoint."""
    self.config = config
    self.context = context
    self.mesh = context.mesh
    self.talker_runtime = context.components["talker"]
    self.code2wav_runtime = context.components["code2wav"]
    self.talker_params = self.talker_runtime.params
    self.code2wav_params = self.code2wav_runtime.params
    self.init_cache = self.talker_runtime.initial_cache()
    self.talker_config = self.talker_runtime.config
    self.batch = self.talker_config.micro_batch_size_to_train_on
    self.max_prefill_length = self.talker_config.max_prefill_predict_length
    self.num_code_groups = config.num_code_groups
    self.codec_eos = config.codec_eos_token_id
    self.codec_vocab = config.talker_vocab_size
    self.suppress_start = self.codec_vocab - _NUM_SUPPRESSED_TOKENS
    self.total_upsample = int(np.prod(config.code2wav_upsample_rates) * np.prod(config.code2wav_upsampling_ratios))

  # ------------------------------------------------------------------ model helpers
  def generate(self, request: AudioGenerationRequest) -> AudioGenerationResult:
    """Adapts generic text outputs to Qwen conditioning, codes, waveform, and replay diagnostics."""
    layer_key = f"hidden_states_{self.config.accept_hidden_layer}"
    required = ("input_embeddings", layer_key)
    if any(key not in request.features for key in required):
      raise ValueError(f"Qwen3-Omni audio generation requires captured features {required}.")
    talker_inputs = {
        "input_embeddings": request.features["input_embeddings"],
        "accept_hidden_states": request.features[layer_key],
    }
    output_path = self.config.audio_output_path if request.output_path is None else request.output_path
    if output_path:
      save_thinker_outputs(
          output_path + ".thinker_outputs.npz",
          self.config,
          self.context.params,
          request.prompt_ids,
          request.generated_ids,
          talker_inputs,
          request.processor_outputs,
      )
    prefill = self._build_from_outputs(request.prompt_ids, request.generated_ids, talker_inputs)
    codes = self.generate_codes(prefill, request.processor_outputs)
    waveform = self.codes_to_waveform(codes)
    if output_path:
      _write_outputs(
          output_path,
          self,
          prefill,
          request.processor_outputs,
          codes,
          waveform,
          request.prompt_ids,
          request.generated_ids,
      )
    return AudioGenerationResult(waveform=waveform, sample_rate=AUDIO_SAMPLE_RATE, diagnostics={"codes": codes})

  def _build_from_outputs(self, prompt_ids, generated_ids, talker_inputs):
    token_embedding = self.context.params["token_embedder"]["embedding"]
    token_embedding = getattr(token_embedding, "value", token_embedding)
    return self.build_prefill_inputs(
        prompt_ids,
        generated_ids,
        talker_inputs["input_embeddings"],
        talker_inputs["accept_hidden_states"],
        token_embedding,
    )

  def _talker(self, params, cache_dict=None):
    return self.talker_runtime.merge(params, cache_dict)

  def _code2wav(self, params):
    return self.code2wav_runtime.merge(params)

  def _rules(self):
    return logical_axis_rules(self.context.logical_axis_rules)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _project(self, params, thinker_embed, thinker_hidden, mm_mask):
    """User-part talker inputs: hidden_projection at multimodal positions, text_projection elsewhere."""
    talker = self._talker(params)
    dtype = self.talker_config.dtype
    with self.mesh, self._rules():
      text = talker.text_projection(thinker_embed.astype(dtype))
      hidden = talker.hidden_projection(thinker_hidden.astype(dtype))
    return jnp.where(mm_mask[..., None], hidden, text)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _text_project(self, params, thinker_embed):
    talker = self._talker(params)
    with self.mesh, self._rules():
      return talker.text_projection(thinker_embed.astype(self.talker_config.dtype))

  @functools.partial(jax.jit, static_argnums=(0,))
  def _codec_embed(self, params, codec_ids):
    talker = self._talker(params)
    with self.mesh, self._rules():
      return talker.codec_embedding(codec_ids)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _prefill(self, params, cache_dict, inputs_embeds, positions, segment_ids, true_length):
    talker = self._talker(params, cache_dict)
    with self.mesh, self._rules():
      logits, hidden = talker(inputs_embeds, positions, segment_ids, model_mode=MODEL_MODE_PREFILL)
    new_cache = nnx.to_pure_dict(nnx.state(talker, nnx.Cache))
    last_logits = jax.lax.dynamic_slice_in_dim(logits, true_length - 1, 1, axis=1)
    last_hidden = jax.lax.dynamic_slice_in_dim(hidden, true_length - 1, 1, axis=1)
    return last_logits, last_hidden, new_cache

  @functools.partial(jax.jit, static_argnums=(0,))
  def _step(self, params, cache_dict, inputs_embed, position):
    talker = self._talker(params, cache_dict)
    with self.mesh, self._rules():
      logits, hidden = talker(inputs_embed, position, None, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    new_cache = nnx.to_pure_dict(nnx.state(talker, nnx.Cache))
    return logits, hidden, new_cache

  @functools.partial(jax.jit, static_argnums=(0,))
  def _residual_codes(self, params, talker_hidden, code0):
    """Runs the code predictor for one frame.

    Args:
      talker_hidden: ``[B, 1, emb]`` normed talker hidden state that produced ``code0``.
      code0: ``[B, 1]`` first code group.

    Returns:
      ``codes [B, num_code_groups]`` and the summed codec embeddings ``[B, 1, emb]`` for the next talker input.
    """
    talker = self._talker(params)
    cp = talker.code_predictor
    n = self.num_code_groups
    dtype = self.talker_config.dtype
    with self.mesh, self._rules():
      first_embed = talker.codec_embedding(code0)  # [B, 1, emb]
      batch, _, emb = first_embed.shape
      window = jnp.zeros((batch, n, emb), dtype=dtype)
      window = window.at[:, 0].set(talker_hidden[:, 0].astype(dtype))
      window = window.at[:, 1].set(first_embed[:, 0].astype(dtype))
      codes = [code0[:, 0]]
      summed = first_embed
      for g in range(n - 1):
        hidden = cp(window)  # [B, n, emb]
        logits = cp.head(g, hidden[:, g + 1 : g + 2])  # predicts code_{g+1}
        code = jnp.argmax(logits[:, 0], axis=-1)  # [B]
        codes.append(code)
        code_embed = cp.embed_residual(g, code[:, None])  # [B, 1, emb]
        summed = summed + code_embed
        if g + 2 < n:
          window = window.at[:, g + 2].set(code_embed[:, 0].astype(dtype))
    return jnp.stack(codes, axis=1), summed

  @functools.partial(jax.jit, static_argnums=(0,))
  def _sample_code0(self, logits, history_mask, rng):
    """Suppress control ids, apply repetition penalty, then greedy / top-k / top-p sampling (HF order)."""
    cfg = self.config
    logits = logits[:, -1].astype(jnp.float32)  # [B, vocab]
    ids = jnp.arange(logits.shape[-1])
    suppressed = (ids >= self.suppress_start) & (ids != self.codec_eos)
    logits = jnp.where(suppressed[None], -jnp.inf, logits)
    penalty = cfg.talker_repetition_penalty
    penalized = jnp.where(logits < 0, logits * penalty, logits / penalty)
    logits = jnp.where(history_mask[None], penalized, logits)
    if not cfg.talker_do_sample:
      return jnp.argmax(logits, axis=-1)
    logits = logits / cfg.talker_temperature
    if cfg.talker_top_k > 0:
      top_values = jax.lax.top_k(logits, cfg.talker_top_k)[0][:, -1:]
      logits = jnp.where(logits < top_values, -jnp.inf, logits)
    if cfg.talker_top_p < 1.0:
      sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
      cumulative = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
      cutoff = jnp.sum(cumulative < cfg.talker_top_p, axis=-1, keepdims=True)
      threshold = jnp.take_along_axis(sorted_logits, cutoff, axis=-1)
      logits = jnp.where(logits < threshold, -jnp.inf, logits)
    return jax.random.categorical(rng, logits, axis=-1)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _decode_codes(self, params, codes):
    code2wav = self._code2wav(params)
    with self.mesh, self._rules():
      return code2wav(codes)

  # ------------------------------------------------------------------ prefill construction
  def build_prefill_inputs(
      self,
      prompt_ids: np.ndarray,
      generated_ids: Sequence[int],
      thinker_input_embeddings: jax.Array,
      thinker_accept_hidden: jax.Array,
      thinker_token_embedding: jax.Array,
  ) -> TalkerPrefillInputs:
    """Assembles the talker prefill embeddings (HF ``_get_talker_user_parts`` / ``_get_talker_assistant_parts``).

    Args:
      prompt_ids: ``[L]`` thinker prompt token ids (true length, no padding).
      generated_ids: thinker-generated ids (the final one is usually ``<|im_end|>``).
      thinker_input_embeddings: ``[1, >=L, D]`` merged thinker input embeddings (HF ``hidden_states[0]``).
      thinker_accept_hidden: ``[1, >=L, D]`` thinker hidden state after ``accept_hidden_layer`` layers.
      thinker_token_embedding: ``[vocab, D]`` thinker token embedding table (for generated / TTS tokens).
    """
    cfg = self.config
    tokens = processor_qwen3_omni.QwenTokens(cfg)
    prompt_ids = np.asarray(prompt_ids, dtype=np.int32)
    seq_len = prompt_ids.shape[0]
    generated = np.asarray(list(generated_ids), dtype=np.int32)
    if generated.shape[0] < 2:
      raise ValueError("Audio generation needs at least one generated thinker token before <|im_end|>.")

    thinker_embed = thinker_input_embeddings[:, :seq_len]
    thinker_hidden = thinker_accept_hidden[:, :seq_len]
    mm_ids = [tok for tok in (tokens.audio_pad, tokens.image_pad, tokens.video_pad) if tok is not None]
    mm_mask = np.isin(prompt_ids, mm_ids)[None, :]

    # Role of every prompt token (== the token following the most recent <|im_start|>).
    im_start_positions = np.where(prompt_ids == cfg.im_start_token_id, np.arange(seq_len), -1)
    previous_im_start = np.clip(np.maximum.accumulate(im_start_positions), 0, None)
    role_ids = prompt_ids[np.clip(previous_im_start + 1, None, seq_len - 1)]
    user_indices = np.nonzero(role_ids == cfg.user_token_id)[0]
    if user_indices.size == 0:
      raise ValueError("Could not find a user turn in the prompt; audio generation needs the chat template.")
    assistant_starts = np.nonzero(
        (prompt_ids[:-1] == cfg.im_start_token_id) & (prompt_ids[1:] == cfg.assistant_token_id)
    )[0]
    if assistant_starts.size == 0:
      raise ValueError("Could not find '<|im_start|>assistant' in the prompt.")
    assistant_start = int(assistant_starts[-1])

    full_user_embeds = self._project(self.talker_params, thinker_embed, thinker_hidden, jnp.asarray(mm_mask))
    talker_user_embeds = jnp.take(full_user_embeds, jnp.asarray(user_indices), axis=1)
    talker_user_ids = prompt_ids[user_indices]

    # Assistant part: header tokens from the prompt + generated tokens (all but the last), embedded by the thinker.
    assistant_token_ids = np.concatenate([prompt_ids[assistant_start:], generated[:-1]])
    assistant_embed = jnp.take(thinker_token_embedding, jnp.asarray(assistant_token_ids), axis=0)[None]
    assistant_hidden = self._text_project(self.talker_params, assistant_embed)  # [1, 3 + n_gen - 1, emb]
    if assistant_hidden.shape[1] < 4:
      raise ValueError("Assistant segment too short for talker prefill (need header + first generated token).")
    tts_ids = jnp.asarray([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]])
    tts_embeds = self._text_project(self.talker_params, jnp.take(thinker_token_embedding, tts_ids, axis=0))
    tts_bos_embed, tts_eos_embed, tts_pad_embed = (tts_embeds[:, i : i + 1] for i in range(3))

    assistant_text_hidden = jnp.concatenate(
        [
            assistant_hidden[:, :3],
            jnp.broadcast_to(tts_pad_embed, (1, 4, tts_pad_embed.shape[-1])),
            tts_bos_embed,
            assistant_hidden[:, 3:4],
        ],
        axis=1,
    )
    speaker_id = cfg.speaker_id[cfg.speaker.lower()]
    codec_special = jnp.asarray(
        [
            [
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
                speaker_id,
                cfg.codec_pad_id,
                cfg.codec_bos_id,
            ]
        ],
        dtype=jnp.int32,
    )
    codec_hidden = self._codec_embed(self.talker_params, codec_special)
    assistant_codec_hidden = jnp.concatenate(
        [jnp.zeros((1, 3, codec_hidden.shape[-1]), dtype=codec_hidden.dtype), codec_hidden], axis=1
    )
    # One trailing entry per later generated token, then the TTS EOS marker. (HF additionally re-positions the
    # marker for batched rows that stopped early; decode.py hands us one stream already truncated at its EOS.)
    trailing_text_hidden = jnp.concatenate([assistant_hidden[:, 4:], tts_eos_embed], axis=1)

    inputs_embeds = jnp.concatenate(
        [talker_user_embeds.astype(assistant_text_hidden.dtype), assistant_text_hidden + assistant_codec_hidden], axis=1
    )
    input_ids = np.concatenate([talker_user_ids, np.full((assistant_text_hidden.shape[1],), cfg.tts_pad_token_id)])
    return TalkerPrefillInputs(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids[None, :].astype(np.int64),
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )

  def talker_positions(self, prefill: TalkerPrefillInputs, processor_outputs) -> tuple[np.ndarray, np.ndarray]:
    """3-D MRoPE positions for the talker prefill sequence (HF talker ``get_rope_index``)."""
    cfg = self.config
    num_videos = getattr(processor_outputs, "num_videos", 0)
    position_ids, deltas = processor_qwen3_omni.get_rope_index(
        input_ids=prefill.input_ids,
        image_grid_thw=getattr(processor_outputs, "pixel_grid_thw", None),
        video_grid_thw=getattr(processor_outputs, "video_grid_thw", None),
        attention_mask=np.ones_like(prefill.input_ids),
        use_audio_in_video=cfg.use_audio and num_videos > 0,
        audio_lengths=getattr(processor_outputs, "audio_lengths", None),
        second_per_grids=getattr(processor_outputs, "video_second_per_grid", None),
        spatial_merge_size=cfg.spatial_merge_size_for_vit,
        position_id_per_seconds=cfg.position_id_per_seconds,
        config=cfg,
    )
    return np.asarray(position_ids), np.asarray(deltas)

  # ------------------------------------------------------------------ generation
  def generate_codes(self, prefill: TalkerPrefillInputs, processor_outputs, rng: jax.Array | None = None) -> np.ndarray:
    """Generates codec codes ``[num_code_groups, T]`` for the talker prefill inputs."""
    cfg = self.config
    rng = jax.random.PRNGKey(0) if rng is None else rng
    batch = self.batch
    true_length = int(prefill.inputs_embeds.shape[1])
    if true_length > self.max_prefill_length:
      raise ValueError(
          f"Talker prefill length {true_length} exceeds its budget {self.max_prefill_length}; "
          "raise max_prefill_predict_length."
      )
    position_ids, deltas = self.talker_positions(prefill, processor_outputs)  # [1, L, 3], [1, 1]
    rope_delta = int(deltas[0, 0])
    pad = self.max_prefill_length - true_length
    embeds = jnp.pad(prefill.inputs_embeds, ((0, 0), (0, pad), (0, 0)))
    pad_pos = position_ids[:, -1:, :] + np.arange(1, pad + 1, dtype=position_ids.dtype)[None, :, None]
    positions = jnp.asarray(np.concatenate([position_ids, pad_pos], axis=1), dtype=jnp.int32)
    segment_ids = jnp.asarray(
        np.concatenate([np.full((1, true_length), DECODING_ACTIVE_SEQUENCE_INDICATOR), np.zeros((1, pad))], axis=1),
        dtype=jnp.int32,
    )

    def bcast(x):  # one stream -> every decode row
      return jnp.broadcast_to(x, (batch,) + tuple(x.shape[1:]))

    history = jnp.zeros((self.codec_vocab,), dtype=bool)
    start = time.time()
    logits, hidden, cache = self._prefill(
        self.talker_params, self.init_cache, bcast(embeds), bcast(positions), bcast(segment_ids), true_length
    )
    rng, sub = jax.random.split(rng)
    code0 = self._sample_code0(logits, history, sub)  # [B]
    max_logging.log(f"[talker] prefill ({true_length} tokens) done in {time.time() - start:.1f}s")

    frames = []
    num_trailing = int(prefill.trailing_text_hidden.shape[1])
    start = time.time()
    for step in range(cfg.talker_max_new_tokens):
      if int(code0[0]) == self.codec_eos:
        break
      history = history.at[code0[0]].set(True)
      codes, summed = self._residual_codes(self.talker_params, hidden, code0[:, None])
      frames.append(np.asarray(codes[0]))
      text_hidden = prefill.trailing_text_hidden[:, step : step + 1] if step < num_trailing else prefill.tts_pad_embed
      inputs_embed = summed + bcast(text_hidden.astype(summed.dtype))
      position = jnp.full((batch, 1), true_length + step + rope_delta, dtype=jnp.int32)
      logits, hidden, cache = self._step(self.talker_params, cache, inputs_embed, position)
      rng, sub = jax.random.split(rng)
      code0 = self._sample_code0(logits, history, sub)
      if (step + 1) % 50 == 0:
        max_logging.log(f"[talker] generated {step + 1} frames ({(step + 1) / 12.5:.1f}s of audio)")
    max_logging.log(f"[talker] generated {len(frames)} frames in {time.time() - start:.1f}s")
    if not frames:
      return np.zeros((self.num_code_groups, 0), dtype=np.int32)
    return np.stack(frames, axis=1).astype(np.int32)  # [num_code_groups, T]

  def codes_to_waveform(self, codes: np.ndarray) -> np.ndarray:
    """Decodes ``[num_code_groups, T]`` codes into a float32 waveform in [-1, 1] (HF ``chunked_decode``)."""
    if codes.shape[-1] == 0:
      return np.zeros((0,), dtype=np.float32)
    codes = jnp.asarray(codes[None], dtype=jnp.int32)  # [1, nq, T]
    wavs = []
    start_index = 0
    total = codes.shape[-1]
    while start_index < total:
      end_index = min(start_index + _CODE2WAV_CHUNK_SIZE, total)
      context = _CODE2WAV_LEFT_CONTEXT if start_index - _CODE2WAV_LEFT_CONTEXT > 0 else start_index
      wav = self._decode_codes(self.code2wav_params, codes[..., start_index - context : end_index])
      wavs.append(np.asarray(wav[0, 0, context * self.total_upsample :], dtype=np.float32))
      start_index = end_index
    return np.concatenate(wavs)


def generate_speech(
    config,
    engine: maxengine.MaxEngine,
    params,
    prompt_ids: np.ndarray,
    generated_ids: Sequence[int],
    talker_inputs: dict[str, Any],
    processor_outputs,
    output_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """End-to-end: thinker outputs -> talker codes -> waveform (+ optional WAV file).

  Args:
    config: thinker config.
    engine / params: the ``MaxEngine`` and params used for text decoding.
    prompt_ids: ``[L]`` prompt token ids (true length).
    generated_ids: thinker-generated ids.
    talker_inputs: legacy mapping of input embeddings and accept-layer hidden states, also used by replay dumps.
    processor_outputs: multimodal preprocessor output (grids / audio lengths for MRoPE), or ``None``.

  Returns ``(codes [num_code_groups, T], waveform [num_samples])``.
  """
  pipeline = Qwen3OmniTalkerPipeline(config, engine.get_generation_context(params))
  token_embedding = params["token_embedder"]["embedding"]
  token_embedding = getattr(token_embedding, "value", token_embedding)
  prefill = pipeline.build_prefill_inputs(
      prompt_ids, generated_ids, talker_inputs["input_embeddings"], talker_inputs["accept_hidden_states"], token_embedding
  )
  codes = pipeline.generate_codes(prefill, processor_outputs)
  waveform = pipeline.codes_to_waveform(codes)
  output_path = output_path or config.audio_output_path
  if output_path:
    _write_outputs(output_path, pipeline, prefill, processor_outputs, codes, waveform, prompt_ids, generated_ids)
  return codes, waveform


def _write_outputs(output_path, pipeline, prefill, processor_outputs, codes, waveform, prompt_ids, generated_ids):
  """Writes the WAV plus a ``*.talker_debug.npz`` dump of the exact talker prefill.

  The dump lets ``tests/utils/qwen3_omni_talker_hf_reference.py`` run the HF talker on identical inputs.
  """
  write_wav(output_path, waveform, AUDIO_SAMPLE_RATE)
  max_logging.log(
      f"[talker] wrote {waveform.shape[0] / AUDIO_SAMPLE_RATE:.2f}s of audio ({codes.shape[-1]} frames) to {output_path}"
  )
  position_ids, deltas = pipeline.talker_positions(prefill, processor_outputs)
  np.savez(
      output_path + ".talker_debug.npz",
      inputs_embeds=np.asarray(prefill.inputs_embeds.astype(jnp.float32)),
      input_ids=prefill.input_ids,
      trailing_text_hidden=np.asarray(prefill.trailing_text_hidden.astype(jnp.float32)),
      tts_pad_embed=np.asarray(prefill.tts_pad_embed.astype(jnp.float32)),
      position_ids=position_ids,
      rope_deltas=deltas,
      codes=codes,
      prompt_ids=np.asarray(prompt_ids),
      generated_ids=np.asarray(list(generated_ids)),
  )


# ----------------------------------------------------------------------------- offline replay
_PROCESSOR_FIELDS = (
    "pixel_grid_thw",
    "video_grid_thw",
    "audio_lengths",
    "video_second_per_grid",
    "num_videos",
    "num_images",
)


def save_thinker_outputs(path, config, params, prompt_ids, generated_ids, talker_inputs, processor_outputs):
  """Dumps everything the talker needs from the thinker so it can be re-run without reloading the thinker."""
  prompt_ids = np.asarray(prompt_ids, dtype=np.int32)
  generated = np.asarray(list(generated_ids), dtype=np.int32)
  token_embedding = params["token_embedder"]["embedding"]
  token_embedding = getattr(token_embedding, "value", token_embedding)
  needed = np.unique(
      np.concatenate([prompt_ids, generated, [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
  )
  payload = {
      "prompt_ids": prompt_ids,
      "generated_ids": generated,
      "input_embeddings": np.asarray(talker_inputs["input_embeddings"].astype(jnp.float32)),
      "accept_hidden_states": np.asarray(talker_inputs["accept_hidden_states"].astype(jnp.float32)),
      "embedding_ids": needed,
      "embedding_rows": np.asarray(jnp.take(token_embedding, jnp.asarray(needed), axis=0).astype(jnp.float32)),
      "vocab_size": np.asarray(token_embedding.shape[0]),
  }
  for field in _PROCESSOR_FIELDS:
    value = getattr(processor_outputs, field, None)
    if value is not None:
      payload["processor_" + field] = np.asarray(value)
  np.savez(path, **payload)
  max_logging.log(f"[talker] saved thinker outputs to {path}")


class _ProcessorOutputs:  # pylint: disable=too-few-public-methods
  """Minimal stand-in for the preprocessor output when replaying from a dump."""

  def __init__(self, dump):
    for field in _PROCESSOR_FIELDS:
      key = "processor_" + field
      value = dump[key] if key in dump else None
      if value is not None and value.ndim == 0:
        value = value.item()
      setattr(self, field, value)


def replay(config, dump_path: str, output_path: str) -> tuple[np.ndarray, np.ndarray]:
  """Re-runs the speech stack from a ``*.thinker_outputs.npz`` or ``*.talker_debug.npz`` dump.

  Loads the (thinker + talker) checkpoint through ``MaxEngine`` but skips thinker inference, so talker changes (or
  fp32 parity runs with ``weight_dtype=float32 dtype=float32 matmul_precision=highest``) iterate quickly.
  """
  engine = maxengine.MaxEngine(config)
  params = engine.load_params(jax.random.PRNGKey(0))
  dump = np.load(dump_path)
  pipeline = Qwen3OmniTalkerPipeline(config, engine.get_generation_context(params))
  if "talker_debug" in dump_path:
    prefill = TalkerPrefillInputs(
        inputs_embeds=jnp.asarray(dump["inputs_embeds"]),
        input_ids=dump["input_ids"],
        trailing_text_hidden=jnp.asarray(dump["trailing_text_hidden"]),
        tts_pad_embed=jnp.asarray(dump["tts_pad_embed"]),
    )
    position_ids, rope_deltas = dump["position_ids"], dump["rope_deltas"]
    pipeline.talker_positions = lambda *_: (position_ids, rope_deltas)  # replay the dumped MRoPE positions
    processor_outputs = None
  else:
    table = np.zeros((int(dump["vocab_size"]), dump["embedding_rows"].shape[1]), dtype=np.float32)
    table[dump["embedding_ids"]] = dump["embedding_rows"]
    prefill = pipeline.build_prefill_inputs(
        dump["prompt_ids"],
        list(dump["generated_ids"]),
        jnp.asarray(dump["input_embeddings"], dtype=jnp.bfloat16),
        jnp.asarray(dump["accept_hidden_states"], dtype=jnp.bfloat16),
        jnp.asarray(table, dtype=jnp.bfloat16),
    )
    processor_outputs = _ProcessorOutputs(dump)
  codes = pipeline.generate_codes(prefill, processor_outputs)
  waveform = pipeline.codes_to_waveform(codes)
  _write_outputs(
      output_path, pipeline, prefill, processor_outputs, codes, waveform, dump["prompt_ids"], dump["generated_ids"]
  )
  if "codes" in dump:
    ref = dump["codes"]
    common = min(ref.shape[-1], codes.shape[-1])
    if common:
      agreement = float((ref[0, :common] == codes[0, :common]).mean())
      max_logging.log(
          f"[talker] replay: {codes.shape[-1]} frames (dump had {ref.shape[-1]}), code0 agreement {agreement:.3f}"
      )
  return codes, waveform


if __name__ == "__main__":
  # Usage: python -m maxtext.inference.qwen3_omni_talker_pipeline <base.yml> model_name=qwen3-omni-30b-a3b \
  #   load_parameters_path=... enable_audio_output=true ... --dump=<...npz> --output=<out.wav>
  _dump = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--dump="))
  _output = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--output="))
  _argv = [a for a in sys.argv if not a.startswith("--dump=") and not a.startswith("--output=")]
  replay(pyconfig.initialize(_argv), _dump, _output)
