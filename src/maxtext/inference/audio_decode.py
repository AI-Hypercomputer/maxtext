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

"""Speech generation: text prompt -> WAV, for models with an `audio_decoder_block`.

Unlike `decode.py`, no tokens are sampled: every step the decoder's final-normed last hidden state goes
through the model's `AudioDecoder` block (`layers.encoders.AudioDecoder`), which returns a continuous audio
latent, the decoder input embedding for the next step and a stop flag. The block's vocoder then turns the
latents into a waveform. The prompt is split into short text fragments which are synthesized one after
another; the model-specific prompt protocol and fragmenting live in `multimodal.processor`.

Example (Ming-3.0-audio-tiny talker):
  python3 -m maxtext.inference.audio_decode src/maxtext/configs/base.yml \\
    model_name=ming-3.0-audio-tiny-talker tokenizer_path=<snapshot>/talker/llm \\
    load_parameters_path=<talker ckpt>/0/items scan_layers=false per_device_batch_size=1 \\
    max_prefill_predict_length=256 max_target_length=1280 attention=dot_product \\
    prompt="Hello, this is MaxText speaking." audio_output_path=/tmp/talker.wav

The decoder runs with MaxText KV caches: the prompt is padded to `max_prefill_predict_length`, so up to
`max_target_length - max_prefill_predict_length` latents fit per fragment. The same fragment is broadcast to
every row of the decode batch and row 0 is used.
"""

import functools
import os
import time
import wave
from typing import Sequence

from absl import app
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
  """Writes a mono waveform in [-1, 1] as 16-bit PCM WAV."""
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  with wave.open(path, "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes((np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


class AudioGenerator:
  """Drives a model's decoder + `AudioDecoder` block to synthesize speech from text fragments."""

  def __init__(self, config):
    if config.scan_layers:
      raise ValueError("audio_decode supports scan_layers=false only.")
    self.config = config
    self.mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    with self.mesh:
      model = model_creation_utils.from_pretrained(config, mesh=self.mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    if model.audio_decoder is None:
      raise ValueError(f"model_name={config.model_name} has no audio_decoder_block.")
    self.graphdef, self.params, self.cache, self.rest = nnx.split(model, nnx.Param, nnx.Cache, ...)
    self.sample_rate = model.audio_decoder.sample_rate
    self.samples_per_latent = model.audio_decoder.samples_per_latent
    self.batch = config.micro_batch_size_to_train_on
    self.max_steps = config.max_target_length - config.max_prefill_predict_length

  # ------------------------------------------------------------------ jitted model calls
  def _merge(self, params, cache=None):
    """Rebuilds the model; without `cache` the KV cache is zero-initialized inside the trace."""
    if cache is None:
      cache = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.cache)
    # `copy=True`: `rest` holds RNG counters, which must not be shared across jit traces.
    return nnx.merge(self.graphdef, params, cache, self.rest, copy=True)

  def _hidden(self, model, embeds, positions, segment_ids, model_mode):
    """Runs the decoder on input embeddings; returns the final-normed hidden states."""
    _, hidden, _ = model.decoder(
        shared_embedding=model.token_embedder,
        decoder_input_tokens=jnp.zeros(embeds.shape[:2], jnp.int32),
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=model_mode,
        decoder_input_embeddings=embeds.astype(self.config.dtype),
        skip_lm_head=True,  # Only the hidden state is used: skip the vocab projection.
    )
    return model.decoder.apply_decoder_norm(hidden)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _prefill(self, params, input_ids, true_length, key):
    """Prefills the padded prompt and runs the first audio decoder step."""
    model = self._merge(params)
    with self.mesh, logical_axis_rules(self.config.logical_axis_rules):
      embeds = model.token_embedder(input_ids)
      positions = jnp.broadcast_to(jnp.arange(input_ids.shape[1], dtype=jnp.int32), input_ids.shape)
      hidden = self._hidden(model, embeds, positions, (positions < true_length).astype(jnp.int32), MODEL_MODE_PREFILL)
      hidden = jax.lax.dynamic_slice_in_dim(hidden, true_length - 1, 1, axis=1)
      state = model.audio_decoder.init_state(self.batch)
      latent, embed, done, state = model.audio_decoder.step(hidden, state, key)
    return latent, embed, done, state, nnx.state(model, nnx.Cache)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _step(self, params, cache, embed, position, state, key):  # pylint: disable=too-many-positional-arguments
    """One autoregressive step: decoder on the previous latent's embedding, then the audio decoder."""
    model = self._merge(params, cache)
    with self.mesh, logical_axis_rules(self.config.logical_axis_rules):
      positions = jnp.full((self.batch, 1), position, jnp.int32)
      hidden = self._hidden(model, embed, positions, None, MODEL_MODE_AUTOREGRESSIVE)
      latent, embed, done, state = model.audio_decoder.step(hidden, state, key)
    return latent, embed, done, state, nnx.state(model, nnx.Cache)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _decode_audio(self, params, latents, num_latents):
    model = self._merge(params)
    with self.mesh, logical_axis_rules(self.config.logical_axis_rules):
      return model.audio_decoder.decode_audio(latents, num_latents)

  # ------------------------------------------------------------------ generation
  def generate_latents(self, input_ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """Autoregressively samples the `[N, latent]` latents of one prompt (`[L]` token ids); N = steps * patch.

    The stop flags are read on the host only every `audio_decoder_stop_check_interval` steps, so the steps in
    between are dispatched asynchronously instead of syncing on every step. Each step's noise depends only on
    `(seed, step)`, so truncating after the first stop gives the same latents as a per-step check.
    """
    prefill_length = self.config.max_prefill_predict_length
    interval = max(1, self.config.audio_decoder_stop_check_interval)
    true_length = int(input_ids.shape[0])
    if true_length > prefill_length:
      raise ValueError(f"Prompt has {true_length} tokens > max_prefill_predict_length={prefill_length}.")
    padded = np.zeros((self.batch, prefill_length), np.int32)
    padded[:, :true_length] = input_ids
    key = jax.random.PRNGKey(seed)

    latent, embed, done, state, cache = self._prefill(
        self.params, jnp.asarray(padded), jnp.int32(true_length), jax.random.fold_in(key, 0)
    )
    latents, dones = [latent], [done[0]]
    checked = 0  # dones[:checked] are known to be False.
    while True:
      if len(latents) - checked >= interval or len(latents) >= self.max_steps:
        hits = np.flatnonzero(jax.device_get(jnp.stack(dones[checked:])))
        if hits.size:
          return np.asarray(jnp.concatenate(latents[: checked + int(hits[0]) + 1], axis=1)[0])
        checked = len(latents)
        if checked >= self.max_steps:
          max_logging.log(f"Reached the step limit ({self.max_steps}) without the audio decoder stopping.")
          return np.asarray(jnp.concatenate(latents, axis=1)[0])
      step = len(latents)
      position = jnp.int32(true_length + step - 1)
      latent, embed, done, state, cache = self._step(
          self.params, cache, embed, position, state, jax.random.fold_in(key, step)
      )
      latents.append(latent)
      dones.append(done[0])

  def latents_to_waveform(self, latents: np.ndarray, bucket: int = 64) -> np.ndarray:
    """`[N, latent]` -> `[N * samples_per_latent]` float32 waveform.

    The vocoder runs on latents edge-padded to a multiple of `bucket` (one compilation per bucket instead of
    per length); `num_latents` lets the block mask the padding so the result is exact.
    """
    n = latents.shape[0]
    padded = np.pad(np.asarray(latents, np.float32), ((0, -n % bucket), (0, 0)), mode="edge")
    waveform = self._decode_audio(self.params, jnp.asarray(padded)[None], jnp.int32(n))
    return np.asarray(waveform[0, : n * self.samples_per_latent], np.float32)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  config = pyconfig.initialize(argv)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  generator = AudioGenerator(config)
  fragments = mm_processor.split_tts_text(config)
  if not fragments:
    raise ValueError("Empty prompt: nothing to synthesize.")

  sample_rate = generator.sample_rate
  chunks = []
  for index, fragment in enumerate(fragments):
    input_ids = mm_processor.get_tts_input_ids(config, tokenizer, fragment)
    start = time.time()
    latents = generator.generate_latents(input_ids, seed=config.audio_decoder_seed + index)
    generated = time.time()
    waveform = generator.latents_to_waveform(latents)
    chunks.append(mm_processor.postprocess_tts_waveform(config, waveform, sample_rate))
    max_logging.log(
        f"Fragment {index}: {fragment!r} -> {input_ids.shape[0]} prompt tokens, {latents.shape[0]} latents,"
        f" {waveform.shape[0] / sample_rate:.2f} s audio"
        f" (decoder {generated - start:.1f} s, vocoder {time.time() - generated:.1f} s)"
    )
  audio = np.concatenate(chunks)
  write_wav(config.audio_output_path, audio, sample_rate)
  max_logging.log(f"Wrote {audio.shape[0] / sample_rate:.2f} s of {sample_rate} Hz audio to {config.audio_output_path}")


if __name__ == "__main__":
  app.run(main)
