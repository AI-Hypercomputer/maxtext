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

"""Ming-3.0-audio-tiny talker inference: text prompt -> speech WAV.

Unlike `decode.py`, the talker does not emit tokens: every step the Qwen2 LLM's last hidden state is turned
into a continuous AudioVAE latent by a flow-matching DiT, and the latent is embedded back (aggregator) as
the next LLM input. Generation stops on the stop head; the AudioVAE decoder converts the latents to audio.
This mirrors the publisher's `talker/inference.py` (`MingTalker.synthesize`, "IP" task: named voice, no
reference audio).

Example:
  python3 -m maxtext.inference.ming_talker_decode src/maxtext/configs/base.yml \
    model_name=ming-3.0-audio-tiny-talker tokenizer_path=<snapshot>/talker/llm \
    load_parameters_path=<talker ckpt>/0/items scan_layers=false per_device_batch_size=1 \
    max_prefill_predict_length=256 max_target_length=1280 attention=dot_product \
    prompt="Hello, this is MaxText speaking." ming_talker_audio_output_path=/tmp/talker.wav

The LLM runs with MaxText KV caches: an AUTOREGRESSIVE-mode model serves the PREFILL call (prompt padded
to `max_prefill_predict_length`) and the AUTOREGRESSIVE steps, so up to
`max_target_length - max_prefill_predict_length` latents (0.08 s of audio each) fit per fragment. The stream
is broadcast to every row of the global decode batch and row 0 is used.
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

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.multimodal import processor_ming_talker
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
  """Writes a mono waveform in [-1, 1] as 16-bit PCM WAV."""
  pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
  with wave.open(str(path), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes(pcm.tobytes())


class MingTalkerGenerator:
  """Loads a converted Ming talker checkpoint and synthesizes speech from text fragments."""

  def __init__(self, config, model=None, mesh=None):
    if not config.ming_talker:
      raise ValueError("ming_talker_decode requires a talker config (model_name=ming-3.0-audio-tiny-talker).")
    if config.scan_layers:
      raise ValueError("ming_talker_decode currently supports scan_layers=false only.")
    self.config = config
    if mesh is None:
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    self.mesh = mesh
    if model is None:
      with self.mesh:
        model = model_creation_utils.from_pretrained(config, mesh=self.mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    self.graphdef, self.params, cache, self.rest = nnx.split(model, nnx.Param, nnx.Cache, ...)
    self.cache_template = cache
    self.init_cache = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), nnx.to_pure_dict(cache))
    self.batch = config.micro_batch_size_to_train_on
    self.prefill_length = config.max_prefill_predict_length
    self.max_steps = min(config.ming_talker_max_steps, config.max_target_length - config.max_prefill_predict_length)
    self.latent_shape = (config.ming_talker_patch_size, config.ming_talker_latent_dim)

  # ------------------------------------------------------------------ jitted model calls
  def _merge(self, params, cache_dict=None):
    cache = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.cache_template)
    if cache_dict is not None:
      nnx.replace_by_pure_dict(cache, cache_dict)
    return nnx.merge(self.graphdef, params, cache, self.rest, copy=True)

  def _rules(self):
    return logical_axis_rules(self.config.logical_axis_rules)

  def _llm(self, model, embeds, positions, segment_ids, model_mode):
    """Runs the talker LLM on input embeddings; returns the final-normed hidden states."""
    tokens = jnp.zeros(embeds.shape[:2], dtype=jnp.int32)
    _, hidden, _ = model.decoder(
        shared_embedding=model.token_embedder,
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=model_mode,
        decoder_input_embeddings=embeds.astype(self.config.dtype),
    )
    return model.decoder.apply_decoder_norm(hidden)

  def _heads(self, model, hidden, history, noise, sde_noise):
    cfg = self.config
    if cfg.ming_talker_sigma * cfg.ming_talker_temperature == 0:
      sde_noise = None  # Deterministic ODE sampling (the default).
    return model.ming_talker.sample_latent(
        hidden,
        history,
        noise,
        cfg.ming_talker_cfg_strength,
        cfg.ming_talker_sigma,
        cfg.ming_talker_temperature,
        sde_noise,
    )

  @functools.partial(jax.jit, static_argnums=(0,))
  def _prefill(self, params, cache_dict, input_ids, true_length, noise, sde_noise):
    """Prefills the padded prompt and samples the first latent."""
    model = self._merge(params, cache_dict)
    with self.mesh, self._rules():
      embeds = model.token_embedder(input_ids)
      positions = jnp.broadcast_to(jnp.arange(input_ids.shape[1], dtype=jnp.int32), input_ids.shape)
      segment_ids = (positions < true_length).astype(jnp.int32)
      hidden = self._llm(model, embeds, positions, segment_ids, MODEL_MODE_PREFILL)
      hidden = jax.lax.dynamic_slice_in_dim(hidden, true_length - 1, 1, axis=1)
      history = jnp.zeros((hidden.shape[0], self.config.ming_talker_history_patch_size, self.latent_shape[1]))
      latent, next_embed, stop_prob = self._heads(model, hidden, history, noise, sde_noise)
    return latent, next_embed, stop_prob, hidden, nnx.to_pure_dict(nnx.state(model, nnx.Cache))

  @functools.partial(jax.jit, static_argnums=(0,))
  def _step(self, params, cache_dict, embed, position, history, noise, sde_noise):
    """One autoregressive talker step: LLM on the previous latent's embedding, then the CFM head."""
    model = self._merge(params, cache_dict)
    with self.mesh, self._rules():
      positions = jnp.full((embed.shape[0], 1), position, dtype=jnp.int32)
      hidden = self._llm(model, embed, positions, None, MODEL_MODE_AUTOREGRESSIVE)
      latent, next_embed, stop_prob = self._heads(model, hidden, history, noise, sde_noise)
    return latent, next_embed, stop_prob, hidden, nnx.to_pure_dict(nnx.state(model, nnx.Cache))

  @functools.partial(jax.jit, static_argnums=(0,))
  def _embed_latent(self, params, latent):
    model = self._merge(params)
    with self.mesh, self._rules():
      return model.ming_talker.aggregator(latent)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _decode_audio(self, params, latents, num_latents):
    model = self._merge(params)
    with self.mesh, self._rules():
      return model.ming_talker.decode_audio(latents, num_latents)

  # ------------------------------------------------------------------ generation
  def generate_latents(
      self,
      input_ids: np.ndarray,
      noise: np.ndarray | None = None,
      seed: int = 0,
      forced_latents: np.ndarray | None = None,
  ):
    """Autoregressively samples latents for one prompt.

    Args:
      input_ids: `[L]` talker prompt token ids.
      noise: optional `[>=steps, P, latent]` CFM initial noise per step (e.g. replayed from an HF golden).
      seed: seed for the noise when `noise` is None.
      forced_latents: optional `[N, P, latent]` teacher-forcing latents (debugging): step `i + 1` is conditioned on
        `forced_latents[i]` instead of the sampled latent, and generation runs for exactly N steps.

    Returns:
      dict with `latents [N, P, latent]`, `stop_probs [N]` and `hidden [N, emb]` (row 0).
    """
    cfg = self.config
    true_length = int(input_ids.shape[0])
    if true_length > self.prefill_length:
      raise ValueError(f"Talker prompt has {true_length} tokens > max_prefill_predict_length={self.prefill_length}.")
    padded = np.zeros((self.batch, self.prefill_length), np.int32)
    padded[:, :true_length] = input_ids
    rng = jax.random.PRNGKey(seed)

    def step_noise(step):
      """Initial CFM noise `[B, P, latent]` and SDE noise `[steps, B, P, latent]` for one talker step."""
      key = jax.random.fold_in(rng, step)
      if noise is not None:
        if step >= noise.shape[0]:
          raise ValueError(f"Replayed noise has only {noise.shape[0]} steps.")
        n = jnp.asarray(noise[step], jnp.float32).reshape(self.latent_shape)
      else:
        n = jax.random.normal(key, self.latent_shape, jnp.float32)
      sde = jax.random.normal(jax.random.fold_in(key, 1), (cfg.ming_talker_cfm_steps, *self.latent_shape), jnp.float32)
      return (
          jnp.broadcast_to(n[None], (self.batch, *self.latent_shape)),
          jnp.broadcast_to(sde[:, None], (cfg.ming_talker_cfm_steps, self.batch, *self.latent_shape)),
      )

    latent, embed, stop, hidden, cache = self._prefill(
        self.params, self.init_cache, jnp.asarray(padded), jnp.int32(true_length), *step_noise(0)
    )
    latents, stops, hiddens = [latent], [float(stop[0])], [hidden]
    history_size = cfg.ming_talker_history_patch_size

    def feedback(sampled, sampled_embed, step):
      if forced_latents is None:
        return sampled, sampled_embed
      forced = jnp.broadcast_to(jnp.asarray(forced_latents[step], jnp.float32)[None], sampled.shape)
      return forced, self._embed_latent(self.params, forced)

    latent, embed = feedback(latent, embed, 0)
    history = jnp.concatenate([jnp.zeros((self.batch, history_size, self.latent_shape[1])), latent], axis=1)
    history = history[:, -history_size:]

    def should_stop():
      if forced_latents is not None:
        return len(latents) >= forced_latents.shape[0]
      # Reference: `if step > 3 and stop > 0.5: break` after appending the latent of `step`.
      return len(stops) - 1 >= cfg.ming_talker_min_steps and stops[-1] > cfg.ming_talker_stop_threshold

    while not should_stop() and len(latents) < self.max_steps:
      step = len(latents)
      position = jnp.int32(true_length + step - 1)
      latent, embed, stop, hidden, cache = self._step(self.params, cache, embed, position, history, *step_noise(step))
      latents.append(latent)
      stops.append(float(stop[0]))
      hiddens.append(hidden)
      latent, embed = feedback(latent, embed, step)
      history = jnp.concatenate([history, latent], axis=1)[:, -history_size:]
    if not should_stop():
      max_logging.log(f"Talker reached the step limit ({self.max_steps}) without the stop head firing.")
    return {
        "latents": np.asarray(jnp.concatenate(latents, axis=1)[0]),
        "stop_probs": np.asarray(stops, np.float32),
        "hidden": np.asarray(jnp.concatenate(hiddens, axis=1)[0], np.float32),
    }

  def latents_to_waveform(self, latents: np.ndarray, bucket: int = 64) -> np.ndarray:
    """`[N, latent]` -> `[N * patch * hop]` float32 waveform.

    The AudioVAE runs on latents edge-padded to a multiple of `bucket` (one compilation per bucket instead of
    per length). The backbone is causal and the ISTFT masks the padded frames, so the result is exact.
    """
    latents = np.asarray(latents, np.float32)
    n = latents.shape[0]
    padded = np.pad(latents, ((0, -n % bucket), (0, 0)), mode="edge")
    waveform = self._decode_audio(self.params, jnp.asarray(padded)[None], jnp.int32(n))
    num_samples = n * self.config.ming_vae_patch_size * self.config.ming_vae_hop_length
    return np.asarray(waveform[0, :num_samples], np.float32)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()

  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  generator = MingTalkerGenerator(config)

  language = config.ming_talker_language or None
  prompt = processor_ming_talker.build_tts_prompt(language)
  instruction = processor_ming_talker.build_ip_instruction(config.ming_talker_ip_name)
  frontend_path = os.environ.get("MING_TALKER_FRONTEND_PATH", "")
  if frontend_path:
    fragments = processor_ming_talker.HFTextFrontend(frontend_path, config.ming_talker_max_fragment_length).fragments(
        config.prompt, language or "en"
    )
  else:
    fragments = processor_ming_talker.split_text(config.prompt, config.ming_talker_max_fragment_length)
  if not fragments:
    raise ValueError("Empty prompt: nothing to synthesize.")

  sample_rate = config.ming_vae_sample_rate
  chunks = []
  for index, fragment in enumerate(fragments):
    input_ids = processor_ming_talker.talker_input_ids(tokenizer, fragment, prompt, instruction)
    start = time.time()
    result = generator.generate_latents(input_ids, seed=config.ming_talker_seed + index)
    generated = time.time()
    waveform = generator.latents_to_waveform(result["latents"])
    chunks.append(processor_ming_talker.trim_trailing_silence(waveform, sample_rate))
    max_logging.log(
        f"Fragment {index}: {fragment!r} -> {input_ids.shape[0]} prompt tokens, {result['latents'].shape[0]} latents,"
        f" {waveform.shape[0] / sample_rate:.2f} s audio (talker {generated - start:.1f} s,"
        f" AudioVAE {time.time() - generated:.1f} s)"
    )
  audio = np.concatenate(chunks)
  output_path = config.ming_talker_audio_output_path or "ming_talker_output.wav"
  write_wav(output_path, audio, sample_rate)
  max_logging.log(f"Wrote {audio.shape[0] / sample_rate:.2f} s of {sample_rate} Hz audio to {output_path}")


if __name__ == "__main__":
  app.run(main)
