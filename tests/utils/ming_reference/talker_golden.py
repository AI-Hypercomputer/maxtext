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

"""HF reference goldens for the Ming-3.0-audio-tiny talker (text -> speech).

Runs the publisher's native talker (`talker/inference.py`: Qwen2 LLM + CFM/DiT + Aggregator + AudioVAE)
on CPU and records everything needed for a MaxText parity check:

* the talker prompt token ids of every text fragment,
* the Gaussian noise used for each autoregressive step (drawn here with numpy so MaxText can replay it),
* per-step normed LLM hidden states, sampled latents and stop probabilities,
* the AudioVAE waveform (before and after the publisher's trailing-silence trimming).

The generation loop mirrors `MingTalker.generate_latents`; only the noise source differs. With the
default `sigma=0.25, temperature=0.0` the SDE term is zero, so the initial noise is the only randomness.

Extra dependencies (not in the MaxText venv): x-transformers, torchaudio, diffusers, onnxruntime, inflect,
pynini (text normalization) and fasttext. Install them into a side directory and pass `--extra_path`.

Example:
  python3 -m tests.utils.ming_reference.talker_golden \
    --talker_path=/dev/shm/hengtaoguo/ming/hf_full/talker --extra_path=/dev/shm/hengtaoguo/pyextra \
    --text="Hello, this is a test of the Ming talker." --output=/dev/shm/hengtaoguo/ming/talker_golden.npz
"""

import argparse
import os
import pickle
import sys
import wave

import numpy as np


def _import_talker(talker_path, extra_path):
  """Imports the publisher's `talker` package from a local HF snapshot directory."""
  if extra_path:
    sys.path.insert(0, extra_path)
  talker_path = os.path.abspath(talker_path)
  sys.path.insert(0, os.path.dirname(talker_path))
  package = os.path.basename(talker_path)
  module = __import__(f"{package}.inference", fromlist=["MingTalker"])
  return module


def write_wav(path, waveform, sample_rate):
  pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
  with wave.open(str(path), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes(pcm.tobytes())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--talker_path", required=True, help="HF snapshot `talker/` directory.")
  parser.add_argument("--extra_path", default=None, help="Side directory with the talker's extra dependencies.")
  parser.add_argument("--text", default="Hello, this is a test of the Ming talker.")
  parser.add_argument("--language", default="en")
  parser.add_argument("--ip_name", default="灵小柔")
  parser.add_argument("--max_steps", type=int, default=200)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--cfg", type=float, default=2.0)
  parser.add_argument("--sigma", type=float, default=0.25)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  parser.add_argument("--output", required=True, help="Output .pkl path; a .wav is written next to it.")
  args = parser.parse_args()

  inference = _import_talker(args.talker_path, args.extra_path)
  import torch  # pylint: disable=import-outside-toplevel

  torch.manual_seed(args.seed)
  dtype = getattr(torch, args.dtype)
  model = inference.MingTalker.from_pretrained(args.talker_path, dtype=dtype, device="cpu")
  tts = inference

  rng = np.random.default_rng(args.seed)
  language, text = tts.resolve_request_language(args.text, language=args.language, taskname="IP")
  prompt = tts.append_optional_target_language(tts.DEFAULT_TTS_PROMPT, language)
  instruction = tts.build_task_instruction("IP", ip=args.ip_name)
  skip_tn = tts.preserve_multilingual_text(tts.target_language_from_prompt(prompt))

  fragments = []
  with torch.inference_mode():
    for fragment in model.text_fragments(text):
      raw_fragment = fragment
      protected = tts.contains_paralinguistic_tag(fragment)
      fragment = tts.protect_paralinguistic_tags(fragment) if protected else fragment
      if not skip_tn:
        if not tts.is_chinese(fragment):
          fragment = tts.normalize_numbers(fragment)
        fragment = tts.normalize_text_kit(model.normalizer.normalize(fragment)).lstrip("，")
      if protected:
        fragment = tts.restore_paralinguistic_tags(fragment)

      embeddings = model.input_embeddings(fragment, prompt, instruction, None, None, None)
      encode = model.tokenizer.encode
      ids = (
          encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
          + encode("<|im_start|>user\n")
          + encode(prompt)
          + encode(" Text input:\n")
          + encode(fragment)
          + encode("<|im_end|>\n")
          + encode("<|im_start|>assistant\n")
          + encode(instruction)
          + encode("<|im_end|>")
          + encode("<audio>")
      )
      assert embeddings.shape[1] == len(ids)

      # Mirrors MingTalker.generate_latents with replayable noise.
      noise_all = rng.standard_normal((args.max_steps, 1, model.patch_size, model.latent_dim)).astype(np.float32)
      history = embeddings.new_zeros(1, model.history_patch_size, model.latent_dim)
      past = None
      latents, hiddens, stops = [], [], []
      for step in range(args.max_steps):
        output = model.model(inputs_embeds=embeddings, past_key_values=past, use_cache=True)
        past = output.past_key_values
        hidden = output.last_hidden_state[:, -1:]
        noise = torch.from_numpy(noise_all[step]).to(dtype)
        timesteps = inference.get_epss_timesteps(model.config.steps, "cpu", dtype)
        sde_noise = torch.zeros((model.config.steps, *noise.shape), dtype=dtype)
        latent = model.cfm.sample(
            hidden, history, noise, timesteps, hidden.new_tensor([args.cfg, args.sigma, args.temperature]), sde_noise
        )
        embeddings = model.aggregator(latent)
        latents.append(latent)
        hiddens.append(hidden.float().numpy())
        history = torch.cat([history, latent], dim=1)[:, -model.history_patch_size :]
        stop = model.stop_head(hidden[:, -1]).softmax(dim=-1)[0, 1]
        stops.append(float(stop))
        if step > 3 and stop.item() > 0.5:
          break
      latents = torch.cat(latents, dim=1)
      waveform, _, _ = model.audio_vae.decode(latents, use_cache=False, last_chunk=True)
      waveform = waveform.detach().float().cpu()
      trimmed = model.silence_holder(waveform, model.audio_vae.config.sample_rate)[0]
      fragments.append(
          {
              "raw_text": raw_fragment,
              "text": fragment,
              "input_ids": np.asarray(ids, np.int32),
              "noise": noise_all[: len(stops)],
              "hidden": np.concatenate(hiddens, axis=1),
              "latents": latents.float().numpy(),
              "stop_probs": np.asarray(stops, np.float32),
              "waveform": waveform.numpy().reshape(-1),
              "waveform_trimmed": trimmed.numpy().reshape(-1),
          }
      )
      print(f"fragment {raw_fragment!r} -> {fragment!r}: {len(ids)} prompt tokens, {len(stops)} latents")

  sample_rate = model.audio_vae.config.sample_rate
  golden = {
      "text": args.text,
      "prompt": prompt,
      "instruction": instruction,
      "sample_rate": sample_rate,
      "cfg": args.cfg,
      "sigma": args.sigma,
      "temperature": args.temperature,
      "fragments": fragments,
  }
  with open(args.output, "wb") as f:
    pickle.dump(golden, f)
  audio = np.concatenate([frag["waveform_trimmed"] for frag in fragments])
  write_wav(os.path.splitext(args.output)[0] + ".wav", audio, sample_rate)
  print(f"Saved {args.output} ({audio.shape[0] / sample_rate:.2f} s audio)")


if __name__ == "__main__":
  main()
