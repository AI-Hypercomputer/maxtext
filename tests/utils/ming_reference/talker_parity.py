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

"""End-to-end parity check of the MaxText Ming talker against HF goldens (`talker_golden.py`).

For every golden fragment: checks the MaxText text frontend reproduces the HF prompt token ids, replays the
HF CFM noise through `MingTalkerGenerator`, and compares per-step LLM hidden states, latents, stop
probabilities and the AudioVAE waveform. Writes the MaxText waveform next to the golden for listening.

Example (TPU, fp32):
  python3 -m tests.utils.ming_reference.talker_parity src/maxtext/configs/base.yml \
    model_name=ming-3.0-audio-tiny-talker tokenizer_path=<snapshot>/talker/llm \
    load_parameters_path=/dev/shm/hengtaoguo/ming/talker_ckpt_fp32/0/items scan_layers=false \
    per_device_batch_size=1 max_prefill_predict_length=256 max_target_length=1280 attention=dot_product \
    weight_dtype=float32 dtype=float32 matmul_precision=highest skip_jax_distributed_system=true \
    --golden=/dev/shm/hengtaoguo/ming/talker_golden.pkl
"""

import os
import pickle
import sys

import jax
import numpy as np

from maxtext.configs import pyconfig
from maxtext.inference.ming_talker_decode import MingTalkerGenerator, write_wav
from maxtext.multimodal import processor_ming_talker


def _rel_err(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main(argv):
  golden_path = None
  rest = []
  for arg in argv:
    if arg.startswith("--golden="):
      golden_path = arg.split("=", 1)[1]
    else:
      rest.append(arg)
  if golden_path is None:
    raise ValueError("--golden=<talker_golden.pkl> is required")
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  config = pyconfig.initialize(rest)
  with open(golden_path, "rb") as f:
    golden = pickle.load(f)

  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  prompt = processor_ming_talker.build_tts_prompt(None)
  instruction = processor_ming_talker.build_ip_instruction(config.ming_talker_ip_name)
  assert prompt == golden["prompt"], (prompt, golden["prompt"])
  assert instruction == golden["instruction"], (instruction, golden["instruction"])

  ok = True
  frontend_path = os.environ.get("MING_TALKER_FRONTEND_PATH", "")
  if frontend_path:
    frontend = processor_ming_talker.HFTextFrontend(frontend_path, config.ming_talker_max_fragment_length)
    fragments = frontend.fragments(golden["text"], "en")
    frontend_match = fragments == [frag["text"] for frag in golden["fragments"]]
    print(f"HF text frontend fragments match: {frontend_match} {fragments}")
    ok &= frontend_match

  generator = MingTalkerGenerator(config)
  audio = []
  for index, frag in enumerate(golden["fragments"]):
    ids = processor_ming_talker.talker_input_ids(tokenizer, frag["text"], prompt, instruction)
    ids_match = np.array_equal(ids, frag["input_ids"])
    ok &= ids_match
    noise = frag["noise"][:, 0]  # [steps, P, latent]
    result = generator.generate_latents(frag["input_ids"], noise=noise)
    n_mt, n_hf = result["latents"].shape[0], frag["latents"].shape[1]
    n = min(n_mt, n_hf)
    hidden_err = [_rel_err(result["hidden"][i], frag["hidden"][0, i]) for i in range(n)]
    latent_err = [_rel_err(result["latents"][i], frag["latents"][0, i]) for i in range(n)]
    stop_diff = np.abs(result["stop_probs"][:n] - frag["stop_probs"][:n]).max()
    # Teacher forcing: condition every step on the HF latents to measure the per-step error without AR drift.
    forced = generator.generate_latents(frag["input_ids"], noise=noise, forced_latents=frag["latents"][0][:, None])
    forced_err = [_rel_err(forced["latents"][i], frag["latents"][0, i]) for i in range(n_hf)]
    forced_hidden_err = [_rel_err(forced["hidden"][i], frag["hidden"][0, i]) for i in range(n_hf)]
    # Decode the HF latents too, isolating the AudioVAE from autoregressive drift.
    wav_from_hf_latents = generator.latents_to_waveform(frag["latents"][0])
    wav = generator.latents_to_waveform(result["latents"])
    vae_err = _rel_err(wav_from_hf_latents, frag["waveform"])
    m = min(wav.shape[0], frag["waveform"].shape[0])
    wav_corr = float(np.corrcoef(wav[:m], frag["waveform"][:m])[0, 1])
    print(f"fragment {index}: {frag['text']!r}")
    print(f"  prompt ids match HF: {ids_match} ({ids.shape[0]} tokens)")
    print(f"  latents: MaxText {n_mt}, HF {n_hf}")
    print(f"  hidden rel err: first {hidden_err[0]:.2e}, max {max(hidden_err):.2e}")
    print(f"  latent rel err: first {latent_err[0]:.2e}, max {max(latent_err):.2e}")
    print(f"  stop prob max abs diff: {stop_diff:.2e}")
    print("  per-step latent rel err:", " ".join(f"{e:.1e}" for e in latent_err))
    print(f"  teacher-forced: hidden rel err max {max(forced_hidden_err):.2e}, latent rel err max {max(forced_err):.2e}")
    print(f"  AudioVAE on HF latents: waveform rel err {vae_err:.2e}")
    print(f"  end-to-end waveform correlation: {wav_corr:.6f}")
    # Free-running latents drift slowly (chaotic autoregression); the per-step (teacher-forced) error is the check.
    ok &= n_mt == n_hf and max(forced_err) < 1e-3 and max(forced_hidden_err) < 1e-3 and vae_err < 1e-3
    audio.append(processor_ming_talker.trim_trailing_silence(wav, golden["sample_rate"]))
  out = os.path.splitext(golden_path)[0] + "_maxtext.wav"
  write_wav(out, np.concatenate(audio), golden["sample_rate"])
  print(f"Wrote {out}")
  print("PARITY", "PASS" if ok else "FAIL")
  if not ok:
    sys.exit(1)


if __name__ == "__main__":
  main(sys.argv)
