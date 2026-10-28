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

"""Multimodal (audio + text) preprocessing for inclusionAI Ming (BailingMoeV3 audio) models.

Mirrors `processing_bailing_moe_v3_audio.BailingMoeV3AudioProcessor`:
  1. Load audio with soundfile, resample to 16 kHz with torchaudio's default sinc/Hann
     interpolation (re-implemented in numpy), and average channels.
  2. whisper `log_mel_spectrogram` (128 mels, n_fft 400, hop 160) without 30 s padding.
  3. Chat-template the prompt and expand `<|audio_pad|>` to the number of audio tokens:
     per 3000-frame chunk, L -> conv2 (s2) -> 3x projector conv (s2).

The talker (text -> speech) frontend lives at the end of this file (see `TALKER_TTS_PROMPT`).
"""

from dataclasses import dataclass
import json
import math
import re

import numpy as np

from maxtext.multimodal import utils as mm_utils

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
CHUNK_SIZE = 3000  # mel frames per whisper chunk (30 s)
NUM_PROJECTOR_CONVS = 3

AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_PAD_ID = 157168
AUDIO_START_ID = 157169
AUDIO_END_ID = 157170
EOS_IDS = (156892, 156895)  # <|endoftext|>, <|role_end|>

SYSTEM_PREFIX = "<role>SYSTEM</role>detailed thinking off<|role_end|>"
USER_PREFIX = "<role>HUMAN</role>"
TURN_END = "<|role_end|>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n<think></think>"


@dataclass
class MingPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of the Ming audio preprocessor."""

  num_audios: int = 0
  audio_values: None | np.ndarray = None  # [1, T_mel, n_mels] float32
  audio_mask: None | np.ndarray = None  # [1, T_mel] int32
  audio_lengths: None | np.ndarray = None  # [num_audios] number of audio tokens per audio


# ---------------------------------------------------------------------------
# Audio loading / resampling / log-mel
# ---------------------------------------------------------------------------


def _sinc_resample_kernel(orig_freq: int, new_freq: int, lowpass_filter_width: int = 6, rolloff: float = 0.99):
  """numpy port of `torchaudio.functional.functional._get_sinc_resample_kernel` (sinc_interp_hann)."""
  gcd = math.gcd(int(orig_freq), int(new_freq))
  orig_freq = int(orig_freq) // gcd
  new_freq = int(new_freq) // gcd
  base_freq = min(orig_freq, new_freq) * rolloff
  width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
  idx = np.arange(-width, width + orig_freq, dtype=np.float64)[None, :] / orig_freq
  t = np.arange(0, -new_freq, -1, dtype=np.float64)[:, None] / new_freq + idx
  t = np.clip(t * base_freq, -lowpass_filter_width, lowpass_filter_width)
  window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
  scale = base_freq / orig_freq
  kernels = np.sinc(t)
  kernels = kernels * window * scale  # [new_freq, kernel_len]
  return kernels, width, orig_freq, new_freq


def resample(waveform: np.ndarray, orig_freq: int, new_freq: int = SAMPLE_RATE) -> np.ndarray:
  """`torchaudio.functional.resample` with default arguments, for a 1D or [C, N] waveform."""
  if orig_freq == new_freq:
    return waveform
  kernels, width, orig, new = _sinc_resample_kernel(orig_freq, new_freq)
  x = np.atleast_2d(np.asarray(waveform, dtype=np.float64))
  length = x.shape[-1]
  x = np.pad(x, ((0, 0), (width, width + orig)))
  num_frames = (x.shape[-1] - kernels.shape[1]) // orig + 1
  idx = np.arange(num_frames)[:, None] * orig + np.arange(kernels.shape[1])[None, :]
  frames = x[:, idx]  # [C, num_frames, kernel_len]
  out = np.einsum("cfk,nk->cfn", frames, kernels).reshape(x.shape[0], -1)
  target_length = int(math.ceil(new * length / orig))
  out = out[:, :target_length].astype(np.float32)
  return out[0] if np.ndim(waveform) == 1 else out


def load_audio(path: str) -> np.ndarray:
  """Loads `path` as mono 16 kHz float32, matching BailingMoeV3AudioProcessor._load_audio."""
  import soundfile as sf  # pylint: disable=import-outside-toplevel

  samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
  waveform = samples.T  # [C, N]
  if sample_rate != SAMPLE_RATE:
    waveform = resample(waveform, sample_rate, SAMPLE_RATE)
  return waveform.mean(axis=0).astype(np.float32)


def log_mel_spectrogram(waveform: np.ndarray, n_mels: int = NUM_MEL_BINS) -> np.ndarray:
  """whisper.log_mel_spectrogram (no padding); returns `[T_mel, n_mels]` float32."""
  mel_filters = mm_utils.mel_filter_bank(
      num_frequency_bins=1 + N_FFT // 2,
      num_mel_filters=n_mels,
      min_frequency=0.0,
      max_frequency=SAMPLE_RATE / 2,
      sampling_rate=SAMPLE_RATE,
      norm="slaney",
      mel_scale="slaney",
  )
  log_spec = mm_utils.spectrogram(
      waveform.astype(np.float32),
      mm_utils.window_function(N_FFT, "hann"),
      frame_length=N_FFT,
      hop_length=HOP_LENGTH,
      power=2.0,
      dither=0.0,
      mel_filters=mel_filters,
      log_mel="log10",
  )
  log_spec = log_spec[:, :-1]
  log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec.T.astype(np.float32)


def get_num_audio_tokens(mel_length: int, chunk_size: int = CHUNK_SIZE, num_convs: int = NUM_PROJECTOR_CONVS) -> int:
  """BailingMoeV3AudioProcessor._num_audio_tokens."""
  total = 0
  for offset in range(0, mel_length, chunk_size):
    length = min(chunk_size, mel_length - offset)
    length = (length + 2 - 3) // 2 + 1
    for _ in range(num_convs):
      length = (length + 2 * 1 - 3) // 2 + 1
    total += length
  return total


def preprocess_audio_ming(audio_path: str) -> tuple[np.ndarray, np.ndarray, int]:
  """Returns (audio_values [1, T_mel, 128], audio_mask [1, T_mel], num_audio_tokens)."""
  mel = log_mel_spectrogram(load_audio(audio_path))
  audio_values = mel[None]
  audio_mask = np.ones(audio_values.shape[:2], dtype=np.int32)
  return audio_values, audio_mask, get_num_audio_tokens(mel.shape[0])


def preprocess_mm_data_ming(config) -> MingPreprocessorOutput:
  """Preprocesses the (single) audio file in `config.audio_path`."""
  out = MingPreprocessorOutput()
  if not config.audio_path:
    return out
  paths = [p for p in config.audio_path.split(",") if p]
  if len(paths) != 1:
    raise ValueError(f"Ming currently supports exactly one audio input, got {len(paths)}.")
  audio_values, audio_mask, num_tokens = preprocess_audio_ming(paths[0])
  out.num_audios = 1
  out.audio_values = audio_values
  out.audio_mask = audio_mask
  out.audio_lengths = np.asarray([num_tokens], dtype=np.int32)
  return out


# ---------------------------------------------------------------------------
# Prompt formatting / token expansion
# ---------------------------------------------------------------------------


def reformat_prompt_ming(prompt: str, audio_placeholder: str = "<|audio|>", num_audios: int = 0) -> str:
  """Applies the BailingMoeV3 chat template (single user turn, thinking off).

  Audio blocks replace `audio_placeholder` occurrences in the prompt, or are prepended to the
  user text (the order used by the HF example: audio first, then text).
  """
  audio_block = AUDIO_START_TOKEN + AUDIO_PAD_TOKEN + AUDIO_END_TOKEN
  if audio_placeholder and audio_placeholder in prompt:
    content = prompt.replace(audio_placeholder, audio_block)
  else:
    content = audio_block * num_audios + prompt
  return SYSTEM_PREFIX + USER_PREFIX + content + TURN_END + ASSISTANT_PREFIX


def get_audio_offsets_ming(processor_output) -> int:
  """Number of extra tokens added when each `<|audio_pad|>` is expanded."""
  lengths = getattr(processor_output, "audio_lengths", None)
  if lengths is None:
    return 0
  return int(np.sum(lengths)) - len(lengths)


def add_extra_tokens_for_ming(tokens: np.ndarray, processor_output) -> np.ndarray:
  """Expands the i-th `<|audio_pad|>` token into `audio_lengths[i]` copies."""
  lengths = getattr(processor_output, "audio_lengths", None)
  if lengths is None:
    return tokens
  tokens = np.asarray(tokens)
  if tokens.ndim != 1:
    raise ValueError(f"add_extra_tokens_for_ming expects a 1D array of tokens, got shape {tokens.shape}")
  new_tokens = []
  audio_idx = 0
  for tok in tokens.tolist():
    if tok == AUDIO_PAD_ID and audio_idx < len(lengths):
      new_tokens.extend([AUDIO_PAD_ID] * int(lengths[audio_idx]))
      audio_idx += 1
    else:
      new_tokens.append(tok)
  if audio_idx != len(lengths):
    raise ValueError(f"Found {audio_idx} {AUDIO_PAD_TOKEN} tokens for {len(lengths)} audios.")
  return np.asarray(new_tokens, dtype=tokens.dtype)


def get_dummy_audio_shape_for_init_ming(config):
  """(batch, T_mel, n_mels) for one full whisper chunk."""
  return (config.micro_batch_size_to_train_on, config.ming_audio_chunk_size, config.num_mel_bins_for_audio)


# -----------------------------------------
# Talker (text -> speech) frontend
# -----------------------------------------
#
# Prompt protocol of the publisher's `talker/inference.py` + `talker/tts_prompt.py` for the default "IP" task
# (named voice, no reference audio); every piece is tokenized separately and concatenated like the reference:
#
#   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt} Text input:\n{text}
#   <|im_end|>\n<|im_start|>assistant\n{instruction}<|im_end|><audio>
#
# The publisher's text normalization (pynini FST) is not ported: text is synthesized as written.

TALKER_TTS_PROMPT = "Please generate speech based on the following description.\n"
# `append_optional_target_language`: zh/en (the training languages) add no "Target language" line.
_TALKER_LANGUAGE_NAMES = {
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "yue": "Chinese",
    "zh-yue": "Chinese",
    "cantonese": "Chinese",
}
_TALKER_NO_TARGET_LANGUAGE_IDS = {"", "zh", "zh-cn", "chinese", "en", "en-us", "en-gb", "english"}


def build_tts_prompt(language: str = "") -> str:
  """`append_optional_target_language(TALKER_TTS_PROMPT, language)`."""
  key = language.strip().lower().replace("_", "-")
  if key in _TALKER_NO_TARGET_LANGUAGE_IDS:
    return TALKER_TTS_PROMPT
  return TALKER_TTS_PROMPT + f"Target language: {_TALKER_LANGUAGE_NAMES.get(key, language.strip())}.\n"


def build_ip_instruction(ip_name: str) -> str:
  """`build_task_instruction("IP", ip=ip_name)`: the assistant-side JSON caption (zh schema)."""
  item = {
      "序号": 1,
      "说话人": "speaker_1",
      "方言": None,
      "风格": None,
      "语速": None,
      "基频": None,
      "音量": None,
      "情感": None,
  }
  item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
  item["IP"] = ip_name
  return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def talker_input_ids(tokenizer, text: str, prompt: str, instruction: str) -> np.ndarray:
  """Talker prompt token ids for one text fragment (`MingTalker.input_embeddings` without speakers)."""
  pieces = (
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
      "<|im_start|>user\n",
      prompt,
      " Text input:\n",
      text,
      "<|im_end|>\n",
      "<|im_start|>assistant\n",
      instruction,
      "<|im_end|>",
      "<audio>",
  )
  return np.asarray([t for piece in pieces for t in tokenizer.encode(piece, add_special_tokens=False)], np.int32)


# Dots that do not end a sentence (the reference `front/text_segment_cut.py` protects decimals and dotted
# capitals such as "U.S.A."): decimals ("3.14"), single-letter abbreviations ("e.g.", "J. K."), common titles.
_TALKER_DOT = "\ue000"  # private-use placeholder, restored after splitting
_TALKER_PROTECTED_DOTS = re.compile(
    r"(?<=\d)\.(?=\d)"  # decimals
    # Initials / single-letter abbreviations and titles (fixed-width look-behinds), unless the dot ends the text.
    r"|(?:(?<=\b[A-Za-z])|(?<=\b(?:Mr|Ms|Dr|St|Jr|Sr|vs))|(?<=\bMrs)|(?<=\bProf))\.(?=\s*\S)"
)
# A sentence ends after a run of terminators ("...", "?!"), so ellipses stay with their sentence.
_TALKER_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])(?![.!?。！？…])\s*")
_TALKER_CLAUSE_END = re.compile(r"(?<=[,，;；、:：])\s*")
_TALKER_CJK = re.compile(r"[\u4e00-\u9fff]")


def _join_text(left: str, right: str) -> str:
  """Joins two text pieces with a space unless the left one ends in CJK text or punctuation."""
  if not left:
    return right
  separator = "" if re.search(r"[\u4e00-\u9fff。！？，；、：]$", left) else " "
  return left + separator + right


def _pack(pieces: list[str], max_length: int) -> list[str]:
  """Greedily concatenates consecutive pieces while the result stays within `max_length` characters."""
  fragments = []
  for piece in pieces:
    if fragments and len(_join_text(fragments[-1], piece)) <= max_length:
      fragments[-1] = _join_text(fragments[-1], piece)
    else:
      fragments.append(piece)
  return fragments


def _split_long(sentence: str, max_length: int) -> list[str]:
  """Splits a sentence longer than `max_length`: at clause punctuation, then words, then characters."""
  if len(sentence) <= max_length:
    return [sentence]
  pieces = []
  for clause in (c.strip() for c in _TALKER_CLAUSE_END.split(sentence) if c.strip()):
    if len(clause) <= max_length:
      pieces.append(clause)
      continue
    for word in clause.split(" "):
      # A single over-long "word" (e.g. unspaced CJK text) is cut into max_length characters.
      pieces.extend(word[i : i + max_length] for i in range(0, len(word), max_length))
  return _pack(pieces, max_length)


def split_text(text: str, max_length: int = 50) -> list[str]:
  """Splits text into fragments of at most `max_length` characters, synthesized one at a time.

  Sentences end after a run of `.!?。！？…` (not inside decimals, initials, abbreviations or ellipses); sentences
  longer than `max_length` fall back to clause punctuation, then whitespace, then characters. Consecutive pieces
  are packed up to `max_length`, and the final fragment gets a sentence terminator like the reference.
  """
  text = _TALKER_PROTECTED_DOTS.sub(_TALKER_DOT, " ".join(text.split()))
  sentences = [s.strip() for s in _TALKER_SENTENCE_END.split(text) if s.strip()]
  pieces = [piece for sentence in sentences for piece in _split_long(sentence, max_length)]
  fragments = [f.replace(_TALKER_DOT, ".") for f in _pack(pieces, max_length)]
  if fragments and not re.search(r"[.!?。！？…]$", fragments[-1]):
    fragments[-1] += "。" if _TALKER_CJK.search(fragments[-1]) else "."
  return fragments


def trim_trailing_silence(speech: np.ndarray, sample_rate: int, threshold: float = 1e-3, keep: float = 0.3):
  """`MingTalker.silence_holder(..., last_chunk=True)` for one complete fragment waveform `[T]`."""
  frame, keep_len = int(sample_rate * 0.1), int(keep * sample_rate)
  framed = speech[: speech.shape[-1] // frame * frame]
  loud = np.nonzero(np.abs(framed.reshape(-1, frame)).mean(axis=-1) > threshold)[0]
  if loud.size == 0:
    return speech[:keep_len]
  # The reference drops the sub-frame remainder of a fragment that ends with speech.
  return framed[: (int(loud[-1]) + 1) * frame + keep_len]
