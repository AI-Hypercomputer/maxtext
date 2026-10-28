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
"""

import math
from dataclasses import dataclass

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
