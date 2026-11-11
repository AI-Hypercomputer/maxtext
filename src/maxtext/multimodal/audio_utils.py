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

"""Utilities for audio output independent of model architecture."""

import wave

import numpy as np


def write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
  """Writes a mono waveform in [-1, 1] as little-endian 16-bit PCM WAV."""
  pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
  with wave.open(str(path), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes(pcm.tobytes())
