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

"""Text frontend of the Ming-3.0-audio-tiny talker (text -> speech).

Mirrors the prompt protocol of the publisher's `talker/inference.py` + `talker/tts_prompt.py` for the
default "IP" (named voice, no reference audio) task:

  <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{prompt} Text input:\\n{text}
  <|im_end|>\\n<|im_start|>assistant\\n{instruction}<|im_end|><audio>

Every piece is tokenized separately and concatenated, exactly like the reference.

Text normalization (number verbalization, pynini FST TN) is optional: pass the HF snapshot's `talker/`
directory as `frontend_path` to reuse the publisher's fragmenter and normalizer (needs `pynini`, `inflect`).
Otherwise a lightweight sentence splitter is used and the text is synthesized as written.
"""

from collections import OrderedDict
import json
import os
import re
import sys

import numpy as np

DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"

_TARGET_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
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
_NO_TARGET_LANGUAGE_IDS = {"zh", "zh-cn", "chinese", "en", "en-us", "en-gb", "english"}
_ZH_FIELDS = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
_BGM_FIELDS = ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")


def build_tts_prompt(language: str | None = None) -> str:
  """`append_optional_target_language(DEFAULT_TTS_PROMPT, language)`: zh/en/None keep the default prompt."""
  if language is None or not str(language).strip():
    return DEFAULT_TTS_PROMPT
  key = str(language).strip().lower().replace("_", "-")
  if key in _NO_TARGET_LANGUAGE_IDS:
    return DEFAULT_TTS_PROMPT
  return DEFAULT_TTS_PROMPT + f"Target language: {_TARGET_LANGUAGE_NAMES.get(key, str(language).strip())}.\n"


def build_ip_instruction(ip_name: str) -> str:
  """`build_task_instruction("IP", ip=ip_name)`: the assistant-side JSON caption (zh schema)."""
  values = (1, "speaker_1", None, None, None, None, None, None)
  item = OrderedDict(zip(_ZH_FIELDS, values))
  item["BGM"] = OrderedDict((key, None) for key in _BGM_FIELDS)
  item["IP"] = ip_name
  return json.dumps(OrderedDict((("audio_sequence", [item]),)), ensure_ascii=False)


def talker_input_ids(tokenizer, text: str, prompt: str, instruction: str | None) -> np.ndarray:
  """Talker prompt token ids for one text fragment (`MingTalker.input_embeddings` without speakers)."""
  encode = lambda s: tokenizer.encode(s, add_special_tokens=False)  # pylint: disable=unnecessary-lambda-assignment
  ids = (
      encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
      + encode("<|im_start|>user\n")
      + encode(prompt)
      + encode(" Text input:\n")
      + encode(text)
      + encode("<|im_end|>\n")
      + encode("<|im_start|>assistant\n")
      + (encode(instruction) + encode("<|im_end|>") if instruction else [])
      + encode("<audio>")
  )
  return np.asarray(ids, dtype=np.int32)


def split_text(text: str, max_length: int = 50) -> list[str]:
  """Lightweight fragmenter: sentence split on terminal punctuation, then pack sentences up to `max_length`."""
  text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
  if not text:
    return []
  sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*", text) if s.strip()]
  fragments, current = [], ""
  for sentence in sentences:
    if current and len(current) + 1 + len(sentence) > max_length:
      fragments.append(current)
      current = sentence
    elif current:
      separator = "" if re.search(r"[\u4e00-\u9fff。！？]$", current) else " "
      current = current + separator + sentence
    else:
      current = sentence
  if current:
    fragments.append(current)

  def add_period(fragment):
    if re.search(r"[.!?。！？,，;；]$", fragment):
      return fragment
    return fragment + ("。" if re.search(r"[\u4e00-\u9fff]", fragment) else ".")

  return [add_period(f) for f in fragments]


class HFTextFrontend:
  """The publisher's fragmenter + text normalization, imported from a local HF `talker/` directory."""

  def __init__(self, frontend_path: str, max_length: int = 50):
    frontend_path = os.path.abspath(frontend_path)
    parent, package = os.path.dirname(frontend_path), os.path.basename(frontend_path)
    if parent not in sys.path:
      sys.path.insert(0, parent)
    self.inference = __import__(f"{package}.inference", fromlist=["MingTalker"])
    self.normalizer = self.inference.TalkerTN()
    self.max_length = max_length

  def fragments(self, text: str, language: str | None) -> list[str]:
    """`MingTalker.synthesize` text handling: fragmenting, paralinguistic-tag protection and TN."""
    m = self.inference
    skip_tn = m.preserve_multilingual_text(language)
    out = []
    for fragment in m.MingTalker.text_fragments(None, text, self.max_length):
      protected = m.contains_paralinguistic_tag(fragment)
      fragment = m.protect_paralinguistic_tags(fragment) if protected else fragment
      if not skip_tn:
        if not m.is_chinese(fragment):
          fragment = m.normalize_numbers(fragment)
        fragment = m.normalize_text_kit(self.normalizer.normalize(fragment)).lstrip("，")
      if protected:
        fragment = m.restore_paralinguistic_tags(fragment)
      out.append(fragment)
    return out


def trim_trailing_silence(speech: np.ndarray, sample_rate: int, threshold: float = 1e-3, keep: float = 0.3):
  """`MingTalker.silence_holder(..., last_chunk=True)` for one complete fragment waveform `[T]`."""
  frame = int(sample_rate * 0.1)
  keep_len = int(keep * sample_rate)
  if speech.shape[-1] < frame:
    return speech[:keep_len]
  num_frames = (speech.shape[-1] - frame) // frame + 1
  framed = speech[: num_frames * frame]
  scores = np.abs(framed.reshape(num_frames, frame)).mean(axis=-1)
  loud = np.nonzero(scores > threshold)[0]
  if loud.size == 0:
    return speech[:keep_len]
  # The reference drops the sub-frame remainder of a fragment that ends with speech.
  return framed[: int(loud[-1]) * frame + frame + keep_len]
