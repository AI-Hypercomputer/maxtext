# Copyright 2023–2025 Google LLC
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

"""Tokenize Op used by Grain"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any
import grain.python as grain
import numpy as np
from MaxText import tokenizer


@dataclasses.dataclass
class TokenizerTransformBase:
  """Base class for tokenizer transforms with common functionality."""

  # pylint: disable=attribute-defined-outside-init
  add_bos: bool
  add_eos: bool
  tokenizer: tokenizer.SentencePieceTokenizerGrain | tokenizer.HFTokenizer

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()

  def _get_processor(self):
    if self._processor is None:
      with self._initialize_processor_lock:
        if self._processor is None:  # Ensures only one thread initializes SPP.
          self._processor = self.tokenizer
    return self._processor

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()


@dataclasses.dataclass
class TokenizeAndTrim(TokenizerTransformBase, grain.MapTransform):
  """Tokenize and trim features to sequence length."""
  # pylint: disable=attribute-defined-outside-init
  feature_names: str | Sequence[str]
  sequence_length: int | Sequence[int]

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.feature_names, str):
      self.feature_names = [self.feature_names]
    if isinstance(self.sequence_length, int):
      self.sequence_length = [self.sequence_length] * len(self.feature_names)

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Maps to each element."""
    processor = self._get_processor()
    for feature_name, sequence_length in zip(self.feature_names, self.sequence_length, strict=True):
      text = element[feature_name]
      token_ids = processor.encode(text)[:sequence_length]
      element[feature_name] = np.asarray(token_ids, dtype=np.int32)
    return element


@dataclasses.dataclass
class TokenizeAndChunk(TokenizerTransformBase, grain.experimental.FlatMapTransform):
  """Tokenize and chunk features into multiple examples of sequence length."""

  # pylint: disable=attribute-defined-outside-init
  feature_name: str
  sequence_length: int
  max_fan_out: int = 2048

  def flat_map(self, element: dict[str, Any]) -> list[dict[str, Any]]:
    processor = self._get_processor()
    text = element[self.feature_name]
    max_len = self.sequence_length

    token_ids = processor.encode(text)

    if not token_ids:
      return []

    output_elements = []
    for i in range(0, len(token_ids), max_len):
      chunk = np.asarray(token_ids[i : i + max_len], dtype=np.int32)
      new_element = {self.feature_name: chunk}
      output_elements.append(new_element)
    
    return output_elements