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
from maxtext.src.maxtext import tokenizer


@dataclasses.dataclass
class TokenizeAndTrim(grain.MapTransform):
  """Tokenize and trim features to sequence length."""

  # pylint: disable=attribute-defined-outside-init
  feature_names: str | Sequence[str]
  sequence_length: int | Sequence[int]
  add_bos: bool
  add_eos: bool
  tokenizer: tokenizer.SentencePieceTokenizerGrain | tokenizer.HFTokenizer

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()
    if isinstance(self.feature_names, str):
      self.feature_names = [self.feature_names]
    if isinstance(self.sequence_length, int):
      self.sequence_length = [self.sequence_length] * len(self.feature_names)

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Maps to each element."""
    if self._processor is None:
      with self._initialize_processor_lock:
        if self._processor is None:  # Ensures only one thread initializes SPP.
          self._processor = self.tokenizer
    for feature_name, sequence_length in zip(self.feature_names, self.sequence_length, strict=True):
      text = element[feature_name]
      token_ids = self._processor.encode(text)[:sequence_length]
      element[feature_name] = np.asarray(token_ids, dtype=np.int32)
    return element

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()
