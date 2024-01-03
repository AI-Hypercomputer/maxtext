import abc
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
import threading
from typing import Any
from sentencepiece import SentencePieceProcessor
import grain.python as grain
import numpy as np

@dataclasses.dataclass
class Tokenize(grain.MapTransform):
  """Tokenize, truncate and pad features to sequence length."""

  feature_names: str | Sequence[str]
  sequence_length: int | Sequence[int]
  model_path: str

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()
    if isinstance(self.feature_names, str):
      self.feature_names = [self.feature_names]
    if isinstance(self.sequence_length, int):
      self.sequence_length = [self.sequence_length] * len(self.feature_names)

  def map(self, features: dict[str, Any]) -> dict[str, Any]:
    if self._processor is None:
      with self._initialize_processor_lock:
        if self._processor is None:  # Ensures only one thread initializes SPP.
          self._processor = SentencePieceProcessor(self.model_path)
    for feature_name, sequence_length in zip(
        self.feature_names, self.sequence_length, strict=True
    ):
      text = features[feature_name]
      token_ids = self._processor.EncodeAsIds(text)
      token_ids = token_ids[:sequence_length]
      features[feature_name] = np.asarray(token_ids, dtype=np.int32)
    return features

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()
