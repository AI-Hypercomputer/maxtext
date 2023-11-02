import abc
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
import threading
from typing import Any
from sentencepiece import SentencePieceProcessor
import grain.python as grain

class AbstractTokenizeAndSplit(grain.MapTransform):
  """Tokenize and split text features.

  The split of the tokenized features will replace the text features.

  This transform makes 2 assumptions:
  - Records are flat dictionaries with 1 or more text features.
  - It follows a DataSourceWithSplitInfo which should produce elements as:
    (example, (split_index, expected_split_count))

  The transform will produces None if the actual example doesn't have the
  corresponding split.
  """

  def __init__(
      self,
      feature_names: str | Sequence[str],
      sequence_length: int | Sequence[int],
  ):
    """Creates a new TokenizeAndSplit transform.

    Args:
      feature_names: One or multiple feature names that contain text.
      sequence_length: One or multiple sequence lengths to use for the text
        features. Output features will have [0, sequence_length] tokens.
    """
    super().__init__()
    if isinstance(feature_names, str):
      feature_names = [feature_names]
    if isinstance(sequence_length, int):
      sequence_length = [sequence_length] * len(feature_names)
    elif len(sequence_length) != len(feature_names):
      raise ValueError(
          f"Number of features and sequence lengths mismatch: {feature_names=},"
          f" {sequence_length=}"
      )
    self._feature_names = feature_names
    self._sequence_length = sequence_length
    self._stats = {
        "empty_splits": 0,
        "discarded_splits": 0,
    }

  def map(
      self, features: tuple[dict[str, Any], tuple[int, int]]
  ) -> dict[str, Any] | None:
    features, (split_index, expected_split_count) = features
    actual_split_count = 0
    for feature_name, sequence_length in zip(
        self._feature_names, self._sequence_length, strict=True
    ):
      text = features[feature_name]
      token_ids = self._tokenize(text)
      start = split_index * sequence_length
      end = (split_index + 1) * sequence_length
      if start >= len(token_ids):
        self._stats["empty_splits"] += 1
        return None
      actual_split_count = max(
          actual_split_count, int(math.ceil(len(token_ids) / sequence_length))
      )
      features[feature_name] = np.asarray(token_ids[start:end])
    if split_index == 0 and actual_split_count > expected_split_count:
      self._stats["discarded_splits"] += (
          actual_split_count - expected_split_count
      )
    return features

  def get_stats(self) -> Mapping[str, int]:
    return copy.copy(self._stats)

  @abc.abstractmethod
  def _tokenize(self, text: str) -> Sequence[int]:
    """Tokenizes the text."""


class SentencePieceTokenizeAndSplit(AbstractTokenizeAndSplit):
  """Tokenize and split text features using a Gemini tokenizer."""

  def __init__(
      self,
      feature_names: str | Sequence[str],
      sequence_length: int | Sequence[int],
      sentencepiece_model_path: str,
  ):
    super().__init__(feature_names, sequence_length)
    self._sentencepiece_model_path = sentencepiece_model_path
    self._initialize_processor_lock = threading.Lock()
    self._tokenizer = None

  def _tokenize(self, text: str) -> Sequence[int]:
    if self._tokenizer is None:
      with self._initialize_processor_lock:
        if self._tokenizer is None:
          self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
          self._tokenizer.Load(filename=self._sentencepiece_model_path)
    return self._tokenizer.EncodeAsIds(text)

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_tokenizer"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._tokenizer = None
    self._initialize_processor_lock = threading.Lock()


@dataclasses.dataclass
class TokenizeAndPad(grain.MapTransform):
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
          # self._processor.Load(filename=self.model_path)
    for feature_name, sequence_length in zip(
        self.feature_names, self.sequence_length, strict=True
    ):
      text = features[feature_name]
      token_ids = self._processor.EncodeAsIds(text)
      token_ids = token_ids[:sequence_length]
      token_ids = token_ids + [self._processor.pad_id()] * (
          sequence_length - len(token_ids)
      )
      features[feature_name] = token_ids
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
