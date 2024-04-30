"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Provides op for tokenizing a dataset."""

from typing import Any, Dict, Iterable

import dataclasses
from absl import logging
import tensorflow as tf
import tensorflow_text as tftxt

import max_logging


Features = Dict[str, tf.Tensor]


def _load_sentencepiece_tokenizer(tokenizer_path: str, add_bos: bool = False, add_eos: bool = True, reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  with tf.io.gfile.GFile(tokenizer_path, "rb") as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def load_tokenizer(tokenizer_path: str, add_bos=False, add_eos=True):
  """Loads the tokenizer at `tokenizer_path` or trains a one from `dataset`."""
  try:
    sp_tokenizer = _load_sentencepiece_tokenizer(tokenizer_path, add_bos, add_eos)
    return sp_tokenizer
  except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
    logging.info("SentencePiece vocab not found, Run train_tokenizer.py")
    return None


@dataclasses.dataclass
class TokenizeOp:
  sp_tokenizer: Any
  data_keys: Iterable[str] = ("inputs", "targets")

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.tokenize(features[k])
    return features
