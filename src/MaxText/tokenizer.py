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

"""Provides op for tokenizing a dataset."""

from typing import Iterable, Literal, Sequence, Collection
from pathlib import Path
import tensorflow as tf
import tensorflow_text as tftxt
from MaxText import max_logging
import transformers
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from sentencepiece import SentencePieceProcessor


Features = dict[str, tf.Tensor]


class TikTokenTokenizer:
  """
  Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
  """

  special_tokens: dict[str, int]

  num_reserved_special_tokens = 256

  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # pylint: disable=line-too-long

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    """
    Initializes the Tokenizer with a Tiktoken model.

    Args:
        model_path (str): The path to the Tiktoken model file.
    """

    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
    self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    self.model = tiktoken.Encoding(
        name=Path(model_path).name,
        pat_str=self.pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )
    self.eos = add_eos
    self.bos = add_bos
    max_logging.log(f"Reloaded tiktoken model from {model_path}")

    self.n_words: int = self.model.n_vocab
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.pad_id: int = -1
    self.stop_tokens = {
        self.special_tokens["<|end_of_text|>"],
        self.special_tokens["<|eot_id|>"],
    }
    max_logging.log(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

  def encode(
      self,
      s: str,
      *,
      allowed_special: Literal["all"] | Collection[str] = (),
      disallowed_special: Literal["all"] | Collection[str] = (),
  ) -> list[int]:
    """
    Encodes a string into a list of token IDs.

    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_tokens (`"all"|set[str]`): allowed special tokens in string
        disallowed_tokens (`"all"|set[str]`): special tokens that raise an error when in string

    Returns:
        list[int]: A list of token IDs.

    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:

    * Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    * Setting `allowed_special` to "all" will treat all text corresponding
      to special tokens to be encoded as special tokens.
    """
    assert isinstance(s, str)

    # The tiktoken tokenizer can handle <=400k chars without
    # pyo3_runtime.PanicException.
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000

    # https://github.com/openai/tiktoken/issues/195
    # Here we iterate over subsequences and split if we exceed the limit
    # of max consecutive non-whitespace or whitespace characters.
    MAX_NO_WHITESPACES_CHARS = 25_000

    substrs = (
        substr
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
        for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
        )
    )
    t: list[int] = []
    for substr in substrs:
      t.extend(
          self.model.encode(
              substr,
              allowed_special=set(allowed_special),
              disallowed_special=disallowed_special,
          )
      )
    if self.bos:
      t.insert(0, self.bos_id)
    if self.eos:
      t.append(self.eos_id)
    return t

  def decode(self, t) -> str:
    """
    Decodes a list of token IDs into a string.

    Args:
        t (list[int]): The list of token IDs to be decoded.

    Returns:
        str: The decoded string.
    """
    # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
    return self.model.decode(t)

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int):
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i, _ in enumerate(s):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]


class SentencePieceTokenizer:
  """
  Tokenizing and encoding/decoding text using the Sentencepiece tokenizer loaded with tensorflow_text
  """

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    max_logging.log(f"Tokenizer path: {model_path}")
    with tf.io.gfile.GFile(model_path, "rb") as model_fp:
      sp_model = model_fp.read()
    self.sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=False)
    self.pad_id = self.sp_tokenizer.string_to_id("<pad>")
    self.unk_id = self.sp_tokenizer.string_to_id("<unk>")

  def encode(self, s: str) -> list[int]:
    return self.sp_tokenizer.tokenize(s)

  def decode(self, t: Sequence[int]) -> str:
    return self.sp_tokenizer.detokenize(t)


class SentencePieceTokenizerGrain:
  """
  Tokenizing and encoding/decoding text using the Sentencepiece tokenizer loaded with sentencepiece
  """

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    max_logging.log(f"Loading sentencepiece tokenizer: {model_path}")
    self._tokenizer_model = SentencePieceProcessor()
    self._tokenizer_model.Load(model_path)
    self.pad_id = self._tokenizer_model.pad_id()
    self.unk_id = self._tokenizer_model.unk_id()
    self.bos_id = self._tokenizer_model.bos_id()
    self.eos_id = self._tokenizer_model.eos_id()
    self.add_bos = add_bos
    self.add_eos = add_eos

  def encode(self, s: str) -> list[int]:
    token_ids = self._tokenizer_model.EncodeAsIds(s)
    if self.add_bos:
      token_ids = [self.bos_id] + token_ids
    if self.add_eos:
      token_ids += [self.eos_id]
    return token_ids

  def decode(self, t: Sequence[int]) -> str:
    return self._tokenizer_model.DecodeIds(t)


class HFTokenizer:
  """
  Tokenizing using huggingface tokenizer
  """

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool, hf_access_token: str):
    max_logging.log(f"Loading HF tokenizer: {model_path}")
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        token=hf_access_token,
    )
    self.pad_id = self.tokenizer.pad_token_id
    self.unk_id = self.tokenizer.unk_token_id
    self.bos_id = self.tokenizer.bos_token_id
    self.eos_id = self.tokenizer.eos_token_id

  def encode(self, s: str) -> list[int]:
    return self.tokenizer.encode(s)

  def decode(self, t: Sequence[int]) -> str:
    return self.tokenizer.decode(t)


def build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token, dataset_type):
  """Loads the tokenizer at `tokenizer_path`"""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  if tokenizer_type == "tiktoken":
    assert "tiktoken" in tokenizer_path, f"Invalid tokenizer type: {tokenizer_type} chosen for {tokenizer_path}"
    return TikTokenTokenizer(tokenizer_path, add_bos, add_eos)
  elif tokenizer_type == "huggingface":
    return HFTokenizer(tokenizer_path, add_bos, add_eos, hf_access_token)
  elif tokenizer_type == "sentencepiece":
    if dataset_type == "tfds":
      return SentencePieceTokenizer(tokenizer_path, add_bos, add_eos)
    else:
      return SentencePieceTokenizerGrain(tokenizer_path, add_bos, add_eos)
  else:
    raise ValueError(f"Invalid tokenizer_type:{tokenizer_type} chosen in config")


def TokenizeOp(tokenizer, features: Features, data_keys: Iterable[str] = ("inputs", "targets")) -> Features:
  """Op for tokenization"""

  def _process_string(string_tensor):
    # Extract string value and decode it if necessary
    string_value = string_tensor.numpy().decode("utf-8")
    # encode and extract the tokenized integers
    modified_string = tokenizer.encode(string_value)
    return [modified_string]

  for k in data_keys:
    if isinstance(tokenizer, (TikTokenTokenizer, HFTokenizer)):
      features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
    elif isinstance(tokenizer, SentencePieceTokenizer):
      features[k] = tokenizer.encode(features[k])
  return features
