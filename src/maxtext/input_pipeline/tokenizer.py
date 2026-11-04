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

import json
import posixpath
from typing import Literal, Sequence, Collection
from pathlib import Path
from maxtext.utils import max_logging
import transformers
import tiktoken
from tiktoken.load import load_tiktoken_bpe, read_file
from sentencepiece import SentencePieceProcessor


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

    try:
      mergeable_ranks = load_tiktoken_bpe(model_path)
    except Exception as e:
      raise ValueError(f"Failed to load tiktoken tokenizer from {model_path}: {e}") from e
    num_base_tokens = len(mergeable_ranks)
    self.special_tokens = self._build_special_tokens(model_path, num_base_tokens)
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
    self._set_special_token_ids()
    max_logging.log(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

  def _build_special_tokens(self, model_path: str, num_base_tokens: int) -> dict[str, int]:
    """Returns the special-token -> id mapping (Llama-3 layout). Subclasses override for other vocabularies."""
    del model_path
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
    return {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

  def _set_special_token_ids(self):
    """Sets bos_id / eos_id / pad_id / stop_tokens from `self.special_tokens`."""
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.pad_id: int = -1
    self.stop_tokens = {
        self.special_tokens["<|end_of_text|>"],
        self.special_tokens["<|eot_id|>"],
    }

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
        allowed_tokens ("all"|set[str]): allowed special tokens in string
        disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

    Returns:
        list[int]: A list of token IDs.

    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:
    - Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    - Setting `allowed_special` to "all" will treat all text corresponding
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
  Tokenizing and encoding/decoding text using the native sentencepiece library.
  Supports both local and GCS (gs://) model paths.
  """

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    max_logging.log(f"Loading sentencepiece tokenizer: {model_path}")
    self._tokenizer_model = SentencePieceProcessor()
    try:
      if model_path.startswith("gs://"):
        from maxtext.utils.gcs_utils import read_bytes_from_gcs  # pylint: disable=import-outside-toplevel

        model_proto = read_bytes_from_gcs(model_path)
        self._tokenizer_model.LoadFromSerializedProto(model_proto)
      else:
        self._tokenizer_model.Load(model_path)
    except Exception as e:
      raise ValueError(f"Failed to load sentencepiece tokenizer from {model_path}: {e}") from e
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
    try:
      self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_path,
          add_bos_token=add_bos,
          add_eos_token=add_eos,
          token=hf_access_token,
      )
    except Exception as e:
      raise ValueError(f"Failed to load Hugging Face tokenizer from {model_path}: {e}") from e
    self.pad_id = self.tokenizer.pad_token_id
    self.unk_id = self.tokenizer.unk_token_id
    self.bos_id = self.tokenizer.bos_token_id
    self.eos_id = self.tokenizer.eos_token_id

  def encode(self, s: str) -> list[int]:
    return self.tokenizer.encode(s)

  def decode(self, t: Sequence[int]) -> str:
    return self.tokenizer.decode(t)


# Kimi-K3 pre-tokenizer regex (Han runs split from other letters); copied from the HF reference
# tokenization_kimi.py `TikTokenTokenizer.pat_str`.
KIMI_K3_TIKTOKEN_PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        (
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"""
            r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
        ),
        (
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"""
            r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
        ),
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

# Fallback special-token table (offsets from len(mergeable_ranks)), used only when tokenizer_config.json is
# missing. Mirrors `added_tokens_decoder` in the moonshotai Kimi-K3 HF checkpoint's tokenizer_config.json.
KIMI_K3_DEFAULT_SPECIAL_TOKEN_OFFSETS = {
    0: "[BOS]",
    1: "[EOS]",
    2: "<|end_of_msg|>",
    3: "<|open|>",
    4: "<|close|>",
    5: "<|sep|>",
    6: "[start_header_id]",
    7: "[end_header_id]",
    9: "[EOT]",
    18: "<|media_begin|>",
    19: "<|media_content|>",
    20: "<|media_end|>",
    21: "<|media_pad|>",
    65: "<osagent_mode>",
    254: "[UNK]",
    255: "[PAD]",
}


def is_kimi_tiktoken_path(tokenizer_path: str) -> bool:
  """Convenience heuristic: True if the tokenizer file or its parent directory name mentions "kimi"."""
  path = str(tokenizer_path).rstrip("/")
  return "kimi" in posixpath.basename(path).lower() or "kimi" in posixpath.basename(posixpath.dirname(path)).lower()


class KimiTikTokenTokenizer(TikTokenTokenizer):
  """Tiktoken tokenizer for Kimi-K3: Han-aware regex and Kimi special tokens.

  Special tokens are read from `tokenizer_config.json` (`added_tokens_decoder`) next to the tiktoken model file,
  like the HF reference tokenization_kimi.py; unnamed reserved slots are `<|reserved_token_{id}|>`. Local and
  remote (gs://, via blobfile) paths are supported. If the config is missing or unreadable a warning is logged and
  KIMI_K3_DEFAULT_SPECIAL_TOKEN_OFFSETS is used. Unlike Llama-3 tiktoken, pad_id is a real token id ([PAD]).
  """

  pat_str = KIMI_K3_TIKTOKEN_PAT_STR

  def _build_special_tokens(self, model_path: str, num_base_tokens: int) -> dict[str, int]:
    id_to_token = {num_base_tokens + off: tok for off, tok in KIMI_K3_DEFAULT_SPECIAL_TOKEN_OFFSETS.items()}
    cfg_path = posixpath.join(posixpath.dirname(str(model_path)), "tokenizer_config.json")
    try:
      tok_cfg = json.loads(read_file(cfg_path).decode("utf-8"))
      added = tok_cfg["added_tokens_decoder"]
      id_to_token = {int(k): (v["content"] if isinstance(v, dict) else str(v)) for k, v in added.items()}
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.warning(
          f"Could not load Kimi special tokens from {cfg_path} ({type(e).__name__}: {e}); "
          "falling back to the built-in Kimi-K3 special-token table."
      )
    return {
        id_to_token.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(num_base_tokens, num_base_tokens + self.num_reserved_special_tokens)
    }

  def _set_special_token_ids(self):
    self.bos_id: int = self.special_tokens["[BOS]"]
    self.eos_id: int = self.special_tokens["[EOS]"]
    self.pad_id: int = self.special_tokens["[PAD]"]
    self.unk_id: int = self.special_tokens["[UNK]"]
    self.stop_tokens = {
        self.eos_id,
        self.special_tokens["<|end_of_msg|>"],
        self.special_tokens["[EOT]"],
    }


def build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token):
  """Loads the tokenizer at `tokenizer_path`"""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  if tokenizer_type == "tiktoken":
    if is_kimi_tiktoken_path(tokenizer_path):
      return KimiTikTokenTokenizer(tokenizer_path, add_bos, add_eos)
    assert "tiktoken" in tokenizer_path, f"Invalid tokenizer type: {tokenizer_type} chosen for {tokenizer_path}"
    return TikTokenTokenizer(tokenizer_path, add_bos, add_eos)
  elif tokenizer_type == "huggingface":
    return HFTokenizer(tokenizer_path, add_bos, add_eos, hf_access_token)
  elif tokenizer_type == "sentencepiece":
    return SentencePieceTokenizer(tokenizer_path, add_bos, add_eos)
  else:
    raise ValueError(f"Invalid tokenizer_type:{tokenizer_type} chosen in config")
