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
import os
from typing import Literal, Sequence, Collection
from pathlib import Path
from maxtext.utils import max_logging
import transformers
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from sentencepiece import SentencePieceProcessor

# Kimi-K3 Tiktoken pattern for Han character splitting matching encoding_k3.py / tokenization_kimi.py
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


class TikTokenTokenizer:
  """
  Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
  """

  special_tokens: dict[str, int]

  num_reserved_special_tokens = 256

  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # pylint: disable=line-too-long

  def __init__(
      self,
      model_path: str,
      add_bos: bool = False,
      add_eos: bool = False,
      pat_str: str | None = None,
      special_tokens: dict[str, int] | None = None,
  ):
    """
    Initializes the Tokenizer with a Tiktoken model.

    Args:
        model_path (str): The path to the Tiktoken model file.
        add_bos (bool): Whether to prepend the beginning-of-sequence token.
        add_eos (bool): Whether to append the end-of-sequence token.
        pat_str (str, optional): Custom regex pattern for tokenization splitting.
        special_tokens (dict[str, int], optional): Custom mapping of special token strings to IDs.
    """
    is_kimi = "kimi" in str(model_path).lower()

    if pat_str is not None:
      self.pat_str = pat_str
    elif is_kimi:
      self.pat_str = KIMI_K3_TIKTOKEN_PAT_STR

    try:
      mergeable_ranks = load_tiktoken_bpe(model_path)
    except Exception as e:
      raise ValueError(f"Failed to load tiktoken tokenizer from {model_path}: {e}") from e
    num_base_tokens = len(mergeable_ranks)

    if special_tokens is not None:
      self.special_tokens = special_tokens
    elif is_kimi:
      # Load from tokenizer_config.json if available in parent directory
      added_tokens_mapping = {}
      model_dir = Path(model_path).parent if os.path.isfile(model_path) else Path(model_path)
      cfg_path = model_dir / "tokenizer_config.json"
      if cfg_path.is_file():
        try:
          with open(cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
          added_tokens = tok_cfg.get("added_tokens_decoder", {})
          for k_id, token_info in added_tokens.items():
            content = token_info.get("content") if isinstance(token_info, dict) else str(token_info)
            added_tokens_mapping[int(k_id)] = content
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      # Standard Kimi-K3 special tokens
      default_kimi_special = {
          num_base_tokens: "[BOS]",
          num_base_tokens + 1: "[EOS]",
          num_base_tokens + 2: "<|end_of_msg|>",
          num_base_tokens + 3: "<|open|>",
          num_base_tokens + 4: "<|close|>",
          num_base_tokens + 5: "<|sep|>",
          num_base_tokens + 6: "[start_header_id]",
          num_base_tokens + 7: "[end_header_id]",
          num_base_tokens + 9: "[EOT]",
          num_base_tokens + 18: "<|media_begin|>",
          num_base_tokens + 19: "<|media_content|>",
          num_base_tokens + 20: "<|media_end|>",
          num_base_tokens + 21: "<|media_pad|>",
          num_base_tokens + 65: "<osagent_mode>",
          num_base_tokens + 254: "[UNK]",
          num_base_tokens + 255: "[PAD]",
      }
      self.special_tokens = {}
      for i in range(num_base_tokens, num_base_tokens + self.num_reserved_special_tokens):
        tok_str = added_tokens_mapping.get(i, default_kimi_special.get(i, f"<|reserved_token_{i}|>"))
        self.special_tokens[tok_str] = i
    else:
      special_tokens_list = [
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
      self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens_list)}

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
    if is_kimi:
      self.bos_id: int = self.special_tokens.get("[BOS]", num_base_tokens)
      self.eos_id: int = self.special_tokens.get("[EOS]", num_base_tokens + 1)
      self.pad_id: int = self.special_tokens.get("[PAD]", num_base_tokens + 255)
      self.unk_id: int = self.special_tokens.get("[UNK]", num_base_tokens + 254)
      self.stop_tokens = {
          self.eos_id,
          self.special_tokens.get("<|end_of_msg|>", self.eos_id),
          self.special_tokens.get("[EOT]", self.eos_id),
      }
    else:
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


class KimiTikTokenTokenizer(TikTokenTokenizer):
  """
  Tiktoken tokenizer specifically configured for Kimi-K3 with Han character splitting.
  """

  pat_str = KIMI_K3_TIKTOKEN_PAT_STR

  def __init__(self, model_path: str, add_bos: bool = False, add_eos: bool = False):
    super().__init__(model_path, add_bos=add_bos, add_eos=add_eos, pat_str=KIMI_K3_TIKTOKEN_PAT_STR)


def build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token):
  """Loads the tokenizer at `tokenizer_path`"""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  if tokenizer_type in ("tiktoken", "kimi_tiktoken", "kimi"):
    if "kimi" in tokenizer_path.lower() or tokenizer_type in ("kimi_tiktoken", "kimi"):
      return KimiTikTokenTokenizer(tokenizer_path, add_bos, add_eos)
    assert "tiktoken" in tokenizer_path, f"Invalid tokenizer type: {tokenizer_type} chosen for {tokenizer_path}"
    return TikTokenTokenizer(tokenizer_path, add_bos, add_eos)
  elif tokenizer_type == "huggingface":
    return HFTokenizer(tokenizer_path, add_bos, add_eos, hf_access_token)
  elif tokenizer_type == "sentencepiece":
    return SentencePieceTokenizer(tokenizer_path, add_bos, add_eos)
  else:
    raise ValueError(f"Invalid tokenizer_type:{tokenizer_type} chosen in config")
