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

"""DeepSeek-V4 SFT loss-mask checks: official chat encoding vs MaxText's SFT transforms.

DeepSeek-V4 ships no Jinja chat template; its official encoder
(tests/utils/deepseek4_reference/encoding_dsv4.py) defines the format. Each
assistant turn is rendered as

  <｜User｜>{user}<｜Assistant｜>{</think> | <think>}{reasoning</think>}{content}<｜end▁of▁sentence｜>

where `<｜Assistant｜>` plus one thinking token (`</think>` in chat mode or for
dropped-thinking history turns, `<think>` in thinking mode) is the generation
prompt, and everything after it up to and including EOS is model output
(`assistant_msg_template` ends with `eos_token`). Hence the trained targets are
the reasoning, the closing `</think>` (thinking turns only), the content and
EOS; BOS, `<｜User｜>`, `<｜Assistant｜>` and `<think>` are never trained.

The per-token labels are derived here from the encoder output alone, then fed
through MaxText's production SFTPromptMasking, packers, PadOrTrimToMaxLength
and ShiftData (grain and HF variants) and compared position by position with
the `targets_segmentation != 0` weights used by train.py's loss_fn:

  (a) per-position weights and total_weights equal the independent labels;
  (b) BOS/User/Assistant/<think> and prompt positions are never weighted;
  (c) segment ids and positions reset per packed document;
  (d) the EOS closing every assistant turn is weighted;
  (e) negative control: a mask shifted by one position fails (a).

DeepSeek-V4's tokenizer sets pad_token == eos_token (id 1). MaxText uses pad_id
both as the SFT prompt mask id and as a ShiftData ignored id, so every
completion EOS target is dropped from the loss. Those checks are xfail(strict)
for pad_id=1 and pass for pad_id=2 (`<｜▁pad▁｜>`).
"""

import copy
import functools
import importlib.util
import json
import os
import re
import types

import grain.python as grain
import numpy as np
import pytest

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils

pytestmark = [pytest.mark.cpu_only]

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "deepseek4_reference")
_TESTDATA_DIR = os.path.join(_REF_DIR, "testdata")


def _load_encoding_module():
  # Loaded by path so the package __init__ (torch reference model) is not imported.
  spec = importlib.util.spec_from_file_location("deepseek4_encoding_dsv4", os.path.join(_REF_DIR, "encoding_dsv4.py"))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


enc = _load_encoding_module()

# Token ids from the DeepSeek-V4 tokenizer.json.
BOS_ID = 0
EOS_ID = 1
PAD_TOKEN_ID = 2  # `<｜▁pad▁｜>`; unused by the shipped tokenizer config.
USER_ID = 128803
ASSISTANT_ID = 128804
THINK_ID = 128821
END_THINK_ID = 128822
DS4_PAD_ID = EOS_ID  # tokenizer_config.json: pad_token == eos_token.

_SPECIAL_IDS = {
    enc.bos_token: BOS_ID,
    enc.eos_token: EOS_ID,
    enc.USER_SP_TOKEN: USER_ID,
    enc.ASSISTANT_SP_TOKEN: ASSISTANT_ID,
    enc.thinking_start_token: THINK_ID,
    enc.thinking_end_token: END_THINK_ID,
}
_NEVER_TRAINED_IDS = (BOS_ID, USER_ID, ASSISTANT_ID, THINK_ID)

CONVERSATIONS = (
    (
        "chat",
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ],
    ),
    (
        "thinking",
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is 17 * 3?"},
            {"role": "assistant", "reasoning_content": "17 * 3 = 51.", "content": "51"},
        ],
    ),
    (
        "thinking",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "Greet back.", "content": "Hello!"},
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "reasoning_content": "2 is prime.", "content": "2"},
        ],
    ),
)

_TURN_RE = re.compile(
    re.escape(enc.ASSISTANT_SP_TOKEN)
    + f"({re.escape(enc.thinking_start_token)}|{re.escape(enc.thinking_end_token)})"
    + f"(.*?{re.escape(enc.eos_token)})",
    re.DOTALL,
)


def _split_official(messages, thinking_mode):
  """Splits encode_messages output into (prompt, completion) text chunks.

  A completion starts right after `<｜Assistant｜>` + thinking token and ends
  at EOS inclusive; each is checked to parse back to its source message.

  Returns:
    (encoded_text, chunks, is_prompt, num_thinking_turns).
  """
  text = enc.encode_messages(messages, thinking_mode=thinking_mode)
  assistants = [m for m in messages if m["role"] == "assistant"]
  matches = list(_TURN_RE.finditer(text))
  assert len(matches) == len(assistants), text
  chunks, is_prompt, cursor, num_thinking = [], [], 0, 0
  for match, msg in zip(matches, assistants):
    completion = match.group(2)
    chunks += [text[cursor : match.start(2)], completion]
    is_prompt += [True, False]
    cursor = match.end(2)
    parse_mode = "thinking" if match.group(1) == enc.thinking_start_token else "chat"
    parsed = enc.parse_message_from_completion_text(completion, thinking_mode=parse_mode)
    assert parsed["content"] == msg["content"], f"completion boundary mismatch: {completion!r}"
    if parse_mode == "thinking":
      num_thinking += 1
      assert parsed["reasoning_content"] == msg["reasoning_content"], f"reasoning mismatch: {completion!r}"
  assert cursor == len(text), "trailing text after the last completion"
  return text, chunks, is_prompt, num_thinking


class ToyTokenizer:
  """Tokenizer-free stand-in: real DeepSeek-V4 special ids, one id per other character."""

  _CHAR_OFFSET = 200_000
  _SPECIAL_RE = re.compile("|".join(re.escape(s) for s in _SPECIAL_IDS))

  def encode(self, text):
    ids, pos = [], 0
    for match in self._SPECIAL_RE.finditer(text):
      ids += [self._CHAR_OFFSET + ord(c) for c in text[pos : match.start()]]
      ids.append(_SPECIAL_IDS[match.group()])
      pos = match.end()
    return ids + [self._CHAR_OFFSET + ord(c) for c in text[pos:]]


def _build_examples(tokenizer):
  """Tokenizes every conversation into SFT chunks plus independent per-token labels."""
  examples = []
  for mode, messages in CONVERSATIONS:
    text, chunks, is_prompt, num_thinking = _split_official(messages, mode)
    # Same per-chunk tokenization as the production grain SFT pipeline.
    element = grain_data_processing._tokenize_sft_chunks(  # pylint: disable=protected-access
        {"messages": chunks, "is_prompt": is_prompt}, "messages", tokenizer
    )
    tokens = np.asarray(sum(element["messages"], []), dtype=np.int32)
    np.testing.assert_array_equal(tokens, tokenizer.encode(text))
    labels = np.concatenate([np.full(len(c), not p) for c, p in zip(element["messages"], is_prompt)])
    assert not labels[0], "a document must start with a prompt token"
    examples.append(
        {
            "element": element,
            "tokens": tokens,
            "labels": labels,
            "num_turns": is_prompt.count(False),
            "num_thinking_turns": num_thinking,
        }
    )
  return examples


# pylint: disable-next=too-many-positional-arguments
def _run_pipeline(examples, pad_id, variant, packing, max_len, batch_size):
  """Runs production SFT transforms and returns the list of shifted batches."""
  elements = [copy.deepcopy(e["element"]) for e in examples]
  masking = input_pipeline_utils.SFTPromptMasking(
      text_column_name="messages", completion_only=True, max_target_length=max_len, unk_id=pad_id
  )
  columns = ("inputs", "targets")
  if variant == "grain":
    # grain_data_processing.sft_preprocessing_pipeline -> format_and_batch.
    cfg = types.SimpleNamespace(
        packing=packing,
        max_target_length=max_len,
        max_segments_per_seq=None,
        grain_packing_type="first_fit",
        grain_use_elastic_iterator=False,
        add_bos=False,
    )
    ds = grain.MapDataset.source(elements).map(masking).to_iter_dataset()
    return list(data_processing_utils.format_and_batch(ds, cfg, batch_size, pad_id, columns, tokenizer_model=None))

  # Same operations list and DataLoader as hf_data_processing.preprocessing_pipeline.
  hf_cfg = types.SimpleNamespace(training_objective="causal_lm")
  operations = [masking]
  if packing:
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=batch_size, length_struct={c: max_len for c in columns}, max_sequences_per_bin=None
        )
    )
    operations.append(input_pipeline_utils.ReformatPacking(columns))
  else:
    operations.append(input_pipeline_utils.PadOrTrimToMaxLength(max_len, pad_id, config=hf_cfg))
    # drop_remainder=True is the preprocessing_pipeline default.
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))
  operations.append(
      hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
          hf_cfg,
          shift=True,
          use_dpo=False,
          use_sft=True,
          packing=packing,
          pad_id=pad_id,
          bos_token_id=BOS_ID,
      )
  )
  sampler = grain.IndexSampler(
      num_records=len(elements),
      num_epochs=1,
      shard_options=grain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False),
      shuffle=False,
      seed=0,
  )
  loader = grain.DataLoader(data_source=elements, operations=operations, sampler=sampler, worker_count=0)
  return list(loader)


def _locate_documents(batches, examples):
  """Finds each example's token run in the batches, independent of MaxText segment ids.

  Returns:
    List of (batch_index, row, start) per example.
  """
  placements = []
  for ex in examples:
    tokens, found = ex["tokens"], []
    for b, batch in enumerate(batches):
      for r, row in enumerate(batch["inputs"]):
        for s in range(len(row) - len(tokens) + 1):
          if np.array_equal(row[s : s + len(tokens)], tokens):
            found.append((b, r, s))
    assert len(found) == 1, f"document found {len(found)} times"
    placements.append(found[0])
  return placements


def _expected_weights(batches, examples):
  """Per-position expected loss weight: 1 iff the next token of the same document is completion."""
  expected = [np.zeros(b["inputs"].shape, dtype=bool) for b in batches]
  for ex, (b, r, s) in zip(examples, _locate_documents(batches, examples)):
    expected[b][r, s : s + len(ex["tokens"]) - 1] = ex["labels"][1:]
  return expected


def _weights_of(batches):
  return [np.asarray(b["targets_segmentation"]) != 0 for b in batches]


def _weighted_target_ids(batches, weights):
  return np.concatenate([np.asarray(b["targets"])[w] for b, w in zip(batches, weights)])


def _completion_target_ids(examples):
  return np.concatenate([ex["tokens"][ex["labels"]] for ex in examples])


def _multiset(ids):
  values, counts = np.unique(ids, return_counts=True)
  return dict(zip(values.tolist(), counts.tolist()))


# (packing, max_len, batch_size): one packed row; rows split across batches; unpacked.
_LAYOUTS = {
    "packed_one_row": (True, 512, 1),
    "packed_multi_row": (True, 200, 2),
    "unpacked": (False, 256, 3),
}
_VARIANTS = ("grain", "hf")

_EOS_DEFECT = (
    "MaxText defect: DeepSeek-V4 pad_id == eos_id == 1. input_pipeline_utils.SFTPromptMasking(unk_id=pad_id) "
    "writes 1 into prompt targets and ShiftData(ignored_ids=[pad_id, ...]) (data_processing_utils.format_and_batch; "
    "hf_data_processing._get_training_objective_transform) zeroes targets_segmentation wherever target == 1, "
    "dropping every completion EOS: total_weights 40 vs 44, weighted EOS 0 vs 4 (toy set, all layouts)."
)
_SEGMENT_DEFECT = (
    "MaxText defect: input_pipeline_utils.PadOrTrimToMaxLength sets inputs_segmentation = (inputs != pad_id); "
    "with pad_id == eos_id == 1 the EOS between turns 1 and 2 of the multi-turn document gets segment 0 "
    "(1 position) and is excluded from attention."
)


def _param_grid(xfail_reason=None, xfail_if=None):
  """(pad_id, variant, layout) grid; xfail(strict, AssertionError only) where xfail_if(pad_id, packing) holds."""
  params = []
  for pad_id in (PAD_TOKEN_ID, DS4_PAD_ID):
    for variant in _VARIANTS:
      for layout, (packing, _, _) in _LAYOUTS.items():
        marks = []
        if xfail_if is not None and xfail_if(pad_id, packing):
          marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=xfail_reason))
        params.append(pytest.param(pad_id, variant, layout, marks=marks, id=f"pad{pad_id}-{variant}-{layout}"))
  return params


def _is_ds4_pad(pad_id, packing):
  del packing
  return pad_id == DS4_PAD_ID


def _is_ds4_pad_unpacked(pad_id, packing):
  return pad_id == DS4_PAD_ID and not packing


@functools.lru_cache(maxsize=None)
def _toy_examples():
  return _build_examples(ToyTokenizer())


def _run(examples, pad_id, variant, layout):
  packing, max_len, batch_size = _LAYOUTS[layout]
  return _run_pipeline(examples, pad_id, variant, packing, max_len, batch_size)


@pytest.mark.parametrize("case", [1, 2, 3, 4])
def test_vendored_encoder_matches_official_golden(case):
  """Vendored encoding_dsv4.py reproduces the upstream encoding/tests goldens."""
  with open(os.path.join(_TESTDATA_DIR, f"test_input_{case}.json"), encoding="utf-8") as f:
    data = json.load(f)
  with open(os.path.join(_TESTDATA_DIR, f"test_output_{case}.txt"), encoding="utf-8") as f:
    gold = f.read()
  if case == 1:
    messages = data["messages"]
    messages[0]["tools"] = data["tools"]
  else:
    messages = data
  mode = "chat" if case == 4 else "thinking"
  assert enc.encode_messages(messages, thinking_mode=mode) == gold


def test_official_split_structure():
  """Completion chunks follow the official format (see module docstring)."""
  for ex in _toy_examples():
    for chunk, is_prompt in zip(ex["element"]["messages"], ex["element"]["is_prompt"]):
      if is_prompt:
        assert chunk[-1] in (THINK_ID, END_THINK_ID) and chunk[-2] == ASSISTANT_ID
        assert EOS_ID not in chunk
      else:
        assert chunk[-1] == EOS_ID and not set(chunk) & set(_NEVER_TRAINED_IDS)
  assert [ex["num_thinking_turns"] for ex in _toy_examples()] == [0, 1, 1]


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid(_EOS_DEFECT, _is_ds4_pad))
def test_loss_mask_matches_official_format(pad_id, variant, layout):
  """(a) exact per-position weights and total_weights vs the independent labels."""
  examples = _toy_examples()
  batches = _run(examples, pad_id, variant, layout)
  weights, expected = _weights_of(batches), _expected_weights(batches, examples)
  total_weights = sum(int(w.sum()) for w in weights)
  expected_total = sum(int(ex["labels"].sum()) for ex in examples)
  assert total_weights == expected_total, f"total_weights {total_weights} vs {expected_total} completion targets"
  for w, e in zip(weights, expected):
    np.testing.assert_array_equal(w, e)
  assert _multiset(_weighted_target_ids(batches, weights)) == _multiset(_completion_target_ids(examples))


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid())
def test_prompt_and_control_tokens_not_trained(pad_id, variant, layout):
  """(b) BOS/User/Assistant/<think> never weighted; </think> only for thinking turns; no prompt leak."""
  examples = _toy_examples()
  batches = _run(examples, pad_id, variant, layout)
  weights, expected = _weights_of(batches), _expected_weights(batches, examples)
  ids = _weighted_target_ids(batches, weights)
  assert not set(ids.tolist()) & set(_NEVER_TRAINED_IDS), "a never-trained control token carries loss weight"
  assert int((ids == END_THINK_ID).sum()) == sum(ex["num_thinking_turns"] for ex in examples)
  for w, e in zip(weights, expected):
    assert not (w & ~e).any(), "a prompt-position target carries loss weight"


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid(_EOS_DEFECT, _is_ds4_pad))
def test_completion_eos_is_trained(pad_id, variant, layout):
  """(d) the EOS closing every assistant turn is a weighted target."""
  examples = _toy_examples()
  batches = _run(examples, pad_id, variant, layout)
  ids = _weighted_target_ids(batches, _weights_of(batches))
  num_eos, num_turns = int((ids == EOS_ID).sum()), sum(ex["num_turns"] for ex in examples)
  assert num_eos == num_turns, f"weighted EOS targets {num_eos} vs {num_turns} assistant turns"


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid(_SEGMENT_DEFECT, _is_ds4_pad_unpacked))
def test_segments_and_positions_reset_per_document(pad_id, variant, layout):
  """(c) each document has one nonzero segment id, unique in its row, and positions 0..n-1.

  The document's final token (its closing EOS) is exempt from the segment check: it only predicts padding.
  """
  examples = _toy_examples()
  batches = _run(examples, pad_id, variant, layout)
  row_segments = {}
  for ex, (b, r, s) in zip(examples, _locate_documents(batches, examples)):
    n = len(ex["tokens"])
    seg = np.asarray(batches[b]["inputs_segmentation"])[r, s : s + n - 1]
    pos = np.asarray(batches[b]["inputs_position"])[r, s : s + n]
    assert seg[0] != 0 and (seg == seg[0]).all(), f"segment ids not constant within a document: {seg}"
    np.testing.assert_array_equal(pos, np.arange(n))
    assert seg[0] not in row_segments.setdefault((b, r), set()), "segment id reused within a row"
    row_segments[(b, r)].add(seg[0])
  if variant == "grain" and _LAYOUTS[layout][0]:
    assert any(len(v) > 1 for v in row_segments.values()), "packing produced no multi-document row"


@pytest.mark.parametrize("variant", _VARIANTS)
def test_negative_control_shifted_mask_is_detected(variant):
  """(e) rolling targets_segmentation by one position must fail the checks of (a)."""
  examples = _toy_examples()
  batches = _run(examples, PAD_TOKEN_ID, variant, "packed_one_row")
  expected = _expected_weights(batches, examples)
  rolled = [np.roll(np.asarray(b["targets_segmentation"]), 1, axis=1) != 0 for b in batches]
  assert any((w != e).any() for w, e in zip(rolled, expected)), "per-position check missed the shift"
  assert _multiset(_weighted_target_ids(batches, rolled)) != _multiset(_completion_target_ids(examples))
  # A bare count is blind to the shift: np.roll preserves the total weight here.
  assert sum(int(w.sum()) for w in rolled) == sum(int(ex["labels"].sum()) for ex in examples)


@functools.lru_cache(maxsize=None)
def _load_ds4_tokenizer():
  """Loads the tokenizer from $DEEPSEEK_V4_TOKENIZER_PATH, else the HF Hub; skips if not found/downloadable."""
  import transformers  # pylint: disable=import-outside-toplevel

  path = os.environ.get("DEEPSEEK_V4_TOKENIZER_PATH", "deepseek-ai/DeepSeek-V4-Flash")
  try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
  except OSError as e:
    pytest.skip(f"DeepSeek-V4 tokenizer unavailable at {path!r} (set DEEPSEEK_V4_TOKENIZER_PATH): {e}")
  return tokenizer


@pytest.mark.external_training  # May download tokenizer.json (~6 MB) from the HF Hub.
def test_real_tokenizer_special_ids_and_pad():
  """Special ids used above match tokenizer.json; the shipped pad token is EOS."""
  tokenizer = _load_ds4_tokenizer()
  for token, token_id in _SPECIAL_IDS.items():
    actual = tokenizer.convert_tokens_to_ids(token)
    assert actual == token_id, f"{token!r} -> {actual}, expected {token_id}"
  assert tokenizer.convert_tokens_to_ids("<｜▁pad▁｜>") == PAD_TOKEN_ID
  assert hf_data_processing._get_pad_id(tokenizer) == DS4_PAD_ID  # pylint: disable=protected-access
  assert tokenizer.encode("hi")[0] != BOS_ID, "tokenizer must not auto-insert BOS into SFT chunks"


@pytest.mark.external_training  # May download tokenizer.json (~6 MB) from the HF Hub.
@pytest.mark.parametrize("pad_id,variant,layout", _param_grid(_EOS_DEFECT, _is_ds4_pad))
def test_real_tokenizer_loss_mask_matches_official_format(pad_id, variant, layout):
  """Same as test_loss_mask_matches_official_format with real BPE token ids."""
  examples = _build_examples(_load_ds4_tokenizer())
  batches = _run(examples, pad_id, variant, layout)
  weights, expected = _weights_of(batches), _expected_weights(batches, examples)
  for w, e in zip(weights, expected):
    np.testing.assert_array_equal(w, e)
  ids = _weighted_target_ids(batches, weights)
  assert not set(ids.tolist()) & set(_NEVER_TRAINED_IDS), "a never-trained control token carries loss weight"
  num_eos, num_turns = int((ids == EOS_ID).sum()), sum(ex["num_turns"] for ex in examples)
  assert num_eos == num_turns, f"weighted EOS targets {num_eos} vs {num_turns} assistant turns"
