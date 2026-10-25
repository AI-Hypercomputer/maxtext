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
(src/maxtext/input_pipeline/encoding_dsv4.py) defines the format. Each
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

Tests run with <｜▁pad▁｜> (id 2) as pad_id (remapped in production by
input_pipeline_utils.fix_pad_token_if_needed, tested in
tests/unit/input_pipeline_utils_test.py).
"""

import copy
import functools
import json
import os
import re
import types
from unittest import mock

import grain.python as grain
import numpy as np
import pytest

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import encoding_dsv4 as enc
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import tokenizer as maxtext_tokenizer

pytestmark = [pytest.mark.cpu_only]

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "deepseek4_reference")
_TESTDATA_DIR = os.path.join(_REF_DIR, "testdata")


# Token ids from the DeepSeek-V4 tokenizer.json.
BOS_ID = 0
EOS_ID = 1
PAD_TOKEN_ID = 2  # `<｜▁pad▁｜>`; unused by the shipped tokenizer config.
USER_ID = 128803
ASSISTANT_ID = 128804
THINK_ID = 128821
END_THINK_ID = 128822
DS4_PAD_ID = PAD_TOKEN_ID  # With SFT pad fix applied, DS4 uses <｜▁pad▁｜> (id 2).

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

_TURN_RE = input_pipeline_utils._DS4_TURN_RE  # pylint: disable=protected-access


def _split_official(messages, thinking_mode):
  """Splits encode_messages output into (prompt, completion) text chunks.

  A completion starts right after `<｜Assistant｜>` + thinking token and ends
  at EOS inclusive; each is checked to parse back to its source message. In
  thinking mode, history turns whose reasoning the encoder dropped stay in the
  prompt.

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
    parse_mode = "thinking" if match.group(1) == enc.thinking_start_token else "chat"
    parsed = enc.parse_message_from_completion_text(completion, thinking_mode=parse_mode)
    assert parsed["content"] == msg["content"], f"completion boundary mismatch: {completion!r}"
    if thinking_mode == "thinking" and parse_mode == "chat":
      # History turn whose reasoning the encoder dropped: part of the next prompt, not trained.
      continue
    chunks += [text[cursor : match.start(2)], completion]
    is_prompt += [True, False]
    cursor = match.end(2)
    if parse_mode == "thinking":
      num_thinking += 1
      assert parsed["reasoning_content"] == msg["reasoning_content"], f"reasoning mismatch: {completion!r}"
  assert cursor == len(text), "trailing text after the last completion"
  return text, chunks, is_prompt, num_thinking


class ToyTokenizer:
  """Tokenizer-free stand-in: real DeepSeek-V4 special ids, one id per other character."""

  name_or_path = "deepseek4"
  chat_template = None
  _CHAR_OFFSET = 200_000
  _SPECIAL_RE = re.compile("|".join(re.escape(s) for s in _SPECIAL_IDS))

  def encode(self, text):
    ids, pos = [], 0
    for match in self._SPECIAL_RE.finditer(text):
      ids += [self._CHAR_OFFSET + ord(c) for c in text[pos : match.start()]]
      ids.append(_SPECIAL_IDS[match.group()])
      pos = match.end()
    return ids + [self._CHAR_OFFSET + ord(c) for c in text[pos:]]

  def convert_tokens_to_ids(self, token):
    return _SPECIAL_IDS.get(token)


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


def _param_grid():
  """(pad_id, variant, layout) grid for DeepSeek-V4 SFT loss-mask verification."""
  params = []
  for pad_id in (DS4_PAD_ID,):
    for variant in _VARIANTS:
      for layout in _LAYOUTS:
        params.append(pytest.param(pad_id, variant, layout, id=f"pad{pad_id}-{variant}-{layout}"))
  return params


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
      else:
        assert chunk[-1] == EOS_ID and not set(chunk) & set(_NEVER_TRAINED_IDS)
  assert [ex["num_thinking_turns"] for ex in _toy_examples()] == [0, 1, 1]


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid())
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


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid())
def test_completion_eos_is_trained(pad_id, variant, layout):
  """(d) the EOS closing every assistant turn is a weighted target."""
  examples = _toy_examples()
  batches = _run(examples, pad_id, variant, layout)
  ids = _weighted_target_ids(batches, _weights_of(batches))
  num_eos, num_turns = int((ids == EOS_ID).sum()), sum(ex["num_turns"] for ex in examples)
  assert num_eos == num_turns, f"weighted EOS targets {num_eos} vs {num_turns} assistant turns"


@pytest.mark.parametrize("pad_id,variant,layout", _param_grid())
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
  path = os.environ.get("DEEPSEEK_V4_TOKENIZER_PATH", "deepseek-ai/DeepSeek-V4-Flash")
  try:
    # Same construction as the grain SFT pipeline: per-chunk encoding with BOS/EOS disabled.
    tokenizer = maxtext_tokenizer.build_tokenizer(path, "huggingface", add_bos=False, add_eos=False, hf_access_token=None)
  except OSError as e:
    pytest.skip(f"DeepSeek-V4 tokenizer unavailable at {path!r} (set DEEPSEEK_V4_TOKENIZER_PATH): {e}")
  tokenizer = tokenizer.tokenizer
  tokenizer.pad_token = "<｜▁pad▁｜>"
  return tokenizer


@pytest.mark.external_training  # May download tokenizer.json (~6 MB) from the HF Hub.
def test_real_tokenizer_special_ids_and_pad():
  """Special ids used above match tokenizer.json; the shipped pad token is EOS."""
  tokenizer = _load_ds4_tokenizer()
  for token, token_id in _SPECIAL_IDS.items():
    actual = tokenizer.convert_tokens_to_ids(token)
    assert actual == token_id, f"{token!r} -> {actual}, expected {token_id}"
  assert tokenizer.convert_tokens_to_ids("<｜▁pad▁｜>") == PAD_TOKEN_ID
  assert hf_data_processing._get_pad_id(tokenizer) == PAD_TOKEN_ID  # pylint: disable=protected-access
  assert tokenizer.encode("hi")[0] != BOS_ID, "tokenizer must not auto-insert BOS into SFT chunks"


@pytest.mark.external_training  # May download tokenizer.json (~6 MB) from the HF Hub.
@pytest.mark.parametrize("pad_id,variant,layout", _param_grid())
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


def test_is_deepseek_v4_by_name():
  """Tests _is_deepseek_v4 detection by model or tokenizer name."""
  positive_names = [
      "deepseek4-284b",
      "deepseek4-1600b",
      "deepseek-ai/DeepSeek-V4-Flash",
      "deepseek-ai/DeepSeek-V4-Pro-0813",
  ]
  negative_names = [
      "deepseek-ai/DeepSeek-V3-0324",
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
      "deepseek3-671b",
  ]
  for name in positive_names:
    assert input_pipeline_utils._is_deepseek_v4(None, name), f"Expected True for {name}"  # pylint: disable=protected-access
  for name in negative_names:
    assert not input_pipeline_utils._is_deepseek_v4(None, name), f"Expected False for {name}"  # pylint: disable=protected-access

  # A DeepSeek-V3.2 tokenizer loaded from a local path has the same special tokens but is not V4.
  v32_vocab = {"<｜User｜>": 128803, "<｜Assistant｜>": 128804, "｜DSML｜": 128793}
  v32_tok = types.SimpleNamespace(chat_template=None, name_or_path="/tmp/local_tokenizer", get_vocab=lambda: v32_vocab)
  assert not input_pipeline_utils._is_deepseek_v4(v32_tok, "deepseek3.2-671b")  # pylint: disable=protected-access
  assert not input_pipeline_utils._is_deepseek_v4(v32_tok, None)  # pylint: disable=protected-access


def test_apply_chat_template_deepseek_v4():
  """Tests that apply_chat_template uses encode_messages for DeepSeek-V4 when chat_template is None."""
  tokenizer = ToyTokenizer()
  for mode, msgs in CONVERSATIONS:
    example = {"messages": copy.deepcopy(msgs)}
    formatted = input_pipeline_utils.apply_chat_template(example, tokenizer, "messages")
    _, expected_chunks, expected_is_prompt, _ = _split_official(msgs, mode)
    assert formatted["messages"] == expected_chunks
    assert formatted["is_prompt"] == expected_is_prompt


def test_apply_chat_template_non_deepseek_without_template_raises():
  """Non-DeepSeek tokenizer without chat_template raises ValueError as expected."""
  tok = types.SimpleNamespace(
      chat_template=None,
      name_or_path="other-model",
      apply_chat_template=mock.MagicMock(
          side_effect=ValueError("Cannot use apply_chat_template without a chat_template")
      ),
  )
  example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]}
  with pytest.raises(ValueError):
    input_pipeline_utils.apply_chat_template(example, tok, "messages")


def test_format_chat_template_grain_deepseek_v4():
  """Tests that _format_chat_template_grain forwards model_name and formats DeepSeek-V4 conversations."""
  tok = types.SimpleNamespace(chat_template=None, name_or_path="custom-local-tok")
  for mode, msgs in CONVERSATIONS:
    element = {"messages": copy.deepcopy(msgs)}
    formatted = grain_data_processing._format_chat_template_grain(  # pylint: disable=protected-access
        element, data_columns=["messages"], tokenizer_model=tok, model_name="deepseek4-1600b"
    )
    _, expected_chunks, expected_is_prompt, _ = _split_official(msgs, mode)
    assert formatted["messages"] == expected_chunks
    assert formatted["is_prompt"] == expected_is_prompt


def test_thinking_history_turns_stay_in_prompt():
  """Only the final thinking turn is trained; the earlier turn (reasoning dropped by the encoder) is prompt."""
  _, msgs = CONVERSATIONS[2]
  formatted = input_pipeline_utils.apply_chat_template(
      {"messages": copy.deepcopy(msgs)}, ToyTokenizer(), "messages", model_name="deepseek4-1600b"
  )
  assert formatted["is_prompt"] == [True, False]
  prompt, completion = formatted["messages"]
  assert "Hello!" in prompt and "Greet back." not in prompt
  assert completion.startswith("2 is prime.") and completion.endswith(enc.eos_token)


def test_grain_sft_tokenizer_disables_bos_and_eos():
  """The grain SFT pipeline builds its tokenizer with add_bos=False and add_eos=False."""
  dataset = mock.MagicMock()
  dataset.map.return_value = dataset
  config = types.SimpleNamespace(
      sft_train_on_completion_only=True, max_target_length=8, chat_template=None, model_name="deepseek4-1600b"
  )
  tok = types.SimpleNamespace(tokenizer=ToyTokenizer(), pad_id=DS4_PAD_ID)
  utils = grain_data_processing.data_processing_utils
  with (
      mock.patch.object(utils, "parse_and_keep_features", return_value=dataset),
      mock.patch.object(utils, "get_tokenizer_and_pad_id", return_value=(tok, DS4_PAD_ID)) as get_tokenizer,
      mock.patch.object(utils, "validate_and_configure_sft_columns"),
      mock.patch.object(utils, "get_local_batch_size", return_value=1),
      mock.patch.object(utils, "format_and_batch", return_value=dataset),
      mock.patch.object(utils, "apply_multiprocessing_and_prefetch", return_value=dataset),
  ):
    grain_data_processing.sft_preprocessing_pipeline(
        dataset, config, ["messages"], tokenize=True, grain_worker_count=0, grain_per_worker_buffer_size=1
    )
  assert get_tokenizer.call_args.kwargs == {"add_bos": False, "add_eos": False}


def test_thinking_mode_tool_calling_keeps_reasoning_without_embedded_tools():
  """Multi-step tool-calling trajectories keep reasoning on earlier assistant tool-call turns."""
  tok = ToyTokenizer()
  with open(os.path.join(_TESTDATA_DIR, "test_input_1.json"), encoding="utf-8") as f:
    data = json.load(f)
  with open(os.path.join(_TESTDATA_DIR, "test_output_1.txt"), encoding="utf-8") as f:
    gold = f.read()

  # Case 1: top-level `tools` field on `example` (not pre-attached to `messages[0]`).
  out_with_top_tools = input_pipeline_utils.apply_chat_template(
      {"messages": copy.deepcopy(data["messages"]), "tools": data["tools"]},
      tok,
      "messages",
  )
  assert out_with_top_tools["is_prompt"] == [True, False, True, False]
  assert "".join(out_with_top_tools["messages"]) == gold
  assert "The user wants to know the weather in Beijing." in out_with_top_tools["messages"][1]

  # Case 2: `tools` omitted entirely, only `tool_calls` and `role == "tool"` present.
  out_no_tools_schema = input_pipeline_utils.apply_chat_template(
      {"messages": copy.deepcopy(data["messages"])},
      tok,
      "messages",
  )
  assert out_no_tools_schema["is_prompt"] == [True, False, True, False]
  assert "The user wants to know the weather in Beijing." in out_no_tools_schema["messages"][1]

  # Case 3: top-level `tools` when `messages` starts directly with `"user"` (no `"system"`/`"developer"`).
  out_no_system = input_pipeline_utils.apply_chat_template(
      {"messages": copy.deepcopy(data["messages"][1:]), "tools": data["tools"]},
      tok,
      "messages",
  )
  assert out_no_system["is_prompt"] == [True, False, True, False]
  assert "".join(out_no_system["messages"]) == gold.replace("You are a helpful assistant.", "", 1)

  # Case 4: `_format_chat_template_grain` accepts tool-calling turns where `"content"` is omitted.
  out_grain = grain_data_processing._format_chat_template_grain(  # pylint: disable=protected-access
      {"messages": copy.deepcopy(data["messages"]), "tools": data["tools"]},
      data_columns=["messages"],
      tokenizer_model=tok,
      model_name="deepseek4-1600b",
  )
  assert out_grain["is_prompt"] == [True, False, True, False]
  assert "".join(out_grain["messages"]) == gold


def test_action_task_keeps_action_token_in_prompt():
  """When a user message has `task: "action"`, `<｜action｜>` stays in the prompt chunk."""
  tok = ToyTokenizer()
  with open(os.path.join(_TESTDATA_DIR, "test_input_4.json"), encoding="utf-8") as f:
    data = json.load(f)
  with open(os.path.join(_TESTDATA_DIR, "test_output_4.txt"), encoding="utf-8") as f:
    gold = f.read()

  out = input_pipeline_utils.apply_chat_template({"messages": copy.deepcopy(data)}, tok, "messages")
  assert out["is_prompt"] == [True, False, True, False]
  assert "".join(out["messages"]) == gold
  expected_prefix = enc.ASSISTANT_SP_TOKEN + enc.thinking_end_token + enc.DS_TASK_SP_TOKENS["action"]
  assert out["messages"][2].endswith(expected_prefix)
  assert out["messages"][3] == "Search" + enc.eos_token
