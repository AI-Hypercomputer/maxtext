# Copyright 2026 Google LLC
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

"""Pure-logic tests for SFTPromptMaskingWindows (no tokenizer / no GCS).

Validates that long SFT examples keep the turn terminator (e.g. <end_of_turn>) in the loss by
windowing instead of head-truncating, and that the fast path matches SFTPromptMasking exactly.
"""
import pytest

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

import numpy as np

from maxtext.input_pipeline.input_pipeline_utils import SFTPromptMasking, SFTPromptMaskingWindows

PAD = 0
EOT = 129  # stand-in for <end_of_turn>


def _loss_tokens(record):
  """Tokens that actually contribute to the loss = targets != pad."""
  t = record["targets"]
  return [int(x) for x in t[t != PAD]]


def test_short_example_matches_sftpromptmasking():
  """An example that fits in max_target_length yields one record, identical to SFTPromptMasking."""
  length = 20
  element = {"text": [[1, 2, 3], [101, 102, 103, EOT]], "is_prompt": [True, False]}
  windows = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=length, unk_id=PAD)
  recs = windows.flat_map({k: list(v) for k, v in element.items()})

  baseline = SFTPromptMasking("text", completion_only=True, max_target_length=length, unk_id=PAD)
  expected = baseline.map({k: list(v) for k, v in element.items()})

  assert len(recs) == 1
  assert np.array_equal(recs[0]["inputs"], expected["inputs"])
  assert np.array_equal(recs[0]["targets"], expected["targets"])


def test_truncate_baseline_drops_eot():
  """Sanity: the old 1:1 transform drops the terminator for an over-length example (the bug)."""
  length = 20
  prompt = [1, 2, 3]
  completion = list(range(100, 129)) + [EOT]  # 30 tokens, terminator last
  baseline = SFTPromptMasking("text", completion_only=True, max_target_length=length, unk_id=PAD)
  out = baseline.map({"text": [prompt, completion], "is_prompt": [True, False]})
  assert len(out["inputs"]) == length
  assert EOT not in set(int(x) for x in out["targets"])  # terminator truncated away → not in loss


def test_long_single_turn_keeps_eot_and_covers_completion_once():
  """A long single-turn example is split into windows; the terminator stays in the loss and every
  completion token is trained exactly once (disjoint loss), with the prompt pinned as context."""
  length = 20
  prompt = [1, 2, 3]
  completion = list(range(100, 129)) + [EOT]  # 30 tokens
  windows = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=2)
  recs = windows.flat_map({"text": [prompt, completion], "is_prompt": [True, False]})

  assert len(recs) >= 2
  for r in recs:
    assert len(r["inputs"]) == len(r["targets"]) <= length
    # prompt pinned (masked) as context in every window
    assert [int(x) for x in r["inputs"][: len(prompt)]] == prompt
    assert [int(x) for x in r["targets"][: len(prompt)]] == [PAD] * len(prompt)

  # disjoint + complete + ordered coverage of the completion
  covered = []
  for r in recs:
    covered += _loss_tokens(r)
  assert covered == completion

  # terminator is in the loss of the LAST window
  assert _loss_tokens(recs[-1])[-1] == EOT


def test_multi_turn_each_turn_terminator_in_loss():
  """A multi-turn over-length example keeps each assistant turn's terminator in the loss."""
  length = 16
  # system+user1 | answer1+EOT | user2 | answer2+EOT  (total > length)
  segments = [
      [1, 2, 3, 4, 5],  # prompt (system+user1)
      [201, 202, 203, EOT],  # completion 1
      [6, 7, 8],  # prompt (user2)
      [211, 212, 213, 214, 215, EOT],  # completion 2
  ]
  is_prompt = [True, False, True, False]
  windows = SFTPromptMaskingWindows("text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=2)
  recs = windows.flat_map({"text": segments, "is_prompt": is_prompt})

  covered = []
  for r in recs:
    assert len(r["inputs"]) == len(r["targets"]) <= length
    covered += _loss_tokens(r)

  # both completions fully covered, each exactly once, terminators included
  assert covered == [201, 202, 203, EOT, 211, 212, 213, 214, 215, EOT]
  assert covered.count(EOT) == 2


def test_fan_out_cap_is_respected():
  """max_fan_out bounds the number of emitted records (runaway guard)."""
  length = 12
  prompt = [1, 2]
  completion = list(range(100, 200))  # very long
  windows = SFTPromptMaskingWindows(
      "text", completion_only=True, max_target_length=length, unk_id=PAD, overlap=1, max_fan_out=3
  )
  recs = windows.flat_map({"text": [prompt, completion], "is_prompt": [True, False]})
  assert len(recs) == 3
  for r in recs:
    assert len(r["inputs"]) <= length


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"PASS {name}")
  print("ALL PASS")
