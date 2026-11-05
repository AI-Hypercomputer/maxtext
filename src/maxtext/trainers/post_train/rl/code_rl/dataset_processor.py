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

"""Dataset processor for multi-turn Python coding RL (`PythonCodeEnv`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

DEFAULT_FORMAT_INSTRUCTIONS: str = "Enclose your final Python solution inside a ```python ... ``` code block."


def _parse_raw_tests(raw_tests: Any) -> Any:
  """Parses raw JSON strings or splits multi-line string tests into structured collections."""
  if isinstance(raw_tests, str):
    stripped = raw_tests.strip()
    if stripped.startswith(("[", "{")):
      try:
        return json.loads(stripped)
      except Exception:  # pylint: disable=broad-except
        return [line.strip() for line in stripped.splitlines() if line.strip()]
    elif stripped:
      return [line.strip() for line in stripped.splitlines() if line.strip()]
    return []
  if isinstance(raw_tests, Mapping):
    return dict(raw_tests)
  if isinstance(raw_tests, (Sequence, Iterable)) and not isinstance(raw_tests, (str, bytes)):
    return list(raw_tests)
  return raw_tests


def _normalize_stdio_tests(parsed: Any) -> dict[str, Any]:
  """Validates and normalizes `tests` payload for `stdio` format."""
  if isinstance(parsed, Mapping):
    inputs = list(parsed.get("inputs", []))
    outputs = list(parsed.get("outputs", []))
    if not inputs or len(inputs) != len(outputs):
      raise ValueError("Dataset record with format='stdio' requires non-empty equal-length 'inputs' and 'outputs'")
    return {
        "format": "stdio",
        "inputs": [str(x) for x in inputs],
        "outputs": [str(x) for x in outputs],
    }
  if (
      isinstance(parsed, Sequence)
      and not isinstance(parsed, (str, bytes))
      and parsed
      and all(isinstance(x, Mapping) for x in parsed)
  ):
    inputs = [str(x.get("input", "")) for x in parsed]
    outputs = [str(x.get("output", "")) for x in parsed]
    if not inputs or len(inputs) != len(outputs):
      raise ValueError("Dataset record with format='stdio' requires non-empty equal-length 'inputs' and 'outputs'")
    return {
        "format": "stdio",
        "inputs": inputs,
        "outputs": outputs,
    }
  raise ValueError("Dataset record with format='stdio' must provide a dict 'tests' with 'inputs' and 'outputs'")


def _normalize_assert_tests(parsed: Any) -> list[str]:
  """Validates and normalizes `tests` payload for `asserts` format."""
  if isinstance(parsed, Mapping):
    if "asserts" in parsed:
      assert_list = [str(x) for x in parsed["asserts"] if str(x).strip()]
    else:
      assert_list = []
  elif isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes)):
    assert_list = [str(x) for x in parsed if str(x).strip()]
  else:
    assert_list = []
  if not assert_list:
    raise ValueError("Dataset record missing non-empty 'tests'")
  return assert_list


def _normalize_tests(raw_tests: Any, fmt: str) -> list[str] | dict[str, Any]:
  """Validates and normalizes `tests` payload for `asserts` or `stdio` format."""
  parsed = _parse_raw_tests(raw_tests)
  if fmt in ("stdio", "stdin"):
    return _normalize_stdio_tests(parsed)
  if isinstance(parsed, Mapping) and "inputs" in parsed and "outputs" in parsed:
    return _normalize_stdio_tests(parsed)
  if (
      isinstance(parsed, Sequence)
      and not isinstance(parsed, (str, bytes))
      and bool(parsed)
      and all(isinstance(x, Mapping) and ("input" in x or "output" in x) for x in parsed)
  ):
    return _normalize_stdio_tests(parsed)
  return _normalize_assert_tests(parsed)


def partition_visible_and_hidden_tests(
    normalized_tests: list[str] | dict[str, Any],
    prompt_id: str = "default",
    *,
    fmt: str = "asserts",
    visible_fraction: float = 0.5,
    min_visible: int = 1,
    max_visible: int = 2,
) -> tuple[list[str] | dict[str, Any], list[str] | dict[str, Any]]:
  """Deterministically partitions `normalized_tests` into disjoint `(visible_tests, hidden_tests)`."""
  if prompt_id in ("asserts", "stdio", "stdin") and fmt == "asserts":
    fmt = prompt_id
  if fmt in ("stdio", "stdin") or (isinstance(normalized_tests, Mapping) and normalized_tests.get("format") == "stdio"):
    assert isinstance(normalized_tests, Mapping)
    inputs = list(normalized_tests.get("inputs", []))
    outputs = list(normalized_tests.get("outputs", []))
    total = len(inputs)
    if total <= 1:
      vis_dict = {"format": "stdio", "inputs": list(inputs), "outputs": list(outputs)}
      hid_dict = {"format": "stdio", "inputs": list(inputs), "outputs": list(outputs)}
      return vis_dict, hid_dict
    k_vis = min(
        total - 1,
        max_visible,
        max(min_visible, int(round(total * visible_fraction))),
    )
    return (
        {"format": "stdio", "inputs": inputs[:k_vis], "outputs": outputs[:k_vis]},
        {"format": "stdio", "inputs": inputs[k_vis:], "outputs": outputs[k_vis:]},
    )

  assert isinstance(normalized_tests, list)
  # Deduplicate while preserving order so visible and hidden sets are disjoint when >=2 unique tests exist
  unique_tests = list(dict.fromkeys(normalized_tests))
  total = len(unique_tests)
  if total <= 1:
    return list(normalized_tests), list(normalized_tests)
  k_vis = min(
      total - 1,
      max_visible,
      max(min_visible, int(round(total * visible_fraction))),
  )
  visible_list = unique_tests[:k_vis]
  hidden_list = unique_tests[k_vis:]
  return visible_list, hidden_list


class MinimalTokenizer:
  """Lightweight tokenizer mock for coding RL dataset processing and testing."""

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    del add_special_tokens
    return [1] * max(1, len(text.split()))

  def decode(self, tokens: Sequence[int], skip_special_tokens: bool = False) -> str:
    del skip_special_tokens
    return " ".join(str(t) for t in tokens)


def process_code_rl_record(record: Mapping[str, Any]) -> dict[str, Any]:
  """Formats a raw problem dictionary into the `PythonCodeEnv` dataset schema.

  Args:
    record: Raw problem mapping containing `problem` (or `question`), `tests`,
      and optional `prompt_id`, `format` (`"asserts"` or `"stdio"`),
      `format_instructions`, `prompts`, `entry_point`, `visible_tests`, and
      `hidden_tests`.

  Returns:
    Normalized dictionary with keys `prompts`, `prompt_id`, `problem`, `tests`,
    `visible_tests`, `hidden_tests`, `format`, and `format_instructions`
    (plus `entry_point` if present).
  """
  problem = str(record.get("problem") or record.get("question") or "").strip()
  if not problem:
    raise ValueError("Dataset record missing 'problem' or 'question'")

  raw_fmt = record.get("format")
  raw_tests = record.get("tests")
  if raw_fmt is None:
    if isinstance(raw_tests, Mapping) and "inputs" in raw_tests:
      fmt = "stdio"
    else:
      fmt = "asserts"
  else:
    fmt = str(raw_fmt).strip().lower()

  if fmt not in ("asserts", "stdio"):
    raise ValueError(f"Unsupported test format: {fmt!r}")

  normalized_tests = _normalize_tests(raw_tests, fmt)
  if isinstance(normalized_tests, Mapping) and normalized_tests.get("format") == "stdio":
    fmt = "stdio"

  format_instructions = str(record.get("format_instructions") or DEFAULT_FORMAT_INSTRUCTIONS).strip()
  prompts = str(record.get("prompts") or f"{problem}\n\n{format_instructions}")
  prompt_id = str(
      record.get("prompt_id")
      if record.get("prompt_id") is not None
      else (record.get("task_id") if record.get("task_id") is not None else "item_0")
  )

  raw_vis = record.get("visible_tests") or record.get("base_tests") or record.get("public_test_cases")
  raw_hid = record.get("hidden_tests") or record.get("plus_tests") or record.get("private_test_cases")
  if raw_vis is not None and raw_hid is not None:
    visible_tests = _normalize_tests(raw_vis, fmt)
    hidden_tests = _normalize_tests(raw_hid, fmt)
  else:
    visible_tests, hidden_tests = partition_visible_and_hidden_tests(normalized_tests, prompt_id, fmt=fmt)

  out: dict[str, Any] = {
      "prompts": prompts,
      "question": problem,
      "prompt_id": prompt_id,
      "problem": problem,
      "tests": normalized_tests,
      "visible_tests": visible_tests,
      "hidden_tests": hidden_tests,
      "format": fmt,
      "format_instructions": format_instructions,
  }
  if record.get("entry_point"):
    out["entry_point"] = str(record["entry_point"])
  return out


def process_data(
    dataset_name: str,
    model_tokenizer: Any,
    template_config: Mapping[str, Any] | None,
    tmvp_config: Any,
    x: Mapping[str, Any],
) -> dict[str, Any]:
  """MaxText RL custom dataset processor entrypoint for `train_rl.py`."""
  del dataset_name, template_config
  out = process_code_rl_record(x)
  rl_cfg = getattr(tmvp_config, "rl", None)
  env_class_path = getattr(rl_cfg, "env_class_path", "") if rl_cfg is not None else ""
  if (
      not env_class_path
      and model_tokenizer is not None
      and hasattr(model_tokenizer, "apply_chat_template")
      and getattr(model_tokenizer, "chat_template", None)
  ):
    if isinstance(out["prompts"], str):
      out["prompts"] = model_tokenizer.apply_chat_template(
          [{"role": "user", "content": out["prompts"]}],
          tokenize=False,
          add_generation_prompt=True,
      )
    else:
      out["prompts"] = [
          model_tokenizer.apply_chat_template(
              [{"role": "user", "content": prompt_text}],
              tokenize=False,
              add_generation_prompt=True,
          )
          for prompt_text in out["prompts"]
      ]
  return out


def format_code_rl_dataset(dataset: Any) -> Any:
  """Dataset processor callable for `rl.dataset.dataset_processor_path` / `processing_fn`.

  Supports:
    - Individual record mappings (`dict`)
    - Grain `MapDataset` / `IterDataset` (via `.map(process_code_rl_record)`)
    - Standard Python sequences/iterables of record mappings
  """
  if isinstance(dataset, Mapping):
    return process_code_rl_record(dataset)
  if hasattr(dataset, "map") and callable(dataset.map):
    return dataset.map(process_code_rl_record)
  if isinstance(dataset, Iterable):
    return [process_code_rl_record(item) for item in dataset]
  raise TypeError(f"Unsupported dataset type for format_code_rl_dataset: {type(dataset)!r}")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
  """Loads and normalizes coding RL records from a JSONL file."""
  file_path = Path(path)
  records: list[dict[str, Any]] = []
  with file_path.open("r", encoding="utf-8") as f:
    for line in f:
      stripped = line.strip()
      if not stripped:
        continue
      raw = json.loads(stripped)
      records.append(process_code_rl_record(raw))
  return records
