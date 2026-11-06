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

"""Extracts documented MaxText train commands from a Markdown section and validates them via pyconfig."""

import argparse
import re
import shlex
import sys


def extract_train_blocks(section_text: str, module_name: str = "maxtext.train") -> list[str]:
  """Extracts fenced bash/sh code blocks containing the target module invocation."""
  blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", section_text, re.S)
  return [b.strip() for b in blocks if module_name in b]


def parse_substitutions(subst_args: list[str]) -> dict[str, str]:
  """Parses VAR=VAL CLI substitutions, raising ValueError if '=' is missing."""
  subs: dict[str, str] = {}
  for item in subst_args:
    if "=" not in item:
      raise ValueError(f"Invalid --subst '{item}': expected 'VAR=VAL' format.")
    k, v = item.split("=", 1)
    subs[k] = v
  return subs


def build_pyconfig_argv(block: str, subs: dict[str, str], module_name: str = "maxtext.train") -> tuple[list[str], bool]:
  """Applies variable substitutions, checks for inline comment truncation, and slices argv after module_name."""
  text = block
  for k, v in sorted(subs.items(), key=lambda kv: -len(kv[0])):
    text = text.replace(f"${{{k}}}", v).replace(f"${k}", v)

  # Detect '#' comment lines inside '\' line continuations that truncate subsequent flags.
  lines = text.splitlines()
  prev_continued = False
  comment_after_backslash = False
  for line in lines:
    stripped = line.strip()
    if prev_continued and stripped.startswith("#"):
      comment_after_backslash = True
    prev_continued = stripped.endswith("\\") and not stripped.startswith("#")

  joined = re.sub(r"\\\s*\n", " ", text)
  cmd_line = ""
  for line in joined.splitlines():
    if module_name in line and not line.strip().startswith("#"):
      cmd_line = line.strip()
      break
  if not cmd_line:
    raise ValueError(f"No executable line containing '{module_name}' found in block.")

  full_argv = shlex.split(cmd_line)
  if module_name not in full_argv:
    raise ValueError(f"Module '{module_name}' not found as a token in parsed command: {full_argv}")
  mod_idx = full_argv.index(module_name)
  return ["train.py", *full_argv[mod_idx + 1 :]], comment_after_backslash


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("doc", help="Markdown file path")
  parser.add_argument("--start", required=True, help="Start substring marker")
  parser.add_argument("--end", required=True, help="End substring marker")
  parser.add_argument("--module", default="maxtext.train", help="Python module token to locate")
  parser.add_argument("--subst", action="append", default=[], help="VAR=VAL shell variable substitution")
  parser.add_argument("--keys", default="model_name,override_model_config", help="Comma-separated config keys to print")
  args = parser.parse_args()

  with open(args.doc, "r", encoding="utf-8") as f:
    doc = f.read()

  if args.start not in doc:
    print(f"Error: --start marker '{args.start}' not found in {args.doc}.", file=sys.stderr)
    return 2
  start_idx = doc.index(args.start)
  if args.end not in doc[start_idx:]:
    print(f"Error: --end marker '{args.end}' not found after --start in {args.doc}.", file=sys.stderr)
    return 2
  end_idx = doc.index(args.end, start_idx)
  section = doc[start_idx:end_idx]

  try:
    subs = parse_substitutions(args.subst)
  except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    return 2

  blocks = extract_train_blocks(section, args.module)
  if not blocks:
    print(f"No '{args.module}' blocks found between '{args.start}' and '{args.end}'.", file=sys.stderr)
    return 1

  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel

  keys = [k.strip() for k in args.keys.split(",") if k.strip()]
  rc = 0
  for i, block in enumerate(blocks, 1):
    argv, truncated = build_pyconfig_argv(block, subs, args.module)
    if truncated:
      print(f"[Block {i}] FAIL: '#' comment inside '\\' continuation truncates subsequent arguments.")
      rc = 1
    cfg = pyconfig.initialize(argv)
    kv_str = ", ".join(f"{k}={getattr(cfg, k, '<unset>')!r}" for k in keys)
    print(f"[Block {i}] ({len(argv) - 1} args) -> {kv_str}")
  return rc


if __name__ == "__main__":
  sys.exit(main())
