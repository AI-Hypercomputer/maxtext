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

"""Rejects review verdicts that do not cover a change's full diff.

A Gate C (or any fresh-reader) review report must contain exactly one fenced
`review-scope` block:

```review-scope
target: PR#1234
base: <parent sha>
head: <reviewed head sha>
verdict: APPROVED          (or CHANGES_REQUESTED)
files:
- path/to/changed_file.py: what was checked in this file
- path/to/other.py: what was checked
```

The gate fails (exit 1) when:
- MISSING_OR_MULTI_SCOPE: zero or several scope blocks (one target per report).
- STALE_HEAD: the reviewed head differs from the live head.
- WRONG_BASE: the review diffed against something other than the parent
  (for example a range-diff against the change's previous version).
- EMPTY_DIFF: the diff against the parent is empty.
- UNCOVERED_FILE: a changed file is missing or has an empty/skip note.
- UNRESOLVED_LOCAL_FINDING: a RED/ORANGE finding (or open YELLOW defect /
  unexplained dismissed YELLOW) in `--findings-file` is unresolved, or that
  file reviewed another head.
- INVALID_VERDICT: `verdict` is not `APPROVED` or `CHANGES_REQUESTED`.
- NOT_APPROVED_FOR_PUSH: `--strict-pre-push` or `--findings-file` requires
  `verdict == APPROVED`.

Exit code: 0 = PASS, 1 = gate failed, 2 = usage or I/O error.
"""

import argparse
from collections.abc import Sequence
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

_SCOPE_BLOCK_RE = re.compile(r"^```review-scope[ \t]*\n(.*?)^```", re.M | re.S)
_SKIP_NOTE_RE = re.compile(
    r"^\s*(?:|-|n/?a|none|skip(?:ped)?|not reviewed|todo|tbd|same as before"
    r"|unchanged|pure rebase|no changes?)\s*\.?\s*$",
    re.I,
)
_FINDING_RE = re.compile(
    r"^- severity=(RED|ORANGE|YELLOW)\s+file=(\S+)\s+line=(\d+)\s+"
    r"status=(open|fixed|dismissed)(?:\s+reason=\"([^\"]*)\")?\s*::\s*(.+)$"
)
_FINDINGS_HEADER_RE = re.compile(r"^LOCAL_REVIEW\s+target=(\S+)\s+head=(\S+)\s*$", re.M)
_VALID_VERDICTS = frozenset({"APPROVED", "CHANGES_REQUESTED"})
_DEFECT_YELLOW_RE = re.compile(
    r"\b(crash|error|bug|wrong|broken|missing\s+dep|strict_deps|build\s+dep|"
    r"validator|dead\s+code|unwired|no-op|corrupt|leak|regression|"
    r"audit_pr_review_hygiene|audit_m3_contracts|"
    r"audit_repo_and_ci_contracts)\b",
    re.IGNORECASE,
)
_PURE_NIT_MARKER_RE = re.compile(r"(\[nit\]|\bpure\s+(?:style\s+)?nit\b)", re.IGNORECASE)


def _default_repo_root() -> str:
  """Resolves MaxText repository root from __file__ ancestors or CWD."""
  starts = (
      os.path.dirname(os.path.abspath(__file__)),
      os.path.abspath(os.getcwd()),
  )
  for start in starts:
    cur = start
    while True:
      if os.path.isfile(os.path.join(cur, "src", "maxtext", "configs", "types.py")):
        return cur
      parent = os.path.dirname(cur)
      if parent == cur:
        break
      cur = parent
  return os.path.abspath(os.getcwd())


def parse_scope_blocks(text: str) -> list[dict[str, Any]]:
  """Parses every `review-scope` block into a dict."""
  blocks: list[dict[str, Any]] = []
  for m in _SCOPE_BLOCK_RE.finditer(text):
    block: dict[str, Any] = {"files": {}}
    in_files = False
    for raw in m.group(1).splitlines():
      line = raw.rstrip()
      if not line.strip():
        continue
      if in_files and line.lstrip().startswith("- "):
        item = line.lstrip()[2:]
        path, _, note = item.partition(":")
        block["files"][path.strip().strip("`")] = note.strip()
        continue
      key, sep, val = line.partition(":")
      if not sep:
        continue
      key = key.strip().lower()
      in_files = key == "files"
      if not in_files:
        block[key] = val.strip()
    blocks.append(block)
  return blocks


def _git(repo: str, *args: str) -> str:
  """Runs git in `repo` and returns stripped stdout."""
  git_bin = shutil.which("git")
  if not git_bin:
    raise OSError("git not found")
  proc = subprocess.run(
      [git_bin, "-c", "safe.directory=*", "-C", repo, *args],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise OSError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
  return proc.stdout.strip()


def git_changed_files(repo: str, base: str, head: str) -> tuple[str, list[str]]:
  """Returns (merge-base sha, files changed from merge-base to head)."""
  mb = _git(repo, "merge-base", base, head)
  names = _git(repo, "diff", "--name-only", f"{mb}..{head}")
  return mb, [n for n in names.splitlines() if n.strip()]


def _same_rev(repo: str, a: str, b: str) -> bool:
  if not a or not b:
    return False
  if repo:
    try:
      return _git(repo, "rev-parse", a) == _git(repo, "rev-parse", b)
    except OSError:
      pass
  n = min(len(a), len(b))
  return n >= 7 and a[:n] == b[:n] or a == b


def _path_matches(listed: str, changed: str) -> bool:
  while listed.startswith("./"):
    listed = listed[2:]
  while changed.startswith("./"):
    changed = changed[2:]
  return changed == listed or changed.endswith("/" + listed) or (listed.endswith("/" + changed))


def check_findings_file(path: str, head: str, repo: str) -> list[dict[str, Any]]:
  """Validates the local MaxText-criteria findings file."""
  issues: list[dict[str, Any]] = []
  with open(path, "r", encoding="utf-8") as f:
    text = f.read()
  hdr = _FINDINGS_HEADER_RE.search(text)
  if not hdr:
    return [
        {
            "code": "UNRESOLVED_LOCAL_FINDING",
            "message": f"{path} lacks the `LOCAL_REVIEW target=.. head=..` header.",
        }
    ]
  if head and not _same_rev(repo, hdr.group(2), head):
    issues.append(
        {
            "code": "UNRESOLVED_LOCAL_FINDING",
            "message": (
                f"Local review ran on head {hdr.group(2)}, not the live head" f" {head}; rerun it on the current head."
            ),
        }
    )
  for line in text.splitlines():
    m = _FINDING_RE.match(line.strip())
    if not m:
      continue
    sev, fpath, ln, status, reason, title = m.groups()
    unexplained = status == "dismissed" and not (reason or "").strip()
    if sev in ("RED", "ORANGE"):
      if status == "open" or unexplained:
        issues.append(
            {
                "code": "UNRESOLVED_LOCAL_FINDING",
                "message": (
                    f"{sev} local finding at {fpath}:{ln} is {status}"
                    f"{'' if status == 'open' else ' without a reason'}: {title}"
                ),
            }
        )
    elif sev == "YELLOW":
      if unexplained:
        issues.append(
            {
                "code": "UNRESOLVED_LOCAL_FINDING",
                "message": (f"YELLOW local finding at {fpath}:{ln} is dismissed without a" f" reason: {title}"),
            }
        )
      elif status == "open" and _DEFECT_YELLOW_RE.search(title) and not _PURE_NIT_MARKER_RE.search(title):
        issues.append(
            {
                "code": "UNRESOLVED_LOCAL_FINDING",
                "message": (
                    f"YELLOW local finding at {fpath}:{ln} describes a"
                    " functional/config/contract defect and cannot be downgraded"
                    f" below ORANGE: {title}"
                ),
            }
        )
  return issues


def run_gate(
    report_text: str,
    changed: Sequence[str],
    expected_base: str,
    expected_head: str,
    *,
    repo: str = "",
    findings_file: str = "",
    allow_globs: Sequence[str] = (),
    strict_pre_push: bool = False,
) -> dict[str, Any]:
  """Runs all scope checks and returns a JSON-able report."""
  issues: list[dict[str, Any]] = []
  blocks = parse_scope_blocks(report_text)
  if len(blocks) != 1:
    issues.append(
        {
            "code": "MISSING_OR_MULTI_SCOPE",
            "message": (f"Found {len(blocks)} `review-scope` blocks; a report must review" " exactly one PR."),
        }
    )
  block = blocks[0] if blocks else {"files": {}}
  if blocks:
    raw_verdict = str(block.get("verdict", "")).strip()
    verdict_upper = raw_verdict.upper()
    if verdict_upper not in _VALID_VERDICTS:
      issues.append(
          {
              "code": "INVALID_VERDICT",
              "message": (
                  f"Verdict {raw_verdict!r} is invalid (APPROVED_WITH_FOLLOWUPS is"
                  " forbidden); use 'APPROVED' or 'CHANGES_REQUESTED'."
              ),
          }
      )
    elif (strict_pre_push or bool(findings_file)) and (verdict_upper != "APPROVED"):
      issues.append(
          {
              "code": "NOT_APPROVED_FOR_PUSH",
              "message": (f"Pre-push gate requires verdict 'APPROVED', got {raw_verdict!r}."),
          }
      )
  if not changed:
    issues.append(
        {
            "code": "EMPTY_DIFF",
            "message": "The diff against the parent is empty; check --base.",
        }
    )
  if blocks and expected_head and not _same_rev(repo, block.get("head", ""), expected_head):
    issues.append(
        {
            "code": "STALE_HEAD",
            "message": (f"Review covers head {block.get('head', '?')!r}, live head is" f" {expected_head!r}."),
        }
    )
  if blocks and expected_base and not _same_rev(repo, block.get("base", ""), expected_base):
    issues.append(
        {
            "code": "WRONG_BASE",
            "message": (
                f"Review diffed against {block.get('base', '?')!r}, but the parent"
                f" is {expected_base!r}. Review the full diff against the parent,"
                " not a range-diff against the previous version."
            ),
        }
    )
  files: dict[str, str] = block.get("files", {})
  for cf in changed:
    if any(fnmatch.fnmatch(cf, g) for g in allow_globs):
      continue
    note = next((n for p, n in files.items() if _path_matches(p, cf)), None)
    if note is None:
      issues.append(
          {
              "code": "UNCOVERED_FILE",
              "message": f"Changed file {cf} is not listed in the review scope.",
          }
      )
    elif _SKIP_NOTE_RE.match(note):
      issues.append(
          {
              "code": "UNCOVERED_FILE",
              "message": f"Changed file {cf} has a skip note: {note!r}.",
          }
      )
  extra = [p for p in files if not any(_path_matches(p, cf) for cf in changed)]
  if findings_file:
    issues.extend(check_findings_file(findings_file, expected_head, repo))
  return {
      "status": "FAIL" if issues else "PASS",
      "target": block.get("target", ""),
      "verdict": block.get("verdict", ""),
      "changed_files": len(changed),
      "listed_not_in_diff": extra,
      "issues": issues,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--report", required=True, help="Review report path.")
  parser.add_argument("--repo-root", default="", help="Git worktree root.")
  parser.add_argument("--base", default="", help="Parent git ref.")
  parser.add_argument("--head", default="", help="Live head SHA.")
  parser.add_argument("--changed-files", default="", help="File with one changed path per line.")
  parser.add_argument("--findings-file", default="", help="Local MaxText-criteria findings.")
  parser.add_argument(
      "--allow-unreviewed",
      nargs="*",
      default=[],
      help="Globs exempt from coverage (for example generated goldens).",
  )
  parser.add_argument(
      "--strict-pre-push",
      action="store_true",
      help="Require verdict == APPROVED in addition to scope coverage.",
  )
  args = parser.parse_args(argv)
  try:
    with open(args.report, "r", encoding="utf-8") as f:
      report_text = f.read()
    expected_base = args.base
    repo_root = args.repo_root
    if not repo_root and args.base and args.head and not args.changed_files:
      repo_root = _default_repo_root()
    if args.changed_files:
      with open(args.changed_files, "r", encoding="utf-8") as f:
        changed = [ln.strip() for ln in f if ln.strip()]
    elif repo_root and args.base and args.head:
      expected_base, changed = git_changed_files(repo_root, args.base, args.head)
    else:
      sys.stderr.write("Error: pass --repo-root/--base/--head or --changed-files.\n")
      return 2
    result = run_gate(
        report_text,
        changed,
        expected_base,
        args.head,
        repo=repo_root,
        findings_file=args.findings_file,
        allow_globs=args.allow_unreviewed,
        strict_pre_push=args.strict_pre_push,
    )
  except OSError as exc:
    sys.stderr.write(f"Error: {exc}\n")
    return 2
  sys.stdout.write(json.dumps(result, indent=2) + "\n")
  return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
