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

"""Validates MaxText .agents/wiki/ structure, links, frontmatter, budgets, live code_refs, and git drift."""

import argparse
import datetime
from pathlib import Path
import re
import subprocess
import sys

ALLOWED_CATEGORIES = ("troubleshooting", "parity", "patterns", "conversion")
ALLOWED_PHASES = ("phase1", "phase2", "phase3", "phase4", "benchmark", "pr_cleanup")
ALLOWED_STACKS = ("linen", "m3")
ALLOWED_STATUSES = ("active", "graduated", "superseded")
REQUIRED_FRONTMATTER_KEYS = (
    "title",
    "category",
    "phases",
    "applies_to",
    "tags",
    "code_refs",
    "config_keys",
    "verified_on",
    "last_verified",
)
CATEGORY_REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "troubleshooting": (
        "## Symptom / When to Use",
        "## Action Ladder",
        "## Guardrails & Cleanup",
        "## Related",
    ),
    "parity": (
        "## Symptom & Divergence Signature",
        "## Fast Isolation Probe",
        "## Root Cause & Fix Pattern",
        "## Verification & Negative Control",
        "## Related",
    ),
    "patterns": (
        "## When You Need This",
        "## Canonical MaxText Implementation",
        "## Common Traps & Invariants",
        "## Related",
    ),
    "conversion": (
        "## When You Need This",
        "## Canonical MaxText Implementation",
        "## Common Traps & Invariants",
        "## Related",
    ),
}
MAX_INDEX_LINES = 150
MIN_LEAF_LINES = 25
COMPACTION_WARN_LINES = 120
MAX_LEAF_LINES = 140
SNAKE_CASE_MD_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*\.md$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LEAKED_PATH_RE = re.compile(r"(?:/usr/local/google/home/[a-zA-Z0-9_]+|/home/[a-zA-Z0-9_]+|/google/src/cloud/)")


def strip_fenced_code_blocks(markdown_text: str) -> str:
  """Removes fenced code blocks so template examples are not parsed as live links or headings."""
  out_lines: list[str] = []
  fence_marker: str | None = None
  for line in markdown_text.splitlines():
    stripped = line.lstrip()
    match = re.match(r"^(`{3,}|~{3,})", stripped)
    if match:
      token = match.group(1)
      if fence_marker is None:
        fence_marker = token
        continue
      if token[0] == fence_marker[0] and len(token) >= len(fence_marker):
        fence_marker = None
        continue
    if fence_marker is None:
      out_lines.append(line)
  return "\n".join(out_lines)


def _strip_quotes(val: str) -> str:
  val = val.strip()
  if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
    return val[1:-1]
  return val


def parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]] | None, str]:
  """Parses simple YAML frontmatter (scalars, inline lists, and block lists) using stdlib only."""
  lines = text.splitlines()
  if not lines or lines[0].strip() != "---":
    return None, text

  end_idx: int | None = None
  for idx in range(1, len(lines)):
    if lines[idx].strip() == "---":
      end_idx = idx
      break
  if end_idx is None:
    return None, text

  meta: dict[str, str | list[str]] = {}
  current_list_key: str | None = None
  for raw_line in lines[1:end_idx]:
    no_comment = re.sub(r"\s+#.*$", "", raw_line).rstrip()
    if not no_comment.strip():
      continue
    if no_comment.startswith("  - ") or no_comment.startswith("- "):
      if current_list_key is None or not isinstance(meta.get(current_list_key), list):
        return None, text
      item = _strip_quotes(no_comment.lstrip()[2:])
      cast_list = meta[current_list_key]
      if isinstance(cast_list, list):
        cast_list.append(item)
      continue

    if ":" not in no_comment:
      return None, text
    key, raw_val = no_comment.split(":", 1)
    key = key.strip()
    raw_val = raw_val.strip()
    current_list_key = None
    if not raw_val:
      meta[key] = []
      current_list_key = key
    elif raw_val.startswith("[") and raw_val.endswith("]"):
      inner = raw_val[1:-1].strip()
      meta[key] = [_strip_quotes(part) for part in inner.split(",") if part.strip()] if inner else []
    else:
      meta[key] = _strip_quotes(raw_val)

  body = "\n".join(lines[end_idx + 1 :])
  return meta, body


def load_known_config_keys(repo_root: Path) -> set[str]:
  """Extracts valid MaxText config keys from base.yml and types.py."""
  keys: set[str] = set()
  for rel in ("src/maxtext/configs/base.yml", "src/maxtext/configs/types.py"):
    cfg_file = repo_root / rel
    if cfg_file.is_file():
      for match in re.finditer(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", cfg_file.read_text(encoding="utf-8"), re.M):
        keys.add(match.group(1))
  return keys


def check_markdown_links(md_path: Path, text_no_code: str) -> list[str]:
  """Verifies all relative markdown links resolve on disk and forbid file:// or absolute paths."""
  errors: list[str] = []
  for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text_no_code):
    target = match.group(1).strip().split()[0]
    if target.startswith(("https://", "http://", "mailto:", "#")):
      if target.startswith("http://"):
        errors.append(f"{md_path}: insecure or internal http:// link '{target}' (use relative path or https://)")
      continue
    if target.startswith(("file://", "/")):
      errors.append(f"{md_path}: non-portable link '{target}' (must be a bare relative path)")
      continue
    rel_part = target.split("#", 1)[0]
    if not rel_part:
      continue
    resolved = (md_path.parent / rel_part).resolve()
    if not resolved.exists():
      errors.append(f"{md_path}: broken relative link '{target}' -> '{resolved}' does not exist")
  return errors


def validate_leaf_frontmatter(
    rel_leaf: Path,
    meta: dict[str, str | list[str]],
    repo_root: Path,
    known_config_keys: set[str],
) -> list[str]:
  """Checks frontmatter types, allowed enums, status lifecycle, and live code_refs / config_keys."""
  errors: list[str] = []
  missing = [k for k in REQUIRED_FRONTMATTER_KEYS if k not in meta]
  if missing:
    errors.append(f"{rel_leaf}: missing frontmatter keys: {missing}")
    return errors

  expected_cat = rel_leaf.parts[0]
  if meta["category"] != expected_cat:
    errors.append(f"{rel_leaf}: frontmatter category '{meta['category']}' must match directory '{expected_cat}'")

  if not isinstance(meta["title"], str) or not meta["title"].strip():
    errors.append(f"{rel_leaf}: 'title' must be a non-empty string")

  status = meta.get("status", "active")
  if not isinstance(status, str) or status not in ALLOWED_STATUSES:
    errors.append(f"{rel_leaf}: 'status' must be one of {ALLOWED_STATUSES}, got '{status}'")
  elif status in ("graduated", "superseded"):
    sup = meta.get("superseded_by", "")
    if not isinstance(sup, str) or not sup.strip():
      errors.append(f"{rel_leaf}: status='{status}' requires non-empty 'superseded_by' reference")

  phases = meta["phases"]
  if not isinstance(phases, list) or not phases or any(p not in ALLOWED_PHASES for p in phases):
    errors.append(f"{rel_leaf}: 'phases' must be a non-empty subset of {ALLOWED_PHASES}, got {phases}")

  stacks = meta["applies_to"]
  if not isinstance(stacks, list) or not stacks or any(s not in ALLOWED_STACKS for s in stacks):
    errors.append(f"{rel_leaf}: 'applies_to' must be a non-empty subset of {ALLOWED_STACKS}, got {stacks}")

  for list_key in ("tags", "verified_on"):
    val = meta[list_key]
    if not isinstance(val, list) or not val:
      errors.append(f"{rel_leaf}: '{list_key}' must be a non-empty list")

  last_verified = meta["last_verified"]
  if not isinstance(last_verified, str) or not DATE_RE.match(last_verified):
    errors.append(f"{rel_leaf}: 'last_verified' must be YYYY-MM-DD, got '{last_verified}'")

  code_refs = meta["code_refs"]
  if not isinstance(code_refs, list):
    errors.append(f"{rel_leaf}: 'code_refs' must be a list")
  else:
    for ref in code_refs:
      if not (repo_root / ref).exists():
        errors.append(f"{rel_leaf}: stale code_refs entry '{ref}' does not exist in repo")

  config_keys = meta["config_keys"]
  if not isinstance(config_keys, list):
    errors.append(f"{rel_leaf}: 'config_keys' must be a list")
  elif known_config_keys:
    for cfg_key in config_keys:
      if cfg_key not in known_config_keys:
        errors.append(f"{rel_leaf}: stale config_keys entry '{cfg_key}' not found in base.yml or types.py")

  return errors


def check_git_drift(
    rel_leaf: Path,
    meta: dict[str, str | list[str]],
    repo_root: Path,
    stale_days: int,
) -> list[str]:
  """Checks git commit history of code_refs since last_verified and flags aged or retired pages."""
  warnings: list[str] = []
  status = meta.get("status", "active")
  if status in ("graduated", "superseded"):
    warnings.append(
        f"{rel_leaf}: status='{status}' (superseded_by='{meta.get('superseded_by')}') — prune once callers migrate"
    )

  last_verified = meta.get("last_verified", "")
  if not isinstance(last_verified, str) or not DATE_RE.match(last_verified):
    return warnings

  try:
    verified_date = datetime.date.fromisoformat(last_verified)
    age_days = (datetime.date.today() - verified_date).days
    if age_days > stale_days:
      warnings.append(f"{rel_leaf}: last_verified={last_verified} ({age_days}d > {stale_days}d stale threshold)")
  except ValueError:
    return warnings

  code_refs = meta.get("code_refs", [])
  if not isinstance(code_refs, list):
    return warnings

  for ref in code_refs:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "log", f"--since={last_verified}T23:59:59Z", "--format=%cs", "--", ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
      continue
    commit_dates = [d.strip() for d in proc.stdout.splitlines() if d.strip() and d.strip() > last_verified]
    if commit_dates:
      warnings.append(
          f"{rel_leaf}: code_ref '{ref}' has {len(commit_dates)} commit(s) after "
          f"last_verified={last_verified} (latest: {commit_dates[0]})"
      )
  return warnings


def audit_wiki(
    repo_root: Path,
    wiki_dir: Path,
    check_drift: bool = False,
    stale_days: int = 180,
) -> tuple[list[str], list[str]]:
  """Returns (errors, advisory_warnings) across wiki_dir."""
  errors: list[str] = []
  warnings: list[str] = []
  index_path = wiki_dir / "index.md"
  protocol_path = wiki_dir / "PROTOCOL.md"

  if not index_path.is_file():
    return [f"Missing required hub file: {index_path}"], warnings
  if not protocol_path.is_file():
    errors.append(f"Missing required protocol file: {protocol_path}")

  index_text = index_path.read_text(encoding="utf-8")
  index_lines = len(index_text.splitlines())
  if index_lines > MAX_INDEX_LINES:
    errors.append(f"{index_path}: {index_lines} lines exceeds cap of {MAX_INDEX_LINES} lines")

  topic_section = index_text.split("## Topic Index", 1)[1] if "## Topic Index" in index_text else ""
  if not topic_section:
    errors.append(f"{index_path}: missing '## Topic Index' section")

  known_config_keys = load_known_config_keys(repo_root)
  for md_file in sorted(wiki_dir.rglob("*.md")):
    raw = md_file.read_text(encoding="utf-8")
    no_code = strip_fenced_code_blocks(raw)
    if LEAKED_PATH_RE.search(no_code):
      errors.append(f"{md_file}: contains machine-specific or personal path (use $USER / placeholders)")
    errors.extend(check_markdown_links(md_file, no_code))

    if md_file in (index_path, protocol_path):
      continue

    rel_leaf = md_file.relative_to(wiki_dir)
    if len(rel_leaf.parts) != 2 or rel_leaf.parts[0] not in ALLOWED_CATEGORIES:
      errors.append(
          f"{rel_leaf}: leaf pages must live directly under one of {ALLOWED_CATEGORIES} (e.g. troubleshooting/foo.md)"
      )
      continue
    if not SNAKE_CASE_MD_RE.match(rel_leaf.name):
      errors.append(f"{rel_leaf}: filename must be snake_case.md")

    line_count = len(raw.splitlines())
    if line_count < MIN_LEAF_LINES or line_count > MAX_LEAF_LINES:
      errors.append(f"{rel_leaf}: {line_count} lines outside allowed [{MIN_LEAF_LINES}, {MAX_LEAF_LINES}] budget")
    elif line_count >= COMPACTION_WARN_LINES:
      warnings.append(
          f"{rel_leaf}: {line_count}/{MAX_LEAF_LINES} lines — compact older bullets into table rows on next merge"
      )

    meta, body = parse_frontmatter(raw)
    if meta is None:
      errors.append(f"{rel_leaf}: missing or malformed YAML frontmatter")
      continue
    errors.extend(validate_leaf_frontmatter(rel_leaf, meta, repo_root, known_config_keys))

    rel_posix = rel_leaf.as_posix()
    status = meta.get("status", "active")
    in_topic_index = f"({rel_posix})" in topic_section
    if status == "active" and not in_topic_index:
      errors.append(f"{rel_leaf}: active page not registered in index.md '## Topic Index' as ({rel_posix})")
    elif status in ("graduated", "superseded") and in_topic_index:
      errors.append(f"{rel_leaf}: status='{status}' page must be removed from active index.md '## Topic Index'")

    if check_drift:
      warnings.extend(check_git_drift(rel_leaf, meta, repo_root, stale_days))

    body_no_code = strip_fenced_code_blocks(body)
    headings = [line.strip() for line in body_no_code.splitlines() if line.startswith("## ")]
    for req_sec in CATEGORY_REQUIRED_SECTIONS[rel_leaf.parts[0]]:
      if not any(h == req_sec or h.startswith(f"{req_sec} ") for h in headings):
        errors.append(f"{rel_leaf}: missing required section heading '{req_sec}'")

  return errors, warnings


def validate_wiki(repo_root: Path, wiki_dir: Path) -> list[str]:
  """Backwards-compatible wrapper returning structural/grounding errors only."""
  errors, _ = audit_wiki(repo_root, wiki_dir, check_drift=False)
  return errors


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  default_repo = Path(__file__).resolve().parents[3]
  parser.add_argument("--repo-root", type=Path, default=default_repo, help="Path to MaxText repository root")
  parser.add_argument("--wiki-dir", type=Path, default=None, help="Override path to .agents/wiki directory")
  parser.add_argument(
      "--check-drift",
      action="store_true",
      help="Audit git history of code_refs since last_verified and flag aged/graduated pages",
  )
  parser.add_argument("--stale-days", type=int, default=180, help="Age threshold in days for --check-drift")
  parser.add_argument("--strict-drift", action="store_true", help="Exit non-zero if advisory drift warnings exist")
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  wiki_dir = (args.wiki_dir or (repo_root / ".agents/wiki")).resolve()
  errors, warnings = audit_wiki(repo_root, wiki_dir, check_drift=args.check_drift, stale_days=args.stale_days)
  for warn in warnings:
    print(f"ADVISORY: {warn}", file=sys.stderr)

  if errors:
    print(f"Wiki validation FAILED ({len(errors)} issue(s)):", file=sys.stderr)
    for err in errors:
      print(f"  - {err}", file=sys.stderr)
    return 1
  if args.strict_drift and warnings:
    print(f"Wiki strict-drift check FAILED ({len(warnings)} advisory warning(s)).", file=sys.stderr)
    return 1

  leaf_count = sum(1 for p in wiki_dir.rglob("*.md") if p.name not in ("index.md", "PROTOCOL.md"))
  print(f"Wiki validation PASSED ({leaf_count} leaf page(s) verified against {repo_root.name}).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
