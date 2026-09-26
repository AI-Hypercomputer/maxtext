# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper script to collect candidate commits since Last Updated in docs/release_notes.md,
verify LLM-synthesized release notes for dropped commits, and deterministically merge them into docs/release_notes.md.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


RTD_BASE_URL = "https://maxtext.readthedocs.io/en/latest"
# `docs/` paths that Sphinx never renders as standalone pages.
NON_PAGE_DOC_PREFIXES = ("docs/_",)
# Keep bullets readable: a doc-heavy commit can otherwise touch a dozen pages.
MAX_DOC_LINKS = 3

EXCLUDED_PATH_PREFIXES = (
    "tests/",
    ".github/",
    ".gemini/",
    "src/maxtext/training_engine/",
    "src/maxtext/experimental/",
    "experimental/",
    "src/dependencies/dockerfiles/",
    "src/dependencies/scripts/",
    "tools/",
)

EXCLUDED_FILES = (
    "pytest.ini",
    "README.md",
    "docs/release_notes.md",
    "*Dockerfile",
)

EXCLUDED_PATHSPECS = [f":(exclude){p}" for p in (*EXCLUDED_PATH_PREFIXES, *EXCLUDED_FILES)]

COMMIT_DELIMITER = "===COMMIT_DELIMITER==="
FIELD_SEP = "===FIELD_SEP==="
FILES_SEP = "===FILES_SEP==="

RELEASE_NOTES_RELPATH = os.path.join("docs", "release_notes.md")
RAW_COMMITS_FILE = "raw_commits.txt"
POLISHED_NOTES_FILE = "new_notes.md"
LAST_UPDATED_RE = re.compile(r"\*{0,2}Last Updated\*{0,2}\s*:\s*\*{0,2}`?([0-9a-f]{7,40})`?", re.IGNORECASE)
MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #\d+ from [^\s]+\s*")

SECTION_ORDER = ("#### Changes", "#### Bug Fixes", "#### Deprecations")
CATEGORY_ORDER = (
    "##### Models",
    "##### Pre-Training",
    "##### Post-Training",
    "##### Multimodal",
    "##### Performance",
    "##### Checkpointing / Goodput",
    "##### Usability",
)


def get_head_commit() -> str:
  """Returns the current HEAD commit hash."""
  result = subprocess.run(
      ["git", "rev-parse", "--short=9", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
  )
  return result.stdout.strip()


def get_last_updated_commit() -> Optional[str]:
  """Reads `**Last Updated**: <commit_hash>` from docs/release_notes.md."""
  if os.path.exists(RELEASE_NOTES_RELPATH):
    try:
      with open(RELEASE_NOTES_RELPATH, "r", encoding="utf-8") as f:
        content = f.read()
      match = LAST_UPDATED_RE.search(content)
      if match:
        commit_hash = match.group(1)
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_hash}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0:
          return commit_hash
    except OSError:
      pass

  return None


def run_git_log() -> str:
  """Runs git log --first-parent for <last_commit>..HEAD with excluded pathspecs."""
  last_commit = get_last_updated_commit()
  if not last_commit:
    print(f"Error: Could not find a valid '**Last Updated**: <commit_hash>' in {RELEASE_NOTES_RELPATH}.")
    sys.exit(1)
  # %h: short commit hash, %P: parent hashes, %s: subject, %b: body
  pretty_format = f"{COMMIT_DELIMITER}%h{FIELD_SEP}%P{FIELD_SEP}%s{FIELD_SEP}%b{FILES_SEP}"
  cmd = [
      "git",
      "log",
      "--first-parent",
      "--diff-merges=first-parent",
      "--no-renames",
      "--name-only",
      f"--pretty=format:{pretty_format}",
      f"{last_commit}..HEAD",
      "--",
      ".",
      *EXCLUDED_PATHSPECS,
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return result.stdout


def is_followup_or_cleanup_subject(subject: str) -> bool:
  """Returns True if a commit subject is a PR review follow-up, comment cleanup, or lint/test-only fix."""
  s = subject.strip().lower()
  desc = re.sub(r"^[a-z]+(?:\([^)]+\))?!?:\s*", "", s).strip()
  cleanup_patterns = [
      r"^(?:address(?:ed|ing)?|review|pr)\s+(?:code\s+)?(?:review\s+|pr\s+)?(?:feedback|comments?)",
      r"^clean\s*up\s+(?:comments?|descriptions?|docstrings?|code|formatting|imports?)",
      r"^(?:fix(?:ed|es)?\s+)?(?:pylint|pyink|lint(?:er|ing)?|formatting|reformat|typos?|nits?)\b",
      r"^format\b[^:]*\blint\b",
      r"^(?:add|fix(?:ed|es)?|improve)\s+(?:[a-z0-9_.-]+\s+){0,2}(?:tests?|testing|test_[a-z0-9_]+)\b"
      r"(?!.*\band\s+(?:add|enable|fix|implement|improve|remove|support|update)\w*\b)",
      r"^(?:update\s+(?:src|tests|docs)/|update$|merge\s+(?:branch|remote-tracking\s+branch)\b|reverts?\b)",
      r"^(?:no\s+public\s+description|suppress\s+new\s+pyrefly|add\s+auto_gha\s+prefix)\b",
      r"\b(?:broken\s+(?:[a-z]+\s+)?links?|dead\s+links?|toctree|sphinx|myst)\b",
  ]
  return any(re.search(pat, desc) for pat in cleanup_patterns)


def normalize_subject(subject: str, body: str = "") -> str:
  """Cleans up commit subjects (extracts Copybara import titles and strips inline bullet lists)."""
  s = subject.strip()
  if s.lower().startswith("copybara import of the project"):
    # Look for the first non-empty line after '<hash> by <author>:'
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and ln.strip() != "--"]
    for i, ln in enumerate(lines):
      if re.match(r"^[0-9a-f]{7,40}\s+by\s+", ln) and i + 1 < len(lines):
        s = lines[i + 1]
        break
  s = re.sub(r"^#+\s*description\s*:?\s*", "", s, flags=re.IGNORECASE).strip()
  # If a single-paragraph commit message includes bullet points (` - `), keep only the headline
  s = re.split(r"\s+-\s+(?=[A-Z`'])", s, maxsplit=1)[0].strip()
  return s


def is_excluded_commit(subject: str, body: str = "") -> bool:
  """Returns True if a commit should be completely excluded from release notes."""
  clean = MERGE_SUBJECT_RE.sub("", subject).strip()
  s = normalize_subject(clean, body).lower()
  if not s or is_followup_or_cleanup_subject(s):
    return True
  if re.match(r"^(?:style|chore|test|ci|build)(?:\([^)]+\))?!?:", s):
    return not any(
        kw in f"{s} {body.lower()}" for kw in ("support", "upgrade", "enable", "migration", "release", "tutorial")
    )
  return False


def resolve_merge_commit(subject: str, body: str, parent1: str, parent2: str) -> tuple[str, str]:
  """Resolves a merge commit's subject and body from its body headline or the merged PR branch."""
  first_line = body.splitlines()[0].strip() if body else ""
  if first_line and not is_followup_or_cleanup_subject(first_line):
    return first_line, body

  cmd = [
      "git",
      "log",
      f"{parent1}..{parent2}",
      "--no-merges",
      "--reverse",
      f"--pretty=format:{COMMIT_DELIMITER}%s{FIELD_SEP}%b",
      "--",
      ".",
      *EXCLUDED_PATHSPECS,
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  branch_commits = []
  for raw in result.stdout.split(COMMIT_DELIMITER):
    if FIELD_SEP not in raw:
      continue
    subj, branch_body = raw.strip().split(FIELD_SEP, 1)
    if not subj.strip():
      continue
    if not is_followup_or_cleanup_subject(subj):
      return subj.strip(), clean_body_text(branch_body) or body
    branch_commits.append((subj.strip(), branch_body))

  if branch_commits:
    subj, branch_body = branch_commits[0]
    return subj, clean_body_text(branch_body) or body

  return subject, body


def extract_rtd_links(changed_files: List[str]) -> str:
  """Extracts ReadTheDocs documentation links from changed `docs/` files."""
  links: List[str] = []
  for f in changed_files:
    if not f.startswith("docs/") or not f.endswith((".md", ".rst", ".ipynb")):
      continue
    if f.startswith(NON_PAGE_DOC_PREFIXES) or os.path.basename(f) == "README.md" or not os.path.exists(f):
      continue
    stem_path = re.sub(r"\.(?:md|rst|ipynb)$", "", f[len("docs/") :])
    url = f"{RTD_BASE_URL}/{stem_path}.html"
    label = os.path.basename(stem_path).replace("_", " ")
    link = f"[{label}]({url})"
    if link not in links:
      links.append(link)
    if len(links) == MAX_DOC_LINKS:
      break
  return ", ".join(links)


def clean_body_text(body: str) -> str:
  """Removes PiperOrigin-RevId and Copybara trailers from commit body."""
  return "\n".join(
      line
      for line in body.splitlines()
      if not line.strip().startswith(("PiperOrigin-RevId:", "COPYBARA_INTEGRATE_REVIEW="))
  ).strip()


def clean_commit_subject(subject: str, body: str) -> str:
  """Cleans up a commit subject by stripping merge prefixes and inline PR numbers."""
  clean_subject = re.sub(r"^Merge pull request #\d+ from [^\s]+\s*", "", subject).strip()
  if not clean_subject and body:
    clean_subject = body.strip().splitlines()[0]
  clean_subject = normalize_subject(clean_subject, body)
  clean_subject = re.sub(r"^PR\s*#\d+\s*:\s*|\s*\(#\d+\)$", "", clean_subject, flags=re.IGNORECASE).strip()
  return clean_subject


def extract_ref_id(commit_hash: str, subject: str, body: str) -> str:
  """Returns `#<PR>` if a GitHub PR number is present, otherwise the short commit hash."""
  for pattern, text in (
      (r"Merge pull request #(\d+)", subject),
      (r"\(#(\d+)\)", subject),
      (r"^PR\s*#(\d+)", subject),
      (r"(?:PR|pull request|#) ?#?(\d{4,5})\b", body),
  ):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
      return f"#{match.group(1)}"
  return commit_hash


def parse_commits(raw_log: str) -> List[Dict[str, Any]]:
  """Parses `run_git_log` output into cleaned changelog candidate entries."""
  commits = []

  for entry in raw_log.split(COMMIT_DELIMITER):
    entry = entry.strip()
    if not entry:
      continue

    # 1. Unpack `<hash>===FIELD_SEP===<parents>===FIELD_SEP===<subject>===FIELD_SEP===<body>===FILES_SEP===<files>`
    parts = entry.split(FIELD_SEP)
    if len(parts) < 4:
      continue
    commit_hash = parts[0].strip()
    parents = parts[1].strip().split()
    subject = parts[2].strip()
    body_and_files = parts[3].split(FILES_SEP, 1)
    raw_body = body_and_files[0]
    files_block = body_and_files[1] if len(body_and_files) > 1 else ""
    changed_files = [line.strip() for line in files_block.splitlines() if line.strip()]

    # 2. Strip internal trailers (`PiperOrigin-RevId`) and extract tracking ref_id (`#<PR>` or `<hash>`)
    body = clean_body_text(raw_body)
    ref_id = extract_ref_id(commit_hash, subject, raw_body)

    # 3. For merge commits (2+ parents), replace `"Merge pull request #..."` with the real PR title
    if len(parents) > 1:
      subject, body = resolve_merge_commit(subject, body, parents[0], parents[1])

    # 4. Exclude review follow-ups, lint cleanups, and minor chores
    if is_excluded_commit(subject, body):
      if len(parents) > 1 and MERGE_SUBJECT_RE.match(subject):
        print(
            f"Warning: dropped merge commit {commit_hash} ({ref_id}) -- no PR title could be "
            f"resolved from its branch. Check it manually if it should appear in the notes.",
            file=sys.stderr,
        )
      continue

    commits.append(
        {
            "ref_id": ref_id,
            "summary": clean_commit_subject(subject, body),
            "doc_link": extract_rtd_links(changed_files),
        }
    )

  return commits


def group_multipart_commits(commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Consolidates numbered multi-part PR series (e.g., 'Feature (1/5): detail') into a single entry."""
  grouped: List[Dict[str, Any]] = []
  series_map: Dict[str, Dict[str, Any]] = {}
  series_re = re.compile(r"^(?:\[([^\]]+?)\s+\d+/\d+\]|(.+?)\s*\([^)]*\b\d+/\d+\))\s*:\s*(.+)$")

  for c in commits:
    match = series_re.match(c["summary"].strip())
    if not match:
      grouped.append(dict(c))
      continue

    prefix = (match.group(1) or match.group(2)).strip()
    detail = match.group(3).strip()
    key = prefix.lower()
    if key not in series_map:
      entry = {**c, "summary": f"**{prefix}**: {detail}"}
      series_map[key] = entry
      grouped.append(entry)
    else:
      existing = series_map[key]
      if detail not in existing["summary"]:
        existing["summary"] += f"; {detail}"
      if c.get("doc_link") and c["doc_link"] not in existing.get("doc_link", ""):
        existing["doc_link"] = f"{existing.get('doc_link', '')}, {c['doc_link']}".lstrip(", ")
      if c.get("ref_id") and c["ref_id"] not in existing.get("ref_id", ""):
        existing["ref_id"] = f"{existing.get('ref_id', '')}, {c['ref_id']}".lstrip(", ")

  return grouped


def format_commit_bullet(item: Dict[str, Any]) -> str:
  """Formats a single changelog bullet line with its `[ref:...]` tracking tag."""
  summary = item["summary"].rstrip(".")
  if summary and summary[0].islower():
    summary = summary[0].upper() + summary[1:]
  ref_suffix = f" [ref:{item['ref_id']}]" if item.get("ref_id") else ""
  if item.get("doc_link"):
    return f"- {summary} ({item['doc_link']}).{ref_suffix}"
  return f"- {summary}.{ref_suffix}"


def generate_changelog_block(commits: List[Dict[str, Any]]) -> str:
  """Generates formatted Markdown bullet lines for candidate commits."""
  if not commits:
    return ""
  consolidated = group_multipart_commits(commits)
  return "\n".join(format_commit_bullet(item) for item in consolidated)


def extract_ref_ids(text: str) -> set[str]:
  """Extracts all reconciliation reference IDs from `[ref:...]` tags in `text`."""
  ids: set[str] = set()
  for match in re.finditer(r"\[ref:([^\]]+)\]", text):
    for token in match.group(1).split(","):
      if token.strip():
        ids.add(token.strip())
  return ids


def reconcile_commits(raw_text: str, polished_text: str) -> List[str]:
  """Compares `[ref:...]` IDs in `raw_text` against `polished_text` and returns any dropped IDs."""
  raw_ids = extract_ref_ids(raw_text)
  polished_ids = extract_ref_ids(polished_text)
  return sorted(raw_ids - polished_ids)


def strip_ref_tags(text: str) -> str:
  """Removes temporary `[ref:...]` reconciliation tags before writing to `docs/release_notes.md`."""
  return re.sub(r"\s*\[ref:[^\]]+\]", "", text).strip()


def parse_unreleased_sections(text: str) -> Dict[str, Dict[str, List[str]]]:
  """Parses a release notes Markdown block into `{h4_section: {h5_category: [bullets]}}`."""
  sections: Dict[str, Dict[str, List[str]]] = {}
  current_h4 = "#### Changes"
  current_h5 = ""

  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith("##### "):
      current_h5 = f"##### {line.lstrip('#').strip()}"
      sections.setdefault(current_h4, {}).setdefault(current_h5, [])
    elif line.startswith("###"):
      current_h4 = f"#### {line.lstrip('#').strip()}"
      current_h5 = ""
      sections.setdefault(current_h4, {}).setdefault(current_h5, [])
    elif line.startswith("- "):
      bullets = sections.setdefault(current_h4, {}).setdefault(current_h5, [])
      bullets.append(line)
    else:
      bullets = sections.setdefault(current_h4, {}).setdefault(current_h5, [])
      if bullets:
        bullets[-1] = f"{bullets[-1]}\n{line}"
      else:
        bullets.append(f"- {line}")

  return sections


def merge_unreleased_notes(existing_text: str, new_text: str) -> str:
  """Merges `new_text` into `existing_text` under matching `####` sections and `#####` subsections."""
  if not existing_text.strip():
    return new_text.strip()

  merged = parse_unreleased_sections(existing_text)
  incoming = parse_unreleased_sections(new_text)

  for h4, submap in incoming.items():
    target_submap = merged.setdefault(h4, {})
    for h5, new_bullets in submap.items():
      existing_bullets = target_submap.setdefault(h5, [])
      target_submap[h5] = [b for b in new_bullets if b not in existing_bullets] + existing_bullets

  all_h4 = list(dict.fromkeys([*SECTION_ORDER, *merged.keys()]))
  out_blocks: List[str] = []

  for h4 in all_h4:
    submap = merged.get(h4, {})
    if not any(submap.values()):
      continue
    sec_lines = [f"{h4}\n"]
    all_h5 = list(dict.fromkeys(["", *CATEGORY_ORDER, *submap.keys()]))
    for h5 in all_h5:
      bullets = submap.get(h5, [])
      if not bullets:
        continue
      if h5:
        sec_lines.append(f"{h5}\n")
      sec_lines.extend(bullets)
    out_blocks.append("\n".join(sec_lines))

  return "\n\n".join(out_blocks)


def update_changelog_files(new_block: str) -> None:
  """Merges new release notes into `## Unreleased` and updates `**Last Updated**` in docs/release_notes.md."""
  if not os.path.exists(RELEASE_NOTES_RELPATH):
    print(f"Warning: {RELEASE_NOTES_RELPATH} does not exist. Skipping update.")
    return

  with open(RELEASE_NOTES_RELPATH, "r", encoding="utf-8") as f:
    content = f.read()

  head_commit = get_head_commit()
  updated_tag = f"**Last Updated**: {head_commit}"
  marker = "<!-- Add new unreleased changes below this line -->"

  # 1. Update existing `**Last Updated**` tag, or insert it under `## Unreleased`
  if LAST_UPDATED_RE.search(content):
    new_content = LAST_UPDATED_RE.sub(updated_tag, content, count=1)
  elif "## Unreleased" in content:
    new_content = content.replace("## Unreleased", f"## Unreleased\n\n{updated_tag}", 1)
  else:
    print(f"Warning: Could not find '## Unreleased' header in {RELEASE_NOTES_RELPATH}. Skipping update.")
    return

  # 2. Merge `new_block` into existing `#### Changes` (`##### <Category>`), `#### Bug Fixes`, and `#### Deprecations`
  anchor = marker if marker in new_content else updated_tag
  anchor_end = new_content.index(anchor) + len(anchor)
  next_h2_match = re.search(r"\n##\s+", new_content[anchor_end:])
  unreleased_end = anchor_end + next_h2_match.start() if next_h2_match else len(new_content)

  existing_unreleased = new_content[anchor_end:unreleased_end].strip()
  merged_unreleased = merge_unreleased_notes(existing_unreleased, new_block)
  new_content = f"{new_content[:anchor_end]}\n\n{merged_unreleased}\n\n{new_content[unreleased_end:].lstrip()}"

  with open(RELEASE_NOTES_RELPATH, "w", encoding="utf-8") as f:
    f.write(new_content)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Collect candidate commits since Last Updated in docs/release_notes.md, "
          "verify LLM-synthesized release notes for dropped commits, and inject them into docs/release_notes.md."
      )
  )
  parser.add_argument(
      "--mode",
      choices=["collect", "verify", "inject"],
      required=True,
      help=(
          "Execution mode: 'collect' writes candidate commits to raw_commits.txt; "
          "'verify' checks that new_notes.md contains all [ref:...] IDs from raw_commits.txt; "
          "'inject' strips [ref:...] tags from new_notes.md and injects into docs/release_notes.md"
      ),
  )
  args = parser.parse_args()

  if args.mode == "collect":
    raw_log = run_git_log()
    commits = parse_commits(raw_log)
    raw_changelog = generate_changelog_block(commits)
    with open(RAW_COMMITS_FILE, "w", encoding="utf-8") as f:
      f.write(f"{raw_changelog}\n" if raw_changelog else "")
    print(f"Wrote {len(extract_ref_ids(raw_changelog))} commits to {RAW_COMMITS_FILE}.")

  elif args.mode == "verify":
    with open(RAW_COMMITS_FILE, "r", encoding="utf-8") as f:
      raw_text = f.read()
    with open(POLISHED_NOTES_FILE, "r", encoding="utf-8") as f:
      polished_text = f.read()
    dropped = reconcile_commits(raw_text, polished_text)
    if dropped:
      print(f"Error: Reconciliation check failed! Dropped commit/PR IDs: {', '.join(dropped)}")
      sys.exit(1)
    print(f"Reconciliation passed (all {len(extract_ref_ids(raw_text))} IDs verified).")

  elif args.mode == "inject":
    with open(POLISHED_NOTES_FILE, "r", encoding="utf-8") as f:
      polished_text = f.read()
    clean_notes = strip_ref_tags(polished_text)
    update_changelog_files(clean_notes)
    print(f"Injected {POLISHED_NOTES_FILE} into {RELEASE_NOTES_RELPATH}.")


if __name__ == "__main__":
  main()
