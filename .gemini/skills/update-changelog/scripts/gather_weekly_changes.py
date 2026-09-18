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

"""Helper script to gather git commits and PRs for weekly changelog updates."""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


REPO_URL = "https://github.com/AI-Hypercomputer/maxtext"

# Categories mapped by conventional commit type or keyword heuristics
CATEGORY_KEYWORDS = {
    "Model Support & Architecture": [
        "deepseek",
        "qwen",
        "gemma",
        "llama",
        "mixtral",
        "gpt",
        "moe",
        "diffusion",
        "vit",
        "mamba",
        "decoder",
        "encoder",
        "architecture",
        "model",
    ],
    "Performance": [
        "fp8",
        "fp4",
        "int8",
        "int4",
        "quant",
        "qwix",
        "aqt",
        "gemm",
        "matmul",
        "throughput",
        "perf",
        "tflops",
        "overlap",
        "collective",
        "double-buffer",
        "memory",
        "muon",
        "sparse_core",
        "sparsecore",
    ],
    "Checkpointing & Goodput": [
        "checkpoint",
        "orbax",
        "goodput",
        "elasticity",
        "restore",
        "slice",
        "multi-tier",
        "resilience",
        "zarr",
        "ocdbt",
    ],
    "Post Training": [
        "rl",
        "grpo",
        "dpo",
        "sft",
        "lora",
        "qlora",
        "reward",
        "rollout",
        "vllm",
        "post-training",
        "post_training",
        "tunix",
        "raiden",
    ],
    "Usability & Infrastructure": [
        "wandb",
        "grain",
        "docker",
        "ci",
        "eval",
        "diloco",
        "profil",
        "tutorial",
        "doc",
        "guide",
        "cli",
        "config",
        "pypi",
        "wheel",
        "airflow",
        "dag",
    ],
}


def run_git_log(repo_dir: str, since: str) -> str:
  """Runs git log and returns raw formatted commit entries including parent hashes."""
  delimiter = "===COMMIT_DELIMITER==="
  field_sep = "===FIELD_SEP==="
  # %H: hash, %P: parent hashes, %an: author, %ad: author date, %s: subject, %b: body
  pretty_format = f"{delimiter}%H{field_sep}%P{field_sep}%an{field_sep}%ad{field_sep}%s{field_sep}%b"
  cmd = [
      "git",
      "-C",
      repo_dir,
      "log",
      f"--since={since}",
      "--date=short",
      f"--pretty=format:{pretty_format}",
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return result.stdout


def extract_pr_number(subject: str, body: str) -> Optional[str]:
  """Extracts PR number from commit subject or body."""
  merge_match = re.search(r"Merge pull request #(\d+)", subject)
  if merge_match:
    return merge_match.group(1)
  paren_match = re.search(r"\(#(\d+)\)", subject)
  if paren_match:
    return paren_match.group(1)
  body_match = re.search(r"(?:PR|pull request|#) ?#?(\d{4,5})\b", body, re.IGNORECASE)
  if body_match:
    return body_match.group(1)
  return None


def clean_body_text(body: str) -> str:
  """Removes PiperOrigin-RevId and other metadata noise from commit body."""
  lines = []
  for line in body.splitlines():
    if line.strip().startswith("PiperOrigin-RevId:"):
      continue
    if line.strip().startswith("COPYBARA_INTEGRATE_REVIEW="):
      continue
    lines.append(line)
  return "\n".join(lines).strip()


def classify_commit(subject: str, body: str) -> Dict[str, Any]:
  """Classifies a commit into a section (Changes vs Bug Fixes) and category."""
  text = f"{subject} {body}".lower()
  clean_subject = re.sub(r"^Merge pull request #\d+ from [^\s]+\s*", "", subject).strip()
  if not clean_subject and body:
    clean_subject = body.strip().splitlines()[0]

  conv_match = re.match(r"^([a-z]+)(?:\(([^)]+)\))?!?:\s*(.*)", clean_subject, re.IGNORECASE)
  commit_type = conv_match.group(1).lower() if conv_match else ""
  scope = conv_match.group(2).lower() if conv_match and conv_match.group(2) else ""
  description = conv_match.group(3) if conv_match else clean_subject

  is_bugfix = (
      commit_type in ("fix", "bugfix", "hotfix")
      or clean_subject.lower().startswith("fix ")
      or clean_subject.lower().startswith("fixed ")
      or "fix:" in clean_subject.lower()
      or "bug" in scope
  )

  lower_desc = description.lower()
  is_minor = (
      commit_type in ("style", "chore", "test", "ci", "build")
      or lower_desc in ("update", "fix linter issues", "lint", "format")
      or lower_desc.startswith("reverts ")
      or lower_desc.startswith("suppress new pyrefly")
      or lower_desc.startswith("add auto_gha prefix")
  ) and not any(kw in text for kw in ["support", "upgrade", "enable", "migration", "release", "tutorial"])

  matched_category = "Usability & Infrastructure"
  search_text = f"{scope} {clean_subject} {body}".lower()
  for category, keywords in CATEGORY_KEYWORDS.items():
    if any(kw in search_text for kw in keywords):
      matched_category = category
      break

  return {
      "clean_subject": description or clean_subject,
      "commit_type": commit_type or ("fix" if is_bugfix else "feat"),
      "scope": scope,
      "section": "Bug Fixes" if is_bugfix else "Changes",
      "category": matched_category,
      "is_minor": is_minor,
  }


def parse_commits(raw_log: str) -> List[Dict[str, Any]]:
  """Parses raw git log output and resolves merge commits with their merged parents."""
  delimiter = "===COMMIT_DELIMITER==="
  field_sep = "===FIELD_SEP==="

  raw_entries: Dict[str, Dict[str, Any]] = {}
  ordered_hashes: List[str] = []
  parent_to_pr: Dict[str, str] = {}

  for entry in raw_log.split(delimiter):
    entry = entry.strip()
    if not entry:
      continue
    parts = entry.split(field_sep)
    if len(parts) < 6:
      continue
    commit_hash = parts[0].strip()
    parents = parts[1].strip().split()
    author = parts[2].strip()
    date = parts[3].strip()
    subject = parts[4].strip()
    body = clean_body_text(parts[5])

    pr_num = extract_pr_number(subject, parts[5])
    is_merge = subject.startswith("Merge pull request #") or len(parents) > 1

    if is_merge and pr_num and len(parents) >= 2:
      second_parent = parents[1]
      parent_to_pr[second_parent] = pr_num

    raw_entries[commit_hash] = {
        "hash": commit_hash,
        "parents": parents,
        "author": author,
        "date": date,
        "subject": subject,
        "body": body,
        "pr_num": pr_num,
        "is_merge": is_merge,
    }
    ordered_hashes.append(commit_hash)

  commits = []
  seen_prs = set()
  seen_hashes = set()

  for commit_hash in ordered_hashes:
    item = raw_entries[commit_hash]
    pr_num = item["pr_num"] or parent_to_pr.get(commit_hash)

    subject = item["subject"]
    body = item["body"]
    if item["is_merge"] and len(item["parents"]) >= 2:
      second_parent = item["parents"][1]
      matched_parent = None
      for h in raw_entries:  # pylint: disable=consider-using-dict-items
        if h.startswith(second_parent) or second_parent.startswith(h):
          matched_parent = raw_entries[h]
          break
      if matched_parent:
        subject = matched_parent["subject"]
        body = matched_parent["body"]
        seen_hashes.add(matched_parent["hash"])
      elif not body and subject.startswith("Merge pull request #"):
        branch_match = re.search(r"from [^:]+:(.+)$", subject)
        if branch_match:
          subject = branch_match.group(1).replace("-", " ").replace("_", " ")

    if commit_hash in seen_hashes:
      continue
    seen_hashes.add(commit_hash)

    if pr_num and pr_num in seen_prs:
      continue
    if pr_num:
      seen_prs.add(pr_num)

    classification = classify_commit(subject, body)
    pr_link = f"[PR #{pr_num}]({REPO_URL}/pull/{pr_num})" if pr_num else f"`{commit_hash[:7]}`"

    commits.append(
        {
            "hash": commit_hash[:9],
            "author": item["author"],
            "date": item["date"],
            "raw_subject": subject,
            "summary": classification["clean_subject"],
            "pr_number": pr_num,
            "pr_link": pr_link,
            "section": classification["section"],
            "category": classification["category"],
            "is_minor": classification["is_minor"],
            "body": body,
        }
    )

  return commits


def get_existing_unreleased_prs(repo_dir: str) -> List[str]:
  """Reads existing PR numbers from docs/release_notes.md Unreleased section."""
  found_prs = set()
  path = os.path.join(repo_dir, "docs", "release_notes.md")
  if os.path.exists(path):
    try:
      with open(path, "r", encoding="utf-8") as f:
        content = f.read()
      unreleased_match = re.search(r"##\s+Unreleased(.*?)(?:\n##\s+|\Z)", content, re.DOTALL | re.IGNORECASE)
      text_to_scan = unreleased_match.group(1) if unreleased_match else content
      for m in re.finditer(r"pull/(\d+)|#(\d+)", text_to_scan):
        pr = m.group(1) or m.group(2)
        if pr:
          found_prs.add(pr)
    except OSError:
      pass
  return sorted(found_prs)


def generate_unreleased_markdown_block(major_commits: List[Dict[str, Any]]) -> str:
  """Generates formatted Markdown lines for new unreleased major commits."""
  if not major_commits:
    return ""
  block_lines = []
  for section in ["Changes", "Bug Fixes"]:
    section_commits = [c for c in major_commits if c["section"] == section]
    if not section_commits:
      continue
    block_lines.append(f"#### {section}")
    block_lines.append("")
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for c in section_commits:
      by_category.setdefault(c["category"], []).append(c)
    for category, items in by_category.items():
      block_lines.append(f"- **{category}**:")
      for item in items:
        summary = item["summary"]
        if summary and summary[0].islower():
          summary = summary[0].upper() + summary[1:]
        block_lines.append(f"  - {summary} ({item['pr_link']}).")
      block_lines.append("")
  return "\n".join(block_lines).strip()


def update_changelog_files(repo_dir: str, major_commits: List[Dict[str, Any]]) -> List[str]:
  """Inserts new major commits into docs/release_notes.md if not already updated."""
  if not major_commits:
    return []
  new_block = generate_unreleased_markdown_block(major_commits)
  if not new_block:
    return []

  updated_files = []
  marker = "<!-- Add new unreleased changes below this line -->"
  targets = [
      os.path.join(repo_dir, "docs", "release_notes.md"),
  ]
  for path in targets:
    if not os.path.exists(path):
      continue
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
    if marker in content:
      replacement = f"{marker}\n\n{new_block}"
      new_content = content.replace(marker, replacement, 1)
    elif "## Unreleased" in content:
      new_content = content.replace("## Unreleased", f"## Unreleased\n\n{new_block}", 1)
    else:
      continue
    if new_content != content:
      with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
      updated_files.append(path)
  return updated_files


def format_markdown_report(commits: List[Dict[str, Any]], since: str, existing_prs: List[str]) -> str:
  """Formats the parsed commits into a Markdown report ready for changelog synthesis."""
  existing_set = set(existing_prs)
  new_commits = [c for c in commits if not c["pr_number"] or c["pr_number"] not in existing_set]
  already_documented = [c for c in commits if c["pr_number"] and c["pr_number"] in existing_set]

  major_commits = [c for c in new_commits if not c["is_minor"]]
  minor_commits = [c for c in new_commits if c["is_minor"]]

  lines = [
      f"# Weekly Changelog Candidate Report (Since {since})",
      "",
      f"- **Total Unique Commits/PRs Analyzed**: {len(commits)}",
      f"- **New Major / Announcement Candidates**: {len(major_commits)}",
      f"- **Already Documented in Unreleased**: {len(already_documented)}",
      f"- **Minor / Internal Commits Filtered**: {len(minor_commits)}",
      "",
      "## New Candidate Major Changes (By Release Notes Category)",
      "",
  ]

  for section in ["Changes", "Bug Fixes"]:
    section_commits = [c for c in major_commits if c["section"] == section]
    if not section_commits:
      continue
    lines.append(f"### {section}")
    lines.append("")

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for c in section_commits:
      by_category.setdefault(c["category"], []).append(c)

    for category, items in by_category.items():
      lines.append(f"- **{category}**:")
      for item in items:
        summary = item["summary"]
        if summary and summary[0].islower():
          summary = summary[0].upper() + summary[1:]
        lines.append(f"  - {summary} ({item['pr_link']}) — *{item['date']}*")
      lines.append("")

  if already_documented:
    lines.append("## Already Present in Changelog (`Unreleased`)")
    lines.append("")
    for item in already_documented:
      lines.append(f"- {item['summary']} ({item['pr_link']})")
    lines.append("")

  if minor_commits:
    lines.append("## Filtered Minor / Maintenance Commits (Excluded from Announcement)")
    lines.append("")
    for item in minor_commits:
      lines.append(f"- [{item['category']}] {item['summary']} ({item['pr_link']})")
    lines.append("")

  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Gather weekly git changes for changelog.")
  parser.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
  parser.add_argument("--since", type=str, default=None, help="Explicit start date (YYYY-MM-DD)")
  parser.add_argument("--repo-dir", type=str, default=".", help="Path to git repository root")
  parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
  parser.add_argument(
      "--update-files",
      action="store_true",
      help="Automatically insert new candidate major changes into docs/release_notes.md",
  )
  args = parser.parse_args()

  if args.since:
    since_date = args.since
  else:
    dt = datetime.date.today() - datetime.timedelta(days=args.days)
    since_date = dt.isoformat()

  raw_log = run_git_log(args.repo_dir, since_date)
  commits = parse_commits(raw_log)
  existing_prs = get_existing_unreleased_prs(args.repo_dir)

  existing_set = set(existing_prs)
  new_commits = [c for c in commits if not c["pr_number"] or c["pr_number"] not in existing_set]
  major_commits = [c for c in new_commits if not c["is_minor"]]

  if args.update_files and major_commits:
    updated = update_changelog_files(args.repo_dir, major_commits)
    if updated:
      sys.stderr.write(f"Updated changelog files: {', '.join(updated)}\n")

  if args.format == "json":
    print(
        json.dumps(
            {
                "since": since_date,
                "existing_unreleased_prs": existing_prs,
                "commits": commits,
            },
            indent=2,
        )
    )
  else:
    print(format_markdown_report(commits, since_date, existing_prs))


if __name__ == "__main__":
  main()
