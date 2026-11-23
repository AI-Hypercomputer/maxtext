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

"""Mines handover docs, git diffs/commits, and overlapping .agents/wiki/ pages for wiki distillation."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

SHARED_EXACT_PATHS = {
    "src/maxtext/configs/base.yml",
    "src/maxtext/configs/pyconfig.py",
    "src/maxtext/configs/types.py",
    "tests/integration/hlo_diff_test.py",
    "tests/unit/pyconfig_test.py",
    "tests/unit/train_nnx_test.py",
}
SHARED_PREFIXES = (
    "benchmarks/api_server/",
    "src/maxtext/checkpoint_conversion/utils/",
    "src/maxtext/common/",
    "src/maxtext/inference/",
    "src/maxtext/kernels/",
    "src/maxtext/layers/",
    "src/maxtext/m3/core/",
    "src/maxtext/m3/train/",
    "src/maxtext/multimodal/",
    "src/maxtext/trainers/",
    "src/maxtext/utils/",
)
COMMIT_SIGNAL_RE = re.compile(
    r"\b(fix|bug|oom|hbm|remat|offload|parity|kl|rope|router|moe|transpose|dtype|fp32|bf16|"
    r"shard|mask|vmem|workaround|gdn|cache|nan|overflow|tie)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
PENDING_DISPOSITIONS = {"", "...", "pending", "(pending)", "tbd", "todo", "-"}
ROOT_ARTIFACT_SUFFIXES = (".md", ".sh", ".py", ".patch")


def _run_git(repo_root: Path, args: list[str]) -> str:
  proc = subprocess.run(
      ["git", "-C", str(repo_root), *args],
      capture_output=True,
      text=True,
      check=False,
  )
  return proc.stdout.strip() if proc.returncode == 0 else ""


def _strip_quotes(val: str) -> str:
  val = val.strip()
  if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
    return val[1:-1]
  return val


def parse_wiki_frontmatter(text: str) -> dict[str, str | list[str]]:
  """Extracts frontmatter key-values from a wiki markdown page."""
  lines = text.splitlines()
  if not lines or lines[0].strip() != "---":
    return {}
  end_idx = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
  if end_idx is None:
    return {}

  meta: dict[str, str | list[str]] = {}
  current_list: str | None = None
  for raw_line in lines[1:end_idx]:
    clean = re.sub(r"\s+#.*$", "", raw_line).rstrip()
    if not clean.strip():
      continue
    if clean.startswith(("  - ", "- ")) and current_list and isinstance(meta.get(current_list), list):
      cast_list = meta[current_list]
      if isinstance(cast_list, list):
        cast_list.append(_strip_quotes(clean.lstrip()[2:]))
      continue
    if ":" not in clean:
      continue
    key, val = [part.strip() for part in clean.split(":", 1)]
    current_list = None
    if not val:
      meta[key] = []
      current_list = key
    elif val.startswith("[") and val.endswith("]"):
      inner = val[1:-1].strip()
      meta[key] = [_strip_quotes(p) for p in inner.split(",") if p.strip()] if inner else []
    else:
      meta[key] = _strip_quotes(val)
  return meta


def resolve_diff_base(repo_root: Path, base_ref: str) -> str:
  """Resolves merge-base(base_ref, HEAD) so behind-main branches only report branch-local changes."""
  merge_base = _run_git(repo_root, ["merge-base", base_ref, "HEAD"])
  if merge_base:
    return merge_base
  fallback = _run_git(repo_root, ["rev-parse", base_ref])
  return fallback or "HEAD"


def is_shared_infra_path(rel_path: str) -> bool:
  """Returns True when rel_path belongs to shared MaxText runtime/config/kernel infrastructure."""
  if rel_path in SHARED_EXACT_PATHS:
    return True
  return rel_path.startswith(SHARED_PREFIXES)


def extract_handover_candidates(handover_paths: list[Path], repo_root: Path) -> list[dict[str, str]]:
  """Parses Section 4 pitfall table rows or bullet items lacking a finalized Wiki Disposition."""
  items: list[dict[str, str]] = []
  for path in handover_paths:
    if not path.is_file():
      continue
    text = path.read_text(encoding="utf-8", errors="replace")
    sec_match = re.search(r"^##\s+4\..*$", text, re.M)
    section = text[sec_match.end() :] if sec_match else ""
    if not section:
      continue
    next_h2 = re.search(r"^##\s+", section, re.M)
    if next_h2:
      section = section[: next_h2.start()]

    rel_name = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else path.name
    for line in section.splitlines():
      stripped = line.strip()
      if stripped.startswith("|") and stripped.endswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells or all(re.match(r"^:?-{2,}:?$", c) for c in cells):
          continue
        if cells[0].lower() in ("stage", "#") or all(c == "..." for c in cells):
          continue
        disposition = cells[-1].strip("` ").lower() if len(cells) >= 5 else ""
        if disposition in PENDING_DISPOSITIONS:
          items.append({"source": rel_name, "entry": " | ".join(cells)})
      elif stripped.startswith(("- ", "* ")) and len(stripped) > 6:
        items.append({"source": rel_name, "entry": stripped[2:].strip()})
  return items


def collect_changed_config_keys(repo_root: Path, diff_base: str, include_working_tree: bool) -> list[str]:
  """Extracts config keys added or modified in src/maxtext/configs/base.yml relative to diff_base."""
  diff_args = ["diff", diff_base] if include_working_tree else ["diff", f"{diff_base}..HEAD"]
  diff_text = _run_git(repo_root, [*diff_args, "--", "src/maxtext/configs/base.yml"])
  keys: set[str] = set()
  for line in diff_text.splitlines():
    if line.startswith("+") and not line.startswith("+++"):
      match = re.match(r"^\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", line)
      if match:
        keys.add(match.group(1))
  return sorted(keys)


def extract_signal_tokens(
    changed_files: list[str],
    config_keys: list[str],
    commits: list[str],
    handover_items: list[dict[str, str]],
) -> set[str]:
  """Collects lowercase alphanumeric tokens across diffs, config keys, commits, and handover notes."""
  corpus = " ".join(
      [
          *changed_files,
          *config_keys,
          *commits,
          *(item["entry"] for item in handover_items),
      ]
  ).lower()
  return set(TOKEN_RE.findall(corpus))


def match_overlapping_wiki_pages(
    wiki_dir: Path,
    shared_files: list[str],
    config_keys: list[str],
    signal_tokens: set[str] | None = None,
) -> list[dict[str, object]]:
  """Finds existing .agents/wiki/ leaf pages sharing code_refs, config_keys, or tags with the branch."""
  if not wiki_dir.is_dir():
    return []
  shared_set = set(shared_files)
  cfg_set = set(config_keys)
  tok_set = signal_tokens or set()
  matches: list[dict[str, object]] = []

  for md_file in sorted(wiki_dir.rglob("*.md")):
    if md_file.name in ("index.md", "PROTOCOL.md"):
      continue
    meta = parse_wiki_frontmatter(md_file.read_text(encoding="utf-8"))
    status = meta.get("status", "active")
    code_refs = meta.get("code_refs", [])
    page_cfgs = meta.get("config_keys", [])
    page_tags = meta.get("tags", [])

    ref_hits = sorted(shared_set.intersection(code_refs)) if isinstance(code_refs, list) else []
    cfg_hits = sorted(cfg_set.intersection(page_cfgs)) if isinstance(page_cfgs, list) else []
    tag_hits = (
        sorted(t for t in page_tags if t.lower() in tok_set) if status == "active" and isinstance(page_tags, list) else []
    )
    if ref_hits or cfg_hits or len(tag_hits) >= 2:
      matches.append(
          {
              "page": md_file.relative_to(wiki_dir).as_posix(),
              "title": meta.get("title", md_file.stem),
              "status": status,
              "matched_code_refs": ref_hits,
              "matched_config_keys": cfg_hits,
              "matched_tags": tag_hits,
          }
      )

  def _score(item: dict[str, object]) -> int:
    refs = len(item["matched_code_refs"])  # type: ignore[arg-type]
    cfgs = len(item["matched_config_keys"])  # type: ignore[arg-type]
    tags = len(item["matched_tags"])  # type: ignore[arg-type]
    return -(3 * refs + 3 * cfgs + tags)

  matches.sort(key=_score)
  return matches


def harvest(
    repo_root: Path,
    wiki_dir: Path,
    base_ref: str,
    handover_args: list[Path],
    include_working_tree: bool = True,
) -> dict[str, object]:
  """Collects deterministic distillation signals from handover docs, git diffs, commits, and wiki overlaps."""
  branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD"
  diff_base = resolve_diff_base(repo_root, base_ref)

  diff_cmd = ["diff", "--name-only", diff_base] if include_working_tree else ["diff", "--name-only", f"{diff_base}..HEAD"]
  changed_files = [p for p in _run_git(repo_root, diff_cmd).splitlines() if p and not p.startswith(".agents/")]
  shared_files = sorted(p for p in changed_files if is_shared_infra_path(p))
  model_files = sorted(p for p in changed_files if not is_shared_infra_path(p))

  config_keys = collect_changed_config_keys(repo_root, diff_base, include_working_tree)

  raw_commits = _run_git(repo_root, ["log", "--oneline", f"{diff_base}..HEAD"]).splitlines()
  notable_commits = [c for c in raw_commits if COMMIT_SIGNAL_RE.search(c)]

  handover_files = list(handover_args) if handover_args else sorted(repo_root.glob("*_handover.md"))
  handover_items = extract_handover_candidates(handover_files, repo_root)

  untracked_all = _run_git(repo_root, ["ls-files", "--others", "--exclude-standard"]).splitlines()
  root_artifacts = sorted(
      p for p in untracked_all if "/" not in p and p.endswith(ROOT_ARTIFACT_SUFFIXES) and p != "README.md"
  )

  tokens = extract_signal_tokens(changed_files, config_keys, notable_commits, handover_items)
  overlaps = match_overlapping_wiki_pages(wiki_dir, shared_files, config_keys, tokens)
  return {
      "repo": repo_root.name,
      "branch": branch,
      "merge_base": diff_base[:10],
      "pending_handover_items": handover_items,
      "overlapping_wiki_pages": overlaps,
      "shared_infra_files": shared_files,
      "changed_base_yml_keys": config_keys,
      "notable_commits": notable_commits,
      "root_debug_artifacts": root_artifacts,
      "model_scoped_file_count": len(model_files),
  }


def format_markdown_report(report: dict[str, object]) -> str:
  """Renders the harvested distillation brief as compact Markdown for human or agent triage."""
  lines = [
      f"# Wiki Distillation Brief (`{report['branch']}` @ `{report['merge_base']}`)",
      "",
      f"- **Shared infrastructure files touched**: {len(report['shared_infra_files'])} "  # type: ignore[arg-type]
      f"(model-scoped files skipped: {report['model_scoped_file_count']})",
      f"- **Changed `base.yml` config keys**: {len(report['changed_base_yml_keys'])}",  # type: ignore[arg-type]
      f"- **Pending handover §4 items**: {len(report['pending_handover_items'])}",  # type: ignore[arg-type]
      "",
      "## 1. Overlapping Existing Wiki Pages (Check `MERGE` Before `NEW`)",
  ]
  overlaps: list[dict[str, object]] = report["overlapping_wiki_pages"]  # type: ignore[assignment]
  if overlaps:
    for ov in overlaps:
      lines.append(
          f"- **`{ov['page']}`** ({ov['title']}) — `code_refs`: `{ov['matched_code_refs']}` | "
          f"`config_keys`: `{ov['matched_config_keys']}` | `tags`: `{ov['matched_tags']}`"
      )
  else:
    lines.append("- *(No existing wiki page overlaps `code_refs`, `config_keys`, or `tags`.)*")

  lines.extend(["", "## 2. Pending Handover Pitfalls (`*_handover.md` §4)"])
  handovers: list[dict[str, str]] = report["pending_handover_items"]  # type: ignore[assignment]
  if handovers:
    for item in handovers:
      lines.append(f"- `{item['source']}`: {item['entry']}")
  else:
    lines.append("- *(No `*_handover.md` pending §4 entries found in repo root.)*")

  lines.extend(["", "## 3. Shared MaxText Infrastructure Files Modified"])
  shared: list[str] = report["shared_infra_files"]  # type: ignore[assignment]
  for path in shared or ["*(None)*"]:
    lines.append(f"- `{path}`" if path != "*(None)*" else f"- {path}")

  lines.extend(["", "## 4. Added / Modified `base.yml` Config Keys"])
  cfgs: list[str] = report["changed_base_yml_keys"]  # type: ignore[assignment]
  lines.append(", ".join(f"`{k}`" for k in cfgs) if cfgs else "- *(None)*")

  lines.extend(["", "## 5. Fix / Pitfall Commits (`merge-base..HEAD`)"])
  commits: list[str] = report["notable_commits"]  # type: ignore[assignment]
  for commit in commits[:20] or ["*(None)*"]:
    lines.append(f"- `{commit}`" if commit != "*(None)*" else f"- {commit}")

  lines.extend(["", "## 6. Untracked Root Verification / Debug Artifacts"])
  artifacts: list[str] = report["root_debug_artifacts"]  # type: ignore[assignment]
  for art in artifacts or ["*(None)*"]:
    lines.append(f"- `{art}`" if art != "*(None)*" else f"- {art}")

  return "\n".join(lines) + "\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  script_wiki_dir = Path(__file__).resolve().parents[1]
  default_repo = Path(__file__).resolve().parents[3]
  parser.add_argument("--repo-root", type=Path, default=default_repo, help="Target MaxText checkout to inspect")
  parser.add_argument(
      "--wiki-dir", type=Path, default=None, help="Path to .agents/wiki (defaults to repo or script wiki)"
  )
  parser.add_argument("--base", default="origin/main", help="Base git ref for merge-base diffing")
  parser.add_argument("--handover", type=Path, action="append", default=[], help="Explicit *_handover.md path(s)")
  parser.add_argument("--no-working-tree", action="store_true", help="Ignore uncommitted working-tree edits")
  parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format")
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  candidate_wiki = repo_root / ".agents/wiki"
  wiki_dir = (args.wiki_dir or (candidate_wiki if candidate_wiki.is_dir() else script_wiki_dir)).resolve()

  report = harvest(
      repo_root=repo_root,
      wiki_dir=wiki_dir,
      base_ref=args.base,
      handover_args=args.handover,
      include_working_tree=not args.no_working_tree,
  )
  if args.format == "json":
    print(json.dumps(report, indent=2))
  else:
    print(format_markdown_report(report), end="")
  return 0


if __name__ == "__main__":
  sys.exit(main())
