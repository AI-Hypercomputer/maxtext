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

"""Helper for the non-blocking MaxText post-submit review GitHub Action.

Provides two subcommands:
  1. `prepass`: resolves `BASE_SHA..TARGET_SHA` (`<target>^1..<target>`) and the
     merged pull request number, runs fast deterministic and cross-model checks
     from `.agents/skills/maxtext-pr-upload/scripts/`, and writes
     `.gemini/postsubmit_prepass.json` and `.gemini/context.json`.
  2. `maybe-comment`: combines Step 1 pre-pass results with Step 2 structured
     Gemini findings and posts at most one idempotent comment on the merged pull
     request when `deterministic_rc != 0` or `>= 1` `P0`/`P1` finding exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

_SAFE_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/~^@{}-]{1,128}$")
_MERGE_PR_RE = re.compile(r"Merge pull request #(\d+)\b")
_SQUASH_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
_PARAM_MAPPING_PATH_RE = re.compile(r"(?:checkpoint_conversion|param_mapping|hf_streaming_load|load_dynamic)")
_MAXTEXT_MODEL_CODE_RE = re.compile(r"^src/maxtext/.*\.(?:py|ya?ml)$")
_SHARDING_SWEEP_PATH_RE = re.compile(
    r"(?:sharding\.py$|m3/core/moe\.py$|m3/models/__init__\.py$|m3/models/.+/modeling_[^/]+\.py$)"
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_SEVERITY_MAP: dict[str, str] = {
    "P0": "P0",
    "CRITICAL": "P0",
    "RED": "P0",
    "🔴": "P0",
    "P1": "P1",
    "HIGH": "P1",
    "ORANGE": "P1",
    "🟠": "P1",
    "P2": "P2",
    "MEDIUM": "P2",
    "MED": "P2",
    "YELLOW": "P2",
    "🟡": "P2",
    "LOW": "P2",
    "NIT": "NIT",
    "TRIVIAL": "NIT",
    "INFO": "NIT",
    "GREEN": "NIT",
    "🟢": "NIT",
}

_MAX_PATCH_CHARS = 400_000
_MAX_COMMENT_CHARS = 60_000
_MAX_PAGES = 50
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

HttpTransport = Callable[
    [str, str, Mapping[str, str], bytes | None],
    tuple[int, Mapping[str, str], str],
]


@dataclasses.dataclass(frozen=True)
class CommitRange:
  """Resolved commit metadata and changed-file lists for `base_sha..target_sha`."""

  target_sha: str
  base_sha: str
  parent_shas: tuple[str, ...]
  commit_subject: str
  commit_message: str
  all_changed_files: tuple[str, ...]
  existing_changed_files: tuple[str, ...]
  unified_diff: str = ""
  u0_diff: str = ""


def _validate_git_ref(ref: str, label: str = "git ref") -> str:
  """Validates that `ref` cannot be interpreted as a CLI flag by `git`."""
  cleaned = ref.strip()
  if not cleaned or cleaned.startswith("-") or not _SAFE_GIT_REF_RE.match(cleaned):
    raise ValueError(f"Invalid {label}: {ref!r}")
  return cleaned


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    check: bool = True,
    timeout_s: int = 120,
) -> subprocess.CompletedProcess[str]:
  """Runs `git` safely inside `repo_root` with `safe.directory=*`."""
  cmd = ["git", "-c", "safe.directory=*", "-C", str(repo_root), *args]
  proc = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      check=False,
      timeout=timeout_s,
  )
  if check and proc.returncode != 0:
    raise RuntimeError(f"git {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}")
  return proc


def _extract_raw_commit_parents(repo_root: Path, commit_sha: str) -> tuple[str, ...]:
  """Reads `parent <sha>` headers directly from the commit object (shallow-safe)."""
  raw_obj = _run_git(repo_root, ["cat-file", "-p", commit_sha]).stdout
  parents: list[str] = []
  for line in raw_obj.splitlines():
    if not line.strip():
      break
    if line.startswith("parent "):
      parent_sha = line[len("parent ") :].strip()
      if parent_sha:
        parents.append(parent_sha)
  return tuple(parents)


def resolve_commit_range(
    repo_root: Path,
    target_ref: str = "HEAD",
    base_ref: str = "",
) -> CommitRange:
  """Resolves `target_sha` and first-parent `base_sha` (`<target_sha>^1` or empty tree)."""
  safe_target = _validate_git_ref(target_ref, "target_ref")
  target_proc = _run_git(repo_root, ["rev-parse", "--verify", f"{safe_target}^{{commit}}"], check=False)
  if target_proc.returncode != 0 and re.fullmatch(r"[0-9a-fA-F]{40}", safe_target):
    _run_git(repo_root, ["fetch", "--no-tags", "--depth=2", "origin", safe_target], check=False)
    target_proc = _run_git(repo_root, ["rev-parse", "--verify", f"{safe_target}^{{commit}}"], check=False)
  if target_proc.returncode != 0:
    raise ValueError(f"Cannot resolve target commit {safe_target!r}: {target_proc.stderr.strip()}")
  target_sha = target_proc.stdout.strip()

  parent_shas = _extract_raw_commit_parents(repo_root, target_sha)

  if base_ref.strip():
    safe_base = _validate_git_ref(base_ref, "base_ref")
    if safe_base == _EMPTY_TREE_SHA:
      base_sha = _EMPTY_TREE_SHA
    else:
      base_proc = _run_git(repo_root, ["rev-parse", "--verify", f"{safe_base}^{{commit}}"], check=False)
      if base_proc.returncode != 0:
        raise ValueError(f"Cannot resolve base commit {safe_base!r}: {base_proc.stderr.strip()}")
      base_sha = base_proc.stdout.strip()
  elif not parent_shas:
    base_sha = _EMPTY_TREE_SHA
  else:
    first_parent = parent_shas[0]
    has_parent = _run_git(repo_root, ["cat-file", "-e", f"{first_parent}^{{commit}}"], check=False)
    if has_parent.returncode != 0:
      _run_git(repo_root, ["fetch", "--no-tags", "--depth=2", "origin", target_sha], check=False)
    base_proc = _run_git(repo_root, ["rev-parse", "--verify", f"{first_parent}^{{commit}}"], check=False)
    if base_proc.returncode != 0:
      raise ValueError(f"Cannot resolve first parent for {target_sha}: {base_proc.stderr.strip()}")
    base_sha = base_proc.stdout.strip()

  commit_subject = _run_git(repo_root, ["log", "-1", "--format=%s", target_sha]).stdout.strip()
  commit_message = _run_git(repo_root, ["log", "-1", "--format=%B", target_sha]).stdout.strip()

  all_files_out = _run_git(repo_root, ["diff", "--name-only", base_sha, target_sha]).stdout
  all_changed = tuple(line.strip() for line in all_files_out.splitlines() if line.strip())

  existing_out = _run_git(
      repo_root,
      ["diff", "--name-only", "--diff-filter=ACMRTUXB", base_sha, target_sha],
  ).stdout
  existing_changed = tuple(line.strip() for line in existing_out.splitlines() if line.strip())
  unified_diff = _run_git(repo_root, ["diff", base_sha, target_sha]).stdout
  u0_diff = _run_git(repo_root, ["diff", "-U0", base_sha, target_sha]).stdout

  return CommitRange(
      target_sha=target_sha,
      base_sha=base_sha,
      parent_shas=parent_shas,
      commit_subject=commit_subject,
      commit_message=commit_message,
      all_changed_files=all_changed,
      existing_changed_files=existing_changed,
      unified_diff=unified_diff,
      u0_diff=u0_diff,
  )


def _default_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: bytes | None,
) -> tuple[int, Mapping[str, str], str]:
  """Executes an HTTPS GitHub REST API call via `urllib.request`."""
  parsed = urllib.parse.urlparse(url)
  if parsed.scheme != "https" or parsed.netloc != "api.github.com":
    raise ValueError(f"Refusing non-GitHub API URL: {url!r}")
  req = urllib.request.Request(url, data=payload, headers=dict(headers), method=method)
  with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 - scheme and host validated above
    body = resp.read().decode("utf-8", errors="replace")
    return int(resp.status), dict(resp.headers.items()), body


class GitHubApiClient:
  """Minimal stdlib GitHub REST client with pagination and retry support."""

  def __init__(
      self,
      token: str = "",
      transport: HttpTransport | None = None,
      max_retries: int = 3,
      retry_backoff_s: float = 1.0,
  ) -> None:
    self._token = token.strip()
    self._transport = transport or _default_transport
    self._max_retries = max(1, max_retries)
    self._retry_backoff_s = max(0.0, retry_backoff_s)

  def _headers(self) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "maxtext-postsubmit-review",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if self._token:
      headers["Authorization"] = f"Bearer {self._token}"
    return headers

  def request_json(
      self,
      method: str,
      path: str,
      body_obj: Mapping[str, Any] | None = None,
  ) -> Any:
    """Sends a GitHub REST request and parses the JSON response."""
    rel = path if path.startswith("/") else f"/{path}"
    url = f"https://api.github.com{rel}"
    headers = self._headers()
    payload: bytes | None = None
    if body_obj is not None:
      headers["Content-Type"] = "application/json; charset=utf-8"
      payload = json.dumps(body_obj).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(self._max_retries):
      try:
        status, _, text = self._transport(method, url, headers, payload)
        if 200 <= status < 300:
          return json.loads(text) if text.strip() else None
        if status >= 500 and attempt + 1 < self._max_retries:
          time.sleep(self._retry_backoff_s * (2**attempt))
          continue
        raise RuntimeError(f"GitHub API {method} {rel} failed with HTTP {status}: {text[:300]}")
      except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        if exc.code >= 500 and attempt + 1 < self._max_retries:
          last_exc = exc
          time.sleep(self._retry_backoff_s * (2**attempt))
          continue
        raise RuntimeError(f"GitHub API {method} {rel} failed with HTTP {exc.code}: {err_body[:300]}") from exc
      except urllib.error.URLError as exc:
        last_exc = exc
        if attempt + 1 < self._max_retries:
          time.sleep(self._retry_backoff_s * (2**attempt))
          continue
        raise RuntimeError(f"GitHub API {method} {rel} network error: {exc}") from exc

    raise RuntimeError(f"GitHub API {method} {rel} failed after retries: {last_exc}")

  def list_paginated(self, path: str, per_page: int = 100) -> list[dict[str, Any]]:
    """Fetches all pages (`per_page=100`) from a GitHub REST list endpoint."""
    collected: list[dict[str, Any]] = []
    sep = "&" if "?" in path else "?"
    for page in range(1, _MAX_PAGES + 1):
      paged_path = f"{path}{sep}per_page={per_page}&page={page}"
      data = self.request_json("GET", paged_path)
      if not isinstance(data, list):
        break
      page_items = [item for item in data if isinstance(item, dict)]
      collected.extend(page_items)
      if len(data) < per_page:
        break
    return collected


def _parse_bool_flag(value: str | bool) -> bool:
  """Parses a boolean or GitHub Actions string boolean (`true`/`1`/`yes`)."""
  if isinstance(value, bool):
    return value
  return value.strip().lower() in ("true", "1", "yes")


def resolve_merged_pr_number(
    repository: str,
    target_sha: str,
    commit_subject: str,
    commit_message: str = "",
    api_client: GitHubApiClient | None = None,
    *,
    is_fork: bool = False,
) -> int | None:
  """Resolves the merged PR number via `/commits/{sha}/pulls` with subject fallback."""
  if api_client is not None and repository.strip():
    try:
      pulls = api_client.list_paginated(f"/repos/{repository}/commits/{target_sha}/pulls", per_page=100)
    except RuntimeError as exc:
      sys.stderr.write(f"Warning: could not query /commits/{target_sha}/pulls: {exc}\n")
      pulls = []

    merged_main_exact = [
        p
        for p in pulls
        if p.get("merged_at")
        and isinstance(p.get("base"), dict)
        and p["base"].get("ref") == "main"
        and p.get("merge_commit_sha") == target_sha
    ]
    merged_main = [
        p for p in pulls if p.get("merged_at") and isinstance(p.get("base"), dict) and p["base"].get("ref") == "main"
    ]
    merged_any = [p for p in pulls if p.get("merged_at")]
    exact_sha_any = [p for p in pulls if p.get("merge_commit_sha") == target_sha]

    for bucket in (merged_main_exact, merged_main, merged_any, exact_sha_any):
      if bucket:
        num = bucket[0].get("number")
        if isinstance(num, int) and num > 0:
          return num

  if is_fork:
    return None

  for text in (commit_subject, commit_message):
    if not text:
      continue
    merge_match = _MERGE_PR_RE.search(text)
    if merge_match:
      return int(merge_match.group(1))

  squash_match = _SQUASH_PR_RE.search(commit_subject)
  if squash_match:
    return int(squash_match.group(1))

  return None


def _python_has_jax(python_exe: str) -> bool:
  """Returns True when `python_exe` can import `jax` and `flax` on CPU."""
  env = {
      **os.environ,
      "JAX_PLATFORMS": "cpu",
      "JAX_ENABLE_COMPILATION_CACHE": "0",
      "TPU_SKIP_MDS_QUERY": "1",
      "ALLOW_MULTIPLE_LIBTPU_LOAD": "true",
  }
  try:
    probe = subprocess.run(
        [python_exe, "-I", "-S", "-m", "site"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if probe.returncode != 0:
      return False
    check_proc = subprocess.run(
        [python_exe],
        input="import importlib.util, sys\nsys.exit(0 if importlib.util.find_spec('jax') else 1)\n",
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    return check_proc.returncode == 0
  except (subprocess.TimeoutExpired, OSError):
    return False


def _timeout_check_result(check_name: str, timeout_s: float | int) -> dict[str, Any]:
  """Builds a non-crashing ERROR check record when a subprocess times out."""
  msg = f"{check_name} timed out after {int(timeout_s)}s."
  return {
      "name": check_name,
      "status": "ERROR",
      "returncode": -1,
      "summary": msg,
      "findings": [
          {
              "source": check_name,
              "code": "CHECK_TIMEOUT",
              "severity": "HIGH",
              "file": "",
              "line": None,
              "message": msg,
          }
      ],
  }


def _locate_script(
    script_name: str,
    repo_root: Path,
    fallback_scripts_dir: Path | None = None,
) -> Path | None:
  """Finds a check script in `.agents/skills/maxtext-pr-upload/scripts/` or fallback."""
  primary = repo_root / ".agents" / "skills" / "maxtext-pr-upload" / "scripts" / script_name
  if primary.is_file():
    return primary
  if fallback_scripts_dir is not None:
    candidate = fallback_scripts_dir / script_name
    if candidate.is_file():
      return candidate
  return None


def _extract_issues_from_json_stdout(
    stdout: str,
    check_name: str,
    *,
    only_high_confidence: bool = False,
) -> list[dict[str, Any]]:
  """Extracts structured issue entries from a check script's JSON stdout."""
  stripped = stdout.strip()
  if not stripped:
    return []
  try:
    payload = json.loads(stripped)
  except json.JSONDecodeError:
    return []
  if not isinstance(payload, dict):
    return []
  raw_issues = payload.get("issues", [])
  if not isinstance(raw_issues, list):
    return []
  dict_items = [item for item in raw_issues if isinstance(item, dict)]
  if only_high_confidence and dict_items:
    high_conf = [
        item
        for item in dict_items
        if str(item.get("severity", "HIGH")).upper() == "HIGH" and str(item.get("confidence", "HIGH")).upper() == "HIGH"
    ]
    dict_items = high_conf
  extracted: list[dict[str, Any]] = []
  for item in dict_items:
    extracted.append(
        {
            "source": check_name,
            "code": str(item.get("code", check_name)),
            "severity": str(item.get("severity", "HIGH")),
            "file": str(item.get("file", item.get("model", ""))),
            "line": item.get("line"),
            "message": str(item.get("message", "")),
        }
    )
  return extracted


def _infer_named_models_regex(changed_files: Sequence[str], head_dump_dir: Path) -> str:
  """Builds `--named` regex for models whose own config or m3 package was edited."""
  all_dumped = [p.stem for p in head_dump_dir.glob("*.json")]
  named: set[str] = set()
  for rel in changed_files:
    yml_match = re.match(r"src/maxtext/configs/models/(.+)\.yml$", rel)
    if yml_match:
      named.add(yml_match.group(1))
    m3_match = re.match(r"src/maxtext/m3/models/(\w+)/", rel)
    if m3_match:
      fam = re.sub(r"[^a-z0-9]", "", m3_match.group(1).lower())
      for model_key in all_dumped:
        if model_key.startswith("m3__"):
          norm_key = re.sub(r"[^a-z0-9]", "", model_key[4:].lower())
          if norm_key.startswith(fam):
            named.add(model_key)
  if not named:
    return ""
  return "^(" + "|".join(re.escape(name) for name in sorted(named)) + ")$"


def _run_abstract_model_diff_check(
    script_path: Path,
    repo_root: Path,
    base_sha: str,
    changed_files: Sequence[str],
    python_exe: str,
) -> dict[str, Any]:
  """Runs 102-model `abstract_model_diff.py` comparing `base_sha` vs `repo_root`."""
  tmp_root = Path(tempfile.mkdtemp(prefix="maxtext_postsubmit_tier_a_"))
  base_wt = tmp_root / f"base_wt_{os.getpid()}"
  base_dump = tmp_root / "base_dump"
  head_dump = tmp_root / "head_dump"
  md_out = tmp_root / "tier_a.md"
  workers = str(min(32, max(2, (os.cpu_count() or 4) * 2)))
  env = {
      **os.environ,
      "JAX_PLATFORMS": "cpu",
      "JAX_ENABLE_COMPILATION_CACHE": "0",
      "TPU_SKIP_MDS_QUERY": "1",
      "ALLOW_MULTIPLE_LIBTPU_LOAD": "true",
  }

  try:
    wt_proc = _run_git(repo_root, ["worktree", "add", "--detach", str(base_wt), base_sha], check=False)
    if wt_proc.returncode != 0:
      return {
          "name": "abstract_model_diff",
          "status": "ERROR",
          "returncode": wt_proc.returncode,
          "summary": f"Could not create temporary base worktree: {wt_proc.stderr.strip()[:240]}",
          "findings": [],
      }

    for wt_dir, dump_dir in ((base_wt, base_dump), (repo_root, head_dump)):
      dump_dir.mkdir(parents=True, exist_ok=True)
      dump_proc = subprocess.run(
          [
              python_exe,
              str(script_path),
              "dump",
              "--worktree",
              str(wt_dir),
              "--out",
              str(dump_dir),
              "--python",
              python_exe,
              "-j",
              workers,
          ],
          capture_output=True,
          text=True,
          env=env,
          check=False,
          timeout=900,
      )
      if dump_proc.returncode != 0:
        return {
            "name": "abstract_model_diff",
            "status": "ERROR",
            "returncode": dump_proc.returncode,
            "summary": f"abstract_model_diff dump failed on {wt_dir.name}: {dump_proc.stderr.strip()[-300:]}",
            "findings": [],
        }

    named_re = _infer_named_models_regex(changed_files, head_dump)
    cmp_cmd = [
        python_exe,
        str(script_path),
        "compare",
        str(base_dump),
        str(head_dump),
        "--md",
        str(md_out),
    ]
    if named_re:
      cmp_cmd.extend(["--named", named_re])

    cmp_proc = subprocess.run(
        cmp_cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=300,
    )
    md_text = md_out.read_text(encoding="utf-8", errors="replace") if md_out.is_file() else cmp_proc.stdout
    headline = md_text.splitlines()[0].lstrip("# ").strip() if md_text.strip() else "abstract_model_diff completed"
    findings: list[dict[str, Any]] = []
    rc = cmp_proc.returncode
    if rc == 1:
      findings.append(
          {
              "source": "abstract_model_diff",
              "code": "CROSS_MODEL_SHAPE_OR_SHARDING_DIFF",
              "severity": "HIGH",
              "file": "src/maxtext/",
              "line": None,
              "message": md_text.strip()[:1500],
          }
      )

    if any(_SHARDING_SWEEP_PATH_RE.search(f) for f in changed_files):
      sweep_md = tmp_root / "sharding_sweep.md"
      sweep_proc = subprocess.run(
          [
              python_exe,
              str(script_path),
              "sharding-sweep",
              "--worktree",
              str(repo_root),
              "--md",
              str(sweep_md),
              "--changed-files",
              ",".join(changed_files),
              "--python",
              python_exe,
          ],
          capture_output=True,
          text=True,
          env=env,
          check=False,
          timeout=300,
      )
      if sweep_proc.returncode == 1:
        rc = 1
        sweep_text = sweep_md.read_text(encoding="utf-8", errors="replace") if sweep_md.is_file() else sweep_proc.stdout
        findings.append(
            {
                "source": "abstract_model_diff",
                "code": "SHARDING_AXIS_SWEEP_FAILURE",
                "severity": "HIGH",
                "file": "src/maxtext/",
                "line": None,
                "message": sweep_text.strip()[:1500],
            }
        )

    status = "PASS" if rc == 0 else ("FAIL" if rc == 1 else "ERROR")
    return {
        "name": "abstract_model_diff",
        "status": status,
        "returncode": rc,
        "summary": headline,
        "findings": findings,
    }
  except subprocess.TimeoutExpired as exc:
    return _timeout_check_result("abstract_model_diff", exc.timeout)
  finally:
    _run_git(repo_root, ["worktree", "remove", "--force", str(base_wt)], check=False)
    _run_git(repo_root, ["worktree", "prune"], check=False)
    shutil.rmtree(tmp_root, ignore_errors=True)


def _format_prepass_digest(
    commit_range: CommitRange,
    pr_number: int | None,
    checks: Sequence[Mapping[str, Any]],
    blocking_findings: Sequence[Mapping[str, Any]],
) -> str:
  """Renders a concise Markdown digest of Step 1 results for Gemini context."""
  pr_label = f"PR #{pr_number}" if pr_number is not None else "no linked PR"
  lines = [
      f"### Step 1 Post-Submit Deterministic Pre-Pass ({commit_range.target_sha[:12]}, {pr_label})",
      f"- Base commit (`^1`): `{commit_range.base_sha}`",
      f"- Changed files: `{len(commit_range.all_changed_files)}`",
      "",
      "| Check | Status | Summary |",
      "| :--- | :--- | :--- |",
  ]
  for chk in checks:
    summary_cell = str(chk.get("summary", "")).replace("\n", " ").replace("|", "\\|")[:200]
    lines.append(f"| `{chk['name']}` | **{chk['status']}** | {summary_cell} |")

  if blocking_findings:
    lines.append("")
    lines.append("#### Deterministic Findings to Verify / Contextualize:")
    for item in blocking_findings[:20]:
      loc = str(item.get("file") or "<repo>")
      if item.get("line"):
        loc = f"{loc}:{item['line']}"
      msg = str(item.get("message", "")).strip().replace("\n", " ")[:300]
      lines.append(f"- **[{item.get('code', 'CHECK_FAIL')}]** `{loc}`: {msg}")
  return "\n".join(lines)


def _run_skill_and_script_self_tests(
    repo_abs: Path,
    python_exe: str,
    py_env: Mapping[str, str],
) -> dict[str, Any]:
  """Runs existing `.github/scripts/*_test.py` and `.agents/skills/*/tests/*_test.py`."""
  candidates: list[Path] = []
  gh_scripts = repo_abs / ".github" / "scripts"
  if gh_scripts.is_dir():
    candidates.extend(sorted(p for p in gh_scripts.glob("*_test.py") if p.is_file()))
  skills_root = repo_abs / ".agents" / "skills"
  if skills_root.is_dir():
    candidates.extend(sorted(p for p in skills_root.glob("*/tests/*_test.py") if p.is_file()))

  if not candidates:
    return {
        "name": "skill_and_script_self_tests",
        "status": "SKIPPED",
        "returncode": 0,
        "summary": "No skill or helper *_test.py files present in checkout.",
        "findings": [],
    }

  test_env = {
      **py_env,
      "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
  }
  try:
    has_pytest_proc = subprocess.run(
        [python_exe],
        input="import importlib.util, sys\nsys.exit(0 if importlib.util.find_spec('pytest') else 1)\n",
        capture_output=True,
        text=True,
        env=test_env,
        check=False,
        timeout=15,
    )
    tmp_cwd = tempfile.gettempdir()
    if has_pytest_proc.returncode == 0:
      proc = subprocess.run(
          [
              python_exe,
              "-m",
              "pytest",
              "-q",
              "--noconftest",
              "-o",
              "addopts=--import-mode=importlib --strict-markers",
              "-p",
              "no:cacheprovider",
              *(str(p) for p in candidates),
          ],
          capture_output=True,
          text=True,
          cwd=tmp_cwd,
          env=test_env,
          check=False,
          timeout=300,
      )
      if proc.returncode == 0:
        return {
            "name": "skill_and_script_self_tests",
            "status": "PASS",
            "returncode": 0,
            "summary": f"Passed {len(candidates)} self-test suite(s) via pytest.",
            "findings": [],
        }
      tail = (proc.stdout + "\n" + proc.stderr).strip()[-600:]
      return {
          "name": "skill_and_script_self_tests",
          "status": "FAIL",
          "returncode": proc.returncode,
          "summary": f"Self-test suite failed under pytest (rc={proc.returncode}).",
          "findings": [
              {
                  "source": "skill_and_script_self_tests",
                  "code": "SKILL_OR_SCRIPT_SELF_TEST_FAILURE",
                  "severity": "HIGH",
                  "file": ".agents/skills/",
                  "line": None,
                  "message": tail,
              }
          ],
      }

    for test_path in candidates:
      proc = subprocess.run(
          [python_exe, str(test_path)],
          capture_output=True,
          text=True,
          cwd=tmp_cwd,
          env=test_env,
          check=False,
          timeout=300,
      )
      if proc.returncode != 0:
        rel = str(test_path.relative_to(repo_abs))
        tail = (proc.stdout + "\n" + proc.stderr).strip()[-600:]
        return {
            "name": "skill_and_script_self_tests",
            "status": "FAIL",
            "returncode": proc.returncode,
            "summary": f"Self-test {rel} failed (rc={proc.returncode}).",
            "findings": [
                {
                    "source": "skill_and_script_self_tests",
                    "code": "SKILL_OR_SCRIPT_SELF_TEST_FAILURE",
                    "severity": "HIGH",
                    "file": rel,
                    "line": None,
                    "message": tail,
                }
            ],
        }
  except subprocess.TimeoutExpired as exc:
    return _timeout_check_result("skill_and_script_self_tests", exc.timeout)
  return {
      "name": "skill_and_script_self_tests",
      "status": "PASS",
      "returncode": 0,
      "summary": f"Passed {len(candidates)} self-test suite(s) via unittest.",
      "findings": [],
  }


def run_prepass(
    repo_root: Path,
    repository: str,
    *,
    target_ref: str = "HEAD",
    base_ref: str = "",
    fallback_scripts_dir: Path | None = None,
    prepass_out: Path | None = None,
    authoritative_prepass_out: Path | None = None,
    context_out: Path | None = None,
    github_output: Path | None = None,
    python_exe: str | None = None,
    api_client: GitHubApiClient | None = None,
    is_fork: bool = False,
) -> dict[str, Any]:
  """Executes Step 1 pre-pass checks and writes `.gemini/` context artifacts."""
  repo_abs = repo_root.resolve()
  py_bin = python_exe or sys.executable
  commit_range = resolve_commit_range(repo_abs, target_ref=target_ref, base_ref=base_ref)
  pr_number = resolve_merged_pr_number(
      repository=repository,
      target_sha=commit_range.target_sha,
      commit_subject=commit_range.commit_subject,
      commit_message=commit_range.commit_message,
      api_client=api_client,
      is_fork=is_fork,
  )

  checks: list[dict[str, Any]] = []
  blocking_findings: list[dict[str, Any]] = []
  py_env = {
      **os.environ,
      "PYTHONPATH": f"{repo_abs / 'src'}:{repo_abs}",
      "JAX_PLATFORMS": "cpu",
      "JAX_ENABLE_COMPILATION_CACHE": "0",
      "TPU_SKIP_MDS_QUERY": "1",
      "ALLOW_MULTIPLE_LIBTPU_LOAD": "true",
  }

  # 1. audit_repo_and_ci_contracts.py
  changed_py = [f for f in commit_range.existing_changed_files if f.endswith(".py")]
  contracts_script = _locate_script("audit_repo_and_ci_contracts.py", repo_abs, fallback_scripts_dir)
  if not changed_py:
    checks.append(
        {
            "name": "audit_repo_and_ci_contracts",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "No non-deleted .py files changed.",
            "findings": [],
        }
    )
  elif contracts_script is None:
    checks.append(
        {
            "name": "audit_repo_and_ci_contracts",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "Script not present in checkout.",
            "findings": [],
        }
    )
  else:
    abs_changed_py = [str(repo_abs / f) for f in changed_py]
    contracts_cmd = [
        py_bin,
        str(contracts_script),
        "--repo-root",
        str(repo_abs),
        "--files",
        *abs_changed_py,
    ]
    if commit_range.parent_shas:
      contracts_cmd.extend(["--base", commit_range.base_sha])
    try:
      proc = subprocess.run(
          contracts_cmd,
          capture_output=True,
          text=True,
          cwd=str(repo_abs),
          env=py_env,
          check=False,
          timeout=300,
      )
      found = _extract_issues_from_json_stdout(
          proc.stdout,
          "audit_repo_and_ci_contracts",
          only_high_confidence=True,
      )
      status = "PASS" if proc.returncode == 0 else ("FAIL" if proc.returncode == 1 else "ERROR")
      if status == "FAIL" and not found:
        found = [
            {
                "source": "audit_repo_and_ci_contracts",
                "code": "REPO_CONTRACT_VIOLATION",
                "severity": "HIGH",
                "file": "",
                "line": None,
                "message": (proc.stderr or proc.stdout).strip()[:500],
            }
        ]
      if status == "FAIL":
        blocking_findings.extend(found)
      checks.append(
          {
              "name": "audit_repo_and_ci_contracts",
              "status": status,
              "returncode": proc.returncode,
              "summary": f"Audited {len(changed_py)} Python file(s); {len(found)} finding(s).",
              "findings": found,
          }
      )
    except subprocess.TimeoutExpired as exc:
      timeout_res = _timeout_check_result("audit_repo_and_ci_contracts", exc.timeout)
      blocking_findings.extend(timeout_res["findings"])
      checks.append(timeout_res)

  # 2. audit_pr_review_hygiene.py
  has_py_or_sh = any(f.endswith((".py", ".sh")) for f in commit_range.all_changed_files)
  hygiene_script = _locate_script("audit_pr_review_hygiene.py", repo_abs, fallback_scripts_dir)
  if not has_py_or_sh:
    checks.append(
        {
            "name": "audit_pr_review_hygiene",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "No .py or .sh files changed.",
            "findings": [],
        }
    )
  elif hygiene_script is None:
    checks.append(
        {
            "name": "audit_pr_review_hygiene",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "Script not present in checkout.",
            "findings": [],
        }
    )
  else:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", encoding="utf-8", delete=False) as tf:
      tf.write(commit_range.u0_diff)
      patch_path = Path(tf.name)
    try:
      proc = subprocess.run(
          [
              py_bin,
              str(hygiene_script),
              "--repo-root",
              str(repo_abs),
              "--diff-file",
              str(patch_path),
          ],
          capture_output=True,
          text=True,
          cwd=str(repo_abs),
          env=py_env,
          check=False,
          timeout=300,
      )
      found = _extract_issues_from_json_stdout(proc.stdout, "audit_pr_review_hygiene")
      status = "PASS" if proc.returncode == 0 else ("FAIL" if proc.returncode == 1 else "ERROR")
      if status == "FAIL" and not found:
        found = [
            {
                "source": "audit_pr_review_hygiene",
                "code": "PR_HYGIENE_VIOLATION",
                "severity": "HIGH",
                "file": "",
                "line": None,
                "message": (proc.stderr or proc.stdout).strip()[:500],
            }
        ]
      if status == "FAIL":
        blocking_findings.extend(found)
      checks.append(
          {
              "name": "audit_pr_review_hygiene",
              "status": status,
              "returncode": proc.returncode,
              "summary": f"{len(found)} hygiene finding(s).",
              "findings": found,
          }
      )
    except subprocess.TimeoutExpired as exc:
      timeout_res = _timeout_check_result("audit_pr_review_hygiene", exc.timeout)
      blocking_findings.extend(timeout_res["findings"])
      checks.append(timeout_res)
    finally:
      patch_path.unlink(missing_ok=True)

  # 3. sweep_param_mapping_contracts.py (conditional on checkpoint/mapping edits)
  needs_param_sweep = any(_PARAM_MAPPING_PATH_RE.search(f) for f in commit_range.all_changed_files)
  needs_model_diff = any(_MAXTEXT_MODEL_CODE_RE.search(f) for f in commit_range.all_changed_files)
  has_jax = _python_has_jax(py_bin) if (needs_param_sweep or (needs_model_diff and commit_range.parent_shas)) else False

  sweep_script = _locate_script("sweep_param_mapping_contracts.py", repo_abs, fallback_scripts_dir)
  if not needs_param_sweep:
    checks.append(
        {
            "name": "sweep_param_mapping_contracts",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "No checkpoint conversion or param_mapping files changed.",
            "findings": [],
        }
    )
  elif sweep_script is None or not has_jax:
    reason = "Script not present in checkout." if sweep_script is None else "JAX not installed in pre-pass venv."
    checks.append(
        {
            "name": "sweep_param_mapping_contracts",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": reason,
            "findings": [],
        }
    )
  else:
    try:
      proc = subprocess.run(
          [py_bin, str(sweep_script), "--maxtext-root", str(repo_abs)],
          capture_output=True,
          text=True,
          cwd=str(repo_abs),
          env=py_env,
          check=False,
          timeout=300,
      )
      found = _extract_issues_from_json_stdout(proc.stdout, "sweep_param_mapping_contracts")
      status = "PASS" if proc.returncode == 0 else ("FAIL" if proc.returncode == 1 else "ERROR")
      if status == "FAIL" and not found:
        found = [
            {
                "source": "sweep_param_mapping_contracts",
                "code": "PARAM_MAPPING_CONTRACT_VIOLATION",
                "severity": "HIGH",
                "file": "src/maxtext/checkpoint_conversion/",
                "line": None,
                "message": (proc.stderr or proc.stdout).strip()[:500],
            }
        ]
      if status == "FAIL":
        blocking_findings.extend(found)
      checks.append(
          {
              "name": "sweep_param_mapping_contracts",
              "status": status,
              "returncode": proc.returncode,
              "summary": f"{len(found)} contract violation(s).",
              "findings": found,
          }
      )
    except subprocess.TimeoutExpired as exc:
      timeout_res = _timeout_check_result("sweep_param_mapping_contracts", exc.timeout)
      blocking_findings.extend(timeout_res["findings"])
      checks.append(timeout_res)

  # 4. abstract_model_diff.py (102-model Tier A shape/sharding diff)
  diff_script = _locate_script("abstract_model_diff.py", repo_abs, fallback_scripts_dir)
  if not needs_model_diff:
    checks.append(
        {
            "name": "abstract_model_diff",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "No src/maxtext/ Python or YAML model/config files changed.",
            "findings": [],
        }
    )
  elif not commit_range.parent_shas:
    checks.append(
        {
            "name": "abstract_model_diff",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "Root commit has no parent commit for base worktree comparison.",
            "findings": [],
        }
    )
  elif diff_script is None or not has_jax:
    reason = "Script not present in checkout." if diff_script is None else "JAX not installed in pre-pass venv."
    checks.append(
        {
            "name": "abstract_model_diff",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": reason,
            "findings": [],
        }
    )
  else:
    tier_a_res = _run_abstract_model_diff_check(
        script_path=diff_script,
        repo_root=repo_abs,
        base_sha=commit_range.base_sha,
        changed_files=commit_range.all_changed_files,
        python_exe=py_bin,
    )
    if tier_a_res["status"] in ("FAIL", "ERROR"):
      blocking_findings.extend(tier_a_res["findings"])
    checks.append(tier_a_res)

  # 5. Self-tests when `.agents/skills/` or `.github/scripts/` changed
  needs_self_tests = any(f.startswith((".agents/skills/", ".github/scripts/")) for f in commit_range.all_changed_files)
  if not needs_self_tests:
    checks.append(
        {
            "name": "skill_and_script_self_tests",
            "status": "SKIPPED",
            "returncode": 0,
            "summary": "No .agents/skills/ or .github/scripts/ files changed.",
            "findings": [],
        }
    )
  else:
    self_test_res = _run_skill_and_script_self_tests(repo_abs, py_bin, py_env)
    if self_test_res["status"] in ("FAIL", "ERROR"):
      blocking_findings.extend(self_test_res["findings"])
    checks.append(self_test_res)

  deterministic_rc = 1 if any(c["status"] in ("FAIL", "ERROR") for c in checks) else 0
  digest_markdown = _format_prepass_digest(commit_range, pr_number, checks, blocking_findings)

  gemini_dir = prepass_out.parent if prepass_out is not None else (repo_abs / ".gemini")
  gemini_dir.mkdir(parents=True, exist_ok=True)

  full_diff = commit_range.unified_diff
  if len(full_diff) > _MAX_PATCH_CHARS:
    full_diff = full_diff[:_MAX_PATCH_CHARS] + "\n... [diff truncated]\n"
  (gemini_dir / "commit_diff.patch").write_text(full_diff, encoding="utf-8")
  (gemini_dir / "commit_message.txt").write_text(commit_range.commit_message + "\n", encoding="utf-8")
  (gemini_dir / "changed_files.txt").write_text(
      "\n".join(commit_range.all_changed_files) + ("\n" if commit_range.all_changed_files else ""),
      encoding="utf-8",
  )

  prepass_payload: dict[str, Any] = {
      "repository": repository,
      "target_sha": commit_range.target_sha,
      "base_sha": commit_range.base_sha,
      "parent_shas": list(commit_range.parent_shas),
      "pr_number": pr_number,
      "commit_subject": commit_range.commit_subject,
      "changed_files": list(commit_range.all_changed_files),
      "deterministic_rc": deterministic_rc,
      "checks": checks,
      "blocking_findings": blocking_findings,
      "digest_markdown": digest_markdown,
  }
  prepass_json_text = json.dumps(prepass_payload, indent=2, sort_keys=True) + "\n"
  resolved_prepass_out = prepass_out or (gemini_dir / "postsubmit_prepass.json")
  resolved_prepass_out.parent.mkdir(parents=True, exist_ok=True)
  resolved_prepass_out.write_text(prepass_json_text, encoding="utf-8")
  if authoritative_prepass_out is not None:
    authoritative_prepass_out.parent.mkdir(parents=True, exist_ok=True)
    authoritative_prepass_out.write_text(prepass_json_text, encoding="utf-8")

  additional_context = (
      f"{digest_markdown}\n\n"
      "### Post-Submit Output Contract (Strict Quiet-Unless-Blocking Policy)\n"
      "Do NOT call `pull_request_review_write` or post any comments directly to GitHub. "
      "Write structured JSON findings (`severity`: `P0` | `P1` | `P2` | `NIT`, `category`, "
      "`file`, `line`, `title`, `body`) to `.gemini/postsubmit_gemini_findings.json` and "
      "echo the same JSON block in your final response so clean merges receive zero noise comments."
  )
  context_payload = {
      "repository": repository,
      "pull_request_number": str(pr_number) if pr_number is not None else "",
      "target_sha": commit_range.target_sha,
      "base_sha": commit_range.base_sha,
      "additional_context": additional_context,
  }
  resolved_context_out = context_out or (gemini_dir / "context.json")
  resolved_context_out.parent.mkdir(parents=True, exist_ok=True)
  resolved_context_out.write_text(json.dumps(context_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  if github_output is not None:
    with github_output.open("a", encoding="utf-8") as out_f:
      out_f.write(f"pr_number={pr_number or ''}\n")
      out_f.write(f"deterministic_rc={deterministic_rc}\n")

  return prepass_payload


def _extract_balanced_json_object(text: str) -> str | None:
  """Extracts the last top-level `{...}` JSON object containing `\"findings\"`."""
  candidates: list[str] = []
  depth = 0
  start = -1
  in_str = False
  escape = False
  for idx, ch in enumerate(text):
    if in_str:
      if escape:
        escape = False
      elif ch == "\\":
        escape = True
      elif ch == '"':
        in_str = False
      continue
    if ch == '"':
      in_str = True
    elif ch == "{":
      if depth == 0:
        start = idx
      depth += 1
    elif ch == "}" and depth > 0:
      depth -= 1
      if depth == 0 and start >= 0:
        snippet = text[start : idx + 1]
        if '"findings"' in snippet:
          candidates.append(snippet)
        start = -1
  return candidates[-1] if candidates else None


def parse_gemini_findings(raw_text: str) -> tuple[list[dict[str, Any]], str]:
  """Parses structured Gemini findings safely without raising on malformed text."""
  stripped = raw_text.strip()
  if not stripped:
    return [], "empty_gemini_output"

  fenced_blobs = [m.group(1).strip() for m in reversed(list(_JSON_FENCE_RE.finditer(stripped)))]
  balanced = _extract_balanced_json_object(stripped)
  json_blobs: list[str] = [*fenced_blobs, stripped]
  if balanced:
    json_blobs.append(balanced)

  raw_items: list[Any] | None = None
  saw_other_json = False
  for blob in json_blobs:
    if not blob:
      continue
    try:
      cand = json.loads(blob)
    except json.JSONDecodeError:
      continue
    if isinstance(cand, dict) and isinstance(cand.get("findings"), list):
      raw_items = cand["findings"]
      break
    if isinstance(cand, list):
      raw_items = cand
      break
    saw_other_json = True

  if raw_items is None:
    return [], ("unexpected_gemini_json_schema" if saw_other_json else "malformed_gemini_json")

  normalized: list[dict[str, Any]] = []
  for item in raw_items:
    if not isinstance(item, dict):
      continue
    raw_sev = str(item.get("severity", "P2")).strip().upper()
    sev = _SEVERITY_MAP.get(raw_sev, "P2")
    title = str(item.get("title", "")).strip()
    body = str(item.get("body", item.get("message", ""))).strip()
    if not title and not body:
      continue
    line_val = item.get("line")
    line_num = int(line_val) if isinstance(line_val, int) and not isinstance(line_val, bool) and line_val > 0 else None
    normalized.append(
        {
            "severity": sev,
            "category": str(item.get("category", "Bug")).strip() or "Bug",
            "file": str(item.get("file", "")).strip(),
            "line": line_num,
            "title": title or "Post-submit finding",
            "body": body,
        }
    )
  return normalized, ""


def resolve_dry_run_flag(event_name: str, dry_run_input: str) -> bool:
  """Resolves `dry_run`: defaults to True on `workflow_dispatch`, False on `push`."""
  norm = dry_run_input.strip().lower()
  if norm in ("true", "1", "yes"):
    return True
  if norm in ("false", "0", "no"):
    return False
  return event_name.strip() == "workflow_dispatch"


def build_postsubmit_comment_body(
    *,
    target_sha: str,
    base_sha: str,
    deterministic_findings: Sequence[Mapping[str, Any]],
    blocking_gemini_findings: Sequence[Mapping[str, Any]],
    failed_checks: Sequence[Mapping[str, Any]],
    run_url: str = "",
) -> str:
  """Formats the single idempotent post-submit comment containing only blocking items."""
  marker = f"<!-- maxtext-postsubmit-review:{target_sha} -->"
  short_sha = target_sha[:12]
  lines = [
      marker,
      f"### Post-Submit Review Alert (`{short_sha}`)",
      "",
      (
          f"Automated post-submit checks on `{short_sha}` (diffed against `{base_sha[:12]}`) "
          "flagged the following high-priority item(s) for follow-up:"
      ),
  ]

  if failed_checks or deterministic_findings:
    lines.extend(["", "#### Deterministic / Cross-Model Check Failures"])
    for chk in failed_checks:
      lines.append(f"- **`{chk['name']}`**: {chk.get('summary', 'Check exited non-zero.')}")
    for item in deterministic_findings[:15]:
      loc = str(item.get("file") or "<repo>")
      if item.get("line"):
        loc = f"{loc}:{item['line']}"
      msg = str(item.get("message", "")).strip()
      if len(msg) > 500:
        msg = msg[:500] + "..."
      lines.append(f"- **[{item.get('code', 'CHECK_FAIL')}]** `{loc}`: {msg}")

  if blocking_gemini_findings:
    lines.extend(["", "#### Blocking Review Findings (`P0` / `P1`)"])
    for idx, finding in enumerate(blocking_gemini_findings, start=1):
      loc = (
          f"`{finding['file']}" + (f":{finding['line']}`" if finding.get("line") else "`") if finding.get("file") else ""
      )
      loc_prefix = f"{loc} — " if loc else ""
      lines.append(
          f"{idx}. **[{finding['severity']}] [{finding['category']}] {finding['title']}**: "
          f"{loc_prefix}{finding['body']}"
      )

  if run_url.strip():
    lines.extend(["", f"_Workflow run: [logs]({run_url.strip()})_"])

  body = "\n".join(lines)
  if len(body) > _MAX_COMMENT_CHARS:
    body = body[:_MAX_COMMENT_CHARS] + "\n\n_... [comment truncated]_"
  return body


def evaluate_and_maybe_comment(
    prepass_json_path: Path,
    *,
    gemini_findings_json_path: Path | None = None,
    gemini_summary_text: str = "",
    repository: str = "",
    event_name: str = "push",
    dry_run_input: str = "",
    is_fork: bool = False,
    check_only: bool = False,
    gemini_outcome: str = "",
    run_url: str = "",
    api_client: GitHubApiClient | None = None,
    step_summary_path: Path | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
  """Evaluates Step 1 + Step 2 findings and posts at most 1 idempotent PR comment."""
  prepass = json.loads(prepass_json_path.read_text(encoding="utf-8"))
  target_sha = str(prepass["target_sha"])
  base_sha = str(prepass["base_sha"])
  repo = repository.strip() or str(prepass.get("repository", "")).strip()
  pr_raw = prepass.get("pr_number")
  pr_number = int(pr_raw) if isinstance(pr_raw, int) and not isinstance(pr_raw, bool) and pr_raw > 0 else None
  deterministic_rc = int(prepass.get("deterministic_rc", 0))
  det_findings = [item for item in prepass.get("blocking_findings", []) if isinstance(item, dict)]
  failed_checks = [
      chk for chk in prepass.get("checks", []) if isinstance(chk, dict) and chk.get("status") in ("FAIL", "ERROR")
  ]

  raw_gemini = ""
  if gemini_findings_json_path is not None and gemini_findings_json_path.is_file():
    raw_gemini = gemini_findings_json_path.read_text(encoding="utf-8", errors="replace")
  if not raw_gemini.strip():
    raw_gemini = gemini_summary_text

  gemini_findings, parse_note = parse_gemini_findings(raw_gemini)
  if parse_note and not check_only and (raw_gemini.strip() or gemini_outcome == "failure"):
    sys.stderr.write(
        f"Note: Gemini output note={parse_note!r} (outcome={gemini_outcome!r}); "
        "proceeding safely with deterministic gate.\n"
    )

  blocking_gemini = [f for f in gemini_findings if f["severity"] in ("P0", "P1")]
  should_comment = (deterministic_rc != 0) or bool(blocking_gemini)
  effective_is_fork = is_fork or _parse_bool_flag(os.environ.get("GITHUB_EVENT_REPOSITORY_FORK", ""))
  dry_run = resolve_dry_run_flag(event_name=event_name, dry_run_input=dry_run_input)
  needs_post = should_comment and (not effective_is_fork) and (pr_number is not None) and bool(repo) and (not dry_run)
  if github_output is not None:
    with github_output.open("a", encoding="utf-8") as out_f:
      out_f.write(f"needs_post={'true' if needs_post else 'false'}\n")
  if check_only:
    return {
        "action": "check_only",
        "posted": False,
        "needs_post": needs_post,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
    }

  marker = f"<!-- maxtext-postsubmit-review:{target_sha} -->"

  if not should_comment:
    msg = "Clean merge: 0 P0/P1 findings; posting 0 comments."
    print(msg)
    if step_summary_path is not None:
      step_summary_path.write_text(f"### Post-Submit Review (`{target_sha[:12]}`)\n\n{msg}\n", encoding="utf-8")
    return {
        "action": "clean_skip",
        "posted": False,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": 0,
        "non_blocking_gemini_count": len(gemini_findings),
        "parse_note": parse_note,
    }

  comment_body = build_postsubmit_comment_body(
      target_sha=target_sha,
      base_sha=base_sha,
      deterministic_findings=det_findings,
      blocking_gemini_findings=blocking_gemini,
      failed_checks=failed_checks,
      run_url=run_url,
  )
  if step_summary_path is not None:
    step_summary_path.write_text(comment_body + "\n", encoding="utf-8")

  if effective_is_fork:
    print("Fork repository detected: skipping POST of post-submit comment.")
    return {
        "action": "fork_skip",
        "posted": False,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
        "comment_body": comment_body,
    }

  if pr_number is None or not repo:
    print(f"Blocking findings on {target_sha[:12]}, but no PR number resolved; skipping comment.")
    return {
        "action": "no_pr_skip",
        "posted": False,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
        "comment_body": comment_body,
    }

  if dry_run:
    print(f"Dry-run mode enabled: skipping POST of post-submit comment to PR #{pr_number}.")
    return {
        "action": "dry_run_skip",
        "posted": False,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
        "comment_body": comment_body,
    }

  if api_client is None:
    raise ValueError("api_client is required when posting live PR comments.")

  try:
    existing_comments = api_client.list_paginated(
        f"/repos/{repo}/issues/{pr_number}/comments",
        per_page=100,
    )
  except RuntimeError as exc:
    sys.stderr.write(
        f"Warning: failed to read existing comments on PR #{pr_number} ({exc}); "
        "skipping POST to preserve idempotency.\n"
    )
    return {
        "action": "comment_read_error_skip",
        "posted": False,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
    }

  for existing in existing_comments:
    if marker in str(existing.get("body", "")):
      print(f"Idempotency guard: marker for {target_sha[:12]} already present on PR #{pr_number}; " "posting 0 comments.")
      return {
          "action": "duplicate_skip",
          "posted": False,
          "pr_number": pr_number,
          "deterministic_rc": deterministic_rc,
          "blocking_gemini_count": len(blocking_gemini),
      }

  try:
    resp = api_client.request_json(
        "POST",
        f"/repos/{repo}/issues/{pr_number}/comments",
        body_obj={"body": comment_body},
    )
  except RuntimeError as exc:
    sys.stderr.write(f"Warning: failed to post comment on PR #{pr_number} ({exc}); " "continuing non-blocking.\n")
    return {
        "action": "comment_post_error_skip",
        "posted": False,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "blocking_gemini_count": len(blocking_gemini),
    }
  html_url = resp.get("html_url", "") if isinstance(resp, dict) else ""
  print(f"Posted post-submit review comment on PR #{pr_number}: {html_url}")
  return {
      "action": "posted",
      "posted": True,
      "pr_number": pr_number,
      "comment_url": html_url,
      "deterministic_rc": deterministic_rc,
      "blocking_gemini_count": len(blocking_gemini),
      "comment_body": comment_body,
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for `prepass` and `maybe-comment`."""
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="subcommand", required=True)

  pre = sub.add_parser("prepass", help="Run Step 1 deterministic and cross-model pre-pass.")
  pre.add_argument("--repo-root", type=Path, default=Path("."))
  pre.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
  pre.add_argument("--target-sha", default="HEAD")
  pre.add_argument("--base-sha", default="")
  pre.add_argument("--fallback-scripts-dir", type=Path, default=None)
  pre.add_argument("--prepass-out", type=Path, default=Path(".gemini/postsubmit_prepass.json"))
  pre.add_argument("--authoritative-prepass-out", type=Path, default=None)
  pre.add_argument("--context-out", type=Path, default=Path(".gemini/context.json"))
  pre.add_argument("--github-output", type=Path, default=None)
  pre.add_argument("--is-fork", default=os.environ.get("GITHUB_EVENT_REPOSITORY_FORK", "false"))

  gate = sub.add_parser("maybe-comment", help="Evaluate severity gate and conditionally comment.")
  gate.add_argument("--prepass-json", type=Path, default=Path(".gemini/postsubmit_prepass.json"))
  gate.add_argument(
      "--gemini-findings-json",
      type=Path,
      default=Path(".gemini/postsubmit_gemini_findings.json"),
  )
  gate.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
  gate.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "push"))
  gate.add_argument("--dry-run-input", default="")
  gate.add_argument("--is-fork", default=os.environ.get("GITHUB_EVENT_REPOSITORY_FORK", "false"))
  gate.add_argument("--check-only", action="store_true")
  gate.add_argument("--github-output", type=Path, default=None)
  gate.add_argument("--gemini-outcome", default="")
  gate.add_argument("--run-url", default="")

  args = parser.parse_args(argv)
  token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
  client = GitHubApiClient(token=token)
  fork_flag = _parse_bool_flag(args.is_fork)

  if args.subcommand == "prepass":
    if not args.prepass_out:
      parser.error("--prepass-out is required")
    res = run_prepass(
        repo_root=args.repo_root,
        repository=args.repository,
        target_ref=args.target_sha,
        base_ref=args.base_sha,
        fallback_scripts_dir=args.fallback_scripts_dir,
        prepass_out=args.prepass_out,
        authoritative_prepass_out=args.authoritative_prepass_out,
        context_out=args.context_out,
        github_output=args.github_output,
        api_client=client,
        is_fork=fork_flag,
    )
    print(
        f"Pre-pass complete for {res['target_sha'][:12]} "
        f"(PR={res['pr_number']}, deterministic_rc={res['deterministic_rc']})."
    )
    return 0

  if not args.prepass_json.is_file():
    if args.github_output is not None:
      with args.github_output.open("a", encoding="utf-8") as out_f:
        out_f.write("needs_post=false\n")
    sys.stderr.write(f"Warning: {args.prepass_json} not found; skipping post-submit comment gate.\n")
    return 0

  step_summary_env = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
  step_summary_path = Path(step_summary_env) if (step_summary_env and not args.check_only) else None
  evaluate_and_maybe_comment(
      prepass_json_path=args.prepass_json,
      gemini_findings_json_path=args.gemini_findings_json,
      gemini_summary_text=os.environ.get("GEMINI_SUMMARY", ""),
      repository=args.repository,
      event_name=args.event_name,
      dry_run_input=args.dry_run_input,
      is_fork=fork_flag,
      check_only=args.check_only,
      gemini_outcome=args.gemini_outcome,
      run_url=args.run_url,
      api_client=None if args.check_only else client,
      step_summary_path=step_summary_path,
      github_output=args.github_output,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
