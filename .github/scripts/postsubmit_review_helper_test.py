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

"""Unit tests for `.github/scripts/postsubmit_review_helper.py` and workflow security."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import postsubmit_review_helper as helper  # pylint: disable=wrong-import-position

_USES_PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}\s+#\s*v\d+")


def _git(repo: Path, *args: str) -> str:
  """Runs a git command in a test fixture repository."""
  env = {
      **os.environ,
      "GIT_AUTHOR_NAME": "Test Bot",
      "GIT_AUTHOR_EMAIL": "test@example.com",
      "GIT_COMMITTER_NAME": "Test Bot",
      "GIT_COMMITTER_EMAIL": "test@example.com",
  }
  proc = subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
      capture_output=True,
      text=True,
      env=env,
      check=True,
  )
  return proc.stdout.strip()


class _FakeTransport:
  """Records GitHub REST requests and serves canned HTTP responses."""

  def __init__(self, routes: Mapping[tuple[str, str], tuple[int, Any]] | None = None) -> None:
    self.routes: dict[tuple[str, str], tuple[int, Any]] = dict(routes or {})
    self.calls: list[tuple[str, str, Any]] = []

  def __call__(
      self,
      method: str,
      url: str,
      headers: Mapping[str, str],
      payload: bytes | None,
  ) -> tuple[int, Mapping[str, str], str]:
    del headers
    rel = url.removeprefix("https://api.github.com")
    decoded = json.loads(payload.decode("utf-8")) if payload else None
    self.calls.append((method, rel, decoded))
    key = (method, rel)
    if key in self.routes:
      status, body_obj = self.routes[key]
      text = body_obj if isinstance(body_obj, str) else json.dumps(body_obj)
      return status, {}, text
    if method == "GET":
      return 200, {}, "[]"
    return 201, {}, json.dumps({"id": 999, "html_url": "https://github.com/org/repo/pull/1#issuecomment-999"})

  @property
  def post_calls(self) -> list[tuple[str, str, Any]]:
    """Returns recorded HTTP POST requests."""
    return [c for c in self.calls if c[0] == "POST"]


def _validate_job_steps(job_id: str, job_cfg: Mapping[str, Any]) -> list[str]:
  """Checks a single workflow job's permissions and step security invariants."""
  violations: list[str] = []
  if not isinstance(job_cfg.get("permissions"), dict):
    violations.append(f"Job {job_id!r} is missing explicit least-privilege permissions.")
  for idx, step in enumerate(job_cfg.get("steps", [])):
    if not isinstance(step, dict):
      continue
    step_label = step.get("name", f"step[{idx}]")
    run_script = step.get("run")
    if isinstance(run_script, str) and "${{" in run_script:
      violations.append(f"Job {job_id!r} {step_label!r} contains '${{{{' interpolation inside run:.")
    uses_val = step.get("uses")
    if not isinstance(uses_val, str) or uses_val.startswith("./"):
      continue
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$", uses_val):
      violations.append(f"Job {job_id!r} {step_label!r} uses unpinned action {uses_val!r}.")
    if uses_val.startswith("actions/checkout@"):
      with_cfg = step.get("with", {})
      if not isinstance(with_cfg, dict) or with_cfg.get("persist-credentials") is not False:
        violations.append(f"Job {job_id!r} {step_label!r} must set persist-credentials: false.")
  return violations


def validate_workflow_yaml_security(yaml_text: str) -> list[str]:
  """Validates GitHub Actions workflow YAML against injection and supply-chain rules."""
  violations: list[str] = []
  try:
    import yaml  # pylint: disable=import-outside-toplevel
  except ImportError:
    yaml = None

  if yaml is not None:
    doc = yaml.safe_load(yaml_text)
    if not isinstance(doc, dict):
      return ["Workflow YAML did not parse to a mapping."]
    top_perms = doc.get("permissions")
    if top_perms != {"contents": "read"}:
      violations.append(f"Expected top-level permissions={{'contents': 'read'}}, got {top_perms!r}")

    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict) or not jobs:
      violations.append("Workflow defines no jobs.")
    else:
      for job_id, job_cfg in jobs.items():
        if isinstance(job_cfg, dict):
          violations.extend(_validate_job_steps(str(job_id), job_cfg))

  # Also verify `# vX.Y` version comments on raw `uses:` lines even when PyYAML strips comments.
  for lineno, raw_line in enumerate(yaml_text.splitlines(), start=1):
    stripped = raw_line.strip().removeprefix("-").strip()
    if stripped.startswith("uses:"):
      rhs = stripped[len("uses:") :].strip()
      if not rhs.startswith("./") and not _USES_PIN_RE.match(rhs):
        violations.append(f"Line {lineno} uses: missing 40-char SHA + '# vX.Y' comment: {raw_line.strip()}")
  return violations


class PostSubmitReviewHelperTest(unittest.TestCase):
  """Tests commit resolution, PR lookup, pre-pass execution, comment gating, and security."""

  def test_resolve_commit_range_merge_commit_vs_linear_commit(self) -> None:
    """Verifies first-parent (^1) resolution for merge and linear commits."""
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp)
      _git(repo, "init", "-b", "main")
      (repo / "README.md").write_text("init\n", encoding="utf-8")
      _git(repo, "add", "README.md")
      _git(repo, "commit", "-m", "Initial commit")
      c0 = _git(repo, "rev-parse", "HEAD")

      _git(repo, "checkout", "-b", "feature", c0)
      (repo / "src" / "maxtext" / "layers").mkdir(parents=True)
      (repo / "src" / "maxtext" / "layers" / "demo.py").write_text("X = 1\n", encoding="utf-8")
      _git(repo, "add", "src/maxtext/layers/demo.py")
      _git(repo, "commit", "-m", "Add demo layer")
      c_feat = _git(repo, "rev-parse", "HEAD")

      _git(repo, "checkout", "main")
      (repo / "docs_note.md").write_text("unrelated main advance\n", encoding="utf-8")
      _git(repo, "add", "docs_note.md")
      _git(repo, "commit", "-m", "Advance main before merge")
      c_main_prev = _git(repo, "rev-parse", "HEAD")

      _git(repo, "merge", "--no-ff", "feature", "-m", "Merge pull request #6155 from org/feature")
      c_merge = _git(repo, "rev-parse", "HEAD")

      merge_range = helper.resolve_commit_range(repo, target_ref=c_merge)
      self.assertEqual(merge_range.target_sha, c_merge)
      self.assertEqual(merge_range.base_sha, c_main_prev)
      self.assertEqual(merge_range.parent_shas, (c_main_prev, c_feat))
      self.assertEqual(merge_range.all_changed_files, ("src/maxtext/layers/demo.py",))
      self.assertEqual(merge_range.existing_changed_files, ("src/maxtext/layers/demo.py",))

      # Now add a 1-parent squash-style commit on main that also deletes a file.
      (repo / "docs_note.md").unlink()
      (repo / "src" / "maxtext" / "layers" / "demo.py").write_text("X = 2\n", encoding="utf-8")
      _git(repo, "add", "-A")
      _git(repo, "commit", "-m", "Refine demo scaling (#6156)")
      c_linear = _git(repo, "rev-parse", "HEAD")

      linear_range = helper.resolve_commit_range(repo, target_ref=c_linear)
      self.assertEqual(linear_range.base_sha, c_merge)
      self.assertEqual(linear_range.parent_shas, (c_merge,))
      self.assertEqual(
          set(linear_range.all_changed_files),
          {"docs_note.md", "src/maxtext/layers/demo.py"},
      )
      self.assertEqual(linear_range.existing_changed_files, ("src/maxtext/layers/demo.py",))

      with self.assertRaises(ValueError):
        helper.resolve_commit_range(repo, target_ref="--upload-pack=evil")

  def test_resolve_merged_pr_number_api_fallback_and_no_pr(self) -> None:
    """Verifies PR number lookup via /commits/{sha}/pulls, subject fallback, and no-PR skip."""
    sha = "a" * 40
    transport = _FakeTransport(
        {
            ("GET", f"/repos/AI-Hypercomputer/maxtext/commits/{sha}/pulls?per_page=100&page=1"): (
                200,
                [
                    {
                        "number": 900,
                        "merged_at": None,
                        "merge_commit_sha": None,
                        "base": {"ref": "main"},
                    },
                    {
                        "number": 6155,
                        "merged_at": "2026-11-19T00:00:00Z",
                        "merge_commit_sha": sha,
                        "base": {"ref": "main"},
                    },
                ],
            )
        }
    )
    client = helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0)
    self.assertEqual(
        helper.resolve_merged_pr_number("AI-Hypercomputer/maxtext", sha, "Some commit", api_client=client),
        6155,
    )

    # Empty API list -> subject fallback for merge commit and squash commit.
    empty_client = helper.GitHubApiClient(token="tok", transport=_FakeTransport(), retry_backoff_s=0.0)
    self.assertEqual(
        helper.resolve_merged_pr_number(
            "AI-Hypercomputer/maxtext",
            sha,
            "Merge pull request #6201 from org/branch",
            api_client=empty_client,
        ),
        6201,
    )
    self.assertEqual(
        helper.resolve_merged_pr_number(
            "AI-Hypercomputer/maxtext",
            sha,
            "Add causal mask check (#6202)",
            api_client=empty_client,
        ),
        6202,
    )
    # Direct commit with no PR -> None (no-PR skip).
    self.assertIsNone(
        helper.resolve_merged_pr_number(
            "AI-Hypercomputer/maxtext",
            sha,
            "Automated version bump",
            api_client=empty_client,
        )
    )

  def _write_prepass_fixture(
      self,
      tmp_dir: Path,
      *,
      deterministic_rc: int = 0,
      pr_number: int | None = 6155,
      target_sha: str = "b" * 40,
      blocking_findings: list[dict[str, Any]] | None = None,
  ) -> Path:
    """Writes a synthetic postsubmit_prepass.json fixture for comment-gate tests."""
    prepass_path = tmp_dir / "postsubmit_prepass.json"
    checks = []
    if deterministic_rc != 0:
      checks.append(
          {
              "name": "audit_repo_and_ci_contracts",
              "status": "FAIL",
              "returncode": 1,
              "summary": "1 contract issue.",
          }
      )
    payload = {
        "repository": "AI-Hypercomputer/maxtext",
        "target_sha": target_sha,
        "base_sha": "c" * 40,
        "pr_number": pr_number,
        "deterministic_rc": deterministic_rc,
        "checks": checks,
        "blocking_findings": blocking_findings or [],
        "digest_markdown": "digest",
    }
    prepass_path.write_text(json.dumps(payload), encoding="utf-8")
    return prepass_path

  def test_clean_merge_posts_zero_comments(self) -> None:
    """Verifies clean merges (rc=0, only P2/NIT findings) perform zero API calls."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      prepass_path = self._write_prepass_fixture(tmp_dir, deterministic_rc=0, pr_number=6155)
      gemini_path = tmp_dir / "gemini.json"
      gemini_path.write_text(
          json.dumps(
              {
                  "verdict": "CLEAN",
                  "findings": [
                      {
                          "severity": "P2",
                          "category": "Tests",
                          "file": "tests/unit/foo_test.py",
                          "line": 12,
                          "title": "Minor assertion suggestion",
                          "body": "Consider checking dtype explicitly.",
                      },
                      {
                          "severity": "NIT",
                          "category": "Bug",
                          "file": "src/maxtext/foo.py",
                          "line": 4,
                          "title": "Docstring typo",
                          "body": "Typo in comment.",
                      },
                  ],
              }
          ),
          encoding="utf-8",
      )
      transport = _FakeTransport()
      client = helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0)
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        res = helper.evaluate_and_maybe_comment(
            prepass_json_path=prepass_path,
            gemini_findings_json_path=gemini_path,
            event_name="push",
            api_client=client,
        )
      self.assertEqual(res["action"], "clean_skip")
      self.assertFalse(res["posted"])
      self.assertEqual(len(transport.post_calls), 0)
      self.assertEqual(len(transport.calls), 0)
      self.assertIn("Clean merge: 0 P0/P1 findings; posting 0 comments.", buf.getvalue())

  def test_blocking_deterministic_or_p0_p1_posts_single_comment_with_marker(self) -> None:
    """Verifies deterministic failures and P0/P1 findings post 1 marked comment."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      sha = "1234567890abcdef" * 2 + "12345678"

      # Case A: deterministic failure (rc=1) with empty Gemini findings.
      prepass_det = self._write_prepass_fixture(
          tmp_dir,
          deterministic_rc=1,
          pr_number=6155,
          target_sha=sha,
          blocking_findings=[
              {
                  "source": "audit_repo_and_ci_contracts",
                  "code": "SILENT_CONFIG_FALLBACK",
                  "severity": "HIGH",
                  "file": "src/maxtext/layers/foo.py",
                  "line": 22,
                  "message": "Unregistered getattr fallback.",
              }
          ],
      )
      transport_a = _FakeTransport()
      client_a = helper.GitHubApiClient(token="tok", transport=transport_a, retry_backoff_s=0.0)
      res_a = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_det,
          gemini_summary_text='{"verdict": "CLEAN", "findings": []}',
          event_name="push",
          api_client=client_a,
      )
      self.assertTrue(res_a["posted"])
      self.assertEqual(len(transport_a.post_calls), 1)
      posted_body_a = transport_a.post_calls[0][2]["body"]
      self.assertIn(f"<!-- maxtext-postsubmit-review:{sha} -->", posted_body_a)
      self.assertIn("SILENT_CONFIG_FALLBACK", posted_body_a)

      # Case B: deterministic_rc=0, P0 + P1 + P2 Gemini findings in fenced markdown summary.
      prepass_clean = self._write_prepass_fixture(
          tmp_dir,
          deterministic_rc=0,
          pr_number=6156,
          target_sha=sha,
      )
      fenced_summary = (
          "Audit complete.\n```json\n"
          + json.dumps(
              {
                  "verdict": "BLOCKING_ISSUES_FOUND",
                  "findings": [
                      {
                          "severity": "P0",
                          "category": "Bug",
                          "file": "src/maxtext/layers/attentions.py",
                          "line": 88,
                          "title": "Cross-document attention leakage under packing",
                          "body": "Missing segment_ids mask when packing=True.",
                      },
                      {
                          "severity": "P1",
                          "category": "Perf",
                          "file": "src/maxtext/train.py",
                          "line": 140,
                          "title": "Host sync inside step loop",
                          "body": "float(loss) blocks device execution every step.",
                      },
                      {
                          "severity": "P2",
                          "category": "Tests",
                          "file": "tests/unit/attn_test.py",
                          "line": 10,
                          "title": "Optional extra assertion",
                          "body": "Non-blocking suggestion.",
                      },
                  ],
              }
          )
          + "\n```\n"
      )
      transport_b = _FakeTransport()
      client_b = helper.GitHubApiClient(token="tok", transport=transport_b, retry_backoff_s=0.0)
      res_b = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_clean,
          gemini_findings_json_path=tmp_dir / "missing.json",
          gemini_summary_text=fenced_summary,
          event_name="push",
          run_url="https://github.com/AI-Hypercomputer/maxtext/actions/runs/777",
          api_client=client_b,
      )
      self.assertTrue(res_b["posted"])
      self.assertEqual(len(transport_b.post_calls), 1)
      posted_body_b = transport_b.post_calls[0][2]["body"]
      self.assertIn(f"<!-- maxtext-postsubmit-review:{sha} -->", posted_body_b)
      self.assertIn("[P0]", posted_body_b)
      self.assertIn("[P1]", posted_body_b)
      self.assertNotIn("Optional extra assertion", posted_body_b)
      self.assertNotIn("[P2]", posted_body_b)

      # Case C: no-PR commit with blocking finding -> 0 POST calls.
      prepass_no_pr = self._write_prepass_fixture(
          tmp_dir,
          deterministic_rc=1,
          pr_number=None,
          target_sha=sha,
      )
      transport_c = _FakeTransport()
      client_c = helper.GitHubApiClient(token="tok", transport=transport_c, retry_backoff_s=0.0)
      res_c = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_no_pr,
          event_name="push",
          api_client=client_c,
      )
      self.assertEqual(res_c["action"], "no_pr_skip")
      self.assertEqual(len(transport_c.post_calls), 0)

  def test_duplicate_run_idempotency_across_paginated_comments(self) -> None:
    """Verifies paginated comment inspection skips duplicate posts for the same SHA."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      sha = "d" * 40
      prepass_path = self._write_prepass_fixture(tmp_dir, deterministic_rc=1, pr_number=6155, target_sha=sha)
      page1 = [{"id": i, "body": f"Earlier comment {i}"} for i in range(1, 101)]
      page2 = [
          {"id": 101, "body": "Another comment"},
          {"id": 102, "body": f"<!-- maxtext-postsubmit-review:{sha} -->\nAlready posted."},
      ]
      transport = _FakeTransport(
          {
              ("GET", "/repos/AI-Hypercomputer/maxtext/issues/6155/comments?per_page=100&page=1"): (200, page1),
              ("GET", "/repos/AI-Hypercomputer/maxtext/issues/6155/comments?per_page=100&page=2"): (200, page2),
          }
      )
      client = helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0)
      res = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_path,
          event_name="push",
          api_client=client,
      )
      self.assertEqual(res["action"], "duplicate_skip")
      self.assertFalse(res["posted"])
      self.assertEqual(len(transport.post_calls), 0)
      self.assertEqual(len(transport.calls), 2)

  def test_malformed_or_empty_gemini_output_fails_safe(self) -> None:
    """Verifies malformed or empty Gemini output exits cleanly with 0 comments when rc=0."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      prepass_clean = self._write_prepass_fixture(tmp_dir, deterministic_rc=0, pr_number=6155)
      bad_gemini = tmp_dir / "bad_gemini.json"
      bad_gemini.write_text("{truncated json output...", encoding="utf-8")

      transport = _FakeTransport()
      client = helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0)
      res = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_clean,
          gemini_findings_json_path=bad_gemini,
          gemini_outcome="failure",
          event_name="push",
          api_client=client,
      )
      self.assertEqual(res["action"], "clean_skip")
      self.assertFalse(res["posted"])
      self.assertEqual(len(transport.post_calls), 0)

      # Empty Gemini output -> also safe 0-comment clean exit.
      res_empty = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_clean,
          gemini_findings_json_path=tmp_dir / "nonexistent.json",
          gemini_summary_text="",
          event_name="push",
          api_client=client,
      )
      self.assertEqual(res_empty["action"], "clean_skip")
      self.assertEqual(len(transport.post_calls), 0)

  def test_workflow_dispatch_dry_run_skips_post_unless_false(self) -> None:
    """Verifies workflow_dispatch defaults to dry_run=True and skips POST unless false."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      prepass_fail = self._write_prepass_fixture(tmp_dir, deterministic_rc=1, pr_number=6155)
      transport = _FakeTransport()
      client = helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0)

      # Default dry_run on workflow_dispatch -> 0 POST calls.
      res_default = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="workflow_dispatch",
          dry_run_input="",
          api_client=client,
      )
      self.assertEqual(res_default["action"], "dry_run_skip")
      self.assertEqual(len(transport.post_calls), 0)

      # Explicit dry_run=true -> 0 POST calls.
      res_true = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="workflow_dispatch",
          dry_run_input="true",
          api_client=client,
      )
      self.assertEqual(res_true["action"], "dry_run_skip")
      self.assertEqual(len(transport.post_calls), 0)

      # Explicit dry_run=false -> 1 POST call.
      res_false = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="workflow_dispatch",
          dry_run_input="false",
          api_client=client,
      )
      self.assertEqual(res_false["action"], "posted")
      self.assertEqual(len(transport.post_calls), 1)

  def test_run_prepass_executes_checks_and_cleans_temporary_worktree(self) -> None:
    """Verifies run_prepass writes context artifacts and removes temporary git worktrees."""
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp) / "repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / "src" / "maxtext" / "layers").mkdir(parents=True)
      (repo / "src" / "maxtext" / "checkpoint_conversion").mkdir(parents=True)
      (repo / "src" / "maxtext" / "layers" / "attn.py").write_text("X = 1\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Base commit")

      (repo / "src" / "maxtext" / "layers" / "attn.py").write_text("X = 2\n", encoding="utf-8")
      (repo / "src" / "maxtext" / "checkpoint_conversion" / "param_mapping.py").write_text(
          "MAP = {}\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #6300 from org/change")

      scripts_dir = Path(tmp) / "fallback_scripts"
      scripts_dir.mkdir()
      for ok_script in (
          "audit_repo_and_ci_contracts.py",
          "audit_pr_review_hygiene.py",
          "sweep_param_mapping_contracts.py",
      ):
        (scripts_dir / ok_script).write_text(
            "import json, sys\nprint(json.dumps({'status': 'PASS', 'issues': []}))\nsys.exit(0)\n",
            encoding="utf-8",
        )
      # Fake abstract_model_diff.py that writes dumps and fails compare (rc=1) to verify
      # both blocking finding capture AND finally: git worktree cleanup.
      (scripts_dir / "abstract_model_diff.py").write_text(
          "import json, pathlib, sys\n"
          "cmd = sys.argv[1]\n"
          "if cmd == 'dump':\n"
          "  out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
          "  out.mkdir(parents=True, exist_ok=True)\n"
          "  (out / 'llama3-8b.json').write_text('{}')\n"
          "  sys.exit(0)\n"
          "if cmd == 'compare':\n"
          "  md = pathlib.Path(sys.argv[sys.argv.index('--md') + 1])\n"
          "  md.write_text('# Tier A abstract diff: 1 models differ\\n## llama3-8b\\n- shape diff\\n')\n"
          "  sys.exit(1)\n"
          "sys.exit(0)\n",
          encoding="utf-8",
      )

      gh_out = Path(tmp) / "github_output.txt"
      with mock.patch.object(helper, "_python_has_jax", return_value=True):
        prepass = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            fallback_scripts_dir=scripts_dir,
            github_output=gh_out,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )

      self.assertEqual(prepass["pr_number"], 6300)
      self.assertEqual(prepass["deterministic_rc"], 1)
      self.assertEqual(len(prepass["blocking_findings"]), 1)
      self.assertEqual(prepass["blocking_findings"][0]["code"], "CROSS_MODEL_SHAPE_OR_SHARDING_DIFF")

      context_json = json.loads((repo / ".gemini" / "context.json").read_text(encoding="utf-8"))
      self.assertEqual(context_json["pull_request_number"], "6300")
      self.assertIn("postsubmit_gemini_findings.json", context_json["additional_context"])

      wt_lines = [
          line for line in _git(repo, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")
      ]
      self.assertEqual(len(wt_lines), 1)
      self.assertIn("pr_number=6300", gh_out.read_text(encoding="utf-8"))

  def test_workflow_yaml_and_gemini_command_security_contracts(self) -> None:
    """Verifies postsubmit_review.yml and maxtext-postsubmit-review.toml security rules."""
    # Negative control: ensure our security validator catches bad workflow YAML.
    bad_yaml = (
        "name: Bad\n"
        "on: [push]\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - run: echo ${{ github.actor }}\n"
    )
    bad_findings = validate_workflow_yaml_security(bad_yaml)
    self.assertGreaterEqual(len(bad_findings), 3)

    repo_root = _SCRIPT_DIR.parents[1]
    wf_path = repo_root / ".github" / "workflows" / "postsubmit_review.yml"
    cmd_path = repo_root / ".gemini" / "commands" / "maxtext-postsubmit-review.toml"
    if not wf_path.is_file():
      # Support running from staged scratch directory before installation into repo.
      wf_path = _SCRIPT_DIR / "postsubmit_review.yml"
      cmd_path = _SCRIPT_DIR / "maxtext-postsubmit-review.toml"

    wf_text = wf_path.read_text(encoding="utf-8")
    self.assertEqual(validate_workflow_yaml_security(wf_text), [])

    # Verify read-only MCP allowlist omits write tools.
    self.assertNotIn("pull_request_review_write", wf_text)
    self.assertNotIn("add_comment_to_pending_review", wf_text)
    self.assertNotIn("add_issue_comment", wf_text)

    import tomllib  # pylint: disable=import-outside-toplevel

    cmd_text = cmd_path.read_text(encoding="utf-8")
    parsed_toml = tomllib.loads(cmd_text)
    self.assertIn("prompt", parsed_toml)
    self.assertIn("description", parsed_toml)
    self.assertIn("postsubmit_gemini_findings.json", parsed_toml["prompt"])
    self.assertIn("maxtext-review-criteria.md", parsed_toml["prompt"])

  def test_high_confidence_contract_filtering_and_explicit_python_arg(self) -> None:
    """Verifies HIGH/HIGH contract filtering and explicit --python forwarding to Tier A."""
    sample_stdout = json.dumps(
        {
            "status": "FAIL",
            "issues": [
                {
                    "code": "SILENT_CONFIG_FALLBACK",
                    "severity": "HIGH",
                    "confidence": "HIGH",
                    "file": "src/maxtext/layers/a.py",
                    "line": 10,
                    "message": "High-confidence violation.",
                },
                {
                    "code": "LOW_WARN",
                    "severity": "LOW",
                    "confidence": "MEDIUM",
                    "file": "src/maxtext/layers/a.py",
                    "line": 20,
                    "message": "Informational warning.",
                },
            ],
        }
    )
    filtered = helper._extract_issues_from_json_stdout(  # pylint: disable=protected-access
        sample_stdout,
        "audit_repo_and_ci_contracts",
        only_high_confidence=True,
    )
    self.assertEqual(len(filtered), 1)
    self.assertEqual(filtered[0]["code"], "SILENT_CONFIG_FALLBACK")

    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp) / "repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / "src" / "maxtext" / "configs").mkdir(parents=True)
      (repo / "src" / "maxtext" / "configs" / "sharding.py").write_text("A = 1\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Base")
      base_sha = _git(repo, "rev-parse", "HEAD")

      (repo / "src" / "maxtext" / "configs" / "sharding.py").write_text("A = 2\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Update sharding")

      arg_log = Path(tmp) / "argv_env_log.jsonl"
      fake_diff = Path(tmp) / "abstract_model_diff.py"
      fake_diff.write_text(
          "import json, os, pathlib, sys\n"
          f"log = pathlib.Path({str(arg_log)!r})\n"
          "entry = {'argv': sys.argv[1:], 'mds': os.environ.get('TPU_SKIP_MDS_QUERY'), "
          "'multi': os.environ.get('ALLOW_MULTIPLE_LIBTPU_LOAD')}\n"
          "with log.open('a', encoding='utf-8') as f:\n"
          "  f.write(json.dumps(entry) + '\\n')\n"
          "cmd = sys.argv[1]\n"
          "if cmd == 'dump':\n"
          "  out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
          "  out.mkdir(parents=True, exist_ok=True)\n"
          "  (out / 'llama3-8b.json').write_text('{}')\n"
          "sys.exit(0)\n",
          encoding="utf-8",
      )
      res = helper._run_abstract_model_diff_check(  # pylint: disable=protected-access
          script_path=fake_diff,
          repo_root=repo,
          base_sha=base_sha,
          changed_files=("src/maxtext/configs/sharding.py",),
          python_exe=sys.executable,
      )
      self.assertEqual(res["status"], "PASS")
      records = [json.loads(line) for line in arg_log.read_text(encoding="utf-8").splitlines()]
      dump_And_sweep = [r for r in records if r["argv"][0] in ("dump", "sharding-sweep")]
      self.assertEqual(len(dump_And_sweep), 3)
      for rec in dump_And_sweep:
        self.assertIn("--python", rec["argv"])
        py_idx = rec["argv"].index("--python")
        self.assertEqual(rec["argv"][py_idx + 1], sys.executable)
        self.assertEqual(rec["mds"], "1")
        self.assertEqual(rec["multi"], "true")

  def test_run_prepass_runs_skill_and_script_self_tests_when_touched(self) -> None:
    """Verifies self-tests in .github/scripts and .agents/skills run when touched."""
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp) / "repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / ".github" / "scripts").mkdir(parents=True)
      (repo / ".agents" / "skills" / "demo" / "tests").mkdir(parents=True)
      (repo / ".github" / "scripts" / "tool.py").write_text("VAL = 1\n", encoding="utf-8")
      test_file = repo / ".agents" / "skills" / "demo" / "tests" / "demo_test.py"
      test_file.write_text(
          "import unittest\nclass T(unittest.TestCase):\n  def test_ok(self):\n    self.assertEqual(1, 1)\n"
          "if __name__ == '__main__':\n  unittest.main()\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Base")

      (repo / ".github" / "scripts" / "tool.py").write_text("VAL = 2\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #6400 from org/skill-edit")

      prepass_ok = helper.run_prepass(
          repo_root=repo,
          repository="AI-Hypercomputer/maxtext",
          api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
      )
      by_name_ok = {c["name"]: c for c in prepass_ok["checks"]}
      self.assertEqual(by_name_ok["skill_and_script_self_tests"]["status"], "PASS")
      self.assertEqual(prepass_ok["deterministic_rc"], 0)

      # Now make the self-test fail and confirm deterministic_rc becomes 1.
      test_file.write_text(
          "import unittest\nclass T(unittest.TestCase):\n  def test_fail(self):\n    self.assertEqual(1, 2)\n"
          "if __name__ == '__main__':\n  unittest.main()\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #6401 from org/broken-skill")

      prepass_fail = helper.run_prepass(
          repo_root=repo,
          repository="AI-Hypercomputer/maxtext",
          api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
      )
      by_name_fail = {c["name"]: c for c in prepass_fail["checks"]}
      self.assertEqual(by_name_fail["skill_and_script_self_tests"]["status"], "FAIL")
      self.assertEqual(prepass_fail["deterministic_rc"], 1)
      codes = {f["code"] for f in prepass_fail["blocking_findings"]}
      self.assertIn("SKILL_OR_SCRIPT_SELF_TEST_FAILURE", codes)

  def test_live_pr1_scripts_integration(self) -> None:
    """Exercises run_prepass + maybe-comment against PR 1's real audit scripts."""
    repo_root = _SCRIPT_DIR.parents[1]
    real_scripts_dir = repo_root / ".agents" / "skills" / "maxtext-pr-upload" / "scripts"
    if not (real_scripts_dir / "audit_repo_and_ci_contracts.py").is_file():
      self.skipTest("PR 1 scripts not present in this checkout")

    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp) / "live_repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / "src" / "maxtext" / "utils").mkdir(parents=True)
      (repo / "src" / "maxtext" / "utils" / "sample.py").write_text(
          "def add_one(x: int) -> int:\n  return x + 1\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Initial clean baseline")

      # 1. Clean Python change -> real audit_repo_and_ci_contracts & audit_pr_review_hygiene PASS.
      (repo / "src" / "maxtext" / "utils" / "sample.py").write_text(
          "def add_one(x: int) -> int:\n  return x + 2\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #7001 from org/clean-py")

      with mock.patch.object(helper, "_python_has_jax", return_value=False):
        res_py = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            fallback_scripts_dir=real_scripts_dir,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )
      checks_py = {c["name"]: c["status"] for c in res_py["checks"]}
      self.assertEqual(checks_py["audit_repo_and_ci_contracts"], "PASS")
      self.assertEqual(checks_py["audit_pr_review_hygiene"], "PASS")
      self.assertEqual(res_py["deterministic_rc"], 0)

      # 2. Zero-.py Markdown/YAML-only change -> real scripts skip cleanly without parser.error (rc=2).
      (repo / "docs").mkdir()
      (repo / "docs" / "notes.md").write_text("# Notes\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #7002 from org/docs-only")

      with mock.patch.object(helper, "_python_has_jax", return_value=False):
        res_md = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            fallback_scripts_dir=real_scripts_dir,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )
      checks_md = {c["name"]: c["status"] for c in res_md["checks"]}
      self.assertEqual(checks_md["audit_repo_and_ci_contracts"], "SKIPPED")
      self.assertEqual(checks_md["audit_pr_review_hygiene"], "SKIPPED")
      self.assertEqual(res_md["deterministic_rc"], 0)

      # 3. Shell benchmark hazard -> real audit_pr_review_hygiene flags BENCHMARK_CRIPPLED_XLA_FLAG.
      (repo / "benchmarks").mkdir()
      (repo / "benchmarks" / "run_moe.sh").write_text(
          "#!/bin/bash\nexport LIBTPU_INIT_ARGS='--xla_tpu_enable_ragged_dot_kernel=false'\n",
          encoding="utf-8",
      )
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #7003 from org/bad-bench")

      with mock.patch.object(helper, "_python_has_jax", return_value=False):
        res_bad = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            fallback_scripts_dir=real_scripts_dir,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )
      self.assertEqual(res_bad["deterministic_rc"], 1)
      transport = _FakeTransport()
      comment_res = helper.evaluate_and_maybe_comment(
          prepass_json_path=repo / ".gemini" / "postsubmit_prepass.json",
          event_name="push",
          api_client=helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0),
      )
      self.assertTrue(comment_res["posted"])
      self.assertEqual(len(transport.post_calls), 1)
      self.assertIn("BENCHMARK_CRIPPLED_XLA_FLAG", transport.post_calls[0][2]["body"])

  def test_parse_gemini_findings_multi_fence_extracts_final_findings_block(self) -> None:
    """Verifies [RED 1.1]: earlier non-findings JSON fences do not shadow the final findings block."""
    multi_fence_response = (
        "Reviewed input context:\n"
        "```json\n"
        '{"repository": "AI-Hypercomputer/maxtext", "target_sha": "abc1234"}\n'
        "```\n"
        "Final structured post-submit findings:\n"
        "```json\n"
        + json.dumps(
            {
                "verdict": "BLOCKING_ISSUES_FOUND",
                "summary": "Found critical bug.",
                "findings": [
                    {
                        "severity": "P0",
                        "category": "Bug",
                        "file": "src/maxtext/layers/moe.py",
                        "line": 77,
                        "title": "Silent expert drop without mask",
                        "body": "Tokens exceeding capacity are zeroed without renorm.",
                    }
                ],
            }
        )
        + "\n```\n"
    )
    findings, note = helper.parse_gemini_findings(multi_fence_response)
    self.assertEqual(note, "")
    self.assertEqual(len(findings), 1)
    self.assertEqual(findings[0]["severity"], "P0")
    self.assertEqual(findings[0]["title"], "Silent expert drop without mask")

    # Non-findings JSON alone still returns empty with unexpected_gemini_json_schema.
    other_only, other_note = helper.parse_gemini_findings('```json\n{"target_sha": "abc"}\n```')
    self.assertEqual(other_only, [])
    self.assertEqual(other_note, "unexpected_gemini_json_schema")

  def test_fork_auth_and_app_token_workflow_guards(self) -> None:
    """Verifies [ORANGE 1.2]: fork push skip, Gemini auth gate, pinned CLI version, and App-token safety."""
    repo_root = _SCRIPT_DIR.parents[1]
    wf_path = repo_root / ".github" / "workflows" / "postsubmit_review.yml"
    if not wf_path.is_file():
      wf_path = _SCRIPT_DIR / "postsubmit_review.yml"
    wf_text = wf_path.read_text(encoding="utf-8")

    import yaml  # pylint: disable=import-outside-toplevel

    doc = yaml.safe_load(wf_text)
    job = doc["jobs"]["postsubmit_review"]
    job_if = str(job.get("if", ""))
    self.assertIn("!github.event.repository.fork", job_if)
    self.assertIn("workflow_dispatch", job_if)

    steps_by_id = {s.get("id"): s for s in job["steps"] if isinstance(s, dict) and s.get("id")}
    gemini_step = steps_by_id["gemini_postsubmit_review"]
    self.assertIn("has_gemini_auth", str(gemini_step.get("if", "")))
    cli_ver_expr = str(gemini_step["with"]["gemini_cli_version"])
    self.assertIn("0.34.0", cli_ver_expr)

    mint_step = steps_by_id["mint_identity_token"]
    self.assertIs(mint_step.get("continue-on-error"), True)
    mint_if = str(mint_step.get("if", ""))
    self.assertIn("steps.check_comment_needed.outputs.needs_post == 'true'", mint_if)
    self.assertIn("!github.event.repository.fork", mint_if)
    self.assertIn("inputs.dry_run", mint_if)

    # Also verify defense-in-depth authoritative prepass JSON outside .gemini/.
    eval_step = job["steps"][-1]
    self.assertIn("${TOOLS_STAGE_DIR}/postsubmit_prepass.json", str(eval_step.get("run", "")))

    # Helper fork safety: on forks, subject #N fallback is disabled and maybe-comment skips POST.
    empty_client = helper.GitHubApiClient(token="tok", transport=_FakeTransport(), retry_backoff_s=0.0)
    self.assertIsNone(
        helper.resolve_merged_pr_number(
            "fork-owner/maxtext",
            "a" * 40,
            "Merge pull request #6155 from org/branch",
            api_client=empty_client,
            is_fork=True,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      clean_dir = tmp_dir / "clean"
      clean_dir.mkdir()
      prepass_clean = self._write_prepass_fixture(clean_dir, deterministic_rc=0, pr_number=6155)
      gh_out_clean = tmp_dir / "gh_out_clean.txt"
      chk_clean = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_clean,
          event_name="push",
          check_only=True,
          github_output=gh_out_clean,
      )
      self.assertFalse(chk_clean["needs_post"])
      self.assertIn("needs_post=false", gh_out_clean.read_text(encoding="utf-8"))

      prepass_fail = self._write_prepass_fixture(tmp_dir, deterministic_rc=1, pr_number=6155)
      gh_out_fail = tmp_dir / "gh_out_fail.txt"
      chk_fail = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="push",
          check_only=True,
          github_output=gh_out_fail,
      )
      self.assertTrue(chk_fail["needs_post"])
      self.assertIn("needs_post=true", gh_out_fail.read_text(encoding="utf-8"))

      transport = _FakeTransport()
      res_fork = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="workflow_dispatch",
          dry_run_input="false",
          is_fork=True,
          api_client=helper.GitHubApiClient(token="tok", transport=transport, retry_backoff_s=0.0),
      )
      self.assertEqual(res_fork["action"], "fork_skip")
      self.assertEqual(len(transport.post_calls), 0)

  def test_post_comment_http_error_and_check_timeout_are_non_blocking(self) -> None:
    """Verifies [ORANGE 1.3]: POST 403/500 and subprocess TimeoutExpired do not crash."""
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = Path(tmp)
      prepass_fail = self._write_prepass_fixture(tmp_dir, deterministic_rc=1, pr_number=6155)
      transport_403 = _FakeTransport(
          {
              ("GET", "/repos/AI-Hypercomputer/maxtext/issues/6155/comments?per_page=100&page=1"): (200, []),
              ("POST", "/repos/AI-Hypercomputer/maxtext/issues/6155/comments"): (
                  403,
                  {"message": "Resource not accessible by integration"},
              ),
          }
      )
      client = helper.GitHubApiClient(token="tok", transport=transport_403, retry_backoff_s=0.0)
      res_post_err = helper.evaluate_and_maybe_comment(
          prepass_json_path=prepass_fail,
          event_name="push",
          api_client=client,
      )
      self.assertEqual(res_post_err["action"], "comment_post_error_skip")
      self.assertFalse(res_post_err["posted"])

      # Subprocess TimeoutExpired inside run_prepass still writes JSON artifacts and records deterministic_rc=1.
      repo = tmp_dir / "timeout_repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / "src" / "maxtext").mkdir(parents=True)
      (repo / "src" / "maxtext" / "mod.py").write_text("X = 1\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Base")
      (repo / "src" / "maxtext" / "mod.py").write_text("X = 2\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Merge pull request #7100 from org/slow")

      scripts_dir = tmp_dir / "slow_scripts"
      scripts_dir.mkdir()
      (scripts_dir / "audit_repo_and_ci_contracts.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

      orig_run = subprocess.run

      def _timeout_on_contracts(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(cmd, list) and any("audit_repo_and_ci_contracts.py" in str(x) for x in cmd):
          raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
        return orig_run(cmd, *args, **kwargs)  # pylint: disable=subprocess-run-check

      auth_copy = tmp_dir / "auth_prepass.json"
      with (
          mock.patch.object(helper, "_python_has_jax", return_value=False),
          mock.patch("subprocess.run", side_effect=_timeout_on_contracts),
      ):
        prepass_timeout = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            fallback_scripts_dir=scripts_dir,
            authoritative_prepass_out=auth_copy,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )
      self.assertEqual(prepass_timeout["deterministic_rc"], 1)
      by_name = {c["name"]: c for c in prepass_timeout["checks"]}
      self.assertEqual(by_name["audit_repo_and_ci_contracts"]["status"], "ERROR")
      self.assertEqual(by_name["audit_repo_and_ci_contracts"]["returncode"], -1)
      self.assertTrue((repo / ".gemini" / "postsubmit_prepass.json").is_file())
      self.assertTrue((repo / ".gemini" / "context.json").is_file())
      self.assertTrue(auth_copy.is_file())

  def test_root_commit_and_short_sha_workflow_resolution(self) -> None:
    """Verifies [ORANGE 1.4]: 0-parent root commits and 7-39 char SHA resolution."""
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp) / "root_repo"
      repo.mkdir()
      _git(repo, "init", "-b", "main")
      (repo / "src" / "maxtext").mkdir(parents=True)
      (repo / "src" / "maxtext" / "init_mod.py").write_text("ROOT_VAL = 1\n", encoding="utf-8")
      _git(repo, "add", ".")
      _git(repo, "commit", "-m", "Root commit (#1)")
      root_sha = _git(repo, "rev-parse", "HEAD")

      root_range = helper.resolve_commit_range(repo, target_ref=root_sha[:9])
      self.assertEqual(root_range.target_sha, root_sha)
      self.assertEqual(root_range.parent_shas, ())
      self.assertEqual(root_range.base_sha, "4b825dc642cb6eb9a060e54bf8d69288fbee4904")
      self.assertEqual(root_range.all_changed_files, ("src/maxtext/init_mod.py",))

      real_scripts_dir = _SCRIPT_DIR.parents[1] / ".agents" / "skills" / "maxtext-pr-upload" / "scripts"
      fallback = real_scripts_dir if real_scripts_dir.is_dir() else None
      with mock.patch.object(helper, "_python_has_jax", return_value=True):
        root_prepass = helper.run_prepass(
            repo_root=repo,
            repository="AI-Hypercomputer/maxtext",
            target_ref=root_sha[:9],
            fallback_scripts_dir=fallback,
            api_client=helper.GitHubApiClient(transport=_FakeTransport(), retry_backoff_s=0.0),
        )
      by_name = {c["name"]: c for c in root_prepass["checks"]}
      self.assertEqual(by_name["abstract_model_diff"]["status"], "SKIPPED")
      self.assertEqual(root_prepass["deterministic_rc"], 0)

      # Verify workflow YAML handles <40-char RAW_SHA via commits API and empty-tree root fallback.
      wf_path = _SCRIPT_DIR.parents[1] / ".github" / "workflows" / "postsubmit_review.yml"
      if not wf_path.is_file():
        wf_path = _SCRIPT_DIR / "postsubmit_review.yml"
      wf_text = wf_path.read_text(encoding="utf-8")
      self.assertIn("4b825dc642cb6eb9a060e54bf8d69288fbee4904", wf_text)
      self.assertIn("commits/${RAW_SHA}", wf_text)


if __name__ == "__main__":
  unittest.main()
