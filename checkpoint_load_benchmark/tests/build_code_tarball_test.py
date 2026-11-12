# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests build_code_tarball.py on temporary git repositories whose `origin` is a local bare repository (no network)."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILDER_RELPATH = "checkpoint_load_benchmark/scripts/build_code_tarball.py"

BRANCH = "bench/code-tarball"
GCS_CODE_DIR = "gs://test-bucket/code"
TARBALL = "maxtext_lance_prs.tar.gz"
BOOTSTRAP = "checkpoint_load_benchmark/bootstrap_tpu7x.py"
QUEUE = "checkpoint_load_benchmark/scripts/tpu7x_task_queue.py"
DECODERS = "src/maxtext/layers/decoders.py"
# The date of every test commit, in a time zone other than UTC.
COMMIT_DATE = "2026-11-12T07:03:27+02:00"

# Committed files under the 4 tarball paths (the test repositories also commit a copy of the builder there).
TARBALL_FILES = {
    "src/maxtext/__init__.py": "",
    # A file name longer than the 100-character name field of a tar header, so git writes its path in a pax header.
    f"src/maxtext/{'long_name_' * 10}.py": "",
    DECODERS: "DECODERS = 'committed'\n",
    BOOTSTRAP: "BOOTSTRAP = 'committed'\n",
    QUEUE: "QUEUE = 'committed'\n",
    "pyproject.toml": "[project]\nname = 'maxtext'\n",
    "README.md": "# MaxText\n",
}
# Committed files outside the tarball paths.
OTHER_FILES = {"tests/unit/decoders_test.py": "DECODERS_TEST = 'committed'\n", "docs/guide.md": "# Guide\n"}
# Keeps the git config of the machine (system hooks, global commit signing or hooks path) out of the test repositories.
ISOLATED_GIT_CONFIG = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
# What `gcloud storage ls` prints when no object matches its URL, and what it prints for an error.
LS_NOT_FOUND = "ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
LS_FORBIDDEN = "ERROR: (gcloud.storage.ls) HTTPError 403: test@example.com does not have storage.objects.list access.\n"


def _git(repo: str, *args: str) -> str:
  """Runs git in `repo` as a test user, with COMMIT_DATE as the commit date, and returns its output."""
  env = {**os.environ, **ISOLATED_GIT_CONFIG, "GIT_AUTHOR_DATE": COMMIT_DATE, "GIT_COMMITTER_DATE": COMMIT_DATE}
  cmd = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args]
  return subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _write(path: str, text: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(text)


class BuildCodeTarballTest(unittest.TestCase):
  """Runs `build_code_tarball.py --out <out>` from a test repository whose HEAD is pushed to `origin`."""

  def setUp(self):
    super().setUp()
    # The git commands of the builder, and of its CLI subprocess, inherit the isolated config too.
    env_patcher = mock.patch.dict(os.environ, ISOLATED_GIT_CONFIG)
    env_patcher.start()
    self.addCleanup(env_patcher.stop)
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    self.repo = os.path.join(tmp_dir, "repo")
    self.out = os.path.join(tmp_dir, "out")
    self.tarball = os.path.join(self.out, TARBALL)
    origin = os.path.join(tmp_dir, "origin.git")
    _git(tmp_dir, "init", "--quiet", "--bare", origin)
    _git(tmp_dir, "init", "--quiet", f"--initial-branch={BRANCH}", self.repo)
    for path, text in {**TARBALL_FILES, **OTHER_FILES}.items():
      _write(os.path.join(self.repo, path), text)
    shutil.copyfile(os.path.join(_REPO_ROOT, _BUILDER_RELPATH), os.path.join(self.repo, _BUILDER_RELPATH))
    _git(self.repo, "add", ".")
    _git(self.repo, "commit", "--quiet", "--message", "Initial commit")
    _git(self.repo, "remote", "add", "origin", origin)
    _git(self.repo, "push", "--quiet", "origin", BRANCH)
    # The builder packages the repository it is in, so the tests run its copy.
    spec = importlib.util.spec_from_file_location("build_code_tarball", os.path.join(self.repo, _BUILDER_RELPATH))
    self.builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.builder)
    # No test runs gcloud: the fake records every `gcloud storage` call and the content of each local file it copies.
    self.gcloud_calls: list[list[str]] = []
    self.uploaded: dict[str, bytes] = {}
    self.ls_returncode, self.ls_stderr = 0, ""  # The remote tarball exists.
    self.failing_destination = None
    patcher = mock.patch.object(self.builder, "run_gcloud_storage", self._fake_gcloud_storage)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _fake_gcloud_storage(self, args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Records a `gcloud storage` call and returns its result: `ls` as the test sets it, `cp` fails only for one URL."""
    self.gcloud_calls.append(list(args))
    if args[0] == "ls":
      returncode, stdout, stderr = self.ls_returncode, "" if self.ls_returncode else f"{args[1]}\n", self.ls_stderr
    else:
      source, destination = args[1:]
      if not source.startswith("gs://"):
        with open(source, "rb") as f:
          self.uploaded[destination] = f.read()
      returncode, stdout, stderr = int(destination == self.failing_destination), "", ""
    if not capture_output:  # Like subprocess.run, which captures nothing unless asked to.
      stdout = stderr = None
    return subprocess.CompletedProcess(["gcloud", "storage", *args], returncode, stdout, stderr)

  def _main(self, *args: str) -> tuple[int, str, str]:
    """Runs `build_code_tarball.py --out <out> <args>` and returns its exit code, stdout and stderr."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
      try:
        exit_code = self.builder.main(["--out", self.out, *args])
      except SystemExit as e:  # argparse rejects invalid arguments.
        exit_code = e.code
    return exit_code, stdout.getvalue(), stderr.getvalue()

  def _commit_locally(self) -> None:
    _write(os.path.join(self.repo, DECODERS), "DECODERS = 'unpushed'\n")
    _git(self.repo, "commit", "--quiet", "--all", "--message", "Unpushed change")

  def _push_with_a_stale_origin_ref(self) -> None:
    """Pushes a new commit, then moves the local origin/<branch> back, as if another clone had pushed it."""
    stale = _git(self.repo, "rev-parse", "HEAD")
    self._commit_locally()
    _git(self.repo, "push", "--quiet", "origin", BRANCH)
    _git(self.repo, "update-ref", f"refs/remotes/origin/{BRANCH}", stale)

  def _hide_working_tree_change(self, path: str, text: str) -> None:
    """Changes a committed file in a way that `git status` does not report (a skip-worktree file)."""
    _git(self.repo, "update-index", "--skip-worktree", path)
    _write(os.path.join(self.repo, path), text)

  def _tarball_names(self) -> list[str]:
    with tarfile.open(self.tarball, "r:gz") as tf:
      return tf.getnames()

  def _read_from_tarball(self, name: str) -> bytes:
    with tarfile.open(self.tarball, "r:gz") as tf:
      return tf.extractfile(name).read()

  def _code_version(self) -> dict:
    return json.loads(self._read_from_tarball("./CODE_VERSION.json"))

  def test_refuses_a_modified_tracked_file_with_a_non_zero_exit(self):
    _write(os.path.join(self.repo, DECODERS), "DECODERS = 'modified'\n")
    cli = subprocess.run(
        [sys.executable, os.path.join(self.repo, _BUILDER_RELPATH), "--out", self.out],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(cli.returncode, 0)
    self.assertIn("uncommitted changes", cli.stderr)
    self.assertIn(DECODERS, cli.stderr)
    self.assertFalse(os.path.exists(self.tarball))

  def test_builds_with_untracked_files_and_leaves_them_out(self):
    _write(os.path.join(self.repo, "src/maxtext/untracked.py"), "UNTRACKED = True\n")
    exit_code, _, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    self.assertNotIn("./src/maxtext/untracked.py", self._tarball_names())

  def test_refuses_an_unpushed_commit(self):
    self._commit_locally()
    exit_code, _, stderr = self._main()
    self.assertNotEqual(exit_code, 0)
    self.assertIn(f"not in origin/{BRANCH}", stderr)
    self.assertFalse(os.path.exists(self.tarball))

  def test_allow_unpushed_builds_and_records_the_commit_as_not_pushed(self):
    self._commit_locally()
    exit_code, _, stderr = self._main("--allow-unpushed")
    self.assertEqual(exit_code, 0, stderr)
    self.assertIs(self._code_version()["git_commit_pushed"], False)

  def test_refuses_a_branch_that_is_not_on_origin(self):
    _git(self.repo, "checkout", "--quiet", "-b", "bench/never-pushed")
    exit_code, _, stderr = self._main()
    self.assertNotEqual(exit_code, 0)
    # The fetch failed, so the error must not claim that HEAD is unpushed, and must offer --no-fetch.
    self.assertIn("`git fetch origin bench/never-pushed` failed", stderr)
    self.assertIn("--no-fetch", stderr)
    self.assertNotIn("is not in origin", stderr)

  def test_allow_unpushed_builds_after_a_failed_fetch_and_records_the_commit_as_not_pushed(self):
    _git(self.repo, "checkout", "--quiet", "-b", "bench/never-pushed")
    exit_code, _, stderr = self._main("--allow-unpushed")
    self.assertEqual(exit_code, 0, stderr)
    self.assertIn("warning: ", stderr)
    self.assertIn("`git fetch origin bench/never-pushed` failed", stderr)
    self.assertIs(self._code_version()["git_commit_pushed"], False)

  def test_refuses_a_detached_head(self):
    _git(self.repo, "checkout", "--quiet", "--detach")
    exit_code, _, stderr = self._main("--allow-unpushed")
    self.assertNotEqual(exit_code, 0)
    self.assertIn("detached", stderr)

  def test_fetches_the_branch_before_checking_that_head_is_pushed(self):
    self._push_with_a_stale_origin_ref()
    exit_code, _, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    self.assertIs(self._code_version()["git_commit_pushed"], True)

  def test_no_fetch_checks_head_against_the_local_origin_branch(self):
    self._push_with_a_stale_origin_ref()
    exit_code, _, stderr = self._main("--no-fetch")
    self.assertNotEqual(exit_code, 0)
    self.assertIn(f"not in origin/{BRANCH}", stderr)

  def test_tarball_holds_the_tracked_files_of_the_four_paths_and_code_version(self):
    exit_code, _, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    expected = ["./CODE_VERSION.json", f"./{_BUILDER_RELPATH}"] + [f"./{path}" for path in TARBALL_FILES]
    self.assertCountEqual(self._tarball_names(), expected)

  def test_tarball_holds_the_committed_content_not_the_working_tree(self):
    self._hide_working_tree_change(DECODERS, "DECODERS = 'working tree'\n")
    exit_code, _, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    self.assertEqual(self._read_from_tarball(f"./{DECODERS}"), b"DECODERS = 'committed'\n")

  def test_code_version_records_the_commit(self):
    exit_code, _, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    code_version = self._code_version()
    self.assertRegex(code_version.pop("built_at_utc"), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
    self.assertEqual(
        code_version,
        {
            "git_branch_name": BRANCH,
            "git_commit": _git(self.repo, "rev-parse", "HEAD"),
            "git_commit_short": _git(self.repo, "rev-parse", "--short", "HEAD"),
            "git_commit_time_utc": "2026-11-12T05:03:27Z",
            "git_tree_sha": _git(self.repo, "rev-parse", "HEAD^{tree}"),
            "git_working_tree_clean": True,
            "git_commit_pushed": True,
            "tarball_paths": ["src/", "checkpoint_load_benchmark/", "pyproject.toml", "README.md"],
            "builder": _BUILDER_RELPATH,
        },
    )

  def test_writes_the_sha256_and_code_version_next_to_the_tarball(self):
    exit_code, stdout, stderr = self._main()
    self.assertEqual(exit_code, 0, stderr)
    with open(self.tarball, "rb") as f:
      sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(f"{self.tarball}.sha256", "r", encoding="utf-8") as f:
      self.assertEqual(f.read(), f"{sha256}  {TARBALL}\n")
    self.assertIn(sha256, stdout)
    with open(os.path.join(self.out, "CODE_VERSION.json"), "r", encoding="utf-8") as f:
      self.assertEqual(json.load(f), self._code_version())

  def test_upload_archives_the_old_tarball_then_uploads_the_build_and_the_committed_scripts(self):
    self._hide_working_tree_change(BOOTSTRAP, "BOOTSTRAP = 'working tree'\n")
    self._hide_working_tree_change(QUEUE, "QUEUE = 'working tree'\n")
    exit_code, _, stderr = self._main("--upload", GCS_CODE_DIR)
    self.assertEqual(exit_code, 0, stderr)
    archived = self.gcloud_calls[1][-1]
    self.assertRegex(archived, rf"^{re.escape(GCS_CODE_DIR)}/archive/maxtext_lance_prs\.\d{{8}}T\d{{6}}Z\.tar\.gz$")
    self.assertEqual(
        self.gcloud_calls,
        [
            ["ls", f"{GCS_CODE_DIR}/{TARBALL}"],
            ["cp", f"{GCS_CODE_DIR}/{TARBALL}", archived],
            ["cp", self.tarball, f"{GCS_CODE_DIR}/{TARBALL}"],
            ["cp", f"{self.tarball}.sha256", f"{GCS_CODE_DIR}/{TARBALL}.sha256"],
            ["cp", os.path.join(self.out, "CODE_VERSION.json"), f"{GCS_CODE_DIR}/CODE_VERSION.json"],
            ["cp", os.path.join(self.out, "bootstrap_tpu7x.py"), f"{GCS_CODE_DIR}/bootstrap_tpu7x.py"],
            ["cp", os.path.join(self.out, "tpu7x_task_queue.py"), f"{GCS_CODE_DIR}/tpu7x_task_queue.py"],
        ],
    )
    self.assertEqual(self.uploaded[f"{GCS_CODE_DIR}/bootstrap_tpu7x.py"], b"BOOTSTRAP = 'committed'\n")
    self.assertEqual(self.uploaded[f"{GCS_CODE_DIR}/tpu7x_task_queue.py"], b"QUEUE = 'committed'\n")

  def test_upload_archives_nothing_without_a_remote_tarball(self):
    self.ls_returncode, self.ls_stderr = 1, LS_NOT_FOUND
    exit_code, _, stderr = self._main("--upload", GCS_CODE_DIR)
    self.assertEqual(exit_code, 0, stderr)
    uploaded_names = (TARBALL, f"{TARBALL}.sha256", "CODE_VERSION.json", "bootstrap_tpu7x.py", "tpu7x_task_queue.py")
    self.assertEqual([call[-1] for call in self.gcloud_calls[1:]], [f"{GCS_CODE_DIR}/{name}" for name in uploaded_names])

  def test_upload_stops_when_gcloud_cannot_tell_whether_there_is_a_remote_tarball(self):
    self.ls_returncode, self.ls_stderr = 1, LS_FORBIDDEN
    exit_code, _, stderr = self._main("--upload", GCS_CODE_DIR)
    self.assertNotEqual(exit_code, 0)
    self.assertIn("HTTPError 403", stderr)
    self.assertEqual(self.gcloud_calls, [["ls", f"{GCS_CODE_DIR}/{TARBALL}"]])

  def test_upload_strips_the_trailing_slash_of_the_code_dir(self):
    exit_code, _, stderr = self._main("--upload", f"{GCS_CODE_DIR}/")
    self.assertEqual(exit_code, 0, stderr)
    gcs_urls = [arg for call in self.gcloud_calls for arg in call[1:] if arg.startswith("gs://")]
    self.assertEqual(len(gcs_urls), 8)  # ls, then the archive copy (2 URLs) and the 5 uploads
    for url in gcs_urls:
      self.assertNotIn("//", url.removeprefix("gs://"))
    self.assertIn(["cp", self.tarball, f"{GCS_CODE_DIR}/{TARBALL}"], self.gcloud_calls)

  def test_upload_rejects_a_code_dir_that_is_not_on_gcs(self):
    exit_code, _, stderr = self._main("--upload", "test-bucket/code")
    self.assertEqual(exit_code, 2)
    self.assertIn("gs://", stderr)
    self.assertEqual(self.gcloud_calls, [])
    self.assertFalse(os.path.exists(self.tarball))

  def test_upload_cannot_be_combined_with_allow_unpushed(self):
    exit_code, _, stderr = self._main("--allow-unpushed", "--upload", GCS_CODE_DIR)
    self.assertEqual(exit_code, 2)
    self.assertIn("--allow-unpushed is for local dry runs", stderr)
    self.assertEqual(self.gcloud_calls, [])
    self.assertFalse(os.path.exists(self.tarball))

  def test_upload_stops_at_the_first_failed_copy(self):
    self.failing_destination = f"{GCS_CODE_DIR}/{TARBALL}"
    exit_code, _, stderr = self._main("--upload", GCS_CODE_DIR)
    self.assertNotEqual(exit_code, 0)
    self.assertIn(self.failing_destination, stderr)
    self.assertEqual(self.gcloud_calls[-1][-1], self.failing_destination)


if __name__ == "__main__":
  unittest.main()
