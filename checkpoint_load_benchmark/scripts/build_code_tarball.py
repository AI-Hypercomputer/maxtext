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

"""Builds the code tarball that the benchmark pods run, from the current commit of this branch.

Every experiment must run exactly one committed version of this branch, so the tarball comes from `git archive` of
HEAD, never from the working tree, and the builder refuses to build when:
  * tracked files have uncommitted changes (untracked files are ignored, and never included),
  * HEAD is detached, or
  * HEAD is not in `origin/<current branch>` after `git fetch origin <branch>` (--no-fetch skips the fetch).
    --allow-unpushed builds anyway and records `git_commit_pushed: false`, for local dry runs only.

It writes to --out:
  * maxtext_lance_prs.tar.gz (the name the pod bootstrap downloads): the files tracked at HEAD under `src/`,
    `checkpoint_load_benchmark/`, `pyproject.toml` and `README.md`, named `./<path>`, plus `./CODE_VERSION.json`: the
    commit it was built from, which the harness records in every phase result.
  * maxtext_lance_prs.tar.gz.sha256 and CODE_VERSION.json.

--upload GCS_CODE_DIR (a gs:// directory; not with --allow-unpushed) then copies GCS_CODE_DIR/maxtext_lance_prs.tar.gz
(if any) to GCS_CODE_DIR/archive/maxtext_lance_prs.<UTC timestamp>.tar.gz, uploads these 3 files, and uploads
bootstrap_tpu7x.py and tpu7x_task_queue.py as committed at HEAD. If gcloud cannot tell whether there is a tarball to
archive, nothing is uploaded.

Usage:
  python3 checkpoint_load_benchmark/scripts/build_code_tarball.py --out /tmp/code_tarball
  python3 checkpoint_load_benchmark/scripts/build_code_tarball.py --out /tmp/code_tarball \\
      --upload gs://mesa-maxtext/mesakh_tpu7x_bench/code
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from typing import Any

TARBALL_STEM = "maxtext_lance_prs"
TARBALL_NAME = f"{TARBALL_STEM}.tar.gz"
# The tracked paths that the tarball holds.
TARBALL_PATHS = ("src/", "checkpoint_load_benchmark/", "pyproject.toml", "README.md")
# The pods also download these from the code directory, so --upload uploads them next to the tarball.
HEAD_FILES = ("checkpoint_load_benchmark/bootstrap_tpu7x.py", "checkpoint_load_benchmark/scripts/tpu7x_task_queue.py")
# What `gcloud storage ls` says when no object matches its URL (its other failures are errors).
GCLOUD_LS_NOT_FOUND = "matched no objects"


class BuildError(Exception):
  """A reason not to build or not to upload the tarball."""


def run_gcloud_storage(args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
  """Runs `gcloud storage <args>`, capturing its output as text if asked to.

  Every gcloud call of this script goes through here.
  """
  return subprocess.run(["gcloud", "storage", *args], capture_output=capture_output, text=True, check=False)


def _git_output(repo: str, *args: str) -> bytes:
  """Returns the stdout of `git <args>` run in `repo`, or raises BuildError with the error of git."""
  result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
  if result.returncode:
    raise BuildError(f"`git {' '.join(args)}` failed: {result.stderr.decode(errors='replace').strip()}")
  return result.stdout


def _git(repo: str, *args: str) -> str:
  return _git_output(repo, *args).decode().rstrip()


def _utc(timestamp: float) -> str:
  return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_pushed(repo: str, branch: str, commit: str, fetch: bool, allow_unpushed: bool) -> bool:
  """Returns whether `commit` is in origin/<branch>, after fetching that branch unless `fetch` is false.

  Raises BuildError if it is not, or if the fetch fails, unless `allow_unpushed`.
  """
  if fetch:
    fetched = subprocess.run(
        ["git", "fetch", "--quiet", "origin", branch], cwd=repo, capture_output=True, text=True, check=False
    )
    if fetched.returncode:
      problem = (
          f"cannot check that HEAD ({commit}) is in origin/{branch}: "
          f"`git fetch origin {branch}` failed: {fetched.stderr.strip()}"
      )
      if not allow_unpushed:
        raise BuildError(
            f"{problem}\nIf the branch is pushed, pass --no-fetch to check the local origin/{branch}; otherwise push it"
            " first, or pass --allow-unpushed for a local dry run"
        )
      print(f"warning: {problem}", file=sys.stderr)
      return False
  is_ancestor = subprocess.run(
      ["git", "merge-base", "--is-ancestor", commit, f"refs/remotes/origin/{branch}"],
      cwd=repo,
      capture_output=True,
      check=False,
  )
  if is_ancestor.returncode and not allow_unpushed:
    raise BuildError(
        f"HEAD ({commit}) is not in origin/{branch}: push it first, or pass --allow-unpushed for a local dry run"
    )
  return is_ancestor.returncode == 0


def build(repo: str, out_dir: str, fetch: bool = True, allow_unpushed: bool = False) -> tuple[dict[str, Any], str]:
  """Builds the tarball of HEAD of `repo` in `out_dir`. Returns its CODE_VERSION.json content and its sha256."""
  changes = _git(repo, "status", "--porcelain", "--untracked-files=no")
  if changes:
    raise BuildError(f"tracked files have uncommitted changes (commit or stash them first):\n{changes}")
  branch = _git(repo, "branch", "--show-current")
  if not branch:
    raise BuildError("HEAD is detached: check out the branch to build")
  commit = _git(repo, "rev-parse", "HEAD")
  pushed = _check_pushed(repo, branch, commit, fetch, allow_unpushed)
  commit_time = int(_git(repo, "show", "--no-patch", "--format=%ct", commit))
  code_version = {
      "git_branch_name": branch,
      "git_commit": commit,
      "git_commit_short": _git(repo, "rev-parse", "--short", commit),
      "git_commit_time_utc": _utc(commit_time),
      "git_tree_sha": _git(repo, "rev-parse", f"{commit}^{{tree}}"),
      "git_working_tree_clean": True,
      "git_commit_pushed": pushed,
      "tarball_paths": list(TARBALL_PATHS),
      "built_at_utc": _utc(time.time()),
      "builder": os.path.relpath(os.path.realpath(__file__), os.path.realpath(repo)),
  }
  code_version_json = (json.dumps(code_version, indent=2) + "\n").encode()

  # The archive of HEAD (resolved once, to `commit`), never of the working tree.
  archive = _git_output(repo, "archive", "--format=tar", commit, "--", *TARBALL_PATHS)
  os.makedirs(out_dir, exist_ok=True)
  tarball = os.path.join(out_dir, TARBALL_NAME)
  with tarfile.open(fileobj=io.BytesIO(archive)) as src, tarfile.open(tarball, "w:gz") as dst:
    info = tarfile.TarInfo("./CODE_VERSION.json")
    info.size, info.mtime, info.mode = len(code_version_json), commit_time, 0o644
    dst.addfile(info, io.BytesIO(code_version_json))
    for member in src.getmembers():
      if member.isdir():
        continue  # Only the files: extracting them creates their directories.
      data = src.extractfile(member) if member.isfile() else None
      member.name = f"./{member.name}"
      member.pax_headers = {}  # git's pax headers (its commit id, a long path) would override the new name.
      dst.addfile(member, data)

  with open(tarball, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()
  with open(f"{tarball}.sha256", "w", encoding="utf-8") as f:
    f.write(f"{sha256}  {TARBALL_NAME}\n")
  with open(os.path.join(out_dir, "CODE_VERSION.json"), "wb") as f:
    f.write(code_version_json)
  return code_version, sha256


def upload(repo: str, out_dir: str, gcs_code_dir: str, commit: str) -> None:
  """Uploads the build in `out_dir` to `gcs_code_dir`, after copying the tarball it replaces to `archive/`.

  The HEAD_FILES are uploaded as committed at `commit` (HEAD), never from the working tree. Every local file is written
  before the first upload. Nothing is uploaded if gcloud cannot tell whether there is a tarball to archive, and the
  uploads stop at the first failed copy.
  """
  for path in HEAD_FILES:
    with open(os.path.join(out_dir, os.path.basename(path)), "wb") as f:
      f.write(_git_output(repo, "show", f"{commit}:{path}"))
  copies = []  # (source, destination) of each `gcloud storage cp`, in order.
  remote_tarball = f"{gcs_code_dir}/{TARBALL_NAME}"
  listed = run_gcloud_storage(["ls", remote_tarball], capture_output=True)
  if listed.returncode == 0:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    copies.append((remote_tarball, f"{gcs_code_dir}/archive/{TARBALL_STEM}.{timestamp}.tar.gz"))
  elif GCLOUD_LS_NOT_FOUND in listed.stderr:
    print(f"No {remote_tarball} to archive")
  else:  # Uploading anyway could replace the tarball without archiving it.
    raise BuildError(f"cannot tell whether there is a {remote_tarball} to archive: {listed.stderr.strip()}")
  for name in (TARBALL_NAME, f"{TARBALL_NAME}.sha256", "CODE_VERSION.json", *map(os.path.basename, HEAD_FILES)):
    copies.append((os.path.join(out_dir, name), f"{gcs_code_dir}/{name}"))
  for source, destination in copies:
    if run_gcloud_storage(["cp", source, destination]).returncode != 0:
      raise BuildError(f"`gcloud storage cp {source} {destination}` failed")
  print(f"Uploaded the build of {commit} to {gcs_code_dir}")


def _gcs_code_dir(value: str) -> str:
  """Returns the --upload directory without trailing slashes, which would put `//` in every object name."""
  code_dir = value.rstrip("/")
  if not code_dir.startswith("gs://"):
    raise argparse.ArgumentTypeError(f"not a gs:// directory: {value!r}")
  return code_dir


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out", required=True, help="Directory to write the tarball, its .sha256 and CODE_VERSION.json to")
  p.add_argument("--no-fetch", action="store_true", help="Check HEAD against the local origin/<branch>, unfetched")
  p.add_argument(
      "--allow-unpushed",
      action="store_true",
      help="Build even if HEAD is not in origin/<branch>, recording git_commit_pushed: false (local dry runs only)",
  )
  p.add_argument(
      "--upload", metavar="GCS_CODE_DIR", type=_gcs_code_dir, help="Upload the build to this code directory (see above)"
  )
  args = p.parse_args(argv)
  if args.upload and args.allow_unpushed:
    p.error("--allow-unpushed is for local dry runs; it cannot be combined with --upload")
  try:
    # The repository that this script is in, whatever the working directory.
    repo = _git(os.path.dirname(os.path.realpath(__file__)), "rev-parse", "--show-toplevel")
    code_version, sha256 = build(repo, args.out, fetch=not args.no_fetch, allow_unpushed=args.allow_unpushed)
    not_pushed = "" if code_version["git_commit_pushed"] else " (not pushed: for local dry runs only)"
    print(f"Built {os.path.join(args.out, TARBALL_NAME)} from {code_version['git_commit']}{not_pushed}")
    print(f"sha256: {sha256}")
    if args.upload:
      upload(repo, args.out, args.upload, code_version["git_commit"])
  except BuildError as e:
    print(f"error: {e}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
