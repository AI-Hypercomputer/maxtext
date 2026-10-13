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

"""Unit tests for the fast-rebuild dependency drift guard script."""

import os
import shutil
import subprocess
import tempfile
import unittest

import pytest

_SCRIPT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".github",
        "scripts",
        "fast_rebuild_drift_guard.sh",
    )
)


def _run_drift_guard(repo_dir, source_sha, allow_drift="false", current_sha="", workflow="", github_output=""):
  """Run the drift guard script in a given git repo.

  Args:
    repo_dir: Path to the git repository.
    source_sha: The source commit SHA to compare against.
    allow_drift: "true" to allow drift (warn instead of fail).
    current_sha: The current commit SHA; empty means the script defaults to HEAD.
    workflow: "pre-training" or "post-training"; empty behaves like pre-training.
    github_output: Path of a file the script appends `overlay_variant=...` to; empty disables it.

  Returns:
    subprocess.CompletedProcess with returncode, stdout, stderr.
  """
  result = subprocess.run(
      ["bash", _SCRIPT_PATH],
      cwd=repo_dir,
      env={
          **os.environ,
          "SOURCE_SHA": source_sha,
          "CURRENT_SHA": current_sha,
          "ALLOW_DRIFT": allow_drift,
          "WORKFLOW": workflow,
          # Always override so a CI job's real GITHUB_OUTPUT file is never written to.
          "GITHUB_OUTPUT": github_output,
          # Not setting GITHUB_TOKEN/SOURCE_RUN_ID — we provide SOURCE_SHA directly
      },
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
  )
  return result


def _create_test_repo():
  """Create a temporary git repo with a controlled commit history.

  Returns:
    (repo_dir, base_sha): Path to the repo and the SHA of the initial commit.
  """
  repo_dir = tempfile.mkdtemp(prefix="drift_guard_test_")

  subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
  subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True, capture_output=True)
  subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True, capture_output=True)

  # Create the directory structure that the drift guard checks
  os.makedirs(os.path.join(repo_dir, "src", "dependencies", "requirements"), exist_ok=True)
  os.makedirs(os.path.join(repo_dir, "src", "maxtext"), exist_ok=True)
  os.makedirs(os.path.join(repo_dir, "tests"), exist_ok=True)

  # Initial commit: base state
  with open(
      os.path.join(repo_dir, "src", "dependencies", "requirements", "tpu-requirements.txt"), "w", encoding="utf-8"
  ) as f:
    f.write("jax>=0.11.0\n")
  with open(os.path.join(repo_dir, "pyproject.toml"), "w", encoding="utf-8") as f:
    f.write('[project]\nname = "maxtext"\n')
  with open(os.path.join(repo_dir, ".dockerignore"), "w", encoding="utf-8") as f:
    f.write(".git\n")
  with open(os.path.join(repo_dir, "src", "maxtext", "train.py"), "w", encoding="utf-8") as f:
    f.write("# training code\n")
  with open(os.path.join(repo_dir, "tests", "test_train.py"), "w", encoding="utf-8") as f:
    f.write("# test code\n")

  subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
  subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

  result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
  base_sha = result.stdout.strip()

  return repo_dir, base_sha


def _read_github_output(path):
  """Parse a GITHUB_OUTPUT-style file (key=value per line) into a dict."""
  with open(path, encoding="utf-8") as f:
    return dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)


def _commit_file(repo_dir, rel_path, content, message):
  """Write a file and commit it; returns the new commit SHA."""
  with open(os.path.join(repo_dir, rel_path), "w", encoding="utf-8") as f:
    f.write(content)
  subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
  subprocess.run(["git", "commit", "-m", message], cwd=repo_dir, check=True, capture_output=True)
  result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
  return result.stdout.strip()


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not os.path.isfile(_SCRIPT_PATH),
    reason="drift guard script is not shipped in the MaxText image; run these tests from a source checkout",
)
class FastRebuildDriftGuardTest(unittest.TestCase):
  """Tests for .github/scripts/fast_rebuild_drift_guard.sh.

  The script lives under .github/, which is not copied into the Docker image, so the
  class is skipped when the tests run from inside the image (maxtext_installed=true).
  """

  def test_no_drift_code_only_changes(self):
    """Fast-rebuild should succeed when only code files changed."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    # Change only code files (not deps)
    with open(os.path.join(repo_dir, "src", "maxtext", "train.py"), "w", encoding="utf-8") as f:
      f.write("# updated training code\n")
    with open(os.path.join(repo_dir, "tests", "test_train.py"), "w", encoding="utf-8") as f:
      f.write("# updated test code\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "code change"], cwd=repo_dir, check=True, capture_output=True)

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("No dependency changes detected", result.stdout)

  def test_drift_detected_requirements_changed(self):
    """Fast-rebuild should fail when requirements files changed."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    # Change a requirements file
    with open(
        os.path.join(repo_dir, "src", "dependencies", "requirements", "tpu-requirements.txt"), "w", encoding="utf-8"
    ) as f:
      f.write("jax>=0.12.0\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "bump jax"], cwd=repo_dir, check=True, capture_output=True)

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 1, f"Expected failure but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("Dependency files changed", result.stdout)
    self.assertIn("tpu-requirements.txt", result.stdout)

  def test_drift_detected_pyproject_changed(self):
    """Fast-rebuild should fail when pyproject.toml changed."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    with open(os.path.join(repo_dir, "pyproject.toml"), "w", encoding="utf-8") as f:
      f.write('[project]\nname = "maxtext"\nversion = "0.3.0"\n')
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "bump version"], cwd=repo_dir, check=True, capture_output=True)

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 1, f"Expected failure but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("pyproject.toml", result.stdout)

  def test_drift_detected_dockerignore_changed(self):
    """Fast-rebuild should fail when .dockerignore changed."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    with open(os.path.join(repo_dir, ".dockerignore"), "w", encoding="utf-8") as f:
      f.write(".git\n.venv\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "update dockerignore"], cwd=repo_dir, check=True, capture_output=True)

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 1, f"Expected failure but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn(".dockerignore", result.stdout)

  def test_drift_detected_new_dockerfile_added(self):
    """Fast-rebuild should fail when a file is added under src/dependencies/.

    Mirrors the real case of a branch adding a new Dockerfile under
    src/dependencies/dockerfiles/: an added file is drift, not only a modified one.
    """
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    os.makedirs(os.path.join(repo_dir, "src", "dependencies", "dockerfiles"), exist_ok=True)
    _commit_file(
        repo_dir,
        os.path.join("src", "dependencies", "dockerfiles", "new_overlay.Dockerfile"),
        "FROM scratch\n",
        "add overlay dockerfile",
    )

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 1, f"Expected failure but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("Dependency files changed", result.stdout)
    self.assertIn("src/dependencies/dockerfiles/new_overlay.Dockerfile", result.stdout)

  def test_overlay_variant_fast_when_code_modified(self):
    """Editing or adding code files keeps the COPY-only fast overlay."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)
    _commit_file(repo_dir, os.path.join("src", "maxtext", "train.py"), "# new training code\n", "edit code")
    _commit_file(repo_dir, os.path.join("tests", "test_new.py"), "# new test\n", "add test")
    output_path = os.path.join(repo_dir, "github_output.txt")

    result = _run_drift_guard(repo_dir, base_sha, github_output=output_path)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("Overlay variant: fast", result.stdout)
    self.assertEqual(_read_github_output(output_path)["overlay_variant"], "fast")

  def test_overlay_variant_clean_when_code_file_deleted(self):
    """A deleted code file must select the clean overlay, which removes old code first."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)
    subprocess.run(
        ["git", "rm", "-q", os.path.join("tests", "test_train.py")], cwd=repo_dir, check=True, capture_output=True
    )
    subprocess.run(["git", "commit", "-m", "remove test"], cwd=repo_dir, check=True, capture_output=True)
    output_path = os.path.join(repo_dir, "github_output.txt")

    result = _run_drift_guard(repo_dir, base_sha, github_output=output_path)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("Overlay variant: clean", result.stdout)
    self.assertIn("tests/test_train.py (removed)", result.stdout)
    self.assertEqual(_read_github_output(output_path)["overlay_variant"], "clean")

  def test_overlay_variant_clean_for_post_training_vllm_change(self):
    """A vLLM adapter change needs the clean overlay for post-training only."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)
    os.makedirs(os.path.join(repo_dir, "src", "maxtext", "integration", "vllm"), exist_ok=True)
    _commit_file(
        repo_dir, os.path.join("src", "maxtext", "integration", "vllm", "adapter.py"), "# adapter\n", "change adapter"
    )
    pre_output = os.path.join(repo_dir, "pre.txt")
    post_output = os.path.join(repo_dir, "post.txt")

    pre = _run_drift_guard(repo_dir, base_sha, workflow="pre-training", github_output=pre_output)
    post = _run_drift_guard(repo_dir, base_sha, workflow="post-training", github_output=post_output)
    self.assertEqual(pre.returncode, 0, pre.stdout + pre.stderr)
    self.assertEqual(post.returncode, 0, post.stdout + post.stderr)
    self.assertEqual(_read_github_output(pre_output)["overlay_variant"], "fast")
    self.assertEqual(_read_github_output(post_output)["overlay_variant"], "clean")
    self.assertIn("vllm", post.stdout)

  def test_overlay_variant_reported_when_drift_allowed(self):
    """ALLOW_DRIFT=true still reports an overlay variant so the build can proceed."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)
    _commit_file(repo_dir, "pyproject.toml", '[project]\nname = "maxtext"\nversion = "2"\n', "bump version")
    output_path = os.path.join(repo_dir, "github_output.txt")

    result = _run_drift_guard(repo_dir, base_sha, allow_drift="true", github_output=output_path)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("::warning::", result.stdout)
    self.assertEqual(_read_github_output(output_path)["overlay_variant"], "fast")

  def test_allow_drift_overrides_failure(self):
    """Fast-rebuild should warn but succeed when allow_drift=true."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    with open(
        os.path.join(repo_dir, "src", "dependencies", "requirements", "tpu-requirements.txt"), "w", encoding="utf-8"
    ) as f:
      f.write("jax>=0.12.0\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "bump jax"], cwd=repo_dir, check=True, capture_output=True)

    result = _run_drift_guard(repo_dir, base_sha, allow_drift="true")
    self.assertEqual(
        result.returncode, 0, f"Expected success with allow_drift but got:\n{result.stdout}\n{result.stderr}"
    )
    self.assertIn("Proceeding with fast-rebuild despite dependency drift", result.stdout)

  def test_missing_source_sha_no_api_vars(self):
    """Drift guard should fail with exit code 2 when SOURCE_SHA is empty and API vars are missing."""
    repo_dir, _ = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    result = subprocess.run(
        ["bash", _SCRIPT_PATH],
        cwd=repo_dir,
        env={**os.environ, "SOURCE_SHA": "", "SOURCE_RUN_ID": "", "GITHUB_TOKEN": "", "GITHUB_REPOSITORY": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertEqual(
        result.returncode, 2, f"Expected exit code 2 but got {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )

  def test_no_changes_same_commit(self):
    """Drift guard should succeed when source and HEAD are the same commit."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")

  def test_current_sha_overrides_head(self):
    """CURRENT_SHA selects the commit to compare instead of HEAD."""
    repo_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, repo_dir, ignore_errors=True)

    code_sha = _commit_file(repo_dir, os.path.join("src", "maxtext", "train.py"), "# updated\n", "code change")
    _commit_file(
        repo_dir, os.path.join("src", "dependencies", "requirements", "tpu-requirements.txt"), "jax>=0.12.0\n", "bump jax"
    )

    # HEAD (deps change) vs base: drift
    result = _run_drift_guard(repo_dir, base_sha)
    self.assertEqual(result.returncode, 1, f"Expected drift at HEAD but got:\n{result.stdout}\n{result.stderr}")

    # code-only commit vs base: no drift, even though HEAD has moved past it
    result = _run_drift_guard(repo_dir, base_sha, current_sha=code_sha)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("No dependency changes detected", result.stdout)

  def test_fetches_missing_commits_from_origin(self):
    """In a shallow clone, commits missing locally are fetched from origin by SHA."""
    origin_dir, base_sha = _create_test_repo()
    self.addCleanup(shutil.rmtree, origin_dir, ignore_errors=True)
    code_sha = _commit_file(origin_dir, os.path.join("src", "maxtext", "train.py"), "# updated\n", "code change")
    _commit_file(origin_dir, os.path.join("src", "maxtext", "train.py"), "# updated again\n", "another code change")
    # GitHub serves any reachable commit by SHA; local file remotes need this opt-in.
    subprocess.run(
        ["git", "config", "uploadpack.allowAnySHA1InWant", "true"], cwd=origin_dir, check=True, capture_output=True
    )

    clone_dir = tempfile.mkdtemp(prefix="drift_guard_clone_")
    self.addCleanup(shutil.rmtree, clone_dir, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--depth=1", f"file://{origin_dir}", clone_dir], check=True, capture_output=True
    )
    for sha in (base_sha, code_sha):
      missing = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=clone_dir, check=False)
      self.assertNotEqual(missing.returncode, 0, f"{sha} should be absent from the shallow clone")

    result = _run_drift_guard(clone_dir, base_sha, current_sha=code_sha)
    self.assertEqual(result.returncode, 0, f"Expected success but got:\n{result.stdout}\n{result.stderr}")
    self.assertIn("not in local history, fetching", result.stdout)
    self.assertIn("No dependency changes detected", result.stdout)


if __name__ == "__main__":
  unittest.main()
