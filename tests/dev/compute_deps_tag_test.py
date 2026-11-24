# Copyright 2026 Google LLC
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

"""Unit tests for ``src/dependencies/scripts/compute_deps_tag.sh``."""

import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "src/dependencies/scripts/compute_deps_tag.sh"

_DEPS_IMAGE_INPUTS = {
    "src/dependencies/requirements/generated_requirements/tpu-requirements.txt": "jax>=0.1\nlibtpu>=0.2\n",
    "src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt": "jax>=0.1\nvllm>=0.3\n",
    "src/dependencies/requirements/generated_requirements/cuda12-requirements.txt": (
        "jax>=0.1\nnvidia-cublas-cu12>=0.4\n"
    ),
    "src/dependencies/extra_deps/pre_train_github_deps.txt": "mlperf-logging @ https://example.invalid/pre.zip\n",
    "src/dependencies/extra_deps/post_train_github_deps.txt": "vllm @ https://example.invalid/post.zip\n",
    "src/dependencies/extra_deps/tf_requirements.txt": "tensorflow>=2.0\n",
    "src/dependencies/scripts/install_pre_train_extra_deps.py": "",
    "src/dependencies/scripts/install_post_train_extra_deps.py": "",
    "src/dependencies/scripts/setup.sh": "#!/bin/bash\n",
    "src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile": "ARG BASEIMAGE=python:3.12-slim-trixie\n",
    "src/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile": "ARG BASEIMAGE=ubuntu:24.04\n",
}

_DEPS_IMAGE_SPECS = {
    "tpu_pre_train": ("tpu", "pre-training"),
    "tpu_post_train": ("tpu", "post-training"),
    "gpu_pre_train": ("gpu", "pre-training"),
}


def _compute_deps_tags(tmp_path, *, tf=False, change=None):
  """Run compute_deps_tag.sh across all three suites against a synthetic checkout."""
  files = {**_DEPS_IMAGE_INPUTS, **(change or {})}
  for rel, content in files.items():
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
  tf_flag = "true" if tf else "false"
  tags = {}
  for key, (device, workflow) in _DEPS_IMAGE_SPECS.items():
    result = subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            f"--device={device}",
            f"--workflow={workflow}",
            f"--tf={tf_flag}",
            f"--repo-root={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tags[key] = result.stdout.strip()
  return tags


def test_setup_change_rebuilds_every_image(tmp_path):
  """setup.sh feeds every image, so changing it must move all three tags."""
  unchanged = _compute_deps_tags(tmp_path / "unchanged")
  assert all(tag.startswith("deps-") for tag in unchanged.values()), unchanged
  assert len(set(unchanged.values())) == len(unchanged), unchanged
  changed = _compute_deps_tags(
      tmp_path / "changed",
      change={"src/dependencies/scripts/setup.sh": "#!/bin/bash\ntrue\n"},
  )
  for key in _DEPS_IMAGE_SPECS:
    assert changed[key] != unchanged[key], key


@pytest.mark.parametrize(
    "changed_file, moves",
    [
        pytest.param(
            "src/dependencies/requirements/generated_requirements/tpu-requirements.txt",
            {"tpu_pre_train"},
            id="tpu-requirements",
        ),
        pytest.param(
            "src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt",
            {"tpu_post_train"},
            id="tpu-post-train-requirements",
        ),
        pytest.param(
            "src/dependencies/requirements/generated_requirements/cuda12-requirements.txt",
            {"gpu_pre_train"},
            id="cuda12-requirements",
        ),
        pytest.param(
            "src/dependencies/extra_deps/pre_train_github_deps.txt",
            {"tpu_pre_train", "gpu_pre_train"},
            id="pre-train-github-deps",
        ),
        pytest.param(
            "src/dependencies/extra_deps/tf_requirements.txt",
            set(),
            id="tf-requirements-ignored-when-tf-false",
        ),
        pytest.param(
            "src/dependencies/scripts/install_post_train_extra_deps.py",
            {"tpu_post_train"},
            id="post-train-installer",
        ),
        pytest.param(
            "src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile",
            {"tpu_pre_train", "tpu_post_train"},
            id="tpu-dockerfile",
        ),
        pytest.param(
            "src/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile",
            {"gpu_pre_train"},
            id="gpu-dockerfile",
        ),
    ],
)
def test_tags_move_only_for_the_images_that_use_the_file(tmp_path, changed_file, moves):
  """Each image hashes only its own inputs, so a bump must rename those images and leave the rest alone."""
  unchanged = _compute_deps_tags(tmp_path / "unchanged")
  changed = _compute_deps_tags(tmp_path / "changed", change={changed_file: "# changed\n"})
  for key in _DEPS_IMAGE_SPECS:
    if key in moves:
      assert changed[key] != unchanged[key], f"{key} should have moved for {changed_file}"
    else:
      assert changed[key] == unchanged[key], f"{key} should not have moved for {changed_file}"


def test_tf_true_hashes_tf_requirements(tmp_path):
  """When --tf=true (release pipelines), tf_requirements.txt moves pre-training tags only."""
  unchanged = _compute_deps_tags(tmp_path / "unchanged", tf=True)
  changed = _compute_deps_tags(
      tmp_path / "changed",
      tf=True,
      change={"src/dependencies/extra_deps/tf_requirements.txt": "# changed\n"},
  )
  assert changed["tpu_pre_train"] != unchanged["tpu_pre_train"]
  assert changed["gpu_pre_train"] != unchanged["gpu_pre_train"]
  assert changed["tpu_post_train"] == unchanged["tpu_post_train"]


def test_print_base_image_mirrors_and_validates(tmp_path):
  """--print-base-image maps Docker Hub tags to Airlock and rejects untrusted refs."""
  for device, expected in [
      ("tpu", "us-docker.pkg.dev/artifact-foundry-prod/docker-3p-trusted/python:3.12-slim-trixie"),
      ("gpu", "us-docker.pkg.dev/artifact-foundry-prod/docker-3p-trusted/ubuntu:24.04"),
  ]:
    res = subprocess.run(
        ["bash", str(_SCRIPT), f"--device={device}", "--print-base-image", f"--repo-root={REPO_ROOT}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == expected

  bad_df = tmp_path / "bad.Dockerfile"
  for line in (
      "ARG BASEIMAGE=evil.io/foo:latest\n",
      "ARG BASEIMAGE=us-docker.pkg.dev/other-project/repo/img:1\n",
      "ARG BASEIMAGE=python:3.12/../../foo\n",
      "ARG BASEIMAGE=\n",
  ):
    bad_df.write_text(line, encoding="utf-8")
    res = subprocess.run(
        ["bash", str(_SCRIPT), "--print-base-image", f"--dockerfile={bad_df}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0, f"expected rejection for {line!r}, got {res.stdout!r}"


def test_dependency_image_extras_match_the_generated_requirement_files():
  """Verify setup.sh and pyproject.toml agree on requirement files for each extra.

  maxtext_{tpu,gpu}_dependencies.Dockerfile installs `-r <extra>-requirements.txt` via setup.sh, and the test job then
  installs the wheel with --no-deps. That is only equivalent while maxtext declares no base dependencies and each extra
  is exactly one generated file; add a base dependency and CI would silently run without it.
  """
  pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
  assert "\ndependencies = []\n" in pyproject, "maxtext gained base dependencies; --no-deps would now skip them"
  generated = "src/dependencies/requirements/generated_requirements"
  for extra, filename in [
      ("tpu", "tpu-requirements.txt"),
      ("tpu-post-train", "tpu-post-train-requirements.txt"),
      ("cuda12", "cuda12-requirements.txt"),
  ]:
    assert f'{extra} = ["{generated}/{filename}"]' in pyproject, f"extra '{extra}' no longer maps to {filename}"
