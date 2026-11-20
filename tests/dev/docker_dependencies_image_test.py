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

"""Tests for multi-stage dependencies-only Docker builds and daily CI workflows."""

import os
import pathlib
import re
import stat
import subprocess
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILES_DIR = REPO_ROOT / "src" / "dependencies" / "dockerfiles"
BUILD_SCRIPT = REPO_ROOT / "src" / "dependencies" / "scripts" / "docker_build_dependency_image.sh"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _split_dockerfile_stages(content: str) -> tuple[str, str]:
  """Splits a multi-stage Dockerfile into `dependencies` and `full` stage bodies."""
  dep_marker = "FROM $BASEIMAGE AS dependencies"
  full_marker = "FROM dependencies AS full"
  dep_idx = content.find(dep_marker)
  full_idx = content.find(full_marker)
  if dep_idx == -1:
    raise AssertionError(f"Missing stage header: {dep_marker}")
  if full_idx == -1:
    raise AssertionError(f"Missing stage header: {full_marker}")
  if dep_idx >= full_idx:
    raise AssertionError(f"Stage {dep_marker!r} must precede {full_marker!r}")
  return content[dep_idx:full_idx], content[full_idx:]


def _extract_run_blocks(workflow_text: str) -> list[str]:
  """Extracts shell bodies under `run: |` blocks in a GitHub Actions workflow."""
  blocks: list[str] = []
  lines = workflow_text.splitlines()
  idx = 0
  while idx < len(lines):
    line = lines[idx]
    match = re.match(r"^(\s*)run:\s*\|\s*$", line)
    if match:
      run_indent = len(match.group(1))
      idx += 1
      block_lines: list[str] = []
      while idx < len(lines):
        cur = lines[idx]
        if not cur.strip():
          block_lines.append(cur)
          idx += 1
          continue
        cur_indent = len(cur) - len(cur.lstrip(" "))
        if cur_indent <= run_indent:
          break
        block_lines.append(cur)
        idx += 1
      blocks.append("\n".join(block_lines))
      continue
    idx += 1
  return blocks


class DockerDependenciesImageTest(unittest.TestCase):
  """Verifies dependencies-only Docker stage separation, build script, and workflows."""

  def test_dockerfiles_separate_dependencies_and_full_stages(self):
    for filename in ("maxtext_tpu_dependencies.Dockerfile", "maxtext_gpu_dependencies.Dockerfile"):
      with self.subTest(dockerfile=filename):
        content = (DOCKERFILES_DIR / filename).read_text(encoding="utf-8")
        deps_stage, full_stage = _split_dockerfile_stages(content)

        # dependencies stage installs deps, removes temporary /deps/src/maxtext, and sets PYTHONPATH.
        self.assertIn("rm -rf /deps/src/maxtext", deps_stage)
        self.assertIn('ENV PYTHONPATH="/deps/src${PYTHONPATH:+:${PYTHONPATH}}"', deps_stage)
        self.assertNotIn("INCLUDE_TEST_ASSETS", deps_stage)

        allowed_vllm_copy = "COPY ${PACKAGE_DIR}/maxtext/integration/vllm/ src/maxtext/integration/vllm/"
        copy_lines = [
            line.strip()
            for line in deps_stage.splitlines()
            if re.match(r"^\s*(COPY|ADD)\b", line) and not line.strip().startswith("#")
        ]
        self.assertGreater(len(copy_lines), 0)
        for copy_line in copy_lines:
          if copy_line == allowed_vllm_copy:
            self.assertLess(deps_stage.index(copy_line), deps_stage.index("rm -rf /deps/src/maxtext"))
            continue
          for forbidden in ("maxtext/", "tests", "benchmarks", "pytest.ini"):
            self.assertNotIn(forbidden, copy_line, msg=f"Code/test asset copied into dependencies stage: {copy_line}")

        # full stage layers MaxText source, tests, benchmarks, and optional test assets.
        self.assertIn("ARG PACKAGE_DIR", full_stage)
        self.assertIn("ARG INCLUDE_TEST_ASSETS", full_stage)
        for expected in ("maxtext/", "tests*/", "pytest.ini*", "benchmarks*/", "maxtext-test-assets"):
          self.assertIn(expected, full_stage)

  def _run_build_script_with_mock_docker(
      self,
      env_overrides: dict[str, str],
      cli_args: list[str] | None = None,
  ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Executes the dependency build script against a mock docker binary."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      args_file = tmp_path / "docker_build_args.txt"
      mock_docker = tmp_path / "docker"
      mock_docker.write_text(
          "#!/usr/bin/env bash\n"
          'if [[ "${1:-}" == "build" ]]; then\n'
          '  printf \'%s\\n\' "$@" > "${MOCK_DOCKER_ARGS_FILE}"\n'
          "fi\n",
          encoding="utf-8",
      )
      mock_docker.chmod(mock_docker.stat().st_mode | stat.S_IXUSR)

      env = os.environ.copy()
      env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
      env["MOCK_DOCKER_ARGS_FILE"] = str(args_file)
      env["DEVICE"] = "tpu"
      env["WORKFLOW"] = "pre-training"
      env["MODE"] = "stable"
      env.pop("INCLUDE_CODE", None)
      env.pop("INCLUDE_TEST_ASSETS", None)
      env.update(env_overrides)

      result = subprocess.run(
          ["bash", str(BUILD_SCRIPT), *(cli_args or [])],
          cwd=REPO_ROOT,
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      recorded_args = args_file.read_text(encoding="utf-8").splitlines() if args_file.exists() else []
      return result, recorded_args

  def test_build_script_include_code_flag_behavior(self):
    res_default, args_default = self._run_build_script_with_mock_docker({})
    self.assertEqual(res_default.returncode, 0, msg=res_default.stderr + res_default.stdout)
    self.assertNotIn("--target", args_default)
    self.assertIn(str(pathlib.Path("src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile")), args_default)
    self.assertIn("It has the dependencies and MaxText code installed.", res_default.stdout)

    res_true, args_true = self._run_build_script_with_mock_docker({"INCLUDE_CODE": "true"})
    self.assertEqual(res_true.returncode, 0, msg=res_true.stderr + res_true.stdout)
    self.assertNotIn("--target", args_true)
    self.assertIn("It has the dependencies and MaxText code installed.", res_true.stdout)

    res_false, args_false = self._run_build_script_with_mock_docker({"INCLUDE_CODE": "false"})
    self.assertEqual(res_false.returncode, 0, msg=res_false.stderr + res_false.stdout)
    self.assertIn("--target", args_false)
    target_idx = args_false.index("--target")
    self.assertEqual(args_false[target_idx + 1], "dependencies")
    self.assertIn("It only has the dependencies installed.", res_false.stdout)

    res_cli_false, args_cli_false = self._run_build_script_with_mock_docker({}, cli_args=["INCLUDE_CODE=false"])
    self.assertEqual(res_cli_false.returncode, 0, msg=res_cli_false.stderr + res_cli_false.stdout)
    self.assertIn("--target", args_cli_false)
    self.assertEqual(args_cli_false[args_cli_false.index("--target") + 1], "dependencies")

    res_gpu_false, args_gpu_false = self._run_build_script_with_mock_docker({"DEVICE": "gpu", "INCLUDE_CODE": "false"})
    self.assertEqual(res_gpu_false.returncode, 0, msg=res_gpu_false.stderr + res_gpu_false.stdout)
    self.assertIn("--target", args_gpu_false)
    self.assertEqual(args_gpu_false[args_gpu_false.index("--target") + 1], "dependencies")
    gpu_dockerfile = str(pathlib.Path("src/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile"))
    self.assertIn(gpu_dockerfile, args_gpu_false)

    res_conflict, args_conflict = self._run_build_script_with_mock_docker(
        {
            "INCLUDE_CODE": "false",
            "INCLUDE_TEST_ASSETS": "true",
        }
    )
    self.assertNotEqual(res_conflict.returncode, 0)
    self.assertEqual(args_conflict, [])
    self.assertIn("INCLUDE_TEST_ASSETS=true cannot be used when INCLUDE_CODE=false", res_conflict.stderr)

    res_invalid, args_invalid = self._run_build_script_with_mock_docker({"INCLUDE_CODE": "maybe"})
    self.assertNotEqual(res_invalid.returncode, 0)
    self.assertEqual(args_invalid, [])
    self.assertIn("Invalid INCLUDE_CODE", res_invalid.stderr)

  def test_daily_workflows_publish_and_promote_dependencies_images(self):
    tpu_text = (WORKFLOWS_DIR / "tpu_docker_images_pipeline.yml").read_text(encoding="utf-8")
    gpu_text = (WORKFLOWS_DIR / "gpu_docker_images_pipeline.yml").read_text(encoding="utf-8")
    build_text = (WORKFLOWS_DIR / "build_and_push_docker_image.yml").read_text(encoding="utf-8")
    promote_text = (WORKFLOWS_DIR / "promote_docker_image.yml").read_text(encoding="utf-8")

    expected_tpu_images = [
        "maxtext_jax_stable",
        "maxtext_jax_nightly",
        "maxtext_post_training_stable",
        "maxtext_post_training_nightly",
    ]
    for img in expected_tpu_images:
      self.assertRegex(
          tpu_text,
          rf"image_name:\s*{img}\n\s*dependencies_image_name:\s*{img}_dependencies\b",
      )
    self.assertIn("dependencies_image_name: ${{ matrix.dependencies_image_name }}", tpu_text)

    expected_gpu_images = ["maxtext_gpu_jax_stable", "maxtext_gpu_jax_nightly"]
    for img in expected_gpu_images:
      matches = re.findall(
          rf"image_name:\s*{img}\n\s*dependencies_image_name:\s*{img}_dependencies\b",
          gpu_text,
      )
      self.assertEqual(len(matches), 2, msg=f"Expected amd64+arm64 dependencies entries for {img}")
    self.assertIn("dependencies_image_name: ${{ matrix.dependencies_image_name }}", gpu_text)

    self.assertIn("dependencies_image_name:", build_text)
    self.assertIn("id: build_push_deps", build_text)
    self.assertIn("target: dependencies", build_text)
    self.assertIn("deps_image_digest: ${{ steps.build_push_deps.outputs.digest }}", build_text)
    self.assertIn('generate_and_export_sbom "${INPUTS_DEPS_IMAGE_NAME}" "${DEPS_IMAGE_DIGEST}"', build_text)
    self.assertIn(
        '"${PROJECT_NAME}" "${STAGING_REPO}" "${PROD_REPO}" "${INPUTS_DEPS_IMAGE_NAME}" "${DEPS_IMAGE_DIGEST}"'
        ' "${SBOM_BUCKET}"',
        build_text,
    )
    self.assertIn('tag_image_set "${INPUTS_DEPS_IMAGE_NAME}" "${DEPS_IMAGE_DIGEST}"', build_text)

    self.assertIn('for deps_image in "${IMAGE}_dependencies" "${nightly_image}_dependencies"; do', promote_text)

    for wf_name, wf_text in (
        ("build_and_push_docker_image.yml", build_text),
        ("tpu_docker_images_pipeline.yml", tpu_text),
        ("gpu_docker_images_pipeline.yml", gpu_text),
        ("promote_docker_image.yml", promote_text),
    ):
      for block in _extract_run_blocks(wf_text):
        self.assertNotIn("${{", block, msg=f"Inline template expression found inside run block in {wf_name}")


if __name__ == "__main__":
  unittest.main()
