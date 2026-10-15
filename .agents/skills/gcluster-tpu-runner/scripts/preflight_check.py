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

"""Pre-flight validator for MaxText TPU workloads (AST syntax, YAML, .dockerignore, Docker imports)."""

import argparse
import ast
import os
import subprocess
import sys
import yaml

REQUIRED_DOCKERIGNORE_PATTERNS = [".git", "*venv*", "*.ckpt", "*.safetensors", "/tmp"]
OVERLAY_PREFIX = "cp -a /src/. /deps/src/ 2>/dev/null || cp -a /src/. /deps/ && cd /deps"


def validate_dockerignore(maxtext_root: str, fix: bool = False) -> tuple[bool, list[str]]:
  """Ensures .dockerignore excludes heavy directories (.git, *venv*, checkpoints)."""
  dockerignore_path = os.path.join(maxtext_root, ".dockerignore")
  existing_lines: list[str] = []
  if os.path.exists(dockerignore_path):
    with open(dockerignore_path, "r", encoding="utf-8") as f:
      existing_lines = [line.strip() for line in f.readlines()]

  missing = [p for p in REQUIRED_DOCKERIGNORE_PATTERNS if p not in existing_lines]
  if missing and fix:
    with open(dockerignore_path, "a", encoding="utf-8") as f:
      if existing_lines and existing_lines[-1] != "":
        f.write("\n")
      for pattern in missing:
        f.write(f"{pattern}\n")
    return True, []
  return len(missing) == 0, missing


def wrap_deps_overlay_command(user_command: str, use_deps_overlay: bool = True) -> str:
  """Prepends the /src -> /deps copy command when using a layered code-overlay image."""
  if not use_deps_overlay or "/deps" in user_command:
    return user_command
  return f"{OVERLAY_PREFIX} && {user_command}"


def check_python_syntax(maxtext_root: str) -> tuple[bool, list[str]]:
  """Parses changed or staged Python files via ast.parse without writing .pyc files."""
  src_dir = os.path.join(maxtext_root, "src")
  if not os.path.isdir(src_dir):
    return False, [f"Missing src/ directory under {maxtext_root}"]

  errors: list[str] = []
  git_cmd = ["git", "-C", maxtext_root, "diff", "--name-only", "HEAD"]
  res = subprocess.run(git_cmd, capture_output=True, text=True, check=False)
  changed_files = [
      os.path.join(maxtext_root, line.strip())
      for line in res.stdout.splitlines()
      if line.strip().endswith(".py") and os.path.exists(os.path.join(maxtext_root, line.strip()))
  ]

  for py_file in changed_files:
    try:
      with open(py_file, "r", encoding="utf-8") as f:
        ast.parse(f.read(), filename=py_file)
    except SyntaxError as e:
      errors.append(f"Syntax error in {py_file}:{e.lineno}: {e.msg}")
  return len(errors) == 0, errors


def check_yaml_configs(maxtext_root: str, config_files: list[str] | None = None) -> tuple[bool, list[str]]:
  """Validates YAML syntax for base.yml and specified model configs."""
  errors: list[str] = []
  targets = list(config_files or [])
  base_yml = os.path.join(maxtext_root, "src", "maxtext", "configs", "base.yml")
  if os.path.exists(base_yml) and base_yml not in targets:
    targets.append(base_yml)

  for yml_path in targets:
    full_path = yml_path if os.path.isabs(yml_path) else os.path.join(maxtext_root, yml_path)
    if not os.path.exists(full_path):
      errors.append(f"Config file not found: {full_path}")
      continue
    try:
      with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
      if not isinstance(data, dict):
        errors.append(f"Config file {full_path} did not parse to a YAML mapping.")
    except yaml.YAMLError as e:
      errors.append(f"YAML syntax error in {full_path}: {e}")
  return len(errors) == 0, errors


def check_imports_docker(
    maxtext_root: str,
    docker_image: str,
    modules: list[str],
    extra_pip_packages: list[str] | None = None,
) -> tuple[bool, str]:
  """Runs a fast CPU-only import check inside the target Docker image."""
  check_script = os.path.join(maxtext_root, ".preflight_import_check.py")
  lines = ["import sys", "sys.path.insert(0, '/workspace/src')", "sys.path.insert(0, '/workspace')"]
  for mod in modules:
    lines.append(f"print('Importing {mod}...', flush=True)")
    lines.append(f"import {mod}")
  lines.append("print('ALL_IMPORTS_PASSED', flush=True)")

  pip_prefix = ""
  if extra_pip_packages:
    pkgs = " ".join(extra_pip_packages)
    pip_prefix = f"pip install --quiet --no-cache-dir {pkgs} >/dev/null 2>&1 || true; "

  try:
    with open(check_script, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        "-e",
        "JAX_PLATFORMS=cpu",
        "-v",
        f"{os.path.abspath(maxtext_root)}:/workspace",
        docker_image,
        "-c",
        f"{pip_prefix}python3 /workspace/.preflight_import_check.py",
    ]
    res = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=180, check=False)
    output = (res.stdout or "") + "\n" + (res.stderr or "")
    return res.returncode == 0 and "ALL_IMPORTS_PASSED" in output, output.strip()
  except subprocess.TimeoutExpired:
    return False, "Docker import check timed out after 180s."
  finally:
    if os.path.exists(check_script):
      os.remove(check_script)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--maxtext-root", default=".", help="Path to MaxText repository root")
  parser.add_argument("--fix-dockerignore", action="store_true", help="Append missing .dockerignore patterns")
  parser.add_argument("--configs", default="", help="Comma-separated YAML config paths to validate")
  parser.add_argument("--docker-image", default="", help="Optional Docker image for container import check")
  parser.add_argument("--check-imports", default="maxtext.train", help="Comma-separated modules to import")
  parser.add_argument("--extra-pip-packages", default="", help="Comma-separated pip packages for Docker check")
  args = parser.parse_args()

  root = os.path.abspath(args.maxtext_root)
  all_ok = True

  di_ok, missing = validate_dockerignore(root, fix=args.fix_dockerignore)
  if not di_ok:
    print(f"[FAIL] .dockerignore missing patterns: {missing}", file=sys.stderr)
    all_ok = False

  py_ok, py_errs = check_python_syntax(root)
  if not py_ok:
    for err in py_errs:
      print(f"[FAIL] {err}", file=sys.stderr)
    all_ok = False

  cfg_list = [c.strip() for c in args.configs.split(",") if c.strip()]
  yml_ok, yml_errs = check_yaml_configs(root, cfg_list)
  if not yml_ok:
    for err in yml_errs:
      print(f"[FAIL] {err}", file=sys.stderr)
    all_ok = False

  if args.docker_image and all_ok:
    mods = [m.strip() for m in args.check_imports.split(",") if m.strip()]
    pkgs = [p.strip() for p in args.extra_pip_packages.split(",") if p.strip()]
    imp_ok, imp_out = check_imports_docker(root, args.docker_image, mods, extra_pip_packages=pkgs)
    if not imp_ok:
      print(f"[FAIL] Docker import check failed:\n{imp_out}", file=sys.stderr)
      all_ok = False

  if not all_ok:
    return 1
  print("PRE-FLIGHT PASSED.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
