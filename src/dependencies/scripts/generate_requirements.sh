#!/bin/bash

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

# Generates requirements using seed-env, then optionally installs extra packages
# with overrides that are incompatible with the JAX seed lock.
#
# Usage:
#   bash src/dependencies/scripts/generate_requirements.sh [options]
#
# Options:
#   --base-requirements       Path to the base requirements file (required)
#   --generated-requirements  Name of the generated requirements file to update (required)
#   --seed-commit             JAX seed commit to use (required
#   --override-requirements   Path to file with uv override-dependencies (optional))
#   --hardware                Target hardware for seed-env (default: tpu)
#   --python-version          Python version to use (default: 3.12)

set -euo pipefail

SEED_ENV="seed-env"

# Defaults
HARDWARE="tpu"
OUTPUT_DIR="generated_artifacts"
BASE_REQUIREMENTS=""
GENERATED_REQUIREMENTS=""
OVERRIDE_REQUIREMENTS=""
PYTHON_VERSION="3.12"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-requirements)     BASE_REQUIREMENTS="$2"; shift 2 ;;
    --generated-requirements) GENERATED_REQUIREMENTS="$2"; shift 2 ;;
    --override-requirements) OVERRIDE_REQUIREMENTS="$2"; shift 2 ;;
    --seed-commit)            SEED_COMMIT="$2"; shift 2 ;;
    --python-version)     PYTHON_VERSION="$2"; shift 2 ;;
    --hardware)          HARDWARE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$BASE_REQUIREMENTS" ]]; then
  echo "Error: --base-requirements is required." >&2
  exit 1
fi
if [[ -z "$GENERATED_REQUIREMENTS" ]]; then
  echo "Error: --generated-requirements is required." >&2
  exit 1
fi
if [[ -z "$SEED_COMMIT" ]]; then
  echo "Error: --seed-commit is required." >&2
  exit 1
fi

ARTIFACT_DIR="$OUTPUT_DIR/python${PYTHON_VERSION//./_}"

echo "=== Running seed-env with $BASE_REQUIREMENTS ==="
"$SEED_ENV" \
  --local-requirements "$BASE_REQUIREMENTS" \
  --host-name MaxText \
  --hardware "$HARDWARE" \
  --python-version "$PYTHON_VERSION" \
  --seed-commit "$SEED_COMMIT" \
  --output-dir "$OUTPUT_DIR"

# Recursively expand a requirements file into a flat list of package specs,
# following -r <file> includes and stripping comments.
expand_requirements() {
  local file="$1"
  local dir
  dir="$(dirname "$file")"
  while IFS= read -r line; do
    [[ "$line" =~ ^\s*# ]] && continue
    [[ -z "${line// }" ]] && continue
    if [[ "$line" =~ ^-r[[:space:]]+(.+)$ ]]; then
      expand_requirements "$dir/${BASH_REMATCH[1]}"
    else
      echo "$line" | sed 's/[[:space:]]*#.*//'
    fi
  done < "$file"
}

strip_exact_pins() {
  # Convert ==X to >=X so uv gets a minimum version floor
  sed 's/^\([A-Za-z0-9._-]*\(\[[^]]*\]\)\?\)[[:space:]]*==[[:space:]]*\([^,; ]*\)/\1>=\3/'
}

if [[ "$BASE_REQUIREMENTS" == *post-train* ]]; then
  echo "=== Cloning vllm and tpu-inference Github repositories ==="
  WORK_DIR="$(pwd)"
  GITHUB_DEPS="src/dependencies/extra_deps/post_train_github_deps.txt"

  TPU_INFERENCE_COMMIT="$(grep '^tpu-inference' "$GITHUB_DEPS" | sed 's|.*/archive/\([a-f0-9]*\)\.zip.*|\1|')"
  VLLM_COMMIT="$(grep '^vllm' "$GITHUB_DEPS" | sed 's|.*@\([a-f0-9]*\)[[:space:]]*$|\1|')"
  echo "Using tpu-inference commit $TPU_INFERENCE_COMMIT, and vllm commit $VLLM_COMMIT"

  git clone https://github.com/vllm-project/tpu-inference.git
  git -C tpu-inference checkout "${TPU_INFERENCE_COMMIT}"

  git clone https://github.com/vllm-project/vllm.git
  git -C vllm checkout "${VLLM_COMMIT}"

  echo "=== Adding tpu-inference and vllm requirements to lock file ==="
  # Exclude tpu-inference (handled separately) and nixl (CUDA-only, not needed for TPU).
  mapfile -t _vllm_pkgs < <(
    expand_requirements "$WORK_DIR/vllm/requirements/tpu.txt" \
      | grep -viE '^(tpu-inference|nixl)' \
      | strip_exact_pins
  )
  mapfile -t _tpu_pkgs < <(
    expand_requirements "$WORK_DIR/tpu-inference/requirements.txt" \
      | strip_exact_pins
  )

  rm -rf "$WORK_DIR/tpu-inference" "$WORK_DIR/vllm"

  UV_TORCH_BACKEND=cpu VLLM_TARGET_DEVICE=tpu uv add \
    --managed-python \
    --no-sync \
    --resolution=lowest \
    --directory "$ARTIFACT_DIR" \
    "${_vllm_pkgs[@]}" \
    "${_tpu_pkgs[@]}"
fi

if [[ -n "$OVERRIDE_REQUIREMENTS" ]]; then
  echo "=== Adding overrides from $OVERRIDE_REQUIREMENTS ==="

  # Append [tool.uv] with overrides to the artifact pyproject.toml.
  # Overrides force specific versions that conflict with the JAX seed lock.
  {
    echo ""
    echo "[tool.uv]"
    echo "override-dependencies = ["
    grep -v '^\s*#' "$OVERRIDE_REQUIREMENTS" | grep -v '^\s*$' | sed 's/.*/"&",/' | sed 's/^/  /'
    echo "]"
  } >> "$ARTIFACT_DIR/pyproject.toml"

  mapfile -t _pkgs < <(grep -v '^\s*#' "$OVERRIDE_REQUIREMENTS" | grep -v '^\s*$')
  uv add \
    --managed-python \
    --no-sync \
    --resolution=lowest \
    --directory "$ARTIFACT_DIR" \
    "${_pkgs[@]}"
fi

echo "=== Exporting updated lock to generated requirements ==="
uv export \
  --managed-python \
  --locked \
  --no-hashes \
  --no-annotate \
  --resolution=lowest \
  --directory "$ARTIFACT_DIR" \
  --output-file "$GENERATED_REQUIREMENTS"

# Remove editable self-install added by uv export; the package is installed separately.
sed -i '/^-e \./d' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"

# uv export pins exact versions with ==; convert to >= so installers can
# pick up compatible newer releases without re-running the full lock.
sed -i 's/^\([A-Za-z0-9][A-Za-z0-9._-]*\)==/\1>=/g' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"

# Replace uv-generated header with custom header
sed -i '1,2c\# Generated by generate_requirements.sh using seed-env tool. Do not edit manually.\n#    See https://maxtext.readthedocs.io/en/latest/development/update_dependencies.html for details.\n' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"

# Post-process TPU post-training requirements to force CPU torch and remove CUDA packages
if [[ "$HARDWARE" == "tpu" && "$GENERATED_REQUIREMENTS" == *post-train* ]]; then
  echo "=== Post-processing TPU post-train requirements ==="
  # Remove nvidia and cuda packages
  sed -i '/^nvidia-/d' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"
  sed -i '/^cuda-/d' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"

  # Replace torch and torchvision with +cpu versions (must use == for local version labels)
  sed -i 's/^\(torch\)>=\([0-9.]*\)\(.*\)$/\1==\2+cpu\3/' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"
  sed -i 's/^\(torchvision\)>=\([0-9.]*\)\(.*\)$/\1==\2+cpu\3/' "$ARTIFACT_DIR/$GENERATED_REQUIREMENTS"
fi

echo "Updated generated requirements: $ARTIFACT_DIR/$GENERATED_REQUIREMENTS"
