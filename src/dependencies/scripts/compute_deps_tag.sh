#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Computes the deterministic content-hash tag for a stable dependencies-only
# MaxText Docker image built from maxtext_{tpu,gpu}_dependencies.Dockerfile
# (--target dependencies).
#
# Usage:
#   bash src/dependencies/scripts/compute_deps_tag.sh \
#     [--device=tpu|gpu] \
#     [--workflow=pre-training|post-training] \
#     [--tf=true|false] \
#     [--base-digest=sha256:...] \
#     [--repo-root=/path/to/maxtext] \
#     [--dockerfile=/path/to/Dockerfile] \
#     [--print-base-image]
#
# Outputs:
#   deps-<16hex> (or deps-<16hex>-b<12hex> with --base-digest), or the
#   Airlock-mirrored default ARG BASEIMAGE when --print-base-image is set.

set -euo pipefail

DEVICE="tpu"
WORKFLOW="pre-training"
TF="true"
BASE_DIGEST=""
REPO_ROOT=""
DOCKERFILE=""
PRINT_BASE_IMAGE="false"

for arg in "$@"; do
  case "${arg}" in
    --device=*)
      DEVICE="${arg#*=}"
      ;;
    --workflow=*)
      WORKFLOW="${arg#*=}"
      ;;
    --tf=*)
      TF="${arg#*=}"
      ;;
    --base-digest=*)
      BASE_DIGEST="${arg#*=}"
      ;;
    --repo-root=*)
      REPO_ROOT="${arg#*=}"
      ;;
    --dockerfile=*)
      DOCKERFILE="${arg#*=}"
      ;;
    --print-base-image)
      PRINT_BASE_IMAGE="true"
      ;;
    -h|--help)
      sed -n '17,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument '${arg}'." >&2
      exit 1
      ;;
  esac
done

DEVICE="${DEVICE,,}"
WORKFLOW="${WORKFLOW,,}"
TF="${TF,,}"

if [[ "${TF}" != "true" && "${TF}" != "false" ]]; then
  echo "ERROR: Invalid --tf='${TF}'. Expected 'true' or 'false'." >&2
  exit 1
fi

if [[ -z "${REPO_ROOT}" ]]; then
  if [[ -d "src/dependencies" ]]; then
    REPO_ROOT="${PWD}"
  else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  fi
fi

if [[ "${PRINT_BASE_IMAGE}" == "true" ]]; then
  if [[ -z "${DOCKERFILE}" ]]; then
    case "${DEVICE}" in
      tpu) DOCKERFILE="${REPO_ROOT}/src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile" ;;
      gpu) DOCKERFILE="${REPO_ROOT}/src/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile" ;;
      *)
        echo "ERROR: Unsupported --device='${DEVICE}'. Expected 'tpu' or 'gpu'." >&2
        exit 1
        ;;
    esac
  fi
  if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "ERROR: Dockerfile not found: ${DOCKERFILE}" >&2
    exit 1
  fi
  default_base=$(sed -n 's/^ARG BASEIMAGE=//p' "${DOCKERFILE}" | head -n1 | cut -d'#' -f1 | tr -d '\r"'\''[:space:]')
  if [[ -z "${default_base}" ]]; then
    echo "ERROR: Could not parse default ARG BASEIMAGE from ${DOCKERFILE}." >&2
    exit 1
  fi
  airlock_registry="us-docker.pkg.dev/artifact-foundry-prod/docker-3p-trusted"
  if [[ "${default_base}" == *..* ]]; then
    echo "ERROR: Unsupported ARG BASEIMAGE '${default_base}' in ${DOCKERFILE}." >&2
    exit 1
  elif [[ "${default_base}" == "${airlock_registry}/"* ]]; then
    printf '%s\n' "${default_base}"
  elif [[ "${default_base}" =~ ^[a-z0-9][a-z0-9_-]*(/[a-z0-9._-]+)*:[a-zA-Z0-9._-]+$ ]]; then
    printf '%s/%s\n' "${airlock_registry}" "${default_base}"
  else
    echo "ERROR: Unsupported ARG BASEIMAGE '${default_base}' in ${DOCKERFILE}." >&2
    exit 1
  fi
  exit 0
fi

case "${DEVICE}:${WORKFLOW}" in
  tpu:pre-training)
    inputs=(
      "src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile"
      "src/dependencies/scripts/setup.sh"
      "src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
      "src/dependencies/extra_deps/pre_train_github_deps.txt"
      "src/dependencies/scripts/install_pre_train_extra_deps.py"
    )
    # TODO: remove once TensorFlow is deprecated.
    if [[ "${TF}" == "true" ]]; then
      inputs+=("src/dependencies/extra_deps/tf_requirements.txt")
    fi
    ;;
  tpu:post-training)
    inputs=(
      "src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile"
      "src/dependencies/scripts/setup.sh"
      "src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt"
      "src/dependencies/extra_deps/post_train_github_deps.txt"
      "src/dependencies/scripts/install_post_train_extra_deps.py"
    )
    ;;
  gpu:pre-training)
    inputs=(
      "src/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile"
      "src/dependencies/scripts/setup.sh"
      "src/dependencies/requirements/generated_requirements/cuda12-requirements.txt"
      "src/dependencies/extra_deps/pre_train_github_deps.txt"
      "src/dependencies/scripts/install_pre_train_extra_deps.py"
    )
    # TODO: remove once TensorFlow is deprecated.
    if [[ "${TF}" == "true" ]]; then
      inputs+=("src/dependencies/extra_deps/tf_requirements.txt")
    fi
    ;;
  *)
    echo "ERROR: Unsupported --device='${DEVICE}' with --workflow='${WORKFLOW}'. Expected tpu:pre-training, tpu:post-training, or gpu:pre-training." >&2
    exit 1
    ;;
esac

for input in "${inputs[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${input}" ]]; then
    echo "ERROR: Required dependency hash input missing: ${REPO_ROOT}/${input}" >&2
    exit 1
  fi
done

# Hash relative paths from REPO_ROOT so the tag is independent of checkout location.
reqs_hash=$(
  cd "${REPO_ROOT}" && {
    printf 'device=%s\nworkflow=%s\nmode=stable\ntf=%s\n' "${DEVICE}" "${WORKFLOW}" "${TF}"
    sha256sum "${inputs[@]}"
  } | sha256sum | cut -c1-16
)

tag="deps-${reqs_hash}"

if [[ -n "${BASE_DIGEST}" ]]; then
  clean_digest="${BASE_DIGEST#sha256:}"
  if [[ ! "${clean_digest}" =~ ^[0-9a-f]{12,64}$ ]]; then
    echo "ERROR: Invalid --base-digest='${BASE_DIGEST}'. Expected sha256:<hex>." >&2
    exit 1
  fi
  tag="${tag}-b${clean_digest:0:12}"
fi

printf '%s\n' "${tag}"
