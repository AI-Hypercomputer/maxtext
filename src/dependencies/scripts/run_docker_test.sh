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

# This script runs MaxText unit, integration, or end-to-end tests in Docker containers
# while allowing users to choose whether to build all images, perform a fast rebuild
# of only code layers, or re-run tests using an existing image.
#
# Supported modes:
#   1. build-all    : Re-run all steps including building the base dependency image and runner image.
#   2. test-only    : Re-run just the tests using the existing image (no build, fast re-run).
#   3. fast-rebuild : Rebuild the runner image using cached dependencies, updating only code.

set -e

# Default parameter values
PACKAGE_DIR="${PACKAGE_DIR:-src}"
MODE="${MODE:-build-all}"
COMMAND="${COMMAND:-python3 -m pytest -vv tests/unit/}"
BASE_IMAGE="${BASE_IMAGE:-maxtext_base_image}"
IMAGE_NAME="${IMAGE_NAME:-maxtext_base_image__runner}"
DEVICE="${DEVICE:-tpu}"
BUILD_MODE="${BUILD_MODE:-stable}"
CLEANUP_IMAGE="${CLEANUP_IMAGE:-false}"
DOCKER_RUN_FLAGS="${DOCKER_RUN_FLAGS:-}"

# Parse command line flags and key=value arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift 1
      ;;
    -c|--command)
      COMMAND="$2"
      shift 2
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift 1
      ;;
    -i|--image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --image=*)
      IMAGE_NAME="${1#*=}"
      shift 1
      ;;
    --base-image=*)
      BASE_IMAGE="${1#*=}"
      shift 1
      ;;
    --cleanup)
      CLEANUP_IMAGE="true"
      shift 1
      ;;
    --no-cleanup)
      CLEANUP_IMAGE="false"
      shift 1
      ;;
    --cleanup=*)
      CLEANUP_IMAGE="${1#*=}"
      shift 1
      ;;
    *=*)
      IFS='=' read -r KEY VALUE <<< "$1"
      export "${KEY}"="${VALUE}"
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [KEY=VALUE...]"
      echo ""
      echo "Options:"
      echo "  -m, --mode <mode>        Execution mode:"
      echo "                             build-all    : Build dependency image + runner image + run tests"
      echo "                             test-only    : Run tests using existing image (no rebuild)"
      echo "                             fast-rebuild : Rebuild only runner code layer + run tests"
      echo "  -c, --command <cmd>      Command to execute inside Docker container"
      echo "                           (default: 'python3 -m pytest -vv tests/unit/')"
      echo "  -i, --image <name>       Runner Docker image name (default: maxtext_base_image__runner)"
      echo "      --base-image <name>  Base dependency Docker image name (default: maxtext_base_image)"
      echo "      --cleanup=<bool>     Remove image after test run (default: false)"
      echo ""
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Normalize mode string
case "${MODE}" in
  build-all|rebuild-all|all|1)
    MODE="build-all"
    ;;
  test-only|reuse-image|no-build|2)
    MODE="test-only"
    ;;
  fast-rebuild|rebuild-code|code-only|3)
    MODE="fast-rebuild"
    ;;
  *)
    echo "ERROR: Invalid mode '${MODE}'. Supported modes: build-all, test-only, fast-rebuild." >&2
    exit 1
    ;;
esac

echo "=========================================================="
echo "MaxText Docker Test Runner"
echo "=========================================================="
echo "Mode           : ${MODE}"
echo "Runner Image   : ${IMAGE_NAME}"
echo "Base Image     : ${BASE_IMAGE}"
echo "Test Command   : ${COMMAND}"
echo "Cleanup Image  : ${CLEANUP_IMAGE}"
echo "=========================================================="

# Check for Docker daemon access
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Permission denied while trying to connect to the Docker daemon." >&2
  echo "Please check your Docker permissions (sudo, docker group, etc.)." >&2
  exit 1
fi

build_base_image() {
  echo "----------------------------------------------------------"
  echo "Step 1: Building base dependency image '${BASE_IMAGE}'..."
  echo "----------------------------------------------------------"
  if [[ -f "${PACKAGE_DIR}/dependencies/scripts/docker_build_dependency_image.sh" ]]; then
    bash "${PACKAGE_DIR}/dependencies/scripts/docker_build_dependency_image.sh" \
      DEVICE="${DEVICE}" MODE="${BUILD_MODE}" LOCAL_IMAGE_NAME="${BASE_IMAGE}"
  else
    echo "ERROR: Base image build script not found under ${PACKAGE_DIR}/dependencies/scripts/" >&2
    exit 1
  fi
}

build_runner_image() {
  local use_cache="$1"
  echo "----------------------------------------------------------"
  echo "Step 2: Building runner image '${IMAGE_NAME}' from '${BASE_IMAGE}'..."
  echo "        Cache enabled: ${use_cache}"
  echo "----------------------------------------------------------"
  if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: Base dependency image '${BASE_IMAGE}' not found locally." >&2
    echo "       Please run with '--mode=build-all' first to build the base dependencies." >&2
    exit 1
  fi

  local cache_args=()
  if [[ "${use_cache}" == "false" ]]; then
    cache_args+=("--no-cache")
  fi

  docker build "${cache_args[@]}" \
    --build-arg BASEIMAGE="${BASE_IMAGE}" \
    --build-arg PACKAGE_DIR="${PACKAGE_DIR}" \
    -f "${PACKAGE_DIR}/dependencies/dockerfiles/maxtext_runner.Dockerfile" \
    -t "${IMAGE_NAME}" .
}

case "${MODE}" in
  build-all)
    build_base_image
    build_runner_image "false"
    ;;
  fast-rebuild)
    build_runner_image "true"
    ;;
  test-only)
    echo "----------------------------------------------------------"
    echo "Skipping image build. Verifying runner image '${IMAGE_NAME}' exists..."
    echo "----------------------------------------------------------"
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
      echo "ERROR: Runner image '${IMAGE_NAME}' not found locally." >&2
      echo "       Cannot re-run tests without an existing image." >&2
      echo "       Use '--mode=build-all' or '--mode=fast-rebuild' to create the image first." >&2
      exit 1
    fi
    ;;
esac

echo "----------------------------------------------------------"
echo "Step 3: Running test command inside container '${IMAGE_NAME}'..."
echo "----------------------------------------------------------"

set +e
docker run --rm -it --network host --privileged \
  -e MAXTEXT_REPO_ROOT=/deps \
  -e MAXTEXT_PKG_DIR=/deps/src/maxtext \
  -v "$(pwd):/deps" \
  ${DOCKER_RUN_FLAGS} \
  "${IMAGE_NAME}" \
  bash -c "${COMMAND}"
TEST_EXIT_CODE=$?
set -e

echo "----------------------------------------------------------"
if [[ "${TEST_EXIT_CODE}" -eq 0 ]]; then
  echo "TEST RESULT: SUCCESS (Exit code 0)"
else
  echo "TEST RESULT: FAILURE (Exit code ${TEST_EXIT_CODE})"
fi

if [[ "${CLEANUP_IMAGE}" == "true" ]]; then
  echo "Cleaning up Docker image '${IMAGE_NAME}' as requested..."
  docker image rm "${IMAGE_NAME}" 2>/dev/null || true
else
  echo "----------------------------------------------------------"
  echo "NOTICE: Docker image '${IMAGE_NAME}' has been preserved."
  echo "  - To re-run tests without rebuilding image  : bash $0 --mode=test-only"
  echo "  - To rebuild only code layers (fast rebuild): bash $0 --mode=fast-rebuild"
  echo "  - To rebuild all dependencies from scratch    : bash $0 --mode=build-all"
  echo "----------------------------------------------------------"
fi

exit "${TEST_EXIT_CODE}"
