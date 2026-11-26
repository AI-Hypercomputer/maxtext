# syntax=docker.io/docker/dockerfile:1.7-labs
#
# Code-overlay Dockerfile for cached code-free dependencies-only images.
# Takes a cached maxtext_*_deps@sha256:<digest> image (TPU or GPU)
# and layers MaxText code, tests, benchmarks, and path env vars on top.
#
# Targets:
#   - overlay-pre-training: COPY --link and ENV only (no RUN), so BuildKit
#     never pulls the base image layers (~10s for TPU and GPU amd64/arm64).
#     When INCLUDE_TEST_ASSETS is enabled in CI, the workflow downloads
#     gs://maxtext-test-assets/* into tests/assets/golden_logits/ on the host
#     runner beforehand so COPY --link tests*/ /deps/tests/ includes them.
#   - overlay-post-training: inherits overlay-pre-training and refreshes the
#     installed MaxText vLLM adapter from /deps/src/maxtext/integration/vllm.
#
# Usage (CI):
#   docker build --target overlay-pre-training \
#     --build-arg SOURCE_IMAGE=<deps_image>@sha256:<digest> \
#     --build-arg PACKAGE_DIR=.wheel \
#     -f maxtext_code_overlay.Dockerfile .

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

ARG SOURCE_IMAGE
FROM ${SOURCE_IMAGE} AS overlay-pre-training

WORKDIR /deps

# Set MaxText path env vars (omitted from code-free *_deps images so
# external overlays at /app auto-resolve via src/maxtext/utils/globals.py).
ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/maxtext
ENV MAXTEXT_REPO_ROOT=/deps

# Copy code and non-installed dependency helper files (e.g. preflight.sh,
# rto_setup.sh) with --link and absolute destinations so BuildKit builds
# independent layers without pulling SOURCE_IMAGE.
# PACKAGE_DIR = unpacked maxtext wheel directory (e.g. .wheel in CI).
ARG PACKAGE_DIR
COPY --link ${PACKAGE_DIR}/dependencies/extra_deps/ /deps/src/dependencies/extra_deps/
COPY --link ${PACKAGE_DIR}/dependencies/requirements/ /deps/src/dependencies/requirements/
COPY --link ${PACKAGE_DIR}/dependencies/scripts/ /deps/src/dependencies/scripts/
COPY --link ${PACKAGE_DIR}/maxtext/ /deps/src/maxtext/

# tests/, pytest.ini, benchmarks/ come from the checkout root (not PACKAGE_DIR),
# matching maxtext_tpu_dependencies.Dockerfile and maxtext_gpu_dependencies.Dockerfile.
COPY --link tests*/ /deps/tests/
COPY --link pytest.ini* /deps/pytest.ini
COPY --link benchmarks*/ /deps/benchmarks/

FROM overlay-pre-training AS overlay-post-training

# Re-install the MaxText vLLM adapter for post-training images so the installed
# package metadata (entry_points, .dist-info) matches the overlaid code.
# This is a --no-deps install of a tiny local package (~5 files) — <1s.
# `python3 -m uv` targets the image's own interpreter, like setup.sh does; a bare
# `uv pip install --python python3` refuses to install outside a virtualenv.
RUN python3 -m uv pip install --no-deps /deps/src/maxtext/integration/vllm
