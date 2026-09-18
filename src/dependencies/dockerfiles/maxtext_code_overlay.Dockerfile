# syntax=docker.io/docker/dockerfile:1.7-labs
#
# Code-overlay Dockerfile for fast-rebuild mode.
# Takes a source image (with all dependencies pre-installed) and replaces
# only the code directories with new code from the build context.
#
# Usage (CI): FROM <source_image>@sha256:<digest>
# The source image must be a previously built maxtext_*:<run_id> image.
#
# This Dockerfile is intentionally minimal — it does NOT install dependencies,
# run setup.sh, or modify system packages. Dependencies come from the source image.
#
# This is the "clean" variant: it removes the old code directories before copying
# the new code, so deleted or renamed files do not survive, and it re-installs the
# vLLM adapter for post-training images. Because of the RUN instructions BuildKit
# must pull the source image's layers, which costs minutes. When no code file was
# removed the drift guard selects maxtext_code_overlay_fast.Dockerfile instead.

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
FROM ${SOURCE_IMAGE}

WORKDIR /deps

# Remove old code layers.
# Note: rm in a new layer creates whiteout entries (~negligible size).
# The deleted files' data still exists in parent layers but is not visible.
# This adds ~5-10% overhead to a 1-2GB image — acceptable trade-off
# for guaranteed dependency identity.
RUN rm -rf /deps/src/maxtext /deps/tests /deps/benchmarks /deps/pytest.ini

# Copy new code from the build context.
# PACKAGE_DIR = .venv/lib/python3.12/site-packages (same as full build).
ARG PACKAGE_DIR
COPY ${PACKAGE_DIR}/maxtext/ src/maxtext/

# tests/, pytest.ini, benchmarks/ come from the checkout root (not PACKAGE_DIR),
# same as the full Dockerfile (maxtext_tpu_dependencies.Dockerfile lines 80-82).
COPY tests*/ tests/
COPY pytest.ini* pytest.ini
COPY benchmarks*/ benchmarks/

# Re-install the MaxText vLLM adapter for post-training images.
# The source image installed it via install_post_train_extra_deps.py from
# /deps/src/maxtext/integration/vllm. After COPY replaces that directory,
# the pip-installed package metadata (entry_points, .dist-info) is stale.
# This is a --no-deps install of a tiny local package (~5 files) — <1s.
# `python3 -m uv` targets the image's own interpreter, like setup.sh does; a bare
# `uv pip install --python python3` refuses to install outside a virtualenv.
ARG WORKFLOW=pre-training
RUN if [ "$WORKFLOW" = "post-training" ]; then \
      python3 -m uv pip install --no-deps /deps/src/maxtext/integration/vllm; \
    fi

# Re-download test assets from GCS if building with test assets.
# The source image already has them in /deps/tests/assets/golden_logits/.
# For most fast-rebuilds the GCS content is unchanged so this is fast.
ARG INCLUDE_TEST_ASSETS=false
RUN if [ "$INCLUDE_TEST_ASSETS" = "true" ]; then \
        echo "Downloading test assets from GCS..."; \
        if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}/golden_logits"; then \
          echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests."; \
        fi; \
    fi
