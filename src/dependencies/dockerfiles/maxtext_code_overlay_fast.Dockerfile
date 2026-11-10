# syntax=docker.io/docker/dockerfile:1.7-labs
#
# Fast code-overlay Dockerfile for fast-rebuild mode.
#
# Same purpose as maxtext_code_overlay.Dockerfile (replace only the code in a
# previously built image), but built from COPY --link instructions only, with no
# RUN. BuildKit therefore never has to pull the source image's layers, and the
# export only uploads the new code layers: seconds instead of minutes.
#
# Trade-off: nothing is deleted from the source image. A file that was removed or
# renamed between the source commit and the current commit would still exist in
# the result. The drift guard (.github/scripts/fast_rebuild_drift_guard.sh) selects
# this variant only when no code file was removed, and for post-training images
# only when src/maxtext/integration/vllm/ is unchanged (its installed copy in
# site-packages would otherwise be stale). Otherwise the clean variant is used.
#
# Test assets already downloaded into the source image (tests/assets/golden_logits)
# are kept, so no GCS download is needed here.

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

ARG PACKAGE_DIR
# Declared so the build args passed by the workflow are consumed; unused here.
ARG WORKFLOW=pre-training
ARG INCLUDE_TEST_ASSETS=false

# Absolute destinations: with --link the copy happens in an independent layer that
# is placed on top of the source image without touching its layers.
COPY --link ${PACKAGE_DIR}/maxtext/ /deps/src/maxtext/
COPY --link tests*/ /deps/tests/
COPY --link pytest.ini* /deps/pytest.ini
COPY --link benchmarks*/ /deps/benchmarks/
