# Copyright 2023-2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Builds a CI test image with every MaxText dependency pre-installed but without
# MaxText itself.
#
# The clean_py_env*.Dockerfile images ship a bare interpreter, so each CI job had
# to resolve and download the full dependency tree from PyPI before it could run a
# single test. This image hoists that work into a single build, keyed on the
# content of the pinned requirements, so a test job only has to install the
# MaxText wheel built for its own commit.
#
# MaxText is deliberately not installed here: it changes on every commit, whereas
# the dependency set changes only when a pin moves. Tests install the wheel with
# `uv pip install --no-deps`, which is safe because the extras in pyproject.toml
# are exactly the generated requirement files installed below.
#
# Usage:
#   docker build \
#     --build-arg BASE_IMAGE=us-docker.pkg.dev/<project>/maxtext-images/maxtext-unit-test-tpu:py312-bookworm \
#     --build-arg MAXTEXT_EXTRA=tpu \
#     -t maxtext-deps-tpu:<deps-hash> \
#     -f src/dependencies/dockerfiles/test_deps_env.Dockerfile .

ARG BASE_IMAGE=maxtext-unit-test-tpu:py312-bookworm
FROM ${BASE_IMAGE}

# Which optional-dependency group to install: tpu, cuda12 or tpu-post-train.
ARG MAXTEXT_EXTRA=tpu
ENV MAXTEXT_EXTRA=${MAXTEXT_EXTRA}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
# Mirrors the container env the test workflow sets, so vLLM and torch resolve the
# same way here as they do in the non-prebuilt install path.
ENV VLLM_TARGET_DEVICE=tpu
ENV UV_TORCH_BACKEND=cpu

# Test jobs symlink this to ./.venv in the workspace. Keeping the dependencies in
# a venv (rather than system site-packages) means the resulting layout is
# identical to the one `uv venv --seed` produces in the non-prebuilt path, so
# everything downstream -- `source .venv/bin/activate`, `.venv/bin/python3`, and
# the LD_LIBRARY_PATH discovery that walks `.venv/lib/**/nvidia` -- is unchanged.
ENV MAXTEXT_DEPS_VENV=/opt/maxtext-venv
ENV VIRTUAL_ENV=${MAXTEXT_DEPS_VENV}
ENV PATH="${MAXTEXT_DEPS_VENV}/bin:${PATH}"

WORKDIR /deps
COPY src/dependencies/ src/dependencies/

# install_post_train_extra_deps resolves this path relative to itself and installs
# it as the maxtext_vllm_adapter distribution, so it has to be present for the
# script to run unmodified. The copy is only a placeholder: it is always replaced
# at test time by the version from the commit's own wheel, because uv compares the
# recorded file:// URL and the two paths differ. maxtext_tpu_dependencies.Dockerfile
# copies the same directory for the same reason.
COPY src/maxtext/integration/vllm/ src/maxtext/integration/vllm/

RUN python3 -m uv venv --seed "${MAXTEXT_DEPS_VENV}"

RUN --mount=type=cache,target=/root/.cache/uv \
    (trap 'kill $SLEEP_PID 2>/dev/null || true; exit 0' TERM INT; while true; do sleep 60 & SLEEP_PID=$!; wait $SLEEP_PID || break; echo "[build keepalive] $(date -u +%H:%M:%S)"; done) & \
    KEEPALIVE_PID=$!; trap 'kill $KEEPALIVE_PID 2>/dev/null || true; wait $KEEPALIVE_PID 2>/dev/null || true' EXIT; \
    export UV_LINK_MODE=copy && \
    uv pip install --resolution=lowest \
      -r "src/dependencies/requirements/generated_requirements/${MAXTEXT_EXTRA}-requirements.txt" && \
    if [ "${MAXTEXT_EXTRA}" = "tpu-post-train" ]; then \
      python3 -m src.dependencies.scripts.install_post_train_extra_deps; \
    else \
      python3 -m src.dependencies.scripts.install_pre_train_extra_deps; \
    fi

RUN uv pip freeze > /opt/maxtext-deps-freeze.txt && \
    echo "MaxText dependency image for '${MAXTEXT_EXTRA}'" && \
    wc -l /opt/maxtext-deps-freeze.txt

WORKDIR /
RUN rm -rf /deps
