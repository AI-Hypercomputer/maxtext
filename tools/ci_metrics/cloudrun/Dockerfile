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

# The CI Pulse collector as a Cloud Run job. Build it from tools/ci_metrics:
#
#   docker build -f cloudrun/Dockerfile -t IMAGE .
#
# python:3.12-slim, pinned by digest so a rebuild gets the same base.
FROM python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Every dependency is pinned with its hash, and nothing outside the list may be installed. The
# index can be a mirror (deploy.sh passes the build machine's pip index); the hashes still decide.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org
COPY cloudrun/requirements.txt cloudrun/requirements.txt
RUN pip install --root-user-action=ignore --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --require-hashes --no-deps -r cloudrun/requirements.txt

COPY collector/ collector/
COPY cloudrun/ cloudrun/

# A user with no home and no shell. The job writes only under the temporary directory.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin ci-pulse
USER 10001

ENTRYPOINT ["python", "-m", "cloudrun.job"]
