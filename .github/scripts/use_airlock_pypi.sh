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

# Points uv and pip at the Airlock PyPI mirror in Artifact Registry when this runner can reach it.
#
# Source this file (do not execute it) so the exported variables reach the installs that follow in
# the same step. CI jobs otherwise download every dependency from public PyPI, whose throughput
# from the runners varies widely. Airlock's trusted repository can lag behind new releases, so
# public PyPI stays configured as an extra index, as in maxtext_tpu_dependencies.Dockerfile. If no
# token is available or the mirror does not answer, nothing is changed and installs use public PyPI.

AIRLOCK_PYPI_INDEX="us-python.pkg.dev/artifact-foundry-prod/python-3p-trusted/simple/"
AIRLOCK_TOKEN=$(gcloud auth print-access-token 2>/dev/null || true)
# Probe a package page: the index root itself answers 404 even when the mirror is reachable.
if [ -n "${AIRLOCK_TOKEN}" ] && \
    curl -fsS -o /dev/null --max-time 20 -H "Authorization: Bearer ${AIRLOCK_TOKEN}" "https://${AIRLOCK_PYPI_INDEX}pip/"; then
  echo "::add-mask::${AIRLOCK_TOKEN}"
  export UV_INDEX_URL="https://oauth2accesstoken:${AIRLOCK_TOKEN}@${AIRLOCK_PYPI_INDEX}"
  export UV_EXTRA_INDEX_URL="https://pypi.org/simple/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
  export PIP_INDEX_URL="${UV_INDEX_URL}"
  export PIP_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL}"
  echo "Package index: Airlock PyPI mirror (https://${AIRLOCK_PYPI_INDEX}), falling back to pypi.org"
else
  echo "Package index: public PyPI (Airlock PyPI mirror not reachable from this runner)"
fi
unset AIRLOCK_TOKEN AIRLOCK_PYPI_INDEX
