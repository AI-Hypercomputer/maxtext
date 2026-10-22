#!/usr/bin/env bash
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

# Shows the CI Pulse page with the views the PoC wrote to the bucket, on this machine only.
#
#   cloudrun/preview.sh [PORT]                 serve on http://localhost:PORT (default 8765)
#   cloudrun/preview.sh --screenshot OUT.png   save a picture of the page and exit
#
# The page comes from gh-pages (dev/bench/ci-pulse/index.html), the data from gs://$BUCKET/views.
# Nothing is published anywhere.

set -euo pipefail

# Only CI_PULSE_* variables are read; generic names like BUCKET are often set for other work.
BUCKET="${CI_PULSE_BUCKET:-mesa-ci-pulse}"
PAGE_REF="${CI_PULSE_PAGE_REF:-origin/gh-pages}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

site="$(mktemp -d -t mesa-ci-pulse-preview-XXXXXX)"
trap 'rm -rf "${site}"' EXIT
git -C "${HERE}" show "${PAGE_REF}:dev/bench/ci-pulse/index.html" >"${site}/index.html"
mkdir -p "${site}/views"
gcloud storage rsync --recursive "gs://${BUCKET}/views" "${site}/views" --quiet
echo "views: $(find "${site}/views" -type f | wc -l) file(s) from gs://${BUCKET}/views"

if [[ "${1:-}" == "--screenshot" ]]; then
  out="$(realpath -m "${2:?usage: preview.sh --screenshot OUT.png}")"
  port=8765
  python3 -m http.server "${port}" --bind 127.0.0.1 --directory "${site}" >/dev/null 2>&1 &
  server=$!
  trap 'kill "${server}" 2>/dev/null; rm -rf "${site}"' EXIT
  sleep 1
  google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --window-size=1600,2400 \
    --virtual-time-budget=15000 --screenshot="${out}" "http://127.0.0.1:${port}/#ci" >/dev/null 2>&1
  echo "screenshot: ${out}"
else
  port="${1:-8765}"
  echo "open http://localhost:${port}/  (Ctrl-C to stop)"
  python3 -m http.server "${port}" --bind 127.0.0.1 --directory "${site}"
fi
