#!/usr/bin/env bash
#
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
#
# =============================================================================
# Refresh the CI Pulse dashboard data by hand.
#
# This is the runbook in tools/ci_metrics/deploy/GOING-LIVE.md section 4.3, as a
# script. It runs one collector tick into a store that lives OUTSIDE this
# repository, then copies only the rendered views into
# dev/bench/ci-pulse/views/, which is what the dashboard reads.
#
#   tools/ci_metrics/deploy/refresh-data.sh
#
# It reads the GitHub API and writes two places on this machine: the store
# directory you name, and dev/bench/ci-pulse/views/. It never pushes, and it
# never commits unless you ask it to with --commit.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STORE="${HOME}/ci-metrics-store"
REPO='AI-Hypercomputer/maxtext'
BACKFILL_DAYS='30'
SINCE=''
UNTIL=''
MAX_RUNS=''
DRY_RUN='no'
DO_COMMIT='no'
DO_SERVE='no'
FORCE='no'
PORT='8000'

usage() {
  cat <<'USAGE'
Refresh the CI Pulse dashboard data from the GitHub API.

  tools/ci_metrics/deploy/refresh-data.sh [options]

Options:
  --store DIR          Where the raw history lives. Kept outside the repository
                       on purpose: only the views are ever published. THIS is
                       what accumulates - it is append-only, and every run adds
                       to it. Use the same one every time and history builds up.
                       Default: ~/ci-metrics-store
  --repo OWNER/NAME    Repository to read. Default: AI-Hypercomputer/maxtext
  --backfill-days N    How far back the FIRST run of an empty store reaches.
                       Ignored once the store has a watermark. Default: 30
  --since YYYY-MM-DD   Collect runs created on or after this day.
  --until YYYY-MM-DD   Collect runs created on or before this day.
  --max-runs N         Stop after N runs; the next run continues where this one
                       left off. Useful for splitting a big backfill.
  --dry-run            Collect and report, then stop. Copies nothing, and the
                       store is left untouched.
  --commit             Commit dev/bench/ci-pulse/views afterwards. Off by
                       default: look at the page before you record anything.
  --serve              Serve dev/bench on localhost when the copy is done, so
                       you can open the page straight away.
  --port N             Port for --serve. Default: 8000
  --force              Copy the views even if the tick reported a problem, or
                       even if the rebuild is smaller than what is already
                       published. Both checks exist to stop you losing history.
  -h, --help           This message.

A token is required. The script uses GITHUB_TOKEN if it is set, and falls back
to `gh auth token`. It needs to read Actions data on the target repository: a
classic token with the `repo` scope, or a fine-grained token with
`Actions: read` and `Metadata: read`.

Examples:
  # First time: thirty days of history.
  tools/ci_metrics/deploy/refresh-data.sh

  # Later: just what is new, then look at it.
  tools/ci_metrics/deploy/refresh-data.sh --serve

  # A big backfill, one bite at a time.
  tools/ci_metrics/deploy/refresh-data.sh --backfill-days 90 --max-runs 300
USAGE
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
  case "$1" in
    --store)         STORE="${2:?--store needs a directory}"; shift 2 ;;
    --repo)          REPO="${2:?--repo needs owner/name}"; shift 2 ;;
    --backfill-days) BACKFILL_DAYS="${2:?--backfill-days needs a number}"; shift 2 ;;
    --since)         SINCE="${2:?--since needs YYYY-MM-DD}"; shift 2 ;;
    --until)         UNTIL="${2:?--until needs YYYY-MM-DD}"; shift 2 ;;
    --max-runs)      MAX_RUNS="${2:?--max-runs needs a number}"; shift 2 ;;
    --port)          PORT="${2:?--port needs a number}"; shift 2 ;;
    --dry-run)       DRY_RUN='yes'; shift ;;
    --commit)        DO_COMMIT='yes'; shift ;;
    --serve)         DO_SERVE='yes'; shift ;;
    --force)         FORCE='yes'; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Where are we
# ---------------------------------------------------------------------------

step 'Checking the working tree'

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die 'not inside a git repository.'
COLLECTOR="${ROOT}/tools/ci_metrics"
TARGET="${ROOT}/dev/bench/ci-pulse/views"

[ -d "${COLLECTOR}/collector" ] \
  || die "no collector at ${COLLECTOR}/collector - is this the right branch?"
[ -f "${ROOT}/dev/bench/ci-pulse/index.html" ] \
  || die "no dashboard at dev/bench/ci-pulse/index.html - this branch does not serve the page, so there is nowhere to put the data."

say "repository  ${ROOT}"
say "branch      $(git -C "${ROOT}" branch --show-current)"
say "store       ${STORE}"
say "target      ${TARGET#"${ROOT}"/}"

# The store must not live inside the repository. data/*.ndjson is the raw
# history and would end up on a public website the first time someone ran
# `git add -A`.
case "$(cd "$(dirname "${STORE}")" 2>/dev/null && pwd || echo "${STORE}")" in
  "${ROOT}"|"${ROOT}"/*)
    die "the store is inside the repository (${STORE}). Raw history must stay out of the working tree - pick a path outside ${ROOT}." ;;
esac

# ---------------------------------------------------------------------------
# What we need
# ---------------------------------------------------------------------------

step 'Checking what the collector needs'

command -v python3 >/dev/null 2>&1 || die 'python3 is not on PATH.'
say "python      $(python3 --version 2>&1)"

python3 -c 'import requests' >/dev/null 2>&1 \
  || die 'the requests package is missing. Install it with: python3 -m pip install "requests>=2.31,<3"'
say 'requests    present'

if [ -z "${GITHUB_TOKEN:-}" ]; then
  if command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
    GITHUB_TOKEN="$(gh auth token)"
    say "token       taken from gh auth token"
  else
    die "no token. Set GITHUB_TOKEN, or log in with: gh auth login. Without one the API allows 60 requests an hour, which is not enough for a single run."
  fi
else
  say 'token       from GITHUB_TOKEN'
fi
export GITHUB_TOKEN

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

step 'Collecting'

# Built with plain ifs, not `test && args+=(...)`: under `set -e` a false test as
# the last command of an AND-list ends the script, which is not what a skipped
# optional flag should do.
args=(--out "${STORE}" --repo "${REPO}")
if [ -n "${SINCE}" ];         then args+=(--since "${SINCE}"); fi
if [ -n "${UNTIL}" ];         then args+=(--until "${UNTIL}"); fi
if [ -n "${BACKFILL_DAYS}" ]; then args+=(--backfill-days "${BACKFILL_DAYS}"); fi
if [ -n "${MAX_RUNS}" ];      then args+=(--max-runs "${MAX_RUNS}"); fi
if [ "${DRY_RUN}" = 'yes' ];  then args+=(--dry-run); fi

say "python3 -m collector.tick ${args[*]}"
say ''

# The tick prints its own report, so let it through unfiltered. PIPESTATUS
# carries its exit code past the tee.
LOG="$(mktemp -t ci-metrics-tick)"
set +e
( cd "${COLLECTOR}" && python3 -m collector.tick "${args[@]}" ) 2>&1 | tee "${LOG}"
TICK_STATUS="${PIPESTATUS[0]}"
set -e

case "${TICK_STATUS}" in
  0) ;;
  1) say ''
     say 'The tick exited 1: it lost data or could not write everything.'
     say 'Whatever it did collect is in the store, and repeating a tick is always safe.'
     if [ "${FORCE}" != 'yes' ]; then
       rm -f "${LOG}"
       die 'stopping before the copy. Run it again, or pass --force to publish what it did get.'
     fi
     say 'Continuing because --force was given.' ;;
  *) rm -f "${LOG}"
     die "the tick exited ${TICK_STATUS}: the command line was wrong, so repeating it would fail the same way." ;;
esac
rm -f "${LOG}"

if [ "${DRY_RUN}" = 'yes' ]; then
  step 'Dry run'
  say 'Nothing was written and nothing was copied.'
  exit 0
fi

# ---------------------------------------------------------------------------
# Publish the views
# ---------------------------------------------------------------------------

step 'Copying the views into the branch'

[ -f "${STORE}/views/meta.json" ] \
  || die "the collector wrote no ${STORE}/views/meta.json, so there is nothing to publish."

python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "${STORE}/views/meta.json" \
  || die 'views/meta.json is not valid JSON. Refusing to publish a broken index.'

# The views are a rendering of the store, not a second copy of the history, so
# they are replaced whole rather than merged. That is safe as long as the store
# is the same one as last time: the store appends, so each rebuild covers
# everything the last one did and more.
#
# It stops being safe when the store is a different or emptier one - a fresh
# clone, a wiped directory, a shorter --since. Then the rebuild is SMALLER than
# what is already published, and replacing it whole would delete history that
# exists nowhere else. So compare the two indexes first and refuse to shrink.
if [ -f "${TARGET}/meta.json" ]; then
  set +e
  python3 - "${TARGET}/meta.json" "${STORE}/views/meta.json" <<'PY'
import json, sys

def read(path):
    with open(path) as fh:
        return json.load(fh)

old, new = read(sys.argv[1]), read(sys.argv[2])
problems = []
for group, body in old.get("groups", {}).items():
    had = set(body.get("months", ()))
    now = set(new.get("groups", {}).get(group, {}).get("months", ()))
    gone = sorted(had - now)
    if gone:
        problems.append(f"{group}: would lose month(s) {', '.join(gone)}")
    was = sum(sum(t.values()) for t in body.get("rows", {}).values())
    is_ = sum(sum(t.values()) for t in new.get("groups", {}).get(group, {}).get("rows", {}).values())
    if is_ < was:
        problems.append(f"{group}: {was:,} row(s) published, {is_:,} in the rebuild")

if problems:
    print()
    print("  The rebuild is smaller than what is already published:")
    for line in problems:
        print(f"    - {line}")
    print()
    print("  This normally means the store is not the one the published views came")
    print("  from. Point --store at the store you used last time and run again; it")
    print("  appends, so nothing is lost. Use --force only if you mean to shrink it.")
    sys.exit(3)
PY
  SHRINK_STATUS=$?
  set -e
  if [ "${SHRINK_STATUS}" != '0' ] && [ "${FORCE}" != 'yes' ]; then
    die 'stopping: the copy would remove published data. See above.'
  fi

  # Whatever happens next, the previous copy is recoverable.
  rm -rf "${STORE}/views-previous"
  cp -R "${TARGET}" "${STORE}/views-previous"
  say "previous views kept at ${STORE}/views-previous"
fi

# Remove the old copy before writing the new one. A plain copy over the top
# would leave behind a month file the collector has since dropped: meta.json
# would stop listing it, and the result is a stale file nothing points at.
rm -rf "${TARGET}"
mkdir -p "$(dirname "${TARGET}")"
cp -R "${STORE}/views" "${TARGET}"

say "copied $(find "${TARGET}" -type f -name '*.json' | wc -l | tr -d ' ') file(s), $(du -sh "${TARGET}" | cut -f1) on disk"

# What is actually in it, read out of the index rather than guessed.
python3 - "${TARGET}/meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print()
print(f"  built at        {meta['generated_at']}")
print(f"  window          {meta['window_days']} days")
print(f"  runs not read   {meta['uncollected_runs']}")
for group, body in meta["groups"].items():
    months = ", ".join(body["months"]) or "none"
    rows = sum(sum(t.values()) for t in body["rows"].values())
    print(f"  {group:<15} {rows:>7,} row(s)   [{months}]")
prs = len(meta.get("pull_requests", {}))
if prs:
    print(f"  {'pull requests':<15} {prs:>7,} file(s)")
PY

step 'What changed in the branch'
git -C "${ROOT}" status --short -- 'dev/bench/ci-pulse/views' || true

# ---------------------------------------------------------------------------
# Commit, if asked
# ---------------------------------------------------------------------------

if [ "${DO_COMMIT}" = 'yes' ]; then
  step 'Committing'
  git -C "${ROOT}" add -- 'dev/bench/ci-pulse/views'
  # Only the views. Anything else that happens to be dirty is not ours to record.
  if git -C "${ROOT}" diff --cached --quiet; then
    say 'The views are unchanged; nothing to commit.'
  else
    git -C "${ROOT}" commit -m 'ci-metrics: refresh dashboard data'
    say 'Committed. Nothing has been pushed.'
  fi
else
  step 'Not committed'
  say 'The views are in the working tree only. When the page looks right:'
  say ''
  say '  git add dev/bench/ci-pulse/views'
  say '  git status --short        # confirm nothing else is staged'
  say "  git commit -m 'ci-metrics: refresh dashboard data'"
fi

# ---------------------------------------------------------------------------
# Look at it
# ---------------------------------------------------------------------------

if [ "${DO_SERVE}" = 'yes' ]; then
  step 'Serving'
  say "Open http://localhost:${PORT}/#ci   (Ctrl-C to stop)"
  say ''
  cd "${ROOT}/dev/bench" && exec python3 -m http.server "${PORT}"
else
  step 'To look at it'
  say '  cd dev/bench && python3 -m http.server 8000'
  say '  open http://localhost:8000/#ci'
  say ''
  say 'Note: the dashboard still reads its baked-in constants, so these files are'
  say 'not on screen yet. That is step 2 of the work order in GOING-LIVE.md.'
fi
