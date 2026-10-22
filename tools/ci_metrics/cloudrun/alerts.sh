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

# Creates or updates the PoC's email alerts (run it through `deploy.sh alerts`):
#   "mesa-ci-pulse-collector failed"       an execution ended with a non-zero exit
#   "mesa-ci-pulse-collector silent 5h"    no execution succeeded for five hours (one missed 4-hourly run)
# Both read the Cloud Run job metric run.googleapis.com/job/completed_execution_count.

set -euo pipefail

# Only CI_PULSE_* variables are read; generic names like PROJECT are often set for other work.
PROJECT="${CI_PULSE_PROJECT:-tpu-prod-env-multipod}"
JOB="${CI_PULSE_JOB:-mesa-ci-pulse-collector}"
# The account gcloud is signed in with, unless CI_PULSE_ALERT_EMAIL says otherwise.
ALERT_EMAIL="${CI_PULSE_ALERT_EMAIL:-$(gcloud config get-value account 2>/dev/null || true)}"
[[ -n "${ALERT_EMAIL}" ]] || { echo "Set CI_PULSE_ALERT_EMAIL, or sign in to gcloud, so the alerts have an address." >&2; exit 2; }
SCHEDULER_JOB="${CI_PULSE_SCHEDULER_JOB:-mesa-ci-pulse-collector-4h}"
# Longer than the 4-hour schedule plus a margin, so a single missed or failed run is noticed.
SILENT_HOURS="${CI_PULSE_SILENT_HOURS:-5}"
CHANNEL_NAME="mesa-ci-pulse email"

gc() { gcloud --project "${PROJECT}" --quiet "$@"; }

channel="$(gc beta monitoring channels list --filter="displayName=\"${CHANNEL_NAME}\"" --format="value(name)" | head -n 1)"
if [[ -z "${channel}" ]]; then
  channel="$(gc beta monitoring channels create --display-name="${CHANNEL_NAME}" --type=email \
    --channel-labels="email_address=${ALERT_EMAIL}" --user-labels="app=ci-pulse,owner=mesakh,purpose=poc" \
    --format="value(name)")"
fi
echo "channel: ${channel}"

filter_for() {
  printf 'resource.type = "cloud_run_job" AND resource.labels.job_name = "%s" AND metric.type = "run.googleapis.com/job/completed_execution_count" AND metric.labels.result = "%s"' "${JOB}" "$1"
}

policy_file="$(mktemp)"
trap 'rm -f "${policy_file}"' EXIT

apply_policy() {
  # Found by the start of its name, so a new silent window updates the same policy in place.
  local prefix="$1" existing
  existing="$(gc monitoring policies list --filter="displayName~\"^${prefix}\"" --format="value(name)" | head -n 1)"
  if [[ -z "${existing}" ]]; then
    gc monitoring policies create --policy-from-file="${policy_file}" --format="value(name)"
  else
    gc monitoring policies update "${existing}" --policy-from-file="${policy_file}" --format="value(name)"
  fi
}

python3 - "${policy_file}" "${channel}" "$(filter_for failed)" "${JOB}" <<'EOF'
import json, sys
path, channel, metric_filter, job = sys.argv[1:]
json.dump({
    "displayName": f"{job} failed",
    "documentation": {
        "mimeType": "text/markdown",
        "content": (
            f"An execution of the Cloud Run job `{job}` (CI Pulse PoC) ended with a non-zero exit code. "
            "Its last log line is a JSON summary: `exit_code`, `error`, `tick_summary`, the tick's own "
            "`tick_budget`, `tick_stopped_early` and `tick_problems`. Exit 1: the tick lost some data, often a "
            "GitHub 5xx, and those runs are read again on the next run. 3: the bucket could not be read or "
            "written. 4: the tick ran past 10 minutes. 5: the tick was killed. 75: another execution held the lock."
        ),
    },
    "combiner": "OR",
    "conditions": [{
        "displayName": f"A {job} execution failed",
        "conditionThreshold": {
            "filter": metric_filter,
            "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM", "crossSeriesReducer": "REDUCE_SUM"}],
            "comparison": "COMPARISON_GT",
            "thresholdValue": 0,
            "duration": "0s",
            "trigger": {"count": 1},
        },
    }],
    "alertStrategy": {"autoClose": "1800s"},
    "notificationChannels": [channel],
    "userLabels": {"app": "ci-pulse", "owner": "mesakh", "purpose": "poc"},
    "severity": "WARNING",
}, open(path, "w"), indent=2)
EOF
echo "failed policy: $(apply_policy "${JOB} failed")"

python3 - "${policy_file}" "${channel}" "$(filter_for succeeded)" "${JOB}" "${SILENT_HOURS}" "${SCHEDULER_JOB}" <<'EOF'
import json, sys
path, channel, metric_filter, job, hours, scheduler_job = sys.argv[1:]
hours = int(hours)
json.dump({
    "displayName": f"{job} silent {hours}h",
    "documentation": {
        "mimeType": "text/markdown",
        "content": (
            f"No execution of the Cloud Run job `{job}` (CI Pulse PoC) succeeded in the last {hours} hours. "
            f"Check that the Cloud Scheduler job `{scheduler_job}` is enabled and that its "
            "recent attempts succeeded, then read the job's executions."
        ),
    },
    "combiner": "OR",
    "conditions": [{
        "displayName": f"No successful {job} execution in {hours} hours",
        "conditionAbsent": {
            "filter": metric_filter,
            "aggregations": [{"alignmentPeriod": "3600s", "perSeriesAligner": "ALIGN_SUM", "crossSeriesReducer": "REDUCE_SUM"}],
            "duration": f"{hours * 3600}s",
            "trigger": {"count": 1},
        },
    }],
    "notificationChannels": [channel],
    "userLabels": {"app": "ci-pulse", "owner": "mesakh", "purpose": "poc"},
    "severity": "WARNING",
}, open(path, "w"), indent=2)
EOF
echo "silent policy: $(apply_policy "${JOB} silent")"
