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

# Sets up, updates and removes the CI Pulse collector PoC in GCP. Every step can be run again.
#
#   cloudrun/deploy.sh infra      bucket, service account, bucket IAM, Artifact Registry repo
#   cloudrun/deploy.sh image      build the image from tools/ci_metrics and push it
#   cloudrun/deploy.sh job        create or update the Cloud Run job with the image just pushed
#   cloudrun/deploy.sh schedule   create or update the Cloud Scheduler job (every 4 hours)
#   cloudrun/deploy.sh alerts     email channel and the two alert policies
#   cloudrun/deploy.sh run        start one execution now and wait for it
#   cloudrun/deploy.sh status     show what exists
#   cloudrun/deploy.sh teardown   delete all of the above (asks first)
#
# Run it from tools/ci_metrics. Every name starts with "mesa-" so the PoC is easy to find.
#
# Settings can be overridden only through CI_PULSE_* variables (CI_PULSE_PROJECT, CI_PULSE_REGION,
# CI_PULSE_BUCKET, ...). Generic names such as REGION or PROJECT are often exported by a shell
# profile for other work, so they are deliberately ignored.

set -euo pipefail

PROJECT="${CI_PULSE_PROJECT:-tpu-prod-env-multipod}"
# Looked up from gcloud when it is needed (see project_number), so it is not written here.
PROJECT_NUMBER="${CI_PULSE_PROJECT_NUMBER:-}"
REGION="${CI_PULSE_REGION:-us-central1}"
BUCKET="${CI_PULSE_BUCKET:-mesa-ci-pulse}"
SA_NAME="${CI_PULSE_SA_NAME:-mesa-ci-pulse-collector}"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
AR_REPO="${CI_PULSE_AR_REPO:-mesa-ci-pulse}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/collector"
JOB="${CI_PULSE_JOB:-mesa-ci-pulse-collector}"
# Every 4 hours at 01:20, 05:20, ... UTC: MaxText's scheduled CI runs at 0 */4 * * *, starts a few minutes
# late and takes about 40-55 minutes, so this collects each scheduled run soon after it has finished.
SCHEDULER_JOB="${CI_PULSE_SCHEDULER_JOB:-mesa-ci-pulse-collector-4h}"
SCHEDULE="${CI_PULSE_SCHEDULE:-20 1-23/4 * * *}"
# The scheduler needs run.jobs.run on the job. Granting roles/run.invoker on the job needs
# run.jobs.setIamPolicy, which only a project admin has, so until one runs
#   gcloud run jobs add-iam-policy-binding "${JOB}" --region "${REGION}" \
#     --member "serviceAccount:${SA_EMAIL}" --role roles/run.invoker --project "${PROJECT}"
# the scheduler runs as the project's default compute service account, which already holds
# roles/run.invoker on the project. After the grant:
#   CI_PULSE_SCHEDULER_SA="${SA_EMAIL}" deploy.sh schedule
SCHEDULER_SA="${CI_PULSE_SCHEDULER_SA:-}"
# Who gets the alert emails, and the "Owner:" in descriptions: the account gcloud is signed in
# with, unless CI_PULSE_ALERT_EMAIL says otherwise.
ALERT_EMAIL="${CI_PULSE_ALERT_EMAIL:-$(gcloud config get-value account 2>/dev/null || true)}"
LABELS="app=ci-pulse,owner=mesakh,purpose=poc"
# No GitHub token: 60 requests an hour, no artifacts (401), so the tick needs these options.
TICK_ARGS="${CI_PULSE_TICK_ARGS:---skip-tests --rate-floor 15 --max-wait 30 --merge-checks list --defer-running --backfill-days 3}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT="$(dirname "${HERE}")"
STATE_DIR="${CI_PULSE_STATE_DIR:-${HOME}/.cache/mesa-ci-pulse}"
mkdir -p "${STATE_DIR}"

gc() { gcloud --project "${PROJECT}" --quiet "$@"; }

# The project number, for the default compute service account's address.
project_number() {
  if [[ -z "${PROJECT_NUMBER}" ]]; then
    PROJECT_NUMBER="$(gc projects describe "${PROJECT}" --format="value(projectNumber)")"
  fi
  echo "${PROJECT_NUMBER}"
}

# One global pip setting of this machine (from any pip.conf), or nothing.
pip_setting() { python3 -m pip config list 2>/dev/null | sed -n "s/^global\.$1='\(.*\)'\$/\1/p" | head -n 1; }

infra() {
  if ! gc storage buckets describe "gs://${BUCKET}" --format="value(name)" >/dev/null 2>&1; then
    gc storage buckets create "gs://${BUCKET}" --location "${REGION}" --uniform-bucket-level-access \
      --public-access-prevention --default-storage-class STANDARD
  fi
  gc storage buckets update "gs://${BUCKET}" --update-labels "${LABELS}"

  if ! gc iam service-accounts describe "${SA_EMAIL}" --format="value(email)" >/dev/null 2>&1; then
    gc iam service-accounts create "${SA_NAME}" --display-name "mesa CI Pulse collector (PoC)" \
      --description "Runs the ${JOB} Cloud Run job. Owner: ${ALERT_EMAIL}. Access: objectAdmin on gs://${BUCKET} only."
    sleep 10
  fi
  gc storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA_EMAIL}" \
    --role roles/storage.objectAdmin --format=none

  if ! gc artifacts repositories describe "${AR_REPO}" --location "${REGION}" --format="value(name)" >/dev/null 2>&1; then
    gc artifacts repositories create "${AR_REPO}" --repository-format docker --location "${REGION}" \
      --description "mesa CI Pulse collector images (PoC). Owner: ${ALERT_EMAIL}" --labels "${LABELS}"
  fi
}

image() {
  local tag index trusted
  tag="$(git -C "${CONTEXT}" rev-parse --short=12 HEAD)"
  if [[ -n "$(git -C "${CONTEXT}" status --porcelain -- .)" ]]; then
    echo "error: tools/ci_metrics has uncommitted changes; commit them so the image tag names the code" >&2
    exit 1
  fi
  # The build machine's pip index (a corp mirror on a Cloudtop), else PyPI. The hashes decide either way.
  index="$(pip_setting index-url)"
  trusted="$(pip_setting trusted-host)"
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null
  # Host networking: on a Cloudtop the default bridge has no route out, so pip could not install.
  docker build --network host --build-arg PIP_INDEX_URL="${index:-https://pypi.org/simple}" \
    --build-arg PIP_TRUSTED_HOST="${trusted:-pypi.org}" -f "${HERE}/Dockerfile" -t "${IMAGE}:${tag}" "${CONTEXT}"
  docker push "${IMAGE}:${tag}"
  docker image inspect "${IMAGE}:${tag}" --format '{{index .RepoDigests 0}}' | tee "${STATE_DIR}/image"
}

job() {
  local image_ref env_file
  image_ref="$(cat "${STATE_DIR}/image")"
  env_file="$(mktemp)"
  trap 'rm -f "${env_file}"' RETURN
  cat >"${env_file}" <<EOF
CI_PULSE_BUCKET: "${BUCKET}"
CI_PULSE_TICK_ARGS: "${TICK_ARGS}"
CI_PULSE_TICK_TIMEOUT_SECONDS: "600"
CI_PULSE_LOCK_STALE_SECONDS: "1800"
EOF
  # Every execution downloads the whole store into /tmp, which Cloud Run keeps in memory: about
  # 550 MB since the store was seeded with the full history on 2026-10-21, plus up to about 1.2 GB
  # while the tick rebuilds one month's views. Cloud Run jobs allow at most 4 GiB with 1 vCPU, so
  # 8 GiB needs 2 vCPUs.
  gc run jobs deploy "${JOB}" --region "${REGION}" --image "${image_ref}" \
    --service-account "${SA_EMAIL}" --tasks 1 --parallelism 1 --max-retries 0 --task-timeout 15m \
    --cpu 2 --memory 8Gi --env-vars-file "${env_file}" --labels "${LABELS}"
}

schedule() {
  local uri="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"
  local scheduler_sa="${SCHEDULER_SA}"
  if [[ -z "${scheduler_sa}" ]]; then
    scheduler_sa="$(project_number)-compute@developer.gserviceaccount.com"
  fi
  local verb=create
  if gc scheduler jobs describe "${SCHEDULER_JOB}" --location "${REGION}" --format="value(name)" >/dev/null 2>&1; then
    verb=update
  fi
  gc scheduler jobs "${verb}" http "${SCHEDULER_JOB}" --location "${REGION}" --schedule "${SCHEDULE}" \
    --time-zone "Etc/UTC" --uri "${uri}" --http-method POST \
    --oauth-service-account-email "${scheduler_sa}" \
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" --attempt-deadline 180s \
    --description "Starts ${JOB} on '${SCHEDULE}' UTC, after MaxText's 4-hourly scheduled CI (CI Pulse PoC). Owner: ${ALERT_EMAIL}"
}

run_now() {
  gc run jobs execute "${JOB}" --region "${REGION}" --wait
}

status() {
  gc storage ls --long "gs://${BUCKET}/**" 2>/dev/null | tail -n 25 || true
  gc run jobs describe "${JOB}" --region "${REGION}" \
    --format="yaml(metadata.name,spec.template.spec.template.spec.containers[0].image)" || true
  gc run jobs executions list --job "${JOB}" --region "${REGION}" --limit 5 || true
  gc scheduler jobs describe "${SCHEDULER_JOB}" --location "${REGION}" \
    --format="yaml(name,schedule,state,lastAttemptTime,status)" || true
}

teardown() {
  read -r -p "Delete ${SCHEDULER_JOB}, ${JOB}, the alert policies, ${AR_REPO}, gs://${BUCKET} and ${SA_EMAIL}? [y/N] " answer
  [[ "${answer}" == "y" ]] || exit 1
  gc scheduler jobs delete "${SCHEDULER_JOB}" --location "${REGION}" || true
  # The first schedule ran hourly under this name; delete it too if it is still there.
  gc scheduler jobs delete mesa-ci-pulse-collector-hourly --location "${REGION}" 2>/dev/null || true
  gc run jobs delete "${JOB}" --region "${REGION}" || true
  for policy in $(gc monitoring policies list --filter='displayName~"^mesa-ci-pulse"' --format="value(name)" 2>/dev/null); do
    gc monitoring policies delete "${policy}" || true
  done
  for channel in $(gc beta monitoring channels list --filter='displayName="mesa-ci-pulse email"' --format="value(name)" 2>/dev/null); do
    gc beta monitoring channels delete "${channel}" || true
  done
  gc artifacts repositories delete "${AR_REPO}" --location "${REGION}" || true
  gc storage rm --recursive "gs://${BUCKET}" || true
  gc iam service-accounts delete "${SA_EMAIL}" || true
}

case "${1:-}" in
  infra | image | job | schedule | alerts | run | status | teardown)
    echo "deploy.sh $1: project=${PROJECT} region=${REGION} bucket=gs://${BUCKET} job=${JOB}" >&2
    ;;
esac

case "${1:-}" in
  infra) infra ;;
  image) image ;;
  job) job ;;
  schedule) schedule ;;
  alerts)
    [[ -n "${ALERT_EMAIL}" ]] || { echo "Set CI_PULSE_ALERT_EMAIL, or sign in to gcloud, so the alerts have an address." >&2; exit 2; }
    CI_PULSE_PROJECT="${PROJECT}" CI_PULSE_JOB="${JOB}" CI_PULSE_ALERT_EMAIL="${ALERT_EMAIL}" CI_PULSE_SCHEDULER_JOB="${SCHEDULER_JOB}" "${HERE}/alerts.sh" ;;
  run) run_now ;;
  status) status ;;
  teardown) teardown ;;
  *)
    sed -n '17,27p' "${BASH_SOURCE[0]}"
    exit 2
    ;;
esac
