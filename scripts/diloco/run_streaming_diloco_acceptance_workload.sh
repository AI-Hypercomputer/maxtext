#!/usr/bin/env bash
#
# Runs inside the Pathways controller container. Prefer invoking this through
# run_streaming_diloco_gcloud.sh instead of calling it directly.

set -Eeuo pipefail

PHASE="${DILOCO_TEST_PHASE:?DILOCO_TEST_PHASE is required}"
EXPECTED_SLICES="${EXPECTED_SLICES:?EXPECTED_SLICES is required}"
MAXTEXT_RUN_NAME="${MAXTEXT_RUN_NAME:?MAXTEXT_RUN_NAME is required}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}"
MEMORY_INTERVAL_SECONDS="${DILOCO_MEMORY_INTERVAL_SECONDS:-5}"
PREFLIGHT_TIMEOUT_SECONDS="${DILOCO_PREFLIGHT_TIMEOUT_SECONDS:-300}"
PHASE_TIMEOUT_SECONDS="${DILOCO_PHASE_TIMEOUT_SECONDS:-14400}"
TIMEOUT_GRACE_SECONDS="${DILOCO_TIMEOUT_GRACE_SECONDS:-60}"
SOURCE_REVISION="${MAXTEXT_SOURCE_REVISION:-unknown}"
LOG_FILE="/tmp/diloco-acceptance-${PHASE}.log"
PIPE_DIRECTORY=""
LOG_PIPE=""

export PYTHONUNBUFFERED=1
export PYTHONPATH="/app/src:${PYTHONPATH:-}"

case "${PHASE}" in
  layout|tiny|tiny-save|tiny-resume|qwen8b) ;;
  *)
    echo "Unknown DILOCO_TEST_PHASE=${PHASE}" >&2
    exit 2
    ;;
esac

if [[ "${PHASE}" != "layout" && ! "${BASE_OUTPUT_DIRECTORY}" =~ ^gs://[^/]+(/.*)?$ ]]; then
  echo "BASE_OUTPUT_DIRECTORY must contain a GCS bucket for ${PHASE}" >&2
  exit 2
fi
if [[ ! "${EXPECTED_SLICES}" =~ ^[0-9]+$ ]] || (( EXPECTED_SLICES < 1 )); then
  echo "EXPECTED_SLICES must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MEMORY_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DILOCO_MEMORY_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi
for timeout_setting in \
  "DILOCO_PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS}" \
  "DILOCO_PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS}" \
  "DILOCO_TIMEOUT_GRACE_SECONDS=${TIMEOUT_GRACE_SECONDS}"; do
  timeout_value="${timeout_setting#*=}"
  if [[ ! "${timeout_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${timeout_setting%%=*} must be a positive integer" >&2
    exit 2
  fi
done
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for bounded Pathways acceptance runs" >&2
  exit 2
fi

echo "DILOCO_ACCEPTANCE phase=${PHASE} run=${MAXTEXT_RUN_NAME} source=${SOURCE_REVISION}"

run_pathways_preflight() {
  timeout \
    --signal=TERM \
    --kill-after="${TIMEOUT_GRACE_SECONDS}s" \
    "${PREFLIGHT_TIMEOUT_SECONDS}s" \
    python3 - "${EXPECTED_SLICES}" <<'PY'
import importlib.metadata
import inspect
import sys

import pathwaysutils

pathwaysutils.initialize()

import jax
from jax.experimental import colocated_python
from pathwaysutils.experimental import reshard as pathways_reshard

expected_slices = int(sys.argv[1])
all_default_devices = list(jax.devices())
accelerators = [device for device in all_default_devices if device.platform != "cpu"]
if not accelerators:
  raise RuntimeError(f"Pathways preflight found no accelerator devices: {all_default_devices}")

slice_indices = {int(device.slice_index) for device in accelerators}
if len(slice_indices) != expected_slices:
  raise RuntimeError(
      f"Expected {expected_slices} Pathways TPU slices, found {len(slice_indices)}: "
      f"{sorted(slice_indices)}"
  )
if not pathwaysutils.is_pathways_backend_used():
  raise RuntimeError("pathwaysutils does not detect an active Pathways backend")

colocated_cpus = list(colocated_python.colocated_cpu_devices(tuple(accelerators)))
if len(colocated_cpus) != len(accelerators):
  raise RuntimeError(
      f"Expected one colocated CPU per TPU device, got {len(colocated_cpus)} "
      f"for {len(accelerators)} accelerators"
  )

incompatible = [
    (tpu, cpu)
    for tpu, cpu in zip(accelerators, colocated_cpus)
    if cpu.platform != "cpu" or cpu.client is not tpu.client
]
if incompatible:
  raise RuntimeError(
      "Colocated CPU devices are not CPU devices on the TPU Pathways IFRT client. "
      f"Incompatible pairs: {incompatible}"
  )

reshard_parameters = inspect.signature(pathways_reshard.reshard).parameters
required_reshard_parameters = {"donate", "may_alias", "cache_resharding_plans"}
missing_parameters = required_reshard_parameters - set(reshard_parameters)
if missing_parameters:
  raise RuntimeError(
      "Installed pathwaysutils.experimental.reshard API is incompatible; "
      f"missing parameters: {sorted(missing_parameters)}"
  )

def package_version(name):
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return "unknown"

print(
    "DILOCO_PREFLIGHT "
    f"jax={jax.__version__} "
    f"jaxlib={package_version('jaxlib')} "
    f"pathwaysutils={package_version('pathwaysutils')} "
    f"accelerators={len(accelerators)} "
    f"colocated_cpus={len(colocated_cpus)} "
    f"slices={sorted(slice_indices)}"
)
print(f"DILOCO_PREFLIGHT accelerator_devices={accelerators}")
print(f"DILOCO_PREFLIGHT colocated_cpu_devices={colocated_cpus}")
print(f"DILOCO_PREFLIGHT reshard_signature={inspect.signature(pathways_reshard.reshard)}")
PY
}

emit_memory_sample() {
  local process_id="$1"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "DILOCO_MEM timestamp=${timestamp} phase=${PHASE} train_pid=${process_id}"

  if [[ -r "/proc/${process_id}/status" ]]; then
    awk '
      /^(VmRSS|VmHWM|VmSize|Threads):/ {
        printf "DILOCO_MEM process_%s=%s%s\n", $1, $2, $3
      }
    ' "/proc/${process_id}/status" 2>/dev/null || true
  fi

  if [[ -r /sys/fs/cgroup/memory.current ]]; then
    echo "DILOCO_MEM cgroup_version=2"
    for metric in memory.current memory.peak memory.max memory.swap.current; do
      if [[ -r "/sys/fs/cgroup/${metric}" ]]; then
        echo "DILOCO_MEM cgroup_${metric}=$(<"/sys/fs/cgroup/${metric}")"
      fi
    done
    if [[ -r /sys/fs/cgroup/memory.events ]]; then
      sed 's/^/DILOCO_MEM cgroup_event_/' /sys/fs/cgroup/memory.events
    fi
  elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    echo "DILOCO_MEM cgroup_version=1"
    for metric in memory.usage_in_bytes memory.max_usage_in_bytes memory.limit_in_bytes memory.failcnt; do
      if [[ -r "/sys/fs/cgroup/memory/${metric}" ]]; then
        echo "DILOCO_MEM cgroup_${metric}=$(<"/sys/fs/cgroup/memory/${metric}")"
      fi
    done
  fi

  {
    ps -eo pid,ppid,rss,vsz,comm --sort=-rss | head -n 8
  } 2>/dev/null | sed 's/^/DILOCO_MEM top_rss /' || true
}

memory_sampler() {
  local process_id="$1"
  while kill -0 "${process_id}" 2>/dev/null; do
    emit_memory_sample "${process_id}"
    sleep "${MEMORY_INTERVAL_SECONDS}"
  done
  emit_memory_sample "${process_id}"
}

require_log_marker() {
  local marker="$1"
  if ! grep -Fq -- "${marker}" "${LOG_FILE}"; then
    echo "DILOCO_ACCEPTANCE missing success marker: ${marker}" >&2
    return 1
  fi
}

verify_workload_log() {
  local failure_pattern
  failure_pattern='RESOURCE_EXHAUSTED|OOMKilled|out of memory|INVALID_ARGUMENT.*layout|Buffer has been deleted|deleted buffer|TransportProtocolError|Incompatible CPU devices|Traceback \(most recent call last\)'

  if grep -Eiq -- "${failure_pattern}" "${LOG_FILE}"; then
    echo "DILOCO_ACCEPTANCE detected a failure signature:" >&2
    grep -Ein -- "${failure_pattern}" "${LOG_FILE}" | tail -n 20 >&2
    return 1
  fi

  case "${PHASE}" in
    layout)
      require_log_marker "1 passed"
      ;;
    tiny|tiny-save)
      require_log_marker "Syncer: Step 3 sync finished"
      require_log_marker "Syncer: Step 6 sync finished"
      require_log_marker "Finished run_threaded_diloco"
      ;;
    tiny-resume)
      require_log_marker "Syncer: every learner checkpoint is aligned at step 6"
      require_log_marker "Syncer: Step 9 sync finished"
      require_log_marker "Finished run_threaded_diloco"
      ;;
    qwen8b)
      require_log_marker "Syncer: Step 37 sync finished"
      require_log_marker "Syncer: Step 74 sync finished"
      require_log_marker "Finished run_threaded_diloco"
      ;;
  esac
}

run_pathways_preflight

workload_command=()
if [[ "${PHASE}" == "layout" ]]; then
  if (( $# != 0 )); then
    echo "Extra MaxText arguments are not valid for the layout phase" >&2
    exit 2
  fi
  workload_command=(
    python3
    -c
    "import pathwaysutils, pytest, sys; pathwaysutils.initialize(); sys.exit(pytest.main(['/app/tests/unit/pathways_null_layout_repro_test.py', '-v', '-s']))"
  )
  export RUN_PATHWAYS_REPRO=1
else
  for argument in "$@"; do
    if [[ ! "${argument}" =~ ^[A-Za-z0-9_.-]+=.+$ ]]; then
      echo "Extra MaxText arguments must be non-empty key=value tokens; got: ${argument}" >&2
      exit 2
    fi
    case "${argument%%=*}" in
      per_device_batch_size|max_target_length) ;;
      *)
        echo "Acceptance override is not allowed: ${argument%%=*}" >&2
        exit 2
        ;;
    esac
  done

  common_args=(
    /app/src/maxtext/configs/base.yml
    "run_name=${MAXTEXT_RUN_NAME}"
    "base_output_directory=${BASE_OUTPUT_DIRECTORY}"
    "num_slices=${EXPECTED_SLICES}"
    dataset_type=synthetic
    reuse_example_batch=1
    packing=false
    dtype=bfloat16
    weight_dtype=bfloat16
    enable_diloco=true
    enable_streaming_diloco=true
    enable_non_spmd_diloco=true
    enable_single_controller=true
    pure_nnx=true
    ici_diloco_parallelism=1
    "dcn_diloco_parallelism=${EXPECTED_SLICES}"
    dcn_data_parallelism=1
    ici_fsdp_parallelism=-1
    use_sequential_layers=false
    communication_overlapping_alpha=0.0
    diloco_outer_lr=0.1
    diloco_outer_momentum=0.9
    eval_interval=-1
    enable_checkpointing=false
    save_checkpoint_on_completion=false
    async_checkpointing=false
    enable_continuous_checkpointing=false
    enable_autocheckpoint=false
    enable_emergency_checkpoint=false
    enable_multi_tier_checkpointing=false
    colocated_python_checkpointing=false
    checkpoint_storage_concurrent_gb=8
    max_num_checkpoints_to_keep=3
    upload_all_profiler_results=false
    enable_goodput_recording=false
    monitor_goodput=false
    monitor_step_time_deviation=false
    enable_gcp_goodput_metrics=false
    enable_gcp_step_deviation_metrics=false
    report_heartbeat_metric_for_gcp_monitoring=false
    report_performance_metric_for_gcp_monitoring=false
    enable_tensorboard=false
    use_vertex_tensorboard=false
    save_config_to_gcs=false
    gcs_metrics=false
    log_period=1
    elastic_enabled=false
  )

  case "${PHASE}" in
    tiny|tiny-save|tiny-resume)
      phase_args=(
        model_name=qwen3-0.6b
        override_model_config=true
        base_emb_dim=128
        base_num_query_heads=4
        base_num_kv_heads=2
        base_mlp_dim=256
        base_num_decoder_layers=2
        head_dim=32
        vocab_size=1024
        logits_via_embedding=true
        per_device_batch_size=1
        max_target_length=128
        num_diloco_fragments=2
        diloco_sync_period=3
        num_communication_overlapping_steps=1
        steps=8
      )
      ;;
    qwen8b)
      phase_args=(
        model_name=qwen3-8b
        per_device_batch_size=1
        max_target_length=512
        num_diloco_fragments=36
        diloco_sync_period=36
        num_communication_overlapping_steps=2
        steps=80
      )
      ;;
  esac

  if [[ "${PHASE}" == "tiny-save" ]]; then
    phase_args+=(
      steps=6
      enable_checkpointing=true
      save_checkpoint_on_completion=true
      checkpoint_period=3
      checkpoint_storage_use_ocdbt=false
      checkpoint_storage_use_zarr3=false
    )
  elif [[ "${PHASE}" == "tiny-resume" ]]; then
    phase_args+=(
      steps=10
      enable_checkpointing=true
      save_checkpoint_on_completion=true
      checkpoint_period=3
      checkpoint_storage_use_ocdbt=false
      checkpoint_storage_use_zarr3=false
    )
  fi

  workload_command=(
    python3
    -m
    maxtext.trainers.pre_train.train
    "${common_args[@]}"
    "${phase_args[@]}"
    "$@"
  )
fi

printf 'DILOCO_ACCEPTANCE command='
printf '%q ' "${workload_command[@]}"
printf '\n'

: >"${LOG_FILE}"
train_pid=""
sampler_pid=""
tee_pid=""
PIPE_DIRECTORY="$(mktemp -d /tmp/diloco-acceptance-pipe.XXXXXX)"
LOG_PIPE="${PIPE_DIRECTORY}/workload-output"

cleanup() {
  if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi
  if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
    kill -TERM "${train_pid}" 2>/dev/null || true
    wait "${train_pid}" 2>/dev/null || true
  fi
  if [[ -n "${tee_pid}" ]] && kill -0 "${tee_pid}" 2>/dev/null; then
    kill -TERM "${tee_pid}" 2>/dev/null || true
    wait "${tee_pid}" 2>/dev/null || true
  fi
  if [[ -n "${LOG_PIPE}" ]]; then
    rm -f -- "${LOG_PIPE}"
  fi
  if [[ -n "${PIPE_DIRECTORY}" ]]; then
    rmdir -- "${PIPE_DIRECTORY}" 2>/dev/null || true
  fi
}

forward_signal() {
  if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
    kill -TERM "${train_pid}" 2>/dev/null || true
  fi
  exit 143
}

trap cleanup EXIT
trap forward_signal INT TERM

mkfifo "${LOG_PIPE}"
tee "${LOG_FILE}" <"${LOG_PIPE}" &
tee_pid="$!"
timeout \
  --signal=TERM \
  --kill-after="${TIMEOUT_GRACE_SECONDS}s" \
  "${PHASE_TIMEOUT_SECONDS}s" \
  "${workload_command[@]}" >"${LOG_PIPE}" 2>&1 &
train_pid="$!"
memory_sampler "${train_pid}" &
sampler_pid="$!"

set +e
wait "${train_pid}"
workload_status="$?"
train_pid=""
set -e

if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
fi
sampler_pid=""
set +e
wait "${tee_pid}"
tee_status="$?"
tee_pid=""
set -e

if (( workload_status != 0 )); then
  echo "DILOCO_ACCEPTANCE_RESULT phase=${PHASE} status=FAIL exit_code=${workload_status}" >&2
  exit "${workload_status}"
fi
if (( tee_status != 0 )); then
  echo "DILOCO_ACCEPTANCE_RESULT phase=${PHASE} status=FAIL tee_exit_code=${tee_status}" >&2
  exit "${tee_status}"
fi
if ! verify_workload_log; then
  echo "DILOCO_ACCEPTANCE_RESULT phase=${PHASE} status=FAIL verification=log-markers" >&2
  exit 1
fi

echo "DILOCO_ACCEPTANCE_RESULT phase=${PHASE} status=PASS"
