# Runs MaxText on a single host with one process per GPU.
#
# The documented single-host GPU flow (docs/run_maxtext/run_maxtext_single_host_gpu.md)
# uses NNODES=1, i.e. one process driving every GPU, and gpu_multi_process_run.sh
# runs one process per *node*. Neither works with the TransformerEngine grouped
# GEMM backend (use_te_grouped_gemm=true): TE's v1 grouped GEMM copies group_sizes
# device-to-host inside its FFI, and that host sync deadlocks when a single
# process drives several devices — the blocked thread never reaches the collective
# rendezvous the other devices are waiting on.
#
# This script uses the one-process-per-GPU layout, which MaxText supports through
# SLURM_STEP_GPUS -> jax.distributed(local_device_ids=...) in
# max_utils.initialize_jax_for_gpu. Note that CUDA_VISIBLE_DEVICES cannot be used
# for this on ROCm: it actually hides the devices, so local_device_ids would point
# at a device the process can no longer see.
#
# Usage (from the repository root):
#   tools/orchestration/gpu_per_device_run.sh <config.yml> [maxtext args...]
#
# Example:
#   tools/orchestration/gpu_per_device_run.sh \
#     src/maxtext/configs/gpu/models/gpt-oss-20b.yml \
#     steps=10 sparse_matmul=true megablox=false use_te_grouped_gemm=true \
#     ici_expert_parallelism=8 ici_fsdp_parallelism=1 \
#     run_name=te_gmm_ep8 base_output_directory=/tmp/te_gmm_ep8_out
#
# Environment:
#   GPU_IDS               comma-separated device indices to run on, e.g. "4,5,6,7".
#                         Use this when some GPUs are busy; NUM_PROCS is then its length.
#   NUM_PROCS             number of processes/GPUs (default: all detected GPUs)
#   JAX_COORDINATOR_PORT  coordinator port (default: 12355)
#   LOG_DIR               directory for per-rank logs (default: a fresh mktemp dir)

set -u -o pipefail

cd "$(dirname "$0")/../.." || exit 1

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <config.yml> [maxtext args...]" >&2
  exit 1
fi

detect_gpus() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid --csv 2>/dev/null | grep -c '^card'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l
  else
    echo 0
  fi
}

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
  NUM_PROCS="${#gpu_ids[@]}"
else
  NUM_PROCS="${NUM_PROCS:-$(detect_gpus)}"
  gpu_ids=()
  for ((i = 0; i < NUM_PROCS; i++)); do gpu_ids+=("${i}"); done
fi
JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-12355}"
LOG_DIR="${LOG_DIR:-$(mktemp -d -t maxtext_per_device_XXXXXX)}"

if [[ "${NUM_PROCS}" -lt 1 ]]; then
  echo "No GPUs detected; set NUM_PROCS explicitly." >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"

# MaxText re-raises when the optional Google Cloud monitoring packages are absent.
# Its own decoupled mode substitutes stubs, so enable it when the deps are missing.
if [[ -z "${DECOUPLE_GCLOUD:-}" ]] && ! python3 -c "import ml_goodput_measurement" >/dev/null 2>&1; then
  echo "ml_goodput_measurement is not installed; setting DECOUPLE_GCLOUD=TRUE to use stubs."
  export DECOUPLE_GCLOUD=TRUE
fi

echo "Launching ${NUM_PROCS} processes on GPUs ${gpu_ids[*]}, one each. Logs: ${LOG_DIR}"

pids=()
for ((i = 0; i < NUM_PROCS; i++)); do
  JAX_COORDINATOR_IP=127.0.0.1 \
  JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT}" \
  NNODES="${NUM_PROCS}" \
  NODE_RANK="${i}" \
  SLURM_STEP_GPUS="${gpu_ids[$i]}" \
  PYTHONPATH="src:${PYTHONPATH:-}" \
    python3 -m maxtext.trainers.pre_train.train "$@" \
      >"${LOG_DIR}/rank${i}.log" 2>&1 &
  pids+=("$!")
done

status=0
for ((i = 0; i < NUM_PROCS; i++)); do
  if ! wait "${pids[$i]}"; then
    echo "rank ${i} FAILED (see ${LOG_DIR}/rank${i}.log)" >&2
    status=1
  fi
done

echo "--- rank 0 summary ---"
grep -E "completed step|number parameters|Total TFLOPs" "${LOG_DIR}/rank0.log" | tail -15

if [[ "${status}" -eq 0 ]]; then
  echo "All ${NUM_PROCS} ranks finished successfully."
else
  echo "Some ranks failed; logs kept in ${LOG_DIR}" >&2
fi
exit "${status}"
