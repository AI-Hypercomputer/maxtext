#!/bin/bash

# This script launches the serving phased profiling repro workload using XPK.
set -e

export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export CLUSTER_NAME="${CLUSTER_NAME:-mlperf-v5p}"
export ZONE="${ZONE:-europe-west4-a}"
export TPU_TYPE="${TPU_TYPE:-v5p-64}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-runner:trkw-2026-06-25-22-36-41}"

export WORKLOAD_NAME="moh-v5p-13574-repro-${RANDOM}"
export PHASED_PROFILING_DIR="gs://runner-maxtext-logs/${WORKLOAD_NAME}/tensorboard"

# Define the script executed inside the container
CONTAINER_COMMAND=$(cat << 'EOF'
echo XPK Start: $(date); _sigterm() { kill -SIGTERM $! 2>/dev/null; }; trap _sigterm SIGTERM;

(
cd /app
pip install -e . --no-deps
mkdir -p scripts

# Create script inside container
cat << 'INNER_EOF' > scripts/run_all.sh
#!/bin/bash
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Configure tpu-inference automatic phased profiling
# (PHASED_PROFILING_DIR is passed into the container via XPK --env)
export PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=50
export PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP=10
export PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=2000

# Install pathways-utils from head
pip install git+https://github.com/google/pathways-utils.git

VLLM_ENABLE_V1_MULTIPROCESSING=0 MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=jax JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --data-parallel-size 8 \
  --tensor-parallel-size 4 \
  --max-model-len 24576 \
  --max-num-batched-tokens 24576 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.4 \
  --no-enable-prefix-caching \
  --trust-remote-code &
SERVER_PID=$!

# Wait for the server to be healthy
echo "Waiting for server to become healthy..."
SERVER_HEALTHY=false
for i in $(seq 1 240); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server is healthy!"
    SERVER_HEALTHY=true
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process died."
    exit 1
  fi
  sleep 10
done

if [ "${SERVER_HEALTHY}" = "false" ]; then
  echo "Server failed to become healthy within timeout."
  exit 1
fi

# Run the benchmark client
vllm bench serve \
  --model Qwen/Qwen3-0.6B \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 8192 \
  --num-prompts 480 \
  --request-rate inf \
  --ignore-eos \
  --profile \
  --trust-remote-code
EXIT_CODE=$?
INNER_EOF

chmod +x scripts/run_all.sh
bash scripts/run_all.sh
) &
PID=$!
while kill -0 $PID 2>/dev/null; do sleep 5; done
wait $PID
EXIT_CODE=$?
echo XPK End: $(date)
echo EXIT_CODE=$EXIT_CODE
exit $EXIT_CODE
EOF
)

echo "Launching XPK workload ${WORKLOAD_NAME} on cluster ${CLUSTER_NAME}..."

/usr/local/google/home/mohitkhatwani/max_venv/bin/python3 \
  /usr/local/google/home/mohitkhatwani/xpk/xpk.py \
  workload create-pathways \
  --workload "${WORKLOAD_NAME}" \
  --cluster "${CLUSTER_NAME}" \
  --zone "${ZONE}" \
  --tpu-type "${TPU_TYPE}" \
  --docker-image "${DOCKER_IMAGE}" \
  --env PHASED_PROFILING_DIR="${PHASED_PROFILING_DIR}" \
  --command "${CONTAINER_COMMAND}"
