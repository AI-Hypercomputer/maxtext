#!/bin/bash
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Configure tpu-inference automatic phased profiling
export PHASED_PROFILING_DIR=${PHASED_PROFILING_DIR:="gs://runner-maxtext-logs/${WORKLOAD_NAME:-mohit-v5p-bench-repro}/tensorboard"}
export PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=${PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR:=50}
export PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP=${PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP:=10}
export PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=${PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD:=1000}

VLLM_ENABLE_V1_MULTIPROCESSING=0 MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=jax JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-model-len 24576 \
  --max-num-batched-tokens 24576 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.4 \
  --no-enable-prefix-caching \
  --trust-remote-code &
SERVER_PID=$!


# 2. Wait for the server to be healthy
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

if [ "$SERVER_HEALTHY" = "false" ]; then
  echo "Server failed to become healthy within timeout."
  exit 1
fi

# 3. Run the benchmark client
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
