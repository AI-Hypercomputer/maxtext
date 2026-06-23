#!/bin/bash

if [ "${JOB_COMPLETION_INDEX:-0}" = "0" ] || [ "${TPU_WORKER_ID:-0}" = "0" ]; then
  IS_LEADER=true
else
  IS_LEADER=false
fi

echo "Starting vLLM server (Leader: $IS_LEADER)..."
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
MODEL_IMPL_TYPE=vllm \
TPU_BACKEND_TYPE=jax \
vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --data-parallel-size 16 \
  --tensor-parallel-size 4 \
  --max-model-len 24576 \
  --max-num-batched-tokens 24576 \
  --max-num-seqs 480 \
  --gpu-memory-utilization 0.4 \
  --no-enable-prefix-caching \
  --trust-remote-code &
SERVER_PID=$!

echo "Waiting for server to become healthy..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server is healthy!"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process died."
    exit 1
  fi
  sleep 10
done

if [ "$IS_LEADER" = "true" ]; then
  echo "Running benchmark client..."
  vllm bench serve \
    --model Qwen/Qwen3-0.6B \
    --host localhost \
    --port 8000 \
    --dataset-name random \
    --random-input-len 16384 \
    --random-output-len 8192 \
    --num-prompts 2 \
    --request-rate inf \
    --ignore-eos \
    --profile \
    --trust-remote-code
  EXIT_CODE=$?
  kill "${SERVER_PID}" || true
else
  echo "Waiting for server process to finish..."
  wait "${SERVER_PID}"
  EXIT_CODE=$?
fi

exit $EXIT_CODE
