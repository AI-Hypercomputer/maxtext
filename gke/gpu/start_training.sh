#! /bin/bash
set -e
set -u
set -o pipefail

: "${NNODES:?Must set NNODES}"
: "${NODE_RANK:?Must set NODE_RANK}"
: "${JAX_COORDINATOR_PORT:?Must set JAX_COORDINATOR_PORT}"
: "${JAX_COORDINATOR_ADDRESS:?Must set JAX_COORDINATOR_ADDRESS}"
: "${GPUS_PER_NODE:?Must set GPUS_PER_NODE}"
: "${RUN_NAME:?Must set RUN_NAME}"


export GPUS_PER_NODE=$GPUS_PER_NODE
export JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT
export JAX_COORDINATOR_ADDRESS=$JAX_COORDINATOR_ADDRESS
export JAX_NUM_PROCESSES=$((NNODES * GPUS_PER_NODE))

set_nccl_gpudirect_tcpx_specific_configuration() {
  if [[ "$USE_GPUDIRECT_TCPX" == "yes" ]]; then
    echo "Using GPUDirect-TCPX"
    export NCCL_CROSS_NIC=0
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    export NCCL_DEBUG=INFO
    export NCCL_NET_GDR_LEVEL=PIX
    export NCCL_P2P_PXN_LEVEL=0
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING,NET,VERSION
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tcpx/lib64"
    export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
    export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000
    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_P2P_NET_CHUNKSIZE=524288
    export NCCL_P2P_PCI_CHUNKSIZE=524288
    export NCCL_P2P_NVL_CHUNKSIZE=1048576
    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=1
    export NCCL_MAX_NCHANNELS=12
    export NCCL_MIN_NCHANNELS=12
    export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
    export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,124-139;eth2:22-35,124-139;eth3:74-87,178-191;eth4:74-87,178-191"
    export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
    export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
    export NCCL_NVLS_ENABLE=0
  else
    echo "NOT using TCPX"
  fi
}

echo "LD_LIBRARY_PATH ${LD_LIBRARY_PATH}"

set_nccl_gpudirect_tcpx_specific_configuration

wait_all_success_or_exit() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pids=("$@")
  while [[ ${#pids[@]} -ne 0 ]]; do
    all_success="true"
    for pid in "${pids[@]}"; do
      code=$(non_blocking_wait "$pid")
      if [[ $code -ne 127 ]]; then
        if [[ $code -ne 0 ]]; then
          echo "PID $pid failed with exit code $code"
          exit "$code"
        fi
      else
        all_success="false"
      fi
    done
    if [[ $all_success == "true" ]]; then
      echo "All pids succeeded"
      break
    fi
    sleep 5
  done
}
non_blocking_wait() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pid=$1
  local code=127 # special code to indicate not-finished
  if [[ ! -d "/proc/$pid" ]]; then
    wait "$pid"
    code=$?
  fi
  echo $code
}

resolve_coordinator_ip() {
  local lookup_attempt=1
  local max_coordinator_lookups=10
  local coordinator_found=false
  local coordinator_ip_address=""

  echo "Coordinator Address $JAX_COORDINATOR_ADDRESS"

  while [[ "$coordinator_found" = false && $lookup_attempt -le $max_coordinator_lookups ]]; do
    coordinator_ip_address=$(nslookup "$JAX_COORDINATOR_ADDRESS" 2>/dev/null | awk '/^Address: / { print $2 }' | head -n 1)
    if [[ -n "$coordinator_ip_address" ]]; then
      coordinator_found=true
      echo "Coordinator IP address: $coordinator_ip_address"
      export JAX_COORDINATOR_IP=$coordinator_ip_address
      return 0
    else
      echo "Failed to recognize coordinator address $JAX_COORDINATOR_ADDRESS on attempt $lookup_attempt, retrying..."
      ((lookup_attempt++))
      sleep 1
    fi
  done

  if [[ "$coordinator_found" = false ]]; then
    echo "Failed to resolve coordinator address after $max_coordinator_lookups attempts."
    return 1
  fi
}

# HLO dump
export XLA_FLAGS="--xla_dump_to=/tmp/xladump"

# Resolving coordinator IP
set +e
resolve_coordinator_ip
set -e

PIDS=()
for ((LOCAL_DEVICE_ID=0; LOCAL_DEVICE_ID <= $((GPUS_PER_NODE - 1)); LOCAL_DEVICE_ID++)); do
   PROCESS_ID=$(($GPUS_PER_NODE*$NODE_RANK + $LOCAL_DEVICE_ID))
   echo "LOCAL_DEVICE_ID=$LOCAL_DEVICE_ID PROCESS_ID=$PROCESS_ID  python MaxText/train.py MaxText/configs/base.yml hardware=gpu \
      run_name=${RUN_NAME}_$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs \
      dataset_path=gs://maxtext-dataset steps=30 enable_checkpointing=False \
      base_emb_dim=6144 base_num_query_heads=24 base_num_kv_heads=24 base_mlp_dim=24576 \
      base_num_decoder_layers=48 head_dim=256 max_target_length=1024 trainable_position_size=16384 \
      mlp_activations=['gelu'] vocab_size=32768 enable_dropout=False logits_via_embedding=True \
      normalize_embedding_logits=False logits_dot_in_fp32=False normalization_layer_epsilon=1.e-05 \
      use_iota_embed=True fused_qkv=True opt_type='adam_pax' decoder_block='gpt3' \
      gradient_clipping_threshold=1. adam_b1=0.9 adam_b2=0.95 adam_eps=1.e-8 adam_weight_decay=0.1 &"
   LOCAL_DEVICE_ID=$LOCAL_DEVICE_ID PROCESS_ID=$PROCESS_ID python MaxText/train.py MaxText/configs/base.yml hardware=gpu \
      run_name=${RUN_NAME}_$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs \
      dataset_path=gs://maxtext-dataset steps=30 enable_checkpointing=False \
      base_emb_dim=6144 base_num_query_heads=24 base_num_kv_heads=24 base_mlp_dim=24576 \
      base_num_decoder_layers=48 head_dim=256 max_target_length=1024 trainable_position_size=16384 \
      vocab_size=32768 enable_dropout=False logits_via_embedding=True \
      normalize_embedding_logits=False logits_dot_in_fp32=False normalization_layer_epsilon=1.e-05 \
      use_iota_embed=True fused_qkv=True opt_type="adam_pax" decoder_block="gpt3" \
      gradient_clipping_threshold=1. adam_b1=0.9 adam_b2=0.95 adam_eps=1.e-8 adam_weight_decay=0.1 attention=dot_product &
   PID=$!
   PIDS+=($PID)
   echo "Launched MaxText/train.py for local_device_id: $LOCAL_DEVICE_ID process_id: $PROCESS_ID and PID $PID"
done

wait_all_success_or_exit "${PIDS[@]}"