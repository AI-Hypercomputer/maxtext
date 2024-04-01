#! /bin/bash
set -e
set -u
set -o pipefail

: "${NNODES:?Must set NNODES}"
: "${NODE_RANK:?Must set NODE_RANK}"
: "${JAX_COORDINATOR_PORT:?Must set JAX_COORDINATOR_PORT}"
: "${JAX_COORDINATOR_ADDRESS:?Must set JAX_COORDINATOR_ADDRESS}"
: "${GPUS_PER_NODE:?Must set GPUS_PER_NODE}"
: "${COMMAND:?Must set COMMAND}"


export GPUS_PER_NODE=$GPUS_PER_NODE
export JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT
export JAX_COORDINATOR_ADDRESS=$JAX_COORDINATOR_ADDRESS

set_nccl_gpudirect_tcpx_specific_configuration() {
  if [[ "$USE_GPUDIRECT" == "tcpx" ]]; then
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
  elif [[ "$USE_GPUDIRECT" == "fastrak" ]]; then
    echo "Using GPUDirect-TCPFasTrak"
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING,NET,VERSION
    export NCCL_DEBUG=INFO
    export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
    export LD_LIBRARY_PATH="/usr/local/fastrak/lib64:${LD_LIBRARY_PATH}"
    export NCCL_FASTRAK_CTRL_DEV=eth0
    export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_CROSS_NIC=0
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    export NCCL_MAX_NCHANNELS=16
    export NCCL_MIN_NCHANNELS=16
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_DYNAMIC_CHUNK_SIZE=524288
    export NCCL_P2P_NET_CHUNKSIZE=524288
    export NCCL_P2P_PCI_CHUNKSIZE=524288
    export NCCL_P2P_NVL_CHUNKSIZE=1048576
    export NCCL_FASTRAK_NUM_FLOWS=8
    export NCCL_FASTRAK_FLOWS_PER_GROUP=2
    export NCCL_BUFFSIZE=4194304
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_NET_GDR_LEVEL=PIX
  else
    echo "NOT using GPUDirect"
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
  local max_coordinator_lookups=500
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

# Resolving coordinator IP
set +e
resolve_coordinator_ip
set -e

PIDS=()
${COMMAND} &
PID=$!
PIDS+=($PID)

wait_all_success_or_exit "${PIDS[@]}"

