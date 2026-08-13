#!/bin/bash

# --- Configuration ---
# Which matrix to run inside the pod. Defaults to the vLLM decode matrix; pass another internal
# script (e.g. run_post_train_matrix_xpk_internal.py) as $1 to drive a different one.
INNER_SCRIPT="${1:-run_vllm_matrix_xpk_internal.py}"
MATRIX_NAME=$(basename "$INNER_SCRIPT" | sed -e 's/^run_//' -e 's/_xpk_internal\.py$//')
# Workload names must be a DNS label, so underscores become dashes there but not in the log dir.
WORKLOAD_PREFIX=$(echo "$MATRIX_NAME" | tr '_' '-')

# Kubernetes caps label values at 63 bytes and JobSet builds one as
# "<workload>-pathways-head-0-0.<workload>", i.e. 2*len(workload)+19 bytes. With the -MMDDHHMM
# suffix that leaves 13 characters for the prefix. Overrun it and the JobSet controller rejects
# every Job it tries to create, leaving the workload admitted but permanently pod-less.
MAX_PREFIX_LEN=13
if [ ${#WORKLOAD_PREFIX} -gt $MAX_PREFIX_LEN ]; then
    WORKLOAD_PREFIX=$(echo "$WORKLOAD_PREFIX" | sed 's/-matrix$//')
fi
WORKLOAD_PREFIX="${WORKLOAD_PREFIX:0:$MAX_PREFIX_LEN}"

export WORKLOAD_NAME="${WORKLOAD_PREFIX}-$(date +%m%d%H%M)"

export CLUSTER_NAME=mesa-v6e32-eu
export TPU_TYPE=v6e-32
export REGION=europe-west4
export ZONE=europe-west4-a
export PROJECT_ID=cienet-cmcs

# Base local log directory
BASE_LOG_DIR="${MATRIX_NAME}_logs"
mkdir -p "$BASE_LOG_DIR"
LOCAL_LOG_FILE="${BASE_LOG_DIR}/${WORKLOAD_NAME}.log"

echo "================================================================="
echo "WORKLOAD NAME: ${WORKLOAD_NAME}"
echo "CLUSTER      : ${CLUSTER_NAME}"
echo "PROJECT      : ${PROJECT_ID}"
echo "ZONE         : ${ZONE}"
echo "LOG FILE     : ${LOCAL_LOG_FILE}"
echo "START TIME   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================="

# Setup Google Cloud Config and Credentials
echo "Configuring gcloud project and zone..."
/usr/bin/gcloud config set project "$PROJECT_ID"
/usr/bin/gcloud config set compute/region "$REGION"

echo "Fetching credentials for cluster ${CLUSTER_NAME}..."
/usr/bin/gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION" --project "$PROJECT_ID"

# Setup Log File Header
{
    echo "================================================================="
    echo "WORKLOAD  : ${WORKLOAD_NAME}"
    echo "START TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="
} > "$LOCAL_LOG_FILE"

# Hugging Face token for tokenizer downloads
# export HF_TOKEN="your_token_here"

# Get the base64 of the internal python script
if [ ! -f "$INNER_SCRIPT" ]; then
    echo "❌ ERROR: $INNER_SCRIPT not found!"
    exit 1
fi
INNER_SCRIPT_B64=$(base64 -w 0 "$INNER_SCRIPT")

# Optionally ship local source files into the image's own checkout at /deps, so a MaxText fix can
# be exercised without rebuilding the shared image. Opt-in and scoped to this workload:
#   MAXTEXT_PATCH_FILES="src/maxtext/a.py src/maxtext/b.py" bash run_vllm_matrix_xpk.sh ...
# Paths are relative to the repo root, which is what /deps mirrors.
PATCH_CMD=""
if [ -n "${MAXTEXT_PATCH_FILES:-}" ]; then
    for f in $MAXTEXT_PATCH_FILES; do
        if [ ! -f "$f" ]; then
            echo "❌ ERROR: patch file $f not found!"
            exit 1
        fi
    done
    PATCH_B64=$(tar -czf - $MAXTEXT_PATCH_FILES | base64 -w 0)
    PATCH_CMD="echo '${PATCH_B64}' | base64 -d | tar -xzvf - -C /deps; "
    echo "Shipping local patches into /deps: ${MAXTEXT_PATCH_FILES}"
fi

# Final command executed INSIDE the K8s pod
XPK_COMMAND="set -xue; \
export JAX_PLATFORMS='proxy,cpu'; \
export JAX_BACKEND_TARGET='grpc://127.0.0.1:29000'; \
export HF_TOKEN='${HF_TOKEN}'; \
export MATRIX_CASE_SLICE='${MATRIX_CASE_SLICE:-}'; \
sed -i 's/\${HF_TOKEN}//g' /deps/src/maxtext/configs/base.yml || true; \
${PATCH_CMD}\
echo '$INNER_SCRIPT_B64' | base64 -d > /tmp/matrix_internal.py; \
python3 /tmp/matrix_internal.py"


echo "Submitting to K8s via XPK..."
if ! xpk workload create-pathways \
    --cluster="$CLUSTER_NAME" \
    --workload="$WORKLOAD_NAME" \
    --tpu-type="$TPU_TYPE" \
    --num-slices=1 \
    --priority=very-high \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --skip-validation \
    --docker-image=gcr.io/cienet-cmcs/mesa_maxtext_base_image_trainrl:latest \
    --command="$XPK_COMMAND" 2>&1 | tee -a "$LOCAL_LOG_FILE"; then
    echo "❌ ERROR: XPK failed to submit workload."
    exit 1
fi

# --- Phase 1: Wait for Pod to Appear in K8s ---
echo -n "Waiting for Pod to be created in K8s"
POD_FOUND=false
POD_NAME=""
ATTEMPTS=0

while [ "$POD_FOUND" = false ]; do
    POD_NAME=$(/usr/bin/kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME}" -o jsonpath='{.items[*].metadata.name}' --no-headers 2>/dev/null | tr ' ' '\n' | grep "pathways-head" | head -n 1)

    if [ -n "$POD_NAME" ]; then
        POD_FOUND=true
        echo -e "\n✅ Found Pod: $POD_NAME"
    else
        ATTEMPTS=$((ATTEMPTS+1))
        if [ $((ATTEMPTS % 12)) -eq 0 ]; then
             echo -e "\n⏳ Still waiting for pod... (Attempt $ATTEMPTS)"
        fi

        if [ $ATTEMPTS -gt 1200 ]; then # 100 minute timeout
            echo -e "\n❌ Error: Timed out waiting for Pod creation."
            exit 1
        fi
        sleep 5
        echo -n "."
    fi
done

# --- Phase 2: Monitor Pod Status & Stream Logs ---
POD_STATUS="Pending"
IS_STREAMING=false

echo "Monitoring Pod Status for $POD_NAME..."
while true; do
    POD_PHASE=$(/usr/bin/kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null)

    if [[ "$POD_PHASE" == "Running" ]] && [[ "$IS_STREAMING" = false ]]; then
        # Ensure the specific jax-tpu container is actually running, not just the proxy sidecars
        CONTAINER_STATE=$(/usr/bin/kubectl get pod "$POD_NAME" -o jsonpath='{.status.containerStatuses[?(@.name=="jax-tpu")].state.running}' 2>/dev/null)
        if [[ -n "$CONTAINER_STATE" ]]; then
            echo "▶ Pod is Running. Streaming logs..."
            IS_STREAMING=true
            # Run logs in the foreground, this will block until the container finishes
            /usr/bin/kubectl logs -f "$POD_NAME" -c jax-tpu 2>&1 | tee -a "$LOCAL_LOG_FILE"
        fi
    fi

    # Check if the main jax-tpu container has terminated (since sidecars might keep the pod "Running")
    TERMINATED_STATE=$(/usr/bin/kubectl get pod "$POD_NAME" -o jsonpath='{.status.containerStatuses[?(@.name=="jax-tpu")].state.terminated}' 2>/dev/null)
    if [[ -n "$TERMINATED_STATE" ]]; then
         EXIT_CODE=$(/usr/bin/kubectl get pod "$POD_NAME" -o jsonpath='{.status.containerStatuses[?(@.name=="jax-tpu")].state.terminated.exitCode}' 2>/dev/null)
         if [[ "$EXIT_CODE" == "0" ]]; then
             POD_STATUS="Succeeded"
         else
             POD_STATUS="Failed"
         fi

         # If it finished before K8s reported 'Running' and we missed the stream, just grab all logs once
         if [[ "$IS_STREAMING" == false ]]; then
             /usr/bin/kubectl logs "$POD_NAME" -c jax-tpu >> "$LOCAL_LOG_FILE" 2>&1
         fi
         break
    fi

    # Also handle if the entire pod fails early before containers are scheduled
    POD_PHASE=$(/usr/bin/kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "$POD_PHASE" == "Failed" ]]; then
         POD_STATUS="Failed"
         if [[ "$IS_STREAMING" == false ]]; then
             /usr/bin/kubectl logs "$POD_NAME" -c jax-tpu >> "$LOCAL_LOG_FILE" 2>&1
         fi
         break
    fi

    sleep 15
done

FULL_POD_LOG_FILE="${BASE_LOG_DIR}/${WORKLOAD_NAME}_full_pod.log"
echo "Dumping full pod logs to $FULL_POD_LOG_FILE..."
/usr/bin/kubectl logs "$POD_NAME" -c jax-tpu > "$FULL_POD_LOG_FILE" 2>&1

# --- Phase 3: Final Status Check ---
if [ "$POD_STATUS" == "Succeeded" ]; then
    echo "✅ SUCCESS: Finished $WORKLOAD_NAME."
    exit 0
else
    echo "❌ ERROR: $WORKLOAD_NAME failed (Status: $POD_STATUS)."
    exit 1
fi
