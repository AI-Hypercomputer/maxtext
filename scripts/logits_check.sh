# gemma4e2b_checkpoint_fix'

source ~/VENVS/ubench/bin/activate

export PROJECT_ID="diesel-patrol-382622"
export BASE_OUTPUT_DIRECTORY="gs://mdonati-uscentral1/maxtext-logs/"
export CLUSTER_NAME=mdonati-xpk-v7-spot
export ZONE=us-central1-c
export BASE_DOCKER_IMAGE="gcr.io/diesel-patrol-382622/maxtextrl-mdonati:latest" # rebuilt on 8/6/26

export WORKLOAD_NAME=logit-scan-full
export CHECKPOINT_DIR="gs://mdonati-uscentral1/maxtext-logs/gemma4-e2b-converted-ckpt/0/items"
export MAXTEXT_COMMAND="pip install torch --index-url https://download.pytorch.org/whl/cpu && \
export PYTHONPATH=/app/src && \
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
model_name=gemma4-e2b \
tokenizer_path=google/gemma-4-E2B \
hf_access_token=\${HF_TOKEN} \
load_parameters_path=${CHECKPOINT_DIR} \
--run_hf_model=true \
--hf_model_path=google/gemma-4-E2B \
--max_kl_div=0.03 \
use_multimodal=false \
scan_layers=true \
remat_policy=full \
per_device_batch_size=1 \
dtype=float32 \
attention=dot_product"

xpk workload create \
--cluster="${CLUSTER_NAME}" \
--project="${PROJECT_ID}" \
--tpu-type=tpu7x-8 \
--zone="${ZONE}" \
--num-slices=1 \
--base-docker-image="${BASE_DOCKER_IMAGE}" \
--script-dir="/usr/local/google/home/mattdonati/common_files/maxtext" \
--workload="${WORKLOAD_NAME}" \
--env="HF_TOKEN=${HF_TOKEN}" \
--command="${MAXTEXT_COMMAND}"

export PROJECT_ID="diesel-patrol-382622"
export BASE_OUTPUT_DIRECTORY="gs://mdonati-uscentral1/maxtext-logs/"
export CLUSTER_NAME=mdonati-xpk-v7-spot
export ZONE=us-central1-c
export BASE_DOCKER_IMAGE="gcr.io/diesel-patrol-382622/maxtextrl-mdonati:latest" # rebuilt on 8/6/26

export WORKLOAD_NAME=logit-scan-none
export CHECKPOINT_DIR="gs://mdonati-uscentral1/maxtext-logs/gemma4-e2b-converted-ckpt/0/items"
export MAXTEXT_COMMAND="pip install torch --index-url https://download.pytorch.org/whl/cpu && \
export PYTHONPATH=/app/src && \
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
model_name=gemma4-e2b \
tokenizer_path=google/gemma-4-E2B \
hf_access_token=\${HF_TOKEN} \
load_parameters_path=${CHECKPOINT_DIR} \
--run_hf_model=true \
--hf_model_path=google/gemma-4-E2B \
--max_kl_div=0.03 \
use_multimodal=false \
scan_layers=true \
remat_policy=none \
per_device_batch_size=1 \
dtype=float32 \
attention=dot_product"

xpk workload create \
--cluster="${CLUSTER_NAME}" \
--project="${PROJECT_ID}" \
--tpu-type=tpu7x-8 \
--zone="${ZONE}" \
--num-slices=1 \
--base-docker-image="${BASE_DOCKER_IMAGE}" \
--script-dir="/usr/local/google/home/mattdonati/common_files/maxtext" \
--workload="${WORKLOAD_NAME}" \
--env="HF_TOKEN=${HF_TOKEN}" \
--command="${MAXTEXT_COMMAND}"

# source ~/VENVS/ubench/bin/activate

# export PROJECT_ID="diesel-patrol-382622"
# export BASE_OUTPUT_DIRECTORY="gs://mdonati-uscentral1/maxtext-logs/"
# export CLUSTER_NAME=mdonati-xpk-v7-spot
# export ZONE=us-central1-c
# export BASE_DOCKER_IMAGE="gcr.io/diesel-patrol-382622/maxtextrl-mdonati:latest" # rebuilt on 8/6/26

# export WORKLOAD_NAME=logit-check-scan
# export CHECKPOINT_DIR="gs://mdonati-uscentral1/maxtext-logs/gemma4-e2b-converted-ckpt/0/items"
# export MAXTEXT_COMMAND="pip install torch --index-url https://download.pytorch.org/whl/cpu && \
# export PYTHONPATH=/app/src && \
# python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
# model_name=gemma4-e2b \
# tokenizer_path=google/gemma-4-E2B \
# hf_access_token=\${HF_TOKEN} \
# load_parameters_path=${CHECKPOINT_DIR} \
# --run_hf_model=true \
# --hf_model_path=google/gemma-4-E2B \
# --max_kl_div=0.03 \
# use_multimodal=false \
# scan_layers=true \
# remat_policy=full \
# per_device_batch_size=1 \
# dtype=float32 \
# attention=dot_product"

# xpk workload create \
# --cluster="${CLUSTER_NAME}" \
# --project="${PROJECT_ID}" \
# --tpu-type=tpu7x-8 \
# --zone="${ZONE}" \
# --num-slices=1 \
# --base-docker-image="${BASE_DOCKER_IMAGE}" \
# --script-dir="/usr/local/google/home/mattdonati/common_files/maxtext" \
# --workload="${WORKLOAD_NAME}" \
# --env="HF_TOKEN=${HF_TOKEN}" \
# --command="${MAXTEXT_COMMAND}"