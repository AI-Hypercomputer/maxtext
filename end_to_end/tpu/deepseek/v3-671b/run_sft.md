<!--
 # Copyright 2023–2025 Google LLC
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
 -->

# Supervised Fine-Tuning (SFT) with Deepseek3-671b Model

To run Deepseek
## using XPK

### Set up environment variables
```
export PROJECT=<Google Cloud Project Name>
export CLUSTER_NAME=<GKE cluster name>
export ZONE=<GKE cluster zone>
export TPU_TYPE=v6e-256
export RUN_NAME=<run name>
export WORKLOAD_NAME=<workload name>
export OUTPUT_PATH=<GCS bucket for output>
STEPS
HF_TOKEN
PRE_TRAINED_MODEL_CHECKPOINT_PATH
```

### Create workload
xpk workload create \
--cluster ${CLUSTER_NAME} \
--docker-image ${DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type ${TPU_TYPE} --num-slices=1 --zone=${ZONE} \
--project=${PROJECT} \
--command "python3 -m MaxText.sft.sft_trainer MaxText/configs/sft.yml run_name=$RUN_NAME base_output_directory=$OUTPUT_PATH model_name=deepseek3-671b load_parameters_path=$PRE_TRAINED_MODEL_CHECKPOINT_PATH hf_access_token=$HF_TOKEN tokenizer_path=deepseek-ai/DeepSeek-V3 per_device_batch_size=1 steps=$STEPS profiler=xplane megablox=False sparse_matmul=False ici_expert_parallelism=16 ici_fsdp_parallelism=16 weight_dtype=bfloat16 dtype=bfloat16"
