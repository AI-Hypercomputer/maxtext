export PROJECT=cloud-tpu-multipod-dev #<your_project_id>
export ZONE=europe-west4-b #<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>

xpk cluster create \
--default-pool-cpu-machine-type=n1-standard-32 \
--cluster ${CLUSTER_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
--reservation=cloudtpu-20240716121201-595617744

xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload hello-world-test \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--command "echo Hello World"

bash docker_build_dependency_image.sh MODE=nightly DEVICE=tpu
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner

GCS_PATH=gs://v5p-subsup-moe #<your_GCS_folder_for_results>
gcloud storage buckets create ${GCS_PATH}  --project ${PROJECT}




xla_tpu_scoped_vmem_limit_kib=81920
BATCH_SIZE=3
MODEL=MOE_48_46B
ici_tensor_parallelism=1
remat_policy=kv_proj
RUN_NAME=$MODEL-per-device-batch-size-$BATCH_SIZE-remat-$remat_policy
NODE_COUNT=1

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export WORKLOAD_NAME=subsup-48e-46b-test6
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/${PROJECT}/${USER}_runner
export OUTPUT_PATH=gs://v5p-subsup-moe/ #<your_GCS_folder_for_results>

xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--docker-image=${LOCAL_IMAGE_NAME} \
--command "\
    export LIBTPU_INIT_ARGS=\"xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"; \
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=15 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    ici_tensor_parallelism=${ici_tensor_parallelism}"


xla_tpu_scoped_vmem_limit_kib=81920
BATCH_SIZE=2
MODEL=MOE_20_84B
ici_tensor_parallelism=1
remat_policy=full
RUN_NAME=$MODEL-per-device-batch-size-$BATCH_SIZE-remat-$remat_policy
NODE_COUNT=1

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export WORKLOAD_NAME=subsup-22e-84b-test
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_dependencies:2024-09-04
export OUTPUT_PATH=gs://v5p-subsup-moe/ #<your_GCS_folder_for_results>

xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image=${LOCAL_IMAGE_NAME} \
--command "\
    export LIBTPU_INIT_ARGS=\"xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"; \
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    gcs_metrics=true \
    ici_tensor_parallelism=${ici_tensor_parallelism}"


xla_tpu_scoped_vmem_limit_kib=81920
BATCH_SIZE=3
MODEL=MOE_22_84B
ici_tensor_parallelism=1
remat_policy=full
RUN_NAME=$MODEL-per-device-batch-size-$BATCH_SIZE-remat-$remat_policy
NODE_COUNT=1

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export WORKLOAD_NAME=subsup-22e-84b-test4
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_dependencies:2024-09-04
export OUTPUT_PATH=gs://v5p-subsup-moe/ #<your_GCS_folder_for_results>

xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image=${LOCAL_IMAGE_NAME} \
--command "\
    export LIBTPU_INIT_ARGS=\"xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"; \
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    gcs_metrics=true \
    ici_tensor_parallelism=${ici_tensor_parallelism}"

xla_tpu_scoped_vmem_limit_kib=81920
BATCH_SIZE=8
MODEL=MOE_64_xB
ici_tensor_parallelism=1
remat_policy=full
RUN_NAME=$MODEL-per-device-batch-size-$BATCH_SIZE-remat-$remat_policy
NODE_COUNT=1

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export WORKLOAD_NAME=subsup-64e-xb-test6
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_dependencies:2024-09-04
export OUTPUT_PATH=gs://v5p-subsup-moe/ #<your_GCS_folder_for_results>

xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image=${LOCAL_IMAGE_NAME} \
--command "\
    export LIBTPU_INIT_ARGS=\"xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"; \
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    gcs_metrics=true \
    ici_tensor_parallelism=${ici_tensor_parallelism}"


xpk workload list \
--cluster ${CLUSTER_NAME}

export WORKLOAD_NAME_TO_DELETE=subsup-20e-xb-test6

xpk workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}