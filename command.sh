# AOT Command:
python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=9 compile_topology=v5p-256 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=8 compile_topology=v5p-1024 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=9 compile_topology=v5p-2048 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=9 compile_topology=v5p-12288 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_large per_device_batch_size=7 compile_topology=v5p-12288 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_large per_device_batch_size=2 compile_topology=v5p-1024 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_large per_device_batch_size=2 remat_policy=qkv_proj_offloaded compile_topology=v5p-1024 compile_topology_num_slices=1 


python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=8 \
remat_policy=save_out_proj \
compile_topology=v5p-1024 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small \
per_device_batch_size=4 \
ici_tensor_parallelism=8 ici_fsdp_parallelism=64 \
compile_topology=v5p-1024 compile_topology_num_slices=1


PROJECT_ID=cloud-tpu-best-effort-colo 
ZONE=europe-west1-c
TPU_NAME=v5p-256-moe-test4
QR_NAME=$TPU_NAME
NODE_PREFIX=${QR_NAME}
TPU_TYPE=v5p-256
VERSION=v2-alpha-tpuv5

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}


gcloud alpha compute tpus queued-resources create ${QR_NAME} \
--accelerator-type=${TPU_TYPE} \
--project=${PROJECT_ID} \
--zone=${ZONE} \
--runtime-version=${VERSION} \
--node-id=${QR_NAME} \
--description noteardown \
--reserved 

gcloud alpha compute tpus queued-resources list --filter=$QR_NAME

# Training Command:
# PROJECT_ID=tpu-prod-env-multipod
# ZONE=us-east5-a
PROJECT_ID=cloud-tpu-best-effort-colo 
ZONE=europe-west1-c
TPU_NAME=v5p-256-moe-test4

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

# Install deps
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable;"

# Start training small model
BATCH_SIZE=6
RUN_NAME=v5p_256_subsup_small_test_per_device_batch_size$BATCH_SIZE
MAXTEXT_OUTPUT_PATH=gs://tony-moe

python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="\
    python3 MaxText/train.py MaxText/configs/base.yml model_name=subsup_small \
    base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane"

yes | gcloud alpha compute tpus queued-resources delete $QR_NAME --async --force

export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump/"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=81920"

BUCKET_NAME=gs://tony-moe/7_29/
MAXTEXT_OUTPUT_PATH=$BUCKET_NAME
TPU_TYPE=v5p-1024
MODEL=subsup_small
BATCH_SIZE=8
VERSION=v2-alpha-tpuv5
xla_tpu_scoped_vmem_limit_kib=81920
ici_tensor_parallelism=8
RUN_NAME=${TPU_TYPE}_${MODEL}_test2_per_device_batch_size$BATCH_SIZE-ici_tensor_parallelism$ici_tensor_parallelism-xla_tpu_scoped_vmem_limit_kib$xla_tpu_scoped_vmem_limit_kib
NODE_COUNT=1

python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME="$RUN_NAME" --BUCKET_NAME="$BUCKET_NAME" \
--TPU_TYPE=$TPU_TYPE --VERSION=$VERSION --CQR_EXTRA_ARGS="--reserved" \
--COMMAND="\
    bash setup.sh MODE=stable;\
    export LIBTPU_INIT_ARGS=\"--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    ici_tensor_parallelism=${ici_tensor_parallelism}"


BUCKET_NAME=gs://tony-moe/7_29/
MAXTEXT_OUTPUT_PATH=$BUCKET_NAME
TPU_TYPE=v5p-1024
MODEL=subsup_large
BATCH_SIZE=2
VERSION=v2-alpha-tpuv5
xla_tpu_scoped_vmem_limit_kib=81920
ici_tensor_parallelism=8
RUN_NAME=${TPU_TYPE}_${MODEL}_test2_per_device_batch_size$BATCH_SIZE-ici_tensor_parallelism$ici_tensor_parallelism-xla_tpu_scoped_vmem_limit_kib$xla_tpu_scoped_vmem_limit_kib
NODE_COUNT=1

python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME="$RUN_NAME" --BUCKET_NAME="$BUCKET_NAME" \
--TPU_TYPE=$TPU_TYPE --VERSION=$VERSION --CQR_EXTRA_ARGS="--reserved" \
--COMMAND="\
    bash setup.sh MODE=stable;\
    export LIBTPU_INIT_ARGS=\"--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=${xla_tpu_scoped_vmem_limit_kib}\"
    python3 MaxText/train.py MaxText/configs/base.yml model_name=${MODEL} \
    base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane \
    ici_tensor_parallelism=${ici_tensor_parallelism}"




MAXTEXT_OUTPUT_PATH=gs://tony-moe
RUN_NAME=test
python3 MaxText/train.py MaxText/configs/base.yml   \
base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME}    \
enable_checkpointing=false async_checkpointing=false    per_device_batch_size=1    \
skip_first_n_steps_for_profiler=5 steps=30    dataset_type=synthetic    \
profiler=xplane gradient_accumulation_steps=10


gcloud alpha compute tpus queued-resources list --filter=v5p-2048_subsup_large_test_per_device_batch_size2-test --zone=europe-west1-c --project=cloud-tpu-best-effort-colo