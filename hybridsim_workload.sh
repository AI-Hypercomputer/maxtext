# bash docker_build_dependency_image.sh

MODEL_NAME=llama3.1-8b
NUM_SLICES=1
PER_DEVICE_BATCH_SIZE=2
MAX_TARGET_LENGTH=2048
QUANTIZATION=bf16

xpk workload create \
--cluster raymondzou-v5p-single-cell \
--zone europe-west1 \
--base-docker-image maxtext_base_image \
--workload htg-llama31-8b-v5p-8-${NUM_SLICES}-${PER_DEVICE_BATCH_SIZE}-bf16-${MAX_TARGET_LENGTH} \
--tpu-type=v5p-2048 \
--num-slices=${NUM_SLICES}  \
--command "python MaxText/train.py MaxText/configs/base.yml model_name=${MODEL_NAME} base_output_directory=gs://hengtaoguo-maxtext-logs dataset_path=gs://hengtaoguo-maxtext-dataset tokenizer_path=assets/tokenizer.llama2 remat_policy=save_qkv_proj steps=20 enable_checkpointing=false per_device_batch_size=${PER_DEVICE_BATCH_SIZE} max_target_length=${MAX_TARGET_LENGTH} profiler=xplane"
# --command "python MaxText/train.py MaxText/configs/base.yml model_name=${MODEL_NAME} base_output_directory=gs://hengtaoguo-maxtext-logs dataset_path=gs://hengtaoguo-maxtext-dataset tokenizer_path=assets/tokenizer.llama2 remat_policy=save_qkv_proj steps=20 enable_checkpointing=false per_device_batch_size=${PER_DEVICE_BATCH_SIZE} max_target_length=${MAX_TARGET_LENGTH} profiler=xplane quantization=${QUANTIZATION}"

# xpk workload delete --project tpu-prod-env-multipod --cluster v5e-256-opm-ase1-lowmtu --filter-by-job htg-llama31-8b-v5e-256-2-1-int8-2048
# xpk workload delete --project tpu-prod-env-multipod --cluster raymondzou-v5p-single-cell --zone europe-west1 --filter-by-job  htg-llama31-8b-v5p-8-1-4-bf16-2048
