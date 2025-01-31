#!/bin/bash


#!/bin/bash

# Multi-Host vlp (TODO: replace these params for your own config)
NAME="jwyang-tpu-sh1"
# NAME="jwyang-v5p8-vm"
ACCELERATOR_TYPE="v5litepod-4"
# ACCELERATOR_TYPE="v5litepod-8"
# ACCELERATOR_TYPE="v5p-8"
RUNTIME_VERSION="v2-alpha-tpuv5-lite"
# PROJECT="tpu-prod-env-automated"
PROJECT="cloud-tpu-inference-test"
# PROJECT="tpu-prod-env-small"
# PROJECT="tpu-prod-env-large-cont"
# ZONE="us-east1-c"
ZONE="us-west1-c"
# ZONE="us-east5-a"

USER=jwyang

# (TODO: replace these params to your own config)
NUM_WORKERS=1
TPU_NAME="t1v-n-63d3a09c"

create_tpu() {
  # A temporary solution to clean up the failed and suspended queued resources.
  # Otherwise, there will be a quota error.
  existing_qr=$(gcloud alpha compute tpus queued-resources list \
    --project ${PROJECT} \
    --zone ${ZONE} \
    --quiet)
  while read -r line; do
    name=$(echo $line | awk '{print $1}')
    status=$(echo $line | awk '{print $5}')
    echo ${name}
    echo ${status}
    if [[ ${status} == "SUSPENDED" || ${status} == "FAILED" ]]; then
      gcloud alpha compute tpus queued-resources delete ${name} \
        --project ${PROJECT} \
        --zone ${ZONE} \
        --quiet
    fi
  done <<< ${existing_qr}

  gcloud alpha compute tpus queued-resources create ${NAME} \
    --description noteardown \
    --node-id ${NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --runtime-version=${RUNTIME_VERSION} \
    --reserved;
}

list_tpu() {
  gcloud compute tpus tpu-vm list --project=${PROJECT} --zone=${ZONE};
}

list_queue_resource() {
  gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE};
}

delete_tpu() {
  gcloud alpha compute tpus tpu-vm delete ${NAME} --project=${PROJECT} --zone=${ZONE};
  gcloud alpha compute tpus queued-resources delete ${NAME} --project=${PROJECT} --zone=${ZONE};
}

ssh_to_tpu() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${1} --project ${PROJECT} -- -o ProxyCommand='corp-ssh-helper %h %p'
}

create_disk() {
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    TPU_WORKER_NAME=${TPU_NAME}-w-${i}
    DISK_NAME=${NAME}-w${i}-ssd

    SIZE=35
    if [[ ${i} == 0 ]]
    then
      SIZE=512
    fi

    gcloud compute disks create ${DISK_NAME} \
      --size ${SIZE} \
      --zone ${ZONE} \
      --type pd-ssd \
      --project=${PROJECT}

    # attach disk to tpu
    gcloud alpha compute instances attach-disk ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --disk=${DISK_NAME} \
      --mode=rw \
      --project=${PROJECT}

    gcloud compute instances set-disk-auto-delete ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --auto-delete \
      --disk=${DISK_NAME} \
      --project=${PROJECT}

    gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${i} --project=${PROJECT} \
      --command="sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb &&
               sudo mkdir -p /mnt/disks/persist &&
               sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist" \
      -- -o ProxyCommand='corp-ssh-helper %h %p'
  done
}

detach_disks() {
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    TPU_WORKER_NAME=${TPU_NAME}-w-${i}
    DISK_NAME=${NAME}-w${i}-ssd

    # attach disk to tpu
    gcloud alpha compute instances detach-disk ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --disk=${DISK_NAME} \
      --project=${PROJECT}
  done
}

check_disks() {
  set -o xtrace
  dir_checks=""
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    dir_checks="$dir_checks $(
      gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${i} --project=${PROJECT} \
        --command="if [ -d /mnt/disks/persist ]; then echo "exists"; fi" \
        -- -o ProxyCommand='corp-ssh-helper %h %p'
    )"
  done
  num_dir_exists=$(echo "$dir_checks" | wc -w)
  echo "Number of workers with disks: $num_dir_exists"
  set +o xtrace
}



############### Scrach ################
copy_relevant_files() {
  # # kill model server process
  # gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
  #   --command="sudo rm /tmp/libtpu_lockfile && sudo lsof -t /dev/vfio/0 > tpu_process_pid.txt && sudo pkill -F tpu_process_pid.txt" \
  #   -- -o ProxyCommand='corp-ssh-helper %h %p'


  # gcloud compute tpus tpu-vm \
  #   scp --zone=${ZONE} --project=${PROJECT} --worker=all \
  #   $PWD/MaxText/maxengine.py \
  #   ${NAME}:~/maxtext/MaxText/maxengine.py \
  #   --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  gcloud compute tpus tpu-vm \
    scp --zone=${ZONE} --project=${PROJECT} --worker=all \
    $PWD/benchmarks/benchmark_serving.py \
    ${NAME}:~/JetStream/benchmarks/benchmark_serving.py \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  gcloud compute tpus tpu-vm \
    scp --zone=${ZONE} --project=${PROJECT} --worker=all \
    $PWD/benchmarks/open_orca_gpt4_tokenized_llama.calibration_1000.pkl \
    ${NAME}:~/JetStream/benchmarks/ \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

}


# # Microbenchmark command
# # source .env/bin/activate
# your_run_name=jwyang_bs1_llama7b
# python MaxText/inference_microbenchmark.py \
#   MaxText/configs/base.yml \
#   base_output_directory=gs://jwyang-data/maxtext-llama2-7b/microbenchmark \
#   run_name=${your_run_name} \
#   per_device_batch_size=12 \
#   save_config_to_gcs=true \
#   model_name=llama2-7b \
#   tokenizer_path=assets/tokenizer.llama2 \
#   inference_microbenchmark_prefill_lengths=1024 \
#   max_prefill_predict_length=1024 \
#   max_target_length=2048 \
#   ici_fsdp_parallelism=1 \
#   ici_tensor_parallelism=-1 \
#   ici_autoregressive_parallelism=1 \
#   weight_dtype=bfloat16 \
#   enable_profiler=true \
#   scan_layers=false \
#   quantization=int8 \
#   quantize_kvcache=true
#   inference_mode=true


# LLaMA2-7B JetStream/Maxtext commands
export model_name=llama2-7b
export tokenizer_path=assets/tokenizer.llama2
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"
export ici_tensor_parallelism=-1
export ici_autoregressive_parallelism=1
export per_device_batch_size=12
export load_parameters_path_chat=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-18-28/checkpoints/0/items
export load_parameters_path=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-19-40/checkpoints/0/items

python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://jwyang-data/maxtext-llama2-7b/microbenchmark \
  load_parameters_path=${load_parameters_path_chat} \
  run_name=$(date +%Y-%m-%d-%H-%M) \
  save_config_to_gcs=true \
  model_name=${model_name} \
  tokenizer_path=${tokenizer_path} \
  inference_microbenchmark_log_file_path=microbenchmark.json \
  inference_microbenchmark_prefill_lengths=1024 \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_loop_iters=1000 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  per_device_batch_size=${per_device_batch_size} \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=${ici_tensor_parallelism} \
  ici_autoregressive_parallelism=${ici_autoregressive_parallelism} \
  enable_profiler=false \
  scan_layers=false \
  weight_dtype=bfloat16
  # quantization=int8
  # quantize_kvcache=True


export model_name=llama2-7b
export dataset_path=/home/jwyang/llama7b_chat_openorca_input.json
python JetStream/benchmarks/benchmark_serving.py \
  --tokenizer ~/maxtext/assets/tokenizer.llama2 \
  --warmup-first true \
  --save-result \
  --save-request-outputs \
  --request-outputs-file-path /home/jwyang/outputs.json \
  --num-prompts 1000 \
  --max-output-length 1024 \
  --dataset openorca \
  --dataset-path ${dataset_path}



# # 13b model
export model_name=llama2-13b
export tokenizer_path=assets/tokenizer.llama2
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"
export ici_tensor_parallelism=-1
export ici_autoregressive_parallelism=1
export per_device_batch_size=1
export load_parameters_path=gs://runner-maxtext-logs/2024-05-16-23-59/unscanned_chkpt/checkpoints/0/items


export experiment_time=$(date +%Y-%m-%d-%H-%M)
echo "export experiment_time=${experiment_time}"
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://morgandu-tpu/maxtext-logs/microbenchmark/${experiment_time} \
  model_name=llama2-13b \
  async_checkpointing=false \
  load_parameters_path=gs://runner-maxtext-logs/2024-05-16-23-59/unscanned_chkpt/checkpoints/0/items \
  run_name=${experiment_time} \
  inference_microbenchmark_log_file_path=${run_name}.json \
  tokenizer_path=assets/tokenizer.llama2 \
  weight_dtype=bfloat16 \
  inference_microbenchmark_prefill_lengths=1024 \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_loop_iters=10 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=-1 \
  ici_autoregressive_parallelism=1 \
  enable_profiler=false \
  scan_layers=false \
  attention=dot_product \
  save_config_to_gcs=true \
  per_device_batch_size=1


  run_name=$(date +%Y-%m-%d-%H-%M) \
  save_config_to_gcs=true \
  model_name=${model_name} \
  tokenizer_path=${tokenizer_path} \
  inference_microbenchmark_log_file_path=microbenchmark.json \
  inference_microbenchmark_prefill_lengths=1024 \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_loop_iters=1000 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  per_device_batch_size=${per_device_batch_size} \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=${ici_tensor_parallelism} \
  ici_autoregressive_parallelism=${ici_autoregressive_parallelism} \
  enable_profiler=false \
  scan_layers=false \
  weight_dtype=bfloat16


python MaxText/inference_microbenchmark.py \
    MaxText/configs/base.yml \
    base_output_directory=gs://morgandu-tpu/maxtext-logs/microbenchmark/${experiment_time} \
    model_name=llama2-13b \
    async_checkpointing=false \
    load_parameters_path=gs://runner-maxtext-logs/2024-05-16-23-59/unscanned_chkpt/checkpoints/0/items \
    run_name=${experiment_time} \
    inference_microbenchmark_log_file_path=${run_name}.json \
    tokenizer_path=assets/tokenizer.llama2 \
    weight_dtype=bfloat16 \
    inference_microbenchmark_prefill_lengths=1024 \
    inference_microbenchmark_stages=prefill,generate \
    inference_microbenchmark_loop_iters=10 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=-1 \
    ici_autoregressive_parallelism=1 \
    enable_profiler=false \
    scan_layers=false \
    attention=dot_product \
    save_config_to_gcs=true \
    per_device_batch_size=1



# # LLaMA2-70B commands
# # source .env/bin/activate
# your_run_name=jwyang_bs1_llama70b
# python MaxText/inference_microbenchmark.py \
#   MaxText/configs/base.yml \
#   base_output_directory=gs://jwyang-data/maxtext-llama2-70b/microbenchmark \
#   run_name=${your_run_name} \
#   per_device_batch_size=1 \
#   save_config_to_gcs=true \
#   model_name=llama2-70b \
#   tokenizer_path=assets/tokenizer.llama2 \
#   inference_microbenchmark_prefill_lengths=32 \
#   max_prefill_predict_length=32 \
#   max_target_length=64 \
#   ici_fsdp_parallelism=1 \
#   ici_tensor_parallelism=-1 \
#   ici_autoregressive_parallelism=1 \
#   weight_dtype=bfloat16 \
#   enable_profiler=true \
#   scan_layers=false \
#   quantization=int8 \
#   quantize_kvcache=true 


export model_name=llama2-70b
export tokenizer_path=assets/tokenizer.llama2
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"
export ici_tensor_parallelism=-1
export ici_autoregressive_parallelism=1
export per_device_batch_size=1
export prefill_length=16
export target_length=32

python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://jwyang-data/maxtext-llama2-70b/microbenchmark \
  run_name=$(date +%Y-%m-%d-%H-%M) \
  save_config_to_gcs=true \
  model_name=${model_name} \
  tokenizer_path=${tokenizer_path} \
  inference_microbenchmark_log_file_path=microbenchmark.json \
  inference_microbenchmark_prefill_lengths=${prefill_length} \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_loop_iters=1000 \
  max_prefill_predict_length=${prefill_length} \
  max_target_length=${target_length} \
  per_device_batch_size=${per_device_batch_size} \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=${ici_tensor_parallelism} \
  ici_autoregressive_parallelism=${ici_autoregressive_parallelism} \
  enable_profiler=false \
  scan_layers=false \
  weight_dtype=bfloat16 \
  quantization=int8 \
  quantize_kvcache=True