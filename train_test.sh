#!/bin/bash
set -euo pipefail

source get_tpu_bucket_name.sh

export TPU_PREFIX="$(get_tpu_name)"
export BUCKET_NAME="$(get_bucket_name)"
export NUM_HOSTS=$(get_num_hosts)

for arg in "$@"; do
    case $arg in
        --lr=*) LR="${arg#*=}" ;;
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --global_batch_size=*) GLOBAL_BATCH_SIZE="${arg#*=}" ;;
        --grad_clip=*) GRAD_CLIP="${arg#*=}" ;;
        --min_lr_ratio=*) MIN_LR_RATIO="${arg#*=}" ;;
        --warmup_ratio=*) WARMUP_RATIO="${arg#*=}" ;;
        --max_to_keep=*) MAX_TO_KEEP="${arg#*=}" ;;
        --data_files=*) DATA_FILES="${arg#*=}" ;;
        --shuffle=*) SHUFFLE="${arg#*=}" ;;
        --tag=*) TAG="${arg#*=}" ;;
        *) echo "[WARN] Unknown arg $arg" ;;
    esac
done

export MODEL_NAME="qwen3-0.6b"
export NUM_STEPS=50000
export SEQ_LEN=8192
export BATCH_SIZE=${BATCH_SIZE:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
export GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / BATCH_SIZE / NUM_HOSTS / 4))
export GRAD_CLIP=${GRAD_CLIP:-1.0}
export LR=${LR:-0.0003}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
export WARMUP_RATIO=${WARMUP_RATIO:-0.05}
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://${BUCKET_NAME}/model_ckpts/maxtext"
export MAX_TO_KEEP=${MAX_TO_KEEP:-1}
export DATA_FILES="${DATA_FILES:-/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record}"
export SHUFFLE="${SHUFFLE:-True}"
export RUN_NAME="${MODEL_NAME}_L200_seqlen_${SEQ_LEN}_bs_${BATCH_SIZE}_grad_accum_${GRAD_ACCUM}_lr_${LR}_min_lr_ratio_${MIN_LR_RATIO}_warmup_ratio_${WARMUP_RATIO}"
if [ ! -z "${TAG:-}" ]; then
    export RUN_NAME="${RUN_NAME}_${TAG}"
fi
export JAX_PLATFORMS=tpu
export SPARSE_MODEL_TRAINING=False

export PYTHONPATH=./src:${PYTHONPATH:-''}
python -u multihost_runner_orig.py \
    --TPU_PREFIX=${TPU_PREFIX} \
    --COMMAND="
    export TPU_LOG_DIR=/home/zephyr/tpu_logs
    export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
    source ~/maxtext_env_py311/bin/activate
    export PYTHONPATH=./src:\${PYTHONPATH:-''}
    ~/maxtext_env_py311/bin/python -u -m src.MaxText.train src/MaxText/configs/base.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files=${DATA_FILES} \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        enable_data_shuffling=${SHUFFLE} \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=${SEQ_LEN} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        model_name=${MODEL_NAME} \
        steps=${NUM_STEPS} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        gradient_clipping_threshold=${GRAD_CLIP} \
        learning_rate=${LR} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        checkpoint_period=500 \
        enable_wandb=True \
        wandb_project_name=llm_pruning \
        wandb_run_name=${TPU_PREFIX}_${RUN_NAME} \
        packing=false \
    "

# bash scripts/convert.sh gen_param_ckpt \
#     --model=${MODEL_NAME} \
#     --orbax_ckpt_name=${RUN_NAME} \
#     --step=49999 \
#     --hf_model_name=Llama-3.1-8B \
#     --direct_run_name=${RUN_NAME}

# bash scripts/convert.sh eval \
#     --model=${MODEL_NAME} \
#     --hf_model_name=Llama-3.1-8B \
#     --direct_run_name=${RUN_NAME}
