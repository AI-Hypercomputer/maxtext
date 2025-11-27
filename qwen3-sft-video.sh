set -xe

RUN_NAME=sft-$(date +%Y-%m-%d-%H-%M-%S)
PER_DEVICE_BATCH_SIZE=1
BASE_OUTPUT_DIRECTORY=gs://aireenmei-multipod/sft
PRE_TRAINED_MODEL=qwen3-omni-30b-a3b
STEPS=20
#PRE_TRAINED_MODEL_CKPT_PATH=gs://maxtext-llama/llama4-17b-16e/2025-06-06/unscanned/0/items
#PRE_TRAINED_MODEL_CKPT_PATH=gs://maxtext-model-checkpoints/llama2-7b/2025-01-23-19-26/unscanned/checkpoints/0/items

PRE_TRAINED_MODEL_TOKENIZER=src/MaxText/assets/qwen3-tokenizer

# SFT with HF pipeline
python3 -m MaxText.sft_trainer MaxText/configs/sft-video-mme.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} \
    tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    tokenizer_type=huggingface \
    megablox=true \
    sparse_matmul=true \
    ici_tensor_parallelism=1 \
    ici_fsdp_parallelism=-1 \
    ici_data_parallelism=1 \
    ici_expert_parallelism=1 \
    ici_autoregressive_parallelism=1 \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    steps=${STEPS} max_target_length=1024 checkpoint_period=100 \
    attention=dot_product scan_layers=false enable_checkpointing=false \
    dataset_type=grain grain_file_type=parquet \
    grain_train_files=gs://aireenmei-us-central1/maxtext-dataset/hf/video_mme_videos/test-00000-of-00001.parquet \
    video_path=/tmp/gcsfuse/maxtext-dataset/hf/video_mme_videos/data \
    grain_worker_count=1
 
    #scan_layers=False load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH}

# synthetic 588 vs 576
