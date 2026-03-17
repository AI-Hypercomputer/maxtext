# WORKING SCRIPT!!!!!

#!/bin/bash

# 1. Suppress CUDA warnings (force Jax to only look for TPU/CPU)
export JAX_PLATFORMS=tpu,cpu
# Silence XLA/TF C++ logging spam
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export CUDA_VISIBLE_DEVICES=""

# 2. Increase GCS resilience to prevent CURL timeout drops
export GCS_READ_CACHE_MAX_RETRIES=10
export GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
export GCS_METADATA_CACHE_MAX_RETRIES=10
export GCS_RESOLVE_REFRESH_SECS=60

source $HOME/maxtext_venv/bin/activate
cd $HOME/custom-gemma

# 3. Run the MaxText training job
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
    run_name="custom-gemma-swa" \
    model_name="brahmai-4b" \
    base_output_directory="gs://zen-check-hns/custom-gemma-4b" \
    per_device_batch_size=4 \
    attention="flash" \
    max_target_length=4096 \
    gradient_accumulation_steps=1 \
    opt_type="adamw" \
    adam_b1=0.9 \
    adam_b2=0.95 \
    adam_eps=1e-8 \
    adam_weight_decay=0.1 \
    learning_rate=3e-4 \
    lr_schedule_type="wsd" \
    warmup_steps_fraction=0.05 \
    wsd_decay_steps_fraction=0.15 \
    wsd_decay_style="linear" \
    steps=100000 \
    dataset_type="grain" \
    grain_train_files="/home/pinakinchoudhary/data/dclm/*/*/*.arrayrecord" \
    grain_file_type="arrayrecord" \
    grain_packing_type="best_fit" \
    grain_worker_count=4 \
    tokenizer_type="huggingface" \
    tokenizer_path="/home/pinakinchoudhary/custom-gemma/brahmai-tokenizer" \
    add_bos=True \
    add_eos=True \
    tokenize_train_data=True \
    enable_checkpointing=True \
    checkpoint_period=10000\
    log_period=50 \
    profiler="xplane" \
    skip_first_n_steps_for_profiler=10 \
    profiler_steps=5 \
    upload_all_profiler_results=True \
