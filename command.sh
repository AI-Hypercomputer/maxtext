RUN_NAME=test
MAXTEXT_OUTPUT_PATH=gs://tony-moe

python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
per_device_batch_size=24 enable_checkpointing=false async_checkpointing=false \
model_name=mixtral-8x1b ici_fsdp_parallelism=4 skip_first_n_steps_for_profiler=5 steps=100 max_target_length=4096  \
tokenizer_path=assets/tokenizer.mistral attention=flash dtype=bfloat16 weight_dtype=bfloat16 opt_type=sgd dataset_type=synthetic \
profiler=xplane


python3 MaxText/train_compile.py MaxText/configs/subsup_small.yml compile_topology=v5p-256 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/subsup_large.yml compile_topology=v5p-2048 compile_topology_num_slices=1
