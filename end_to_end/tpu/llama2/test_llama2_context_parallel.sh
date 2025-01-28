#!/bin/bash


idx=$(date +%Y-%m-%d-%H-%M)
python3 MaxText/train.py MaxText/configs/base.yml ici_context_parallelism=-1 ici_fsdp_parallelism=1 enable_checkpointing=false base_output_directory=gs://mazumdera-test-bucket-us-east5/maxtext/seqpara/${idx} dataset_path=gs://max-datasets-rogue run_name=context_test enable_goodput_recording=false monitor_goodput=false per_device_batch_size=10 steps=30 profiler=xplane profiler_steps=20 max_target_length=65536