#!/bin/bash
export CLUSTER_NAME=a3plus-benchmark
export WORKLOAD_NAME=yangyuwei-maxtext-llama2-70b-32n
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yangyuwei/maxtext-fastrak:06-25-2024-nightly
export DEVICE_TYPE=h100-mega-80gb-8
export NUM_NODES=32

python3 xpk.py workload create --cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} --docker-image ${LOCAL_IMAGE_NAME} \
--device-type ${DEVICE_TYPE} --num-nodes ${NUM_NODES} \
--command "nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python MaxText/train.py MaxText/configs/base.yml hardware=gpu run_name=maxtext-llama2-70b-06-27-2024-profile steps=30 model_name=llama2-70b enable_checkpointing=false attention=cudnn_flash_te dataset_type=synthetic async_checkpointing=false enable_profiler=true base_output_directory=gs://runner-maxtext-logs use_iota_embed=true scan_layers=true per_device_batch_size=4 remat_policy=save_qkv_proj logits_dot_in_fp32=false max_target_length=4096 num_slices=32 dcn_fsdp_parallelism=32 ici_fsdp_parallelism=8 ici_data_parallelism=1 dcn_data_parallelism=1 ici_tensor_parallelism=1 dcn_tensor_parallelism=1" \
--env-file=env.txt \
--priority=high \
--scheduler=gke.io/topology-aware-auto

