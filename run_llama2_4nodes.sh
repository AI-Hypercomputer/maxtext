#!/bin/bash 

# Build and upload image
# bash docker_build_dependency_image.sh DEVICE=gpu LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# docker push gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=yooh/maxtext-tcpx

# Clone Yuwei's XPK branch
git clone -b yangyuwei-xpk-gpu https://github.com/google/xpk.git

# Write env file
cat << EOF > xpk/env1.txt
export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-llama-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true"
EOF

# python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-llama-$(date +%Y-%m-%d-%H-%M) --command "nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default model_name=llama2-7b attention=dot_product async_checkpointing=False hardware=gpu steps=30" --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=$NUM_SLICE --env-file=xpk/env1.txt --priority=high
python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-llama-$(date +%Y-%m-%d-%H-%M) \
    --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=4 --env-file=xpk/env1.txt --priority=high \
    --command "python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs 
        load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default 
        model_name=llama2-7b attention=dot_product async_checkpointing=False hardware=gpu steps=30" 


