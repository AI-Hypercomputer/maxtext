#!/bin/bash 

# Build and upload image
# bash docker_build_dependency_image.sh DEVICE=gpu LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# docker push gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=yooh/maxtext-tcpx

# Clone Yuwei's XPK branch
git clone -b yangyuwei-xpk-gpu https://github.com/google/xpk.git

# Write env file
cat << EOF > xpk/env2.txt
export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-gpt-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true --xla_gpu_pgle_profile_file_or_directory_path=profile.pbtxt"
EOF

python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-gpt-$(date +%Y-%m-%d-%H-%M) \
    --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=1 --env-file=xpk/env2.txt --priority=high \
    --command "nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi 
        --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml hardware=gpu \
        run_name=yooh-gpt-$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs \
        dataset_path=gs://maxtext-dataset steps=30 enable_checkpointing=False tokenizer_path=assets/tokenizer \
        base_emb_dim=6144 base_num_query_heads=24 base_num_kv_heads=24 base_mlp_dim=24576 \
        base_num_decoder_layers=48 head_dim=256 max_target_length=1024 trainable_position_size=16384 \
        vocab_size=32768 enable_dropout=False logits_via_embedding=True per_device_batch_size=8.0 \
        normalize_embedding_logits=False logits_dot_in_fp32=False normalization_layer_epsilon=1.e-05 \
        use_iota_embed=True fused_qkv=True opt_type=adam_pax decoder_block=gpt3 \
        gradient_clipping_threshold=1. adam_b1=0.9 adam_b2=0.95 adam_eps=1.e-8 adam_weight_decay=0.1 attention=dot_product"



