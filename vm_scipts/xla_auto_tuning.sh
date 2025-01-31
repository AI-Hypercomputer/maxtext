for xla_gpu_enable_pipelined_all_gather in true false
do 
  for xla_gpu_enable_pipelined_reduce_scatter in true false
  do
    for xla_gpu_enable_pipelined_all_reduce in true false
    do
      for xla_gpu_enable_while_loop_double_buffering in true false
      do
        for xla_gpu_enable_triton_softmax_fusion in true false
        do 
          for xla_gpu_enable_all_gather_combine_by_dim in true false
          do
            for xla_gpu_enable_reduce_scatter_combine_by_dim in true false
            do
              export XLA_FLAGS="--xla_dump_to=/tmp/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true
                                --xla_gpu_graph_level=0 --xla_gpu_enable_highest_priority_async_stream=true
                                --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728
                                --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=${xla_gpu_enable_pipelined_all_gather}
                                --xla_gpu_enable_pipelined_reduce_scatter=${xla_gpu_enable_pipelined_reduce_scatter} --xla_gpu_enable_pipelined_all_reduce=${xla_gpu_enable_pipelined_all_reduce}
                                --xla_gpu_enable_while_loop_double_buffering=${xla_gpu_enable_while_loop_double_buffering} --xla_gpu_enable_triton_softmax_fusion=${xla_gpu_enable_triton_softmax_fusion}
                                --xla_gpu_enable_all_gather_combine_by_dim=${xla_gpu_enable_all_gather_combine_by_dim} --xla_gpu_enable_reduce_scatter_combine_by_dim=${xla_gpu_enable_reduce_scatter_combine_by_dim}
                                --xla_disable_hlo_passes=rematerialization"
              echo ${XLA_FLAGS}
              export TF_FORCE_GPU_ALLOW_GROWTH=true
              export BASE_OUTPUT_DIRECTORY=gs://jwyang/maxtext
              export ASYNC_CHECKPOINTING=false
              export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
              export PER_DEVICE_BATCH_SIZE=140
              python3 MaxText/inference_microbenchmark.py MaxText/configs/base.yml  \
                      base_output_directory=${BASE_OUTPUT_DIRECTORY}  \
                      model_name='llama2-70b' \
                      max_prefill_predict_length=1024  \
                      max_target_length=2048  \
                      attention=dot_product \
                      scan_layers=false \
                      hardware=gpu \
                      async_checkpointing=${ASYNC_CHECKPOINTING} \
                      per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
                      inference_microbenchmark_prefill_lengths=1024  \
                      inference_microbenchmark_stages=prefill,generate \
                      inference_microbenchmark_loop_iters=64 \
                      run_name=$(date +%Y-%m-%d-%H-%M) \
                      ici_fsdp_parallelism=1 \
                      ici_autoregressive_parallelism=-1 \
                      ici_tensor_parallelism=1 \
                      weight_dtype=bfloat16 \
                      quantization=int8 quantize_kvcache=True |& tee ${xla_gpu_enable_pipelined_all_gather}_${xla_gpu_enable_pipelined_reduce_scatter}_${xla_gpu_enable_pipelined_all_reduce}_${xla_gpu_enable_while_loop_double_buffering}_${xla_gpu_enable_triton_softmax_fusion}_${xla_gpu_enable_all_gather_combine_by_dim}_${xla_gpu_enable_reduce_scatter_combine_by_dim}.txt
            done
          done
        done
      done
    done
  done
done