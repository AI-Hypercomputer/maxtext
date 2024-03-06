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

python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-llama-$(date +%Y-%m-%d-%H-%M) --command "nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default model_name=llama2-7b attention=dot_product async_checkpointing=False hardware=gpu steps=30" --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=$NUM_SLICE --env-file=xpk/env1.txt --priority=high



# root@gke-maxtext-a3-20nod-maxtext-a3-20nod-92d4d6cc-6vff:/app# export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-llama-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
#                 --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
#                 --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
#                 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
#                 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
#                 --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true"
# nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default model_name=llama2-7b attention=dot_product async_checkpointing=False hardware=gpu steps=30 run_name=yooh-llama-$(date +%Y-%m-%d-%H-%M)
# 2024-03-06 06:16:13.595529: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# 2024-03-06 06:16:15.194880: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1960] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
# ['MaxText/train.py', 'MaxText/configs/base.yml', 'dataset_path=gs://maxtext-dataset', 'base_output_directory=gs://runner-maxtext-logs', 'load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default', 'model_name=llama2-7b', 'attention=dot_product', 'async_checkpointing=False', 'hardware=gpu', 'steps=30', 'run_name=yooh-llama-2024-03-06-06-16']
# Updating keys from env and command line: ['run_name', 'model_name', 'load_parameters_path', 'async_checkpointing', 'attention', 'base_output_directory', 'hardware', 'dataset_path', 'steps']
# Running Model: llama2-7b
# Updating following parameters in config

# base_emb_dim: 4096
# base_num_query_heads: 32
# base_num_kv_heads: 32
# base_mlp_dim: 11008
# base_num_decoder_layers: 32
# head_dim: 128
# mlp_activations: ['silu', 'linear']
# vocab_size: 32000
# enable_dropout: False
# logits_via_embedding: False
# normalization_layer_epsilon: 1e-05
# decoder_block: llama2
# Updating keys from model: ['base_emb_dim', 'base_num_query_heads', 'base_num_kv_heads', 'base_mlp_dim', 'base_num_decoder_layers', 'head_dim', 'mlp_activations', 'vocab_size', 'enable_dropout', 'logits_via_embedding', 'normalization_layer_epsilon', 'decoder_block']
# Attempting to initialize the jax distributed system for GPU backend...
# Jax distributed system initialized on GPU!
# I0306 06:16:16.672545 136871327533376 xla_bridge.py:704] Unable to initialize backend 'rocm': NOT_FOUND: Could not find registered platform with name: "rocm". Available platform names are: CUDA
# I0306 06:16:16.673780 136871327533376 xla_bridge.py:704] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: dlopen hook: 'libtpu.so': cannot open shared object file: No such file or directory
# System Information: Jax Version: 0.4.25
# System Information: Jaxlib Version: 0.4.25
# System Information: Jax Backend: cuda 12030
# Config param adam_b1: 0.9
# Config param adam_b2: 0.95
# Config param adam_eps: 1e-08
# Config param adam_eps_root: 0.0
# Config param adam_weight_decay: 0.1
# Config param async_checkpointing: False
# Config param attention: dot_product
# Config param autoregressive_decode_assert: 
# Config param base_emb_dim: 4096
# Config param base_mlp_dim: 11008
# Config param base_num_decoder_layers: 32
# Config param base_num_kv_heads: 32
# Config param base_num_query_heads: 32
# Config param base_output_directory: gs://runner-maxtext-logs
# Config param checkpoint_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/checkpoints/
# Config param checkpoint_period: 10000
# Config param collect_stack_trace: False
# Config param compile_topology: 
# Config param compile_topology_num_slices: -1
# Config param compiled_trainstep_file: 
# Config param cosine_learning_rate_final_fraction: 0.1
# Config param data_sharding: (('data', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'autoregressive'),)
# Config param data_shuffle_seed: 0
# Config param dataset_name: c4/en:3.0.1
# Config param dataset_path: gs://maxtext-dataset
# Config param dataset_type: c4
# Config param dcn_autoregressive_parallelism: 1
# Config param dcn_data_parallelism: -1
# Config param dcn_fsdp_parallelism: 1
# Config param dcn_fsdp_transpose_parallelism: 1
# Config param dcn_sequence_parallelism: 1
# Config param dcn_tensor_parallelism: 1
# Config param decode_sampling_nucleus_p: -1
# Config param decode_sampling_strategy: greedy
# Config param decode_sampling_temperature: 1.0
# Config param decode_sampling_top_k: 0
# Config param decoder_block: llama2
# Config param dropout_rate: 0
# Config param dtype: bfloat16
# Config param emb_dim: 4096
# Config param enable_checkpointing: True
# Config param enable_data_shuffling: True
# Config param enable_dropout: False
# Config param enable_profiler: False
# Config param eval_dataset_name: c4/en:3.0.1
# Config param eval_interval: -1
# Config param eval_per_device_batch_size: 0
# Config param eval_split: validation
# Config param force_unroll: False
# Config param fused_mlp: False
# Config param fused_qkv: False
# Config param gcs_metrics: False
# Config param global_batch_size_to_load: 96
# Config param global_batch_size_to_train_on: 96
# Config param global_parameter_scale: 1
# Config param gradient_clipping_threshold: 1.0
# Config param grain_worker_count: 4
# Config param hardware: gpu
# Config param head_dim: 128
# Config param ici_autoregressive_parallelism: 1
# Config param ici_data_parallelism: 1
# Config param ici_fsdp_parallelism: -1
# Config param ici_fsdp_transpose_parallelism: 1
# Config param ici_sequence_parallelism: 1
# Config param ici_tensor_parallelism: 1
# Config param init_weights_seed: 0
# Config param jax_cache_dir: ~/jax_cache
# Config param learning_rate: 3e-05
# Config param learning_rate_schedule_steps: 30
# Config param load_from_prefill_dir: False
# Config param load_full_state_path: 
# Config param load_parameters_path: gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default
# Config param log_period: 100
# Config param logical_axis_rules: (('activation_batch', ('data', 'fsdp', 'fsdp_transpose')), ('activation_heads', ('tensor', 'sequence')), ('activation_length', 'sequence'), ('activation_embed', 'tensor'), ('activation_mlp', 'tensor'), ('activation_kv', 'tensor'), ('activation_vocab', ('tensor', 'sequence')), ('activation_vocab', 'tensor'), ('activation_vocab', 'sequence'), ('mlp', ('fsdp_transpose', 'tensor', 'autoregressive')), ('vocab', ('tensor', 'autoregressive')), ('embed', ('fsdp', 'fsdp_transpose', 'sequence')), ('embed', ('fsdp', 'sequence')), ('heads', ('tensor', 'autoregressive')), ('kv', ()), ('cache_batch', ()), ('cache_heads', ('autoregressive',)), ('cache_kv', ()), ('cache_sequence', ()))
# Config param logits_dot_in_fp32: True
# Config param logits_via_embedding: False
# Config param max_corpus_chars: 10000000
# Config param max_prefill_predict_length: 64
# Config param max_target_length: 2048
# Config param mesh_axes: ['data', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'autoregressive']
# Config param metrics_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/metrics/
# Config param metrics_file: 
# Config param mlp_activations: ['silu', 'linear']
# Config param mlp_dim: 11008
# Config param model_name: llama2-7b
# Config param normalization_layer_epsilon: 1e-05
# Config param normalize_embedding_logits: True
# Config param num_decoder_layers: 32
# Config param num_experts: 1
# Config param num_experts_per_tok: 1
# Config param num_kv_heads: 32
# Config param num_query_heads: 32
# Config param num_slices: 1
# Config param opt_type: adamw
# Config param param_scan_axis: 1
# Config param per_device_batch_size: 12.0
# Config param prefill_cache_dir: 
# Config param profiler_steps: 5
# Config param prompt: I love to
# Config param quantization: 
# Config param quantization_local_shard_count: 1
# Config param record_internal_nn_metrics: 0
# Config param remat_policy: full
# Config param reuse_example_batch: 0
# Config param run_name: yooh-llama-2024-03-06-06-16
# Config param save_config_to_gcs: False
# Config param scan_layers: True
# Config param skip_first_n_steps_for_profiler: 1
# Config param stack_trace_interval_seconds: 600
# Config param stack_trace_to_cloud: False
# Config param steps: 30
# Config param target_eval_loss: 0.0
# Config param tensorboard_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/tensorboard/
# Config param tokenizer_path: assets/tokenizer.llama2
# Config param trainable_position_size: -1
# Config param upload_all_profiler_results: False
# Config param use_iota_embed: False
# Config param use_untrainable_positional_embedding: False
# Config param vocab_size: 32000
# Config param warmup_steps_fraction: 0.1
# Config param weight_dtype: float32
# 2024-03-06 06:16:17.601566: I external/xla/xla/service/dump.cc:507] HloModule dump enabled with path prefix: , suffix: before_optimizations
# 2024-03-06 06:16:17.766521: I external/xla/xla/stream_executor/cuda/cuda_dnn.cc:517] Loaded cuDNN version 8907
# 2024-03-06 06:16:18.997294: W external/xla/xla/service/gpu/nvptx_compiler.cc:742] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.3.107). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
# I0306 06:16:19.961535 136871327533376 compilation_cache.py:213] Writing jit_convert_element_type to persistent compilation cache with key jit_convert_element_type-0f89eee2747ef7ba9074b605e660307809239d9f633e27134826f114900a4ba7.
# I0306 06:16:23.261688 136871327533376 compilation_cache.py:213] Writing jit__threefry_seed to persistent compilation cache with key jit__threefry_seed-cd17e7771358c6834450dd9bb7b126d39294a349694fc5d79bfbaf95228196c6.
# I0306 06:16:25.769990 136871327533376 compilation_cache.py:213] Writing jit_concatenate to persistent compilation cache with key jit_concatenate-2629693bee6d607a0db345df671748764d4b33f8c19c2d9f7e3dca80e4380a72.
# Creating checkpoint manager...
# I0306 06:16:33.668231 136871327533376 compilation_cache.py:213] Writing jit__psum to persistent compilation cache with key jit__psum-46046f29cf54a3ba134a23e142d7462c41f039624f1956db829fa8d326307d4b.
# Checkpoint manager created!
# Num_devices: 8, shape (1, 8, 1, 1, 1, 1)
# I0306 06:16:34.163375 136871327533376 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:16:34.590125 136871327533376 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 06:16:34.652602 136871327533376 reader.py:261] Creating a tf.data.Dataset reading 1024 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 06:16:34.752325 136871327533376 logging_logger.py:49] Constructing tf.data.Dataset c4 for split train, from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:16:34.914938 136871327533376 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:16:35.309167 136871327533376 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 06:16:35.359775 136871327533376 reader.py:261] Creating a tf.data.Dataset reading 8 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 06:16:35.380727 136871327533376 logging_logger.py:49] Constructing tf.data.Dataset c4 for split validation, from gs://maxtext-dataset/c4/en/3.0.1
# Tokenizer path: assets/tokenizer.llama2
# normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
# checkpoint manager exists so trying to load this run's existing checkpoint
# restoring params from load_parameters_from_path='gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default'
# I0306 06:16:36.689786 136871327533376 checkpointer.py:164] Restoring item from gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1709705797.219256     158 gcs_resource.cc:99] Using default AdmissionQueue with limit 32
# I0000 00:00:1709705797.223200    2094 google_auth_provider.cc:179] Running on GCE, using service account 455207029971-compute@developer.gserviceaccount.com
# W0306 06:17:20.557305 136871327533376 transform_utils.py:229] The transformations API will eventually be replaced by an upgraded design. The current API will not be removed until this point, but it will no longer be actively worked on.
# I0306 06:17:20.611339 136871327533376 checkpointer.py:167] Finished restoring checkpoint from gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default.
# I0306 06:17:25.942031 136871327533376 compilation_cache.py:213] Writing jit__unnamed_wrapped_function_ to persistent compilation cache with key jit__unnamed_wrapped_function_-0213e11eccdbfa895e48c3206056466a92e097dd94377dc7285c753c2713f48c.
# number parameters: 6.738 billion
# I0306 06:17:30.775450 136871327533376 compilation_cache.py:213] Writing jit_fold_in to persistent compilation cache with key jit_fold_in-1c7c03b448a22e4a29fcfba276db6b2892ce6c07c8405d4a01593c7e70e8b4b0.
# I0306 06:17:45.650760 136871327533376 compilation_cache.py:213] Writing jit_train_step to persistent compilation cache with key jit_train_step-3fa60b3014914e3da1ccf0142e6c85bec9edd2256d2c12399075e1a0a58b31da.
# I0306 06:18:04.678130 136871327533376 compilation_cache.py:213] Writing jit_clip to persistent compilation cache with key jit_clip-21ce239e1b51e1785c8c18c22488b0069bcc909916d3f1dd34b9123411d64175.
# I0306 06:18:07.047491 136871327533376 compilation_cache.py:213] Writing jit_true_divide to persistent compilation cache with key jit_true_divide-69687829a7c933910b48848cb9174734440d876495e59b7cb04f455db30df199.
# I0306 06:18:09.323183 136871327533376 compilation_cache.py:213] Writing jit__lambda_ to persistent compilation cache with key jit__lambda_-1623f24023e366d7c49df04ced286c3c9cc628c4436801b947c24fabee3ddafd.
# I0306 06:18:11.545138 136871327533376 compilation_cache.py:213] Writing jit_integer_pow to persistent compilation cache with key jit_integer_pow-d05e7c8a309bbc20faaaed3fd9b4ed0bb46cbba33bfc7dd817b8f4bd7aab2d28.
# I0306 06:18:13.739756 136871327533376 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-d5ff748d04c434c25c53e4dcc4406ec6205c845af9c00132218815d108bac983.
# I0306 06:18:16.021788 136871327533376 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-b4ab909d5a8c01561dfee05c65116260fc6cf2a952bb92a5926993e8f49830b1.
# I0306 06:18:18.216284 136871327533376 compilation_cache.py:213] Writing jit_cos to persistent compilation cache with key jit_cos-a3effe71916c93f625accd8bfeb73a11d8763fb3033e040857574c6cae309eeb.
# I0306 06:18:20.378414 136871327533376 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-a3943ce5b64e18899ae5b64e7c4bc2a1c83b313e48a49250b911dcbd2b61d796.
# I0306 06:18:22.566450 136871327533376 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-1fa587291c9270970fad238eddd3aa7f347cdfb3e548a0d022a9db8820a6e296.
# I0306 06:18:24.820323 136871327533376 compilation_cache.py:213] Writing jit__where to persistent compilation cache with key jit__where-812a199b1ff1a60b1343306ea0b79ab7adb30be9c898c7c8270c4c42d763f530.
# I0306 06:18:24.936837 136871327533376 checkpointer.py:133] Saving item to gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/checkpoints/0.
# W0306 06:18:26.010290 136871327533376 type_handlers.py:399] SaveArgs.aggregate is deprecated, please use custom TypeHandler (https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler) or contact Orbax team to migrate before May 1st, 2024.
# I0306 06:21:11.491789 136871327533376 utils.py:598] Finished saving checkpoint to `gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/checkpoints/0`.
# I0306 06:21:11.496040 136871327533376 checkpoint_manager.py:819] Finished synchronous save.
# Per train step, total TFLOPs will be 1033.20, split as 96.17% learnable weight flops and 3.83% attention flops
# saved a checkpoint at step 0
# completed step: 0, seconds: 36.240, TFLOP/s/device: 28.510, loss: 1.966
# To see full metrics 'tensorboard --logdir=gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-16/tensorboard/'
# completed step: 1, seconds: 192.846, TFLOP/s/device: 5.358, loss: 2.008
# completed step: 2, seconds: 3.033, TFLOP/s/device: 340.650, loss: 1.989
# completed step: 3, seconds: 3.565, TFLOP/s/device: 289.823, loss: 1.977
# completed step: 4, seconds: 3.964, TFLOP/s/device: 260.669, loss: 1.989
# Capture range started in the application.
# completed step: 5, seconds: 4.003, TFLOP/s/device: 258.112, loss: 1.909
# completed step: 6, seconds: 18.009, TFLOP/s/device: 57.370, loss: 1.922
# Capture range ended in the application.
# Generating '/tmp/nsys-report-f7f2.qdstrm'
# [1/1] [========================100%] nsys_profile.out.nsys-rep
# Generated:
#     /app/nsys_profile.out.nsys-rep
# completed step: 7, seconds: 3.580, TFLOP/s/device: 288.581, loss: 2.014
# completed step: 8, seconds: 10.231, TFLOP/s/device: 100.991, loss: 2.025
# completed step: 9, seconds: 3.952, TFLOP/s/device: 261.447, loss: 1.898
# completed step: 10, seconds: 4.027, TFLOP/s/device: 256.571, loss: 1.939
# completed step: 11, seconds: 3.990, TFLOP/s/device: 258.929, loss: 1.963
# completed step: 12, seconds: 4.027, TFLOP/s/device: 256.541, loss: 1.998
# completed step: 13, seconds: 4.011, TFLOP/s/device: 257.569, loss: 1.958
# completed step: 14, seconds: 4.201, TFLOP/s/device: 245.914, loss: 1.948
# completed step: 15, seconds: 4.178, TFLOP/s/device: 247.300, loss: 1.958
# completed step: 16, seconds: 4.036, TFLOP/s/device: 255.980, loss: 1.993
# completed step: 17, seconds: 4.006, TFLOP/s/device: 257.921, loss: 1.982
# completed step: 18, seconds: 4.224, TFLOP/s/device: 244.588, loss: 1.990
# completed step: 19, seconds: 3.993, TFLOP/s/device: 258.727, loss: 1.906
# completed step: 20, seconds: 4.117, TFLOP/s/device: 250.982, loss: 1.999
# completed step: 21, seconds: 4.400, TFLOP/s/device: 234.832, loss: 1.973
# completed step: 22, seconds: 4.016, TFLOP/s/device: 257.261, loss: 1.957
# completed step: 23, seconds: 3.894, TFLOP/s/device: 265.329, loss: 1.928
# completed step: 24, seconds: 4.220, TFLOP/s/device: 244.812, loss: 1.998
# completed step: 25, seconds: 3.800, TFLOP/s/device: 271.870, loss: 1.906
# completed step: 26, seconds: 4.019, TFLOP/s/device: 257.061, loss: 2.019
# completed step: 27, seconds: 3.898, TFLOP/s/device: 265.065, loss: 2.002
# completed step: 28, seconds: 3.823, TFLOP/s/device: 270.241, loss: 1.937
# completed step: 29, seconds: 3.889, TFLOP/s/device: 265.695, loss: 1.857