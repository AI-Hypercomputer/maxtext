#!/bin/bash 

# Build and upload image
# bash docker_build_dependency_image.sh DEVICE=gpu LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# docker push gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=yooh/maxtext-tcpx

# Clone Yuwei's XPK branch
git clone -b yangyuwei-xpk-gpu https://github.com/google/xpk.git

# Write env file
cat << EOF > xpk/env1.txt
export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-gpt-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true"
EOF

python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-gpt-$(date +%Y-%m-%d-%H-%M) \
    --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=1 --env-file=xpk/env1.txt --priority=high \
    --command "python3 MaxText/train.py MaxText/configs/models/gpt3-22b.yml hardware=gpu per_device_batch_size=2\
        run_name=yooh-gpt-$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs \
        dataset_path=gs://maxtext-dataset steps=30 enable_checkpointing=False attention=dot_product"










# root@gke-maxtext-a3-20nod-maxtext-a3-20nod-92d4d6cc-nnf7:/app# python3 MaxText/train.py MaxText/configs/models/gpt3-22b.yml hardware=gpu per_device_batch_size=1\
#         run_name=yooh-gpt-$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs \
#         dataset_path=gs://maxtext-dataset steps=30 enable_checkpointing=False attention=dot_product
# 2024-03-06 18:29:24.687214: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# 2024-03-06 18:29:26.039328: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1960] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
# Updating keys from env and command line: ['per_device_batch_size', 'hardware', 'run_name', 'enable_checkpointing', 'attention', 'base_output_directory', 'dataset_path', 'steps']
# Running Model: default
# Updating keys from model: []
# Attempting to initialize the jax distributed system for GPU backend...
# Jax distributed system initialized on GPU!
# I0306 18:29:27.479860 135886601294464 xla_bridge.py:704] Unable to initialize backend 'rocm': NOT_FOUND: Could not find registered platform with name: "rocm". Available platform names are: CUDA
# I0306 18:29:27.480640 135886601294464 xla_bridge.py:704] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
# System Information: Jax Version: 0.4.25
# System Information: Jaxlib Version: 0.4.25
# System Information: Jax Backend: cuda 12030
# Config param adam_b1: 0.9
# Config param adam_b2: 0.95
# Config param adam_eps: 1e-08
# Config param adam_eps_root: 0.0
# Config param adam_weight_decay: 0.1
# Config param async_checkpointing: True
# Config param attention: dot_product
# Config param autoregressive_decode_assert: 
# Config param base_emb_dim: 6144
# Config param base_mlp_dim: 24576
# Config param base_num_decoder_layers: 48
# Config param base_num_kv_heads: 24
# Config param base_num_query_heads: 24
# Config param base_output_directory: gs://runner-maxtext-logs
# Config param checkpoint_dir: gs://runner-maxtext-logs/yooh-gpt-2024-03-06-18-29/checkpoints/
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
# Config param decoder_block: gpt3
# Config param dropout_rate: 0
# Config param dtype: bfloat16
# Config param emb_dim: 6144
# Config param enable_checkpointing: False
# Config param enable_data_shuffling: True
# Config param enable_dropout: False
# Config param enable_profiler: False
# Config param eval_dataset_name: c4/en:3.0.1
# Config param eval_interval: -1
# Config param eval_per_device_batch_size: 0
# Config param eval_split: validation
# Config param force_unroll: False
# Config param fused_mlp: False
# Config param fused_qkv: True
# Config param gcs_metrics: False
# Config param global_batch_size_to_load: 8
# Config param global_batch_size_to_train_on: 8
# Config param global_parameter_scale: 1
# Config param gradient_clipping_threshold: 1.0
# Config param grain_worker_count: 4
# Config param hardware: gpu
# Config param head_dim: 256
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
# Config param load_parameters_path: 
# Config param log_period: 100
# Config param logical_axis_rules: (('activation_batch', ('data', 'fsdp', 'fsdp_transpose')), ('activation_heads', ('tensor', 'sequence')), ('activation_length', 'sequence'), ('activation_embed', 'tensor'), ('activation_mlp', 'tensor'), ('activation_kv', 'tensor'), ('activation_vocab', ('tensor', 'sequence')), ('activation_vocab', 'tensor'), ('activation_vocab', 'sequence'), ('mlp', ('fsdp_transpose', 'tensor', 'autoregressive')), ('vocab', ('tensor', 'autoregressive')), ('embed', ('fsdp', 'fsdp_transpose', 'sequence')), ('embed', ('fsdp', 'sequence')), ('heads', ('tensor', 'autoregressive')), ('kv', ()), ('cache_batch', ()), ('cache_heads', ('autoregressive',)), ('cache_kv', ()), ('cache_sequence', ()))
# Config param logits_dot_in_fp32: False
# Config param logits_via_embedding: True
# Config param max_corpus_chars: 10000000
# Config param max_prefill_predict_length: 64
# Config param max_target_length: 1024
# Config param mesh_axes: ['data', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'autoregressive']
# Config param metrics_dir: gs://runner-maxtext-logs/yooh-gpt-2024-03-06-18-29/metrics/
# Config param metrics_file: 
# Config param mlp_activations: ['gelu']
# Config param mlp_dim: 24576
# Config param model_name: default
# Config param normalization_layer_epsilon: 1e-05
# Config param normalize_embedding_logits: False
# Config param num_decoder_layers: 48
# Config param num_experts: 1
# Config param num_experts_per_tok: 1
# Config param num_kv_heads: 24
# Config param num_query_heads: 24
# Config param num_slices: 1
# Config param opt_type: adam_pax
# Config param param_scan_axis: 1
# Config param per_device_batch_size: 1.0
# Config param prefill_cache_dir: 
# Config param profiler_steps: 5
# Config param prompt: I love to
# Config param quantization: 
# Config param quantization_local_shard_count: 1
# Config param record_internal_nn_metrics: 0
# Config param remat_policy: full
# Config param reuse_example_batch: 0
# Config param run_name: yooh-gpt-2024-03-06-18-29
# Config param save_config_to_gcs: False
# Config param scan_layers: True
# Config param skip_first_n_steps_for_profiler: 1
# Config param stack_trace_interval_seconds: 600
# Config param stack_trace_to_cloud: False
# Config param steps: 30
# Config param target_eval_loss: 0.0
# Config param tensorboard_dir: gs://runner-maxtext-logs/yooh-gpt-2024-03-06-18-29/tensorboard/
# Config param tokenizer_path: assets/tokenizer
# Config param trainable_position_size: 16384
# Config param upload_all_profiler_results: False
# Config param use_iota_embed: True
# Config param use_untrainable_positional_embedding: False
# Config param vocab_size: 32768
# Config param warmup_steps_fraction: 0.1
# Config param weight_dtype: float32
# 2024-03-06 18:29:27.607939: I external/xla/xla/stream_executor/cuda/cuda_dnn.cc:517] Loaded cuDNN version 8907
# 2024-03-06 18:29:27.612422: W external/xla/xla/service/gpu/nvptx_compiler.cc:742] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.3.107). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
# Checkpointing disabled, not creating checkpoint manager.
# Num_devices: 8, shape (1, 8, 1, 1, 1, 1)
# I0306 18:29:28.831793 135886601294464 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 18:29:29.321984 135886601294464 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 18:29:29.376520 135886601294464 reader.py:261] Creating a tf.data.Dataset reading 1024 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 18:29:29.458355 135886601294464 logging_logger.py:49] Constructing tf.data.Dataset c4 for split train, from gs://maxtext-dataset/c4/en/3.0.1
# I0306 18:29:29.615486 135886601294464 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 18:29:30.031535 135886601294464 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 18:29:30.080556 135886601294464 reader.py:261] Creating a tf.data.Dataset reading 8 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 18:29:30.100734 135886601294464 logging_logger.py:49] Constructing tf.data.Dataset c4 for split validation, from gs://maxtext-dataset/c4/en/3.0.1
# Tokenizer path: assets/tokenizer
# No existing checkpoints found, not restoring checkpoint.
# I0306 18:29:33.250982 135886601294464 compilation_cache.py:213] Writing jit__unnamed_wrapped_function_ to persistent compilation cache with key jit__unnamed_wrapped_function_-95b8e6e58ee29df9e8baa0ff64b0ab3bbbda9dfd64af90493e2b053fe2607f2c.
# number parameters: 22.049 billion
# Per train step, total TFLOPs will be 137.33, split as 98.65% learnable weight flops and 1.35% attention flops
# /usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:914: UserWarning: Some donated buffers were not usable: ShapedArray(bfloat16[768,48,24576]), ShapedArray(bfloat16[24576,48,768]), ShapedArray(bfloat16[24,48,256,768]), ShapedArray(bfloat16[768,48,3,24,256]), ShapedArray(bfloat16[768,48,24576]), ShapedArray(bfloat16[24576,48,768]), ShapedArray(bfloat16[24,48,256,768]), ShapedArray(bfloat16[768,48,3,24,256]).
# See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
#   warnings.warn("Some donated buffers were not usable:"
# 2024-03-06 18:29:46.880537: W external/xla/xla/service/hlo_rematerialization.cc:2941] Can't reduce memory use below 22.27GiB (23917347918 bytes) by rematerialization; only reduced to 37.90GiB (40693879420 bytes), down from 38.20GiB (41022266148 bytes) originally
# I0306 18:29:51.448759 135886601294464 compilation_cache.py:213] Writing jit_train_step to persistent compilation cache with key jit_train_step-13d6f1d7eb05a9bb9bc98344010e3651b11d743c295a92ac6687aecb313a9f60.
# 2024-03-06 18:30:07.422282: W external/xla/xla/service/hlo_rematerialization.cc:2941] Can't reduce memory use below 31.89GiB (34245402088 bytes) by rematerialization; only reduced to 37.90GiB (40693837400 bytes), down from 38.20GiB (41021187800 bytes) originally
# I0306 18:30:10.862313 135886601294464 compilation_cache.py:213] Writing jit_train_step to persistent compilation cache with key jit_train_step-3b91bf09fb9e4536e5b92db2fe278de9c9e607cd8b943aab52edafe1dde77056.
# completed step: 0, seconds: 27.762, TFLOP/s/device: 4.947, loss: 875.434
# To see full metrics 'tensorboard --logdir=gs://runner-maxtext-logs/yooh-gpt-2024-03-06-18-29/tensorboard/'
# completed step: 1, seconds: 5.524, TFLOP/s/device: 24.860, loss: 871.627
# completed step: 2, seconds: 1.147, TFLOP/s/device: 119.678, loss: 528.254
# completed step: 3, seconds: 0.771, TFLOP/s/device: 178.149, loss: 322.881
# completed step: 4, seconds: 0.770, TFLOP/s/device: 178.318, loss: 272.197
# completed step: 5, seconds: 0.770, TFLOP/s/device: 178.236, loss: 259.840
# completed step: 6, seconds: 1.044, TFLOP/s/device: 131.546, loss: 247.975
# completed step: 7, seconds: 1.589, TFLOP/s/device: 86.411, loss: 253.525
# completed step: 8, seconds: 1.303, TFLOP/s/device: 105.405, loss: 268.391
# completed step: 9, seconds: 1.305, TFLOP/s/device: 105.202, loss: 255.067
# completed step: 10, seconds: 1.002, TFLOP/s/device: 137.025, loss: 251.749
# completed step: 11, seconds: 1.204, TFLOP/s/device: 114.093, loss: 244.524
# completed step: 12, seconds: 1.404, TFLOP/s/device: 97.780, loss: 249.867
# completed step: 13, seconds: 1.203, TFLOP/s/device: 114.130, loss: 243.242
# completed step: 14, seconds: 1.202, TFLOP/s/device: 114.244, loss: 249.613
# completed step: 15, seconds: 1.104, TFLOP/s/device: 124.430, loss: 239.558
# completed step: 16, seconds: 1.103, TFLOP/s/device: 124.501, loss: 220.819
# completed step: 17, seconds: 1.302, TFLOP/s/device: 105.448, loss: 226.801
# completed step: 18, seconds: 1.104, TFLOP/s/device: 124.402, loss: 225.090
# completed step: 19, seconds: 1.305, TFLOP/s/device: 105.217, loss: 223.080
# completed step: 20, seconds: 1.204, TFLOP/s/device: 114.023, loss: 210.506
# completed step: 21, seconds: 1.303, TFLOP/s/device: 105.381, loss: 204.511
# completed step: 22, seconds: 1.304, TFLOP/s/device: 105.334, loss: 204.759
# completed step: 23, seconds: 1.105, TFLOP/s/device: 124.319, loss: 197.995
# completed step: 24, seconds: 1.104, TFLOP/s/device: 124.390, loss: 195.862
# completed step: 25, seconds: 1.204, TFLOP/s/device: 114.089, loss: 201.334
# completed step: 26, seconds: 1.104, TFLOP/s/device: 124.402, loss: 191.151
# completed step: 27, seconds: 1.404, TFLOP/s/device: 97.819, loss: 194.334
# completed step: 28, seconds: 1.505, TFLOP/s/device: 91.225, loss: 194.649
# completed step: 29, seconds: 1.505, TFLOP/s/device: 91.240, loss: 184.291
