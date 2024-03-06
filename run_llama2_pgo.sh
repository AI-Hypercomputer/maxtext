#!/bin/bash 

# Build and upload image
# bash docker_build_dependency_image.sh DEVICE=gpu LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# docker push gcr.io/supercomputer-testing/yooh/maxtext-tcpx:latest
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=yooh/maxtext-tcpx

# Clone Yuwei's XPK branch
git clone -b yangyuwei-xpk-gpu https://github.com/google/xpk.git

# Write env file
cat << EOF > xpk/env2.txt
export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-llama-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true --xla_gpu_pgle_profile_file_or_directory_path=profile.pbtxt"
EOF

python3 xpk/xpk.py workload create --cluster maxtext-a3-20nodes --workload yooh-llama-$(date +%Y-%m-%d-%H-%M) --command "nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default model_name=llama2-7b attention=dot_product async_checkpointing=False steps=30 hardware=gpu" --docker-image=gcr.io/supercomputer-testing/yooh/maxtext-tcpx --device-type=h100-80gb-8 --num-slices=1 --env-file=xpk/env2.txt --priority=high





# root@gke-maxtext-a3-20nod-maxtext-a3-20nod-92d4d6cc-6vff:/app# export XLA_FLAGS="--xla_dump_to=gs://runner-maxtext-logs/yooh-llama-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
#                 --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
#                 --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
#                 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
#                 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
#                 --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_all_to_all=true --xla_gpu_pgle_profile_file_or_directory_path=profile.pbtxt"
# nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 MaxText/train.py MaxText/configs/base.yml dataset_path=gs://maxtext-dataset base_output_directory=gs://runner-maxtext-logs load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default model_name=llama2-7b attention=dot_product async_checkpointing=False hardware=gpu steps=30 run_name=yooh-llama-$(date +%Y-%m-%d-%H-%M)
# 2024-03-06 06:26:42.468543: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# 2024-03-06 06:26:44.035032: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1960] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
# ['MaxText/train.py', 'MaxText/configs/base.yml', 'dataset_path=gs://maxtext-dataset', 'base_output_directory=gs://runner-maxtext-logs', 'load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default', 'model_name=llama2-7b', 'attention=dot_product', 'async_checkpointing=False', 'hardware=gpu', 'steps=30', 'run_name=yooh-llama-2024-03-06-06-26']
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
# I0306 06:26:45.559332 135530450430272 xla_bridge.py:704] Unable to initialize backend 'rocm': NOT_FOUND: Could not find registered platform with name: "rocm". Available platform names are: CUDA
# I0306 06:26:45.560552 135530450430272 xla_bridge.py:704] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: dlopen hook: 'libtpu.so': cannot open shared object file: No such file or directory
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
# Config param checkpoint_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/checkpoints/
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
# Config param metrics_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/metrics/
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
# Config param run_name: yooh-llama-2024-03-06-06-26
# Config param save_config_to_gcs: False
# Config param scan_layers: True
# Config param skip_first_n_steps_for_profiler: 1
# Config param stack_trace_interval_seconds: 600
# Config param stack_trace_to_cloud: False
# Config param steps: 30
# Config param target_eval_loss: 0.0
# Config param tensorboard_dir: gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/tensorboard/
# Config param tokenizer_path: assets/tokenizer.llama2
# Config param trainable_position_size: -1
# Config param upload_all_profiler_results: False
# Config param use_iota_embed: False
# Config param use_untrainable_positional_embedding: False
# Config param vocab_size: 32000
# Config param warmup_steps_fraction: 0.1
# Config param weight_dtype: float32
# 2024-03-06 06:26:46.451355: I external/xla/xla/service/dump.cc:507] HloModule dump enabled with path prefix: , suffix: before_optimizations
# 2024-03-06 06:26:46.595839: I external/xla/xla/stream_executor/cuda/cuda_dnn.cc:517] Loaded cuDNN version 8907
# 2024-03-06 06:26:47.055259: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:26:47.055369: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:26:47.055379: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_convert_element_type
# 2024-03-06 06:26:47.716879: W external/xla/xla/service/gpu/nvptx_compiler.cc:742] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.3.107). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
# I0306 06:26:48.603058 135530450430272 compilation_cache.py:213] Writing jit_convert_element_type to persistent compilation cache with key jit_convert_element_type-6002910eaf7a8da75702dc43b182f551754a1f85ffed9efc2f86445a6ae94db5.
# 2024-03-06 06:26:49.980380: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:26:49.980493: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:26:49.980504: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit__threefry_seed
# I0306 06:26:51.523830 135530450430272 compilation_cache.py:213] Writing jit__threefry_seed to persistent compilation cache with key jit__threefry_seed-3b3ee14c783e38df6153ddc916312f39c8e0b587f3ae86d14e4a3e4bf5dfaa08.
# 2024-03-06 06:26:52.195550: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:26:52.195655: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:26:52.195664: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_concatenate
# I0306 06:26:53.717827 135530450430272 compilation_cache.py:213] Writing jit_concatenate to persistent compilation cache with key jit_concatenate-d20232f0079a2321832d61fcc04b468c8b09d022d29c2adcc15f1ab1d641fe85.
# Creating checkpoint manager...
# 2024-03-06 06:26:59.882312: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:26:59.882425: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:26:59.882434: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit__psum
# I0306 06:27:01.370934 135530450430272 compilation_cache.py:213] Writing jit__psum to persistent compilation cache with key jit__psum-6f7e7527982f7e12ff369051ff4973124f99069bb2eb799ad0c0b9acd4b872dd.
# Checkpoint manager created!
# Num_devices: 8, shape (1, 8, 1, 1, 1, 1)
# I0306 06:27:01.849420 135530450430272 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:27:02.283119 135530450430272 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 06:27:02.344485 135530450430272 reader.py:261] Creating a tf.data.Dataset reading 1024 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 06:27:02.445581 135530450430272 logging_logger.py:49] Constructing tf.data.Dataset c4 for split train, from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:27:02.599652 135530450430272 dataset_info.py:610] Load dataset info from gs://maxtext-dataset/c4/en/3.0.1
# I0306 06:27:03.006815 135530450430272 dataset_info.py:702] For 'c4/en/3.0.1': fields info.[splits] differ on disk and in the code. Keeping the one from code.
# I0306 06:27:03.060315 135530450430272 reader.py:261] Creating a tf.data.Dataset reading 8 files located in folders: gs://maxtext-dataset/c4/en/3.0.1.
# I0306 06:27:03.082809 135530450430272 logging_logger.py:49] Constructing tf.data.Dataset c4 for split validation, from gs://maxtext-dataset/c4/en/3.0.1
# Tokenizer path: assets/tokenizer.llama2
# normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
# checkpoint manager exists so trying to load this run's existing checkpoint
# restoring params from load_parameters_from_path='gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default'
# I0306 06:27:04.436551 135530450430272 checkpointer.py:164] Restoring item from gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1709706424.935781    4137 gcs_resource.cc:99] Using default AdmissionQueue with limit 32
# I0000 00:00:1709706424.939854    6073 google_auth_provider.cc:179] Running on GCE, using service account 455207029971-compute@developer.gserviceaccount.com
# W0306 06:27:56.755874 135530450430272 transform_utils.py:229] The transformations API will eventually be replaced by an upgraded design. The current API will not be removed until this point, but it will no longer be actively worked on.
# I0306 06:27:56.954647 135530450430272 checkpointer.py:167] Finished restoring checkpoint from gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default.
# 2024-03-06 06:28:00.406426: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:00.406603: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:00.407029: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit__unnamed_wrapped_function_
# I0306 06:28:02.833588 135530450430272 compilation_cache.py:213] Writing jit__unnamed_wrapped_function_ to persistent compilation cache with key jit__unnamed_wrapped_function_-6d140cc2cea3676d14fbc0f61b4a5e1bb0973bb3df6065b2fac6868d4f33cf88.
# number parameters: 6.738 billion
# 2024-03-06 06:28:05.679104: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:05.679244: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:05.679266: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_fold_in
# I0306 06:28:07.523111 135530450430272 compilation_cache.py:213] Writing jit_fold_in to persistent compilation cache with key jit_fold_in-7de27f6fbed65dd90a61a8adc62619f2523e006eaa9aa235b54a11bb794202d6.
# 2024-03-06 06:28:17.185105: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:17.185280: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:17.185831: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name wrapped_transpose.10 not in module jit_train_step
# I0306 06:28:22.607545 135530450430272 compilation_cache.py:213] Writing jit_train_step to persistent compilation cache with key jit_train_step-5cd23515a7c311acde846b6a593454ebfacd431354d019469f57ec100c8243e9.
# 2024-03-06 06:28:39.966612: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:39.966783: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:39.966796: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_clip
# I0306 06:28:41.635436 135530450430272 compilation_cache.py:213] Writing jit_clip to persistent compilation cache with key jit_clip-c69b3250bbdbccca111cbc3b42840b277885b6820ee7615fb337dab9322906ed.
# 2024-03-06 06:28:42.283017: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:42.283179: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:42.283192: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_true_divide
# I0306 06:28:43.871892 135530450430272 compilation_cache.py:213] Writing jit_true_divide to persistent compilation cache with key jit_true_divide-9d405a09963c076064e2ac55bef06bd797467ff4a0a7f636cc17f766ab13fc94.
# 2024-03-06 06:28:44.490095: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:44.490223: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:44.490235: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit__lambda_
# I0306 06:28:46.085944 135530450430272 compilation_cache.py:213] Writing jit__lambda_ to persistent compilation cache with key jit__lambda_-2243300209c81a4d13ae46d3d07ecd02b9f69e3a6d1404edc17b627f9680a58b.
# 2024-03-06 06:28:46.715806: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:46.716043: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:46.716063: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_integer_pow
# I0306 06:28:48.226547 135530450430272 compilation_cache.py:213] Writing jit_integer_pow to persistent compilation cache with key jit_integer_pow-0aa754b98e38da7573313289fc4ec55296b12caa57523d5c5739ae81b857942d.
# 2024-03-06 06:28:48.872005: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:48.872159: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:48.872167: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_fn
# I0306 06:28:50.491325 135530450430272 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-f0f2977ccc95f95478418a76c0c1710319edaa337ca0c41b2298bd05cba37692.
# 2024-03-06 06:28:51.118640: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:51.118807: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:51.118820: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_fn
# I0306 06:28:52.730741 135530450430272 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-d2c6de4bd10f7073000e9085942f4726fa772805a52fc16ffdaeac3941c91603.
# 2024-03-06 06:28:53.354088: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:53.354324: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:53.354344: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_cos
# I0306 06:28:54.919690 135530450430272 compilation_cache.py:213] Writing jit_cos to persistent compilation cache with key jit_cos-55f753a95fb2bc7a95610f4b6cf26a84b6b3de9d6a635b922655532eeed875d2.
# 2024-03-06 06:28:55.556103: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:55.556257: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:55.556271: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_fn
# I0306 06:28:57.263653 135530450430272 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-ea9ee39c265bb3f3819d41fec9f883e17b143cd12e7a61c61e06eb7b951ae70c.
# 2024-03-06 06:28:57.918425: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:28:57.918591: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:28:57.918604: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit_fn
# I0306 06:28:59.457576 135530450430272 compilation_cache.py:213] Writing jit_fn to persistent compilation cache with key jit_fn-835c00c5b9e0ddd5f517431c2b8551060ae1ce969608ecf1edbe5bed3d9d422f.
# 2024-03-06 06:29:00.101329: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:567] Using PGLE profile from profile.pbtxt
# 2024-03-06 06:29:00.101482: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:711] Found profile, using profile guided latency estimator
# 2024-03-06 06:29:00.101496: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:714] PGLE profile may not applicable to the module, but will still be used : cost name all-gather-start not in module jit__where
# I0306 06:29:01.765120 135530450430272 compilation_cache.py:213] Writing jit__where to persistent compilation cache with key jit__where-31dc9a5ae9d70f98600c38d16821c2aefcd457a95641982dc23b904471a6ef53.
# I0306 06:29:01.874489 135530450430272 checkpointer.py:133] Saving item to gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/checkpoints/0.
# W0306 06:29:02.938695 135530450430272 type_handlers.py:399] SaveArgs.aggregate is deprecated, please use custom TypeHandler (https://orbax.readthedocs.io/en/latest/custom_handlers.html#typehandler) or contact Orbax team to migrate before May 1st, 2024.
# I0306 06:31:32.217803 135530450430272 utils.py:598] Finished saving checkpoint to `gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/checkpoints/0`.
# I0306 06:31:32.222628 135530450430272 checkpoint_manager.py:819] Finished synchronous save.
# Per train step, total TFLOPs will be 1033.20, split as 96.17% learnable weight flops and 3.83% attention flops
# saved a checkpoint at step 0
# completed step: 0, seconds: 36.350, TFLOP/s/device: 28.423, loss: 1.966
# To see full metrics 'tensorboard --logdir=gs://runner-maxtext-logs/yooh-llama-2024-03-06-06-26/tensorboard/'
# completed step: 1, seconds: 176.734, TFLOP/s/device: 5.846, loss: 2.008
# completed step: 2, seconds: 3.053, TFLOP/s/device: 338.396, loss: 1.989
# completed step: 3, seconds: 3.585, TFLOP/s/device: 288.187, loss: 1.977
# completed step: 4, seconds: 3.603, TFLOP/s/device: 286.754, loss: 1.989
# Capture range started in the application.
# completed step: 5, seconds: 3.587, TFLOP/s/device: 288.011, loss: 1.909
# completed step: 6, seconds: 19.463, TFLOP/s/device: 53.085, loss: 1.922
# Capture range ended in the application.
# Generating '/tmp/nsys-report-18dd.qdstrm'
# [1/1] [========================100%] nsys_profile.out.nsys-rep
# Generated:
#     /app/nsys_profile.out.nsys-rep
# completed step: 7, seconds: 3.604, TFLOP/s/device: 286.671, loss: 2.014
# completed step: 8, seconds: 10.688, TFLOP/s/device: 96.665, loss: 2.025
# completed step: 9, seconds: 4.110, TFLOP/s/device: 251.368, loss: 1.898
# completed step: 10, seconds: 4.029, TFLOP/s/device: 256.417, loss: 1.938
# completed step: 11, seconds: 4.197, TFLOP/s/device: 246.200, loss: 1.963
# completed step: 12, seconds: 4.416, TFLOP/s/device: 233.959, loss: 1.998
# completed step: 13, seconds: 4.022, TFLOP/s/device: 256.882, loss: 1.958
# completed step: 14, seconds: 4.495, TFLOP/s/device: 229.866, loss: 1.948
# completed step: 15, seconds: 4.109, TFLOP/s/device: 251.452, loss: 1.958
# completed step: 16, seconds: 4.221, TFLOP/s/device: 244.771, loss: 1.993
# completed step: 17, seconds: 4.311, TFLOP/s/device: 239.674, loss: 1.982
# completed step: 18, seconds: 4.123, TFLOP/s/device: 250.618, loss: 1.990
# completed step: 19, seconds: 4.107, TFLOP/s/device: 251.551, loss: 1.906
# completed step: 20, seconds: 4.312, TFLOP/s/device: 239.632, loss: 1.998
# completed step: 21, seconds: 4.309, TFLOP/s/device: 239.760, loss: 1.973
# completed step: 22, seconds: 4.315, TFLOP/s/device: 239.429, loss: 1.957
# completed step: 23, seconds: 4.099, TFLOP/s/device: 252.065, loss: 1.928
# completed step: 24, seconds: 4.417, TFLOP/s/device: 233.890, loss: 1.998
# completed step: 25, seconds: 4.205, TFLOP/s/device: 245.687, loss: 1.906
# completed step: 26, seconds: 4.511, TFLOP/s/device: 229.021, loss: 2.019
# completed step: 27, seconds: 4.199, TFLOP/s/device: 246.073, loss: 2.002
# completed step: 28, seconds: 4.422, TFLOP/s/device: 233.674, loss: 1.937
# completed step: 29, seconds: 4.301, TFLOP/s/device: 240.214, loss: 1.857