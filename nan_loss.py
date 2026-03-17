 pinakinchoudhary@t1v-n-dfaea27c-w-0  ~/custom-gemma   main ●  bash run.sh
I0317 09:59:16.236806 125344476638336 pyconfig.py:413] Config param act_quantization_calibration_method: absmax
I0317 09:59:16.237003 125344476638336 pyconfig.py:413] Config param activation_dropout_for_audio: 0.0
I0317 09:59:16.237056 125344476638336 pyconfig.py:413] Config param activation_function_for_audio: gelu
I0317 09:59:16.237078 125344476638336 pyconfig.py:413] Config param activations_in_float32: False
I0317 09:59:16.237096 125344476638336 pyconfig.py:413] Config param adam_b1: 0.9
I0317 09:59:16.237115 125344476638336 pyconfig.py:413] Config param adam_b2: 0.95
I0317 09:59:16.237132 125344476638336 pyconfig.py:413] Config param adam_eps: 1e-08
I0317 09:59:16.237149 125344476638336 pyconfig.py:413] Config param adam_eps_root: 0.0
I0317 09:59:16.237163 125344476638336 pyconfig.py:413] Config param adam_weight_decay: 0.1
I0317 09:59:16.237177 125344476638336 pyconfig.py:413] Config param adamw_mask: []
I0317 09:59:16.237191 125344476638336 pyconfig.py:413] Config param add_bos: True
I0317 09:59:16.237208 125344476638336 pyconfig.py:413] Config param add_eos: True
I0317 09:59:16.237224 125344476638336 pyconfig.py:413] Config param allow_split_physical_axes: False
I0317 09:59:16.237238 125344476638336 pyconfig.py:413] Config param ar_cache_axis_order: 1,2,0,3
I0317 09:59:16.237252 125344476638336 pyconfig.py:413] Config param async_checkpointing: True
I0317 09:59:16.237267 125344476638336 pyconfig.py:413] Config param async_scheduling: False
I0317 09:59:16.237281 125344476638336 pyconfig.py:413] Config param attention: flash
I0317 09:59:16.237294 125344476638336 pyconfig.py:413] Config param attention_bias: False
I0317 09:59:16.237309 125344476638336 pyconfig.py:413] Config param attention_dropout_for_audio: 0.0
I0317 09:59:16.237323 125344476638336 pyconfig.py:413] Config param attention_out: RematLocation.REMAT
I0317 09:59:16.237349 125344476638336 pyconfig.py:413] Config param attention_sink: False
I0317 09:59:16.237364 125344476638336 pyconfig.py:413] Config param attention_type: global
I0317 09:59:16.237378 125344476638336 pyconfig.py:413] Config param attn_logits_soft_cap: None
I0317 09:59:16.237392 125344476638336 pyconfig.py:413] Config param audio_path:
I0317 09:59:16.237407 125344476638336 pyconfig.py:413] Config param audio_placeholder: <|audio|>
I0317 09:59:16.237421 125344476638336 pyconfig.py:413] Config param autoregressive_decode_assert:
I0317 09:59:16.237439 125344476638336 pyconfig.py:413] Config param base_config: None
I0317 09:59:16.237454 125344476638336 pyconfig.py:413] Config param base_emb_dim: 2560
I0317 09:59:16.237468 125344476638336 pyconfig.py:413] Config param base_mlp_dim: 10240
I0317 09:59:16.237482 125344476638336 pyconfig.py:413] Config param base_moe_mlp_dim: 7168
I0317 09:59:16.237497 125344476638336 pyconfig.py:413] Config param base_num_decoder_layers: 36
I0317 09:59:16.237511 125344476638336 pyconfig.py:413] Config param base_num_kv_heads: 4
I0317 09:59:16.237525 125344476638336 pyconfig.py:413] Config param base_num_query_heads: 8
I0317 09:59:16.237539 125344476638336 pyconfig.py:413] Config param base_output_directory: /home/pinakinchoudhary/custom-gemma/tmp
I0317 09:59:16.237553 125344476638336 pyconfig.py:413] Config param batch_size: 1
I0317 09:59:16.237567 125344476638336 pyconfig.py:413] Config param batch_split_factor: 1
I0317 09:59:16.237581 125344476638336 pyconfig.py:413] Config param beta_fast: 32
I0317 09:59:16.237595 125344476638336 pyconfig.py:413] Config param beta_slow: 1
I0317 09:59:16.237608 125344476638336 pyconfig.py:413] Config param bwd_quantization_calibration_method: absmax
I0317 09:59:16.237622 125344476638336 pyconfig.py:413] Config param capacity_factor: -1.0
I0317 09:59:16.237637 125344476638336 pyconfig.py:413] Config param cast_logits_to_fp32: True
I0317 09:59:16.237652 125344476638336 pyconfig.py:413] Config param chat_template:
I0317 09:59:16.237666 125344476638336 pyconfig.py:413] Config param chat_template_path:
I0317 09:59:16.237679 125344476638336 pyconfig.py:413] Config param checkpoint_conversion_fn: None
I0317 09:59:16.237693 125344476638336 pyconfig.py:413] Config param checkpoint_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/checkpoints/
I0317 09:59:16.237707 125344476638336 pyconfig.py:413] Config param checkpoint_is_quantized: False
I0317 09:59:16.237722 125344476638336 pyconfig.py:413] Config param checkpoint_period: 10000
I0317 09:59:16.237735 125344476638336 pyconfig.py:413] Config param checkpoint_storage_concurrent_gb: 96
I0317 09:59:16.237749 125344476638336 pyconfig.py:413] Config param checkpoint_storage_target_data_file_size_bytes: 2147483648
I0317 09:59:16.237761 125344476638336 pyconfig.py:413] Config param checkpoint_storage_use_ocdbt: True
I0317 09:59:16.237775 125344476638336 pyconfig.py:413] Config param checkpoint_storage_use_zarr3: True
I0317 09:59:16.237788 125344476638336 pyconfig.py:413] Config param chips_per_vm: 4
I0317 09:59:16.237802 125344476638336 pyconfig.py:413] Config param chunk_attn_window_size: 0
I0317 09:59:16.237814 125344476638336 pyconfig.py:413] Config param collect_stack_trace: False
I0317 09:59:16.237828 125344476638336 pyconfig.py:413] Config param colocated_python_checkpointing: False
I0317 09:59:16.237841 125344476638336 pyconfig.py:413] Config param colocated_python_data_input: False
I0317 09:59:16.237855 125344476638336 pyconfig.py:413] Config param compile_topology:
I0317 09:59:16.237869 125344476638336 pyconfig.py:413] Config param compile_topology_num_slices: -1
I0317 09:59:16.237883 125344476638336 pyconfig.py:413] Config param compiled_trainstep_file:
I0317 09:59:16.237895 125344476638336 pyconfig.py:413] Config param compute_axis_order: 0,1,2,3
I0317 09:59:16.237909 125344476638336 pyconfig.py:413] Config param constant_bound_config: []
I0317 09:59:16.237922 125344476638336 pyconfig.py:413] Config param context: RematLocation.REMAT
I0317 09:59:16.237937 125344476638336 pyconfig.py:413] Config param context_parallel_load_balance: True
I0317 09:59:16.237951 125344476638336 pyconfig.py:413] Config param context_parallel_size: 1
I0317 09:59:16.237966 125344476638336 pyconfig.py:413] Config param context_parallel_strategy: all_gather
I0317 09:59:16.237978 125344476638336 pyconfig.py:413] Config param conv_chunksize_for_audio: 500
I0317 09:59:16.237991 125344476638336 pyconfig.py:413] Config param conv_stride_for_vit: 14
I0317 09:59:16.238003 125344476638336 pyconfig.py:413] Config param cost_estimate_flops_bwd: -1
I0317 09:59:16.238108 125344476638336 pyconfig.py:413] Config param cost_estimate_flops_fwd: -1
I0317 09:59:16.238124 125344476638336 pyconfig.py:413] Config param custom_mesh:
I0317 09:59:16.238136 125344476638336 pyconfig.py:413] Config param d_model_for_audio: 256
I0317 09:59:16.238148 125344476638336 pyconfig.py:413] Config param data_sharding: (('data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'context_autoregressive', 'tensor
', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive'),)
I0317 09:59:16.238167 125344476638336 pyconfig.py:413] Config param data_shuffle_seed: 0
I0317 09:59:16.238180 125344476638336 pyconfig.py:413] Config param dataset_name: c4/en:3.0.1
I0317 09:59:16.238193 125344476638336 pyconfig.py:413] Config param dataset_path:
I0317 09:59:16.238205 125344476638336 pyconfig.py:413] Config param dataset_type: DatasetType.GRAIN
I0317 09:59:16.238224 125344476638336 pyconfig.py:413] Config param dcn_autoregressive_parallelism: 1
I0317 09:59:16.238236 125344476638336 pyconfig.py:413] Config param dcn_context_autoregressive_parallelism: 1
I0317 09:59:16.238249 125344476638336 pyconfig.py:413] Config param dcn_context_parallelism: 1
I0317 09:59:16.238261 125344476638336 pyconfig.py:413] Config param dcn_data_parallelism: -1
I0317 09:59:16.238274 125344476638336 pyconfig.py:413] Config param dcn_diloco_parallelism: 1
I0317 09:59:16.238287 125344476638336 pyconfig.py:413] Config param dcn_expert_parallelism: 1
I0317 09:59:16.238299 125344476638336 pyconfig.py:413] Config param dcn_fsdp_parallelism: 1
I0317 09:59:16.238312 125344476638336 pyconfig.py:413] Config param dcn_fsdp_transpose_parallelism: 1
I0317 09:59:16.238324 125344476638336 pyconfig.py:413] Config param dcn_parallelism: [1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
I0317 09:59:16.238337 125344476638336 pyconfig.py:413] Config param dcn_pipeline_parallelism: 1
I0317 09:59:16.238350 125344476638336 pyconfig.py:413] Config param dcn_sequence_parallelism: 1
I0317 09:59:16.238362 125344476638336 pyconfig.py:413] Config param dcn_tensor_parallelism: 1
I0317 09:59:16.238375 125344476638336 pyconfig.py:413] Config param dcn_tensor_sequence_parallelism: 1
I0317 09:59:16.238387 125344476638336 pyconfig.py:413] Config param dcn_tensor_transpose_parallelism: 1
I0317 09:59:16.238399 125344476638336 pyconfig.py:413] Config param debug: {'rl': False}
I0317 09:59:16.238412 125344476638336 pyconfig.py:413] Config param debug_sharding: False
I0317 09:59:16.238425 125344476638336 pyconfig.py:413] Config param decode_sampling_nucleus_p: -1
I0317 09:59:16.238440 125344476638336 pyconfig.py:413] Config param decode_sampling_strategy: SamplingStrategy.GREEDY
I0317 09:59:16.238455 125344476638336 pyconfig.py:413] Config param decode_sampling_temperature: 1.0
I0317 09:59:16.238468 125344476638336 pyconfig.py:413] Config param decode_sampling_top_k: 0
I0317 09:59:16.238481 125344476638336 pyconfig.py:413] Config param decoder_block: DecoderBlockType.GEMMA3
I0317 09:59:16.238497 125344476638336 pyconfig.py:413] Config param decoder_layer_input: RematLocation.DEVICE
I0317 09:59:16.238513 125344476638336 pyconfig.py:413] Config param deepstack_visual_indexes_for_vit: []
I0317 09:59:16.238526 125344476638336 pyconfig.py:413] Config param diloco_outer_lr: 0.3
I0317 09:59:16.238539 125344476638336 pyconfig.py:413] Config param diloco_outer_momentum: 0.9
I0317 09:59:16.238552 125344476638336 pyconfig.py:413] Config param diloco_sync_period: 36
I0317 09:59:16.238564 125344476638336 pyconfig.py:413] Config param distill_alpha: 0.5
I0317 09:59:16.238577 125344476638336 pyconfig.py:413] Config param distill_beta: 0.0
I0317 09:59:16.238590 125344476638336 pyconfig.py:413] Config param distill_layer_indices: None
I0317 09:59:16.238602 125344476638336 pyconfig.py:413] Config param distill_temperature: 1.0
I0317 09:59:16.238615 125344476638336 pyconfig.py:413] Config param downsample_hidden_size_for_audio: 256
I0317 09:59:16.238628 125344476638336 pyconfig.py:413] Config param dpo_beta: 0.1
I0317 09:59:16.238642 125344476638336 pyconfig.py:413] Config param dpo_label_smoothing: 0.0
I0317 09:59:16.238655 125344476638336 pyconfig.py:413] Config param dq_reduction_steps: 0
I0317 09:59:16.238667 125344476638336 pyconfig.py:413] Config param dropout_rate: 0.0
I0317 09:59:16.238680 125344476638336 pyconfig.py:413] Config param dtype: bfloat16
I0317 09:59:16.238725 125344476638336 pyconfig.py:413] Config param dtype_mm: float32
I0317 09:59:16.238739 125344476638336 pyconfig.py:413] Config param dump_hlo: False
I0317 09:59:16.238751 125344476638336 pyconfig.py:413] Config param dump_hlo_delete_local_after: True
I0317 09:59:16.238764 125344476638336 pyconfig.py:413] Config param dump_hlo_gcs_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/xla_dump
I0317 09:59:16.238777 125344476638336 pyconfig.py:413] Config param dump_hlo_local_dir: /tmp/xla_dump/
I0317 09:59:16.238789 125344476638336 pyconfig.py:413] Config param dump_hlo_local_module_name: jit_train_step
I0317 09:59:16.238803 125344476638336 pyconfig.py:413] Config param dump_hlo_module_name: jit_train_step
I0317 09:59:16.238816 125344476638336 pyconfig.py:413] Config param dump_hlo_upload_all: False
I0317 09:59:16.238829 125344476638336 pyconfig.py:413] Config param dump_hlo_xla_flags:
I0317 09:59:16.238842 125344476638336 pyconfig.py:413] Config param dump_jaxpr: False
I0317 09:59:16.238856 125344476638336 pyconfig.py:413] Config param dump_jaxpr_delete_local_after: True
I0317 09:59:16.238868 125344476638336 pyconfig.py:413] Config param dump_jaxpr_gcs_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/jaxpr_dump
I0317 09:59:16.238881 125344476638336 pyconfig.py:413] Config param dump_jaxpr_local_dir: /tmp/jaxpr_dump/
I0317 09:59:16.238893 125344476638336 pyconfig.py:413] Config param dump_step: -1
I0317 09:59:16.238905 125344476638336 pyconfig.py:413] Config param emb_dim: 2560
I0317 09:59:16.238918 125344476638336 pyconfig.py:413] Config param enable_checkpoint_cloud_logger: False
I0317 09:59:16.238931 125344476638336 pyconfig.py:413] Config param enable_checkpointing: False
I0317 09:59:16.238944 125344476638336 pyconfig.py:413] Config param enable_continuous_checkpointing: False
I0317 09:59:16.238956 125344476638336 pyconfig.py:413] Config param enable_data_shuffling: True
I0317 09:59:16.238969 125344476638336 pyconfig.py:413] Config param enable_diloco: False
I0317 09:59:16.238981 125344476638336 pyconfig.py:413] Config param enable_dp_attention: False
I0317 09:59:16.238994 125344476638336 pyconfig.py:413] Config param enable_dropout: True
I0317 09:59:16.239017 125344476638336 pyconfig.py:413] Config param enable_emergency_checkpoint: False
I0317 09:59:16.239030 125344476638336 pyconfig.py:413] Config param enable_gcp_goodput_metrics: True
I0317 09:59:16.239042 125344476638336 pyconfig.py:413] Config param enable_gcp_step_deviation_metrics: True
I0317 09:59:16.239055 125344476638336 pyconfig.py:413] Config param enable_goodput_recording: False
I0317 09:59:16.239068 125344476638336 pyconfig.py:413] Config param enable_jax_profiler: False
I0317 09:59:16.239081 125344476638336 pyconfig.py:413] Config param enable_llm_inference_pool: False
I0317 09:59:16.239093 125344476638336 pyconfig.py:413] Config param enable_model_warmup: False
I0317 09:59:16.239105 125344476638336 pyconfig.py:413] Config param enable_multi_tier_checkpointing: False
I0317 09:59:16.239118 125344476638336 pyconfig.py:413] Config param enable_nnx: False
I0317 09:59:16.239131 125344476638336 pyconfig.py:413] Config param enable_orbax_v1: False
I0317 09:59:16.239143 125344476638336 pyconfig.py:413] Config param enable_padding_causal_mask: True
I0317 09:59:16.239156 125344476638336 pyconfig.py:413] Config param enable_pathways_goodput: False
I0317 09:59:16.239168 125344476638336 pyconfig.py:413] Config param enable_prefix_caching: False
I0317 09:59:16.239181 125344476638336 pyconfig.py:413] Config param enable_rampup_batch_size: False
I0317 09:59:16.239193 125344476638336 pyconfig.py:413] Config param enable_single_controller: False
I0317 09:59:16.239205 125344476638336 pyconfig.py:413] Config param enable_single_replica_ckpt_restoring: False
I0317 09:59:16.239218 125344476638336 pyconfig.py:413] Config param enable_tensorboard: True
I0317 09:59:16.239230 125344476638336 pyconfig.py:413] Config param enable_tunix_perf_metrics: False
I0317 09:59:16.239243 125344476638336 pyconfig.py:413] Config param encoder_attention_heads_for_audio: 4
I0317 09:59:16.239255 125344476638336 pyconfig.py:413] Config param encoder_ffn_dim_for_audio: 512
I0317 09:59:16.239268 125344476638336 pyconfig.py:413] Config param encoder_layers_for_audio: 2
I0317 09:59:16.239280 125344476638336 pyconfig.py:413] Config param engram: RematLocation.REMAT
I0317 09:59:16.239294 125344476638336 pyconfig.py:413] Config param engram_head_dim: 1280
I0317 09:59:16.239306 125344476638336 pyconfig.py:413] Config param engram_kernel_size: 4
I0317 09:59:16.239320 125344476638336 pyconfig.py:413] Config param engram_layers: []
I0317 09:59:16.239333 125344476638336 pyconfig.py:413] Config param engram_max_ngram_size: 3
I0317 09:59:16.239345 125344476638336 pyconfig.py:413] Config param engram_num_heads: 8
I0317 09:59:16.239357 125344476638336 pyconfig.py:413] Config param engram_seed: 0
I0317 09:59:16.239370 125344476638336 pyconfig.py:413] Config param engram_vocab_bases: []
I0317 09:59:16.239382 125344476638336 pyconfig.py:413] Config param eval_corr_lst: False
I0317 09:59:16.239395 125344476638336 pyconfig.py:413] Config param eval_data_columns: ['text']
I0317 09:59:16.239408 125344476638336 pyconfig.py:413] Config param eval_dataset_name: c4/en:3.0.1
I0317 09:59:16.239420 125344476638336 pyconfig.py:413] Config param eval_image_column: image
I0317 09:59:16.239436 125344476638336 pyconfig.py:413] Config param eval_interval: -1
I0317 09:59:16.239448 125344476638336 pyconfig.py:413] Config param eval_make_lst: False
I0317 09:59:16.239460 125344476638336 pyconfig.py:413] Config param eval_per_device_batch_size: 1
I0317 09:59:16.239473 125344476638336 pyconfig.py:413] Config param eval_sampling_strategy: greedy
I0317 09:59:16.239486 125344476638336 pyconfig.py:413] Config param eval_split: validation
I0317 09:59:16.239498 125344476638336 pyconfig.py:413] Config param eval_steps: -1
I0317 09:59:16.239510 125344476638336 pyconfig.py:413] Config param expansion_factor_real_data: -1.0
I0317 09:59:16.239523 125344476638336 pyconfig.py:413] Config param expert_shard_attention_option: fsdp
I0317 09:59:16.239535 125344476638336 pyconfig.py:413] Config param final_logits_soft_cap: None
I0317 09:59:16.239548 125344476638336 pyconfig.py:413] Config param first_num_dense_layers: 0
I0317 09:59:16.239560 125344476638336 pyconfig.py:413] Config param float32_logits: False
I0317 09:59:16.239573 125344476638336 pyconfig.py:413] Config param float32_qk_product: False
I0317 09:59:16.239584 125344476638336 pyconfig.py:413] Config param float32_weight_sum: True
I0317 09:59:16.239597 125344476638336 pyconfig.py:413] Config param force_q_layout: False
I0317 09:59:16.239609 125344476638336 pyconfig.py:413] Config param force_unroll: False
I0317 09:59:16.239621 125344476638336 pyconfig.py:413] Config param freeze_audio_encoder_params: True
I0317 09:59:16.239634 125344476638336 pyconfig.py:413] Config param freeze_vision_encoder_params: True
I0317 09:59:16.239646 125344476638336 pyconfig.py:413] Config param fused_mlp: False
I0317 09:59:16.239659 125344476638336 pyconfig.py:413] Config param fused_qkv: False
I0317 09:59:16.239671 125344476638336 pyconfig.py:413] Config param gcs_metrics: False
I0317 09:59:16.239684 125344476638336 pyconfig.py:413] Config param gdn_chunk_size: 64
I0317 09:59:16.239696 125344476638336 pyconfig.py:413] Config param gdn_conv_kernel_dim: 4
I0317 09:59:16.239708 125344476638336 pyconfig.py:413] Config param gdn_key_head_dim: 128
I0317 09:59:16.239721 125344476638336 pyconfig.py:413] Config param gdn_num_key_heads: 16
I0317 09:59:16.239733 125344476638336 pyconfig.py:413] Config param gdn_num_value_heads: 32
I0317 09:59:16.239747 125344476638336 pyconfig.py:413] Config param gdn_value_head_dim: 128
I0317 09:59:16.239759 125344476638336 pyconfig.py:413] Config param generate_padding_batch_eval: False
I0317 09:59:16.239771 125344476638336 pyconfig.py:413] Config param generate_padding_batch_train: False
I0317 09:59:16.239783 125344476638336 pyconfig.py:413] Config param generate_slice: v5e-16
I0317 09:59:16.239796 125344476638336 pyconfig.py:413] Config param generation_configs: {}
I0317 09:59:16.239809 125344476638336 pyconfig.py:413] Config param global_batch_size_to_eval_on: 8
I0317 09:59:16.239822 125344476638336 pyconfig.py:413] Config param global_batch_size_to_load: 8
I0317 09:59:16.239834 125344476638336 pyconfig.py:413] Config param global_batch_size_to_load_eval: 8
I0317 09:59:16.239846 125344476638336 pyconfig.py:413] Config param global_batch_size_to_load_increment: None
I0317 09:59:16.239859 125344476638336 pyconfig.py:413] Config param global_batch_size_to_load_start: None
I0317 09:59:16.239871 125344476638336 pyconfig.py:413] Config param global_batch_size_to_train_on: 8
I0317 09:59:16.239884 125344476638336 pyconfig.py:413] Config param global_parameter_scale: 1
I0317 09:59:16.239896 125344476638336 pyconfig.py:413] Config param global_rampup_samples: 500
I0317 09:59:16.239908 125344476638336 pyconfig.py:413] Config param goodput_upload_interval_seconds: 30
I0317 09:59:16.239921 125344476638336 pyconfig.py:413] Config param grad_dtype: float32
I0317 09:59:16.239976 125344476638336 pyconfig.py:413] Config param gradient_accumulation_steps: 1
I0317 09:59:16.239990 125344476638336 pyconfig.py:413] Config param gradient_clipping_threshold: 1.0
I0317 09:59:16.240003 125344476638336 pyconfig.py:413] Config param grain_data_source_max_workers: 16
I0317 09:59:16.240028 125344476638336 pyconfig.py:413] Config param grain_eval_files:
I0317 09:59:16.240042 125344476638336 pyconfig.py:413] Config param grain_file_type: arrayrecord
I0317 09:59:16.240055 125344476638336 pyconfig.py:413] Config param grain_num_threads: 16
I0317 09:59:16.240068 125344476638336 pyconfig.py:413] Config param grain_num_threads_eval: 16
I0317 09:59:16.240080 125344476638336 pyconfig.py:413] Config param grain_packing_type: best_fit
I0317 09:59:16.240093 125344476638336 pyconfig.py:413] Config param grain_per_worker_buffer_size: 1
I0317 09:59:16.240105 125344476638336 pyconfig.py:413] Config param grain_per_worker_buffer_size_eval: 1
I0317 09:59:16.240117 125344476638336 pyconfig.py:413] Config param grain_prefetch_buffer_size: 500
I0317 09:59:16.240130 125344476638336 pyconfig.py:413] Config param grain_prefetch_buffer_size_eval: 500
I0317 09:59:16.240142 125344476638336 pyconfig.py:413] Config param grain_ram_budget_mb: 1024
I0317 09:59:16.240155 125344476638336 pyconfig.py:413] Config param grain_train_files: /home/pinakinchoudhary/data/dclm/global-shard_01_of_10/local-shard_0_of_10/*.arrayrecord
I0317 09:59:16.240168 125344476638336 pyconfig.py:413] Config param grain_train_mixture_config_path:
I0317 09:59:16.240180 125344476638336 pyconfig.py:413] Config param grain_worker_count: 4
I0317 09:59:16.240193 125344476638336 pyconfig.py:413] Config param grain_worker_count_eval: 1
I0317 09:59:16.240205 125344476638336 pyconfig.py:413] Config param grpo_beta: 0.08
I0317 09:59:16.240218 125344476638336 pyconfig.py:413] Config param grpo_epsilon: 0.2
I0317 09:59:16.240231 125344476638336 pyconfig.py:413] Config param hardware: tpu
I0317 09:59:16.240244 125344476638336 pyconfig.py:413] Config param hbm_utilization_vllm: 0.72
I0317 09:59:16.240257 125344476638336 pyconfig.py:413] Config param head_dim: 256
I0317 09:59:16.240269 125344476638336 pyconfig.py:413] Config param heartbeat_reporting_interval_in_seconds: 5
I0317 09:59:16.240282 125344476638336 pyconfig.py:413] Config param hf_data_dir: None
I0317 09:59:16.240294 125344476638336 pyconfig.py:413] Config param hf_eval_files: None
I0317 09:59:16.240307 125344476638336 pyconfig.py:413] Config param hf_eval_split: None
I0317 09:59:16.240319 125344476638336 pyconfig.py:413] Config param hf_name: None
I0317 09:59:16.240333 125344476638336 pyconfig.py:413] Config param hf_path:
I0317 09:59:16.240346 125344476638336 pyconfig.py:413] Config param hf_train_files: None
I0317 09:59:16.240358 125344476638336 pyconfig.py:413] Config param hidden_size_for_vit: 1408
I0317 09:59:16.240370 125344476638336 pyconfig.py:413] Config param hide_profiler_step_metric: False
I0317 09:59:16.240383 125344476638336 pyconfig.py:413] Config param ici_autoregressive_parallelism: 1
I0317 09:59:16.240396 125344476638336 pyconfig.py:413] Config param ici_context_autoregressive_parallelism: 1
I0317 09:59:16.240409 125344476638336 pyconfig.py:413] Config param ici_context_parallelism: 1
I0317 09:59:16.240421 125344476638336 pyconfig.py:413] Config param ici_data_parallelism: 1
I0317 09:59:16.240436 125344476638336 pyconfig.py:413] Config param ici_diloco_parallelism: 1
I0317 09:59:16.240449 125344476638336 pyconfig.py:413] Config param ici_expert_parallelism: 1
I0317 09:59:16.240461 125344476638336 pyconfig.py:413] Config param ici_fsdp_parallelism: -1
I0317 09:59:16.240474 125344476638336 pyconfig.py:413] Config param ici_fsdp_transpose_parallelism: 1
I0317 09:59:16.240486 125344476638336 pyconfig.py:413] Config param ici_parallelism: [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
I0317 09:59:16.240500 125344476638336 pyconfig.py:413] Config param ici_pipeline_parallelism: 1
I0317 09:59:16.240513 125344476638336 pyconfig.py:413] Config param ici_sequence_parallelism: 1
I0317 09:59:16.240527 125344476638336 pyconfig.py:413] Config param ici_tensor_parallelism: 1
I0317 09:59:16.240539 125344476638336 pyconfig.py:413] Config param ici_tensor_sequence_parallelism: 1
I0317 09:59:16.240551 125344476638336 pyconfig.py:413] Config param ici_tensor_transpose_parallelism: 1
I0317 09:59:16.240564 125344476638336 pyconfig.py:413] Config param image_path:
I0317 09:59:16.240578 125344476638336 pyconfig.py:413] Config param image_placeholder: <|image|>
I0317 09:59:16.240590 125344476638336 pyconfig.py:413] Config param image_size_for_vit: 896
I0317 09:59:16.240602 125344476638336 pyconfig.py:413] Config param index_head_dim: 128
I0317 09:59:16.240616 125344476638336 pyconfig.py:413] Config param index_n_heads: 64
I0317 09:59:16.240628 125344476638336 pyconfig.py:413] Config param index_topk: 2048
I0317 09:59:16.240641 125344476638336 pyconfig.py:413] Config param indexer_loss_scaling_factor: 0.0
I0317 09:59:16.240654 125344476638336 pyconfig.py:413] Config param inference_benchmark_test: False
I0317 09:59:16.240666 125344476638336 pyconfig.py:413] Config param inference_metadata_file:
I0317 09:59:16.240679 125344476638336 pyconfig.py:413] Config param inference_microbenchmark_log_file_path:
I0317 09:59:16.240691 125344476638336 pyconfig.py:413] Config param inference_microbenchmark_loop_iters: 10
I0317 09:59:16.240703 125344476638336 pyconfig.py:413] Config param inference_microbenchmark_num_samples: [1, 2, 3, 4, 5]
I0317 09:59:16.240717 125344476638336 pyconfig.py:413] Config param inference_microbenchmark_prefill_lengths: 64,128,256,512,1024
I0317 09:59:16.240729 125344476638336 pyconfig.py:413] Config param inference_microbenchmark_stages: prefill,generate
I0317 09:59:16.240741 125344476638336 pyconfig.py:413] Config param inference_server: MaxtextInterleavedServer
I0317 09:59:16.240754 125344476638336 pyconfig.py:413] Config param inhomogeneous_layer_cycle_interval: 1
I0317 09:59:16.240766 125344476638336 pyconfig.py:413] Config param init_weights_seed: 0
I0317 09:59:16.240778 125344476638336 pyconfig.py:413] Config param input_data_sharding_logical_axes: ['activation_embed_and_logits_batch', 'activation_norm_length']
I0317 09:59:16.240791 125344476638336 pyconfig.py:413] Config param interleave_moe_layer_step: 1
I0317 09:59:16.240804 125344476638336 pyconfig.py:413] Config param intermediate_size_for_vit: 5632
I0317 09:59:16.240816 125344476638336 pyconfig.py:413] Config param internal_compile: False
I0317 09:59:16.240828 125344476638336 pyconfig.py:413] Config param internal_compile_num_devices: -1
I0317 09:59:16.240840 125344476638336 pyconfig.py:413] Config param jax_cache_dir: ~/jax_cache
I0317 09:59:16.240853 125344476638336 pyconfig.py:413] Config param jax_debug_log_modules:
I0317 09:59:16.240865 125344476638336 pyconfig.py:413] Config param jax_distributed_initialization_timeout: 300
I0317 09:59:16.240878 125344476638336 pyconfig.py:413] Config param jax_profiler_port: 9999
I0317 09:59:16.240890 125344476638336 pyconfig.py:413] Config param key_proj: RematLocation.REMAT
I0317 09:59:16.240903 125344476638336 pyconfig.py:413] Config param kv_cache_buffer: 256
I0317 09:59:16.240916 125344476638336 pyconfig.py:413] Config param kv_lora_rank: 512
I0317 09:59:16.240928 125344476638336 pyconfig.py:413] Config param kv_quant_axis: KvQuantAxis.HEADS_AND_DKV
I0317 09:59:16.240944 125344476638336 pyconfig.py:413] Config param kv_quant_dtype: int8
I0317 09:59:16.240957 125344476638336 pyconfig.py:413] Config param learning_rate: 0.0003
I0317 09:59:16.240971 125344476638336 pyconfig.py:413] Config param learning_rate_final_fraction: 0.1
I0317 09:59:16.240983 125344476638336 pyconfig.py:413] Config param learning_rate_schedule_steps: 50
I0317 09:59:16.240995 125344476638336 pyconfig.py:413] Config param load_balance_loss_weight: 0.0
I0317 09:59:16.241019 125344476638336 pyconfig.py:413] Config param load_checkpoint_only_once: False
I0317 09:59:16.241031 125344476638336 pyconfig.py:413] Config param load_from_prefill_dir: False
I0317 09:59:16.241044 125344476638336 pyconfig.py:413] Config param load_full_state_path:
I0317 09:59:16.241056 125344476638336 pyconfig.py:413] Config param load_parameters_path:
I0317 09:59:16.241069 125344476638336 pyconfig.py:413] Config param local_checkpoint_directory:
I0317 09:59:16.241081 125344476638336 pyconfig.py:413] Config param local_checkpoint_period: 0
I0317 09:59:16.241095 125344476638336 pyconfig.py:413] Config param local_rope_max_timescale: 10000
I0317 09:59:16.241108 125344476638336 pyconfig.py:413] Config param log_config: True
I0317 09:59:16.241121 125344476638336 pyconfig.py:413] Config param log_period: 100
I0317 09:59:16.241133 125344476638336 pyconfig.py:413] Config param logical_axis_rules: (('activation_batch', ('data', 'fsdp', 'fsdp_transpose', 'expert')), ('activation_batch_no_exp',
 ('data', 'fsdp', 'fsdp_transpose')), ('activation_embed_and_logits_batch', ('data', 'stage', 'fsdp', 'fsdp_transpose', 'expert')), ('activation_embed_and_logits_batch_sequence', ('dat
a', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')), ('activation_heads', ('tensor', 'tensor_transpose', 'sequence', 'tensor_sequence', 'autoregressive')), ('activ
ation_kv_heads', ('tensor', 'tensor_transpose', 'sequence', 'tensor_sequence')), ('activation_length', ('sequence', 'context', 'expert')), ('activation_length', ('context', 'expert')),
 ('activation_attn_length', ('sequence', 'context', 'expert')), ('activation_attn_length', ('context', 'expert')), ('activation_attn_length_no_exp', ('sequence', 'context')), ('activat
ion_attn_length_no_exp', ('context',)), ('activation_length_no_exp', ('sequence', 'context')), ('activation_length_no_exp', ('context',)), ('activation_norm_length', ('tensor_sequence'
, 'context', 'sequence')), ('activation_q_length', ('context', 'expert')), ('activation_q_length_no_exp', ('context',)), ('prefill_activation_length', ('sequence', 'context')), ('prefi
ll_activation_norm_length', ('tensor_sequence', 'context', 'sequence')), ('activation_kv_length', ()), ('activation_attn_embed', ('tensor', 'tensor_transpose')), ('activation_embed', (
'tensor', 'tensor_transpose')), ('activation_mlp', ('tensor', 'tensor_transpose', 'tensor_sequence')), ('activation_kv', ('tensor', 'tensor_transpose', 'tensor_sequence')), ('activatio
n_prefill_kv_batch', ('data', 'fsdp', 'fsdp_transpose', 'expert')), ('activation_kv_batch', ('data', 'fsdp', 'fsdp_transpose', 'expert')), ('activation_kv_batch_no_exp', ('data', 'fsdp
', 'fsdp_transpose')), ('activation_kv_head_dim', ('tensor', 'tensor_transpose', 'tensor_sequence')), ('activation_vocab', ('tensor', 'tensor_transpose', 'tensor_sequence')), ('activat
ion_vocab', ('tensor', 'tensor_transpose')), ('activation_vocab', 'tensor_sequence'), ('activation_vocab', ('sequence', 'context')), ('activation_stage', 'stage'), ('activation_exp', (
'expert',)), ('decode_batch', ('data', 'fsdp', 'fsdp_transpose', 'expert')), ('decode_length', ('sequence',)), ('mlp', ('fsdp_transpose', 'tensor', 'tensor_sequence', 'autoregressive')
), ('mlp_no_fsdp', ('tensor', 'tensor_sequence', 'autoregressive')), ('vocab', ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive')), ('heads', ('tensor', 'tensor_transp
ose', 'tensor_sequence', 'autoregressive')), ('q_heads', ('tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive')), ('kv_heads', ('tensor', 'tensor_transpose', 'tensor_seque
nce', 'autoregressive')), ('embed', ('fsdp', 'fsdp_transpose', 'sequence', 'tensor_transpose', 'context', 'expert')), ('embed', ('fsdp', 'sequence', 'tensor_transpose', 'context', 'exp
ert')), ('embed', ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')), ('embed', ('fsdp', 'sequence', 'context', 'expert')), ('embed_no_exp', ('fsdp', 'fsdp_transpose', 'seque
nce', 'tensor_transpose', 'context')), ('embed_no_exp', ('fsdp', 'sequence', 'tensor_transpose', 'context')), ('embed_no_exp', ('fsdp', 'fsdp_transpose', 'sequence', 'context')), ('emb
ed_no_exp', ('fsdp', 'sequence', 'context')), ('embed_tensor_transpose', ('tensor_transpose',)), ('q_lora', ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'tensor_transpose', 'exper
t')), ('q_lora', ('fsdp', 'sequence', 'context', 'tensor_transpose', 'expert')), ('q_lora', ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')), ('q_lora', ('fsdp', 'sequence'
, 'context', 'expert')), ('q_lora_up_proj', ()), ('kv_lora', ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'tensor_transpose', 'expert')), ('kv_lora', ('fsdp', 'sequence', 'context
', 'tensor_transpose', 'expert')), ('kv_lora', ('fsdp', 'fsdp_transpose', 'sequence', 'context', 'expert')), ('kv_lora', ('fsdp', 'sequence', 'context', 'expert')), ('kv_lora_up_proj',
 ()), ('norm', ('tensor', 'tensor_transpose')), ('layers', 'stage'), ('qkv', ()), ('kv', ()), ('kv_head_dim', ()), ('cache_batch_prefill', ()), ('cache_batch', ()), ('cache_heads_none'
, ()), ('cache_heads', ('autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence')), ('cache_heads', ('autoregressive', 'tensor', 'tensor_sequence')), ('cache_kv', ()), ('cache
_sequence', ()), ('exp', 'expert'), ('exp_with_fsdp', 'fsdp'), ('paged_kv_heads', ('tensor',)), ('num_pages', ()), ('tokens_per_page', ()), ('paged_kv_head_dim_size', ()), ('dense_laye
rs', ()), ('moe_layers', ()), ('engram_dim', ('tensor',)), ('mhc', ()), ('diloco', 'diloco'))
I0317 09:59:16.241237 125344476638336 pyconfig.py:413] Config param logits_dot_in_fp32: False
I0317 09:59:16.241252 125344476638336 pyconfig.py:413] Config param logits_via_embedding: True
I0317 09:59:16.241265 125344476638336 pyconfig.py:413] Config param lora_input_adapters_path:
I0317 09:59:16.241277 125344476638336 pyconfig.py:413] Config param loss_algo: grpo
I0317 09:59:16.241289 125344476638336 pyconfig.py:413] Config param lr_schedule_type: LearningRateScheduleType.WSD
I0317 09:59:16.241307 125344476638336 pyconfig.py:413] Config param managed_mldiagnostics: False
I0317 09:59:16.241320 125344476638336 pyconfig.py:413] Config param managed_mldiagnostics_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/managed-mldiagnostics
I0317 09:59:16.241333 125344476638336 pyconfig.py:413] Config param managed_mldiagnostics_run_group:
I0317 09:59:16.241345 125344476638336 pyconfig.py:413] Config param matmul_precision: MatmulPrecision.DEFAULT
I0317 09:59:16.241363 125344476638336 pyconfig.py:413] Config param max_checkify: False
I0317 09:59:16.241375 125344476638336 pyconfig.py:413] Config param max_corpus_chars: 10000000
I0317 09:59:16.241388 125344476638336 pyconfig.py:413] Config param max_num_batched_tokens: None
I0317 09:59:16.241400 125344476638336 pyconfig.py:413] Config param max_num_checkpoints_to_keep: None
I0317 09:59:16.241412 125344476638336 pyconfig.py:413] Config param max_num_images_per_example: -1
I0317 09:59:16.241425 125344476638336 pyconfig.py:413] Config param max_num_seqs: None
I0317 09:59:16.241440 125344476638336 pyconfig.py:413] Config param max_position_embeddings: 163840
I0317 09:59:16.241453 125344476638336 pyconfig.py:413] Config param max_prefill_predict_length: 64
I0317 09:59:16.241465 125344476638336 pyconfig.py:413] Config param max_sample_len_for_audio: 10000
I0317 09:59:16.241477 125344476638336 pyconfig.py:413] Config param max_segments_per_seq: -1
I0317 09:59:16.241490 125344476638336 pyconfig.py:413] Config param max_source_positions_for_audio: 1500
I0317 09:59:16.241503 125344476638336 pyconfig.py:413] Config param max_target_length: 4096
I0317 09:59:16.241515 125344476638336 pyconfig.py:413] Config param max_timescale_for_audio: 10000.0
I0317 09:59:16.241528 125344476638336 pyconfig.py:413] Config param megablox: True
I0317 09:59:16.241540 125344476638336 pyconfig.py:413] Config param merge_gating_gmm: False
I0317 09:59:16.241553 125344476638336 pyconfig.py:413] Config param mesh_axes: ['diloco', 'data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'context_autoregressive', 't
ensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive']
I0317 09:59:16.241568 125344476638336 pyconfig.py:413] Config param metrics_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/metrics/
I0317 09:59:16.241580 125344476638336 pyconfig.py:413] Config param metrics_file:
I0317 09:59:16.241593 125344476638336 pyconfig.py:413] Config param mhc_expansion_rate: 1
I0317 09:59:16.241605 125344476638336 pyconfig.py:413] Config param micro_batch_size: -1
I0317 09:59:16.241617 125344476638336 pyconfig.py:413] Config param micro_batch_size_to_eval_on: 8
I0317 09:59:16.241629 125344476638336 pyconfig.py:413] Config param micro_batch_size_to_train_on: 8
I0317 09:59:16.241642 125344476638336 pyconfig.py:413] Config param mla_kv: RematLocation.REMAT
I0317 09:59:16.241656 125344476638336 pyconfig.py:413] Config param mla_naive_kvcache: True
I0317 09:59:16.241668 125344476638336 pyconfig.py:413] Config param mla_q: RematLocation.REMAT
I0317 09:59:16.241681 125344476638336 pyconfig.py:413] Config param mlp_activations: ['silu', 'linear']
I0317 09:59:16.241694 125344476638336 pyconfig.py:413] Config param mlp_activations_limit: -1.0
I0317 09:59:16.241707 125344476638336 pyconfig.py:413] Config param mlp_bias: False
I0317 09:59:16.241719 125344476638336 pyconfig.py:413] Config param mlp_dim: 10240
I0317 09:59:16.241732 125344476638336 pyconfig.py:413] Config param mlpwi: RematLocation.REMAT
I0317 09:59:16.241745 125344476638336 pyconfig.py:413] Config param mlpwi_0: RematLocation.REMAT
I0317 09:59:16.241758 125344476638336 pyconfig.py:413] Config param mlpwi_1: RematLocation.REMAT
I0317 09:59:16.241770 125344476638336 pyconfig.py:413] Config param mlpwo: RematLocation.REMAT
I0317 09:59:16.241783 125344476638336 pyconfig.py:413] Config param moba: False
I0317 09:59:16.241795 125344476638336 pyconfig.py:413] Config param moba_chunk_size: 1024
I0317 09:59:16.241808 125344476638336 pyconfig.py:413] Config param moba_topk: 8
I0317 09:59:16.241821 125344476638336 pyconfig.py:413] Config param model_call_mode:
I0317 09:59:16.241833 125344476638336 pyconfig.py:413] Config param model_name: brahmai-4b
I0317 09:59:16.241846 125344476638336 pyconfig.py:413] Config param moe_fsdp_use_two_stage_all_gather: False
I0317 09:59:16.241858 125344476638336 pyconfig.py:413] Config param moe_mlp_dim: 7168
I0317 09:59:16.241870 125344476638336 pyconfig.py:413] Config param monitor_goodput: False
I0317 09:59:16.241882 125344476638336 pyconfig.py:413] Config param monitor_step_time_deviation: True
I0317 09:59:16.241895 125344476638336 pyconfig.py:413] Config param mrope_section: [24, 20, 20]
I0317 09:59:16.241908 125344476638336 pyconfig.py:413] Config param mscale: 1.0
I0317 09:59:16.241922 125344476638336 pyconfig.py:413] Config param mtc_data_parallelism: 0
I0317 09:59:16.241934 125344476638336 pyconfig.py:413] Config param mtp_eval_target_module: 0
I0317 09:59:16.241947 125344476638336 pyconfig.py:413] Config param mtp_loss_scaling_factor: 0.1
I0317 09:59:16.241960 125344476638336 pyconfig.py:413] Config param mtp_num_layers: 0
I0317 09:59:16.241972 125344476638336 pyconfig.py:413] Config param mu_dtype: float32
I0317 09:59:16.241993 125344476638336 pyconfig.py:413] Config param multi_sampling: False
I0317 09:59:16.242035 125344476638336 pyconfig.py:413] Config param multi_tier_checkpointing_backup_interval_minutes: 0
I0317 09:59:16.242049 125344476638336 pyconfig.py:413] Config param muon_beta: 0.95
I0317 09:59:16.242064 125344476638336 pyconfig.py:413] Config param muon_consistent_rms: None
I0317 09:59:16.242078 125344476638336 pyconfig.py:413] Config param muon_weight_decay: 0.0
I0317 09:59:16.242092 125344476638336 pyconfig.py:413] Config param n_routing_groups: -1
I0317 09:59:16.242105 125344476638336 pyconfig.py:413] Config param n_window_for_audio: 50
I0317 09:59:16.242119 125344476638336 pyconfig.py:413] Config param n_window_infer_for_audio: 800
I0317 09:59:16.242132 125344476638336 pyconfig.py:413] Config param nope_layer_interval: -1
I0317 09:59:16.242146 125344476638336 pyconfig.py:413] Config param norm_topk_prob: False
I0317 09:59:16.242159 125344476638336 pyconfig.py:413] Config param normalization_layer_epsilon: 1e-06
I0317 09:59:16.242173 125344476638336 pyconfig.py:413] Config param normalize_embedding_logits: True
I0317 09:59:16.242186 125344476638336 pyconfig.py:413] Config param num_attention_heads_for_vit: 16
I0317 09:59:16.242199 125344476638336 pyconfig.py:413] Config param num_batches: 4
I0317 09:59:16.242212 125344476638336 pyconfig.py:413] Config param num_channels_for_vit: 3
I0317 09:59:16.242225 125344476638336 pyconfig.py:413] Config param num_conv_layers_for_audio: 3
I0317 09:59:16.242237 125344476638336 pyconfig.py:413] Config param num_decoder_layers: 36
I0317 09:59:16.242249 125344476638336 pyconfig.py:413] Config param num_diloco_replicas: 1
I0317 09:59:16.242262 125344476638336 pyconfig.py:413] Config param num_epoch: 1
I0317 09:59:16.242275 125344476638336 pyconfig.py:413] Config param num_eval_passes: 1
I0317 09:59:16.242288 125344476638336 pyconfig.py:413] Config param num_experts: 1
I0317 09:59:16.242300 125344476638336 pyconfig.py:413] Config param num_experts_per_tok: 1
I0317 09:59:16.242313 125344476638336 pyconfig.py:413] Config param num_generations: 2
I0317 09:59:16.242325 125344476638336 pyconfig.py:413] Config param num_hidden_layers_for_vit: 34
I0317 09:59:16.242338 125344476638336 pyconfig.py:413] Config param num_iterations: 1
I0317 09:59:16.242350 125344476638336 pyconfig.py:413] Config param num_kv_heads: 4
I0317 09:59:16.242362 125344476638336 pyconfig.py:413] Config param num_layers_per_pipeline_stage: 1
I0317 09:59:16.242375 125344476638336 pyconfig.py:413] Config param num_mel_bins_for_audio: 128
I0317 09:59:16.242387 125344476638336 pyconfig.py:413] Config param num_pipeline_microbatches: -1
I0317 09:59:16.242400 125344476638336 pyconfig.py:413] Config param num_pipeline_repeats: -1
I0317 09:59:16.242412 125344476638336 pyconfig.py:413] Config param num_position_embeddings_for_vit: 1024
I0317 09:59:16.242425 125344476638336 pyconfig.py:413] Config param num_query_heads: 8
I0317 09:59:16.242440 125344476638336 pyconfig.py:413] Config param num_samplers_slices: -1
I0317 09:59:16.242453 125344476638336 pyconfig.py:413] Config param num_slices: 1
I0317 09:59:16.242465 125344476638336 pyconfig.py:413] Config param num_target_devices: 8
I0317 09:59:16.242478 125344476638336 pyconfig.py:413] Config param num_test_batches: 5
I0317 09:59:16.242491 125344476638336 pyconfig.py:413] Config param num_trainer_slices: -1
I0317 09:59:16.242504 125344476638336 pyconfig.py:413] Config param num_vocab_tiling: 1
I0317 09:59:16.242516 125344476638336 pyconfig.py:413] Config param opt_type: OptimizerType.ADAMW
I0317 09:59:16.242534 125344476638336 pyconfig.py:413] Config param optimize_mesh_for_tpu_v6e: False
I0317 09:59:16.242547 125344476638336 pyconfig.py:413] Config param optimizer_memory_host_offload: False
I0317 09:59:16.242559 125344476638336 pyconfig.py:413] Config param original_max_position_embeddings: 4096
I0317 09:59:16.242572 125344476638336 pyconfig.py:413] Config param out_hidden_size_for_vit: 512
I0317 09:59:16.242585 125344476638336 pyconfig.py:413] Config param out_proj: RematLocation.REMAT
I0317 09:59:16.242599 125344476638336 pyconfig.py:413] Config param output_dim_for_audio: 512
I0317 09:59:16.242611 125344476638336 pyconfig.py:413] Config param override_logical_axis_rules: False
I0317 09:59:16.242624 125344476638336 pyconfig.py:413] Config param override_model_config: False
I0317 09:59:16.242636 125344476638336 pyconfig.py:413] Config param packing: True
I0317 09:59:16.242649 125344476638336 pyconfig.py:413] Config param pagedattn_head_dim_alignment: 128
I0317 09:59:16.242661 125344476638336 pyconfig.py:413] Config param pagedattn_max_pages_per_group: -1
I0317 09:59:16.242674 125344476638336 pyconfig.py:413] Config param pagedattn_num_pages: 64
I0317 09:59:16.242686 125344476638336 pyconfig.py:413] Config param pagedattn_pages_per_compute_block: 4
I0317 09:59:16.242699 125344476638336 pyconfig.py:413] Config param pagedattn_tokens_per_page: 32
I0317 09:59:16.242712 125344476638336 pyconfig.py:413] Config param param_scan_axis: 1
I0317 09:59:16.242725 125344476638336 pyconfig.py:413] Config param parameter_memory_host_offload: False
I0317 09:59:16.242737 125344476638336 pyconfig.py:413] Config param partial_rotary_factor: 1.0
I0317 09:59:16.242750 125344476638336 pyconfig.py:413] Config param patch_size_for_vit: 14
I0317 09:59:16.242762 125344476638336 pyconfig.py:413] Config param penalty_incorrect_answer: -1.0
I0317 09:59:16.242774 125344476638336 pyconfig.py:413] Config param penalty_incorrect_format: -0.5
I0317 09:59:16.242788 125344476638336 pyconfig.py:413] Config param per_device_batch_size: 1
I0317 09:59:16.242800 125344476638336 pyconfig.py:413] Config param per_device_batch_size_increment: 2.0
I0317 09:59:16.242813 125344476638336 pyconfig.py:413] Config param per_device_batch_size_start: 4.0
I0317 09:59:16.242825 125344476638336 pyconfig.py:413] Config param pipeline_delay_activation_forwarding: False
I0317 09:59:16.242839 125344476638336 pyconfig.py:413] Config param pipeline_fsdp_ag_once: False
I0317 09:59:16.242851 125344476638336 pyconfig.py:413] Config param pipeline_fsdp_ag_per_repeat: False
I0317 09:59:16.242864 125344476638336 pyconfig.py:413] Config param pipeline_parallel_layers: 36
I0317 09:59:16.242877 125344476638336 pyconfig.py:413] Config param pixel_shuffle_ratio_for_vit: 0.5
I0317 09:59:16.242889 125344476638336 pyconfig.py:413] Config param posemb_type_for_vit: learn
I0317 09:59:16.242902 125344476638336 pyconfig.py:413] Config param position_id_per_seconds: 25
I0317 09:59:16.242915 125344476638336 pyconfig.py:413] Config param prefill_cache_axis_order: 1,2,0,3
I0317 09:59:16.242927 125344476638336 pyconfig.py:413] Config param prefill_cache_dir:
I0317 09:59:16.242940 125344476638336 pyconfig.py:413] Config param prefill_chunk_size: 256
I0317 09:59:16.242952 125344476638336 pyconfig.py:413] Config param prefill_slice: v5e-16
I0317 09:59:16.242965 125344476638336 pyconfig.py:413] Config param prefix_caching_dram_byte: 100000000000
I0317 09:59:16.242978 125344476638336 pyconfig.py:413] Config param prefix_caching_hbm_byte: 10000000000
I0317 09:59:16.242990 125344476638336 pyconfig.py:413] Config param profile_cleanly: True
I0317 09:59:16.243003 125344476638336 pyconfig.py:413] Config param profile_periodically_period: -1
I0317 09:59:16.243047 125344476638336 pyconfig.py:413] Config param profile_power_events: False
I0317 09:59:16.243061 125344476638336 pyconfig.py:413] Config param profiler: ProfilerType.NONE
I0317 09:59:16.243078 125344476638336 pyconfig.py:413] Config param profiler_steps: 5
I0317 09:59:16.243093 125344476638336 pyconfig.py:413] Config param projector_dropout_for_vit: 0.0
I0317 09:59:16.243107 125344476638336 pyconfig.py:413] Config param projector_input_dim_for_vit: 4096
I0317 09:59:16.243121 125344476638336 pyconfig.py:413] Config param projector_output_dim_for_vit: 4096
I0317 09:59:16.243135 125344476638336 pyconfig.py:413] Config param prometheus_port: 0
I0317 09:59:16.243149 125344476638336 pyconfig.py:413] Config param prompt: I love to
I0317 09:59:16.243161 125344476638336 pyconfig.py:413] Config param pure_nnx_decoder: False
I0317 09:59:16.243173 125344476638336 pyconfig.py:413] Config param q_lora_rank: 0
I0317 09:59:16.243186 125344476638336 pyconfig.py:413] Config param qk_clip_threshold: 100.0
I0317 09:59:16.243199 125344476638336 pyconfig.py:413] Config param qk_nope_head_dim: 128
I0317 09:59:16.243211 125344476638336 pyconfig.py:413] Config param qk_rope_head_dim: 64
I0317 09:59:16.243224 125344476638336 pyconfig.py:413] Config param qkv_proj: RematLocation.REMAT
I0317 09:59:16.243236 125344476638336 pyconfig.py:413] Config param quant_cfg_path:
I0317 09:59:16.243249 125344476638336 pyconfig.py:413] Config param quantization: QuantizationType.NONE
I0317 09:59:16.243266 125344476638336 pyconfig.py:413] Config param quantization_local_shard_count: 8
I0317 09:59:16.243279 125344476638336 pyconfig.py:413] Config param quantize_kvcache: False
I0317 09:59:16.243292 125344476638336 pyconfig.py:413] Config param query_proj: RematLocation.REMAT
I0317 09:59:16.243305 125344476638336 pyconfig.py:413] Config param ragged_block_size: 256
I0317 09:59:16.243318 125344476638336 pyconfig.py:413] Config param rampup_end_step: 0
I0317 09:59:16.243331 125344476638336 pyconfig.py:413] Config param rampup_samples_per_increment_to_load: None
I0317 09:59:16.243343 125344476638336 pyconfig.py:413] Config param reasoning_end_token: </reasoning>
I0317 09:59:16.243356 125344476638336 pyconfig.py:413] Config param reasoning_start_token: <reasoning>
I0317 09:59:16.243368 125344476638336 pyconfig.py:413] Config param record_internal_nn_metrics: 0
I0317 09:59:16.243382 125344476638336 pyconfig.py:413] Config param remat_policy: full
I0317 09:59:16.243394 125344476638336 pyconfig.py:413] Config param remat_policy_for_vit: minimal
I0317 09:59:16.243407 125344476638336 pyconfig.py:413] Config param replicate_quant_scale: False
I0317 09:59:16.243419 125344476638336 pyconfig.py:413] Config param replicator_backup_interval_minutes: 0
I0317 09:59:16.243435 125344476638336 pyconfig.py:413] Config param report_heartbeat_metric_for_gcp_monitoring: False
I0317 09:59:16.243448 125344476638336 pyconfig.py:413] Config param report_performance_metric_for_gcp_monitoring: False
I0317 09:59:16.243461 125344476638336 pyconfig.py:413] Config param reshape_q: False
I0317 09:59:16.243473 125344476638336 pyconfig.py:413] Config param return_log_prob: False
I0317 09:59:16.243486 125344476638336 pyconfig.py:413] Config param reuse_example_batch: 0
I0317 09:59:16.243499 125344476638336 pyconfig.py:413] Config param reward_exact_format_match: 3.0
I0317 09:59:16.243512 125344476638336 pyconfig.py:413] Config param reward_partial_format_match: 0.5
I0317 09:59:16.243525 125344476638336 pyconfig.py:413] Config param reward_ratio_guess_to_answer_high: 0.5
I0317 09:59:16.243538 125344476638336 pyconfig.py:413] Config param reward_ratio_guess_to_answer_low: 0.25
I0317 09:59:16.243551 125344476638336 pyconfig.py:413] Config param reward_white_space_format_match: 1.5
I0317 09:59:16.243564 125344476638336 pyconfig.py:413] Config param rl: {'num_generations': 2, 'num_iterations': 1, 'grpo_beta': 0.08, 'grpo_epsilon': 0.2, 'loss_algo': 'grpo'}
I0317 09:59:16.243579 125344476638336 pyconfig.py:413] Config param rollout_data_parallelism: -1
I0317 09:59:16.243591 125344476638336 pyconfig.py:413] Config param rollout_expert_parallelism: 1
I0317 09:59:16.243603 125344476638336 pyconfig.py:413] Config param rollout_tensor_parallelism: -1
I0317 09:59:16.243616 125344476638336 pyconfig.py:413] Config param rope_attention_scaling: False
I0317 09:59:16.243629 125344476638336 pyconfig.py:413] Config param rope_factor: 40
I0317 09:59:16.243642 125344476638336 pyconfig.py:413] Config param rope_interleave: True
I0317 09:59:16.243654 125344476638336 pyconfig.py:413] Config param rope_linear_scaling_factor: 8.0
I0317 09:59:16.243666 125344476638336 pyconfig.py:413] Config param rope_max_timescale: 1000000
I0317 09:59:16.243679 125344476638336 pyconfig.py:413] Config param rope_min_timescale: 1
I0317 09:59:16.243691 125344476638336 pyconfig.py:413] Config param rope_theta_for_vit: 10000
I0317 09:59:16.243704 125344476638336 pyconfig.py:413] Config param rope_truncate: True
I0317 09:59:16.243716 125344476638336 pyconfig.py:413] Config param rope_type: RopeType.DEFAULT
I0317 09:59:16.243729 125344476638336 pyconfig.py:413] Config param rope_use_scale: True
I0317 09:59:16.243742 125344476638336 pyconfig.py:413] Config param routed_bias: False
I0317 09:59:16.243754 125344476638336 pyconfig.py:413] Config param routed_bias_update_rate: 0.0
I0317 09:59:16.243767 125344476638336 pyconfig.py:413] Config param routed_scaling_factor: 1.0
I0317 09:59:16.243780 125344476638336 pyconfig.py:413] Config param routed_score_func:
I0317 09:59:16.243793 125344476638336 pyconfig.py:413] Config param run_name: custom-gemma-swa
I0317 09:59:16.243806 125344476638336 pyconfig.py:413] Config param sa_block_kv: 512
I0317 09:59:16.243818 125344476638336 pyconfig.py:413] Config param sa_block_kv_compute: 512
I0317 09:59:16.243831 125344476638336 pyconfig.py:413] Config param sa_block_kv_dkv: 512
I0317 09:59:16.243844 125344476638336 pyconfig.py:413] Config param sa_block_kv_dkv_compute: 512
I0317 09:59:16.243856 125344476638336 pyconfig.py:413] Config param sa_block_kv_dq: 512
I0317 09:59:16.243869 125344476638336 pyconfig.py:413] Config param sa_block_q: 512
I0317 09:59:16.243881 125344476638336 pyconfig.py:413] Config param sa_block_q_dkv: 512
I0317 09:59:16.243894 125344476638336 pyconfig.py:413] Config param sa_block_q_dq: 512
I0317 09:59:16.243906 125344476638336 pyconfig.py:413] Config param sa_k_layout: HEAD_DIM_MINOR
I0317 09:59:16.243920 125344476638336 pyconfig.py:413] Config param sa_q_layout: HEAD_DIM_MINOR
I0317 09:59:16.243932 125344476638336 pyconfig.py:413] Config param sa_use_fused_bwd_kernel: False
I0317 09:59:16.243945 125344476638336 pyconfig.py:413] Config param sa_v_layout: HEAD_DIM_MINOR
I0317 09:59:16.243958 125344476638336 pyconfig.py:413] Config param sampler_devices_fraction: 0.5
I0317 09:59:16.243972 125344476638336 pyconfig.py:413] Config param save_checkpoint_on_completion: True
I0317 09:59:16.243985 125344476638336 pyconfig.py:413] Config param save_config_to_gcs: False
I0317 09:59:16.243997 125344476638336 pyconfig.py:413] Config param save_quantized_params_path:
I0317 09:59:16.244062 125344476638336 pyconfig.py:413] Config param scale_embedding_for_audio: True
I0317 09:59:16.244075 125344476638336 pyconfig.py:413] Config param scan_layers: True
I0317 09:59:16.244088 125344476638336 pyconfig.py:413] Config param scan_layers_per_stage: False
I0317 09:59:16.244100 125344476638336 pyconfig.py:413] Config param scan_pipeline_iterations: True
I0317 09:59:16.244112 125344476638336 pyconfig.py:413] Config param scan_pipeline_repeats: True
I0317 09:59:16.244125 125344476638336 pyconfig.py:413] Config param set_remat_policy_on_layers_per_stage: False
I0317 09:59:16.244137 125344476638336 pyconfig.py:413] Config param set_remat_policy_on_pipeline_iterations: True
I0317 09:59:16.244149 125344476638336 pyconfig.py:413] Config param sft_train_on_completion_only: False
I0317 09:59:16.244162 125344476638336 pyconfig.py:413] Config param shard_exp_on_fsdp: False
I0317 09:59:16.244174 125344476638336 pyconfig.py:413] Config param shard_mode: ShardMode.AUTO
I0317 09:59:16.244191 125344476638336 pyconfig.py:413] Config param shard_optimizer_over_data: False
I0317 09:59:16.244204 125344476638336 pyconfig.py:413] Config param sharding_strategy: None
I0317 09:59:16.244216 125344476638336 pyconfig.py:413] Config param sharding_tolerance: 0.02
I0317 09:59:16.244230 125344476638336 pyconfig.py:413] Config param shardy: True
I0317 09:59:16.244242 125344476638336 pyconfig.py:413] Config param share_kv_projections: False
I0317 09:59:16.244254 125344476638336 pyconfig.py:413] Config param shared_experts: 1
I0317 09:59:16.244267 125344476638336 pyconfig.py:413] Config param sinkhorn_iterations: 20
I0317 09:59:16.244279 125344476638336 pyconfig.py:413] Config param skip_first_n_steps_for_profiler: 1
I0317 09:59:16.244292 125344476638336 pyconfig.py:413] Config param skip_jax_distributed_system: False
I0317 09:59:16.244304 125344476638336 pyconfig.py:413] Config param sliding_window_size: 1024
I0317 09:59:16.244317 125344476638336 pyconfig.py:413] Config param solution_end_token: </answer>
I0317 09:59:16.244330 125344476638336 pyconfig.py:413] Config param solution_start_token: <answer>
I0317 09:59:16.244343 125344476638336 pyconfig.py:413] Config param source_checkpoint_layout: orbax
I0317 09:59:16.244356 125344476638336 pyconfig.py:413] Config param sparse_indexer_loss: False
I0317 09:59:16.244368 125344476638336 pyconfig.py:413] Config param sparse_matmul: True
I0317 09:59:16.244380 125344476638336 pyconfig.py:413] Config param spatial_merge_size_for_vit: 2
I0317 09:59:16.244392 125344476638336 pyconfig.py:413] Config param stack_prefill_result_cache: False
I0317 09:59:16.244405 125344476638336 pyconfig.py:413] Config param stack_trace_interval_seconds: 600
I0317 09:59:16.244418 125344476638336 pyconfig.py:413] Config param stack_trace_to_cloud: False
I0317 09:59:16.244435 125344476638336 pyconfig.py:413] Config param step_deviation_interval_seconds: 30
I0317 09:59:16.244447 125344476638336 pyconfig.py:413] Config param steps: 50
I0317 09:59:16.244460 125344476638336 pyconfig.py:413] Config param stop_strings: None
I0317 09:59:16.244473 125344476638336 pyconfig.py:413] Config param student_overrides: {}
I0317 09:59:16.244486 125344476638336 pyconfig.py:413] Config param subslice_shape:
I0317 09:59:16.244498 125344476638336 pyconfig.py:413] Config param swap_space_vllm_gb: 2
I0317 09:59:16.244510 125344476638336 pyconfig.py:413] Config param target_eval_loss: 0.0
I0317 09:59:16.244523 125344476638336 pyconfig.py:413] Config param teacher_overrides: {}
I0317 09:59:16.244535 125344476638336 pyconfig.py:413] Config param temperature_tuning: False
I0317 09:59:16.244548 125344476638336 pyconfig.py:413] Config param temporal_patch_size_for_vit: 2
I0317 09:59:16.244562 125344476638336 pyconfig.py:413] Config param tensorboard_dir: /home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/tensorboard/
I0317 09:59:16.244575 125344476638336 pyconfig.py:413] Config param tensors_on_device: None
I0317 09:59:16.244587 125344476638336 pyconfig.py:413] Config param tensors_to_offload: None
I0317 09:59:16.244600 125344476638336 pyconfig.py:413] Config param tile_size_for_vit: 336
I0317 09:59:16.244612 125344476638336 pyconfig.py:413] Config param tokenize_eval_data: True
I0317 09:59:16.244625 125344476638336 pyconfig.py:413] Config param tokenize_train_data: True
I0317 09:59:16.244638 125344476638336 pyconfig.py:413] Config param tokenizer_path: /home/pinakinchoudhary/custom-gemma/brahmai-tokenizer
I0317 09:59:16.244651 125344476638336 pyconfig.py:413] Config param tokenizer_type: TokenizerType.HUGGINGFACE
I0317 09:59:16.244667 125344476638336 pyconfig.py:413] Config param topk_routing_group: -1
I0317 09:59:16.244679 125344476638336 pyconfig.py:413] Config param train_data_columns: ['text']
I0317 09:59:16.244692 125344476638336 pyconfig.py:413] Config param train_fraction: 1.0
I0317 09:59:16.244704 125344476638336 pyconfig.py:413] Config param train_image_column: image
I0317 09:59:16.244717 125344476638336 pyconfig.py:413] Config param train_split: train
I0317 09:59:16.244729 125344476638336 pyconfig.py:413] Config param trainable_position_size: -1
I0317 09:59:16.244742 125344476638336 pyconfig.py:413] Config param trainer_devices_fraction: 0.5
I0317 09:59:16.244755 125344476638336 pyconfig.py:413] Config param upload_all_profiler_results: False
I0317 09:59:16.244767 125344476638336 pyconfig.py:413] Config param use_2d_fsdp_sharding: False
I0317 09:59:16.244780 125344476638336 pyconfig.py:413] Config param use_audio: False
I0317 09:59:16.244793 125344476638336 pyconfig.py:413] Config param use_audio_in_video: False
I0317 09:59:16.244805 125344476638336 pyconfig.py:413] Config param use_batch_split_schedule: False
I0317 09:59:16.244817 125344476638336 pyconfig.py:413] Config param use_chat_template: False
I0317 09:59:16.244830 125344476638336 pyconfig.py:413] Config param use_chunked_prefill: False
I0317 09:59:16.244842 125344476638336 pyconfig.py:413] Config param use_custom_sort_vjp: True
I0317 09:59:16.244854 125344476638336 pyconfig.py:413] Config param use_dpo: False
I0317 09:59:16.244867 125344476638336 pyconfig.py:413] Config param use_grpo: True
I0317 09:59:16.244880 125344476638336 pyconfig.py:413] Config param use_iota_embed: False
I0317 09:59:16.244892 125344476638336 pyconfig.py:413] Config param use_jax_splash: False
I0317 09:59:16.244904 125344476638336 pyconfig.py:413] Config param use_max_logit_estimate: -1
I0317 09:59:16.244916 125344476638336 pyconfig.py:413] Config param use_mrope: False
I0317 09:59:16.244929 125344476638336 pyconfig.py:413] Config param use_multimodal: False
I0317 09:59:16.244942 125344476638336 pyconfig.py:413] Config param use_pathways: True
I0317 09:59:16.244954 125344476638336 pyconfig.py:413] Config param use_post_attn_norm: True
I0317 09:59:16.244967 125344476638336 pyconfig.py:413] Config param use_post_ffw_norm: True
I0317 09:59:16.244979 125344476638336 pyconfig.py:413] Config param use_qk_clip: False
I0317 09:59:16.244992 125344476638336 pyconfig.py:413] Config param use_qk_norm: False
I0317 09:59:16.245015 125344476638336 pyconfig.py:413] Config param use_qk_norm_in_gdn: True
I0317 09:59:16.245028 125344476638336 pyconfig.py:413] Config param use_qwix_quantization: False
I0317 09:59:16.245040 125344476638336 pyconfig.py:413] Config param use_ragged_attention: False
I0317 09:59:16.245053 125344476638336 pyconfig.py:413] Config param use_random_routing: False
I0317 09:59:16.245065 125344476638336 pyconfig.py:413] Config param use_replicator_service: False
I0317 09:59:16.245078 125344476638336 pyconfig.py:413] Config param use_ring_of_experts: False
I0317 09:59:16.245090 125344476638336 pyconfig.py:413] Config param use_sft: False
I0317 09:59:16.245102 125344476638336 pyconfig.py:413] Config param use_sparse_indexer: False
I0317 09:59:16.245115 125344476638336 pyconfig.py:413] Config param use_splash_scheduler: False
I0317 09:59:16.245128 125344476638336 pyconfig.py:413] Config param use_tokamax_gmm: False
I0317 09:59:16.245142 125344476638336 pyconfig.py:413] Config param use_tokamax_splash: False
I0317 09:59:16.245157 125344476638336 pyconfig.py:413] Config param use_truncation: True
I0317 09:59:16.245169 125344476638336 pyconfig.py:413] Config param use_tunix_gradient_accumulation: False
I0317 09:59:16.245182 125344476638336 pyconfig.py:413] Config param use_untrainable_positional_embedding: False
I0317 09:59:16.245194 125344476638336 pyconfig.py:413] Config param use_vertex_tensorboard: False
I0317 09:59:16.245207 125344476638336 pyconfig.py:413] Config param using_pipeline_parallelism: False
I0317 09:59:16.245219 125344476638336 pyconfig.py:413] Config param v_head_dim: 128
I0317 09:59:16.245231 125344476638336 pyconfig.py:413] Config param value_proj: RematLocation.REMAT
I0317 09:59:16.245244 125344476638336 pyconfig.py:413] Config param vertex_tensorboard_project:
I0317 09:59:16.245256 125344476638336 pyconfig.py:413] Config param vertex_tensorboard_region:
I0317 09:59:16.245269 125344476638336 pyconfig.py:413] Config param video_path:
I0317 09:59:16.245281 125344476638336 pyconfig.py:413] Config param video_placeholder: <|video|>
I0317 09:59:16.245294 125344476638336 pyconfig.py:413] Config param vision_output_dim_for_vit: 4096
I0317 09:59:16.245306 125344476638336 pyconfig.py:413] Config param vllm_additional_config: {}
I0317 09:59:16.245319 125344476638336 pyconfig.py:413] Config param vllm_hf_config_path:
I0317 09:59:16.245332 125344476638336 pyconfig.py:413] Config param vllm_hf_overrides: {}
I0317 09:59:16.245344 125344476638336 pyconfig.py:413] Config param vocab_size: 262144
I0317 09:59:16.245357 125344476638336 pyconfig.py:413] Config param warmup_steps_fraction: 0.05
I0317 09:59:16.245369 125344476638336 pyconfig.py:413] Config param weight_dtype: float32
I0317 09:59:16.245390 125344476638336 pyconfig.py:413] Config param weight_quantization_calibration_method: absmax
I0317 09:59:16.245402 125344476638336 pyconfig.py:413] Config param wi_combine_scopes: False
I0317 09:59:16.245416 125344476638336 pyconfig.py:413] Config param wi_tile_dlhs_batch_seq: 512
I0317 09:59:16.245430 125344476638336 pyconfig.py:413] Config param wi_tile_dlhs_buffer_count: 2
I0317 09:59:16.245444 125344476638336 pyconfig.py:413] Config param wi_tile_dlhs_embed_dim: 1024
I0317 09:59:16.245456 125344476638336 pyconfig.py:413] Config param wi_tile_dlhs_mlp_dim: 1024
I0317 09:59:16.245469 125344476638336 pyconfig.py:413] Config param wi_tile_drhs_batch_seq: 512
I0317 09:59:16.245481 125344476638336 pyconfig.py:413] Config param wi_tile_drhs_buffer_count: 2
I0317 09:59:16.245494 125344476638336 pyconfig.py:413] Config param wi_tile_drhs_embed_dim: 1024
I0317 09:59:16.245506 125344476638336 pyconfig.py:413] Config param wi_tile_drhs_mlp_dim: 1024
I0317 09:59:16.245518 125344476638336 pyconfig.py:413] Config param wi_tile_fwd_batch_seq: 512
I0317 09:59:16.245532 125344476638336 pyconfig.py:413] Config param wi_tile_fwd_buffer_count: 2
I0317 09:59:16.245545 125344476638336 pyconfig.py:413] Config param wi_tile_fwd_embed_dim: 1024
I0317 09:59:16.245558 125344476638336 pyconfig.py:413] Config param wi_tile_fwd_mlp_dim: 1024
I0317 09:59:16.245570 125344476638336 pyconfig.py:413] Config param wo_combine_scopes: False
I0317 09:59:16.245582 125344476638336 pyconfig.py:413] Config param wo_tile_dlhs_batch_seq: 512
I0317 09:59:16.245595 125344476638336 pyconfig.py:413] Config param wo_tile_dlhs_buffer_count: 2
I0317 09:59:16.245607 125344476638336 pyconfig.py:413] Config param wo_tile_dlhs_embed_dim: 1024
I0317 09:59:16.245620 125344476638336 pyconfig.py:413] Config param wo_tile_dlhs_mlp_dim: 1024
I0317 09:59:16.245632 125344476638336 pyconfig.py:413] Config param wo_tile_drhs_batch_seq: 512
I0317 09:59:16.245645 125344476638336 pyconfig.py:413] Config param wo_tile_drhs_buffer_count: 2
I0317 09:59:16.245657 125344476638336 pyconfig.py:413] Config param wo_tile_drhs_embed_dim: 1024
I0317 09:59:16.245670 125344476638336 pyconfig.py:413] Config param wo_tile_drhs_mlp_dim: 1024
I0317 09:59:16.245683 125344476638336 pyconfig.py:413] Config param wo_tile_fwd_batch_seq: 512
I0317 09:59:16.245695 125344476638336 pyconfig.py:413] Config param wo_tile_fwd_buffer_count: 2
I0317 09:59:16.245708 125344476638336 pyconfig.py:413] Config param wo_tile_fwd_embed_dim: 1024
I0317 09:59:16.245720 125344476638336 pyconfig.py:413] Config param wo_tile_fwd_mlp_dim: 1024
I0317 09:59:16.245737 125344476638336 pyconfig.py:413] Config param wsd_decay_steps_fraction: 0.15
I0317 09:59:16.245749 125344476638336 pyconfig.py:413] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0317 09:59:16.245770 125344476638336 pyconfig.py:413] Config param xprof_e2e_enable_fw_power_level_event: False
I0317 09:59:16.245783 125344476638336 pyconfig.py:413] Config param xprof_e2e_enable_fw_thermal_event: False
I0317 09:59:16.245795 125344476638336 pyconfig.py:413] Config param xprof_e2e_enable_fw_throttle_event: False
I0317 09:59:16.245808 125344476638336 pyconfig.py:413] Config param xprof_tpu_power_trace_level: 0
I0317 09:59:16.245826 125344476638336 pyconfig.py:413] Config param z_loss_multiplier: 0.0
I0317 09:59:16.246152 125344476638336 max_utils.py:750] System Information: Jax Version: 0.9.1
I0317 09:59:16.246187 125344476638336 max_utils.py:751] System Information: Jaxlib Version: 0.9.1
I0317 09:59:16.246221 125344476638336 max_utils.py:752] System Information: Jax Backend: PJRT C API
TFRT TPU v6 lite
Built on Mar 4 2026 11:32:08 (1772652728) cl/878335365
I0317 09:59:16.246249 125344476638336 train_utils.py:310] WARNING: 'base_output_directory' might be pointing your local file system
I0317 09:59:16.325568 125344476638336 maxtext_utils.py:1391] Num_devices: 8, shape (1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1)
I0317 09:59:16.626461 125344476638336 checkpointing.py:662] Setting up checkpoint logger...
I0317 09:59:16.626570 125344476638336 checkpointing.py:227] Checkpointing disabled, not creating checkpoint manager.
I0317 09:59:16.715489 125344476638336 grain_data_processing.py:46] Found 144 files for train/eval with grain
I0317 09:59:16.794246 125344476638336 tokenizer.py:245] Tokenizer path: /home/pinakinchoudhary/custom-gemma/brahmai-tokenizer
I0317 09:59:16.794354 125344476638336 tokenizer.py:224] Loading HF tokenizer: /home/pinakinchoudhary/custom-gemma/brahmai-tokenizer
/home/pinakinchoudhary/maxtext_venv/lib/python3.12/site-packages/flax/core/lift.py:310: RuntimeWarning: kwargs are not supported in scan, so "bidirectional_mask" is(are) ignored
  warnings.warn(msg.format(name, ', '.join(kwargs.keys())), RuntimeWarning)
I0317 09:59:19.183594 125344476638336 checkpointing.py:650] No existing checkpoints found, not restoring checkpoint.
I0317 09:59:40.970564 125344476638336 max_utils.py:741] Total memory size: 13.6 GB, Output size: 5.7 GB, Temp size: 7.9 GB, Argument size: 5.7 GB, Host temp size: 0.0 GB.
Per train step:
 Total TFLOPs: 102.77
 split as 97.29% learnable weight flops and 2.71% attention flops
I0317 09:59:40.972985 125344476638336 metric_logger.py:275] number parameters: 4.069 billion
Token indices sequence length is longer than the specified maximum sequence length for this model (23966 > 8192). Running this sequence through the model will result in indexing errors
Token indices sequence length is longer than the specified maximum sequence length for this model (8258 > 8192). Running this sequence through the model will result in indexing errors
Token indices sequence length is longer than the specified maximum sequence length for this model (16323 > 8192). Running this sequence through the model will result in indexing errors
Token indices sequence length is longer than the specified maximum sequence length for this model (12043 > 8192). Running this sequence through the model will result in indexing errors
I0317 10:00:12.743028 125344476638336 max_utils.py:700]
Memstats: After params initialized:
I0317 10:00:12.743185 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_0(process=0,(0,0,0,0))
I0317 10:00:12.743223 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_1(process=0,(1,0,0,0))
I0317 10:00:12.743248 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_2(process=0,(0,1,0,0))
I0317 10:00:12.743271 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_3(process=0,(1,1,0,0))
I0317 10:00:12.743292 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_4(process=0,(0,2,0,0))
I0317 10:00:12.743313 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_5(process=0,(1,2,0,0))
I0317 10:00:12.743333 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_6(process=0,(0,3,0,0))
I0317 10:00:12.743352 125344476638336 max_utils.py:706]         Using (GB) 5.86 / 31.25 (18.752000%) on TPU_7(process=0,(1,3,0,0))
I0317 10:00:13.224641 125344476638336 metric_logger.py:181] completed step: 0, seconds: 31.731, TFLOP/s/device: 3.239, Tokens/s/device: 129.087, total_weights: 30193, loss: 12.938
I0317 10:00:13.226566 125344476638336 metric_logger.py:255] To see full metrics 'tensorboard --logdir=/home/pinakinchoudhary/custom-gemma/tmp/custom-gemma-swa/tensorboard/'
I0317 10:00:13.658792 125344476638336 metric_logger.py:181] completed step: 1, seconds: 0.277, TFLOP/s/device: 370.707, Tokens/s/device: 14774.896, total_weights: 31850, loss: 12.939
I0317 10:00:14.093450 125344476638336 metric_logger.py:181] completed step: 2, seconds: 0.215, TFLOP/s/device: 477.762, Tokens/s/device: 19041.686, total_weights: 27633, loss: 11.412
I0317 10:00:14.526789 125344476638336 metric_logger.py:181] completed step: 3, seconds: 0.434, TFLOP/s/device: 237.056, Tokens/s/device: 9448.085, total_weights: 29413, loss: 14.140
W0317 10:00:14.527971 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:14.969613 125344476638336 metric_logger.py:181] completed step: 4, seconds: 0.434, TFLOP/s/device: 236.694, Tokens/s/device: 9433.680, total_weights: 31345, loss: 14.640
I0317 10:00:15.412527 125344476638336 metric_logger.py:181] completed step: 5, seconds: 0.434, TFLOP/s/device: 237.045, Tokens/s/device: 9447.649, total_weights: 30851, loss: 12.286
W0317 10:00:15.413712 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:15.852387 125344476638336 metric_logger.py:181] completed step: 6, seconds: 0.443, TFLOP/s/device: 231.978, Tokens/s/device: 9245.716, total_weights: 29291, loss: 12.318
I0317 10:00:16.291563 125344476638336 metric_logger.py:181] completed step: 7, seconds: 0.443, TFLOP/s/device: 232.190, Tokens/s/device: 9254.155, total_weights: 31632, loss: 12.373
I0317 10:00:16.725589 125344476638336 metric_logger.py:181] completed step: 8, seconds: 0.439, TFLOP/s/device: 233.854, Tokens/s/device: 9320.466, total_weights: 31165, loss: 11.938
W0317 10:00:16.725888 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:16.726686 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:16.726898 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:17.152287 125344476638336 metric_logger.py:181] completed step: 9, seconds: 0.440, TFLOP/s/device: 233.790, Tokens/s/device: 9317.922, total_weights: 29579, loss: nan
W0317 10:00:17.152580 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.152882 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.153438 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.153641 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.153833 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:17.579063 125344476638336 metric_logger.py:181] completed step: 10, seconds: 0.435, TFLOP/s/device: 236.448, Tokens/s/device: 9423.869, total_weights: 30708, loss: nan
W0317 10:00:17.579360 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.579662 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.580190 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.580389 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:17.580584 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:18.005612 125344476638336 metric_logger.py:181] completed step: 11, seconds: 0.426, TFLOP/s/device: 241.018, Tokens/s/device: 9606.004, total_weights: 29800, loss: nan
W0317 10:00:18.005908 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.006210 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.006779 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.006931 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.007139 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:18.432328 125344476638336 metric_logger.py:181] completed step: 12, seconds: 0.427, TFLOP/s/device: 240.946, Tokens/s/device: 9603.144, total_weights: 28410, loss: nan
W0317 10:00:18.432615 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.432885 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.433367 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.433592 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.433808 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:18.859127 125344476638336 metric_logger.py:181] completed step: 13, seconds: 0.427, TFLOP/s/device: 240.839, Tokens/s/device: 9598.868, total_weights: 28975, loss: nan
W0317 10:00:18.859441 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.859753 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.860303 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.860453 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:18.860634 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:19.285658 125344476638336 metric_logger.py:181] completed step: 14, seconds: 0.427, TFLOP/s/device: 240.823, Tokens/s/device: 9598.260, total_weights: 32387, loss: nan
W0317 10:00:19.285947 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.286265 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.286801 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.286957 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.287192 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:19.712659 125344476638336 metric_logger.py:181] completed step: 15, seconds: 0.427, TFLOP/s/device: 240.825, Tokens/s/device: 9598.305, total_weights: 31613, loss: nan
W0317 10:00:19.712948 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.713245 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.713776 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.713949 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:19.714165 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:20.139541 125344476638336 metric_logger.py:181] completed step: 16, seconds: 0.426, TFLOP/s/device: 241.135, Tokens/s/device: 9610.692, total_weights: 31594, loss: nan
W0317 10:00:20.139830 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.140132 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.140717 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.140872 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.141067 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:20.567310 125344476638336 metric_logger.py:181] completed step: 17, seconds: 0.427, TFLOP/s/device: 240.544, Tokens/s/device: 9587.117, total_weights: 31411, loss: nan
W0317 10:00:20.567605 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.567896 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.568457 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.568619 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.568814 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:20.994041 125344476638336 metric_logger.py:181] completed step: 18, seconds: 0.427, TFLOP/s/device: 240.658, Tokens/s/device: 9591.675, total_weights: 30726, loss: nan
W0317 10:00:20.994331 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.994639 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.995216 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.995368 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:20.995569 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:21.420979 125344476638336 metric_logger.py:181] completed step: 19, seconds: 0.427, TFLOP/s/device: 240.475, Tokens/s/device: 9584.380, total_weights: 31291, loss: nan
W0317 10:00:21.421299 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.421588 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.422084 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.422291 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.422528 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:21.847368 125344476638336 metric_logger.py:181] completed step: 20, seconds: 0.427, TFLOP/s/device: 240.782, Tokens/s/device: 9596.596, total_weights: 28764, loss: nan
W0317 10:00:21.847658 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.847958 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.848504 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.848686 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:21.848874 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:22.274351 125344476638336 metric_logger.py:181] completed step: 21, seconds: 0.427, TFLOP/s/device: 240.585, Tokens/s/device: 9588.756, total_weights: 28393, loss: nan
W0317 10:00:22.274636 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.274945 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.275514 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.275663 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.275837 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:22.702340 125344476638336 metric_logger.py:181] completed step: 22, seconds: 0.426, TFLOP/s/device: 240.993, Tokens/s/device: 9605.013, total_weights: 29020, loss: nan
W0317 10:00:22.702629 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.702935 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.703490 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.703647 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:22.703820 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:23.129366 125344476638336 metric_logger.py:181] completed step: 23, seconds: 0.427, TFLOP/s/device: 240.726, Tokens/s/device: 9594.371, total_weights: 30999, loss: nan
W0317 10:00:23.129666 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.129981 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.130558 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.130720 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.130908 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:23.556332 125344476638336 metric_logger.py:181] completed step: 24, seconds: 0.428, TFLOP/s/device: 240.370, Tokens/s/device: 9580.188, total_weights: 32512, loss: nan
W0317 10:00:23.556632 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.556923 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.557443 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.557665 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.557843 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:23.983988 125344476638336 metric_logger.py:181] completed step: 25, seconds: 0.427, TFLOP/s/device: 240.416, Tokens/s/device: 9582.026, total_weights: 27832, loss: nan
W0317 10:00:23.984314 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.984747 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.985310 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.985465 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:23.985665 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
^BdI0317 10:00:24.410596 125344476638336 metric_logger.py:181] completed step: 26, seconds: 0.427, TFLOP/s/device: 240.728, Tokens/s/device: 9594.461, total_weights: 31056, loss: nan
W0317 10:00:24.410885 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.411213 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.411763 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.411912 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.412121 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:24.838197 125344476638336 metric_logger.py:181] completed step: 27, seconds: 0.428, TFLOP/s/device: 240.292, Tokens/s/device: 9577.097, total_weights: 31812, loss: nan
W0317 10:00:24.838491 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.838791 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.839344 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.839503 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:24.839701 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:25.265308 125344476638336 metric_logger.py:181] completed step: 28, seconds: 0.426, TFLOP/s/device: 241.164, Tokens/s/device: 9611.820, total_weights: 31826, loss: nan
W0317 10:00:25.265597 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.265900 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.266457 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.266649 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.266824 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:25.692470 125344476638336 metric_logger.py:181] completed step: 29, seconds: 0.428, TFLOP/s/device: 240.129, Tokens/s/device: 9570.563, total_weights: 32001, loss: nan
W0317 10:00:25.692765 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.693115 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.693632 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.693782 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:25.693959 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:26.120195 125344476638336 metric_logger.py:181] completed step: 30, seconds: 0.427, TFLOP/s/device: 240.538, Tokens/s/device: 9586.893, total_weights: 32236, loss: nan
W0317 10:00:26.120487 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.120950 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.121548 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.121704 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.121876 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:26.547616 125344476638336 metric_logger.py:181] completed step: 31, seconds: 0.427, TFLOP/s/device: 240.568, Tokens/s/device: 9588.082, total_weights: 29198, loss: nan
W0317 10:00:26.547908 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.548211 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.548757 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.548904 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.549104 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:26.974152 125344476638336 metric_logger.py:181] completed step: 32, seconds: 0.427, TFLOP/s/device: 240.493, Tokens/s/device: 9585.098, total_weights: 31256, loss: nan
W0317 10:00:26.974449 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.974762 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.975327 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.975478 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:26.975679 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:27.400712 125344476638336 metric_logger.py:181] completed step: 33, seconds: 0.428, TFLOP/s/device: 240.293, Tokens/s/device: 9577.120, total_weights: 28986, loss: nan
W0317 10:00:27.401035 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.401350 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.401893 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.402057 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.402253 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:27.827598 125344476638336 metric_logger.py:181] completed step: 34, seconds: 0.426, TFLOP/s/device: 241.075, Tokens/s/device: 9608.302, total_weights: 31782, loss: nan
W0317 10:00:27.827888 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.828207 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.828677 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.828897 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:27.829114 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:28.254675 125344476638336 metric_logger.py:181] completed step: 35, seconds: 0.427, TFLOP/s/device: 240.767, Tokens/s/device: 9596.012, total_weights: 29331, loss: nan
W0317 10:00:28.254979 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.255324 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.255868 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.256029 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.256208 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:28.682040 125344476638336 metric_logger.py:181] completed step: 36, seconds: 0.426, TFLOP/s/device: 240.968, Tokens/s/device: 9604.022, total_weights: 31414, loss: nan
W0317 10:00:28.682348 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.682658 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.683270 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.683443 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:28.683632 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:29.108956 125344476638336 metric_logger.py:181] completed step: 37, seconds: 0.428, TFLOP/s/device: 240.373, Tokens/s/device: 9580.301, total_weights: 29036, loss: nan
W0317 10:00:29.109279 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.109589 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.110160 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.110347 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.110530 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:29.536212 125344476638336 metric_logger.py:181] completed step: 38, seconds: 0.427, TFLOP/s/device: 240.586, Tokens/s/device: 9588.778, total_weights: 32273, loss: nan
W0317 10:00:29.536513 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.536825 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.537374 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.537519 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.537719 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:29.962504 125344476638336 metric_logger.py:181] completed step: 39, seconds: 0.427, TFLOP/s/device: 240.604, Tokens/s/device: 9589.497, total_weights: 32439, loss: nan
W0317 10:00:29.962820 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.963122 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.963686 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.963858 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:29.964068 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:30.388930 125344476638336 metric_logger.py:181] completed step: 40, seconds: 0.427, TFLOP/s/device: 240.852, Tokens/s/device: 9599.385, total_weights: 31008, loss: nan
W0317 10:00:30.389257 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.389554 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.390122 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.390326 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.390502 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:30.815785 125344476638336 metric_logger.py:181] completed step: 41, seconds: 0.427, TFLOP/s/device: 240.813, Tokens/s/device: 9597.833, total_weights: 30693, loss: nan
W0317 10:00:30.816084 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.816384 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.816941 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.817113 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:30.817306 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:31.242075 125344476638336 metric_logger.py:181] completed step: 42, seconds: 0.426, TFLOP/s/device: 240.978, Tokens/s/device: 9604.405, total_weights: 31528, loss: nan
W0317 10:00:31.242374 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.242678 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.243245 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.243406 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.243597 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:31.668638 125344476638336 metric_logger.py:181] completed step: 43, seconds: 0.427, TFLOP/s/device: 240.792, Tokens/s/device: 9597.001, total_weights: 30214, loss: nan
W0317 10:00:31.668933 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.669243 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.669795 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.669950 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:31.670154 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:32.096307 125344476638336 metric_logger.py:181] completed step: 44, seconds: 0.426, TFLOP/s/device: 241.373, Tokens/s/device: 9620.150, total_weights: 31196, loss: nan
W0317 10:00:32.096617 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.096906 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.097473 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.097631 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.097822 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:32.522935 125344476638336 metric_logger.py:181] completed step: 45, seconds: 0.427, TFLOP/s/device: 240.667, Tokens/s/device: 9592.012, total_weights: 31722, loss: nan
W0317 10:00:32.523262 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.523585 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.524140 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.524294 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.524491 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:32.949953 125344476638336 metric_logger.py:181] completed step: 46, seconds: 0.428, TFLOP/s/device: 240.323, Tokens/s/device: 9578.329, total_weights: 31080, loss: nan
W0317 10:00:32.950282 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.950595 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.951149 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.951307 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:32.951522 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:33.377063 125344476638336 metric_logger.py:181] completed step: 47, seconds: 0.427, TFLOP/s/device: 240.904, Tokens/s/device: 9601.478, total_weights: 30170, loss: nan
W0317 10:00:33.377369 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.377801 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.378406 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.378561 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.378756 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:33.803863 125344476638336 metric_logger.py:181] completed step: 48, seconds: 0.427, TFLOP/s/device: 240.820, Tokens/s/device: 9598.103, total_weights: 31784, loss: nan
W0317 10:00:33.804179 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.804467 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.805047 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.805200 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:33.805367 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
I0317 10:00:34.230409 125344476638336 metric_logger.py:181] completed step: 49, seconds: 0.428, TFLOP/s/device: 240.360, Tokens/s/device: 9579.808, total_weights: 29182, loss: nan
W0317 10:00:34.230695 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:34.230954 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:34.231483 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:34.231664 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
W0317 10:00:34.231835 125344476638336 x2num.py:13] NaN or Inf found in input tensor.
 pinakinchoudhary@t1v-n-dfaea27c-w-0  ~/custom-gemma   main ●  tmux capture-pane -pS - > session_dump.txt

