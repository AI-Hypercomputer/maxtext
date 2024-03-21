#export XLA_FLAGS="--xla_dump_to=v5e256_host_offloaded_maxtext_20240321"
#python3 MaxText/train_compile.py MaxText/configs/base.yml per_device_batch_size=1 ici_fsdp_parallelism=16 ici_tensor_parallelism=16 max_target_length=2048 fused_qkv=true fused_mlp=true remat_policy=minimal_offloaded use_iota_embed=true global_parameter_scale=128 compile_topology=v5e-256 compile_topology_num_slices=1

export M_ENABLE_CHECKPOINTING=false
export M_DATASET_TYPE=synthetic

python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=1 ici_fsdp_parallelism=16 ici_tensor_parallelism=16 max_target_length=2048 fused_qkv=true fused_mlp=true remat_policy=minimal_offloaded use_iota_embed=true global_parameter_scale=128
