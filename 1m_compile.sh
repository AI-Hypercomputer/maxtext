set -e 

export M_MAX_TARGET_LENGTH=1048576
echo "Doing ${M_MAX_TARGET_LENGTH}"
time python3 MaxText/train_compile.py MaxText/configs/base.yml \
    per_device_batch_size=0.0078125\
    ici_sequence_parallelism=16 ici_tensor_parallelism=8\
    base_output_directory=gs://runner-maxtext-logs\
    use_iota_embed=true\
    attention='flash'\
    fused_qkv=False fused_mlp=True\
    compile_topology=v5p-256 compile_topology_num_slices=1\
    base_num_kv_heads=128 base_num_query_heads=128\
    base_emb_dim=4096 base_mlp_dim=28872\
    base_num_decoder_layers=32
echo "Done ${M_MAX_TARGET_LENGTH}"