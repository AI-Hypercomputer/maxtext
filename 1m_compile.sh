set -e 


for M_MAX_TARGET_LENGTH in 32768 65536 131072 262144 524288 1048576
do
    export M_MAX_TARGET_LENGTH=${M_MAX_TARGET_LENGTH}
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
        compiled_trainstep_file="compiled_${M_MAX_TARGET_LENGTH}.pickle"\
        base_num_decoder_layers=32 > file_${M_MAX_TARGET_LENGTH}.txt
    echo "Done ${M_MAX_TARGET_LENGTH}"
    echo 
    echo
done