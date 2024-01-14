echo "Running v5p_int8.sh".
# This config will work out of the box for any number of v5e-256 slices.

# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#


# Stop execution if any command exits with error
set -e

export PLATFORM="gke"
export PER_DEVICE_BATCH_SIZE=4
export GLOBAL_PARAMETER_SCALE=8
export INT8_TRAINING=true
export MARCELLO_FLAG=true
# Shards: 0 will use ttf when int8 is true (bf16 when false), else ttt (setting to 1 is basicaly non-local)
export local_aqt_shards_mlp1=0 
export local_aqt_shards_mlp2=0
export local_aqt_shards_query_proj=0
export local_aqt_shards_key_proj=0
export local_aqt_shards_value_proj=0
export local_aqt_shards_attention_out_proj=0
export remat_policy=full
export steps=5
export reuse_example_batch=1
export dataset_type="synthetic"
export learning_rate=0.001

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

libtpu_init_args="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

if [[ $MARCELLO_FLAG == true ]]; then
    echo "Using Marcello flag"
    libtpu_init_args="${libtpu_init_args} --xla_jf_crs_combiner_threshold_count=0"
fi

export LIBTTPU_INIT_ARGS=$libtpu_init_args

# Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=$steps per_device_batch_size=$PER_DEVICE_BATCH_SIZE enable_checkpointing=false\
    enable_profiler=true remat_policy=$remat_policy global_parameter_scale=$GLOBAL_PARAMETER_SCALE\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=$reuse_example_batch\
    dataset_type=$dataset_type attention='flash' int8_training=$INT8_TRAINING\
    learning_rate=$learning_rate\
    local_aqt_shards_mlp1=$local_aqt_shards_mlp1\
    local_aqt_shards_mlp2=$local_aqt_shards_mlp2\
    local_aqt_shards_query_proj=$local_aqt_shards_query_proj\
    local_aqt_shards_key_proj=$local_aqt_shards_key_proj\
    local_aqt_shards_value_proj=$local_aqt_shards_value_proj\
    local_aqt_shards_attention_out_proj=$local_aqt_shards_attention_out_proj\
