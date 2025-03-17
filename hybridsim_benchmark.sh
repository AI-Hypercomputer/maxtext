set -ex

# Run this script inside maxtext folder in https://github.com/AI-Hypercomputer/maxtext
MODEL_NAME="llama3.1-8b"
export M_COMPILE_TOPOLOGY=v5e-256 
export M_COMPILE_TOPOLOGY_NUM_SLICES=1
PER_DEVICE_BATCH_SIZE="2"
QUANTIZATION=""
SEQUENCE_LENGTH="2048"
JAX_VERSION="$(python3 -c 'import jax; print(jax.__version__)')"
REMAT_POLICY="full"
# REMAT_POLICY="custom"

# HLO_NAME=${MODEL_NAME}-${M_COMPILE_TOPOLOGY}-${M_COMPILE_TOPOLOGY_NUM_SLICES}-${PER_DEVICE_BATCH_SIZE}-bf16-${SEQUENCE_LENGTH}-${JAX_VERSION}-${REMAT_POLICY}
HLO_NAME=${MODEL_NAME}-${M_COMPILE_TOPOLOGY}

LOCAL_DIR="/tmp/hlo/${HLO_NAME}"
REMOTE_DIR="gs://hengtaoguo-maxtext-logs/hlo/test/${HLO_NAME}"
echo $LOCAL_DIR

export XLA_FLAGS="--xla_dump_to=${LOCAL_DIR} --xla_dump_large_constants=true"
echo ${XLA_FLAGS}

# Run train_compile to dump HLO to LOCAL_DIR
python MaxText/train_compile.py MaxText/configs/base.yml model_name=$MODEL_NAME\
  base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
  tokenizer_path=assets/tokenizer.llama2 remat_policy=$REMAT_POLICY\
  steps=20 enable_checkpointing=false use_iota_embed=true\
  max_target_length=$SEQUENCE_LENGTH per_device_batch_size=$PER_DEVICE_BATCH_SIZE\

echo "Dumped HLO to ${LOCAL_DIR}"
echo "=========="

# # ==================================================================================
# ### Step 1: Provide the HLO graph folder ###
# local_dir=$LOCAL_DIR

# ### Step 2: De-sanitize ###
# find $local_dir -type f -exec sed -i 's/e0897fcce0/PUFFERFISH/g' {} +
# find $local_dir -type f -exec sed -i 's/2093c6a02/VIPERFISH/g' {} +
# find $local_dir -type f -exec sed -i 's/b5ee2/barna/g' {} +
# find $local_dir -type f -exec sed -i 's/cc8675309/GHOSTLITE/g' {} +

# ### Step 2.5: Remove LIBTPU flag formatting issue in tpu_comp_env.txt - see b/309949420 
# search_string="tpu_comp_env.txt"

# # Loop through files in the directory 
# find "$local_dir" -name "*$search_string*" -print0 | while IFS= read -r -d '' filename; do 
# # Remove lines with "//" sed -i '/\/\//d' "$filename" 
# # Replace "var1 var2" with "var1:var2" 
# sed -i -E 's/\b(\w+)\s+\b(\w+)\b/\1:\ \2/g' "$filename" 
# done

# ==================================================================================
### Rename files to remove the ".cl_731502260." suffix
for file in $LOCAL_DIR/*; do
  if [[ "$file" == *".cl_731502260."* ]]; then
    new_file="${file//.cl_731502260./.}"
    mv "$file" "$new_file"
  fi
done

# ==================================================================================
### Copy to GCS REMOTE DIR
gcloud storage cp -r ${LOCAL_DIR} ${REMOTE_DIR}
echo ${LOCAL_DIR}
echo "https://pantheon.corp.google.com/storage/browser/hengtaoguo-maxtext-logs/hlo/benchmark/${HLO_NAME}"

echo "=========="
echo "input_dir=${REMOTE_DIR}"
echo "bash cloud/tpu/tools/multipod/multipod_tests/unsanitize_and_move_dir.sh \$input_dir"


# gcloud auth configure-docker 
# export HYBRIDSIM_DOCKER_IMAGE=gcr.io/tpu-prod-env-multipod/hybridsim:$USER-latest
# # export HYBRIDSIM_DOCKER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/test_hybridsim_jax:2024-12-10
# docker pull $HYBRIDSIM_DOCKER_IMAGE

# xla_dump_dir="/tmp/xla_dump"

# xla_dump_dir=/tmp/hlo/llama3.1-8b-v4-128
# module_name_pattern=jit_train_step
# deepsea_chip_config_name=megacore # megacore
# use_megascale_xla=false # true or false 

# # Run HybridSim
# docker run -v $xla_dump_dir:$xla_dump_dir --network=host --privileged \
# $HYBRIDSIM_DOCKER_IMAGE \
# --xla_dump_dir=$xla_dump_dir \
# --module_name_pattern=$module_name_pattern \
# --deepsea_chip_config_name=$deepsea_chip_config_name 2>&1 | tee ./hybridsim_log.txt

# # # echo "=========="
# # # cat hybridsim_log.txt | grep -E "xla_dump_dir|estimated_cost_ns"
