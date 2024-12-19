MODEL_NAME="llama3.1-8b"
# ACCELERATOR="v5p"
# NUM_ACCELERATOR="256"
export M_COMPILE_TOPOLOGY=v5e-256
export M_COMPILE_TOPOLOGY_NUM_SLICES=1
PER_DEVICE_BATCH_SIZE="2"
QUANTIZATION=""
SEQUENCE_LENGTH="2048"
JAX_VERSION="$(python3 -c 'import jax; print(jax.__version__)')"

HLO_NAME=new-${MODEL_NAME}-${M_COMPILE_TOPOLOGY}-${M_COMPILE_TOPOLOGY_NUM_SLICES}-${PER_DEVICE_BATCH_SIZE}-bf16-${SEQUENCE_LENGTH}-${JAX_VERSION}

export XLA_FLAGS="--xla_dump_to=/tmp/hlo/${HLO_NAME} --xla_dump_large_constants=true"
# export XLA_FLAGS="--xla_dump_to=/tmp/hlo/${HLO_NAME}"
echo ${XLA_FLAGS}

python MaxText/train_compile.py MaxText/configs/base.yml model_name=$MODEL_NAME\
  base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
  tokenizer_path=assets/tokenizer.llama2 remat_policy=save_qkv_proj\
  steps=20 enable_checkpointing=false use_iota_embed=true\
  max_target_length=$SEQUENCE_LENGTH per_device_batch_size=$PER_DEVICE_BATCH_SIZE\
  quantization=$QUANTIZATION

# gpt3-6b
# python MaxText/train_compile.py MaxText/configs/base.yml model_name=$MODEL_NAME\
#   base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
#   steps=20 enable_checkpointing=false\
#   per_device_batch_size=$PER_DEVICE_BATCH_SIZE\
#   max_target_length=$SEQUENCE_LENGTH quantization=$QUANTIZATION

# Copy to GCS
gcloud storage cp -r /tmp/hlo/${HLO_NAME} gs://hengtaoguo-maxtext-logs/hlo/benchmark/${HLO_NAME}

echo gs://hengtaoguo-maxtext-logs/hlo/benchmark/${HLO_NAME}
echo http://pantheon.corp.google.com/storage/browser/hengtaoguo-maxtext-logs/hlo/benchmark/${HLO_NAME}

# HLO_NAME="llama3-70b-v5e-256-1-1-bf16-2048"

# xla_dump_dir=/tmp/hlo/${HLO_NAME}
# module_name_pattern=jit_train_step
# deepsea_chip_config_name=megacore # megacore
# use_megascale_xla=false # true or false 

# export HYBRIDSIM_DOCKER_IMAGE=gcr.io/tpu-prod-env-multipod/hybridsim:$USER-latest
# # Once below
# # blaze run -c opt //cloud/tpu/multipod/hybridsim:hybridsim_cloud_image_push -- --dst="${HYBRIDSIM_DOCKER_IMAGE}"

# # Run HybridSim
# docker run -v $xla_dump_dir:$xla_dump_dir --network=host --privileged \
# $HYBRIDSIM_DOCKER_IMAGE \
# --xla_dump_dir=$xla_dump_dir \
# --module_name_pattern=$module_name_pattern \
# --deepsea_chip_config_name=$deepsea_chip_config_name 2>&1 | tee ./hybridsim_log.txt

# echo "=========="
# cat hybridsim_log.txt | grep -E "xla_dump_dir|estimated_cost_ns"


# Default will pick stable versions of dependencies
# bash docker_build_dependency_image.sh

# XPK run
# xpk workload create \
# --cluster v5e-256-opm-ase1 \
# --base-docker-image maxtext_base_image \
# --workload hengtaoguo-v5e-llama2-7b-1 \
# --tpu-type=v5litepod-256 \
# --num-slices=1  \
# --command "python MaxText/train.py MaxText/configs/base.yml model_name=llama2-7b \
#   base_output_directory=gs://hengtaoguo-maxtext-logs \
#   dataset_path=gs://hengtaoguo-maxtext-dataset tokenizer_path=assets/tokenizer.llama2 \
#   per_device_batch_size=$PER_DEVICE_BATCH_SIZE remat_policy=save_qkv_proj steps=20 \
#   enable_checkpointing=false use_iota_embed=true profiler=xplane"