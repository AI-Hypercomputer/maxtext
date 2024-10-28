MODEL_NAME="llama2_7b"
ACCELERATOR="v5e"
NUM_ACCELERATOR="256"
M_COMPILE_TOPOLOGY_NUM_SLICES="1"
PER_DEVICE_BATCH_SIZE="8"
QUANTIZATION="int8"
SEQUENCE_LENGTH="2048"

JAX_VERSION="$(python3 -c 'import jax; print(jax.__version__)')"

HLO_NAME=${MODEL_NAME}_${ACCELERATOR}_${NUM_ACCELERATOR}_${M_COMPILE_TOPOLOGY_NUM_SLICES}_${PER_DEVICE_BATCH_SIZE}_${QUANTIZATION}_${SEQUENCE_LENGTH}_${JAX_VERSION}

echo ${HLO_NAME}
echo "export XLA_FLAGS="--xla_dump_to=/tmp/hlo/${HLO_NAME}""
export XLA_FLAGS="--xla_dump_to=/tmp/hlo/${HLO_NAME}"

bash MaxText/configs/v5e/${MODEL_NAME}.sh \
    EXECUTABLE=train_compile.py \
    M_COMPILE_TOPOLOGY=${ACCELERATOR}-${NUM_ACCELERATOR} \
    M_COMPILE_TOPOLOGY_NUM_SLICES=${M_COMPILE_TOPOLOGY_NUM_SLICES} \
    QUANTIZATION=${QUANTIZATION} \
    SEQUENCE_LENGTH=${SEQUENCE_LENGTH} \
    PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}

gcloud storage cp -r /tmp/hlo/${HLO_NAME} gs://hengtaoguo-maxtext-logs/hlo/benchmark/${HLO_NAME}

xla_dump_dir=/tmp/hlo/${HLO_NAME}
module_name_pattern=jit_train_step
deepsea_chip_config_name=megacore

export HYBRIDSIM_DOCKER_IMAGE=gcr.io/tpu-prod-env-multipod/hybridsim:$USER-latest

# Run HybridSim
docker run -v $xla_dump_dir:$xla_dump_dir --network=host --privileged \
$HYBRIDSIM_DOCKER_IMAGE \
--xla_dump_dir=$xla_dump_dir \
--module_name_pattern=$module_name_pattern \
--deepsea_chip_config_name=$deepsea_chip_config_name 2>&1 | tee ./hybridsim_log.txt

echo "=========="
cat hybridsim_log.txt | grep -E "xla_dump_dir|estimated_cost_ns"
