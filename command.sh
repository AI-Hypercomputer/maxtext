export RUN=$(date +%Y-%m-%d-%H-%M-%S)
export EMBED=$1
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export TPU_PREMAPPED_BUFFER_SIZE=4294967296 
#export XLA_FLAGS="--xla_dump_to=/tmp/hlo_${RUN}_${EMBED}"
#export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_${RUN}_${EMBED}"
export TPU_VMODULE=3
python3 pedagogical_examples/shardings.py --ici_fsdp_parallelism=256 --batch_size=64  --embedding_dimension=$EMBED
#python3 pedagogical_examples/shardings.py --ici_fsdp_parallelism=16 --ici_tensor_parallelism=16 --batch_size=64  --embedding_dimension=$EMBED
