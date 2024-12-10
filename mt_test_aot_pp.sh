export NUM_NODES=1
export TARGET_NUM_NODES=128
#export TARGET_NUM_NODES=126
#export TARGET_NUM_NODES=96

export WORKLOAD_NAME=lance-$(echo $MODEL_NAME | sed 's/\.//g')-aot

export PER_DEVICE_BATCH_SIZE=5
export ICI_TP=8

export DCN_PP=2
#export DCN_FSDP=$TARGET_NUM_NODES
export DCN_FSDP=$(expr $TARGET_NUM_NODES / $DCN_PP)


#export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)
export NUM_LAYERS_PER_PP_STAGE=14
# export NUM_LAYERS_PER_PP_STAGE=16
export MBS_PER_STAGE=$PER_DEVICE_BATCH_SIZE

export JAX_ENABLE_PGLE=false
# export REMAT_POLICY=save_qkv_proj
export REMAT_POLICY=full

#export ATTENTION=dot_product
export ATTENTION=cudnn_flash_te

export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance