export TPU_VISIBLE_CHIPS=4,5,6,7
MODEL_NAME=dit-xl-2-256
NOW=$(date +%m%d%H%M)
OUTPUT_PATH=${GS_BUCKET?}/output/${MODEL_NAME?}/${NOW?}/
PYTHONPATH=src/ /home/liyinn_google_com/anaconda3/envs/maxtext/bin/python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME?} \
    decoder_block=dit \
    base_output_directory=${OUTPUT_PATH} \
    scan_layers=false \
    use_multimodal=false \
    hf_access_token=${HF_TOKEN?} \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_ocdbt=False \
    checkpoint_storage_use_zarr3=False \
    --eager_load_method=torch \
    --lazy_load_tensors=False \
    --save_dtype=float32 \
    > output_checkpoint_${NOW?}.txt 2>&1

