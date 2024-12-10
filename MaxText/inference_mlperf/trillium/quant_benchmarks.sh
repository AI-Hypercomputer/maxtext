export TOKENIZER_PATH=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 
export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_
export DATA_DISK_DIR=/tmp
export LOGLEVEL=INFO


# int8 baseline w/ int8 kv cache
export kv_quant_dtype=int8
export quantization=int8
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance


# int8 baseline w/ int4 kv cache
# export kv_quant_dtype=int4
# export quantization=int8
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance


# a16 w4 w/ int8 kv cache
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance

# # a16 w4 w/ int4 kv cache
# export kv_quant_dtype=int4
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance


# a8 w4 w/ int8 kv cache
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance

# # a8 w4 w/ int4 kv cache
# export kv_quant_dtype=int4
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance


# # a4 w4 w/ int8 kv cache
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance

# # a4 w4 w/ int4 kv cache
# export kv_quant_dtype=int4
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance