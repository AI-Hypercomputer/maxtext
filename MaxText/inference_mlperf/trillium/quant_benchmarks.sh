export TOKENIZER_PATH=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 
export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_
export DATA_DISK_DIR=/tmp
export LOGLEVEL=INFO


# int8 kv cache
export kv_quant_dtype=int8

## int8 baseline
export quantization=int8
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

## a16w4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

## a8w4
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

# ## a4w4
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy


# int4 kv cache
export kv_quant_dtype=int4

# ## int8 baseline
export quantization=int8
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

# ## a16w4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

# ## a8w4
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy

# ## a4w4
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy