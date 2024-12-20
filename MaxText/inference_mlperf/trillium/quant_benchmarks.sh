#!/bin/bash

set -x

export TOKENIZER_PATH=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 
# export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_
export DATA_DISK_DIR=/tmp
export LOGLEVEL=WARNING
export FAST_EVAL=true

# ## int8 baseline
# export kv_quant_dtype=int8
# export quantization=int8
# export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance | tee perf_int8_int8_a8w8.out
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int8_int8_a8w8.out

# ## a16w4
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int4_weight_only.json
# export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/int4_weight_only
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int8_intmp_a16w4.out
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int8_intmp_a16w4.out

# ## a16w8
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int8_weight_only.json
# export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/int8_weight_only
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int8_intmp_a16w8.out
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int8_intmp_a16w8.out

# ## a8w4
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
# export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/a8w4
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int8_intmp_a8w4.out
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int8_intmp_a8w4.out

# ## a4w4
# export kv_quant_dtype=int8
# export quantization=intmp
# export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
# export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/a4w4
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int8_intmp_a4w4.out
# bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int8_intmp_a4w4.out



#########################################################################################
# int4 kv cache
## int8 baseline
export kv_quant_dtype=int4
export quantization=int8
export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int4_int8_a8w8.out
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int4_int8_a8w8.out

## a16w4
export kv_quant_dtype=int4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int4_weight_only.json
export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/int4_weight_only
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int4_intmp_a16w4.out
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int4_intmp_a16w4.out

## a16w8
export kv_quant_dtype=int4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int8_weight_only.json
export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/int8_weight_only
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int4_intmp_a16w8.out
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int4_intmp_a16w8.out

## a8w4
export kv_quant_dtype=int4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json
export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/a8w4
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int4_intmp_a8w4.out
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int4_intmp_a8w4.out

## a4w4
export kv_quant_dtype=int4
export quantization=intmp
export quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json
export CHECKPOINT=gs://patemotter/checkpoints/quantized/llama2-70b-chat/a4w4
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance 2>&1 | tee perf_int4_intmp_a4w4.out
bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -b accuracy 2>&1 | tee acc_int4_intmp_a4w4.out
