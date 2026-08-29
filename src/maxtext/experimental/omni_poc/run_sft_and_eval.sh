#!/bin/bash
set -e

export PYTHONPATH=src:$PYTHONPATH

echo "=== Starting SFT Training ==="
python3 -m maxtext.experimental.omni_poc.train_sft_omni \
  src/maxtext/experimental/omni_poc/sft-omni-gemma3-qwen3-xpk-128.yml

echo "=== Starting SFT Evaluation ==="
python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
  src/maxtext/experimental/omni_poc/sft-omni-gemma3-qwen3-xpk-128.yml \
  load_parameters_path=gs://yuchenhou-maxtext-logs/omni-gemma3-qwen3/multimodal/sft_after_chartnet/omni_sft_chartqa_v4_128/checkpoints/1100/items \
  base_output_directory=gs://yuchenhou-maxtext-logs/omni-gemma3-qwen3/multimodal/sft_after_chartnet \
  run_name=eval_final_results \
  --ckpt_type=sft \
  --num_examples=2500 \
  --hf_eval_split=test

echo "=== SFT Training and Evaluation Complete ==="
