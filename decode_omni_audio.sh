#!/bin/bash
# Qwen3-Omni decode with speech output on the v5p-8 dev VM.
# Usage: bash decode_omni_audio.sh <mode: text|video> [extra maxtext args...]
set -x
MODE=${1:-text}; shift || true
cd /home/hengtaoguo_google_com/projects/maxtext
export PYTHONPATH=src:.
export HF_HOME=/dev/shm/hengtaoguo
export JAX_PLATFORMS=tpu,cpu
export TMPDIR=/dev/shm
# The JetStream HF tokenizer needs a real token (an empty one yields "Bearer " header errors).
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}
PY=/home/hengtaoguo_google_com/projects/venv1/bin/python
CKPT=${CKPT:-/dev/shm/hengtaoguo/checkpoints/qwen3-omni-30b-a3b-audio/unscanned/0/items}

COMMON=(
  src/maxtext/configs/base.yml
  model_name=qwen3-omni-30b-a3b
  tokenizer_path=Qwen/Qwen3-Omni-30B-A3B-Instruct
  tokenizer_type=huggingface
  load_parameters_path=${CKPT}
  per_device_batch_size=1
  scan_layers=false
  use_multimodal=true
  add_bos=false
  attention=dot_product
  skip_jax_distributed_system=true
  jax_cache_dir=/dev/shm/hengtaoguo/jax_cache
  enable_audio_output=true
)

if [ "$MODE" = "text" ]; then
  $PY -m maxtext.inference.decode "${COMMON[@]}" \
    run_name=qwen3_omni_decode_audio_text \
    prompt='Say hello and introduce yourself in one short sentence.' \
    max_prefill_predict_length=64 max_target_length=192 \
    audio_output_path=/dev/shm/hengtaoguo/omni_text.wav "$@"
else
  $PY -m maxtext.inference.decode "${COMMON[@]}" \
    run_name=qwen3_omni_decode_audio_video \
    use_audio_in_video=true \
    prompt='What can you see and hear? Answer in one short sentence.' \
    video_path=/home/hengtaoguo_google_com/projects/mini.mp4 \
    max_prefill_predict_length=2048 max_target_length=2304 \
    audio_output_path=/dev/shm/hengtaoguo/omni_video.wav "$@"
fi
