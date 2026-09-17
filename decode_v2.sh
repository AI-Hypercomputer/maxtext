MODEL_NAME=dit-xl-2-256
if [ -z "$1" ]; then
  echo "Error: checkpoint_path is required."
  echo "Usage: $0 <checkpoint_path>"
  exit 1
fi
export checkpoint_path=$1

NOW=$(date +%Y-%m-%d-%H-%M)

TPU_VISIBLE_CHIPS=4,5,6,7 PYTHONPATH=src/ /home/liyinn_google_com/anaconda3/envs/maxtext/bin/python3 -m maxtext.inference.sampler_v2 \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME?} \
    tokenizer_path= \
    tokenizer_type=huggingface  \
    load_parameters_path=${checkpoint_path} \
    per_device_batch_size=1 \
    scan_layers=false \
    use_multimodal=false \
    prompt='white shark,umbrella' \
    max_prefill_predict_length=256 \
    max_target_length=256 \
    ici_tensor_parallelism=4 \
    override_model_config=true \
    attention='dot_product' \
    hf_access_token=${HF_TOKEN} \
    --verbosity=1 --alsologtostderr \
    > output_decode_v2_${MODEL_NAME?}_${NOW}.txt 2>&1
