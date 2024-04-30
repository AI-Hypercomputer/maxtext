set -e -x

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

gsutil cp -r gs://mazumdera-test-bucket/lg/vocab_102400 assets

# CHECKPOINT_PATH="gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-22-48/checkpoints/105000/items"
# # # gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-22-48/checkpoints/100000
# # # gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-16-15/checkpoints/77000
BASE_OUTPUT_DIRECTORY="gs://aireenmei-multipod/openllm/"
# RUN_NAME1=$(date +%Y-%m-%d-%H-%M)

# echo "generating param only checkpoint"

# python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#   run_name=${RUN_NAME1} \
#   load_parameters_path=${CHECKPOINT_PATH} \
#   model_name='llama2-7b' force_unroll=true

#Successfully generated decode checkpoint at: gs://aireenmei-multipod/openllm/2024-04-28-06-49/checkpoints/0/items

echo "OpenLLM decode"

#DECODE_CHECKPOINT_PATH=${BASE_OUTPUT_DIRECTORY}${RUN_NAME1}"/checkpoints/0/items"
DECODE_CHECKPOINT_PATH=gs://aireenmei-multipod/openllm/2024-04-28-06-49/checkpoints/0/items
RUN_NAME2=$(date +%Y-%m-%d-%H-%M)
python3 MaxText/decode_openllm.py MaxText/configs/base.yml load_parameters_path=${DECODE_CHECKPOINT_PATH} \
 run_name=${RUN_NAME2} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
 attention=dot_product scan_layers=false \
 per_device_batch_size=1
 #max_prefill_predict_length=

 