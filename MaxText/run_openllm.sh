# git clone https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .

# CHECKPOINT_PATH="gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-22-48/checkpoints/100000/items"
# # gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-22-48/checkpoints/100000
# # gs://mazumdera-test-bucket-us-east5/lg-llama2/hf/mazumdera-llama27b-lg-2024-04-22-16-15/checkpoints/77000
# BASE_OUTPUT_DIRECTORY="gs://aireenmei-multipod/openllm/"
# RUN_NAME=$(date +%Y-%m-%d-%H-%M)

# echo "generating param only checkpoint"

# python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#   run_name=${RUN_NAME} \
#   load_parameters_path=${CHECKPOINT_PATH} \
#   model_name='llama2-7b' force_unroll=true

# Successfully generated decode checkpoint at: gs://aireenmei-multipod/openllm/2024-04-26-21-01/checkpoints/0/items

echo "OpenLLM decode"

RUN_NAME=$(date +%Y-%m-%d-%H-%M)
python3 MaxText/decode_openllm.py MaxText/configs/base.yml load_parameters_path=gs://aireenmei-multipod/openllm/2024-04-26-20-55/checkpoints/0/items \
 run_name=${RUN_NAME}

 