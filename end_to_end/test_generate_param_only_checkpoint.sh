#!/bin/bash

set -uex

helpFunction()
{
  echo ""
  echo "Usage: $0 <dataset_path> <output_path>"
  echo -e "\t-n dry_run is true "
  echo -e "\t-r runid: run_test_model_0b"
  echo -e "\t-d dataset_path: gs://test-maxtext-dataset"
  echo -e "\t-o output_path: gs://test-maxtext-output"
  echo -e "\t-i ici_tensor_parallelism: 8"
  echo -e "\t-a attention: flash"
  echo -e "\t-q quantization: int8"
  exit 1 # Exit script after printing help
}

# Default option values
dry_run=false
run_id=test_model_0b_$(date +%Y-%m-%d-%H)
dataset_path=gs://test-maxtext-dataset
base_output_directory=gs://test-maxtext-output
ici_tensor_parallelism=8
attention=flash
quantization=""

while getopts "nr:d:o:t:i:a:q:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      r ) run_id="$OPTARG" ;;
      d ) dataset_path="$OPTARG";;
      o ) base_output_directory="$OPTARG";;
      i ) ici_tensor_parallelism="$OPTARG" ;;
      a ) attention="$OPTARG" ;;
      q ) quantization="int8" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

echo
echo "Running: ./$0 dataset_path=${dataset_path} base_output_directory=${base_output_directory}"
echo "          dry_run=${dry_run} run_id=${run_id}  "
echo "          ici_tensor_parallelism=${ici_tensor_parallelism} attention=${attention} quantization=${quantization}"
echo

if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

training_ckpt_run_id=${run_id}-ckpt-train-steps-5
decode_ckpt_run_id=${run_id}-decode-ckpt-train-steps-5
model_params="base_emb_dim=384 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=192 base_num_decoder_layers=8 head_dim=128"

echo
echo "Create a test training checkpoint"
echo
$cmd python3 MaxText/train.py MaxText/configs/base.yml \
run_name=${training_ckpt_run_id} \
base_output_directory=${base_output_directory} \
dataset_path=${dataset_path} attention=${attention} \
steps=5 checkpoint_period=3 async_checkpointing=false \
quantization=${quantization} \
${model_params} \


if [ $? -eq 0 ]
then
  echo
  echo "Successfully created a training checkpoint"
  echo "Checkpoint path:  ${base_output_directory}/${training_ckpt_run_id}/checkpoints/3/items"
else
  echo
  echo "Could not create a training checkpoint" >&2
  exit 1
fi

echo
echo "Generate a decode checkpoint from the test training checkpoint"
echo

$cmd python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml \
run_name=${decode_ckpt_run_id} attention=${attention} \
base_output_directory=${base_output_directory} \
dataset_path=${dataset_path} async_checkpointing=false \
load_full_state_path=${base_output_directory}/${training_ckpt_run_id}/checkpoints/3/items \
quantization=${quantization} \
${model_params} \


if [ $? -eq 0 ]
then
  echo "Successfully created an decode checkpoint"
  echo "Checkpoint path:  ${base_output_directory}/${decode_ckpt_run_id}/checkpoints/0/items"

else
  echo "Could not create an decode checkpoint" >&2
  exit 1
fi

echo
echo "Run decode using the generated checkpoint"
echo
$cmd python3 MaxText/decode.py MaxText/configs/base.yml \
run_name=${run_id}-decode-steps-50 \
base_output_directory=${base_output_directory} \
dataset_path=${dataset_path} \
load_parameters_path=${base_output_directory}/${decode_ckpt_run_id}/checkpoints/0/items \
attention=dot_product ici_tensor_parallelism=${ici_tensor_parallelism} steps=50 \
metrics_file=/tmp/${run_id}_metrics.txt async_checkpointing=false max_target_length=128 per_device_batch_size=1 \
quantization=${quantization} \
${model_params} \

if [ $? -eq 0 ]
then
  echo "Successfully ran decode using decode optimized checkpoint"
else
  echo "Could not run decode decode optimized checkpoint" >&2
  exit 1
fi
