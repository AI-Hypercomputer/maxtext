
## Create TPU VM.
Follow these [instructions](https://cloud.google.com/tpu/docs/v5e-inference#tpu-vm) to create TPU v5e-8 VM and ssh into the VM


## Clone repo
```
git clone https://github.com/mlcommons/inference.git
```

## Install loadgen
```
apt-get install python3-dev
apt-get install build-essential -y
cd loadgen/ && pip install .
```

## Install eval dependencies
```
pip install \
transformers==4.31.0 \
nltk==3.8.1 \
evaluate==0.4.0 \
absl-py==1.4.0 \
rouge-score==0.1.2 \
sentencepiece==0.1.99 \
accelerate==0.21.0
```

## Download data file
```
cd /
export DATA_DISK_DIR=/loadgen_run_data
mkdir -p ${DATA_DISK_DIR}
cd ${DATA_DISK_DIR}
gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl .
mv open_orca_gpt4_tokenized_llama.calibration_1000.pkl processed-calibration-data.pkl

gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl .
mv open_orca_gpt4_tokenized_llama.sampled_24576.pkl processed-data.pkl
cd /inference_mlperf4.1
```

## Install Maxtext 
```
cd /
git clone git@github.com:google/maxtext.git
cd maxtext
git checkout offline_inf
cd maxtext/MaxText
```

## Checkpoint generation

Steps to get a quantized llama2-70B checkpoint for v5e-8

Note llama2-70B model takes about 140G of memory and will not fit into a v5e-8. It must be downloaded onto a large machine (such as v5p-8) and quantized to a smaller quantized checkpoint to be loaded onto a v5e-8 machine.

* Obtain a llama2-70b checkpoint and convert it to a maxtext inference checkpoint. Please follow maxtext instructions specified here: https://github.com/google/maxtext/blob/main/getting_started/Run_Llama2.md

* Convert the checkpoint into a quantized checkpoint

To create an int8 DRQ checkpoint run the following step:

1. Define paths to load maxtext checkpoint from and save quantized checkpoint to.

```
export LOAD_PARAMS_PATH=gs://${USER}-bkt/llama2-70b-chat/param-only-decode-ckpt-maxtext/checkpoints/0/items

export SAVE_QUANT_PARAMS_PATH=gs://${USER}-bkt/quantized/llama2-70b-chat
```

2. Run the following maxtext script to generate and save an int8 quantized checkpoint

```
# Set appropriate tokenizer path. For example, LLama2 models tokenizer.llama2. You can find 
# other tokenizers under maxtext/assets/ directory.
export TOKENIZER_PATH=maxtext/assets/tokenizer.llama2
cd maxtext && \
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${LOAD_PARAMS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-70b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11 attention=dot_product quantization=int8 save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH}
```

Your checkpoint is generated at `$SAVE_QUANT_PARAMS_PATH`. This is used to set `load_parameters_path` param below in `MAXENGINE_ARGS` env variable. 

## HF login
```
huggingface-cli login
```

## Loadgen settings
```
cd Google/code/llama2-70b/tpu_v5e_8_jetstream_maxtext/scripts/
export API_URL=0.0.0.0:9000
export DATA_DISK_DIR=/loadgen_run_data
export DATASET_TYPE=full # for calibration run, DATASET_TYPE=calibration

export MODEL_NAME=llama70b
export TOTAL_SAMPLE_COUNT=24576 # for calibration run, TOTAL_SAMPLE_COUNT=1000
export LOG_INTERVAL=1000
export BATCH_SIZE_EXP=8
export USER_CONFIG=user.conf
```

## Offline Setup
```
cd /
git clone git@github.com:google/maxtext.git
cd maxtext
git checkout offline_inf
cd maxtext/MaxText

# For v5e use
export BATCH_AND_PREFILL_LEN=“256,80|512,40|1024,20”

# For v6 use
export BATCH_AND_PREFILL_LEN=“256,216|512,108|1024,54”

# Set appropriate tokenizer path. For example, LLama2 models tokenizer.llama2. You can find 
# other tokenizers under maxtext/assets/ directory.
export TOKENIZER_PATH=maxtext/assets/tokenizer.llama2

export MAXENGINE_ARGS="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH}  quantization=int8 quantize_kvcache=True load_parameters_path=${SAVE_QUANT_PARAMS_PATH} checkpoint_is_quantized=True compute_axis_order=0,1,2,3 ar_cache_axis_order=0,1,2,3"
```

## Run offline performance

```
bash ./llama_offline_performance_run.sh
```

## Run offline accuracy
```
bash ./llama_offline_accuracy_run.sh
```

## Run offline audit
```
bash ./llama_offline_audit_run.sh
```

## Run server performance
```
bash ./generate_server_performance_run.sh
```

## Run server accuracy
```
bash ./generate_server_accuracy_run.sh
```

## Run server audit
```
bash ./generate_server_audit_run.sh
```

