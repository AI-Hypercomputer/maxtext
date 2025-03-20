# Run offline performance benchmarks.


## Create TPU VM.
Follow these [instructions](https://cloud.google.com/tpu/docs/v5e-inference#tpu-vm) to create TPU v5e-8 VM and ssh into the VM


### Setup a virtual env
sudo apt install python3.10-venv
python -m venv .env
source .env/bin/activate

### Install loadgen
```
sudo apt-get install python3-dev
sudo apt-get install build-essential -y
git clone git@github.com:mlcommons/inference.git
cd inference/
cd loadgen/ && pip install .
```

### Download datasets

```
export DATA_DISK_DIR=~/loadgen_run_data
mkdir -p ${DATA_DISK_DIR}
cd ${DATA_DISK_DIR}
```

#### LLama2-70b:

```
gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl .
mv open_orca_gpt4_tokenized_llama.calibration_1000.pkl processed-calibration-data.pkl

gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl .
mv open_orca_gpt4_tokenized_llama.sampled_24576.pkl processed-data.pkl
```

#### Mixtral-8x7b:
```
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl .
mv mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl mixtral-processed-calibration-data.pkl

wget https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl .
mv 09292024_mixtral_15k_mintoken2_v1.pkl mixtral-processed-data.pkl
```

### Install Maxtext
```
cd ~
git clone git@github.com:google/maxtext.git
cd maxtext
bash setup.sh
pip install -r MaxText/inference_mlperf/requirements.txt
```

### Generate quantized checkpoint

* Download HF checkpoints and convert to Maxtext/Orbax format:
#### LLama2-70b:
Steps to get a quantized llama2-70B checkpoint for v5e-8

Note llama2-70B model takes about 140G of memory and will not fit into a v5e-8. It must be downloaded onto a large machine (such as v5p-8) and quantized to a smaller quantized checkpoint to be loaded onto a v5e-8 machine.

* Obtain a llama2-70b checkpoint and convert it to a maxtext inference checkpoint. Please follow maxtext instructions specified here: https://github.com/google/maxtext/blob/main/getting_started/Run_Llama2.md

#### Mixtral-8x7b:
Run the mixtral-8x7B script to generate a new bf16 checkpoint - https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh
For example, here is a bf16 checkpoint generaed by the script -- "gs://ml-auto-solutions/output/multipod/maxtext/chained_tests_mixtral-8x7b_stable-2024-09-15-04-01-09/unscanned_ckpt/checkpoints/0/items"

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

#### CPU based quantization with Llama3.1-405b:

The llama3.1-405b model takes about 800GB of memory. This does not fit in TPU machines and must be downloaded onto a large CPU machine (such as `m1-ultramem-160`) and quantized to a smaller quantized checkpoint (~400GB) to be loaded to TPUs for serving. After obtaining a llama3.1-405b checkpoint and converting it to a maxtext inference checkpoint, you can convert the checkpoint to a quantized checkpoint:

1. Define paths to load maxtext checkpoint from and save quantized checkpoint to

```
export LOAD_PARAMS_PATH=gs://${USER}-bkt/llama3.1-405b/param-only-decode-ckpt-maxtext/checkpoints/0/items

export SAVE_QUANT_PARAMS_PATH=gs://${USER}-bkt/quantized/llama3.1-405b
```

2. Run the following maxtext script to generate and save an int8 quantized checkpoint

```
export TOKENIZER_PATH=assets/tokenizer_llama3.tiktoken
export MODEL_SIZE=llama3.1-405b
export QUANTIZE_TYPE=int8

cd maxtext && \
python3 MaxText/load_and_quantize_checkpoint.py MaxText/configs/base.yml tokenizer_path=${TOKENIZER} load_parameters_path=${LOAD_PARAMS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=${MODEL_SIZE} ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 attention=dot_product quantization=${QUANTIZE_TYPE} save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH} async_checkpointing=false
```

The quantized checkpoint is saved at `${SAVE_QUANT_PARAMS_PATH}`

### HuggingFace login
```
export HUGGING_FACE_TOKEN=<your_hugging_face_token>
huggingface-cli login --token $HUGGING_FACE_TOKEN
```

### Run Offline Benchmarks

#### For trillium
#### LLama2-70b:
```
cd ~/maxtext/MaxText/inference_mlperf/trillium
```

##### Test Run
```
bash benchmarks_llama2-70b-trillium_2x4.sh -b=performance -t
```

##### Performance Only:
```
bash benchmarks_llama2-70b-trillium_2x4.sh -b=performance
```

##### Accuracy Only:
```
bash benchmarks_llama2-70b-trillium_2x4.sh -b=accuracy
```

##### Audit Only:
```
bash benchmarks_llama2-70b-trillium_2x4.sh -b=audit
```

##### Run all benchmarks:
```
bash benchmarks_llama2-70b-trillium_2x4.sh -b=all
```

#### Mixtral-8x7b:
```
export PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="256,144|512,72|2048,18"
export MAXENGINE_ARGS="model_name=mixtral-8x7b tokenizer_path=${TOKENIZER_PATH}  quantization=int8 quantize_kvcache=True load_parameters_path=${SAVE_QUANT_PARAMS_PATH} checkpoint_is_quantized=True megablox=False sparse_matmul=False capacity_factor=1 model_call_mode=inference compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
```

##### Test Run
```
bash ./mixtral_offline_run.sh -t
```

##### Performance Only:
```
bash ./mixtral_offline_run.sh
```

##### Accuracy Only:
```
bash ./mixtral_offline_run.sh -a
```

##### Audit Only:
```
bash ./mixtral_offline_run.sh -d
```

### Profiling

```
# Capture profile
bash ./llama_offline_run.sh -p -e
python -m jax.collect_profile 9999 2000 --log_dir /tmp/profiles --no_perfetto_link

# View profile
tensorboard --logdir /tmp/profiles
```
