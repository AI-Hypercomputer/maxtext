export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export TPU_VMODULE=tpu_configuration_ops_impl=3
export JAX_USE_PJRT_C_API_ON_TPU=1
export TF_CPP_MIN_LOG_LEVEL=0
python3 MaxText/train.py MaxText/configs/base.yml run_name=llama2-finetuning-maxtext base_output_directory=gs://mazumdera-test-bucket-us-west4/maxtext/llama2/02062024/1  dataset_path=gs://maxtext-dataset load_parameters_path=gs://maxtext-llama-us-west4/llama2-70b/maxtext-ckpt/1/0/default model_name="llama2-70b" per_device_batch_size=.0625 assets_path=gs://maxtext-llama-us-west4/llama2-70b steps=5 async_checkpointing=false

# python3 MaxText/train.py MaxText/configs/base.yml run_name=llama2-finetuning-maxtext base_output_directory=gs://mazumdera-test-bucket-us-central2/maxtext/llama2/02062024/1  dataset_path=gs://maxtext-dataset model_name="llama2-70b" per_device_batch_size=.0625 assets_path=gs://maxtext-llama/llama2-7b steps=5 enable_checkpointing=false