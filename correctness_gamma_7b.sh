#export M_LOAD_PARAMETERS_PATH=gs://maxtext-gamma/7b/2024-02-29-17-56/0/default
export M_LOAD_PARAMETERS_PATH=/home/rwitten/2024-02-29-17-56/0/default
#export TPU_STDERR_LOG_LEVEL=0
#export TPU_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_LOG_LEVEL=0
export M_ICI_FSDP_PARALLELISM=1
export M_ICI_AUTOREGRESSIVE_PARALLELISM=-1
#python MaxText/maxengine_server.py MaxText/configs/base.yml assets_path=gs://maxtext-gamma/gamma  per_device_batch_size=24 run_name=runner_2024-02-29-18-00 max_prefill_predict_length=1024 max_target_length=2048 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-7b attention=recommended
#python MaxText/maxengine_server.py MaxText/configs/base.yml assets_path=gs://maxtext-gamma/gamma  per_device_batch_size=24 run_name=runner_2024-02-29-18-00 max_prefill_predict_length=1024 max_target_length=2048 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-7b attention=recommended
python MaxText/maxengine_server.py MaxText/configs/base.yml assets_path=gs://maxtext-gamma/gamma  per_device_batch_size=7 run_name=runner_2024-02-29-18-00 max_prefill_predict_length=1024 max_target_length=2048 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-7b attention=recommended
