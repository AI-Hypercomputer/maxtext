set -e
idx=$(date +%Y-%m-%d-%H-%M)

base_ckpt_path=gs://mohitkhatwani-maxtext-chkpts/maxtext-gamma-7b-pt-final-orbax/0/default
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
DATASET_PATH=gs://maxtext-dataset

export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIRECTORY \
load_parameters_path=$base_ckpt_path model_name=gamma-7b dataset_path=$DATASET_PATH scan_layers=false \
per_device_batch_size=1 ici_tensor_parallelism=4 async_checkpointing=false assets_path=gs://maxtext-gamma/gamma \
steps=10 metrics_file='metrics.txt'

export LOSS_THRESHOLD=4.0
# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD
