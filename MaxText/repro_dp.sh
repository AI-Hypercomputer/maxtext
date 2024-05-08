export LIBTPU_INIT_ARGS="--xla_jf_bounds_check"
(python3 MaxText/train.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False dcn_pipeline_parallelism=4 ici_data_parallelism=4 base_num_decoder_layers=64 scan_layers=True   per_device_batch_size=12 base_emb_dim=256 enable_profiler=True scan_pipeline_iterations=True decoder_block=simple) 2>&1 | tee /tmp/log
gcs_path="gs://mattdavidow-maxtext-br/pipeline-debug-a7/slice-${MEGASCALE_SLICE_ID}/"
gsutil cp /tmp/log ${gcs_path}
#megascale_gcs_path="gs://mattdavidow-maxtext-br/pipeline-megascale-debug-a2-${MEGASCALE_SLICE_ID}/"
megascale_gcs_path="gs://mattdavidow-maxtext-br/pipeline-megascale-debug-a3/"
gsutil -m cp -r /tmp/xla_dump/megascale_debug $megascale_gcs_path
