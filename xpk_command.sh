python3 xpk/xpk.py workload create \
--cluster v5e-bodaborg   \
--docker-image gcr.io/tpu-prod-env-multipod/mattdavidow_runner \
--workload mattdavidow-xaot-test-2 \
--tpu-type=v5litepod-16 \
--num-slices=2  \
--command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue/ steps=5 per_device_batch_size=4 load_xaot=True xaot_save_name=xaot_2xv5e-16.pickle enable_checkpointing=False" \
--zone=us-west4-a