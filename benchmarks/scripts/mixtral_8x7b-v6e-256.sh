python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-multipod' --zone='europe-west4' --device_type=v6e-256 --num_slices=1  --cluster_name='mlperf-v6e-256' \
--model_name="mixtral_8x7b_dropped_int8" --libtpu_version=20241009 --base_docker_image maxtext_base_image


python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-automated' --zone='us-east5' --device_type=v6e-256 --num_slices=1  --cluster_name='bodaborg-v6e-256' --base_output_directory="gs://maxtext-experiments-tpem/" \
--model_name="mixtral_8x7b_dropped_int8" --libtpu_version=20241028 --base_docker_image maxtext_base_image