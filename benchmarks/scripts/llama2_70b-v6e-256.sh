
python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-multipod' --zone='europe-west4' --device_type=v6e-256 --num_slices=1 --cluster_name='mlperf-v6e-256' \
--model_name="llama2_70b_4096" --base_output_directory="gs://maxtext-experiments-tpem/" --libtpu_version=20241009 --base_docker_image maxtext_base_image