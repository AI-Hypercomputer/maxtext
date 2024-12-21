
python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-multipod' --zone='europe-west4' --device_type=v6e-256 --num_slices=2 --cluster_name='mlperf-v6e-256' \
--model_name="llama3_1_405b_8192_fsdp_dcn" --base_output_directory="gs://maxtext-experiments-tpem/" --libtpu_version=20241009 --base_docker_image maxtext_base_image

python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-automated' --zone='us-east5' --device_type=v6e-256 --num_slices=4  --cluster_name='bodaborg-v6e-256' --base_output_directory="gs://maxtext-experiments-tpem/" \
--model_name="llama3_1_405b_8192_fsdp_dcn_c4" --base_output_directory="gs://maxtext-experiments-tpem/" --libtpu_version=20241028 --base_docker_image maxtext_base_image

python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-one-vm' --zone='us-east5' --device_type=v6e-256 --num_slices=4  --cluster_name='bodaborg-v6e-256' --base_output_directory="gs://maxtext-experiments-tpem/" \
--model_name="llama3_1_8b_8192_c4" --base_output_directory="gs://maxtext-experiments-tpem/" --libtpu_version=20241028 --base_docker_image maxtext_base_image

python3 benchmarks/benchmark_runner.py --project='tpu-prod-env-one-vm' --zone='us-east5' --device_type=v6e-256 --num_slices=4  --cluster_name='bodaborg-v6e-256' --base_output_directory="gs://maxtext-experiments-tpem/" \
--model_name="llama3_70b_8192" --base_output_directory="gs://maxtext-experiments-tpem/" --libtpu_version=20241028 --base_docker_image maxtext_base_image
