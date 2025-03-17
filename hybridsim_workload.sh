# HYBRIDSIM_DOCKER_IMAGE=gcr.io/tpu-prod-env-multipod/hybridsim:$USER-latest
# HYBRIDSIM_DOCKER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/test_hybridsim_jax:2024-12-10
# HYBRIDSIM_DOCKER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/cloud_hybridsim_gcloud_python:2025-03-13
HYBRIDSIM_DOCKER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/test_hybridsim_profile:2025-03-16
module_name_pattern=jit_train_step # This is the module name for MaxText
# gcs_path=gs://hengtaoguo-maxtext-logs/hlo/test/llama3.1-8b-v4-128
# gcs_output_path=gs://hengtaoguo-maxtext-logs/profile_test/testrun-v4-128.txt
# gcs_xplane_path=gs://hengtaoguo-maxtext-logs/profile_test/testrun-v4-128.xplane.pb

gcs_path=gs://hengtaoguo-maxtext-logs/hlo/test/llama3.1-8b-v5e-256
gcs_output_path=gs://hengtaoguo-maxtext-logs/profile_test/testrun-v5e-256.txt
gcs_xplane_path=gs://hengtaoguo-maxtext-logs/profile_test/testrun-v5e-256.xplane.pb

# PROJECT=tpu-prod-env-multipod
# CLUSTER=v4-8-maxtext
# ZONE=us-central2-b
# TPU_TYPE=v4-8
# WORKLOAD_NAME=hengtaoguo-hybridsim-gke-profile5

# PROJECT=tpu-prod-env-multipod
# CLUSTER=v5e-256-opm-ase1
# ZONE=asia-southeast1-b
# TPU_TYPE=v5e-256
# WORKLOAD_NAME=hengtaoguo-hybridsim-gke-profile5

python3 ../xpk/xpk.py workload create \
--project $PROJECT \
--cluster $CLUSTER \
--zone $ZONE \
--workload ${WORKLOAD_NAME} \
--base-docker-image $HYBRIDSIM_DOCKER_IMAGE \
--tpu-type=$TPU_TYPE \
--num-slices=1 \
--command="/usr/hybridsim/run/cloud_hybrid_sim_client_main.par \
--uid= \
--enforce_kernel_ipv6_support=false \
--rpc_security_protocol= \
--rpc_default_rate_acl=INSECURE \
--rpc_security_protocol_empty_check_enabled=false \
--envelope_enabled=false \
--deepsea_chips_per_host_bounds=1,1,1 \
--deepsea_host_bounds=1,1,1 \
--vmodule=hybrid_sim=2 \
--alsologtostderr \
--gcs_path=$gcs_path \
--gcs_output_path=$gcs_output_path \
--gcs_xplane_path=$gcs_xplane_path \
--module_name_pattern=$module_name_pattern \
--deepsea_chip_config_name=default 2>&1 | tee ./hybridsim_log.txt; \
cat hybridsim_log.txt | grep estimated_cost_ns;"

python3 ../xpk/xpk.py workload delete --project tpu-prod-env-multipod \
  --cluster $CLUSTER --zone $ZONE --workload hengtaoguo-hybridsim-gke-profile5