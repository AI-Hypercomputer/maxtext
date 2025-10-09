# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define user specific configurations for recipes here."""

import os
import maxtext_trillium_model_configs as model_configs
import maxtext_v5e_model_configs as v5e_model_configs
import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

# Cluster and XPK Path Configuration
# Define the target cluster and the path to the xpk tool.
cluster_config = XpkClusterConfig(
    cluster_name="pw-scale-test-v5e-32",
    project="cloud-tpu-multipod-dev",
    zone="us-south1",
    device_type="v5litepod-32",
)
xpk_path = "../xpk"

# User and GCS Path Configurations
user = os.environ.get("USER", "default_user")
project = cluster_config.project
region = "-".join(cluster_config.zone.split("-")[:-1])
base_output_directory = f"gs://{user}-{project}-{region}"

# Docker Image Configurations
# Specify the container images for the pathways job.
# proxy_image = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
# server_image = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest"
# colocated_python_image = None
runner_image = "gcr.io/cloud-tpu-multipod-dev/sujinesh_maxtext_latest"

server_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server_maxtext:latest"
proxy_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server_maxtext:latest"
# proxy_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:sidecar"
# server_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:sidecar"
# colocated_python_image="gcr.io/cloud-tpu-multipod-dev/sujinesh_sidecar_maxtext:latest"
colocated_python_image="gcr.io/cloud-tpu-multipod-dev/ksadi_sidecar_maxtext:latest"

# Whether to run in headless mode.
headless=True
# headless=False

# Pathways Configuration
# This object bundles all the image settings for the runner.
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner_image,
    colocated_python_sidecar_image=colocated_python_image,
    headless=headless,
)

# Model and Slice Configurations
# Define the list of models and slice configurations to run.
list_of_models = [
    # model_configs.default_basic_1,
    v5e_model_configs.llama3_1_8b_8192_v5e_32,
]
num_slices_list = [1]

# Default Workload Configurations
# These parameters are used in the WorkloadConfig and can be easily tuned here.
workload_config_defaults = {
    "max_restarts": 0,
    "libtpu_type": None,
    "libtpu_nightly_version": "",
    "base_docker_image": "",
    "num_steps": 25,
}