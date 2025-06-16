"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Define user specific configurations for recipes here."""

import os
import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

# Cluster Parameters
cluster_config = XpkClusterConfig(
    cluster_name="pw-scale-test-v5e-32",
    project="cloud-tpu-multipod-dev",
    zone="us-south1",
    device_type="v5litepod-32",
)

# Path to your local xpk checkout
xpk_path = os.path.join("~", "xpk")

# User and project details
user = os.environ["USER"]
project = cluster_config.project
country = "us"  # Used for GCS bucket naming

# GCS bucket for output
base_output_directory = f"gs://{user}-{project}-{country}/pw_small_run/"

# Docker Image URIs
proxy_image = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_proxy_server"
server_image = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_server"
runner = "gcr.io/tpu-prod-env-multipod/sujinesh_maxtext_latest"
colocated_python_image = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/remote_python_sidecar_server"

# Pathways-specific configurations
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner,
    colocated_python_sidecar_image=colocated_python_image,
)
