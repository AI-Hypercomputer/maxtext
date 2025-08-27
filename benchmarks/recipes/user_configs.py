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

import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

# cluster_config = XpkClusterConfig(
#     cluster_name="bodaborg-v6e-256-lcscld-c",
#     project="tpu-prod-env-one-vm",
#     zone="southamerica-west1-a",
#     device_type="v6e-256",
# )
# base_output_directory = f"gs://sujinesh-southamerica1/sept_2025_scale_test/"

cluster_config = XpkClusterConfig(
    cluster_name="tpu-cpq-yucmhke-v6e-256-qual",
    project="tpu-prod-env-multipod",
    zone="us-east5-a",
    device_type="v6e-256",
)
base_output_directory = f"gs://maxtext-scale-test-2025-08-29/pause_resume/"

xpk_path = "../xpk"

user = os.environ["USER"]
region = "-".join(cluster_config.zone.split("-")[:-1])
proxy_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_proxy_server:latest"
)
server_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_server:latest"
)
colocated_python_image = f"gcr.io/cloud-tpu-multipod-dev/sujinesh_sidecar_debug:latest"
runner = f"gcr.io/tpu-prod-env-one-vm/sujinesh_latest:latest"
headless = False
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner,
    colocated_python_sidecar_image=colocated_python_image,
    headless=headless,
)
headless_workload_name = f"{user[:3]}-headless"
