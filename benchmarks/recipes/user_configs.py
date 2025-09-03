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

import src/MaxText_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

cluster_config = XpkClusterConfig(
    cluster_name="test-v5e-32-cluster",
    project="cloud-tpu-cluster",
    zone="us-south1-a",
    device_type="v5litepod-32",
)
xpk_path = "~/xpk"

user = os.environ["USER"]
region = "-".join(cluster_config.zone.split("-")[:-1])
proxy_image = (
    f"us-docker.pkg.dev/path/to/{user}/proxy_server"
)
server_image = (
    f"us-docker.pkg.dev/path/to/{user}/server"
)
colocated_python_image = f"gcr.io/{cluster_config.project}/path/to/{user}/colocated_python_sidecar"
runner = f"gcr.io/{cluster_config.project}/{user}_src/MaxText_latest:latest"
base_output_directory = f"gs://{user}-{region}/{user}"
headless = True
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner,
    colocated_python_sidecar_image=colocated_python_image,
    headless=headless,
)
headless_workload_name = f"{user[:3]}-headless"
