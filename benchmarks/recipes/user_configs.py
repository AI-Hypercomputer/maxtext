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
import sys
import args_helper as helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig

cluster_config = XpkClusterConfig(
    cluster_name="bodaborg-v6e-256-tt-c",
    project="tpu-prod-env-multipod",
    zone="us-west1-c",
    device_type="v6e-256",
)
xpk_path = "../xpk"

user = os.environ["USER"]
region = "-".join(cluster_config.zone.split("-")[:-1])
proxy_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/{user}/"
    "unsanitized_proxy_server:latest"
)
server_image = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/{user}/"
    "unsanitized_server:latest"
)
colocated_python_image = None # f"gcr.io/{cluster_config.project}/{user}/colocated_python_sidecar_latest:latest"
runner = f"gcr.io/{cluster_config.project}/{user}_maxtext_latest:latest"
BASE_OUTPUT_DIRECTORY = f"gs://{user}-{region}/{user}"
headless = False
pathways_config = mxr.PathwaysConfig(
    server_image=server_image,
    proxy_server_image=proxy_image,
    runner_image=runner,
    colocated_python_sidecar_image=colocated_python_image or None,
    headless=headless,
)
headless_workload_name = f"{user[:3]}-headless-v5e"




# This needs to be set to True to test restore and if you want to restore from
# a previous run, you'll need to set RESUME_CHECKPOINT_NAMES below.
TEST_RESTORE = False
MAX_RESTARTS = 100

BENCHMARK_STEPS = 41
RESTORE_BENCHMARK_STEPS = 20  # Define steps for restore run

# RESUME_CHECKPOINT_NAMES = {
#     "pathways": {
#         # Key is number of slices, value is a dictionary of run_name,
#         # base_output_directory, and num_steps.
#         32: {
#             "run_name":  "restoring_run_name",
#             "base_output_directory": f"gs://{BASE_OUTPUT_DIRECTORY}/...",
#             "num_steps": BENCHMARK_STEPS + RESTORE_BENCHMARK_STEPS,
#         }
#     },
#     # "mcjax": {
#         # 32: {}
#     # },
# }

