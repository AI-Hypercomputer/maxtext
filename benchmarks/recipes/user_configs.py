# Copyright 2023–2026 Google LLC
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

"""User-configurable settings for running MaxText benchmarks.

This file defines the `UserConfig` dataclass, which serves as a centralized
location for users to specify their GCP environment, desired models, and
execution parameters for running benchmarks with Cluster Toolkit (gcluster). The
`USER_CONFIG` instance at the bottom of the file is the main object to be
modified for custom runs.
"""

import argparse
import dataclasses
from typing import Any

from benchmarks import maxtext_trillium_model_configs as v6e_model_configs
from benchmarks import maxtext_v5e_model_configs as v5e_model_configs
from benchmarks import maxtext_v5p_model_configs as v5p_model_configs
from benchmarks.recipes import parser_utils
from benchmarks.recipes.pw_utils import build_user_models, get_cluster_config, get_pathways_config


AVAILABLE_MODELS_FRAMEWORKS = ["mcjax", "pathways"]

AVAILABLE_MODELS = {
    "v6e": v6e_model_configs.trillium_model_dict,
    "v5litepod": v5e_model_configs.v5e_model_dict,
    "v5p": v5p_model_configs.v5p_model_dict,
}


@dataclasses.dataclass
class UserConfig:
  """The default configuration can be modified here."""

  # gcp configuration
  user: str = "user_name"
  cluster_name: str = "v6e-256-cluster"
  project: str = "tpu-prod-env-cluster"
  zone: str = "us-east5-b"
  location: str | None = None
  device_type: str = "v6e-256"
  priority: str = "medium"
  base_output_directory: str = None

  # Images for env
  server_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
  proxy_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
  runner: str = "us-docker.pkg.dev/path/to/maxtext_runner"
  colocated_python_image: str = None
  worker_flags: str = ""
  proxy_flags: str = ""
  server_flags: str = ""

  # model configuration
  benchmark_steps: int = 20
  headless: bool = False
  selected_model_framework: list[str] = dataclasses.field(default_factory=lambda: ["pathways"])  # pathways, mcjax
  selected_model_names: list[str] = dataclasses.field(default_factory=lambda: ["llama3_1_8b_8192"])
  num_slices_list: list[int] = dataclasses.field(default_factory=lambda: [2])

  # BigQuery configuration
  bq_enable: bool = False
  bq_db_project: str = ""
  bq_db_dataset: str = ""

  # other configuration
  xpk_path: str | None = None
  delete: bool = False
  max_restarts: int = 0
  temp_key: str = None
  workload_id: str = None
  skip_validation: bool = False

  # MaxText training overrides (None = not set). Must match parser_utils.MAXTEXT_FLAG_SPECS.
  # Elastic training
  elastic_enabled: bool | None = None
  enable_single_controller: bool | None = None
  elastic_min_slice_count: int | None = None
  # Goodput
  enable_pathways_goodput: bool | None = None
  enable_goodput_recording: bool | None = None
  goodput_upload_interval_seconds: int | None = None
  monitor_goodput: bool | None = None
  # Checkpointing
  enable_checkpointing: bool | None = None
  async_checkpointing: bool | None = None
  enable_checkpoint_cloud_logger: bool | None = None
  checkpoint_period: int | None = None
  # Colocated Python data input / dataset
  colocated_python_data_input: bool | None = None
  tokenizer_path: str | None = None
  dataset_type: str | None = None
  grain_train_files: str | None = None
  # Generic MaxText key=value overrides (value None = remove key from the model config).
  maxtext_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Derived attributes initialized in __post_init__
  cluster_config: Any = dataclasses.field(init=False, default=None)
  region: str = dataclasses.field(init=False, default="")
  pathways_config: Any = dataclasses.field(init=False, default=None)
  headless_workload_name: str = dataclasses.field(init=False, default="")
  models: dict[str, list[Any]] = dataclasses.field(init=False, default_factory=dict)

  def __post_init__(self):
    """Automatically generate derived attributes after the object is created."""
    if self.location:
      self.zone = self.location if len(self.location.split("-")) == 3 else ""
    self.cluster_config = get_cluster_config(self.cluster_name, self.project, self.zone, self.device_type, self.location)

    loc = self.location or self.zone
    loc_parts = loc.split("-")
    self.region = "-".join(loc_parts[:-1]) if len(loc_parts) >= 3 else loc
    self.pathways_config = get_pathways_config(
        self.server_image,
        self.proxy_image,
        self.runner,
        self.colocated_python_image,
        self.headless,
        self.server_flags,
        self.proxy_flags,
        self.worker_flags,
    )
    self.headless_workload_name = f"{self.user[:3]}-headless"
    if not self.base_output_directory:
      # Default prefix format (ends with a hyphen)
      self.base_output_directory = f"gs://{self.user}-{self.region}/{self.user}-"
    elif not self.base_output_directory.endswith("/"):
      # Ensure user-provided directories always end with a slash
      self.base_output_directory += "/"

    device_base_type = self.device_type.split("-", maxsplit=1)[0]
    self.models = build_user_models(
        self.selected_model_framework,
        self.selected_model_names,
        device_base_type,
        AVAILABLE_MODELS_FRAMEWORKS,
        AVAILABLE_MODELS,
    )
    # Fail fast on conflicting overrides, before any cluster interaction.
    self.tuning_param_overrides()

  def tuning_param_overrides(self) -> dict[str, Any]:
    """Returns MaxText overrides: --maxtext_override entries plus the typed flags that are set."""
    typed = {name: getattr(self, name) for name in parser_utils.MAXTEXT_FLAG_NAMES if getattr(self, name) is not None}
    conflicts = sorted(
        k
        for k in typed.keys() & self.maxtext_overrides.keys()
        if typed[k] != _parse_override_like_flag(k, self.maxtext_overrides[k])
    )
    if conflicts:
      raise ValueError(f"Conflicting values for {conflicts} between typed flags and --maxtext_override.")
    return {**self.maxtext_overrides, **typed}


_FLAG_TYPES = {name: arg_type for name, arg_type, _ in parser_utils.MAXTEXT_FLAG_SPECS}


def _parse_override_like_flag(key: str, value: Any) -> Any:
  """Parses a --maxtext_override value with the matching typed flag's parser, so equal values compare equal.

  For example `--maxtext_override enable_checkpointing=true` parses to True like `--enable_checkpointing=true`.
  Unset (None) and unparsable values are returned unchanged, so they still differ from any typed value.
  """
  if value is None:
    return None
  try:
    return _FLAG_TYPES[key](value)
  except (argparse.ArgumentTypeError, TypeError, ValueError):
    return value


# Define the required configuration here
USER_CONFIG = UserConfig(
    user="user_name",
    cluster_name="v6e-256-cluster",
    project="tpu-prod-env-cluster",
    zone="us-east5-b",
    device_type="v6e-256",
    benchmark_steps=20,
    num_slices_list=[2],
    server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server",
    proxy_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server",
    runner="us-docker.pkg.dev/path/to/maxtext_runner",
    selected_model_framework=["pathways"],
    selected_model_names=["llama3_1_8b_8192"],
    priority="medium",
    base_output_directory=None,  # GCS Bucket path
    # Optional parameters, useful for single controller data loading optimizations
    # proxy_flags="--sidecar_name=external",
)
