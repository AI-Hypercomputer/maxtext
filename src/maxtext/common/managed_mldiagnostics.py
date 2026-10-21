# Copyright 2025 Google LLC
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

"""Managed ML Diagnostics run lifecycle management for MaxText."""

import json
from typing import Any

from maxtext.common.gcloud_stub import mldiagnostics_modules
from maxtext.utils import max_logging

mldiag, _mldiag_is_stub = mldiagnostics_modules()

from maxtext.configs.pyconfig import KEYS_NO_LOGGING


def mldiagnostics_available() -> bool:
  """Returns whether the real google_cloud_mldiagnostics SDK is loaded (not the stub)."""
  return not _mldiag_is_stub


def _config_items(config: Any) -> dict[str, Any]:
  """Extracts configuration key-value pairs as a flat dictionary."""
  if hasattr(config, "get_keys"):
    return config.get_keys()
  return config.model_dump(mode="json")


class ManagedMLDiagnostics:
  """Centralized ML Diagnostics Run manager implemented as a Singleton."""

  _instance = None

  def __new__(cls, *args: Any, **kwargs: Any):
    if cls._instance is None:
      cls._instance = super(ManagedMLDiagnostics, cls).__new__(cls)
    return cls._instance

  def __init__(self, config: Any, sampler_config: Any | None = None):
    if hasattr(self, "_initialized"):
      return
    self._initialized = True
    if not config.managed_mldiagnostics:
      return

    if not mldiagnostics_available():
      max_logging.warning(
          "managed_mldiagnostics=True, but the google_cloud_mldiagnostics SDK"
          " is not available (not installed, or MaxText is running in decoupled"
          " mode); no ML Diagnostics run will be created."
      )

    def should_log_key(key: str, value: Any) -> bool:
      if key in KEYS_NO_LOGGING:
        return False
      try:
        json.dumps(value, allow_nan=False)
      except (TypeError, ValueError):
        return False
      return True

    config_dict = {key: value for key, value in _config_items(config).items() if should_log_key(key, value)}

    if sampler_config is not None and sampler_config is not config:
      for key, value in _config_items(sampler_config).items():
        if should_log_key(key, value) and config_dict.get(key) != value:
          config_dict[f"sampler.{key}"] = value

    region = config.managed_mldiagnostics_region if config.managed_mldiagnostics_region else None
    mldiag.machinelearning_run(
        name=f"{config.run_name}",
        run_group=config.managed_mldiagnostics_run_group,
        configs=config_dict,
        gcs_path=config.managed_mldiagnostics_dir,
        on_demand_xprof=config.managed_mldiagnostics_on_demand_profiling,
        region=region,
        metric_only_run=config.managed_mldiagnostics_metric_only_run,
        application_framework="MAXTEXT",
    )
