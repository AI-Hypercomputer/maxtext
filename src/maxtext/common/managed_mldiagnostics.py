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

"""Create the managed mldiagnostics run."""

import json
from typing import Any

from maxtext.common.gcloud_stub import mldiagnostics_modules
from maxtext.utils import max_logging

# `mldiagnostics_modules()` never returns None: when the SDK is missing or
# MaxText runs decoupled from gcloud it hands back a no-op stub and sets the flag.
mldiag, _mldiag_is_stub = mldiagnostics_modules()

from maxtext.configs.pyconfig import KEYS_NO_LOGGING

_MISSING = object()


def mldiagnostics_available() -> bool:
  """Returns whether the real google_cloud_mldiagnostics SDK is loaded (not the stub)."""
  return not _mldiag_is_stub


def _config_items(config) -> dict[str, Any]:
  """Returns the config as a flat dict.

  Pretrain passes a `pyconfig.HyperParameters` (`get_keys()`); the RL trainer
  passes the pydantic
  `RLConfig` returned by `pyconfig.initialize_pydantic` (`model_dump()`).
  """
  if hasattr(config, "get_keys"):
    return config.get_keys()
  return config.model_dump(mode="json")


class ManagedMLDiagnostics:
  """ML Diagnostics Run, implemented with the Singleton pattern.

  Ensures that only one instance of the class can exist.
  """

  _instance = None  # Class attribute to hold the single instance

  def __new__(cls, *args: Any, **kwargs: Any):
    """Overrides the instance creation method.

    If an instance already exists, it is returned instead of creating a new one.
    """
    if cls._instance is None:
      cls._instance = super(ManagedMLDiagnostics, cls).__new__(cls)

    return cls._instance

  def __init__(self, config, sampler_config=None):
    """Creates the ML Diagnostics run once; later calls are no-ops.

    Args:
      config: The (trainer) config. Its keys are uploaded as the run config and
        `config.managed_mldiagnostics_dir` becomes the run's `gcs_path`, under
        which the SDK stores profiles.
      sampler_config: Optional second config (the RL sampler). Keys whose values
        differ from `config` are uploaded as `sampler.<key>`.
    """
    # We need a flag to ensure __init__ only runs once,
    # as the object is returned multiple times by __new__.
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

    # Set up the managed mldiagnostics for profiling and metrics uploading.
    def should_log_key(key, value):
      if key in KEYS_NO_LOGGING:
        return False
      try:
        # Verify the value can be serialized to json. If not, we'll skip it.
        json.dumps(value, allow_nan=False)
      except (TypeError, ValueError):
        return False
      return True

    config_dict = {key: value for key, value in _config_items(config).items() if should_log_key(key, value)}
    if sampler_config is not None and sampler_config is not config:
      for key, value in _config_items(sampler_config).items():
        if should_log_key(key, value) and config_dict.get(key, _MISSING) != value:
          config_dict[f"sampler.{key}"] = value

    # Create a run for the managed mldiagnostics, and upload the configuration.
    region = config.managed_mldiagnostics_region if config.managed_mldiagnostics_region else None
    mldiag.machinelearning_run(
        name=f"{config.run_name}",
        run_group=config.managed_mldiagnostics_run_group,
        configs=config_dict,
        gcs_path=config.managed_mldiagnostics_dir,
        on_demand_xprof=config.managed_mldiagnostics_on_demand_profiling,
        region=region,
    )
