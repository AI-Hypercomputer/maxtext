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
from typing import Any

import simplejson as json

from google_cloud_mldiagnostics import machinelearning_run

from MaxText.pyconfig import KEYS_NO_LOGGING


class ManagedMLDiagnostics:
  """
  ML Diagnostics Run, implemented with the Singleton pattern.
  Ensures that only one instance of can exist.
  """

  enabled = False

  _instance = None  # Class attribute to hold the single instance

  def __new__(cls, *args: Any, **kwargs: Any):
    """
    Overrides the instance creation method.
    If an instance already exists, it is returned instead of creating a new one.
    """
    if cls._instance is None:
      cls._instance = super(ManagedMLDiagnostics, cls).__new__(cls)

    return cls._instance

  def __init__(self, config):
    """
    Initializes the ManagedMLDiagnostics, ensuring this method runs only once.
    """
    # We need a flag to ensure __init__ only runs once,
    # as the object is returned multiple times by __new__.
    if hasattr(self, "_initialized"):
      return
    self._initialized = True
    self.enabled = config.managed_mldiagnostics
    if not self.enabled:
      return

    # Set up the managed mldiagnostics for profiling and metrics uploading.
    def should_log_key(key, value):
      if key in KEYS_NO_LOGGING:
        return False
      try:
        # Verify the value can be serialized to json. If not, we'll skip it.
        json.dumps(value, allow_nan=False)
      except TypeError:
        return False
      return True

    config_dict = {key: value for key, value in config.get_keys().items() if should_log_key(key, value)}

    # Create a run for the managed mldiagnostics, and upload the configuration.
    machinelearning_run(
        # Use different names if on all devices.
        name=f"{config.run_name}",
        run_group=config.managed_mldiagnostics_run_group,
        configs=config_dict,
        gcs_path=config.managed_mldiagnostics_dir,
        # TODO: b/455623960 - Remove the following once multi-region and prod support are enabled.
        region="us-central1",
        environment="autopush",  # Default would be "prod" for formal launch.
    )
