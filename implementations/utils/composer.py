# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities get Composer configs."""

from typing import Mapping
from implementations.utils import tpu as util
import requests


def get_composer_data(project: str, region: str, env: str) -> Mapping[str, str]:
  """Get composer metadata.

  Args:
   project: The project name of the composer.
   region: The region of the composer.
   env: The environment name of the composer.

  Returns:
  A dict mapping metadata.
  """
  request_endpoint = f"https://composer.googleapis.com/v1beta1/projects/{project}/locations/{region}/environments/{env}"
  response = requests.get(request_endpoint, headers=util.get_headers())
  print("response.json()", response.json())
  return response.json()


def get_airflow_url(project: str, region: str, env: str) -> str:
  """Get Airflow web UI.

  Args:
   project: The project name of the composer.
   region: The region of the composer.
   env: The environment name of the composer.

  Returns:
  The URL of Airflow.
  """
  configs = get_composer_data(project, region, env)
  return configs["config"]["airflowUri"]
