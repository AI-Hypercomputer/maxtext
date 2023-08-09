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

"""Config file for Google Cloud Project (GCP)."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class GCPConfig:
  """This is a class to set up configs of GCP.

  Attributes:
    project_name: The name of a project to run a test job.
    zone: The zone to run a test job.
    database_name: The name of database for metrics.
  """

  project_name: str
  zone: str
  database_name: Optional[str] = None
