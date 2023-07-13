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

"""Config file for a test job."""

import dataclasses
from typing import Iterable


@dataclasses.dataclass
class TestConfig:
  """This is a class to set up configs of a test job.

  Attributes:
    time_out_in_min: The time duration of a test job in minutes.
    set_up_cmd: The commands to run during setup.
    run_model_cmd: The commands to run during training or serving.
  """

  time_out_in_min: int
  set_up_cmd: Iterable[str]
  run_model_cmd: Iterable[str]
