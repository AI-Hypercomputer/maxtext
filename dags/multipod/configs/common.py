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

"""Utilities to construct common configs."""

from typing import Tuple
import enum

UPGRADE_PIP = "pip install --upgrade pip"


class SetupMode(enum.Enum):
  STABLE = "stable"
  NIGHTLY = "nightly"


class Platform(enum.Enum):
  GCE = "gce"
  GKE = "gke"


def download_maxtext() -> Tuple[str]:
  """Download MaxText repo."""
  return (
      UPGRADE_PIP,
      "git clone https://github.com/google/maxtext.git /tmp/maxtext",
  )


def setup_maxtext(mode: SetupMode, platform: Platform) -> Tuple[str]:
  """Common set up for MaxText repo."""
  return download_maxtext() + (
      f"cd /tmp/maxtext && bash setup.sh MODE={mode.value} && bash preflight.sh PLATFORM={platform.value}",
  )
