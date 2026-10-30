# Copyright 2026 Google LLC
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

"""m3 configuration: pure-Python Pydantic v2 runtime configs, CLI reflection parser, and `pyconfig` bridge.

Modules:
  * `training_config.py` — `TrainingConfig`, `MeshConfig`, `TransformConfig`, `DataPipelineConfig`.
  * `cli_parser.py` — `M3ArgumentParser` (`--preset`, dotted leaf overrides, `--save_config_json`).
  * `pyconfig_bridge.py` — `from_pyconfig`, `ConfigMigrationError`, `format_cli_migration_hint`.
"""

from maxtext.m3.configs.cli_parser import M3ArgumentParser
from maxtext.m3.configs.pyconfig_bridge import (
    ConfigMigrationError,
    format_cli_migration_hint,
    from_pyconfig,
)
from maxtext.m3.configs.training_config import (
    DataPipelineConfig,
    MeshConfig,
    TrainingConfig,
    TransformConfig,
)

__all__ = [
    "ConfigMigrationError",
    "DataPipelineConfig",
    "M3ArgumentParser",
    "MeshConfig",
    "TrainingConfig",
    "TransformConfig",
    "format_cli_migration_hint",
    "from_pyconfig",
]
