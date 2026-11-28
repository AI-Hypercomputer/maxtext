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

"""m3 configuration types.

Modules:
  * `training_config.py` — frozen Pydantic `TrainingConfig` (mesh, optimizer,
    checkpoint, data, transforms) for LLM pre-training, plus
    `validate_training_config`. Model architecture lives in
    `maxtext.m3.models.model_config.ModelConfig`.
  * `config.py`, `diffusers_config.py` — YAML-driven configs for the diffusion
    pipelines (`maxtext/inference/sample.py`).
  * `glm5_3_flash_config.py` — GLM-5 runtime config and sharding rules.

Keep this package free of package-level imports so config types stay cheap to
import.
"""
