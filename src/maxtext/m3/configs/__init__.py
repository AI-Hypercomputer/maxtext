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

"""m3 configuration: slim Pydantic types plus YAML defaults.

Planned contents:
  * `config.py` — `ModelConfig`, `TransformConfig`, `RootConfig`, `load_config`.
  * `base.yml` — base defaults that model YAMLs override.

m3 configs are intentionally small and additive: fields are added as
architectures require them, not preemptively. Interop with the legacy
`maxtext.configs.pyconfig.HyperParameters` surface (required by the training
engine and the RL path) is handled by a separate bridge, not by widening these
types.
"""
